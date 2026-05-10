from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PredictionRecord:
    sender_id: str
    ticker: str
    direction: str
    confidence: float
    date: str


class ReputationTracker:
    def __init__(self):
        self.predictions: list[PredictionRecord] = []
        self.outcomes: list[dict] = []
        self.belief_changes: list[dict] = []
        self._reputation_cache: dict[str, float] = {}
        self._calibration_cache: dict[str, float] = {}

    def record_prediction(self, sender_id, ticker, direction, confidence, date) -> None:
        self.predictions.append(
            PredictionRecord(
                sender_id=str(sender_id),
                ticker=str(ticker),
                direction=str(direction),
                confidence=float(max(0.0, min(1.0, confidence))),
                date=str(pd.Timestamp(date))[:10],
            )
        )
        self._clear_accuracy_caches()

    def record_outcome(self, ticker, realized_return, date) -> None:
        self.outcomes.append(
            {
                "ticker": str(ticker),
                "realized_return": float(realized_return),
                "date": str(pd.Timestamp(date))[:10],
            }
        )
        self._clear_accuracy_caches()

    def record_belief_change(self, receiver_id, sender_id, ticker, before, after, date) -> None:
        self.belief_changes.append(
            {
                "receiver_id": receiver_id,
                "sender_id": sender_id,
                "ticker": ticker,
                "before": float(before),
                "after": float(after),
                "abs_change": float(abs(after - before)),
                "date": str(pd.Timestamp(date))[:10],
            }
        )

    def get_reputation(self, sender_id) -> float:
        if sender_id in self._reputation_cache:
            return self._reputation_cache[sender_id]
        matched = self._matched_predictions(sender_id)
        if not matched:
            self._reputation_cache[sender_id] = 0.5
            return 0.5
        value = float(np.mean([self._is_correct(pred.direction, outcome) for pred, outcome in matched]))
        self._reputation_cache[sender_id] = value
        return value

    def get_calibration_error(self, sender_id) -> float:
        if sender_id in self._calibration_cache:
            return self._calibration_cache[sender_id]
        matched = self._matched_predictions(sender_id)
        if not matched:
            self._calibration_cache[sender_id] = 0.25
            return 0.25
        errors = []
        for pred, outcome in matched:
            actual = 1.0 if self._is_correct(pred.direction, outcome) else 0.0
            errors.append((pred.confidence - actual) ** 2)
        value = float(np.mean(errors))
        self._calibration_cache[sender_id] = value
        return value

    def get_influence_score(self, sender_id) -> float:
        changes = [row["abs_change"] for row in self.belief_changes if row["sender_id"] == sender_id]
        return float(np.mean(changes)) if changes else 0.0

    def to_dataframe(self) -> pd.DataFrame:
        sender_ids = sorted({pred.sender_id for pred in self.predictions})
        rows = []
        for sender_id in sender_ids:
            sender_predictions = [pred for pred in self.predictions if pred.sender_id == sender_id]
            rows.append(
                {
                    "sender_id": sender_id,
                    "prediction_count": len(sender_predictions),
                    "avg_confidence": float(np.mean([pred.confidence for pred in sender_predictions])),
                    "reputation": self.get_reputation(sender_id),
                    "calibration_error": self.get_calibration_error(sender_id),
                    "influence_score": self.get_influence_score(sender_id),
                }
            )
        return pd.DataFrame(
            rows,
            columns=[
                "sender_id",
                "prediction_count",
                "avg_confidence",
                "reputation",
                "calibration_error",
                "influence_score",
            ],
        )

    def _matched_predictions(self, sender_id) -> list[tuple[PredictionRecord, float]]:
        matched = []
        for pred in self.predictions:
            if pred.sender_id != sender_id:
                continue
            outcome = self._first_outcome_after(pred.ticker, pred.date)
            if outcome is not None:
                matched.append((pred, outcome))
        return matched

    def _first_outcome_after(self, ticker, date) -> float | None:
        start = pd.Timestamp(date)
        candidates = [
            row
            for row in self.outcomes
            if row["ticker"] == ticker and pd.Timestamp(row["date"]) > start
        ]
        if not candidates:
            return None
        first = min(candidates, key=lambda row: pd.Timestamp(row["date"]))
        return float(first["realized_return"])

    def _is_correct(self, direction: str, realized_return: float) -> bool:
        if direction == "bullish":
            return realized_return > 0
        if direction == "bearish":
            return realized_return < 0
        return abs(realized_return) <= 0.002

    def _clear_accuracy_caches(self) -> None:
        self._reputation_cache.clear()
        self._calibration_cache.clear()

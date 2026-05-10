from __future__ import annotations

import pandas as pd

from aggregation import clamp_signal, reputation_weighted


class BeliefState:
    """Current signed conviction per ticker, in the interval [-1, 1]."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._beliefs: dict[str, float] = {}
        self._history: list[dict] = []

    def update_from_own_signal(self, ticker, signal, date: str | None = None) -> None:
        self._beliefs[str(ticker)] = clamp_signal(float(signal))
        self._record(str(ticker), self._beliefs[str(ticker)], "own", date)

    def update_from_messages(
        self,
        messages,
        reputation_tracker,
        method: str = "reputation_weighted",
        current_date: str | None = None,
    ) -> None:
        grouped: dict[str, list] = {}
        for msg in messages:
            for ticker in msg.tickers:
                grouped.setdefault(str(ticker), []).append(msg)

        for ticker, ticker_messages in grouped.items():
            own_signal = self.get_belief(ticker)
            if method != "reputation_weighted":
                raise ValueError(f"unknown belief aggregation method: {method}")
            belief = reputation_weighted(own_signal, ticker_messages, reputation_tracker, current_date=current_date)
            self._beliefs[ticker] = belief
            self._record(ticker, belief, method, current_date)

    def get_belief(self, ticker) -> float:
        return float(self._beliefs.get(str(ticker), 0.0))

    def to_weights(self, tickers, max_weight=0.35) -> dict:
        positive = {ticker: max(0.0, self.get_belief(ticker)) for ticker in tickers}
        total = sum(positive.values())
        if total <= 1e-12:
            return {ticker: 0.0 for ticker in tickers}
        weights = {ticker: min(max_weight, value / total) for ticker, value in positive.items()}
        weight_sum = sum(weights.values())
        if weight_sum > 0.95:
            weights = {ticker: value / weight_sum * 0.95 for ticker, value in weights.items()}
        return weights

    def snapshot(self, date: str) -> None:
        for ticker, belief in self._beliefs.items():
            self._record(ticker, belief, "snapshot", date)

    def to_dataframe(self) -> pd.DataFrame:
        columns = ["date", "agent", "ticker", "belief", "source"]
        return pd.DataFrame(self._history, columns=columns)

    def _record(self, ticker: str, belief: float, source: str, date: str | None) -> None:
        if date is None:
            return
        self._history.append(
            {
                "date": str(pd.Timestamp(date))[:10],
                "agent": self.agent_id,
                "ticker": ticker,
                "belief": float(belief),
                "source": source,
            }
        )

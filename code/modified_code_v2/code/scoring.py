from __future__ import annotations

import ast
import math

import numpy as np
import pandas as pd


def brier_score(probability: float, outcome: float) -> float:
    probability = clamp_probability(probability)
    outcome = 1.0 if float(outcome) > 0.0 else 0.0
    return float((probability - outcome) ** 2)


def log_score(probability: float, outcome: float, eps: float = 1e-6) -> float:
    probability = min(max(float(probability), eps), 1.0 - eps)
    outcome = 1.0 if float(outcome) > 0.0 else 0.0
    return float(outcome * math.log(probability) + (1.0 - outcome) * math.log(1.0 - probability))


def update_reputation(old_rep: float, score: float, rho: float = 0.1) -> float:
    return float((1.0 - rho) * float(old_rep) + rho * float(score))


def build_forecast_scores(messages: pd.DataFrame, prices: pd.DataFrame, horizon_days: int = 20) -> pd.DataFrame:
    columns = [
        "message_id",
        "date",
        "sender_id",
        "ticker",
        "direction",
        "probability_up",
        "confidence",
        "horizon_days",
        "realized_return",
        "outcome_up",
        "brier_score",
        "log_score",
        "proper_score",
    ]
    if messages.empty or prices.empty:
        return pd.DataFrame(columns=columns)

    price_lookup = _price_lookup(prices)
    rows = []
    for _, msg in messages.iterrows():
        tickers = parse_list(msg.get("tickers", []))
        if not tickers:
            continue
        date = str(pd.Timestamp(msg.get("timestamp", msg.get("date", ""))))[:10]
        direction = str(msg.get("direction", "neutral"))
        confidence = clamp_probability(msg.get("confidence", 0.5))
        probability_up = direction_to_probability(direction, confidence)
        for ticker in tickers:
            realized_return = forward_return(price_lookup, ticker, date, horizon_days)
            if realized_return is None:
                continue
            outcome_up = 1.0 if realized_return > 0.0 else 0.0
            brier = brier_score(probability_up, outcome_up)
            log = log_score(probability_up, outcome_up)
            rows.append(
                {
                    "message_id": msg.get("message_id", msg.get("event_id", "")),
                    "date": date,
                    "sender_id": msg.get("sender_id", msg.get("agent", "")),
                    "ticker": ticker,
                    "direction": direction,
                    "probability_up": probability_up,
                    "confidence": confidence,
                    "horizon_days": int(horizon_days),
                    "realized_return": realized_return,
                    "outcome_up": outcome_up,
                    "brier_score": brier,
                    "log_score": log,
                    "proper_score": 1.0 - brier,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def scoring_summary(forecast_scores: pd.DataFrame) -> pd.DataFrame:
    columns = ["sender_id", "forecast_count", "mean_brier", "mean_log_score", "proper_reputation"]
    if forecast_scores.empty:
        return pd.DataFrame(columns=columns)
    summary = (
        forecast_scores.groupby("sender_id")
        .agg(
            forecast_count=("message_id", "count"),
            mean_brier=("brier_score", "mean"),
            mean_log_score=("log_score", "mean"),
            proper_reputation=("proper_score", "mean"),
        )
        .reset_index()
    )
    return summary[columns]


def direction_to_probability(direction: str, confidence: float) -> float:
    confidence = clamp_probability(confidence)
    if direction == "bullish":
        return 0.5 + 0.5 * confidence
    if direction == "bearish":
        return 0.5 - 0.5 * confidence
    return 0.5


def clamp_probability(value) -> float:
    try:
        return float(min(1.0, max(0.0, float(value))))
    except (TypeError, ValueError):
        return 0.5


def parse_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    text = str(value or "").strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item)]
        except (SyntaxError, ValueError):
            pass
    return [part.strip() for part in text.split(",") if part.strip()]


def _price_lookup(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    work = prices.copy()
    work["date"] = pd.to_datetime(work["date"])
    return {
        ticker: group.sort_values("date").reset_index(drop=True)
        for ticker, group in work.groupby("ticker")
    }


def forward_return(price_lookup: dict[str, pd.DataFrame], ticker: str, date: str, horizon_days: int) -> float | None:
    group = price_lookup.get(str(ticker))
    if group is None or group.empty:
        return None
    date_ts = pd.Timestamp(date)
    candidates = group[group["date"] >= date_ts]
    if candidates.empty:
        return None
    start_index = int(candidates.index[0])
    end_index = start_index + int(horizon_days)
    if end_index >= len(group):
        return None
    start_price = float(group.loc[start_index, "close"])
    end_price = float(group.loc[end_index, "close"])
    if start_price <= 0:
        return None
    return float(end_price / start_price - 1.0)

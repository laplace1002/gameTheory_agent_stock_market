from __future__ import annotations

import numpy as np
import pandas as pd

from aggregation import black_litterman
from scoring import direction_to_probability, parse_list


def build_black_litterman_outputs(
    messages: pd.DataFrame,
    prices: pd.DataFrame,
    reputation_scores: pd.DataFrame,
    horizon_days: int = 20,
) -> dict[str, pd.DataFrame]:
    views = extract_agent_views(messages, prices, reputation_scores, horizon_days=horizon_days)
    weights = black_litterman_weights_from_views(views, prices)
    return {"agent_views": views, "bl_agent_view_weights": weights}


def extract_agent_views(
    messages: pd.DataFrame,
    prices: pd.DataFrame,
    reputation_scores: pd.DataFrame,
    horizon_days: int = 20,
) -> pd.DataFrame:
    columns = [
        "date",
        "agent",
        "ticker",
        "view_type",
        "direction",
        "probability_up",
        "expected_return",
        "confidence",
        "reputation",
        "omega",
        "source_message_id",
    ]
    if messages.empty or prices.empty:
        return pd.DataFrame(columns=columns)

    vol = _annualized_volatility(prices)
    rep = _reputation_lookup(reputation_scores)
    rows = []
    for _, msg in messages.iterrows():
        tickers = parse_list(msg.get("tickers", []))
        if not tickers:
            continue
        agent = str(msg.get("sender_id", msg.get("agent", "")))
        direction = str(msg.get("direction", "neutral"))
        confidence = float(max(0.0, min(1.0, float(msg.get("confidence", 0.5) or 0.5))))
        probability_up = direction_to_probability(direction, confidence)
        sign = 1.0 if direction == "bullish" else (-1.0 if direction == "bearish" else 0.0)
        reputation = float(rep.get(agent, 0.5))
        for ticker in tickers:
            ticker_vol = float(vol.get(ticker, 0.18))
            expected_return = sign * confidence * ticker_vol * np.sqrt(horizon_days / 252.0)
            omega = max(1e-5, (ticker_vol**2) * horizon_days / 252.0 / max(0.05, reputation * confidence))
            rows.append(
                {
                    "date": str(pd.Timestamp(msg.get("timestamp", msg.get("date", ""))))[:10],
                    "agent": agent,
                    "ticker": ticker,
                    "view_type": "absolute",
                    "direction": direction,
                    "probability_up": probability_up,
                    "expected_return": float(expected_return),
                    "confidence": confidence,
                    "reputation": reputation,
                    "omega": float(omega),
                    "source_message_id": msg.get("message_id", msg.get("event_id", "")),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def black_litterman_weights_from_views(views: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    columns = ["ticker", "posterior_return", "weight", "view_count", "avg_confidence"]
    if views.empty or prices.empty:
        return pd.DataFrame(columns=columns)

    tickers = sorted(prices["ticker"].astype(str).unique())
    returns = prices.copy()
    returns["date"] = pd.to_datetime(returns["date"])
    wide = returns.pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index().pct_change().dropna()
    wide = wide.reindex(columns=tickers).fillna(0.0)
    if wide.empty:
        return pd.DataFrame(columns=columns)

    sigma = wide.cov().to_numpy(dtype=float) * 252.0
    sigma = sigma + np.eye(len(tickers)) * 1e-6
    pi = wide.mean().to_numpy(dtype=float) * 252.0
    ticker_index = {ticker: idx for idx, ticker in enumerate(tickers)}

    rows = []
    q = []
    omega_diag = []
    for _, view in views.iterrows():
        ticker = str(view.get("ticker", ""))
        if ticker not in ticker_index:
            continue
        p = np.zeros(len(tickers))
        p[ticker_index[ticker]] = 1.0
        rows.append(p)
        q.append(float(view.get("expected_return", 0.0)) * 252.0 / 20.0)
        omega_diag.append(float(view.get("omega", 0.05)))
    if not rows:
        return pd.DataFrame(columns=columns)

    P = np.vstack(rows)
    omega = np.diag(np.maximum(omega_diag, 1e-5))
    posterior = black_litterman(pi, sigma, P, q, omega)
    raw = np.maximum(posterior, 0.0)
    if raw.sum() <= 1e-12:
        raw = np.ones_like(raw)
    weights = raw / raw.sum()
    summary = views.groupby("ticker").agg(view_count=("ticker", "count"), avg_confidence=("confidence", "mean"))
    out = []
    for ticker, mu, weight in zip(tickers, posterior, weights):
        out.append(
            {
                "ticker": ticker,
                "posterior_return": float(mu),
                "weight": float(weight),
                "view_count": int(summary.loc[ticker, "view_count"]) if ticker in summary.index else 0,
                "avg_confidence": float(summary.loc[ticker, "avg_confidence"]) if ticker in summary.index else 0.0,
            }
        )
    return pd.DataFrame(out, columns=columns)


def _annualized_volatility(prices: pd.DataFrame) -> dict[str, float]:
    work = prices.copy()
    work["date"] = pd.to_datetime(work["date"])
    vols = {}
    for ticker, group in work.sort_values("date").groupby("ticker"):
        returns = group["close"].pct_change().dropna()
        vols[str(ticker)] = float(returns.std() * np.sqrt(252.0)) if len(returns) else 0.18
    return vols


def _reputation_lookup(reputation_scores: pd.DataFrame) -> dict[str, float]:
    if reputation_scores.empty or "sender_id" not in reputation_scores.columns:
        return {}
    return reputation_scores.set_index("sender_id")["reputation"].astype(float).to_dict()

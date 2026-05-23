from __future__ import annotations

import json

import numpy as np
import pandas as pd


TRAINING_GRIDS = {
    "MomentumAgent": [
        {"short_lookback": 10, "long_lookback": 40, "short_weight": 0.75},
        {"short_lookback": 20, "long_lookback": 60, "short_weight": 0.70},
        {"short_lookback": 30, "long_lookback": 90, "short_weight": 0.65},
    ],
    "MeanReversionAgent": [
        {"lookback": 10},
        {"lookback": 20},
        {"lookback": 40},
    ],
    "LowVolatilityAgent": [
        {"vol_lookback": 20, "trend_lookback": 40},
        {"vol_lookback": 40, "trend_lookback": 40},
        {"vol_lookback": 60, "trend_lookback": 60},
    ],
    "DrawdownBuyerAgent": [
        {"drawdown_lookback": 60, "rebound_lookback": 5},
        {"drawdown_lookback": 120, "rebound_lookback": 5},
        {"drawdown_lookback": 180, "rebound_lookback": 10},
    ],
}


def walk_forward_train_agents(prices: pd.DataFrame, agents: list, train_days: int = 80) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame(columns=training_columns())
    work = prices.copy()
    work["date"] = pd.to_datetime(work["date"])
    train_dates = sorted(work["date"].unique())[: max(20, int(train_days))]
    train_prices = work[work["date"].isin(train_dates)].copy()
    rows = []
    for agent in agents:
        rows.extend(train_agent_tree(agent, train_prices))
    return pd.DataFrame(rows, columns=training_columns())


def train_agent_tree(agent, train_prices: pd.DataFrame) -> list[dict]:
    rows = []
    strategy = getattr(agent, "own_strategy", agent)
    rows.extend(train_single_strategy(strategy, display_agent=getattr(agent, "name", strategy.__class__.__name__), train_prices=train_prices))
    for member in getattr(strategy, "members", []) or []:
        rows.extend(train_single_strategy(member, display_agent=getattr(agent, "name", strategy.__class__.__name__), train_prices=train_prices))
    return rows


def train_single_strategy(strategy, display_agent: str, train_prices: pd.DataFrame) -> list[dict]:
    strategy_name = getattr(strategy, "name", strategy.__class__.__name__)
    grid = TRAINING_GRIDS.get(strategy_name)
    if not grid:
        return [
            {
                "agent": display_agent,
                "strategy": strategy_name,
                "fixed_identity": strategy_name,
                "train_window_start": str(train_prices["date"].min())[:10] if not train_prices.empty else "",
                "train_window_end": str(train_prices["date"].max())[:10] if not train_prices.empty else "",
                "candidate_params_json": "{}",
                "selected_params_json": "{}",
                "validation_score": np.nan,
                "param_hash": "",
                "status": "not_trainable",
            }
        ]

    scored = []
    for params in grid:
        score = score_strategy_params(strategy_name, params, train_prices)
        scored.append((score, params))
    best_score, best_params = max(scored, key=lambda item: item[0])
    apply_params(strategy, best_params)
    param_hash = stable_param_hash(best_params)
    return [
        {
            "agent": display_agent,
            "strategy": strategy_name,
            "fixed_identity": strategy_name,
            "train_window_start": str(train_prices["date"].min())[:10],
            "train_window_end": str(train_prices["date"].max())[:10],
            "candidate_params_json": json.dumps([params for _, params in scored], ensure_ascii=False, sort_keys=True),
            "selected_params_json": json.dumps(best_params, ensure_ascii=False, sort_keys=True),
            "validation_score": float(best_score),
            "param_hash": param_hash,
            "status": "trained_frozen",
        }
    ]


def score_strategy_params(strategy_name: str, params: dict, prices: pd.DataFrame) -> float:
    if prices.empty:
        return -1e9
    scores = []
    for _, group in prices.sort_values("date").groupby("ticker"):
        close = group["close"].astype(float)
        if len(close) < 15:
            continue
        if strategy_name == "MomentumAgent":
            short = int(params["short_lookback"])
            long = int(params["long_lookback"])
            if len(close) <= long + 2:
                continue
            signal = close.iloc[-2] / close.iloc[-short - 2] - 1.0
            future = close.iloc[-1] / close.iloc[-2] - 1.0
            scores.append(signal * future)
        elif strategy_name == "MeanReversionAgent":
            lookback = int(params["lookback"])
            if len(close) <= lookback + 2:
                continue
            ma = close.rolling(lookback).mean().iloc[-2]
            signal = (ma - close.iloc[-2]) / max(1e-9, ma)
            future = close.iloc[-1] / close.iloc[-2] - 1.0
            scores.append(signal * future)
        elif strategy_name == "LowVolatilityAgent":
            vol_lookback = int(params["vol_lookback"])
            trend_lookback = int(params["trend_lookback"])
            if len(close) <= max(vol_lookback, trend_lookback) + 2:
                continue
            returns = close.pct_change()
            vol = returns.tail(vol_lookback).std()
            trend = close.iloc[-2] / close.iloc[-trend_lookback - 2] - 1.0
            future = close.iloc[-1] / close.iloc[-2] - 1.0
            scores.append(max(0.0, trend) * future / max(1e-6, vol))
        elif strategy_name == "DrawdownBuyerAgent":
            dd_lookback = int(params["drawdown_lookback"])
            rebound_lookback = int(params["rebound_lookback"])
            if len(close) <= dd_lookback + 2:
                continue
            high = close.rolling(dd_lookback).max().iloc[-2]
            drawdown = (high - close.iloc[-2]) / max(1e-9, high)
            rebound = close.iloc[-2] / close.iloc[-rebound_lookback - 2] - 1.0
            future = close.iloc[-1] / close.iloc[-2] - 1.0
            scores.append(max(0.0, drawdown) * (1.0 + max(0.0, rebound)) * future)
    if not scores:
        return -1e9
    return float(np.mean(scores))


def apply_params(strategy, params: dict) -> None:
    for key, value in params.items():
        if hasattr(strategy, key):
            setattr(strategy, key, value)


def stable_param_hash(params: dict) -> str:
    import hashlib

    raw = json.dumps(params, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def training_columns() -> list[str]:
    return [
        "agent",
        "strategy",
        "fixed_identity",
        "train_window_start",
        "train_window_end",
        "candidate_params_json",
        "selected_params_json",
        "validation_score",
        "param_hash",
        "status",
    ]

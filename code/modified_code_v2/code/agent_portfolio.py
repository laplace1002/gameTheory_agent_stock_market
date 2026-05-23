from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from portfolio import performance_metrics


class AgentPortfolioManager:
    name = "base"

    def allocate(self, date, agent_metrics, agent_returns, corr_matrix, previous_weights):
        raise NotImplementedError

    def _normalize(self, weights: dict[str, float], agents: list[str]) -> dict[str, float]:
        cleaned = {agent: max(0.0, float(weights.get(agent, 0.0))) for agent in agents}
        total = sum(cleaned.values())
        if total <= 1e-12:
            return equal_weights(agents)
        return {agent: value / total for agent, value in cleaned.items()}


class EqualAgentManager(AgentPortfolioManager):
    name = "equal_agent_portfolio"

    def allocate(self, date, agent_metrics, agent_returns, corr_matrix, previous_weights):
        return equal_weights(list(agent_returns.columns))


@dataclass
class HedgeAgentManager(AgentPortfolioManager):
    learning_rate: float = 2.0
    min_weight: float = 0.02

    name = "hedge_agent_portfolio"

    def allocate(self, date, agent_metrics, agent_returns, corr_matrix, previous_weights):
        agents = list(agent_returns.columns)
        if agent_returns.empty:
            return equal_weights(agents)
        cumulative_returns = (1.0 + agent_returns.fillna(0.0)).prod() - 1.0
        drawdown_penalty = _drawdown_penalty(agent_returns)
        scores = cumulative_returns - 0.35 * drawdown_penalty
        raw = {
            agent: math.exp(self.learning_rate * float(scores.get(agent, 0.0)))
            for agent in agents
        }
        weights = self._normalize(raw, agents)
        floored = {agent: max(self.min_weight, weight) for agent, weight in weights.items()}
        return self._normalize(floored, agents)


def build_agent_portfolio_outputs(
    equity: pd.DataFrame,
    initial_cash: float = 100000.0,
    rebalance_every: int = 5,
) -> dict[str, pd.DataFrame]:
    agent_return_history = agent_returns_from_equity(equity)
    corr = agent_return_correlation(agent_return_history)

    managers = [EqualAgentManager(), HedgeAgentManager()]
    manager_frames = []
    weight_frames = []
    for manager in managers:
        manager_curve, weight_history = simulate_manager(
            agent_return_history=agent_return_history,
            manager=manager,
            initial_cash=initial_cash,
            rebalance_every=rebalance_every,
        )
        manager_frames.append(manager_curve)
        weight_frames.append(weight_history)

    manager_equity = pd.concat(manager_frames, ignore_index=True) if manager_frames else _empty_manager_equity()
    meta_weights = pd.concat(weight_frames, ignore_index=True) if weight_frames else _empty_meta_weights()
    comparison = experiment_comparison(equity, manager_equity, meta_weights, corr)
    return {
        "agent_equity_curve": equity.copy(),
        "agent_return_history": agent_return_history,
        "agent_return_correlation": corr,
        "manager_equity_curve": manager_equity,
        "meta_weight_history": meta_weights,
        "experiment_comparison": comparison,
    }


def agent_returns_from_equity(equity: pd.DataFrame) -> pd.DataFrame:
    if equity.empty:
        return pd.DataFrame(columns=["date", "agent", "return"])
    work = equity.copy()
    work["date"] = pd.to_datetime(work["date"])
    work = work.sort_values(["agent", "date"])
    work["return"] = work.groupby("agent")["equity"].pct_change().fillna(0.0)
    return work[["date", "agent", "return"]].assign(date=lambda df: df["date"].dt.strftime("%Y-%m-%d"))


def agent_return_correlation(agent_return_history: pd.DataFrame) -> pd.DataFrame:
    if agent_return_history.empty:
        return pd.DataFrame()
    wide = _returns_wide(agent_return_history)
    corr = wide.corr().fillna(0.0).copy()
    values = corr.to_numpy(copy=True)
    np.fill_diagonal(values, 1.0)
    corr = pd.DataFrame(values, index=corr.index, columns=corr.columns)
    corr.index.name = "agent"
    return corr.reset_index()


def simulate_manager(
    agent_return_history: pd.DataFrame,
    manager: AgentPortfolioManager,
    initial_cash: float,
    rebalance_every: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = _returns_wide(agent_return_history)
    agents = list(returns.columns)
    if returns.empty or not agents:
        return _empty_manager_equity(), _empty_meta_weights()

    weights = equal_weights(agents)
    equity_value = float(initial_cash)
    equity_rows = []
    weight_rows = []
    for step, (date, row) in enumerate(returns.iterrows()):
        day_return = float(sum(weights.get(agent, 0.0) * row.get(agent, 0.0) for agent in agents))
        equity_value *= 1.0 + day_return
        equity_rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "manager": manager.name,
                "equity": equity_value,
                "daily_return": day_return,
            }
        )

        history = returns.iloc[: step + 1]
        if step == 0 or (step > 0 and step % max(1, rebalance_every) == 0):
            metrics = _agent_metrics_from_returns(history, initial_cash)
            corr = history.corr().fillna(0.0)
            weights = manager.allocate(date, metrics, history, corr, weights)
            weights = manager._normalize(weights, agents)

        for agent, weight in weights.items():
            weight_rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "manager": manager.name,
                    "agent": agent,
                    "weight": float(weight),
                }
            )
    return pd.DataFrame(equity_rows), pd.DataFrame(weight_rows)


def experiment_comparison(
    agent_equity: pd.DataFrame,
    manager_equity: pd.DataFrame,
    meta_weights: pd.DataFrame,
    corr: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    if not agent_equity.empty:
        agent_metrics = performance_metrics(agent_equity).rename(columns={"agent": "portfolio"})
        agent_metrics.insert(0, "experiment_type", "single_agent")
        frames.append(agent_metrics)
    if not manager_equity.empty:
        manager_as_equity = manager_equity.rename(columns={"manager": "agent"})[["date", "agent", "equity"]]
        manager_metrics = performance_metrics(manager_as_equity).rename(columns={"agent": "portfolio"})
        manager_metrics.insert(0, "experiment_type", "agent_portfolio")
        frames.append(manager_metrics)
    if not frames:
        return pd.DataFrame()

    comparison = pd.concat(frames, ignore_index=True)
    avg_corr = _average_off_diagonal_corr(corr)
    hhi = _hhi_by_manager(meta_weights)
    comparison["average_agent_correlation"] = avg_corr
    comparison["final_weight_hhi"] = comparison["portfolio"].map(hhi).fillna(np.nan)

    best_agent_return = comparison.loc[comparison["experiment_type"] == "single_agent", "total_return"].max()
    comparison["regret_to_best_agent"] = best_agent_return - comparison["total_return"]
    return comparison


def equal_weights(agents: list[str]) -> dict[str, float]:
    if not agents:
        return {}
    weight = 1.0 / len(agents)
    return {agent: weight for agent in agents}


def _returns_wide(agent_return_history: pd.DataFrame) -> pd.DataFrame:
    if agent_return_history.empty:
        return pd.DataFrame()
    work = agent_return_history.copy()
    work["date"] = pd.to_datetime(work["date"])
    wide = work.pivot_table(index="date", columns="agent", values="return", aggfunc="last").sort_index()
    return wide.fillna(0.0)


def _drawdown_penalty(returns: pd.DataFrame) -> pd.Series:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return drawdown.min().abs().fillna(0.0)


def _agent_metrics_from_returns(returns: pd.DataFrame, initial_cash: float) -> pd.DataFrame:
    rows = []
    for agent in returns.columns:
        values = (1.0 + returns[agent].fillna(0.0)).cumprod() * initial_cash
        frame = pd.DataFrame({"date": returns.index, "agent": agent, "equity": values})
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    return performance_metrics(pd.concat(rows, ignore_index=True))


def _average_off_diagonal_corr(corr: pd.DataFrame) -> float:
    if corr.empty or "agent" not in corr.columns:
        return np.nan
    matrix = corr.set_index("agent")
    if matrix.shape[0] <= 1:
        return np.nan
    values = matrix.to_numpy(dtype=float)
    mask = ~np.eye(values.shape[0], dtype=bool)
    return float(np.nanmean(values[mask]))


def _hhi_by_manager(meta_weights: pd.DataFrame) -> dict[str, float]:
    if meta_weights.empty:
        return {}
    latest = meta_weights.copy()
    latest["date"] = pd.to_datetime(latest["date"])
    latest = latest.sort_values("date").groupby(["manager", "agent"], as_index=False).tail(1)
    return latest.groupby("manager")["weight"].apply(lambda values: float(np.square(values).sum())).to_dict()


def _empty_manager_equity() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "manager", "equity", "daily_return"])


def _empty_meta_weights() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "manager", "agent", "weight"])

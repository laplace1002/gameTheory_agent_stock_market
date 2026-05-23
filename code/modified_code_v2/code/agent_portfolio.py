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

    def loss_rows(self, date, agent_returns, previous_weights) -> list[dict]:
        return []

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
    drawdown_penalty: float = 0.35
    cvar_penalty: float = 0.20

    name = "hedge_agent_portfolio"

    def allocate(self, date, agent_metrics, agent_returns, corr_matrix, previous_weights):
        agents = list(agent_returns.columns)
        if agent_returns.empty:
            return equal_weights(agents)
        losses = _latest_agent_losses(agent_returns, self.drawdown_penalty, self.cvar_penalty)
        raw = {}
        for agent in agents:
            prior = float(previous_weights.get(agent, 1.0 / len(agents))) if previous_weights else 1.0 / len(agents)
            raw[agent] = max(1e-12, prior) * math.exp(-self.learning_rate * float(losses.get(agent, 0.0)))
        weights = self._normalize(raw, agents)
        floored = {agent: max(self.min_weight, weight) for agent, weight in weights.items()}
        return self._normalize(floored, agents)

    def loss_rows(self, date, agent_returns, previous_weights) -> list[dict]:
        losses = _latest_agent_losses(agent_returns, self.drawdown_penalty, self.cvar_penalty, detailed=True)
        rows = []
        for agent, values in losses.items():
            rows.append(
                {
                    "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                    "manager": self.name,
                    "agent": agent,
                    "previous_weight": float(previous_weights.get(agent, 0.0)) if previous_weights else 0.0,
                    **values,
                }
            )
        return rows


@dataclass
class CorrelationAwareAgentManager(AgentPortfolioManager):
    learning_rate: float = 1.25
    corr_penalty: float = 0.55
    min_weight: float = 0.01

    name = "correlation_aware_agent_portfolio"

    def allocate(self, date, agent_metrics, agent_returns, corr_matrix, previous_weights):
        agents = list(agent_returns.columns)
        if not agents:
            return {}
        mean_returns = agent_returns.tail(60).mean().fillna(0.0)
        avg_corr = _average_corr_by_agent(corr_matrix, agents)
        scores = {agent: float(mean_returns.get(agent, 0.0)) * 252.0 - self.corr_penalty * avg_corr.get(agent, 0.0) for agent in agents}
        raw = {agent: math.exp(self.learning_rate * scores[agent]) for agent in agents}
        weights = self._normalize(raw, agents)
        return self._normalize({agent: max(self.min_weight, weight) for agent, weight in weights.items()}, agents)


@dataclass
class DrawdownConstrainedAgentManager(HedgeAgentManager):
    max_agent_weight: float = 0.25
    cvar_limit: float = 0.025

    name = "drawdown_constrained_agent_portfolio"

    def allocate(self, date, agent_metrics, agent_returns, corr_matrix, previous_weights):
        agents = list(agent_returns.columns)
        weights = super().allocate(date, agent_metrics, agent_returns, corr_matrix, previous_weights)
        cvar = _rolling_cvar(agent_returns)
        capped = {}
        for agent in agents:
            cap = self.max_agent_weight
            if cvar.get(agent, 0.0) > self.cvar_limit:
                cap = min(cap, 0.12)
            capped[agent] = min(cap, weights.get(agent, 0.0))
        return self._normalize(capped, agents)


def build_agent_portfolio_outputs(
    equity: pd.DataFrame,
    initial_cash: float = 100000.0,
    rebalance_every: int = 5,
) -> dict[str, pd.DataFrame]:
    agent_return_history = agent_returns_from_equity(equity)
    corr = agent_return_correlation(agent_return_history)

    managers = [
        EqualAgentManager(),
        HedgeAgentManager(),
        CorrelationAwareAgentManager(),
        DrawdownConstrainedAgentManager(),
    ]
    manager_frames = []
    weight_frames = []
    loss_frames = []
    for manager in managers:
        manager_curve, weight_history, loss_history = simulate_manager(
            agent_return_history=agent_return_history,
            manager=manager,
            initial_cash=initial_cash,
            rebalance_every=rebalance_every,
        )
        manager_frames.append(manager_curve)
        weight_frames.append(weight_history)
        loss_frames.append(loss_history)

    manager_equity = pd.concat(manager_frames, ignore_index=True) if manager_frames else _empty_manager_equity()
    meta_weights = pd.concat(weight_frames, ignore_index=True) if weight_frames else _empty_meta_weights()
    loss_history = pd.concat(loss_frames, ignore_index=True) if loss_frames else _empty_manager_loss_history()
    comparison = experiment_comparison(equity, manager_equity, meta_weights, corr)
    return {
        "agent_equity_curve": equity.copy(),
        "agent_return_history": agent_return_history,
        "agent_return_correlation": corr,
        "manager_equity_curve": manager_equity,
        "meta_weight_history": meta_weights,
        "manager_loss_history": loss_history,
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    returns = _returns_wide(agent_return_history)
    agents = list(returns.columns)
    if returns.empty or not agents:
        return _empty_manager_equity(), _empty_meta_weights(), _empty_manager_loss_history()

    weights = equal_weights(agents)
    equity_value = float(initial_cash)
    equity_rows = []
    weight_rows = []
    loss_rows = []
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
            loss_rows.extend(manager.loss_rows(date, history, weights))
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
    return pd.DataFrame(equity_rows), pd.DataFrame(weight_rows), pd.DataFrame(loss_rows)


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


def _latest_agent_losses(
    returns: pd.DataFrame,
    drawdown_penalty: float,
    cvar_penalty: float,
    detailed: bool = False,
) -> dict:
    if returns.empty:
        return {}
    latest = returns.fillna(0.0).iloc[-1]
    drawdowns = _drawdown_penalty(returns)
    cvar = _rolling_cvar(returns)
    out = {}
    for agent in returns.columns:
        ret = float(latest.get(agent, 0.0))
        log_loss = -math.log(max(1e-6, 1.0 + ret))
        dd_component = float(drawdown_penalty * drawdowns.get(agent, 0.0))
        cvar_component = float(cvar_penalty * cvar.get(agent, 0.0))
        total = log_loss + dd_component + cvar_component
        if detailed:
            out[agent] = {
                "agent_return": ret,
                "log_loss": log_loss,
                "drawdown_penalty": dd_component,
                "cvar_penalty": cvar_component,
                "total_loss": total,
            }
        else:
            out[agent] = total
    return out


def _rolling_cvar(returns: pd.DataFrame, alpha: float = 0.95, window: int = 60) -> dict[str, float]:
    if returns.empty:
        return {}
    tail = returns.fillna(0.0).tail(window)
    out = {}
    quantile = 1.0 - float(alpha)
    for agent in tail.columns:
        losses = -tail[agent]
        cutoff = losses.quantile(1.0 - quantile) if len(losses) > 1 else losses.iloc[-1]
        tail_losses = losses[losses >= cutoff]
        out[agent] = float(tail_losses.mean()) if not tail_losses.empty else 0.0
    return out


def _average_corr_by_agent(corr_matrix: pd.DataFrame, agents: list[str]) -> dict[str, float]:
    if corr_matrix.empty:
        return {agent: 0.0 for agent in agents}
    corr = corr_matrix.reindex(index=agents, columns=agents).fillna(0.0).abs()
    out = {}
    for agent in agents:
        others = [other for other in agents if other != agent]
        out[agent] = float(corr.loc[agent, others].mean()) if others else 0.0
    return out


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


def _empty_manager_loss_history() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "manager",
            "agent",
            "previous_weight",
            "agent_return",
            "log_loss",
            "drawdown_penalty",
            "cvar_penalty",
            "total_loss",
        ]
    )

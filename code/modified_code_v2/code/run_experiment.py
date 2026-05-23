from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ai_agent_stock_market_matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from aggregation import hedge_weights
from agent_portfolio import build_agent_portfolio_outputs
from agents import (
    CommunicatingAgent,
    CommitteeTeamAgent,
    ContrarianAgent,
    DrawdownBuyerAgent,
    DynamicTeamAgent,
    FreeRiderAgent,
    LowVolatilityAgent,
    MeanReversionAgent,
    MomentumAgent,
    PersuaderAgent,
    RandomAgent,
    SocialGraphAgent,
    TruthfulReporterAgent,
)
from data_loader import load_prices
from message import MessageBoard
from portfolio import performance_metrics
from reputation import ReputationTracker
from social_graph import SocialGraph
from strategy_spec import strategy_spec_for_agent


EXPERIMENTS = {
    "baseline": {"channels": []},
    "public_only": {"channels": ["public"]},
    "private_only": {"channels": ["private"]},
    "moments_only": {"channels": ["moments"]},
    "full_social": {"channels": ["public", "private", "moments"]},
}

DEFAULT_SCENARIOS = {
    "baseline_isolated": {
        "topology": "isolated",
        "friendships": [],
        "groups": {},
        "friend_request_rules": {"strategy_similarity_threshold": 0.45, "max_friends": 3},
    },
    "chain_influence": {
        "topology": "chain",
        "friendships": [
            ["MomentumAgent", "TruthfulReporterAgent"],
            ["MeanReversionAgent", "PersuaderAgent"],
            ["LowVolatilityAgent", "FreeRiderAgent"],
        ],
        "groups": {
            "Trend Lab": ["MomentumAgent", "TruthfulReporterAgent", "DynamicTeamAgent"],
            "Risk Desk": ["LowVolatilityAgent", "FreeRiderAgent", "CommitteeTeamAgent"],
        },
        "friend_request_rules": {"strategy_similarity_threshold": 0.45, "max_friends": 4},
    },
    "dense_market": {
        "topology": "dense",
        "friendships": "all",
        "groups": {
            "Market Hall": [
                "MomentumAgent",
                "MeanReversionAgent",
                "LowVolatilityAgent",
                "DrawdownBuyerAgent",
                "RandomAgent",
                "CommitteeTeamAgent",
                "DynamicTeamAgent",
                "TruthfulReporterAgent",
                "PersuaderAgent",
                "FreeRiderAgent",
                "ContrarianAgent",
                "SocialGraphAgent",
            ]
        },
        "friend_request_rules": {"strategy_similarity_threshold": 0.20, "max_friends": 12},
    },
    "echo_chambers": {
        "topology": "echo_chambers",
        "friendships": [
            ["MomentumAgent", "TruthfulReporterAgent"],
            ["MeanReversionAgent", "PersuaderAgent"],
            ["MeanReversionAgent", "ContrarianAgent"],
            ["LowVolatilityAgent", "FreeRiderAgent"],
            ["CommitteeTeamAgent", "DynamicTeamAgent"],
        ],
        "groups": {
            "Momentum 圈": ["MomentumAgent", "TruthfulReporterAgent", "DynamicTeamAgent"],
            "Reversion 圈": ["MeanReversionAgent", "PersuaderAgent", "ContrarianAgent"],
            "Risk 圈": ["LowVolatilityAgent", "FreeRiderAgent", "CommitteeTeamAgent"],
        },
        "friend_request_rules": {"strategy_similarity_threshold": 0.60, "max_friends": 5},
    },
    "core_periphery": {
        "topology": "core_periphery",
        "core": ["SocialGraphAgent", "DynamicTeamAgent", "CommitteeTeamAgent"],
        "friendships": [
            ["SocialGraphAgent", "DynamicTeamAgent"],
            ["SocialGraphAgent", "CommitteeTeamAgent"],
            ["DynamicTeamAgent", "CommitteeTeamAgent"],
        ],
        "groups": {
            "Core Desk": ["SocialGraphAgent", "DynamicTeamAgent", "CommitteeTeamAgent"],
            "Outer Desk": ["MomentumAgent", "MeanReversionAgent", "LowVolatilityAgent", "DrawdownBuyerAgent", "RandomAgent"],
        },
        "friend_request_rules": {"strategy_similarity_threshold": 0.35, "max_friends": 6},
    },
}


class EventRecorder:
    """Single ordered event stream shared by Trading Hall and ChatLab."""

    def __init__(self):
        self.rows: list[dict] = []
        self._seq = 0

    def add(
        self,
        date,
        event_type: str,
        agent: str = "",
        counterparty: str = "",
        channel: str = "",
        source: str = "",
        detail: str = "",
        ticker: str = "",
        side: str = "",
        shares=0,
        price=0.0,
        notional=0.0,
        cash=None,
        equity=None,
        pnl=None,
        pnl_pct=None,
        payload=None,
    ) -> int:
        self._seq += 1
        date_str = str(pd.Timestamp(date))[:10]
        event_time = pd.Timestamp(date_str) + pd.Timedelta(hours=9, minutes=30, seconds=self._seq)
        row = {
            "event_id": self._seq,
            "date": date_str,
            "event_time": event_time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": source or self._infer_source(event_type),
            "event_type": event_type,
            "agent": str(agent or ""),
            "counterparty": str(counterparty or ""),
            "channel": str(channel or ""),
            "ticker": str(ticker or ""),
            "side": str(side or ""),
            "shares": int(shares or 0),
            "price": float(price or 0.0),
            "notional": float(notional or 0.0),
            "cash": _safe_optional_float(cash),
            "equity": _safe_optional_float(equity),
            "pnl": _safe_optional_float(pnl),
            "pnl_pct": _safe_optional_float(pnl_pct),
            "detail": str(detail or ""),
            "payload": json.dumps(payload or {}, ensure_ascii=False, default=str),
        }
        self.rows.append(row)
        return self._seq

    def to_dataframe(self) -> pd.DataFrame:
        columns = [
            "event_id",
            "date",
            "event_time",
            "source",
            "event_type",
            "agent",
            "counterparty",
            "channel",
            "ticker",
            "side",
            "shares",
            "price",
            "notional",
            "cash",
            "equity",
            "pnl",
            "pnl_pct",
            "detail",
            "payload",
        ]
        return pd.DataFrame(self.rows, columns=columns)

    def _infer_source(self, event_type: str) -> str:
        if event_type in {"message"}:
            return "chat"
        if event_type.startswith("friend"):
            return "social"
        if event_type in {"trade", "decision", "hold", "pnl_snapshot"}:
            return "portfolio"
        return "system"


def build_agents(social_graph: SocialGraph) -> list:
    """Create a population where every trading strategy also has ChatLab visibility."""
    agents = [
        CommunicatingAgent(MomentumAgent(), name="MomentumAgent", message_probability=0.85),
        CommunicatingAgent(MeanReversionAgent(), name="MeanReversionAgent", message_probability=0.80),
        CommunicatingAgent(LowVolatilityAgent(), name="LowVolatilityAgent", message_probability=0.65),
        CommunicatingAgent(DrawdownBuyerAgent(), name="DrawdownBuyerAgent", message_probability=0.70),
        CommunicatingAgent(RandomAgent(seed=42), name="RandomAgent", message_probability=0.45),
        CommunicatingAgent(CommitteeTeamAgent(), name="CommitteeTeamAgent", message_probability=0.75),
        CommunicatingAgent(DynamicTeamAgent(), name="DynamicTeamAgent", message_probability=0.75),
        TruthfulReporterAgent(MomentumAgent()),
        PersuaderAgent(MeanReversionAgent()),
        FreeRiderAgent(LowVolatilityAgent()),
        ContrarianAgent(DrawdownBuyerAgent()),
    ]
    social_agent = SocialGraphAgent(
        own_strategy=DynamicTeamAgent(),
        social_graph=social_graph,
        all_agents=[],
        alpha=0.55,
    )
    agents.append(social_agent)
    social_agent.all_agents = agents
    return agents


def run(
    prices_path,
    out_dir="outputs",
    initial_cash=100000,
    fee_rate=0.001,
    experiment="full_social",
    rebalance_every=5,
    scenario_name="core_periphery",
):
    if experiment not in EXPERIMENTS:
        known = ", ".join(sorted(EXPERIMENTS))
        raise ValueError(f"unknown experiment '{experiment}', choose one of: {known}")

    scenario = load_scenario(scenario_name)
    out_dir = Path(out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    prices = load_prices(prices_path)
    recorder = EventRecorder()
    board = MessageBoard()
    reputation = ReputationTracker()
    social_graph = SocialGraph()
    agents = build_agents(social_graph)
    social_graph.load_scenario(scenario, agents)
    board.sync_groups(social_graph)
    setup_date = str(prices["date"].min())[:10]
    social_events = _run_friend_request_round(
        agents=agents,
        social_graph=social_graph,
        board=board,
        rules=scenario.get("friend_request_rules", {}),
        date=setup_date,
        recorder=recorder,
    )
    channels = EXPERIMENTS[experiment]["channels"]
    _set_agent_channels(agents, channels)

    equity, trade_log, aggregation_history, state_history, strategy_history, drift_log = _simulate_population(
        prices=prices,
        agents=agents,
        channels=channels,
        board=board,
        reputation=reputation,
        social_graph=social_graph,
        initial_cash=initial_cash,
        fee_rate=fee_rate,
        rebalance_every=rebalance_every,
        recorder=recorder,
    )
    metrics = performance_metrics(equity).sort_values("sharpe", ascending=False)
    portfolio_outputs = build_agent_portfolio_outputs(
        equity=equity,
        initial_cash=initial_cash,
        rebalance_every=rebalance_every,
    )

    total_returns = metrics.set_index("agent")["total_return"].to_dict()
    social_graph.update_weights(total_returns)
    reputation_scores = reputation.to_dataframe()
    social_edges = social_graph.to_dataframe()
    centrality_scores = _centrality_dataframe(social_graph, experiment, scenario_name)
    belief_history = _belief_history_dataframe(agents)
    aggregation_history = _append_hedge_weights(
        aggregation_history,
        metrics,
        reputation_scores,
        last_date=str(prices["date"].max())[:10],
    )
    agent_registry = _agent_registry_dataframe(agents)
    friendships = social_graph.friendships_dataframe()
    group_memberships = social_graph.groups_dataframe()
    social_events_df = pd.DataFrame(
        social_events,
        columns=["event_id", "date", "event_type", "sender", "receiver", "status", "similarity", "detail"],
    )
    event_log = recorder.to_dataframe()

    prices.to_csv(out_dir / "tables" / "market_history.csv", index=False)
    equity.to_csv(out_dir / "tables" / "equity_curve.csv", index=False)
    portfolio_outputs["agent_equity_curve"].to_csv(out_dir / "tables" / "agent_equity_curve.csv", index=False)
    portfolio_outputs["agent_return_history"].to_csv(out_dir / "tables" / "agent_return_history.csv", index=False)
    portfolio_outputs["agent_return_correlation"].to_csv(out_dir / "tables" / "agent_return_correlation.csv", index=False)
    portfolio_outputs["manager_equity_curve"].to_csv(out_dir / "tables" / "manager_equity_curve.csv", index=False)
    portfolio_outputs["meta_weight_history"].to_csv(out_dir / "tables" / "meta_weight_history.csv", index=False)
    portfolio_outputs["experiment_comparison"].to_csv(out_dir / "tables" / "experiment_comparison.csv", index=False)
    trade_log.to_csv(out_dir / "tables" / "trade_log.csv", index=False)
    metrics.to_csv(out_dir / "tables" / "performance_metrics.csv", index=False)
    board.to_dataframe().to_csv(out_dir / "tables" / "message_log.csv", index=False)
    social_edges.to_csv(out_dir / "tables" / "social_graph_edges.csv", index=False)
    reputation_scores.to_csv(out_dir / "tables" / "reputation_scores.csv", index=False)
    belief_history.to_csv(out_dir / "tables" / "belief_history.csv", index=False)
    centrality_scores.to_csv(out_dir / "tables" / "centrality_scores.csv", index=False)
    aggregation_history.to_csv(out_dir / "tables" / "aggregation_history.csv", index=False)
    agent_registry.to_csv(out_dir / "tables" / "agent_registry.csv", index=False)
    friendships.to_csv(out_dir / "tables" / "friendships.csv", index=False)
    group_memberships.to_csv(out_dir / "tables" / "group_memberships.csv", index=False)
    social_events_df.to_csv(out_dir / "tables" / "social_events.csv", index=False)
    event_log.to_csv(out_dir / "tables" / "unified_event_log.csv", index=False)
    state_history.to_csv(out_dir / "tables" / "agent_state_history.csv", index=False)
    strategy_history.to_csv(out_dir / "tables" / "strategy_choice_history.csv", index=False)
    drift_log.to_csv(out_dir / "tables" / "drift_log.csv", index=False)

    _write_figures(prices, equity, metrics, out_dir, portfolio_outputs["manager_equity_curve"])

    print(f"实验完成：{experiment} / {scenario_name}")
    print("核心结果已保存到：")
    print(out_dir / "tables" / "unified_event_log.csv")
    return equity, trade_log, metrics


def run_all_scenarios(
    prices_path,
    out_dir="outputs",
    initial_cash=100000,
    fee_rate=0.001,
    experiment="full_social",
    rebalance_every=5,
):
    out_dir = Path(out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    metric_frames = []
    event_frames = []
    state_frames = []
    strategy_frames = []
    for scenario_name in sorted(load_scenarios()):
        scenario_out = out_dir / scenario_name
        run(
            prices_path=prices_path,
            out_dir=scenario_out,
            initial_cash=initial_cash,
            fee_rate=fee_rate,
            experiment=experiment,
            rebalance_every=rebalance_every,
            scenario_name=scenario_name,
        )
        metrics = pd.read_csv(scenario_out / "tables" / "performance_metrics.csv")
        metrics.insert(0, "scenario", scenario_name)
        metric_frames.append(metrics)
        events = pd.read_csv(scenario_out / "tables" / "unified_event_log.csv")
        events.insert(0, "scenario", scenario_name)
        event_frames.append(events)
        states = pd.read_csv(scenario_out / "tables" / "agent_state_history.csv")
        states.insert(0, "scenario", scenario_name)
        state_frames.append(states)
        strategy_path = scenario_out / "tables" / "strategy_choice_history.csv"
        if strategy_path.exists():
            strategies = pd.read_csv(strategy_path)
            strategies.insert(0, "scenario", scenario_name)
            strategy_frames.append(strategies)

    if metric_frames:
        comparison = pd.concat(metric_frames, ignore_index=True)
        comparison.to_csv(out_dir / "tables" / "scenario_comparison.csv", index=False)
        summary = (
            comparison.groupby("scenario")
            .agg(
                best_sharpe=("sharpe", "max"),
                average_return=("total_return", "mean"),
                average_drawdown=("max_drawdown", "mean"),
            )
            .reset_index()
        )
        summary.to_csv(out_dir / "tables" / "scenario_summary.csv", index=False)
    if event_frames:
        pd.concat(event_frames, ignore_index=True).to_csv(out_dir / "tables" / "all_scenarios_event_log.csv", index=False)
    if state_frames:
        pd.concat(state_frames, ignore_index=True).to_csv(out_dir / "tables" / "all_scenarios_agent_state_history.csv", index=False)
    if strategy_frames:
        pd.concat(strategy_frames, ignore_index=True).to_csv(out_dir / "tables" / "all_scenarios_strategy_choice_history.csv", index=False)
    print("全部社交图谱场景已完成。")


def load_scenarios() -> dict:
    path = Path(__file__).parent.parent / "config" / "social_scenarios.yaml"
    if not path.exists():
        return DEFAULT_SCENARIOS
    with open(path, encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    return loaded.get("scenarios", DEFAULT_SCENARIOS)


def load_scenario(scenario_name: str) -> dict:
    scenarios = load_scenarios()
    if scenario_name not in scenarios:
        known = ", ".join(sorted(scenarios))
        raise ValueError(f"unknown scenario '{scenario_name}', choose one of: {known}")
    return scenarios[scenario_name]


def _simulate_population(
    prices,
    agents,
    channels,
    board,
    reputation,
    social_graph,
    initial_cash,
    fee_rate,
    rebalance_every,
    recorder: EventRecorder,
):
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["date", "ticker"])
    dates = sorted(prices["date"].unique())
    tickers = sorted(prices["ticker"].unique())
    portfolios = {
        agent.name: {"cash": float(initial_cash), "positions": {ticker: 0 for ticker in tickers}}
        for agent in agents
    }
    equity_rows = []
    state_rows = []
    trade_rows = []
    aggregation_rows = []
    strategy_rows = []
    drift_rows = []
    last_actions = {agent.name: "等待市场开盘" for agent in agents}

    for step, date in enumerate(dates):
        today = prices[prices["date"] == date].set_index("ticker")
        date_str = str(date)[:10]
        should_rebalance = step > 80 and step % rebalance_every == 0
        if should_rebalance:
            history = prices[prices["date"] <= date]
            for ticker, realized_return in _recent_ticker_returns(history).items():
                reputation.record_outcome(ticker, realized_return, date_str)

            chosen_rows = _choose_social_strategies(
                agents=agents,
                board=board,
                reputation=reputation,
                social_graph=social_graph,
                channels=channels,
                portfolios=portfolios,
                today=today,
                initial_cash=initial_cash,
                current_date=date_str,
                recorder=recorder,
            )
            strategy_rows.extend(chosen_rows)

            _publish_messages(agents, history, board, reputation, social_graph, channels, date_str, step, recorder)
            for agent in _communicating_agents(agents):
                if getattr(agent, "current_social_strategy", "cooperate") == "independent":
                    continue
                agent.update_belief(
                    board,
                    reputation,
                    current_date=date_str,
                    channels=channels,
                    agent_groups=social_graph.get_agent_groups(agent.name),
                )

            aggregation_rows.extend(_centrality_rows(social_graph, date_str, method="pagerank"))
            for agent in agents:
                portfolio = portfolios[agent.name]
                value_before = _portfolio_value(portfolio["cash"], portfolio["positions"], today)
                decision = agent.target_weights(history, portfolio["cash"], portfolio["positions"])
                spec, violations = _strategy_contract_result(agent, decision, rebalance_every)
                decision.strategy_spec_version = spec.version
                decision.param_hash = spec.param_hash()
                for violation in violations:
                    drift_rows.append(
                        {
                            "date": date_str,
                            "agent": agent.name,
                            "strategy_spec": spec.name,
                            "strategy_spec_version": spec.version,
                            "param_hash": decision.param_hash,
                            "violation": violation,
                            "target_weights_json": json.dumps(decision.target_weights, ensure_ascii=False),
                        }
                    )
                recorder.add(
                    date_str,
                    event_type="decision",
                    source="portfolio",
                    agent=agent.name,
                    cash=portfolio["cash"],
                    equity=value_before,
                    pnl=value_before - initial_cash,
                    pnl_pct=value_before / initial_cash - 1,
                    detail=f"目标仓位：{_target_summary(decision.target_weights)}。{decision.note}",
                    payload={
                        "target_weights": decision.target_weights,
                        "note": decision.note,
                        "strategy_spec": spec.name,
                        "strategy_spec_version": decision.strategy_spec_version,
                        "param_hash": decision.param_hash,
                        "drift_violations": violations,
                    },
                )
                portfolio["cash"], new_trades = _execute_decision(
                    date_str=date_str,
                    agent_name=agent.name,
                    decision=decision,
                    target_weights=decision.target_weights,
                    value=value_before,
                    cash=portfolio["cash"],
                    positions=portfolio["positions"],
                    today=today,
                    fee_rate=fee_rate,
                )
                if new_trades:
                    for trade in new_trades:
                        trade_rows.append(trade)
                        last_actions[agent.name] = _trade_detail(trade)
                        recorder.add(
                            date_str,
                            event_type="trade",
                            source="portfolio",
                            agent=agent.name,
                            ticker=trade["ticker"],
                            side=trade["side"],
                            shares=trade["shares"],
                            price=trade["price"],
                            notional=trade["notional"],
                            cash=trade["cash_after"],
                            equity=trade["equity_after"],
                            pnl=trade["equity_after"] - initial_cash,
                            pnl_pct=trade["equity_after"] / initial_cash - 1,
                            detail=_trade_detail(trade),
                            payload=trade,
                        )
                else:
                    last_actions[agent.name] = f"HOLD：无交易；{_target_summary(decision.target_weights)}"
                    recorder.add(
                        date_str,
                        event_type="hold",
                        source="portfolio",
                        agent=agent.name,
                        cash=portfolio["cash"],
                        equity=value_before,
                        pnl=value_before - initial_cash,
                        pnl_pct=value_before / initial_cash - 1,
                        detail=last_actions[agent.name],
                        payload={"target_weights": decision.target_weights},
                    )

            for agent in _communicating_agents(agents):
                agent.belief.snapshot(date_str)

        for agent in agents:
            portfolio = portfolios[agent.name]
            today_value = _portfolio_value(portfolio["cash"], portfolio["positions"], today)
            pnl = today_value - initial_cash
            pnl_pct = today_value / initial_cash - 1
            position_summary = _position_summary(portfolio["positions"])
            equity_rows.append([date_str, agent.name, today_value, portfolio["cash"]])
            state_rows.append(
                {
                    "date": date_str,
                    "agent": agent.name,
                    "equity": today_value,
                    "cash": portfolio["cash"],
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "positions_json": json.dumps(portfolio["positions"], ensure_ascii=False),
                    "position_summary": position_summary,
                    "last_action": last_actions.get(agent.name, ""),
                    "friend_count": len(social_graph.get_friends(agent.name)),
                    "friends": ", ".join(social_graph.get_friends(agent.name)),
                }
            )
            recorder.add(
                date_str,
                event_type="pnl_snapshot",
                source="portfolio",
                agent=agent.name,
                cash=portfolio["cash"],
                equity=today_value,
                pnl=pnl,
                pnl_pct=pnl_pct,
                detail=f"PnL {pnl:+,.2f} ({pnl_pct:+.2%})，持仓 {position_summary or 'cash only'}",
                payload={"positions": portfolio["positions"]},
            )

    equity = pd.DataFrame(equity_rows, columns=["date", "agent", "equity", "cash"])
    trades = pd.DataFrame(
        trade_rows,
        columns=[
            "date",
            "agent",
            "ticker",
            "side",
            "shares",
            "price",
            "fee",
            "note",
            "notional",
            "cash_after",
            "equity_after",
        ],
    )
    aggregation = pd.DataFrame(aggregation_rows, columns=["date", "method", "agent", "value"])
    state_history = pd.DataFrame(
        state_rows,
        columns=[
            "date",
            "agent",
            "equity",
            "cash",
            "pnl",
            "pnl_pct",
            "positions_json",
            "position_summary",
            "last_action",
            "friend_count",
            "friends",
        ],
    )
    strategy_history = pd.DataFrame(
        strategy_rows,
        columns=[
            "date",
            "agent",
            "strategy",
            "expected_utility",
            "cooperate_utility",
            "compete_utility",
            "observe_utility",
            "independent_utility",
            "reason",
            "reputation",
            "influence_score",
            "pagerank",
            "friend_count",
            "visible_messages",
            "return_to_date",
            "graph_density",
        ],
    )
    drift_log = pd.DataFrame(
        drift_rows,
        columns=[
            "date",
            "agent",
            "strategy_spec",
            "strategy_spec_version",
            "param_hash",
            "violation",
            "target_weights_json",
        ],
    )
    return equity, trades, aggregation, state_history, strategy_history, drift_log



def _choose_social_strategies(
    agents,
    board,
    reputation,
    social_graph,
    channels,
    portfolios,
    today,
    initial_cash,
    current_date,
    recorder: EventRecorder,
) -> list[dict]:
    """Heuristic utility model for cooperate / compete / observe / independent."""
    rows = []
    centrality = social_graph.centrality()
    names = [agent.name for agent in agents]
    n = max(1, len(names))
    edge_count = len(list(social_graph.graph.edges()))
    graph_density = float(edge_count / max(1, n * (n - 1)))
    for agent in _communicating_agents(agents):
        friends = social_graph.get_friends(agent.name)
        friend_count = len(friends)
        friend_norm = min(1.0, friend_count / max(1, n - 1))
        visible = board.get_visible(
            agent.name,
            channels=channels or agent.enabled_channels,
            since=str((pd.Timestamp(current_date) - pd.Timedelta(days=30)))[:10],
            agent_groups=social_graph.get_agent_groups(agent.name),
        )
        visible_count = len(visible)
        visible_norm = min(1.0, visible_count / 18.0)
        reputation_score = float(reputation.get_reputation(agent.name))
        influence_score = float(reputation.get_influence_score(agent.name))
        pagerank = float(centrality.get(agent.name, 0.0))
        portfolio = portfolios[agent.name]
        value = _portfolio_value(portfolio["cash"], portfolio["positions"], today)
        return_to_date = value / initial_cash - 1
        positive_return = max(0.0, min(1.0, return_to_date * 5.0))
        negative_return = max(0.0, min(1.0, -return_to_date * 5.0))

        cooperate_u = 0.34 * friend_norm + 0.24 * visible_norm + 0.24 * reputation_score + 0.10 * graph_density + 0.08 * positive_return
        compete_u = 0.30 * min(1.0, pagerank * n) + 0.24 * min(1.0, influence_score * 10) + 0.26 * positive_return + 0.20 * (1.0 - graph_density)
        observe_u = 0.44 * visible_norm + 0.22 * (1.0 - reputation_score) + 0.20 * friend_norm + 0.14 * negative_return
        independent_u = 0.18 * (1.0 - visible_norm) + 0.12 * (1.0 - friend_norm) + 0.12 * positive_return + 0.10 * (1.0 - graph_density)

        labels = {agent.name, agent.__class__.__name__, getattr(getattr(agent, "own_strategy", None), "name", "")}
        if labels & {"TruthfulReporterAgent", "MomentumAgent", "CommitteeTeamAgent", "DynamicTeamAgent", "SocialGraphAgent"}:
            cooperate_u += 0.10
        if labels & {"PersuaderAgent", "ContrarianAgent", "MeanReversionAgent"}:
            compete_u += 0.12
        if labels & {"FreeRiderAgent", "LowVolatilityAgent"}:
            observe_u += 0.12
        if labels & {"RandomAgent"}:
            independent_u += 0.08

        utilities = {
            "cooperate": cooperate_u,
            "compete": compete_u,
            "observe": observe_u,
            "independent": independent_u,
        }
        strategy = max(utilities, key=utilities.get)
        agent.current_social_strategy = strategy
        reason = _strategy_reason(strategy, friend_count, visible_count, reputation_score, pagerank, return_to_date, graph_density)
        row = {
            "date": current_date,
            "agent": agent.name,
            "strategy": strategy,
            "expected_utility": float(utilities[strategy]),
            "cooperate_utility": float(cooperate_u),
            "compete_utility": float(compete_u),
            "observe_utility": float(observe_u),
            "independent_utility": float(independent_u),
            "reason": reason,
            "reputation": reputation_score,
            "influence_score": influence_score,
            "pagerank": pagerank,
            "friend_count": friend_count,
            "visible_messages": visible_count,
            "return_to_date": float(return_to_date),
            "graph_density": graph_density,
        }
        rows.append(row)
        recorder.add(
            current_date,
            event_type="strategy_choice",
            source="strategy",
            agent=agent.name,
            detail=f"选择 {strategy}：{reason}",
            equity=value,
            cash=portfolio["cash"],
            pnl=value - initial_cash,
            pnl_pct=value / initial_cash - 1,
            payload={"utilities": utilities, "visible_messages": visible_count, "friend_count": friend_count},
        )
    return rows


def _strategy_reason(strategy, friend_count, visible_count, reputation_score, pagerank, return_to_date, graph_density) -> str:
    if strategy == "cooperate":
        return f"好友 {friend_count}、可见消息 {visible_count}、声誉 {reputation_score:.2f}，合作可提高信息融合。"
    if strategy == "compete":
        return f"PageRank {pagerank:.3f}、收益 {return_to_date:+.2%}，竞争可扩大影响力。"
    if strategy == "observe":
        return f"可见消息 {visible_count} 且声誉 {reputation_score:.2f}，先观察以利用外部信息。"
    return f"图密度 {graph_density:.2f}、好友 {friend_count}、收益 {return_to_date:+.2%}，社交边际收益不足。"


def _strategy_adjusted_channel(agent, channel, social_graph):
    strategy = getattr(agent, "current_social_strategy", "cooperate")
    if strategy == "independent":
        return None
    if strategy == "compete" and "public" in agent.enabled_channels:
        return "public"
    if strategy == "observe":
        return channel
    if strategy == "cooperate" and "private" in agent.enabled_channels and social_graph.get_friends(agent.name):
        return "private"
    return channel


def _strategy_allows_publish(agent, step: int) -> bool:
    strategy = getattr(agent, "current_social_strategy", "cooperate")
    if strategy == "independent":
        return False
    if strategy == "observe":
        return (step + sum(ord(ch) for ch in agent.name)) % 4 == 0
    return True


def _publish_messages(agents, history, board, reputation, social_graph, channels, date_str, step, recorder: EventRecorder) -> None:
    if not channels:
        return
    for agent in _communicating_agents(agents):
        if not _strategy_allows_publish(agent, step):
            continue
        channel = _choose_channel(agent, channels, step)
        channel = _strategy_adjusted_channel(agent, channel, social_graph)
        if channel is None:
            continue
        receivers = _private_receivers(agent.name, agents, social_graph) if channel == "private" else []
        if channel == "private" and not receivers:
            continue
        msg = agent.generate_message(history, board, date_str, channel=channel, receiver_ids=receivers)
        if msg is None:
            continue
        social_strategy = getattr(agent, "current_social_strategy", "cooperate")
        if social_strategy == "compete":
            msg.confidence = min(1.0, msg.confidence + 0.10)
            msg.natural_language = f"[compete] {msg.natural_language}"
        elif social_strategy == "cooperate":
            msg.natural_language = f"[cooperate] {msg.natural_language}"
        elif social_strategy == "observe":
            msg.natural_language = f"[observe] {msg.natural_language}"
        receiver_text = ", ".join(receivers) if receivers else ("朋友圈可见" if channel == "moments" else "群聊")
        event_id = recorder.add(
            date_str,
            event_type="message",
            source="chat",
            agent=msg.sender_id,
            counterparty=receiver_text,
            channel=msg.channel,
            ticker=", ".join(msg.tickers),
            detail=msg.natural_language,
            payload={
                "message_id": msg.message_id,
                "receiver_ids": msg.receiver_ids,
                "tickers": msg.tickers,
                "direction": msg.direction,
                "confidence": msg.confidence,
                "position_intent": msg.position_intent,
                "social_strategy": social_strategy,
                "evidence": msg.evidence,
            },
        )
        msg.event_id = event_id
        board.post(msg)
        for ticker in msg.tickers:
            reputation.record_prediction(msg.sender_id, ticker, msg.direction, msg.confidence, msg.timestamp)


def _execute_decision(date_str, agent_name, decision, target_weights, value, cash, positions, today, fee_rate):
    trade_rows = []
    for ticker in positions:
        if ticker not in today.index:
            continue
        price = float(today.loc[ticker, "close"])
        current_value = positions[ticker] * price
        target_value = value * target_weights.get(ticker, 0.0)
        diff_value = target_value - current_value
        if abs(diff_value) < value * 0.005:
            continue
        shares = int(diff_value / price)
        if shares == 0:
            continue
        cost = shares * price
        fee = abs(cost) * fee_rate
        if shares > 0 and cash >= cost + fee:
            positions[ticker] += shares
            cash -= cost + fee
            equity_after = _portfolio_value(cash, positions, today)
            trade_rows.append(
                {
                    "date": date_str,
                    "agent": agent_name,
                    "ticker": ticker,
                    "side": "BUY",
                    "shares": int(shares),
                    "price": price,
                    "fee": fee,
                    "note": decision.note,
                    "notional": abs(cost),
                    "cash_after": cash,
                    "equity_after": equity_after,
                }
            )
        elif shares < 0 and positions[ticker] >= abs(shares):
            positions[ticker] += shares
            cash -= cost
            cash -= fee
            equity_after = _portfolio_value(cash, positions, today)
            trade_rows.append(
                {
                    "date": date_str,
                    "agent": agent_name,
                    "ticker": ticker,
                    "side": "SELL",
                    "shares": int(abs(shares)),
                    "price": price,
                    "fee": fee,
                    "note": decision.note,
                    "notional": abs(cost),
                    "cash_after": cash,
                    "equity_after": equity_after,
                }
            )
    return cash, trade_rows


def _portfolio_value(cash, positions, today) -> float:
    return float(cash + sum(positions.get(ticker, 0) * today.loc[ticker, "close"] for ticker in positions if ticker in today.index))


def _recent_ticker_returns(history, lookback=5) -> dict:
    returns = {}
    for ticker, group in history.groupby("ticker"):
        group = group.sort_values("date")
        if len(group) <= lookback:
            returns[ticker] = 0.0
            continue
        returns[ticker] = float(group["close"].iloc[-1] / group["close"].iloc[-lookback - 1] - 1)
    return returns


def _set_agent_channels(agents, channels) -> None:
    for agent in _communicating_agents(agents):
        agent.enabled_channels = list(channels)


def _communicating_agents(agents) -> list:
    return [agent for agent in agents if isinstance(agent, CommunicatingAgent)]


def _run_friend_request_round(agents, social_graph, board, rules, date, recorder: EventRecorder, n_rounds=3) -> list[dict]:
    communicating = _communicating_agents(agents)
    events = []
    for round_index in range(n_rounds):
        for agent in communicating:
            candidates = [
                candidate
                for candidate in communicating
                if candidate.name != agent.name and not social_graph.are_friends(agent.name, candidate.name)
            ]
            candidates.sort(key=lambda candidate: -social_graph._strategy_similarity(agent, candidate))
            for candidate in candidates[:2]:
                social_graph.send_friend_request(agent.name, candidate.name)
                detail = f"第 {round_index + 1} 轮：{agent.name} 申请添加 {candidate.name} 为好友"
                event_id = recorder.add(
                    date,
                    event_type="friend_request",
                    source="social",
                    agent=agent.name,
                    counterparty=candidate.name,
                    channel="friend_request",
                    detail=detail,
                    payload={"round": round_index + 1, "status": "pending"},
                )
                events.append(
                    {
                        "event_id": event_id,
                        "date": date,
                        "event_type": "friend_request",
                        "sender": agent.name,
                        "receiver": candidate.name,
                        "status": "pending",
                        "similarity": None,
                        "detail": detail,
                    }
                )
        decisions = social_graph.process_requests_detailed(agents, rules)
        for row in decisions:
            event_type = "friend_accept" if row["status"] == "accepted" else "friend_reject"
            detail = f"{row['receiver']} {'通过' if row['status'] == 'accepted' else '拒绝'}了 {row['sender']} 的好友申请：{row['reason']}"
            event_id = recorder.add(
                date,
                event_type=event_type,
                source="social",
                agent=row["receiver"],
                counterparty=row["sender"],
                channel="friend_request",
                detail=detail,
                payload=row,
            )
            events.append(
                {
                    "event_id": event_id,
                    "date": date,
                    "event_type": event_type,
                    "sender": row["sender"],
                    "receiver": row["receiver"],
                    "status": row["status"],
                    "similarity": row["similarity"],
                    "detail": detail,
                }
            )
        board.sync_groups(social_graph)
    return events


def _choose_channel(agent, channels, step) -> str:
    if len(channels) == 1:
        return channels[0]
    preferences = {
        "MomentumAgent": ["public", "moments", "private"],
        "MeanReversionAgent": ["public", "private", "moments"],
        "LowVolatilityAgent": ["moments", "private", "public"],
        "DrawdownBuyerAgent": ["moments", "public", "private"],
        "RandomAgent": ["public", "moments", "private"],
        "CommitteeTeamAgent": ["public", "private", "moments"],
        "DynamicTeamAgent": ["private", "public", "moments"],
        "TruthfulReporterAgent": ["public", "moments", "private"],
        "PersuaderAgent": ["public", "private", "moments"],
        "FreeRiderAgent": ["moments", "public", "private"],
        "ContrarianAgent": ["moments", "public", "private"],
        "SocialGraphAgent": ["private", "public", "moments"],
    }
    ordered = [channel for channel in preferences.get(agent.name, channels) if channel in channels]
    if not ordered:
        ordered = list(channels)
    return ordered[(step + sum(ord(ch) for ch in agent.name)) % len(ordered)]


def _private_receivers(agent_name, agents, social_graph) -> list:
    return [
        agent.name
        for agent in _communicating_agents(agents)
        if agent.name != agent_name and social_graph.are_friends(agent_name, agent.name)
    ]


def _centrality_rows(social_graph, date, method) -> list[dict]:
    return [
        {"date": date, "method": method, "agent": agent, "value": float(value)}
        for agent, value in social_graph.centrality().items()
    ]


def _centrality_dataframe(social_graph, experiment, scenario_name) -> pd.DataFrame:
    rows = [
        {"experiment": experiment, "scenario": scenario_name, "agent": agent, "pagerank": float(value)}
        for agent, value in social_graph.centrality().items()
    ]
    return pd.DataFrame(rows, columns=["experiment", "scenario", "agent", "pagerank"])


def _belief_history_dataframe(agents) -> pd.DataFrame:
    frames = [agent.belief.to_dataframe() for agent in _communicating_agents(agents)]
    if not frames:
        return pd.DataFrame(columns=["date", "agent", "ticker", "belief", "source"])
    return pd.concat(frames, ignore_index=True)


def _agent_registry_dataframe(agents) -> pd.DataFrame:
    rows = []
    for agent in agents:
        spec = strategy_spec_for_agent(agent)
        rows.append(
            {
                "agent": agent.name,
                "class": agent.__class__.__name__,
                "communicating": isinstance(agent, CommunicatingAgent),
                "own_strategy": getattr(getattr(agent, "own_strategy", None), "name", ""),
                "enabled_channels": ", ".join(getattr(agent, "enabled_channels", [])),
                "strategy_spec": spec.name,
                "strategy_family": spec.family,
                "strategy_spec_version": spec.version,
                "param_hash": spec.param_hash(),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "agent",
            "class",
            "communicating",
            "own_strategy",
            "enabled_channels",
            "strategy_spec",
            "strategy_family",
            "strategy_spec_version",
            "param_hash",
        ],
    )


def _append_hedge_weights(aggregation_history, metrics, reputation_scores, last_date) -> pd.DataFrame:
    agent_scores = metrics.set_index("agent")["total_return"].to_dict()
    drawdowns = metrics.set_index("agent")["max_drawdown"].to_dict()
    if reputation_scores.empty:
        calibration = {}
    else:
        calibration = reputation_scores.set_index("sender_id")["calibration_error"].to_dict()
    weights = hedge_weights(agent_scores, drawdowns, calibration)
    hedge_rows = pd.DataFrame(
        [
            {"date": last_date, "method": "hedge_weight", "agent": agent, "value": value}
            for agent, value in weights.items()
        ],
        columns=["date", "method", "agent", "value"],
    )
    return pd.concat([aggregation_history, hedge_rows], ignore_index=True)


def _write_figures(prices, equity, metrics, out_dir, manager_equity=None) -> None:
    plt.figure(figsize=(10, 5))
    for agent, group in equity.groupby("agent"):
        group = group.sort_values("date")
        plt.plot(pd.to_datetime(group["date"]), group["equity"] / group["equity"].iloc[0], label=agent)
    plt.title("Agent Equity Curves")
    plt.xlabel("Date")
    plt.ylabel("Normalized Equity")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_dir / "figures" / "equity_curves.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(metrics["agent"], metrics["total_return"])
    plt.title("Total Return by Agent")
    plt.xlabel("Agent")
    plt.ylabel("Total Return")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "figures" / "total_return_by_agent.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(metrics["agent"], metrics["sharpe"])
    plt.title("Sharpe Ratio by Agent")
    plt.xlabel("Agent")
    plt.ylabel("Sharpe Ratio")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "figures" / "sharpe_by_agent.png", dpi=180)
    plt.close()

    norm_prices = prices.copy()
    norm_prices["date"] = pd.to_datetime(norm_prices["date"])
    plt.figure(figsize=(10, 5))
    for ticker, group in norm_prices.groupby("ticker"):
        group = group.sort_values("date")
        plt.plot(group["date"], group["close"] / group["close"].iloc[0], label=ticker)
    plt.title("Underlying Market Prices")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "figures" / "market_prices.png", dpi=180)
    plt.close()

    if manager_equity is not None and not manager_equity.empty:
        plt.figure(figsize=(10, 5))
        for manager, group in manager_equity.groupby("manager"):
            group = group.sort_values("date")
            plt.plot(pd.to_datetime(group["date"]), group["equity"] / group["equity"].iloc[0], label=manager)
        plt.title("Agent Portfolio Manager Equity Curves")
        plt.xlabel("Date")
        plt.ylabel("Normalized Equity")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "figures" / "manager_equity_curves.png", dpi=180)
        plt.close()


def _strategy_contract_result(agent, decision, rebalance_every) -> tuple:
    spec = strategy_spec_for_agent(agent, rebalance_every=rebalance_every)
    violations = spec.validate_action({"target_weights": getattr(decision, "target_weights", {})})
    return spec, violations


def _safe_optional_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _target_summary(weights: dict, top_n: int = 3) -> str:
    if not weights:
        return "空仓"
    items = sorted(weights.items(), key=lambda item: float(item[1]), reverse=True)
    parts = [f"{ticker} {float(weight):.1%}" for ticker, weight in items[:top_n] if float(weight) > 0.001]
    return ", ".join(parts) if parts else "现金/观望"


def _position_summary(positions: dict, top_n: int = 4) -> str:
    non_zero = {ticker: int(shares) for ticker, shares in positions.items() if int(shares) != 0}
    if not non_zero:
        return ""
    items = sorted(non_zero.items(), key=lambda item: abs(item[1]), reverse=True)[:top_n]
    return ", ".join(f"{ticker}:{shares}" for ticker, shares in items)


def _trade_detail(trade: dict) -> str:
    return (
        f"{trade['side']} {int(trade['shares'])} {trade['ticker']} "
        f"@ {float(trade['price']):.2f}，成交额 {float(trade['notional']):,.2f}，手续费 {float(trade['fee']):,.2f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default="data/sample_synthetic_prices.csv")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--initial_cash", type=float, default=100000)
    parser.add_argument("--fee_rate", type=float, default=0.001)
    parser.add_argument("--rebalance_every", type=int, default=5)
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), default="full_social")
    parser.add_argument("--scenario", choices=sorted(load_scenarios()) + ["all"], default="core_periphery")
    args = parser.parse_args()
    if args.scenario == "all":
        run_all_scenarios(args.prices, args.out, args.initial_cash, args.fee_rate, args.experiment, args.rebalance_every)
    else:
        run(args.prices, args.out, args.initial_cash, args.fee_rate, args.experiment, args.rebalance_every, args.scenario)

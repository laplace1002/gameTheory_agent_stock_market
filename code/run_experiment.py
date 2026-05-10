import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ai_agent_stock_market_matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from aggregation import hedge_weights
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


EXPERIMENTS = {
    "baseline": {"channels": []},
    "public_only": {"channels": ["public"]},
    "private_only": {"channels": ["private"]},
    "moments_only": {"channels": ["moments"]},
    "full_social": {"channels": ["public", "private", "moments"]},
}


def build_agents(social_graph: SocialGraph) -> list:
    base_agents = [
        MomentumAgent(),
        MeanReversionAgent(),
        LowVolatilityAgent(),
        DrawdownBuyerAgent(),
        RandomAgent(seed=42),
        CommitteeTeamAgent(),
        DynamicTeamAgent(),
    ]
    communicating_agents = [
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
    agents = base_agents + communicating_agents + [social_agent]
    social_agent.all_agents = agents
    return agents


def run(
    prices_path,
    out_dir="outputs",
    initial_cash=100000,
    fee_rate=0.001,
    experiment="full_social",
    rebalance_every=5,
    scenario_name="case1",
):
    if experiment not in EXPERIMENTS:
        known = ", ".join(sorted(EXPERIMENTS))
        raise ValueError(f"unknown experiment '{experiment}', choose one of: {known}")

    scenario = load_scenario(scenario_name)
    out_dir = Path(out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    prices = load_prices(prices_path)
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
    )
    channels = EXPERIMENTS[experiment]["channels"]
    _set_agent_channels(agents, channels)

    equity, trade_log, aggregation_history = _simulate_population(
        prices=prices,
        agents=agents,
        channels=channels,
        board=board,
        reputation=reputation,
        social_graph=social_graph,
        initial_cash=initial_cash,
        fee_rate=fee_rate,
        rebalance_every=rebalance_every,
    )
    metrics = performance_metrics(equity).sort_values("sharpe", ascending=False)

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
    social_events = pd.DataFrame(
        social_events,
        columns=["date", "event_type", "sender", "receiver", "status", "detail"],
    )

    prices.to_csv(out_dir / "tables" / "market_history.csv", index=False)
    equity.to_csv(out_dir / "tables" / "equity_curve.csv", index=False)
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
    social_events.to_csv(out_dir / "tables" / "social_events.csv", index=False)

    _write_figures(prices, equity, metrics, out_dir)

    print(f"实验完成：{experiment} / {scenario_name}")
    print("核心结果已保存到：")
    print(out_dir / "tables" / "performance_metrics.csv")
    return equity, trade_log, metrics


def load_scenarios() -> dict:
    path = Path(__file__).parent.parent / "config" / "social_scenarios.yaml"
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)["scenarios"]


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
    trade_rows = []
    aggregation_rows = []

    for step, date in enumerate(dates):
        today = prices[prices["date"] == date].set_index("ticker")
        if step > 80 and step % rebalance_every == 0:
            date_str = str(date)[:10]
            history = prices[prices["date"] <= date]
            for ticker, realized_return in _recent_ticker_returns(history).items():
                reputation.record_outcome(ticker, realized_return, date_str)

            _publish_messages(agents, history, board, reputation, social_graph, channels, date_str, step)
            for agent in _communicating_agents(agents):
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
                value = _portfolio_value(portfolio["cash"], portfolio["positions"], today)
                decision = agent.target_weights(history, portfolio["cash"], portfolio["positions"])
                portfolio["cash"], new_trades = _execute_decision(
                    date_str=date_str,
                    agent_name=agent.name,
                    decision=decision,
                    target_weights=decision.target_weights,
                    value=value,
                    cash=portfolio["cash"],
                    positions=portfolio["positions"],
                    today=today,
                    fee_rate=fee_rate,
                )
                trade_rows.extend(new_trades)

            for agent in _communicating_agents(agents):
                agent.belief.snapshot(date_str)

        for agent in agents:
            portfolio = portfolios[agent.name]
            today_value = _portfolio_value(portfolio["cash"], portfolio["positions"], today)
            equity_rows.append([str(date)[:10], agent.name, today_value, portfolio["cash"]])

    equity = pd.DataFrame(equity_rows, columns=["date", "agent", "equity", "cash"])
    trades = pd.DataFrame(
        trade_rows,
        columns=["date", "agent", "ticker", "side", "shares", "price", "fee", "note"],
    )
    aggregation = pd.DataFrame(aggregation_rows, columns=["date", "method", "agent", "value"])
    return equity, trades, aggregation


def _publish_messages(agents, history, board, reputation, social_graph, channels, date_str, step) -> None:
    if not channels:
        return
    for agent in _communicating_agents(agents):
        channel = _choose_channel(agent, channels, step)
        receivers = _private_receivers(agent.name, agents, social_graph) if channel == "private" else []
        if channel == "private" and not receivers:
            continue
        msg = agent.generate_message(history, board, date_str, channel=channel, receiver_ids=receivers)
        if msg is None:
            continue
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
            trade_rows.append([date_str, agent_name, ticker, "BUY", shares, price, fee, decision.note])
        elif shares < 0 and positions[ticker] >= abs(shares):
            positions[ticker] += shares
            cash -= cost
            cash -= fee
            trade_rows.append([date_str, agent_name, ticker, "SELL", abs(shares), price, fee, decision.note])
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


def _run_friend_request_round(agents, social_graph, board, rules, date, n_rounds=3) -> list[dict]:
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
                events.append(
                    {
                        "date": date,
                        "event_type": "friend_request",
                        "sender": agent.name,
                        "receiver": candidate.name,
                        "status": "pending",
                        "detail": f"round {round_index + 1}: {agent.name} requested {candidate.name}",
                    }
                )
        accepted = social_graph.process_requests(agents, rules)
        for sender, receiver in accepted:
            events.append(
                {
                    "date": date,
                    "event_type": "friend_accept",
                    "sender": sender,
                    "receiver": receiver,
                    "status": "accepted",
                    "detail": f"{receiver} accepted {sender}",
                }
            )
        board.sync_groups(social_graph)
    return events


def _choose_channel(agent, channels, step) -> str:
    if len(channels) == 1:
        return channels[0]
    preferences = {
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
        rows.append(
            {
                "agent": agent.name,
                "class": agent.__class__.__name__,
                "communicating": isinstance(agent, CommunicatingAgent),
                "own_strategy": getattr(getattr(agent, "own_strategy", None), "name", ""),
            }
        )
    return pd.DataFrame(rows, columns=["agent", "class", "communicating", "own_strategy"])


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


def _write_figures(prices, equity, metrics, out_dir) -> None:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default="data/sample_synthetic_prices.csv")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--initial_cash", type=float, default=100000)
    parser.add_argument("--fee_rate", type=float, default=0.001)
    parser.add_argument("--rebalance_every", type=int, default=5)
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), default="full_social")
    parser.add_argument("--scenario", choices=sorted(load_scenarios()), default="case1")
    args = parser.parse_args()
    run(args.prices, args.out, args.initial_cash, args.fee_rate, args.experiment, args.rebalance_every, args.scenario)

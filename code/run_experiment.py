import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_prices
from agents import (
    MomentumAgent,
    MeanReversionAgent,
    LowVolatilityAgent,
    DrawdownBuyerAgent,
    RandomAgent,
    CommitteeTeamAgent,
    DynamicTeamAgent,
)
from portfolio import simulate_agent, performance_metrics


def run(prices_path, out_dir="outputs", initial_cash=100000, fee_rate=0.001):
    out_dir = Path(out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    prices = load_prices(prices_path)
    agents = [
        MomentumAgent(),
        MeanReversionAgent(),
        LowVolatilityAgent(),
        DrawdownBuyerAgent(),
        RandomAgent(seed=42),
        CommitteeTeamAgent(),
        DynamicTeamAgent(),
    ]

    equities = []
    trades = []
    for agent in agents:
        eq, tr = simulate_agent(prices, agent, initial_cash=initial_cash, fee_rate=fee_rate)
        equities.append(eq)
        trades.append(tr)
    equity = pd.concat(equities, ignore_index=True)
    trade_log = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    metrics = performance_metrics(equity)
    metrics = metrics.sort_values("sharpe", ascending=False)

    equity.to_csv(out_dir / "tables" / "equity_curve.csv", index=False)
    trade_log.to_csv(out_dir / "tables" / "trade_log.csv", index=False)
    metrics.to_csv(out_dir / "tables" / "performance_metrics.csv", index=False)

    plt.figure(figsize=(10, 5))
    for agent, g in equity.groupby("agent"):
        g = g.sort_values("date")
        plt.plot(pd.to_datetime(g["date"]), g["equity"] / g["equity"].iloc[0], label=agent)
    plt.title("Agent Equity Curves")
    plt.xlabel("Date")
    plt.ylabel("Normalized Equity")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "figures" / "equity_curves.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(metrics["agent"], metrics["total_return"])
    plt.title("Total Return by Agent")
    plt.xlabel("Agent")
    plt.ylabel("Total Return")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "figures" / "total_return_by_agent.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(metrics["agent"], metrics["sharpe"])
    plt.title("Sharpe Ratio by Agent")
    plt.xlabel("Agent")
    plt.ylabel("Sharpe Ratio")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "figures" / "sharpe_by_agent.png", dpi=180)
    plt.close()

    norm_prices = prices.copy()
    norm_prices["date"] = pd.to_datetime(norm_prices["date"])
    plt.figure(figsize=(10, 5))
    for ticker, g in norm_prices.groupby("ticker"):
        g = g.sort_values("date")
        plt.plot(g["date"], g["close"] / g["close"].iloc[0], label=ticker)
    plt.title("Underlying Market Prices")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "figures" / "market_prices.png", dpi=180)
    plt.close()

    print("实验完成。核心结果已保存到：")
    print(out_dir / "tables" / "performance_metrics.csv")
    return equity, trade_log, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default="data/sample_synthetic_prices.csv")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--initial_cash", type=float, default=100000)
    parser.add_argument("--fee_rate", type=float, default=0.001)
    args = parser.parse_args()
    run(args.prices, args.out, args.initial_cash, args.fee_rate)

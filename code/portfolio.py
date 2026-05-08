import numpy as np
import pandas as pd


def simulate_agent(prices: pd.DataFrame, agent, initial_cash=100000.0, fee_rate=0.001, rebalance_every=5):
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["date", "ticker"])
    dates = sorted(prices["date"].unique())
    cash = float(initial_cash)
    positions = {t: 0 for t in sorted(prices["ticker"].unique())}
    equity_rows = []
    trade_rows = []

    for step, d in enumerate(dates):
        today = prices[prices["date"] == d].set_index("ticker")
        value = cash + sum(positions.get(t, 0) * today.loc[t, "close"] for t in positions if t in today.index)
        if step > 80 and step % rebalance_every == 0:
            history = prices[prices["date"] <= d]
            decision = agent.target_weights(history, cash, positions)
            target = decision.target_weights
            for ticker in positions:
                if ticker not in today.index:
                    continue
                price = float(today.loc[ticker, "close"])
                current_value = positions[ticker] * price
                target_value = value * target.get(ticker, 0.0)
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
                    trade_rows.append([str(d)[:10], agent.name, ticker, "BUY", shares, price, fee, decision.note])
                elif shares < 0 and positions[ticker] >= abs(shares):
                    positions[ticker] += shares
                    cash -= cost
                    cash -= fee
                    trade_rows.append([str(d)[:10], agent.name, ticker, "SELL", abs(shares), price, fee, decision.note])
        today_value = cash + sum(positions.get(t, 0) * today.loc[t, "close"] for t in positions if t in today.index)
        equity_rows.append([str(d)[:10], agent.name, today_value, cash])

    equity = pd.DataFrame(equity_rows, columns=["date", "agent", "equity", "cash"])
    trades = pd.DataFrame(trade_rows, columns=["date", "agent", "ticker", "side", "shares", "price", "fee", "note"])
    return equity, trades


def performance_metrics(equity: pd.DataFrame, risk_free_rate=0.0):
    equity = equity.copy()
    equity["date"] = pd.to_datetime(equity["date"])
    out = []
    for agent, g in equity.groupby("agent"):
        g = g.sort_values("date")
        r = g["equity"].pct_change().dropna()
        if len(r) == 0:
            continue
        total_return = g["equity"].iloc[-1] / g["equity"].iloc[0] - 1
        ann_return = (1 + total_return) ** (252 / max(1, len(r))) - 1
        ann_vol = r.std() * np.sqrt(252)
        sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 1e-12 else np.nan
        running_max = g["equity"].cummax()
        drawdown = g["equity"] / running_max - 1
        max_drawdown = drawdown.min()
        out.append([agent, total_return, ann_return, ann_vol, sharpe, max_drawdown, len(r)])
    return pd.DataFrame(out, columns=["agent", "total_return", "annual_return", "annual_volatility", "sharpe", "max_drawdown", "days"])

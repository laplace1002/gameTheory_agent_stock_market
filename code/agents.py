from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class AgentDecision:
    agent: str
    date: str
    target_weights: dict
    note: str


class BaseAgent:
    name = "BaseAgent"

    def target_weights(self, history: pd.DataFrame, cash: float, positions: dict) -> AgentDecision:
        raise NotImplementedError

    def _normalize(self, scores: dict, long_only=True, max_weight=0.35):
        if not scores:
            return {}
        cleaned = {}
        for k, v in scores.items():
            if np.isnan(v) or np.isinf(v):
                v = 0
            cleaned[k] = max(0, v) if long_only else v
        s = sum(abs(v) for v in cleaned.values())
        if s <= 1e-12:
            return {k: 0 for k in cleaned}
        weights = {k: min(max_weight, abs(v) / s) for k, v in cleaned.items()}
        total = sum(weights.values())
        if total > 0.95:
            weights = {k: v / total * 0.95 for k, v in weights.items()}
        return weights


class MomentumAgent(BaseAgent):
    name = "MomentumAgent"

    def target_weights(self, history, cash, positions):
        latest_date = str(history["date"].max())[:10]
        scores = {}
        for ticker, g in history.groupby("ticker"):
            g = g.sort_values("date")
            if len(g) < 40:
                scores[ticker] = 0
                continue
            ret_20 = g["close"].iloc[-1] / g["close"].iloc[-21] - 1
            ret_60 = g["close"].iloc[-1] / g["close"].iloc[-61] - 1 if len(g) >= 70 else ret_20
            scores[ticker] = 0.7 * ret_20 + 0.3 * ret_60
        return AgentDecision(self.name, latest_date, self._normalize(scores), "追涨型：偏好近期上涨更强的股票。")


class MeanReversionAgent(BaseAgent):
    name = "MeanReversionAgent"

    def target_weights(self, history, cash, positions):
        latest_date = str(history["date"].max())[:10]
        scores = {}
        for ticker, g in history.groupby("ticker"):
            g = g.sort_values("date")
            if len(g) < 25:
                scores[ticker] = 0
                continue
            ma20 = g["close"].rolling(20).mean().iloc[-1]
            price = g["close"].iloc[-1]
            scores[ticker] = max(0, (ma20 - price) / ma20)
        return AgentDecision(self.name, latest_date, self._normalize(scores), "均值回归型：偏好短期跌到均线下方的股票。")


class LowVolatilityAgent(BaseAgent):
    name = "LowVolatilityAgent"

    def target_weights(self, history, cash, positions):
        latest_date = str(history["date"].max())[:10]
        scores = {}
        for ticker, g in history.groupby("ticker"):
            g = g.sort_values("date")
            if len(g) < 45:
                scores[ticker] = 0
                continue
            r = g["close"].pct_change().dropna()
            vol = r.tail(40).std()
            trend = g["close"].iloc[-1] / g["close"].iloc[-41] - 1
            scores[ticker] = max(0, trend) / (vol + 1e-6)
        return AgentDecision(self.name, latest_date, self._normalize(scores), "稳健型：偏好波动较低且趋势不差的股票。")


class DrawdownBuyerAgent(BaseAgent):
    name = "DrawdownBuyerAgent"

    def target_weights(self, history, cash, positions):
        latest_date = str(history["date"].max())[:10]
        scores = {}
        for ticker, g in history.groupby("ticker"):
            g = g.sort_values("date")
            if len(g) < 120:
                scores[ticker] = 0
                continue
            high = g["close"].rolling(120).max().iloc[-1]
            price = g["close"].iloc[-1]
            drawdown = (high - price) / high
            rebound = g["close"].iloc[-1] / g["close"].iloc[-6] - 1 if len(g) >= 10 else 0
            scores[ticker] = max(0, drawdown) * (1 + max(0, rebound))
        return AgentDecision(self.name, latest_date, self._normalize(scores), "逢低型：偏好相对高点回撤较大的股票。")


class RandomAgent(BaseAgent):
    name = "RandomAgent"

    def __init__(self, seed=7):
        self.rng = np.random.default_rng(seed)

    def target_weights(self, history, cash, positions):
        latest_date = str(history["date"].max())[:10]
        tickers = sorted(history["ticker"].unique())
        scores = {t: float(self.rng.random()) for t in tickers}
        return AgentDecision(self.name, latest_date, self._normalize(scores), "基准型：随机分配权重，用于对照。")


class CommitteeTeamAgent(BaseAgent):
    name = "CommitteeTeamAgent"

    def __init__(self, members=None):
        self.members = members or [MomentumAgent(), MeanReversionAgent(), LowVolatilityAgent(), DrawdownBuyerAgent()]

    def target_weights(self, history, cash, positions):
        latest_date = str(history["date"].max())[:10]
        all_weights = []
        for member in self.members:
            all_weights.append(member.target_weights(history, cash, positions).target_weights)
        tickers = sorted(history["ticker"].unique())
        avg = {t: float(np.mean([w.get(t, 0) for w in all_weights])) for t in tickers}
        return AgentDecision(self.name, latest_date, self._normalize(avg), "组织型：多个 agent 先独立判断，再平均投票形成团队决策。")


class DynamicTeamAgent(BaseAgent):
    name = "DynamicTeamAgent"

    def __init__(self, members=None):
        self.members = members or [MomentumAgent(), MeanReversionAgent(), LowVolatilityAgent(), DrawdownBuyerAgent()]
        self.member_scores = {m.name: 1.0 for m in self.members}

    def update_scores(self, recent_returns: dict):
        for name, value in recent_returns.items():
            self.member_scores[name] = max(0.2, self.member_scores.get(name, 1.0) * (1 + value))

    def target_weights(self, history, cash, positions):
        latest_date = str(history["date"].max())[:10]
        tickers = sorted(history["ticker"].unique())
        weighted = {t: 0.0 for t in tickers}
        score_sum = sum(self.member_scores.get(m.name, 1.0) for m in self.members)
        for member in self.members:
            w = member.target_weights(history, cash, positions).target_weights
            alpha = self.member_scores.get(member.name, 1.0) / score_sum
            for t in tickers:
                weighted[t] += alpha * w.get(t, 0)
        return AgentDecision(self.name, latest_date, self._normalize(weighted), "动态组织型：过去表现较好的 agent 在团队里获得更高权重。")

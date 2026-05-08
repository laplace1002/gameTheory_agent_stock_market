from dataclasses import dataclass
import numpy as np
import pandas as pd

from aggregation import clamp_signal, message_signal
from belief import BeliefState
from message import make_message


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


class CommunicatingAgent(BaseAgent):
    """Base class for agents that read/write market messages."""

    name = "CommunicatingAgent"

    def __init__(
        self,
        own_strategy=None,
        name: str | None = None,
        enabled_channels=None,
        message_probability: float = 1.0,
    ):
        self.own_strategy = own_strategy or MomentumAgent()
        if name is not None:
            self.name = name
        self.enabled_channels = list(enabled_channels or ["public", "private", "moments"])
        self.message_probability = float(max(0.0, min(1.0, message_probability)))
        self.belief = BeliefState(self.name)

    def generate_message(self, history, board, date: str, channel: str | None = None, receiver_ids=None):
        channel = self._choose_channel(channel)
        if channel is None or not self._should_publish(date):
            return None
        decision = self.own_strategy.target_weights(history, cash=0.0, positions={})
        ticker, weight = self._top_weight(decision.target_weights)
        if ticker is None:
            return None

        direction = "bullish" if weight > 0.02 else "neutral"
        confidence = min(0.9, 0.45 + weight)
        evidence = self._evidence(history, ticker, weight)
        return make_message(
            sender_id=self.name,
            timestamp=date,
            channel=channel,
            receiver_ids=receiver_ids,
            tickers=[ticker],
            direction=direction,
            confidence=confidence,
            evidence=evidence,
            position_intent="increase" if direction == "bullish" else "hold",
            natural_language=f"{self.name} reports a {direction} signal on {ticker}.",
            claim_type="trend",
        )

    def update_belief(self, board, reputation, current_date: str | None = None, channels=None, agent_groups=None) -> None:
        since = self._message_window_start(current_date)
        visible = board.get_visible(
            self.name,
            channels=channels or self.enabled_channels,
            since=since,
            agent_groups=agent_groups,
        )
        if not visible:
            return
        before = {
            (msg.sender_id, ticker): self.belief.get_belief(ticker)
            for msg in visible
            for ticker in msg.tickers
        }
        self.belief.update_from_messages(visible, reputation, current_date=current_date)
        for msg in visible:
            for ticker in msg.tickers:
                reputation.record_belief_change(
                    receiver_id=self.name,
                    sender_id=msg.sender_id,
                    ticker=ticker,
                    before=before[(msg.sender_id, ticker)],
                    after=self.belief.get_belief(ticker),
                    date=current_date,
                )

    def target_weights(self, history, cash, positions):
        own_decision = self.own_strategy.target_weights(history, cash, positions)
        latest_date = own_decision.date
        tickers = sorted(history["ticker"].unique())
        prior_beliefs = {ticker: self.belief.get_belief(ticker) for ticker in tickers}
        social_weights = self.belief.to_weights(tickers)
        own_signals = self._weights_to_signals(own_decision.target_weights, tickers)

        if sum(social_weights.values()) <= 1e-12:
            target = own_decision.target_weights
        else:
            target = self._blend_weights(own_decision.target_weights, social_weights, own_alpha=0.65, tickers=tickers)

        for ticker, own_signal in own_signals.items():
            combined_signal = clamp_signal(0.65 * own_signal + 0.35 * prior_beliefs.get(ticker, 0.0))
            self.belief.update_from_own_signal(ticker, combined_signal, latest_date)
        return AgentDecision(self.name, latest_date, self._normalize(target), f"{self.name}: own strategy blended with communicated beliefs.")

    def _choose_channel(self, requested_channel):
        if requested_channel in self.enabled_channels:
            return requested_channel
        if self.enabled_channels:
            return self.enabled_channels[0]
        return None

    def _message_window_start(self, current_date: str | None, days: int = 30) -> str | None:
        if current_date is None:
            return None
        return str((pd.Timestamp(current_date) - pd.Timedelta(days=days)))[:10]

    def _should_publish(self, date: str) -> bool:
        if self.message_probability >= 1.0:
            return True
        bucket = (pd.Timestamp(date).dayofyear + sum(ord(ch) for ch in self.name)) % 100
        return bucket < int(self.message_probability * 100)

    def _top_weight(self, weights: dict):
        if not weights:
            return None, 0.0
        ticker = max(weights, key=lambda key: weights.get(key, 0.0))
        return ticker, float(weights.get(ticker, 0.0))

    def _weights_to_signals(self, weights: dict, tickers: list[str]) -> dict:
        if not tickers:
            return {}
        values = np.array([float(weights.get(ticker, 0.0)) for ticker in tickers], dtype=float)
        center = float(values.mean()) if len(values) else 0.0
        scale = float(values.max() - values.min())
        if scale <= 1e-12:
            return {ticker: 0.0 for ticker in tickers}
        return {ticker: clamp_signal((float(weights.get(ticker, 0.0)) - center) / scale) for ticker in tickers}

    def _blend_weights(self, own_weights: dict, belief_weights: dict, own_alpha: float, tickers: list[str]) -> dict:
        return {
            ticker: own_alpha * float(own_weights.get(ticker, 0.0)) + (1.0 - own_alpha) * float(belief_weights.get(ticker, 0.0))
            for ticker in tickers
        }

    def _evidence(self, history, ticker: str, weight: float) -> list:
        ticker_history = history[history["ticker"] == ticker].sort_values("date")
        if len(ticker_history) < 2:
            return [{"type": "target_weight", "quality": min(1.0, max(0.1, weight))}]
        lookback = min(20, len(ticker_history) - 1)
        recent_return = ticker_history["close"].iloc[-1] / ticker_history["close"].iloc[-lookback - 1] - 1
        volatility = ticker_history["close"].pct_change().tail(lookback).std()
        return [
            {"type": "target_weight", "quality": min(1.0, max(0.1, weight))},
            {"type": "recent_return", "value": float(recent_return), "quality": min(1.0, abs(float(recent_return)) * 5.0)},
            {"type": "volatility", "value": float(volatility), "quality": max(0.1, min(1.0, 1.0 - float(volatility)))},
        ]


class TruthfulReporterAgent(CommunicatingAgent):
    """Publishes its own signal with minimal strategic bias."""

    def __init__(self, own_strategy=None):
        super().__init__(own_strategy=own_strategy or MomentumAgent(), name="TruthfulReporterAgent")


class PersuaderAgent(CommunicatingAgent):
    """Raises confidence on its preferred view to maximize influence."""

    def __init__(self, own_strategy=None):
        super().__init__(own_strategy=own_strategy or MeanReversionAgent(), name="PersuaderAgent")

    def generate_message(self, history, board, date: str, channel: str | None = None, receiver_ids=None):
        msg = super().generate_message(history, board, date, channel=channel, receiver_ids=receiver_ids)
        if msg is None:
            return None
        msg.confidence = min(1.0, max(0.8, msg.confidence + 0.15))
        msg.position_intent = "increase"
        msg.natural_language = f"{self.name} strongly recommends increasing exposure to {msg.tickers[0]}."
        return msg


class FreeRiderAgent(CommunicatingAgent):
    """Consumes messages but publishes only occasionally."""

    def __init__(self, own_strategy=None):
        super().__init__(
            own_strategy=own_strategy or LowVolatilityAgent(),
            name="FreeRiderAgent",
            message_probability=0.12,
        )


class ContrarianAgent(CommunicatingAgent):
    """Discounts crowded messages and leans against consensus."""

    def __init__(self, own_strategy=None):
        super().__init__(own_strategy=own_strategy or DrawdownBuyerAgent(), name="ContrarianAgent")

    def update_belief(self, board, reputation, current_date: str | None = None, channels=None, agent_groups=None) -> None:
        since = self._message_window_start(current_date)
        visible = board.get_visible(
            self.name,
            channels=channels or self.enabled_channels,
            since=since,
            agent_groups=agent_groups,
        )
        if not visible:
            return
        grouped = {}
        for msg in visible:
            for ticker in msg.tickers:
                grouped.setdefault(ticker, []).append(msg)
        for ticker, messages in grouped.items():
            consensus = float(np.mean([message_signal(msg, current_date=current_date) for msg in messages]))
            before = self.belief.get_belief(ticker)
            after = clamp_signal(0.5 * before - 0.5 * consensus)
            self.belief.update_from_own_signal(ticker, after, current_date)
            for msg in messages:
                reputation.record_belief_change(self.name, msg.sender_id, ticker, before, after, current_date)


class SocialGraphAgent(CommunicatingAgent):
    """Mixes an own strategy with incoming weighted influencers in the social graph."""

    def __init__(self, own_strategy=None, social_graph=None, all_agents=None, alpha=0.5):
        super().__init__(own_strategy=own_strategy or DynamicTeamAgent(), name="SocialGraphAgent")
        self.social_graph = social_graph
        self.all_agents = list(all_agents or [])
        self.alpha = float(max(0.0, min(1.0, alpha)))

    def target_weights(self, history, cash, positions):
        own_decision = self.own_strategy.target_weights(history, cash, positions)
        latest_date = own_decision.date
        tickers = sorted(history["ticker"].unique())
        agent_map = {agent.name: agent for agent in self.all_agents if agent is not self}
        influencer_weights = {ticker: 0.0 for ticker in tickers}

        influencers = self.social_graph.get_influencers(self.name) if self.social_graph is not None else []
        for influencer_name, graph_weight in influencers:
            influencer = agent_map.get(influencer_name)
            if influencer is None:
                continue
            decision = influencer.target_weights(history, cash, positions)
            for ticker in tickers:
                influencer_weights[ticker] += graph_weight * decision.target_weights.get(ticker, 0.0)

        if influencers:
            target = self._blend_weights(own_decision.target_weights, influencer_weights, own_alpha=1.0 - self.alpha, tickers=tickers)
        else:
            target = own_decision.target_weights

        belief_weights = self.belief.to_weights(tickers)
        if sum(belief_weights.values()) > 1e-12:
            target = self._blend_weights(target, belief_weights, own_alpha=0.75, tickers=tickers)

        own_signals = self._weights_to_signals(target, tickers)
        for ticker, signal in own_signals.items():
            self.belief.update_from_own_signal(ticker, signal, latest_date)

        return AgentDecision(self.name, latest_date, self._normalize(target), "社交图谱型：按有向图影响力混合团队与通信信念。")

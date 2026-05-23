from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json


@dataclass(frozen=True)
class StrategySpec:
    name: str
    family: str
    allowed_features: tuple[str, ...]
    trainable_params: dict = field(default_factory=dict)
    fixed_rules: tuple[str, ...] = field(default_factory=tuple)
    max_weight: float = 0.35
    rebalance_every: int = 5
    version: str = "v1"

    def param_hash(self) -> str:
        payload = {
            "name": self.name,
            "family": self.family,
            "trainable_params": self.trainable_params,
            "fixed_rules": self.fixed_rules,
            "max_weight": self.max_weight,
            "rebalance_every": self.rebalance_every,
            "version": self.version,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]

    def validate_action(self, action: dict) -> list[str]:
        weights = action.get("target_weights", {}) if isinstance(action, dict) else {}
        violations = []
        if not isinstance(weights, dict):
            return ["target_weights must be a dict"]

        gross = 0.0
        for ticker, value in weights.items():
            try:
                weight = float(value)
            except (TypeError, ValueError):
                violations.append(f"{ticker}: weight is not numeric")
                continue
            gross += abs(weight)
            if weight < -1e-12:
                violations.append(f"{ticker}: short weight {weight:.4f} violates long-only rule")
            if weight > self.max_weight + 1e-12:
                violations.append(f"{ticker}: weight {weight:.4f} exceeds max_weight {self.max_weight:.4f}")

        if gross > 0.950001:
            violations.append(f"gross exposure {gross:.4f} exceeds 0.9500 cash buffer rule")
        return violations


def default_strategy_spec(agent_name: str, rebalance_every: int = 5) -> StrategySpec:
    specs = {
        "MomentumAgent": StrategySpec(
            name="MomentumAgent",
            family="momentum",
            allowed_features=("close", "20d_return", "60d_return"),
            trainable_params={"ret_20_weight": 0.7, "ret_60_weight": 0.3},
            fixed_rules=("long_only", "rank_recent_winners", "cash_buffer"),
            rebalance_every=rebalance_every,
        ),
        "MeanReversionAgent": StrategySpec(
            name="MeanReversionAgent",
            family="mean_reversion",
            allowed_features=("close", "20d_moving_average"),
            trainable_params={"lookback": 20},
            fixed_rules=("long_only", "buy_below_moving_average", "cash_buffer"),
            rebalance_every=rebalance_every,
        ),
        "LowVolatilityAgent": StrategySpec(
            name="LowVolatilityAgent",
            family="low_volatility",
            allowed_features=("close", "40d_volatility", "40d_trend"),
            trainable_params={"vol_lookback": 40, "trend_floor": 0.0},
            fixed_rules=("long_only", "prefer_low_vol_positive_trend", "cash_buffer"),
            rebalance_every=rebalance_every,
        ),
        "DrawdownBuyerAgent": StrategySpec(
            name="DrawdownBuyerAgent",
            family="drawdown_value",
            allowed_features=("close", "120d_high", "5d_rebound"),
            trainable_params={"drawdown_lookback": 120, "rebound_lookback": 5},
            fixed_rules=("long_only", "buy_large_drawdowns", "cash_buffer"),
            rebalance_every=rebalance_every,
        ),
        "RandomAgent": StrategySpec(
            name="RandomAgent",
            family="random_baseline",
            allowed_features=("ticker_universe",),
            trainable_params={"seeded_rng": True},
            fixed_rules=("long_only", "seeded_random_scores", "cash_buffer"),
            rebalance_every=rebalance_every,
        ),
        "CommitteeTeamAgent": StrategySpec(
            name="CommitteeTeamAgent",
            family="committee",
            allowed_features=("member_target_weights",),
            trainable_params={"aggregation": "equal_average"},
            fixed_rules=("long_only", "average_fixed_member_strategies", "cash_buffer"),
            rebalance_every=rebalance_every,
        ),
        "DynamicTeamAgent": StrategySpec(
            name="DynamicTeamAgent",
            family="dynamic_committee",
            allowed_features=("member_target_weights", "recent_member_returns"),
            trainable_params={"score_floor": 0.2},
            fixed_rules=("long_only", "weight_fixed_members_by_score", "cash_buffer"),
            rebalance_every=rebalance_every,
        ),
        "TruthfulReporterAgent": StrategySpec(
            name="TruthfulReporterAgent",
            family="communicating_momentum",
            allowed_features=("own_signal", "visible_messages", "reputation"),
            trainable_params={"own_alpha": 0.65},
            fixed_rules=("long_only", "blend_own_strategy_with_beliefs", "cash_buffer"),
            rebalance_every=rebalance_every,
        ),
        "PersuaderAgent": StrategySpec(
            name="PersuaderAgent",
            family="communicating_mean_reversion",
            allowed_features=("own_signal", "visible_messages", "reputation"),
            trainable_params={"own_alpha": 0.65, "message_confidence_bias": 0.15},
            fixed_rules=("long_only", "blend_own_strategy_with_beliefs", "cash_buffer"),
            rebalance_every=rebalance_every,
        ),
        "FreeRiderAgent": StrategySpec(
            name="FreeRiderAgent",
            family="communicating_low_volatility",
            allowed_features=("own_signal", "visible_messages", "reputation"),
            trainable_params={"own_alpha": 0.65, "message_probability": 0.12},
            fixed_rules=("long_only", "consume_more_than_publish", "cash_buffer"),
            rebalance_every=rebalance_every,
        ),
        "ContrarianAgent": StrategySpec(
            name="ContrarianAgent",
            family="communicating_contrarian",
            allowed_features=("own_signal", "visible_messages", "message_consensus"),
            trainable_params={"consensus_alpha": -0.5},
            fixed_rules=("long_only", "discount_crowded_messages", "cash_buffer"),
            rebalance_every=rebalance_every,
        ),
        "SocialGraphAgent": StrategySpec(
            name="SocialGraphAgent",
            family="social_graph",
            allowed_features=("own_signal", "influencer_weights", "visible_messages", "pagerank"),
            trainable_params={"graph_alpha": 0.55},
            fixed_rules=("long_only", "mix_fixed_strategy_with_graph_influence", "cash_buffer"),
            rebalance_every=rebalance_every,
        ),
    }
    return specs.get(
        agent_name,
        StrategySpec(
            name=agent_name,
            family="custom",
            allowed_features=("close",),
            trainable_params={},
            fixed_rules=("long_only", "cash_buffer"),
            rebalance_every=rebalance_every,
        ),
    )


def strategy_spec_for_agent(agent, rebalance_every: int = 5) -> StrategySpec:
    spec = getattr(agent, "strategy_spec", None)
    if isinstance(spec, StrategySpec):
        return spec
    return default_strategy_spec(getattr(agent, "name", agent.__class__.__name__), rebalance_every=rebalance_every)

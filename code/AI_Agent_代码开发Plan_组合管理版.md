# AI Agent 交易平台代码开发 Plan：Portfolio of Agents 版本

## 核心目标

把现有的社交交易平台升级为一个“agent 组合管理”研究平台。投资者不只是直接管理股票组合，而是管理多个固定策略 agent 的资本权重。每个 agent 坚持自己的 strategy spec；训练只允许在自己的参数空间里校准，不能因为市场波动而改变身份。

## 当前代码包状态

已有模块：

- `agents.py`：Momentum、MeanReversion、LowVolatility、DrawdownBuyer、Random、CommitteeTeam、DynamicTeam、TruthfulReporter、Persuader、FreeRider、Contrarian、SocialGraphAgent。
- `aggregation.py`：`reputation_weighted`、`degroot_consensus`、`black_litterman`、`hedge_weights`。
- `run_experiment.py`：baseline/public/private/moments/full_social，以及多个 social scenario。
- `dashboard_app.py`：统一事件流、ChatLab、好友申请、朋友圈、实时 LLM mode、本地 replay mode。

关键缺口：

1. 缺少上层 `AgentPortfolioManager`。
2. 缺少 agent 收益相关矩阵和分散化指标。
3. 缺少 walk-forward training / frozen execution 协议。
4. 缺少单一 generalist LLM/open model、普通策略组合和 agent 组合之间的对照实验。
5. Dashboard 缺少 Portfolio Hall。

## 开发阶段

### P1. 固定策略契约

新增 `code/strategy_spec.py`：

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class StrategySpec:
    name: str
    family: str
    allowed_features: tuple[str, ...]
    trainable_params: dict
    fixed_rules: tuple[str, ...]
    max_weight: float = 0.35
    rebalance_every: int = 5
    version: str = "v1"

    def validate_action(self, action: dict) -> list[str]:
        ...
```

验收：每个 agent 的每次决策都记录 `strategy_spec_version` 和 `param_hash`。若 LLM 或策略动作违反固定规则，写入 `drift_log.csv`。

### P2. Walk-forward training

新增 `code/strategy_training.py`：

- train window：选择 agent 自己的参数。
- validation window：选择 best params。
- test window：参数冻结，只执行。

输出：`training_params.csv`。

### P3. Agent equity 与 manager equity 分离

新增输出：

```text
agent_equity_curve.csv
agent_return_history.csv
agent_return_correlation.csv
meta_weight_history.csv
manager_equity_curve.csv
```

### P4. AgentPortfolioManager

新增 `code/agent_portfolio.py`：

```python
class AgentPortfolioManager:
    def allocate(self, date, agent_metrics, agent_returns, corr_matrix, previous_weights):
        raise NotImplementedError

class EqualAgentManager(AgentPortfolioManager): ...
class HedgeAgentManager(AgentPortfolioManager): ...
class CorrelationAwareAgentManager(AgentPortfolioManager): ...
class BlackLittermanAgentManager(AgentPortfolioManager): ...
```

方法来源：

- Equal：基准。
- Hedge：Freund & Schapire / online expert advice。
- Correlation-aware：Markowitz mean-variance。
- Black-Litterman：Black-Litterman views。

### P5. 对照实验网格

新增 `config/experiment_grid.yaml`：

```yaml
experiments:
  - id: B0_equal_asset
    type: asset_baseline
  - id: B1_single_agents
    type: single_agent
  - id: B2_single_generalist_llm
    type: llm_generalist
  - id: B3_strategy_portfolio_no_agent
    type: strategy_portfolio
  - id: A1_equal_agent_portfolio
    type: agent_portfolio
    manager: equal
  - id: A2_hedge_agent_portfolio
    type: agent_portfolio
    manager: hedge
  - id: A3_correlation_agent_portfolio
    type: agent_portfolio
    manager: correlation_aware
  - id: A4_full_social_agent_portfolio
    type: agent_portfolio
    manager: hedge
    communication: full_social
```

### P6. 评价指标

新增 `code/evaluation.py`：

- total return
- annual return
- annual volatility
- Sharpe
- max drawdown
- turnover
- transaction cost drag
- agent return correlation
- diversification ratio
- regret to best agent
- HHI weight concentration
- strategy drift score
- calibration error

### P7. LLM constrained outputs

新增 `code/schemas.py`，所有 LLM 输出必须满足 JSON Schema。无效输出回退到 HOLD。

### P8. Dashboard Portfolio Hall

在 `dashboard_app.py` 新增：

- manager equity curves
- meta weight history
- agent correlation heatmap
- diversification metrics
- training params table
- drift alerts
- ablation comparison

## MVP 优先级

最小可行版本只做：

1. `StrategySpec`
2. `EqualAgentManager`
3. `HedgeAgentManager`
4. `agent_return_correlation.csv`
5. `manager_equity_curve.csv`
6. `experiment_comparison.csv`
7. dashboard 中的 Portfolio Hall 基础版

这样已经可以回答：个人投资者能否通过 AI agent 团队管理 portfolio of agents，并利用低相关策略改善风险调整收益。

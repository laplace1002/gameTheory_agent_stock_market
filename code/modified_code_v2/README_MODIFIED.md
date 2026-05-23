# 同步交易大厅 / ChatLab 改版说明

这版把 dashboard 从静态图表改成了统一事件流驱动的同步界面。交易大厅、ChatLab、好友申请列表和 PnL 状态可以读取同一个事件游标。

当前默认模式是 **LLM API 实时交互**：ChatLab 不再依赖模拟消息数据集。左侧边栏的 `一键启动所有 Agent` 会并发调用所有 Agent 的 LLM API，每个 Agent 在同一轮里生成交易、公聊、私聊、朋友圈和好友申请等完整行为，并写入实时统一事件流。旧的 `outputs/tables/unified_event_log.csv` 仍保留为“本地事件回放”模式。

## 主要新增文件

- `outputs/tables/unified_event_log.csv`：统一事件时间轴，包含交易、PnL、群聊、私聊、朋友圈、好友申请、接受/拒绝、社交策略选择。
- `outputs/tables/agent_state_history.csv`：每个日期每个 Agent 的 cash、equity、PnL、持仓、好友数、最新动作。
- `outputs/tables/strategy_choice_history.csv`：每个 rebalance tick 的 cooperate / compete / observe / independent 策略效用与选择原因。
- `code/strategy_spec.py`：固定策略契约，记录每个 agent 的 spec version、参数 hash，并校验漂移。
- `code/agent_portfolio.py`：agent 组合管理层，包含 Equal 和 Hedge 两个 meta-manager。
- `outputs/tables/agent_return_correlation.csv`：agent 日收益相关矩阵。
- `outputs/tables/manager_equity_curve.csv`：agent portfolio manager 的权益曲线。
- `outputs/tables/meta_weight_history.csv`：manager 分配到各 agent 的权重历史。
- `outputs/tables/experiment_comparison.csv`：single agent 与 agent portfolio 的统一指标对比。
- `outputs/tables/drift_log.csv`：违反固定策略契约的动作日志；空表表示没有漂移告警。
- `config/social_scenarios.yaml`：不同社交图谱场景配置。
- `config/experiment_grid.yaml`：研究实验网格，列出基准、单 agent、普通策略组合与 agent portfolio 对照组。
- `data/sample_synthetic_prices.csv`：离线可运行的合成行情样例。

## 运行

```bash
pip install -r requirements.txt
streamlit run code/dashboard_app.py
```

实时 LLM 模式只从 `.env` / `env.txt` 文件或系统环境变量读取配置，页面侧边栏不会显示 API 配置项。推荐在 `code/modified_code_v2/env.txt` 中放：

```dotenv
OPENAI_API_KEY=你的 API key
OPENAI_MODEL=你的模型名
OPENAI_BASE_URL=https://api.openai.com/v1
LLM_TEMPERATURE=0.7
LLM_TIMEOUT=60
```

`.env` 和 `env.txt` 都已加入 `.gitignore`，不要提交到 GitHub。

如果要看旧的离线回放，再运行：

```bash
python code/run_experiment.py --experiment full_social --scenario core_periphery --out outputs --prices data/sample_synthetic_prices.csv
```

## Dashboard 操作

侧边栏先选择数据源：

- `LLM API 实时交互`：默认模式，ChatLab 中的 agent 消息由 LLM API 实时生成。
- `本地事件回放`：读取 `outputs/tables/*.csv` 的离线实验结果。

在左侧边栏的 `LLM Agent 总控` 中：

- 点击 `一键启动所有 Agent`：立即并发调用所有 Agent，每个 Agent 生成一套完整行为包。
- 打开 `持续自动运行全部行为`：按设置的间隔持续运行完整行为轮次，让统一事件流自动增长。
- 点击 `停止自动生成`：停止后续自动轮次。
- 点击 `继续上一次生成`：继续当前生成目录，不新建 run 文件夹。

每次点击 `一键启动所有 Agent` 都会创建一个新的目录：

```text
outputs/live_state/runs/run_YYYYMMDD_HHMMSS_xxxxxx/
```

目录中会保存该次启动后的 round 数据、原始 LLM plans、交易记录、聊天记录、好友申请和社交图谱。`停止自动生成` 不会删除这个目录；`继续上一次生成` 会继续写入同一个目录。

每个完整行为包包含：

- 交易：BUY / SELL / HOLD，并更新 cash、equity、PnL 和持仓。
- 群聊：写入 public ChatLab。
- 私聊：写入 private ChatLab；如果双方还不是好友，会触发好友申请和通过事件。
- 朋友圈：写入 moments，并按当前好友关系决定可见范围。
- 好友申请：更新好友表和社交图谱。

`ChatLab` 页内只保留折叠的 `单 Agent 调试`，用于临时测试一个 Agent 的单条消息；正常自动交互应使用边栏总控。

实时模式会从 YAML 场景加载初始社交图谱。之后私聊、好友申请、通过好友等事件会动态更新社交图谱；实时事件流、好友关系、群组和最终社交图谱会保存到 `outputs/live_state/`。该目录已被 `.gitignore` 忽略，不会提交到 GitHub。刷新 dashboard 后会从这个状态继续，不会重新从空会话开始。

播放模式：

- `最新状态`：直接看最新事件。
- `手动回放`：拖动事件游标，交易大厅和 ChatLab 会同步跳到同一时点。
- `实时滚动`：按统一事件流自动前进；可设置每次前进事件数、刷新间隔、是否循环，以及聊天/事件窗是否自动滚到底部。

ChatLab 中点击左侧 Agent 按钮后，可以查看：

- 群聊：所有 public 消息。
- 私聊：选择私聊对象，查看双方可见的 private 消息。
- 朋友圈：查看该 Agent 与好友可见的 moments。
- 好友申请：按申请聚合展示。发起和通过不会拆成两条；同一条中会显示发送时间、通过/处理时间和当前状态。

## 社交策略

每个 Agent 在 rebalance tick 会根据好友数、可见消息数、PageRank、声誉、影响力、收益和图密度估计四种策略效用：

- cooperate：合作共享，偏向私聊/融合信息。
- compete：竞争影响力，偏向群聊并提高信心表达。
- observe：少发言，多读取信息。
- independent：社交边际收益不足，减少发言并主要靠自身交易策略。

这些选择会写入 `strategy_choice_history.csv`，并作为 `strategy_choice` 事件进入 dashboard 的同步事件流。

## Portfolio of Agents

离线实验现在会额外生成 agent 组合管理结果。每个 agent 仍独立执行自己的固定策略，组合管理器只在上层分配资本权重：

- `equal_agent_portfolio`：等权分配到所有 agent。
- `hedge_agent_portfolio`：根据 agent 累计收益和回撤惩罚做在线专家权重。

Dashboard 中的 `Portfolio Hall` 可以查看 manager 权益曲线、meta weight 历史、agent 收益相关热力图、策略契约版本和漂移告警。

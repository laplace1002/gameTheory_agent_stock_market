# 同步交易大厅 / ChatLab 改版说明

这版把 dashboard 从静态图表改成了统一事件流驱动的同步界面。交易大厅、ChatLab、好友申请列表和 PnL 状态可以读取同一个事件游标。

当前默认模式是 **LLM API 实时交互**：ChatLab 不再依赖模拟消息数据集。左侧边栏的 `一键启动所有 Agent` 会并发调用所有 Agent 的 LLM API，每个 Agent 在同一轮里生成交易、公聊、私聊、朋友圈和好友申请等完整行为，并写入实时统一事件流。旧的 `outputs/tables/unified_event_log.csv` 仍保留为“本地事件回放”模式。

实时 LLM 模式的股票池和交易价格默认读取 `data/TRD_Dalyr.xlsx`。Dashboard 会把 Excel 中的真实 ticker 作为可交易范围，并按轮次推进到对应历史交易日，用当日 close 做成交校验、持仓估值和 LLM prompt 中的行情截面。需要切换行情文件时，可在环境变量中设置 `LIVE_MARKET_DATA_PATH`。

## 主要新增文件

- `outputs/tables/unified_event_log.csv`：统一事件时间轴，包含交易、PnL、群聊、私聊、朋友圈、好友申请、接受/拒绝、社交策略选择。
- `outputs/tables/agent_state_history.csv`：每个日期每个 Agent 的 cash、equity、PnL、持仓、好友数、最新动作。
- `outputs/tables/strategy_choice_history.csv`：每个 rebalance tick 的 cooperate / compete / observe / independent 策略效用与选择原因。
- `code/strategy_spec.py`：固定策略契约，记录每个 agent 的 spec version、参数 hash，并校验漂移。
- `code/agent_portfolio.py`：agent 组合管理层，包含 Equal 和 Hedge 两个 meta-manager。
- `code/strategy_training.py`：walk-forward 参数训练协议，只训练允许参数，不改变 agent 固定策略身份。
- `code/scoring.py`：用 Brier score / log score 评价 agent forecast，并回写 proper scoring reputation。
- `code/view_extractor.py`：把 agent 消息转成 Black-Litterman views，再输出基于观点的资产权重。
- `code/schemas.py`：LLM/结构化动作 schema 校验工具。
- `outputs/tables/agent_return_correlation.csv`：agent 日收益相关矩阵。
- `outputs/tables/manager_equity_curve.csv`：agent portfolio manager 的权益曲线。
- `outputs/tables/meta_weight_history.csv`：manager 分配到各 agent 的权重历史。
- `outputs/tables/manager_loss_history.csv`：online Hedge / risk manager 的 loss 分解，包括 log loss、drawdown penalty 和 CVaR penalty。
- `outputs/tables/training_params.csv`：每个可训练策略的候选参数、冻结参数和训练窗口。
- `outputs/tables/forecast_scores.csv`：每条可验证 forecast 的 Brier score、log score 和 proper score。
- `outputs/tables/agent_views.csv`：从消息中提取出的结构化 agent views。
- `outputs/tables/bl_agent_view_weights.csv`：Black-Litterman posterior return 与资产权重。
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
LLM_MAX_WORKERS=3
```

`.env` 和 `env.txt` 都已加入 `.gitignore`，不要提交到 GitHub。

如果要看旧的离线回放，再运行：

```bash
python code/run_experiment.py --experiment full_social --scenario core_periphery --out outputs --prices data/TRD_Dalyr.xlsx
```

`data/TRD_Dalyr.xlsx` 是 CSMAR 风格日行情文件。`data_loader.py` 会把 `Stkcd / Trddt / Opnprc / Hiprc / Loprc / Clsprc / Dnshrtrd` 标准化为实验需要的 `ticker / date / open / high / low / close / volume`。

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

如果实时模式出现 API timeout，单个超时 Agent 会自动回退为 HOLD，不会中断整轮。频繁超时时，建议把 `env.txt` 里的 `LLM_TIMEOUT` 调到 `120`，并把 `LLM_MAX_WORKERS` 调到 `3`，减少并发请求压力。

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
- `hedge_agent_portfolio`：按 online expert aggregation 更新 agent 权重。
- `correlation_aware_agent_portfolio`：惩罚与其他 agent 高相关的资本分配。
- `drawdown_constrained_agent_portfolio`：加入单 agent 权重上限、回撤和 CVaR 惩罚。

Dashboard 中的 `Portfolio Hall` 可以查看 manager 权益曲线、meta weight 历史、agent 收益相关热力图、策略契约版本、walk-forward 训练参数、Hedge loss 分解、proper scoring 和 Black-Litterman agent views。

## 从实时 LLM run 生成报告研究表

如果已经用 dashboard 跑完 12 个实时社交图谱场景，可以一键把这些 live 结果转成 Portfolio Hall 和论文报告可用的分类输出：

```bash
/Users/ella/miniforge3/bin/python code/build_live_research_outputs.py \
  --runs-root /Users/ella/Desktop/agent_runs/不同社交图谱下多agent实时记录 \
  --output-root /Users/ella/Desktop/agent_runs/研究汇总_资源受限个人投资者AI_Agent组合管理
```

输出目录结构：

- `01_per_scenario/<scenario>/tables/`：每个社交图谱的 Portfolio Hall 兼容表，可在 dashboard 的“本地事件回放”中选择。
- `02_cross_scenario_comparison/`：跨社交图谱的 manager、single agent 和社交活动比较 CSV。
- `03_figures/`：报告可直接引用的跨场景图。
- `04_report_tables/README_REPORT_OUTPUTS.md`：报告写作时优先看的摘要表。

## 不开 Dashboard 批量跑实时 LLM 场景

如果要直接跑完 12 个社交图谱，每个图谱 50 轮，并把每个 run 保存到 `outputs/live_state/runs/`：

```bash
/Users/ella/miniforge3/bin/python code/run_live_social_grid.py \
  --prices data/TRD_Dalyr.xlsx \
  --config config/social_scenarios.yaml \
  --out outputs/live_state/runs \
  --scenario all \
  --rounds 50 \
  --max-workers 2 \
  --friend-decision llm \
  --agent-retries 5 \
  --repair-retries 2 \
  --round-retries 0 \
  --retry-backoff 5 \
  --max-retries 3 \
  --resume
```

这个脚本按社交图谱顺序串行执行：一个 scenario 完整跑完后才会进入下一个。默认是严格有效 run 模式：每一轮必须 12 个 agent 都成功返回可解析 JSON 并参与本轮；任一 agent 或好友判断中途报错，会先按 `--agent-retries` 原地重试该单次调用。若该轮仍失败，脚本会回滚到本轮开始前并继续重跑同一轮；`--round-retries 0` 表示一直重试到该轮通过，不删除整个 run。

每个有效 scenario 会创建独立 run 目录，结构与 dashboard 实时 run 一致：`metadata.json`、`rounds/round_*/` 和 `tables/`。脚本还会在 `outputs/live_state/runs/RUN_RECORDS.md` 中记录每个 run_id 对应的社交图谱、尝试次数、完成轮数和路径。

失败 round 的重试记录会写入该 run 目录下的 `ROUND_RETRY_LOG.md`。

如果中途失败或手动停止，推荐用同一条命令加 `--resume` 继续。它会先读取 `RUN_RECORDS.md` 跳过已经完成 50 轮的 scenario；如果当前 scenario 有未完成 run 目录，则读取该 run 的 `tables/`，恢复事件、持仓、好友图谱和已完成轮数，从下一轮继续。也可以用 `--start-at barbell_two_camps` 从指定场景开始。

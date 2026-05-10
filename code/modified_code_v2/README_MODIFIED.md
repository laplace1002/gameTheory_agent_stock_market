# 同步交易大厅 / ChatLab 改版说明

这版把 dashboard 从静态图表改成了统一事件流驱动的同步界面。交易大厅、ChatLab、好友申请列表和 PnL 状态都读取同一个 `outputs/tables/unified_event_log.csv` 游标。

## 主要新增文件

- `outputs/tables/unified_event_log.csv`：统一事件时间轴，包含交易、PnL、群聊、私聊、朋友圈、好友申请、接受/拒绝、社交策略选择。
- `outputs/tables/agent_state_history.csv`：每个日期每个 Agent 的 cash、equity、PnL、持仓、好友数、最新动作。
- `outputs/tables/strategy_choice_history.csv`：每个 rebalance tick 的 cooperate / compete / observe / independent 策略效用与选择原因。
- `config/social_scenarios.yaml`：不同社交图谱场景配置。
- `data/sample_synthetic_prices.csv`：离线可运行的合成行情样例。

## 运行

```bash
pip install -r requirements.txt
python code/run_experiment.py --experiment full_social --scenario core_periphery --out outputs --prices data/sample_synthetic_prices.csv
streamlit run code/dashboard_app.py
```

## Dashboard 操作

侧边栏有三种播放模式：

- `最新状态`：直接看最新事件。
- `手动回放`：拖动事件游标，交易大厅和 ChatLab 会同步跳到同一时点。
- `实时滚动`：按统一事件流自动前进；可设置每次前进事件数、刷新间隔、是否循环，以及聊天/事件窗是否自动滚到底部。

ChatLab 中点击左侧 Agent 按钮后，可以查看：

- 群聊：所有 public 消息。
- 私聊：选择私聊对象，查看双方可见的 private 消息。
- 朋友圈：查看该 Agent 与好友可见的 moments。
- 好友申请：查看该 Agent 发送、收到、同意、拒绝的好友申请事件。

## 社交策略

每个 Agent 在 rebalance tick 会根据好友数、可见消息数、PageRank、声誉、影响力、收益和图密度估计四种策略效用：

- cooperate：合作共享，偏向私聊/融合信息。
- compete：竞争影响力，偏向群聊并提高信心表达。
- observe：少发言，多读取信息。
- independent：社交边际收益不足，减少发言并主要靠自身交易策略。

这些选择会写入 `strategy_choice_history.csv`，并作为 `strategy_choice` 事件进入 dashboard 的同步事件流。

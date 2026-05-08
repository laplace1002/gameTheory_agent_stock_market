# AI Agent 虚拟股票市场项目包

## 项目一句话简介

本项目搭建一个“虚拟股票市场实验平台”：多个 AI agent 使用纸面资金，在真实股票历史数据上进行模拟交易，并比较不同 agent 单独行动、团队协作、动态组织后的表现差异。

## 和上一版相比的主要修改

1. 研究场景从普通虚拟交易市场，改为更具体的“虚拟股票市场”。
2. 数据使用方式改为“真实股票历史行情数据 + 纸面资金交易”。
3. agent 不再只有买家、卖家、套利者、监管者，而是变成不同投资思想的 agent：
   - 动量型 agent
   - 均值回归型 agent
   - 低波动风控型 agent
   - 逢低型 agent
   - 随机基准 agent
   - 委员会团队 agent
   - 动态团队 agent
4. 增加“分开 vs 结合”的实验：比较单个 agent 和多个 agent 组成团队后的效果。
5. 增加可展示的平台：`code/dashboard_app.py` 可以用 Streamlit 展示成绩、资金曲线和交易记录。

## 文件结构

- `report/AI_Agent_虚拟股票市场项目报告_修改版.docx`：中文版学术报告，正文不包含代码。
- `code/`：Python 项目代码。
- `data/`：数据文件和数据说明。
- `outputs/`：示例实验结果，包括表格和图。
- `prompts/`：LLM agent 提示词模板。
- `config/`：实验参数配置。
- `docs/汇报提纲.md`：课堂汇报思路。

## 推荐运行步骤

### 第一步：安装依赖

```bash
pip install -r requirements.txt
```

### 第二步：下载真实数据

```bash
python code/download_real_data.py --tickers AAPL MSFT GOOGL AMZN NVDA TSLA --start 2022-01-01 --end 2025-12-31 --out data/market_prices.csv
```

### 第三步：运行实验

```bash
python code/run_experiment.py --prices data/market_prices.csv --out outputs
```

如果没有网络，也可以先跑离线演示数据：

```bash
python code/run_experiment.py --prices data/sample_synthetic_prices.csv --out outputs
```

### 第四步：展示平台

```bash
streamlit run code/dashboard_app.py
```

## 重要说明

本项目只用于课程展示、经济学模拟和 AI agent 行为研究。它不是投资系统，不构成投资建议。

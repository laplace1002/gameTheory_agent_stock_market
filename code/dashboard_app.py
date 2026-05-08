from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="AI Agent 虚拟股票市场实验平台", layout="wide")
st.title("AI Agent 虚拟股票市场实验平台")
st.caption("用途：展示多个 AI agent 在纸面资金约束下的交易成绩、风险和协作效果。")

out_dir = Path("outputs/tables")
metrics_path = out_dir / "performance_metrics.csv"
equity_path = out_dir / "equity_curve.csv"
trade_path = out_dir / "trade_log.csv"

if not metrics_path.exists() or not equity_path.exists():
    st.warning("还没有找到实验结果。请先运行：python code/run_experiment.py")
    st.stop()

metrics = pd.read_csv(metrics_path)
equity = pd.read_csv(equity_path)
trades = pd.read_csv(trade_path) if trade_path.exists() else pd.DataFrame()

c1, c2, c3 = st.columns(3)
best = metrics.sort_values("sharpe", ascending=False).iloc[0]
c1.metric("Sharpe 第一名", best["agent"], f"{best['sharpe']:.2f}")
c2.metric("总收益率", best["agent"], f"{best['total_return']:.2%}")
c3.metric("最大回撤", best["agent"], f"{best['max_drawdown']:.2%}")

st.subheader("资金曲线")
fig = px.line(equity, x="date", y="equity", color="agent")
st.plotly_chart(fig, use_container_width=True)

st.subheader("成绩对比表")
st.dataframe(metrics, use_container_width=True)

st.subheader("交易记录")
st.dataframe(trades.tail(200), use_container_width=True)

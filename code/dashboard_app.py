from pathlib import Path
import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def network_figure(edges: pd.DataFrame, centrality_scores: pd.DataFrame) -> go.Figure:
    if edges.empty:
        fig = go.Figure()
        fig.update_layout(height=520, showlegend=False)
        return fig

    nodes = sorted(set(edges["source"]) | set(edges["target"]))
    center = {row["agent"]: row.get("pagerank", 0.05) for _, row in centrality_scores.iterrows()} if not centrality_scores.empty else {}
    positions = {
        node: (
            math.cos(2 * math.pi * index / len(nodes)),
            math.sin(2 * math.pi * index / len(nodes)),
        )
        for index, node in enumerate(nodes)
    }
    edge_x = []
    edge_y = []
    for _, row in edges.iterrows():
        x0, y0 = positions[row["source"]]
        x1, y1 = positions[row["target"]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = [positions[node][0] for node in nodes]
    node_y = [positions[node][1] for node in nodes]
    sizes = [18 + 90 * float(center.get(node, 0.05)) for node in nodes]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="#8a8f98"), hoverinfo="none"))
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=nodes,
            textposition="top center",
            marker=dict(size=sizes, color=sizes, colorscale="Viridis", showscale=False),
            hoverinfo="text",
        )
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=520, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
    return fig


st.set_page_config(page_title="AI Agent 虚拟股票市场实验平台", layout="wide")
st.title("AI Agent 虚拟股票市场实验平台")

OUT_DIR = Path("outputs/tables")


def read_table(name: str) -> pd.DataFrame:
    path = OUT_DIR / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


metrics = read_table("performance_metrics.csv")
equity = read_table("equity_curve.csv")
trades = read_table("trade_log.csv")
messages = read_table("message_log.csv")
social_edges = read_table("social_graph_edges.csv")
reputation = read_table("reputation_scores.csv")
belief = read_table("belief_history.csv")
aggregation = read_table("aggregation_history.csv")
centrality = read_table("centrality_scores.csv")
market = read_table("market_history.csv")
registry = read_table("agent_registry.csv")

if metrics.empty or equity.empty:
    st.warning("还没有找到实验结果。请先运行：python code/run_experiment.py --experiment full_social")
    st.stop()

overview_tab, market_tab, agents_tab, chat_tab, network_tab, aggregation_tab, trades_tab, evaluation_tab = st.tabs(
    ["Overview", "Market", "Agents", "Chat Lab", "Belief Network", "Aggregation", "Trades", "Evaluation"]
)

with overview_tab:
    best = metrics.sort_values("sharpe", ascending=False).iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sharpe 第一名", best["agent"], f"{best['sharpe']:.2f}")
    c2.metric("总收益率", best["agent"], f"{best['total_return']:.2%}")
    c3.metric("最大回撤", best["agent"], f"{best['max_drawdown']:.2%}")
    c4.metric("Agent 数量", len(metrics["agent"].unique()))
    st.plotly_chart(px.line(equity, x="date", y="equity", color="agent"), width="stretch", key="overview_equity_curve")
    st.dataframe(metrics, width="stretch")

with market_tab:
    if market.empty:
        st.info("market_history.csv 尚未生成。")
    else:
        market["date"] = pd.to_datetime(market["date"])
        normalized = market.sort_values(["ticker", "date"]).copy()
        normalized["normalized_close"] = normalized.groupby("ticker")["close"].transform(lambda values: values / values.iloc[0])
        c1, c2 = st.columns([2, 1])
        c1.plotly_chart(px.line(normalized, x="date", y="normalized_close", color="ticker"), width="stretch", key="market_price_curve")
        returns = normalized.sort_values(["ticker", "date"]).copy()
        returns["return"] = returns.groupby("ticker")["close"].pct_change()
        volatility = returns.groupby("ticker")["return"].std().reset_index(name="volatility")
        c2.plotly_chart(px.bar(volatility, x="ticker", y="volatility"), width="stretch", key="market_volatility_bar")
        heatmap_data = returns.pivot_table(index="date", columns="ticker", values="return").tail(80)
        st.plotly_chart(
            px.imshow(heatmap_data.T, aspect="auto", color_continuous_scale="RdBu", zmin=-0.05, zmax=0.05),
            width="stretch",
            key="market_return_heatmap",
        )

with agents_tab:
    c1, c2 = st.columns([1, 2])
    if not registry.empty:
        c1.dataframe(registry, width="stretch", hide_index=True)
    latest_equity = equity.sort_values("date").groupby("agent").tail(1)
    c2.plotly_chart(px.bar(latest_equity, x="agent", y="equity", color="agent"), width="stretch", key="agents_latest_equity")
    st.plotly_chart(px.line(equity, x="date", y="equity", color="agent"), width="stretch", key="agents_equity_curve")

with chat_tab:
    if messages.empty:
        st.info("当前实验没有消息记录。")
    else:
        channels = sorted(messages["channel"].dropna().unique())
        selected_channels = st.multiselect("Channel", channels, default=channels)
        filtered = messages[messages["channel"].isin(selected_channels)] if selected_channels else messages
        st.dataframe(filtered.sort_values("timestamp", ascending=False), width="stretch", hide_index=True)
        st.plotly_chart(px.histogram(filtered, x="sender_id", color="channel", barmode="group"), width="stretch", key="chat_message_histogram")

with network_tab:
    c1, c2 = st.columns([2, 1])
    c1.plotly_chart(network_figure(social_edges, centrality), width="stretch", key="network_social_graph")
    if not social_edges.empty:
        c2.dataframe(social_edges.sort_values("weight", ascending=False), width="stretch", hide_index=True)
    if not centrality.empty:
        st.plotly_chart(px.bar(centrality, x="agent", y="pagerank", color="agent"), width="stretch", key="network_pagerank_bar")

with aggregation_tab:
    if aggregation.empty:
        st.info("aggregation_history.csv 尚未生成。")
    else:
        st.plotly_chart(
            px.line(aggregation, x="date", y="value", color="agent", line_dash="method"),
            width="stretch",
            key="aggregation_history_line",
        )
        hedge = aggregation[aggregation["method"] == "hedge_weight"]
        if not hedge.empty:
            st.plotly_chart(px.bar(hedge, x="agent", y="value", color="agent"), width="stretch", key="aggregation_hedge_bar")
    if not belief.empty:
        st.plotly_chart(px.line(belief, x="date", y="belief", color="agent", line_dash="ticker"), width="stretch", key="aggregation_belief_line")

with trades_tab:
    if trades.empty:
        st.info("当前实验没有交易记录。")
    else:
        trade_values = trades.copy()
        trade_values["notional"] = trade_values["shares"].abs() * trade_values["price"]
        turnover = trade_values.groupby("agent")["notional"].sum().reset_index(name="gross_notional")
        st.plotly_chart(px.bar(turnover, x="agent", y="gross_notional", color="agent"), width="stretch", key="trades_turnover_bar")
        st.dataframe(trades.sort_values("date", ascending=False).head(300), width="stretch", hide_index=True)

with evaluation_tab:
    c1, c2 = st.columns(2)
    c1.dataframe(metrics, width="stretch", hide_index=True)
    if reputation.empty:
        c2.info("当前实验没有声誉记录。")
    else:
        c2.dataframe(reputation, width="stretch", hide_index=True)
        st.plotly_chart(
            px.scatter(reputation, x="calibration_error", y="reputation", size="prediction_count", color="sender_id"),
            width="stretch",
            key="evaluation_reputation_scatter",
        )

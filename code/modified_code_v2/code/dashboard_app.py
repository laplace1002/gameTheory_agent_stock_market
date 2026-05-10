from __future__ import annotations

import ast
import html
import json
import math
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components


APP_CSS = """
<style>
:root {
    --bg: #f5f6f7;
    --panel: #ffffff;
    --ink: #1f2328;
    --muted: #75808a;
    --line: #e5e7eb;
    --wechat-green: #95ec69;
    --wechat-dark: #111827;
    --buy: #008a3d;
    --sell: #d92d20;
}
.block-container { padding-top: 1.2rem; max-width: 1500px; }
.main-title {
    padding: 16px 18px;
    border-radius: 18px;
    background: linear-gradient(135deg, #111827, #253044);
    color: white;
    margin-bottom: 14px;
}
.main-title h1 { margin: 0; font-size: 28px; }
.main-title p { margin: 6px 0 0 0; color: #d1d5db; }
.sync-bar {
    border: 1px solid var(--line);
    background: var(--panel);
    border-radius: 16px;
    padding: 12px 16px;
    margin: 10px 0 14px 0;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
}
.sync-event { font-size: 14px; color: var(--muted); }
.sync-event b { color: var(--ink); }
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
    margin-bottom: 14px;
}
.kpi-card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 14px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
}
.kpi-label { color: var(--muted); font-size: 12px; margin-bottom: 4px; }
.kpi-value { color: var(--ink); font-size: 22px; font-weight: 800; }
.kpi-sub { color: var(--muted); font-size: 12px; margin-top: 3px; }
.hall-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(255px, 1fr));
    gap: 12px;
}
.agent-card {
    background: #fff;
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 14px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
}
.agent-top { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.agent-avatar {
    width: 38px; height: 38px; border-radius: 12px;
    background: linear-gradient(135deg, #dbeafe, #c7d2fe);
    display: flex; align-items: center; justify-content: center;
    font-weight: 900; color: #1f2937;
}
.agent-name { font-weight: 800; color: var(--ink); line-height: 1.1; }
.agent-friends { color: var(--muted); font-size: 12px; }
.agent-row { display: flex; justify-content: space-between; gap: 8px; border-top: 1px solid #f0f2f5; padding: 6px 0; font-size: 13px; }
.agent-row span:first-child { color: var(--muted); }
.agent-row span:last-child { color: var(--ink); font-weight: 700; text-align: right; }
.pnl-pos { color: var(--buy) !important; }
.pnl-neg { color: var(--sell) !important; }
.last-action {
    margin-top: 10px;
    background: #f8fafc;
    border: 1px solid #edf1f5;
    border-radius: 12px;
    padding: 9px;
    min-height: 54px;
    color: #334155;
    font-size: 12px;
    line-height: 1.45;
}
.badge {
    display: inline-block; padding: 2px 7px; border-radius: 999px;
    font-size: 11px; font-weight: 800; margin-right: 4px;
}
.badge-buy { background: #dcfce7; color: #166534; }
.badge-sell { background: #fee2e2; color: #991b1b; }
.badge-chat { background: #dbeafe; color: #1d4ed8; }
.badge-social { background: #fef3c7; color: #92400e; }
.badge-hold { background: #f1f5f9; color: #475569; }
.badge-pnl { background: #ede9fe; color: #5b21b6; }
.wechat-layout {
    display: grid;
    grid-template-columns: minmax(230px, 290px) 1fr;
    gap: 14px;
    align-items: start;
}
.wechat-sidebar {
    background: #ededed;
    border: 1px solid #d9d9d9;
    border-radius: 18px;
    padding: 12px;
}
.wechat-sidebar-title {
    font-weight: 900; color: #111827; margin-bottom: 8px;
}
.agent-mini {
    display: flex; align-items: center; gap: 9px;
    background: white; border: 1px solid #e5e7eb; border-radius: 14px;
    padding: 8px; margin-bottom: 8px;
}
.agent-mini-avatar {
    width: 32px; height: 32px; border-radius: 10px;
    background: #dbeafe; display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 900;
}
.agent-mini-main { flex: 1; min-width: 0; }
.agent-mini-name { font-weight: 800; font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.agent-mini-sub { color: #6b7280; font-size: 11px; }
.section-panel {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 14px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
}
.small-muted { color: var(--muted); font-size: 12px; }
</style>
"""

COMPONENT_CSS = """
<style>
* { box-sizing: border-box; }
body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1f2328; }
.feed-panel {
    height: 100%; overflow-y: auto; padding: 10px; background: #0f172a;
    border-radius: 16px; border: 1px solid #1f2937;
}
.feed-row {
    background: #111827; color: #d1d5db; border: 1px solid #253044;
    border-radius: 12px; padding: 9px 10px; margin-bottom: 8px;
    font-size: 13px; line-height: 1.45;
}
.feed-row.current { border-color: #95ec69; box-shadow: 0 0 0 1px #95ec69 inset; }
.feed-time { color: #9ca3af; font-size: 11px; margin-bottom: 2px; }
.feed-agent { color: #fff; font-weight: 800; }
.feed-detail { margin-top: 3px; }
.badge { display: inline-block; padding: 2px 7px; border-radius: 999px; font-size: 11px; font-weight: 800; margin-right: 4px; }
.badge-buy { background: #dcfce7; color: #166534; }
.badge-sell { background: #fee2e2; color: #991b1b; }
.badge-chat { background: #dbeafe; color: #1d4ed8; }
.badge-social { background: #fef3c7; color: #92400e; }
.badge-hold { background: #f1f5f9; color: #475569; }
.badge-pnl { background: #ede9fe; color: #5b21b6; }
.wechat-phone {
    height: 100%; overflow: hidden; display: flex; flex-direction: column;
    background: #ededed; border: 1px solid #d7d7d7; border-radius: 18px;
}
.wechat-header {
    height: 46px; background: #f7f7f7; border-bottom: 1px solid #d7d7d7;
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; color: #111827; flex: 0 0 auto;
}
.wechat-body { flex: 1 1 auto; overflow-y: auto; padding: 14px; }
.wechat-input {
    height: 48px; background: #f7f7f7; border-top: 1px solid #d7d7d7;
    display: flex; align-items: center; gap: 8px; padding: 8px 12px; color: #9ca3af; font-size: 13px;
}
.bubble-wrap { display: flex; gap: 8px; align-items: flex-start; margin: 10px 0; }
.bubble-wrap.self { flex-direction: row-reverse; }
.avatar {
    width: 32px; height: 32px; border-radius: 8px; background: #c7d2fe;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 900; color: #1f2937; flex: 0 0 auto;
}
.bubble {
    max-width: 73%; border-radius: 12px; padding: 9px 10px;
    font-size: 13px; line-height: 1.45; word-break: break-word;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}
.bubble.received { background: white; border-top-left-radius: 3px; }
.bubble.sent { background: #95ec69; border-top-right-radius: 3px; }
.bubble-meta { color: #6b7280; font-size: 11px; margin-top: 5px; }
.chip { display: inline-block; padding: 2px 6px; border-radius: 999px; background: rgba(255,255,255,.6); color: #374151; font-size: 11px; margin: 0 4px 5px 0; }
.moment-post { display: flex; gap: 10px; padding: 12px 0; border-bottom: 1px solid #d9d9d9; }
.moment-avatar { width: 38px; height: 38px; border-radius: 8px; background: #c7d2fe; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 900; flex: 0 0 auto; }
.moment-name { color: #576b95; font-weight: 900; margin-bottom: 4px; }
.moment-text { color: #111827; font-size: 13px; line-height: 1.45; }
.moment-meta { color: #8a8a8a; font-size: 11px; margin-top: 6px; }
.request-card { background: white; border-radius: 12px; border: 1px solid #e5e7eb; padding: 10px; margin-bottom: 9px; display: flex; gap: 10px; align-items: center; }
.request-main { flex: 1; min-width: 0; }
.request-title { font-weight: 900; font-size: 13px; }
.request-detail { color: #6b7280; font-size: 12px; line-height: 1.35; margin-top: 2px; }
.status-accepted { color: #166534; font-weight: 900; }
.status-rejected { color: #991b1b; font-weight: 900; }
.status-pending { color: #92400e; font-weight: 900; }
.empty { color: #6b7280; text-align: center; padding: 60px 20px; }
</style>
"""


def main():
    st.set_page_config(page_title="同步交易大厅 / ChatLab", layout="wide")
    st.markdown(APP_CSS, unsafe_allow_html=True)
    st.markdown(
        "<div class='main-title'><h1>同步交易大厅 / ChatLab</h1>"
        "<p>同一个事件游标驱动交易、PnL、群聊、私聊、朋友圈和好友申请列表。</p></div>",
        unsafe_allow_html=True,
    )


    def read_table(out_dir: Path, name: str) -> pd.DataFrame:
        path = out_dir / name
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()


    def discover_output_dirs() -> list[Path]:
        candidates = []
        root = Path("outputs")
        if (root / "tables").exists():
            candidates.append(root / "tables")
        if root.exists():
            for child in sorted(root.iterdir()):
                if child.is_dir() and (child / "tables").exists():
                    candidates.append(child / "tables")
        return candidates or [Path("outputs/tables")]


    output_dirs = discover_output_dirs()
    selected_out = st.sidebar.selectbox("结果目录", [str(path) for path in output_dirs], index=0)
    OUT_DIR = Path(selected_out)

    metrics = read_table(OUT_DIR, "performance_metrics.csv")
    equity = read_table(OUT_DIR, "equity_curve.csv")
    trades = read_table(OUT_DIR, "trade_log.csv")
    messages = read_table(OUT_DIR, "message_log.csv")
    social_edges = read_table(OUT_DIR, "social_graph_edges.csv")
    centrality = read_table(OUT_DIR, "centrality_scores.csv")
    friendships = read_table(OUT_DIR, "friendships.csv")
    groups = read_table(OUT_DIR, "group_memberships.csv")
    social_events = read_table(OUT_DIR, "social_events.csv")
    event_log = read_table(OUT_DIR, "unified_event_log.csv")
    state_history = read_table(OUT_DIR, "agent_state_history.csv")
    market = read_table(OUT_DIR, "market_history.csv")
    registry = read_table(OUT_DIR, "agent_registry.csv")
    strategy_history = read_table(OUT_DIR, "strategy_choice_history.csv")

    if event_log.empty:
        event_log = build_legacy_event_log(trades, messages, social_events, equity)
    if event_log.empty or (metrics.empty and equity.empty and state_history.empty):
        st.warning("还没有找到可播放的实验结果。请先运行：python code/run_experiment.py --experiment full_social --scenario core_periphery")
        st.stop()

    event_log = normalize_events(event_log)
    state_history = normalize_state_history(state_history, equity, trades)
    agents = sorted(infer_agents(state_history, equity, registry, event_log))

    cursor_id, auto_mode, auto_scroll, visible_event_types = render_global_controls(event_log)
    current_event = event_log[event_log["event_id"] <= cursor_id].tail(1)
    current_date = current_event["date"].iloc[0] if not current_event.empty else event_log["date"].iloc[-1]
    events_until = event_log[event_log["event_id"] <= cursor_id].copy()
    state_now = state_as_of(state_history, current_date)

    render_sync_bar(cursor_id, event_log, current_event, current_date, OUT_DIR)

    tab_overview, tab_hall, tab_chat, tab_social, tab_tables = st.tabs(["实时总控", "交易大厅", "ChatLab", "社交关系", "研究表格"])

    with tab_overview:
        render_overview(metrics, state_now, events_until, event_log, current_date, strategy_history)

    with tab_hall:
        render_trading_hall(state_now, events_until, cursor_id, visible_event_types, auto_scroll)

    with tab_chat:
        render_chatlab(agents, events_until, friendships, cursor_id, auto_scroll)

    with tab_social:
        render_social_view(social_edges, centrality, friendships, groups, events_until, cursor_id, auto_scroll)

    with tab_tables:
        render_tables(event_log, state_history, messages, trades, social_events, metrics, strategy_history)

    run_auto_advance(event_log, cursor_id, auto_mode)


# ----------------------------- data normalization -----------------------------


def normalize_events(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "event_id" not in df.columns:
        df["event_id"] = range(1, len(df) + 1)
    df["event_id"] = pd.to_numeric(df["event_id"], errors="coerce").fillna(0).astype(int)
    if "date" not in df.columns:
        if "event_time" in df.columns:
            df["date"] = pd.to_datetime(df["event_time"], errors="coerce").dt.strftime("%Y-%m-%d")
        else:
            df["date"] = ""
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "event_time" not in df.columns:
        df["event_time"] = df["date"].astype(str) + " 09:30:00"
    for col in [
        "source",
        "event_type",
        "agent",
        "counterparty",
        "channel",
        "ticker",
        "side",
        "detail",
        "payload",
    ]:
        if col not in df.columns:
            df[col] = ""
    for col in ["shares", "price", "notional", "cash", "equity", "pnl", "pnl_pct"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("event_id").reset_index(drop=True)
    return df


def normalize_state_history(state_df: pd.DataFrame, equity_df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    if not state_df.empty:
        out = state_df.copy()
    elif not equity_df.empty:
        out = equity_df.copy()
        out["pnl"] = out.groupby("agent")["equity"].transform(lambda values: values - values.iloc[0])
        out["pnl_pct"] = out.groupby("agent")["equity"].transform(lambda values: values / values.iloc[0] - 1)
        out["positions_json"] = "{}"
        out["position_summary"] = ""
        out["last_action"] = ""
        out["friend_count"] = 0
        out["friends"] = ""
    else:
        return pd.DataFrame()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in ["equity", "cash", "pnl", "pnl_pct"]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    for col in ["positions_json", "position_summary", "last_action", "friends"]:
        if col not in out.columns:
            out[col] = ""
    if "friend_count" not in out.columns:
        out["friend_count"] = 0
    return out.sort_values(["date", "agent"])


def build_legacy_event_log(trades_df, messages_df, social_events_df, equity_df) -> pd.DataFrame:
    rows = []
    seq = 0

    def add(date, event_type, agent, detail, source, **extra):
        nonlocal seq
        seq += 1
        date_str = str(pd.Timestamp(date))[:10]
        rows.append(
            {
                "event_id": seq,
                "date": date_str,
                "event_time": f"{date_str} 09:30:{seq % 60:02d}",
                "source": source,
                "event_type": event_type,
                "agent": agent,
                "detail": detail,
                **extra,
            }
        )

    if not social_events_df.empty:
        for _, row in social_events_df.iterrows():
            add(
                row.get("date", ""),
                row.get("event_type", "friend_request"),
                row.get("sender", ""),
                row.get("detail", ""),
                "social",
                counterparty=row.get("receiver", ""),
                channel="friend_request",
            )
    if not messages_df.empty:
        for _, row in messages_df.iterrows():
            add(
                row.get("timestamp", ""),
                "message",
                row.get("sender_id", ""),
                row.get("natural_language", ""),
                "chat",
                channel=row.get("channel", ""),
                ticker=", ".join(parse_list_value(row.get("tickers", ""))),
                payload=json.dumps({"receiver_ids": parse_list_value(row.get("receiver_ids", ""))}, ensure_ascii=False),
            )
    if not trades_df.empty:
        for _, row in trades_df.iterrows():
            detail = f"{row.get('side', '')} {row.get('shares', '')} {row.get('ticker', '')} @ {row.get('price', '')}"
            add(
                row.get("date", ""),
                "trade",
                row.get("agent", ""),
                detail,
                "portfolio",
                side=row.get("side", ""),
                ticker=row.get("ticker", ""),
                shares=row.get("shares", 0),
                price=row.get("price", 0),
                notional=float(row.get("shares", 0)) * float(row.get("price", 0)),
            )
    if not equity_df.empty:
        for _, row in equity_df.iterrows():
            add(
                row.get("date", ""),
                "pnl_snapshot",
                row.get("agent", ""),
                f"Equity {float(row.get('equity', 0)):,.2f}",
                "portfolio",
                equity=row.get("equity", 0),
                cash=row.get("cash", 0),
            )
    return pd.DataFrame(rows)


def infer_agents(state_df, equity_df, registry_df, event_df) -> set[str]:
    names = set()
    for df, col in [(state_df, "agent"), (equity_df, "agent"), (registry_df, "agent"), (event_df, "agent")]:
        if not df.empty and col in df.columns:
            names.update(df[col].dropna().astype(str).tolist())
    names.discard("")
    return names


# ----------------------------- global playback -----------------------------


def render_global_controls(events: pd.DataFrame):
    min_id = int(events["event_id"].min())
    max_id = int(events["event_id"].max())
    if "event_cursor" not in st.session_state:
        st.session_state.event_cursor = max_id
    if "selected_chat_agent" not in st.session_state:
        first_agent = str(events["agent"].dropna().iloc[0]) if not events["agent"].dropna().empty else ""
        st.session_state.selected_chat_agent = first_agent

    mode = st.sidebar.radio("播放模式", ["最新状态", "手动回放", "实时滚动"], index=0)
    auto_scroll = st.sidebar.toggle("聊天/事件窗自动滚到底部", value=True)
    all_types = sorted(events["event_type"].dropna().astype(str).unique().tolist())
    default_types = [t for t in all_types if t not in {"pnl_snapshot"}]
    visible_event_types = st.sidebar.multiselect("交易大厅事件类型", all_types, default=default_types or all_types)

    if mode == "最新状态":
        cursor_id = max_id
        st.session_state.event_cursor = cursor_id
        auto_mode = False
    elif mode == "手动回放":
        cursor_id = st.sidebar.slider("事件游标", min_value=min_id, max_value=max_id, value=int(st.session_state.event_cursor))
        st.session_state.event_cursor = cursor_id
        auto_mode = False
    else:
        c1, c2 = st.sidebar.columns(2)
        if c1.button("从头播放"):
            st.session_state.event_cursor = min_id
        if c2.button("跳到最新"):
            st.session_state.event_cursor = max_id
        cursor_id = int(max(min_id, min(max_id, st.session_state.event_cursor)))
        st.sidebar.slider("当前事件", min_value=min_id, max_value=max_id, value=cursor_id, disabled=True)
        st.sidebar.slider("每次前进事件数", 1, 40, 8, key="playback_step")
        st.sidebar.slider("刷新间隔秒", 0.2, 3.0, 0.8, 0.1, key="playback_interval")
        st.sidebar.toggle("到结尾后循环", value=False, key="playback_loop")
        auto_mode = True

    return cursor_id, auto_mode, auto_scroll, visible_event_types


def run_auto_advance(events: pd.DataFrame, cursor_id: int, auto_mode: bool) -> None:
    if not auto_mode:
        return
    max_id = int(events["event_id"].max())
    min_id = int(events["event_id"].min())
    step = int(st.session_state.get("playback_step", 8))
    interval = float(st.session_state.get("playback_interval", 0.8))
    loop = bool(st.session_state.get("playback_loop", False))
    next_id = cursor_id + step
    if next_id > max_id:
        next_id = min_id if loop else max_id
    st.session_state.event_cursor = next_id
    time.sleep(interval)
    rerun_app()


def rerun_app() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:  # pragma: no cover - old streamlit compatibility
        st.experimental_rerun()


def render_sync_bar(cursor_id, events, current_event, current_date, out_dir):
    max_id = int(events["event_id"].max())
    if current_event.empty:
        detail = "暂无事件"
        event_type = ""
        agent = ""
    else:
        row = current_event.iloc[0]
        detail = row.get("detail", "")
        event_type = row.get("event_type", "")
        agent = row.get("agent", "")
    st.markdown(
        "<div class='sync-bar'>"
        f"<div class='sync-event'><b>同步游标</b> {cursor_id:,} / {max_id:,} &nbsp; "
        f"<b>日期</b> {html.escape(str(current_date))} &nbsp; "
        f"<b>当前事件</b> {html.escape(str(event_type))} / {html.escape(str(agent))}</div>"
        f"<div class='small-muted'>{html.escape(str(detail))}</div>"
        f"<div class='small-muted'>读取目录：{html.escape(str(out_dir))}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


# ----------------------------- overview / hall -----------------------------


def render_overview(metrics, state_now, events_until, events_all, current_date, strategy_history=None):
    total_events = len(events_until)
    message_count = int((events_until["event_type"] == "message").sum()) if not events_until.empty else 0
    trade_count = int((events_until["event_type"] == "trade").sum()) if not events_until.empty else 0
    social_count = int(events_until["event_type"].astype(str).str.startswith("friend").sum()) if not events_until.empty else 0
    if not state_now.empty:
        best = state_now.sort_values("pnl", ascending=False).iloc[0]
        avg_pnl = state_now["pnl"].mean()
    else:
        best = None
        avg_pnl = 0.0
    cards = [
        ("当前日期", str(current_date), f"已播放 {total_events:,} 个事件"),
        ("最佳 PnL", best["agent"] if best is not None else "-", money(best["pnl"]) if best is not None else "-"),
        ("平均 PnL", money(avg_pnl), "当前游标下所有 agent"),
        ("交易 / 聊天 / 社交", f"{trade_count} / {message_count} / {social_count}", "统一事件流统计"),
    ]
    st.markdown(render_kpi_cards(cards), unsafe_allow_html=True)
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown("#### 当前 Agent 状态")
        if state_now.empty:
            st.info("暂无状态快照。")
        else:
            table = state_now[["agent", "equity", "cash", "pnl", "pnl_pct", "position_summary", "last_action"]].copy()
            table = table.sort_values("pnl", ascending=False)
            st.dataframe(table, use_container_width=True, hide_index=True)
    with c2:
        st.markdown("#### 已播放事件分布")
        if events_until.empty:
            st.info("暂无事件。")
        else:
            counts = events_until.groupby(["source", "event_type"]).size().reset_index(name="count")
            fig = go.Figure()
            for source, group in counts.groupby("source"):
                fig.add_bar(name=source, x=group["event_type"], y=group["count"])
            fig.update_layout(height=310, barmode="stack", margin=dict(l=10, r=10, t=20, b=80))
            st.plotly_chart(fig, use_container_width=True)
        if strategy_history is not None and not strategy_history.empty:
            work = strategy_history.copy()
            work["date_ts"] = pd.to_datetime(work["date"], errors="coerce")
            latest_date = pd.Timestamp(current_date)
            work = work[work["date_ts"] <= latest_date]
            if not work.empty:
                latest_strategy = work.sort_values(["date_ts", "agent"]).groupby("agent", as_index=False).tail(1)
                st.markdown("#### 当前社交策略")
                st.dataframe(
                    latest_strategy[["agent", "strategy", "expected_utility", "reason"]].sort_values("agent"),
                    use_container_width=True,
                    hide_index=True,
                )


def render_trading_hall(state_now, events_until, cursor_id, visible_event_types, auto_scroll):
    st.markdown("#### 交易大厅：PnL、持仓、买卖动作与实时事件流")
    if state_now.empty:
        st.info("暂无 Agent 状态。")
        return
    latest_by_agent = latest_activity_by_agent(events_until)
    cards = []
    for _, row in state_now.sort_values("agent").iterrows():
        agent = str(row["agent"])
        pnl_class = "pnl-pos" if float(row.get("pnl", 0)) >= 0 else "pnl-neg"
        action = latest_by_agent.get(agent, row.get("last_action", ""))
        badge_class, badge_text = badge_for_detail(action)
        cards.append(
            "<div class='agent-card'>"
            "<div class='agent-top'>"
            f"<div class='agent-avatar'>{html.escape(short_name(agent))}</div>"
            "<div>"
            f"<div class='agent-name'>{html.escape(agent)}</div>"
            f"<div class='agent-friends'>好友 {int(row.get('friend_count', 0) or 0)} · {html.escape(truncate(str(row.get('friends', '')), 36))}</div>"
            "</div></div>"
            f"<div class='agent-row'><span>Equity</span><span>{money(row.get('equity', 0))}</span></div>"
            f"<div class='agent-row'><span>Cash</span><span>{money(row.get('cash', 0))}</span></div>"
            f"<div class='agent-row'><span>PnL</span><span class='{pnl_class}'>{money(row.get('pnl', 0))} / {percent(row.get('pnl_pct', 0))}</span></div>"
            f"<div class='agent-row'><span>持仓</span><span>{html.escape(str(row.get('position_summary', '') or 'cash only'))}</span></div>"
            f"<div class='last-action'><span class='badge {badge_class}'>{html.escape(badge_text)}</span>{html.escape(str(action or '等待事件'))}</div>"
            "</div>"
        )
    st.markdown(f"<div class='hall-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)

    st.markdown("#### 同步事件流")
    feed = events_until.copy()
    if visible_event_types:
        feed = feed[feed["event_type"].isin(visible_event_types)]
    render_event_feed(feed.tail(220), cursor_id, height=520, auto_scroll=auto_scroll)


def latest_activity_by_agent(events: pd.DataFrame) -> dict[str, str]:
    if events.empty:
        return {}
    useful = events[~events["event_type"].isin(["pnl_snapshot"])]
    out = {}
    for agent, rows in useful.groupby("agent"):
        row = rows.sort_values("event_id").tail(1).iloc[0]
        out[str(agent)] = str(row.get("detail", ""))
    return out


# ----------------------------- ChatLab -----------------------------


def render_chatlab(agents, events_until, friendships_df, cursor_id, auto_scroll):
    st.markdown("#### ChatLab：点击 Agent 后同步查看公聊、私聊、朋友圈、好友申请")
    if not agents:
        st.info("没有 Agent。")
        return
    if st.session_state.get("selected_chat_agent") not in agents:
        st.session_state.selected_chat_agent = agents[0]

    left, right = st.columns([0.27, 0.73], gap="large")
    with left:
        st.markdown("<div class='wechat-sidebar-title'>Agent 微信列表</div>", unsafe_allow_html=True)
        search = st.text_input("搜索 Agent", value="", label_visibility="collapsed")
        filtered = [agent for agent in agents if search.lower() in agent.lower()]
        counts = message_counts_by_agent(events_until)
        for agent in filtered:
            label = f"💬 {agent} · {counts.get(agent, 0)}"
            if st.button(label, key=f"select_agent_{agent}", use_container_width=True):
                st.session_state.selected_chat_agent = agent
        selected = st.session_state.selected_chat_agent
        st.markdown(
            f"<div class='agent-mini'><div class='agent-mini-avatar'>{html.escape(short_name(selected))}</div>"
            f"<div class='agent-mini-main'><div class='agent-mini-name'>{html.escape(selected)}</div>"
            f"<div class='agent-mini-sub'>当前查看对象</div></div></div>",
            unsafe_allow_html=True,
        )

    with right:
        selected = st.session_state.selected_chat_agent
        public_tab, private_tab, moments_tab, request_tab = st.tabs(["群聊", "私聊", "朋友圈", "好友申请"])
        chat_events = events_until[events_until["event_type"] == "message"].copy()
        with public_tab:
            public = chat_events[chat_events["channel"] == "public"]
            render_chat_window(public, selected, "Market Group Chat", cursor_id, auto_scroll, empty="当前游标下还没有群聊消息。")
        with private_tab:
            private = chat_events[chat_events["channel"] == "private"].copy()
            contacts = private_contacts(private, selected)
            if contacts:
                contact = st.selectbox("选择私聊对象", contacts, key=f"private_contact_{selected}")
                convo = filter_private_conversation(private, selected, contact)
                render_chat_window(convo, selected, f"{contact}", cursor_id, auto_scroll, empty="当前游标下还没有这段私聊。")
            else:
                render_empty_phone("Private Messages", "当前游标下没有可见私聊。")
        with moments_tab:
            friends = friends_for_agent(friendships_df, selected)
            moments = visible_moments(chat_events, selected, friends)
            render_moments_window(moments, selected, friends, cursor_id, auto_scroll)
        with request_tab:
            req = friend_request_events(events_until, selected)
            render_requests_window(req, selected, cursor_id, auto_scroll)


def render_chat_window(messages_df, viewer, title, cursor_id, auto_scroll, empty="暂无消息。"):
    if messages_df.empty:
        render_empty_phone(title, empty)
        return
    rows = messages_df.sort_values("event_id").tail(180)
    bubbles = []
    for _, row in rows.iterrows():
        sender = str(row.get("agent", ""))
        is_self = sender == viewer
        wrap_class = "bubble-wrap self" if is_self else "bubble-wrap"
        bubble_class = "sent" if is_self else "received"
        payload = parse_payload(row.get("payload", ""))
        direction = payload.get("direction", "")
        confidence = payload.get("confidence", "")
        receivers = payload.get("receiver_ids", [])
        receiver_note = ""
        if row.get("channel") == "private" and receivers:
            receiver_note = " → " + ", ".join(map(str, receivers[:3]))
            if len(receivers) > 3:
                receiver_note += f" +{len(receivers)-3}"
        chips = [str(row.get("ticker", ""))]
        if direction:
            chips.append(str(direction))
        if confidence != "":
            try:
                chips.append(f"{float(confidence):.0%}")
            except (TypeError, ValueError):
                pass
        chip_html = "".join(f"<span class='chip'>{html.escape(chip)}</span>" for chip in chips if chip)
        bubbles.append(
            f"<div class='{wrap_class}'>"
            f"<div class='avatar'>{html.escape(short_name(sender))}</div>"
            f"<div class='bubble {bubble_class}'>"
            f"{chip_html}<div>{html.escape(str(row.get('detail', '')))}</div>"
            f"<div class='bubble-meta'>#{int(row.get('event_id', 0))} · {html.escape(str(row.get('date', '')))} · {html.escape(sender + receiver_note)}</div>"
            "</div></div>"
        )
    render_phone_component(title, "".join(bubbles), cursor_id, auto_scroll, height=640)


def render_moments_window(moments_df, viewer, friends, cursor_id, auto_scroll):
    title = f"朋友圈 · {viewer}"
    if moments_df.empty:
        friend_text = ", ".join(friends) if friends else "暂无好友"
        render_empty_phone(title, f"当前游标下没有可见朋友圈。可见好友：{friend_text}")
        return
    posts = []
    for _, row in moments_df.sort_values("event_id", ascending=False).head(120).iterrows():
        sender = str(row.get("agent", ""))
        payload = parse_payload(row.get("payload", ""))
        meta = f"#{int(row.get('event_id', 0))} · {row.get('date', '')} · {row.get('ticker', '')} · {payload.get('direction', '')}"
        posts.append(
            "<div class='moment-post'>"
            f"<div class='moment-avatar'>{html.escape(short_name(sender))}</div>"
            "<div>"
            f"<div class='moment-name'>{html.escape(sender)}</div>"
            f"<div class='moment-text'>{html.escape(str(row.get('detail', '')))}</div>"
            f"<div class='moment-meta'>{html.escape(meta)}</div>"
            "</div></div>"
        )
    render_phone_component(title, "".join(posts), cursor_id, auto_scroll, height=640, input_text="朋友圈仅展示，无输入框")


def render_requests_window(req_df, viewer, cursor_id, auto_scroll):
    title = f"新的朋友 · {viewer}"
    if req_df.empty:
        render_empty_phone(title, "当前游标下没有与该 Agent 相关的好友申请。")
        return
    cards = []
    for _, row in req_df.sort_values("event_id").tail(160).iterrows():
        event_type = str(row.get("event_type", ""))
        if event_type == "friend_accept":
            status = "accepted"
            label = "已通过"
        elif event_type == "friend_reject":
            status = "rejected"
            label = "已拒绝"
        else:
            status = "pending"
            label = "等待验证"
        sender = str(row.get("agent", ""))
        counterparty = str(row.get("counterparty", ""))
        cards.append(
            "<div class='request-card'>"
            f"<div class='avatar'>{html.escape(short_name(sender or counterparty))}</div>"
            "<div class='request-main'>"
            f"<div class='request-title'>{html.escape(sender)} ↔ {html.escape(counterparty)}</div>"
            f"<div class='request-detail'>{html.escape(str(row.get('detail', '')))}</div>"
            f"<div class='request-detail'>#{int(row.get('event_id', 0))} · {html.escape(str(row.get('date', '')))}</div>"
            "</div>"
            f"<div class='status-{status}'>{label}</div>"
            "</div>"
        )
    render_phone_component(title, "".join(cards), cursor_id, auto_scroll, height=640, input_text="好友申请由仿真策略自动处理")


def render_empty_phone(title, text):
    render_phone_component(title, f"<div class='empty'>{html.escape(text)}</div>", 0, False, height=420)


def render_phone_component(title, body_html, cursor_id, auto_scroll, height=620, input_text="Message disabled in replay mode"):
    should_scroll = "true" if auto_scroll else "false"
    full = f"""
    {COMPONENT_CSS}
    <div class='wechat-phone'>
      <div class='wechat-header'>{html.escape(title)}</div>
      <div id='wechatBody{cursor_id}' class='wechat-body'>{body_html}</div>
      <div class='wechat-input'>＋ <span>{html.escape(input_text)}</span></div>
    </div>
    <script>
      const body = document.getElementById('wechatBody{cursor_id}');
      if (body && {should_scroll}) {{ body.scrollTop = body.scrollHeight; }}
    </script>
    """
    components.html(full, height=height, scrolling=False)


# ----------------------------- social/network -----------------------------


def render_social_view(edges, centrality_scores, friendships_df, groups_df, events_until, cursor_id, auto_scroll):
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown("#### 社交图谱")
        st.plotly_chart(network_figure(edges, centrality_scores), use_container_width=True)
    with c2:
        st.markdown("#### 好友申请实时流")
        req = events_until[events_until["event_type"].astype(str).str.startswith("friend")]
        render_event_feed(req.tail(180), cursor_id, height=430, auto_scroll=auto_scroll)
    f1, f2 = st.columns(2)
    with f1:
        st.markdown("#### 好友关系")
        if friendships_df.empty:
            st.info("暂无好友关系。")
        else:
            st.dataframe(friendships_df, use_container_width=True, hide_index=True)
    with f2:
        st.markdown("#### 群 / 朋友圈可见范围")
        if groups_df.empty:
            st.info("暂无群组。")
        else:
            st.dataframe(groups_df, use_container_width=True, hide_index=True)


def network_figure(edges: pd.DataFrame, centrality_scores: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if edges.empty:
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
        return fig
    nodes = sorted(set(edges["source"].dropna().astype(str)) | set(edges["target"].dropna().astype(str)))
    if not nodes:
        return fig
    center = {}
    if not centrality_scores.empty and {"agent", "pagerank"} <= set(centrality_scores.columns):
        center = {str(row["agent"]): float(row.get("pagerank", 0.05)) for _, row in centrality_scores.iterrows()}
    positions = {
        node: (
            math.cos(2 * math.pi * index / len(nodes)),
            math.sin(2 * math.pi * index / len(nodes)),
        )
        for index, node in enumerate(nodes)
    }
    edge_x, edge_y = [], []
    for _, row in edges.iterrows():
        source, target = str(row["source"]), str(row["target"])
        if source not in positions or target not in positions:
            continue
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1), hoverinfo="none"))
    sizes = [22 + 150 * float(center.get(node, 0.03)) for node in nodes]
    fig.add_trace(
        go.Scatter(
            x=[positions[node][0] for node in nodes],
            y=[positions[node][1] for node in nodes],
            mode="markers+text",
            text=nodes,
            textposition="top center",
            marker=dict(size=sizes, showscale=False),
            hovertext=[f"{node}<br>pagerank={center.get(node, 0):.4f}" for node in nodes],
            hoverinfo="text",
        )
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=520, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
    return fig


# ----------------------------- generic renderers -----------------------------


def render_event_feed(events_df, cursor_id, height=500, auto_scroll=True):
    if events_df.empty:
        body = "<div class='feed-panel'><div class='feed-row'>当前视图没有事件。</div></div>"
    else:
        rows = []
        for _, row in events_df.sort_values("event_id").iterrows():
            badge_class, badge_text = badge_for_event(row)
            current = " current" if int(row.get("event_id", 0)) == int(cursor_id) else ""
            meta = f"#{int(row.get('event_id', 0))} · {row.get('event_time', '')} · {row.get('event_type', '')}"
            counterparty = str(row.get("counterparty", ""))
            if counterparty:
                counterparty = f" → {counterparty}"
            money_line = ""
            if pd.notna(row.get("pnl")) and str(row.get("event_type")) in {"trade", "pnl_snapshot", "hold", "decision"}:
                money_line = f"<div class='feed-time'>PnL {money(row.get('pnl', 0))} / {percent(row.get('pnl_pct', 0))}</div>"
            rows.append(
                f"<div class='feed-row{current}'>"
                f"<div class='feed-time'>{html.escape(meta)}</div>"
                f"<span class='badge {badge_class}'>{html.escape(badge_text)}</span>"
                f"<span class='feed-agent'>{html.escape(str(row.get('agent', '')))}{html.escape(counterparty)}</span>"
                f"<div class='feed-detail'>{html.escape(str(row.get('detail', '')))}</div>"
                f"{money_line}"
                "</div>"
            )
        body = f"<div id='feedBody{cursor_id}' class='feed-panel'>{''.join(rows)}</div>"
    should_scroll = "true" if auto_scroll else "false"
    full = f"""
    {COMPONENT_CSS}
    {body}
    <script>
      const feed = document.getElementById('feedBody{cursor_id}');
      if (feed && {should_scroll}) {{ feed.scrollTop = feed.scrollHeight; }}
    </script>
    """
    components.html(full, height=height, scrolling=False)


def render_kpi_cards(cards) -> str:
    html_cards = []
    for label, value, sub in cards:
        html_cards.append(
            "<div class='kpi-card'>"
            f"<div class='kpi-label'>{html.escape(str(label))}</div>"
            f"<div class='kpi-value'>{html.escape(str(value))}</div>"
            f"<div class='kpi-sub'>{html.escape(str(sub))}</div>"
            "</div>"
        )
    return f"<div class='kpi-grid'>{''.join(html_cards)}</div>"


def render_tables(event_log, state_history, messages, trades, social_events, metrics, strategy_history):
    st.markdown("#### 研究数据表")
    table_name = st.selectbox(
        "选择表",
        ["unified_event_log", "agent_state_history", "strategy_choice_history", "message_log", "trade_log", "social_events", "performance_metrics"],
    )
    table_map = {
        "unified_event_log": event_log,
        "agent_state_history": state_history,
        "strategy_choice_history": strategy_history if strategy_history is not None else pd.DataFrame(),
        "message_log": messages,
        "trade_log": trades,
        "social_events": social_events,
        "performance_metrics": metrics,
    }
    df = table_map[table_name]
    if df.empty:
        st.info("这张表为空。")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)


# ----------------------------- filtering helpers -----------------------------


def state_as_of(state_df: pd.DataFrame, date: str) -> pd.DataFrame:
    if state_df.empty:
        return pd.DataFrame()
    work = state_df.copy()
    work["date_ts"] = pd.to_datetime(work["date"], errors="coerce")
    cutoff = pd.Timestamp(date)
    work = work[work["date_ts"] <= cutoff]
    if work.empty:
        return pd.DataFrame()
    return work.sort_values(["date_ts", "agent"]).groupby("agent", as_index=False).tail(1).drop(columns=["date_ts"])


def message_counts_by_agent(events_df: pd.DataFrame) -> dict[str, int]:
    chat = events_df[events_df["event_type"] == "message"]
    counts = chat.groupby("agent").size().astype(int).to_dict() if not chat.empty else {}
    for _, row in chat.iterrows():
        for receiver in event_receivers(row):
            counts[receiver] = counts.get(receiver, 0) + 1
    return counts


def private_contacts(private_df: pd.DataFrame, selected: str) -> list[str]:
    contacts = set()
    if private_df.empty:
        return []
    for _, row in private_df.iterrows():
        sender = str(row.get("agent", ""))
        receivers = event_receivers(row)
        if sender == selected:
            contacts.update(receiver for receiver in receivers if receiver != selected)
        elif selected in receivers:
            contacts.add(sender)
    contacts.discard("")
    return sorted(contacts)


def filter_private_conversation(private_df: pd.DataFrame, selected: str, contact: str) -> pd.DataFrame:
    if private_df.empty:
        return private_df
    mask = []
    for _, row in private_df.iterrows():
        sender = str(row.get("agent", ""))
        receivers = event_receivers(row)
        ok = (sender == selected and contact in receivers) or (sender == contact and selected in receivers)
        mask.append(ok)
    return private_df[mask]


def visible_moments(chat_events: pd.DataFrame, selected: str, friends: list[str]) -> pd.DataFrame:
    moments = chat_events[chat_events["channel"] == "moments"].copy()
    if moments.empty:
        return moments
    visible = set(friends) | {selected}
    return moments[moments["agent"].isin(visible)]


def friend_request_events(events_df: pd.DataFrame, selected: str) -> pd.DataFrame:
    req = events_df[events_df["event_type"].astype(str).str.startswith("friend")].copy()
    if req.empty:
        return req
    return req[(req["agent"] == selected) | (req["counterparty"] == selected)]


def friends_for_agent(friendships_df: pd.DataFrame, agent: str) -> list[str]:
    friends = []
    if friendships_df.empty:
        return friends
    for _, row in friendships_df.iterrows():
        a = str(row.get("agent_a", ""))
        b = str(row.get("agent_b", ""))
        if a == agent:
            friends.append(b)
        elif b == agent:
            friends.append(a)
    return sorted(set(friends))


def event_receivers(row) -> list[str]:
    payload = parse_payload(row.get("payload", ""))
    receivers = payload.get("receiver_ids", [])
    if not receivers:
        receivers = parse_list_value(row.get("counterparty", ""))
    return [str(value) for value in receivers if str(value) and str(value) not in {"群聊", "朋友圈可见"}]


# ----------------------------- formatting helpers -----------------------------


def parse_payload(value) -> dict:
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    text = str(value)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
            return parsed if isinstance(parsed, dict) else {}
        except (SyntaxError, ValueError):
            return {}


def parse_list_value(value) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except (SyntaxError, ValueError):
            pass
    return [part.strip() for part in text.split(",") if part.strip()]


def money(value) -> str:
    try:
        return f"{float(value):+,.2f}" if float(value) < 0 else f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "-"


def percent(value) -> str:
    try:
        return f"{float(value):+.2%}"
    except (TypeError, ValueError):
        return "-"


def short_name(agent: str) -> str:
    clean = str(agent).replace("Agent", "")
    caps = "".join(ch for ch in clean if ch.isupper())
    return (caps[:3] or clean[:3]).upper()


def truncate(text: str, limit: int = 42) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def badge_for_event(row) -> tuple[str, str]:
    event_type = str(row.get("event_type", ""))
    side = str(row.get("side", "")).upper()
    if event_type == "trade" and side == "BUY":
        return "badge-buy", "BUY"
    if event_type == "trade" and side == "SELL":
        return "badge-sell", "SELL"
    if event_type == "message":
        return "badge-chat", str(row.get("channel", "CHAT") or "CHAT").upper()
    if event_type.startswith("friend"):
        return "badge-social", "FRIEND"
    if event_type in {"hold", "decision"}:
        return "badge-hold", event_type.upper()
    if event_type == "pnl_snapshot":
        return "badge-pnl", "PNL"
    return "badge-hold", event_type.upper() or "EVENT"


def badge_for_detail(detail: str) -> tuple[str, str]:
    upper = str(detail).upper()
    if upper.startswith("BUY"):
        return "badge-buy", "BUY"
    if upper.startswith("SELL"):
        return "badge-sell", "SELL"
    if upper.startswith("HOLD"):
        return "badge-hold", "HOLD"
    if "好友" in str(detail) or "FRIEND" in upper:
        return "badge-social", "SOCIAL"
    if "REPORT" in upper or "MESSAGE" in upper or "SIGNAL" in upper:
        return "badge-chat", "CHAT"
    return "badge-pnl", "LIVE"


main()

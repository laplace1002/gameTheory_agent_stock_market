from __future__ import annotations

import ast
import html
import json
import math
import os
import re
import socket
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import yaml

from data_loader import load_prices


LIVE_SOURCE_LABEL = "LLM API 实时交互"
REPLAY_SOURCE_LABEL = "本地事件回放"
LIVE_STATE_DIR = Path("outputs/live_state")
LIVE_STATE_PATH = LIVE_STATE_DIR / "state.json"
LIVE_TABLE_DIR = LIVE_STATE_DIR / "tables"
LIVE_RUNS_DIR = LIVE_STATE_DIR / "runs"
DEFAULT_LIVE_SCENARIO = "core_periphery"
DEFAULT_STARTING_CASH = 100_000.0
LIVE_STATE_SCHEMA_VERSION = 3
LIVE_MARKET_DATA_PATH = Path(
    os.getenv("LIVE_MARKET_DATA_PATH", Path(__file__).resolve().parent.parent / "data" / "TRD_Dalyr.xlsx")
)

FALLBACK_LIVE_TICKER_PRICES = {
    "AAPL": 190.0,
    "MSFT": 430.0,
    "NVDA": 920.0,
    "TSLA": 175.0,
    "SPY": 520.0,
    "QQQ": 445.0,
    "XLE": 95.0,
    "UUP": 29.0,
}


def load_live_market_history_once(path: Path = LIVE_MARKET_DATA_PATH) -> tuple[pd.DataFrame, str]:
    try:
        market = load_prices(str(path))
    except Exception as exc:
        return pd.DataFrame(), f"{exc.__class__.__name__}: {exc}"
    return market, ""


LIVE_MARKET_HISTORY, LIVE_MARKET_LOAD_ERROR = load_live_market_history_once()


def latest_live_ticker_prices(market: pd.DataFrame = LIVE_MARKET_HISTORY) -> dict[str, float]:
    if market.empty:
        return dict(FALLBACK_LIVE_TICKER_PRICES)
    latest = market.sort_values("date").groupby("ticker", as_index=False).tail(1)
    return {
        str(row["ticker"]): round(float(row["close"]), 2)
        for _, row in latest.sort_values("ticker").iterrows()
        if pd.notna(row.get("close"))
    }


LIVE_TICKER_PRICES = latest_live_ticker_prices()

RELATIONSHIP_LABELS = {
    "friendship": "好友关系",
    "influence": "影响关系",
}

RELATIONSHIP_COLORS = {
    "friendship": "#16a34a",
    "influence": "#64748b",
}

LIVE_AGENT_PROFILES = {
    "MomentumAgent": "追涨型交易 agent，偏好近期强势资产，表达直接，重视趋势延续。",
    "MeanReversionAgent": "均值回归 agent，偏好被过度抛售后的反弹机会，谨慎质疑市场共识。",
    "LowVolatilityAgent": "低波动 agent，重视回撤、仓位控制和风险预算。",
    "DrawdownBuyerAgent": "回撤买入 agent，寻找跌深后的价值修复机会。",
    "RandomAgent": "探索型 agent，会引入随机视角，偶尔提出非共识交易想法。",
    "TruthfulReporterAgent": "诚实报告 agent，只分享自己认为证据较强的市场信号。",
    "PersuaderAgent": "说服型 agent，会主动影响其他人，但仍要给出可解释理由。",
    "FreeRiderAgent": "观察型 agent，倾向先听别人观点，再选择性发言。",
    "ContrarianAgent": "逆向 agent，经常挑战拥挤交易和过度乐观叙事。",
    "CommitteeTeamAgent": "委员会 agent，会综合多个信号后给出平衡判断。",
    "DynamicTeamAgent": "动态协作 agent，会根据当前对话调整合作、观察或竞争。",
    "SocialGraphAgent": "社交图谱 agent，关注好友关系、影响力和信息传播路径。",
}


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
html, body { height: 100%; overflow: hidden; }
.feed-panel {
    height: 100%; overflow-y: auto; padding: 10px; background: #0f172a;
    border-radius: 16px; border: 1px solid #1f2937;
    overscroll-behavior: contain;
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
    min-height: 0;
    height: 100%; overflow: hidden; display: flex; flex-direction: column;
    background: #ededed; border: 1px solid #d7d7d7; border-radius: 18px;
}
.wechat-header {
    height: 46px; background: #f7f7f7; border-bottom: 1px solid #d7d7d7;
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; color: #111827; flex: 0 0 auto;
}
.wechat-body {
    flex: 1 1 auto;
    min-height: 0;
    overflow-y: auto;
    padding: 14px;
    overscroll-behavior: contain;
    scroll-behavior: smooth;
}
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
    load_dotenv_files()
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


    data_source = st.sidebar.radio("数据源", [LIVE_SOURCE_LABEL, REPLAY_SOURCE_LABEL], index=0)
    live_api_mode = data_source == LIVE_SOURCE_LABEL
    if live_api_mode:
        if LIVE_MARKET_HISTORY.empty:
            st.sidebar.warning(f"实时行情读取失败，已回退备用价格：{LIVE_MARKET_LOAD_ERROR}")
        else:
            dates = live_market_dates()
            st.sidebar.caption(
                f"实时行情：{LIVE_MARKET_DATA_PATH.name} | "
                f"{len(live_tradable_tickers())} tickers | "
                f"{pd.Timestamp(dates[0]).strftime('%Y-%m-%d')} 至 {pd.Timestamp(dates[-1]).strftime('%Y-%m-%d')}"
            )

    if live_api_mode:
        OUT_DIR = Path("LLM API realtime")
        (
            metrics,
            equity,
            trades,
            messages,
            social_edges,
            centrality,
            friendships,
            groups,
            social_events,
            event_log,
            state_history,
            market,
            registry,
            strategy_history,
        ) = load_live_api_tables()
        manager_equity = pd.DataFrame()
        meta_weights = pd.DataFrame()
        agent_corr = pd.DataFrame()
        experiment_comparison = pd.DataFrame()
        drift_log = pd.DataFrame()
        manager_loss = pd.DataFrame()
        training_params = pd.DataFrame()
        forecast_scores = pd.DataFrame()
        agent_views = pd.DataFrame()
        bl_weights = pd.DataFrame()
    else:
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
        manager_equity = read_table(OUT_DIR, "manager_equity_curve.csv")
        meta_weights = read_table(OUT_DIR, "meta_weight_history.csv")
        agent_corr = read_table(OUT_DIR, "agent_return_correlation.csv")
        experiment_comparison = read_table(OUT_DIR, "experiment_comparison.csv")
        drift_log = read_table(OUT_DIR, "drift_log.csv")
        manager_loss = read_table(OUT_DIR, "manager_loss_history.csv")
        training_params = read_table(OUT_DIR, "training_params.csv")
        forecast_scores = read_table(OUT_DIR, "forecast_scores.csv")
        agent_views = read_table(OUT_DIR, "agent_views.csv")
        bl_weights = read_table(OUT_DIR, "bl_agent_view_weights.csv")

    if event_log.empty:
        event_log = build_legacy_event_log(trades, messages, social_events, equity)
    if event_log.empty or (metrics.empty and equity.empty and state_history.empty):
        st.warning("还没有找到可播放的实验结果。请先运行：python code/run_experiment.py --experiment full_social --scenario core_periphery")
        st.stop()

    event_log = normalize_events(event_log)
    state_history = normalize_state_history(state_history, equity, trades)
    agents = sorted(infer_agents(state_history, equity, registry, event_log))
    if live_api_mode:
        agents = list(LIVE_AGENT_PROFILES)
        render_live_sidebar_controls(event_log, agents, friendships)

    cursor_id, auto_mode, auto_scroll, visible_event_types = render_global_controls(event_log)
    current_event = event_log[event_log["event_id"] <= cursor_id].tail(1)
    current_date = current_event["date"].iloc[0] if not current_event.empty else event_log["date"].iloc[-1]
    events_until = event_log[event_log["event_id"] <= cursor_id].copy()
    state_now = state_as_of(state_history, current_date)

    render_sync_bar(cursor_id, event_log, current_event, current_date, OUT_DIR)

    tab_overview, tab_hall, tab_portfolio, tab_chat, tab_social, tab_tables = st.tabs(
        ["实时总控", "交易大厅", "Portfolio Hall", "ChatLab", "社交关系", "研究表格"]
    )

    with tab_overview:
        render_overview(metrics, state_now, events_until, event_log, current_date, strategy_history)

    with tab_hall:
        render_trading_hall(state_now, events_until, cursor_id, visible_event_types, auto_scroll)

    with tab_portfolio:
        render_portfolio_hall(
            equity,
            manager_equity,
            meta_weights,
            agent_corr,
            experiment_comparison,
            registry,
            drift_log,
            manager_loss,
            training_params,
            forecast_scores,
            agent_views,
            bl_weights,
        )

    with tab_chat:
        render_chatlab(agents, events_until, friendships, cursor_id, auto_scroll, live_api_mode=live_api_mode)

    with tab_social:
        render_social_view(
            social_edges,
            centrality,
            friendships,
            groups,
            events_until,
            cursor_id,
            auto_scroll,
            live_api_mode=live_api_mode,
        )

    with tab_tables:
        render_tables(
            event_log,
            state_history,
            messages,
            trades,
            social_events,
            metrics,
            strategy_history,
            manager_equity,
            meta_weights,
            agent_corr,
            experiment_comparison,
            drift_log,
            manager_loss,
            training_params,
            forecast_scores,
            agent_views,
            bl_weights,
        )

    if live_api_mode:
        run_live_agent_autoplay(event_log, agents, friendships)
    run_auto_advance(event_log, cursor_id, auto_mode)


# ----------------------------- live LLM API data source -----------------------------


def load_dotenv_files() -> None:
    for env_path in dotenv_search_paths():
        if env_path.exists():
            load_dotenv_file(env_path)


def dotenv_values() -> dict[str, str]:
    values = {}
    for env_path in dotenv_search_paths():
        if not env_path.exists():
            continue
        values.update(parse_env_file(env_path))
    return values


def get_llm_setting(name: str, default: str = "") -> str:
    return os.getenv(name) or dotenv_values().get(name, default)


def has_llm_api_config() -> bool:
    return bool(get_llm_setting("OPENAI_API_KEY") and get_llm_setting("OPENAI_MODEL", "gpt-4o-mini"))


def dotenv_search_paths() -> list[Path]:
    script_dir = Path(__file__).resolve().parent
    search_dirs = [Path.cwd(), script_dir, script_dir.parent, script_dir.parents[2]]
    names = [".env", "env.txt"]
    candidates = [directory / name for directory in search_dirs for name in names]
    unique = []
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def load_dotenv_file(path: Path) -> None:
    for key, value in parse_env_file(path).items():
        if key and key not in os.environ:
            os.environ[key] = value


def parse_env_file(path: Path) -> dict[str, str]:
    values = {}
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def load_live_api_tables():
    init_live_session_state()
    event_log = pd.DataFrame(st.session_state.live_events)
    state_history = pd.DataFrame(st.session_state.live_state_history)
    registry = pd.DataFrame(
        [{"agent": agent, "description": persona} for agent, persona in LIVE_AGENT_PROFILES.items()]
    )
    friendships = pd.DataFrame(st.session_state.live_friendships, columns=["agent_a", "agent_b"])
    groups = live_groups_dataframe()
    social_edges = live_social_edges_dataframe()
    centrality = live_centrality_dataframe(social_edges)
    messages = live_messages_table(event_log)
    trades = live_trades_table(event_log)
    equity = live_equity_table(state_history)
    return (
        live_metrics_table(state_history),
        equity,
        trades,
        messages,
        social_edges,
        centrality,
        friendships,
        groups,
        live_social_events_table(event_log),
        event_log,
        state_history,
        LIVE_MARKET_HISTORY.copy(),
        registry,
        pd.DataFrame(),
    )


def init_live_session_state() -> None:
    if st.session_state.get("live_state_loaded"):
        migrate_live_session_state()
        return
    state = normalize_live_state(load_live_state_from_disk() or create_initial_live_state())
    st.session_state.live_events = state["events"]
    st.session_state.live_state_history = state["state_history"]
    st.session_state.live_friendships = [tuple(pair) for pair in state["friendships"]]
    st.session_state.live_groups = state.get("groups", {})
    st.session_state.live_influence_edges = state.get("influence_edges", [])
    st.session_state.live_scenario = state.get("scenario", DEFAULT_LIVE_SCENARIO)
    auto_state = state.get("auto", {})
    st.session_state.live_auto_agents = bool(auto_state.get("enabled", False))
    st.session_state.live_auto_interval = int(auto_state.get("interval", 12))
    st.session_state.live_auto_channel = "full"
    st.session_state.live_auto_last_ts = float(auto_state.get("last_ts", 0.0))
    st.session_state.live_auto_round = int(auto_state.get("round", 0))
    st.session_state.live_current_run = state.get("current_run", {})
    st.session_state.live_active_run_id = ""
    st.session_state.live_active_round_id = ""
    st.session_state.live_state_loaded = True
    st.session_state.live_state_schema_version = int(state.get("schema_version", LIVE_STATE_SCHEMA_VERSION))
    migrate_live_session_state()
    refresh_live_friend_counts()
    persist_live_state()


def normalize_live_state(state: dict) -> dict:
    now = current_event_timestamp()
    existing_agents = {row.get("agent") for row in state.get("state_history", [])}
    for agent in LIVE_AGENT_PROFILES:
        if agent in existing_agents:
            continue
        state.setdefault("state_history", []).append(
            {
                "date": now[:10],
                "agent": agent,
                "equity": DEFAULT_STARTING_CASH,
                "cash": DEFAULT_STARTING_CASH,
                "pnl": 0.0,
                "pnl_pct": 0.0,
                "positions_json": "{}",
                "position_summary": "cash only",
                "last_action": "等待 LLM API 交互",
                "friend_count": 0,
                "friends": "",
            }
        )
    state["friendships"] = [
        list(pair)
        for pair in sorted(
            {
                tuple(sorted([str(pair[0]), str(pair[1])]))
                for pair in state.get("friendships", [])
                if len(pair) == 2 and str(pair[0]) in LIVE_AGENT_PROFILES and str(pair[1]) in LIVE_AGENT_PROFILES
            }
        )
    ]
    state["schema_version"] = LIVE_STATE_SCHEMA_VERSION
    state.setdefault("groups", {})
    state.setdefault("influence_edges", [])
    state.setdefault("current_run", {})
    state.setdefault("auto", {"enabled": False, "interval": 12, "channel": "full", "last_ts": 0.0, "round": 0})
    normalize_live_state_history_money(state.get("state_history", []))
    return state


def migrate_live_session_state() -> None:
    changed = False
    rows = st.session_state.get("live_state_history", [])
    if normalize_live_state_history_money(rows):
        changed = True
    if st.session_state.get("live_auto_channel") != "full":
        st.session_state.live_auto_channel = "full"
        changed = True
    if int(st.session_state.get("live_state_schema_version", 0)) < LIVE_STATE_SCHEMA_VERSION:
        st.session_state.live_state_schema_version = LIVE_STATE_SCHEMA_VERSION
        changed = True
    if "live_current_run" not in st.session_state:
        st.session_state.live_current_run = {}
        changed = True
    st.session_state.setdefault("live_active_run_id", "")
    st.session_state.setdefault("live_active_round_id", "")
    if changed:
        refresh_live_friend_counts()
        persist_live_state()


def normalize_live_state_history_money(rows: list[dict]) -> bool:
    changed = False
    for row in rows:
        if row.get("agent") not in LIVE_AGENT_PROFILES:
            continue
        positions = parse_positions_json(row.get("positions_json", "{}"))
        cash = safe_float(row.get("cash", 0.0))
        equity = safe_float(row.get("equity", 0.0))
        if not positions and cash == 0.0 and equity == 0.0:
            row["cash"] = DEFAULT_STARTING_CASH
            row["equity"] = DEFAULT_STARTING_CASH
            row["pnl"] = 0.0
            row["pnl_pct"] = 0.0
            row["position_summary"] = "cash only"
            changed = True
        elif positions and not row.get("position_summary"):
            row["position_summary"] = format_positions(positions)
            changed = True
    return changed


def load_live_state_from_disk() -> dict | None:
    if not LIVE_STATE_PATH.exists():
        return None
    try:
        state = json.loads(LIVE_STATE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(state, dict) or not state.get("events"):
        return None
    state.setdefault("state_history", [])
    state.setdefault("friendships", [])
    state.setdefault("groups", {})
    state.setdefault("influence_edges", [])
    state.setdefault("current_run", {})
    state.setdefault("auto", {})
    return state


def create_initial_live_state() -> dict:
    scenario_name = get_llm_setting("LIVE_SCENARIO", DEFAULT_LIVE_SCENARIO)
    scenario = load_live_scenario(scenario_name)
    if not scenario:
        scenario_name = DEFAULT_LIVE_SCENARIO
        scenario = load_live_scenario(scenario_name)
    now = current_event_timestamp()
    friendships = initial_friendships_from_scenario(scenario)
    groups = initial_groups_from_scenario(scenario)
    influence_edges = initial_influence_edges_from_scenario(scenario)
    state_history = [
        {
            "date": now[:10],
            "agent": agent,
            "equity": DEFAULT_STARTING_CASH,
            "cash": DEFAULT_STARTING_CASH,
            "pnl": 0.0,
            "pnl_pct": 0.0,
            "positions_json": "{}",
            "position_summary": "cash only",
            "last_action": "等待 LLM API 交互",
            "friend_count": 0,
            "friends": "",
        }
        for agent in LIVE_AGENT_PROFILES
    ]
    return {
        "schema_version": LIVE_STATE_SCHEMA_VERSION,
        "scenario": scenario_name,
        "events": [
            {
                "event_id": 1,
                "date": now[:10],
                "event_time": now,
                "source": "system",
                "event_type": "session_start",
                "agent": "System",
                "counterparty": "",
                "channel": "system",
                "ticker": "",
                "side": "",
                "shares": 0,
                "price": 0.0,
                "notional": 0.0,
                "cash": 0.0,
                "equity": 0.0,
                "pnl": 0.0,
                "pnl_pct": 0.0,
                "detail": f"LLM API 实时交互会话已启动，初始社交图谱来自 YAML 场景 {scenario_name}。",
                "payload": json.dumps({"scenario": scenario_name}, ensure_ascii=False),
            }
        ],
        "state_history": state_history,
        "friendships": [list(pair) for pair in friendships],
        "groups": groups,
        "influence_edges": influence_edges,
        "current_run": {},
        "auto": {"enabled": False, "interval": 12, "channel": "full", "last_ts": 0.0, "round": 0},
    }


def load_live_scenario(name: str) -> dict:
    path = Path(__file__).resolve().parent.parent / "config" / "social_scenarios.yaml"
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    scenarios = data.get("scenarios", data)
    return scenarios.get(name, {}) if isinstance(scenarios, dict) else {}


def load_live_scenarios() -> dict:
    path = Path(__file__).resolve().parent.parent / "config" / "social_scenarios.yaml"
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    scenarios = data.get("scenarios", data)
    return scenarios if isinstance(scenarios, dict) else {}


def scenario_summary(name: str, scenario: dict) -> str:
    topology = scenario.get("topology", "custom")
    friendships = scenario.get("friendships", [])
    if friendships == "all":
        friend_count = "all"
    else:
        friend_count = str(len(friendships or []))
    influence_count = len(scenario.get("influence_edges", []) or [])
    group_count = len(scenario.get("groups", {}) or {})
    return f"{name} | topology={topology}, friendships={friend_count}, influence_edges={influence_count}, groups={group_count}"


def initial_friendships_from_scenario(scenario: dict) -> list[tuple[str, str]]:
    friendships = scenario.get("friendships", []) if scenario else []
    agents = list(LIVE_AGENT_PROFILES)
    if friendships == "all":
        return [(a, b) for index, a in enumerate(agents) for b in agents[index + 1 :]]
    pairs = []
    for pair in friendships or []:
        if len(pair) != 2:
            continue
        a, b = str(pair[0]), str(pair[1])
        if a in LIVE_AGENT_PROFILES and b in LIVE_AGENT_PROFILES and a != b:
            pairs.append(tuple(sorted([a, b])))
    return sorted(set(pairs))


def initial_groups_from_scenario(scenario: dict) -> dict[str, list[str]]:
    groups = {}
    for group, members in (scenario.get("groups", {}) if scenario else {}).items():
        groups[str(group)] = [str(member) for member in members if str(member) in LIVE_AGENT_PROFILES]
    return groups


def initial_influence_edges_from_scenario(scenario: dict) -> list[dict]:
    agents = list(LIVE_AGENT_PROFILES)
    edges = []
    explicit_edges = (scenario or {}).get("influence_edges", []) or []
    if explicit_edges:
        for edge in explicit_edges:
            if len(edge) < 2:
                continue
            source, target = str(edge[0]), str(edge[1])
            if source in LIVE_AGENT_PROFILES and target in LIVE_AGENT_PROFILES and source != target:
                edges.append(
                    {
                        "source": source,
                        "target": target,
                        "weight": float(edge[2]) if len(edge) >= 3 else 1.0,
                        "kind": "influence",
                    }
                )
        return dedupe_influence_edges(edges)

    topology = (scenario or {}).get("topology", "")
    if topology == "chain":
        for source, target in zip(agents, agents[1:] + agents[:1]):
            edges.append({"source": source, "target": target, "weight": 1.0, "kind": "influence"})
    elif topology == "dense":
        for source in agents:
            for target in agents:
                if source != target:
                    edges.append({"source": source, "target": target, "weight": 0.6, "kind": "influence"})
    elif topology == "core_periphery":
        core = [agent for agent in (scenario or {}).get("core", []) if agent in LIVE_AGENT_PROFILES]
        core = core or ["SocialGraphAgent", "DynamicTeamAgent", "CommitteeTeamAgent"]
        for source, target in zip(core, core[1:] + core[:1]):
            if source != target:
                edges.append({"source": source, "target": target, "weight": 1.0, "kind": "influence"})
        periphery = [agent for agent in agents if agent not in core]
        for index, target in enumerate(periphery):
            source = core[index % len(core)]
            edges.append({"source": source, "target": target, "weight": 0.7, "kind": "influence"})
    return dedupe_influence_edges(edges)


def dedupe_influence_edges(edges: list[dict]) -> list[dict]:
    deduped = {}
    for edge in edges:
        source, target = str(edge.get("source", "")), str(edge.get("target", ""))
        if source not in LIVE_AGENT_PROFILES or target not in LIVE_AGENT_PROFILES or source == target:
            continue
        key = (source, target)
        deduped[key] = {
            "source": source,
            "target": target,
            "weight": float(edge.get("weight", 1.0)),
            "kind": "influence",
        }
    return [deduped[key] for key in sorted(deduped)]


def persist_live_state() -> None:
    LIVE_STATE_DIR.mkdir(parents=True, exist_ok=True)
    state = {
        "schema_version": LIVE_STATE_SCHEMA_VERSION,
        "scenario": st.session_state.get("live_scenario", DEFAULT_LIVE_SCENARIO),
        "events": st.session_state.get("live_events", []),
        "state_history": st.session_state.get("live_state_history", []),
        "friendships": [list(pair) for pair in st.session_state.get("live_friendships", [])],
        "groups": st.session_state.get("live_groups", {}),
        "influence_edges": st.session_state.get("live_influence_edges", []),
        "current_run": st.session_state.get("live_current_run", {}),
        "auto": current_live_auto_state(),
    }
    LIVE_STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    export_live_tables(state)


def current_live_auto_state() -> dict:
    return {
        "enabled": bool(st.session_state.get("live_auto_agents", False)),
        "interval": int(st.session_state.get("live_auto_interval", 12)),
        "channel": "full",
        "last_ts": float(st.session_state.get("live_auto_last_ts", 0.0)),
        "round": int(st.session_state.get("live_auto_round", 0)),
    }


def request_live_agent_start() -> None:
    st.session_state.live_start_all_requested = True
    st.session_state.live_auto_agents = True
    st.session_state.live_auto_agents_enabled = True


def request_live_agent_stop() -> None:
    st.session_state.live_auto_agents = False
    st.session_state.live_auto_agents_enabled = False
    pause_live_run()


def request_live_agent_resume() -> None:
    st.session_state.live_resume_requested = True
    st.session_state.live_auto_agents = True
    st.session_state.live_auto_agents_enabled = True


def begin_new_live_run() -> dict:
    now = current_event_timestamp()
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    run = {
        "id": run_id,
        "status": "running",
        "started_at": now,
        "updated_at": now,
        "round_count": 0,
        "path": str(LIVE_RUNS_DIR / run_id),
    }
    run_dir = LIVE_RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_live_run_metadata(run)
    st.session_state.live_current_run = run
    return run


def write_live_run_metadata(run: dict, **updates) -> None:
    run_id = str(run.get("id", ""))
    if not run_id:
        return
    run_dir = LIVE_RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = run_dir / "metadata.json"
    existing = {}
    if metadata_path.exists():
        try:
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
    metadata = {**existing, **run, **updates, "updated_at": current_event_timestamp()}
    (run_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_live_run() -> dict:
    run = st.session_state.get("live_current_run") or {}
    if not run.get("id"):
        run = begin_new_live_run()
    run["status"] = "running" if st.session_state.get("live_auto_agents", False) else run.get("status", "running")
    run["updated_at"] = current_event_timestamp()
    st.session_state.live_current_run = run
    return run


def pause_live_run() -> None:
    run = st.session_state.get("live_current_run") or {}
    if run:
        run["status"] = "paused"
        run["updated_at"] = current_event_timestamp()
        st.session_state.live_current_run = run
        write_live_run_metadata(run)


def live_event_context(event_id: int | None = None) -> dict:
    market_date = live_market_date_for_event(event_id or next_live_event_id())
    return {
        "run_id": st.session_state.get("live_active_run_id", ""),
        "round_id": st.session_state.get("live_active_round_id", ""),
        "market_date": market_date.strftime("%Y-%m-%d") if market_date is not None else "",
    }


def export_live_tables(state: dict) -> None:
    LIVE_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(state.get("events", [])).to_csv(LIVE_TABLE_DIR / "unified_event_log.csv", index=False)
    pd.DataFrame(state.get("state_history", [])).to_csv(LIVE_TABLE_DIR / "agent_state_history.csv", index=False)
    pd.DataFrame(state.get("friendships", []), columns=["agent_a", "agent_b"]).to_csv(
        LIVE_TABLE_DIR / "friendships.csv",
        index=False,
    )
    live_groups_dataframe(state.get("groups", {})).to_csv(LIVE_TABLE_DIR / "group_memberships.csv", index=False)
    live_social_edges_dataframe(state).to_csv(LIVE_TABLE_DIR / "social_graph_edges.csv", index=False)
    events = pd.DataFrame(state.get("events", []))
    state_history = pd.DataFrame(state.get("state_history", []))
    live_trades_table(events).to_csv(LIVE_TABLE_DIR / "trade_log.csv", index=False)
    live_equity_table(state_history).to_csv(LIVE_TABLE_DIR / "equity_curve.csv", index=False)
    live_metrics_table(state_history).to_csv(LIVE_TABLE_DIR / "performance_metrics.csv", index=False)
    LIVE_MARKET_HISTORY.to_csv(LIVE_TABLE_DIR / "market_history.csv", index=False)


def live_groups_dataframe(groups: dict | None = None) -> pd.DataFrame:
    groups = groups if groups is not None else st.session_state.get("live_groups", {})
    rows = []
    for group, members in (groups or {}).items():
        for agent in members:
            rows.append({"group": group, "agent": agent})
    return pd.DataFrame(rows, columns=["group", "agent"])


def live_social_edges_dataframe(state: dict | None = None) -> pd.DataFrame:
    if state is None:
        influence_edges = st.session_state.get("live_influence_edges", [])
        friendships = st.session_state.get("live_friendships", [])
    else:
        influence_edges = state.get("influence_edges", [])
        friendships = [tuple(pair) for pair in state.get("friendships", [])]
    rows = [
        {
            "source": edge.get("source", ""),
            "target": edge.get("target", ""),
            "weight": float(edge.get("weight", 1.0)),
            "kind": edge.get("kind", "influence"),
        }
        for edge in influence_edges
    ]
    for a, b in friendships:
        rows.append({"source": a, "target": b, "weight": 1.0, "kind": "friendship"})
        rows.append({"source": b, "target": a, "weight": 1.0, "kind": "friendship"})
    return pd.DataFrame(rows, columns=["source", "target", "weight", "kind"])


def live_centrality_dataframe(edges: pd.DataFrame) -> pd.DataFrame:
    agents = list(LIVE_AGENT_PROFILES)
    if edges.empty:
        return pd.DataFrame({"agent": agents, "pagerank": [1 / len(agents)] * len(agents)})
    scores = {agent: 1.0 for agent in agents}
    for _, row in edges.iterrows():
        scores[str(row.get("target", ""))] = scores.get(str(row.get("target", "")), 1.0) + float(row.get("weight", 1.0))
    total = sum(scores.values()) or 1.0
    return pd.DataFrame(
        [{"agent": agent, "pagerank": float(scores.get(agent, 0.0)) / total} for agent in sorted(scores)]
    )


def current_event_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def live_messages_table(event_log: pd.DataFrame) -> pd.DataFrame:
    if event_log.empty:
        return pd.DataFrame()
    messages = event_log[event_log["event_type"] == "message"].copy()
    if messages.empty:
        return pd.DataFrame()
    rows = []
    for _, row in messages.iterrows():
        payload = parse_payload(row.get("payload", ""))
        rows.append(
            {
                "message_id": int(row.get("event_id", 0)),
                "timestamp": row.get("event_time", ""),
                "sender_id": row.get("agent", ""),
                "channel": row.get("channel", ""),
                "receiver_ids": payload.get("receiver_ids", []),
                "tickers": [row.get("ticker", "")] if row.get("ticker", "") else [],
                "direction": payload.get("direction", ""),
                "confidence": payload.get("confidence", ""),
                "natural_language": row.get("detail", ""),
            }
        )
    return pd.DataFrame(rows)


def live_social_events_table(event_log: pd.DataFrame) -> pd.DataFrame:
    if event_log.empty:
        return pd.DataFrame()
    social = event_log[event_log["event_type"].astype(str).str.startswith("friend")].copy()
    if social.empty:
        return pd.DataFrame()
    return social.rename(columns={"agent": "sender", "counterparty": "receiver"})[
        ["date", "event_type", "sender", "receiver", "detail"]
    ]


def live_trades_table(event_log: pd.DataFrame) -> pd.DataFrame:
    if event_log.empty:
        return pd.DataFrame()
    trades = event_log[event_log["event_type"] == "trade"].copy()
    if trades.empty:
        return pd.DataFrame()
    return trades[
        ["date", "agent", "ticker", "side", "shares", "price", "notional", "cash", "equity", "pnl", "pnl_pct", "detail"]
    ].reset_index(drop=True)


def live_equity_table(state_history: pd.DataFrame) -> pd.DataFrame:
    if state_history.empty:
        return pd.DataFrame()
    columns = ["date", "agent", "equity", "cash", "pnl", "pnl_pct"]
    available = [column for column in columns if column in state_history.columns]
    return state_history[available].copy()


def live_metrics_table(state_history: pd.DataFrame) -> pd.DataFrame:
    if state_history.empty:
        return pd.DataFrame()
    latest = state_history.copy()
    latest["_seq"] = range(len(latest))
    latest = latest.sort_values(["agent", "_seq"]).groupby("agent", as_index=False).tail(1)
    return latest[["agent", "equity", "cash", "pnl", "pnl_pct"]].sort_values("pnl", ascending=False)


def append_live_message_event(agent: str, channel: str, receivers: list[str], detail: str, ticker: str = "") -> None:
    now = current_event_timestamp()
    event_id = next_live_event_id()
    market_date = live_market_date_string(event_id, now[:10])
    st.session_state.live_events.append(
        {
            "event_id": event_id,
            "date": market_date,
            "event_time": now,
            "source": "llm_api",
            "event_type": "message",
            "agent": agent,
            "counterparty": ", ".join(receivers),
            "channel": channel,
            "ticker": ticker,
            "side": "",
            "shares": 0,
            "price": 0.0,
            "notional": 0.0,
            "cash": 0.0,
            "equity": 0.0,
            "pnl": 0.0,
            "pnl_pct": 0.0,
            "detail": detail,
            "payload": json.dumps({"receiver_ids": receivers, "llm_generated": True}, ensure_ascii=False),
            **live_event_context(event_id),
        }
    )
    update_live_agent_action(agent, f"{channel}: {truncate(detail, 90)}")
    update_live_graph_from_message(agent, channel, receivers, detail)
    st.session_state.event_cursor = event_id
    persist_live_state()


def append_live_trade_event(agent: str, trade_plan: dict, detail_prefix: str = "") -> None:
    trade = normalize_trade_plan(agent, trade_plan)
    state = latest_live_agent_state(agent)
    positions = parse_positions_json(state.get("positions_json", "{}"))
    cash = safe_float(state.get("cash", DEFAULT_STARTING_CASH), DEFAULT_STARTING_CASH)
    side, ticker, price, requested_shares = trade["side"], trade["ticker"], trade["price"], trade["shares"]
    available = int(positions.get(ticker, 0))
    shares = requested_shares

    if side == "BUY":
        affordable = int(cash // price)
        shares = max(0, min(shares, affordable))
    elif side == "SELL":
        shares = max(0, min(shares, available))

    if side not in {"BUY", "SELL"} or shares <= 0:
        append_live_hold_event(agent, trade, reason=trade.get("rationale") or "本轮没有可执行交易。")
        return

    notional = round(shares * price, 2)
    if side == "BUY":
        positions[ticker] = available + shares
        cash = round(cash - notional, 2)
    else:
        remaining = available - shares
        if remaining > 0:
            positions[ticker] = remaining
        else:
            positions.pop(ticker, None)
        cash = round(cash + notional, 2)

    event_id = next_live_event_id()
    equity = calculate_live_equity(cash, positions, event_id)
    pnl = round(equity - DEFAULT_STARTING_CASH, 2)
    pnl_pct = pnl / DEFAULT_STARTING_CASH if DEFAULT_STARTING_CASH else 0.0
    now = current_event_timestamp()
    market_date = live_market_date_string(event_id, now[:10])
    rationale = trade.get("rationale", "")
    detail = f"{side} {shares} {ticker} @ {price:.2f}"
    if rationale:
        detail = f"{detail}；{rationale}"
    if detail_prefix:
        detail = f"{detail_prefix}{detail}"
    st.session_state.live_events.append(
        {
            "event_id": event_id,
            "date": market_date,
            "event_time": now,
            "source": "llm_api",
            "event_type": "trade",
            "agent": agent,
            "counterparty": "",
            "channel": "portfolio",
            "ticker": ticker,
            "side": side,
            "shares": shares,
            "price": price,
            "notional": notional,
            "cash": cash,
            "equity": equity,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "detail": detail,
            "payload": json.dumps({"positions": positions, "llm_generated": True}, ensure_ascii=False),
            **live_event_context(event_id),
        }
    )
    append_live_state_snapshot(agent, cash, equity, pnl, pnl_pct, positions, detail)
    st.session_state.event_cursor = event_id
    persist_live_state()


def append_live_hold_event(agent: str, trade_plan: dict, reason: str = "") -> None:
    state = latest_live_agent_state(agent)
    positions = parse_positions_json(state.get("positions_json", "{}"))
    cash = safe_float(state.get("cash", DEFAULT_STARTING_CASH), DEFAULT_STARTING_CASH)
    event_id = next_live_event_id()
    equity = calculate_live_equity(cash, positions, event_id)
    pnl = round(equity - DEFAULT_STARTING_CASH, 2)
    pnl_pct = pnl / DEFAULT_STARTING_CASH if DEFAULT_STARTING_CASH else 0.0
    now = current_event_timestamp()
    market_date = live_market_date_string(event_id, now[:10])
    ticker = normalize_ticker(trade_plan.get("ticker") or default_ticker_for_agent(agent))
    detail = reason or trade_plan.get("rationale") or "HOLD：等待更清晰的信号。"
    if not str(detail).upper().startswith("HOLD"):
        detail = f"HOLD：{detail}"
    st.session_state.live_events.append(
        {
            "event_id": event_id,
            "date": market_date,
            "event_time": now,
            "source": "llm_api",
            "event_type": "hold",
            "agent": agent,
            "counterparty": "",
            "channel": "portfolio",
            "ticker": ticker,
            "side": "HOLD",
            "shares": 0,
            "price": 0.0,
            "notional": 0.0,
            "cash": cash,
            "equity": equity,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "detail": detail,
            "payload": json.dumps({"positions": positions, "llm_generated": True}, ensure_ascii=False),
            **live_event_context(event_id),
        }
    )
    append_live_state_snapshot(agent, cash, equity, pnl, pnl_pct, positions, detail)
    st.session_state.event_cursor = event_id
    persist_live_state()


def append_live_state_snapshot(
    agent: str,
    cash: float,
    equity: float,
    pnl: float,
    pnl_pct: float,
    positions: dict[str, int],
    last_action: str,
) -> None:
    st.session_state.live_state_history.append(
        {
            "date": live_market_date_string(max(1, next_live_event_id() - 1), current_event_timestamp()[:10]),
            "agent": agent,
            "equity": round(equity, 2),
            "cash": round(cash, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": pnl_pct,
            "positions_json": json.dumps({k: v for k, v in sorted(positions.items()) if v}, ensure_ascii=False),
            "position_summary": format_positions(positions),
            "last_action": truncate(last_action, 140),
            "friend_count": 0,
            "friends": "",
            **live_event_context(max(1, next_live_event_id() - 1)),
        }
    )
    refresh_live_friend_counts()


def append_live_friendship(agent_a: str, agent_b: str) -> None:
    if not agent_a or not agent_b or agent_a == agent_b:
        return
    pair = tuple(sorted([agent_a, agent_b]))
    existing = {tuple(sorted(pair_item)) for pair_item in st.session_state.live_friendships}
    if pair in existing:
        return
    st.session_state.live_friendships.append(pair)
    now = current_event_timestamp()
    event_id = next_live_event_id()
    st.session_state.live_events.append(
        {
            "event_id": event_id,
            "date": now[:10],
            "event_time": now,
            "source": "user",
            "event_type": "friend_accept",
            "agent": agent_a,
            "counterparty": agent_b,
            "channel": "friend_request",
            "ticker": "",
            "side": "",
            "shares": 0,
            "price": 0.0,
            "notional": 0.0,
            "cash": 0.0,
            "equity": 0.0,
            "pnl": 0.0,
            "pnl_pct": 0.0,
            "detail": f"{agent_a} 与 {agent_b} 已建立好友关系。",
            "payload": "{}",
            **live_event_context(),
        }
    )
    refresh_live_friend_counts()
    st.session_state.event_cursor = event_id
    persist_live_state()


def append_live_event(row: dict) -> None:
    st.session_state.live_events.append(row)


def reset_live_social_graph_from_yaml() -> None:
    scenario_name = st.session_state.get("live_scenario") or get_llm_setting("LIVE_SCENARIO", DEFAULT_LIVE_SCENARIO)
    scenario = load_live_scenario(str(scenario_name))
    if not scenario:
        scenario_name = DEFAULT_LIVE_SCENARIO
        scenario = load_live_scenario(DEFAULT_LIVE_SCENARIO)
    st.session_state.live_scenario = str(scenario_name)
    st.session_state.live_friendships = initial_friendships_from_scenario(scenario)
    st.session_state.live_groups = initial_groups_from_scenario(scenario)
    st.session_state.live_influence_edges = initial_influence_edges_from_scenario(scenario)
    refresh_live_friend_counts()
    now = current_event_timestamp()
    event_id = next_live_event_id()
    append_live_event(
        {
            "event_id": event_id,
            "date": now[:10],
            "event_time": now,
            "source": "system",
            "event_type": "social_graph_reload",
            "agent": "System",
            "counterparty": "",
            "channel": "system",
            "ticker": "",
            "side": "",
            "shares": 0,
            "price": 0.0,
            "notional": 0.0,
            "cash": 0.0,
            "equity": 0.0,
            "pnl": 0.0,
            "pnl_pct": 0.0,
            "detail": f"已按 YAML 场景 {scenario_name} 重新加载好友关系、群组和初始影响关系。",
            "payload": json.dumps({"scenario": scenario_name}, ensure_ascii=False),
        }
    )
    st.session_state.event_cursor = event_id
    persist_live_state()


def update_live_graph_from_message(agent: str, channel: str, receivers: list[str], detail: str) -> None:
    if channel == "public":
        reinforce_live_influence(agent, [name for name in LIVE_AGENT_PROFILES if name != agent])
        return
    if channel != "private":
        reinforce_live_influence(agent, receivers)
        return
    for receiver in receivers:
        if not receiver or receiver == agent:
            continue
        if live_are_friends(agent, receiver):
            reinforce_live_influence(agent, [receiver])
            continue
        add_live_friend_request_with_decision(
            agent,
            receiver,
            "私聊触达非好友，仅记录申请意向，等待对方判断。",
            pd.DataFrame(st.session_state.get("live_events", [])),
        )
        reinforce_live_influence(agent, [receiver])


def append_live_friend_request_event(agent: str, receiver: str, reason: str, trigger: str = "") -> int | None:
    if not agent or not receiver or agent == receiver or live_are_friends(agent, receiver):
        return None
    if has_recent_friend_request(agent, receiver):
        return None
    now = current_event_timestamp()
    request_id = next_live_event_id()
    append_live_event(
        {
            "event_id": request_id,
            "date": now[:10],
            "event_time": now,
            "source": "llm_api",
            "event_type": "friend_request",
            "agent": agent,
            "counterparty": receiver,
            "channel": "friend_request",
            "ticker": "",
            "side": "",
            "shares": 0,
            "price": 0.0,
            "notional": 0.0,
            "cash": 0.0,
            "equity": 0.0,
            "pnl": 0.0,
            "pnl_pct": 0.0,
            "detail": f"{agent} 向 {receiver} 发起好友申请：{truncate(reason, 120)}",
            "payload": json.dumps(
                {
                    "reason": truncate(reason, 160),
                    "trigger": truncate(trigger, 160),
                    "status": "pending",
                },
                ensure_ascii=False,
            ),
            **live_event_context(),
        }
    )
    return request_id


def has_recent_friend_request(agent: str, receiver: str, window: int = 80) -> bool:
    pair = {agent, receiver}
    events = st.session_state.get("live_events", [])[-window:]
    for event in reversed(events):
        if not str(event.get("event_type", "")).startswith("friend"):
            continue
        if {str(event.get("agent", "")), str(event.get("counterparty", ""))} == pair:
            return True
    return False


def add_live_friend_request_with_decision(agent: str, receiver: str, reason: str, recent_events: pd.DataFrame | None = None) -> None:
    request_id = append_live_friend_request_event(agent, receiver, reason)
    if request_id is None:
        return
    decision = decide_live_friend_request(agent, receiver, reason, recent_events)
    if decision.get("accept"):
        accept_live_friend_request(agent, receiver, decision.get("reason", "对方判断该连接有信息价值。"))
    else:
        reject_live_friend_request(agent, receiver, decision.get("reason", "对方判断当前没有建立好友关系的必要。"))


def accept_live_friend_request(agent: str, receiver: str, reason: str) -> None:
    pair = tuple(sorted([agent, receiver]))
    if pair not in {tuple(sorted(item)) for item in st.session_state.get("live_friendships", [])}:
        st.session_state.live_friendships.append(pair)
    accept_id = next_live_event_id()
    now = current_event_timestamp()
    append_live_event(
        {
            "event_id": accept_id,
            "date": now[:10],
            "event_time": now,
            "source": "llm_api",
            "event_type": "friend_accept",
            "agent": receiver,
            "counterparty": agent,
            "channel": "friend_request",
            "ticker": "",
            "side": "",
            "shares": 0,
            "price": 0.0,
            "notional": 0.0,
            "cash": 0.0,
            "equity": 0.0,
            "pnl": 0.0,
            "pnl_pct": 0.0,
            "detail": f"{receiver} 通过了 {agent} 的好友申请：{truncate(reason, 120)}",
            "payload": json.dumps({"reason": truncate(reason, 160), "status": "accepted"}, ensure_ascii=False),
            **live_event_context(),
        }
    )
    refresh_live_friend_counts()


def reject_live_friend_request(agent: str, receiver: str, reason: str) -> None:
    reject_id = next_live_event_id()
    now = current_event_timestamp()
    append_live_event(
        {
            "event_id": reject_id,
            "date": now[:10],
            "event_time": now,
            "source": "llm_api",
            "event_type": "friend_reject",
            "agent": receiver,
            "counterparty": agent,
            "channel": "friend_request",
            "ticker": "",
            "side": "",
            "shares": 0,
            "price": 0.0,
            "notional": 0.0,
            "cash": 0.0,
            "equity": 0.0,
            "pnl": 0.0,
            "pnl_pct": 0.0,
            "detail": f"{receiver} 拒绝了 {agent} 的好友申请：{truncate(reason, 120)}",
            "payload": json.dumps({"reason": truncate(reason, 160), "status": "rejected"}, ensure_ascii=False),
            **live_event_context(),
        }
    )


def decide_live_friend_request(agent: str, receiver: str, reason: str, recent_events: pd.DataFrame | None = None) -> dict:
    if not truthy_env("LLM_FRIEND_DECISION", default=True) or not has_llm_api_config():
        return heuristic_friend_decision(agent, receiver, reason)
    try:
        context = recent_events if recent_events is not None else pd.DataFrame()
        return call_llm_friend_decision(agent, receiver, reason, context)
    except RuntimeError:
        return heuristic_friend_decision(agent, receiver, reason)


def heuristic_friend_decision(agent: str, receiver: str, reason: str) -> dict:
    current_friends = [pair for pair in st.session_state.get("live_friendships", []) if receiver in pair]
    text = str(reason or "")
    useful_terms = ["相关", "互补", "风险", "回撤", "低波", "趋势", "信息", "验证", "分歧", "对冲", "合作"]
    noisy_terms = ["随便", "热闹", "所有人", "无理由", "诗", "火焰", "余烬", "旧帐篷"]
    accept = len(current_friends) < 5 and any(term in text for term in useful_terms) and not any(term in text for term in noisy_terms)
    if accept:
        return {"accept": True, "reason": "申请理由包含可验证的信息互补价值。"}
    return {"accept": False, "reason": "申请理由不足或当前好友数已较高，暂不建立连接。"}


def live_are_friends(agent_a: str, agent_b: str) -> bool:
    pair = tuple(sorted([agent_a, agent_b]))
    return pair in {tuple(sorted(item)) for item in st.session_state.get("live_friendships", [])}


def reinforce_live_influence(source: str, targets: list[str]) -> None:
    if not targets:
        return
    edges = st.session_state.get("live_influence_edges", [])
    for target in targets:
        if not target or target == source:
            continue
        found = False
        for edge in edges:
            if edge.get("source") == source and edge.get("target") == target:
                edge["weight"] = min(5.0, float(edge.get("weight", 1.0)) + 0.08)
                found = True
                break
        if not found:
            edges.append({"source": source, "target": target, "weight": 0.8, "kind": "influence"})
    st.session_state.live_influence_edges = edges


def next_live_event_id() -> int:
    events = st.session_state.get("live_events", [])
    if not events:
        return 1
    return max(int(event.get("event_id", 0)) for event in events) + 1


def update_live_agent_action(agent: str, action: str) -> None:
    rows = st.session_state.get("live_state_history", [])
    if not rows:
        return
    for row in reversed(rows):
        if row.get("agent") == agent:
            row["date"] = current_event_timestamp()[:10]
            row["last_action"] = action
            break


def refresh_live_friend_counts() -> None:
    friend_lookup = {agent: [] for agent in LIVE_AGENT_PROFILES}
    unique_pairs = sorted({tuple(sorted(pair)) for pair in st.session_state.get("live_friendships", [])})
    st.session_state.live_friendships = unique_pairs
    for a, b in unique_pairs:
        friend_lookup.setdefault(a, []).append(b)
        friend_lookup.setdefault(b, []).append(a)
    for row in st.session_state.get("live_state_history", []):
        friends = sorted(friend_lookup.get(row.get("agent"), []))
        row["friend_count"] = len(friends)
        row["friends"] = ", ".join(friends)


def latest_live_agent_state(agent: str) -> dict:
    for row in reversed(st.session_state.get("live_state_history", [])):
        if row.get("agent") == agent:
            return dict(row)
    return {
        "agent": agent,
        "cash": DEFAULT_STARTING_CASH,
        "equity": DEFAULT_STARTING_CASH,
        "pnl": 0.0,
        "pnl_pct": 0.0,
        "positions_json": "{}",
        "position_summary": "cash only",
        "last_action": "等待 LLM API 交互",
    }


def normalize_trade_plan(agent: str, trade_plan: dict | None) -> dict:
    plan = trade_plan if isinstance(trade_plan, dict) else {}
    side = str(plan.get("side", "HOLD")).strip().upper()
    if side not in {"BUY", "SELL", "HOLD"}:
        side = "HOLD"
    ticker = normalize_ticker(plan.get("ticker") or default_ticker_for_agent(agent))
    price = safe_float(plan.get("price", 0.0), 0.0)
    reference_price = live_reference_price(ticker, next_live_event_id())
    if price <= 0 or price < reference_price * 0.2 or price > reference_price * 5:
        price = reference_price
    shares = int(max(0, safe_float(plan.get("shares", 0), 0.0)))
    if side in {"BUY", "SELL"} and shares <= 0:
        shares = 5 + stable_index(agent + ticker + side, 11)
    return {
        "side": side,
        "ticker": ticker,
        "shares": min(shares, 100),
        "price": round(price, 2),
        "rationale": truncate(str(plan.get("rationale", "") or ""), 160),
    }


def trade_plan_from_agent_plan(agent: str, plan: dict) -> dict:
    trade = plan.get("trade") if isinstance(plan.get("trade"), dict) else {}
    normalized = normalize_trade_plan(agent, trade)
    text = collect_plan_text(plan)
    inferred_side = infer_trade_side_from_text(text)

    if normalized["side"] == "HOLD" and inferred_side in {"BUY", "SELL"}:
        ticker = str(trade.get("ticker") or infer_ticker_from_text(text) or default_ticker_for_agent(agent))
        inferred = {
            **trade,
            "side": inferred_side,
            "ticker": ticker,
            "shares": trade.get("shares") or infer_shares_from_text(agent, ticker, text),
            "price": trade.get("price", 0),
            "rationale": trade.get("rationale")
            or f"从 Agent 文本中识别到交易意图：{truncate(text, 100)}",
        }
        return inferred

    if normalized["side"] in {"BUY", "SELL"}:
        if not trade.get("ticker"):
            trade = {**trade, "ticker": infer_ticker_from_text(text) or normalized["ticker"]}
        if not trade.get("rationale"):
            trade = {**trade, "rationale": f"结构化交易指令：{normalized['side']} {normalized['ticker']}"}
    return trade


def collect_plan_text(plan: dict) -> str:
    parts = []
    trade = plan.get("trade")
    if isinstance(trade, dict):
        parts.extend(str(trade.get(key, "")) for key in ["side", "ticker", "rationale"])
    for key in ["public_message", "moment"]:
        value = plan.get(key)
        if value:
            parts.append(str(value))
    for key in ["private_message", "friend_request"]:
        value = plan.get(key)
        if isinstance(value, dict):
            parts.extend(str(item) for item in value.values() if item)
        elif value:
            parts.append(str(value))
    return " ".join(parts)


def infer_trade_side_from_text(text: str) -> str:
    work = str(text or "").lower()
    buy_terms = [
        "加仓",
        "买入",
        "增持",
        "建仓",
        "加多",
        "做多",
        "重仓接",
        "接回调",
        "先接",
        "进场接",
        "继续买",
    ]
    sell_terms = ["减仓", "卖出", "清仓", "止盈", "加空", "做空", "开空", "short", "sell"]
    hold_terms = ["持有现金", "现金继续", "继续拿现金", "不动", "观望", "等待", "不碰", "再考虑", "维持"]
    if any(term in work for term in buy_terms) or re.search(r"\bbuy\b", work):
        return "BUY"
    if any(term in work for term in sell_terms):
        return "SELL"
    if any(term in work for term in hold_terms):
        return "HOLD"
    return ""


def infer_ticker_from_text(text: str) -> str:
    work = str(text or "").upper()
    for ticker in live_tradable_tickers():
        if ticker in work:
            return ticker
    if any(term in work for term in ["AI", "半导体", "芯片", "算力"]):
        return default_ticker_for_theme("growth")
    if any(term in work for term in ["能源", "原油", "石油", "OIL"]):
        return default_ticker_for_theme("cyclical")
    if any(term in work for term in ["美元", "避险", "USD"]):
        return default_ticker_for_theme("defensive")
    if any(term in work for term in ["科技", "纳指", "NASDAQ", "QQQ"]):
        return default_ticker_for_theme("growth")
    if any(term in work for term in ["大盘", "指数", "标普", "SPY"]):
        return default_ticker_for_theme("market")
    return ""


def infer_shares_from_text(agent: str, ticker: str, text: str) -> int:
    price = live_reference_price(normalize_ticker(ticker), next_live_event_id())
    percent_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*仓位", str(text or ""))
    if percent_match:
        percent = min(50.0, max(1.0, safe_float(percent_match.group(1), 5.0)))
        return max(1, int(DEFAULT_STARTING_CASH * percent / 100.0 / price))
    if "翻倍" in str(text):
        return 20 + stable_index(agent + ticker, 16)
    if any(term in str(text) for term in ["重仓", "大幅"]):
        return 20 + stable_index(agent + ticker, 21)
    return 5 + stable_index(agent + ticker, 16)


def normalize_ticker(value) -> str:
    ticker = "".join(ch for ch in str(value or "").upper() if ch.isalnum() or ch in {".", "-"}).strip(".-")
    if ticker in live_ticker_prices():
        return ticker
    tickers = live_tradable_tickers()
    return tickers[stable_index(ticker or tickers[0], len(tickers))]


def default_ticker_for_agent(agent: str) -> str:
    tickers = live_tradable_tickers()
    return tickers[stable_index(agent, len(tickers))]


def default_ticker_for_theme(theme: str) -> str:
    tickers = live_tradable_tickers()
    preferred = {
        "growth": ["300750", "002594", "002415", "000333"],
        "cyclical": ["601899", "000002"],
        "defensive": ["600036", "601318", "600519"],
        "market": ["000333", "600036", "601318"],
    }.get(theme, [])
    for ticker in preferred:
        if ticker in tickers:
            return ticker
    return tickers[stable_index(theme, len(tickers))]


def live_ticker_prices() -> dict[str, float]:
    return LIVE_TICKER_PRICES or dict(FALLBACK_LIVE_TICKER_PRICES)


def live_tradable_tickers() -> list[str]:
    return sorted(live_ticker_prices())


def live_market_dates() -> list[pd.Timestamp]:
    if LIVE_MARKET_HISTORY.empty:
        return []
    return sorted(pd.to_datetime(LIVE_MARKET_HISTORY["date"]).dropna().unique())


def live_market_step(event_id: int) -> int:
    active_round = str(st.session_state.get("live_active_round_id", ""))
    match = re.search(r"round_(\d+)", active_round)
    if match:
        return max(0, int(match.group(1)) - 1)
    round_number = int(st.session_state.get("live_auto_round", 0) or 0)
    if round_number > 0:
        return max(0, round_number - 1)
    events_per_step = max(1, len(LIVE_AGENT_PROFILES))
    return max(0, (max(1, int(event_id)) - 1) // events_per_step)


def live_market_date_for_event(event_id: int) -> pd.Timestamp | None:
    dates = live_market_dates()
    if not dates:
        return None
    step = min(live_market_step(event_id), len(dates) - 1)
    return pd.Timestamp(dates[step])


def live_market_date_string(event_id: int, fallback: str = "") -> str:
    market_date = live_market_date_for_event(event_id)
    return market_date.strftime("%Y-%m-%d") if market_date is not None else fallback


def live_market_row(ticker: str, event_id: int) -> pd.Series | None:
    if LIVE_MARKET_HISTORY.empty:
        return None
    normalized = str(ticker)
    target_date = live_market_date_for_event(event_id)
    work = LIVE_MARKET_HISTORY[LIVE_MARKET_HISTORY["ticker"].astype(str) == normalized].copy()
    if work.empty:
        return None
    work["date"] = pd.to_datetime(work["date"])
    if target_date is not None:
        eligible = work[work["date"] <= target_date]
        if not eligible.empty:
            return eligible.sort_values("date").iloc[-1]
    return work.sort_values("date").iloc[-1]


def live_reference_price(ticker: str, event_id: int) -> float:
    normalized = normalize_ticker(ticker)
    row = live_market_row(normalized, event_id)
    if row is not None and pd.notna(row.get("close")):
        return round(max(0.01, float(row["close"])), 2)
    return round(max(0.01, live_ticker_prices().get(normalized, next(iter(live_ticker_prices().values())))), 2)


def calculate_live_equity(cash: float, positions: dict[str, int], event_id: int) -> float:
    holdings = sum(int(shares) * live_reference_price(ticker, event_id) for ticker, shares in positions.items())
    return round(cash + holdings, 2)


def parse_positions_json(value) -> dict[str, int]:
    if isinstance(value, dict):
        raw = value
    else:
        try:
            raw = json.loads(str(value or "{}"))
        except json.JSONDecodeError:
            raw = {}
    positions = {}
    for ticker, shares in raw.items() if isinstance(raw, dict) else []:
        normalized = normalize_ticker(ticker)
        amount = int(max(0, safe_float(shares, 0.0)))
        if amount:
            positions[normalized] = positions.get(normalized, 0) + amount
    return positions


def format_positions(positions: dict[str, int]) -> str:
    active = [f"{ticker} {shares}" for ticker, shares in sorted(positions.items()) if shares]
    return ", ".join(active) if active else "cash only"


def call_llm_agent(agent: str, channel: str, receivers: list[str], instruction: str, recent_events: pd.DataFrame) -> str:
    load_dotenv_files()
    api_key = get_llm_setting("OPENAI_API_KEY")
    base_url = get_llm_setting("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = get_llm_setting("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        raise RuntimeError("缺少 API key。请在 .env、env.txt 或系统环境变量中设置 OPENAI_API_KEY。")
    if not model:
        raise RuntimeError("缺少模型名。请在 .env、env.txt 或系统环境变量中设置 OPENAI_MODEL。")

    payload = {
        "model": model,
        "temperature": env_float("LLM_TEMPERATURE", 0.35),
        "messages": [
            {
                "role": "system",
                "content": (
                    f"你是 {agent}。你的角色设定：{LIVE_AGENT_PROFILES.get(agent, '交易 agent')} "
                    "你正在一个多智能体股票交易社交系统中互动。"
                    "请用中文、第一人称、简洁地发言。不要输出 Markdown，不要解释自己是 AI。"
                    "语言必须像交易员工作消息，禁止诗句、隐喻、口号、hashtag 和文艺化表达。"
                    "你的回复会直接进入交易大厅 ChatLab。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"频道：{channel}\n"
                    f"接收方：{', '.join(receivers) if receivers else '所有人'}\n"
                    f"最近对话：\n{recent_chat_context(recent_events)}\n\n"
                    f"本轮任务：{instruction}\n"
                    "请生成这一条 agent 消息，长度控制在 1 句话，必须具体到 ticker、方向或风险点。"
                ),
            },
        ],
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=env_int("LLM_TIMEOUT", 60)) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM API HTTP {exc.code}: {detail}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise RuntimeError(f"LLM API 读取超时：超过 {env_int('LLM_TIMEOUT', 60)} 秒未返回。") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM API 连接失败：{exc.reason}") from exc
    try:
        content = str(data["choices"][0]["message"]["content"]).strip()
        return sanitize_agent_text(content, "我本轮只更新交易判断，不发布额外观点。", limit=120)
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"LLM API 返回格式无法解析：{data}") from exc


def call_llm_agent_plan(
    agent: str,
    agents: list[str],
    friendships_df: pd.DataFrame,
    recent_events: pd.DataFrame,
) -> dict:
    load_dotenv_files()
    api_key = get_llm_setting("OPENAI_API_KEY")
    base_url = get_llm_setting("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = get_llm_setting("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        raise RuntimeError("缺少 API key。请在 .env、env.txt 或系统环境变量中设置 OPENAI_API_KEY。")
    if not model:
        raise RuntimeError("缺少模型名。请在 .env、env.txt 或系统环境变量中设置 OPENAI_MODEL。")

    peers = [name for name in agents if name != agent]
    friends = friends_for_agent(friendships_df, agent)
    ticker_example = default_ticker_for_agent(agent)
    payload = {
        "model": model,
        "temperature": env_float("LLM_TEMPERATURE", 0.35),
        "messages": [
            {
                "role": "system",
                "content": (
                    f"你是 {agent}。角色设定：{LIVE_AGENT_PROFILES.get(agent, '交易 agent')} "
                    "你在一个多智能体股票交易社交系统中同时交易和社交。"
                    "你必须只输出一个 JSON 对象，不要输出 Markdown、代码块或额外解释。"
                    "语言必须像研究员/交易员工作记录：具体、克制、可验证。"
                    "禁止诗句、隐喻、成语改写、口号、hashtag、玄学表达和文艺化措辞。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "本轮需要一次性决定你的所有行为：交易、公聊、私聊、朋友圈、好友申请。\n"
                    f"行情数据源：{LIVE_MARKET_DATA_PATH}\n"
                    f"可交易 ticker：{', '.join(live_tradable_tickers())}\n"
                    f"当前真实历史行情截面：\n{live_market_context(next_live_event_id())}\n"
                    f"其他 Agent：{', '.join(peers)}\n"
                    f"当前好友：{', '.join(friends) if friends else '暂无'}\n"
                    f"你的组合状态：{agent_portfolio_context(agent)}\n"
                    f"最近事件：\n{recent_event_context(recent_events)}\n\n"
                    "输出 JSON schema：\n"
                    "{\n"
                    f'  "trade": {{"side": "BUY|SELL|HOLD", "ticker": "{ticker_example}", "shares": 1, "price": 0, "rationale": "交易理由"}},\n'
                    '  "public_message": "发到群聊的一句话",\n'
                    '  "private_message": {"to": "某个 Agent", "content": "私聊内容"},\n'
                    '  "moment": "朋友圈动态",\n'
                    '  "friend_request": {"to": "某个非好友 Agent", "reason": "申请理由"}\n'
                    "}\n"
                    "要求：trade 必须存在，优先给出可执行的 BUY 或 SELL，只有风险理由明确时才 HOLD；"
                    "public_message 必须存在，最多 40 个汉字，必须包含 ticker、方向和原因；"
                    "private_message/moment 最多 50 个汉字；"
                    "如果消息里说加仓、买入、减仓、卖出，trade.side 必须与消息一致；"
                    "friend_request 只能在你能说清楚信息互补、风险验证或策略分歧价值时提出，理由最多 50 个汉字；"
                    "private_message、moment、friend_request 可为空字符串或 null。"
                ),
            },
        ],
    }
    if truthy_env("LLM_JSON_MODE", default=False):
        payload["response_format"] = {"type": "json_object"}
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=env_int("LLM_TIMEOUT", 60)) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM API HTTP {exc.code}: {detail}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise RuntimeError(f"LLM API 读取超时：超过 {env_int('LLM_TIMEOUT', 60)} 秒未返回。") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM API 连接失败：{exc.reason}") from exc
    try:
        content = str(data["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"LLM API 返回格式无法解析：{data}") from exc
    plan = extract_json_object(content)
    if not plan:
        plan = {"trade": {"side": "HOLD", "rationale": "LLM 输出不是 JSON，已回退 HOLD。"}, "public_message": content}
    return normalize_agent_plan(agent, plan)


def fallback_llm_plan(agent: str, reason: str) -> dict:
    ticker = default_ticker_for_agent(agent)
    short_reason = truncate(str(reason or "LLM 调用失败"), 120)
    return normalize_agent_plan(
        agent,
        {
            "trade": {
                "side": "HOLD",
                "ticker": ticker,
                "shares": 0,
                "price": 0,
                "rationale": f"LLM 本轮不可用，回退 HOLD：{short_reason}",
            },
            "public_message": f"{ticker} 本轮 LLM 响应超时，我先保持观望。",
            "private_message": None,
            "moment": None,
            "friend_request": None,
        },
    )


def call_llm_friend_decision(agent: str, receiver: str, reason: str, recent_events: pd.DataFrame) -> dict:
    load_dotenv_files()
    api_key = get_llm_setting("OPENAI_API_KEY")
    base_url = get_llm_setting("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = get_llm_setting("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key or not model:
        raise RuntimeError("缺少 LLM API 配置。")

    receiver_state = agent_portfolio_context(receiver)
    current_friendships = pd.DataFrame(st.session_state.get("live_friendships", []), columns=["agent_a", "agent_b"])
    receiver_friends = friends_for_agent(current_friendships, receiver)
    payload = {
        "model": model,
        "temperature": env_float("LLM_TEMPERATURE", 0.35),
        "messages": [
            {
                "role": "system",
                "content": (
                    f"你是 {receiver}，需要审批 {agent} 的好友申请。"
                    "只输出 JSON。判断标准是研究价值，不是礼貌：是否有互补信息、风险校验、策略分歧价值、可控好友数量。"
                    "禁止诗化表达。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"申请方：{agent}\n"
                    f"申请理由：{reason}\n"
                    f"你的当前好友：{', '.join(receiver_friends) if receiver_friends else '暂无'}\n"
                    f"你的组合状态：{receiver_state}\n"
                    f"最近事件：\n{recent_event_context(recent_events, limit=16)}\n\n"
                    '输出：{"accept": true|false, "reason": "不超过40字的具体原因"}'
                ),
            },
        ],
    }
    if truthy_env("LLM_JSON_MODE", default=False):
        payload["response_format"] = {"type": "json_object"}
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=env_int("LLM_TIMEOUT", 60)) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM API HTTP {exc.code}: {detail}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise RuntimeError(f"LLM API 读取超时：超过 {env_int('LLM_TIMEOUT', 60)} 秒未返回。") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM API 连接失败：{exc.reason}") from exc
    try:
        content = str(data["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"LLM API 返回格式无法解析：{data}") from exc
    parsed = extract_json_object(content)
    if not parsed:
        raise RuntimeError("好友审批 LLM 未返回 JSON。")
    return {
        "accept": bool(parsed.get("accept")),
        "reason": sanitize_agent_text(parsed.get("reason"), "理由不够具体。", limit=60),
    }


def extract_json_object(text: str) -> dict:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def recent_chat_context(events: pd.DataFrame, limit: int = 18) -> str:
    if events.empty:
        return "暂无历史对话。"
    chat = events[events["event_type"] == "message"].sort_values("event_id").tail(limit)
    if chat.empty:
        return "暂无历史对话。"
    lines = []
    for _, row in chat.iterrows():
        receivers = event_receivers(row)
        target = f" -> {', '.join(receivers)}" if receivers else ""
        detail = sanitize_agent_text(row.get("detail", ""), "[漂移文本已忽略]", limit=100)
        lines.append(
            f"#{int(row.get('event_id', 0))} [{row.get('channel', '')}] "
            f"{row.get('agent', '')}{target}: {detail}"
        )
    return "\n".join(lines)


def recent_event_context(events: pd.DataFrame, limit: int = 28) -> str:
    if events.empty:
        return "暂无历史事件。"
    work = events[events["event_type"].isin(["message", "trade", "hold", "friend_request", "friend_accept"])].tail(limit)
    if work.empty:
        return "暂无历史事件。"
    lines = []
    for _, row in work.iterrows():
        event_type = row.get("event_type", "")
        channel = row.get("channel", "")
        ticker = row.get("ticker", "")
        side = row.get("side", "")
        detail = sanitize_agent_text(row.get("detail", ""), "[漂移文本已忽略]", limit=120)
        lines.append(
            f"#{int(row.get('event_id', 0))} [{event_type}/{channel}] "
            f"{row.get('agent', '')} {side} {ticker}: {detail}"
        )
    return "\n".join(lines)


def live_market_context(event_id: int, max_tickers: int = 12) -> str:
    if LIVE_MARKET_HISTORY.empty:
        prices = live_ticker_prices()
        return "真实行情文件读取失败，临时使用备用价格：" + ", ".join(
            f"{ticker} close={price:.2f}" for ticker, price in list(prices.items())[:max_tickers]
        )
    target_date = live_market_date_for_event(event_id)
    tickers = live_tradable_tickers()[:max_tickers]
    lines = []
    for ticker in tickers:
        work = LIVE_MARKET_HISTORY[LIVE_MARKET_HISTORY["ticker"].astype(str) == ticker].copy()
        if work.empty:
            continue
        work["date"] = pd.to_datetime(work["date"])
        if target_date is not None:
            work = work[work["date"] <= target_date]
        work = work.sort_values("date")
        if work.empty:
            continue
        close = float(work["close"].iloc[-1])
        ret5 = period_return(work["close"], 5)
        ret20 = period_return(work["close"], 20)
        volume = float(work["volume"].iloc[-1]) if "volume" in work.columns else 0.0
        lines.append(
            f"{ticker}: date={work['date'].iloc[-1].strftime('%Y-%m-%d')}, "
            f"close={close:.2f}, ret5={ret5:.2%}, ret20={ret20:.2%}, volume={volume:.0f}"
        )
    if not lines:
        return "暂无可用真实行情截面。"
    return "\n".join(lines)


def period_return(series: pd.Series, periods: int) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) <= periods:
        return 0.0
    start = float(values.iloc[-periods - 1])
    end = float(values.iloc[-1])
    return (end / start - 1.0) if start else 0.0


def agent_portfolio_context(agent: str) -> str:
    state = latest_live_agent_state(agent)
    return (
        f"cash={money(state.get('cash', DEFAULT_STARTING_CASH))}, "
        f"equity={money(state.get('equity', DEFAULT_STARTING_CASH))}, "
        f"pnl={money(state.get('pnl', 0))}, "
        f"positions={state.get('position_summary') or 'cash only'}"
    )


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
    out["_seq"] = range(len(out))
    return out.sort_values(["date", "agent", "_seq"]).drop(columns=["_seq"])


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


def render_live_sidebar_controls(events: pd.DataFrame, agents: list[str], friendships_df: pd.DataFrame) -> None:
    st.sidebar.markdown("### LLM Agent 总控")
    if has_llm_api_config():
        st.sidebar.caption("API：已读取 env.txt / .env / 系统环境变量")
    else:
        st.sidebar.warning("API：缺少 OPENAI_API_KEY 或 OPENAI_MODEL")

    if st.session_state.pop("live_force_auto_off", False):
        st.session_state.live_auto_agents = False
        st.session_state.live_auto_agents_enabled = False
    if "live_auto_agents_enabled" not in st.session_state:
        st.session_state.live_auto_agents_enabled = bool(st.session_state.get("live_auto_agents", False))

    scenarios = load_live_scenarios()
    if scenarios:
        st.sidebar.markdown("### 初始社交图谱")
        scenario_names = sorted(scenarios)
        current = st.session_state.get("live_scenario", DEFAULT_LIVE_SCENARIO)
        index = scenario_names.index(current) if current in scenario_names else 0
        selected_scenario = st.sidebar.selectbox(
            "选择网络实验场景",
            scenario_names,
            index=index,
            format_func=lambda name: scenario_summary(name, scenarios.get(name, {})),
            key="live_scenario_selector",
        )
        st.sidebar.caption(
            "用于研究不同初始关系对信息传播、好友演化、交易同步和收益风险的影响。"
        )
        if st.sidebar.button("应用该初始图谱", use_container_width=True, key="apply_live_social_scenario"):
            st.session_state.live_scenario = selected_scenario
            reset_live_social_graph_from_yaml()
            rerun_app()

    st.sidebar.slider("自动轮次间隔秒", 3, 120, step=1, key="live_auto_interval")
    st.sidebar.number_input("一键启动轮数", min_value=1, max_value=50, value=1, step=1, key="live_start_rounds")
    st.sidebar.caption(
        f"LLM_TIMEOUT={env_int('LLM_TIMEOUT', 60)} 秒；"
        f"LLM_MAX_WORKERS={env_int('LLM_MAX_WORKERS', len(agents))}。"
        "如果频繁超时，建议在 env.txt 设置 LLM_TIMEOUT=120、LLM_MAX_WORKERS=3。"
    )
    auto_enabled = st.sidebar.toggle("持续自动运行全部行为", key="live_auto_agents_enabled")
    st.session_state.live_auto_agents = bool(auto_enabled)
    st.sidebar.caption("每轮会并发调用所有 Agent，并生成交易、公聊、私聊、朋友圈和好友申请。")
    st.sidebar.caption("Portfolio Hall 的 manager 曲线/相关矩阵来自离线实验，请把数据源切到“本地事件回放”查看。")

    st.sidebar.button(
        "一键启动所有 Agent",
        type="primary",
        use_container_width=True,
        key="start_all_live_agents",
        on_click=request_live_agent_start,
    )
    if st.session_state.pop("live_start_all_requested", False):
        if not has_llm_api_config():
            st.sidebar.error("无法启动：请先检查 env.txt 里的 LLM API 配置。")
            st.session_state.live_force_auto_off = True
        else:
            run = begin_new_live_run()
            requested_rounds = int(st.session_state.get("live_start_rounds", 1))
            with st.spinner(f"正在并发启动所有 Agent 的完整行为，共 {requested_rounds} 轮..."):
                try:
                    generate_live_agent_rounds(agents=agents, n_rounds=requested_rounds)
                    st.session_state.live_auto_last_ts = time.time()
                    persist_live_state()
                    rerun_app()
                except RuntimeError as exc:
                    st.session_state.live_auto_agents = False
                    st.session_state.live_force_auto_off = True
                    st.session_state.live_active_run_id = ""
                    st.session_state.live_active_round_id = ""
                    run["status"] = "error"
                    run["error"] = str(exc)
                    run["updated_at"] = current_event_timestamp()
                    st.session_state.live_current_run = run
                    persist_live_state()
                    st.sidebar.error(str(exc))

    if st.sidebar.button("停止自动生成", use_container_width=True, key="stop_all_live_agents", on_click=request_live_agent_stop):
        persist_live_state()
        st.sidebar.success("已停止自动生成。")

    st.sidebar.button(
        "继续上一次生成",
        use_container_width=True,
        key="resume_live_agents",
        on_click=request_live_agent_resume,
    )
    if st.session_state.pop("live_resume_requested", False):
        if not has_llm_api_config():
            st.sidebar.error("无法继续：请先检查 env.txt 里的 LLM API 配置。")
            st.session_state.live_force_auto_off = True
        elif not (st.session_state.get("live_current_run") or {}).get("id"):
            st.sidebar.error("没有可继续的上一轮生成。请先点一次“一键启动所有 Agent”。")
            st.session_state.live_force_auto_off = True
        else:
            requested_rounds = int(st.session_state.get("live_start_rounds", 1))
            with st.spinner(f"正在继续上一次 Agent 生成，共 {requested_rounds} 轮..."):
                try:
                    generate_live_agent_rounds(agents=agents, n_rounds=requested_rounds)
                    st.session_state.live_auto_last_ts = time.time()
                    persist_live_state()
                    rerun_app()
                except RuntimeError as exc:
                    st.session_state.live_auto_agents = False
                    st.session_state.live_force_auto_off = True
                    st.session_state.live_active_run_id = ""
                    st.session_state.live_active_round_id = ""
                    pause_live_run()
                    persist_live_state()
                    st.sidebar.error(str(exc))

    round_count = int(st.session_state.get("live_auto_round", 0))
    st.sidebar.caption(f"已完成自动轮次：{round_count}")
    run = st.session_state.get("live_current_run") or {}
    if run.get("id"):
        st.sidebar.caption(f"当前生成目录：outputs/live_state/runs/{run['id']}")
    sync_live_auto_settings()


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


# ----------------------------- Portfolio Hall -----------------------------


def render_portfolio_hall(
    agent_equity,
    manager_equity,
    meta_weights,
    agent_corr,
    experiment_comparison,
    registry,
    drift_log,
    manager_loss=None,
    training_params=None,
    forecast_scores=None,
    agent_views=None,
    bl_weights=None,
):
    st.markdown("#### Portfolio Hall：Agent 组合管理")
    if manager_equity.empty and agent_equity.empty:
        st.info("暂无 agent portfolio 输出。请先运行离线实验。")
        return

    cards = portfolio_hall_cards(experiment_comparison, manager_equity, meta_weights, drift_log)
    st.markdown(render_kpi_cards(cards), unsafe_allow_html=True)

    c1, c2 = st.columns([1.25, 1])
    with c1:
        st.markdown("#### Manager Equity")
        if manager_equity.empty:
            st.info("缺少 manager_equity_curve.csv。")
        else:
            work = manager_equity.copy()
            work["date"] = pd.to_datetime(work["date"], errors="coerce")
            fig = go.Figure()
            for manager, group in work.groupby("manager"):
                group = group.sort_values("date")
                base = safe_float(group["equity"].iloc[0], 1.0) or 1.0
                fig.add_scatter(
                    x=group["date"],
                    y=group["equity"] / base,
                    mode="lines",
                    name=str(manager),
                )
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=30), yaxis_title="Normalized equity")
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### Agent Return Correlation")
        if agent_corr.empty or "agent" not in agent_corr.columns:
            st.info("缺少 agent_return_correlation.csv。")
        else:
            matrix = agent_corr.set_index("agent")
            fig = go.Figure(
                data=go.Heatmap(
                    z=matrix.values,
                    x=list(matrix.columns),
                    y=list(matrix.index),
                    zmin=-1,
                    zmax=1,
                    colorscale="RdBu",
                    reversescale=True,
                )
            )
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=90))
            st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns([1.1, 1])
    with c3:
        st.markdown("#### Meta Weight History")
        if meta_weights.empty:
            st.info("缺少 meta_weight_history.csv。")
        else:
            managers = sorted(meta_weights["manager"].dropna().astype(str).unique())
            selected_manager = st.selectbox("Manager", managers, key="portfolio_hall_manager")
            weights = meta_weights[meta_weights["manager"].astype(str) == selected_manager].copy()
            weights["date"] = pd.to_datetime(weights["date"], errors="coerce")
            fig = go.Figure()
            for agent, group in weights.groupby("agent"):
                group = group.sort_values("date")
                fig.add_scatter(x=group["date"], y=group["weight"], mode="lines", stackgroup="one", name=str(agent))
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=30), yaxis_title="Weight")
            st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.markdown("#### Strategy Contract")
        if registry.empty:
            st.info("缺少 agent_registry.csv。")
        else:
            cols = [
                col
                for col in ["agent", "strategy_family", "strategy_spec_version", "param_hash", "own_strategy"]
                if col in registry.columns
            ]
            st.dataframe(registry[cols].sort_values("agent"), use_container_width=True, hide_index=True)

    st.markdown("#### Experiment Comparison")
    if experiment_comparison.empty:
        st.info("缺少 experiment_comparison.csv。")
    else:
        show_cols = [
            col
            for col in [
                "experiment_type",
                "portfolio",
                "total_return",
                "annual_return",
                "annual_volatility",
                "sharpe",
                "max_drawdown",
                "average_agent_correlation",
                "final_weight_hhi",
                "regret_to_best_agent",
            ]
            if col in experiment_comparison.columns
        ]
        st.dataframe(
            experiment_comparison[show_cols].sort_values("sharpe", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    if not drift_log.empty:
        st.markdown("#### Drift Alerts")
        st.dataframe(drift_log.sort_values(["date", "agent"]), use_container_width=True, hide_index=True)

    c5, c6 = st.columns([1, 1])
    with c5:
        st.markdown("#### Walk-Forward Training")
        if training_params is None or training_params.empty:
            st.info("缺少 training_params.csv。")
        else:
            cols = [col for col in ["agent", "strategy", "fixed_identity", "selected_params_json", "validation_score", "status"] if col in training_params.columns]
            st.dataframe(training_params[cols], use_container_width=True, hide_index=True)
    with c6:
        st.markdown("#### Hedge Loss Breakdown")
        if manager_loss is None or manager_loss.empty:
            st.info("缺少 manager_loss_history.csv。")
        else:
            latest = manager_loss.copy()
            latest["date"] = pd.to_datetime(latest["date"], errors="coerce")
            latest = latest.sort_values("date").groupby(["manager", "agent"], as_index=False).tail(1)
            cols = [col for col in ["manager", "agent", "previous_weight", "agent_return", "total_loss"] if col in latest.columns]
            st.dataframe(latest[cols].sort_values(["manager", "total_loss"]), use_container_width=True, hide_index=True)

    c7, c8 = st.columns([1, 1])
    with c7:
        st.markdown("#### Proper Scoring")
        if forecast_scores is None or forecast_scores.empty:
            st.info("缺少 forecast_scores.csv。")
        else:
            summary = (
                forecast_scores.groupby("sender_id")
                .agg(forecasts=("message_id", "count"), mean_brier=("brier_score", "mean"), proper_score=("proper_score", "mean"))
                .reset_index()
                .sort_values("proper_score", ascending=False)
            )
            st.dataframe(summary, use_container_width=True, hide_index=True)
    with c8:
        st.markdown("#### Black-Litterman Agent Views")
        if bl_weights is None or bl_weights.empty:
            st.info("缺少 bl_agent_view_weights.csv。")
        else:
            st.dataframe(bl_weights.sort_values("weight", ascending=False), use_container_width=True, hide_index=True)
            if agent_views is not None and not agent_views.empty:
                st.caption(f"已提取 {len(agent_views):,} 条 agent views。")


def portfolio_hall_cards(experiment_comparison, manager_equity, meta_weights, drift_log):
    best_manager = "-"
    best_sharpe = "-"
    if not experiment_comparison.empty and {"experiment_type", "sharpe", "portfolio"}.issubset(experiment_comparison.columns):
        managers = experiment_comparison[experiment_comparison["experiment_type"] == "agent_portfolio"].copy()
        if not managers.empty:
            row = managers.sort_values("sharpe", ascending=False).iloc[0]
            best_manager = str(row["portfolio"])
            best_sharpe = f"{safe_float(row['sharpe']):.2f}"

    final_equity = "-"
    if not manager_equity.empty:
        latest = manager_equity.copy()
        latest["date"] = pd.to_datetime(latest["date"], errors="coerce")
        last_date = latest["date"].max()
        row = latest[latest["date"] == last_date].sort_values("equity", ascending=False).iloc[0]
        final_equity = f"{row.get('manager', '-')}: {money(row.get('equity', 0))}"

    hhi_text = "-"
    if not meta_weights.empty:
        latest = meta_weights.copy()
        latest["date"] = pd.to_datetime(latest["date"], errors="coerce")
        latest = latest.sort_values("date").groupby(["manager", "agent"], as_index=False).tail(1)
        hhi = latest.groupby("manager")["weight"].apply(lambda values: float((values**2).sum()))
        if not hhi.empty:
            manager = hhi.idxmin()
            hhi_text = f"{manager}: {hhi.loc[manager]:.3f}"

    return [
        ("Best Manager", best_manager, f"Sharpe {best_sharpe}"),
        ("Latest Manager Equity", final_equity, "manager_equity_curve"),
        ("Lowest HHI", hhi_text, "meta weight concentration"),
        ("Drift Alerts", f"{len(drift_log):,}", "strategy contract violations"),
    ]


# ----------------------------- ChatLab -----------------------------


def render_chatlab(agents, events_until, friendships_df, cursor_id, auto_scroll, live_api_mode=False):
    st.markdown("#### ChatLab：点击 Agent 后同步查看公聊、私聊、朋友圈、好友申请")
    if not agents:
        st.info("没有 Agent。")
        return
    if st.session_state.get("selected_chat_agent") not in agents:
        st.session_state.selected_chat_agent = agents[0]
    if live_api_mode:
        render_live_llm_controls(agents, events_until, friendships_df)

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


def render_live_llm_controls(agents, events_until, friendships_df):
    selected = st.session_state.get("selected_chat_agent", agents[0])
    with st.expander("单 Agent 调试", expanded=False):
        if has_llm_api_config():
            st.caption("API 状态：已从 env.txt / .env / 系统环境变量读取配置。")
        else:
            st.warning("API 状态：缺少 OPENAI_API_KEY 或 OPENAI_MODEL。请检查 env.txt。")
        st.caption("全局的一键启动和持续自动运行已移到左侧边栏；这里仅保留单个 Agent 的人工调试入口。")

        c1, c2, c3 = st.columns([1, 1, 1])
        sender_index = agents.index(selected) if selected in agents else 0
        sender = c1.selectbox("发言 Agent", agents, index=sender_index, key="live_sender_agent")
        channel = c2.selectbox(
            "频道",
            ["public", "private", "moments"],
            format_func=lambda value: {"public": "群聊", "private": "私聊", "moments": "朋友圈"}[value],
            key="live_channel",
        )
        ticker = c3.text_input("Ticker / 主题", value="", key="live_ticker").strip().upper()

        possible_receivers = [agent for agent in agents if agent != sender]
        if channel == "private":
            receivers = st.multiselect("私聊接收方", possible_receivers, key="live_receivers")
        elif channel == "moments":
            receivers = friends_for_agent(friendships_df, sender)
            st.caption("朋友圈可见好友：" + (", ".join(receivers) if receivers else "暂无好友，仅自己可见"))
        else:
            receivers = []

        instruction = st.text_area(
            "给这个 Agent 的本轮任务 / 用户输入",
            value="结合当前对话，给出你的市场观点或回应其他 agent。",
            height=90,
            key="live_instruction",
        )
        b1, b2 = st.columns([1, 1])
        if b1.button("调用 LLM 生成并发送", type="primary", use_container_width=True):
            if channel == "private" and not receivers:
                st.error("私聊需要至少选择一个接收方。")
            else:
                with st.spinner(f"正在调用 LLM 生成 {sender} 的消息..."):
                    try:
                        content = call_llm_agent(sender, channel, receivers, instruction, events_until)
                        append_live_message_event(sender, channel, receivers, content, ticker)
                        st.session_state.selected_chat_agent = sender
                        st.success("消息已写入实时事件流。")
                        rerun_app()
                    except RuntimeError as exc:
                        st.error(str(exc))
        if b2.button("直接发送用户输入", use_container_width=True):
            if channel == "private" and not receivers:
                st.error("私聊需要至少选择一个接收方。")
            else:
                append_live_message_event(sender, channel, receivers, instruction, ticker)
                st.session_state.selected_chat_agent = sender
                rerun_app()

        with st.popover("好友关系"):
            friend_target = st.selectbox("添加好友", possible_receivers, key="live_friend_target")
            if st.button("建立好友关系", use_container_width=True):
                append_live_friendship(sender, friend_target)
                rerun_app()
            current_friends = friends_for_agent(friendships_df, sender)
            st.caption("当前好友：" + (", ".join(current_friends) if current_friends else "暂无"))


def run_live_agent_autoplay(events_until: pd.DataFrame, agents: list[str], friendships_df: pd.DataFrame) -> None:
    if not st.session_state.get("live_auto_agents", False):
        return
    if not has_llm_api_config():
        st.session_state.live_auto_agents = False
        st.session_state.live_force_auto_off = True
        st.error("自动生成已停止：缺少 OPENAI_API_KEY 或 OPENAI_MODEL。请检查 env.txt。")
        return
    interval = int(st.session_state.get("live_auto_interval", 12))
    last_ts = float(st.session_state.get("live_auto_last_ts", 0.0))
    now_ts = time.time()
    if now_ts - last_ts < interval:
        time.sleep(max(0.2, min(2.0, interval - (now_ts - last_ts))))
        rerun_app()
        return
    try:
        generate_live_agent_round(agents=agents, events_until=events_until, friendships_df=friendships_df)
        st.session_state.live_auto_last_ts = time.time()
        persist_live_state()
    except RuntimeError as exc:
        st.session_state.live_auto_agents = False
        st.session_state.live_force_auto_off = True
        st.session_state.live_active_run_id = ""
        st.session_state.live_active_round_id = ""
        pause_live_run()
        persist_live_state()
        st.error(f"自动生成已停止：{exc}")
        return
    time.sleep(0.2)
    rerun_app()


def generate_live_agent_round(
    agents: list[str],
    events_until: pd.DataFrame,
    friendships_df: pd.DataFrame,
    channel: str = "full",
) -> None:
    usable_agents = [agent for agent in agents if agent in LIVE_AGENT_PROFILES]
    if not usable_agents:
        raise RuntimeError("没有可用于 LLM 交互的 agent。")
    run = ensure_live_run()
    round_number = int(run.get("round_count", 0)) + 1
    round_id = f"round_{round_number:03d}"
    st.session_state.live_active_run_id = run["id"]
    st.session_state.live_active_round_id = round_id
    start_event_id = next_live_event_id()
    context = events_until.copy()
    if not context.empty and "run_id" in context.columns:
        context = context[context["run_id"].astype(str) == str(run["id"])].copy()
    jobs = [{"sender": sender} for sender in usable_agents]

    max_workers = max(1, min(len(jobs), env_int("LLM_MAX_WORKERS", len(jobs))))
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                call_llm_agent_plan,
                job["sender"],
                usable_agents,
                friendships_df,
                context,
            ): job
            for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                results.append({**job, "plan": future.result(), "error": ""})
            except Exception as exc:
                error = f"{exc.__class__.__name__}: {exc}"
                results.append({**job, "plan": fallback_llm_plan(job["sender"], error), "error": error})

    order = {job["sender"]: index for index, job in enumerate(jobs)}
    for row in sorted(results, key=lambda item: order[item["sender"]]):
        append_live_agent_plan_events(row["sender"], row["plan"], usable_agents, friendships_df)
    end_event_id = next_live_event_id() - 1
    run["round_count"] = round_number
    run["status"] = "running" if st.session_state.get("live_auto_agents", False) else "paused"
    run["updated_at"] = current_event_timestamp()
    st.session_state.live_current_run = run
    st.session_state.live_auto_round = int(st.session_state.get("live_auto_round", 0)) + 1
    export_live_run_round(run, round_id, start_event_id, end_event_id, results)
    st.session_state.live_active_run_id = ""
    st.session_state.live_active_round_id = ""
    persist_live_state()


def generate_live_agent_rounds(agents: list[str], n_rounds: int) -> None:
    total = max(1, int(n_rounds))
    progress = st.progress(0, text=f"LLM round 0 / {total}")
    try:
        for index in range(total):
            generate_live_agent_round(
                agents=agents,
                events_until=current_live_events_dataframe(),
                friendships_df=current_live_friendships_dataframe(),
            )
            progress.progress((index + 1) / total, text=f"LLM round {index + 1} / {total}")
    finally:
        progress.empty()


def current_live_events_dataframe() -> pd.DataFrame:
    events = pd.DataFrame(st.session_state.get("live_events", []))
    return normalize_events(events) if not events.empty else events


def current_live_friendships_dataframe() -> pd.DataFrame:
    return pd.DataFrame(st.session_state.get("live_friendships", []), columns=["agent_a", "agent_b"])


def sync_live_auto_settings() -> None:
    persist_live_state()


def export_live_run_round(run: dict, round_id: str, start_event_id: int, end_event_id: int, results: list[dict]) -> None:
    run_id = str(run.get("id", ""))
    if not run_id:
        return
    run_dir = LIVE_RUNS_DIR / run_id
    round_dir = run_dir / "rounds" / round_id
    round_dir.mkdir(parents=True, exist_ok=True)
    events = pd.DataFrame(st.session_state.get("live_events", []))
    states = pd.DataFrame(st.session_state.get("live_state_history", []))
    if events.empty:
        round_events = pd.DataFrame()
        run_events = pd.DataFrame()
    else:
        event_ids = pd.to_numeric(events.get("event_id"), errors="coerce").fillna(0).astype(int)
        round_events = events[(event_ids >= start_event_id) & (event_ids <= end_event_id)].copy()
        run_events = events[events.get("run_id", "").astype(str) == run_id].copy() if "run_id" in events.columns else pd.DataFrame()
    if states.empty or "run_id" not in states.columns:
        round_states = pd.DataFrame()
        run_states = pd.DataFrame()
    else:
        round_states = states[
            (states["run_id"].astype(str) == run_id) & (states["round_id"].astype(str) == round_id)
        ].copy()
        run_states = states[states["run_id"].astype(str) == run_id].copy()

    round_events.to_csv(round_dir / "unified_event_log.csv", index=False)
    round_states.to_csv(round_dir / "agent_state_history.csv", index=False)
    live_trades_table(round_events).to_csv(round_dir / "trade_log.csv", index=False)
    live_messages_table(round_events).to_csv(round_dir / "message_log.csv", index=False)
    live_social_events_table(round_events).to_csv(round_dir / "social_events.csv", index=False)
    LIVE_MARKET_HISTORY.to_csv(round_dir / "market_history.csv", index=False)
    (round_dir / "raw_llm_plans.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    run_events.to_csv(tables_dir / "unified_event_log.csv", index=False)
    run_states.to_csv(tables_dir / "agent_state_history.csv", index=False)
    live_trades_table(run_events).to_csv(tables_dir / "trade_log.csv", index=False)
    live_messages_table(run_events).to_csv(tables_dir / "message_log.csv", index=False)
    live_social_events_table(run_events).to_csv(tables_dir / "social_events.csv", index=False)
    LIVE_MARKET_HISTORY.to_csv(tables_dir / "market_history.csv", index=False)
    live_social_edges_dataframe().to_csv(tables_dir / "social_graph_edges.csv", index=False)
    pd.DataFrame(st.session_state.get("live_friendships", []), columns=["agent_a", "agent_b"]).to_csv(
        tables_dir / "friendships.csv",
        index=False,
    )
    live_groups_dataframe().to_csv(tables_dir / "group_memberships.csv", index=False)
    write_live_run_metadata(
        run,
        last_round_id=round_id,
        last_round_start_event_id=start_event_id,
        last_round_end_event_id=end_event_id,
    )


def append_live_agent_plan_events(agent: str, plan: dict, agents: list[str], friendships_df: pd.DataFrame) -> None:
    if not isinstance(plan, dict):
        plan = {}
    plan = normalize_agent_plan(agent, plan)
    trade = trade_plan_from_agent_plan(agent, plan)
    append_live_trade_event(agent, trade)
    ticker = normalize_trade_plan(agent, trade).get("ticker", "")

    public_message = coerce_plan_text(plan.get("public_message"))
    if not public_message:
        public_message = f"我本轮完成了交易判断，当前组合：{agent_portfolio_context(agent)}。"
    append_live_message_event(agent, "public", [], public_message, ticker=ticker)

    private_message = plan.get("private_message")
    if isinstance(private_message, dict):
        target = normalize_agent_target(private_message.get("to"), agent, agents)
        content = coerce_plan_text(private_message.get("content"))
        if target and content:
            if not live_are_friends(agent, target):
                add_live_friend_request_with_decision(
                    agent,
                    target,
                    f"希望私聊验证观点：{truncate(content, 60)}",
                    pd.DataFrame(st.session_state.get("live_events", [])),
                )
            if live_are_friends(agent, target):
                append_live_message_event(agent, "private", [target], content, ticker=ticker)

    moment = coerce_plan_text(plan.get("moment"))
    if moment:
        current_friendships = pd.DataFrame(st.session_state.get("live_friendships", []), columns=["agent_a", "agent_b"])
        append_live_message_event(agent, "moments", friends_for_agent(current_friendships, agent), moment, ticker=ticker)

    friend_request = plan.get("friend_request")
    if isinstance(friend_request, dict):
        target = normalize_agent_target(friend_request.get("to"), agent, agents)
        reason = coerce_plan_text(friend_request.get("reason")) or "希望建立信息共享关系。"
        if target and not live_are_friends(agent, target):
            add_live_friend_request_with_decision(
                agent,
                target,
                reason,
                pd.DataFrame(st.session_state.get("live_events", [])),
            )
            reinforce_live_influence(agent, [target])
            st.session_state.event_cursor = next_live_event_id() - 1
            persist_live_state()


def coerce_plan_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "null", "none"}:
        return ""
    return sanitize_agent_text(text, "", limit=320)


def normalize_agent_plan(agent: str, plan: dict) -> dict:
    if not isinstance(plan, dict):
        plan = {}
    normalized = dict(plan)
    trade = normalized.get("trade") if isinstance(normalized.get("trade"), dict) else {}
    normalized["trade"] = {
        **trade,
        "rationale": sanitize_agent_text(
            trade.get("rationale"),
            f"{agent} 按当前信号执行结构化交易判断。",
            limit=120,
        ),
    }
    normalized["public_message"] = sanitize_agent_text(
        normalized.get("public_message"),
        public_fallback_from_trade(agent, normalized["trade"]),
        limit=80,
    )
    normalized["moment"] = sanitize_agent_text(normalized.get("moment"), "", limit=90)
    private = normalized.get("private_message")
    if isinstance(private, dict):
        normalized["private_message"] = {
            "to": private.get("to", ""),
            "content": sanitize_agent_text(private.get("content"), "", limit=90),
        }
    else:
        normalized["private_message"] = None
    request = normalized.get("friend_request")
    if isinstance(request, dict):
        normalized["friend_request"] = {
            "to": request.get("to", ""),
            "reason": sanitize_agent_text(request.get("reason"), "", limit=90),
        }
        if not normalized["friend_request"]["reason"]:
            normalized["friend_request"] = None
    else:
        normalized["friend_request"] = None
    return normalized


def sanitize_agent_text(value, fallback: str = "", limit: int = 120) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"null", "none"}:
        return fallback
    text = re.sub(r"#[A-Za-z0-9_.-]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("；；", "；").replace("。。", "。")
    if looks_like_drifty_text(text):
        return fallback
    return truncate(text, limit)


def looks_like_drifty_text(text: str) -> bool:
    work = str(text or "")
    banned_terms = [
        "旧帐篷",
        "余烬",
        "裂变",
        "拾薪",
        "火焰",
        "痕",
        "星辰",
        "号角",
        "战鼓",
        "迷雾",
        "命运",
        "诗",
        "寓言",
        "炼金",
        "残响",
        "烟火",
        "众人拾薪",
        "潮汐",
        "旷野",
        "灯塔",
    ]
    if "#" in work:
        return True
    if any(term in work for term in banned_terms):
        return True
    punctuation_count = sum(work.count(mark) for mark in ["，", "。", "；", "！"])
    return len(work) > 90 and punctuation_count >= 4 and not any(ticker in work.upper() for ticker in LIVE_TICKER_PRICES)


def public_fallback_from_trade(agent: str, trade: dict) -> str:
    normalized = normalize_trade_plan(agent, trade)
    side = normalized.get("side", "HOLD")
    ticker = normalized.get("ticker", default_ticker_for_agent(agent))
    if side == "BUY":
        return f"{ticker} 趋势信号占优，我小幅买入并控制仓位。"
    if side == "SELL":
        return f"{ticker} 风险收益变差，我减仓降低敞口。"
    return f"{ticker} 信号不够清晰，我本轮保持观望。"


def normalize_agent_target(value, sender: str, agents: list[str]) -> str:
    target = str(value or "").strip()
    if target in agents and target != sender:
        return target
    candidates = [agent for agent in agents if agent != sender]
    if not candidates:
        return ""
    return candidates[stable_index(target or sender, len(candidates))]


def auto_receivers_for(
    sender: str,
    agents: list[str],
    friendships_df: pd.DataFrame,
    channel: str,
    index: int,
) -> list[str]:
    if channel == "public":
        return []
    friends = friends_for_agent(friendships_df, sender)
    if channel == "moments":
        return friends
    candidates = friends or [agent for agent in agents if agent != sender]
    if not candidates:
        return []
    return [candidates[index % len(candidates)]]


def auto_agent_instruction(sender: str, channel: str, receivers: list[str]) -> str:
    if channel == "private":
        target = ", ".join(receivers) if receivers else "对方"
        return f"你正在和 {target} 私聊。请结合最近上下文，推进一次具体交易讨论或信息交换。"
    if channel == "moments":
        return "你正在发朋友圈。请发布一条市场观察、持仓态度或社交策略动态。"
    return "你正在群聊。请回应最近市场/其他 agent 观点，并给出一个清晰的交易或风险判断。"


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
    for item in aggregate_friend_requests(req_df)[-160:]:
        status = item["status"]
        label = {"accepted": "已通过", "rejected": "已拒绝", "pending": "等待验证"}.get(status, "等待验证")
        sender = item["sender"]
        counterparty = item["counterparty"]
        sent_line = f"发送时间：{item['request_time']}" if item.get("request_time") else "发送时间：未知"
        decided_label = "通过时间" if status == "accepted" else "处理时间"
        decided_line = f"{decided_label}：{item['decision_time']}" if item.get("decision_time") else ""
        decision_html = f"<div class='request-detail'>{html.escape(decided_line)}</div>" if decided_line else ""
        cards.append(
            "<div class='request-card'>"
            f"<div class='avatar'>{html.escape(short_name(sender or counterparty))}</div>"
            "<div class='request-main'>"
            f"<div class='request-title'>{html.escape(sender)} → {html.escape(counterparty)}</div>"
            f"<div class='request-detail'>{html.escape(item.get('detail', ''))}</div>"
            f"<div class='request-detail'>{html.escape(sent_line)}</div>"
            f"{decision_html}"
            "</div>"
            f"<div class='status-{status}'>{label}</div>"
            "</div>"
        )
    render_phone_component(title, "".join(cards), cursor_id, auto_scroll, height=640, input_text="好友申请由仿真策略自动处理")


def aggregate_friend_requests(req_df: pd.DataFrame) -> list[dict]:
    items = {}
    standalone = []
    for _, row in req_df.sort_values("event_id").iterrows():
        event_type = str(row.get("event_type", ""))
        agent = str(row.get("agent", ""))
        counterparty = str(row.get("counterparty", ""))
        if not agent or not counterparty:
            continue
        if event_type == "friend_request":
            key = (agent, counterparty)
            items[key] = {
                "sender": agent,
                "counterparty": counterparty,
                "status": "pending",
                "request_time": str(row.get("event_time", "") or row.get("date", "")),
                "decision_time": "",
                "detail": str(row.get("detail", "")),
                "event_id": int(row.get("event_id", 0)),
            }
        elif event_type in {"friend_accept", "friend_reject"}:
            key = (counterparty, agent)
            status = "accepted" if event_type == "friend_accept" else "rejected"
            if key not in items:
                items[key] = {
                    "sender": counterparty,
                    "counterparty": agent,
                    "status": status,
                    "request_time": "",
                    "decision_time": str(row.get("event_time", "") or row.get("date", "")),
                    "detail": str(row.get("detail", "")),
                    "event_id": int(row.get("event_id", 0)),
                }
            else:
                items[key]["status"] = status
                items[key]["decision_time"] = str(row.get("event_time", "") or row.get("date", ""))
                items[key]["event_id"] = max(items[key]["event_id"], int(row.get("event_id", 0)))
        else:
            standalone.append(
                {
                    "sender": agent,
                    "counterparty": counterparty,
                    "status": "pending",
                    "request_time": str(row.get("event_time", "") or row.get("date", "")),
                    "decision_time": "",
                    "detail": str(row.get("detail", "")),
                    "event_id": int(row.get("event_id", 0)),
                }
            )
    return sorted([*items.values(), *standalone], key=lambda item: item.get("event_id", 0))


def render_empty_phone(title, text):
    render_phone_component(title, f"<div class='empty'>{html.escape(text)}</div>", 0, False, height=420)


def render_phone_component(title, body_html, cursor_id, auto_scroll, height=620, input_text="Message disabled in replay mode"):
    should_scroll = "true" if auto_scroll else "false"
    body_id = dom_id("wechat_body", title, cursor_id, input_text)
    full = f"""
    {COMPONENT_CSS}
    <div class='wechat-phone'>
      <div class='wechat-header'>{html.escape(title)}</div>
      <div id='{body_id}' class='wechat-body'>{body_html}</div>
      <div class='wechat-input'>＋ <span>{html.escape(input_text)}</span></div>
    </div>
    <script>
      const body = document.getElementById('{body_id}');
      function scrollChatToBottom() {{
        if (!body || !{should_scroll}) return;
        body.scrollTop = body.scrollHeight;
      }}
      requestAnimationFrame(scrollChatToBottom);
      setTimeout(scrollChatToBottom, 80);
      setTimeout(scrollChatToBottom, 260);
    </script>
    """
    components.html(full, height=height, scrolling=False)


# ----------------------------- social/network -----------------------------


def render_social_view(
    edges,
    centrality_scores,
    friendships_df,
    groups_df,
    events_until,
    cursor_id,
    auto_scroll,
    live_api_mode=False,
):
    edges = normalize_social_edges(edges)
    influence_edges = edges[edges["kind"] == "influence"].copy() if not edges.empty else pd.DataFrame()
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown("#### 社交图谱")
        st.plotly_chart(network_figure(edges, centrality_scores), use_container_width=True, key="social_graph_network")
        st.caption("绿色实线是好友关系，灰色虚线是影响关系；好友关系来自 YAML 初始配置和通过的好友申请，影响关系来自消息传播。")
    with c2:
        st.markdown("#### 关系类型")
        unique_friendships = count_unique_friendships(friendships_df, edges)
        influence_count = int(len(influence_edges)) if not influence_edges.empty else 0
        m1, m2 = st.columns(2)
        m1.metric("好友关系", unique_friendships)
        m2.metric("影响关系", influence_count)
        if live_api_mode and st.button("按 YAML 重新加载初始图谱", use_container_width=True):
            reset_live_social_graph_from_yaml()
            rerun_app()
        st.markdown("#### 好友申请实时流")
        req = events_until[events_until["event_type"].astype(str).str.startswith("friend")]
        render_event_feed(req.tail(180), cursor_id, height=360, auto_scroll=auto_scroll)
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("#### 好友关系")
        if friendships_df.empty:
            st.info("暂无好友关系。")
        else:
            st.dataframe(friendships_df, use_container_width=True, hide_index=True)
    with f2:
        st.markdown("#### 影响关系")
        if influence_edges.empty:
            st.info("暂无影响关系。")
        else:
            view = influence_edges[["source", "target", "weight"]].copy()
            view["weight"] = view["weight"].astype(float).round(2)
            st.dataframe(view, use_container_width=True, hide_index=True)
    with f3:
        st.markdown("#### 群 / 朋友圈可见范围")
        if groups_df.empty:
            st.info("暂无群组。")
        else:
            st.dataframe(groups_df, use_container_width=True, hide_index=True)


def network_figure(edges: pd.DataFrame, centrality_scores: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    edges = normalize_social_edges(edges)
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
    for kind in ["influence", "friendship"]:
        subset = edges[edges["kind"] == kind]
        edge_x, edge_y = [], []
        for _, row in subset.iterrows():
            source, target = str(row["source"]), str(row["target"])
            if source not in positions or target not in positions:
                continue
            x0, y0 = positions[source]
            x1, y1 = positions[target]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        if not edge_x:
            continue
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(
                    width=3 if kind == "friendship" else 1.4,
                    color=RELATIONSHIP_COLORS.get(kind, "#64748b"),
                    dash="solid" if kind == "friendship" else "dot",
                ),
                hoverinfo="none",
                name=RELATIONSHIP_LABELS.get(kind, kind),
            )
        )
    relationship_stats = node_relationship_stats(edges)
    sizes = [22 + 150 * float(center.get(node, 0.03)) for node in nodes]
    fig.add_trace(
        go.Scatter(
            x=[positions[node][0] for node in nodes],
            y=[positions[node][1] for node in nodes],
            mode="markers+text",
            text=nodes,
            textposition="top center",
            marker=dict(size=sizes, showscale=False),
            hovertext=[
                (
                    f"{node}<br>"
                    f"好友数={relationship_stats.get(node, {}).get('friends', 0)}<br>"
                    f"影响发出={relationship_stats.get(node, {}).get('influence_out', 0)}<br>"
                    f"影响收到={relationship_stats.get(node, {}).get('influence_in', 0)}<br>"
                    f"pagerank={center.get(node, 0):.4f}"
                )
                for node in nodes
            ],
            hoverinfo="text",
            name="Agent",
        )
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        height=520,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=36, b=10),
    )
    return fig


def normalize_social_edges(edges: pd.DataFrame) -> pd.DataFrame:
    columns = ["source", "target", "weight", "kind"]
    if edges is None or edges.empty:
        return pd.DataFrame(columns=columns)
    work = edges.copy()
    for column in columns:
        if column not in work.columns:
            work[column] = "influence" if column == "kind" else (1.0 if column == "weight" else "")
    work["kind"] = work["kind"].fillna("influence").astype(str).str.strip().str.lower()
    work.loc[~work["kind"].isin(RELATIONSHIP_LABELS), "kind"] = "influence"
    work["source"] = work["source"].fillna("").astype(str)
    work["target"] = work["target"].fillna("").astype(str)
    work["weight"] = pd.to_numeric(work["weight"], errors="coerce").fillna(1.0)
    work = work[(work["source"] != "") & (work["target"] != "")]
    return work[columns]


def count_unique_friendships(friendships_df: pd.DataFrame, edges: pd.DataFrame) -> int:
    pairs = set()
    if friendships_df is not None and not friendships_df.empty and {"agent_a", "agent_b"} <= set(friendships_df.columns):
        for _, row in friendships_df.iterrows():
            a, b = str(row.get("agent_a", "")), str(row.get("agent_b", ""))
            if a and b and a != b:
                pairs.add(tuple(sorted([a, b])))
    for _, row in normalize_social_edges(edges).iterrows():
        if row.get("kind") != "friendship":
            continue
        a, b = str(row.get("source", "")), str(row.get("target", ""))
        if a and b and a != b:
            pairs.add(tuple(sorted([a, b])))
    return len(pairs)


def node_relationship_stats(edges: pd.DataFrame) -> dict[str, dict[str, int]]:
    stats = {
        agent: {"friends": 0, "influence_out": 0, "influence_in": 0}
        for agent in sorted(set(edges["source"]) | set(edges["target"]))
    }
    friend_sets = {agent: set() for agent in stats}
    for _, row in edges.iterrows():
        source, target, kind = str(row["source"]), str(row["target"]), str(row["kind"])
        if source not in stats:
            stats[source] = {"friends": 0, "influence_out": 0, "influence_in": 0}
            friend_sets[source] = set()
        if target not in stats:
            stats[target] = {"friends": 0, "influence_out": 0, "influence_in": 0}
            friend_sets[target] = set()
        if kind == "friendship":
            friend_sets[source].add(target)
            friend_sets[target].add(source)
        else:
            stats[source]["influence_out"] += 1
            stats[target]["influence_in"] += 1
    for agent, friends in friend_sets.items():
        stats[agent]["friends"] = len(friends)
    return stats


# ----------------------------- generic renderers -----------------------------


def render_event_feed(events_df, cursor_id, height=500, auto_scroll=True):
    feed_id = dom_id("feed", cursor_id, len(events_df))
    if events_df.empty:
        body = f"<div id='{feed_id}' class='feed-panel'><div class='feed-row'>当前视图没有事件。</div></div>"
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
        body = f"<div id='{feed_id}' class='feed-panel'>{''.join(rows)}</div>"
    should_scroll = "true" if auto_scroll else "false"
    full = f"""
    {COMPONENT_CSS}
    {body}
    <script>
      const feed = document.getElementById('{feed_id}');
      function scrollFeedToBottom() {{
        if (!feed || !{should_scroll}) return;
        feed.scrollTop = feed.scrollHeight;
      }}
      requestAnimationFrame(scrollFeedToBottom);
      setTimeout(scrollFeedToBottom, 80);
      setTimeout(scrollFeedToBottom, 260);
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


def render_tables(
    event_log,
    state_history,
    messages,
    trades,
    social_events,
    metrics,
    strategy_history,
    manager_equity=None,
    meta_weights=None,
    agent_corr=None,
    experiment_comparison=None,
    drift_log=None,
    manager_loss=None,
    training_params=None,
    forecast_scores=None,
    agent_views=None,
    bl_weights=None,
):
    st.markdown("#### 研究数据表")
    table_name = st.selectbox(
        "选择表",
        [
            "unified_event_log",
            "agent_state_history",
            "strategy_choice_history",
            "message_log",
            "trade_log",
            "social_events",
            "performance_metrics",
            "manager_equity_curve",
            "meta_weight_history",
            "agent_return_correlation",
            "experiment_comparison",
            "drift_log",
            "manager_loss_history",
            "training_params",
            "forecast_scores",
            "agent_views",
            "bl_agent_view_weights",
        ],
    )
    table_map = {
        "unified_event_log": event_log,
        "agent_state_history": state_history,
        "strategy_choice_history": strategy_history if strategy_history is not None else pd.DataFrame(),
        "message_log": messages,
        "trade_log": trades,
        "social_events": social_events,
        "performance_metrics": metrics,
        "manager_equity_curve": manager_equity if manager_equity is not None else pd.DataFrame(),
        "meta_weight_history": meta_weights if meta_weights is not None else pd.DataFrame(),
        "agent_return_correlation": agent_corr if agent_corr is not None else pd.DataFrame(),
        "experiment_comparison": experiment_comparison if experiment_comparison is not None else pd.DataFrame(),
        "drift_log": drift_log if drift_log is not None else pd.DataFrame(),
        "manager_loss_history": manager_loss if manager_loss is not None else pd.DataFrame(),
        "training_params": training_params if training_params is not None else pd.DataFrame(),
        "forecast_scores": forecast_scores if forecast_scores is not None else pd.DataFrame(),
        "agent_views": agent_views if agent_views is not None else pd.DataFrame(),
        "bl_agent_view_weights": bl_weights if bl_weights is not None else pd.DataFrame(),
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
    work["_seq"] = range(len(work))
    work["date_ts"] = pd.to_datetime(work["date"], errors="coerce")
    cutoff = pd.Timestamp(date)
    work = work[work["date_ts"] <= cutoff]
    if work.empty:
        return pd.DataFrame()
    return (
        work.sort_values(["date_ts", "agent", "_seq"])
        .groupby("agent", as_index=False)
        .tail(1)
        .drop(columns=["date_ts", "_seq"])
    )


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


def dom_id(prefix: str, *parts) -> str:
    text = "_".join(str(part) for part in parts)
    return f"{prefix}_{abs(hash(text))}"


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, default)))
    except (TypeError, ValueError):
        return default


def truthy_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def stable_index(text: str, modulo: int) -> int:
    if modulo <= 0:
        return 0
    return sum((index + 1) * ord(char) for index, char in enumerate(str(text))) % modulo


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


if __name__ == "__main__":
    main()

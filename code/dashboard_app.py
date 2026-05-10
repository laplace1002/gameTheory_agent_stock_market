from pathlib import Path
import ast
import html
import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PIXEL_CSS = """
<style>
.pixel-floor {
    background: #151923;
    border: 4px solid #2f3545;
    box-shadow: 0 0 0 4px #0b0d12 inset;
    padding: 16px;
    image-rendering: pixelated;
}
.pixel-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 14px;
}
.agent-card {
    min-height: 250px;
    border: 3px solid #10131a;
    box-shadow: 6px 6px 0 #090b10;
    padding: 12px;
    color: #f4f0d8;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    position: relative;
    overflow: hidden;
}
.agent-card:before {
    content: "";
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    height: 38px;
    background: repeating-linear-gradient(90deg, rgba(0,0,0,.28) 0 10px, rgba(255,255,255,.06) 10px 20px);
}
.scene-momentum { background: linear-gradient(180deg, #163b5c 0%, #225e78 54%, #2e2d42 55%); }
.scene-reversion { background: linear-gradient(180deg, #49305f 0%, #745083 54%, #303039 55%); }
.scene-risk { background: linear-gradient(180deg, #1e5a52 0%, #2c7466 54%, #243039 55%); }
.scene-value { background: linear-gradient(180deg, #654b2a 0%, #8a6a39 54%, #2e2b27 55%); }
.scene-random { background: linear-gradient(180deg, #5b3440 0%, #9b4d58 54%, #2c2933 55%); }
.scene-team { background: linear-gradient(180deg, #334264 0%, #5c6584 54%, #252b3a 55%); }
.scene-social { background: linear-gradient(180deg, #3b6251 0%, #7b7d4a 54%, #262b2a 55%); }
.pixel-avatar {
    width: 58px;
    height: 58px;
    background: #f0c887;
    border: 4px solid #151923;
    box-shadow: 4px 4px 0 rgba(0,0,0,.45);
    position: relative;
    margin-bottom: 8px;
}
.pixel-avatar:before {
    content: "";
    position: absolute;
    width: 8px;
    height: 8px;
    left: 12px;
    top: 20px;
    background: #151923;
    box-shadow: 24px 0 0 #151923, 12px 18px 0 #8e3f3f, 20px 18px 0 #8e3f3f;
}
.pixel-avatar:after {
    content: "";
    position: absolute;
    left: -4px;
    top: -12px;
    width: 66px;
    height: 16px;
    background: #24314a;
    border: 4px solid #151923;
}
.agent-name {
    font-size: 15px;
    font-weight: 800;
    text-shadow: 2px 2px 0 #111;
    margin-bottom: 6px;
    position: relative;
}
.pixel-chip {
    display: inline-block;
    padding: 2px 6px;
    margin: 2px 4px 2px 0;
    border: 2px solid rgba(0,0,0,.5);
    background: rgba(255,255,255,.13);
    color: #fff4c2;
    font-size: 11px;
}
.stat-row {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    padding: 3px 0;
    border-bottom: 1px solid rgba(255,255,255,.16);
    position: relative;
}
.stat-label { color: #d4d9e8; }
.stat-value { color: #fff; font-weight: 800; text-align: right; }
.pnl-pos { color: #9ff0a0; }
.pnl-neg { color: #ff9d9d; }
.activity-line {
    margin-top: 8px;
    padding: 8px;
    min-height: 42px;
    background: rgba(8,10,14,.58);
    border: 2px solid rgba(255,255,255,.18);
    position: relative;
    font-size: 12px;
}
.pixel-console {
    background: #10131a;
    color: #dff2ff;
    border: 3px solid #31384b;
    box-shadow: 5px 5px 0 #07080c;
    padding: 12px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}
.feed-row {
    border-bottom: 1px solid #2b3243;
    padding: 6px 0;
}

/* PixelFlow ticker tape */
.ticker-tape-wrap {
    overflow: hidden;
    background: #0a0c10;
    border: 3px solid #1e2535;
    padding: 6px 0;
    margin-bottom: 14px;
    box-shadow: 0 3px 0 #060709;
}
.ticker-tape {
    display: inline-block;
    white-space: nowrap;
    animation: scroll-left 28s linear infinite;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 13px;
    color: #ffe066;
}
@keyframes scroll-left {
    0% { transform: translateX(100vw); }
    100% { transform: translateX(-100%); }
}
.ticker-item { margin: 0 28px; }
.ticker-up { color: #7eff9a; }
.ticker-down { color: #ff7e7e; }
.ticker-flat { color: #ffe066; }

/* PixelFlow trading floor */
.trading-floor {
    background: #0d1117;
    background-image:
        linear-gradient(rgba(30,37,53,.9) 1px, transparent 1px),
        linear-gradient(90deg, rgba(30,37,53,.9) 1px, transparent 1px);
    background-size: 32px 32px;
    border: 4px solid #1e2535;
    box-shadow: 0 0 0 4px #060709 inset;
    padding: 18px;
    image-rendering: pixelated;
}
.trading-floor-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    margin-bottom: 14px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}
.floor-title {
    color: #fff0a8;
    font-size: 18px;
    font-weight: 900;
    text-shadow: 2px 2px 0 #060709;
}
.floor-subtitle {
    color: #90a2c8;
    font-size: 12px;
    text-align: right;
}
.trading-layout {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(280px, 360px);
    gap: 16px;
    align-items: start;
}
.agent-action {
    margin-top: 8px;
    padding: 8px;
    min-height: 44px;
    background: rgba(8,10,14,.66);
    border: 2px solid rgba(255,255,255,.18);
    position: relative;
    font-size: 12px;
    line-height: 1.45;
}
.agent-card .pixel-chip {
    max-width: 100%;
    overflow-wrap: anywhere;
}
.feed-panel-title {
    color: #fff0a8;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 13px;
    font-weight: 900;
    margin-bottom: 8px;
}
.feed-scroll {
    max-height: 640px;
    overflow-y: auto;
    background: #080a0e;
    border: 2px solid #1e2535;
    padding: 8px;
}
.feed-scroll::-webkit-scrollbar { width: 6px; }
.feed-scroll::-webkit-scrollbar-thumb { background: #2f3a50; }
.feed-entry {
    border-bottom: 1px solid #1a2030;
    padding: 7px 0;
    font-size: 12px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    color: #c8d0e0;
    animation: fade-in .4s ease;
}
@keyframes fade-in {
    from { opacity: 0; transform: translateY(4px); }
    to { opacity: 1; transform: translateY(0); }
}
.tag-buy { color: #7eff9a; font-weight: 800; }
.tag-sell { color: #ff7e7e; font-weight: 800; }
.tag-msg { color: #7ec8ff; font-weight: 800; }
.tag-social { color: #ffe066; font-weight: 800; }

/* Pixel phones */
.phone-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
    padding: 16px;
    margin-bottom: 12px;
    background: #0d1117;
    border: 3px solid #1e2535;
    box-shadow: 4px 4px 0 #060709;
}
.phone-icon {
    width: 78px;
    text-align: center;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 11px;
    color: #c8d0e0;
}
.phone-body {
    width: 64px;
    height: 100px;
    background: #1a2030;
    border: 3px solid #3a4560;
    box-shadow: 4px 4px 0 #060709;
    border-radius: 6px 6px 10px 10px;
    position: relative;
    margin: 0 auto 4px;
}
.phone-screen {
    position: absolute;
    top: 8px;
    left: 5px;
    right: 5px;
    bottom: 14px;
    background: #0a1520;
    border: 2px solid #2a3550;
    overflow: hidden;
}
.phone-screen-content {
    font-size: 7px;
    color: #7ec8ff;
    padding: 2px;
    line-height: 1.4;
    overflow-wrap: anywhere;
    text-align: left;
}
.phone-notch {
    position: absolute;
    top: 2px;
    left: 50%;
    transform: translateX(-50%);
    width: 18px;
    height: 4px;
    background: #060709;
    border-radius: 2px;
}
.phone-home {
    position: absolute;
    bottom: 3px;
    left: 50%;
    transform: translateX(-50%);
    width: 14px;
    height: 5px;
    background: #2a3550;
    border-radius: 3px;
}
.phone-badge {
    position: absolute;
    top: -4px;
    right: -4px;
    min-width: 14px;
    height: 14px;
    padding: 0 3px;
    background: #ff4444;
    border-radius: 50%;
    font-size: 9px;
    color: #fff;
    text-align: center;
    line-height: 14px;
    font-weight: 800;
}
.phone-label {
    width: 78px;
    overflow-wrap: anywhere;
    line-height: 1.2;
}

/* WeChat-style chat */
.wechat-window {
    background: #ededed;
    border: 3px solid #1e2535;
    box-shadow: 6px 6px 0 #060709;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    max-height: 520px;
    display: flex;
    flex-direction: column;
}
.wechat-header {
    background: #2c2c2c;
    color: #fff;
    padding: 10px 14px;
    font-size: 14px;
    font-weight: 600;
    border-bottom: 2px solid #1a1a1a;
    display: flex;
    align-items: center;
    gap: 8px;
}
.wechat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 12px;
    background: #ededed;
}
.wechat-messages::-webkit-scrollbar { width: 4px; }
.wechat-messages::-webkit-scrollbar-thumb { background: #bbb; }
.bubble-wrap {
    display: flex;
    margin-bottom: 10px;
    align-items: flex-start;
    gap: 8px;
}
.bubble-wrap.self { flex-direction: row-reverse; }
.bubble-avatar {
    width: 36px;
    height: 36px;
    background: #5b8dd9;
    border-radius: 4px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 10px;
    color: #fff;
    font-weight: 800;
}
.bubble {
    max-width: 68%;
    padding: 8px 12px;
    border-radius: 4px;
    font-size: 13px;
    line-height: 1.5;
    position: relative;
    overflow-wrap: anywhere;
}
.bubble.received { background: #fff; box-shadow: 1px 1px 3px rgba(0,0,0,.1); }
.bubble.sent { background: #95ec69; box-shadow: 1px 1px 3px rgba(0,0,0,.1); }
.bubble-meta { font-size: 10px; color: #777; margin-top: 3px; }
.bubble-tag {
    display: inline-block;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 10px;
    font-weight: 700;
    margin-right: 4px;
}
.tag-bullish { background: #d4f7d4; color: #2a7a2a; }
.tag-bearish { background: #ffd4d4; color: #7a2a2a; }
.tag-neutral { background: #e8e8e8; color: #555; }
.wechat-empty {
    padding: 18px;
    color: #666;
    font-size: 13px;
}

/* Moments */
.moments-post {
    background: #fff;
    border-bottom: 1px solid #e0e0e0;
    padding: 12px;
    display: flex;
    gap: 10px;
}
.moments-avatar {
    width: 40px;
    height: 40px;
    background: #5b8dd9;
    border-radius: 4px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    color: #fff;
    font-weight: 800;
}
.moments-content { flex: 1; }
.moments-name { font-weight: 700; color: #576b95; font-size: 13px; }
.moments-text { font-size: 13px; color: #333; margin: 4px 0; overflow-wrap: anywhere; }
.moments-meta { font-size: 11px; color: #999; }

@media (max-width: 980px) {
    .trading-layout { grid-template-columns: 1fr; }
    .feed-scroll { max-height: 300px; }
    .trading-floor-header { align-items: flex-start; flex-direction: column; }
    .floor-subtitle { text-align: left; }
}
</style>
"""


AGENT_SCENES = {
    "MomentumAgent": ("scene-momentum", "ticker roof"),
    "TruthfulReporterAgent": ("scene-momentum", "signal desk"),
    "MeanReversionAgent": ("scene-reversion", "chart archive"),
    "PersuaderAgent": ("scene-reversion", "broadcast room"),
    "LowVolatilityAgent": ("scene-risk", "risk lab"),
    "FreeRiderAgent": ("scene-risk", "quiet terminal"),
    "DrawdownBuyerAgent": ("scene-value", "value vault"),
    "ContrarianAgent": ("scene-value", "contrarian booth"),
    "RandomAgent": ("scene-random", "random station"),
    "CommitteeTeamAgent": ("scene-team", "committee table"),
    "DynamicTeamAgent": ("scene-team", "team cockpit"),
    "SocialGraphAgent": ("scene-social", "network hub"),
}


def parse_list_value(value) -> list:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return [text]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


def money(value) -> str:
    if pd.isna(value):
        return "$0"
    return f"${float(value):,.0f}"


def percent(value) -> str:
    if pd.isna(value):
        return "0.00%"
    return f"{float(value):+.2%}"


def signed_money(value) -> str:
    if pd.isna(value):
        return "$0"
    sign = "+" if float(value) >= 0 else "-"
    return f"{sign}${abs(float(value)):,.0f}"


def as_date_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_datetime(df[column], errors="coerce")


def available_dates(equity_df: pd.DataFrame) -> list[str]:
    dates = pd.to_datetime(equity_df["date"], errors="coerce").dropna().sort_values().unique()
    return [str(date)[:10] for date in dates]


def holdings_until(trades_df: pd.DataFrame, selected_date: str) -> dict[str, dict[str, int]]:
    if trades_df.empty:
        return {}
    trades_work = trades_df.copy()
    trades_work["date"] = as_date_series(trades_work, "date")
    cutoff = pd.Timestamp(selected_date)
    trades_work = trades_work[trades_work["date"] <= cutoff]
    if trades_work.empty:
        return {}
    trades_work["signed_shares"] = trades_work.apply(
        lambda row: int(row["shares"]) if row["side"] == "BUY" else -int(row["shares"]),
        axis=1,
    )
    grouped = trades_work.groupby(["agent", "ticker"])["signed_shares"].sum().reset_index()
    grouped = grouped[grouped["signed_shares"] != 0]
    holdings: dict[str, dict[str, int]] = {}
    for _, row in grouped.iterrows():
        holdings.setdefault(row["agent"], {})[row["ticker"]] = int(row["signed_shares"])
    return holdings


def latest_equity_until(equity_df: pd.DataFrame, selected_date: str) -> pd.DataFrame:
    equity_work = equity_df.copy()
    equity_work["date"] = as_date_series(equity_work, "date")
    cutoff = pd.Timestamp(selected_date)
    current = equity_work[equity_work["date"] <= cutoff].sort_values(["agent", "date"]).groupby("agent").tail(1)
    initial = equity_work.sort_values(["agent", "date"]).groupby("agent").head(1)[["agent", "equity"]]
    initial = initial.rename(columns={"equity": "initial_equity"})
    current = current.merge(initial, on="agent", how="left")
    current["pnl"] = current["equity"] - current["initial_equity"]
    current["pnl_pct"] = current["pnl"] / current["initial_equity"].replace(0, pd.NA)
    return current.sort_values("agent")


def build_activity_feed(trades_df, messages_df, social_events_df, selected_date: str, mode="day") -> pd.DataFrame:
    rows = []
    cutoff = pd.Timestamp(selected_date)

    if not trades_df.empty:
        trade_work = trades_df.copy()
        trade_work["date"] = as_date_series(trade_work, "date")
        trade_work = trade_work[trade_work["date"].eq(cutoff) if mode == "day" else trade_work["date"].le(cutoff)]
        for _, row in trade_work.iterrows():
            rows.append(
                {
                    "date": str(row["date"])[:10],
                    "agent": row["agent"],
                    "type": "trade",
                    "channel": "",
                    "detail": f"{row['side']} {int(row['shares'])} {row['ticker']} @ {float(row['price']):.2f}",
                }
            )

    if not messages_df.empty:
        msg_work = messages_df.copy()
        msg_work["timestamp"] = as_date_series(msg_work, "timestamp")
        msg_work = msg_work[msg_work["timestamp"].eq(cutoff) if mode == "day" else msg_work["timestamp"].le(cutoff)]
        for _, row in msg_work.iterrows():
            tickers = ", ".join(parse_list_value(row.get("tickers", "")))
            receivers = ", ".join(parse_list_value(row.get("receiver_ids", ""))) or "broadcast"
            rows.append(
                {
                    "date": str(row["timestamp"])[:10],
                    "agent": row["sender_id"],
                    "type": "message",
                    "channel": row["channel"],
                    "detail": f"{row['channel']} -> {receivers}: {row['direction']} {tickers} ({float(row['confidence']):.0%})",
                }
            )

    if not social_events_df.empty:
        social_work = social_events_df.copy()
        social_work["date"] = as_date_series(social_work, "date")
        social_work = social_work[social_work["date"].eq(cutoff) if mode == "day" else social_work["date"].le(cutoff)]
        for _, row in social_work.iterrows():
            rows.append(
                {
                    "date": str(row["date"])[:10],
                    "agent": row["sender"],
                    "type": row["event_type"],
                    "channel": "social",
                    "detail": row["detail"],
                }
            )

    feed = pd.DataFrame(rows, columns=["date", "agent", "type", "channel", "detail"])
    if feed.empty:
        return feed
    return feed.sort_values(["date", "agent", "type"]).tail(160)


def latest_activity_by_agent(feed: pd.DataFrame) -> dict[str, str]:
    if feed.empty:
        return {}
    latest = feed.groupby("agent").tail(1)
    return {row["agent"]: row["detail"] for _, row in latest.iterrows()}


def friend_map(friendships_df: pd.DataFrame) -> dict[str, list[str]]:
    friends: dict[str, list[str]] = {}
    if friendships_df.empty:
        return friends
    for _, row in friendships_df.iterrows():
        a = row["agent_a"]
        b = row["agent_b"]
        friends.setdefault(a, []).append(b)
        friends.setdefault(b, []).append(a)
    return {agent: sorted(values) for agent, values in friends.items()}


def group_map(groups_df: pd.DataFrame) -> dict[str, list[str]]:
    if groups_df.empty:
        return {}
    return {
        group: sorted(members["agent"].tolist())
        for group, members in groups_df.groupby("group")
    }


def holdings_text(holdings: dict[str, int]) -> str:
    if not holdings:
        return "cash only"
    top = sorted(holdings.items(), key=lambda item: abs(item[1]), reverse=True)[:4]
    return " / ".join(f"{ticker}:{shares}" for ticker, shares in top)


def safe_float(value, default: float = 0.0) -> float:
    if value is None or pd.isna(value):
        return default
    return float(value)


def short_agent_name(agent: str) -> str:
    clean = str(agent).replace("Agent", "")
    capitals = "".join(char for char in clean if char.isupper())
    if len(capitals) >= 2:
        return capitals[:3]
    return clean[:3].upper()


def truncate_text(value, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def trend_class(value: float) -> str:
    if value > 0:
        return "ticker-up"
    if value < 0:
        return "ticker-down"
    return "ticker-flat"


def render_ticker_tape(equity_df: pd.DataFrame, market_df: pd.DataFrame) -> str:
    items = []
    if not equity_df.empty:
        for _, row in equity_df.sort_values("agent").iterrows():
            pnl_pct = safe_float(row.get("pnl_pct"))
            items.append(
                f"<span class='ticker-item {trend_class(pnl_pct)}'>"
                f"{html.escape(str(row['agent']))} {percent(pnl_pct)}</span>"
            )

    if not market_df.empty:
        market_work = market_df.copy()
        market_work["date"] = as_date_series(market_work, "date")
        market_work = market_work.dropna(subset=["date"]).sort_values(["ticker", "date"])
        for ticker, rows in market_work.groupby("ticker"):
            recent = rows.tail(2)
            latest = safe_float(recent.iloc[-1]["close"])
            if len(recent) > 1:
                previous = safe_float(recent.iloc[0]["close"], latest)
                change = (latest - previous) / previous if previous else 0.0
            else:
                change = 0.0
            items.append(
                f"<span class='ticker-item {trend_class(change)}'>"
                f"{html.escape(str(ticker))} {latest:.2f} {percent(change)}</span>"
            )

    if not items:
        items.append("<span class='ticker-item ticker-flat'>MARKET WAITING</span>")
    return f"<div class='ticker-tape-wrap'><div class='ticker-tape'>{''.join(items)}</div></div>"


def feed_tag(feed_type: str, detail: str) -> tuple[str, str]:
    upper_detail = str(detail).upper()
    if upper_detail.startswith("BUY"):
        return "BUY", "tag-buy"
    if upper_detail.startswith("SELL"):
        return "SELL", "tag-sell"
    if feed_type == "message":
        return "MSG", "tag-msg"
    return str(feed_type or "SOCIAL").upper(), "tag-social"


def activity_markup(activity: str) -> str:
    tag, tag_class = feed_tag("", activity)
    detail = str(activity)
    if tag in {"BUY", "SELL"} and detail.upper().startswith(tag):
        detail = detail[len(tag) :].strip()
    return f"<span class='{tag_class}'>{html.escape(tag)}</span> {html.escape(detail)}"


def feed_entries_markup(feed: pd.DataFrame, limit: int = 50) -> str:
    if feed.empty:
        return "<div class='feed-entry'>No market activity for this view.</div>"
    entries = []
    for _, row in feed.tail(limit).iloc[::-1].iterrows():
        tag, tag_class = feed_tag(str(row.get("type", "")), str(row.get("detail", "")))
        channel = str(row.get("channel", ""))
        channel_text = f" / {html.escape(channel)}" if channel else ""
        entries.append(
            f"<div class='feed-entry'><span class='{tag_class}'>{html.escape(tag)}</span> "
            f"<b>{html.escape(str(row.get('date', '')))}</b>{channel_text}<br>"
            f"{html.escape(str(row.get('agent', '')))}: {html.escape(str(row.get('detail', '')))}</div>"
        )
    return "".join(entries)


def render_trading_floor(
    states: pd.DataFrame,
    holdings: dict,
    activities: dict,
    friends: dict,
    feed: pd.DataFrame | None = None,
    ticker_html: str = "",
) -> None:
    cards = []
    for _, row in states.iterrows():
        agent = row["agent"]
        scene_class, scene_name = AGENT_SCENES.get(agent, ("scene-team", "trading desk"))
        pnl_class = "pnl-pos" if float(row["pnl"]) >= 0 else "pnl-neg"
        friend_text = ", ".join(friends.get(agent, [])[:4]) or "no friends"
        activity = activities.get(agent, "waiting for next market tick")
        cards.append(
            f"<div class='agent-card {scene_class}'>"
            "<div class='pixel-avatar'></div>"
            f"<div class='agent-name'>{html.escape(agent)}</div>"
            f"<span class='pixel-chip'>{html.escape(scene_name)}</span>"
            f"<span class='pixel-chip'>{html.escape(friend_text)}</span>"
            f"<div class='stat-row'><span class='stat-label'>money</span><span class='stat-value'>{money(row['cash'])}</span></div>"
            f"<div class='stat-row'><span class='stat-label'>equity</span><span class='stat-value'>{money(row['equity'])}</span></div>"
            f"<div class='stat-row'><span class='stat-label'>P/L</span><span class='stat-value {pnl_class}'>{signed_money(row['pnl'])} / {percent(row['pnl_pct'])}</span></div>"
            f"<div class='stat-row'><span class='stat-label'>holdings</span><span class='stat-value'>{html.escape(holdings_text(holdings.get(agent, {})))}</span></div>"
            f"<div class='agent-action'>{activity_markup(activity)}</div>"
            "</div>"
        )
    feed_html = feed_entries_markup(feed if feed is not None else pd.DataFrame())
    floor_html = (
        f"{ticker_html}"
        "<div class='trading-floor'>"
        "<div class='trading-floor-header'>"
        "<div class='floor-title'>PixelFlow Trading Hall</div>"
        f"<div class='floor-subtitle'>{len(states)} agents online / live social-trading feed</div>"
        "</div>"
        "<div class='trading-layout'>"
        f"<div class='pixel-grid'>{''.join(cards)}</div>"
        "<div>"
        "<div class='feed-panel-title'>LIVE ACTIVITY</div>"
        f"<div class='feed-scroll'>{feed_html}</div>"
        "</div>"
        "</div>"
        "</div>"
    )
    st.markdown(floor_html, unsafe_allow_html=True)


def render_agent_cards(states: pd.DataFrame, holdings: dict, activities: dict, friends: dict) -> None:
    render_trading_floor(states, holdings, activities, friends)


def render_feed(feed: pd.DataFrame) -> None:
    if feed.empty:
        st.info("这个日期没有动态。")
        return
    rows = []
    for _, row in feed.tail(80).iterrows():
        rows.append(
            f"<div class='feed-row'><b>{html.escape(row['date'])}</b> "
            f"<span class='pixel-chip'>{html.escape(row['type'])}</span> "
            f"<span class='pixel-chip'>{html.escape(row['channel'])}</span> "
            f"{html.escape(row['agent'])}: {html.escape(row['detail'])}</div>"
        )
    st.markdown(f"<div class='pixel-console'>{''.join(rows)}</div>", unsafe_allow_html=True)


def render_message_console(message_rows: pd.DataFrame) -> None:
    if message_rows.empty:
        st.info("没有符合条件的消息。")
        return
    rows = []
    for _, row in message_rows.sort_values("timestamp", ascending=False).head(120).iterrows():
        tickers = ", ".join(parse_list_value(row.get("tickers", "")))
        receivers = ", ".join(parse_list_value(row.get("receiver_ids", ""))) or "broadcast"
        text = row.get("natural_language", "")
        rows.append(
            f"<div class='feed-row'><b>{html.escape(str(row['timestamp']))}</b> "
            f"<span class='pixel-chip'>{html.escape(str(row['channel']))}</span> "
            f"{html.escape(str(row['sender_id']))} -> {html.escape(receivers)} "
            f"<span class='pixel-chip'>{html.escape(tickers)}</span> "
            f"{html.escape(str(text))}</div>"
        )
    st.markdown(f"<div class='pixel-console'>{''.join(rows)}</div>", unsafe_allow_html=True)


def render_phone_grid(agents: list[str], message_counts: dict[str, int], latest_msgs: dict[str, str]) -> None:
    phones = []
    for agent in agents:
        count = int(message_counts.get(agent, 0))
        badge = f"<div class='phone-badge'>{min(count, 99)}</div>" if count else ""
        latest = truncate_text(latest_msgs.get(agent, "no messages"), 58)
        label = truncate_text(str(agent).replace("Agent", ""), 16)
        phones.append(
            "<div class='phone-icon'>"
            "<div class='phone-body'>"
            "<div class='phone-notch'></div>"
            f"{badge}"
            f"<div class='phone-screen'><div class='phone-screen-content'>{html.escape(latest)}</div></div>"
            "<div class='phone-home'></div>"
            "</div>"
            f"<div class='phone-label'>{html.escape(label)}</div>"
            "</div>"
        )
    st.markdown(f"<div class='phone-grid'>{''.join(phones)}</div>", unsafe_allow_html=True)


def message_direction(row) -> str:
    direction = row.get("direction", "neutral")
    if direction is None or pd.isna(direction):
        return "neutral"
    direction = str(direction).lower()
    if direction not in {"bullish", "bearish", "neutral"}:
        return "neutral"
    return direction


def message_body(row) -> str:
    tickers = ", ".join(parse_list_value(row.get("tickers", ""))) or "market"
    text = row.get("natural_language", "")
    confidence = safe_float(row.get("confidence"))
    direction = message_direction(row)
    return (
        f"<span class='bubble-tag tag-{direction}'>{html.escape(direction)}</span>"
        f"<span class='bubble-tag tag-neutral'>{confidence:.0%}</span>"
        f"<span class='bubble-tag tag-neutral'>{html.escape(tickers)}</span>"
        f"<div>{html.escape(str(text))}</div>"
    )


def render_wechat_chat(messages_df: pd.DataFrame, viewer_agent: str, channel: str) -> None:
    title_lookup = {
        "public": "Public Market Group",
        "private": "Private Messages",
        "moments": "Moments",
    }
    header = f"{title_lookup.get(channel, channel)} - {viewer_agent}"
    if messages_df.empty:
        st.markdown(
            "<div class='wechat-window'>"
            f"<div class='wechat-header'>{html.escape(header)}</div>"
            "<div class='wechat-empty'>No messages in this view.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    chat = messages_df.copy()
    chat["timestamp"] = as_date_series(chat, "timestamp")
    chat = chat.dropna(subset=["timestamp"]).sort_values("timestamp").tail(120)
    bubbles = []
    for _, row in chat.iterrows():
        sender = str(row.get("sender_id", "unknown"))
        is_self = sender == viewer_agent
        wrap_class = "bubble-wrap self" if is_self else "bubble-wrap"
        bubble_class = "sent" if is_self else "received"
        avatar = short_agent_name(sender)
        meta = f"{sender} / {str(row['timestamp'])[:10]}"
        bubbles.append(
            f"<div class='{wrap_class}'>"
            f"<div class='bubble-avatar'>{html.escape(avatar)}</div>"
            f"<div class='bubble {bubble_class}'>"
            f"{message_body(row)}"
            f"<div class='bubble-meta'>{html.escape(meta)}</div>"
            "</div>"
            "</div>"
        )

    st.markdown(
        "<div class='wechat-window'>"
        f"<div class='wechat-header'>{html.escape(header)}</div>"
        f"<div class='wechat-messages'>{''.join(bubbles)}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_moments(messages_df: pd.DataFrame, viewer_agent: str, friends: list[str]) -> None:
    visible_friends = set(friends)
    moments = messages_df.copy()
    if not moments.empty:
        moments["timestamp"] = as_date_series(moments, "timestamp")
        moments = moments[moments["sender_id"].isin(visible_friends)].sort_values("timestamp", ascending=False).head(80)

    header = f"Moments - {viewer_agent}"
    if moments.empty:
        friend_text = ", ".join(friends) if friends else "no friends"
        st.markdown(
            "<div class='wechat-window'>"
            f"<div class='wechat-header'>{html.escape(header)}</div>"
            f"<div class='wechat-empty'>No friend moments visible. Friends: {html.escape(friend_text)}</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    posts = []
    for _, row in moments.iterrows():
        sender = str(row.get("sender_id", "unknown"))
        text = row.get("natural_language", "")
        tickers = ", ".join(parse_list_value(row.get("tickers", ""))) or "market"
        direction = message_direction(row)
        meta = f"{str(row['timestamp'])[:10]} / {direction} / {tickers}"
        posts.append(
            "<div class='moments-post'>"
            f"<div class='moments-avatar'>{html.escape(short_agent_name(sender))}</div>"
            "<div class='moments-content'>"
            f"<div class='moments-name'>{html.escape(sender)}</div>"
            f"<div class='moments-text'>{html.escape(str(text))}</div>"
            f"<div class='moments-meta'>{html.escape(meta)}</div>"
            "</div>"
            "</div>"
        )

    st.markdown(
        "<div class='wechat-window'>"
        f"<div class='wechat-header'>{html.escape(header)}</div>"
        f"<div class='wechat-messages'>{''.join(posts)}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


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
friendships = read_table("friendships.csv")
groups = read_table("group_memberships.csv")
social_events = read_table("social_events.csv")

if metrics.empty or equity.empty:
    st.warning("还没有找到实验结果。请先运行：python code/run_experiment.py --experiment full_social")
    st.stop()

pixel_tab, overview_tab, market_tab, agents_tab, chat_tab, network_tab, aggregation_tab, trades_tab, evaluation_tab = st.tabs(
    ["PixelFlow 交易大厅", "Overview", "Market", "Agents", "Chat Lab", "Belief Network", "Aggregation", "Trades", "Evaluation"]
)

with pixel_tab:
    st.markdown(PIXEL_CSS, unsafe_allow_html=True)
    dates = available_dates(equity)
    c1, c2, c3, c4 = st.columns([1, 2, 1, 2])
    live_mode = c1.toggle("Live latest", value=True)
    if live_mode:
        selected_date = dates[-1]
        c2.metric("Current tick", selected_date)
    else:
        selected_date = c2.select_slider("Replay date", options=dates, value=dates[-1])
    feed_mode = c3.radio("Feed", ["day", "history"], index=0, horizontal=True)

    states = latest_equity_until(equity, selected_date)
    holdings = holdings_until(trades, selected_date)
    feed_all = build_activity_feed(trades, messages, social_events, selected_date, mode="history")
    activities = latest_activity_by_agent(feed_all)
    agent_options = ["All"] + sorted(states["agent"].unique().tolist())
    selected_agent = c4.selectbox("Floor focus", agent_options)
    feed = build_activity_feed(trades, messages, social_events, selected_date, mode=feed_mode)
    if selected_agent != "All" and not feed.empty:
        feed = feed[feed["agent"] == selected_agent]
    market_until = market.copy()
    if not market_until.empty:
        market_until["date"] = as_date_series(market_until, "date")
        market_until = market_until[market_until["date"] <= pd.Timestamp(selected_date)]
    ticker_html = render_ticker_tape(states, market_until)
    render_trading_floor(states, holdings, activities, friend_map(friendships), feed, ticker_html)

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
    st.markdown(PIXEL_CSS, unsafe_allow_html=True)
    if messages.empty:
        st.info("当前实验没有消息记录。")
    else:
        senders = set(messages["sender_id"].dropna().astype(str))
        receivers = set()
        for receiver_value in messages["receiver_ids"].dropna():
            receivers.update(parse_list_value(receiver_value))
        comm_agents = sorted((senders | receivers) or set(equity["agent"].dropna().astype(str)))

        latest_messages = messages.copy()
        latest_messages["timestamp"] = as_date_series(latest_messages, "timestamp")
        latest_messages = latest_messages.dropna(subset=["timestamp"]).sort_values("timestamp").groupby("sender_id").tail(1)
        latest_msgs = {
            str(row["sender_id"]): str(row.get("natural_language", ""))
            for _, row in latest_messages.iterrows()
        }
        msg_counts = messages.groupby("sender_id").size().astype(int).to_dict()
        render_phone_grid(comm_agents, msg_counts, latest_msgs)

        selected_phone = st.selectbox("选择 Agent 查看聊天", comm_agents, key="phone_select_agent")
        public_chat, private_chat, moments_chat = st.tabs(["群聊", "私信", "朋友圈"])

        with public_chat:
            public = messages[messages["channel"] == "public"]
            render_wechat_chat(public, selected_phone, "public")

        with private_chat:
            private = messages[messages["channel"] == "private"].copy()
            if not private.empty:
                private["receivers_parsed"] = private["receiver_ids"].apply(parse_list_value)
                private = private[
                    (private["sender_id"] == selected_phone)
                    | private["receivers_parsed"].apply(lambda values: selected_phone in values)
                ]
            render_wechat_chat(private, selected_phone, "private")

        with moments_chat:
            friends_of = friend_map(friendships).get(selected_phone, [])
            render_moments(messages[messages["channel"] == "moments"], selected_phone, friends_of)

        st.plotly_chart(px.histogram(messages, x="sender_id", color="channel", barmode="group"), width="stretch", key="chat_message_histogram")

with network_tab:
    c1, c2 = st.columns([2, 1])
    c1.plotly_chart(network_figure(social_edges, centrality), width="stretch", key="network_social_graph")
    if not social_edges.empty:
        c2.dataframe(social_edges.sort_values("weight", ascending=False), width="stretch", hide_index=True)
    f1, f2 = st.columns(2)
    if not friendships.empty:
        f1.dataframe(friendships, width="stretch", hide_index=True)
    if not groups.empty:
        f2.dataframe(groups, width="stretch", hide_index=True)
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

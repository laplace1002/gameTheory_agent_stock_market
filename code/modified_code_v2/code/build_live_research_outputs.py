from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-agent-runs")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/agent-runs-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agent_portfolio import build_agent_portfolio_outputs
from portfolio import performance_metrics
from scoring import brier_score, build_forecast_scores, log_score, parse_list
from view_extractor import build_black_litterman_outputs


LIVE_TICKER_PRICES = {
    "AAPL": 190.0,
    "MSFT": 430.0,
    "NVDA": 920.0,
    "TSLA": 175.0,
    "SPY": 520.0,
    "QQQ": 445.0,
    "XLE": 95.0,
    "UUP": 29.0,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build report-ready research outputs from live LLM agent runs.")
    parser.add_argument("--runs-root", required=True, help="Directory that contains one folder per social graph scenario.")
    parser.add_argument("--output-root", required=True, help="Directory where classified research outputs will be written.")
    parser.add_argument("--horizon-events", type=int, default=60)
    args = parser.parse_args()

    runs_root = Path(args.runs_root).expanduser()
    output_root = Path(args.output_root).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    per_scenario_dir = output_root / "01_per_scenario"
    comparison_dir = output_root / "02_cross_scenario_comparison"
    figure_dir = output_root / "03_figures"
    report_dir = output_root / "04_report_tables"
    for directory in [per_scenario_dir, comparison_dir, figure_dir, report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    scenario_summaries = []
    manager_frames = []
    agent_frames = []
    generated_paths = []
    for run_dir in discover_run_dirs(runs_root):
        scenario = clean_scenario_name(run_dir.name)
        print(f"Processing {scenario}: {run_dir}")
        result = process_live_run(run_dir, scenario, per_scenario_dir / scenario, args.horizon_events)
        scenario_summaries.append(result["social_summary"])
        manager_frames.append(result["manager_comparison"])
        agent_frames.append(result["agent_comparison"])
        generated_paths.extend(result["paths"])

    scenario_summary = pd.DataFrame(scenario_summaries)
    manager_comparison = concat_frames(manager_frames)
    agent_comparison = concat_frames(agent_frames)
    best_managers = best_manager_by_scenario(manager_comparison)

    scenario_summary.to_csv(comparison_dir / "live_scenario_social_summary.csv", index=False)
    manager_comparison.to_csv(comparison_dir / "manager_scenario_comparison.csv", index=False)
    agent_comparison.to_csv(comparison_dir / "agent_scenario_comparison.csv", index=False)
    best_managers.to_csv(comparison_dir / "best_manager_by_scenario.csv", index=False)

    write_report_markdown(report_dir, scenario_summary, manager_comparison, best_managers)
    write_figures(figure_dir, scenario_summary, manager_comparison)
    write_manifest(output_root, runs_root, generated_paths)
    print(f"Done. Research outputs written to: {output_root}")


def discover_run_dirs(runs_root: Path) -> list[Path]:
    dirs = []
    for child in runs_root.iterdir():
        if not child.is_dir():
            continue
        if (child / "tables" / "unified_event_log.csv").exists():
            dirs.append(child)
    return sorted(dirs, key=lambda path: clean_scenario_name(path.name))


def process_live_run(run_dir: Path, scenario: str, output_dir: Path, horizon_events: int) -> dict:
    tables = run_dir / "tables"
    output_tables = output_dir / "tables"
    output_tables.mkdir(parents=True, exist_ok=True)

    events = read_csv(tables / "unified_event_log.csv")
    states = read_csv(tables / "agent_state_history.csv")
    messages = read_csv(tables / "message_log.csv")
    trades = read_csv(tables / "trade_log.csv")
    social_events = read_csv(tables / "social_events.csv")
    social_edges = read_csv(tables / "social_graph_edges.csv")
    friendships = read_csv(tables / "friendships.csv")
    market_history = read_csv(tables / "market_history.csv")
    tradable_tickers = sorted(market_history["ticker"].astype(str).unique()) if not market_history.empty and "ticker" in market_history.columns else sorted(LIVE_TICKER_PRICES)

    equity = live_equity_curve_from_states(states)
    if equity.empty:
        raise RuntimeError(f"{scenario} has no live equity history")
    metrics = performance_metrics(equity).sort_values("sharpe", ascending=False)
    portfolio_outputs = build_agent_portfolio_outputs(equity=equity, initial_cash=100000.0, rebalance_every=1)

    enriched_messages = enrich_live_messages(messages, events, tradable_tickers)
    live_prices = market_history if not market_history.empty else synthetic_live_prices(events)
    forecast_scores = (
        build_forecast_scores(enriched_messages, live_prices, horizon_days=20)
        if not market_history.empty
        else live_forecast_scores(enriched_messages, events, horizon_events=horizon_events)
    )
    reputation_scores = live_reputation_scores(forecast_scores, enriched_messages)
    bl_outputs = build_black_litterman_outputs(enriched_messages, live_prices, reputation_scores, horizon_days=20)
    social_summary = live_social_summary(scenario, events, states, trades, social_events, social_edges, friendships)

    write_table(output_tables, "equity_curve.csv", equity)
    write_table(output_tables, "performance_metrics.csv", metrics)
    write_table(output_tables, "agent_return_history.csv", portfolio_outputs["agent_return_history"])
    write_table(output_tables, "agent_return_correlation.csv", portfolio_outputs["agent_return_correlation"])
    write_table(output_tables, "manager_equity_curve.csv", portfolio_outputs["manager_equity_curve"])
    write_table(output_tables, "meta_weight_history.csv", portfolio_outputs["meta_weight_history"])
    write_table(output_tables, "manager_loss_history.csv", portfolio_outputs["manager_loss_history"])
    write_table(output_tables, "experiment_comparison.csv", portfolio_outputs["experiment_comparison"])
    write_table(output_tables, "message_log_enriched.csv", enriched_messages)
    write_table(output_tables, "forecast_scores.csv", forecast_scores)
    write_table(output_tables, "reputation_scores.csv", reputation_scores)
    write_table(output_tables, "agent_views.csv", bl_outputs["agent_views"])
    write_table(output_tables, "bl_agent_view_weights.csv", bl_outputs["bl_agent_view_weights"])
    write_table(output_tables, "social_summary.csv", pd.DataFrame([social_summary]))
    if not market_history.empty:
        write_table(output_tables, "market_history.csv", market_history)

    copy_if_exists(run_dir / "metadata.json", output_dir / "metadata.json")
    copy_live_source_tables(tables, output_tables)

    comparison = portfolio_outputs["experiment_comparison"].copy()
    comparison.insert(0, "scenario", scenario)
    manager_comparison = comparison[comparison["experiment_type"] == "agent_portfolio"].copy()
    agent_comparison = comparison[comparison["experiment_type"] == "single_agent"].copy()
    return {
        "social_summary": social_summary,
        "manager_comparison": manager_comparison,
        "agent_comparison": agent_comparison,
        "paths": [str(output_tables / name) for name in output_file_names()],
    }


def live_equity_curve_from_states(states: pd.DataFrame) -> pd.DataFrame:
    columns = ["date", "agent", "equity", "cash"]
    if states.empty:
        return pd.DataFrame(columns=columns)
    work = states.copy()
    work["_row_order"] = np.arange(len(work))
    work["agent"] = work["agent"].astype(str)
    work["equity"] = pd.to_numeric(work["equity"], errors="coerce")
    work["cash"] = pd.to_numeric(work["cash"], errors="coerce").fillna(0.0)
    work = work.dropna(subset=["agent", "equity"]).sort_values(["agent", "_row_order"])
    work["step"] = work.groupby("agent").cumcount()
    parsed_dates = pd.to_datetime(work.get("date", ""), errors="coerce")
    if parsed_dates.notna().sum() and parsed_dates.dt.strftime("%Y-%m-%d").nunique() > 1:
        work["date"] = parsed_dates.dt.strftime("%Y-%m-%d")
    else:
        base_date = pd.Timestamp("2026-01-01")
        work["date"] = work["step"].map(lambda step: (base_date + pd.Timedelta(days=int(step))).strftime("%Y-%m-%d"))
    return work[columns].sort_values(["date", "agent"]).reset_index(drop=True)


def enrich_live_messages(messages: pd.DataFrame, events: pd.DataFrame, tradable_tickers: list[str] | None = None) -> pd.DataFrame:
    columns = [
        "message_id",
        "timestamp",
        "sender_id",
        "channel",
        "receiver_ids",
        "tickers",
        "horizon",
        "claim_type",
        "direction",
        "confidence",
        "evidence",
        "position_intent",
        "natural_language",
        "expires_at",
        "event_id",
    ]
    if messages.empty:
        return pd.DataFrame(columns=columns)
    work = messages.copy()
    tradable_tickers = tradable_tickers or sorted(LIVE_TICKER_PRICES)
    event_dates = {}
    if not events.empty and {"event_id", "date"} <= set(events.columns):
        event_dates = {
            int(event_id): str(date)
            for event_id, date in zip(pd.to_numeric(events["event_id"], errors="coerce").fillna(0).astype(int), events["date"])
        }
    if "event_id" not in work.columns:
        work["event_id"] = pd.to_numeric(work.get("message_id"), errors="coerce").fillna(0).astype(int)
    if "timestamp" not in work.columns:
        work["timestamp"] = ""
    work["timestamp"] = work.apply(
        lambda row: event_dates.get(int(pd.to_numeric(pd.Series([row.get("event_id", 0)]), errors="coerce").fillna(0).iloc[0]), row.get("timestamp", "")),
        axis=1,
    )
    work["message_id"] = work.get("message_id", work["event_id"])
    work["sender_id"] = work.get("sender_id", work.get("agent", ""))
    work["natural_language"] = work.get("natural_language", work.get("detail", "")).fillna("").astype(str)
    work["tickers"] = work.apply(lambda row: ensure_ticker_list(row.get("tickers", ""), row.get("natural_language", ""), tradable_tickers), axis=1)
    work["direction"] = work.apply(lambda row: infer_direction(row.get("natural_language", ""), row.get("direction", "")), axis=1)
    work["confidence"] = work.apply(lambda row: infer_confidence(row.get("natural_language", ""), row.get("confidence", "")), axis=1)
    work["horizon"] = "medium"
    work["claim_type"] = "llm_live_forecast"
    work["evidence"] = work.apply(lambda row: [{"type": "llm_live_message", "quality": float(row["confidence"])}], axis=1)
    work["position_intent"] = work["direction"].map({"bullish": "increase", "bearish": "decrease", "neutral": "hold"}).fillna("hold")
    work["expires_at"] = ""
    for column in columns:
        if column not in work.columns:
            work[column] = ""
    return work[columns]


def live_forecast_scores(messages: pd.DataFrame, events: pd.DataFrame, horizon_events: int) -> pd.DataFrame:
    rows = []
    if messages.empty:
        return pd.DataFrame(columns=forecast_score_columns())
    max_event = int(pd.to_numeric(events.get("event_id", pd.Series([0])), errors="coerce").max()) if not events.empty else 0
    for _, msg in messages.iterrows():
        event_id = int(pd.to_numeric(pd.Series([msg.get("event_id", 0)]), errors="coerce").fillna(0).iloc[0])
        future_id = min(max_event, event_id + horizon_events)
        direction = str(msg.get("direction", "neutral"))
        confidence = float(msg.get("confidence", 0.5) or 0.5)
        probability_up = direction_to_probability(direction, confidence)
        for ticker in parse_list(msg.get("tickers", [])):
            realized = live_reference_price(ticker, future_id) / live_reference_price(ticker, max(1, event_id)) - 1.0
            outcome = 1.0 if realized > 0 else 0.0
            brier = brier_score(probability_up, outcome)
            rows.append(
                {
                    "message_id": msg.get("message_id", event_id),
                    "event_id": event_id,
                    "sender_id": msg.get("sender_id", ""),
                    "ticker": ticker,
                    "direction": direction,
                    "probability_up": probability_up,
                    "confidence": confidence,
                    "horizon_events": horizon_events,
                    "realized_return": realized,
                    "outcome_up": outcome,
                    "brier_score": brier,
                    "log_score": log_score(probability_up, outcome),
                    "proper_score": 1.0 - brier,
                }
            )
    return pd.DataFrame(rows, columns=forecast_score_columns())


def live_reputation_scores(forecast_scores: pd.DataFrame, messages: pd.DataFrame) -> pd.DataFrame:
    columns = ["sender_id", "prediction_count", "avg_confidence", "reputation", "calibration_error", "influence_score", "proper_reputation"]
    if messages.empty:
        return pd.DataFrame(columns=columns)
    base = (
        messages.groupby("sender_id")
        .agg(prediction_count=("message_id", "count"), avg_confidence=("confidence", "mean"))
        .reset_index()
    )
    if forecast_scores.empty:
        base["reputation"] = 0.5
        base["calibration_error"] = 0.25
        base["influence_score"] = 0.0
        base["proper_reputation"] = 0.5
        return base[columns]
    scores = (
        forecast_scores.groupby("sender_id")
        .agg(calibration_error=("brier_score", "mean"), proper_reputation=("proper_score", "mean"))
        .reset_index()
    )
    out = base.merge(scores, on="sender_id", how="left")
    out["calibration_error"] = out["calibration_error"].fillna(0.25)
    out["proper_reputation"] = out["proper_reputation"].fillna(0.5)
    out["reputation"] = out["proper_reputation"]
    out["influence_score"] = 0.0
    return out[columns]


def synthetic_live_prices(events: pd.DataFrame) -> pd.DataFrame:
    max_event = int(pd.to_numeric(events.get("event_id", pd.Series([1])), errors="coerce").max()) if not events.empty else 1
    step = max(1, max_event // 120)
    event_ids = list(range(1, max_event + 1, step))
    rows = []
    base_date = pd.Timestamp("2026-01-01")
    for idx, event_id in enumerate(event_ids):
        date = base_date + pd.Timedelta(days=idx)
        for ticker in LIVE_TICKER_PRICES:
            price = live_reference_price(ticker, event_id)
            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "open": price,
                    "high": price * 1.002,
                    "low": price * 0.998,
                    "close": price,
                    "volume": 1_000_000,
                }
            )
    return pd.DataFrame(rows)


def live_social_summary(
    scenario: str,
    events: pd.DataFrame,
    states: pd.DataFrame,
    trades: pd.DataFrame,
    social_events: pd.DataFrame,
    social_edges: pd.DataFrame,
    friendships: pd.DataFrame,
) -> dict:
    event_type_counts = events["event_type"].value_counts().to_dict() if not events.empty else {}
    channel_counts = events[events["event_type"] == "message"]["channel"].value_counts().to_dict() if not events.empty else {}
    final_states = states.sort_index().groupby("agent", as_index=False).tail(1) if not states.empty else pd.DataFrame()
    avg_pnl = float(pd.to_numeric(final_states.get("pnl", pd.Series(dtype=float)), errors="coerce").mean()) if not final_states.empty else np.nan
    best_agent = ""
    best_pnl = np.nan
    if not final_states.empty:
        final_states["_pnl"] = pd.to_numeric(final_states.get("pnl", 0.0), errors="coerce")
        best = final_states.sort_values("_pnl", ascending=False).iloc[0]
        best_agent = str(best.get("agent", ""))
        best_pnl = float(best.get("_pnl", np.nan))
    return {
        "scenario": scenario,
        "rounds": states.get("round_id", pd.Series(dtype=str)).nunique() if not states.empty and "round_id" in states.columns else np.nan,
        "events": len(events),
        "messages": int(event_type_counts.get("message", 0)),
        "trades": int(event_type_counts.get("trade", 0)),
        "holds": int(event_type_counts.get("hold", 0)),
        "friend_requests": int(event_type_counts.get("friend_request", 0)),
        "friend_accepts": int(event_type_counts.get("friend_accept", 0)),
        "friend_rejects": int(event_type_counts.get("friend_reject", 0)),
        "public_messages": int(channel_counts.get("public", 0)),
        "private_messages": int(channel_counts.get("private", 0)),
        "moments": int(channel_counts.get("moments", 0)),
        "final_friendships": len(friendships),
        "final_influence_edges": int((social_edges.get("kind", pd.Series(dtype=str)) == "influence").sum()) if not social_edges.empty else 0,
        "avg_final_pnl": avg_pnl,
        "best_final_agent": best_agent,
        "best_final_pnl": best_pnl,
    }


def best_manager_by_scenario(manager_comparison: pd.DataFrame) -> pd.DataFrame:
    if manager_comparison.empty:
        return pd.DataFrame()
    return (
        manager_comparison.sort_values(["scenario", "sharpe"], ascending=[True, False])
        .groupby("scenario", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def write_figures(figure_dir: Path, scenario_summary: pd.DataFrame, manager_comparison: pd.DataFrame) -> None:
    if not manager_comparison.empty:
        best = best_manager_by_scenario(manager_comparison)
        plt.figure(figsize=(12, 5))
        plt.bar(best["scenario"], best["sharpe"])
        plt.xticks(rotation=40, ha="right")
        plt.ylabel("Best manager Sharpe")
        plt.title("Best Agent-Portfolio Manager by Social Graph")
        plt.tight_layout()
        plt.savefig(figure_dir / "best_manager_sharpe_by_scenario.png", dpi=180)
        plt.close()

        pivot = manager_comparison.pivot_table(index="scenario", columns="portfolio", values="total_return", aggfunc="mean")
        plt.figure(figsize=(12, 6))
        plt.imshow(pivot.fillna(0.0).to_numpy(), aspect="auto", cmap="RdYlGn")
        plt.colorbar(label="Total return")
        plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.title("Manager Total Return Heatmap")
        plt.tight_layout()
        plt.savefig(figure_dir / "manager_total_return_heatmap.png", dpi=180)
        plt.close()

    if not scenario_summary.empty:
        plt.figure(figsize=(12, 5))
        plt.bar(scenario_summary["scenario"], scenario_summary["final_friendships"])
        plt.xticks(rotation=40, ha="right")
        plt.ylabel("Final friendships")
        plt.title("Final Social Graph Density Proxy")
        plt.tight_layout()
        plt.savefig(figure_dir / "final_friendships_by_scenario.png", dpi=180)
        plt.close()


def write_report_markdown(report_dir: Path, scenario_summary: pd.DataFrame, manager_comparison: pd.DataFrame, best_managers: pd.DataFrame) -> None:
    lines = [
        "# Live LLM Agent Research Outputs",
        "",
        "## How To Read",
        "",
        "- `01_per_scenario/<scenario>/tables/`: Portfolio Hall-compatible tables generated from each live LLM run.",
        "- `02_cross_scenario_comparison/manager_scenario_comparison.csv`: cross-scenario manager results.",
        "- `02_cross_scenario_comparison/live_scenario_social_summary.csv`: social activity and graph statistics.",
        "- `03_figures/`: report-ready comparison figures.",
        "",
        "## Best Manager By Scenario",
        "",
    ]
    if not best_managers.empty:
        lines.append(markdown_table(best_managers[["scenario", "portfolio", "total_return", "sharpe", "max_drawdown"]]))
    lines.extend(["", "## Social Summary", ""])
    if not scenario_summary.empty:
        cols = ["scenario", "rounds", "messages", "trades", "friend_requests", "friend_accepts", "final_friendships", "avg_final_pnl"]
        lines.append(markdown_table(scenario_summary[[col for col in cols if col in scenario_summary.columns]]))
    (report_dir / "README_REPORT_OUTPUTS.md").write_text("\n".join(lines), encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    display = df.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.6g}")
        else:
            display[column] = display[column].fillna("").astype(str)
    headers = [str(column) for column in display.columns]
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in display.iterrows():
        rows.append("| " + " | ".join(escape_markdown_cell(row[column]) for column in display.columns) + " |")
    return "\n".join(rows)


def escape_markdown_cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_manifest(output_root: Path, runs_root: Path, generated_paths: list[str]) -> None:
    manifest = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "runs_root": str(runs_root),
        "output_root": str(output_root),
        "generated_file_count": len(generated_paths),
        "generated_paths": generated_paths[:200],
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def write_table(directory: Path, name: str, df: pd.DataFrame) -> None:
    df.to_csv(directory / name, index=False)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def copy_live_source_tables(src_dir: Path, dst_dir: Path) -> None:
    dashboard_tables = [
        "unified_event_log.csv",
        "agent_state_history.csv",
        "message_log.csv",
        "trade_log.csv",
        "social_events.csv",
        "social_graph_edges.csv",
        "friendships.csv",
        "group_memberships.csv",
    ]
    for name in dashboard_tables:
        copy_if_exists(src_dir / name, dst_dir / name)
    copy_if_exists(src_dir / "unified_event_log.csv", dst_dir / "raw_unified_event_log.csv")
    copy_if_exists(src_dir / "social_graph_edges.csv", dst_dir / "raw_social_graph_edges.csv")
    copy_if_exists(src_dir / "friendships.csv", dst_dir / "raw_friendships.csv")


def clean_scenario_name(name: str) -> str:
    return "_".join(str(name).strip().split())


def ensure_ticker_list(value, text: str, tradable_tickers: list[str] | None = None) -> list[str]:
    tradable_tickers = tradable_tickers or sorted(LIVE_TICKER_PRICES)
    tradable_set = set(tradable_tickers)
    tickers = [ticker for ticker in parse_list(value) if ticker in tradable_set]
    if tickers:
        return tickers
    upper = str(text or "").upper()
    return [ticker for ticker in tradable_tickers if ticker in upper] or [tradable_tickers[0]]


def infer_direction(text: str, existing="") -> str:
    existing = str(existing or "").strip()
    if existing in {"bullish", "bearish", "neutral"}:
        return existing
    work = str(text or "").lower()
    if any(term in work for term in ["buy", "bull", "加仓", "买入", "增持", "看多", "追涨", "上行"]):
        return "bullish"
    if any(term in work for term in ["sell", "bear", "减仓", "卖出", "看空", "下行", "回撤"]):
        return "bearish"
    return "neutral"


def infer_confidence(text: str, existing="") -> float:
    try:
        if not pd.isna(existing) and str(existing).strip() != "":
            return float(max(0.0, min(1.0, float(existing))))
    except (TypeError, ValueError):
        pass
    work = str(text or "")
    if any(term in work for term in ["确认", "明确", "强", "突破", "高置信"]):
        return 0.75
    if any(term in work for term in ["等待", "观望", "不确定", "验证"]):
        return 0.45
    return 0.62


def direction_to_probability(direction: str, confidence: float) -> float:
    if direction == "bullish":
        return 0.5 + 0.5 * confidence
    if direction == "bearish":
        return 0.5 - 0.5 * confidence
    return 0.5


def live_reference_price(ticker: str, event_id: int) -> float:
    base = LIVE_TICKER_PRICES.get(str(ticker), LIVE_TICKER_PRICES["SPY"])
    drift = ((stable_index(f"{ticker}-{event_id}", 17) - 8) / 1000.0) + ((event_id % 7) - 3) / 2000.0
    return round(max(1.0, base * (1 + drift)), 2)


def stable_index(text: str, modulo: int) -> int:
    if modulo <= 0:
        return 0
    return sum((index + 1) * ord(char) for index, char in enumerate(str(text))) % modulo


def forecast_score_columns() -> list[str]:
    return [
        "message_id",
        "event_id",
        "sender_id",
        "ticker",
        "direction",
        "probability_up",
        "confidence",
        "horizon_events",
        "realized_return",
        "outcome_up",
        "brier_score",
        "log_score",
        "proper_score",
    ]


def output_file_names() -> list[str]:
    return [
        "equity_curve.csv",
        "performance_metrics.csv",
        "agent_return_history.csv",
        "agent_return_correlation.csv",
        "manager_equity_curve.csv",
        "meta_weight_history.csv",
        "manager_loss_history.csv",
        "experiment_comparison.csv",
        "message_log_enriched.csv",
        "forecast_scores.csv",
        "reputation_scores.csv",
        "agent_views.csv",
        "bl_agent_view_weights.csv",
        "social_summary.csv",
    ]


if __name__ == "__main__":
    main()

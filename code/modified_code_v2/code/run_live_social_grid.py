from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import socket
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from data_loader import load_prices


DEFAULT_STARTING_CASH = 100_000.0
DEFAULT_OUT = Path("outputs/live_state/runs")
RUN_RECORD_FILE = "RUN_RECORDS.md"

AGENT_PROFILES = {
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all live LLM social graph scenarios without Streamlit.")
    parser.add_argument("--prices", default="data/TRD_Dalyr.xlsx")
    parser.add_argument("--config", default="config/social_scenarios.yaml")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--scenario", default="all", help="all or comma-separated scenario names")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--initial-cash", type=float, default=DEFAULT_STARTING_CASH)
    parser.add_argument("--fee-rate", type=float, default=0.0)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--friend-decision", choices=["llm", "heuristic"], default="llm")
    parser.add_argument("--sleep-between-rounds", type=float, default=0.0)
    parser.add_argument("--run-prefix", default="run")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--agent-retries", type=int, default=None, help="Retries for one failed agent inside a round.")
    parser.add_argument("--repair-retries", type=int, default=None, help="Retries to repair truncated non-JSON agent output.")
    parser.add_argument(
        "--round-retries",
        type=int,
        default=None,
        help="Retries for a failed round inside the same run. 0 means unlimited.",
    )
    parser.add_argument("--retry-backoff", type=float, default=None, help="Seconds to wait before retrying one failed call.")
    parser.add_argument("--resume", action="store_true", help="Skip scenarios already marked completed in RUN_RECORDS.md.")
    parser.add_argument("--start-at", default="", help="Start from this scenario in the selected order.")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Keep a run even if an agent call fails. By default failed runs are deleted and retried.",
    )
    args = parser.parse_args()

    load_dotenv_files()
    ensure_llm_config()

    prices = load_prices(args.prices)
    scenarios = load_scenarios(args.config)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    selected = select_scenarios(scenarios, args.scenario)
    if args.start_at:
        selected = start_at_scenario(selected, args.start_at)
    skipped = []
    if args.resume:
        completed = completed_scenarios_from_records(out_root, min_rounds=args.rounds)
        skipped = [name for name in selected if name in completed]
        selected = [name for name in selected if name not in completed]

    print(f"Loaded prices: {args.prices} | rows={len(prices)} | tickers={prices['ticker'].nunique()}")
    if skipped:
        print(f"Resume skip completed scenarios: {', '.join(skipped)}")
    print(f"Scenarios: {', '.join(selected)}")
    if not selected:
        print("No scenarios left to run.")
        return
    print(f"Rounds per scenario: {args.rounds}")
    for scenario_name in selected:
        run_scenario_with_retries(
            scenario_name=scenario_name,
            scenario=scenarios[scenario_name],
            prices=prices,
            out_root=out_root,
            rounds=args.rounds,
            initial_cash=args.initial_cash,
            fee_rate=args.fee_rate,
            max_workers=args.max_workers or env_int("LLM_MAX_WORKERS", 3),
            friend_decision=args.friend_decision,
            run_prefix=args.run_prefix,
            max_retries=args.max_retries,
            agent_retries=args.agent_retries or env_int("LLM_AGENT_RETRIES", 3),
            repair_retries=args.repair_retries if args.repair_retries is not None else env_int("LLM_REPAIR_RETRIES", 2),
            round_retries=args.round_retries if args.round_retries is not None else env_int("LLM_ROUND_RETRIES", 0),
            retry_backoff=args.retry_backoff if args.retry_backoff is not None else env_float("LLM_RETRY_BACKOFF", 3.0),
            strict=not args.allow_partial,
            sleep_between_rounds=args.sleep_between_rounds,
            resume_existing=args.resume,
        )


class InvalidRunError(RuntimeError):
    pass


def run_scenario_with_retries(
    scenario_name: str,
    scenario: dict,
    prices: pd.DataFrame,
    out_root: Path,
    rounds: int,
    initial_cash: float,
    fee_rate: float,
    max_workers: int,
    friend_decision: str,
    run_prefix: str,
    max_retries: int,
    agent_retries: int,
    repair_retries: int,
    round_retries: int,
    retry_backoff: float,
    strict: bool,
    sleep_between_rounds: float,
    resume_existing: bool,
) -> None:
    attempts = max(1, int(max_retries))
    last_error = ""
    for attempt in range(1, attempts + 1):
        runner = LiveScenarioRunner(
            scenario_name=scenario_name,
            scenario=scenario,
            prices=prices,
            out_root=out_root,
            rounds=rounds,
            initial_cash=initial_cash,
            fee_rate=fee_rate,
            max_workers=max_workers,
            friend_decision=friend_decision,
            run_prefix=run_prefix,
            strict=strict,
            attempt=attempt,
            agent_retries=agent_retries,
            repair_retries=repair_retries,
            round_retries=round_retries,
            retry_backoff=retry_backoff,
        )
        if attempt == 1 and resume_existing:
            existing = find_latest_incomplete_run(out_root, scenario_name, target_rounds=rounds)
            if existing is not None:
                runner.load_existing_run(existing)
                append_run_record(out_root, runner, attempt, status="resumed")
        try:
            runner.run(sleep_between_rounds=sleep_between_rounds)
            append_run_record(out_root, runner, attempt, status="completed")
            return
        except Exception as exc:
            last_error = f"{exc.__class__.__name__}: {exc}"
            runner.write_metadata(status="invalid_preserved", round_count=runner.completed_rounds, error=last_error)
            append_run_record(out_root, runner, attempt, status="invalid_preserved", error=last_error)
            print(f"Invalid run preserved for {scenario_name}, attempt {attempt}/{attempts}: {last_error}")
            if attempt >= attempts:
                append_run_record(out_root, runner, attempt, status="failed", error=last_error)
                raise RuntimeError(f"{scenario_name} failed after {attempts} attempts. Last error: {last_error}") from exc


class LiveScenarioRunner:
    def __init__(
        self,
        scenario_name: str,
        scenario: dict,
        prices: pd.DataFrame,
        out_root: Path,
        rounds: int,
        initial_cash: float,
        fee_rate: float,
        max_workers: int,
        friend_decision: str,
        run_prefix: str,
        strict: bool = True,
        attempt: int = 1,
        agent_retries: int = 3,
        repair_retries: int = 2,
        round_retries: int = 0,
        retry_backoff: float = 3.0,
    ) -> None:
        self.scenario_name = scenario_name
        self.scenario = scenario or {}
        self.prices = prices.copy()
        self.prices["date"] = pd.to_datetime(self.prices["date"])
        self.tickers = sorted(self.prices["ticker"].astype(str).unique())
        self.dates = sorted(self.prices["date"].dropna().unique())
        self.rounds = max(1, int(rounds))
        self.initial_cash = float(initial_cash)
        self.fee_rate = max(0.0, float(fee_rate))
        self.max_workers = max(1, int(max_workers))
        self.friend_decision = friend_decision
        self.strict = bool(strict)
        self.attempt = max(1, int(attempt))
        self.agent_retries = max(1, int(agent_retries))
        self.repair_retries = max(0, int(repair_retries))
        self.round_retries = max(0, int(round_retries))
        self.retry_backoff = max(0.0, float(retry_backoff))
        self.completed_rounds = 0
        self.run_id = f"{run_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{scenario_name}_attempt{self.attempt}"
        self.run_dir = out_root / self.run_id
        self.tables_dir = self.run_dir / "tables"
        self.rounds_dir = self.run_dir / "rounds"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(exist_ok=True)
        self.rounds_dir.mkdir(exist_ok=True)
        self.agents = list(AGENT_PROFILES)
        self.events: list[dict] = []
        self.state_history: list[dict] = []
        self.friendships = initial_friendships_from_scenario(self.scenario, self.agents)
        self.groups = initial_groups_from_scenario(self.scenario, self.agents)
        self.influence_edges = initial_influence_edges_from_scenario(self.scenario, self.agents)
        self.states = {
            agent: {
                "cash": self.initial_cash,
                "positions": {},
                "equity": self.initial_cash,
                "pnl": 0.0,
                "pnl_pct": 0.0,
                "last_action": "等待批量 LLM 运行",
            }
            for agent in self.agents
        }

    def load_existing_run(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_id = run_dir.name
        self.tables_dir = self.run_dir / "tables"
        self.rounds_dir = self.run_dir / "rounds"
        metadata = read_json(self.run_dir / "metadata.json")
        self.attempt = int(metadata.get("attempt", self.attempt) or self.attempt)

        events_df = read_csv(self.tables_dir / "unified_event_log.csv")
        states_df = read_csv(self.tables_dir / "agent_state_history.csv")
        friendships_df = read_csv(self.tables_dir / "friendships.csv")
        groups_df = read_csv(self.tables_dir / "group_memberships.csv")
        social_edges_df = read_csv(self.tables_dir / "social_graph_edges.csv")

        self.events = dataframe_records(events_df)
        self.state_history = dataframe_records(states_df)
        if not states_df.empty and "round_id" in states_df.columns:
            round_numbers = states_df["round_id"].astype(str).str.extract(r"round_(\d+)")[0]
            self.completed_rounds = int(pd.to_numeric(round_numbers, errors="coerce").dropna().max()) if not round_numbers.dropna().empty else 0
        else:
            self.completed_rounds = int(metadata.get("round_count", 0) or 0)

        if not friendships_df.empty and {"agent_a", "agent_b"} <= set(friendships_df.columns):
            self.friendships = {
                tuple(sorted([str(row["agent_a"]), str(row["agent_b"])]))
                for _, row in friendships_df.iterrows()
                if str(row.get("agent_a", "")) in self.agents and str(row.get("agent_b", "")) in self.agents
            }
        if not groups_df.empty and {"group", "agent"} <= set(groups_df.columns):
            groups: dict[str, list[str]] = {}
            for _, row in groups_df.iterrows():
                agent = str(row.get("agent", ""))
                if agent in self.agents:
                    groups.setdefault(str(row.get("group", "")), []).append(agent)
            self.groups = groups
        if not social_edges_df.empty and "kind" in social_edges_df.columns:
            influence = social_edges_df[social_edges_df["kind"].astype(str) == "influence"].copy()
            self.influence_edges = [
                {
                    "source": str(row.get("source", "")),
                    "target": str(row.get("target", "")),
                    "weight": safe_float(row.get("weight", 1.0), 1.0),
                    "kind": "influence",
                }
                for _, row in influence.iterrows()
                if str(row.get("source", "")) in self.agents and str(row.get("target", "")) in self.agents
            ]
        self.restore_agent_states_from_history(states_df)
        print(f"Found existing incomplete run for {self.scenario_name}: {self.run_dir} (completed_rounds={self.completed_rounds})")

    def restore_agent_states_from_history(self, states_df: pd.DataFrame) -> None:
        restored = {}
        if not states_df.empty and "agent" in states_df.columns:
            work = states_df.copy()
            work["_seq"] = range(len(work))
            latest = work.sort_values(["agent", "_seq"]).groupby("agent", as_index=False).tail(1)
            for _, row in latest.iterrows():
                agent = str(row.get("agent", ""))
                if agent not in self.agents:
                    continue
                positions = parse_positions_json(row.get("positions_json", "{}"))
                restored[agent] = {
                    "cash": safe_float(row.get("cash"), self.initial_cash),
                    "positions": positions,
                    "equity": safe_float(row.get("equity"), self.initial_cash),
                    "pnl": safe_float(row.get("pnl"), 0.0),
                    "pnl_pct": safe_float(row.get("pnl_pct"), 0.0),
                    "last_action": str(row.get("last_action", "")),
                }
        for agent in self.agents:
            if agent in restored:
                self.states[agent] = restored[agent]

    def run(self, sleep_between_rounds: float = 0.0) -> None:
        print(f"\n=== {self.scenario_name} -> {self.run_dir} ===")
        if not self.events:
            self.write_metadata(status="running", round_count=0)
            self.append_system_start()
            self.export_tables()
        else:
            self.write_metadata(status="running", round_count=self.completed_rounds)
            print(f"Resuming {self.scenario_name} from round {self.completed_rounds + 1:03d}/{self.rounds}")
        for round_index in range(self.completed_rounds + 1, self.rounds + 1):
            started = time.time()
            self.run_round_until_valid(round_index)
            self.completed_rounds = round_index
            self.write_metadata(status="running", round_count=round_index)
            self.export_tables()
            elapsed = time.time() - started
            print(f"{self.scenario_name} round {round_index:03d}/{self.rounds} done in {elapsed:.1f}s")
            if sleep_between_rounds > 0:
                time.sleep(sleep_between_rounds)
        self.write_metadata(status="completed", round_count=self.rounds)
        print(f"Completed {self.scenario_name}: {self.run_dir}")

    def run_round_until_valid(self, round_number: int) -> None:
        tries = 0
        while True:
            tries += 1
            snapshot = self.snapshot_state()
            try:
                self.run_round(round_number)
                if tries > 1:
                    self.append_round_retry_log(round_number, tries, "passed", "")
                return
            except Exception as exc:
                error = f"{exc.__class__.__name__}: {exc}"
                self.restore_snapshot(snapshot)
                self.delete_round_dir(round_number)
                self.append_round_retry_log(round_number, tries, "retry", error)
                self.write_metadata(status="round_retrying", round_count=self.completed_rounds, error=error)
                if self.round_retries > 0 and tries >= self.round_retries:
                    raise InvalidRunError(
                        f"round_{round_number:03d} failed after {tries} round attempts: {error}"
                    ) from exc
                print(
                    f"{self.scenario_name} round {round_number:03d} invalid; "
                    f"retrying same round attempt {tries + 1}"
                    f"{'' if self.round_retries == 0 else f'/{self.round_retries}'}: {error}"
                )
                time.sleep(self.retry_backoff * min(tries, 6))

    def snapshot_state(self) -> dict:
        return {
            "events": copy.deepcopy(self.events),
            "state_history": copy.deepcopy(self.state_history),
            "friendships": copy.deepcopy(self.friendships),
            "groups": copy.deepcopy(self.groups),
            "influence_edges": copy.deepcopy(self.influence_edges),
            "states": copy.deepcopy(self.states),
        }

    def restore_snapshot(self, snapshot: dict) -> None:
        self.events = snapshot["events"]
        self.state_history = snapshot["state_history"]
        self.friendships = snapshot["friendships"]
        self.groups = snapshot["groups"]
        self.influence_edges = snapshot["influence_edges"]
        self.states = snapshot["states"]

    def delete_round_dir(self, round_number: int) -> None:
        round_dir = self.rounds_dir / f"round_{round_number:03d}"
        if round_dir.exists():
            shutil.rmtree(round_dir)

    def append_round_retry_log(self, round_number: int, attempt: int, status: str, error: str) -> None:
        path = self.run_dir / "ROUND_RETRY_LOG.md"
        if not path.exists():
            path.write_text(
                "\n".join(
                    [
                        "# Round Retry Log",
                        "",
                        "| time | scenario | run_id | round | attempt | status | error |",
                        "| --- | --- | --- | ---: | ---: | --- | --- |",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                "| "
                + " | ".join(
                    [
                        sanitize_markdown_cell(current_timestamp()),
                        sanitize_markdown_cell(self.scenario_name),
                        sanitize_markdown_cell(self.run_id),
                        str(round_number),
                        str(attempt),
                        sanitize_markdown_cell(status),
                        sanitize_markdown_cell(error),
                    ]
                )
                + " |\n"
            )

    def append_system_start(self) -> None:
        now = current_timestamp()
        self.events.append(
            {
                "event_id": self.next_event_id(),
                "date": self.market_date_for_round(1),
                "event_time": now,
                "source": "batch_runner",
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
                "detail": f"批量实时 LLM run 启动，场景={self.scenario_name}，行情=data/TRD_Dalyr.xlsx。",
                "payload": json.dumps({"scenario": self.scenario_name}, ensure_ascii=False),
                "run_id": self.run_id,
                "round_id": "",
                "market_date": self.market_date_for_round(1),
            }
        )
        for agent in self.agents:
            self.append_state_snapshot(agent, round_id="", detail="初始状态", round_number=1)

    def run_round(self, round_number: int) -> None:
        round_id = f"round_{round_number:03d}"
        start_event_id = self.next_event_id()
        recent_events = self.recent_run_events(limit=36)
        jobs = [{"sender": agent} for agent in self.agents]
        results = []
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(jobs))) as executor:
            futures = {
                executor.submit(self.call_llm_agent_plan_with_retries, job["sender"], round_number, recent_events): job
                for job in jobs
            }
            for future in as_completed(futures):
                job = futures[future]
                try:
                    results.append({**job, "plan": future.result(), "error": ""})
                except Exception as exc:
                    error = f"{exc.__class__.__name__}: {exc}"
                    if self.strict:
                        results.append({**job, "plan": None, "error": error})
                    else:
                        results.append({**job, "plan": self.fallback_plan(job["sender"], error, round_number), "error": error})

        self.validate_round_results(round_number, results)

        order = {agent: idx for idx, agent in enumerate(self.agents)}
        for row in sorted(results, key=lambda item: order[item["sender"]]):
            self.append_agent_plan_events(row["sender"], row["plan"], round_number, round_id)
        end_event_id = self.next_event_id() - 1
        self.export_round(round_id, start_event_id, end_event_id, results)

    def call_llm_agent_plan_with_retries(self, agent: str, round_number: int, recent_events: list[dict]) -> dict:
        last_error = ""
        for attempt in range(1, self.agent_retries + 1):
            try:
                return self.call_llm_agent_plan(agent, round_number, recent_events)
            except Exception as exc:
                last_error = f"{exc.__class__.__name__}: {exc}"
                if attempt < self.agent_retries:
                    wait = self.retry_backoff * attempt
                    print(
                        f"{self.scenario_name} round {round_number:03d} {agent} retry "
                        f"{attempt}/{self.agent_retries - 1}: {last_error}"
                    )
                    time.sleep(wait)
        raise RuntimeError(f"{agent} failed after {self.agent_retries} call attempts: {last_error}")

    def call_llm_agent_plan(self, agent: str, round_number: int, recent_events: list[dict]) -> dict:
        api_key = get_llm_setting("OPENAI_API_KEY")
        base_url = get_llm_setting("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        model = get_llm_setting("OPENAI_MODEL", "gpt-4o-mini")
        peers = [name for name in self.agents if name != agent]
        friends = friends_for_agent(self.friendships, agent)
        ticker_example = self.default_ticker_for_agent(agent)
        payload = {
            "model": model,
            "temperature": env_float("LLM_TEMPERATURE", 0.35),
            "max_tokens": env_int("LLM_MAX_TOKENS", 700),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"你是 {agent}。角色设定：{AGENT_PROFILES[agent]} "
                        "你在一个多智能体股票交易社交系统中同时交易和社交。"
                        "你必须只输出一个 JSON 对象，不要输出 Markdown、代码块或额外解释。"
                        "语言必须像研究员/交易员工作记录：具体、克制、可验证。"
                        "禁止诗句、隐喻、成语改写、口号、hashtag、玄学表达和文艺化措辞。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"场景：{self.scenario_name}\n"
                        f"轮次：{round_number}/{self.rounds}\n"
                        f"交易日：{self.market_date_for_round(round_number)}\n"
                        f"可交易 ticker：{', '.join(self.tickers)}\n"
                        f"当前真实历史行情截面：\n{self.market_context(round_number)}\n"
                        f"其他 Agent：{', '.join(peers)}\n"
                        f"当前好友：{', '.join(friends) if friends else '暂无'}\n"
                        f"你的组合状态：{self.agent_portfolio_context(agent)}\n"
                        f"最近事件：\n{format_recent_events(recent_events)}\n\n"
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
                        "friend_request 只能在你能说清楚信息互补、风险验证或策略分歧价值时提出，理由最多 50 个汉字。"
                    ),
                },
            ],
        }
        if truthy_env("LLM_JSON_MODE", default=False):
            payload["response_format"] = {"type": "json_object"}
        data = post_chat_completion(base_url, api_key, payload)
        content = str(data["choices"][0]["message"]["content"]).strip()
        plan = extract_json_object(content)
        if not plan:
            plan = self.repair_agent_plan(agent, round_number, content, ticker_example)
        if not plan:
            if self.strict:
                raise RuntimeError(f"{agent} returned non-JSON content: {content[:180]}")
            plan = {
                "trade": {"side": "HOLD", "ticker": ticker_example, "rationale": "LLM 输出不是 JSON，已回退 HOLD。"},
                "public_message": content,
            }
        return self.normalize_agent_plan(agent, plan, round_number)

    def repair_agent_plan(self, agent: str, round_number: int, partial_content: str, ticker_example: str) -> dict:
        if not str(partial_content or "").strip() or self.repair_retries <= 0:
            return {}
        api_key = get_llm_setting("OPENAI_API_KEY")
        base_url = get_llm_setting("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        model = get_llm_setting("OPENAI_MODEL", "gpt-4o-mini")
        repair_prompt = (
            "下面是一个被截断或格式损坏的 JSON。请只返回一个完整、合法的 JSON 对象，不要解释，不要代码块。\n"
            "必须保留同一个交易意图；如果某字段缺失，用空字符串或 null 补齐。\n"
            "schema：\n"
            "{\n"
            f'  "trade": {{"side": "BUY|SELL|HOLD", "ticker": "{ticker_example}", "shares": 1, "price": 0, "rationale": "交易理由"}},\n'
            '  "public_message": "一句话",\n'
            '  "private_message": {"to": "某个 Agent", "content": "私聊内容"} 或 null,\n'
            '  "moment": "朋友圈动态" 或 null,\n'
            '  "friend_request": {"to": "某个非好友 Agent", "reason": "申请理由"} 或 null\n'
            "}\n"
            f"agent={agent}, round={round_number}, valid_tickers={', '.join(self.tickers)}\n"
            f"损坏内容：\n{str(partial_content)[:1200]}"
        )
        payload = {
            "model": model,
            "temperature": 0.0,
            "max_tokens": env_int("LLM_REPAIR_MAX_TOKENS", 700),
            "messages": [
                {"role": "system", "content": "你是 JSON 修复器。只输出合法 JSON 对象。"},
                {"role": "user", "content": repair_prompt},
            ],
        }
        if truthy_env("LLM_JSON_MODE", default=False):
            payload["response_format"] = {"type": "json_object"}
        last_error = ""
        for attempt in range(1, self.repair_retries + 1):
            try:
                data = post_chat_completion(base_url, api_key, payload)
                repaired = str(data["choices"][0]["message"]["content"]).strip()
                parsed = extract_json_object(repaired)
                if parsed:
                    print(f"{self.scenario_name} round {round_number:03d} {agent} repaired truncated JSON")
                    return parsed
                last_error = f"repair returned non-JSON: {repaired[:160]}"
            except Exception as exc:
                last_error = f"{exc.__class__.__name__}: {exc}"
            if attempt < self.repair_retries:
                time.sleep(self.retry_backoff * attempt)
        print(f"{self.scenario_name} round {round_number:03d} {agent} JSON repair failed: {last_error}")
        return {}

    def validate_round_results(self, round_number: int, results: list[dict]) -> None:
        expected = set(self.agents)
        seen = {str(row.get("sender", "")) for row in results}
        missing = sorted(expected - seen)
        errors = [f"{row.get('sender')}: {row.get('error')}" for row in results if row.get("error")]
        null_plans = sorted(str(row.get("sender", "")) for row in results if not isinstance(row.get("plan"), dict))
        if not self.strict and not missing and not null_plans:
            return
        if missing or errors or null_plans:
            details = []
            if missing:
                details.append(f"missing_agents={missing}")
            if null_plans:
                details.append(f"null_plans={null_plans}")
            if errors:
                details.append("errors=[" + "; ".join(errors) + "]")
            raise InvalidRunError(f"round_{round_number:03d} invalid: " + " | ".join(details))

    def append_agent_plan_events(self, agent: str, plan: dict, round_number: int, round_id: str) -> None:
        plan = self.normalize_agent_plan(agent, plan, round_number)
        trade = self.normalize_trade_plan(agent, plan.get("trade"), round_number)
        executed_ticker = self.append_trade_or_hold(agent, trade, round_number, round_id)
        ticker = executed_ticker or trade["ticker"]
        self.append_message(agent, "public", [], plan.get("public_message", ""), ticker, round_number, round_id)

        private = plan.get("private_message") if isinstance(plan.get("private_message"), dict) else {}
        private_to = str(private.get("to", "")).strip()
        private_content = str(private.get("content", "")).strip()
        if private_to in self.agents and private_to != agent and private_content:
            self.append_message(agent, "private", [private_to], private_content, ticker, round_number, round_id)
            if not are_friends(self.friendships, agent, private_to):
                self.add_friend_request(
                    agent,
                    private_to,
                    "私聊触达非好友，申请建立连接以验证信息价值。",
                    round_number,
                    round_id,
                )

        moment = str(plan.get("moment", "") or "").strip()
        if moment:
            self.append_message(agent, "moments", friends_for_agent(self.friendships, agent), moment, ticker, round_number, round_id)

        request = plan.get("friend_request") if isinstance(plan.get("friend_request"), dict) else {}
        target = str(request.get("to", "")).strip()
        reason = str(request.get("reason", "")).strip()
        if target in self.agents and target != agent and reason:
            self.add_friend_request(agent, target, reason, round_number, round_id)

    def normalize_agent_plan(self, agent: str, plan: dict, round_number: int) -> dict:
        plan = plan if isinstance(plan, dict) else {}
        out = {
            "trade": self.normalize_trade_plan(agent, plan.get("trade"), round_number),
            "public_message": sanitize_text(plan.get("public_message"), "", 80),
            "moment": sanitize_text(plan.get("moment"), "", 90),
            "private_message": None,
            "friend_request": None,
        }
        if not out["public_message"]:
            out["public_message"] = f"{out['trade']['ticker']} 本轮信号有限，我按风险约束调整仓位。"
        private = plan.get("private_message")
        if isinstance(private, dict):
            out["private_message"] = {
                "to": str(private.get("to", "")).strip(),
                "content": sanitize_text(private.get("content"), "", 90),
            }
        request = plan.get("friend_request")
        if isinstance(request, dict):
            out["friend_request"] = {
                "to": str(request.get("to", "")).strip(),
                "reason": sanitize_text(request.get("reason"), "", 90),
            }
        return out

    def normalize_trade_plan(self, agent: str, trade_plan: dict | None, round_number: int) -> dict:
        plan = trade_plan if isinstance(trade_plan, dict) else {}
        side = str(plan.get("side", "HOLD")).upper().strip()
        if side not in {"BUY", "SELL", "HOLD"}:
            side = "HOLD"
        ticker = self.normalize_ticker(plan.get("ticker") or self.default_ticker_for_agent(agent))
        price = safe_float(plan.get("price"), 0.0)
        reference = self.price(ticker, round_number)
        if price <= 0 or price < reference * 0.2 or price > reference * 5:
            price = reference
        shares = int(max(0, safe_float(plan.get("shares"), 0.0)))
        if side in {"BUY", "SELL"} and shares <= 0:
            shares = 5 + stable_index(agent + ticker + side, 11)
        return {
            "side": side,
            "ticker": ticker,
            "shares": min(shares, 100),
            "price": round(price, 2),
            "rationale": sanitize_text(plan.get("rationale"), "", 160),
        }

    def append_trade_or_hold(self, agent: str, trade: dict, round_number: int, round_id: str) -> str:
        state = self.states[agent]
        positions = dict(state["positions"])
        cash = float(state["cash"])
        side, ticker, price, requested = trade["side"], trade["ticker"], trade["price"], int(trade["shares"])
        available = int(positions.get(ticker, 0))
        shares = requested
        if side == "BUY":
            affordable = int(cash // max(price * (1 + self.fee_rate), 0.01))
            shares = max(0, min(shares, affordable))
        elif side == "SELL":
            shares = max(0, min(shares, available))

        if side not in {"BUY", "SELL"} or shares <= 0:
            return self.append_hold(agent, trade, round_number, round_id)

        gross = round(shares * price, 2)
        fee = round(gross * self.fee_rate, 2)
        if side == "BUY":
            positions[ticker] = available + shares
            cash = round(cash - gross - fee, 2)
        else:
            remaining = available - shares
            if remaining > 0:
                positions[ticker] = remaining
            else:
                positions.pop(ticker, None)
            cash = round(cash + gross - fee, 2)
        equity = self.calculate_equity(cash, positions, round_number)
        pnl = round(equity - self.initial_cash, 2)
        pnl_pct = pnl / self.initial_cash if self.initial_cash else 0.0
        detail = f"{side} {shares} {ticker} @ {price:.2f}"
        if trade.get("rationale"):
            detail += f"；{trade['rationale']}"
        self.states[agent] = {
            "cash": cash,
            "positions": positions,
            "equity": equity,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "last_action": detail,
        }
        event_id = self.next_event_id()
        self.events.append(self.event_row(event_id, round_number, round_id, "trade", agent, "", "portfolio", ticker, side, shares, price, gross, cash, equity, pnl, pnl_pct, detail, {"positions": positions}))
        self.append_state_snapshot(agent, round_id, detail, round_number)
        return ticker

    def append_hold(self, agent: str, trade: dict, round_number: int, round_id: str) -> str:
        state = self.states[agent]
        positions = dict(state["positions"])
        cash = float(state["cash"])
        equity = self.calculate_equity(cash, positions, round_number)
        pnl = round(equity - self.initial_cash, 2)
        pnl_pct = pnl / self.initial_cash if self.initial_cash else 0.0
        ticker = trade.get("ticker") or self.default_ticker_for_agent(agent)
        detail = trade.get("rationale") or "HOLD：本轮没有可执行交易。"
        if not str(detail).upper().startswith("HOLD"):
            detail = f"HOLD：{detail}"
        self.states[agent] = {
            "cash": cash,
            "positions": positions,
            "equity": equity,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "last_action": detail,
        }
        event_id = self.next_event_id()
        self.events.append(self.event_row(event_id, round_number, round_id, "hold", agent, "", "portfolio", ticker, "HOLD", 0, 0.0, 0.0, cash, equity, pnl, pnl_pct, detail, {"positions": positions}))
        self.append_state_snapshot(agent, round_id, detail, round_number)
        return ticker

    def append_message(self, agent: str, channel: str, receivers: list[str], detail: str, ticker: str, round_number: int, round_id: str) -> None:
        detail = sanitize_text(detail, "", 120)
        if not detail:
            return
        event_id = self.next_event_id()
        self.events.append(self.event_row(event_id, round_number, round_id, "message", agent, ", ".join(receivers), channel, ticker, "", 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, detail, {"receiver_ids": receivers, "llm_generated": True}))
        self.reinforce_influence(agent, receivers if channel != "public" else [name for name in self.agents if name != agent])

    def add_friend_request(self, agent: str, receiver: str, reason: str, round_number: int, round_id: str) -> None:
        if not agent or not receiver or agent == receiver or are_friends(self.friendships, agent, receiver):
            return
        if self.has_recent_friend_request(agent, receiver):
            return
        request_id = self.next_event_id()
        self.events.append(self.event_row(request_id, round_number, round_id, "friend_request", agent, receiver, "friend_request", "", "", 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, f"{agent} 向 {receiver} 发起好友申请：{sanitize_text(reason, '', 120)}", {"reason": reason, "status": "pending"}))
        decision = self.decide_friend_request(agent, receiver, reason, round_number)
        if decision["accept"]:
            self.friendships.add(tuple(sorted([agent, receiver])))
            event_type = "friend_accept"
            detail = f"{receiver} 通过了 {agent} 的好友申请：{decision['reason']}"
            status = "accepted"
        else:
            event_type = "friend_reject"
            detail = f"{receiver} 拒绝了 {agent} 的好友申请：{decision['reason']}"
            status = "rejected"
        event_id = self.next_event_id()
        self.events.append(self.event_row(event_id, round_number, round_id, event_type, receiver, agent, "friend_request", "", "", 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, detail, {"reason": decision["reason"], "status": status}))

    def decide_friend_request(self, agent: str, receiver: str, reason: str, round_number: int) -> dict:
        if self.friend_decision == "heuristic":
            return heuristic_friend_decision(self.friendships, agent, receiver, reason)
        last_error = ""
        for attempt in range(1, self.agent_retries + 1):
            try:
                return self.call_llm_friend_decision(agent, receiver, reason, round_number)
            except Exception as exc:
                last_error = f"{exc.__class__.__name__}: {exc}"
                if attempt < self.agent_retries:
                    print(
                        f"{self.scenario_name} round {round_number:03d} friend decision "
                        f"{agent}->{receiver} retry {attempt}/{self.agent_retries - 1}: {last_error}"
                    )
                    time.sleep(self.retry_backoff * attempt)
        if self.strict:
            raise InvalidRunError(f"friend decision failed for {agent}->{receiver} after {self.agent_retries} attempts: {last_error}")
        fallback = heuristic_friend_decision(self.friendships, agent, receiver, reason)
        fallback["reason"] = f"LLM 判断失败，使用启发式：{fallback['reason']}"
        fallback["error"] = last_error
        return fallback

    def call_llm_friend_decision(self, agent: str, receiver: str, reason: str, round_number: int) -> dict:
        api_key = get_llm_setting("OPENAI_API_KEY")
        base_url = get_llm_setting("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        model = get_llm_setting("OPENAI_MODEL", "gpt-4o-mini")
        payload = {
            "model": model,
            "temperature": env_float("LLM_TEMPERATURE", 0.35),
            "max_tokens": env_int("LLM_MAX_TOKENS", 300),
            "messages": [
                {
                    "role": "system",
                    "content": "你是好友申请接收方的风控/社交决策模块。只输出 JSON，不要 Markdown。",
                },
                {
                    "role": "user",
                    "content": (
                        f"接收方：{receiver}\n申请方：{agent}\n申请理由：{reason}\n"
                        f"接收方组合状态：{self.agent_portfolio_context(receiver)}\n"
                        f"当前好友：{', '.join(friends_for_agent(self.friendships, receiver)) or '暂无'}\n"
                        f"最近事件：\n{format_recent_events(self.recent_run_events(limit=24))}\n"
                        '请判断是否接受好友申请。输出：{"accept": true|false, "reason": "不超过40字的具体理由"}'
                    ),
                },
            ],
        }
        if truthy_env("LLM_JSON_MODE", default=False):
            payload["response_format"] = {"type": "json_object"}
        data = post_chat_completion(base_url, api_key, payload)
        parsed = extract_json_object(str(data["choices"][0]["message"]["content"]).strip()) or {}
        return {
            "accept": bool(parsed.get("accept", False)),
            "reason": sanitize_text(parsed.get("reason"), "理由不够具体。", 60),
        }

    def fallback_plan(self, agent: str, error: str, round_number: int) -> dict:
        ticker = self.default_ticker_for_agent(agent)
        return self.normalize_agent_plan(
            agent,
            {
                "trade": {"side": "HOLD", "ticker": ticker, "shares": 0, "price": 0, "rationale": f"LLM 不可用，回退 HOLD：{error}"},
                "public_message": f"{ticker} 本轮 LLM 调用失败，我先保持观望。",
                "private_message": None,
                "moment": None,
                "friend_request": None,
            },
            round_number,
        )

    def event_row(
        self,
        event_id: int,
        round_number: int,
        round_id: str,
        event_type: str,
        agent: str,
        counterparty: str,
        channel: str,
        ticker: str,
        side: str,
        shares: int,
        price: float,
        notional: float,
        cash: float,
        equity: float,
        pnl: float,
        pnl_pct: float,
        detail: str,
        payload: dict,
    ) -> dict:
        market_date = self.market_date_for_round(round_number)
        return {
            "event_id": event_id,
            "date": market_date,
            "event_time": current_timestamp(),
            "source": "llm_api_batch",
            "event_type": event_type,
            "agent": agent,
            "counterparty": counterparty,
            "channel": channel,
            "ticker": ticker,
            "side": side,
            "shares": shares,
            "price": round(float(price), 2),
            "notional": round(float(notional), 2),
            "cash": round(float(cash), 2),
            "equity": round(float(equity), 2),
            "pnl": round(float(pnl), 2),
            "pnl_pct": float(pnl_pct),
            "detail": detail,
            "payload": json.dumps(payload, ensure_ascii=False),
            "run_id": self.run_id,
            "round_id": round_id,
            "market_date": market_date,
        }

    def append_state_snapshot(self, agent: str, round_id: str, detail: str, round_number: int) -> None:
        state = self.states[agent]
        friends = sorted(friends_for_agent(self.friendships, agent))
        self.state_history.append(
            {
                "date": self.market_date_for_round(round_number),
                "agent": agent,
                "equity": round(float(state["equity"]), 2),
                "cash": round(float(state["cash"]), 2),
                "pnl": round(float(state["pnl"]), 2),
                "pnl_pct": float(state["pnl_pct"]),
                "positions_json": json.dumps(state["positions"], ensure_ascii=False),
                "position_summary": format_positions(state["positions"]),
                "last_action": sanitize_text(detail, "", 160),
                "friend_count": len(friends),
                "friends": ", ".join(friends),
                "run_id": self.run_id,
                "round_id": round_id,
            }
        )

    def market_context(self, round_number: int) -> str:
        target_date = pd.Timestamp(self.market_date_for_round(round_number))
        lines = []
        for ticker in self.tickers:
            work = self.prices[(self.prices["ticker"].astype(str) == ticker) & (self.prices["date"] <= target_date)]
            if work.empty:
                continue
            work = work.sort_values("date")
            close = float(work["close"].iloc[-1])
            ret5 = period_return(work["close"], 5)
            ret20 = period_return(work["close"], 20)
            volume = float(work["volume"].iloc[-1])
            lines.append(f"{ticker}: date={work['date'].iloc[-1].strftime('%Y-%m-%d')}, close={close:.2f}, ret5={ret5:.2%}, ret20={ret20:.2%}, volume={volume:.0f}")
        return "\n".join(lines) if lines else "暂无可用行情。"

    def price(self, ticker: str, round_number: int) -> float:
        ticker = self.normalize_ticker(ticker)
        target_date = pd.Timestamp(self.market_date_for_round(round_number))
        work = self.prices[(self.prices["ticker"].astype(str) == ticker) & (self.prices["date"] <= target_date)]
        if work.empty:
            latest = self.prices[self.prices["ticker"].astype(str) == ticker].sort_values("date")
        else:
            latest = work.sort_values("date")
        if latest.empty:
            return 1.0
        return round(max(0.01, float(latest["close"].iloc[-1])), 2)

    def calculate_equity(self, cash: float, positions: dict[str, int], round_number: int) -> float:
        holdings = sum(int(shares) * self.price(ticker, round_number) for ticker, shares in positions.items())
        return round(float(cash) + holdings, 2)

    def market_date_for_round(self, round_number: int) -> str:
        if not self.dates:
            return datetime.now().strftime("%Y-%m-%d")
        index = min(max(0, int(round_number) - 1), len(self.dates) - 1)
        return pd.Timestamp(self.dates[index]).strftime("%Y-%m-%d")

    def default_ticker_for_agent(self, agent: str) -> str:
        return self.tickers[stable_index(agent, len(self.tickers))]

    def normalize_ticker(self, value) -> str:
        ticker = "".join(ch for ch in str(value or "").upper() if ch.isalnum() or ch in {".", "-"}).strip(".-")
        if ticker in self.tickers:
            return ticker
        return self.tickers[stable_index(ticker or self.tickers[0], len(self.tickers))]

    def agent_portfolio_context(self, agent: str) -> str:
        state = self.states[agent]
        return (
            f"cash={state['cash']:.2f}, equity={state['equity']:.2f}, pnl={state['pnl']:.2f}, "
            f"positions={format_positions(state['positions'])}"
        )

    def has_recent_friend_request(self, agent: str, receiver: str, window: int = 120) -> bool:
        pair = {agent, receiver}
        for event in reversed(self.events[-window:]):
            if not str(event.get("event_type", "")).startswith("friend"):
                continue
            if {str(event.get("agent", "")), str(event.get("counterparty", ""))} == pair:
                return True
        return False

    def reinforce_influence(self, source: str, targets: list[str]) -> None:
        for target in targets:
            if not target or target == source or target not in self.agents:
                continue
            found = False
            for edge in self.influence_edges:
                if edge["source"] == source and edge["target"] == target:
                    edge["weight"] = min(5.0, float(edge.get("weight", 1.0)) + 0.08)
                    found = True
                    break
            if not found:
                self.influence_edges.append({"source": source, "target": target, "weight": 0.8, "kind": "influence"})

    def recent_run_events(self, limit: int) -> list[dict]:
        return [event for event in self.events if event.get("run_id") == self.run_id][-limit:]

    def next_event_id(self) -> int:
        if not self.events:
            return 1
        return max(int(event.get("event_id", 0)) for event in self.events) + 1

    def export_round(self, round_id: str, start_event_id: int, end_event_id: int, results: list[dict]) -> None:
        round_dir = self.rounds_dir / round_id
        round_dir.mkdir(exist_ok=True)
        events = pd.DataFrame(self.events)
        states = pd.DataFrame(self.state_history)
        if events.empty:
            round_events = pd.DataFrame()
        else:
            ids = pd.to_numeric(events["event_id"], errors="coerce").fillna(0).astype(int)
            round_events = events[(ids >= start_event_id) & (ids <= end_event_id)].copy()
        round_states = states[states["round_id"].astype(str) == round_id].copy() if not states.empty else pd.DataFrame()
        round_events.to_csv(round_dir / "unified_event_log.csv", index=False)
        round_states.to_csv(round_dir / "agent_state_history.csv", index=False)
        live_trades_table(round_events).to_csv(round_dir / "trade_log.csv", index=False)
        live_messages_table(round_events).to_csv(round_dir / "message_log.csv", index=False)
        live_social_events_table(round_events).to_csv(round_dir / "social_events.csv", index=False)
        self.prices.to_csv(round_dir / "market_history.csv", index=False)
        (round_dir / "raw_llm_plans.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    def export_tables(self) -> None:
        events = pd.DataFrame(self.events)
        states = pd.DataFrame(self.state_history)
        events.to_csv(self.tables_dir / "unified_event_log.csv", index=False)
        states.to_csv(self.tables_dir / "agent_state_history.csv", index=False)
        live_trades_table(events).to_csv(self.tables_dir / "trade_log.csv", index=False)
        live_messages_table(events).to_csv(self.tables_dir / "message_log.csv", index=False)
        live_social_events_table(events).to_csv(self.tables_dir / "social_events.csv", index=False)
        self.social_edges_dataframe().to_csv(self.tables_dir / "social_graph_edges.csv", index=False)
        pd.DataFrame(sorted(self.friendships), columns=["agent_a", "agent_b"]).to_csv(self.tables_dir / "friendships.csv", index=False)
        groups_dataframe(self.groups).to_csv(self.tables_dir / "group_memberships.csv", index=False)
        live_equity_table(states).to_csv(self.tables_dir / "equity_curve.csv", index=False)
        live_metrics_table(states).to_csv(self.tables_dir / "performance_metrics.csv", index=False)
        self.prices.to_csv(self.tables_dir / "market_history.csv", index=False)

    def social_edges_dataframe(self) -> pd.DataFrame:
        rows = []
        for a, b in sorted(self.friendships):
            rows.append({"source": a, "target": b, "weight": 1.0, "kind": "friendship"})
            rows.append({"source": b, "target": a, "weight": 1.0, "kind": "friendship"})
        rows.extend(self.influence_edges)
        return pd.DataFrame(rows, columns=["source", "target", "weight", "kind"])

    def write_metadata(self, status: str, round_count: int, error: str = "") -> None:
        metadata = {
            "id": self.run_id,
            "status": status,
            "scenario": self.scenario_name,
            "round_count": round_count,
            "target_rounds": self.rounds,
            "attempt": self.attempt,
            "strict": self.strict,
            "path": str(self.run_dir),
            "updated_at": current_timestamp(),
            "prices_rows": len(self.prices),
            "tickers": self.tickers,
            "friend_decision": self.friend_decision,
            "agent_retries": self.agent_retries,
            "repair_retries": self.repair_retries,
            "round_retries": self.round_retries,
            "retry_backoff": self.retry_backoff,
            "error": error,
        }
        (self.run_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def load_scenarios(path: str | Path) -> dict:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    scenarios = data.get("scenarios", data)
    if not isinstance(scenarios, dict):
        raise ValueError(f"Invalid scenarios file: {path}")
    return scenarios


def select_scenarios(scenarios: dict, value: str) -> list[str]:
    if value == "all":
        return sorted(scenarios)
    selected = [name.strip() for name in value.split(",") if name.strip()]
    missing = [name for name in selected if name not in scenarios]
    if missing:
        raise ValueError(f"Unknown scenarios: {missing}")
    return selected


def start_at_scenario(selected: list[str], scenario_name: str) -> list[str]:
    if scenario_name not in selected:
        raise ValueError(f"--start-at scenario is not in selected scenario list: {scenario_name}")
    return selected[selected.index(scenario_name) :]


def completed_scenarios_from_records(out_root: Path, min_rounds: int) -> set[str]:
    path = out_root / RUN_RECORD_FILE
    if not path.exists():
        return set()
    completed = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or "---" in line or "status" in line:
            continue
        parts = [part.strip().replace("\\|", "|") for part in line.strip().strip("|").split("|")]
        if len(parts) < 7:
            continue
        status, scenario, completed_rounds, target_rounds = parts[1], parts[2], parts[5], parts[6]
        if status != "completed":
            continue
        try:
            done_rounds = int(completed_rounds)
            target = int(target_rounds)
        except ValueError:
            continue
        if done_rounds >= min_rounds and target >= min_rounds:
            completed.add(scenario)
    return completed


def find_latest_incomplete_run(out_root: Path, scenario_name: str, target_rounds: int) -> Path | None:
    candidates = []
    for metadata_path in out_root.glob(f"*_{scenario_name}_attempt*/metadata.json"):
        metadata = read_json(metadata_path)
        if metadata.get("scenario") != scenario_name:
            continue
        status = str(metadata.get("status", ""))
        round_count = int(metadata.get("round_count", 0) or 0)
        if status == "completed" and round_count >= target_rounds:
            continue
        run_dir = metadata_path.parent
        if not (run_dir / "tables" / "unified_event_log.csv").exists():
            continue
        if not (run_dir / "tables" / "agent_state_history.csv").exists():
            continue
        updated = str(metadata.get("updated_at", ""))
        candidates.append((updated, metadata_path.stat().st_mtime, run_dir))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (item[0], item[1]))[-1][2]


def append_run_record(out_root: Path, runner: LiveScenarioRunner, attempt: int, status: str, error: str = "") -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / RUN_RECORD_FILE
    if not path.exists():
        path.write_text(
            "\n".join(
                [
                    "# Live LLM Run Records",
                    "",
                    "| time | status | scenario | run_id | attempt | completed_rounds | target_rounds | path | error |",
                    "| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    row = [
        current_timestamp(),
        status,
        runner.scenario_name,
        runner.run_id,
        str(attempt),
        str(runner.completed_rounds),
        str(runner.rounds),
        str(runner.run_dir),
        sanitize_markdown_cell(error),
    ]
    with path.open("a", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(sanitize_markdown_cell(item) for item in row) + " |\n")


def delete_run_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def sanitize_markdown_cell(value) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")[:500]


def initial_friendships_from_scenario(scenario: dict, agents: list[str]) -> set[tuple[str, str]]:
    friendships = scenario.get("friendships", []) if scenario else []
    if friendships == "all":
        return {tuple(sorted([a, b])) for index, a in enumerate(agents) for b in agents[index + 1 :]}
    pairs = set()
    for pair in friendships or []:
        if len(pair) != 2:
            continue
        a, b = str(pair[0]), str(pair[1])
        if a in agents and b in agents and a != b:
            pairs.add(tuple(sorted([a, b])))
    return pairs


def initial_groups_from_scenario(scenario: dict, agents: list[str]) -> dict[str, list[str]]:
    groups = {}
    for group, members in (scenario.get("groups", {}) if scenario else {}).items():
        groups[str(group)] = [str(member) for member in members if str(member) in agents]
    return groups


def initial_influence_edges_from_scenario(scenario: dict, agents: list[str]) -> list[dict]:
    edges = []
    explicit_edges = (scenario or {}).get("influence_edges", []) or []
    for edge in explicit_edges:
        if len(edge) < 2:
            continue
        source, target = str(edge[0]), str(edge[1])
        if source in agents and target in agents and source != target:
            edges.append({"source": source, "target": target, "weight": float(edge[2]) if len(edge) >= 3 else 1.0, "kind": "influence"})
    if edges:
        return dedupe_edges(edges)
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
        core = [agent for agent in (scenario or {}).get("core", []) if agent in agents] or agents[:3]
        for source, target in zip(core, core[1:] + core[:1]):
            edges.append({"source": source, "target": target, "weight": 1.0, "kind": "influence"})
        for index, target in enumerate([agent for agent in agents if agent not in core]):
            edges.append({"source": core[index % len(core)], "target": target, "weight": 0.7, "kind": "influence"})
    return dedupe_edges(edges)


def dedupe_edges(edges: list[dict]) -> list[dict]:
    out = {}
    for edge in edges:
        key = (edge["source"], edge["target"])
        out[key] = edge
    return [out[key] for key in sorted(out)]


def friends_for_agent(friendships: set[tuple[str, str]], agent: str) -> list[str]:
    friends = []
    for a, b in friendships:
        if a == agent:
            friends.append(b)
        elif b == agent:
            friends.append(a)
    return sorted(friends)


def are_friends(friendships: set[tuple[str, str]], a: str, b: str) -> bool:
    return tuple(sorted([a, b])) in friendships


def heuristic_friend_decision(friendships: set[tuple[str, str]], agent: str, receiver: str, reason: str) -> dict:
    receiver_degree = len(friends_for_agent(friendships, receiver))
    useful_terms = ["风险", "验证", "互补", "分歧", "信息", "策略", "回撤", "趋势"]
    useful = any(term in str(reason) for term in useful_terms)
    accept = receiver_degree < 6 and useful
    return {"accept": accept, "reason": "信息互补且好友数未过高。" if accept else "理由不足或当前连接已过密。"}


def post_chat_completion(base_url: str, api_key: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=env_int("LLM_TIMEOUT", 120)) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM API HTTP {exc.code}: {detail}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise RuntimeError(f"LLM API timeout after {env_int('LLM_TIMEOUT', 120)}s") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM API connection failed: {exc.reason}") from exc


def extract_json_object(text: str) -> dict:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def sanitize_text(value, fallback: str = "", limit: int = 120) -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace("#", "")
    if looks_poetic(text):
        return fallback
    return text[:limit]


def looks_poetic(text: str) -> bool:
    work = str(text or "")
    punctuation_count = sum(work.count(mark) for mark in "，。；、！")
    return len(work) > 80 and punctuation_count >= 4


def format_recent_events(events: list[dict]) -> str:
    if not events:
        return "暂无历史事件。"
    lines = []
    for row in events:
        event_type = row.get("event_type", "")
        channel = row.get("channel", "")
        ticker = row.get("ticker", "")
        side = row.get("side", "")
        detail = sanitize_text(row.get("detail", ""), "[漂移文本已忽略]", 120)
        lines.append(f"#{row.get('event_id', 0)} [{event_type}/{channel}] {row.get('agent', '')} {side} {ticker}: {detail}")
    return "\n".join(lines)


def live_trades_table(event_log: pd.DataFrame) -> pd.DataFrame:
    columns = ["date", "agent", "ticker", "side", "shares", "price", "notional", "cash", "equity", "pnl", "pnl_pct", "detail"]
    if event_log.empty or "event_type" not in event_log.columns:
        return pd.DataFrame(columns=columns)
    trades = event_log[event_log["event_type"] == "trade"].copy()
    return trades[columns].reset_index(drop=True) if not trades.empty else pd.DataFrame(columns=columns)


def live_messages_table(event_log: pd.DataFrame) -> pd.DataFrame:
    columns = ["message_id", "timestamp", "sender_id", "channel", "receiver_ids", "tickers", "direction", "confidence", "natural_language"]
    if event_log.empty or "event_type" not in event_log.columns:
        return pd.DataFrame(columns=columns)
    messages = event_log[event_log["event_type"] == "message"].copy()
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
                "direction": "",
                "confidence": "",
                "natural_language": row.get("detail", ""),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def live_social_events_table(event_log: pd.DataFrame) -> pd.DataFrame:
    columns = ["date", "event_type", "sender", "receiver", "detail"]
    if event_log.empty or "event_type" not in event_log.columns:
        return pd.DataFrame(columns=columns)
    social = event_log[event_log["event_type"].astype(str).str.startswith("friend")].copy()
    if social.empty:
        return pd.DataFrame(columns=columns)
    out = social.rename(columns={"agent": "sender", "counterparty": "receiver"})
    return out[columns]


def live_equity_table(state_history: pd.DataFrame) -> pd.DataFrame:
    columns = ["date", "agent", "equity", "cash", "pnl", "pnl_pct"]
    if state_history.empty:
        return pd.DataFrame(columns=columns)
    return state_history[[column for column in columns if column in state_history.columns]].copy()


def live_metrics_table(state_history: pd.DataFrame) -> pd.DataFrame:
    columns = ["agent", "equity", "cash", "pnl", "pnl_pct"]
    if state_history.empty:
        return pd.DataFrame(columns=columns)
    latest = state_history.copy()
    latest["_seq"] = range(len(latest))
    latest = latest.sort_values(["agent", "_seq"]).groupby("agent", as_index=False).tail(1)
    return latest[columns].sort_values("pnl", ascending=False)


def groups_dataframe(groups: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for group, members in (groups or {}).items():
        for agent in members:
            rows.append({"group": group, "agent": agent})
    return pd.DataFrame(rows, columns=["group", "agent"])


def parse_payload(value) -> dict:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value or "{}"))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def dataframe_records(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    clean = df.where(pd.notna(df), "")
    return clean.to_dict(orient="records")


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
        text = str(ticker)
        try:
            amount = int(shares)
        except (TypeError, ValueError):
            amount = 0
        if amount:
            positions[text] = amount
    return positions


def format_positions(positions: dict[str, int]) -> str:
    active = [f"{ticker} {shares}" for ticker, shares in sorted(positions.items()) if shares]
    return ", ".join(active) if active else "cash only"


def period_return(series: pd.Series, periods: int) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) <= periods:
        return 0.0
    start = float(values.iloc[-periods - 1])
    end = float(values.iloc[-1])
    return end / start - 1.0 if start else 0.0


def stable_index(text: str, modulo: int) -> int:
    if modulo <= 0:
        return 0
    return sum((index + 1) * ord(char) for index, char in enumerate(str(text))) % modulo


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_dotenv_files() -> None:
    for path in [Path(".env"), Path("env.txt")]:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_llm_setting(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def ensure_llm_config() -> None:
    if not get_llm_setting("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY. Put it in env.txt, .env, or environment variables.")
    if not get_llm_setting("OPENAI_MODEL", "gpt-4o-mini"):
        raise RuntimeError("Missing OPENAI_MODEL.")


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def truthy_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def current_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    main()

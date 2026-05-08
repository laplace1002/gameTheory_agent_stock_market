from dataclasses import asdict, dataclass, field
from datetime import timedelta
import math
from typing import Iterable
from uuid import uuid4

import pandas as pd


CHANNELS = {"public", "private", "moments"}
DIRECTION_SCORE = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}


@dataclass
class Message:
    message_id: str
    timestamp: str
    sender_id: str
    channel: str
    receiver_ids: list = field(default_factory=list)
    tickers: list = field(default_factory=list)
    horizon: str = "medium"
    claim_type: str = "meta"
    direction: str = "neutral"
    confidence: float = 0.5
    evidence: list = field(default_factory=list)
    position_intent: str = "unknown"
    natural_language: str = ""
    expires_at: str = ""


def make_message(
    sender_id: str,
    timestamp: str,
    channel: str,
    tickers: Iterable[str],
    direction: str,
    confidence: float,
    evidence: list,
    position_intent: str,
    natural_language: str,
    receiver_ids: Iterable[str] | None = None,
    horizon: str = "medium",
    claim_type: str = "trend",
    ttl_days: int = 20,
) -> Message:
    start = pd.Timestamp(timestamp)
    expires_at = (start + timedelta(days=ttl_days)).strftime("%Y-%m-%d")
    return Message(
        message_id=str(uuid4()),
        timestamp=start.strftime("%Y-%m-%d"),
        sender_id=sender_id,
        channel=channel,
        receiver_ids=list(receiver_ids or []),
        tickers=list(tickers),
        horizon=horizon,
        claim_type=claim_type,
        direction=direction,
        confidence=float(max(0.0, min(1.0, confidence))),
        evidence=evidence,
        position_intent=position_intent,
        natural_language=natural_language,
        expires_at=expires_at,
    )


def evidence_quality(evidence: list) -> float:
    if not evidence:
        return 0.45

    scores = []
    for item in evidence:
        if isinstance(item, dict):
            value = item.get("quality", item.get("strength", item.get("confidence", 0.7)))
            scores.append(float(max(0.0, min(1.0, value))))
        elif isinstance(item, (int, float)):
            scores.append(float(max(0.0, min(1.0, item))))
        else:
            scores.append(0.65)

    base = sum(scores) / max(1, len(scores))
    breadth_bonus = min(0.2, 0.04 * len(evidence))
    return float(max(0.0, min(1.0, base + breadth_bonus)))


class MessageBoard:
    def __init__(self):
        self._messages: list[Message] = []
        self._group_members: dict[str, set[str]] = {}
        self._friends: set[frozenset] = set()

    def post(self, msg: Message) -> None:
        if msg.channel not in CHANNELS:
            raise ValueError(f"unknown message channel: {msg.channel}")
        if msg.direction not in DIRECTION_SCORE:
            raise ValueError(f"unknown message direction: {msg.direction}")
        msg.confidence = float(max(0.0, min(1.0, msg.confidence)))
        self._messages.append(msg)

    def get_public(self, since: str = None) -> list[Message]:
        return self._select(channel="public", since=since)

    def get_private(self, receiver_id: str, since: str = None) -> list[Message]:
        return [
            msg
            for msg in self._select(channel="private", since=since)
            if receiver_id in msg.receiver_ids and self._are_friends(msg.sender_id, receiver_id)
        ]

    def get_moments(self, since: str = None, viewer_groups: Iterable[str] | None = None) -> list[Message]:
        messages = self._select(channel="moments", since=since)
        if viewer_groups is None:
            return messages
        groups = set(viewer_groups)
        if not groups:
            return []
        return [
            msg
            for msg in messages
            if any(msg.sender_id in self._group_members.get(group, set()) for group in groups)
        ]

    def get_visible(
        self,
        receiver_id: str,
        channels: Iterable[str] | None = None,
        since: str = None,
        agent_groups: Iterable[str] | None = None,
    ) -> list[Message]:
        allowed = set(channels or CHANNELS)
        visible: list[Message] = []
        if "public" in allowed:
            visible.extend(self.get_public(since=since))
        if "private" in allowed:
            visible.extend(self.get_private(receiver_id=receiver_id, since=since))
        if "moments" in allowed:
            visible.extend(self.get_moments(since=since, viewer_groups=agent_groups))
        return [msg for msg in visible if msg.sender_id != receiver_id]

    def sync_groups(self, social_graph) -> None:
        self._group_members = {
            group: set(members)
            for group, members in getattr(social_graph, "_groups", {}).items()
        }
        self._friends = getattr(social_graph, "_friends", set())

    def to_dataframe(self) -> pd.DataFrame:
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
        ]
        if not self._messages:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame([asdict(msg) for msg in self._messages], columns=columns)

    def signal_vector(self, msg: Message, current_date: str) -> float:
        if msg.expires_at and pd.Timestamp(current_date) > pd.Timestamp(msg.expires_at):
            return 0.0
        age_days = max(0, (pd.Timestamp(current_date) - pd.Timestamp(msg.timestamp)).days)
        decay = math.exp(-age_days / 15.0)
        direction = DIRECTION_SCORE.get(msg.direction, 0.0)
        return float(direction * msg.confidence * evidence_quality(msg.evidence) * decay)

    def _select(self, channel: str, since: str = None) -> list[Message]:
        selected = [msg for msg in self._messages if msg.channel == channel]
        if since is None:
            return selected
        cutoff = pd.Timestamp(since)
        return [msg for msg in selected if pd.Timestamp(msg.timestamp) >= cutoff]

    def _are_friends(self, sender_id: str, receiver_id: str) -> bool:
        return frozenset([sender_id, receiver_id]) in self._friends

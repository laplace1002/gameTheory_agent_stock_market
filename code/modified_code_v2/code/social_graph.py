from __future__ import annotations

import itertools

import pandas as pd

try:
    import networkx as nx
except ImportError:  # pragma: no cover - used only when dependencies are absent.
    nx = None


class _SimpleDiGraph:
    def __init__(self):
        self._nodes = set()
        self._edges = {}

    def add_node(self, node):
        self._nodes.add(node)

    def add_edge(self, source, target, weight=1.0):
        self.add_node(source)
        self.add_node(target)
        self._edges[(source, target)] = {"weight": float(weight)}

    def has_node(self, node):
        return node in self._nodes

    def has_edge(self, source, target):
        return (source, target) in self._edges

    def nodes(self):
        return list(self._nodes)

    def edges(self, data=False):
        if data:
            return [(s, t, attrs) for (s, t), attrs in self._edges.items()]
        return list(self._edges)

    def in_edges(self, node, data=False):
        rows = [(s, t, attrs) for (s, t), attrs in self._edges.items() if t == node]
        if data:
            return rows
        return [(s, t) for s, t, _ in rows]

    def successors(self, node):
        return [target for (source, target) in self._edges if source == node]


class SocialGraph:
    def __init__(self):
        self.graph = nx.DiGraph() if nx is not None else _SimpleDiGraph()
        self._friends: set[frozenset] = set()
        self._groups: dict[str, set[str]] = {}
        self._pending_requests: list[tuple[str, str]] = []

    def build_initial_graph(self, agents: list) -> None:
        self._reset_influence_graph()
        names = [getattr(agent, "name", str(agent)) for agent in agents]
        for name in names:
            self.graph.add_node(name)

        topo_edges = [
            ("MomentumAgent", "MeanReversionAgent", 0.9),
            ("MeanReversionAgent", "LowVolatilityAgent", 0.9),
            ("LowVolatilityAgent", "DrawdownBuyerAgent", 0.9),
            ("DrawdownBuyerAgent", "MomentumAgent", 0.9),
            ("MomentumAgent", "DynamicTeamAgent", 1.0),
            ("MeanReversionAgent", "DynamicTeamAgent", 1.0),
            ("LowVolatilityAgent", "CommitteeTeamAgent", 1.0),
            ("DrawdownBuyerAgent", "CommitteeTeamAgent", 1.0),
            ("CommitteeTeamAgent", "SocialGraphAgent", 1.2),
            ("DynamicTeamAgent", "SocialGraphAgent", 1.2),
        ]
        for source, target, weight in topo_edges:
            if source in names and target in names and not self.graph.has_edge(source, target):
                self.graph.add_edge(source, target, weight=weight)

    def load_scenario(self, scenario: dict, agents: list) -> None:
        self._friends = set()
        self._groups = {}
        self._pending_requests = []
        names = [getattr(agent, "name", str(agent)) for agent in agents]
        name_set = set(names)

        topology = (scenario or {}).get("topology", "default")
        if topology == "default":
            self.build_initial_graph(agents)
        else:
            self._reset_influence_graph()
            for name in names:
                self.graph.add_node(name)
            self._apply_topology(topology, names, scenario or {})

        for edge in (scenario or {}).get("influence_edges", []) or []:
            if len(edge) < 2:
                continue
            source, target = edge[0], edge[1]
            weight = float(edge[2]) if len(edge) >= 3 else 1.0
            if source in name_set and target in name_set and source != target:
                self.graph.add_edge(source, target, weight=weight)

        friendships = (scenario or {}).get("friendships", [])
        if friendships == "all":
            communicating = self._communicating_names(agents)
            for i, source in enumerate(communicating):
                for target in communicating[i + 1:]:
                    self._add_friendship(source, target, name_set)
        else:
            for pair in friendships or []:
                if len(pair) != 2:
                    continue
                self._add_friendship(pair[0], pair[1], name_set)

        for group_name, members in ((scenario or {}).get("groups") or {}).items():
            self._groups[group_name] = {member for member in members if member in name_set}

    def are_friends(self, a: str, b: str) -> bool:
        return frozenset([a, b]) in self._friends

    def get_friends(self, agent_name: str) -> list[str]:
        friends = []
        for pair in self._friends:
            if agent_name in pair:
                friends.extend(name for name in pair if name != agent_name)
        return sorted(friends)

    def get_group_members(self, group_name: str) -> list[str]:
        return sorted(self._groups.get(group_name, set()))

    def get_agent_groups(self, agent_name: str) -> list[str]:
        return sorted(group for group, members in self._groups.items() if agent_name in members)

    def send_friend_request(self, sender: str, receiver: str) -> None:
        if sender == receiver or self.are_friends(sender, receiver):
            return
        request = (sender, receiver)
        if request not in self._pending_requests:
            self._pending_requests.append(request)

    def process_requests(self, agents: list, rules: dict) -> list[tuple[str, str]]:
        decisions = self.process_requests_detailed(agents, rules)
        return [(row["sender"], row["receiver"]) for row in decisions if row["status"] == "accepted"]

    def process_requests_detailed(self, agents: list, rules: dict) -> list[dict]:
        decisions = []
        agent_map = {agent.name: agent for agent in agents}
        threshold = float((rules or {}).get("strategy_similarity_threshold", 0.5))
        max_friends = int((rules or {}).get("max_friends", 5))

        for sender, receiver in list(self._pending_requests):
            similarity = self._strategy_similarity(agent_map.get(sender), agent_map.get(receiver))
            status = "accepted"
            reason = f"strategy similarity {similarity:.2f} >= threshold {threshold:.2f}"
            if self.are_friends(sender, receiver):
                status = "rejected"
                reason = "already friends"
            elif self._friend_count(sender) >= max_friends:
                status = "rejected"
                reason = f"{sender} reached max_friends={max_friends}"
            elif self._friend_count(receiver) >= max_friends:
                status = "rejected"
                reason = f"{receiver} reached max_friends={max_friends}"
            elif similarity < threshold:
                status = "rejected"
                reason = f"strategy similarity {similarity:.2f} < threshold {threshold:.2f}"

            if status == "accepted":
                self._friends.add(frozenset([sender, receiver]))

            decisions.append(
                {
                    "sender": sender,
                    "receiver": receiver,
                    "status": status,
                    "similarity": float(similarity),
                    "reason": reason,
                }
            )

        self._pending_requests.clear()
        return decisions

    def get_influencers(self, agent_name: str) -> list[tuple[str, float]]:
        if not self.graph.has_node(agent_name):
            return []
        inbound = list(self.graph.in_edges(agent_name, data=True))
        total = sum(float(attrs.get("weight", 1.0)) for _, _, attrs in inbound)
        if total <= 1e-12:
            return []
        return [(source, float(attrs.get("weight", 1.0)) / total) for source, _, attrs in inbound]

    def get_followers(self, agent_name: str) -> list[str]:
        if not self.graph.has_node(agent_name):
            return []
        return list(self.graph.successors(agent_name))

    def update_weights(self, recent_returns: dict) -> None:
        for source, target, attrs in list(self.graph.edges(data=True)):
            score = float(recent_returns.get(source, 0.0))
            next_weight = float(attrs.get("weight", 1.0)) * (1.0 + score)
            attrs["weight"] = float(max(0.1, min(5.0, next_weight)))

    def centrality(self) -> dict:
        nodes = list(self.graph.nodes())
        if not nodes:
            return {}
        if nx is not None:
            return nx.pagerank(self.graph, weight="weight")

        scores = {node: 1.0 / len(nodes) for node in nodes}
        damping = 0.85
        for _ in range(30):
            next_scores = {node: (1.0 - damping) / len(nodes) for node in nodes}
            for source in nodes:
                followers = list(self.graph.successors(source))
                if not followers:
                    continue
                weights = [self._edge_weight(source, target) for target in followers]
                total = sum(weights) or 1.0
                for target, weight in zip(followers, weights):
                    next_scores[target] += damping * scores[source] * weight / total
            scores = next_scores
        return scores

    def to_dataframe(self) -> pd.DataFrame:
        rows = [
            {"source": source, "target": target, "weight": float(attrs.get("weight", 1.0))}
            for source, target, attrs in self.graph.edges(data=True)
        ]
        return pd.DataFrame(rows, columns=["source", "target", "weight"])

    def friendships_dataframe(self) -> pd.DataFrame:
        rows = []
        for pair in sorted(self._friends, key=lambda item: tuple(sorted(item))):
            agent_a, agent_b = sorted(pair)
            rows.append({"agent_a": agent_a, "agent_b": agent_b})
        return pd.DataFrame(rows, columns=["agent_a", "agent_b"])

    def groups_dataframe(self) -> pd.DataFrame:
        rows = []
        for group_name, members in sorted(self._groups.items()):
            for member in sorted(members):
                rows.append({"group": group_name, "agent": member})
        return pd.DataFrame(rows, columns=["group", "agent"])

    def _apply_topology(self, topology: str, names: list[str], scenario: dict) -> None:
        if topology == "isolated":
            return
        if topology == "dense":
            weight = float(scenario.get("default_weight", 1.0))
            for source, target in itertools.permutations(names, 2):
                self.graph.add_edge(source, target, weight=weight)
            return
        if topology == "chain":
            for source, target in zip(names, names[1:]):
                self.graph.add_edge(source, target, weight=1.0)
            return
        if topology == "ring":
            for index, source in enumerate(names):
                self.graph.add_edge(source, names[(index + 1) % len(names)], weight=1.0)
            return
        if topology == "star":
            center = scenario.get("center", names[0] if names else "")
            if center not in names:
                center = names[0] if names else ""
            for name in names:
                if name != center:
                    self.graph.add_edge(center, name, weight=1.4)
                    self.graph.add_edge(name, center, weight=0.7)
            return
        if topology == "core_periphery":
            core = [name for name in scenario.get("core", []) if name in names]
            if not core:
                core = names[: max(1, min(3, len(names)))]
            periphery = [name for name in names if name not in core]
            for source, target in itertools.permutations(core, 2):
                self.graph.add_edge(source, target, weight=1.4)
            for outer in periphery:
                for core_name in core:
                    self.graph.add_edge(core_name, outer, weight=1.0)
                    self.graph.add_edge(outer, core_name, weight=0.45)
            return
        if topology == "echo_chambers":
            groups = scenario.get("groups") or {}
            used = set()
            chambers = []
            for members in groups.values():
                chamber = [name for name in members if name in names]
                if chamber:
                    chambers.append(chamber)
                    used.update(chamber)
            leftovers = [name for name in names if name not in used]
            if leftovers:
                midpoint = max(1, len(leftovers) // 2)
                chambers.extend([leftovers[:midpoint], leftovers[midpoint:]])
            for chamber in chambers:
                for source, target in itertools.permutations(chamber, 2):
                    self.graph.add_edge(source, target, weight=1.3)
            if len(chambers) >= 2:
                for left, right in zip(chambers, chambers[1:]):
                    if left and right:
                        self.graph.add_edge(left[0], right[0], weight=0.25)
                        self.graph.add_edge(right[0], left[0], weight=0.25)
            return
        self.build_initial_graph([type("AgentName", (), {"name": name})() for name in names])

    def _reset_influence_graph(self) -> None:
        self.graph = nx.DiGraph() if nx is not None else _SimpleDiGraph()

    def _add_friendship(self, source: str, target: str, names: set[str]) -> None:
        if source in names and target in names and source != target:
            self._friends.add(frozenset([source, target]))

    def _communicating_names(self, agents: list) -> list[str]:
        return sorted(agent.name for agent in agents if hasattr(agent, "belief"))

    def _friend_count(self, agent_name: str) -> int:
        return sum(1 for pair in self._friends if agent_name in pair)

    def _strategy_similarity(self, a, b) -> float:
        if a is None or b is None:
            return 0.0
        a_groups = self._strategy_groups(a)
        b_groups = self._strategy_groups(b)
        if a_groups & b_groups:
            return 1.0
        if "team" in a_groups or "team" in b_groups:
            return 0.6
        if "risk" in a_groups and "value" in b_groups:
            return 0.45
        if "value" in a_groups and "risk" in b_groups:
            return 0.45
        return 0.2

    def _strategy_groups(self, agent) -> set[str]:
        strategy_groups = {
            "trend": {"MomentumAgent", "TruthfulReporterAgent"},
            "reversion": {"MeanReversionAgent", "PersuaderAgent", "ContrarianAgent"},
            "risk": {"LowVolatilityAgent", "FreeRiderAgent"},
            "value": {"DrawdownBuyerAgent"},
            "random": {"RandomAgent"},
            "team": {"CommitteeTeamAgent", "DynamicTeamAgent", "SocialGraphAgent"},
        }
        labels = {agent.__class__.__name__, getattr(agent, "name", "")}
        own_strategy = getattr(agent, "own_strategy", None)
        if own_strategy is not None:
            labels.add(own_strategy.__class__.__name__)
            labels.add(getattr(own_strategy, "name", ""))
        return {group for group, members in strategy_groups.items() if labels & members}

    def _edge_weight(self, source: str, target: str) -> float:
        for edge_source, edge_target, attrs in self.graph.edges(data=True):
            if edge_source == source and edge_target == target:
                return float(attrs.get("weight", 1.0))
        return 0.0

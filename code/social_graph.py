from __future__ import annotations

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
            ("MomentumAgent", "MeanReversionAgent"),
            ("MeanReversionAgent", "LowVolatilityAgent"),
            ("LowVolatilityAgent", "DrawdownBuyerAgent"),
            ("DrawdownBuyerAgent", "MomentumAgent"),
            ("MomentumAgent", "DynamicTeamAgent"),
            ("MeanReversionAgent", "DynamicTeamAgent"),
            ("LowVolatilityAgent", "CommitteeTeamAgent"),
            ("DrawdownBuyerAgent", "CommitteeTeamAgent"),
            ("CommitteeTeamAgent", "SocialGraphAgent"),
            ("DynamicTeamAgent", "SocialGraphAgent"),
        ]
        for source, target in topo_edges:
            if source in names and target in names and not self.graph.has_edge(source, target):
                self.graph.add_edge(source, target, weight=1.0)

    def load_scenario(self, scenario: dict, agents: list) -> None:
        self.build_initial_graph(agents)
        self._friends = set()
        self._groups = {}
        self._pending_requests = []

        names = {getattr(agent, "name", str(agent)) for agent in agents}
        friendships = scenario.get("friendships", [])
        if friendships == "all":
            communicating = self._communicating_names(agents)
            for i, source in enumerate(communicating):
                for target in communicating[i + 1:]:
                    self._add_friendship(source, target, names)
        else:
            for pair in friendships or []:
                if len(pair) != 2:
                    continue
                self._add_friendship(pair[0], pair[1], names)

        for group_name, members in (scenario.get("groups") or {}).items():
            self._groups[group_name] = {member for member in members if member in names}

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
        accepted = []
        agent_map = {agent.name: agent for agent in agents}
        threshold = float(rules.get("strategy_similarity_threshold", 0.5))
        max_friends = int(rules.get("max_friends", 5))

        for sender, receiver in self._pending_requests:
            if self.are_friends(sender, receiver):
                continue
            if self._friend_count(sender) >= max_friends or self._friend_count(receiver) >= max_friends:
                continue
            similarity = self._strategy_similarity(agent_map.get(sender), agent_map.get(receiver))
            if similarity >= threshold:
                self._friends.add(frozenset([sender, receiver]))
                accepted.append((sender, receiver))

        self._pending_requests.clear()
        return accepted

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
        return 0.2

    def _strategy_groups(self, agent) -> set[str]:
        strategy_groups = {
            "trend": {"MomentumAgent", "TruthfulReporterAgent"},
            "reversion": {"MeanReversionAgent", "PersuaderAgent", "ContrarianAgent"},
            "risk": {"LowVolatilityAgent", "FreeRiderAgent"},
            "value": {"DrawdownBuyerAgent"},
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

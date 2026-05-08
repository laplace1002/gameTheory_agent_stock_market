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

    def build_initial_graph(self, agents: list) -> None:
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

    def _edge_weight(self, source: str, target: str) -> float:
        for edge_source, edge_target, attrs in self.graph.edges(data=True):
            if edge_source == source and edge_target == target:
                return float(attrs.get("weight", 1.0))
        return 0.0

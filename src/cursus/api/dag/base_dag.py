from typing import Dict, List, Any, Optional, Type, Set, Tuple
from collections import deque

import logging


logger = logging.getLogger(__name__)


class PipelineDAG:
    """
    Represents a pipeline topology as a directed acyclic graph (DAG).
    Each node is a step name; edges define dependencies.
    """

    def __init__(
        self, nodes: Optional[List[str]] = None, edges: Optional[List[tuple]] = None
    ):
        """
        nodes: List of step names (str)
        edges: List of (from_step, to_step) tuples
        """
        self.nodes = nodes or []
        self.edges = edges or []
        self.adj_list = {n: [] for n in self.nodes}
        self.reverse_adj = {n: [] for n in self.nodes}

        for src, dst in self.edges:
            self.adj_list[src].append(dst)
            self.reverse_adj[dst].append(src)

    def add_node(self, node: str) -> None:
        """Add a single node to the DAG."""
        if node not in self.nodes:
            self.nodes.append(node)
            self.adj_list[node] = []
            self.reverse_adj[node] = []
            logger.info(f"Added node: {node}")

    def add_edge(self, src: str, dst: str) -> None:
        """Add a directed edge from src to dst."""
        # Ensure both nodes exist
        if src not in self.nodes:
            self.add_node(src)
        if dst not in self.nodes:
            self.add_node(dst)

        # Add the edge if it doesn't already exist
        edge = (src, dst)
        if edge not in self.edges:
            self.edges.append(edge)
            self.adj_list[src].append(dst)
            self.reverse_adj[dst].append(src)
            logger.info(f"Added edge: {src} -> {dst}")

    def get_dependencies(self, node: str) -> List[str]:
        """Return immediate dependencies (parents) of a node."""
        return self.reverse_adj.get(node, [])

    def topological_sort(self) -> List[str]:
        """Return nodes in topological order."""

        in_degree = {n: 0 for n in self.nodes}
        for src, dst in self.edges:
            in_degree[dst] += 1

        queue = deque([n for n in self.nodes if in_degree[n] == 0])
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in self.adj_list[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        if len(order) != len(self.nodes):
            raise ValueError("DAG has cycles or disconnected nodes")
        return order

from typing import Dict, List, Any, Optional, Type, Set, Tuple
from collections import deque

import logging


logger = logging.getLogger(__name__)


class PipelineDAG:
    """
    Represents a pipeline topology as a directed acyclic graph (DAG).
    Each node is a step name; edges define dependencies.

    Node declaration vs. auto-creation: a node is "declared" when it is passed in the ``nodes=``
    constructor arg or added via :meth:`add_node`. By default :meth:`add_edge` auto-creates any
    endpoint that was not declared (lenient mode) — convenient, but it means a single typo in an
    edge name (``add_edge("A", "TabularPreprocessing_traning")``) silently spawns a phantom,
    unconfigured node and orphans the real one, and construction never raises. Two guards exist:

    * :meth:`validate_node_declarations` reports every edge endpoint that was never declared
      (always available, non-fatal) — call it to surface likely typos / forgotten ``add_node``.
    * ``strict=True`` turns the same condition into an immediate ``ValueError`` at ``add_edge``
      (and constructor) time, so undeclared endpoints can never enter the graph.
    """

    def __init__(
        self,
        nodes: Optional[List[str]] = None,
        edges: Optional[List[tuple]] = None,
        strict: bool = False,
    ):
        """
        nodes: List of step names (str)
        edges: List of (from_step, to_step) tuples
        strict: when True, every edge endpoint MUST be a declared node — an undeclared endpoint
            raises ValueError instead of being auto-created. Default False preserves the lenient,
            auto-creating behavior.
        """
        self.strict = strict
        self.nodes = nodes or []
        self.edges = edges or []
        self.adj_list = {n: [] for n in self.nodes}
        self.reverse_adj = {n: [] for n in self.nodes}
        # The set of EXPLICITLY declared nodes (the nodes= arg + every add_node). Endpoints that
        # add_edge auto-creates are deliberately NOT added here, so validate_node_declarations can
        # tell a real node from a typo even after the graph is built.
        self._declared_nodes: Set[str] = set(self.nodes)

        # In strict mode, an edge may not reference an undeclared endpoint.
        if self.strict:
            undeclared = self.validate_node_declarations()
            if undeclared:
                raise ValueError(
                    f"strict DAG: edge endpoint(s) {undeclared} are not in the declared node list. "
                    f"Add them to nodes=[...] (or call add_node) before wiring edges, or construct "
                    f"the DAG with strict=False to auto-create them."
                )

        # Build adjacency. Edge endpoints not present in ``nodes`` (dangling edges) are
        # tolerated here — they get an adjacency entry but are NOT added to ``nodes`` — so
        # construction never raises an opaque KeyError. Dangling edges are reported with a
        # clear message by the serializer's validation layer (and by validate_node_declarations).
        for src, dst in self.edges:
            self.adj_list.setdefault(src, []).append(dst)
            self.reverse_adj.setdefault(dst, []).append(src)

    def add_node(self, node: str) -> None:
        """Add a single node to the DAG. This DECLARES the node (see class docstring)."""
        if node not in self.nodes:
            self.nodes.append(node)
            self.adj_list.setdefault(node, [])
            self.reverse_adj.setdefault(node, [])
            logger.info(f"Added node: {node}")
        # Mark declared even if the node was previously auto-created by an edge: an explicit
        # add_node promotes it from auto-created to declared.
        self._declared_nodes.add(node)

    def add_edge(self, src: str, dst: str) -> None:
        """Add a directed edge from src to dst.

        Lenient (default): auto-creates either endpoint if it is not yet a node — but does NOT
        mark it declared, so validate_node_declarations will still surface it as a likely typo.
        strict=True: raises ValueError if either endpoint was never declared via add_node.
        """
        if self.strict:
            undeclared = [n for n in (src, dst) if n not in self._declared_nodes]
            if undeclared:
                raise ValueError(
                    f"add_edge({src!r}, {dst!r}): endpoint(s) {undeclared} were never declared via "
                    f"add_node (strict mode). Declare every node before adding edges, or construct "
                    f"the DAG with strict=False to auto-create them."
                )

        # Ensure both nodes exist (auto-create in lenient mode — intentionally NOT declared).
        if src not in self.nodes:
            self.nodes.append(src)
            self.adj_list.setdefault(src, [])
            self.reverse_adj.setdefault(src, [])
        if dst not in self.nodes:
            self.nodes.append(dst)
            self.adj_list.setdefault(dst, [])
            self.reverse_adj.setdefault(dst, [])

        # Add the edge if it doesn't already exist
        edge = (src, dst)
        if edge not in self.edges:
            self.edges.append(edge)
            self.adj_list[src].append(dst)
            self.reverse_adj[dst].append(src)
            logger.info(f"Added edge: {src} -> {dst}")

    def validate_node_declarations(self) -> List[str]:
        """Return edge endpoints that were never explicitly declared via add_node / the nodes= arg.

        Because add_edge auto-creates missing endpoints (lenient mode), a typo in an edge name
        silently produces a phantom, unconfigured node — and the serializer's dangling-edge check
        cannot catch it, since add_edge has already promoted the typo into ``nodes``. This is the
        only reliable detector: it compares every edge endpoint against the DECLARED set. An empty
        list means every edge endpoint was declared; any member is a likely typo or a forgotten
        add_node. Non-fatal — strict=True turns the same condition into a raise at add_edge time.
        """
        undeclared: List[str] = []
        seen: Set[str] = set()
        for src, dst in self.edges:
            for endpoint in (src, dst):
                if endpoint not in self._declared_nodes and endpoint not in seen:
                    seen.add(endpoint)
                    undeclared.append(endpoint)
        return undeclared

    def get_dependencies(self, node: str) -> List[str]:
        """Return immediate dependencies (parents) of a node."""
        return self.reverse_adj.get(node, [])

    def topological_sort(self) -> List[str]:
        """Return nodes in topological order.

        Tolerates edges whose endpoints are not in ``nodes`` (dangling edges) by ignoring
        the unknown endpoints here rather than raising an opaque ``KeyError`` — structural
        problems like dangling edges are surfaced with clear messages by the serializer's
        validation layer.
        """

        in_degree = {n: 0 for n in self.nodes}
        for _src, dst in self.edges:
            if dst in in_degree:
                in_degree[dst] += 1

        queue = deque([n for n in self.nodes if in_degree[n] == 0])
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in self.adj_list.get(node, []):
                if neighbor not in in_degree:
                    continue
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        if len(order) != len(self.nodes):
            raise ValueError("DAG has cycles or disconnected nodes")
        return order

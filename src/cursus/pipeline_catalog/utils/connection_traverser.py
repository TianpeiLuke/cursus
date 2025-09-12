"""
Connection Traversal Utilities

Navigate pipeline connections following Zettelkasten principles.
Supports manual linking over search by providing curated navigation
paths through the connection graph.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import deque, defaultdict
from pydantic import BaseModel

from .catalog_registry import CatalogRegistry

logger = logging.getLogger(__name__)


class PipelineConnection(BaseModel):
    """Represents a connection between two pipelines."""

    target_id: str
    connection_type: str  # alternatives, related, used_in
    annotation: str
    source_id: Optional[str] = None


class ConnectionTraverser:
    """
    Navigate pipeline connections following Zettelkasten principles.

    Supports manual linking over search by providing curated navigation
    paths through the connection graph.
    """

    def __init__(self, registry: CatalogRegistry):
        """
        Initialize with registry instance.

        Args:
            registry: CatalogRegistry instance for accessing pipeline data
        """
        self.registry = registry
        self._connection_cache = {}
        self._cache_valid = False

    def get_alternatives(self, pipeline_id: str) -> List[PipelineConnection]:
        """
        Get alternative pipelines for the same task.

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            List of alternative pipeline connections
        """
        return self._get_connections_by_type(pipeline_id, "alternatives")

    def get_related(self, pipeline_id: str) -> List[PipelineConnection]:
        """
        Get conceptually related pipelines.

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            List of related pipeline connections
        """
        return self._get_connections_by_type(pipeline_id, "related")

    def get_compositions(self, pipeline_id: str) -> List[PipelineConnection]:
        """
        Get pipelines that can use this pipeline in composition.

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            List of composition pipeline connections
        """
        return self._get_connections_by_type(pipeline_id, "used_in")

    def get_all_connections(
        self, pipeline_id: str
    ) -> Dict[str, List[PipelineConnection]]:
        """
        Get all connections for a pipeline organized by type.

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            Dictionary mapping connection types to connection lists
        """
        try:
            connections = self.registry.get_pipeline_connections(pipeline_id)
            result = {}

            for conn_type, conn_list in connections.items():
                result[conn_type] = [
                    PipelineConnection(
                        target_id=conn["id"],
                        connection_type=conn_type,
                        annotation=conn["annotation"],
                        source_id=pipeline_id,
                    )
                    for conn in conn_list
                ]

            return result

        except Exception as e:
            logger.error(f"Failed to get all connections for {pipeline_id}: {e}")
            return {}

    def _get_connections_by_type(
        self, pipeline_id: str, connection_type: str
    ) -> List[PipelineConnection]:
        """
        Get connections of a specific type for a pipeline.

        Args:
            pipeline_id: Pipeline identifier
            connection_type: Type of connection to retrieve

        Returns:
            List of pipeline connections
        """
        try:
            connections = self.registry.get_pipeline_connections(pipeline_id)
            conn_list = connections.get(connection_type, [])

            return [
                PipelineConnection(
                    target_id=conn["id"],
                    connection_type=connection_type,
                    annotation=conn["annotation"],
                    source_id=pipeline_id,
                )
                for conn in conn_list
            ]

        except Exception as e:
            logger.error(
                f"Failed to get {connection_type} connections for {pipeline_id}: {e}"
            )
            return []

    def traverse_connection_path(
        self, start_id: str, connection_types: List[str], max_depth: int = 3
    ) -> List[List[str]]:
        """
        Traverse connection paths following specified types.

        Args:
            start_id: Starting pipeline identifier
            connection_types: List of connection types to follow
            max_depth: Maximum traversal depth

        Returns:
            List of connection paths (each path is a list of pipeline IDs)
        """
        try:
            if max_depth <= 0:
                return [[start_id]]

            paths = []
            visited = set()

            def dfs(current_id: str, current_path: List[str], depth: int):
                if depth >= max_depth:
                    paths.append(current_path)
                    return

                if current_id in visited:
                    paths.append(current_path)
                    return

                visited.add(current_id)

                # Get connections for current pipeline
                connections = self.registry.get_pipeline_connections(current_id)
                has_connections = False

                for conn_type in connection_types:
                    if conn_type in connections:
                        for conn in connections[conn_type]:
                            target_id = conn["id"]
                            if target_id not in current_path:  # Avoid cycles
                                has_connections = True
                                dfs(target_id, current_path + [target_id], depth + 1)

                if not has_connections:
                    paths.append(current_path)

                visited.remove(current_id)

            dfs(start_id, [start_id], 0)
            return paths

        except Exception as e:
            logger.error(f"Failed to traverse connection path from {start_id}: {e}")
            return [[start_id]]

    def find_shortest_path(self, start_id: str, end_id: str) -> Optional[List[str]]:
        """
        Find shortest connection path between two pipelines.

        Args:
            start_id: Starting pipeline identifier
            end_id: Target pipeline identifier

        Returns:
            Shortest path as list of pipeline IDs, or None if no path exists
        """
        try:
            if start_id == end_id:
                return [start_id]

            # BFS to find shortest path
            queue = deque([(start_id, [start_id])])
            visited = {start_id}

            while queue:
                current_id, path = queue.popleft()

                # Get all connections from current pipeline
                connections = self.registry.get_pipeline_connections(current_id)

                for conn_type, conn_list in connections.items():
                    for conn in conn_list:
                        target_id = conn["id"]

                        if target_id == end_id:
                            return path + [target_id]

                        if target_id not in visited:
                            visited.add(target_id)
                            queue.append((target_id, path + [target_id]))

            return None  # No path found

        except Exception as e:
            logger.error(
                f"Failed to find shortest path from {start_id} to {end_id}: {e}"
            )
            return None

    def get_connection_subgraph(
        self, pipeline_id: str, depth: int = 2
    ) -> Dict[str, Any]:
        """
        Get subgraph of connections around a pipeline.

        Args:
            pipeline_id: Central pipeline identifier
            depth: Depth of subgraph to extract

        Returns:
            Subgraph data with nodes and edges
        """
        try:
            nodes = {}
            edges = []
            visited = set()

            def collect_subgraph(current_id: str, current_depth: int):
                if current_depth > depth or current_id in visited:
                    return

                visited.add(current_id)

                # Add node
                node_data = self.registry.get_pipeline_node(current_id)
                if node_data:
                    nodes[current_id] = {
                        "id": current_id,
                        "title": node_data.get("title", current_id),
                        "framework": node_data.get("zettelkasten_metadata", {}).get(
                            "framework", "unknown"
                        ),
                        "complexity": node_data.get("zettelkasten_metadata", {}).get(
                            "complexity", "unknown"
                        ),
                        "depth": current_depth,
                    }

                # Add edges and recurse
                connections = self.registry.get_pipeline_connections(current_id)
                for conn_type, conn_list in connections.items():
                    for conn in conn_list:
                        target_id = conn["id"]

                        # Add edge
                        edges.append(
                            {
                                "source": current_id,
                                "target": target_id,
                                "type": conn_type,
                                "annotation": conn["annotation"],
                            }
                        )

                        # Recurse if within depth limit
                        if current_depth < depth:
                            collect_subgraph(target_id, current_depth + 1)

            collect_subgraph(pipeline_id, 0)

            return {
                "center_node": pipeline_id,
                "depth": depth,
                "nodes": nodes,
                "edges": edges,
                "node_count": len(nodes),
                "edge_count": len(edges),
            }

        except Exception as e:
            logger.error(f"Failed to get connection subgraph for {pipeline_id}: {e}")
            return {
                "center_node": pipeline_id,
                "depth": depth,
                "nodes": {},
                "edges": [],
                "node_count": 0,
                "edge_count": 0,
            }

    def find_connection_clusters(self) -> List[List[str]]:
        """
        Find clusters of connected pipelines.

        Returns:
            List of clusters, each cluster is a list of pipeline IDs
        """
        try:
            all_pipelines = set(self.registry.get_all_pipelines())
            clusters = []
            visited = set()

            def dfs_cluster(pipeline_id: str, current_cluster: List[str]):
                if pipeline_id in visited:
                    return

                visited.add(pipeline_id)
                current_cluster.append(pipeline_id)

                # Get all connections (both outgoing and incoming)
                connections = self.registry.get_pipeline_connections(pipeline_id)

                # Follow outgoing connections
                for conn_type, conn_list in connections.items():
                    for conn in conn_list:
                        target_id = conn["id"]
                        if target_id not in visited:
                            dfs_cluster(target_id, current_cluster)

                # Find incoming connections
                for other_id in all_pipelines:
                    if other_id != pipeline_id and other_id not in visited:
                        other_connections = self.registry.get_pipeline_connections(
                            other_id
                        )
                        for conn_type, conn_list in other_connections.items():
                            for conn in conn_list:
                                if conn["id"] == pipeline_id:
                                    dfs_cluster(other_id, current_cluster)

            for pipeline_id in all_pipelines:
                if pipeline_id not in visited:
                    cluster = []
                    dfs_cluster(pipeline_id, cluster)
                    if cluster:
                        clusters.append(cluster)

            # Sort clusters by size (largest first)
            clusters.sort(key=len, reverse=True)
            return clusters

        except Exception as e:
            logger.error(f"Failed to find connection clusters: {e}")
            return []

    def get_bidirectional_connections(
        self, pipeline_id: str
    ) -> List[PipelineConnection]:
        """
        Get bidirectional connections for a pipeline.

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            List of bidirectional connections
        """
        try:
            bidirectional = []

            # Get outgoing connections
            outgoing = self.registry.get_pipeline_connections(pipeline_id)

            for conn_type, conn_list in outgoing.items():
                for conn in conn_list:
                    target_id = conn["id"]

                    # Check if target has connection back to source
                    target_connections = self.registry.get_pipeline_connections(
                        target_id
                    )

                    for (
                        target_conn_type,
                        target_conn_list,
                    ) in target_connections.items():
                        for target_conn in target_conn_list:
                            if target_conn["id"] == pipeline_id:
                                bidirectional.append(
                                    PipelineConnection(
                                        target_id=target_id,
                                        connection_type=conn_type,
                                        annotation=f"Bidirectional: {conn['annotation']} | {target_conn['annotation']}",
                                        source_id=pipeline_id,
                                    )
                                )
                                break

            return bidirectional

        except Exception as e:
            logger.error(
                f"Failed to get bidirectional connections for {pipeline_id}: {e}"
            )
            return []

    def analyze_connection_patterns(self) -> Dict[str, Any]:
        """
        Analyze connection patterns across the entire registry.

        Returns:
            Analysis results with pattern statistics
        """
        try:
            all_pipelines = self.registry.get_all_pipelines()

            # Connection type statistics
            connection_type_counts = defaultdict(int)
            total_connections = 0

            # Hub analysis (pipelines with many connections)
            connection_counts = {}

            # Framework connection patterns
            framework_connections = defaultdict(lambda: defaultdict(int))

            for pipeline_id in all_pipelines:
                connections = self.registry.get_pipeline_connections(pipeline_id)
                pipeline_connection_count = 0

                # Get pipeline framework
                node = self.registry.get_pipeline_node(pipeline_id)
                source_framework = (
                    node.get("zettelkasten_metadata", {}).get("framework", "unknown")
                    if node
                    else "unknown"
                )

                for conn_type, conn_list in connections.items():
                    connection_type_counts[conn_type] += len(conn_list)
                    total_connections += len(conn_list)
                    pipeline_connection_count += len(conn_list)

                    # Analyze framework connections
                    for conn in conn_list:
                        target_node = self.registry.get_pipeline_node(conn["id"])
                        target_framework = (
                            target_node.get("zettelkasten_metadata", {}).get(
                                "framework", "unknown"
                            )
                            if target_node
                            else "unknown"
                        )
                        framework_connections[source_framework][target_framework] += 1

                connection_counts[pipeline_id] = pipeline_connection_count

            # Find hubs (top 20% by connection count)
            sorted_by_connections = sorted(
                connection_counts.items(), key=lambda x: x[1], reverse=True
            )
            hub_threshold = max(1, len(sorted_by_connections) // 5)  # Top 20%
            hubs = sorted_by_connections[:hub_threshold]

            # Find isolated nodes
            isolated_nodes = [
                pid for pid, count in connection_counts.items() if count == 0
            ]

            # Calculate connection density
            max_possible_connections = len(all_pipelines) * (len(all_pipelines) - 1)
            connection_density = (
                total_connections / max_possible_connections
                if max_possible_connections > 0
                else 0.0
            )

            return {
                "total_pipelines": len(all_pipelines),
                "total_connections": total_connections,
                "connection_density": connection_density,
                "connection_type_distribution": dict(connection_type_counts),
                "average_connections_per_pipeline": (
                    total_connections / len(all_pipelines) if all_pipelines else 0
                ),
                "hub_pipelines": [
                    {"id": pid, "connections": count} for pid, count in hubs
                ],
                "isolated_pipelines": isolated_nodes,
                "framework_connection_matrix": dict(framework_connections),
                "most_connected_pipeline": (
                    sorted_by_connections[0] if sorted_by_connections else None
                ),
                "connection_distribution": {
                    "0_connections": len(
                        [c for c in connection_counts.values() if c == 0]
                    ),
                    "1-3_connections": len(
                        [c for c in connection_counts.values() if 1 <= c <= 3]
                    ),
                    "4-6_connections": len(
                        [c for c in connection_counts.values() if 4 <= c <= 6]
                    ),
                    "7+_connections": len(
                        [c for c in connection_counts.values() if c >= 7]
                    ),
                },
            }

        except Exception as e:
            logger.error(f"Failed to analyze connection patterns: {e}")
            return {"error": str(e)}

    def suggest_missing_connections(self, pipeline_id: str) -> List[Dict[str, Any]]:
        """
        Suggest potentially missing connections based on similarity.

        Args:
            pipeline_id: Pipeline identifier to analyze

        Returns:
            List of connection suggestions
        """
        try:
            suggestions = []

            # Get pipeline metadata
            node = self.registry.get_pipeline_node(pipeline_id)
            if not node:
                return suggestions

            pipeline_framework = node.get("zettelkasten_metadata", {}).get(
                "framework", ""
            )
            pipeline_complexity = node.get("zettelkasten_metadata", {}).get(
                "complexity", ""
            )
            pipeline_tags = node.get("multi_dimensional_tags", {})

            # Get existing connections
            existing_connections = set()
            connections = self.registry.get_pipeline_connections(pipeline_id)
            for conn_list in connections.values():
                for conn in conn_list:
                    existing_connections.add(conn["id"])

            # Analyze all other pipelines
            all_pipelines = self.registry.get_all_pipelines()

            for other_id in all_pipelines:
                if other_id == pipeline_id or other_id in existing_connections:
                    continue

                other_node = self.registry.get_pipeline_node(other_id)
                if not other_node:
                    continue

                other_framework = other_node.get("zettelkasten_metadata", {}).get(
                    "framework", ""
                )
                other_complexity = other_node.get("zettelkasten_metadata", {}).get(
                    "complexity", ""
                )
                other_tags = other_node.get("multi_dimensional_tags", {})

                # Calculate similarity score
                similarity_score = 0.0
                reasons = []

                # Framework similarity
                if pipeline_framework == other_framework:
                    similarity_score += 0.3
                    reasons.append(f"Same framework ({pipeline_framework})")

                # Complexity similarity
                if pipeline_complexity == other_complexity:
                    similarity_score += 0.2
                    reasons.append(f"Same complexity ({pipeline_complexity})")

                # Tag similarity
                for tag_category, tags in pipeline_tags.items():
                    other_category_tags = other_tags.get(tag_category, [])
                    common_tags = set(tags) & set(other_category_tags)
                    if common_tags:
                        similarity_score += len(common_tags) * 0.1
                        reasons.append(
                            f"Common {tag_category}: {', '.join(common_tags)}"
                        )

                # Suggest connection if similarity is high enough
                if similarity_score >= 0.4:
                    # Determine connection type based on similarity
                    if similarity_score >= 0.7:
                        suggested_type = "alternatives"
                    elif similarity_score >= 0.5:
                        suggested_type = "related"
                    else:
                        suggested_type = "related"

                    suggestions.append(
                        {
                            "target_id": other_id,
                            "target_title": other_node.get("title", other_id),
                            "suggested_type": suggested_type,
                            "similarity_score": similarity_score,
                            "reasons": reasons,
                        }
                    )

            # Sort by similarity score
            suggestions.sort(key=lambda x: x["similarity_score"], reverse=True)

            # Return top 5 suggestions
            return suggestions[:5]

        except Exception as e:
            logger.error(
                f"Failed to suggest missing connections for {pipeline_id}: {e}"
            )
            return []

    def clear_cache(self) -> None:
        """Clear the internal connection cache."""
        self._connection_cache = {}
        self._cache_valid = False
        logger.debug("Connection traverser cache cleared")

"""
Unit tests for ConnectionTraverser class.

Tests connection navigation and analysis functionality following
Zettelkasten principles for manual linking over search.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any, List

from cursus.pipeline_catalog.core.connection_traverser import ConnectionTraverser, PipelineConnection
from cursus.pipeline_catalog.core.catalog_registry import CatalogRegistry


class TestConnectionTraverser:
    """Test suite for ConnectionTraverser class."""

    @pytest.fixture
    def mock_registry(self):
        """Create mock CatalogRegistry."""
        return Mock(spec=CatalogRegistry)

    @pytest.fixture
    def sample_connections(self):
        """Sample connection data for testing."""
        return {
            "alternatives": [
                {"id": "alt_pipeline", "annotation": "Alternative implementation"}
            ],
            "related": [
                {"id": "related_pipeline", "annotation": "Related functionality"}
            ],
            "used_in": [
                {"id": "composite_pipeline", "annotation": "Used in composition"}
            ]
        }

    @pytest.fixture
    def sample_nodes(self):
        """Sample node data for testing."""
        return {
            "test_pipeline": {
                "id": "test_pipeline",
                "title": "Test Pipeline",
                "zettelkasten_metadata": {
                    "framework": "xgboost",
                    "complexity": "simple"
                },
                "multi_dimensional_tags": {
                    "framework_tags": ["xgboost"],
                    "task_tags": ["training"]
                }
            },
            "alt_pipeline": {
                "id": "alt_pipeline",
                "title": "Alternative Pipeline",
                "zettelkasten_metadata": {
                    "framework": "xgboost",
                    "complexity": "simple"
                },
                "multi_dimensional_tags": {
                    "framework_tags": ["xgboost"],
                    "task_tags": ["training"]
                }
            },
            "related_pipeline": {
                "id": "related_pipeline",
                "title": "Related Pipeline",
                "zettelkasten_metadata": {
                    "framework": "pytorch",
                    "complexity": "standard"
                },
                "multi_dimensional_tags": {
                    "framework_tags": ["pytorch"],
                    "task_tags": ["evaluation"]
                }
            }
        }

    @pytest.fixture
    def traverser(self, mock_registry):
        """Create ConnectionTraverser instance."""
        return ConnectionTraverser(mock_registry)

    def test_init(self, mock_registry):
        """Test ConnectionTraverser initialization."""
        traverser = ConnectionTraverser(mock_registry)
        
        assert traverser.registry == mock_registry
        assert traverser._connection_cache == {}
        assert not traverser._cache_valid

    def test_get_alternatives(self, traverser, mock_registry, sample_connections):
        """Test getting alternative pipelines."""
        mock_registry.get_pipeline_connections.return_value = sample_connections
        
        alternatives = traverser.get_alternatives("test_pipeline")
        
        assert len(alternatives) == 1
        assert alternatives[0].target_id == "alt_pipeline"
        assert alternatives[0].connection_type == "alternatives"
        assert alternatives[0].annotation == "Alternative implementation"
        assert alternatives[0].source_id == "test_pipeline"
        mock_registry.get_pipeline_connections.assert_called_once_with("test_pipeline")

    def test_get_related(self, traverser, mock_registry, sample_connections):
        """Test getting related pipelines."""
        mock_registry.get_pipeline_connections.return_value = sample_connections
        
        related = traverser.get_related("test_pipeline")
        
        assert len(related) == 1
        assert related[0].target_id == "related_pipeline"
        assert related[0].connection_type == "related"
        assert related[0].annotation == "Related functionality"
        assert related[0].source_id == "test_pipeline"

    def test_get_compositions(self, traverser, mock_registry, sample_connections):
        """Test getting composition pipelines."""
        mock_registry.get_pipeline_connections.return_value = sample_connections
        
        compositions = traverser.get_compositions("test_pipeline")
        
        assert len(compositions) == 1
        assert compositions[0].target_id == "composite_pipeline"
        assert compositions[0].connection_type == "used_in"
        assert compositions[0].annotation == "Used in composition"
        assert compositions[0].source_id == "test_pipeline"

    def test_get_all_connections(self, traverser, mock_registry, sample_connections):
        """Test getting all connections organized by type."""
        mock_registry.get_pipeline_connections.return_value = sample_connections
        
        all_connections = traverser.get_all_connections("test_pipeline")
        
        assert len(all_connections) == 3
        assert "alternatives" in all_connections
        assert "related" in all_connections
        assert "used_in" in all_connections
        
        assert len(all_connections["alternatives"]) == 1
        assert all_connections["alternatives"][0].target_id == "alt_pipeline"
        
        assert len(all_connections["related"]) == 1
        assert all_connections["related"][0].target_id == "related_pipeline"
        
        assert len(all_connections["used_in"]) == 1
        assert all_connections["used_in"][0].target_id == "composite_pipeline"

    def test_get_all_connections_error(self, traverser, mock_registry):
        """Test getting all connections with error."""
        mock_registry.get_pipeline_connections.side_effect = Exception("Registry error")
        
        all_connections = traverser.get_all_connections("test_pipeline")
        
        assert all_connections == {}

    def test_get_connections_by_type_error(self, traverser, mock_registry):
        """Test getting connections by type with error."""
        mock_registry.get_pipeline_connections.side_effect = Exception("Registry error")
        
        alternatives = traverser.get_alternatives("test_pipeline")
        
        assert alternatives == []

    def test_traverse_connection_path_simple(self, traverser, mock_registry):
        """Test simple connection path traversal."""
        # Mock connection data for path traversal
        connections_map = {
            "start": {
                "related": [{"id": "middle", "annotation": "to middle"}]
            },
            "middle": {
                "related": [{"id": "end", "annotation": "to end"}]
            },
            "end": {
                "related": []
            }
        }
        
        def mock_get_connections(pipeline_id):
            return connections_map.get(pipeline_id, {})
        
        mock_registry.get_pipeline_connections.side_effect = mock_get_connections
        
        paths = traverser.traverse_connection_path("start", ["related"], max_depth=3)
        
        assert len(paths) > 0
        # Should find path: start -> middle -> end
        assert ["start", "middle", "end"] in paths

    def test_traverse_connection_path_max_depth(self, traverser, mock_registry):
        """Test connection path traversal with max depth limit."""
        connections_map = {
            "start": {
                "related": [{"id": "middle", "annotation": "to middle"}]
            },
            "middle": {
                "related": [{"id": "end", "annotation": "to end"}]
            }
        }
        
        mock_registry.get_pipeline_connections.side_effect = lambda pid: connections_map.get(pid, {})
        
        paths = traverser.traverse_connection_path("start", ["related"], max_depth=1)
        
        # Should stop at depth 1
        assert all(len(path) <= 2 for path in paths)

    def test_traverse_connection_path_error(self, traverser, mock_registry):
        """Test connection path traversal with error."""
        mock_registry.get_pipeline_connections.side_effect = Exception("Registry error")
        
        paths = traverser.traverse_connection_path("start", ["related"])
        
        assert paths == [["start"]]

    def test_find_shortest_path_direct(self, traverser, mock_registry):
        """Test finding shortest path between directly connected pipelines."""
        mock_registry.get_pipeline_connections.return_value = {
            "related": [{"id": "target", "annotation": "direct connection"}]
        }
        
        path = traverser.find_shortest_path("source", "target")
        
        assert path == ["source", "target"]

    def test_find_shortest_path_same_node(self, traverser, mock_registry):
        """Test finding shortest path to same node."""
        path = traverser.find_shortest_path("source", "source")
        
        assert path == ["source"]

    def test_find_shortest_path_multi_hop(self, traverser, mock_registry):
        """Test finding shortest path through multiple hops."""
        connections_map = {
            "source": {
                "related": [{"id": "middle", "annotation": "to middle"}]
            },
            "middle": {
                "related": [{"id": "target", "annotation": "to target"}]
            }
        }
        
        mock_registry.get_pipeline_connections.side_effect = lambda pid: connections_map.get(pid, {})
        
        path = traverser.find_shortest_path("source", "target")
        
        assert path == ["source", "middle", "target"]

    def test_find_shortest_path_no_path(self, traverser, mock_registry):
        """Test finding shortest path when no path exists."""
        mock_registry.get_pipeline_connections.return_value = {}
        
        path = traverser.find_shortest_path("source", "target")
        
        assert path is None

    def test_find_shortest_path_error(self, traverser, mock_registry):
        """Test finding shortest path with error."""
        mock_registry.get_pipeline_connections.side_effect = Exception("Registry error")
        
        path = traverser.find_shortest_path("source", "target")
        
        assert path is None

    def test_get_connection_subgraph(self, traverser, mock_registry, sample_nodes):
        """Test getting connection subgraph."""
        connections_map = {
            "center": {
                "related": [{"id": "node1", "annotation": "to node1"}],
                "alternatives": [{"id": "node2", "annotation": "to node2"}]
            },
            "node1": {
                "related": [{"id": "node3", "annotation": "to node3"}]
            },
            "node2": {
                "related": []
            },
            "node3": {
                "related": []
            }
        }
        
        mock_registry.get_pipeline_connections.side_effect = lambda pid: connections_map.get(pid, {})
        mock_registry.get_pipeline_node.side_effect = lambda pid: sample_nodes.get(pid, {
            "title": f"Node {pid}",
            "zettelkasten_metadata": {"framework": "test", "complexity": "simple"}
        })
        
        subgraph = traverser.get_connection_subgraph("center", depth=2)
        
        assert subgraph["center_node"] == "center"
        assert subgraph["depth"] == 2
        assert "center" in subgraph["nodes"]
        assert "node1" in subgraph["nodes"]
        assert "node2" in subgraph["nodes"]
        assert subgraph["node_count"] > 0
        assert subgraph["edge_count"] > 0
        
        # Check edge structure
        edges = subgraph["edges"]
        assert any(edge["source"] == "center" and edge["target"] == "node1" for edge in edges)
        assert any(edge["source"] == "center" and edge["target"] == "node2" for edge in edges)

    def test_get_connection_subgraph_error(self, traverser, mock_registry):
        """Test getting connection subgraph with error."""
        mock_registry.get_pipeline_connections.side_effect = Exception("Registry error")
        
        subgraph = traverser.get_connection_subgraph("center")
        
        assert subgraph["center_node"] == "center"
        assert subgraph["nodes"] == {}
        assert subgraph["edges"] == []
        assert subgraph["node_count"] == 0
        assert subgraph["edge_count"] == 0

    def test_find_connection_clusters(self, traverser, mock_registry):
        """Test finding connection clusters."""
        # Mock pipeline list
        mock_registry.get_all_pipelines.return_value = ["p1", "p2", "p3", "p4", "p5"]
        
        # Mock connections to form clusters: {p1, p2, p3} and {p4} and {p5}
        connections_map = {
            "p1": {"related": [{"id": "p2", "annotation": "p1->p2"}]},
            "p2": {"related": [{"id": "p3", "annotation": "p2->p3"}]},
            "p3": {"related": []},
            "p4": {"related": []},
            "p5": {"related": []}
        }
        
        mock_registry.get_pipeline_connections.side_effect = lambda pid: connections_map.get(pid, {})
        
        clusters = traverser.find_connection_clusters()
        
        assert len(clusters) == 3
        # Largest cluster should be first
        assert len(clusters[0]) >= len(clusters[1])
        assert len(clusters[1]) >= len(clusters[2])
        
        # Check that all pipelines are in some cluster
        all_clustered = set()
        for cluster in clusters:
            all_clustered.update(cluster)
        assert all_clustered == {"p1", "p2", "p3", "p4", "p5"}

    def test_find_connection_clusters_error(self, traverser, mock_registry):
        """Test finding connection clusters with error."""
        mock_registry.get_all_pipelines.side_effect = Exception("Registry error")
        
        clusters = traverser.find_connection_clusters()
        
        assert clusters == []

    def test_get_bidirectional_connections(self, traverser, mock_registry):
        """Test getting bidirectional connections."""
        # Mock connections where p1 -> p2 and p2 -> p1
        connections_map = {
            "p1": {"related": [{"id": "p2", "annotation": "p1 to p2"}]},
            "p2": {"related": [{"id": "p1", "annotation": "p2 to p1"}]}
        }
        
        mock_registry.get_pipeline_connections.side_effect = lambda pid: connections_map.get(pid, {})
        
        bidirectional = traverser.get_bidirectional_connections("p1")
        
        assert len(bidirectional) == 1
        assert bidirectional[0].target_id == "p2"
        assert bidirectional[0].connection_type == "related"
        assert "Bidirectional" in bidirectional[0].annotation

    def test_get_bidirectional_connections_error(self, traverser, mock_registry):
        """Test getting bidirectional connections with error."""
        mock_registry.get_pipeline_connections.side_effect = Exception("Registry error")
        
        bidirectional = traverser.get_bidirectional_connections("p1")
        
        assert bidirectional == []

    def test_analyze_connection_patterns(self, traverser, mock_registry, sample_nodes):
        """Test analyzing connection patterns."""
        mock_registry.get_all_pipelines.return_value = ["p1", "p2", "p3"]
        
        connections_map = {
            "p1": {
                "related": [{"id": "p2", "annotation": "p1->p2"}],
                "alternatives": [{"id": "p3", "annotation": "p1->p3"}]
            },
            "p2": {"related": [{"id": "p3", "annotation": "p2->p3"}]},
            "p3": {"related": []}
        }
        
        mock_registry.get_pipeline_connections.side_effect = lambda pid: connections_map.get(pid, {})
        mock_registry.get_pipeline_node.side_effect = lambda pid: sample_nodes.get(pid, {
            "zettelkasten_metadata": {"framework": "test"}
        })
        
        analysis = traverser.analyze_connection_patterns()
        
        assert "total_pipelines" in analysis
        assert "total_connections" in analysis
        assert "connection_density" in analysis
        assert "connection_type_distribution" in analysis
        assert "hub_pipelines" in analysis
        assert "isolated_pipelines" in analysis
        assert "framework_connection_matrix" in analysis
        
        assert analysis["total_pipelines"] == 3
        assert analysis["total_connections"] == 3  # p1: 2 connections, p2: 1 connection, p3: 0 connections
        assert "related" in analysis["connection_type_distribution"]
        assert "alternatives" in analysis["connection_type_distribution"]

    def test_analyze_connection_patterns_error(self, traverser, mock_registry):
        """Test analyzing connection patterns with error."""
        mock_registry.get_all_pipelines.side_effect = Exception("Registry error")
        
        analysis = traverser.analyze_connection_patterns()
        
        assert "error" in analysis

    def test_suggest_missing_connections(self, traverser, mock_registry, sample_nodes):
        """Test suggesting missing connections."""
        mock_registry.get_pipeline_node.side_effect = lambda pid: sample_nodes.get(pid)
        mock_registry.get_pipeline_connections.return_value = {"related": [], "alternatives": [], "used_in": []}
        mock_registry.get_all_pipelines.return_value = ["test_pipeline", "alt_pipeline", "related_pipeline"]
        
        suggestions = traverser.suggest_missing_connections("test_pipeline")
        
        assert len(suggestions) > 0
        # Should suggest alt_pipeline due to same framework and complexity
        alt_suggestion = next((s for s in suggestions if s["target_id"] == "alt_pipeline"), None)
        assert alt_suggestion is not None
        assert alt_suggestion["similarity_score"] > 0.4
        assert "Same framework" in " ".join(alt_suggestion["reasons"])

    def test_suggest_missing_connections_no_node(self, traverser, mock_registry):
        """Test suggesting missing connections when node doesn't exist."""
        mock_registry.get_pipeline_node.return_value = None
        
        suggestions = traverser.suggest_missing_connections("nonexistent")
        
        assert suggestions == []

    def test_suggest_missing_connections_error(self, traverser, mock_registry):
        """Test suggesting missing connections with error."""
        mock_registry.get_pipeline_node.side_effect = Exception("Registry error")
        
        suggestions = traverser.suggest_missing_connections("test_pipeline")
        
        assert suggestions == []

    def test_clear_cache(self, traverser):
        """Test clearing connection cache."""
        # Set up cache
        traverser._connection_cache = {"test": "data"}
        traverser._cache_valid = True
        
        # Clear cache
        traverser.clear_cache()
        
        assert traverser._connection_cache == {}
        assert not traverser._cache_valid

    def test_pipeline_connection_model(self):
        """Test PipelineConnection model."""
        connection = PipelineConnection(
            target_id="target",
            connection_type="related",
            annotation="Test connection",
            source_id="source"
        )
        
        assert connection.target_id == "target"
        assert connection.connection_type == "related"
        assert connection.annotation == "Test connection"
        assert connection.source_id == "source"

    def test_pipeline_connection_model_optional_source(self):
        """Test PipelineConnection model with optional source."""
        connection = PipelineConnection(
            target_id="target",
            connection_type="related",
            annotation="Test connection"
        )
        
        assert connection.target_id == "target"
        assert connection.connection_type == "related"
        assert connection.annotation == "Test connection"
        assert connection.source_id is None

    def test_complex_traversal_scenario(self, traverser, mock_registry):
        """Test complex traversal scenario with cycles and multiple paths."""
        # Create a more complex connection graph
        connections_map = {
            "A": {"related": [{"id": "B", "annotation": "A->B"}, {"id": "C", "annotation": "A->C"}]},
            "B": {"related": [{"id": "D", "annotation": "B->D"}]},
            "C": {"related": [{"id": "D", "annotation": "C->D"}, {"id": "A", "annotation": "C->A"}]},  # Cycle
            "D": {"related": []}
        }
        
        mock_registry.get_pipeline_connections.side_effect = lambda pid: connections_map.get(pid, {})
        
        # Test path traversal
        paths = traverser.traverse_connection_path("A", ["related"], max_depth=3)
        
        assert len(paths) > 0
        # Should handle cycles properly
        assert all(len(path) <= 4 for path in paths)  # max_depth + 1
        
        # Test shortest path
        shortest = traverser.find_shortest_path("A", "D")
        assert shortest is not None
        assert len(shortest) == 3  # A -> B -> D or A -> C -> D

    def test_empty_registry_scenarios(self, traverser, mock_registry):
        """Test behavior with empty registry."""
        mock_registry.get_all_pipelines.return_value = []
        mock_registry.get_pipeline_connections.return_value = {}
        mock_registry.get_pipeline_node.return_value = None
        
        # Test various methods with empty registry
        assert traverser.get_alternatives("test") == []
        assert traverser.get_all_connections("test") == {}
        assert traverser.find_connection_clusters() == []
        assert traverser.analyze_connection_patterns()["total_pipelines"] == 0
        assert traverser.suggest_missing_connections("test") == []
        
        subgraph = traverser.get_connection_subgraph("test")
        assert subgraph["node_count"] == 0
        assert subgraph["edge_count"] == 0

"""
Unit tests for ConnectionTraverser class.

Tests the connection traversal utilities that navigate pipeline connections
following Zettelkasten principles.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from cursus.pipeline_catalog.utils.connection_traverser import (
    ConnectionTraverser, PipelineConnection
)
from cursus.pipeline_catalog.utils.catalog_registry import CatalogRegistry

class TestPipelineConnection:
    """Test suite for PipelineConnection model."""
    
    def test_pipeline_connection_creation(self):
        """Test creating PipelineConnection instance."""
        connection = PipelineConnection(
            target_id="target_pipeline",
            connection_type="related",
            annotation="Test connection",
            source_id="source_pipeline"
        )
        
        assert connection.target_id == "target_pipeline"
        assert connection.connection_type == "related"
        assert connection.annotation == "Test connection"
        assert connection.source_id == "source_pipeline"
    
    def test_pipeline_connection_without_source(self):
        """Test creating PipelineConnection without source_id."""
        connection = PipelineConnection(
            target_id="target_pipeline",
            connection_type="alternatives",
            annotation="Alternative approach"
        )
        
        assert connection.target_id == "target_pipeline"
        assert connection.connection_type == "alternatives"
        assert connection.annotation == "Alternative approach"
        assert connection.source_id is None

class TestConnectionTraverser:
    """Test suite for ConnectionTraverser class."""
    
    @pytest.fixture
    def mock_registry(self):
        """Create mock CatalogRegistry for testing."""
        registry = Mock(spec=CatalogRegistry)
        
        # Mock pipeline connections data
        connections_data = {
            "pipeline_a": {
                "alternatives": [
                    {"id": "pipeline_b", "annotation": "Alternative framework"}
                ],
                "related": [
                    {"id": "pipeline_c", "annotation": "Related functionality"}
                ],
                "used_in": [
                    {"id": "pipeline_d", "annotation": "Used in composition"}
                ]
            },
            "pipeline_b": {
                "alternatives": [
                    {"id": "pipeline_a", "annotation": "Alternative approach"}
                ],
                "related": [],
                "used_in": []
            },
            "pipeline_c": {
                "alternatives": [],
                "related": [
                    {"id": "pipeline_a", "annotation": "Related to A"}
                ],
                "used_in": []
            },
            "pipeline_d": {
                "alternatives": [],
                "related": [],
                "used_in": []
            }
        }
        
        # Mock pipeline nodes data
        nodes_data = {
            "pipeline_a": {
                "id": "pipeline_a",
                "title": "Pipeline A",
                "zettelkasten_metadata": {
                    "framework": "xgboost",
                    "complexity": "simple"
                }
            },
            "pipeline_b": {
                "id": "pipeline_b",
                "title": "Pipeline B",
                "zettelkasten_metadata": {
                    "framework": "pytorch",
                    "complexity": "simple"
                }
            },
            "pipeline_c": {
                "id": "pipeline_c",
                "title": "Pipeline C",
                "zettelkasten_metadata": {
                    "framework": "xgboost",
                    "complexity": "standard"
                }
            },
            "pipeline_d": {
                "id": "pipeline_d",
                "title": "Pipeline D",
                "zettelkasten_metadata": {
                    "framework": "generic",
                    "complexity": "advanced"
                }
            }
        }
        
        def mock_get_pipeline_connections(pipeline_id):
            return connections_data.get(pipeline_id, {"alternatives": [], "related": [], "used_in": []})
        
        def mock_get_pipeline_node(pipeline_id):
            return nodes_data.get(pipeline_id)
        
        def mock_get_all_pipelines():
            return list(nodes_data.keys())
        
        registry.get_pipeline_connections.side_effect = mock_get_pipeline_connections
        registry.get_pipeline_node.side_effect = mock_get_pipeline_node
        registry.get_all_pipelines.side_effect = mock_get_all_pipelines
        
        return registry
    
    @pytest.fixture
    def traverser(self, mock_registry):
        """Create ConnectionTraverser instance with mock registry."""
        return ConnectionTraverser(mock_registry)
    
    def test_init(self, mock_registry):
        """Test ConnectionTraverser initialization."""
        traverser = ConnectionTraverser(mock_registry)
        
        assert traverser.registry == mock_registry
        assert traverser._connection_cache == {}
        assert not traverser._cache_valid
    
    def test_get_alternatives(self, traverser):
        """Test retrieving alternative connections."""
        alternatives = traverser.get_alternatives("pipeline_a")
        
        assert len(alternatives) == 1
        assert alternatives[0].target_id == "pipeline_b"
        assert alternatives[0].connection_type == "alternatives"
        assert alternatives[0].annotation == "Alternative framework"
        assert alternatives[0].source_id == "pipeline_a"
    
    def test_get_related(self, traverser):
        """Test retrieving related connections."""
        related = traverser.get_related("pipeline_a")
        
        assert len(related) == 1
        assert related[0].target_id == "pipeline_c"
        assert related[0].connection_type == "related"
        assert related[0].annotation == "Related functionality"
        assert related[0].source_id == "pipeline_a"
    
    def test_get_compositions(self, traverser):
        """Test retrieving composition connections."""
        compositions = traverser.get_compositions("pipeline_a")
        
        assert len(compositions) == 1
        assert compositions[0].target_id == "pipeline_d"
        assert compositions[0].connection_type == "used_in"
        assert compositions[0].annotation == "Used in composition"
        assert compositions[0].source_id == "pipeline_a"
    
    def test_get_all_connections(self, traverser):
        """Test retrieving all connections for a pipeline."""
        all_connections = traverser.get_all_connections("pipeline_a")
        
        assert "alternatives" in all_connections
        assert "related" in all_connections
        assert "used_in" in all_connections
        
        assert len(all_connections["alternatives"]) == 1
        assert len(all_connections["related"]) == 1
        assert len(all_connections["used_in"]) == 1
        
        # Check specific connection details
        alt_conn = all_connections["alternatives"][0]
        assert alt_conn.target_id == "pipeline_b"
        assert alt_conn.connection_type == "alternatives"
    
    def test_get_connections_by_type_empty(self, traverser):
        """Test retrieving connections when none exist."""
        related = traverser.get_related("pipeline_b")
        assert len(related) == 0
    
    def test_traverse_connection_path_simple(self, traverser):
        """Test traversing connection paths with simple case."""
        paths = traverser.traverse_connection_path(
            "pipeline_a", 
            ["alternatives"], 
            max_depth=2
        )
        
        assert len(paths) > 0
        # Should find path from pipeline_a to pipeline_b via alternatives
        assert any("pipeline_b" in path for path in paths)
    
    def test_traverse_connection_path_max_depth(self, traverser):
        """Test traversing connection paths respects max depth."""
        paths = traverser.traverse_connection_path(
            "pipeline_a", 
            ["alternatives", "related"], 
            max_depth=1
        )
        
        # With max_depth=1, should only include direct connections
        for path in paths:
            assert len(path) <= 2  # Start node + 1 connection
    
    def test_traverse_connection_path_zero_depth(self, traverser):
        """Test traversing connection paths with zero depth."""
        paths = traverser.traverse_connection_path(
            "pipeline_a", 
            ["alternatives"], 
            max_depth=0
        )
        
        assert len(paths) == 1
        assert paths[0] == ["pipeline_a"]
    
    def test_find_shortest_path_direct(self, traverser):
        """Test finding shortest path between directly connected pipelines."""
        path = traverser.find_shortest_path("pipeline_a", "pipeline_b")
        
        assert path is not None
        assert path == ["pipeline_a", "pipeline_b"]
    
    def test_find_shortest_path_same_node(self, traverser):
        """Test finding shortest path from node to itself."""
        path = traverser.find_shortest_path("pipeline_a", "pipeline_a")
        
        assert path == ["pipeline_a"]
    
    def test_find_shortest_path_no_path(self, traverser):
        """Test finding shortest path when no path exists."""
        # Create isolated node
        traverser.registry.get_pipeline_connections.return_value = {
            "alternatives": [], "related": [], "used_in": []
        }
        
        path = traverser.find_shortest_path("pipeline_a", "isolated_pipeline")
        assert path is None
    
    def test_get_connection_subgraph(self, traverser):
        """Test extracting connection subgraph."""
        subgraph = traverser.get_connection_subgraph("pipeline_a", depth=1)
        
        assert subgraph["center_node"] == "pipeline_a"
        assert subgraph["depth"] == 1
        assert "nodes" in subgraph
        assert "edges" in subgraph
        
        # Should include pipeline_a and its direct connections
        assert "pipeline_a" in subgraph["nodes"]
        assert subgraph["node_count"] > 0
        assert subgraph["edge_count"] > 0
    
    def test_get_connection_subgraph_depth_zero(self, traverser):
        """Test extracting subgraph with depth zero."""
        subgraph = traverser.get_connection_subgraph("pipeline_a", depth=0)
        
        assert subgraph["center_node"] == "pipeline_a"
        assert subgraph["depth"] == 0
        assert "pipeline_a" in subgraph["nodes"]
        assert subgraph["node_count"] == 1
    
    def test_find_connection_clusters(self, traverser):
        """Test finding connection clusters."""
        clusters = traverser.find_connection_clusters()
        
        assert len(clusters) > 0
        # Should find at least one cluster containing connected pipelines
        assert any(len(cluster) > 1 for cluster in clusters)
        
        # Clusters should be sorted by size (largest first)
        if len(clusters) > 1:
            assert len(clusters[0]) >= len(clusters[1])
    
    def test_get_bidirectional_connections(self, traverser):
        """Test finding bidirectional connections."""
        bidirectional = traverser.get_bidirectional_connections("pipeline_a")
        
        # pipeline_a and pipeline_b have bidirectional alternatives connections
        assert len(bidirectional) > 0
        
        # Check if pipeline_b is in bidirectional connections
        target_ids = [conn.target_id for conn in bidirectional]
        assert "pipeline_b" in target_ids
    
    def test_analyze_connection_patterns(self, traverser):
        """Test analyzing connection patterns across registry."""
        analysis = traverser.analyze_connection_patterns()
        
        assert "total_pipelines" in analysis
        assert "total_connections" in analysis
        assert "connection_density" in analysis
        assert "connection_type_distribution" in analysis
        assert "hub_pipelines" in analysis
        assert "isolated_pipelines" in analysis
        
        assert analysis["total_pipelines"] == 4
        assert analysis["total_connections"] > 0
        assert 0 <= analysis["connection_density"] <= 1
    
    def test_suggest_missing_connections(self, traverser):
        """Test suggesting missing connections based on similarity."""
        suggestions = traverser.suggest_missing_connections("pipeline_a")
        
        assert isinstance(suggestions, list)
        # Each suggestion should have required fields
        for suggestion in suggestions:
            assert "target_id" in suggestion
            assert "target_title" in suggestion
            assert "suggested_type" in suggestion
            assert "similarity_score" in suggestion
            assert "reasons" in suggestion
    
    def test_suggest_missing_connections_no_node(self, traverser):
        """Test suggesting connections for non-existent pipeline."""
        traverser.registry.get_pipeline_node.return_value = None
        
        suggestions = traverser.suggest_missing_connections("nonexistent_pipeline")
        assert suggestions == []
    
    def test_clear_cache(self, traverser):
        """Test clearing connection cache."""
        # Populate cache
        traverser._connection_cache = {"test": "data"}
        traverser._cache_valid = True
        
        # Clear cache
        traverser.clear_cache()
        
        assert traverser._connection_cache == {}
        assert not traverser._cache_valid
    
    @patch('src.cursus.pipeline_catalog.utils.connection_traverser.logger')
    def test_error_handling(self, mock_logger, traverser):
        """Test error handling and logging."""
        # Mock registry to raise exception
        traverser.registry.get_pipeline_connections.side_effect = Exception("Connection error")
        
        # Test method that should handle the error gracefully
        alternatives = traverser.get_alternatives("pipeline_a")
        
        assert alternatives == []
        mock_logger.error.assert_called()
    
    def test_connection_patterns_framework_analysis(self, traverser):
        """Test framework connection pattern analysis."""
        analysis = traverser.analyze_connection_patterns()
        
        assert "framework_connection_matrix" in analysis
        framework_matrix = analysis["framework_connection_matrix"]
        
        # Should have framework connections
        assert isinstance(framework_matrix, dict)
    
    def test_connection_distribution_analysis(self, traverser):
        """Test connection distribution analysis."""
        analysis = traverser.analyze_connection_patterns()
        
        assert "connection_distribution" in analysis
        distribution = analysis["connection_distribution"]
        
        # Should categorize pipelines by connection count
        assert "0_connections" in distribution
        assert "1-3_connections" in distribution
        assert "4-6_connections" in distribution
        assert "7+_connections" in distribution
        
        # Total should equal number of pipelines
        total_categorized = sum(distribution.values())
        assert total_categorized == analysis["total_pipelines"]

class TestConnectionTraverserIntegration:
    """Integration tests for ConnectionTraverser with realistic scenarios."""
    
    def test_complex_traversal_scenario(self):
        """Test complex connection traversal scenario."""
        # Create mock registry with complex connection graph
        registry = Mock(spec=CatalogRegistry)
        
        # Complex connection graph: A -> B -> C -> D, with cross-connections
        connections = {
            "A": {
                "alternatives": [{"id": "B", "annotation": "Alt to B"}],
                "related": [{"id": "C", "annotation": "Related to C"}],
                "used_in": []
            },
            "B": {
                "alternatives": [{"id": "A", "annotation": "Alt to A"}],
                "related": [{"id": "D", "annotation": "Related to D"}],
                "used_in": [{"id": "C", "annotation": "Used in C"}]
            },
            "C": {
                "alternatives": [],
                "related": [{"id": "A", "annotation": "Related to A"}],
                "used_in": [{"id": "D", "annotation": "Used in D"}]
            },
            "D": {
                "alternatives": [],
                "related": [{"id": "B", "annotation": "Related to B"}],
                "used_in": []
            }
        }
        
        nodes = {
            "A": {"id": "A", "title": "Pipeline A", "zettelkasten_metadata": {"framework": "xgboost", "complexity": "simple"}},
            "B": {"id": "B", "title": "Pipeline B", "zettelkasten_metadata": {"framework": "pytorch", "complexity": "simple"}},
            "C": {"id": "C", "title": "Pipeline C", "zettelkasten_metadata": {"framework": "xgboost", "complexity": "standard"}},
            "D": {"id": "D", "title": "Pipeline D", "zettelkasten_metadata": {"framework": "generic", "complexity": "advanced"}}
        }
        
        registry.get_pipeline_connections.side_effect = lambda pid: connections.get(pid, {"alternatives": [], "related": [], "used_in": []})
        registry.get_pipeline_node.side_effect = lambda pid: nodes.get(pid)
        registry.get_all_pipelines.return_value = list(nodes.keys())
        
        traverser = ConnectionTraverser(registry)
        
        # Test finding path from A to D
        path = traverser.find_shortest_path("A", "D")
        assert path is not None
        assert len(path) >= 2
        assert path[0] == "A"
        assert path[-1] == "D"
        
        # Test subgraph extraction
        subgraph = traverser.get_connection_subgraph("A", depth=2)
        assert subgraph["node_count"] >= 3  # Should include A and connected nodes
        
        # Test cluster detection
        clusters = traverser.find_connection_clusters()
        assert len(clusters) == 1  # All nodes should be in one connected cluster
        assert len(clusters[0]) == 4  # All 4 nodes
    
    def test_isolated_nodes_scenario(self):
        """Test scenario with isolated nodes."""
        registry = Mock(spec=CatalogRegistry)
        
        # Two separate clusters + one isolated node
        connections = {
            "A": {"alternatives": [{"id": "B", "annotation": "Alt to B"}], "related": [], "used_in": []},
            "B": {"alternatives": [{"id": "A", "annotation": "Alt to A"}], "related": [], "used_in": []},
            "C": {"alternatives": [{"id": "D", "annotation": "Alt to D"}], "related": [], "used_in": []},
            "D": {"alternatives": [{"id": "C", "annotation": "Alt to C"}], "related": [], "used_in": []},
            "E": {"alternatives": [], "related": [], "used_in": []}  # Isolated
        }
        
        nodes = {
            "A": {"id": "A", "title": "Pipeline A"},
            "B": {"id": "B", "title": "Pipeline B"},
            "C": {"id": "C", "title": "Pipeline C"},
            "D": {"id": "D", "title": "Pipeline D"},
            "E": {"id": "E", "title": "Pipeline E"}
        }
        
        registry.get_pipeline_connections.side_effect = lambda pid: connections.get(pid, {"alternatives": [], "related": [], "used_in": []})
        registry.get_pipeline_node.side_effect = lambda pid: nodes.get(pid)
        registry.get_all_pipelines.return_value = list(nodes.keys())
        
        traverser = ConnectionTraverser(registry)
        
        # Test cluster detection
        clusters = traverser.find_connection_clusters()
        assert len(clusters) == 3  # Two clusters of 2 nodes each + 1 isolated node
        
        cluster_sizes = sorted([len(cluster) for cluster in clusters], reverse=True)
        assert cluster_sizes == [2, 2, 1]
        
        # Test connection analysis
        analysis = traverser.analyze_connection_patterns()
        assert len(analysis["isolated_pipelines"]) == 1
        assert "E" in analysis["isolated_pipelines"]

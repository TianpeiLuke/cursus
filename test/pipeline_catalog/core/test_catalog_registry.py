"""
Unit tests for CatalogRegistry class.

Tests the Zettelkasten-inspired registry management functionality including
CRUD operations, connection management, and integrity validation.
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, Any

from cursus.pipeline_catalog.core.catalog_registry import CatalogRegistry
from cursus.pipeline_catalog.shared_dags.enhanced_metadata import EnhancedDAGMetadata, ZettelkastenMetadata


class TestCatalogRegistry:
    """Test suite for CatalogRegistry class."""

    @pytest.fixture
    def mock_sync(self):
        """Create mock DAGMetadataRegistrySync."""
        with patch('cursus.pipeline_catalog.core.catalog_registry.DAGMetadataRegistrySync') as mock_sync_class:
            mock_sync = Mock()
            mock_sync_class.return_value = mock_sync
            yield mock_sync

    @pytest.fixture
    def mock_registry_data(self):
        """Create mock registry data with proper structure."""
        return {
            "version": "1.0.0",
            "nodes": {
                "test_pipeline": {
                    "id": "test_pipeline",
                    "title": "Test Pipeline",
                    "description": "A test pipeline",
                    "atomic_properties": {
                        "single_responsibility": "Test functionality",
                        "independence_level": "high",
                        "node_count": 3,
                        "edge_count": 2
                    },
                    "zettelkasten_metadata": {
                        "framework": "xgboost",
                        "complexity": "simple",
                        "use_case": "Test functionality",
                        "features": ["feature1"],
                        "mods_compatible": True
                    },
                    "multi_dimensional_tags": {
                        "framework_tags": ["xgboost"],
                        "task_tags": ["training"],
                        "complexity_tags": ["simple"]
                    },
                    "connections": {
                        "alternatives": [],
                        "related": [],
                        "used_in": []
                    },
                    "source_file": "test.py",
                    "migration_source": "test",
                    "created_date": "2025-09-18",
                    "priority": "medium"
                }
            },
            "tag_index": {
                "framework_tags": {
                    "xgboost": ["test_pipeline"]
                },
                "task_tags": {
                    "training": ["test_pipeline"]
                },
                "complexity_tags": {
                    "simple": ["test_pipeline"]
                }
            },
            "metadata": {
                "total_pipelines": 1,
                "last_updated": "2025-09-18T09:00:00Z",
                "frameworks": ["xgboost"],
                "complexity_levels": ["simple"],
                "task_types": ["training"]
            }
        }

    @pytest.fixture
    def registry(self, mock_sync):
        """Create CatalogRegistry instance with mocked sync."""
        return CatalogRegistry(registry_path="test_registry.json")

    def test_init_default_path(self, mock_sync):
        """Test initialization with default registry path."""
        registry = CatalogRegistry()
        assert registry.registry_path == "catalog_index.json"
        assert not registry._cache_valid

    def test_init_custom_path(self, mock_sync):
        """Test initialization with custom registry path."""
        registry = CatalogRegistry(registry_path="custom_registry.json")
        assert registry.registry_path == "custom_registry.json"
        assert not registry._cache_valid

    def test_load_registry_success(self, registry, mock_sync, mock_registry_data):
        """Test successful registry loading."""
        mock_sync.load_registry.return_value = mock_registry_data
        
        result = registry.load_registry()
        
        assert result == mock_registry_data
        assert registry._cache_valid
        assert registry._cache == mock_registry_data
        mock_sync.load_registry.assert_called_once()

    def test_load_registry_failure(self, registry, mock_sync):
        """Test registry loading failure."""
        mock_sync.load_registry.side_effect = Exception("Load failed")
        
        with pytest.raises(Exception, match="Load failed"):
            registry.load_registry()

    def test_save_registry(self, registry, mock_sync):
        """Test saving registry data."""
        test_data = {"test": "data"}
        
        registry.save_registry(test_data)
        
        mock_sync._save_registry.assert_called_once_with(test_data)
        assert registry._cache == test_data
        assert registry._cache_valid

    def test_get_pipeline_node_exists(self, registry, mock_sync, mock_registry_data):
        """Test getting existing pipeline node."""
        registry._cache = mock_registry_data
        registry._cache_valid = True
        
        node = registry.get_pipeline_node("test_pipeline")
        
        assert node is not None
        assert node["id"] == "test_pipeline"
        assert node["title"] == "Test Pipeline"
        assert node["zettelkasten_metadata"]["framework"] == "xgboost"

    def test_get_pipeline_node_not_exists(self, registry, mock_sync, mock_registry_data):
        """Test getting non-existent pipeline node."""
        registry._cache = mock_registry_data
        registry._cache_valid = True
        
        node = registry.get_pipeline_node("nonexistent_pipeline")
        assert node is None

    def test_get_pipeline_node_load_error(self, registry, mock_sync):
        """Test getting pipeline node with load error."""
        mock_sync.load_registry.side_effect = Exception("Load error")
        
        node = registry.get_pipeline_node("test_pipeline")
        assert node is None

    def test_get_all_pipelines(self, registry, mock_sync, mock_registry_data):
        """Test getting all pipeline IDs."""
        registry._cache = mock_registry_data
        registry._cache_valid = True
        
        pipelines = registry.get_all_pipelines()
        
        assert pipelines == ["test_pipeline"]

    def test_get_all_pipelines_empty_registry(self, registry, mock_sync):
        """Test getting pipelines from empty registry."""
        registry._cache = {"nodes": {}}
        registry._cache_valid = True
        
        pipelines = registry.get_all_pipelines()
        
        assert pipelines == []

    def test_get_all_pipelines_load_error(self, registry, mock_sync):
        """Test getting all pipelines with load error."""
        mock_sync.load_registry.side_effect = Exception("Load error")
        
        pipelines = registry.get_all_pipelines()
        assert pipelines == []

    def test_add_or_update_enhanced_node_success(self, registry, mock_sync):
        """Test successful enhanced node addition."""
        mock_metadata = Mock(spec=EnhancedDAGMetadata)
        mock_metadata.zettelkasten_metadata = Mock()
        mock_metadata.zettelkasten_metadata.source_file = "test.py"
        mock_metadata.zettelkasten_metadata.atomic_id = "test_id"
        
        mock_sync.sync_metadata_to_registry.return_value = None
        
        result = registry.add_or_update_enhanced_node(mock_metadata)
        
        assert result is True
        mock_sync.sync_metadata_to_registry.assert_called_once_with(
            dag_metadata=mock_metadata,
            pipeline_file_path="test.py"
        )

    def test_add_or_update_enhanced_node_failure(self, registry, mock_sync):
        """Test failed enhanced node addition."""
        mock_metadata = Mock(spec=EnhancedDAGMetadata)
        mock_metadata.zettelkasten_metadata = Mock()
        mock_metadata.zettelkasten_metadata.source_file = "test.py"
        
        mock_sync.sync_metadata_to_registry.side_effect = Exception("Sync error")
        
        result = registry.add_or_update_enhanced_node(mock_metadata)
        
        assert result is False

    def test_add_pipeline_node_success(self, registry, mock_sync, mock_registry_data):
        """Test successful pipeline node addition."""
        registry._cache = mock_registry_data.copy()
        registry._cache_valid = True
        
        new_node = {
            "id": "new_pipeline",
            "title": "New Pipeline",
            "description": "A new test pipeline",
            "atomic_properties": {
                "single_responsibility": "New functionality",
                "independence_level": "medium",
                "node_count": 2,
                "edge_count": 1
            },
            "zettelkasten_metadata": {
                "framework": "pytorch",
                "complexity": "standard"
            },
            "multi_dimensional_tags": {
                "framework_tags": ["pytorch"],
                "task_tags": ["evaluation"],
                "complexity_tags": ["standard"]
            }
        }
        
        # Mock the helper methods
        registry._update_tag_index_for_pipeline = Mock()
        registry._update_registry_metadata = Mock()
        
        result = registry.add_pipeline_node("new_pipeline", new_node)
        
        assert result is True
        assert "new_pipeline" in registry._cache["nodes"]
        assert registry._cache["nodes"]["new_pipeline"] == new_node
        registry._update_tag_index_for_pipeline.assert_called_once()
        registry._update_registry_metadata.assert_called_once()
        mock_sync._save_registry.assert_called_once()

    def test_add_pipeline_node_missing_fields(self, registry, mock_sync, mock_registry_data):
        """Test adding pipeline node with missing required fields."""
        registry._cache = mock_registry_data.copy()
        registry._cache_valid = True
        
        incomplete_node = {
            "id": "incomplete_pipeline",
            "title": "Incomplete Pipeline"
            # Missing required fields
        }
        
        result = registry.add_pipeline_node("incomplete_pipeline", incomplete_node)
        
        assert result is False

    def test_remove_pipeline_node_success(self, registry, mock_sync, mock_registry_data):
        """Test successful pipeline node removal."""
        registry._cache = mock_registry_data.copy()
        registry._cache_valid = True
        
        # Mock the helper methods
        registry._remove_from_tag_index = Mock()
        registry._remove_connections_to_pipeline = Mock()
        registry._update_registry_metadata = Mock()
        
        result = registry.remove_pipeline_node("test_pipeline")
        
        assert result is True
        assert "test_pipeline" not in registry._cache["nodes"]
        registry._remove_from_tag_index.assert_called_once_with("test_pipeline")
        registry._remove_connections_to_pipeline.assert_called_once_with("test_pipeline")
        registry._update_registry_metadata.assert_called_once()
        mock_sync._save_registry.assert_called_once()

    def test_remove_pipeline_node_not_exists(self, registry, mock_sync, mock_registry_data):
        """Test removing non-existent pipeline node."""
        registry._cache = mock_registry_data.copy()
        registry._cache_valid = True
        
        result = registry.remove_pipeline_node("nonexistent_pipeline")
        
        assert result is False

    def test_update_pipeline_node_success(self, registry, mock_sync, mock_registry_data):
        """Test successful pipeline node update."""
        registry._cache = mock_registry_data.copy()
        registry._cache_valid = True
        
        updated_node = {
            "id": "test_pipeline",
            "title": "Updated Test Pipeline",
            "description": "An updated test pipeline",
            "atomic_properties": {
                "single_responsibility": "Updated functionality",
                "independence_level": "high",
                "node_count": 4,
                "edge_count": 3
            },
            "zettelkasten_metadata": {
                "framework": "pytorch",
                "complexity": "advanced"
            },
            "multi_dimensional_tags": {
                "framework_tags": ["pytorch"],
                "task_tags": ["training"],
                "complexity_tags": ["advanced"]
            }
        }
        
        # Mock the helper methods
        registry._update_tag_index_for_pipeline = Mock()
        registry._update_registry_metadata = Mock()
        
        result = registry.update_pipeline_node("test_pipeline", updated_node)
        
        assert result is True
        assert registry._cache["nodes"]["test_pipeline"] == updated_node
        registry._update_tag_index_for_pipeline.assert_called_once()
        registry._update_registry_metadata.assert_called_once()
        mock_sync._save_registry.assert_called_once()

    def test_update_pipeline_node_not_exists(self, registry, mock_sync, mock_registry_data):
        """Test updating non-existent pipeline node."""
        registry._cache = mock_registry_data.copy()
        registry._cache_valid = True
        
        node_data = {"id": "nonexistent", "title": "Test"}
        
        result = registry.update_pipeline_node("nonexistent_pipeline", node_data)
        
        assert result is False

    def test_get_pipelines_by_framework(self, registry, mock_sync, mock_registry_data):
        """Test getting pipelines by framework."""
        registry._cache = mock_registry_data.copy()
        registry._cache_valid = True
        
        pipelines = registry.get_pipelines_by_framework("xgboost")
        
        assert pipelines == ["test_pipeline"]
        
        # Test non-existent framework
        empty_result = registry.get_pipelines_by_framework("nonexistent")
        assert empty_result == []

    def test_get_pipelines_by_complexity(self, registry, mock_sync, mock_registry_data):
        """Test getting pipelines by complexity."""
        registry._cache = mock_registry_data.copy()
        registry._cache_valid = True
        
        pipelines = registry.get_pipelines_by_complexity("simple")
        
        assert pipelines == ["test_pipeline"]
        
        # Test non-existent complexity
        empty_result = registry.get_pipelines_by_complexity("nonexistent")
        assert empty_result == []

    def test_get_pipeline_connections(self, registry, mock_sync, mock_registry_data):
        """Test getting pipeline connections."""
        registry._cache = mock_registry_data.copy()
        registry._cache_valid = True
        
        connections = registry.get_pipeline_connections("test_pipeline")
        
        expected = {
            "alternatives": [],
            "related": [],
            "used_in": []
        }
        assert connections == expected

    def test_get_pipeline_connections_not_exists(self, registry, mock_sync):
        """Test getting connections for non-existent pipeline."""
        registry._cache = {"nodes": {}}
        registry._cache_valid = True
        
        connections = registry.get_pipeline_connections("nonexistent_pipeline")
        
        assert connections == {}

    def test_add_connection_success(self, registry, mock_sync, mock_registry_data):
        """Test successful connection addition."""
        # Add a target pipeline to the registry
        mock_registry_data["nodes"]["target_pipeline"] = {
            "id": "target_pipeline",
            "title": "Target Pipeline",
            "connections": {"alternatives": [], "related": [], "used_in": []}
        }
        
        registry._cache = mock_registry_data.copy()
        registry._cache_valid = True
        
        # Mock the helper methods
        registry._update_registry_metadata = Mock()
        
        result = registry.add_connection(
            "test_pipeline", 
            "target_pipeline", 
            "related", 
            "Test connection"
        )
        
        assert result is True
        
        # Verify connection was added
        connections = registry._cache["nodes"]["test_pipeline"]["connections"]["related"]
        assert len(connections) == 1
        assert connections[0]["id"] == "target_pipeline"
        assert connections[0]["annotation"] == "Test connection"
        
        registry._update_registry_metadata.assert_called_once()
        mock_sync._save_registry.assert_called_once()

    def test_add_connection_invalid_type(self, registry, mock_sync, mock_registry_data):
        """Test adding connection with invalid type."""
        registry._cache = mock_registry_data.copy()
        registry._cache_valid = True
        
        result = registry.add_connection(
            "test_pipeline",
            "target_pipeline", 
            "invalid_type",
            "Test connection"
        )
        
        assert result is False

    def test_add_connection_source_not_exists(self, registry, mock_sync, mock_registry_data):
        """Test adding connection with non-existent source."""
        registry._cache = mock_registry_data.copy()
        registry._cache_valid = True
        
        result = registry.add_connection(
            "nonexistent_source",
            "test_pipeline",
            "related",
            "Test connection"
        )
        
        assert result is False

    def test_add_connection_target_not_exists(self, registry, mock_sync, mock_registry_data):
        """Test adding connection with non-existent target."""
        registry._cache = mock_registry_data.copy()
        registry._cache_valid = True
        
        result = registry.add_connection(
            "test_pipeline",
            "nonexistent_target",
            "related",
            "Test connection"
        )
        
        assert result is False

    def test_remove_connection_success(self, registry, mock_sync, mock_registry_data):
        """Test successful connection removal."""
        # Add a connection first
        mock_registry_data["nodes"]["test_pipeline"]["connections"]["related"] = [
            {"id": "target_pipeline", "annotation": "Test connection"}
        ]
        
        registry._cache = mock_registry_data.copy()
        registry._cache_valid = True
        
        # Mock the helper methods
        registry._update_registry_metadata = Mock()
        
        result = registry.remove_connection("test_pipeline", "target_pipeline", "related")
        
        assert result is True
        
        # Verify connection was removed
        connections = registry._cache["nodes"]["test_pipeline"]["connections"]["related"]
        assert len(connections) == 0
        
        registry._update_registry_metadata.assert_called_once()
        mock_sync._save_registry.assert_called_once()

    def test_remove_connection_not_exists(self, registry, mock_sync, mock_registry_data):
        """Test removing non-existent connection."""
        registry._cache = mock_registry_data.copy()
        registry._cache_valid = True
        
        result = registry.remove_connection("test_pipeline", "nonexistent_target", "related")
        
        assert result is False

    def test_validate_registry_integrity_valid(self, registry, mock_sync, mock_registry_data):
        """Test registry integrity validation with valid registry."""
        registry._cache = mock_registry_data.copy()
        registry._cache_valid = True
        
        result = registry.validate_registry_integrity()
        
        assert result["is_valid"] is True
        assert result["errors"] == []
        assert result["total_nodes"] == 1
        assert result["isolated_nodes"] == 1  # test_pipeline has no connections

    def test_validate_registry_integrity_orphaned_connection(self, registry, mock_sync, mock_registry_data):
        """Test registry integrity validation with orphaned connection."""
        # Add orphaned connection
        mock_registry_data["nodes"]["test_pipeline"]["connections"]["related"] = [
            {"id": "nonexistent_pipeline", "annotation": "Orphaned connection"}
        ]
        
        registry._cache = mock_registry_data.copy()
        registry._cache_valid = True
        
        result = registry.validate_registry_integrity()
        
        assert result["is_valid"] is False
        assert len(result["errors"]) > 0
        assert any("Orphaned connection" in error for error in result["errors"])

    def test_get_registry_statistics(self, registry, mock_sync):
        """Test getting registry statistics."""
        mock_stats = {
            "total_pipelines": 1,
            "total_connections": 0,
            "tag_categories": 3,
            "frameworks": ["xgboost"],
            "complexity_levels": ["simple"]
        }
        mock_sync.get_registry_statistics.return_value = mock_stats
        
        result = registry.get_registry_statistics()
        
        assert result == mock_stats
        mock_sync.get_registry_statistics.assert_called_once()

    def test_clear_cache(self, registry, mock_sync):
        """Test clearing registry cache."""
        # Set up cache
        registry._cache = {"test": "data"}
        registry._cache_valid = True
        
        # Clear cache
        registry.clear_cache()
        
        assert not registry._cache_valid
        assert registry._cache == {}

    def test_convert_zettelkasten_to_node_data(self, registry, mock_sync):
        """Test converting ZettelkastenMetadata to node data."""
        zk_metadata = Mock(spec=ZettelkastenMetadata)
        zk_metadata.atomic_id = "test_id"
        zk_metadata.title = "Test Title"
        zk_metadata.single_responsibility = "Test responsibility"
        zk_metadata.independence_level = "high"
        zk_metadata.node_count = 5
        zk_metadata.edge_count = 4
        zk_metadata.framework = "pytorch"
        zk_metadata.complexity = "advanced"
        zk_metadata.use_case = "Test use case"
        zk_metadata.features = ["feature1", "feature2"]
        zk_metadata.mods_compatible = True
        zk_metadata.framework_tags = ["pytorch"]
        zk_metadata.task_tags = ["training"]
        zk_metadata.complexity_tags = ["advanced"]
        zk_metadata.manual_connections = {"related": ["related_id"]}
        zk_metadata.curated_connections = {"related_id": "Related pipeline"}
        zk_metadata.source_file = "test.py"
        zk_metadata.migration_source = "legacy"
        zk_metadata.created_date = "2025-09-18"
        zk_metadata.priority = "high"
        
        result = registry._convert_zettelkasten_to_node_data(zk_metadata)
        
        assert result["id"] == "test_id"
        assert result["title"] == "Test Title"
        assert result["description"] == "Test responsibility"
        assert result["atomic_properties"]["single_responsibility"] == "Test responsibility"
        assert result["zettelkasten_metadata"]["framework"] == "pytorch"
        assert result["multi_dimensional_tags"]["framework_tags"] == ["pytorch"]
        assert len(result["connections"]["related"]) == 1

    def test_add_or_update_node_legacy(self, registry, mock_sync):
        """Test adding node using legacy ZettelkastenMetadata method."""
        zk_metadata = Mock(spec=ZettelkastenMetadata)
        zk_metadata.atomic_id = "legacy_pipeline"
        zk_metadata.title = "Legacy Pipeline"
        zk_metadata.single_responsibility = "Legacy functionality"
        zk_metadata.independence_level = "medium"
        zk_metadata.node_count = 3
        zk_metadata.edge_count = 2
        zk_metadata.framework = "xgboost"
        zk_metadata.complexity = "simple"
        zk_metadata.use_case = None
        zk_metadata.features = []
        zk_metadata.mods_compatible = False
        zk_metadata.framework_tags = ["xgboost"]
        zk_metadata.task_tags = ["evaluation"]
        zk_metadata.complexity_tags = ["simple"]
        zk_metadata.manual_connections = {}
        zk_metadata.curated_connections = {}
        zk_metadata.source_file = "legacy.py"
        zk_metadata.migration_source = "old_system"
        zk_metadata.created_date = "2025-09-17"
        zk_metadata.priority = "low"
        
        # Mock the add_pipeline_node method to return True
        registry.add_pipeline_node = Mock(return_value=True)
        
        result = registry.add_or_update_node(zk_metadata)
        
        assert result is True
        registry.add_pipeline_node.assert_called_once()

    def test_helper_methods_called(self, registry, mock_sync):
        """Test that helper methods are properly called."""
        # Test _update_tag_index_for_pipeline
        registry._cache = {"tag_index": {}}
        registry._remove_from_tag_index = Mock()
        
        node_data = {
            "multi_dimensional_tags": {
                "framework_tags": ["pytorch"],
                "task_tags": ["training"]
            }
        }
        
        registry._update_tag_index_for_pipeline("test_id", node_data)
        
        # Verify tag index was updated
        assert "framework_tags" in registry._cache["tag_index"]
        assert "pytorch" in registry._cache["tag_index"]["framework_tags"]
        assert "test_id" in registry._cache["tag_index"]["framework_tags"]["pytorch"]

    def test_remove_connections_to_pipeline(self, registry, mock_sync):
        """Test removing all connections to a specific pipeline."""
        registry._cache = {
            "nodes": {
                "pipeline1": {
                    "connections": {
                        "related": [
                            {"id": "target_pipeline", "annotation": "test"},
                            {"id": "other_pipeline", "annotation": "test2"}
                        ],
                        "alternatives": [
                            {"id": "target_pipeline", "annotation": "test3"}
                        ]
                    }
                },
                "pipeline2": {
                    "connections": {
                        "related": [
                            {"id": "other_pipeline", "annotation": "test4"}
                        ]
                    }
                }
            }
        }
        
        registry._remove_connections_to_pipeline("target_pipeline")
        
        # Verify connections to target_pipeline were removed
        pipeline1_related = registry._cache["nodes"]["pipeline1"]["connections"]["related"]
        assert len(pipeline1_related) == 1
        assert pipeline1_related[0]["id"] == "other_pipeline"
        
        pipeline1_alternatives = registry._cache["nodes"]["pipeline1"]["connections"]["alternatives"]
        assert len(pipeline1_alternatives) == 0
        
        # Verify other connections remain
        pipeline2_related = registry._cache["nodes"]["pipeline2"]["connections"]["related"]
        assert len(pipeline2_related) == 1
        assert pipeline2_related[0]["id"] == "other_pipeline"

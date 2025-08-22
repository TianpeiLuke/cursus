"""
Unit tests for CatalogRegistry class.

Tests the central registry manager that implements Zettelkasten principles
for pipeline discovery and navigation.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.cursus.pipeline_catalog.utils.catalog_registry import CatalogRegistry
from src.cursus.pipeline_catalog.shared_dags.enhanced_metadata import (
    EnhancedDAGMetadata, ZettelkastenMetadata, ComplexityLevel, PipelineFramework
)


class TestCatalogRegistry:
    """Test suite for CatalogRegistry class."""
    
    @pytest.fixture
    def temp_registry_file(self):
        """Create a temporary registry file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            registry_data = {
                "version": "1.0",
                "metadata": {
                    "total_pipelines": 2,
                    "frameworks": ["xgboost", "pytorch"],
                    "complexity_levels": ["simple", "standard"],
                    "last_updated": "2024-01-01T00:00:00"
                },
                "nodes": {
                    "xgboost_training_simple": {
                        "id": "xgboost_training_simple",
                        "title": "XGBoost Training Simple",
                        "description": "Simple XGBoost training pipeline",
                        "atomic_properties": {
                            "single_responsibility": "Train XGBoost model",
                            "independence_level": "fully_self_contained",
                            "node_count": 3,
                            "edge_count": 2
                        },
                        "zettelkasten_metadata": {
                            "framework": "xgboost",
                            "complexity": "simple",
                            "use_case": "Basic model training",
                            "features": ["training"],
                            "mods_compatible": True
                        },
                        "multi_dimensional_tags": {
                            "framework_tags": ["xgboost"],
                            "task_tags": ["training"],
                            "complexity_tags": ["simple"]
                        },
                        "connections": {
                            "alternatives": [],
                            "related": [{"id": "pytorch_training_simple", "annotation": "Similar training approach"}],
                            "used_in": []
                        },
                        "source_file": "xgboost/simple_training.py",
                        "created_date": "2024-01-01",
                        "priority": "high"
                    },
                    "pytorch_training_simple": {
                        "id": "pytorch_training_simple",
                        "title": "PyTorch Training Simple",
                        "description": "Simple PyTorch training pipeline",
                        "atomic_properties": {
                            "single_responsibility": "Train PyTorch model",
                            "independence_level": "fully_self_contained",
                            "node_count": 4,
                            "edge_count": 3
                        },
                        "zettelkasten_metadata": {
                            "framework": "pytorch",
                            "complexity": "simple",
                            "use_case": "Basic neural network training",
                            "features": ["training"],
                            "mods_compatible": False
                        },
                        "multi_dimensional_tags": {
                            "framework_tags": ["pytorch"],
                            "task_tags": ["training"],
                            "complexity_tags": ["simple"]
                        },
                        "connections": {
                            "alternatives": [],
                            "related": [{"id": "xgboost_training_simple", "annotation": "Alternative ML approach"}],
                            "used_in": []
                        },
                        "source_file": "pytorch/simple_training.py",
                        "created_date": "2024-01-01",
                        "priority": "medium"
                    }
                },
                "tag_index": {
                    "framework_tags": {
                        "xgboost": ["xgboost_training_simple"],
                        "pytorch": ["pytorch_training_simple"]
                    },
                    "task_tags": {
                        "training": ["xgboost_training_simple", "pytorch_training_simple"]
                    },
                    "complexity_tags": {
                        "simple": ["xgboost_training_simple", "pytorch_training_simple"]
                    }
                }
            }
            json.dump(registry_data, f)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def registry(self, temp_registry_file):
        """Create CatalogRegistry instance with test data."""
        return CatalogRegistry(temp_registry_file)
    
    @pytest.fixture
    def sample_zettelkasten_metadata(self):
        """Create sample ZettelkastenMetadata for testing."""
        return ZettelkastenMetadata(
            atomic_id="test_pipeline_simple",
            title="Test Pipeline",
            single_responsibility="Test pipeline functionality",
            framework="xgboost",
            complexity="simple",
            features=["testing"],
            framework_tags=["xgboost"],
            task_tags=["testing"],
            complexity_tags=["simple"],
            source_file="test/test_pipeline.py"
        )
    
    @pytest.fixture
    def sample_enhanced_metadata(self):
        """Create sample EnhancedDAGMetadata for testing."""
        zettelkasten_metadata = ZettelkastenMetadata(
            atomic_id="enhanced_test_pipeline",
            title="Enhanced Test Pipeline",
            single_responsibility="Enhanced test functionality",
            framework="pytorch",
            complexity="standard",
            features=["training", "evaluation"],
            framework_tags=["pytorch"],
            task_tags=["training", "evaluation"],
            complexity_tags=["standard"]
        )
        
        return EnhancedDAGMetadata(
            description="Enhanced test pipeline",
            complexity=ComplexityLevel.STANDARD,
            features=["training", "evaluation"],
            framework=PipelineFramework.PYTORCH,
            node_count=5,
            edge_count=4,
            zettelkasten_metadata=zettelkasten_metadata
        )
    
    def test_init(self, temp_registry_file):
        """Test CatalogRegistry initialization."""
        registry = CatalogRegistry(temp_registry_file)
        assert registry.registry_path == temp_registry_file
        assert not registry._cache_valid
        assert registry._cache == {}
    
    def test_load_registry(self, registry):
        """Test loading registry data."""
        data = registry.load_registry()
        
        assert data["version"] == "1.0"
        assert len(data["nodes"]) == 2
        assert "xgboost_training_simple" in data["nodes"]
        assert "pytorch_training_simple" in data["nodes"]
        assert registry._cache_valid
    
    def test_save_registry(self, registry, temp_registry_file):
        """Test saving registry data."""
        # Load initial data
        data = registry.load_registry()
        
        # Modify data
        data["metadata"]["total_pipelines"] = 3
        
        # Save modified data
        registry.save_registry(data)
        
        # Verify changes were saved
        with open(temp_registry_file, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data["metadata"]["total_pipelines"] == 3
    
    def test_get_pipeline_node(self, registry):
        """Test retrieving pipeline node data."""
        node = registry.get_pipeline_node("xgboost_training_simple")
        
        assert node is not None
        assert node["id"] == "xgboost_training_simple"
        assert node["title"] == "XGBoost Training Simple"
        assert node["zettelkasten_metadata"]["framework"] == "xgboost"
    
    def test_get_pipeline_node_not_found(self, registry):
        """Test retrieving non-existent pipeline node."""
        node = registry.get_pipeline_node("nonexistent_pipeline")
        assert node is None
    
    def test_get_all_pipelines(self, registry):
        """Test retrieving all pipeline IDs."""
        pipelines = registry.get_all_pipelines()
        
        assert len(pipelines) == 2
        assert "xgboost_training_simple" in pipelines
        assert "pytorch_training_simple" in pipelines
    
    def test_add_or_update_enhanced_node(self, registry, sample_enhanced_metadata):
        """Test adding/updating node with EnhancedDAGMetadata."""
        with patch.object(registry._sync, 'sync_metadata_to_registry') as mock_sync:
            result = registry.add_or_update_enhanced_node(sample_enhanced_metadata)
            
            assert result is True
            mock_sync.assert_called_once()
            assert not registry._cache_valid  # Cache should be cleared
    
    def test_add_or_update_enhanced_node_failure(self, registry, sample_enhanced_metadata):
        """Test handling failure when adding enhanced node."""
        with patch.object(registry._sync, 'sync_metadata_to_registry', side_effect=Exception("Sync failed")):
            result = registry.add_or_update_enhanced_node(sample_enhanced_metadata)
            
            assert result is False
    
    def test_add_or_update_node(self, registry, sample_zettelkasten_metadata):
        """Test adding/updating node with ZettelkastenMetadata."""
        with patch.object(registry, 'add_pipeline_node', return_value=True) as mock_add:
            result = registry.add_or_update_node(sample_zettelkasten_metadata)
            
            assert result is True
            mock_add.assert_called_once()
    
    def test_add_pipeline_node(self, registry):
        """Test adding a new pipeline node."""
        node_data = {
            "id": "new_pipeline",
            "title": "New Pipeline",
            "description": "A new test pipeline",
            "atomic_properties": {
                "single_responsibility": "Test new functionality",
                "independence_level": "fully_self_contained",
                "node_count": 2,
                "edge_count": 1
            },
            "zettelkasten_metadata": {
                "framework": "generic",
                "complexity": "simple",
                "features": ["testing"]
            },
            "multi_dimensional_tags": {
                "framework_tags": ["generic"],
                "task_tags": ["testing"],
                "complexity_tags": ["simple"]
            },
            "connections": {"alternatives": [], "related": [], "used_in": []}
        }
        
        result = registry.add_pipeline_node("new_pipeline", node_data)
        assert result is True
        
        # Verify node was added
        node = registry.get_pipeline_node("new_pipeline")
        assert node is not None
        assert node["id"] == "new_pipeline"
    
    def test_add_pipeline_node_missing_fields(self, registry):
        """Test adding pipeline node with missing required fields."""
        incomplete_node_data = {
            "id": "incomplete_pipeline",
            "title": "Incomplete Pipeline"
            # Missing required fields
        }
        
        result = registry.add_pipeline_node("incomplete_pipeline", incomplete_node_data)
        assert result is False
    
    def test_remove_pipeline_node(self, registry):
        """Test removing a pipeline node."""
        # Verify node exists
        assert registry.get_pipeline_node("xgboost_training_simple") is not None
        
        # Remove node
        result = registry.remove_pipeline_node("xgboost_training_simple")
        assert result is True
        
        # Verify node was removed
        assert registry.get_pipeline_node("xgboost_training_simple") is None
    
    def test_remove_pipeline_node_not_found(self, registry):
        """Test removing non-existent pipeline node."""
        result = registry.remove_pipeline_node("nonexistent_pipeline")
        assert result is False
    
    def test_update_pipeline_node(self, registry):
        """Test updating an existing pipeline node."""
        # Get original node
        original_node = registry.get_pipeline_node("xgboost_training_simple")
        
        # Modify node data
        updated_node = original_node.copy()
        updated_node["title"] = "Updated XGBoost Training"
        
        # Update node
        result = registry.update_pipeline_node("xgboost_training_simple", updated_node)
        assert result is True
        
        # Verify update
        node = registry.get_pipeline_node("xgboost_training_simple")
        assert node["title"] == "Updated XGBoost Training"
    
    def test_update_pipeline_node_not_found(self, registry):
        """Test updating non-existent pipeline node."""
        node_data = {"id": "nonexistent", "title": "Test"}
        result = registry.update_pipeline_node("nonexistent", node_data)
        assert result is False
    
    def test_get_pipelines_by_framework(self, registry):
        """Test retrieving pipelines by framework."""
        xgboost_pipelines = registry.get_pipelines_by_framework("xgboost")
        assert len(xgboost_pipelines) == 1
        assert "xgboost_training_simple" in xgboost_pipelines
        
        pytorch_pipelines = registry.get_pipelines_by_framework("pytorch")
        assert len(pytorch_pipelines) == 1
        assert "pytorch_training_simple" in pytorch_pipelines
        
        # Test non-existent framework
        empty_pipelines = registry.get_pipelines_by_framework("nonexistent")
        assert len(empty_pipelines) == 0
    
    def test_get_pipelines_by_complexity(self, registry):
        """Test retrieving pipelines by complexity."""
        simple_pipelines = registry.get_pipelines_by_complexity("simple")
        assert len(simple_pipelines) == 2
        assert "xgboost_training_simple" in simple_pipelines
        assert "pytorch_training_simple" in simple_pipelines
        
        # Test non-existent complexity
        empty_pipelines = registry.get_pipelines_by_complexity("advanced")
        assert len(empty_pipelines) == 0
    
    def test_get_pipeline_connections(self, registry):
        """Test retrieving pipeline connections."""
        connections = registry.get_pipeline_connections("xgboost_training_simple")
        
        assert "related" in connections
        assert len(connections["related"]) == 1
        assert connections["related"][0]["id"] == "pytorch_training_simple"
        assert connections["related"][0]["annotation"] == "Similar training approach"
    
    def test_add_connection(self, registry):
        """Test adding a connection between pipelines."""
        result = registry.add_connection(
            "xgboost_training_simple",
            "pytorch_training_simple",
            "alternatives",
            "Alternative framework approach"
        )
        assert result is True
        
        # Verify connection was added
        connections = registry.get_pipeline_connections("xgboost_training_simple")
        alternatives = connections["alternatives"]
        assert len(alternatives) == 1
        assert alternatives[0]["id"] == "pytorch_training_simple"
        assert alternatives[0]["annotation"] == "Alternative framework approach"
    
    def test_add_connection_invalid_source(self, registry):
        """Test adding connection with invalid source pipeline."""
        result = registry.add_connection(
            "nonexistent_source",
            "pytorch_training_simple",
            "related",
            "Test connection"
        )
        assert result is False
    
    def test_add_connection_invalid_target(self, registry):
        """Test adding connection with invalid target pipeline."""
        result = registry.add_connection(
            "xgboost_training_simple",
            "nonexistent_target",
            "related",
            "Test connection"
        )
        assert result is False
    
    def test_add_connection_invalid_type(self, registry):
        """Test adding connection with invalid connection type."""
        result = registry.add_connection(
            "xgboost_training_simple",
            "pytorch_training_simple",
            "invalid_type",
            "Test connection"
        )
        assert result is False
    
    def test_remove_connection(self, registry):
        """Test removing a connection between pipelines."""
        # First add a connection
        registry.add_connection(
            "xgboost_training_simple",
            "pytorch_training_simple",
            "alternatives",
            "Test connection"
        )
        
        # Then remove it
        result = registry.remove_connection(
            "xgboost_training_simple",
            "pytorch_training_simple",
            "alternatives"
        )
        assert result is True
        
        # Verify connection was removed
        connections = registry.get_pipeline_connections("xgboost_training_simple")
        assert len(connections["alternatives"]) == 0
    
    def test_remove_connection_not_found(self, registry):
        """Test removing non-existent connection."""
        result = registry.remove_connection(
            "xgboost_training_simple",
            "nonexistent_target",
            "related"
        )
        assert result is False
    
    def test_validate_registry_integrity(self, registry):
        """Test registry integrity validation."""
        validation_result = registry.validate_registry_integrity()
        
        assert validation_result["is_valid"] is True
        assert validation_result["total_nodes"] == 2
        assert len(validation_result["errors"]) == 0
    
    def test_validate_registry_integrity_with_orphaned_connection(self, registry):
        """Test registry validation with orphaned connections."""
        # Add orphaned connection
        data = registry.load_registry()
        data["nodes"]["xgboost_training_simple"]["connections"]["related"].append({
            "id": "nonexistent_pipeline",
            "annotation": "Orphaned connection"
        })
        registry.save_registry(data)
        
        validation_result = registry.validate_registry_integrity()
        
        assert validation_result["is_valid"] is False
        assert len(validation_result["errors"]) > 0
        assert any("Orphaned connection" in error for error in validation_result["errors"])
    
    def test_get_registry_statistics(self, registry):
        """Test retrieving registry statistics."""
        with patch.object(registry._sync, 'get_registry_statistics') as mock_stats:
            mock_stats.return_value = {"total_pipelines": 2, "total_connections": 2}
            
            stats = registry.get_registry_statistics()
            
            assert stats["total_pipelines"] == 2
            assert stats["total_connections"] == 2
            mock_stats.assert_called_once()
    
    def test_clear_cache(self, registry):
        """Test clearing the registry cache."""
        # Load data to populate cache
        registry.load_registry()
        assert registry._cache_valid is True
        
        # Clear cache
        registry.clear_cache()
        assert registry._cache_valid is False
        assert registry._cache == {}
    
    def test_convert_zettelkasten_to_node_data(self, registry, sample_zettelkasten_metadata):
        """Test converting ZettelkastenMetadata to node data format."""
        node_data = registry._convert_zettelkasten_to_node_data(sample_zettelkasten_metadata)
        
        assert node_data["id"] == sample_zettelkasten_metadata.atomic_id
        assert node_data["description"] == sample_zettelkasten_metadata.single_responsibility
        assert node_data["zettelkasten_metadata"]["framework"] == sample_zettelkasten_metadata.framework
        assert node_data["multi_dimensional_tags"]["framework_tags"] == sample_zettelkasten_metadata.framework_tags
    
    @patch('src.cursus.pipeline_catalog.utils.catalog_registry.logger')
    def test_error_handling(self, mock_logger, registry):
        """Test error handling and logging."""
        # Test with invalid registry path
        with patch.object(registry, 'load_registry', side_effect=Exception("Load failed")):
            result = registry.get_all_pipelines()
            assert result == []
            mock_logger.error.assert_called()
    
    def test_update_tag_index_for_pipeline(self, registry):
        """Test updating tag index for a specific pipeline."""
        node_data = {
            "multi_dimensional_tags": {
                "framework_tags": ["new_framework"],
                "task_tags": ["new_task"],
                "complexity_tags": ["new_complexity"]
            }
        }
        
        # Load registry to populate cache
        registry.load_registry()
        
        # Update tag index
        registry._update_tag_index_for_pipeline("test_pipeline", node_data)
        
        # Verify tag index was updated
        tag_index = registry._cache.get("tag_index", {})
        assert "test_pipeline" in tag_index["framework_tags"]["new_framework"]
        assert "test_pipeline" in tag_index["task_tags"]["new_task"]
        assert "test_pipeline" in tag_index["complexity_tags"]["new_complexity"]
    
    def test_remove_connections_to_pipeline(self, registry):
        """Test removing all connections pointing to a specific pipeline."""
        # Load registry
        registry.load_registry()
        
        # Remove connections to pytorch_training_simple
        registry._remove_connections_to_pipeline("pytorch_training_simple")
        
        # Verify connections were removed
        xgboost_connections = registry._cache["nodes"]["xgboost_training_simple"]["connections"]
        related_connections = xgboost_connections["related"]
        
        # Should not contain connection to pytorch_training_simple
        target_ids = [conn["id"] for conn in related_connections]
        assert "pytorch_training_simple" not in target_ids


class TestCatalogRegistryIntegration:
    """Integration tests for CatalogRegistry with real file operations."""
    
    def test_full_workflow(self):
        """Test complete workflow from creation to cleanup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = Path(temp_dir) / "test_registry.json"
            
            # Create registry
            registry = CatalogRegistry(str(registry_path))
            
            # Add a pipeline
            node_data = {
                "id": "test_pipeline",
                "title": "Test Pipeline",
                "description": "Integration test pipeline",
                "atomic_properties": {
                    "single_responsibility": "Test integration",
                    "independence_level": "fully_self_contained",
                    "node_count": 1,
                    "edge_count": 0
                },
                "zettelkasten_metadata": {
                    "framework": "generic",
                    "complexity": "simple",
                    "features": ["testing"]
                },
                "multi_dimensional_tags": {
                    "framework_tags": ["generic"],
                    "task_tags": ["testing"],
                    "complexity_tags": ["simple"]
                },
                "connections": {"alternatives": [], "related": [], "used_in": []}
            }
            
            # Test add
            assert registry.add_pipeline_node("test_pipeline", node_data) is True
            
            # Test retrieve
            retrieved_node = registry.get_pipeline_node("test_pipeline")
            assert retrieved_node is not None
            assert retrieved_node["id"] == "test_pipeline"
            
            # Test update
            node_data["title"] = "Updated Test Pipeline"
            assert registry.update_pipeline_node("test_pipeline", node_data) is True
            
            updated_node = registry.get_pipeline_node("test_pipeline")
            assert updated_node["title"] == "Updated Test Pipeline"
            
            # Test remove
            assert registry.remove_pipeline_node("test_pipeline") is True
            assert registry.get_pipeline_node("test_pipeline") is None

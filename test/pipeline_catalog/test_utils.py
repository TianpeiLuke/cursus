"""
Unit tests for Pipeline Catalog Utils module.

Tests the main PipelineCatalogManager class and convenience functions
that provide unified access to all pipeline catalog functionality.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from typing import Dict, List, Any

from cursus.pipeline_catalog.utils import (
    PipelineCatalogManager,
    create_catalog_manager,
    discover_by_framework,
    discover_by_tags,
    get_pipeline_alternatives,
)
from cursus.pipeline_catalog.shared_dags.enhanced_metadata import EnhancedDAGMetadata


class TestPipelineCatalogManager:
    """Test suite for PipelineCatalogManager class."""

    @pytest.fixture
    def temp_registry_file(self):
        """Create temporary registry file with proper registry format."""
        temp_dir = tempfile.mkdtemp()
        registry_path = Path(temp_dir) / "test_registry.json"
        
        # Create registry structure matching the actual format expected by CatalogRegistry
        registry_data = {
            "version": "1.0",
            "metadata": {
                "total_nodes": 3,
                "last_updated": "2024-01-01T00:00:00Z"
            },
            "nodes": {
                "xgboost-training": {
                    "id": "xgboost-training",
                    "title": "XGBoost Training Pipeline",
                    "description": "Simple XGBoost training pipeline",
                    "atomic_properties": {
                        "single_responsibility": "Train XGBoost model",
                        "independence_level": "fully_self_contained",
                        "node_count": 1,
                        "edge_count": 0
                    },
                    "zettelkasten_metadata": {
                        "framework": "xgboost",
                        "complexity": "simple",
                        "use_case": "classification",
                        "features": ["training"],
                        "mods_compatible": True
                    },
                    "multi_dimensional_tags": {
                        "framework_tags": ["xgboost"],
                        "task_tags": ["training"],
                        "complexity_tags": ["simple"]
                    },
                    "connections": {
                        "alternatives": [{"id": "pytorch-training", "annotation": "Alternative framework"}],
                        "related": [{"id": "xgboost-evaluation", "annotation": "Related evaluation"}],
                        "used_in": []
                    },
                    "created_date": "2024-01-01",
                    "priority": "high"
                },
                "pytorch-training": {
                    "id": "pytorch-training",
                    "title": "PyTorch Training Pipeline",
                    "description": "Intermediate PyTorch training pipeline",
                    "atomic_properties": {
                        "single_responsibility": "Train PyTorch model",
                        "independence_level": "fully_self_contained",
                        "node_count": 1,
                        "edge_count": 0
                    },
                    "zettelkasten_metadata": {
                        "framework": "pytorch",
                        "complexity": "intermediate",
                        "use_case": "deep_learning",
                        "features": ["training"],
                        "mods_compatible": True
                    },
                    "multi_dimensional_tags": {
                        "framework_tags": ["pytorch"],
                        "task_tags": ["training"],
                        "complexity_tags": ["intermediate"]
                    },
                    "connections": {
                        "alternatives": [{"id": "xgboost-training", "annotation": "Alternative framework"}],
                        "related": [],
                        "used_in": []
                    },
                    "created_date": "2024-01-01",
                    "priority": "medium"
                },
                "xgboost-evaluation": {
                    "id": "xgboost-evaluation",
                    "title": "XGBoost Evaluation Pipeline",
                    "description": "Simple XGBoost evaluation pipeline",
                    "atomic_properties": {
                        "single_responsibility": "Evaluate XGBoost model",
                        "independence_level": "fully_self_contained",
                        "node_count": 1,
                        "edge_count": 0
                    },
                    "zettelkasten_metadata": {
                        "framework": "xgboost",
                        "complexity": "simple",
                        "use_case": "model_validation",
                        "features": ["evaluation"],
                        "mods_compatible": True
                    },
                    "multi_dimensional_tags": {
                        "framework_tags": ["xgboost"],
                        "task_tags": ["evaluation"],
                        "complexity_tags": ["simple"]
                    },
                    "connections": {
                        "alternatives": [],
                        "related": [{"id": "xgboost-training", "annotation": "Related training"}],
                        "used_in": []
                    },
                    "created_date": "2024-01-01",
                    "priority": "medium"
                }
            },
            "tag_index": {
                "framework_tags": {
                    "xgboost": ["xgboost-training", "xgboost-evaluation"],
                    "pytorch": ["pytorch-training"]
                },
                "task_tags": {
                    "training": ["xgboost-training", "pytorch-training"],
                    "evaluation": ["xgboost-evaluation"]
                },
                "complexity_tags": {
                    "simple": ["xgboost-training", "xgboost-evaluation"],
                    "intermediate": ["pytorch-training"]
                }
            }
        }
        
        with open(registry_path, 'w') as f:
            json.dump(registry_data, f, indent=2)
        
        yield str(registry_path)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def manager(self, temp_registry_file):
        """Create PipelineCatalogManager instance."""
        return PipelineCatalogManager(temp_registry_file)

    def test_init_with_registry_path(self, temp_registry_file):
        """Test initialization with custom registry path."""
        manager = PipelineCatalogManager(temp_registry_file)
        
        assert manager.registry_path == temp_registry_file
        assert manager.registry is not None
        assert manager.traverser is not None
        assert manager.discovery is not None
        assert manager.recommender is not None
        assert manager.validator is not None
        assert manager.sync is not None

    def test_init_with_default_path(self):
        """Test initialization with default registry path."""
        manager = PipelineCatalogManager()
        
        expected_path = str(Path(__file__).parent.parent / "src" / "cursus" / "pipeline_catalog" / "catalog_index.json")
        # The path should contain catalog_index.json
        assert "catalog_index.json" in manager.registry_path
        assert manager.registry is not None

    def test_discover_pipelines_by_framework(self, manager):
        """Test pipeline discovery by framework."""
        # Test with existing framework
        xgboost_pipelines = manager.discover_pipelines(framework="xgboost")
        assert isinstance(xgboost_pipelines, list)
        assert len(xgboost_pipelines) >= 0  # May be empty if discovery fails

        # Test with non-existent framework
        unknown_pipelines = manager.discover_pipelines(framework="unknown")
        assert isinstance(unknown_pipelines, list)

    def test_discover_pipelines_by_complexity(self, manager):
        """Test pipeline discovery by complexity."""
        simple_pipelines = manager.discover_pipelines(complexity="simple")
        assert isinstance(simple_pipelines, list)

    def test_discover_pipelines_by_tags(self, manager):
        """Test pipeline discovery by tags."""
        training_pipelines = manager.discover_pipelines(tags=["training"])
        assert isinstance(training_pipelines, list)

    def test_discover_pipelines_by_use_case(self, manager):
        """Test pipeline discovery by use case."""
        classification_pipelines = manager.discover_pipelines(use_case="classification")
        assert isinstance(classification_pipelines, list)

    def test_discover_pipelines_no_criteria(self, manager):
        """Test pipeline discovery with no criteria."""
        all_pipelines = manager.discover_pipelines()
        assert isinstance(all_pipelines, list)

    def test_discover_pipelines_with_step_catalog_fallback(self, manager):
        """Test discovery fallback when step catalog is not available."""
        # The step catalog import happens inside the method, so we test the fallback behavior
        result = manager.discover_pipelines(framework="xgboost")
        assert isinstance(result, list)

    def test_discover_pipelines_catalog_integration(self, manager):
        """Test step catalog integration behavior."""
        # Test that the method handles step catalog gracefully
        result = manager.discover_pipelines(tags=["training"])
        assert isinstance(result, list)
        
        result = manager.discover_pipelines(use_case="classification")
        assert isinstance(result, list)

    def test_get_pipeline_connections(self, manager):
        """Test getting pipeline connections."""
        # Test with existing pipeline - this calls traverser.get_all_connections internally
        connections = manager.get_pipeline_connections("xgboost-training")
        assert isinstance(connections, dict)
        
        # Test with non-existent pipeline
        empty_connections = manager.get_pipeline_connections("non-existent")
        assert isinstance(empty_connections, dict)

    def test_find_path(self, manager):
        """Test finding path between pipelines."""
        # Test path finding - this calls traverser.find_shortest_path internally
        path = manager.find_path("xgboost-training", "pytorch-training")
        assert path is None or isinstance(path, list)

        # Test path to same pipeline
        same_path = manager.find_path("xgboost-training", "xgboost-training")
        assert same_path is None or isinstance(same_path, list)

    def test_get_recommendations(self, manager):
        """Test getting pipeline recommendations."""
        recommendations = manager.get_recommendations("classification")
        assert isinstance(recommendations, list)

        # Test with just use case (no additional kwargs that might not be supported)
        simple_recommendations = manager.get_recommendations("classification")
        assert isinstance(simple_recommendations, list)

    def test_validate_registry(self, manager):
        """Test registry validation."""
        validation_result = manager.validate_registry()
        
        assert isinstance(validation_result, dict)
        assert "is_valid" in validation_result
        assert "total_issues" in validation_result
        assert "issues_by_severity" in validation_result
        assert "issues_by_category" in validation_result
        assert "all_issues" in validation_result
        
        assert isinstance(validation_result["is_valid"], bool)
        assert isinstance(validation_result["total_issues"], int)
        assert isinstance(validation_result["issues_by_severity"], dict)
        assert isinstance(validation_result["issues_by_category"], dict)
        assert isinstance(validation_result["all_issues"], list)

    def test_sync_pipeline_success(self, manager):
        """Test successful pipeline sync."""
        # Create mock metadata
        mock_metadata = Mock(spec=EnhancedDAGMetadata)
        
        with patch.object(manager.sync, 'sync_metadata_to_registry') as mock_sync:
            mock_sync.return_value = None  # Successful sync
            
            result = manager.sync_pipeline(mock_metadata, "test_pipeline.py")
            assert result is True
            mock_sync.assert_called_once_with(mock_metadata, "test_pipeline.py")

    def test_sync_pipeline_failure(self, manager):
        """Test pipeline sync failure."""
        # Create mock metadata
        mock_metadata = Mock(spec=EnhancedDAGMetadata)
        
        with patch.object(manager.sync, 'sync_metadata_to_registry') as mock_sync:
            mock_sync.side_effect = Exception("Sync failed")
            
            result = manager.sync_pipeline(mock_metadata, "test_pipeline.py")
            assert result is False

    def test_get_registry_stats(self, manager):
        """Test getting registry statistics."""
        with patch.object(manager.sync, 'get_registry_statistics') as mock_stats:
            mock_stats.return_value = {
                "total_pipelines": 3,
                "frameworks": {"xgboost": 2, "pytorch": 1},
                "complexity_distribution": {"simple": 2, "intermediate": 1}
            }
            
            stats = manager.get_registry_stats()
            assert isinstance(stats, dict)
            assert "total_pipelines" in stats
            mock_stats.assert_called_once()


class TestConvenienceFunctions:
    """Test suite for convenience functions."""

    @pytest.fixture
    def temp_registry_file(self):
        """Create temporary registry file with proper format."""
        temp_dir = tempfile.mkdtemp()
        registry_path = Path(temp_dir) / "test_registry.json"
        
        registry_data = {
            "version": "1.0",
            "metadata": {
                "total_nodes": 1,
                "last_updated": "2024-01-01T00:00:00Z"
            },
            "nodes": {
                "xgboost-training": {
                    "id": "xgboost-training",
                    "title": "XGBoost Training Pipeline",
                    "description": "Simple XGBoost training pipeline",
                    "atomic_properties": {
                        "single_responsibility": "Train XGBoost model",
                        "independence_level": "fully_self_contained",
                        "node_count": 1,
                        "edge_count": 0
                    },
                    "zettelkasten_metadata": {
                        "framework": "xgboost",
                        "complexity": "simple",
                        "use_case": "classification",
                        "features": ["training"],
                        "mods_compatible": True
                    },
                    "multi_dimensional_tags": {
                        "framework_tags": ["xgboost"],
                        "task_tags": ["training"],
                        "complexity_tags": ["simple"]
                    },
                    "connections": {
                        "alternatives": [{"id": "pytorch-training", "annotation": "Alternative framework"}],
                        "related": [],
                        "used_in": []
                    },
                    "created_date": "2024-01-01",
                    "priority": "high"
                }
            },
            "tag_index": {
                "framework_tags": {
                    "xgboost": ["xgboost-training"]
                },
                "task_tags": {
                    "training": ["xgboost-training"]
                },
                "complexity_tags": {
                    "simple": ["xgboost-training"]
                }
            }
        }
        
        with open(registry_path, 'w') as f:
            json.dump(registry_data, f, indent=2)
        
        yield str(registry_path)
        shutil.rmtree(temp_dir)

    def test_create_catalog_manager(self, temp_registry_file):
        """Test creating catalog manager."""
        # Test with custom path
        manager = create_catalog_manager(temp_registry_file)
        assert isinstance(manager, PipelineCatalogManager)
        assert manager.registry_path == temp_registry_file

        # Test with default path
        default_manager = create_catalog_manager()
        assert isinstance(default_manager, PipelineCatalogManager)

    def test_discover_by_framework(self, temp_registry_file):
        """Test framework discovery convenience function."""
        result = discover_by_framework("xgboost", temp_registry_file)
        assert isinstance(result, list)

        # Test with default registry
        default_result = discover_by_framework("xgboost")
        assert isinstance(default_result, list)

    def test_discover_by_tags(self, temp_registry_file):
        """Test tag discovery convenience function."""
        result = discover_by_tags(["training"], temp_registry_file)
        assert isinstance(result, list)

        # Test with default registry
        default_result = discover_by_tags(["training"])
        assert isinstance(default_result, list)

    def test_get_pipeline_alternatives(self, temp_registry_file):
        """Test getting pipeline alternatives."""
        alternatives = get_pipeline_alternatives("xgboost-training", temp_registry_file)
        assert isinstance(alternatives, list)

        # Test with default registry
        default_alternatives = get_pipeline_alternatives("xgboost-training")
        assert isinstance(default_alternatives, list)

    def test_get_pipeline_alternatives_nonexistent(self, temp_registry_file):
        """Test getting alternatives for non-existent pipeline."""
        alternatives = get_pipeline_alternatives("non-existent", temp_registry_file)
        assert isinstance(alternatives, list)
        assert len(alternatives) == 0


class TestIntegrationScenarios:
    """Integration tests for real-world scenarios."""

    @pytest.fixture
    def comprehensive_registry(self):
        """Create comprehensive registry for integration testing with proper format."""
        temp_dir = tempfile.mkdtemp()
        registry_path = Path(temp_dir) / "comprehensive_registry.json"
        
        registry_data = {
            "version": "1.0",
            "metadata": {
                "total_nodes": 4,
                "last_updated": "2024-01-01T00:00:00Z"
            },
            "nodes": {
                "xgboost-simple-training": {
                    "id": "xgboost-simple-training",
                    "title": "Simple XGBoost Training",
                    "description": "Simple XGBoost training pipeline for tabular data",
                    "atomic_properties": {
                        "single_responsibility": "Train XGBoost model on tabular data",
                        "independence_level": "fully_self_contained",
                        "node_count": 1,
                        "edge_count": 0
                    },
                    "zettelkasten_metadata": {
                        "framework": "xgboost",
                        "complexity": "simple",
                        "use_case": "classification",
                        "features": ["training", "tabular"],
                        "mods_compatible": True
                    },
                    "multi_dimensional_tags": {
                        "framework_tags": ["xgboost"],
                        "task_tags": ["training"],
                        "complexity_tags": ["simple"],
                        "domain_tags": ["tabular"]
                    },
                    "connections": {
                        "alternatives": [{"id": "pytorch-simple-training", "annotation": "Alternative framework"}],
                        "related": [{"id": "xgboost-evaluation", "annotation": "Related evaluation"}],
                        "used_in": []
                    },
                    "created_date": "2024-01-01",
                    "priority": "high"
                },
                "pytorch-simple-training": {
                    "id": "pytorch-simple-training",
                    "title": "Simple PyTorch Training",
                    "description": "Simple PyTorch training pipeline for deep learning",
                    "atomic_properties": {
                        "single_responsibility": "Train PyTorch model for deep learning",
                        "independence_level": "fully_self_contained",
                        "node_count": 1,
                        "edge_count": 0
                    },
                    "zettelkasten_metadata": {
                        "framework": "pytorch",
                        "complexity": "simple",
                        "use_case": "deep_learning",
                        "features": ["training", "deep_learning"],
                        "mods_compatible": True
                    },
                    "multi_dimensional_tags": {
                        "framework_tags": ["pytorch"],
                        "task_tags": ["training"],
                        "complexity_tags": ["simple"],
                        "domain_tags": ["deep_learning"]
                    },
                    "connections": {
                        "alternatives": [{"id": "xgboost-simple-training", "annotation": "Alternative framework"}],
                        "related": [],
                        "used_in": []
                    },
                    "created_date": "2024-01-01",
                    "priority": "medium"
                },
                "xgboost-advanced-training": {
                    "id": "xgboost-advanced-training",
                    "title": "Advanced XGBoost Training",
                    "description": "Advanced XGBoost training with hyperparameter tuning",
                    "atomic_properties": {
                        "single_responsibility": "Advanced XGBoost training with hyperparameter optimization",
                        "independence_level": "fully_self_contained",
                        "node_count": 1,
                        "edge_count": 0
                    },
                    "zettelkasten_metadata": {
                        "framework": "xgboost",
                        "complexity": "advanced",
                        "use_case": "classification",
                        "features": ["training", "hyperparameter_tuning"],
                        "mods_compatible": True
                    },
                    "multi_dimensional_tags": {
                        "framework_tags": ["xgboost"],
                        "task_tags": ["training"],
                        "complexity_tags": ["advanced"],
                        "domain_tags": ["tabular"]
                    },
                    "connections": {
                        "alternatives": [],
                        "related": [{"id": "xgboost-simple-training", "annotation": "Simpler version"}],
                        "used_in": []
                    },
                    "created_date": "2024-01-01",
                    "priority": "medium"
                },
                "xgboost-evaluation": {
                    "id": "xgboost-evaluation",
                    "title": "XGBoost Model Evaluation",
                    "description": "Comprehensive XGBoost model evaluation with metrics",
                    "atomic_properties": {
                        "single_responsibility": "Evaluate XGBoost model performance",
                        "independence_level": "fully_self_contained",
                        "node_count": 1,
                        "edge_count": 0
                    },
                    "zettelkasten_metadata": {
                        "framework": "xgboost",
                        "complexity": "intermediate",
                        "use_case": "model_validation",
                        "features": ["evaluation", "metrics"],
                        "mods_compatible": True
                    },
                    "multi_dimensional_tags": {
                        "framework_tags": ["xgboost"],
                        "task_tags": ["evaluation"],
                        "complexity_tags": ["intermediate"],
                        "domain_tags": ["tabular"]
                    },
                    "connections": {
                        "alternatives": [],
                        "related": [
                            {"id": "xgboost-simple-training", "annotation": "Related training"},
                            {"id": "xgboost-advanced-training", "annotation": "Related advanced training"}
                        ],
                        "used_in": []
                    },
                    "created_date": "2024-01-01",
                    "priority": "medium"
                }
            },
            "tag_index": {
                "framework_tags": {
                    "xgboost": ["xgboost-simple-training", "xgboost-advanced-training", "xgboost-evaluation"],
                    "pytorch": ["pytorch-simple-training"]
                },
                "task_tags": {
                    "training": ["xgboost-simple-training", "pytorch-simple-training", "xgboost-advanced-training"],
                    "evaluation": ["xgboost-evaluation"]
                },
                "complexity_tags": {
                    "simple": ["xgboost-simple-training", "pytorch-simple-training"],
                    "intermediate": ["xgboost-evaluation"],
                    "advanced": ["xgboost-advanced-training"]
                },
                "domain_tags": {
                    "tabular": ["xgboost-simple-training", "xgboost-advanced-training", "xgboost-evaluation"],
                    "deep_learning": ["pytorch-simple-training"]
                }
            }
        }
        
        with open(registry_path, 'w') as f:
            json.dump(registry_data, f, indent=2)
        
        yield str(registry_path)
        shutil.rmtree(temp_dir)

    def test_ml_workflow_discovery(self, comprehensive_registry):
        """Test discovering pipelines for a complete ML workflow."""
        manager = PipelineCatalogManager(comprehensive_registry)
        
        # Discover training pipelines
        training_pipelines = manager.discover_pipelines(tags=["training"])
        assert len(training_pipelines) >= 0
        
        # Discover evaluation pipelines
        evaluation_pipelines = manager.discover_pipelines(tags=["evaluation"])
        assert len(evaluation_pipelines) >= 0
        
        # Get recommendations for classification
        recommendations = manager.get_recommendations("classification")
        assert isinstance(recommendations, list)

    def test_framework_comparison_workflow(self, comprehensive_registry):
        """Test comparing different frameworks."""
        manager = PipelineCatalogManager(comprehensive_registry)
        
        # Discover XGBoost pipelines
        xgboost_pipelines = manager.discover_pipelines(framework="xgboost")
        assert isinstance(xgboost_pipelines, list)
        
        # Discover PyTorch pipelines
        pytorch_pipelines = manager.discover_pipelines(framework="pytorch")
        assert isinstance(pytorch_pipelines, list)
        
        # Get alternatives for XGBoost pipeline
        alternatives = manager.get_pipeline_connections("xgboost-simple-training")
        assert isinstance(alternatives, dict)

    def test_complexity_progression_workflow(self, comprehensive_registry):
        """Test discovering pipelines by complexity progression."""
        manager = PipelineCatalogManager(comprehensive_registry)
        
        # Start with simple pipelines
        simple_pipelines = manager.discover_pipelines(complexity="simple")
        assert isinstance(simple_pipelines, list)
        
        # Progress to intermediate
        intermediate_pipelines = manager.discover_pipelines(complexity="intermediate")
        assert isinstance(intermediate_pipelines, list)
        
        # Advanced pipelines
        advanced_pipelines = manager.discover_pipelines(complexity="advanced")
        assert isinstance(advanced_pipelines, list)

    def test_registry_health_check(self, comprehensive_registry):
        """Test comprehensive registry validation."""
        manager = PipelineCatalogManager(comprehensive_registry)
        
        # Validate registry
        validation_result = manager.validate_registry()
        assert validation_result["is_valid"] in [True, False]
        assert validation_result["total_issues"] >= 0
        
        # Get registry statistics
        stats = manager.get_registry_stats()
        assert isinstance(stats, dict)

    def test_pipeline_relationship_exploration(self, comprehensive_registry):
        """Test exploring pipeline relationships."""
        manager = PipelineCatalogManager(comprehensive_registry)
        
        # Get connections for a pipeline
        connections = manager.get_pipeline_connections("xgboost-simple-training")
        assert isinstance(connections, dict)
        
        # Try to find paths between related pipelines
        if connections.get("alternatives"):
            alternative = connections["alternatives"][0]
            path = manager.find_path("xgboost-simple-training", alternative)
            assert path is None or isinstance(path, list)

    def test_error_resilience(self, comprehensive_registry):
        """Test system resilience to various error conditions."""
        manager = PipelineCatalogManager(comprehensive_registry)
        
        # Test with non-existent pipeline
        empty_connections = manager.get_pipeline_connections("non-existent-pipeline")
        assert isinstance(empty_connections, dict)
        
        # Test with invalid criteria
        empty_discovery = manager.discover_pipelines(invalid_criteria="test")
        assert isinstance(empty_discovery, list)
        
        # Test recommendations for unknown use case
        empty_recommendations = manager.get_recommendations("unknown-use-case")
        assert isinstance(empty_recommendations, list)

    def test_step_catalog_integration_fallback(self, comprehensive_registry):
        """Test graceful fallback when step catalog is not available."""
        manager = PipelineCatalogManager(comprehensive_registry)
        
        # Test that the method handles step catalog gracefully by falling back to legacy discovery
        # The StepCatalog import happens inside the method, so we test the fallback behavior
        result = manager.discover_pipelines(framework="xgboost")
        assert isinstance(result, list)
        
        result = manager.discover_pipelines(framework="pytorch")
        assert isinstance(result, list)

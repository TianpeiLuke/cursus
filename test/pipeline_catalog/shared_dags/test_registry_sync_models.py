"""
Model tests for registry synchronization infrastructure.

Tests basic models and utility functions for the DAGMetadataRegistrySync system.
"""

import pytest
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from src.cursus.pipeline_catalog.shared_dags.registry_sync import (
    DAGMetadataRegistrySync, RegistryValidationError,
    create_empty_registry, validate_registry_file
)
from src.cursus.pipeline_catalog.shared_dags.enhanced_metadata import (
    EnhancedDAGMetadata, ZettelkastenMetadata, ComplexityLevel, PipelineFramework
)


class TestRegistryValidationError:
    """Test suite for RegistryValidationError exception."""
    
    def test_registry_validation_error_creation(self):
        """Test creating RegistryValidationError."""
        error = RegistryValidationError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)


class TestRegistrySyncUtilityFunctions:
    """Test suite for utility functions in registry_sync module."""
    
    @pytest.fixture
    def temp_registry_path(self):
        """Create temporary registry path for testing."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        yield temp_path
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def sample_registry_data(self):
        """Create sample registry data for testing."""
        return {
            "version": "1.0",
            "description": "Test registry",
            "metadata": {
                "total_pipelines": 1,
                "frameworks": ["xgboost"],
                "complexity_levels": ["simple"],
                "last_updated": "2024-01-15T10:00:00",
                "connection_types": ["alternatives", "related", "used_in"]
            },
            "nodes": {
                "xgb_simple_training": {
                    "id": "xgb_simple_training",
                    "title": "XGBoost Simple Training",
                    "description": "Simple XGBoost training pipeline",
                    "atomic_properties": {
                        "single_responsibility": "Train XGBoost model",
                        "independence_level": "fully_self_contained",
                        "node_count": 1,
                        "edge_count": 0,
                        "input_interface": ["data"],
                        "output_interface": ["model"],
                        "side_effects": "none"
                    },
                    "zettelkasten_metadata": {
                        "framework": "xgboost",
                        "complexity": "simple",
                        "features": ["training"],
                        "mods_compatible": False,
                        "use_case": "tabular_training"
                    },
                    "multi_dimensional_tags": {
                        "framework_tags": ["xgboost"],
                        "task_tags": ["training"],
                        "complexity_tags": ["simple"],
                        "domain_tags": ["tabular"],
                        "pattern_tags": ["atomic_workflow"]
                    },
                    "connections": {
                        "alternatives": [],
                        "related": [],
                        "used_in": []
                    },
                    "source_file": "xgboost/simple_training.py",
                    "created_date": "2024-01-15",
                    "priority": "medium"
                }
            },
            "connection_graph_metadata": {
                "total_connections": 0,
                "connection_density": 0.0,
                "independent_pipelines": 1,
                "composition_opportunities": 0,
                "alternative_groups": 0,
                "isolated_nodes": ["xgb_simple_training"]
            },
            "tag_index": {
                "framework_tags": {
                    "xgboost": ["xgb_simple_training"]
                },
                "task_tags": {
                    "training": ["xgb_simple_training"]
                },
                "complexity_tags": {
                    "simple": ["xgb_simple_training"]
                },
                "domain_tags": {
                    "tabular": ["xgb_simple_training"]
                }
            }
        }
    
    def test_create_empty_registry_function(self, temp_registry_path):
        """Test create_empty_registry utility function."""
        # Ensure file doesn't exist
        Path(temp_registry_path).unlink(missing_ok=True)
        
        create_empty_registry(temp_registry_path)
        
        # Verify file was created
        assert Path(temp_registry_path).exists()
        
        # Verify structure
        with open(temp_registry_path, 'r') as f:
            registry = json.load(f)
        
        assert registry["version"] == "1.0"
        assert registry["metadata"]["total_pipelines"] == 0
        assert registry["nodes"] == {}
    
    def test_validate_registry_file_valid(self, temp_registry_path, sample_registry_data):
        """Test validate_registry_file with valid registry."""
        with open(temp_registry_path, 'w') as f:
            json.dump(sample_registry_data, f)
        
        errors = validate_registry_file(temp_registry_path)
        assert len(errors) == 0
    
    def test_validate_registry_file_orphaned_connections(self, temp_registry_path, sample_registry_data):
        """Test validate_registry_file with orphaned connections."""
        # Add orphaned connection
        sample_registry_data["nodes"]["xgb_simple_training"]["connections"]["alternatives"] = [
            {"id": "non_existent_pipeline", "annotation": "Orphaned connection"}
        ]
        
        with open(temp_registry_path, 'w') as f:
            json.dump(sample_registry_data, f)
        
        errors = validate_registry_file(temp_registry_path)
        assert len(errors) == 1
        assert "Orphaned connection" in errors[0]
        assert "non_existent_pipeline" in errors[0]
    
    def test_validate_registry_file_tag_index_inconsistency(self, temp_registry_path, sample_registry_data):
        """Test validate_registry_file with tag index inconsistency."""
        # Add reference to non-existent pipeline in tag index
        sample_registry_data["tag_index"]["framework_tags"]["pytorch"] = ["non_existent_pipeline"]
        
        with open(temp_registry_path, 'w') as f:
            json.dump(sample_registry_data, f)
        
        errors = validate_registry_file(temp_registry_path)
        assert len(errors) == 1
        assert "Tag index references non-existent pipeline" in errors[0]
        assert "non_existent_pipeline" in errors[0]
    
    def test_validate_registry_file_error(self, temp_registry_path):
        """Test validate_registry_file with registry error."""
        # Create invalid JSON
        with open(temp_registry_path, 'w') as f:
            f.write("invalid json")
        
        errors = validate_registry_file(temp_registry_path)
        assert len(errors) == 1
        assert "Registry validation failed" in errors[0]

"""
Core tests for DAGMetadataRegistrySync class.

Tests the main functionality of the DAGMetadataRegistrySync class including
initialization, registry operations, validation, and synchronization methods.
"""

import pytest
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from cursus.pipeline_catalog.shared_dags.registry_sync import (
    DAGMetadataRegistrySync,
    RegistryValidationError,
    create_empty_registry,
    validate_registry_file,
)
from cursus.pipeline_catalog.shared_dags.enhanced_metadata import (
    EnhancedDAGMetadata,
    ZettelkastenMetadata,
    ComplexityLevel,
    PipelineFramework,
)


class TestDAGMetadataRegistrySync:
    """Test suite for DAGMetadataRegistrySync class."""

    @pytest.fixture
    def temp_registry_path(self):
        """Create temporary registry path for testing."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
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
                "connection_types": ["alternatives", "related", "used_in"],
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
                        "side_effects": "none",
                    },
                    "zettelkasten_metadata": {
                        "framework": "xgboost",
                        "complexity": "simple",
                        "features": ["training"],
                        "mods_compatible": False,
                        "use_case": "tabular_training",
                    },
                    "multi_dimensional_tags": {
                        "framework_tags": ["xgboost"],
                        "task_tags": ["training"],
                        "complexity_tags": ["simple"],
                        "domain_tags": ["tabular"],
                        "pattern_tags": ["atomic_workflow"],
                    },
                    "connections": {"alternatives": [], "related": [], "used_in": []},
                    "source_file": "xgboost/simple_training.py",
                    "created_date": "2024-01-15",
                    "priority": "medium",
                }
            },
            "connection_graph_metadata": {
                "total_connections": 0,
                "connection_density": 0.0,
                "independent_pipelines": 1,
                "composition_opportunities": 0,
                "alternative_groups": 0,
                "isolated_nodes": ["xgb_simple_training"],
            },
            "tag_index": {
                "framework_tags": {"xgboost": ["xgb_simple_training"]},
                "task_tags": {"training": ["xgb_simple_training"]},
                "complexity_tags": {"simple": ["xgb_simple_training"]},
                "domain_tags": {"tabular": ["xgb_simple_training"]},
            },
        }

    @pytest.fixture
    def sample_enhanced_metadata(self):
        """Create sample EnhancedDAGMetadata for testing."""
        return EnhancedDAGMetadata(
            description="Test XGBoost training pipeline",
            complexity=ComplexityLevel.SIMPLE,
            features=["training"],
            framework=PipelineFramework.XGBOOST,
            node_count=3,
            edge_count=2,
        )

    def test_init_with_existing_registry(
        self, temp_registry_path, sample_registry_data
    ):
        """Test initialization with existing registry file."""
        # Create registry file
        with open(temp_registry_path, "w") as f:
            json.dump(sample_registry_data, f)

        sync = DAGMetadataRegistrySync(temp_registry_path)
        assert sync.registry_path == Path(temp_registry_path)
        assert sync.registry_path.exists()

    def test_init_creates_new_registry(self, temp_registry_path):
        """Test initialization creates new registry if none exists."""
        # Ensure file doesn't exist
        Path(temp_registry_path).unlink(missing_ok=True)

        sync = DAGMetadataRegistrySync(temp_registry_path)

        # Should create the file
        assert sync.registry_path.exists()

        # Should have proper structure
        registry = sync.load_registry()
        assert "version" in registry
        assert "metadata" in registry
        assert "nodes" in registry
        assert registry["metadata"]["total_pipelines"] == 0

    def test_create_empty_registry(self, temp_registry_path):
        """Test creating empty registry structure."""
        sync = DAGMetadataRegistrySync(temp_registry_path)
        sync._create_empty_registry()

        with open(temp_registry_path, "r") as f:
            registry = json.load(f)

        # Verify structure
        assert registry["version"] == "1.0"
        assert "description" in registry
        assert registry["metadata"]["total_pipelines"] == 0
        assert registry["nodes"] == {}
        assert "connection_graph_metadata" in registry
        assert "tag_index" in registry

        # Verify connection types
        assert "alternatives" in registry["metadata"]["connection_types"]
        assert "related" in registry["metadata"]["connection_types"]
        assert "used_in" in registry["metadata"]["connection_types"]

    def test_load_registry_success(self, temp_registry_path, sample_registry_data):
        """Test successful registry loading."""
        with open(temp_registry_path, "w") as f:
            json.dump(sample_registry_data, f)

        sync = DAGMetadataRegistrySync(temp_registry_path)
        registry = sync.load_registry()

        assert registry["version"] == "1.0"
        assert registry["metadata"]["total_pipelines"] == 1
        assert "xgb_simple_training" in registry["nodes"]

    def test_load_registry_file_not_found(self, temp_registry_path):
        """Test loading registry when file doesn't exist."""
        # Ensure file doesn't exist
        Path(temp_registry_path).unlink(missing_ok=True)

        sync = DAGMetadataRegistrySync(temp_registry_path)
        registry = sync.load_registry()

        # Should create new registry
        assert registry["metadata"]["total_pipelines"] == 0
        assert registry["nodes"] == {}

    def test_load_registry_invalid_json(self, temp_registry_path):
        """Test loading registry with invalid JSON."""
        with open(temp_registry_path, "w") as f:
            f.write("invalid json content")

        sync = DAGMetadataRegistrySync(temp_registry_path)

        with pytest.raises(RegistryValidationError, match="Invalid JSON"):
            sync.load_registry()

    def test_load_registry_missing_required_fields(self, temp_registry_path):
        """Test loading registry with missing required fields."""
        invalid_registry = {"version": "1.0"}  # Missing metadata and nodes

        with open(temp_registry_path, "w") as f:
            json.dump(invalid_registry, f)

        sync = DAGMetadataRegistrySync(temp_registry_path)

        with pytest.raises(RegistryValidationError, match="Missing required field"):
            sync.load_registry()

    def test_validate_registry_structure_valid(self, sample_registry_data):
        """Test validating valid registry structure."""
        sync = DAGMetadataRegistrySync()

        # Should not raise exception
        sync._validate_registry_structure(sample_registry_data)

    def test_validate_registry_structure_missing_metadata_field(
        self, sample_registry_data
    ):
        """Test validating registry with missing metadata field."""
        del sample_registry_data["metadata"]["total_pipelines"]

        sync = DAGMetadataRegistrySync()

        with pytest.raises(
            RegistryValidationError, match="Missing required metadata field"
        ):
            sync._validate_registry_structure(sample_registry_data)

    def test_validate_node_structure_valid(self, sample_registry_data):
        """Test validating valid node structure."""
        sync = DAGMetadataRegistrySync()
        node_data = sample_registry_data["nodes"]["xgb_simple_training"]

        # Should not raise exception
        sync._validate_node_structure("xgb_simple_training", node_data)

    def test_validate_node_structure_missing_field(self, sample_registry_data):
        """Test validating node with missing required field."""
        node_data = sample_registry_data["nodes"]["xgb_simple_training"]
        del node_data["title"]

        sync = DAGMetadataRegistrySync()

        with pytest.raises(RegistryValidationError, match="missing required field"):
            sync._validate_node_structure("xgb_simple_training", node_data)

    def test_validate_node_structure_missing_atomic_property(
        self, sample_registry_data
    ):
        """Test validating node with missing atomic property."""
        node_data = sample_registry_data["nodes"]["xgb_simple_training"]
        del node_data["atomic_properties"]["single_responsibility"]

        sync = DAGMetadataRegistrySync()

        with pytest.raises(RegistryValidationError, match="missing atomic property"):
            sync._validate_node_structure("xgb_simple_training", node_data)

    def test_sync_metadata_to_registry_new_pipeline(
        self, temp_registry_path, sample_enhanced_metadata
    ):
        """Test syncing new pipeline metadata to registry."""
        sync = DAGMetadataRegistrySync(temp_registry_path)

        # Sync metadata
        sync.sync_metadata_to_registry(sample_enhanced_metadata, "test/pipeline.py")

        # Verify registry was updated
        registry = sync.load_registry()
        atomic_id = sample_enhanced_metadata.zettelkasten_metadata.atomic_id

        assert atomic_id in registry["nodes"]
        node = registry["nodes"][atomic_id]
        assert node["description"] == sample_enhanced_metadata.description
        assert node["file"] == "test/pipeline.py"
        assert "last_updated" in node

    def test_sync_metadata_to_registry_update_existing(
        self, temp_registry_path, sample_registry_data, sample_enhanced_metadata
    ):
        """Test syncing updated pipeline metadata to registry."""
        # Create registry with existing data
        with open(temp_registry_path, "w") as f:
            json.dump(sample_registry_data, f)

        sync = DAGMetadataRegistrySync(temp_registry_path)

        # Update the enhanced metadata to match existing ID
        sample_enhanced_metadata.zettelkasten_metadata.atomic_id = "xgb_simple_training"
        sample_enhanced_metadata.description = "Updated description"

        # Sync metadata
        sync.sync_metadata_to_registry(sample_enhanced_metadata, "updated/pipeline.py")

        # Verify registry was updated
        registry = sync.load_registry()
        node = registry["nodes"]["xgb_simple_training"]
        assert node["description"] == "Updated description"
        assert node["file"] == "updated/pipeline.py"

    @patch("cursus.pipeline_catalog.shared_dags.registry_sync.logger")
    def test_sync_metadata_to_registry_error(
        self, mock_logger, temp_registry_path, sample_enhanced_metadata
    ):
        """Test error handling in sync_metadata_to_registry."""
        sync = DAGMetadataRegistrySync(temp_registry_path)

        # Mock save_registry to raise exception
        with patch.object(sync, "_save_registry", side_effect=Exception("Save error")):
            with pytest.raises(Exception, match="Save error"):
                sync.sync_metadata_to_registry(
                    sample_enhanced_metadata, "test/pipeline.py"
                )

            mock_logger.error.assert_called_once()

    def test_sync_registry_to_metadata_success(
        self, temp_registry_path, sample_registry_data
    ):
        """Test syncing registry data back to metadata."""
        with open(temp_registry_path, "w") as f:
            json.dump(sample_registry_data, f)

        sync = DAGMetadataRegistrySync(temp_registry_path)
        enhanced_metadata = sync.sync_registry_to_metadata("xgb_simple_training")

        assert enhanced_metadata is not None
        assert enhanced_metadata.description == "Simple XGBoost training pipeline"
        assert enhanced_metadata.framework == PipelineFramework.XGBOOST
        assert enhanced_metadata.complexity == ComplexityLevel.SIMPLE

        # Check Zettelkasten metadata
        zm = enhanced_metadata.zettelkasten_metadata
        assert zm.atomic_id == "xgb_simple_training"
        assert zm.framework_tags == ["xgboost"]
        assert zm.task_tags == ["training"]

    def test_sync_registry_to_metadata_not_found(
        self, temp_registry_path, sample_registry_data
    ):
        """Test syncing registry data for non-existent pipeline."""
        with open(temp_registry_path, "w") as f:
            json.dump(sample_registry_data, f)

        sync = DAGMetadataRegistrySync(temp_registry_path)
        enhanced_metadata = sync.sync_registry_to_metadata("non_existent_pipeline")

        assert enhanced_metadata is None

    @patch("cursus.pipeline_catalog.shared_dags.registry_sync.logger")
    def test_sync_registry_to_metadata_error(self, mock_logger, temp_registry_path):
        """Test error handling in sync_registry_to_metadata."""
        sync = DAGMetadataRegistrySync(temp_registry_path)

        # Mock load_registry to raise exception
        with patch.object(sync, "load_registry", side_effect=Exception("Load error")):
            result = sync.sync_registry_to_metadata("test_pipeline")

            assert result is None
            mock_logger.error.assert_called_once()

    def test_extract_zettelkasten_metadata_from_node(self, sample_registry_data):
        """Test extracting ZettelkastenMetadata from registry node."""
        sync = DAGMetadataRegistrySync()
        node = sample_registry_data["nodes"]["xgb_simple_training"]

        # Add some connections to test extraction
        node["connections"] = {
            "alternatives": [
                {"id": "alt_pipeline", "annotation": "Alternative approach"}
            ],
            "related": [
                {"id": "related_pipeline", "annotation": "Related functionality"}
            ],
        }

        zm = sync._extract_zettelkasten_metadata_from_node(node)

        assert zm.atomic_id == "xgb_simple_training"
        assert zm.title == "XGBoost Simple Training"
        assert zm.single_responsibility == "Train XGBoost model"
        assert zm.framework == "xgboost"
        assert zm.complexity == "simple"
        assert zm.framework_tags == ["xgboost"]
        assert zm.task_tags == ["training"]

        # Check connections were extracted
        assert "alternatives" in zm.manual_connections
        assert "alt_pipeline" in zm.manual_connections["alternatives"]
        assert zm.curated_connections["alt_pipeline"] == "Alternative approach"

    def test_validate_consistency_success(
        self, temp_registry_path, sample_registry_data, sample_enhanced_metadata
    ):
        """Test successful consistency validation."""
        with open(temp_registry_path, "w") as f:
            json.dump(sample_registry_data, f)

        sync = DAGMetadataRegistrySync(temp_registry_path)

        # Update enhanced metadata to match registry
        sample_enhanced_metadata.description = "Simple XGBoost training pipeline"
        sample_enhanced_metadata.zettelkasten_metadata.atomic_id = "xgb_simple_training"
        sample_enhanced_metadata.zettelkasten_metadata.single_responsibility = (
            "Train XGBoost model"
        )

        errors = sync.validate_consistency(
            sample_enhanced_metadata, "xgb_simple_training"
        )
        assert len(errors) == 0

    def test_validate_consistency_mismatches(
        self, temp_registry_path, sample_registry_data, sample_enhanced_metadata
    ):
        """Test consistency validation with mismatches."""
        with open(temp_registry_path, "w") as f:
            json.dump(sample_registry_data, f)

        sync = DAGMetadataRegistrySync(temp_registry_path)

        # Keep mismatched data in enhanced metadata
        errors = sync.validate_consistency(
            sample_enhanced_metadata, "xgb_simple_training"
        )

        # Should find mismatches
        assert len(errors) > 0
        assert any("Description mismatch" in error for error in errors)
        assert any("Atomic ID mismatch" in error for error in errors)

    def test_validate_consistency_pipeline_not_found(
        self, temp_registry_path, sample_enhanced_metadata
    ):
        """Test consistency validation for non-existent pipeline."""
        sync = DAGMetadataRegistrySync(temp_registry_path)

        errors = sync.validate_consistency(
            sample_enhanced_metadata, "non_existent_pipeline"
        )

        assert len(errors) == 1
        assert "not found in registry" in errors[0]

    def test_update_registry_metadata(self, sample_registry_data):
        """Test updating registry-level metadata."""
        sync = DAGMetadataRegistrySync()

        # Add another node to test statistics
        sample_registry_data["nodes"]["pytorch_training"] = {
            "zettelkasten_metadata": {"framework": "pytorch", "complexity": "standard"},
            "connections": {
                "alternatives": [
                    {"id": "xgb_simple_training", "annotation": "XGBoost alternative"}
                ]
            },
        }

        sync._update_registry_metadata(sample_registry_data)

        metadata = sample_registry_data["metadata"]
        assert metadata["total_pipelines"] == 2
        assert "pytorch" in metadata["frameworks"]
        assert "standard" in metadata["complexity_levels"]
        assert "last_updated" in metadata

        # Check connection graph metadata
        graph_meta = sample_registry_data["connection_graph_metadata"]
        assert graph_meta["total_connections"] == 1
        assert graph_meta["independent_pipelines"] == 2
        assert graph_meta["connection_density"] > 0

    def test_update_tag_index(self, sample_registry_data):
        """Test updating tag index."""
        sync = DAGMetadataRegistrySync()

        # Create ZettelkastenMetadata with tags
        zm = ZettelkastenMetadata(
            atomic_id="test_pipeline",
            single_responsibility="Test pipeline",
            framework_tags=["pytorch"],
            task_tags=["evaluation"],
            complexity_tags=["advanced"],
            domain_tags=["nlp"],
        )

        sync._update_tag_index(sample_registry_data, "test_pipeline", zm)

        tag_index = sample_registry_data["tag_index"]

        # Check new tags were added
        assert "pytorch" in tag_index["framework_tags"]
        assert "test_pipeline" in tag_index["framework_tags"]["pytorch"]
        assert "evaluation" in tag_index["task_tags"]
        assert "test_pipeline" in tag_index["task_tags"]["evaluation"]
        assert "nlp" in tag_index["domain_tags"]
        assert "test_pipeline" in tag_index["domain_tags"]["nlp"]

    def test_remove_pipeline_from_registry_success(
        self, temp_registry_path, sample_registry_data
    ):
        """Test successful pipeline removal."""
        with open(temp_registry_path, "w") as f:
            json.dump(sample_registry_data, f)

        sync = DAGMetadataRegistrySync(temp_registry_path)
        result = sync.remove_pipeline_from_registry("xgb_simple_training")

        assert result is True

        # Verify pipeline was removed
        registry = sync.load_registry()
        assert "xgb_simple_training" not in registry["nodes"]
        assert registry["metadata"]["total_pipelines"] == 0

    def test_remove_pipeline_from_registry_not_found(
        self, temp_registry_path, sample_registry_data
    ):
        """Test removing non-existent pipeline."""
        with open(temp_registry_path, "w") as f:
            json.dump(sample_registry_data, f)

        sync = DAGMetadataRegistrySync(temp_registry_path)
        result = sync.remove_pipeline_from_registry("non_existent_pipeline")

        assert result is False

    @patch("cursus.pipeline_catalog.shared_dags.registry_sync.logger")
    def test_remove_pipeline_from_registry_error(self, mock_logger, temp_registry_path):
        """Test error handling in remove_pipeline_from_registry."""
        sync = DAGMetadataRegistrySync(temp_registry_path)

        # Mock load_registry to raise exception
        with patch.object(sync, "load_registry", side_effect=Exception("Load error")):
            result = sync.remove_pipeline_from_registry("test_pipeline")

            assert result is False
            mock_logger.error.assert_called_once()

    def test_remove_from_tag_index(self, sample_registry_data):
        """Test removing pipeline from tag index."""
        sync = DAGMetadataRegistrySync()

        # Add pipeline to multiple tags
        tag_index = sample_registry_data["tag_index"]
        tag_index["framework_tags"]["pytorch"] = [
            "xgb_simple_training",
            "other_pipeline",
        ]
        tag_index["task_tags"]["evaluation"] = ["xgb_simple_training"]

        sync._remove_from_tag_index(sample_registry_data, "xgb_simple_training")

        # Check pipeline was removed from all tags
        assert "xgb_simple_training" not in tag_index["framework_tags"]["pytorch"]
        assert (
            "other_pipeline" in tag_index["framework_tags"]["pytorch"]
        )  # Other pipeline should remain

        # Empty tag should be removed
        assert "evaluation" not in tag_index["task_tags"]

    def test_get_registry_statistics(self, temp_registry_path, sample_registry_data):
        """Test getting registry statistics."""
        with open(temp_registry_path, "w") as f:
            json.dump(sample_registry_data, f)

        sync = DAGMetadataRegistrySync(temp_registry_path)
        stats = sync.get_registry_statistics()

        assert stats["total_pipelines"] == 1
        assert "xgboost" in stats["frameworks"]
        assert "simple" in stats["complexity_levels"]
        assert stats["total_connections"] == 0
        assert stats["connection_density"] == 0.0
        assert "xgb_simple_training" in stats["isolated_nodes"]

        # Check tag statistics
        assert "tag_statistics" in stats
        assert "framework_tags" in stats["tag_statistics"]
        assert stats["tag_statistics"]["framework_tags"]["total_tags"] == 1

        # Check most common tags
        most_common = stats["tag_statistics"]["framework_tags"]["most_common_tags"]
        assert len(most_common) == 1
        assert most_common[0][0] == "xgboost"
        assert most_common[0][1] == 1

    @patch("cursus.pipeline_catalog.shared_dags.registry_sync.logger")
    def test_get_registry_statistics_error(self, mock_logger, temp_registry_path):
        """Test error handling in get_registry_statistics."""
        sync = DAGMetadataRegistrySync(temp_registry_path)

        # Mock load_registry to raise exception
        with patch.object(sync, "load_registry", side_effect=Exception("Load error")):
            stats = sync.get_registry_statistics()

            assert "error" in stats
            assert stats["error"] == "Load error"
            mock_logger.error.assert_called_once()

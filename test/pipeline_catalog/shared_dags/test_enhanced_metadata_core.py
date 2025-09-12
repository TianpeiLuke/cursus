"""
Core tests for EnhancedDAGMetadata class.

Tests the main functionality of the EnhancedDAGMetadata class including
creation, validation, conversion methods, and core operations.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
from datetime import datetime

from cursus.pipeline_catalog.shared_dags.enhanced_metadata import (
    EnhancedDAGMetadata,
    ZettelkastenMetadata,
    ComplexityLevel,
    PipelineFramework,
    DAGMetadataAdapter,
    validate_enhanced_dag_metadata,
)
from cursus.pipeline_catalog.shared_dags import DAGMetadata


class TestEnhancedDAGMetadata:
    """Test suite for EnhancedDAGMetadata class."""

    @pytest.fixture
    def sample_enhanced_metadata(self):
        """Create sample EnhancedDAGMetadata for testing."""
        return EnhancedDAGMetadata(
            description="XGBoost training pipeline for tabular data",
            complexity=ComplexityLevel.SIMPLE,
            features=["training"],
            framework=PipelineFramework.XGBOOST,
            node_count=3,
            edge_count=2,
        )

    @pytest.fixture
    def sample_zettelkasten_metadata(self):
        """Create sample ZettelkastenMetadata for testing."""
        return ZettelkastenMetadata(
            atomic_id="custom_pipeline",
            title="Custom Pipeline",
            single_responsibility="Custom functionality",
            framework="pytorch",
            complexity="standard",
            features=["training", "evaluation"],
            framework_tags=["pytorch"],
            task_tags=["training", "evaluation"],
            complexity_tags=["standard"],
        )

    def test_enhanced_dag_metadata_creation(self, sample_enhanced_metadata):
        """Test creating EnhancedDAGMetadata instance."""
        edm = sample_enhanced_metadata

        assert edm.description == "XGBoost training pipeline for tabular data"
        assert edm.complexity == ComplexityLevel.SIMPLE
        assert edm.features == ["training"]
        assert edm.framework == PipelineFramework.XGBOOST
        assert edm.node_count == 3
        assert edm.edge_count == 2

        # Should have auto-created Zettelkasten metadata
        assert edm.zettelkasten_metadata is not None
        assert edm.zettelkasten_metadata.atomic_id == "xgboost_training_simple"
        assert edm.zettelkasten_metadata.framework == "xgboost"
        assert edm.zettelkasten_metadata.complexity == "simple"

    def test_enhanced_dag_metadata_with_custom_zettelkasten(
        self, sample_zettelkasten_metadata
    ):
        """Test creating EnhancedDAGMetadata with custom ZettelkastenMetadata."""
        edm = EnhancedDAGMetadata(
            description="Custom pipeline description",
            complexity=ComplexityLevel.STANDARD,
            features=["training", "evaluation"],
            framework=PipelineFramework.PYTORCH,
            node_count=5,
            edge_count=4,
            zettelkasten_metadata=sample_zettelkasten_metadata,
        )

        assert edm.zettelkasten_metadata == sample_zettelkasten_metadata
        assert edm.zettelkasten_metadata.atomic_id == "custom_pipeline"

    def test_complexity_validation_enum(self):
        """Test complexity validation with enum values."""
        # Valid enum value
        edm = EnhancedDAGMetadata(
            description="Test pipeline",
            complexity=ComplexityLevel.ADVANCED,
            features=["training"],
            framework=PipelineFramework.XGBOOST,
            node_count=1,
            edge_count=0,
        )
        assert edm.complexity == ComplexityLevel.ADVANCED

    def test_complexity_validation_string(self):
        """Test complexity validation with string values."""
        # Valid string value
        edm = EnhancedDAGMetadata(
            description="Test pipeline",
            complexity="standard",
            features=["training"],
            framework=PipelineFramework.XGBOOST,
            node_count=1,
            edge_count=0,
        )
        assert edm.complexity == "standard"

    def test_framework_validation_enum(self):
        """Test framework validation with enum values."""
        # Valid enum value
        edm = EnhancedDAGMetadata(
            description="Test pipeline",
            complexity=ComplexityLevel.SIMPLE,
            features=["training"],
            framework=PipelineFramework.PYTORCH,
            node_count=1,
            edge_count=0,
        )
        assert edm.framework == PipelineFramework.PYTORCH

    def test_framework_validation_string(self):
        """Test framework validation with string values."""
        # Valid string value
        edm = EnhancedDAGMetadata(
            description="Test pipeline",
            complexity=ComplexityLevel.SIMPLE,
            features=["training"],
            framework="sklearn",
            node_count=1,
            edge_count=0,
        )
        assert edm.framework == "sklearn"

    def test_generate_atomic_id(self, sample_enhanced_metadata):
        """Test atomic ID generation."""
        edm = sample_enhanced_metadata
        atomic_id = edm._generate_atomic_id()

        assert atomic_id == "xgboost_training_simple"

    def test_generate_atomic_id_with_enum(self):
        """Test atomic ID generation with enum values."""
        edm = EnhancedDAGMetadata(
            description="Test pipeline",
            complexity=ComplexityLevel.ADVANCED,
            features=["evaluation", "training"],
            framework=PipelineFramework.PYTORCH,
            node_count=1,
            edge_count=0,
        )

        atomic_id = edm._generate_atomic_id()
        assert atomic_id == "pytorch_evaluation_advanced"

    def test_generate_title(self, sample_enhanced_metadata):
        """Test title generation."""
        edm = sample_enhanced_metadata
        title = edm._generate_title()

        assert title == "Xgboost Training Simple"

    def test_generate_title_with_enum(self):
        """Test title generation with enum values."""
        edm = EnhancedDAGMetadata(
            description="Test pipeline",
            complexity=ComplexityLevel.ADVANCED,
            features=["evaluation"],
            framework=PipelineFramework.PYTORCH,
            node_count=1,
            edge_count=0,
        )

        title = edm._generate_title()
        assert title == "Pytorch Evaluation Advanced"

    def test_to_registry_node(self, sample_enhanced_metadata):
        """Test converting to registry node format."""
        edm = sample_enhanced_metadata
        node = edm.to_registry_node()

        assert node["id"] == "xgboost_training_simple"
        assert node["title"] == "Xgboost Training Simple"
        assert node["description"] == "XGBoost training pipeline for tabular data"

        # Check atomic properties
        assert "atomic_properties" in node
        atomic_props = node["atomic_properties"]
        assert (
            atomic_props["single_responsibility"]
            == "XGBoost training pipeline for tabular data"
        )
        assert atomic_props["independence_level"] == "fully_self_contained"
        assert atomic_props["node_count"] == 1
        assert atomic_props["edge_count"] == 0

        # Check zettelkasten metadata
        assert "zettelkasten_metadata" in node
        zk_meta = node["zettelkasten_metadata"]
        assert zk_meta["framework"] == "xgboost"
        assert zk_meta["complexity"] == "simple"
        assert zk_meta["features"] == ["training"]

        # Check multi-dimensional tags
        assert "multi_dimensional_tags" in node
        tags = node["multi_dimensional_tags"]
        assert tags["framework_tags"] == ["xgboost"]
        assert tags["task_tags"] == ["training"]
        assert tags["complexity_tags"] == ["simple"]

    def test_add_connection(self, sample_enhanced_metadata):
        """Test adding connections through EnhancedDAGMetadata."""
        edm = sample_enhanced_metadata

        edm.add_connection(
            target_id="pytorch_training",
            connection_type="alternatives",
            annotation="Alternative framework",
            confidence=0.8,
        )

        # Verify connection was added to zettelkasten metadata
        zm = edm.zettelkasten_metadata
        assert "alternatives" in zm.manual_connections
        assert "pytorch_training" in zm.manual_connections["alternatives"]
        assert zm.curated_connections["pytorch_training"] == "Alternative framework"
        assert zm.connection_confidence["pytorch_training"] == 0.8

    def test_update_tags(self, sample_enhanced_metadata):
        """Test updating tags through EnhancedDAGMetadata."""
        edm = sample_enhanced_metadata

        # Update domain tags
        edm.update_tags("domain_tags", ["tabular", "structured"])

        assert edm.zettelkasten_metadata.domain_tags == ["tabular", "structured"]

    def test_update_tags_invalid_category(self, sample_enhanced_metadata):
        """Test updating tags with invalid category."""
        edm = sample_enhanced_metadata

        with pytest.raises(ValueError, match="Invalid tag category"):
            edm.update_tags("invalid_category", ["tag1", "tag2"])

    def test_to_dict(self, sample_enhanced_metadata):
        """Test converting to dictionary format."""
        edm = sample_enhanced_metadata
        data_dict = edm.to_dict()

        # Should include base DAGMetadata fields
        assert data_dict["description"] == "XGBoost training pipeline for tabular data"
        assert data_dict["complexity"] == ComplexityLevel.SIMPLE
        assert data_dict["features"] == ["training"]
        assert data_dict["framework"] == PipelineFramework.XGBOOST
        assert data_dict["node_count"] == 3
        assert data_dict["edge_count"] == 2

        # Should include Zettelkasten metadata
        assert "zettelkasten_metadata" in data_dict
        assert data_dict["zettelkasten_metadata"] is not None

    def test_to_legacy_dag_metadata(self, sample_enhanced_metadata):
        """Test converting back to legacy DAGMetadata format."""
        edm = sample_enhanced_metadata
        legacy = edm.to_legacy_dag_metadata()

        assert isinstance(legacy, DAGMetadata)
        assert legacy.description == "XGBoost training pipeline for tabular data"
        assert legacy.complexity == "simple"
        assert legacy.features == ["training"]
        assert legacy.framework == "xgboost"
        assert legacy.node_count == 3
        assert legacy.edge_count == 2

    def test_build_connections(self, sample_enhanced_metadata):
        """Test building connections from manual linking metadata."""
        edm = sample_enhanced_metadata

        # Add some connections
        edm.add_connection("alt_pipeline", "alternatives", "Alternative approach")
        edm.add_connection("related_pipeline", "related", "Related functionality")

        connections = edm._build_connections()

        assert "alternatives" in connections
        assert "related" in connections
        assert "used_in" in connections

        assert len(connections["alternatives"]) == 1
        assert connections["alternatives"][0]["id"] == "alt_pipeline"
        assert connections["alternatives"][0]["annotation"] == "Alternative approach"

        assert len(connections["related"]) == 1
        assert connections["related"][0]["id"] == "related_pipeline"
        assert connections["related"][0]["annotation"] == "Related functionality"

    def test_extract_dependencies(self, sample_enhanced_metadata):
        """Test extracting dependencies from framework and features."""
        edm = sample_enhanced_metadata
        deps = edm._extract_dependencies()

        # Should include framework dependencies
        assert "xgboost" in deps
        assert "sagemaker" in deps

        # Should not have duplicates
        assert len(deps) == len(set(deps))

    def test_extract_dependencies_pytorch(self):
        """Test extracting dependencies for PyTorch framework."""
        edm = EnhancedDAGMetadata(
            description="PyTorch pipeline",
            complexity=ComplexityLevel.SIMPLE,
            features=["training", "calibration"],
            framework=PipelineFramework.PYTORCH,
            node_count=1,
            edge_count=0,
        )

        deps = edm._extract_dependencies()

        assert "torch" in deps
        assert "sagemaker" in deps
        assert "sklearn" in deps  # From calibration feature


class TestDAGMetadataAdapter:
    """Test suite for DAGMetadataAdapter class."""

    @pytest.fixture
    def legacy_metadata(self):
        """Create legacy DAGMetadata for testing."""
        return DAGMetadata(
            description="Legacy XGBoost training pipeline",
            complexity="standard",
            features=["training", "evaluation"],
            framework="xgboost",
            node_count=4,
            edge_count=3,
            extra_metadata={
                "input_interface": ["data", "config"],
                "output_interface": ["model", "metrics"],
            },
        )

    def test_from_legacy_dag_metadata(self, legacy_metadata):
        """Test converting from legacy DAGMetadata to EnhancedDAGMetadata."""
        enhanced = DAGMetadataAdapter.from_legacy_dag_metadata(legacy_metadata)

        assert isinstance(enhanced, EnhancedDAGMetadata)
        assert enhanced.description == "Legacy XGBoost training pipeline"
        assert enhanced.complexity == ComplexityLevel.STANDARD
        assert enhanced.features == ["training", "evaluation"]
        assert enhanced.framework == PipelineFramework.XGBOOST
        assert enhanced.node_count == 4
        assert enhanced.edge_count == 3

        # Check Zettelkasten metadata was created
        zm = enhanced.zettelkasten_metadata
        assert zm.atomic_id == "xgboost_training_standard"
        assert zm.single_responsibility == "Legacy XGBoost training pipeline"
        assert zm.input_interface == ["data", "config"]
        assert zm.output_interface == ["model", "metrics"]
        assert zm.framework_tags == ["xgboost"]
        assert zm.task_tags == ["training", "evaluation"]
        assert zm.complexity_tags == ["standard"]

    def test_from_legacy_dag_metadata_unknown_values(self):
        """Test converting legacy metadata with unknown complexity/framework."""
        legacy = DAGMetadata(
            description="Unknown pipeline",
            complexity="unknown_complexity",
            features=["unknown_feature"],
            framework="unknown_framework",
            node_count=1,
            edge_count=0,
        )

        enhanced = DAGMetadataAdapter.from_legacy_dag_metadata(legacy)

        # Should default to standard complexity and generic framework
        assert enhanced.complexity == ComplexityLevel.STANDARD
        assert enhanced.framework == PipelineFramework.GENERIC

        zm = enhanced.zettelkasten_metadata
        assert zm.atomic_id == "generic_unknown_feature_standard"

    def test_to_legacy_dag_metadata(self, legacy_metadata):
        """Test converting back to legacy format."""
        enhanced = DAGMetadataAdapter.from_legacy_dag_metadata(legacy_metadata)
        converted_back = DAGMetadataAdapter.to_legacy_dag_metadata(enhanced)

        assert isinstance(converted_back, DAGMetadata)
        assert converted_back.description == legacy_metadata.description
        assert converted_back.complexity == legacy_metadata.complexity
        assert converted_back.features == legacy_metadata.features
        assert converted_back.framework == legacy_metadata.framework
        assert converted_back.node_count == legacy_metadata.node_count
        assert converted_back.edge_count == legacy_metadata.edge_count


class TestValidateEnhancedDAGMetadata:
    """Test suite for validate_enhanced_dag_metadata function."""

    @pytest.fixture
    def valid_enhanced_metadata(self):
        """Create valid EnhancedDAGMetadata for testing."""
        return EnhancedDAGMetadata(
            description="Valid XGBoost training pipeline",
            complexity=ComplexityLevel.SIMPLE,
            features=["training"],
            framework=PipelineFramework.XGBOOST,
            node_count=3,
            edge_count=2,
        )

    def test_validate_enhanced_dag_metadata_valid(self, valid_enhanced_metadata):
        """Test validation of valid enhanced metadata."""
        result = validate_enhanced_dag_metadata(valid_enhanced_metadata)
        assert result is True

    def test_validate_enhanced_dag_metadata_invalid_connection_type(
        self, valid_enhanced_metadata
    ):
        """Test validation with invalid connection type."""
        # Add invalid connection type
        valid_enhanced_metadata.zettelkasten_metadata.manual_connections[
            "invalid_type"
        ] = ["target"]

        with pytest.raises(ValueError, match="Invalid connection type"):
            validate_enhanced_dag_metadata(valid_enhanced_metadata)

    def test_validate_enhanced_dag_metadata_missing_annotation(
        self, valid_enhanced_metadata
    ):
        """Test validation with missing connection annotation."""
        # Add connection without annotation
        zm = valid_enhanced_metadata.zettelkasten_metadata
        zm.manual_connections["alternatives"] = ["target_pipeline"]
        # Don't add to curated_connections (missing annotation)

        with patch(
            "cursus.pipeline_catalog.shared_dags.enhanced_metadata.logger"
        ) as mock_logger:
            result = validate_enhanced_dag_metadata(valid_enhanced_metadata)
            assert result is True
            mock_logger.warning.assert_called_once()

    def test_validate_enhanced_dag_metadata_validation_error(
        self, valid_enhanced_metadata
    ):
        """Test validation with core validation error."""
        # Make metadata invalid by clearing features
        valid_enhanced_metadata.features = []

        with pytest.raises(ValueError, match="Features list cannot be empty"):
            validate_enhanced_dag_metadata(valid_enhanced_metadata)

    @patch("cursus.pipeline_catalog.shared_dags.enhanced_metadata.logger")
    def test_validate_enhanced_dag_metadata_exception(
        self, mock_logger, valid_enhanced_metadata
    ):
        """Test validation with unexpected exception."""
        # Mock _validate to raise exception
        with patch.object(
            valid_enhanced_metadata,
            "_validate",
            side_effect=Exception("Validation error"),
        ):
            with pytest.raises(Exception, match="Validation error"):
                validate_enhanced_dag_metadata(valid_enhanced_metadata)

            mock_logger.error.assert_called_once()

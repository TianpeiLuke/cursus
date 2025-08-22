"""
Unit tests for enhanced metadata model classes.

Tests the enum classes and basic model functionality for the enhanced
metadata system.
"""

import pytest
from unittest.mock import patch
from src.cursus.pipeline_catalog.shared_dags.enhanced_metadata import (
    ComplexityLevel, PipelineFramework, ZettelkastenMetadata
)


class TestComplexityLevel:
    """Test suite for ComplexityLevel enum."""
    
    def test_complexity_level_values(self):
        """Test ComplexityLevel enum values."""
        assert ComplexityLevel.SIMPLE.value == "simple"
        assert ComplexityLevel.STANDARD.value == "standard"
        assert ComplexityLevel.ADVANCED.value == "advanced"
        assert ComplexityLevel.COMPREHENSIVE.value == "comprehensive"


class TestPipelineFramework:
    """Test suite for PipelineFramework enum."""
    
    def test_pipeline_framework_values(self):
        """Test PipelineFramework enum values."""
        assert PipelineFramework.XGBOOST.value == "xgboost"
        assert PipelineFramework.PYTORCH.value == "pytorch"
        assert PipelineFramework.SKLEARN.value == "sklearn"
        assert PipelineFramework.GENERIC.value == "generic"
        assert PipelineFramework.FRAMEWORK_AGNOSTIC.value == "framework_agnostic"


class TestZettelkastenMetadata:
    """Test suite for ZettelkastenMetadata class."""
    
    @pytest.fixture
    def sample_zettelkasten_metadata(self):
        """Create sample ZettelkastenMetadata for testing."""
        return ZettelkastenMetadata(
            atomic_id="xgb_simple_training",
            title="XGBoost Simple Training",
            single_responsibility="Train XGBoost model on tabular data",
            framework="xgboost",
            complexity="simple",
            features=["training"],
            framework_tags=["xgboost"],
            task_tags=["training"],
            complexity_tags=["simple"],
            source_file="xgboost/simple_training.py"
        )
    
    def test_zettelkasten_metadata_creation(self, sample_zettelkasten_metadata):
        """Test creating ZettelkastenMetadata instance."""
        zm = sample_zettelkasten_metadata
        
        assert zm.atomic_id == "xgb_simple_training"
        assert zm.title == "XGBoost Simple Training"
        assert zm.single_responsibility == "Train XGBoost model on tabular data"
        assert zm.framework == "xgboost"
        assert zm.complexity == "simple"
        assert zm.features == ["training"]
        assert zm.framework_tags == ["xgboost"]
        assert zm.task_tags == ["training"]
        assert zm.complexity_tags == ["simple"]
        assert zm.source_file == "xgboost/simple_training.py"
    
    def test_zettelkasten_metadata_defaults(self):
        """Test ZettelkastenMetadata with default values."""
        zm = ZettelkastenMetadata(
            atomic_id="test_pipeline",
            single_responsibility="Test functionality"
        )
        
        assert zm.atomic_id == "test_pipeline"
        assert zm.single_responsibility == "Test functionality"
        assert zm.title == ""
        assert zm.input_interface == []
        assert zm.output_interface == []
        assert zm.side_effects == "none"
        assert zm.independence_level == "fully_self_contained"
        assert zm.node_count == 1
        assert zm.edge_count == 0
        assert zm.pattern_tags == ["atomic_workflow", "independent"]  # Default pattern tags
    
    def test_validate_atomic_id(self):
        """Test atomic_id validation."""
        # Valid atomic_id
        zm = ZettelkastenMetadata(
            atomic_id="valid_id",
            single_responsibility="Test"
        )
        assert zm.atomic_id == "valid_id"
        
        # Invalid atomic_id (empty string)
        with pytest.raises(ValueError, match="atomic_id must be a non-empty string"):
            ZettelkastenMetadata(
                atomic_id="",
                single_responsibility="Test"
            )
    
    def test_validate_single_responsibility_verbose(self):
        """Test single_responsibility validation for verbose descriptions."""
        verbose_responsibility = "This is a very long and verbose single responsibility description that exceeds the recommended fifteen word limit for concise descriptions"
        
        with patch('src.cursus.pipeline_catalog.shared_dags.enhanced_metadata.logger') as mock_logger:
            zm = ZettelkastenMetadata(
                atomic_id="test_pipeline",
                single_responsibility=verbose_responsibility
            )
            
            assert zm.single_responsibility == verbose_responsibility
            mock_logger.warning.assert_called_once()
    
    def test_add_connection(self, sample_zettelkasten_metadata):
        """Test adding manual connections."""
        zm = sample_zettelkasten_metadata
        
        # Add connection
        zm.add_connection(
            target_id="pytorch_simple_training",
            connection_type="alternatives",
            annotation="Alternative framework approach",
            confidence=0.9
        )
        
        # Verify connection was added
        assert "alternatives" in zm.manual_connections
        assert "pytorch_simple_training" in zm.manual_connections["alternatives"]
        assert zm.curated_connections["pytorch_simple_training"] == "Alternative framework approach"
        assert zm.connection_confidence["pytorch_simple_training"] == 0.9
    
    def test_add_connection_invalid_type(self, sample_zettelkasten_metadata):
        """Test adding connection with invalid type."""
        zm = sample_zettelkasten_metadata
        
        with pytest.raises(ValueError, match="Invalid connection type"):
            zm.add_connection(
                target_id="target_pipeline",
                connection_type="invalid_type",
                annotation="Invalid connection"
            )
    
    def test_add_connection_duplicate(self, sample_zettelkasten_metadata):
        """Test adding duplicate connection."""
        zm = sample_zettelkasten_metadata
        
        # Add connection twice
        zm.add_connection("target", "alternatives", "First annotation")
        zm.add_connection("target", "alternatives", "Second annotation")
        
        # Should not duplicate the target_id
        assert zm.manual_connections["alternatives"].count("target") == 1
        # Should update the annotation
        assert zm.curated_connections["target"] == "Second annotation"
    
    def test_get_all_tags(self, sample_zettelkasten_metadata):
        """Test getting all tags organized by category."""
        zm = sample_zettelkasten_metadata
        zm.domain_tags = ["tabular"]
        zm.pattern_tags = ["atomic_workflow"]
        
        all_tags = zm.get_all_tags()
        
        assert "framework_tags" in all_tags
        assert "task_tags" in all_tags
        assert "complexity_tags" in all_tags
        assert "domain_tags" in all_tags
        assert "pattern_tags" in all_tags
        
        assert all_tags["framework_tags"] == ["xgboost"]
        assert all_tags["task_tags"] == ["training"]
        assert all_tags["complexity_tags"] == ["simple"]
        assert all_tags["domain_tags"] == ["tabular"]
        assert all_tags["pattern_tags"] == ["atomic_workflow"]
    
    def test_get_flat_tags(self, sample_zettelkasten_metadata):
        """Test getting all tags as flat list."""
        zm = sample_zettelkasten_metadata
        zm.domain_tags = ["tabular"]
        zm.pattern_tags = ["atomic_workflow", "independent"]
        
        flat_tags = zm.get_flat_tags()
        
        expected_tags = ["xgboost", "training", "simple", "tabular", "atomic_workflow", "independent"]
        assert set(flat_tags) == set(expected_tags)
        assert len(flat_tags) == len(set(flat_tags))  # No duplicates

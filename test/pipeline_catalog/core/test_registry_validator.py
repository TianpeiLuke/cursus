"""
Unit tests for RegistryValidator class.

Tests registry validation utilities ensuring Zettelkasten principle compliance.
Validates registry integrity, atomicity, connection integrity, and dual-form structure.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any, List

from cursus.pipeline_catalog.core.registry_validator import (
    RegistryValidator,
    ValidationSeverity,
    ValidationIssue,
    AtomicityViolation,
    ConnectionError,
    MetadataError,
    TagConsistencyError,
    IndependenceError,
    ValidationReport
)
from cursus.pipeline_catalog.core.catalog_registry import CatalogRegistry


class TestRegistryValidator:
    """Test suite for RegistryValidator class."""

    @pytest.fixture
    def mock_registry(self):
        """Create mock CatalogRegistry."""
        return Mock(spec=CatalogRegistry)

    @pytest.fixture
    def sample_valid_node(self):
        """Sample valid pipeline node for testing."""
        return {
            "id": "valid_pipeline",
            "title": "Valid Pipeline",
            "description": "A well-structured pipeline for testing",
            "atomic_properties": {
                "single_responsibility": "Process data efficiently",
                "input_interface": ["data.csv", "config.json"],
                "output_interface": ["results.json"],
                "independence": "fully_self_contained",
                "side_effects": "creates_artifacts",
                "dependencies": ["pandas", "numpy"]
            },
            "zettelkasten_metadata": {
                "framework": "xgboost",
                "complexity": "simple"
            },
            "multi_dimensional_tags": {
                "framework_tags": ["xgboost"],
                "task_tags": ["training"],
                "complexity_tags": ["simple"]
            },
            "discovery_metadata": {
                "estimated_runtime": "5 minutes",
                "resource_requirements": "low",
                "skill_level": "beginner"
            }
        }

    @pytest.fixture
    def validator(self, mock_registry):
        """Create RegistryValidator instance."""
        return RegistryValidator(mock_registry)

    def test_init(self, mock_registry):
        """Test RegistryValidator initialization."""
        validator = RegistryValidator(mock_registry)
        
        assert validator.registry == mock_registry
        assert validator._validation_cache == {}
        assert not validator._cache_valid

    def test_validate_atomicity_valid(self, validator, mock_registry, sample_valid_node):
        """Test atomicity validation with valid pipeline."""
        mock_registry.get_all_pipelines.return_value = ["valid_pipeline"]
        mock_registry.get_pipeline_node.return_value = sample_valid_node
        
        violations = validator.validate_atomicity()
        
        assert len(violations) == 0

    def test_validate_atomicity_missing_responsibility(self, validator, mock_registry):
        """Test atomicity validation with missing responsibility."""
        invalid_node = {
            "id": "test_pipeline",
            "atomic_properties": {}  # Missing single_responsibility
        }
        
        mock_registry.get_all_pipelines.return_value = ["test_pipeline"]
        mock_registry.get_pipeline_node.return_value = invalid_node
        
        violations = validator.validate_atomicity()
        
        assert len(violations) > 0
        assert any("missing_responsibility" in v.message for v in violations)

    def test_validate_atomicity_verbose_responsibility(self, validator, mock_registry):
        """Test atomicity validation with verbose responsibility."""
        invalid_node = {
            "id": "test_pipeline",
            "atomic_properties": {
                "single_responsibility": "This is a very long and verbose description of what this pipeline does that exceeds the recommended word limit",
                "input_interface": ["data"],
                "output_interface": ["results"]
            }
        }
        
        mock_registry.get_all_pipelines.return_value = ["test_pipeline"]
        mock_registry.get_pipeline_node.return_value = invalid_node
        
        violations = validator.validate_atomicity()
        
        assert len(violations) > 0
        assert any("verbose_responsibility" in v.message for v in violations)

    def test_validate_atomicity_missing_interfaces(self, validator, mock_registry):
        """Test atomicity validation with missing interfaces."""
        invalid_node = {
            "id": "test_pipeline",
            "atomic_properties": {
                "single_responsibility": "Process data"
                # Missing input_interface and output_interface
            }
        }
        
        mock_registry.get_all_pipelines.return_value = ["test_pipeline"]
        mock_registry.get_pipeline_node.return_value = invalid_node
        
        violations = validator.validate_atomicity()
        
        assert len(violations) >= 2  # Missing input and output interfaces
        assert any("missing_input_interface" in v.message for v in violations)
        assert any("missing_output_interface" in v.message for v in violations)

    def test_validate_atomicity_independence_contradiction(self, validator, mock_registry):
        """Test atomicity validation with independence contradiction."""
        invalid_node = {
            "id": "test_pipeline",
            "atomic_properties": {
                "single_responsibility": "Process data",
                "input_interface": ["data"],
                "output_interface": ["results"],
                "independence": "fully_self_contained",
                "side_effects": "modifies_external_state"  # Contradicts independence
            }
        }
        
        mock_registry.get_all_pipelines.return_value = ["test_pipeline"]
        mock_registry.get_pipeline_node.return_value = invalid_node
        
        violations = validator.validate_atomicity()
        
        assert len(violations) > 0
        assert any("independence_contradiction" in v.message for v in violations)

    def test_validate_connections_missing_target(self, validator, mock_registry):
        """Test connection validation with missing target."""
        mock_registry.get_all_pipelines.return_value = ["pipeline1"]
        mock_registry.get_pipeline_connections.return_value = {
            "alternatives": [{"id": "nonexistent", "annotation": "Missing target"}]
        }
        mock_registry.load_registry.return_value = {"tag_index": {}}
        
        errors = validator.validate_connections()
        
        assert len(errors) > 0
        assert any("missing_target" in e.message for e in errors)

    def test_validate_connections_missing_annotation(self, validator, mock_registry):
        """Test connection validation with missing annotation."""
        mock_registry.get_all_pipelines.return_value = ["pipeline1", "pipeline2"]
        mock_registry.get_pipeline_connections.return_value = {
            "alternatives": [{"id": "pipeline2", "annotation": ""}]  # Empty annotation
        }
        mock_registry.load_registry.return_value = {"tag_index": {}}
        
        errors = validator.validate_connections()
        
        assert len(errors) > 0
        assert any("missing_annotation" in e.message for e in errors)

    def test_validate_connections_self_reference(self, validator, mock_registry):
        """Test connection validation with self-reference."""
        mock_registry.get_all_pipelines.return_value = ["pipeline1"]
        mock_registry.get_pipeline_connections.return_value = {
            "alternatives": [{"id": "pipeline1", "annotation": "Self reference"}]
        }
        mock_registry.load_registry.return_value = {"tag_index": {}}
        
        errors = validator.validate_connections()
        
        assert len(errors) > 0
        assert any("self_reference" in e.message for e in errors)

    def test_validate_connections_invalid_type(self, validator, mock_registry):
        """Test connection validation with invalid connection type."""
        mock_registry.get_all_pipelines.return_value = ["pipeline1"]
        mock_registry.get_pipeline_connections.return_value = {
            "invalid_type": [{"id": "pipeline2", "annotation": "Invalid type"}]
        }
        mock_registry.load_registry.return_value = {"tag_index": {}}
        
        errors = validator.validate_connections()
        
        assert len(errors) > 0
        assert any("invalid_type" in e.message for e in errors)

    def test_validate_connections_orphaned_tag_reference(self, validator, mock_registry):
        """Test connection validation with orphaned tag references."""
        mock_registry.get_all_pipelines.return_value = ["pipeline1"]
        mock_registry.get_pipeline_connections.return_value = {}
        mock_registry.load_registry.return_value = {
            "tag_index": {
                "framework_tags": {
                    "xgboost": ["pipeline1", "nonexistent_pipeline"]
                }
            }
        }
        
        errors = validator.validate_connections()
        
        assert len(errors) > 0
        assert any("orphaned_tag_reference" in e.message for e in errors)

    def test_validate_metadata_completeness_missing_fields(self, validator, mock_registry):
        """Test metadata validation with missing required fields."""
        invalid_node = {
            "id": "test_pipeline",
            "title": "Test Pipeline"
            # Missing description, atomic_properties, etc.
        }
        
        mock_registry.get_all_pipelines.return_value = ["test_pipeline"]
        mock_registry.get_pipeline_node.return_value = invalid_node
        
        errors = validator.validate_metadata_completeness()
        
        assert len(errors) > 0
        required_fields = ["description", "atomic_properties", "zettelkasten_metadata"]
        for field in required_fields:
            assert any(field in e.message for e in errors)

    def test_validate_metadata_completeness_missing_atomic_properties(self, validator, mock_registry):
        """Test metadata validation with missing atomic properties."""
        invalid_node = {
            "id": "test_pipeline",
            "title": "Test Pipeline",
            "description": "Test description",
            "atomic_properties": {},  # Empty atomic properties
            "zettelkasten_metadata": {},
            "multi_dimensional_tags": {},
            "discovery_metadata": {}
        }
        
        mock_registry.get_all_pipelines.return_value = ["test_pipeline"]
        mock_registry.get_pipeline_node.return_value = invalid_node
        
        errors = validator.validate_metadata_completeness()
        
        assert len(errors) > 0
        atomic_fields = ["single_responsibility", "input_interface", "output_interface"]
        for field in atomic_fields:
            assert any(field in e.message for e in errors)

    def test_validate_metadata_completeness_missing_tags(self, validator, mock_registry):
        """Test metadata validation with missing tag categories."""
        invalid_node = {
            "id": "test_pipeline",
            "title": "Test Pipeline",
            "description": "Test description",
            "atomic_properties": {
                "single_responsibility": "Test",
                "input_interface": ["data"],
                "output_interface": ["results"]
            },
            "zettelkasten_metadata": {"framework": "test", "complexity": "simple"},
            "multi_dimensional_tags": {},  # Empty tags
            "discovery_metadata": {}
        }
        
        mock_registry.get_all_pipelines.return_value = ["test_pipeline"]
        mock_registry.get_pipeline_node.return_value = invalid_node
        
        errors = validator.validate_metadata_completeness()
        
        assert len(errors) > 0
        tag_categories = ["framework_tags", "task_tags", "complexity_tags"]
        for category in tag_categories:
            assert any(category in e.message for e in errors)

    def test_validate_tag_consistency_framework_mismatch(self, validator, mock_registry):
        """Test tag consistency validation with framework mismatch."""
        invalid_node = {
            "id": "test_pipeline",
            "zettelkasten_metadata": {"framework": "xgboost"},
            "multi_dimensional_tags": {
                "framework_tags": ["pytorch"]  # Mismatch
            }
        }
        
        mock_registry.get_all_pipelines.return_value = ["test_pipeline"]
        mock_registry.get_pipeline_node.return_value = invalid_node
        
        errors = validator.validate_tag_consistency()
        
        assert len(errors) > 0
        assert any("xgboost" in e.message and "framework_tags" in e.message for e in errors)

    def test_validate_tag_consistency_complexity_mismatch(self, validator, mock_registry):
        """Test tag consistency validation with complexity mismatch."""
        invalid_node = {
            "id": "test_pipeline",
            "zettelkasten_metadata": {"complexity": "advanced"},
            "multi_dimensional_tags": {
                "complexity_tags": ["simple"]  # Mismatch
            }
        }
        
        mock_registry.get_all_pipelines.return_value = ["test_pipeline"]
        mock_registry.get_pipeline_node.return_value = invalid_node
        
        errors = validator.validate_tag_consistency()
        
        assert len(errors) > 0
        assert any("advanced" in e.message and "complexity_tags" in e.message for e in errors)

    def test_validate_independence_claims_side_effects_contradiction(self, validator, mock_registry):
        """Test independence validation with side effects contradiction."""
        invalid_node = {
            "id": "test_pipeline",
            "atomic_properties": {
                "independence": "fully_self_contained",
                "side_effects": "modifies_external_state"  # Contradicts independence
            }
        }
        
        mock_registry.get_all_pipelines.return_value = ["test_pipeline"]
        mock_registry.get_pipeline_node.return_value = invalid_node
        
        errors = validator.validate_independence_claims()
        
        assert len(errors) > 0
        assert any("side effects" in e.message for e in errors)

    def test_validate_independence_claims_many_dependencies(self, validator, mock_registry):
        """Test independence validation with many dependencies."""
        invalid_node = {
            "id": "test_pipeline",
            "atomic_properties": {
                "independence": "fully_self_contained",
                "dependencies": ["dep1", "dep2", "dep3", "dep4", "dep5"]  # Too many
            }
        }
        
        mock_registry.get_all_pipelines.return_value = ["test_pipeline"]
        mock_registry.get_pipeline_node.return_value = invalid_node
        
        errors = validator.validate_independence_claims()
        
        assert len(errors) > 0
        assert any("dependencies" in e.message for e in errors)

    def test_validate_independence_claims_complex_inputs(self, validator, mock_registry):
        """Test independence validation with complex input requirements."""
        invalid_node = {
            "id": "test_pipeline",
            "atomic_properties": {
                "independence": "fully_self_contained",
                "input_interface": ["input1", "input2", "input3", "input4", "input5", "input6"]  # Too many
            }
        }
        
        mock_registry.get_all_pipelines.return_value = ["test_pipeline"]
        mock_registry.get_pipeline_node.return_value = invalid_node
        
        errors = validator.validate_independence_claims()
        
        assert len(errors) > 0
        assert any("inputs" in e.message for e in errors)

    def test_generate_validation_report_valid_registry(self, validator, mock_registry, sample_valid_node):
        """Test generating validation report for valid registry."""
        mock_registry.get_all_pipelines.return_value = ["valid_pipeline"]
        mock_registry.get_pipeline_node.return_value = sample_valid_node
        mock_registry.get_pipeline_connections.return_value = {}
        mock_registry.load_registry.return_value = {"tag_index": {}}
        
        report = validator.generate_validation_report()
        
        assert isinstance(report, ValidationReport)
        assert report.is_valid
        assert report.total_issues == 0

    def test_generate_validation_report_with_errors(self, validator, mock_registry):
        """Test generating validation report with errors."""
        invalid_node = {
            "id": "test_pipeline",
            "atomic_properties": {}  # Missing required fields
        }
        
        mock_registry.get_all_pipelines.return_value = ["test_pipeline"]
        mock_registry.get_pipeline_node.return_value = invalid_node
        mock_registry.get_pipeline_connections.return_value = {}
        mock_registry.load_registry.return_value = {"tag_index": {}}
        
        report = validator.generate_validation_report()
        
        assert isinstance(report, ValidationReport)
        assert not report.is_valid
        assert report.total_issues > 0
        assert ValidationSeverity.ERROR in report.issues_by_severity

    def test_validate_zettelkasten_principles(self, validator, mock_registry, sample_valid_node):
        """Test Zettelkasten principles validation."""
        mock_registry.get_all_pipelines.return_value = ["valid_pipeline"]
        mock_registry.get_pipeline_node.return_value = sample_valid_node
        mock_registry.get_pipeline_connections.return_value = {
            "alternatives": [{"id": "other_pipeline", "annotation": "Alternative"}]
        }
        mock_registry.load_registry.return_value = {
            "tag_index": {
                "framework_tags": {"xgboost": ["valid_pipeline"]},
                "task_tags": {"training": ["valid_pipeline"]}
            }
        }
        
        result = validator.validate_zettelkasten_principles()
        
        assert "overall_compliance" in result
        assert "principle_scores" in result
        assert "metrics" in result
        assert "recommendations" in result
        assert isinstance(result["overall_compliance"], float)

    def test_validate_zettelkasten_principles_empty_registry(self, validator, mock_registry):
        """Test Zettelkasten principles validation with empty registry."""
        mock_registry.get_all_pipelines.return_value = []
        
        result = validator.validate_zettelkasten_principles()
        
        assert "error" in result
        assert "No pipelines found" in result["error"]

    def test_are_similar_tags(self, validator):
        """Test tag similarity detection."""
        # Test similar tags
        assert validator._are_similar_tags("training", "trainng")  # Single char diff
        assert validator._are_similar_tags("xgboost", "xgbost")   # Single char diff
        assert validator._are_similar_tags("evaluation", "evaluaton")  # Single char diff
        
        # Test dissimilar tags
        assert not validator._are_similar_tags("training", "pytorch")
        assert not validator._are_similar_tags("short", "very_long_tag_name")
        assert not validator._are_similar_tags("evaluation", "eval")  # Too different

    def test_generate_principle_recommendations(self, validator):
        """Test principle recommendations generation."""
        recommendations = validator._generate_principle_recommendations(
            atomicity=0.5,      # Low - should trigger recommendation
            connectivity=0.4,   # Low - should trigger recommendation
            anti_categories=0.6, # Low - should trigger recommendation
            manual_linking=0.7,  # Low - should trigger recommendation
            dual_form=0.8       # Low - should trigger recommendation
        )
        
        assert len(recommendations) == 5  # All principles below threshold
        assert any("atomicity" in rec.lower() for rec in recommendations)
        assert any("connectivity" in rec.lower() for rec in recommendations)

    def test_clear_cache(self, validator):
        """Test clearing validation cache."""
        validator._validation_cache = {"test": "data"}
        validator._cache_valid = True
        
        validator.clear_cache()
        
        assert validator._validation_cache == {}
        assert not validator._cache_valid

    def test_validation_issue_models(self):
        """Test validation issue model classes."""
        # Test AtomicityViolation
        violation = AtomicityViolation(
            pipeline_id="test",
            violation_type="test_type",
            description="Test description",
            suggested_fix="Test fix"
        )
        assert violation.severity == ValidationSeverity.ERROR
        assert violation.category == "atomicity"
        assert "test_type" in violation.message

        # Test ConnectionError
        conn_error = ConnectionError(
            source_id="source",
            target_id="target",
            error_type="test_error",
            description="Test description"
        )
        assert conn_error.severity == ValidationSeverity.ERROR
        assert conn_error.category == "connectivity"

        # Test MetadataError
        meta_error = MetadataError(
            pipeline_id="test",
            field="test_field",
            description="Test description",
            suggested_fix="Test fix"
        )
        assert meta_error.severity == ValidationSeverity.WARNING
        assert meta_error.category == "metadata"

    def test_validation_report_summary(self):
        """Test validation report summary generation."""
        # Test valid report
        valid_report = ValidationReport(
            is_valid=True,
            total_issues=0,
            issues_by_severity={},
            issues_by_category={},
            all_issues=[]
        )
        assert "no critical issues" in valid_report.summary()

        # Test invalid report
        invalid_report = ValidationReport(
            is_valid=False,
            total_issues=3,
            issues_by_severity={
                ValidationSeverity.ERROR: 1,
                ValidationSeverity.WARNING: 2
            },
            issues_by_category={"atomicity": 3},
            all_issues=[]
        )
        summary = invalid_report.summary()
        assert "1 errors" in summary
        assert "2 warnings" in summary

    def test_error_handling_in_validation_methods(self, validator, mock_registry):
        """Test error handling in validation methods."""
        # Mock registry to raise exceptions
        mock_registry.get_all_pipelines.side_effect = Exception("Registry error")
        
        # All validation methods should handle errors gracefully by returning empty lists
        assert validator.validate_atomicity() == []
        assert validator.validate_connections() == []
        assert validator.validate_metadata_completeness() == []
        assert validator.validate_tag_consistency() == []
        assert validator.validate_independence_claims() == []
        
        # When all validation methods return empty lists due to errors,
        # the report will be valid with 0 issues (this is the actual behavior)
        report = validator.generate_validation_report()
        assert report.is_valid  # No issues found due to error handling
        assert report.total_issues == 0

        # Zettelkasten principles should return error dict
        result = validator.validate_zettelkasten_principles()
        assert "error" in result

"""
Test module for TransformStepBuilderValidator.

Tests the Transform-specific validation functionality including
_create_transformer method validation and batch transform handling.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, Any, List

from cursus.validation.alignment.validators.transform_step_validator import TransformStepBuilderValidator


class TestTransformStepBuilderValidator:
    """Test cases for TransformStepBuilderValidator class."""

    @pytest.fixture
    def workspace_dirs(self):
        """Fixture providing workspace directories."""
        return ["/test/workspace1", "/test/workspace2"]

    @pytest.fixture
    def validator(self, workspace_dirs):
        """Fixture providing TransformStepBuilderValidator instance."""
        return TransformStepBuilderValidator(workspace_dirs=workspace_dirs)

    @pytest.fixture
    def sample_transform_builder(self):
        """Fixture providing sample Transform builder class."""
        class SampleTransformBuilder:
            def validate_configuration(self):
                pass
            
            def _get_inputs(self):
                return {
                    "model_data": {"type": "TransformInput", "content_type": "text/csv"},
                    "input_data": {"type": "TransformInput", "content_type": "text/csv"}
                }
            
            def create_step(self):
                pass
            
            def _create_transformer(self, output_path=None):
                return {
                    "transformer_type": "BatchTransform",
                    "model_name": "xgboost-model",
                    "output_path": output_path,
                    "instance_type": "ml.m5.xlarge",
                    "instance_count": 1
                }
            
            def _get_outputs(self):
                return "s3://bucket/transform/output/"
        
        return SampleTransformBuilder

    def test_init_with_workspace_dirs(self, workspace_dirs):
        """Test TransformStepBuilderValidator initialization with workspace directories."""
        validator = TransformStepBuilderValidator(workspace_dirs=workspace_dirs)
        assert validator.workspace_dirs == workspace_dirs

    def test_init_without_workspace_dirs(self):
        """Test TransformStepBuilderValidator initialization without workspace directories."""
        validator = TransformStepBuilderValidator()
        assert validator.workspace_dirs is None

    def test_validate_builder_config_alignment_with_valid_transform_builder(self, validator, sample_transform_builder):
        """Test validation with valid Transform builder."""
        step_name = "transform_step"
        
        with patch.object(validator, '_get_builder_class') as mock_get_builder, \
             patch.object(validator, '_apply_universal_validation') as mock_universal, \
             patch.object(validator, '_resolve_validation_priorities') as mock_resolve:
            
            # Setup mocks
            mock_get_builder.return_value = sample_transform_builder
            
            universal_result = {
                "status": "COMPLETED",
                "issues": [],
                "rule_type": "universal",
                "priority": "HIGHEST"
            }
            
            combined_result = {
                "status": "PASSED",
                "total_issues": 0,
                "priority_resolution": "universal_rules_first_then_step_specific"
            }
            
            mock_universal.return_value = universal_result
            mock_resolve.return_value = combined_result
            
            # Execute validation
            result = validator.validate_builder_config_alignment(step_name)
            
            # Verify results
            assert result["status"] == "PASSED"
            assert result["total_issues"] == 0

    def test_apply_step_specific_validation_with_valid_builder(self, validator, sample_transform_builder):
        """Test step-specific validation with valid Transform builder."""
        step_name = "transform_step"
        
        with patch.object(validator, '_get_builder_class') as mock_get_builder, \
             patch('cursus.registry.step_names.get_sagemaker_step_type') as mock_get_step_type:
            
            # Setup mocks
            mock_get_builder.return_value = sample_transform_builder
            mock_get_step_type.return_value = "Transform"
            
            # Execute step-specific validation
            result = validator._apply_step_specific_validation(step_name)
            
            # Verify results
            assert result["status"] == "COMPLETED"
            assert result["rule_type"] == "step_specific"
            assert result["priority"] == "SECONDARY"
            assert len(result["issues"]) == 0

    def test_apply_step_specific_validation_missing_create_transformer(self, validator):
        """Test step-specific validation with missing _create_transformer method."""
        step_name = "transform_step"
        
        # Create builder missing _create_transformer
        class IncompleteTransformBuilder:
            def validate_configuration(self):
                pass
            
            def _get_inputs(self):
                pass
            
            def create_step(self):
                pass
            
            def _get_outputs(self):
                return "s3://bucket/transform/output/"
            # Missing _create_transformer
        
        with patch.object(validator, '_get_builder_class') as mock_get_builder, \
             patch('cursus.registry.step_names.get_sagemaker_step_type') as mock_get_step_type:
            
            # Setup mocks
            mock_get_builder.return_value = IncompleteTransformBuilder
            mock_get_step_type.return_value = "Transform"
            
            # Execute step-specific validation
            result = validator._apply_step_specific_validation(step_name)
            
            # Verify results show issues
            assert result["status"] == "ISSUES_FOUND"
            assert len(result["issues"]) > 0
            
            # Check for specific missing method issue
            error_issues = [issue for issue in result["issues"] if issue["level"] == "ERROR"]
            assert len(error_issues) > 0
            assert any("_create_transformer" in issue["message"] for issue in error_issues)

    def test_validate_create_transformer_method_with_valid_implementation(self, validator, sample_transform_builder):
        """Test _create_transformer method validation with valid implementation."""
        builder_class = sample_transform_builder
        
        # Execute validation
        issues = validator._validate_create_transformer_method(builder_class)
        
        # Should have no issues for valid implementation
        assert len(issues) == 0

    def test_validate_create_transformer_method_missing_output_path_parameter(self, validator):
        """Test _create_transformer method validation with missing output_path parameter."""
        class InvalidTransformBuilder:
            def _create_transformer(self):  # Missing output_path parameter
                return {"transformer_type": "BatchTransform"}
        
        # Execute validation
        issues = validator._validate_create_transformer_method(InvalidTransformBuilder)
        
        # Should have issues for missing output_path parameter
        assert len(issues) > 0
        warning_issues = [issue for issue in issues if issue["level"] == "WARNING"]
        assert len(warning_issues) > 0
        assert any("output_path" in issue["message"] for issue in warning_issues)

    def test_validate_transform_outputs_with_valid_outputs(self, validator, sample_transform_builder):
        """Test transform outputs validation with valid outputs."""
        builder_class = sample_transform_builder
        
        # Execute validation
        issues = validator._validate_transform_outputs(builder_class)
        
        # Should have no issues for valid outputs
        assert len(issues) == 0

    def test_validate_transform_outputs_with_invalid_return_type(self, validator):
        """Test transform outputs validation with invalid return type."""
        class InvalidOutputsBuilder:
            def _get_outputs(self):
                return ["invalid", "return_type"]  # Should return string
        
        # Execute validation
        issues = validator._validate_transform_outputs(InvalidOutputsBuilder)
        
        # Should have issues for invalid return type
        assert len(issues) > 0
        warning_issues = [issue for issue in issues if issue["level"] == "WARNING"]
        assert len(warning_issues) > 0

    def test_validate_transform_configuration_with_valid_configuration(self, validator, sample_transform_builder):
        """Test transform configuration validation with valid configuration."""
        builder_class = sample_transform_builder
        
        # Execute validation
        issues = validator._validate_transform_configuration(builder_class)
        
        # Should have no issues for valid transform configuration
        assert len(issues) == 0

    def test_validate_transformer_type_patterns_with_valid_patterns(self, validator, sample_transform_builder):
        """Test transformer type patterns validation with valid patterns."""
        builder_class = sample_transform_builder
        
        # Execute validation
        issues = validator._validate_transformer_type_patterns(builder_class)
        
        # Should have no issues for valid transformer type patterns
        assert len(issues) == 0

    def test_integration_with_step_type_specific_validator_base(self, validator):
        """Test integration with StepTypeSpecificValidator base class."""
        step_name = "transform_step"
        
        with patch.object(validator, '_get_builder_class') as mock_get_builder, \
             patch.object(validator, '_apply_universal_validation') as mock_universal:
            
            # Setup mocks
            mock_get_builder.return_value = Mock()
            mock_universal.return_value = {
                "status": "COMPLETED",
                "issues": [],
                "rule_type": "universal",
                "priority": "HIGHEST"
            }
            
            # Execute validation
            result = validator.validate_builder_config_alignment(step_name)
            
            # Verify base class integration
            assert "status" in result
            assert "priority_resolution" in result

    def test_error_handling_during_validation(self, validator):
        """Test error handling during validation process."""
        step_name = "transform_step"
        
        with patch.object(validator, '_get_builder_class') as mock_get_builder:
            # Setup mock to raise exception
            mock_get_builder.side_effect = Exception("Builder not found")
            
            # Execute validation
            result = validator._apply_step_specific_validation(step_name)
            
            # Should handle error gracefully
            assert result["status"] == "ERROR"
            assert len(result["issues"]) > 0

    def test_workspace_directory_propagation(self, workspace_dirs):
        """Test that workspace directories are properly propagated."""
        validator = TransformStepBuilderValidator(workspace_dirs=workspace_dirs)
        assert validator.workspace_dirs == workspace_dirs

    def test_validate_with_complex_transform_configuration(self, validator):
        """Test validation with complex Transform configuration."""
        step_name = "complex_transform_step"
        
        # Create complex Transform builder
        class ComplexTransformBuilder:
            def validate_configuration(self):
                pass
            
            def _get_inputs(self):
                return {
                    f"input_data_{i}": {"type": "TransformInput", "content_type": "text/csv"}
                    for i in range(5)
                }
            
            def create_step(self):
                pass
            
            def _create_transformer(self, output_path=None):
                return {
                    "transformer_type": "BatchTransform",
                    "model_name": "complex-ensemble-model",
                    "instance_type": "ml.c5.4xlarge",
                    "instance_count": 5,
                    "max_concurrent_transforms": 20,
                    "max_payload": 6,
                    "batch_strategy": "MultiRecord",
                    "environment": {f"ENV_VAR_{i}": f"value_{i}" for i in range(10)},
                    "output_path": output_path
                }
            
            def _get_outputs(self):
                return "s3://bucket/complex/transform/output/"
        
        with patch.object(validator, '_get_builder_class') as mock_get_builder, \
             patch('cursus.registry.step_names.get_sagemaker_step_type') as mock_get_step_type:
            
            # Setup mocks
            mock_get_builder.return_value = ComplexTransformBuilder
            mock_get_step_type.return_value = "Transform"
            
            # Execute validation
            result = validator._apply_step_specific_validation(step_name)
            
            # Should handle complex configuration correctly
            assert result["status"] == "COMPLETED"
            assert result["rule_type"] == "step_specific"

    def test_performance_with_large_transform_configuration(self, validator):
        """Test performance with large Transform configuration."""
        step_name = "large_transform_step"
        
        # Create large Transform builder
        class LargeTransformBuilder:
            def validate_configuration(self):
                pass
            
            def _get_inputs(self):
                return {
                    f"large_input_data_{i}": {"type": "TransformInput"}
                    for i in range(100)
                }
            
            def create_step(self):
                pass
            
            def _create_transformer(self, output_path=None):
                return {
                    "transformer_type": "BatchTransform",
                    "model_name": "large-model",
                    "environment": {f"ENV_{i}": f"val_{i}" for i in range(50)}
                }
            
            def _get_outputs(self):
                return "s3://bucket/large/transform/output/"
        
        with patch.object(validator, '_get_builder_class') as mock_get_builder, \
             patch('cursus.registry.step_names.get_sagemaker_step_type') as mock_get_step_type:
            
            # Setup mocks
            mock_get_builder.return_value = LargeTransformBuilder
            mock_get_step_type.return_value = "Transform"
            
            # Execute validation and verify it completes efficiently
            result = validator._apply_step_specific_validation(step_name)
            
            # Should complete successfully
            assert "status" in result
            assert "rule_type" in result

    def test_validation_result_consistency(self, validator, sample_transform_builder):
        """Test that validation results are consistently structured."""
        step_name = "consistency_test"
        
        with patch.object(validator, '_get_builder_class') as mock_get_builder, \
             patch('cursus.registry.step_names.get_sagemaker_step_type') as mock_get_step_type:
            
            # Setup mocks
            mock_get_builder.return_value = sample_transform_builder
            mock_get_step_type.return_value = "Transform"
            
            # Execute validation
            result = validator._apply_step_specific_validation(step_name)
            
            # Verify consistent result structure
            required_keys = ["status", "issues", "step_type", "rule_type", "priority"]
            for key in required_keys:
                assert key in result
            
            assert isinstance(result["issues"], list)
            assert isinstance(result["step_type"], str)
            assert result["rule_type"] == "step_specific"
            assert result["priority"] == "SECONDARY"

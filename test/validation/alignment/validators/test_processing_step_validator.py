"""
Test module for ProcessingStepBuilderValidator.

Tests the processing-specific validation functionality including
_create_processor method validation and ProcessingInput/ProcessingOutput handling.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, Any, List

from cursus.validation.alignment.validators.processing_step_validator import ProcessingStepBuilderValidator


class TestProcessingStepBuilderValidator:
    """Test cases for ProcessingStepBuilderValidator class."""

    @pytest.fixture
    def workspace_dirs(self):
        """Fixture providing workspace directories."""
        return ["/test/workspace1", "/test/workspace2"]

    @pytest.fixture
    def validator(self, workspace_dirs):
        """Fixture providing ProcessingStepBuilderValidator instance."""
        return ProcessingStepBuilderValidator(workspace_dirs=workspace_dirs)

    @pytest.fixture
    def sample_processing_builder(self):
        """Fixture providing sample processing builder class."""
        class SampleProcessingBuilder:
            def validate_configuration(self):
                pass
            
            def _get_inputs(self):
                return [
                    {"logical_name": "training_data", "type": "ProcessingInput"},
                    {"logical_name": "validation_data", "type": "ProcessingInput"}
                ]
            
            def create_step(self):
                pass
            
            def _create_processor(self):  # Fix signature to match actual implementation
                return {"processor_type": "ScriptProcessor"}
            
            def _get_outputs(self):
                return [
                    {"logical_name": "processed_data", "type": "ProcessingOutput"},
                    {"logical_name": "feature_store", "type": "ProcessingOutput"}
                ]
            
            def _get_job_arguments(self):
                return ["--epochs", "100", "--learning-rate", "0.01"]
        
        return SampleProcessingBuilder

    def test_init_with_workspace_dirs(self, workspace_dirs):
        """Test ProcessingStepBuilderValidator initialization with workspace directories."""
        validator = ProcessingStepBuilderValidator(workspace_dirs=workspace_dirs)
        assert validator.workspace_dirs == workspace_dirs

    def test_init_without_workspace_dirs(self):
        """Test ProcessingStepBuilderValidator initialization without workspace directories."""
        validator = ProcessingStepBuilderValidator()
        assert validator.workspace_dirs is None

    def test_validate_builder_config_alignment_with_valid_processing_builder(self, validator, sample_processing_builder):
        """Test validation with valid processing builder."""
        step_name = "processing_step"
        
        with patch.object(validator, '_get_builder_class') as mock_get_builder, \
             patch.object(validator, '_apply_universal_validation') as mock_universal, \
             patch.object(validator, '_resolve_validation_priorities') as mock_resolve:
            
            # Setup mocks
            mock_get_builder.return_value = sample_processing_builder
            
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

    def test_apply_step_specific_validation_with_valid_builder(self, validator, sample_processing_builder):
        """Test step-specific validation with valid processing builder."""
        step_name = "Processing"  # Use valid step name from registry
        
        with patch.object(validator, '_get_builder_class') as mock_get_builder, \
             patch('cursus.registry.step_names.get_sagemaker_step_type') as mock_get_step_type:
            
            # Setup mocks
            mock_get_builder.return_value = sample_processing_builder
            mock_get_step_type.return_value = "Processing"
            
            # Execute step-specific validation
            result = validator._apply_step_specific_validation(step_name)
            
            # Verify results
            assert result["status"] == "COMPLETED"
            assert result["rule_type"] == "step_specific"
            assert result["priority"] == "SECONDARY"
            assert len(result["issues"]) == 0

    def test_apply_step_specific_validation_missing_create_processor(self, validator):
        """Test step-specific validation with missing _create_processor method."""
        step_name = "Processing"  # Use valid step name from registry
        
        # Create builder missing _create_processor
        class IncompleteProcessingBuilder:
            def validate_configuration(self):
                pass
            
            def _get_inputs(self):
                pass
            
            def create_step(self):
                pass
            
            def _get_outputs(self):
                return []
            # Missing _create_processor
        
        with patch.object(validator, '_get_builder_class') as mock_get_builder, \
             patch('cursus.registry.step_names.get_sagemaker_step_type') as mock_get_step_type:
            
            # Setup mocks
            mock_get_builder.return_value = IncompleteProcessingBuilder
            mock_get_step_type.return_value = "Processing"
            
            # Execute step-specific validation
            result = validator._apply_step_specific_validation(step_name)
            
            # Verify results show issues
            assert result["status"] == "ISSUES_FOUND"
            assert len(result["issues"]) > 0
            
            # Check for specific missing method issue
            error_issues = [issue for issue in result["issues"] if issue["level"] == "ERROR"]
            assert len(error_issues) > 0
            assert any("_create_processor" in issue["message"] for issue in error_issues)

    def test_apply_step_specific_validation_missing_get_outputs(self, validator):
        """Test step-specific validation with missing _get_outputs method."""
        step_name = "Processing"  # Use valid step name from registry
        
        # Create builder missing _get_outputs
        class IncompleteProcessingBuilder:
            def validate_configuration(self):
                pass
            
            def _get_inputs(self):
                pass
            
            def create_step(self):
                pass
            
            def _create_processor(self):
                return {"processor_type": "ScriptProcessor"}
            # Missing _get_outputs
        
        with patch.object(validator, '_get_builder_class') as mock_get_builder, \
             patch('cursus.registry.step_names.get_sagemaker_step_type') as mock_get_step_type:
            
            # Setup mocks
            mock_get_builder.return_value = IncompleteProcessingBuilder
            mock_get_step_type.return_value = "Processing"
            
            # Execute step-specific validation
            result = validator._apply_step_specific_validation(step_name)
            
            # Verify results show issues
            assert result["status"] == "ISSUES_FOUND"
            assert len(result["issues"]) > 0
            
            # Check for specific missing method issue
            error_issues = [issue for issue in result["issues"] if issue["level"] == "ERROR"]
            assert len(error_issues) > 0
            assert any("_get_outputs" in issue["message"] for issue in error_issues)

    def test_validate_create_processor_method_with_valid_implementation(self, validator, sample_processing_builder):
        """Test _create_processor method validation with valid implementation."""
        builder_class = sample_processing_builder
        
        # Execute validation
        issues = validator._validate_create_processor_method(builder_class)
        
        # Should have no issues for valid implementation
        assert len(issues) == 0

    def test_validate_create_processor_method_with_invalid_return_type(self, validator):
        """Test _create_processor method validation with invalid return type."""
        class InvalidProcessingBuilder:
            def _create_processor(self):
                return "invalid_return_type"  # Should return dict-like object
        
        # Execute validation
        issues = validator._validate_create_processor_method(InvalidProcessingBuilder)
        
        # Should have issues for invalid return type
        assert len(issues) > 0
        warning_issues = [issue for issue in issues if issue["level"] == "WARNING"]
        assert len(warning_issues) > 0

    def test_validate_processing_outputs_with_valid_outputs(self, validator, sample_processing_builder):
        """Test processing outputs validation with valid outputs."""
        builder_class = sample_processing_builder
        
        # Execute validation
        issues = validator._validate_processing_outputs(builder_class)
        
        # Should have no issues for valid outputs
        assert len(issues) == 0

    def test_validate_processing_outputs_with_invalid_return_type(self, validator):
        """Test processing outputs validation with invalid return type."""
        class InvalidOutputsBuilder:
            def _get_outputs(self):
                return "invalid_return_type"  # Should return list
        
        # Execute validation
        issues = validator._validate_processing_outputs(InvalidOutputsBuilder)
        
        # Should have issues for invalid return type
        assert len(issues) > 0
        warning_issues = [issue for issue in issues if issue["level"] == "WARNING"]
        assert len(warning_issues) > 0

    def test_validate_job_arguments_override_with_valid_arguments(self, validator, sample_processing_builder):
        """Test job arguments override validation with valid arguments."""
        builder_class = sample_processing_builder
        
        # Execute validation
        issues = validator._validate_job_arguments_override(builder_class)
        
        # Should have no issues for valid job arguments override
        assert len(issues) == 0

    def test_validate_processing_input_output_handling_with_valid_handling(self, validator, sample_processing_builder):
        """Test processing input/output handling validation."""
        builder_class = sample_processing_builder
        
        # Execute validation
        issues = validator._validate_processing_input_output_handling(builder_class)
        
        # Should have no issues for valid input/output handling
        assert len(issues) == 0

    def test_integration_with_step_type_specific_validator_base(self, validator):
        """Test integration with StepTypeSpecificValidator base class."""
        step_name = "processing_step"
        
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
        step_name = "Processing"  # Use valid step name from registry
        
        with patch.object(validator, '_get_builder_class') as mock_get_builder:
            # Setup mock to raise exception
            mock_get_builder.side_effect = Exception("Builder not found")
            
            # Execute validation
            result = validator._apply_step_specific_validation(step_name)
            
            # Should handle error gracefully
            assert result["status"] == "ERROR"
            assert "error" in result  # Error results have "error" key, not "issues"

    def test_workspace_directory_propagation(self, workspace_dirs):
        """Test that workspace directories are properly propagated."""
        validator = ProcessingStepBuilderValidator(workspace_dirs=workspace_dirs)
        assert validator.workspace_dirs == workspace_dirs

    def test_validate_with_complex_processing_configuration(self, validator):
        """Test validation with complex processing configuration."""
        step_name = "Processing"  # Use valid step name from registry
        
        # Create complex processing builder
        class ComplexProcessingBuilder:
            def validate_configuration(self):
                pass
            
            def _get_inputs(self):
                return [
                    {"logical_name": f"input_{i}", "type": "ProcessingInput"} 
                    for i in range(10)
                ]
            
            def create_step(self):
                pass
            
            def _create_processor(self):
                return {
                    "processor_type": "ScriptProcessor",
                    "source_dir": "/complex/source",
                    "entry_point": "complex_process.py",
                    "instance_type": "ml.m5.xlarge",
                    "instance_count": 2
                }
            
            def _get_outputs(self):
                return [
                    {"logical_name": f"output_{i}", "type": "ProcessingOutput"} 
                    for i in range(5)
                ]
            
            def _get_job_arguments(self):
                return [f"--param-{i}" for i in range(20)]
        
        with patch.object(validator, '_get_builder_class') as mock_get_builder, \
             patch('cursus.registry.step_names.get_sagemaker_step_type') as mock_get_step_type:
            
            # Setup mocks
            mock_get_builder.return_value = ComplexProcessingBuilder
            mock_get_step_type.return_value = "Processing"
            
            # Execute validation
            result = validator._apply_step_specific_validation(step_name)
            
            # Should handle complex configuration correctly
            assert result["status"] == "COMPLETED"
            assert result["rule_type"] == "step_specific"

    def test_performance_with_large_processing_configuration(self, validator):
        """Test performance with large processing configuration."""
        step_name = "Processing"  # Use valid step name from registry
        
        # Create large processing builder
        class LargeProcessingBuilder:
            def validate_configuration(self):
                pass
            
            def _get_inputs(self):
                return [
                    {"logical_name": f"large_input_{i}", "type": "ProcessingInput"} 
                    for i in range(100)
                ]
            
            def create_step(self):
                pass
            
            def _create_processor(self):
                return {"processor_type": "ScriptProcessor"}
            
            def _get_outputs(self):
                return [
                    {"logical_name": f"large_output_{i}", "type": "ProcessingOutput"} 
                    for i in range(50)
                ]
        
        with patch.object(validator, '_get_builder_class') as mock_get_builder, \
             patch('cursus.registry.step_names.get_sagemaker_step_type') as mock_get_step_type:
            
            # Setup mocks
            mock_get_builder.return_value = LargeProcessingBuilder
            mock_get_step_type.return_value = "Processing"
            
            # Execute validation and verify it completes efficiently
            result = validator._apply_step_specific_validation(step_name)
            
            # Should complete successfully
            assert "status" in result
            assert "rule_type" in result

    def test_validation_result_consistency(self, validator, sample_processing_builder):
        """Test that validation results are consistently structured."""
        step_name = "Processing"  # Use valid step name from registry
        
        with patch.object(validator, '_get_builder_class') as mock_get_builder, \
             patch('cursus.registry.step_names.get_sagemaker_step_type') as mock_get_step_type:
            
            # Setup mocks
            mock_get_builder.return_value = sample_processing_builder
            mock_get_step_type.return_value = "Processing"
            
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

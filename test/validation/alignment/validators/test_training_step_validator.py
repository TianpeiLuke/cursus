"""
Test module for TrainingStepBuilderValidator.

Tests the training-specific validation functionality including
_create_estimator method validation and TrainingInput handling.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, Any, List

from cursus.validation.alignment.validators.training_step_validator import TrainingStepBuilderValidator


class TestTrainingStepBuilderValidator:
    """Test cases for TrainingStepBuilderValidator class."""

    @pytest.fixture
    def workspace_dirs(self):
        """Fixture providing workspace directories."""
        return ["/test/workspace1", "/test/workspace2"]

    @pytest.fixture
    def validator(self, workspace_dirs):
        """Fixture providing TrainingStepBuilderValidator instance."""
        return TrainingStepBuilderValidator(workspace_dirs=workspace_dirs)

    @pytest.fixture
    def sample_training_builder(self):
        """Fixture providing sample training builder class."""
        class SampleTrainingBuilder:
            def validate_configuration(self):
                pass
            
            def _get_inputs(self):
                return {
                    "training_data": {"type": "TrainingInput", "content_type": "text/csv"},
                    "validation_data": {"type": "TrainingInput", "content_type": "text/csv"}
                }
            
            def create_step(self):
                pass
            
            def _create_estimator(self, output_path=None):
                return {
                    "estimator_type": "XGBoost",
                    "framework_version": "1.3-1",
                    "output_path": output_path
                }
            
            def _get_outputs(self):
                return "s3://bucket/model/output/"
        
        return SampleTrainingBuilder

    def test_init_with_workspace_dirs(self, workspace_dirs):
        """Test TrainingStepBuilderValidator initialization with workspace directories."""
        validator = TrainingStepBuilderValidator(workspace_dirs=workspace_dirs)
        assert validator.workspace_dirs == workspace_dirs

    def test_init_without_workspace_dirs(self):
        """Test TrainingStepBuilderValidator initialization without workspace directories."""
        validator = TrainingStepBuilderValidator()
        assert validator.workspace_dirs == []

    def test_validate_builder_config_alignment_with_valid_training_builder(self, validator, sample_training_builder):
        """Test validation with valid training builder."""
        step_name = "training_step"
        
        with patch.object(validator, '_get_builder_class') as mock_get_builder, \
             patch.object(validator, '_apply_universal_validation') as mock_universal, \
             patch.object(validator, '_resolve_validation_priorities') as mock_resolve:
            
            # Setup mocks
            mock_get_builder.return_value = sample_training_builder
            
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

    def test_apply_step_specific_validation_with_valid_builder(self, validator, sample_training_builder):
        """Test step-specific validation with valid training builder."""
        step_name = "training_step"
        
        with patch.object(validator, '_get_builder_class') as mock_get_builder, \
             patch('cursus.validation.alignment.config.validation_ruleset.get_sagemaker_step_type') as mock_get_step_type:
            
            # Setup mocks
            mock_get_builder.return_value = sample_training_builder
            mock_get_step_type.return_value = "Training"
            
            # Execute step-specific validation
            result = validator._apply_step_specific_validation(step_name)
            
            # Verify results
            assert result["status"] == "COMPLETED"
            assert result["rule_type"] == "step_specific"
            assert result["priority"] == "SECONDARY"
            assert len(result["issues"]) == 0

    def test_apply_step_specific_validation_missing_create_estimator(self, validator):
        """Test step-specific validation with missing _create_estimator method."""
        step_name = "training_step"
        
        # Create builder missing _create_estimator
        class IncompleteTrainingBuilder:
            def validate_configuration(self):
                pass
            
            def _get_inputs(self):
                pass
            
            def create_step(self):
                pass
            
            def _get_outputs(self):
                return "s3://bucket/model/output/"
            # Missing _create_estimator
        
        with patch.object(validator, '_get_builder_class') as mock_get_builder, \
             patch('cursus.validation.alignment.config.validation_ruleset.get_sagemaker_step_type') as mock_get_step_type:
            
            # Setup mocks
            mock_get_builder.return_value = IncompleteTrainingBuilder
            mock_get_step_type.return_value = "Training"
            
            # Execute step-specific validation
            result = validator._apply_step_specific_validation(step_name)
            
            # Verify results show issues
            assert result["status"] == "ISSUES_FOUND"
            assert len(result["issues"]) > 0
            
            # Check for specific missing method issue
            error_issues = [issue for issue in result["issues"] if issue["level"] == "ERROR"]
            assert len(error_issues) > 0
            assert any("_create_estimator" in issue["message"] for issue in error_issues)

    def test_apply_step_specific_validation_missing_get_outputs(self, validator):
        """Test step-specific validation with missing _get_outputs method."""
        step_name = "training_step"
        
        # Create builder missing _get_outputs
        class IncompleteTrainingBuilder:
            def validate_configuration(self):
                pass
            
            def _get_inputs(self):
                pass
            
            def create_step(self):
                pass
            
            def _create_estimator(self, output_path=None):
                return {"estimator_type": "XGBoost"}
            # Missing _get_outputs
        
        with patch.object(validator, '_get_builder_class') as mock_get_builder, \
             patch('cursus.validation.alignment.config.validation_ruleset.get_sagemaker_step_type') as mock_get_step_type:
            
            # Setup mocks
            mock_get_builder.return_value = IncompleteTrainingBuilder
            mock_get_step_type.return_value = "Training"
            
            # Execute step-specific validation
            result = validator._apply_step_specific_validation(step_name)
            
            # Verify results show issues
            assert result["status"] == "ISSUES_FOUND"
            assert len(result["issues"]) > 0
            
            # Check for specific missing method issue
            error_issues = [issue for issue in result["issues"] if issue["level"] == "ERROR"]
            assert len(error_issues) > 0
            assert any("_get_outputs" in issue["message"] for issue in error_issues)

    def test_validate_create_estimator_method_with_valid_implementation(self, validator, sample_training_builder):
        """Test _create_estimator method validation with valid implementation."""
        builder_class = sample_training_builder
        
        # Execute validation
        issues = validator._validate_create_estimator_method(builder_class)
        
        # Should have no issues for valid implementation
        assert len(issues) == 0

    def test_validate_create_estimator_method_missing_output_path_parameter(self, validator):
        """Test _create_estimator method validation with missing output_path parameter."""
        class InvalidTrainingBuilder:
            def _create_estimator(self):  # Missing output_path parameter
                return {"estimator_type": "XGBoost"}
        
        # Execute validation
        issues = validator._validate_create_estimator_method(InvalidTrainingBuilder)
        
        # Should have issues for missing output_path parameter
        assert len(issues) > 0
        warning_issues = [issue for issue in issues if issue["level"] == "WARNING"]
        assert len(warning_issues) > 0
        assert any("output_path" in issue["message"] for issue in warning_issues)

    def test_validate_create_estimator_method_with_invalid_return_type(self, validator):
        """Test _create_estimator method validation with invalid return type."""
        class InvalidTrainingBuilder:
            def _create_estimator(self, output_path=None):
                return "invalid_return_type"  # Should return dict-like object
        
        # Execute validation
        issues = validator._validate_create_estimator_method(InvalidTrainingBuilder)
        
        # Should have issues for invalid return type
        assert len(issues) > 0
        warning_issues = [issue for issue in issues if issue["level"] == "WARNING"]
        assert len(warning_issues) > 0

    def test_validate_training_outputs_with_valid_outputs(self, validator, sample_training_builder):
        """Test training outputs validation with valid outputs."""
        builder_class = sample_training_builder
        
        # Execute validation
        issues = validator._validate_training_outputs(builder_class)
        
        # Should have no issues for valid outputs
        assert len(issues) == 0

    def test_validate_training_outputs_with_invalid_return_type(self, validator):
        """Test training outputs validation with invalid return type."""
        class InvalidOutputsBuilder:
            def _get_outputs(self):
                return ["invalid", "return_type"]  # Should return string
        
        # Execute validation
        issues = validator._validate_training_outputs(InvalidOutputsBuilder)
        
        # Should have issues for invalid return type
        assert len(issues) > 0
        warning_issues = [issue for issue in issues if issue["level"] == "WARNING"]
        assert len(warning_issues) > 0

    def test_detect_estimator_type_patterns_xgboost(self, validator):
        """Test estimator type pattern detection for XGBoost."""
        builder_class_dict = {
            "_create_estimator": lambda output_path=None: {
                "estimator_type": "XGBoost",
                "framework_version": "1.3-1",
                "hyperparameters": {"max_depth": 6, "n_estimators": 100}
            }
        }
        
        # Execute pattern detection
        patterns = validator._detect_estimator_type_patterns(builder_class_dict)
        
        # Verify XGBoost pattern detected
        assert "estimator_type" in patterns
        assert patterns["estimator_type"] == "XGBoost"
        assert "xgboost_training" in patterns
        assert patterns["xgboost_training"] is True

    def test_detect_estimator_type_patterns_pytorch(self, validator):
        """Test estimator type pattern detection for PyTorch."""
        builder_class_dict = {
            "_create_estimator": lambda output_path=None: {
                "estimator_type": "PyTorch",
                "framework_version": "1.8.1",
                "py_version": "py38"
            }
        }
        
        # Execute pattern detection
        patterns = validator._detect_estimator_type_patterns(builder_class_dict)
        
        # Verify PyTorch pattern detected
        assert "estimator_type" in patterns
        assert patterns["estimator_type"] == "PyTorch"
        assert "pytorch_training" in patterns
        assert patterns["pytorch_training"] is True

    def test_detect_estimator_type_patterns_tensorflow(self, validator):
        """Test estimator type pattern detection for TensorFlow."""
        builder_class_dict = {
            "_create_estimator": lambda output_path=None: {
                "estimator_type": "TensorFlow",
                "framework_version": "2.6.0",
                "py_version": "py38"
            }
        }
        
        # Execute pattern detection
        patterns = validator._detect_estimator_type_patterns(builder_class_dict)
        
        # Verify TensorFlow pattern detected
        assert "estimator_type" in patterns
        assert patterns["estimator_type"] == "TensorFlow"
        assert "tensorflow_training" in patterns
        assert patterns["tensorflow_training"] is True

    def test_validate_training_input_patterns(self, validator, sample_training_builder):
        """Test training input pattern validation."""
        builder_class = sample_training_builder
        
        # Execute validation
        issues = validator._validate_training_input_patterns(builder_class)
        
        # Should have no issues for valid input patterns
        assert len(issues) == 0

    def test_validate_hyperparameter_patterns(self, validator):
        """Test hyperparameter pattern validation."""
        class HyperparameterBuilder:
            def _create_estimator(self, output_path=None):
                return {
                    "estimator_type": "XGBoost",
                    "hyperparameters": {
                        "max_depth": 6,
                        "n_estimators": 100,
                        "learning_rate": 0.1,
                        "subsample": 0.8
                    }
                }
        
        # Execute validation
        issues = validator._validate_hyperparameter_patterns(HyperparameterBuilder)
        
        # Should have no issues for valid hyperparameter patterns
        assert len(issues) == 0

    def test_validate_framework_specific_xgboost(self, validator):
        """Test framework-specific validation for XGBoost."""
        class XGBoostBuilder:
            def _create_estimator(self, output_path=None):
                return {
                    "estimator_type": "XGBoost",
                    "framework_version": "1.3-1",
                    "hyperparameters": {
                        "max_depth": 6,
                        "n_estimators": 100,
                        "objective": "reg:squarederror"
                    }
                }
        
        # Execute validation
        issues = validator._validate_framework_specific_xgboost(XGBoostBuilder)
        
        # Should have no issues for valid XGBoost configuration
        assert len(issues) == 0

    def test_validate_framework_specific_pytorch(self, validator):
        """Test framework-specific validation for PyTorch."""
        class PyTorchBuilder:
            def _create_estimator(self, output_path=None):
                return {
                    "estimator_type": "PyTorch",
                    "framework_version": "1.8.1",
                    "py_version": "py38",
                    "entry_point": "train.py",
                    "source_dir": "/path/to/source"
                }
        
        # Execute validation
        issues = validator._validate_framework_specific_pytorch(PyTorchBuilder)
        
        # Should have no issues for valid PyTorch configuration
        assert len(issues) == 0

    def test_validate_training_job_optimization(self, validator):
        """Test training job optimization validation."""
        class OptimizedTrainingBuilder:
            def _create_estimator(self, output_path=None):
                return {
                    "estimator_type": "XGBoost",
                    "max_run": 86400,  # 24 hours
                    "use_spot_instances": True,
                    "max_wait": 172800,  # 48 hours
                    "checkpoint_s3_uri": "s3://bucket/checkpoints/"
                }
        
        # Execute validation
        issues = validator._validate_training_job_optimization(OptimizedTrainingBuilder)
        
        # Should have no issues for valid optimization configuration
        assert len(issues) == 0

    def test_integration_with_step_type_specific_validator_base(self, validator):
        """Test integration with StepTypeSpecificValidator base class."""
        step_name = "training_step"
        
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
        step_name = "training_step"
        
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
        validator = TrainingStepBuilderValidator(workspace_dirs=workspace_dirs)
        assert validator.workspace_dirs == workspace_dirs

    def test_validate_with_complex_training_configuration(self, validator):
        """Test validation with complex training configuration."""
        step_name = "complex_training_step"
        
        # Create complex training builder
        class ComplexTrainingBuilder:
            def validate_configuration(self):
                pass
            
            def _get_inputs(self):
                return {
                    f"training_data_{i}": {"type": "TrainingInput", "content_type": "text/csv"}
                    for i in range(5)
                }
            
            def create_step(self):
                pass
            
            def _create_estimator(self, output_path=None):
                return {
                    "estimator_type": "PyTorch",
                    "framework_version": "1.8.1",
                    "py_version": "py38",
                    "entry_point": "complex_train.py",
                    "source_dir": "/complex/source",
                    "hyperparameters": {f"param_{i}": i for i in range(20)},
                    "instance_type": "ml.p3.2xlarge",
                    "instance_count": 4,
                    "output_path": output_path
                }
            
            def _get_outputs(self):
                return "s3://bucket/complex/model/output/"
        
        with patch.object(validator, '_get_builder_class') as mock_get_builder, \
             patch('cursus.validation.alignment.config.validation_ruleset.get_sagemaker_step_type') as mock_get_step_type:
            
            # Setup mocks
            mock_get_builder.return_value = ComplexTrainingBuilder
            mock_get_step_type.return_value = "Training"
            
            # Execute validation
            result = validator._apply_step_specific_validation(step_name)
            
            # Should handle complex configuration correctly
            assert result["status"] == "COMPLETED"
            assert result["rule_type"] == "step_specific"

    def test_performance_with_large_training_configuration(self, validator):
        """Test performance with large training configuration."""
        step_name = "large_training_step"
        
        # Create large training builder
        class LargeTrainingBuilder:
            def validate_configuration(self):
                pass
            
            def _get_inputs(self):
                return {
                    f"large_training_data_{i}": {"type": "TrainingInput"}
                    for i in range(100)
                }
            
            def create_step(self):
                pass
            
            def _create_estimator(self, output_path=None):
                return {
                    "estimator_type": "XGBoost",
                    "hyperparameters": {f"param_{i}": i for i in range(100)}
                }
            
            def _get_outputs(self):
                return "s3://bucket/large/model/output/"
        
        with patch.object(validator, '_get_builder_class') as mock_get_builder, \
             patch('cursus.validation.alignment.config.validation_ruleset.get_sagemaker_step_type') as mock_get_step_type:
            
            # Setup mocks
            mock_get_builder.return_value = LargeTrainingBuilder
            mock_get_step_type.return_value = "Training"
            
            # Execute validation and verify it completes efficiently
            result = validator._apply_step_specific_validation(step_name)
            
            # Should complete successfully
            assert "status" in result
            assert "rule_type" in result

    def test_validation_result_consistency(self, validator, sample_training_builder):
        """Test that validation results are consistently structured."""
        step_name = "consistency_test"
        
        with patch.object(validator, '_get_builder_class') as mock_get_builder, \
             patch('cursus.validation.alignment.config.validation_ruleset.get_sagemaker_step_type') as mock_get_step_type:
            
            # Setup mocks
            mock_get_builder.return_value = sample_training_builder
            mock_get_step_type.return_value = "Training"
            
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

    def test_validate_estimator_output_path_usage(self, validator, sample_training_builder):
        """Test that _create_estimator properly uses output_path parameter."""
        builder_class = sample_training_builder
        
        # Execute validation
        issues = validator._validate_estimator_output_path_usage(builder_class)
        
        # Should have no issues for proper output_path usage
        assert len(issues) == 0

    def test_validate_estimator_output_path_usage_not_used(self, validator):
        """Test validation when _create_estimator doesn't use output_path parameter."""
        class NoOutputPathBuilder:
            def _create_estimator(self, output_path=None):
                return {"estimator_type": "XGBoost"}  # Doesn't use output_path
        
        # Execute validation
        issues = validator._validate_estimator_output_path_usage(NoOutputPathBuilder)
        
        # Should have warning for not using output_path
        assert len(issues) > 0
        warning_issues = [issue for issue in issues if issue["level"] == "WARNING"]
        assert len(warning_issues) > 0
        assert any("output_path" in issue["message"] for issue in warning_issues)

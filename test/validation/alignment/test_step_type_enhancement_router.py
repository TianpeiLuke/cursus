"""
Unit tests for step_type_enhancement_router.py module.

Tests step type enhancement routing and orchestration functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from cursus.validation.alignment.step_type_enhancement_router import (
    StepTypeEnhancementRouter
)
from cursus.validation.alignment.core_models import (
    ValidationResult,
    StepTypeAwareAlignmentIssue,
    SeverityLevel
)

class TestStepTypeEnhancementRouter:
    """Test step type enhancement router functionality."""

    @pytest.fixture
    def router(self):
        """Set up test fixtures."""
        return StepTypeEnhancementRouter()
        
    @pytest.fixture
    def mock_validation_result(self):
        """Mock validation result fixture."""
        return ValidationResult(
            is_valid=True,
            issues=[],
            summary={"message": "Test validation result"},
            metadata={"script_name": "test_script.py"}
        )
        
    @pytest.fixture
    def mock_issue(self):
        """Mock alignment issue fixture."""
        return StepTypeAwareAlignmentIssue(
            level=SeverityLevel.WARNING,
            category="test_category",
            message="Test message",
            recommendation="Test suggestion",
            step_type="Training",
            framework_context="xgboost"
        )

    def test_router_initialization(self, router):
        """Test router initialization with all enhancers."""
        assert router._enhancers is not None
        
        # Check that all expected step types have enhancer classes defined
        expected_step_types = [
            "Processing", "Training", "CreateModel", 
            "Transform", "RegisterModel", "Utility", "Base"
        ]
        
        for step_type in expected_step_types:
            assert step_type in router._enhancer_classes
            assert router._enhancer_classes[step_type] is not None

    @patch('cursus.validation.alignment.step_type_enhancement_router.detect_step_type_from_registry')
    def test_enhance_validation_training_step(self, mock_detect_step_type, router, mock_validation_result):
        """Test validation enhancement for training step."""
        # Setup
        mock_detect_step_type.return_value = "Training"
        script_name = "xgboost_training.py"
        
        # Mock the training enhancer
        mock_training_enhancer = Mock()
        mock_training_enhancer.enhance_validation.return_value = mock_validation_result
        router._enhancers["Training"] = mock_training_enhancer
        
        # Execute
        result = router.enhance_validation(script_name, mock_validation_result)
        
        # Verify
        mock_detect_step_type.assert_called_once_with(script_name)
        mock_training_enhancer.enhance_validation.assert_called_once_with(
            mock_validation_result, script_name
        )
        assert result == mock_validation_result

    @patch('cursus.validation.alignment.step_type_enhancement_router.detect_step_type_from_registry')
    def test_enhance_validation_processing_step(self, mock_detect_step_type, router, mock_validation_result):
        """Test validation enhancement for processing step."""
        # Setup
        mock_detect_step_type.return_value = "Processing"
        script_name = "tabular_preprocessing.py"
        
        # Mock the processing enhancer
        mock_processing_enhancer = Mock()
        mock_processing_enhancer.enhance_validation.return_value = mock_validation_result
        router._enhancers["Processing"] = mock_processing_enhancer
        
        # Execute
        result = router.enhance_validation(script_name, mock_validation_result)
        
        # Verify
        mock_detect_step_type.assert_called_once_with(script_name)
        mock_processing_enhancer.enhance_validation.assert_called_once_with(
            mock_validation_result, script_name
        )
        assert result == mock_validation_result

    @patch('cursus.validation.alignment.step_type_enhancement_router.detect_step_type_from_registry')
    def test_enhance_validation_unknown_step_type(self, mock_detect_step_type, router, mock_validation_result):
        """Test validation enhancement for unknown step type."""
        # Setup
        mock_detect_step_type.return_value = "UnknownStepType"
        script_name = "unknown_script.py"
        
        # Execute
        result = router.enhance_validation(script_name, mock_validation_result)
        
        # Verify - should return original result unchanged
        mock_detect_step_type.assert_called_once_with(script_name)
        assert result == mock_validation_result

    @patch('cursus.validation.alignment.step_type_enhancement_router.detect_step_type_from_registry')
    def test_enhance_validation_none_step_type(self, mock_detect_step_type, router, mock_validation_result):
        """Test validation enhancement when step type detection returns None."""
        # Setup
        mock_detect_step_type.return_value = None
        script_name = "unknown_script.py"
        
        # Execute
        result = router.enhance_validation(script_name, mock_validation_result)
        
        # Verify - should return original result unchanged
        mock_detect_step_type.assert_called_once_with(script_name)
        assert result == mock_validation_result

    def test_get_step_type_requirements_training(self, router):
        """Test getting requirements for training step type."""
        requirements = router.get_step_type_requirements("Training")
        
        assert "input_types" in requirements
        assert "output_types" in requirements
        assert "required_methods" in requirements
        assert "required_patterns" in requirements
        
        # Check specific training requirements
        assert "TrainingInput" in requirements["input_types"]
        assert "model_artifacts" in requirements["output_types"]
        assert "_create_estimator" in requirements["required_methods"]
        assert "training_loop" in requirements["required_patterns"]

    def test_get_step_type_requirements_processing(self, router):
        """Test getting requirements for processing step type."""
        requirements = router.get_step_type_requirements("Processing")
        
        assert "input_types" in requirements
        assert "output_types" in requirements
        assert "required_methods" in requirements
        assert "required_patterns" in requirements
        
        # Check specific processing requirements
        assert "ProcessingInput" in requirements["input_types"]
        assert "ProcessingOutput" in requirements["output_types"]
        assert "_create_processor" in requirements["required_methods"]
        assert "data_transformation" in requirements["required_patterns"]

    def test_get_step_type_requirements_createmodel(self, router):
        """Test getting requirements for CreateModel step type."""
        requirements = router.get_step_type_requirements("CreateModel")
        
        assert "input_types" in requirements
        assert "output_types" in requirements
        assert "required_methods" in requirements
        assert "required_patterns" in requirements
        
        # Check specific CreateModel requirements
        assert "model_artifacts" in requirements["input_types"]
        assert "model_endpoint" in requirements["output_types"]
        assert "_create_model" in requirements["required_methods"]
        assert "model_loading" in requirements["required_patterns"]

    def test_get_step_type_requirements_transform(self, router):
        """Test getting requirements for Transform step type."""
        requirements = router.get_step_type_requirements("Transform")
        
        assert "input_types" in requirements
        assert "output_types" in requirements
        assert "required_methods" in requirements
        assert "required_patterns" in requirements
        
        # Check specific Transform requirements
        assert "TransformInput" in requirements["input_types"]
        assert "transform_results" in requirements["output_types"]
        assert "_create_transformer" in requirements["required_methods"]
        assert "batch_processing" in requirements["required_patterns"]

    def test_get_step_type_requirements_registermodel(self, router):
        """Test getting requirements for RegisterModel step type."""
        requirements = router.get_step_type_requirements("RegisterModel")
        
        assert "input_types" in requirements
        assert "output_types" in requirements
        assert "required_methods" in requirements
        assert "required_patterns" in requirements
        
        # Check specific RegisterModel requirements
        assert "model_artifacts" in requirements["input_types"]
        assert "registered_model" in requirements["output_types"]
        assert "_create_model_package" in requirements["required_methods"]
        assert "model_metadata" in requirements["required_patterns"]

    def test_get_step_type_requirements_utility(self, router):
        """Test getting requirements for Utility step type."""
        requirements = router.get_step_type_requirements("Utility")
        
        assert "input_types" in requirements
        assert "output_types" in requirements
        assert "required_methods" in requirements
        assert "required_patterns" in requirements
        
        # Check specific Utility requirements
        assert "various" in requirements["input_types"]
        assert "prepared_files" in requirements["output_types"]
        assert "_prepare_files" in requirements["required_methods"]
        assert "file_preparation" in requirements["required_patterns"]

    def test_get_step_type_requirements_base(self, router):
        """Test getting requirements for Base step type."""
        requirements = router.get_step_type_requirements("Base")
        
        assert "input_types" in requirements
        assert "output_types" in requirements
        assert "required_methods" in requirements
        assert "required_patterns" in requirements
        
        # Check specific Base requirements
        assert "base_inputs" in requirements["input_types"]
        assert "base_outputs" in requirements["output_types"]
        assert "create_step" in requirements["required_methods"]
        assert "foundation_patterns" in requirements["required_patterns"]

    def test_get_step_type_requirements_unknown(self, router):
        """Test getting requirements for unknown step type."""
        requirements = router.get_step_type_requirements("UnknownStepType")
        
        # Should return empty dict for unknown step type
        assert requirements == {}

    @patch('cursus.validation.alignment.step_type_enhancement_router.detect_step_type_from_registry')
    def test_enhance_validation_with_exception(self, mock_detect_step_type, router, mock_validation_result):
        """Test validation enhancement when enhancer raises exception."""
        # Setup
        mock_detect_step_type.return_value = "Training"
        script_name = "xgboost_training.py"
        
        # Mock the training enhancer to raise exception
        mock_training_enhancer = Mock()
        mock_training_enhancer.enhance_validation.side_effect = Exception("Test exception")
        router.enhancers["Training"] = mock_training_enhancer
        
        # Execute - should handle exception gracefully
        result = router.enhance_validation(script_name, mock_validation_result)
        
        # Verify - should return original result when exception occurs
        assert result == mock_validation_result

    def test_all_step_types_have_requirements(self, router):
        """Test that all step types in enhancers have corresponding requirements."""
        for step_type in router.enhancers.keys():
            requirements = router.get_step_type_requirements(step_type)
            
            # Each step type should have requirements (except for unknown types)
            if step_type != "Base":  # Base might have minimal requirements
                assert isinstance(requirements, dict)
                if requirements:  # If requirements exist, they should have expected keys
                    expected_keys = ["input_types", "output_types", "required_methods", "required_patterns"]
                    for key in expected_keys:
                        assert key in requirements

    @patch('cursus.validation.alignment.step_type_enhancement_router.detect_step_type_from_registry')
    def test_enhance_validation_preserves_original_issues(self, mock_detect_step_type, router):
        """Test that enhancement preserves original validation issues."""
        # Setup
        mock_detect_step_type.return_value = "Training"
        script_name = "xgboost_training.py"
        
        # Create validation result with existing issues
        original_issue = StepTypeAwareAlignmentIssue(
            level=SeverityLevel.ERROR,
            category="original_issue",
            message="Original issue message",
            recommendation="Original suggestion"
        )
        validation_result_with_issues = ValidationResult(
            is_valid=False,
            issues=[original_issue],
            summary={"message": "Validation with existing issues"},
            metadata={"script_name": script_name}
        )
        
        # Mock the training enhancer to add additional issues
        additional_issue = StepTypeAwareAlignmentIssue(
            level=SeverityLevel.WARNING,
            category="training_issue",
            message="Training issue message",
            recommendation="Training suggestion",
            step_type="Training"
        )
        enhanced_result = ValidationResult(
            is_valid=False,
            issues=[original_issue, additional_issue],
            summary={"message": "Enhanced validation result"},
            metadata={"script_name": script_name}
        )
        
        mock_training_enhancer = Mock()
        mock_training_enhancer.enhance_validation.return_value = enhanced_result
        router.enhancers["Training"] = mock_training_enhancer
        
        # Execute
        result = router.enhance_validation(script_name, validation_result_with_issues)
        
        # Verify that both original and new issues are present
        assert len(result.issues) == 2
        assert original_issue in result.issues
        assert additional_issue in result.issues

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

"""
Unit tests for step_type_enhancement_router.py module.

Tests step type enhancement routing and orchestration functionality.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from cursus.validation.alignment.step_type_enhancement_router import (
    StepTypeEnhancementRouter
)
from cursus.validation.alignment.core_models import (
    ValidationResult,
    StepTypeAwareAlignmentIssue,
    SeverityLevel
)

class TestStepTypeEnhancementRouter(unittest.TestCase):
    """Test step type enhancement router functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.router = StepTypeEnhancementRouter()
        
        # Mock validation result
        self.mock_validation_result = ValidationResult(
            is_valid=True,
            issues=[],
            summary={"message": "Test validation result"},
            metadata={"script_name": "test_script.py"}
        )
        
        # Mock alignment issues
        self.mock_issue = StepTypeAwareAlignmentIssue(
            level=SeverityLevel.WARNING,
            category="test_category",
            message="Test message",
            recommendation="Test suggestion",
            step_type="Training",
            framework_context="xgboost"
        )

    def test_router_initialization(self):
        """Test router initialization with all enhancers."""
        self.assertIsNotNone(self.router._enhancers)
        
        # Check that all expected step types have enhancer classes defined
        expected_step_types = [
            "Processing", "Training", "CreateModel", 
            "Transform", "RegisterModel", "Utility", "Base"
        ]
        
        for step_type in expected_step_types:
            self.assertIn(step_type, self.router._enhancer_classes)
            self.assertIsNotNone(self.router._enhancer_classes[step_type])

    @patch('src.cursus.validation.alignment.step_type_enhancement_router.detect_step_type_from_registry')
    def test_enhance_validation_training_step(self, mock_detect_step_type):
        """Test validation enhancement for training step."""
        # Setup
        mock_detect_step_type.return_value = "Training"
        script_name = "xgboost_training.py"
        
        # Mock the training enhancer
        mock_training_enhancer = Mock()
        mock_training_enhancer.enhance_validation.return_value = self.mock_validation_result
        self.router._enhancers["Training"] = mock_training_enhancer
        
        # Execute
        result = self.router.enhance_validation(script_name, self.mock_validation_result)
        
        # Verify
        mock_detect_step_type.assert_called_once_with(script_name)
        mock_training_enhancer.enhance_validation.assert_called_once_with(
            self.mock_validation_result, script_name
        )
        self.assertEqual(result, self.mock_validation_result)

    @patch('src.cursus.validation.alignment.step_type_enhancement_router.detect_step_type_from_registry')
    def test_enhance_validation_processing_step(self, mock_detect_step_type):
        """Test validation enhancement for processing step."""
        # Setup
        mock_detect_step_type.return_value = "Processing"
        script_name = "tabular_preprocessing.py"
        
        # Mock the processing enhancer
        mock_processing_enhancer = Mock()
        mock_processing_enhancer.enhance_validation.return_value = self.mock_validation_result
        self.router._enhancers["Processing"] = mock_processing_enhancer
        
        # Execute
        result = self.router.enhance_validation(script_name, self.mock_validation_result)
        
        # Verify
        mock_detect_step_type.assert_called_once_with(script_name)
        mock_processing_enhancer.enhance_validation.assert_called_once_with(
            self.mock_validation_result, script_name
        )
        self.assertEqual(result, self.mock_validation_result)

    @patch('src.cursus.validation.alignment.step_type_enhancement_router.detect_step_type_from_registry')
    def test_enhance_validation_unknown_step_type(self, mock_detect_step_type):
        """Test validation enhancement for unknown step type."""
        # Setup
        mock_detect_step_type.return_value = "UnknownStepType"
        script_name = "unknown_script.py"
        
        # Execute
        result = self.router.enhance_validation(script_name, self.mock_validation_result)
        
        # Verify - should return original result unchanged
        mock_detect_step_type.assert_called_once_with(script_name)
        self.assertEqual(result, self.mock_validation_result)

    @patch('src.cursus.validation.alignment.step_type_enhancement_router.detect_step_type_from_registry')
    def test_enhance_validation_none_step_type(self, mock_detect_step_type):
        """Test validation enhancement when step type detection returns None."""
        # Setup
        mock_detect_step_type.return_value = None
        script_name = "unknown_script.py"
        
        # Execute
        result = self.router.enhance_validation(script_name, self.mock_validation_result)
        
        # Verify - should return original result unchanged
        mock_detect_step_type.assert_called_once_with(script_name)
        self.assertEqual(result, self.mock_validation_result)

    def test_get_step_type_requirements_training(self):
        """Test getting requirements for training step type."""
        requirements = self.router.get_step_type_requirements("Training")
        
        self.assertIn("input_types", requirements)
        self.assertIn("output_types", requirements)
        self.assertIn("required_methods", requirements)
        self.assertIn("required_patterns", requirements)
        
        # Check specific training requirements
        self.assertIn("TrainingInput", requirements["input_types"])
        self.assertIn("model_artifacts", requirements["output_types"])
        self.assertIn("_create_estimator", requirements["required_methods"])
        self.assertIn("training_loop", requirements["required_patterns"])

    def test_get_step_type_requirements_processing(self):
        """Test getting requirements for processing step type."""
        requirements = self.router.get_step_type_requirements("Processing")
        
        self.assertIn("input_types", requirements)
        self.assertIn("output_types", requirements)
        self.assertIn("required_methods", requirements)
        self.assertIn("required_patterns", requirements)
        
        # Check specific processing requirements
        self.assertIn("ProcessingInput", requirements["input_types"])
        self.assertIn("ProcessingOutput", requirements["output_types"])
        self.assertIn("_create_processor", requirements["required_methods"])
        self.assertIn("data_transformation", requirements["required_patterns"])

    def test_get_step_type_requirements_createmodel(self):
        """Test getting requirements for CreateModel step type."""
        requirements = self.router.get_step_type_requirements("CreateModel")
        
        self.assertIn("input_types", requirements)
        self.assertIn("output_types", requirements)
        self.assertIn("required_methods", requirements)
        self.assertIn("required_patterns", requirements)
        
        # Check specific CreateModel requirements
        self.assertIn("model_artifacts", requirements["input_types"])
        self.assertIn("model_endpoint", requirements["output_types"])
        self.assertIn("_create_model", requirements["required_methods"])
        self.assertIn("model_loading", requirements["required_patterns"])

    def test_get_step_type_requirements_transform(self):
        """Test getting requirements for Transform step type."""
        requirements = self.router.get_step_type_requirements("Transform")
        
        self.assertIn("input_types", requirements)
        self.assertIn("output_types", requirements)
        self.assertIn("required_methods", requirements)
        self.assertIn("required_patterns", requirements)
        
        # Check specific Transform requirements
        self.assertIn("TransformInput", requirements["input_types"])
        self.assertIn("transform_results", requirements["output_types"])
        self.assertIn("_create_transformer", requirements["required_methods"])
        self.assertIn("batch_processing", requirements["required_patterns"])

    def test_get_step_type_requirements_registermodel(self):
        """Test getting requirements for RegisterModel step type."""
        requirements = self.router.get_step_type_requirements("RegisterModel")
        
        self.assertIn("input_types", requirements)
        self.assertIn("output_types", requirements)
        self.assertIn("required_methods", requirements)
        self.assertIn("required_patterns", requirements)
        
        # Check specific RegisterModel requirements
        self.assertIn("model_artifacts", requirements["input_types"])
        self.assertIn("registered_model", requirements["output_types"])
        self.assertIn("_create_model_package", requirements["required_methods"])
        self.assertIn("model_metadata", requirements["required_patterns"])

    def test_get_step_type_requirements_utility(self):
        """Test getting requirements for Utility step type."""
        requirements = self.router.get_step_type_requirements("Utility")
        
        self.assertIn("input_types", requirements)
        self.assertIn("output_types", requirements)
        self.assertIn("required_methods", requirements)
        self.assertIn("required_patterns", requirements)
        
        # Check specific Utility requirements
        self.assertIn("various", requirements["input_types"])
        self.assertIn("prepared_files", requirements["output_types"])
        self.assertIn("_prepare_files", requirements["required_methods"])
        self.assertIn("file_preparation", requirements["required_patterns"])

    def test_get_step_type_requirements_base(self):
        """Test getting requirements for Base step type."""
        requirements = self.router.get_step_type_requirements("Base")
        
        self.assertIn("input_types", requirements)
        self.assertIn("output_types", requirements)
        self.assertIn("required_methods", requirements)
        self.assertIn("required_patterns", requirements)
        
        # Check specific Base requirements
        self.assertIn("base_inputs", requirements["input_types"])
        self.assertIn("base_outputs", requirements["output_types"])
        self.assertIn("create_step", requirements["required_methods"])
        self.assertIn("foundation_patterns", requirements["required_patterns"])

    def test_get_step_type_requirements_unknown(self):
        """Test getting requirements for unknown step type."""
        requirements = self.router.get_step_type_requirements("UnknownStepType")
        
        # Should return empty dict for unknown step type
        self.assertEqual(requirements, {})

    @patch('src.cursus.validation.alignment.step_type_enhancement_router.detect_step_type_from_registry')
    def test_enhance_validation_with_exception(self, mock_detect_step_type):
        """Test validation enhancement when enhancer raises exception."""
        # Setup
        mock_detect_step_type.return_value = "Training"
        script_name = "xgboost_training.py"
        
        # Mock the training enhancer to raise exception
        mock_training_enhancer = Mock()
        mock_training_enhancer.enhance_validation.side_effect = Exception("Test exception")
        self.router.enhancers["Training"] = mock_training_enhancer
        
        # Execute - should handle exception gracefully
        result = self.router.enhance_validation(script_name, self.mock_validation_result)
        
        # Verify - should return original result when exception occurs
        self.assertEqual(result, self.mock_validation_result)

    def test_all_step_types_have_requirements(self):
        """Test that all step types in enhancers have corresponding requirements."""
        for step_type in self.router.enhancers.keys():
            requirements = self.router.get_step_type_requirements(step_type)
            
            # Each step type should have requirements (except for unknown types)
            if step_type != "Base":  # Base might have minimal requirements
                self.assertIsInstance(requirements, dict)
                if requirements:  # If requirements exist, they should have expected keys
                    expected_keys = ["input_types", "output_types", "required_methods", "required_patterns"]
                    for key in expected_keys:
                        self.assertIn(key, requirements)

    @patch('src.cursus.validation.alignment.step_type_enhancement_router.detect_step_type_from_registry')
    def test_enhance_validation_preserves_original_issues(self, mock_detect_step_type):
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
        self.router.enhancers["Training"] = mock_training_enhancer
        
        # Execute
        result = self.router.enhance_validation(script_name, validation_result_with_issues)
        
        # Verify that both original and new issues are present
        self.assertEqual(len(result.issues), 2)
        self.assertIn(original_issue, result.issues)
        self.assertIn(additional_issue, result.issues)

if __name__ == '__main__':
    unittest.main()

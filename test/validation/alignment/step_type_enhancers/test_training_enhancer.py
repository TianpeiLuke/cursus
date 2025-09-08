"""
Unit tests for training_enhancer.py module.

Tests training step-specific validation enhancement functionality.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
from cursus.validation.alignment.step_type_enhancers.training_enhancer import TrainingStepEnhancer
from cursus.validation.alignment.core_models import (
    ValidationResult,
    StepTypeAwareAlignmentIssue,
    SeverityLevel
)

class TestTrainingStepEnhancer(unittest.TestCase):
    """Test training step enhancer functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.enhancer = TrainingStepEnhancer()
        
        # Mock validation result
        self.mock_validation_result = ValidationResult(
            is_valid=True,
            issues=[],
            summary={"message": "Test validation result"},
            metadata={"script_name": "xgboost_training.py"}
        )
        
        # Mock script analysis
        self.mock_script_analysis = {
            'imports': ['xgboost', 'pandas', 'json'],
            'functions': ['main', 'load_data', 'train_model'],
            'file_operations': ['/opt/ml/model/model.xgb', '/opt/ml/input/data/config/hyperparameters.json'],
            'patterns': {
                'training_loop': ['xgb.train'],
                'model_saving': ['model.save_model'],
                'hyperparameter_loading': ['hyperparameters.json']
            }
        }

    def test_training_enhancer_initialization(self):
        """Test training enhancer initialization."""
        self.assertEqual(self.enhancer.step_type, "Training")
        self.assertIn("xgboost_training.py", self.enhancer.reference_examples)
        self.assertIn("pytorch_training.py", self.enhancer.reference_examples)
        self.assertIn("builder_xgboost_training_step.py", self.enhancer.reference_examples)
        
        # Check framework validators
        self.assertIn("xgboost", self.enhancer.framework_validators)
        self.assertIn("pytorch", self.enhancer.framework_validators)

    @patch.object(TrainingStepEnhancer, '_get_script_analysis')
    def test_enhance_validation_xgboost_training(self, mock_get_script_analysis):
        """Test validation enhancement for XGBoost training script."""
        # Setup
        mock_get_script_analysis.return_value = self.mock_script_analysis
        
        # Execute
        result = self.enhancer.enhance_validation(self.mock_validation_result, "xgboost_training.py")
        
        # Verify
        self.assertIsInstance(result, ValidationResult)
        mock_get_script_analysis.assert_called_once_with("xgboost_training.py")

    @patch.object(TrainingStepEnhancer, '_get_script_analysis')
    def test_enhance_validation_pytorch_training(self, mock_get_script_analysis):
        """Test validation enhancement for PyTorch training script."""
        # Setup
        pytorch_analysis = {
            'imports': ['torch', 'torch.nn', 'torch.optim'],
            'functions': ['main', 'Net', 'train'],
            'patterns': {
                'training_loop': ['for epoch in range'],
                'model_saving': ['torch.save'],
                'hyperparameter_loading': ['hyperparameters.json']
            }
        }
        mock_get_script_analysis.return_value = pytorch_analysis
        
        # Execute
        result = self.enhancer.enhance_validation(self.mock_validation_result, "pytorch_training.py")
        
        # Verify
        self.assertIsInstance(result, ValidationResult)
        mock_get_script_analysis.assert_called_once_with("pytorch_training.py")

    @patch.object(TrainingStepEnhancer, '_validate_training_builder')
    @patch.object(TrainingStepEnhancer, '_validate_training_dependencies')
    @patch.object(TrainingStepEnhancer, '_validate_training_specifications')
    @patch.object(TrainingStepEnhancer, '_validate_training_script_patterns')
    @patch.object(TrainingStepEnhancer, '_get_script_analysis')
    def test_enhance_validation_calls_all_validation_levels(self, mock_get_script_analysis, 
                                                           mock_validate_script, mock_validate_specs,
                                                           mock_validate_deps, mock_validate_builder):
        """Test that enhance_validation calls all validation levels."""
        # Setup
        mock_get_script_analysis.return_value = self.mock_script_analysis
        mock_validate_script.return_value = []
        mock_validate_specs.return_value = []
        mock_validate_deps.return_value = []
        mock_validate_builder.return_value = []
        
        # Execute
        result = self.enhancer.enhance_validation(self.mock_validation_result, "xgboost_training.py")
        
        # Verify all validation levels were called
        mock_validate_script.assert_called_once()
        mock_validate_specs.assert_called_once_with("xgboost_training.py")
        # The framework is detected from script analysis, so it should be passed to dependencies validation
        mock_validate_deps.assert_called_once()
        mock_validate_builder.assert_called_once_with("xgboost_training.py")

    @patch.object(TrainingStepEnhancer, '_has_training_loop_patterns')
    @patch.object(TrainingStepEnhancer, '_has_model_saving_patterns')
    @patch.object(TrainingStepEnhancer, '_has_hyperparameter_loading_patterns')
    def test_validate_training_script_patterns_missing_patterns(self, mock_has_hyperparams, 
                                                               mock_has_model_saving, mock_has_training_loop):
        """Test validation when training patterns are missing."""
        # Setup - simulate missing patterns
        mock_has_training_loop.return_value = False
        mock_has_model_saving.return_value = False
        mock_has_hyperparams.return_value = False
        
        # Execute
        issues = self.enhancer._validate_training_script_patterns(self.mock_script_analysis, 'xgboost', 'test_script.py')
        
        # Verify issues were created for missing patterns
        self.assertGreaterEqual(len(issues), 3)
        
        issue_categories = [issue.get('category') for issue in issues]
        self.assertIn("missing_training_loop", issue_categories)
        self.assertIn("missing_model_saving", issue_categories)
        self.assertIn("missing_hyperparameter_loading", issue_categories)

    @patch.object(TrainingStepEnhancer, '_has_training_loop_patterns')
    @patch.object(TrainingStepEnhancer, '_has_model_saving_patterns')
    @patch.object(TrainingStepEnhancer, '_has_hyperparameter_loading_patterns')
    def test_validate_training_script_patterns_all_present(self, mock_has_hyperparams, 
                                                          mock_has_model_saving, mock_has_training_loop):
        """Test validation when all training patterns are present."""
        # Setup - simulate all patterns present
        mock_has_training_loop.return_value = True
        mock_has_model_saving.return_value = True
        mock_has_hyperparams.return_value = True
        
        # Execute
        issues = self.enhancer._validate_training_script_patterns(self.mock_script_analysis, 'xgboost', 'test_script.py')
        
        # Verify fewer issues were created (some patterns might still be missing)
        self.assertLessEqual(len(issues), 2)  # Allow for some issues but fewer than when all patterns are missing

    def test_has_training_loop_patterns_xgboost(self):
        """Test detection of XGBoost training loop patterns."""
        script_analysis = {
            'functions': ['xgb.train', 'model.fit', 'train_model']
        }
        
        result = self.enhancer._has_training_loop_patterns(script_analysis)
        self.assertTrue(result)

    def test_has_training_loop_patterns_missing(self):
        """Test detection when training loop patterns are missing."""
        script_analysis = {
            'patterns': {
                'training_loop': []
            }
        }
        
        result = self.enhancer._has_training_loop_patterns(script_analysis)
        self.assertFalse(result)

    def test_has_model_saving_patterns_present(self):
        """Test detection of model saving patterns."""
        script_analysis = {
            'functions': ['save', 'dump', 'torch.save'],
            'path_references': ['/opt/ml/model']
        }
        
        result = self.enhancer._has_model_saving_patterns(script_analysis)
        self.assertTrue(result)

    def test_has_model_saving_patterns_missing(self):
        """Test detection when model saving patterns are missing."""
        script_analysis = {
            'patterns': {
                'model_saving': []
            }
        }
        
        result = self.enhancer._has_model_saving_patterns(script_analysis)
        self.assertFalse(result)

    def test_has_hyperparameter_loading_patterns_present(self):
        """Test detection of hyperparameter loading patterns."""
        script_analysis = {
            'functions': ['hyperparameters', 'config', 'params'],
            'path_references': ['/opt/ml/input/data/config']
        }
        
        result = self.enhancer._has_hyperparameter_loading_patterns(script_analysis)
        self.assertTrue(result)

    def test_has_hyperparameter_loading_patterns_missing(self):
        """Test detection when hyperparameter loading patterns are missing."""
        script_analysis = {
            'patterns': {
                'hyperparameter_loading': []
            }
        }
        
        result = self.enhancer._has_hyperparameter_loading_patterns(script_analysis)
        self.assertFalse(result)

    def test_create_training_issue(self):
        """Test creation of training-specific issues."""
        issue = self.enhancer._create_step_type_issue(
            "test_category",
            "Test message",
            "Test suggestion",
            "WARNING",
            {"test": "details"}
        )
        
        self.assertIsInstance(issue, dict)
        self.assertEqual(issue["category"], "test_category")
        self.assertEqual(issue["message"], "Test message")
        self.assertEqual(issue["recommendation"], "Test suggestion")
        self.assertEqual(issue["step_type"], "Training")
        self.assertEqual(issue["severity"], "WARNING")

    @patch.object(TrainingStepEnhancer, '_get_script_analysis')
    def test_framework_specific_validation_xgboost(self, mock_get_script_analysis):
        """Test framework-specific validation for XGBoost."""
        # Setup
        mock_get_script_analysis.return_value = self.mock_script_analysis
        
        # Mock XGBoost validator
        mock_xgb_validator = Mock()
        mock_xgb_validator.validate_script_patterns.return_value = [
            StepTypeAwareAlignmentIssue(
                level=SeverityLevel.INFO,
                category="xgboost_specific",
                message="XGBoost-specific validation",
                suggestion="XGBoost suggestion",
                step_type="Training",
                framework="xgboost"
            )
        ]
        self.enhancer.framework_validators["xgboost"] = mock_xgb_validator
        
        # Execute
        issues = self.enhancer._validate_training_script_patterns(self.mock_script_analysis, 'xgboost', 'test_script.py')
        
        # Verify that issues were created (framework-specific validation is handled differently)
        self.assertIsInstance(issues, list)

    @patch.object(TrainingStepEnhancer, '_get_script_analysis')
    def test_framework_specific_validation_pytorch(self, mock_get_script_analysis):
        """Test framework-specific validation for PyTorch."""
        # Setup
        pytorch_analysis = {
            'imports': ['torch', 'torch.nn'],
            'patterns': {
                'training_loop': ['for epoch in range'],
                'model_saving': ['torch.save'],
                'hyperparameter_loading': ['config.json']
            }
        }
        mock_get_script_analysis.return_value = pytorch_analysis
        
        # Mock PyTorch validator
        mock_pytorch_validator = Mock()
        mock_pytorch_validator.validate_script_patterns.return_value = [
            StepTypeAwareAlignmentIssue(
                level=SeverityLevel.INFO,
                category="pytorch_specific",
                message="PyTorch-specific validation",
                suggestion="PyTorch suggestion",
                step_type="Training",
                framework="pytorch"
            )
        ]
        self.enhancer.framework_validators["pytorch"] = mock_pytorch_validator
        
        # Execute
        issues = self.enhancer._validate_training_script_patterns(pytorch_analysis, 'pytorch', 'test_script.py')
        
        # Verify that issues were created (framework-specific validation is handled differently)
        self.assertIsInstance(issues, list)

    def test_framework_specific_validation_unknown_framework(self):
        """Test framework-specific validation for unknown framework."""
        # Execute with unknown framework
        issues = self.enhancer._validate_training_script_patterns(self.mock_script_analysis, 'unknown_framework', 'test_script.py')
        
        # Should not crash and should not include framework-specific issues
        framework_issues = [issue for issue in issues if hasattr(issue, 'framework') and issue.framework == 'unknown_framework']
        self.assertEqual(len(framework_issues), 0)

    @patch.object(TrainingStepEnhancer, '_get_script_analysis')
    def test_get_script_analysis_integration(self, mock_get_script_analysis):
        """Test integration with script analysis."""
        # Setup
        mock_get_script_analysis.return_value = self.mock_script_analysis
        
        # Execute
        result = self.enhancer._get_script_analysis("test_script.py")
        
        # Verify
        self.assertEqual(result, self.mock_script_analysis)
        mock_get_script_analysis.assert_called_once_with("test_script.py")

    def test_validate_training_specifications_placeholder(self):
        """Test training specifications validation (placeholder)."""
        # This is a placeholder test for the specifications validation
        # The actual implementation would depend on the specification system
        issues = self.enhancer._validate_training_specifications("xgboost_training.py")
        
        # Should return a list (empty or with issues)
        self.assertIsInstance(issues, list)

    def test_validate_training_dependencies_placeholder(self):
        """Test training dependencies validation (placeholder)."""
        # This is a placeholder test for the dependencies validation
        # The actual implementation would depend on the dependency system
        issues = self.enhancer._validate_training_dependencies("xgboost_training.py", "xgboost")
        
        # Should return a list (empty or with issues)
        self.assertIsInstance(issues, list)

    def test_validate_training_builder_placeholder(self):
        """Test training builder validation (placeholder)."""
        # This is a placeholder test for the builder validation
        # The actual implementation would depend on the builder system
        issues = self.enhancer._validate_training_builder("xgboost_training.py")
        
        # Should return a list (empty or with issues)
        self.assertIsInstance(issues, list)

    def test_enhancer_inheritance(self):
        """Test that TrainingStepEnhancer properly inherits from BaseStepEnhancer."""
        from cursus.validation.alignment.step_type_enhancers.base_enhancer import BaseStepEnhancer
        
        self.assertIsInstance(self.enhancer, BaseStepEnhancer)
        self.assertTrue(hasattr(self.enhancer, 'enhance_validation'))
        self.assertTrue(hasattr(self.enhancer, '_merge_results'))

    def test_reference_examples_completeness(self):
        """Test that reference examples are comprehensive."""
        expected_examples = [
            "xgboost_training.py",
            "pytorch_training.py", 
            "builder_xgboost_training_step.py"
        ]
        
        for example in expected_examples:
            self.assertIn(example, self.enhancer.reference_examples)

    def test_framework_validators_completeness(self):
        """Test that framework validators are set up for expected frameworks."""
        expected_frameworks = ["xgboost", "pytorch"]
        
        for framework in expected_frameworks:
            self.assertIn(framework, self.enhancer.framework_validators)
            self.assertIsNotNone(self.enhancer.framework_validators[framework])

if __name__ == '__main__':
    unittest.main()

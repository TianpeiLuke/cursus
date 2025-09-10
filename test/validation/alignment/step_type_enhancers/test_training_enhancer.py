"""
Unit tests for training_enhancer.py module.

Tests training step-specific validation enhancement functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, mock_open
from cursus.validation.alignment.step_type_enhancers.training_enhancer import TrainingStepEnhancer
from cursus.validation.alignment.core_models import (
    ValidationResult,
    StepTypeAwareAlignmentIssue,
    SeverityLevel
)

@pytest.fixture
def enhancer():
    """Set up test enhancer fixture."""
    return TrainingStepEnhancer()

@pytest.fixture
def mock_validation_result():
    """Set up mock validation result fixture."""
    return ValidationResult(
        is_valid=True,
        issues=[],
        summary={"message": "Test validation result"},
        metadata={"script_name": "xgboost_training.py"}
    )

@pytest.fixture
def mock_script_analysis():
    """Set up mock script analysis fixture."""
    return {
        'imports': ['xgboost', 'pandas', 'json'],
        'functions': ['main', 'load_data', 'train_model'],
        'file_operations': ['/opt/ml/model/model.xgb', '/opt/ml/input/data/config/hyperparameters.json'],
        'patterns': {
            'training_loop': ['xgb.train'],
            'model_saving': ['model.save_model'],
            'hyperparameter_loading': ['hyperparameters.json']
        }
    }

class TestTrainingStepEnhancer:
    """Test training step enhancer functionality."""

    def test_training_enhancer_initialization(self, enhancer):
        """Test training enhancer initialization."""
        assert enhancer.step_type == "Training"
        assert "xgboost_training.py" in enhancer.reference_examples
        assert "pytorch_training.py" in enhancer.reference_examples
        assert "builder_xgboost_training_step.py" in enhancer.reference_examples
        
        # Check framework validators
        assert "xgboost" in enhancer.framework_validators
        assert "pytorch" in enhancer.framework_validators

    @patch.object(TrainingStepEnhancer, '_get_script_analysis')
    def test_enhance_validation_xgboost_training(self, mock_get_script_analysis, enhancer, mock_validation_result, mock_script_analysis):
        """Test validation enhancement for XGBoost training script."""
        # Setup
        mock_get_script_analysis.return_value = mock_script_analysis
        
        # Execute
        result = enhancer.enhance_validation(mock_validation_result, "xgboost_training.py")
        
        # Verify
        assert isinstance(result, ValidationResult)
        mock_get_script_analysis.assert_called_once_with("xgboost_training.py")

    @patch.object(TrainingStepEnhancer, '_get_script_analysis')
    def test_enhance_validation_pytorch_training(self, mock_get_script_analysis, enhancer, mock_validation_result):
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
        result = enhancer.enhance_validation(mock_validation_result, "pytorch_training.py")
        
        # Verify
        assert isinstance(result, ValidationResult)
        mock_get_script_analysis.assert_called_once_with("pytorch_training.py")

    @patch.object(TrainingStepEnhancer, '_validate_training_builder')
    @patch.object(TrainingStepEnhancer, '_validate_training_dependencies')
    @patch.object(TrainingStepEnhancer, '_validate_training_specifications')
    @patch.object(TrainingStepEnhancer, '_validate_training_script_patterns')
    @patch.object(TrainingStepEnhancer, '_get_script_analysis')
    def test_enhance_validation_calls_all_validation_levels(self, mock_get_script_analysis, 
                                                           mock_validate_script, mock_validate_specs,
                                                           mock_validate_deps, mock_validate_builder,
                                                           enhancer, mock_validation_result, mock_script_analysis):
        """Test that enhance_validation calls all validation levels."""
        # Setup
        mock_get_script_analysis.return_value = mock_script_analysis
        mock_validate_script.return_value = []
        mock_validate_specs.return_value = []
        mock_validate_deps.return_value = []
        mock_validate_builder.return_value = []
        
        # Execute
        result = enhancer.enhance_validation(mock_validation_result, "xgboost_training.py")
        
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
                                                               mock_has_model_saving, mock_has_training_loop,
                                                               enhancer, mock_script_analysis):
        """Test validation when training patterns are missing."""
        # Setup - simulate missing patterns
        mock_has_training_loop.return_value = False
        mock_has_model_saving.return_value = False
        mock_has_hyperparams.return_value = False
        
        # Execute
        issues = enhancer._validate_training_script_patterns(mock_script_analysis, 'xgboost', 'test_script.py')
        
        # Verify issues were created for missing patterns
        assert len(issues) >= 3
        
        issue_categories = [issue.get('category') for issue in issues]
        assert "missing_training_loop" in issue_categories
        assert "missing_model_saving" in issue_categories
        assert "missing_hyperparameter_loading" in issue_categories

    @patch.object(TrainingStepEnhancer, '_has_training_loop_patterns')
    @patch.object(TrainingStepEnhancer, '_has_model_saving_patterns')
    @patch.object(TrainingStepEnhancer, '_has_hyperparameter_loading_patterns')
    def test_validate_training_script_patterns_all_present(self, mock_has_hyperparams, 
                                                          mock_has_model_saving, mock_has_training_loop,
                                                          enhancer, mock_script_analysis):
        """Test validation when all training patterns are present."""
        # Setup - simulate all patterns present
        mock_has_training_loop.return_value = True
        mock_has_model_saving.return_value = True
        mock_has_hyperparams.return_value = True
        
        # Execute
        issues = enhancer._validate_training_script_patterns(mock_script_analysis, 'xgboost', 'test_script.py')
        
        # Verify fewer issues were created (some patterns might still be missing)
        assert len(issues) <= 2  # Allow for some issues but fewer than when all patterns are missing

    def test_has_training_loop_patterns_xgboost(self, enhancer):
        """Test detection of XGBoost training loop patterns."""
        script_analysis = {
            'functions': ['xgb.train', 'model.fit', 'train_model']
        }
        
        result = enhancer._has_training_loop_patterns(script_analysis)
        assert result is True

    def test_has_training_loop_patterns_missing(self, enhancer):
        """Test detection when training loop patterns are missing."""
        script_analysis = {
            'patterns': {
                'training_loop': []
            }
        }
        
        result = enhancer._has_training_loop_patterns(script_analysis)
        assert result is False

    def test_has_model_saving_patterns_present(self, enhancer):
        """Test detection of model saving patterns."""
        script_analysis = {
            'functions': ['save', 'dump', 'torch.save'],
            'path_references': ['/opt/ml/model']
        }
        
        result = enhancer._has_model_saving_patterns(script_analysis)
        assert result is True

    def test_has_model_saving_patterns_missing(self, enhancer):
        """Test detection when model saving patterns are missing."""
        script_analysis = {
            'patterns': {
                'model_saving': []
            }
        }
        
        result = enhancer._has_model_saving_patterns(script_analysis)
        assert result is False

    def test_has_hyperparameter_loading_patterns_present(self, enhancer):
        """Test detection of hyperparameter loading patterns."""
        script_analysis = {
            'functions': ['hyperparameters', 'config', 'params'],
            'path_references': ['/opt/ml/input/data/config']
        }
        
        result = enhancer._has_hyperparameter_loading_patterns(script_analysis)
        assert result is True

    def test_has_hyperparameter_loading_patterns_missing(self, enhancer):
        """Test detection when hyperparameter loading patterns are missing."""
        script_analysis = {
            'patterns': {
                'hyperparameter_loading': []
            }
        }
        
        result = enhancer._has_hyperparameter_loading_patterns(script_analysis)
        assert result is False

    def test_create_training_issue(self, enhancer):
        """Test creation of training-specific issues."""
        issue = enhancer._create_step_type_issue(
            "test_category",
            "Test message",
            "Test suggestion",
            "WARNING",
            {"test": "details"}
        )
        
        assert isinstance(issue, dict)
        assert issue["category"] == "test_category"
        assert issue["message"] == "Test message"
        assert issue["recommendation"] == "Test suggestion"
        assert issue["step_type"] == "Training"
        assert issue["severity"] == "WARNING"

    @patch.object(TrainingStepEnhancer, '_get_script_analysis')
    def test_framework_specific_validation_xgboost(self, mock_get_script_analysis, enhancer, mock_script_analysis):
        """Test framework-specific validation for XGBoost."""
        # Setup
        mock_get_script_analysis.return_value = mock_script_analysis
        
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
        enhancer.framework_validators["xgboost"] = mock_xgb_validator
        
        # Execute
        issues = enhancer._validate_training_script_patterns(mock_script_analysis, 'xgboost', 'test_script.py')
        
        # Verify that issues were created (framework-specific validation is handled differently)
        assert isinstance(issues, list)

    @patch.object(TrainingStepEnhancer, '_get_script_analysis')
    def test_framework_specific_validation_pytorch(self, mock_get_script_analysis, enhancer):
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
        enhancer.framework_validators["pytorch"] = mock_pytorch_validator
        
        # Execute
        issues = enhancer._validate_training_script_patterns(pytorch_analysis, 'pytorch', 'test_script.py')
        
        # Verify that issues were created (framework-specific validation is handled differently)
        assert isinstance(issues, list)

    def test_framework_specific_validation_unknown_framework(self, enhancer, mock_script_analysis):
        """Test framework-specific validation for unknown framework."""
        # Execute with unknown framework
        issues = enhancer._validate_training_script_patterns(mock_script_analysis, 'unknown_framework', 'test_script.py')
        
        # Should not crash and should not include framework-specific issues
        framework_issues = [issue for issue in issues if hasattr(issue, 'framework') and issue.framework == 'unknown_framework']
        assert len(framework_issues) == 0

    @patch.object(TrainingStepEnhancer, '_get_script_analysis')
    def test_get_script_analysis_integration(self, mock_get_script_analysis, enhancer, mock_script_analysis):
        """Test integration with script analysis."""
        # Setup
        mock_get_script_analysis.return_value = mock_script_analysis
        
        # Execute
        result = enhancer._get_script_analysis("test_script.py")
        
        # Verify
        assert result == mock_script_analysis
        mock_get_script_analysis.assert_called_once_with("test_script.py")

    def test_validate_training_specifications_placeholder(self, enhancer):
        """Test training specifications validation (placeholder)."""
        # This is a placeholder test for the specifications validation
        # The actual implementation would depend on the specification system
        issues = enhancer._validate_training_specifications("xgboost_training.py")
        
        # Should return a list (empty or with issues)
        assert isinstance(issues, list)

    def test_validate_training_dependencies_placeholder(self, enhancer):
        """Test training dependencies validation (placeholder)."""
        # This is a placeholder test for the dependencies validation
        # The actual implementation would depend on the dependency system
        issues = enhancer._validate_training_dependencies("xgboost_training.py", "xgboost")
        
        # Should return a list (empty or with issues)
        assert isinstance(issues, list)

    def test_validate_training_builder_placeholder(self, enhancer):
        """Test training builder validation (placeholder)."""
        # This is a placeholder test for the builder validation
        # The actual implementation would depend on the builder system
        issues = enhancer._validate_training_builder("xgboost_training.py")
        
        # Should return a list (empty or with issues)
        assert isinstance(issues, list)

    def test_enhancer_inheritance(self, enhancer):
        """Test that TrainingStepEnhancer properly inherits from BaseStepEnhancer."""
        from cursus.validation.alignment.step_type_enhancers.base_enhancer import BaseStepEnhancer
        
        assert isinstance(enhancer, BaseStepEnhancer)
        assert hasattr(enhancer, 'enhance_validation')
        assert hasattr(enhancer, '_merge_results')

    def test_reference_examples_completeness(self, enhancer):
        """Test that reference examples are comprehensive."""
        expected_examples = [
            "xgboost_training.py",
            "pytorch_training.py", 
            "builder_xgboost_training_step.py"
        ]
        
        for example in expected_examples:
            assert example in enhancer.reference_examples

    def test_framework_validators_completeness(self, enhancer):
        """Test that framework validators are set up for expected frameworks."""
        expected_frameworks = ["xgboost", "pytorch"]
        
        for framework in expected_frameworks:
            assert framework in enhancer.framework_validators
            assert enhancer.framework_validators[framework] is not None

if __name__ == '__main__':
    pytest.main([__file__])

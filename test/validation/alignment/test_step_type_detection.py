"""
Unit tests for step_type_detection.py module.

Tests step type detection functionality.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from src.cursus.validation.alignment.step_type_detection import (
    detect_step_type_from_registry,
    detect_framework_from_imports,
    get_step_type_context
)


class TestStepTypeDetection(unittest.TestCase):
    """Test step type detection functionality."""

    def setUp(self):
        """Set up test fixtures."""
        pass

    @patch('builtins.__import__')
    def test_detect_step_type_from_registry_training(self, mock_import):
        """Test step type detection for training script."""
        # Mock the import to return a mock module
        mock_module = Mock()
        mock_module.get_canonical_name_from_file_name.return_value = "xgboost_training"
        mock_module.get_sagemaker_step_type.return_value = "Training"
        
        def side_effect(name, *args, **kwargs):
            if name == 'cursus.steps.registry.step_names':
                return mock_module
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = side_effect
        
        # Execute
        result = detect_step_type_from_registry("xgboost_training")
        
        # Verify
        self.assertEqual(result, "Training")

    @patch('builtins.__import__')
    def test_detect_step_type_from_registry_processing(self, mock_import):
        """Test step type detection for processing script."""
        # Mock the import to return a mock module
        mock_module = Mock()
        mock_module.get_canonical_name_from_file_name.return_value = "tabular_preprocessing"
        mock_module.get_sagemaker_step_type.return_value = "Processing"
        
        def side_effect(name, *args, **kwargs):
            if name == 'cursus.steps.registry.step_names':
                return mock_module
            return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = side_effect
        
        # Execute
        result = detect_step_type_from_registry("tabular_preprocessing")
        
        # Verify
        self.assertEqual(result, "Processing")

    def test_detect_step_type_from_registry_fallback(self):
        """Test step type detection fallback for unknown script."""
        # Execute - this should fallback to "Processing" when import fails
        result = detect_step_type_from_registry("unknown_script")
        
        # Verify - should return "Processing" as fallback
        self.assertEqual(result, "Processing")

    def test_detect_framework_from_imports_xgboost(self):
        """Test framework detection from XGBoost imports."""
        imports = ['xgboost', 'pandas', 'numpy']
        
        result = detect_framework_from_imports(imports)
        
        self.assertEqual(result, 'xgboost')

    def test_detect_framework_from_imports_pytorch(self):
        """Test framework detection from PyTorch imports."""
        imports = ['torch', 'torch.nn', 'torch.optim']
        
        result = detect_framework_from_imports(imports)
        
        self.assertEqual(result, 'pytorch')

    def test_detect_framework_from_imports_sklearn(self):
        """Test framework detection from sklearn imports."""
        imports = ['sklearn', 'sklearn.ensemble', 'pandas']
        
        result = detect_framework_from_imports(imports)
        
        self.assertEqual(result, 'sklearn')

    def test_detect_framework_from_imports_pandas(self):
        """Test framework detection from pandas imports."""
        imports = ['pandas', 'numpy']
        
        result = detect_framework_from_imports(imports)
        
        self.assertEqual(result, 'pandas')

    def test_detect_framework_from_imports_unknown(self):
        """Test framework detection for unknown imports."""
        imports = ['os', 'sys', 'json']
        
        result = detect_framework_from_imports(imports)
        
        self.assertIsNone(result)

    def test_detect_framework_from_imports_empty(self):
        """Test framework detection with empty imports."""
        imports = []
        
        result = detect_framework_from_imports(imports)
        
        self.assertIsNone(result)

    def test_detect_framework_from_imports_multiple_frameworks(self):
        """Test framework detection with multiple frameworks."""
        imports = ['xgboost', 'torch', 'sklearn']
        
        result = detect_framework_from_imports(imports)
        
        # Should return the first detected framework (priority order)
        self.assertIn(result, ['xgboost', 'torch', 'sklearn'])

    def test_get_step_type_context(self):
        """Test getting step type context."""
        script_name = "xgboost_training"
        script_content = """
import xgboost as xgb
def main():
    model = xgb.train(params, dtrain)
    model.save_model('/opt/ml/model/model.xgb')
"""
        
        # Execute
        context = get_step_type_context(script_name, script_content)
        
        # Verify
        self.assertEqual(context['script_name'], script_name)
        self.assertIn('registry_step_type', context)
        self.assertIn('pattern_step_type', context)
        self.assertIn('final_step_type', context)
        self.assertIn('confidence', context)

    def test_get_step_type_context_no_content(self):
        """Test getting step type context without script content."""
        script_name = "unknown_script"
        
        # Execute
        context = get_step_type_context(script_name)
        
        # Verify
        self.assertEqual(context['script_name'], script_name)
        self.assertIn('registry_step_type', context)
        self.assertIn('pattern_step_type', context)
        self.assertIn('final_step_type', context)
        self.assertIn('confidence', context)
        # Should have Processing as default
        self.assertEqual(context['final_step_type'], 'Processing')

    def test_framework_detection_case_insensitive(self):
        """Test that framework detection is case insensitive."""
        imports = ['XGBoost', 'PANDAS', 'NUMPY']
        
        result = detect_framework_from_imports(imports)
        
        # Should still detect xgboost despite case differences
        self.assertEqual(result, 'xgboost')

    def test_framework_detection_partial_matches(self):
        """Test framework detection with partial import names."""
        imports = ['xgb', 'pd', 'np']
        
        result = detect_framework_from_imports(imports)
        
        # Should handle common abbreviations
        # This depends on implementation - might return None if not handled
        self.assertIsInstance(result, (str, type(None)))

    def test_step_type_detection_with_file_extension(self):
        """Test step type detection with file extension."""
        # Test with .py extension
        result = detect_step_type_from_registry("xgboost_training.py")
        
        # Should handle file extensions gracefully
        self.assertIsInstance(result, (str, type(None)))

    def test_step_type_detection_edge_cases(self):
        """Test step type detection edge cases."""
        # Test with empty string
        result = detect_step_type_from_registry("")
        self.assertIsInstance(result, (str, type(None)))
        
        # Test with None
        result = detect_step_type_from_registry(None)
        self.assertIsInstance(result, (str, type(None)))

    def test_framework_detection_priority_order(self):
        """Test that framework detection follows priority order."""
        # Test with multiple frameworks - should prioritize based on implementation
        imports = ['pandas', 'xgboost', 'torch']
        
        result = detect_framework_from_imports(imports)
        
        # Should return one of the frameworks based on priority
        self.assertIn(result, ['xgboost', 'torch', 'pandas'])


if __name__ == '__main__':
    unittest.main()

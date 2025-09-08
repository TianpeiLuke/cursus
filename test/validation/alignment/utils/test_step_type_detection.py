"""
Test suite for step type detection functionality.
"""

import unittest
from unittest.mock import Mock, patch

from cursus.validation.alignment.alignment_utils import (
    detect_step_type_from_registry,
    detect_framework_from_imports,
    detect_step_type_from_script_patterns,
    get_step_type_context,
    ImportStatement
)

class TestStepTypeDetection(unittest.TestCase):
    """Test step type detection functions."""
    
    def test_detect_step_type_from_registry_training(self):
        """Test step type detection for training scripts."""
        # Mock the entire import since it's done inside the function
        mock_module = Mock()
        mock_module.get_canonical_name_from_file_name.return_value = "xgboost_training"
        mock_module.get_sagemaker_step_type.return_value = "Training"
        
        with patch.dict('sys.modules', {'cursus.steps.registry.step_names': mock_module}):
            step_type = detect_step_type_from_registry("xgboost_training")
            self.assertEqual(step_type, "Training")
    
    def test_detect_step_type_from_registry_processing(self):
        """Test step type detection for processing scripts."""
        mock_module = Mock()
        mock_module.get_canonical_name_from_file_name.return_value = "tabular_preprocessing"
        mock_module.get_sagemaker_step_type.return_value = "Processing"
        
        with patch.dict('sys.modules', {'cursus.steps.registry.step_names': mock_module}):
            step_type = detect_step_type_from_registry("tabular_preprocessing")
            self.assertEqual(step_type, "Processing")
    
    def test_detect_step_type_from_registry_unknown(self):
        """Test step type detection for unknown scripts."""
        # Test fallback to Processing when registry fails
        step_type = detect_step_type_from_registry("unknown_script")
        self.assertEqual(step_type, "Processing")  # Default fallback
    
    def test_detect_framework_from_imports_xgboost(self):
        """Test framework detection for XGBoost imports."""
        imports = [
            ImportStatement(module_name="xgboost", import_alias="xgb", line_number=1),
            ImportStatement(module_name="pandas", import_alias="pd", line_number=2)
        ]
        
        framework = detect_framework_from_imports(imports)
        self.assertEqual(framework, "xgboost")
    
    def test_detect_framework_from_imports_pytorch(self):
        """Test framework detection for PyTorch imports."""
        imports = [
            ImportStatement(module_name="torch", import_alias=None, line_number=1),
            ImportStatement(module_name="torch.nn", import_alias="nn", line_number=2),
            ImportStatement(module_name="torch.optim", import_alias=None, line_number=3)
        ]
        
        framework = detect_framework_from_imports(imports)
        self.assertEqual(framework, "pytorch")
    
    def test_detect_framework_from_imports_sklearn(self):
        """Test framework detection for sklearn imports."""
        imports = [
            ImportStatement(module_name="sklearn.ensemble", import_alias=None, line_number=1),
            ImportStatement(module_name="sklearn.model_selection", import_alias=None, line_number=2)
        ]
        
        framework = detect_framework_from_imports(imports)
        self.assertEqual(framework, "sklearn")
    
    def test_detect_framework_from_imports_pandas(self):
        """Test framework detection for pandas imports."""
        imports = [
            ImportStatement(module_name="pandas", import_alias="pd", line_number=1),
            ImportStatement(module_name="numpy", import_alias="np", line_number=2)
        ]
        
        framework = detect_framework_from_imports(imports)
        self.assertEqual(framework, "pandas")
    
    def test_detect_framework_from_imports_string_list(self):
        """Test framework detection with string list input."""
        imports = ["xgboost", "pandas", "numpy"]
        
        framework = detect_framework_from_imports(imports)
        self.assertEqual(framework, "xgboost")
    
    def test_detect_framework_from_imports_mixed_priority(self):
        """Test framework detection with mixed imports - priority order."""
        imports = [
            ImportStatement(module_name="pandas", import_alias="pd", line_number=1),
            ImportStatement(module_name="xgboost", import_alias="xgb", line_number=2),
            ImportStatement(module_name="torch", import_alias=None, line_number=3)
        ]
        
        # XGBoost should have higher priority than pandas
        framework = detect_framework_from_imports(imports)
        self.assertEqual(framework, "xgboost")
    
    def test_detect_framework_from_imports_no_ml_framework(self):
        """Test framework detection with no ML framework imports."""
        imports = [
            ImportStatement(module_name="os", import_alias=None, line_number=1),
            ImportStatement(module_name="json", import_alias=None, line_number=2),
            ImportStatement(module_name="boto3", import_alias=None, line_number=3)
        ]
        
        framework = detect_framework_from_imports(imports)
        self.assertIsNone(framework)
    
    def test_detect_step_type_from_script_patterns_training(self):
        """Test step type detection from script patterns - training."""
        script_content = """
import xgboost as xgb
import pandas as pd

def main():
    # Load training data
    train_data = pd.read_csv('/opt/ml/input/data/train/train.csv')
    
    # Create DMatrix
    dtrain = xgb.DMatrix(train_data)
    
    # Train model
    model = xgb.train({}, dtrain)
    
    # Save model
    model.save_model('/opt/ml/model/model.xgb')
        """
        
        step_type = detect_step_type_from_script_patterns(script_content)
        self.assertEqual(step_type, "Training")
    
    def test_detect_step_type_from_script_patterns_processing(self):
        """Test step type detection from script patterns - processing."""
        script_content = """
import pandas as pd
import os

def main():
    # Load data
    input_path = os.environ.get('SM_CHANNEL_INPUT')
    data = pd.read_csv(f'{input_path}/data.csv')
    
    # Process data
    processed_data = data.dropna()
    
    # Save processed data
    output_path = os.environ.get('SM_CHANNEL_OUTPUT')
    processed_data.to_csv(f'{output_path}/processed.csv', index=False)
        """
        
        step_type = detect_step_type_from_script_patterns(script_content)
        self.assertEqual(step_type, "Processing")
    
    def test_detect_step_type_from_script_patterns_unknown(self):
        """Test step type detection from script patterns - unknown."""
        script_content = """
import json
import boto3

def main():
    # Some generic code
    client = boto3.client('s3')
    data = {'key': 'value'}
    print(json.dumps(data))
        """
        
        step_type = detect_step_type_from_script_patterns(script_content)
        self.assertIsNone(step_type)
    
    def test_get_step_type_context_training(self):
        """Test getting step type context for training."""
        context = get_step_type_context("training_script")
        
        self.assertIn("script_name", context)
        self.assertIn("registry_step_type", context)
        self.assertIn("pattern_step_type", context)
        self.assertIn("final_step_type", context)
        self.assertIn("confidence", context)
        
        # Check that context is returned
        self.assertEqual(context["script_name"], "training_script")
        self.assertIsNotNone(context["final_step_type"])
    
    def test_get_step_type_context_processing(self):
        """Test getting step type context for processing."""
        context = get_step_type_context("processing_script")
        
        self.assertIn("script_name", context)
        self.assertIn("final_step_type", context)
        self.assertEqual(context["script_name"], "processing_script")
        self.assertIsNotNone(context["final_step_type"])
    
    def test_get_step_type_context_createmodel(self):
        """Test getting step type context for CreateModel."""
        context = get_step_type_context("createmodel_script")
        
        self.assertIn("script_name", context)
        self.assertIn("final_step_type", context)
        self.assertEqual(context["script_name"], "createmodel_script")
    
    def test_get_step_type_context_transform(self):
        """Test getting step type context for Transform."""
        context = get_step_type_context("transform_script")
        
        self.assertIn("script_name", context)
        self.assertIn("final_step_type", context)
        self.assertEqual(context["script_name"], "transform_script")
    
    def test_get_step_type_context_registermodel(self):
        """Test getting step type context for RegisterModel."""
        context = get_step_type_context("registermodel_script")
        
        self.assertIn("script_name", context)
        self.assertIn("final_step_type", context)
        self.assertEqual(context["script_name"], "registermodel_script")
    
    def test_get_step_type_context_utility(self):
        """Test getting step type context for Utility."""
        context = get_step_type_context("utility_script")
        
        self.assertIn("script_name", context)
        self.assertIn("final_step_type", context)
        self.assertEqual(context["script_name"], "utility_script")
    
    def test_get_step_type_context_unknown(self):
        """Test getting step type context for unknown step type."""
        context = get_step_type_context("UnknownStepType")
        
        # Should return context with default values
        self.assertIn("script_name", context)
        self.assertIn("final_step_type", context)
        self.assertEqual(context["script_name"], "UnknownStepType")
        self.assertEqual(context["final_step_type"], "Processing")  # Default fallback

class TestImportStatementHandling(unittest.TestCase):
    """Test handling of different import statement formats."""
    
    def test_detect_framework_with_import_objects(self):
        """Test framework detection with ImportStatement objects."""
        imports = [
            ImportStatement(module_name="xgboost", import_alias="xgb", line_number=1),
            ImportStatement(module_name="pandas", import_alias="pd", line_number=2)
        ]
        
        framework = detect_framework_from_imports(imports)
        self.assertEqual(framework, "xgboost")
    
    def test_detect_framework_with_string_list(self):
        """Test framework detection with string list."""
        imports = ["torch", "torch.nn", "torch.optim"]
        
        framework = detect_framework_from_imports(imports)
        self.assertEqual(framework, "pytorch")
    
    def test_detect_framework_with_mixed_types(self):
        """Test framework detection with mixed import types."""
        imports = [
            "pandas",
            ImportStatement(module_name="xgboost", import_alias="xgb", line_number=2),
            "numpy"
        ]
        
        framework = detect_framework_from_imports(imports)
        self.assertEqual(framework, "xgboost")
    
    def test_detect_framework_empty_imports(self):
        """Test framework detection with empty imports."""
        framework = detect_framework_from_imports([])
        self.assertIsNone(framework)
    
    def test_detect_framework_none_imports(self):
        """Test framework detection with None imports."""
        # This should handle None gracefully
        with self.assertRaises(TypeError):
            detect_framework_from_imports(None)

if __name__ == '__main__':
    unittest.main()

"""
Unit tests for framework_patterns.py module.

Tests framework detection and pattern recognition functionality.
"""

import unittest
from unittest.mock import patch, mock_open
from src.cursus.validation.alignment.framework_patterns import (
    detect_framework_from_script_content,
    detect_training_patterns,
    detect_xgboost_patterns,
    detect_pytorch_patterns,
    detect_sklearn_patterns,
    detect_pandas_patterns,
    get_framework_patterns,
    detect_framework_from_imports
)


class TestFrameworkPatterns(unittest.TestCase):
    """Test framework pattern detection functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.xgboost_script_content = """
import xgboost as xgb
import pandas as pd
import os

def main():
    # Load training data
    train_data = pd.read_csv('/opt/ml/input/data/training/train.csv')
    
    # Create DMatrix
    dtrain = xgb.DMatrix(train_data.drop('target', axis=1), label=train_data['target'])
    
    # Load hyperparameters
    with open('/opt/ml/input/data/config/hyperparameters.json', 'r') as f:
        hyperparams = json.load(f)
    
    # Train model
    model = xgb.train(hyperparams, dtrain)
    
    # Save model
    model.save_model('/opt/ml/model/model.xgb')

if __name__ == "__main__":
    main()
"""

        self.pytorch_script_content = """
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    model = Net()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(10):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    # Save model
    torch.save(model.state_dict(), '/opt/ml/model/model.pth')

if __name__ == "__main__":
    main()
"""

        self.processing_script_content = """
import pandas as pd
import numpy as np
import os

def main():
    # Load data
    input_path = os.environ.get('SM_CHANNEL_INPUT', '/opt/ml/processing/input')
    output_path = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/processing/output')
    
    # Read data
    df = pd.read_csv(f'{input_path}/data.csv')
    
    # Process data
    df_processed = df.dropna()
    df_processed = df_processed.fillna(0)
    
    # Save processed data
    df_processed.to_csv(f'{output_path}/processed_data.csv', index=False)

if __name__ == "__main__":
    main()
"""

    def test_detect_framework_from_script_content_xgboost(self):
        """Test XGBoost framework detection."""
        framework = detect_framework_from_script_content(self.xgboost_script_content)
        self.assertEqual(framework, 'xgboost')

    def test_detect_framework_from_script_content_pytorch(self):
        """Test PyTorch framework detection."""
        framework = detect_framework_from_script_content(self.pytorch_script_content)
        self.assertEqual(framework, 'pytorch')

    def test_detect_framework_from_script_content_processing(self):
        """Test processing framework detection (pandas)."""
        framework = detect_framework_from_script_content(self.processing_script_content)
        self.assertEqual(framework, 'pandas')

    def test_detect_framework_from_script_content_unknown(self):
        """Test unknown framework detection."""
        unknown_content = "print('hello world')"
        framework = detect_framework_from_script_content(unknown_content)
        self.assertIsNone(framework)

    def test_detect_training_patterns(self):
        """Test training pattern detection."""
        # Create mock script analysis
        script_analysis = {
            'functions': ['main', 'xgb.train', 'model.save_model'],
            'path_references': ['/opt/ml/model/model.xgb', '/opt/ml/input/data/config/hyperparameters.json']
        }
        
        patterns = detect_training_patterns(script_analysis)
        
        self.assertIn('has_training_loop', patterns)
        self.assertIn('has_model_saving', patterns)
        self.assertIn('has_hyperparameter_loading', patterns)
        self.assertIn('has_evaluation', patterns)
        
        # Check boolean patterns
        self.assertTrue(patterns['has_training_loop'])
        self.assertTrue(patterns['has_model_saving'])
        self.assertTrue(patterns['has_hyperparameter_loading'])

    def test_detect_xgboost_patterns(self):
        """Test XGBoost-specific pattern detection."""
        # Create mock script analysis
        script_analysis = {
            'imports': ['xgboost', 'xgb'],
            'functions': ['main', 'xgb.DMatrix', 'xgb.train', 'model.save_model']
        }
        
        patterns = detect_xgboost_patterns(script_analysis)
        
        self.assertIn('has_xgboost_import', patterns)
        self.assertIn('has_dmatrix_usage', patterns)
        self.assertIn('has_xgb_train', patterns)
        self.assertIn('has_model_loading', patterns)
        
        # Check boolean patterns
        self.assertTrue(patterns['has_xgboost_import'])
        self.assertTrue(patterns['has_dmatrix_usage'])
        self.assertTrue(patterns['has_xgb_train'])

    def test_detect_pytorch_patterns(self):
        """Test PyTorch-specific pattern detection."""
        # Create mock script analysis
        script_analysis = {
            'imports': ['torch', 'torch.nn', 'torch.optim'],
            'functions': ['Net', 'nn.Module', 'optim.Adam', 'nn.CrossEntropyLoss', 'zero_grad']
        }
        
        patterns = detect_pytorch_patterns(script_analysis)
        
        self.assertIn('has_torch_import', patterns)
        self.assertIn('has_nn_module', patterns)
        self.assertIn('has_optimizer', patterns)
        self.assertIn('has_loss_function', patterns)
        self.assertIn('has_training_loop', patterns)
        
        # Check boolean patterns
        self.assertTrue(patterns['has_torch_import'])
        self.assertTrue(patterns['has_nn_module'])
        self.assertTrue(patterns['has_optimizer'])
        self.assertTrue(patterns['has_loss_function'])
        self.assertTrue(patterns['has_training_loop'])

    def test_detect_pandas_patterns(self):
        """Test pandas pattern detection."""
        # Create mock script analysis
        script_analysis = {
            'imports': ['pandas', 'pd'],
            'functions': ['pd.read_csv', 'df.dropna', 'df.to_csv', 'DataFrame']
        }
        
        patterns = detect_pandas_patterns(script_analysis)
        
        self.assertIn('has_pandas_import', patterns)
        self.assertIn('has_dataframe_operations', patterns)
        self.assertIn('has_data_loading', patterns)
        self.assertIn('has_data_saving', patterns)
        
        # Check boolean patterns
        self.assertTrue(patterns['has_pandas_import'])
        self.assertTrue(patterns['has_dataframe_operations'])
        self.assertTrue(patterns['has_data_loading'])
        self.assertTrue(patterns['has_data_saving'])

    def test_get_framework_patterns(self):
        """Test framework-specific pattern retrieval."""
        # Create mock script analysis
        script_analysis = {
            'imports': ['xgboost'],
            'functions': ['xgb.train']
        }
        
        # Test XGBoost patterns
        xgb_patterns = get_framework_patterns('xgboost', script_analysis)
        self.assertIn('has_xgboost_import', xgb_patterns)
        self.assertTrue(xgb_patterns['has_xgboost_import'])
        
        # Test unknown framework
        unknown_patterns = get_framework_patterns('unknown', script_analysis)
        self.assertEqual(unknown_patterns, {})

    def test_empty_script_analysis(self):
        """Test pattern detection with empty script analysis."""
        empty_analysis = {'functions': [], 'path_references': []}
        patterns = detect_training_patterns(empty_analysis)
        
        self.assertFalse(patterns['has_training_loop'])
        self.assertFalse(patterns['has_model_saving'])
        self.assertFalse(patterns['has_hyperparameter_loading'])
        self.assertFalse(patterns['has_evaluation'])

    def test_multiple_framework_detection(self):
        """Test script with multiple frameworks."""
        mixed_content = """
import xgboost as xgb
import torch
import pandas as pd

# This script uses multiple frameworks
"""
        framework = detect_framework_from_script_content(mixed_content)
        # Should detect the first/primary framework
        self.assertIn(framework, ['xgboost', 'pytorch', 'pandas'])

    def test_detect_framework_from_imports(self):
        """Test framework detection from imports."""
        # Test XGBoost
        xgb_imports = ['xgboost', 'pandas']
        framework = detect_framework_from_imports(xgb_imports)
        self.assertEqual(framework, 'xgboost')
        
        # Test PyTorch
        pytorch_imports = ['torch', 'torch.nn']
        framework = detect_framework_from_imports(pytorch_imports)
        self.assertEqual(framework, 'pytorch')
        
        # Test sklearn
        sklearn_imports = ['sklearn', 'sklearn.ensemble']
        framework = detect_framework_from_imports(sklearn_imports)
        self.assertEqual(framework, 'sklearn')
        
        # Test pandas
        pandas_imports = ['pandas', 'numpy']
        framework = detect_framework_from_imports(pandas_imports)
        self.assertEqual(framework, 'pandas')
        
        # Test unknown
        unknown_imports = ['os', 'sys']
        framework = detect_framework_from_imports(unknown_imports)
        self.assertIsNone(framework)


if __name__ == '__main__':
    unittest.main()

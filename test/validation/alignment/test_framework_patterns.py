"""
Unit tests for framework_patterns.py

Tests framework-specific pattern detection functionality including:
- Training pattern detection
- XGBoost pattern detection  
- PyTorch pattern detection
- Scikit-learn pattern detection
- Pandas pattern detection
- Framework detection from script content and imports
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any, List

from cursus.validation.alignment.framework_patterns import (
    detect_training_patterns,
    detect_xgboost_patterns,
    detect_pytorch_patterns,
    detect_sklearn_patterns,
    detect_pandas_patterns,
    get_framework_patterns,
    get_all_framework_patterns,
    detect_framework_from_script_content,
    detect_framework_from_imports
)


class TestDetectTrainingPatterns:
    """Test training pattern detection functionality."""
    
    def test_detect_training_patterns_empty_analysis(self):
        """Test training pattern detection with empty analysis."""
        script_analysis = {}
        result = detect_training_patterns(script_analysis)
        
        expected = {
            'has_training_loop': False,
            'has_model_saving': False,
            'has_hyperparameter_loading': False,
            'has_data_loading': False,
            'has_evaluation': False
        }
        assert result == expected
    
    def test_detect_training_patterns_with_training_loop(self):
        """Test detection of training loop patterns."""
        script_analysis = {
            'functions': ['def train_model()', 'def fit_data()', 'for epoch in range(10)'],
            'path_references': []
        }
        result = detect_training_patterns(script_analysis)
        
        assert result['has_training_loop'] is True
        assert result['has_model_saving'] is False
    
    def test_detect_training_patterns_with_model_saving(self):
        """Test detection of model saving patterns."""
        script_analysis = {
            'functions': ['model.save()', 'joblib.dump(model)'],
            'path_references': ['/opt/ml/model/model.pkl']
        }
        result = detect_training_patterns(script_analysis)
        
        assert result['has_model_saving'] is True
        assert result['has_training_loop'] is False
    
    def test_detect_training_patterns_with_hyperparameter_loading(self):
        """Test detection of hyperparameter loading patterns."""
        script_analysis = {
            'functions': ['load_hyperparameters()', 'config.get()'],
            'path_references': ['/opt/ml/input/data/config/hyperparameters.json']
        }
        result = detect_training_patterns(script_analysis)
        
        assert result['has_hyperparameter_loading'] is True
    
    def test_detect_training_patterns_with_data_loading(self):
        """Test detection of data loading patterns."""
        script_analysis = {
            'functions': ['pd.read_csv()', 'load_data()'],
            'path_references': ['/opt/ml/input/data/train/train.csv']
        }
        result = detect_training_patterns(script_analysis)
        
        assert result['has_data_loading'] is True
    
    def test_detect_training_patterns_with_evaluation(self):
        """Test detection of evaluation patterns."""
        script_analysis = {
            'functions': ['evaluate_model()', 'calculate_accuracy()', 'validation_loss()'],
            'path_references': []
        }
        result = detect_training_patterns(script_analysis)
        
        assert result['has_evaluation'] is True
    
    def test_detect_training_patterns_comprehensive(self):
        """Test comprehensive training pattern detection."""
        script_analysis = {
            'functions': [
                'def train_model()', 'model.save()', 'load_hyperparameters()',
                'pd.read_csv()', 'evaluate_model()'
            ],
            'path_references': [
                '/opt/ml/model/model.pkl',
                '/opt/ml/input/data/config/hyperparameters.json',
                '/opt/ml/input/data/train/train.csv'
            ]
        }
        result = detect_training_patterns(script_analysis)
        
        # All patterns should be detected
        for pattern_key in result:
            assert result[pattern_key] is True


class TestDetectXGBoostPatterns:
    """Test XGBoost pattern detection functionality."""
    
    def test_detect_xgboost_patterns_empty_analysis(self):
        """Test XGBoost pattern detection with empty analysis."""
        script_analysis = {}
        result = detect_xgboost_patterns(script_analysis)
        
        expected = {
            'has_xgboost_import': False,
            'has_dmatrix_usage': False,
            'has_xgb_train': False,
            'has_booster_usage': False,
            'has_model_loading': False
        }
        assert result == expected
    
    def test_detect_xgboost_patterns_with_imports(self):
        """Test detection of XGBoost imports."""
        script_analysis = {
            'imports': ['import xgboost as xgb', 'from xgboost import DMatrix'],
            'functions': []
        }
        result = detect_xgboost_patterns(script_analysis)
        
        assert result['has_xgboost_import'] is True
    
    def test_detect_xgboost_patterns_with_dmatrix(self):
        """Test detection of DMatrix usage."""
        script_analysis = {
            'imports': [],
            'functions': ['dtrain = xgb.DMatrix(X_train, y_train)', 'DMatrix(data)']
        }
        result = detect_xgboost_patterns(script_analysis)
        
        assert result['has_dmatrix_usage'] is True
    
    def test_detect_xgboost_patterns_with_training(self):
        """Test detection of XGBoost training."""
        script_analysis = {
            'imports': [],
            'functions': ['model = xgb.train(params, dtrain)', 'train(params)']
        }
        result = detect_xgboost_patterns(script_analysis)
        
        assert result['has_xgb_train'] is True
    
    def test_detect_xgboost_patterns_with_booster(self):
        """Test detection of Booster usage."""
        script_analysis = {
            'imports': [],
            'functions': ['booster = xgb.Booster()', 'Booster.load_model()']
        }
        result = detect_xgboost_patterns(script_analysis)
        
        assert result['has_booster_usage'] is True
    
    def test_detect_xgboost_patterns_with_model_loading(self):
        """Test detection of model loading."""
        script_analysis = {
            'imports': [],
            'functions': ['model.load_model()', 'pickle.load(f)', 'joblib.load()']
        }
        result = detect_xgboost_patterns(script_analysis)
        
        assert result['has_model_loading'] is True
    
    def test_detect_xgboost_patterns_comprehensive(self):
        """Test comprehensive XGBoost pattern detection."""
        script_analysis = {
            'imports': ['import xgboost as xgb'],
            'functions': [
                'dtrain = xgb.DMatrix(X_train, y_train)',
                'model = xgb.train(params, dtrain)',
                'booster = xgb.Booster()',
                'model.load_model(filename)'
            ]
        }
        result = detect_xgboost_patterns(script_analysis)
        
        # All patterns should be detected
        for pattern_key in result:
            assert result[pattern_key] is True


class TestDetectPyTorchPatterns:
    """Test PyTorch pattern detection functionality."""
    
    def test_detect_pytorch_patterns_empty_analysis(self):
        """Test PyTorch pattern detection with empty analysis."""
        script_analysis = {}
        result = detect_pytorch_patterns(script_analysis)
        
        expected = {
            'has_torch_import': False,
            'has_nn_module': False,
            'has_optimizer': False,
            'has_loss_function': False,
            'has_model_loading': False,
            'has_training_loop': False
        }
        assert result == expected
    
    def test_detect_pytorch_patterns_with_imports(self):
        """Test detection of PyTorch imports."""
        script_analysis = {
            'imports': ['import torch', 'import pytorch_lightning'],
            'functions': []
        }
        result = detect_pytorch_patterns(script_analysis)
        
        assert result['has_torch_import'] is True
    
    def test_detect_pytorch_patterns_with_nn_module(self):
        """Test detection of nn.Module usage."""
        script_analysis = {
            'imports': [],
            'functions': ['class Net(nn.Module):', 'torch.nn.Linear()']
        }
        result = detect_pytorch_patterns(script_analysis)
        
        assert result['has_nn_module'] is True
    
    def test_detect_pytorch_patterns_with_optimizer(self):
        """Test detection of optimizer usage."""
        script_analysis = {
            'imports': [],
            'functions': ['optimizer = torch.optim.Adam()', 'SGD(params)']
        }
        result = detect_pytorch_patterns(script_analysis)
        
        assert result['has_optimizer'] is True
    
    def test_detect_pytorch_patterns_with_loss_function(self):
        """Test detection of loss function usage."""
        script_analysis = {
            'imports': [],
            'functions': ['criterion = nn.CrossEntropyLoss()', 'MSELoss()']
        }
        result = detect_pytorch_patterns(script_analysis)
        
        assert result['has_loss_function'] is True
    
    def test_detect_pytorch_patterns_with_model_loading(self):
        """Test detection of model loading."""
        script_analysis = {
            'imports': [],
            'functions': ['torch.load(path)', 'model.load_state_dict()']
        }
        result = detect_pytorch_patterns(script_analysis)
        
        assert result['has_model_loading'] is True
    
    def test_detect_pytorch_patterns_with_training_loop(self):
        """Test detection of training loop patterns."""
        script_analysis = {
            'imports': [],
            'functions': ['output = model.forward()', 'loss.backward()', 'optimizer.zero_grad()']
        }
        result = detect_pytorch_patterns(script_analysis)
        
        assert result['has_training_loop'] is True


class TestDetectSklearnPatterns:
    """Test Scikit-learn pattern detection functionality."""
    
    def test_detect_sklearn_patterns_empty_analysis(self):
        """Test sklearn pattern detection with empty analysis."""
        script_analysis = {}
        result = detect_sklearn_patterns(script_analysis)
        
        expected = {
            'has_sklearn_import': False,
            'has_preprocessing': False,
            'has_model_training': False,
            'has_model_evaluation': False,
            'has_pipeline': False
        }
        assert result == expected
    
    def test_detect_sklearn_patterns_with_imports(self):
        """Test detection of sklearn imports."""
        script_analysis = {
            'imports': ['import sklearn', 'from sklearn.ensemble import RandomForestClassifier'],
            'functions': []
        }
        result = detect_sklearn_patterns(script_analysis)
        
        assert result['has_sklearn_import'] is True
    
    def test_detect_sklearn_patterns_with_preprocessing(self):
        """Test detection of preprocessing usage."""
        script_analysis = {
            'imports': [],
            'functions': ['StandardScaler().fit_transform()', 'LabelEncoder()']
        }
        result = detect_sklearn_patterns(script_analysis)
        
        assert result['has_preprocessing'] is True
    
    def test_detect_sklearn_patterns_with_model_training(self):
        """Test detection of model training."""
        script_analysis = {
            'imports': [],
            'functions': ['model.fit(X, y)', 'RandomForestClassifier()']
        }
        result = detect_sklearn_patterns(script_analysis)
        
        assert result['has_model_training'] is True
    
    def test_detect_sklearn_patterns_with_evaluation(self):
        """Test detection of model evaluation."""
        script_analysis = {
            'imports': [],
            'functions': ['model.score(X_test, y_test)', 'accuracy_score()']
        }
        result = detect_sklearn_patterns(script_analysis)
        
        assert result['has_model_evaluation'] is True
    
    def test_detect_sklearn_patterns_with_pipeline(self):
        """Test detection of pipeline usage."""
        script_analysis = {
            'imports': [],
            'functions': ['Pipeline([steps])', 'make_pipeline()']
        }
        result = detect_sklearn_patterns(script_analysis)
        
        assert result['has_pipeline'] is True


class TestDetectPandasPatterns:
    """Test Pandas pattern detection functionality."""
    
    def test_detect_pandas_patterns_empty_analysis(self):
        """Test pandas pattern detection with empty analysis."""
        script_analysis = {}
        result = detect_pandas_patterns(script_analysis)
        
        expected = {
            'has_pandas_import': False,
            'has_dataframe_operations': False,
            'has_data_loading': False,
            'has_data_saving': False,
            'has_data_transformation': False
        }
        assert result == expected
    
    def test_detect_pandas_patterns_with_imports(self):
        """Test detection of pandas imports."""
        script_analysis = {
            'imports': ['import pandas as pd', 'from pandas import DataFrame'],
            'functions': []
        }
        result = detect_pandas_patterns(script_analysis)
        
        assert result['has_pandas_import'] is True
    
    def test_detect_pandas_patterns_with_dataframe_operations(self):
        """Test detection of DataFrame operations."""
        script_analysis = {
            'imports': [],
            'functions': ['df = pd.DataFrame()', 'df.head()']
        }
        result = detect_pandas_patterns(script_analysis)
        
        assert result['has_dataframe_operations'] is True
    
    def test_detect_pandas_patterns_with_data_loading(self):
        """Test detection of data loading."""
        script_analysis = {
            'imports': [],
            'functions': ['pd.read_csv()', 'pd.read_json()']
        }
        result = detect_pandas_patterns(script_analysis)
        
        assert result['has_data_loading'] is True
    
    def test_detect_pandas_patterns_with_data_saving(self):
        """Test detection of data saving."""
        script_analysis = {
            'imports': [],
            'functions': ['df.to_csv()', 'df.to_excel()']
        }
        result = detect_pandas_patterns(script_analysis)
        
        assert result['has_data_saving'] is True
    
    def test_detect_pandas_patterns_with_data_transformation(self):
        """Test detection of data transformation."""
        script_analysis = {
            'imports': [],
            'functions': ['df.groupby()', 'df.merge()', 'df.apply()']
        }
        result = detect_pandas_patterns(script_analysis)
        
        assert result['has_data_transformation'] is True


class TestGetFrameworkPatterns:
    """Test framework pattern retrieval functionality."""
    
    def test_get_framework_patterns_xgboost(self):
        """Test getting XGBoost patterns."""
        script_analysis = {
            'imports': ['import xgboost'],
            'functions': ['xgb.train()']
        }
        result = get_framework_patterns('xgboost', script_analysis)
        
        assert 'has_xgboost_import' in result
        assert result['has_xgboost_import'] is True
    
    def test_get_framework_patterns_pytorch(self):
        """Test getting PyTorch patterns."""
        script_analysis = {
            'imports': ['import torch'],
            'functions': ['nn.Module']
        }
        result = get_framework_patterns('pytorch', script_analysis)
        
        assert 'has_torch_import' in result
        assert result['has_torch_import'] is True
    
    def test_get_framework_patterns_sklearn(self):
        """Test getting sklearn patterns."""
        script_analysis = {
            'imports': ['import sklearn'],
            'functions': ['fit()']
        }
        result = get_framework_patterns('sklearn', script_analysis)
        
        assert 'has_sklearn_import' in result
        assert result['has_sklearn_import'] is True
    
    def test_get_framework_patterns_pandas(self):
        """Test getting pandas patterns."""
        script_analysis = {
            'imports': ['import pandas'],
            'functions': ['DataFrame()']
        }
        result = get_framework_patterns('pandas', script_analysis)
        
        assert 'has_pandas_import' in result
        assert result['has_pandas_import'] is True
    
    def test_get_framework_patterns_training(self):
        """Test getting training patterns."""
        script_analysis = {
            'functions': ['train()'],
            'path_references': []
        }
        result = get_framework_patterns('training', script_analysis)
        
        assert 'has_training_loop' in result
        assert result['has_training_loop'] is True
    
    def test_get_framework_patterns_case_insensitive(self):
        """Test case insensitive framework name matching."""
        script_analysis = {
            'imports': ['import xgboost'],
            'functions': []
        }
        result = get_framework_patterns('XGBOOST', script_analysis)
        
        assert 'has_xgboost_import' in result
        assert result['has_xgboost_import'] is True
    
    def test_get_framework_patterns_unknown_framework(self):
        """Test getting patterns for unknown framework."""
        script_analysis = {'imports': [], 'functions': []}
        result = get_framework_patterns('unknown', script_analysis)
        
        assert result == {}


class TestGetAllFrameworkPatterns:
    """Test comprehensive framework pattern detection."""
    
    def test_get_all_framework_patterns_empty_analysis(self):
        """Test getting all patterns with empty analysis."""
        script_analysis = {}
        result = get_all_framework_patterns(script_analysis)
        
        expected_frameworks = ['xgboost', 'pytorch', 'sklearn', 'pandas', 'training']
        assert set(result.keys()) == set(expected_frameworks)
        
        # All patterns should be False for empty analysis
        for framework_patterns in result.values():
            for pattern_value in framework_patterns.values():
                assert pattern_value is False
    
    def test_get_all_framework_patterns_comprehensive(self):
        """Test getting all patterns with comprehensive analysis."""
        script_analysis = {
            'imports': [
                'import xgboost as xgb',
                'import torch',
                'import sklearn',
                'import pandas as pd'
            ],
            'functions': [
                'xgb.train()', 'nn.Module', 'fit()', 'pd.DataFrame()', 'train_model()'
            ],
            'path_references': ['/opt/ml/model/']
        }
        result = get_all_framework_patterns(script_analysis)
        
        # Check that all frameworks are detected
        assert result['xgboost']['has_xgboost_import'] is True
        assert result['pytorch']['has_torch_import'] is True
        assert result['sklearn']['has_sklearn_import'] is True
        assert result['pandas']['has_pandas_import'] is True
        assert result['training']['has_training_loop'] is True


class TestDetectFrameworkFromScriptContent:
    """Test framework detection from script content."""
    
    def test_detect_framework_from_script_content_empty(self):
        """Test framework detection with empty content."""
        result = detect_framework_from_script_content("")
        assert result is None
        
        result = detect_framework_from_script_content(None)
        assert result is None
    
    def test_detect_framework_from_script_content_xgboost(self):
        """Test XGBoost framework detection."""
        script_content = """
        import xgboost as xgb
        dtrain = xgb.DMatrix(X_train, y_train)
        model = xgb.train(params, dtrain)
        """
        result = detect_framework_from_script_content(script_content)
        assert result == 'xgboost'
    
    def test_detect_framework_from_script_content_pytorch(self):
        """Test PyTorch framework detection."""
        script_content = """
        import torch
        import torch.nn as nn
        class Net(nn.Module):
            def forward(self, x):
                return x
        """
        result = detect_framework_from_script_content(script_content)
        assert result == 'pytorch'
    
    def test_detect_framework_from_script_content_sklearn(self):
        """Test sklearn framework detection."""
        script_content = """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        model = RandomForestClassifier()
        """
        result = detect_framework_from_script_content(script_content)
        assert result == 'sklearn'
    
    def test_detect_framework_from_script_content_pandas(self):
        """Test pandas framework detection."""
        script_content = """
        import pandas as pd
        df = pd.read_csv('data.csv')
        df.to_csv('output.csv')
        """
        result = detect_framework_from_script_content(script_content)
        assert result == 'pandas'
    
    def test_detect_framework_from_script_content_priority(self):
        """Test framework detection priority when multiple frameworks present."""
        script_content = """
        import xgboost as xgb
        import pandas as pd
        df = pd.read_csv('data.csv')
        model = xgb.train(params, dtrain)
        """
        result = detect_framework_from_script_content(script_content)
        # XGBoost should have priority over pandas
        assert result == 'xgboost'
    
    def test_detect_framework_from_script_content_scoring(self):
        """Test framework detection scoring system."""
        # Script with more pandas usage than xgboost
        script_content = """
        import pandas as pd
        import xgboost
        df = pd.read_csv('data.csv')
        df = df.groupby('col').sum()
        df.to_csv('output.csv')
        pd.DataFrame()
        """
        result = detect_framework_from_script_content(script_content)
        # Pandas should win due to more usage patterns
        assert result == 'pandas'


class TestDetectFrameworkFromImports:
    """Test framework detection from import statements."""
    
    def test_detect_framework_from_imports_empty(self):
        """Test framework detection with empty imports."""
        result = detect_framework_from_imports([])
        assert result is None
        
        result = detect_framework_from_imports(None)
        assert result is None
    
    def test_detect_framework_from_imports_xgboost(self):
        """Test XGBoost detection from imports."""
        imports = ['import xgboost as xgb', 'from xgboost import DMatrix']
        result = detect_framework_from_imports(imports)
        assert result == 'xgboost'
    
    def test_detect_framework_from_imports_pytorch(self):
        """Test PyTorch detection from imports."""
        imports = ['import torch', 'from torch import nn']
        result = detect_framework_from_imports(imports)
        assert result == 'pytorch'
    
    def test_detect_framework_from_imports_sklearn(self):
        """Test sklearn detection from imports."""
        imports = ['from sklearn.ensemble import RandomForestClassifier']
        result = detect_framework_from_imports(imports)
        assert result == 'sklearn'
    
    def test_detect_framework_from_imports_pandas(self):
        """Test pandas detection from imports."""
        imports = ['import pandas as pd']
        result = detect_framework_from_imports(imports)
        assert result == 'pandas'
    
    def test_detect_framework_from_imports_priority(self):
        """Test framework detection priority from imports."""
        imports = [
            'import xgboost as xgb',
            'import torch',
            'import sklearn',
            'import pandas as pd'
        ]
        result = detect_framework_from_imports(imports)
        # XGBoost should have highest priority
        assert result == 'xgboost'
    
    def test_detect_framework_from_imports_case_insensitive(self):
        """Test case insensitive import detection."""
        imports = ['import XGBoost', 'from TORCH import nn']
        result = detect_framework_from_imports(imports)
        assert result == 'xgboost'
    
    def test_detect_framework_from_imports_no_match(self):
        """Test no framework detection when no matches."""
        imports = ['import os', 'import sys', 'from collections import defaultdict']
        result = detect_framework_from_imports(imports)
        assert result is None


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_pattern_detection_with_none_values(self):
        """Test pattern detection with None values in analysis."""
        script_analysis = {
            'functions': [None, 'valid_function()'],
            'imports': [None, 'import pandas'],
            'path_references': [None, '/valid/path']
        }
        
        # Should not raise exceptions
        result = detect_training_patterns(script_analysis)
        assert isinstance(result, dict)
        
        result = detect_pandas_patterns(script_analysis)
        assert result['has_pandas_import'] is True
    
    def test_pattern_detection_with_mixed_types(self):
        """Test pattern detection with mixed data types."""
        script_analysis = {
            'functions': ['string_function', 123, {'dict': 'function'}],
            'imports': ['import torch', 456, ['list', 'import']],
            'path_references': ['/path/string', 789, {'path': 'dict'}]
        }
        
        # Should handle mixed types gracefully
        result = detect_pytorch_patterns(script_analysis)
        assert result['has_torch_import'] is True
    
    def test_large_analysis_performance(self):
        """Test performance with large analysis data."""
        # Create large analysis with many entries
        large_functions = [f'function_{i}()' for i in range(1000)]
        large_functions.extend(['xgb.train()', 'torch.nn.Module'])
        
        large_imports = [f'import module_{i}' for i in range(1000)]
        large_imports.extend(['import xgboost', 'import torch'])
        
        script_analysis = {
            'functions': large_functions,
            'imports': large_imports,
            'path_references': []
        }
        
        # Should complete in reasonable time
        result = get_all_framework_patterns(script_analysis)
        assert result['xgboost']['has_xgboost_import'] is True
        assert result['pytorch']['has_torch_import'] is True
    
    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters."""
        script_analysis = {
            'functions': ['función_española()', 'import_模块()', 'xgb.train()'],
            'imports': ['import pandas_数据', 'import torch'],
            'path_references': ['/路径/with/unicode', '/opt/ml/model/']
        }
        
        result = get_all_framework_patterns(script_analysis)
        assert result['pytorch']['has_torch_import'] is True
        assert result['training']['has_model_saving'] is True


class TestFrameworkPatterns:
    """Main test class for framework patterns - used by other test modules."""
    
    def test_framework_patterns_integration(self):
        """Test framework patterns integration."""
        script_analysis = {
            'imports': ['import xgboost as xgb', 'import torch'],
            'functions': ['xgb.train()', 'nn.Module'],
            'path_references': []
        }
        
        all_patterns = get_all_framework_patterns(script_analysis)
        assert 'xgboost' in all_patterns
        assert 'pytorch' in all_patterns
        assert all_patterns['xgboost']['has_xgboost_import'] is True
        assert all_patterns['pytorch']['has_torch_import'] is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

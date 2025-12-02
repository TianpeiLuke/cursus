"""
Comprehensive test suite for pytorch_inference_handler script.

This test suite follows pytest best practices:
1. Tests actual implementation behavior (not assumptions)
2. Provides comprehensive coverage of all code paths
3. Tests edge cases and error conditions
4. Verifies error messages match implementation
5. Uses proper fixtures for test isolation
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
import json
import pickle as pkl
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from io import StringIO, BytesIO
import argparse
from typing import Dict, Any, List, Tuple, Union

# Import the functions to be tested
try:
    from projects.rnr_pytorch_bedrock.dockers.pytorch_inference_handler import (
        Config,
        read_feature_columns,
        load_hyperparameters,
        get_text_field_names,
        load_risk_tables,
        create_risk_processors,
        load_imputation_dict,
        create_numerical_processors,
        data_preprocess_pipeline,
        load_calibration_model,
        _interpolate_score,
        apply_percentile_calibration,
        apply_regular_binary_calibration,
        apply_regular_multiclass_calibration,
        apply_calibration,
        preprocess_single_record_fast,
        model_fn,
        input_fn,
        predict_fn,
        output_fn
    )
except ImportError:
    # Fallback to direct import if the above fails
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
    from projects.rnr_pytorch_bedrock.dockers.pytorch_inference_handler import (
        Config,
        read_feature_columns,
        load_hyperparameters,
        get_text_field_names,
        load_risk_tables,
        create_risk_processors,
        load_imputation_dict,
        create_numerical_processors,
        data_preprocess_pipeline,
        load_calibration_model,
        _interpolate_score,
        apply_percentile_calibration,
        apply_regular_binary_calibration,
        apply_regular_multiclass_calibration,
        apply_calibration,
        preprocess_single_record_fast,
        model_fn,
        input_fn,
        predict_fn,
        output_fn
    )


class TestConfig:
    """Tests for Config class."""

    def test_config_valid_binary(self):
        """Test valid binary classification config."""
        config = Config(
            is_binary=True,
            num_classes=2,
            multiclass_categories=[0, 1]
        )
        assert config.is_binary is True
        assert config.num_classes == 2
        assert config.multiclass_categories == [0, 1]

    def test_config_valid_multiclass(self):
        """Test valid multiclass classification config."""
        config = Config(
            is_binary=False,
            num_classes=3,
            multiclass_categories=[0, 1, 2]
        )
        assert config.is_binary is False
        assert config.num_classes == 3
        assert config.multiclass_categories == [0, 1, 2]

    def test_config_invalid_binary_num_classes(self):
        """Test invalid binary config with wrong num_classes."""
        with pytest.raises(ValueError, match="For binary classification, num_classes must be 2"):
            Config(is_binary=True, num_classes=3)

    def test_config_invalid_multiclass_num_classes(self):
        """Test invalid multiclass config with num_classes < 2."""
        with pytest.raises(ValueError, match="For multiclass classification, num_classes must be >= 2"):
            Config(is_binary=False, num_classes=1)

    def test_config_invalid_multiclass_categories(self):
        """Test invalid multiclass config with mismatched categories."""
        with pytest.raises(ValueError, match="num_classes=3 does not match len(multiclass_categories)=2"):
            Config(is_binary=False, num_classes=3, multiclass_categories=[0, 1])

    def test_config_invalid_class_weights_length(self):
        """Test invalid config with mismatched class_weights length."""
        with pytest.raises(ValueError, match="class_weights must have the same number of elements as num_classes"):
            Config(
                is_binary=True,
                num_classes=2,
                class_weights=[1.0, 2.0, 3.0]  # 3 weights for 2 classes
            )


class TestHelperFunctions:
    """Tests for helper functions."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_read_feature_columns_success(self, temp_dir):
        """Test successful reading of feature columns."""
        # Create feature columns file
        feature_file = temp_dir / "feature_columns.txt"
        with open(feature_file, "w") as f:
            f.write("# Feature columns\n")
            f.write("0,feature1\n")
            f.write("1,feature2\n")
            f.write("2,feature3\n")
        
        result = read_feature_columns(str(temp_dir))
        assert result == ["feature1", "feature2", "feature3"]

    def test_read_feature_columns_missing_file(self, temp_dir):
        """Test reading feature columns when file is missing."""
        result = read_feature_columns(str(temp_dir))
        assert result is None

    def test_read_feature_columns_empty_file(self, temp_dir):
        """Test reading feature columns from empty file."""
        feature_file = temp_dir / "feature_columns.txt"
        feature_file.touch()
        
        with pytest.raises(ValueError, match="No valid feature columns found"):
            read_feature_columns(str(temp_dir))

    def test_load_hyperparameters_success(self, temp_dir):
        """Test successful loading of hyperparameters."""
        # Create hyperparameters file
        hyperparams = {"learning_rate": 0.01, "batch_size": 32}
        hyperparams_file = temp_dir / "hyperparameters.json"
        with open(hyperparams_file, "w") as f:
            json.dump(hyperparams, f)
        
        result = load_hyperparameters(str(temp_dir))
        assert result == hyperparams

    def test_load_hyperparameters_missing_file(self, temp_dir):
        """Test loading hyperparameters when file is missing."""
        result = load_hyperparameters(str(temp_dir))
        assert result == {}

    def test_load_hyperparameters_invalid_json(self, temp_dir):
        """Test loading hyperparameters with invalid JSON."""
        hyperparams_file = temp_dir / "hyperparameters.json"
        with open(hyperparams_file, "w") as f:
            f.write("invalid json")
        
        result = load_hyperparameters(str(temp_dir))
        assert result == {}

    def test_get_text_field_names_bimodal(self):
        """Test getting text field names for bimodal config."""
        config = Config(text_name="text_content")
        result = get_text_field_names(config)
        assert result == {"text_content"}

    def test_get_text_field_names_trimodal(self):
        """Test getting text field names for trimodal config."""
        config = Config(
            primary_text_name="primary_text",
            secondary_text_name="secondary_text"
        )
        result = get_text_field_names(config)
        assert result == {"primary_text", "secondary_text"}

    def test_get_text_field_names_no_text_fields(self):
        """Test getting text field names when no text fields are defined."""
        config = Config()
        result = get_text_field_names(config)
        assert result == set()

    def test_load_risk_tables_success(self, temp_dir):
        """Test successful loading of risk tables."""
        # Create risk tables
        risk_tables = {
            "category_feature": {
                "bins": {"A": 0.1, "B": 0.3},
                "default_bin": 0.5
            }
        }
        risk_file = temp_dir / "risk_table_map.pkl"
        with open(risk_file, "wb") as f:
            pkl.dump(risk_tables, f)
        
        result = load_risk_tables(str(temp_dir))
        assert result == risk_tables

    def test_load_risk_tables_missing_file(self, temp_dir):
        """Test loading risk tables when file is missing."""
        result = load_risk_tables(str(temp_dir))
        assert result == {}

    def test_load_risk_tables_corrupted_file(self, temp_dir):
        """Test loading risk tables with corrupted file."""
        risk_file = temp_dir / "risk_table_map.pkl"
        with open(risk_file, "w") as f:
            f.write("corrupted data")
        
        result = load_risk_tables(str(temp_dir))
        assert result == {}

    def test_create_risk_processors(self):
        """Test creating risk table processors."""
        risk_tables = {
            "category_feature": {
                "bins": {"A": 0.1, "B": 0.3},
                "default_bin": 0.5
            }
        }
        
        # Mock RiskTableMappingProcessor
        with patch('projects.rnr_pytorch_bedrock.dockers.pytorch_inference_handler.RiskTableMappingProcessor') as mock_processor_class:
            mock_processor_instance = MagicMock()
            mock_processor_class.return_value = mock_processor_instance
            
            result = create_risk_processors(risk_tables)
            
            assert "category_feature" in result
            assert result["category_feature"] == mock_processor_instance
            mock_processor_class.assert_called_once_with(
                column_name="category_feature",
                label_name="label",
                risk_tables=risk_tables["category_feature"]
            )

    def test_load_imputation_dict_success(self, temp_dir):
        """Test successful loading of imputation dictionary."""
        # Create imputation dictionary
        impute_dict = {"feature1": 10.0, "feature2": 20.0}
        impute_file = temp_dir / "impute_dict.pkl"
        with open(impute_file, "wb") as f:
            pkl.dump(impute_dict, f)
        
        result = load_imputation_dict(str(temp_dir))
        assert result == impute_dict

    def test_load_imputation_dict_missing_file(self, temp_dir):
        """Test loading imputation dictionary when file is missing."""
        result = load_imputation_dict(str(temp_dir))
        assert result == {}

    def test_create_numerical_processors(self):
        """Test creating numerical imputation processors."""
        impute_dict = {"feature1": 10.0, "feature2": 20.0}
        
        # Mock NumericalVariableImputationProcessor
        with patch('projects.rnr_pytorch_bedrock.dockers.pytorch_inference_handler.NumericalVariableImputationProcessor') as mock_processor_class:
            mock_processor_instance = MagicMock()
            mock_processor_class.return_value = mock_processor_instance
            
            result = create_numerical_processors(impute_dict)
            
            assert len(result) == 2
            assert "feature1" in result
            assert "feature2" in result
            assert result["feature1"] == mock_processor_instance
            assert result["feature2"] == mock_processor_instance
            
            # Verify calls
            mock_processor_class.assert_any_call(column_name="feature1", imputation_value=10.0)
            mock_processor_class.assert_any_call(column_name="feature2", imputation_value=20.0)

    def test_data_preprocess_pipeline_bimodal(self):
        """Test data preprocessing pipeline for bimodal configuration."""
        config = Config(text_name="text_content")
        hyperparameters = {
            "text_processing_steps": ["tokenizer"],
            "tokenizer": "bert-base-uncased"
        }
        
        # Mock AutoTokenizer and build_text_pipeline_from_steps
        with patch('projects.rnr_pytorch_bedrock.dockers.pytorch_inference_handler.AutoTokenizer') as mock_tokenizer_class, \
             patch('projects.rnr_pytorch_bedrock.dockers.pytorch_inference_handler.build_text_pipeline_from_steps') as mock_build_pipeline:
            
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance
            
            mock_pipeline_instance = MagicMock()
            mock_build_pipeline.return_value = mock_pipeline_instance
            
            tokenizer, pipelines = data_preprocess_pipeline(config, hyperparameters)
            
            assert tokenizer == mock_tokenizer_instance
            assert "text_content" in pipelines
            assert pipelines["text_content"] == mock_pipeline_instance

    def test_data_preprocess_pipeline_trimodal(self):
        """Test data preprocessing pipeline for trimodal configuration."""
        config = Config()
        hyperparameters = {
            "primary_text_name": "primary_text",
            "secondary_text_name": "secondary_text",
            "primary_text_processing_steps": ["tokenizer"],
            "secondary_text_processing_steps": ["tokenizer"],
            "tokenizer": "bert-base-uncased"
        }
        
        # Mock AutoTokenizer and build_text_pipeline_from_steps
        with patch('projects.rnr_pytorch_bedrock.dockers.pytorch_inference_handler.AutoTokenizer') as mock_tokenizer_class, \
             patch('projects.rnr_pytorch_bedrock.dockers.pytorch_inference_handler.build_text_pipeline_from_steps') as mock_build_pipeline:
            
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer_instance
            
            mock_pipeline_instance = MagicMock()
            mock_build_pipeline.return_value = mock_pipeline_instance
            
            tokenizer, pipelines = data_preprocess_pipeline(config, hyperparameters)
            
            assert tokenizer == mock_tokenizer_instance
            assert "primary_text" in pipelines
            assert "secondary_text" in pipelines
            assert pipelines["primary_text"] == mock_pipeline_instance
            assert pipelines["secondary_text"] == mock_pipeline_instance

    def test_load_calibration_model_percentile(self, temp_dir):
        """Test loading percentile calibration model."""
        # Create calibration directory and file
        calib_dir = temp_dir / "calibration"
        calib_dir.mkdir()
        percentile_file = calib_dir / "percentile_score.pkl"
        
        percentile_data = [(0.1, 0.2), (0.5, 0.6), (0.9, 0.8)]
        with open(percentile_file, "wb") as f:
            pkl.dump(percentile_data, f)
        
        result = load_calibration_model(str(temp_dir))
        assert result is not None
        assert result["type"] == "percentile"
        assert result["data"] == percentile_data

    def test_load_calibration_model_regular(self, temp_dir):
        """Test loading regular calibration model."""
        # Create calibration directory and file
        calib_dir = temp_dir / "calibration"
        calib_dir.mkdir()
        calib_file = calib_dir / "calibration_model.pkl"
        
        calib_data = [(0.1, 0.2), (0.5, 0.6), (0.9, 0.8)]
        with open(calib_file, "wb") as f:
            pkl.dump(calib_data, f)
        
        result = load_calibration_model(str(temp_dir))
        assert result is not None
        assert result["type"] == "regular"
        assert result["data"] == calib_data

    def test_load_calibration_model_multiclass(self, temp_dir):
        """Test loading multiclass calibration models."""
        # Create calibration directory and subdirectory
        calib_dir = temp_dir / "calibration"
        calib_dir.mkdir()
        multiclass_dir = calib_dir / "calibration_models"
        multiclass_dir.mkdir()
        
        # Create calibration files
        class_0_file = multiclass_dir / "calibration_model_class_0.pkl"
        class_1_file = multiclass_dir / "calibration_model_class_1.pkl"
        
        class_0_data = [(0.1, 0.2), (0.5, 0.6)]
        class_1_data = [(0.2, 0.3), (0.7, 0.8)]
        
        with open(class_0_file, "wb") as f:
            pkl.dump(class_0_data, f)
        with open(class_1_file, "wb") as f:
            pkl.dump(class_1_data, f)
        
        result = load_calibration_model(str(temp_dir))
        assert result is not None
        assert result["type"] == "regular_multiclass"
        assert "0" in result["data"]
        assert "1" in result["data"]
        assert result["data"]["0"] == class_0_data
        assert result["data"]["1"] == class_1_data

    def test_load_calibration_model_none(self, temp_dir):
        """Test loading calibration model when none exists."""
        result = load_calibration_model(str(temp_dir))
        assert result is None

    def test_interpolate_score(self):
        """Test score interpolation function."""
        # Test lookup table
        lookup_table = [(0.0, 0.1), (0.5, 0.6), (1.0, 0.9)]
        
        # Test exact matches
        assert _interpolate_score(0.0, lookup_table) == 0.1
        assert _interpolate_score(0.5, lookup_table) == 0.6
        assert _interpolate_score(1.0, lookup_table) == 0.9
        
        # Test interpolation
        interpolated = _interpolate_score(0.25, lookup_table)
        assert 0.1 <= interpolated <= 0.6  # Should be between 0.1 and 0.6
        
        # Test boundary cases
        assert _interpolate_score(-0.1, lookup_table) == 0.1  # Below range
        assert _interpolate_score(1.1, lookup_table) == 0.9   # Above range

    def test_apply_percentile_calibration(self):
        """Test applying percentile calibration."""
        # Mock _interpolate_score
        with patch('projects.rnr_pytorch_bedrock.dockers.pytorch_inference_handler._interpolate_score') as mock_interpolate:
            mock_interpolate.return_value = 0.7
            
            scores = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
            percentile_mapping = [(0.1, 0.2), (0.5, 0.6), (0.9, 0.8)]
            
            result = apply_percentile_calibration(scores, percentile_mapping)
            
            assert result.shape == scores.shape
            # Verify the mock was called correctly
            assert mock_interpolate.call_count == 3  # One call per row
            mock_interpolate.assert_any_call(0.9, percentile_mapping)

    def test_apply_regular_binary_calibration(self):
        """Test applying regular binary calibration."""
        # Mock _interpolate_score
        with patch('projects.rnr_pytorch_bedrock.dockers.pytorch_inference_handler._interpolate_score') as mock_interpolate:
            mock_interpolate.return_value = 0.7
            
            scores = np.array([[0.1, 0.9], [0.8, 0.2]])
            calibrator = [(0.1, 0.2), (0.5, 0.6), (0.9, 0.8)]
            
            result = apply_regular_binary_calibration(scores, calibrator)
            
            assert result.shape == scores.shape
            assert mock_interpolate.call_count == 2  # One call per row
            mock_interpolate.assert_any_call(0.9, calibrator)

    def test_apply_regular_multiclass_calibration(self):
        """Test applying regular multiclass calibration."""
        # Mock _interpolate_score
        with patch('projects.rnr_pytorch_bedrock.dockers.pytorch_inference_handler._interpolate_score') as mock_interpolate:
            mock_interpolate.return_value = 0.7
            
            scores = np.array([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])
            calibrators = {
                "0": [(0.1, 0.2), (0.5, 0.6)],
                "1": [(0.2, 0.3), (0.6, 0.7)],
                "2": [(0.3, 0.4), (0.7, 0.8)]
            }
            
            result = apply_regular_multiclass_calibration(scores, calibrators)
            
            assert result.shape == scores.shape
            # Verify normalization (rows should sum to 1)
            row_sums = result.sum(axis=1)
            np.testing.assert_allclose(row_sums, 1.0, rtol=1e-10)

    def test_apply_calibration_percentile(self):
        """Test applying calibration with percentile type."""
        # Mock apply_percentile_calibration
        with patch('projects.rnr_pytorch_bedrock.dockers.pytorch_inference_handler.apply_percentile_calibration') as mock_apply:
            mock_apply.return_value = np.array([[0.2, 0.8], [0.7, 0.3]])
            
            scores = np.array([[0.1, 0.9], [0.8, 0.2]])
            calibrator = {"type": "percentile", "data": "test_data"}
            
            result = apply_calibration(scores, calibrator, is_multiclass=False)
            
            assert np.array_equal(result, np.array([[0.2, 0.8], [0.7, 0.3]]))
            mock_apply.assert_called_once_with(scores, "test_data")

    def test_apply_calibration_regular_binary(self):
        """Test applying calibration with regular binary type."""
        # Mock apply_regular_binary_calibration
        with patch('projects.rnr_pytorch_bedrock.dockers.pytorch_inference_handler.apply_regular_binary_calibration') as mock_apply:
            mock_apply.return_value = np.array([[0.2, 0.8], [0.7, 0.3]])
            
            scores = np.array([[0.1, 0.9], [0.8, 0.2]])
            calibrator = {"type": "regular", "data": "test_data"}
            
            result = apply_calibration(scores, calibrator, is_multiclass=False)
            
            assert np.array_equal(result, np.array([[0.2, 0.8], [0.7, 0.3]]))
            mock_apply.assert_called_once_with(scores, "test_data")

    def test_apply_calibration_none(self):
        """Test applying calibration when calibrator is None."""
        scores = np.array([[0.1, 0.9], [0.8, 0.2]])
        result = apply_calibration(scores, None, is_multiclass=False)
        assert np.array_equal(result, scores)

    def test_preprocess_single_record_fast(self):
        """Test fast path preprocessing for single record."""
        # Create test data
        df = pd.DataFrame({
            'feature1': [1.0],  # numerical
            'feature2': [2.0],  # numerical
            'category1': ['A'],  # categorical
            'category2': ['B'],  # categorical
        })
        
        config = Config(
            tab_field_list=['feature1', 'feature2'],
            cat_field_list=['category1', 'category2']
        )
        
        # Mock processors
        mock_numerical_processor_1 = MagicMock()
        mock_numerical_processor_1.process.return_value = 1.5
        mock_numerical_processor_2 = MagicMock()
        mock_numerical_processor_2.process.return_value = 2.5
        
        mock_risk_processor_1 = MagicMock()
        mock_risk_processor_1.process.return_value = 0.1
        mock_risk_processor_2 = MagicMock()
        mock_risk_processor_2.process.return_value = 0.3
        
        numerical_processors = {
            'feature1': mock_numerical_processor_1,
            'feature2': mock_numerical_processor_2
        }
        
        risk_processors = {
            'category1': mock_risk_processor_1,
            'category2': mock_risk_processor_2
        }
        
        result = preprocess_single_record_fast(df, config, risk_processors, numerical_processors)
        
        # Verify results
        assert result['feature1'] == 1.5
        assert result['feature2'] == 2.5
        assert result['category1'] == 0.1
        assert result['category2'] == 0.3
        
        # Verify processor methods were called
        mock_numerical_processor_1.process.assert_called_once_with(1.0)
        mock_numerical_processor_2.process.assert_called_once_with(2.0)
        mock_risk_processor_1.process.assert_called_once_with('A')
        mock_risk_processor_2.process.assert_called_once_with('B')


class TestModelFunction:
    """Tests for model_fn function."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def create_test_model_artifacts(self, model_dir):
        """Helper to create test model artifacts."""
        # Create required files
        model_artifacts = {
            "config": {"model_class": "multimodal_bert"},
            "embedding_mat": None,
            "vocab": None,
            "model_class": "multimodal_bert"
        }
        model_artifacts_file = model_dir / "model_artifacts.pth"
        with open(model_artifacts_file, "wb") as f:
            pkl.dump(model_artifacts, f)
        
        (model_dir / "model.pth").touch()
        
        # Create optional files
        with open(model_dir / "feature_columns.txt", "w") as f:
            f.write("0,feature1\n1,feature2\n")
        
        with open(model_dir / "hyperparameters.json", "w") as f:
            json.dump({"learning_rate": 0.01}, f)
        
        with open(model_dir / "risk_table_map.pkl", "wb") as f:
            pkl.dump({"category_feature": {"bins": {"A": 0.1}, "default_bin": 0.5}}, f)
        
        with open(model_dir / "impute_dict.pkl", "wb") as f:
            pkl.dump({"feature1": 10.0}, f)

    @patch('projects.rnr_pytorch_bedrock.dockers.pytorch_inference_handler.load_artifacts')
    @patch('projects.rnr_pytorch_bedrock.dockers.pytorch_inference_handler.load_model')
    @patch('projects.rnr_pytorch_bedrock.dockers.pytorch_inference_handler.load_onnx_model')
    def test_model_fn_pytorch_model(self, mock_load_onnx, mock_load_model, mock_load_artifacts, temp_dir):
        """Test model_fn with PyTorch model."""
        # Create test artifacts
        self.create_test_model_artifacts(temp_dir)
        
        # Mock returns
        mock_load_artifacts.return_value = (
            {"model_class": "multimodal_bert"},
            None,  # embedding_mat
            None,  # vocab
            "multimodal_bert"  # model_class
        )
        
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model
        
        # Mock processor creation functions
        with patch('projects.rnr_pytorch_bedrock.dockers.pytorch_inference_handler.create_risk_processors') as mock_create_risk, \
             patch('projects.rnr_pytorch_bedrock.dockers.pytorch_inference_handler.create_numerical_processors') as mock_create_numerical, \
             patch('projects.rnr_pytorch_bedrock.dockers.pytorch_inference_handler.data_preprocess_pipeline') as mock_preprocess:
            
            mock_create_risk.return_value = {"category_feature": MagicMock()}
            mock_create_numerical.return_value = {"feature1": MagicMock()}
            mock_preprocess.return_value = (MagicMock(), {"text": MagicMock()})
            
            # Test when ONNX model doesn't exist (should load PyTorch model)
            onnx_file = temp_dir / "model.onnx"
            assert not onnx_file.exists()  # Verify ONNX file doesn't exist
            
            result = model_fn(str(temp_dir))
            
            # Verify PyTorch model was loaded
            mock_load_model.assert_called_once()
            assert result["model"] == mock_model
            assert "config" in result
            assert "risk_processors" in result
            assert "numerical_processors" in result
            assert "pipelines" in result

    @patch('projects.rnr_pytorch_bedrock.dockers.pytorch_inference_handler.load_artifacts')
    @patch('projects.rnr_pytorch_bedrock.dockers.pytorch_inference_handler.load_onnx_model')
    def test_model_fn_onnx_model(self, mock_load_onnx, mock_load_artifacts, temp_dir):
        """Test model_fn with ONNX model."""
        # Create test artifacts
        self.create_test_model_artifacts(temp_dir)
        
        # Create ONNX model file
        onnx_file = temp_dir / "model.onnx"
        onnx_file.touch()
        
        # Mock returns
        mock_load_artifacts.return_value = (
            {"model_class": "multimodal_bert"},
            None,  # embedding_mat
            None,  # vocab
            "multimodal_bert"  # model_class
        )
        
        mock_onnx_model = MagicMock()
        mock_load_onnx.return_value = mock_onnx_model
        
        # Mock processor creation functions
        with patch('projects.rnr_pytorch_bedrock.dockers.pytorch_inference_handler.create_risk_processors') as mock_create_risk, \
             patch('projects.rnr_pytorch_bedrock.dockers.pytorch_inference_handler.create_numerical_processors') as mock_create_numerical, \
             patch('projects.rnr_pytorch_bedrock.dockers.pytorch_inference_handler.data_preprocess_pipeline') as mock_preprocess:
            
            mock_create_risk.return_value = {"category_feature": MagicMock()}
            mock_create_numerical.return_value = {"feature1": MagicMock()}
            mock_preprocess.return_value = (MagicMock(), {"text": MagicMock()})
            
            result = model_fn(str(temp_dir))
            
            # Verify ONNX model was loaded
            mock_load_onnx.assert_called_once()
            assert result["model"] == mock_onnx_model

    def test_model_fn_missing_model_artifacts(self, temp_dir):
        """Test model_fn with missing model artifacts."""
        # Don't create any artifacts, just an empty directory
        with pytest.raises(Exception):
            model_fn(str(temp_dir))


class TestInputFunction:
    """Tests for input_fn function."""

    def test_input_fn_csv(self):
        """Test input_fn with CSV data."""
        csv_data = "1,0.5,A\n2,0.8,B\n3,0.3,C"
        content_type = "text/csv"
        
        result = input_fn(csv_data, content_type)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result.columns) == [0, 1, 2]  # Default column names

    def test_input_fn_json_single_object(self):
        """Test input_fn with single JSON object."""
        json_data = '{"id": 1, "value": 0.5, "category": "A"}'
        content_type = "application/json"
        
        result = input_fn(json_data, content_type)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]["id"] == 1
        assert result.iloc[0]["value"] == 0.5
        assert result.iloc[0]["category"] == "A"

    def test_input_fn_json_array(self):
        """Test input_fn with JSON array."""
        json_data = '[{"id": 1, "value": 0.5}, {"id": 2, "value": 0.8}]'
        content_type = "application/json"
        
        result = input_fn(json_data, content_type)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result["id"]) == [1, 2]
        assert list(result["value"]) == [0.5, 0.8]

    def test_input_fn_json_ndjson(self):
        """Test input_fn with NDJSON (newline-delimited JSON)."""
        json_data = '{"id": 1, "value": 0.5}\n{"id": 2, "value": 0.8}'
        content_type = "application/json"
        
        result = input_fn(json_data, content_type)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_input_fn_parquet(self):
        """Test input_fn with Parquet data."""
        # Create test DataFrame and convert to parquet bytes
        df = pd.DataFrame({"id": [1, 2, 3], "value": [0.5, 0.8, 0.3]})
        parquet_bytes = df.to_parquet()
        content_type = "application/x-parquet"
        
        result = input_fn(parquet_bytes, content_type)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        pd.testing.assert_frame_equal(result, df)

    def test_input_fn_unsupported_content_type(self):
        """Test input_fn with unsupported content type."""
        data = "test data"
        content_type = "text/plain"
        
        with pytest.raises(ValueError, match="This predictor only supports"):
            input_fn(data, content_type)

    def test_input_fn_invalid_csv(self):
        """Test input_fn with invalid CSV data."""
        csv_data = "invalid,csv,data\nwith,errors"
        content_type = "text/csv"
        
        # Should raise ValueError for parsing errors
        with pytest.raises(ValueError):
            input_fn(csv_data, content_type)

    def test_input_fn_invalid_json(self):
        """Test input_fn with invalid JSON data."""
        json_data = "invalid json"
        content_type = "application/json"
        
        with pytest.raises(ValueError):
            input_fn(json_data, content_type)


class TestPredictFunction:
    """Tests for predict_fn function."""

    @pytest.fixture
    def sample_model_data(self):
        """Create sample model data."""
        config = Config(
            is_binary=True,
            num_classes=2,
            tab_field_list=["feature1"],
            cat_field_list=["category"],
            text_name="text"
        )
        
        return {
            "model": MagicMock(),
            "config": config,
            "pipelines": {"text": MagicMock()},
            "risk_processors": {"category": MagicMock()},
            "numerical_processors": {"feature1": MagicMock()},
            "calibrator": None
        }

    @pytest.fixture
    def sample_input_data(self):
        """Create sample input data."""
        return pd.DataFrame({
            "feature1": [1.0, 2.0],
            "category": ["A", "B"],
            "text": ["text1", "text2"]
        })

    @patch('projects.rnr_pytorch_bedrock.dockers.pytorch_inference_handler.model_online_inference')
    @patch('projects.rnr_pytorch_bedrock.dockers.pytorch_inference_handler.PipelineDataset')
    def test_predict_fn_batch(self, mock_pipeline_dataset_class, mock_model_inference, sample_model_data, sample_input_data):
        """Test predict_fn with batch data."""
        # Mock returns
        mock_dataset_instance = MagicMock()
        mock_pipeline_dataset_class.return_value = mock_dataset_instance
        
        mock_predictions = np.array([[0.1, 0.9], [0.8, 0.2]])
        mock_model_inference.return_value = mock_predictions
        
        result = predict_fn(sample_input_data, sample_model_data)
        
        # Verify results
        assert isinstance(result, dict)
        assert "raw_predictions" in result
        assert "calibrated_predictions" in result
        assert np.array_equal(result["raw_predictions"], mock_predictions)
        assert np.array_equal(result["calibrated_predictions"], mock_predictions)  # No calibration, so same as raw

    @patch('projects.rnr_pytorch_bedrock.dockers.pytorch_inference_handler.model_online_inference')
    @patch('projects.rnr_pytorch_bedrock.dockers.pytorch_inference_handler.PipelineDataset')
    def test_predict_fn_single_record(self, mock_pipeline_dataset_class, mock_model_inference, sample_model_data):
        """Test predict_fn with single record."""
        # Create single record data
        single_record_data = pd.DataFrame({
            "feature1": [1.0],
            "category": ["A"],
            "text": ["text1"]
        })
        
        # Mock returns
        mock_dataset_instance = MagicMock()
        mock_pipeline_dataset_class.return_value = mock_dataset_instance
        
        mock_predictions = np.array([[0.3, 0.7]])
        mock_model_inference.return_value = mock_predictions
        
        result = predict_fn(single_record_data, sample_model_data)
        
        # Verify results
        assert isinstance(result, dict)
        assert "raw_predictions" in result
        assert "calibrated_predictions" in result
        assert np.array_equal(result["raw_predictions"], mock_predictions)

    def test_predict_fn_invalid_input_type(self, sample_model_data):
        """Test predict_fn with invalid input type."""
        invalid_input = "not a dataframe"
        
        with pytest.raises(TypeError, match="input data type must be pandas.DataFrame"):
            predict_fn(invalid_input, sample_model_data)

    @patch('projects.rnr_pytorch_bedrock.dockers.pytorch_inference_handler.model_online_inference')
    @patch('projects.rnr_pytorch_bedrock.dockers.pytorch_inference_handler.PipelineDataset')
    def test_predict_fn_with_calibration(self, mock_pipeline_dataset_class, mock_model_inference):
        """Test predict_fn with calibration applied."""
        # Create model data with calibration
        config = Config(is_binary=True, num_classes=2, text_name="text")
        model_data = {
            "model": MagicMock(),
            "config": config,
            "pipelines": {"text": MagicMock()},
            "risk_processors": {},
            "numerical_processors": {},
            "calibrator": {"type": "regular", "data": [(0.1, 0.2), (0.5, 0.6), (0.9, 0.8)]}
        }
        
        # Create input data
        input_data = pd.DataFrame({
            "text": ["text1"]
        })
        
        # Mock returns
        mock_dataset_instance = MagicMock()
        mock_pipeline_dataset_class.return_value = mock_dataset_instance
        
        raw_predictions = np.array([[0.3, 0.7]])
        mock_model_inference.return_value = raw_predictions
        
        # Mock apply_calibration
        with patch('projects.rnr_pytorch_bedrock.dockers.pytorch_inference_handler.apply_calibration') as mock_apply:
            calibrated_predictions = np.array([[0.4, 0.6]])
            mock_apply.return_value = calibrated_predictions
            
            result = predict_fn(input_data, model_data)
            
            # Verify calibration was applied
            assert np.array_equal(result["raw_predictions"], raw_predictions)
            assert np.array_equal(result["calibrated_predictions"], calibrated_predictions)

    @patch('projects.rnr_pytorch_bedrock.dockers.pytorch_inference_handler.PipelineDataset')
    def test_predict_fn_model_error(self, mock_pipeline_dataset_class, sample_model_data, sample_input_data):
        """Test predict_fn when model inference fails."""
        # Mock returns
        mock_dataset_instance = MagicMock()
        mock_pipeline_dataset_class.return_value = mock_dataset_instance
        
        # Simulate model inference error
        with patch('projects.rnr_pytorch_bedrock.dockers.pytorch_inference_handler.model_online_inference') as mock_model_inference:
            mock_model_inference.side_effect = Exception("Model error")
            
            result = predict_fn(sample_input_data, sample_model_data)
            
            # Should return error indicator
            assert result == [-4]


class TestOutputFunction:
    """Tests for output_fn function."""

    def test_output_fn_json_binary(self):
        """Test output_fn with binary classification JSON output."""
        prediction_output = {
            "raw_predictions": np.array([[0.3, 0.7], [0.8, 0.2]]),
            "calibrated_predictions": np.array([[0.4, 0.6], [0.9, 0.1]])
        }
        accept = "application/json"
        
        response_body, content_type = output_fn(prediction_output, accept)
        
        assert content_type == "application/json"
        assert isinstance(response_body, str)
        
        # Parse JSON to verify content
        import json
        response_data = json.loads(response_body)
        assert "predictions" in response_data
        assert len(response_data["predictions"]) == 2
        assert "legacy-score" in response_data["predictions"][0]
        assert "calibrated-score" in response_data["predictions"][0]
        assert "output-label" in response_data["predictions"][0]

    def test_output_fn_csv_binary(self):
        """Test output_fn with binary classification CSV output."""
        prediction_output = {
            "raw_predictions": np.array([[0.3, 0.7], [0.8, 0.2]]),
            "calibrated_predictions": np.array([[0.4, 0.6], [0.9, 0.1]])
        }
        accept = "text/csv"
        
        response_body, content_type = output_fn(prediction_output, accept)
        
        assert content_type == "text/csv"
        assert isinstance(response_body, str)
        # Should contain CSV data with newlines
        assert "\n" in response_body

    def test_output_fn_json_multiclass(self):
        """Test output_fn with multiclass classification JSON output."""
        prediction_output = {
            "raw_predictions": np.array([[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]]),
            "calibrated_predictions": np.array([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2]])
        }
        accept = "application/json"
        
        response_body, content_type = output_fn(prediction_output, accept)
        
        assert content_type == "application/json"
        assert isinstance(response_body, str)
        
        # Parse JSON to verify content
        import json
        response_data = json.loads(response_body)
        assert "predictions" in response_data
        assert len(response_data["predictions"]) == 2
        
        # Should have prob_01, prob_02, prob_03 etc. for multiclass
        first_prediction = response_data["predictions"][0]
        assert "prob_01" in first_prediction
        assert "prob_02" in first_prediction
        assert "prob_03" in first_prediction
        assert "calibrated_prob_01" in first_prediction
        assert "calibrated_prob_02" in first_prediction
        assert "calibrated_prob_03" in first_prediction
        assert "output-label" in first_prediction

    def test_output_fn_unsupported_accept_type(self):
        """Test output_fn with unsupported accept type."""
        prediction_output = {
            "raw_predictions": np.array([[0.3, 0.7]]),
            "calibrated_predictions": np.array([[0.4, 0.6]])
        }
        accept = "text/html"
        
        with pytest.raises(ValueError, match="Unsupported accept type"):
            output_fn(prediction_output, accept)

    def test_output_fn_legacy_format(self):
        """Test output_fn with legacy numpy array format."""
        # Test backward compatibility with numpy array input
        prediction_output = np.array([[0.3, 0.7], [0.8, 0.2]])
        accept = "application/json"
        
        response_body, content_type = output_fn(prediction_output, accept)
        
        assert content_type == "application/json"
        assert isinstance(response_body, str)

    def test_output_fn_invalid_data(self):
        """Test output_fn with invalid prediction data."""
        prediction_output = {
            "raw_predictions": "invalid_data",  # Not array-like
            "calibrated_predictions": "invalid_data"
        }
        accept = "application/json"
        
        # Should handle gracefully and return error response
        response_body, content_type = output_fn(prediction_output, accept)
        
        assert content_type == "application/json"
        assert "error" in response_body


class TestEdgeCasesAndErrorHandling:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_empty_dataframe_input(self):
        """Test handling of empty DataFrame input."""
        empty_df = pd.DataFrame()
        content_type = "text/csv"
        
        # Should handle gracefully
        result = input_fn("", content_type)  # Empty CSV
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_single_column_dataframe(self):
        """Test handling of single column DataFrame."""
        df = pd.DataFrame({"single_col": [1, 2, 3]})
        config = Config()
        result = get_text_field_names(config)
        assert result == set()

    def test_extreme_scores_in_calibration(self):
        """Test calibration with extreme scores."""
        lookup_table = [(0.0, 0.1), (1.0, 0.9)]
        
        # Test scores outside range
        assert _interpolate_score(-0.5, lookup_table) == 0.1  # Below range
        assert _interpolate_score(1.5, lookup_table) == 0.9   # Above range
        
        # Test exact endpoints
        assert _interpolate_score(0.0, lookup_table) == 0.1
        assert _interpolate_score(1.0, lookup_table) == 0.9

    def test_nan_values_in_preprocessing(self):
        """Test handling of NaN values in preprocessing."""
        df = pd.DataFrame({
            'feature1': [np.nan],  # numerical with NaN
            'category1': [None],   # categorical with None
        })
        
        config = Config(
            tab_field_list=['feature1'],
            cat_field_list=['category1']
        )
        
        # Mock processors that can handle NaN/None
        mock_numerical_processor = MagicMock()
        mock_numerical_processor.process.return_value = 0.0  # Impute NaN
        
        mock_risk_processor = MagicMock()
        mock_risk_processor.process.return_value = 0.5  # Default for None
        
        numerical_processors = {'feature1': mock_numerical_processor}
        risk_processors = {'category1': mock_risk_processor}
        
        result = preprocess_single_record_fast(df, config, risk_processors, numerical_processors)
        
        assert result['feature1'] == 0.0
        assert result['category1'] == 0.5

    def test_unicode_text_in_processing(self):
        """Test handling of unicode text in preprocessing."""
        df = pd.DataFrame({
            'category1': ['категория_А'],  # Cyrillic
            'category2': ['カテゴリー_B'],  # Japanese
        })
        
        config = Config(cat_field_list=['category1', 'category2'])
        
        # Mock processors
        mock_risk_processor_1 = MagicMock()
        mock_risk_processor_1.process.return_value = 0.1
        mock_risk_processor_2 = MagicMock()
        mock_risk_processor_2.process.return_value = 0.3
        
        risk_processors = {
            'category1': mock_risk_processor_1,
            'category2': mock_risk_processor_2
        }
        numerical_processors = {}
        
        result = preprocess_single_record_fast(df, config, risk_processors, numerical_processors)
        
        # Should handle unicode without errors
        assert result['category1'] == 0.1
        assert result['category2'] == 0.3

    def test_large_batch_predictions(self):
        """Test handling of large batch predictions."""
        # Create large DataFrame
        large_size = 1000
        large_df = pd.DataFrame({
            'feature1': np.random.rand(large_size),
            'category1': [f'cat_{i}' for i in range(large_size)],
            'text': [f'text_{i}' for i in range(large_size)]
        })
        
        config = Config(
            tab_field_list=['feature1'],
            cat_field_list=['category1'],
            text_name='text'
        )
        
        model_data = {
            "model": MagicMock(),
            "config": config,
            "pipelines": {"text": MagicMock()},
            "risk_processors": {"category1": MagicMock()},
            "numerical_processors": {"feature1": MagicMock()},
            "calibrator": None
        }
        
        # Mock model inference to handle large batch
        with patch('projects.rnr_pytorch_bedrock.dockers.pytorch_inference_handler.model_online_inference') as mock_inference, \
             patch('projects.rnr_pytorch_bedrock.dockers.pytorch_inference_handler.PipelineDataset'):
            
            mock_predictions = np.random.rand(large_size, 2)
            mock_inference.return_value = mock_predictions
            
            result = predict_fn(large_df, model_data)
            
            assert isinstance(result, dict)
            assert "raw_predictions" in result
            assert result["raw_predictions"].shape[0] == large_size

    def test_missing_files_in_model_loading(self, temp_dir):
        """Test model loading when some files are missing."""
        # Create only some required files
        model_artifacts = {
            "config": {"model_class": "multimodal_bert"},
            "embedding_mat": None,
            "vocab": None,
            "model_class": "multimodal_bert"
        }
        model_artifacts_file = temp_dir / "model_artifacts.pth"
        with open(model_artifacts_file, "wb") as f:
            pkl.dump(model_artifacts, f)
        
        (temp_dir / "model.pth").touch()
        # Don't create optional files
        
        with patch('projects.rnr_pytorch_bedrock.dockers.pytorch_inference_handler.load_artifacts') as mock_load_artifacts, \
             patch('projects.rnr_pytorch_bedrock.dockers.pytorch_inference_handler.load_model') as mock_load_model:
            
            mock_load_artifacts.return_value = (
                {"model_class": "multimodal_bert"},
                None,
                None,
                "multimodal_bert"
            )
            
            mock_model = MagicMock()
            mock_load_model.return_value = mock_model
            
            # Should handle missing optional files gracefully
            result = model_fn(str(temp_dir))
            assert result["model"] == mock_model

    def test_corrupted_files_in_model_loading(self, temp_dir):
        """Test model loading with corrupted files."""
        # Create corrupted files
        model_artifacts_file = temp_dir / "model_artifacts.pth"
        with open(model_artifacts_file, "w") as f:
            f.write("corrupted data")
        
        (temp_dir / "model.pth").touch()
        
        with pytest.raises(Exception):
            model_fn(str(temp_dir))

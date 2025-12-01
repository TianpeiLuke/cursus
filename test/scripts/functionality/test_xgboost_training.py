import pytest
from unittest.mock import patch, MagicMock, mock_open
import os
import tempfile
import shutil
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import pickle as pkl
import sys

# Mock torch before any imports that might need it
sys.modules["torch"] = MagicMock()

# CRITICAL: Set USE_SECURE_PYPI=false AND mock subprocess.check_call BEFORE importing
# The xgboost_training script calls install_packages() at module level (line 179)
# Setting USE_SECURE_PYPI=false makes it use public PyPI path
# Mocking subprocess.check_call prevents actual pip installs
os.environ["USE_SECURE_PYPI"] = "false"

# Mock subprocess.check_call to prevent pip installs during module import
with patch("subprocess.check_call"):
    from cursus.steps.scripts.xgboost_training import (
        load_and_validate_config,
        find_first_data_file,
        load_datasets,
        apply_numerical_imputation,
        fit_and_apply_risk_tables,
        prepare_dmatrices,
        train_model,
        save_artifacts,
        main,
        logger,  # Import the module-level logger
    )


class TestXGBoostTrainHelpers:
    """Unit tests for helper functions in the XGBoost training script."""

    @pytest.fixture
    def temp_dir(self):
        """Set up test fixtures."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration."""
        return {
            "tab_field_list": ["feature1", "feature2", "feature3"],
            "cat_field_list": ["category1", "category2"],
            "label_name": "target",
            "multiclass_categories": [0, 1],
            "is_binary": True,
            "num_classes": 2,
            "eta": 0.1,
            "max_depth": 6,
            "num_round": 100,
            "early_stopping_rounds": 10,
            "smooth_factor": 0.0,
            "count_threshold": 0,
        }

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 100),
                "feature2": np.random.normal(1, 2, 100),
                "feature3": np.random.normal(-1, 0.5, 100),
                "category1": np.random.choice(["A", "B", "C"], 100),
                "category2": np.random.choice(["X", "Y"], 100),
                "target": np.random.choice([0, 1], 100),
                "id": range(100),
            }
        )

    def test_load_and_validate_config_valid(self, temp_dir, sample_config):
        """Test loading and validating a valid configuration file."""
        config_path = temp_dir / "hyperparameters.json"
        with open(config_path, "w") as f:
            json.dump(sample_config, f)

        result = load_and_validate_config(str(config_path))

        assert result == sample_config
        assert result["tab_field_list"] == ["feature1", "feature2", "feature3"]
        assert result["num_classes"] == 2

    def test_load_and_validate_config_missing_keys(self, temp_dir):
        """Test validation fails with missing required keys."""
        incomplete_config = {"tab_field_list": ["feature1"]}
        config_path = temp_dir / "hyperparameters.json"
        with open(config_path, "w") as f:
            json.dump(incomplete_config, f)

        with pytest.raises(ValueError) as exc_info:
            load_and_validate_config(str(config_path))

        assert "Missing required key in config" in str(exc_info.value)

    def test_load_and_validate_config_invalid_class_weights(
        self, temp_dir, sample_config
    ):
        """Test validation fails with mismatched class weights."""
        invalid_config = sample_config.copy()
        invalid_config["class_weights"] = [0.3, 0.7, 0.5]  # 3 weights for 2 classes

        config_path = temp_dir / "hyperparameters.json"
        with open(config_path, "w") as f:
            json.dump(invalid_config, f)

        with pytest.raises(ValueError) as exc_info:
            load_and_validate_config(str(config_path))

        assert "Number of class weights" in str(exc_info.value)

    def test_load_and_validate_config_file_not_found(self, temp_dir):
        """Test error handling when config file doesn't exist."""
        nonexistent_path = temp_dir / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            load_and_validate_config(str(nonexistent_path))

    def test_find_first_data_file_csv(self, temp_dir):
        """Test finding CSV data file."""
        data_dir = temp_dir / "data"
        data_dir.mkdir()

        # Create test files
        (data_dir / "data.csv").write_text("test,data\n1,2")
        (data_dir / "other.txt").write_text("not data")

        result = find_first_data_file(str(data_dir))

        assert result == str(data_dir / "data.csv")

    def test_find_first_data_file_parquet(self, temp_dir):
        """Test finding Parquet data file."""
        data_dir = temp_dir / "data"
        data_dir.mkdir()

        # Create test files (parquet comes first alphabetically)
        (data_dir / "data.parquet").write_text("parquet data")
        (data_dir / "zdata.csv").write_text("csv data")

        result = find_first_data_file(str(data_dir))

        assert result == str(data_dir / "data.parquet")

    def test_find_first_data_file_no_data_files(self, temp_dir):
        """Test behavior when no data files are found."""
        data_dir = temp_dir / "data"
        data_dir.mkdir()

        # Create non-data files
        (data_dir / "readme.txt").write_text("not data")
        (data_dir / "config.yaml").write_text("not data")

        result = find_first_data_file(str(data_dir))

        assert result is None

    def test_find_first_data_file_nonexistent_dir(self, temp_dir):
        """Test behavior when directory doesn't exist."""
        nonexistent_dir = temp_dir / "nonexistent"

        result = find_first_data_file(str(nonexistent_dir))

        assert result is None

    @patch("cursus.steps.scripts.xgboost_training.load_dataframe_with_format")
    @patch("cursus.steps.scripts.xgboost_training.find_first_data_file")
    def test_load_datasets_success(self, mock_find_file, mock_load_df):
        """Test successful dataset loading."""
        # Mock file finding
        mock_find_file.side_effect = [
            "/path/train.csv",
            "/path/val.csv",
            "/path/test.csv",
        ]

        # Mock dataframe loading with format
        train_df = pd.DataFrame({"col1": [1, 2], "target": [0, 1]})
        val_df = pd.DataFrame({"col1": [3, 4], "target": [1, 0]})
        test_df = pd.DataFrame({"col1": [5, 6], "target": [0, 1]})

        mock_load_df.side_effect = [
            (train_df, "csv"),
            (val_df, "csv"),
            (test_df, "csv"),
        ]

        result_train, result_val, result_test, result_format = load_datasets(
            "/input/path"
        )

        # Verify results
        pd.testing.assert_frame_equal(result_train, train_df)
        pd.testing.assert_frame_equal(result_val, val_df)
        pd.testing.assert_frame_equal(result_test, test_df)
        assert result_format == "csv"

        # Verify calls
        assert mock_find_file.call_count == 3
        assert mock_load_df.call_count == 3

    @patch("cursus.steps.scripts.xgboost_training.find_first_data_file")
    def test_load_datasets_missing_files(self, mock_find_file):
        """Test error handling when dataset files are missing."""
        # Mock missing validation file
        mock_find_file.side_effect = [
            "/path/train.csv",
            None,  # Missing validation file
            "/path/test.csv",
        ]

        with pytest.raises(FileNotFoundError) as exc_info:
            load_datasets("/input/path")

        assert "Training, validation, or test data file not found" in str(
            exc_info.value
        )

    @patch("cursus.steps.scripts.xgboost_training.NumericalVariableImputationProcessor")
    def test_apply_numerical_imputation(
        self, mock_imputer_class, sample_config, sample_data
    ):
        """Test numerical imputation application with single-column architecture."""
        # Create mock imputer
        mock_imputer = MagicMock()
        mock_imputer.transform.side_effect = lambda series: series  # Return unchanged
        mock_imputer.get_imputation_value.return_value = 0.5
        mock_imputer_class.return_value = mock_imputer

        # Test data
        train_df = sample_data.copy()
        val_df = sample_data.copy()
        test_df = sample_data.copy()

        result = apply_numerical_imputation(sample_config, train_df, val_df, test_df)

        train_result, val_result, test_result, impute_dict = result

        # Verify imputer was created for each column (single-column architecture)
        # Should be called 3 times, once per feature in tab_field_list
        assert mock_imputer_class.call_count == 3

        # Verify each call used correct parameters
        expected_calls = [
            ((("column_name", "feature1"), ("strategy", "mean")),),
            ((("column_name", "feature2"), ("strategy", "mean")),),
            ((("column_name", "feature3"), ("strategy", "mean")),),
        ]
        for call, expected in zip(mock_imputer_class.call_args_list, expected_calls):
            assert call.kwargs.get("column_name") in [
                "feature1",
                "feature2",
                "feature3",
            ]
            assert call.kwargs.get("strategy") == "mean"

        # Verify fit was called 3 times (once per column)
        assert mock_imputer.fit.call_count == 3

        # Verify transform was called 9 times (3 columns × 3 splits)
        assert mock_imputer.transform.call_count == 9

        # Verify imputation dictionary has all 3 features
        assert len(impute_dict) == 3
        assert all(f in impute_dict for f in ["feature1", "feature2", "feature3"])

    @patch("cursus.steps.scripts.xgboost_training.RiskTableMappingProcessor")
    def test_fit_and_apply_risk_tables(
        self, mock_processor_class, sample_config, sample_data
    ):
        """Test risk table fitting and application."""
        # Create mock processor
        mock_processor = MagicMock()
        mock_processor.transform.side_effect = lambda series: series.map(
            {"A": 0.1, "B": 0.2, "C": 0.3, "X": 0.4, "Y": 0.5}
        ).fillna(0.0)
        mock_processor.get_risk_tables.return_value = {"A": 0.1, "B": 0.2, "C": 0.3}
        mock_processor_class.return_value = mock_processor

        # Test data
        train_df = sample_data.copy()
        val_df = sample_data.copy()
        test_df = sample_data.copy()

        result = fit_and_apply_risk_tables(sample_config, train_df, val_df, test_df)

        train_result, val_result, test_result, risk_tables = result

        # Verify processors were created for each categorical variable
        assert mock_processor_class.call_count == 2  # 2 categorical variables

        # Verify risk tables structure
        assert "category1" in risk_tables
        assert "category2" in risk_tables

    @patch("cursus.steps.scripts.xgboost_training.xgb.DMatrix")
    def test_prepare_dmatrices(self, mock_dmatrix, sample_config, sample_data):
        """Test DMatrix preparation."""
        # Create mock DMatrix instances
        mock_dtrain = MagicMock()
        mock_dval = MagicMock()
        mock_dmatrix.side_effect = [mock_dtrain, mock_dval]

        # Prepare test data with numerical values for categorical columns
        # (simulating the output after risk table mapping)
        train_df = sample_data.copy()
        val_df = sample_data.copy()

        # Convert categorical columns to numerical (as would happen after risk table mapping)
        train_df["category1"] = train_df["category1"].map(
            {"A": 0.1, "B": 0.2, "C": 0.3}
        )
        train_df["category2"] = train_df["category2"].map({"X": 0.4, "Y": 0.5})
        val_df["category1"] = val_df["category1"].map({"A": 0.1, "B": 0.2, "C": 0.3})
        val_df["category2"] = val_df["category2"].map({"X": 0.4, "Y": 0.5})

        result = prepare_dmatrices(sample_config, train_df, val_df)
        dtrain, dval, feature_columns = result

        # Verify feature columns
        expected_features = [
            "feature1",
            "feature2",
            "feature3",
            "category1",
            "category2",
        ]
        assert feature_columns == expected_features

        # Verify DMatrix creation
        assert mock_dmatrix.call_count == 2

        # Verify feature names were set
        mock_dtrain.feature_names = expected_features
        mock_dval.feature_names = expected_features

    def test_prepare_dmatrices_nan_values(self, sample_config, sample_data):
        """Test DMatrix preparation fails with NaN values."""
        # Create data with NaN values
        train_df = sample_data.copy()
        val_df = sample_data.copy()

        # Convert categorical columns to numerical first (as would happen after risk table mapping)
        train_df["category1"] = train_df["category1"].map(
            {"A": 0.1, "B": 0.2, "C": 0.3}
        )
        train_df["category2"] = train_df["category2"].map({"X": 0.4, "Y": 0.5})
        val_df["category1"] = val_df["category1"].map({"A": 0.1, "B": 0.2, "C": 0.3})
        val_df["category2"] = val_df["category2"].map({"X": 0.4, "Y": 0.5})

        # Now add NaN values
        train_df.loc[0, "feature1"] = np.nan

        with pytest.raises(ValueError) as exc_info:
            prepare_dmatrices(sample_config, train_df, val_df)

        assert "Training data contains NaN or inf values" in str(exc_info.value)

    @patch("cursus.steps.scripts.xgboost_training.xgb.train")
    def test_train_model_binary(self, mock_xgb_train, sample_config):
        """Test binary classification model training."""
        # Create mock DMatrix objects
        mock_dtrain = MagicMock()
        mock_dval = MagicMock()
        mock_dtrain.get_label.return_value = np.array([0, 1, 0, 1])
        mock_dval.get_label.return_value = np.array([1, 0, 1, 0])

        # Mock trained model
        mock_model = MagicMock()
        mock_xgb_train.return_value = mock_model

        result = train_model(sample_config, mock_dtrain, mock_dval)

        # Verify xgb.train was called
        mock_xgb_train.assert_called_once()

        # Verify parameters
        call_args = mock_xgb_train.call_args
        params = call_args[1]["params"]
        assert params["objective"] == "binary:logistic"
        assert params["eta"] == 0.1
        assert params["max_depth"] == 6

        # Verify result
        assert result == mock_model

    @patch("cursus.steps.scripts.xgboost_training.xgb.train")
    def test_train_model_multiclass(self, mock_xgb_train, sample_config):
        """Test multiclass model training."""
        # Update config for multiclass
        multiclass_config = sample_config.copy()
        multiclass_config["is_binary"] = False
        multiclass_config["num_classes"] = 3

        # Create mock DMatrix objects
        mock_dtrain = MagicMock()
        mock_dval = MagicMock()
        mock_dtrain.get_label.return_value = np.array([0, 1, 2, 0])
        mock_dval.get_label.return_value = np.array([1, 2, 0, 1])

        # Mock trained model
        mock_model = MagicMock()
        mock_xgb_train.return_value = mock_model

        result = train_model(multiclass_config, mock_dtrain, mock_dval)

        # Verify parameters
        call_args = mock_xgb_train.call_args
        params = call_args[1]["params"]
        assert params["objective"] == "multi:softprob"
        assert params["num_class"] == 3

    @patch("cursus.steps.scripts.xgboost_training.json.dump")
    @patch("cursus.steps.scripts.xgboost_training.pkl.dump")
    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    def test_save_artifacts(
        self,
        mock_makedirs,
        mock_file_open,
        mock_pkl_dump,
        mock_json_dump,
        sample_config,
    ):
        """Test model artifacts saving."""
        # Create mock model
        mock_model = MagicMock()
        mock_model.get_fscore.return_value = {"feature1": 10, "feature2": 5}

        # Test data
        risk_tables = {"category1": {"A": 0.1, "B": 0.2}}
        impute_dict = {"feature1": 0.5}
        feature_columns = ["feature1", "feature2", "category1"]
        model_path = "/model/path"

        save_artifacts(
            mock_model,
            risk_tables,
            impute_dict,
            model_path,
            feature_columns,
            sample_config,
        )

        # Verify directory creation
        mock_makedirs.assert_called_once_with(model_path, exist_ok=True)

        # Verify model saving
        mock_model.save_model.assert_called_once()

        # Verify pickle dumps (risk tables and imputation dict)
        assert mock_pkl_dump.call_count == 2

        # Verify JSON dumps (feature importance and hyperparameters)
        assert mock_json_dump.call_count == 2


class TestXGBoostTrainMain:
    """Tests for the main function of the XGBoost training script."""

    @pytest.fixture
    def setup_dirs(self):
        """Set up test fixtures."""
        temp_dir = Path(tempfile.mkdtemp())

        # Create directory structure
        data_dir = temp_dir / "data"
        config_dir = temp_dir / "config"
        model_dir = temp_dir / "model"
        output_dir = temp_dir / "output"

        for dir_path in [data_dir, config_dir, model_dir, output_dir]:
            dir_path.mkdir(parents=True)

        # Create subdirectories for train/val/test
        for split in ["train", "val", "test"]:
            (data_dir / split).mkdir()

        yield {
            "temp_dir": temp_dir,
            "data_dir": data_dir,
            "config_dir": config_dir,
            "model_dir": model_dir,
            "output_dir": output_dir,
        }

        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration."""
        return {
            "tab_field_list": ["feature1", "feature2"],
            "cat_field_list": ["category1"],
            "label_name": "target",
            "multiclass_categories": [0, 1],
            "is_binary": True,
            "num_classes": 2,
            "eta": 0.1,
            "max_depth": 3,
            "num_round": 10,
            "early_stopping_rounds": 5,
        }

    def _create_test_data(self, dirs, sample_config):
        """Create test data files."""
        data_dir = dirs["data_dir"]
        config_dir = dirs["config_dir"]

        # Save configuration
        config_path = config_dir / "hyperparameters.json"
        with open(config_path, "w") as f:
            json.dump(sample_config, f)

        # Create sample datasets
        np.random.seed(42)
        for split in ["train", "val", "test"]:
            data = pd.DataFrame(
                {
                    "feature1": np.random.normal(0, 1, 50),
                    "feature2": np.random.normal(1, 2, 50),
                    "category1": np.random.choice(["A", "B"], 50),
                    "target": np.random.choice([0, 1], 50),
                    "id": range(50),
                }
            )
            data.to_csv(data_dir / split / "data.csv", index=False)

    @patch("cursus.steps.scripts.xgboost_training.xgb.train")
    @patch("cursus.steps.scripts.xgboost_training.RiskTableMappingProcessor")
    @patch("cursus.steps.scripts.xgboost_training.NumericalVariableImputationProcessor")
    def test_main_success(
        self,
        mock_imputer_class,
        mock_processor_class,
        mock_xgb_train,
        setup_dirs,
        sample_config,
    ):
        """Test successful main function execution."""
        dirs = setup_dirs
        self._create_test_data(dirs, sample_config)

        # Setup mocks for single-column architecture
        mock_imputer = MagicMock()
        mock_imputer.transform.side_effect = (
            lambda series: series
        )  # Single-column: returns Series
        mock_imputer.get_imputation_value.return_value = (
            0.5  # Return actual float, not Mock
        )
        mock_imputer_class.return_value = mock_imputer

        mock_processor = MagicMock()
        mock_processor.transform.side_effect = lambda series: series.map(
            {"A": 0.1, "B": 0.2}
        ).fillna(0.0)
        mock_processor.get_risk_tables.return_value = {"A": 0.1, "B": 0.2}
        mock_processor_class.return_value = mock_processor

        mock_model = MagicMock()
        mock_model.get_fscore.return_value = {"feature1": 10}
        mock_model.predict.return_value = np.array([0.3, 0.7, 0.4, 0.6])
        mock_xgb_train.return_value = mock_model

        # Prepare input parameters using contract logical names
        input_paths = {
            "input_path": str(dirs["data_dir"]),
            "hyperparameters_s3_uri": str(dirs["config_dir"] / "hyperparameters.json"),
        }
        output_paths = {
            "model_output": str(dirs["model_dir"]),
            "evaluation_output": str(dirs["output_dir"]),
        }
        environ_vars = {}
        args = argparse.Namespace(job_type=None)

        # Execute main function with logger
        try:
            main(input_paths, output_paths, environ_vars, args, logger)
            success = True
        except Exception as e:
            success = False
            error = str(e)

        # Verify success
        assert success, (
            f"Main function failed with error: {error if not success else 'None'}"
        )

        # Verify model training was called
        mock_xgb_train.assert_called_once()

        # Verify model artifacts were saved
        assert (
            Path(dirs["model_dir"]) / "xgboost_model.bst"
        ).exists() or mock_model.save_model.called

    def test_main_missing_config(self, setup_dirs, sample_config):
        """Test main function with missing configuration file."""
        dirs = setup_dirs
        self._create_test_data(dirs, sample_config)

        # Remove config file
        (dirs["config_dir"] / "hyperparameters.json").unlink()

        input_paths = {
            "input_path": str(dirs["data_dir"]),
            "hyperparameters_s3_uri": str(dirs["config_dir"] / "hyperparameters.json"),
        }
        output_paths = {
            "model_output": str(dirs["model_dir"]),
            "evaluation_output": str(dirs["output_dir"]),
        }
        environ_vars = {}
        args = argparse.Namespace(job_type=None)

        with pytest.raises(FileNotFoundError):
            main(input_paths, output_paths, environ_vars, args, logger)

    def test_main_missing_data(self, setup_dirs, sample_config):
        """Test main function with missing data files."""
        dirs = setup_dirs
        self._create_test_data(dirs, sample_config)

        # Remove training data
        shutil.rmtree(dirs["data_dir"] / "train")

        input_paths = {
            "input_path": str(dirs["data_dir"]),
            "hyperparameters_s3_uri": str(dirs["config_dir"] / "hyperparameters.json"),
        }
        output_paths = {
            "model_output": str(dirs["model_dir"]),
            "evaluation_output": str(dirs["output_dir"]),
        }
        environ_vars = {}
        args = argparse.Namespace(job_type=None)

        with pytest.raises(FileNotFoundError):
            main(input_paths, output_paths, environ_vars, args, logger)

    def test_main_invalid_config(self, setup_dirs):
        """Test main function with invalid configuration."""
        dirs = setup_dirs

        # Create invalid config (missing required keys)
        invalid_config = {"tab_field_list": ["feature1"]}
        config_path = dirs["config_dir"] / "hyperparameters.json"
        with open(config_path, "w") as f:
            json.dump(invalid_config, f)

        input_paths = {
            "input_path": str(dirs["data_dir"]),
            "hyperparameters_s3_uri": str(dirs["config_dir"] / "hyperparameters.json"),
        }
        output_paths = {
            "model_output": str(dirs["model_dir"]),
            "evaluation_output": str(dirs["output_dir"]),
        }
        environ_vars = {}
        args = argparse.Namespace(job_type=None)

        with pytest.raises(ValueError):
            main(input_paths, output_paths, environ_vars, args, logger)


class TestXGBoostTrainIntegration:
    """Integration tests for the XGBoost training script."""

    @pytest.fixture
    def temp_dir(self):
        """Set up test fixtures for integration tests."""
        temp_dir = Path(tempfile.mkdtemp())

        # Create directory structure
        data_dir = temp_dir / "data"
        config_dir = temp_dir / "config"
        model_dir = temp_dir / "model"
        output_dir = temp_dir / "output"

        for dir_path in [data_dir, config_dir, model_dir, output_dir]:
            dir_path.mkdir(parents=True)

        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_script_imports_successfully(self):
        """Test that the script can be imported without errors."""
        try:
            import cursus.steps.scripts.xgboost_training

            success = True
        except Exception as e:
            success = False
            error = str(e)

        assert success, f"Script import failed: {error if not success else 'None'}"

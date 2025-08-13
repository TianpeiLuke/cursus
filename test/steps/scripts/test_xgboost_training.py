import unittest
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

# Import the functions to be tested
from src.cursus.steps.scripts.xgboost_training import (
    load_and_validate_config,
    find_first_data_file,
    load_datasets,
    apply_numerical_imputation,
    fit_and_apply_risk_tables,
    prepare_dmatrices,
    train_model,
    save_artifacts,
    main
)


class TestXGBoostTrainHelpers(unittest.TestCase):
    """Unit tests for helper functions in the XGBoost training script."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create sample configuration
        self.sample_config = {
            "tab_field_list": ["feature1", "feature2", "feature3"],
            "cat_field_list": ["category1", "category2"],
            "label_name": "target",
            "is_binary": True,
            "num_classes": 2,
            "eta": 0.1,
            "max_depth": 6,
            "num_round": 100,
            "early_stopping_rounds": 10,
            "smooth_factor": 0.0,
            "count_threshold": 0
        }
        
        # Create sample data
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            "feature1": np.random.normal(0, 1, 100),
            "feature2": np.random.normal(1, 2, 100),
            "feature3": np.random.normal(-1, 0.5, 100),
            "category1": np.random.choice(["A", "B", "C"], 100),
            "category2": np.random.choice(["X", "Y"], 100),
            "target": np.random.choice([0, 1], 100),
            "id": range(100)
        })

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_load_and_validate_config_valid(self):
        """Test loading and validating a valid configuration file."""
        config_path = self.temp_dir / "hyperparameters.json"
        with open(config_path, "w") as f:
            json.dump(self.sample_config, f)
        
        result = load_and_validate_config(str(config_path))
        
        self.assertEqual(result, self.sample_config)
        self.assertEqual(result["tab_field_list"], ["feature1", "feature2", "feature3"])
        self.assertEqual(result["num_classes"], 2)

    def test_load_and_validate_config_missing_keys(self):
        """Test validation fails with missing required keys."""
        incomplete_config = {"tab_field_list": ["feature1"]}
        config_path = self.temp_dir / "hyperparameters.json"
        with open(config_path, "w") as f:
            json.dump(incomplete_config, f)
        
        with self.assertRaises(ValueError) as context:
            load_and_validate_config(str(config_path))
        
        self.assertIn("Missing required key in config", str(context.exception))

    def test_load_and_validate_config_invalid_class_weights(self):
        """Test validation fails with mismatched class weights."""
        invalid_config = self.sample_config.copy()
        invalid_config["class_weights"] = [0.3, 0.7, 0.5]  # 3 weights for 2 classes
        
        config_path = self.temp_dir / "hyperparameters.json"
        with open(config_path, "w") as f:
            json.dump(invalid_config, f)
        
        with self.assertRaises(ValueError) as context:
            load_and_validate_config(str(config_path))
        
        self.assertIn("Number of class weights", str(context.exception))

    def test_load_and_validate_config_file_not_found(self):
        """Test error handling when config file doesn't exist."""
        nonexistent_path = self.temp_dir / "nonexistent.json"
        
        with self.assertRaises(FileNotFoundError):
            load_and_validate_config(str(nonexistent_path))

    def test_find_first_data_file_csv(self):
        """Test finding CSV data file."""
        data_dir = self.temp_dir / "data"
        data_dir.mkdir()
        
        # Create test files
        (data_dir / "data.csv").write_text("test,data\n1,2")
        (data_dir / "other.txt").write_text("not data")
        
        result = find_first_data_file(str(data_dir))
        
        self.assertEqual(result, str(data_dir / "data.csv"))

    def test_find_first_data_file_parquet(self):
        """Test finding Parquet data file."""
        data_dir = self.temp_dir / "data"
        data_dir.mkdir()
        
        # Create test files (parquet comes first alphabetically)
        (data_dir / "data.parquet").write_text("parquet data")
        (data_dir / "zdata.csv").write_text("csv data")
        
        result = find_first_data_file(str(data_dir))
        
        self.assertEqual(result, str(data_dir / "data.parquet"))

    def test_find_first_data_file_no_data_files(self):
        """Test behavior when no data files are found."""
        data_dir = self.temp_dir / "data"
        data_dir.mkdir()
        
        # Create non-data files
        (data_dir / "readme.txt").write_text("not data")
        (data_dir / "config.yaml").write_text("not data")
        
        result = find_first_data_file(str(data_dir))
        
        self.assertIsNone(result)

    def test_find_first_data_file_nonexistent_dir(self):
        """Test behavior when directory doesn't exist."""
        nonexistent_dir = self.temp_dir / "nonexistent"
        
        result = find_first_data_file(str(nonexistent_dir))
        
        self.assertIsNone(result)

    @patch('src.cursus.steps.scripts.xgboost_train.pd.read_csv')
    @patch('src.cursus.steps.scripts.xgboost_train.find_first_data_file')
    def test_load_datasets_success(self, mock_find_file, mock_read_csv):
        """Test successful dataset loading."""
        # Mock file finding
        mock_find_file.side_effect = [
            "/path/train.csv",
            "/path/val.csv", 
            "/path/test.csv"
        ]
        
        # Mock dataframe reading
        train_df = pd.DataFrame({"col1": [1, 2], "target": [0, 1]})
        val_df = pd.DataFrame({"col1": [3, 4], "target": [1, 0]})
        test_df = pd.DataFrame({"col1": [5, 6], "target": [0, 1]})
        
        mock_read_csv.side_effect = [train_df, val_df, test_df]
        
        result_train, result_val, result_test = load_datasets("/input/path")
        
        # Verify results
        pd.testing.assert_frame_equal(result_train, train_df)
        pd.testing.assert_frame_equal(result_val, val_df)
        pd.testing.assert_frame_equal(result_test, test_df)
        
        # Verify calls
        self.assertEqual(mock_find_file.call_count, 3)
        self.assertEqual(mock_read_csv.call_count, 3)

    @patch('src.cursus.steps.scripts.xgboost_train.find_first_data_file')
    def test_load_datasets_missing_files(self, mock_find_file):
        """Test error handling when dataset files are missing."""
        # Mock missing validation file
        mock_find_file.side_effect = [
            "/path/train.csv",
            None,  # Missing validation file
            "/path/test.csv"
        ]
        
        with self.assertRaises(FileNotFoundError) as context:
            load_datasets("/input/path")
        
        self.assertIn("Training, validation, or test data file not found", str(context.exception))

    @patch('src.cursus.steps.scripts.xgboost_train.NumericalVariableImputationProcessor')
    def test_apply_numerical_imputation(self, mock_imputer_class):
        """Test numerical imputation application."""
        # Create mock imputer
        mock_imputer = MagicMock()
        mock_imputer.transform.side_effect = lambda df: df  # Return unchanged for simplicity
        mock_imputer.get_params.return_value = {"imputation_dict": {"feature1": 0.5}}
        mock_imputer_class.return_value = mock_imputer
        
        # Test data
        train_df = self.sample_data.copy()
        val_df = self.sample_data.copy()
        test_df = self.sample_data.copy()
        
        result = apply_numerical_imputation(
            self.sample_config, train_df, val_df, test_df
        )
        
        train_result, val_result, test_result, impute_dict = result
        
        # Verify imputer was created correctly
        mock_imputer_class.assert_called_once_with(
            variables=self.sample_config['tab_field_list'], 
            strategy='mean'
        )
        
        # Verify imputer methods were called
        mock_imputer.fit.assert_called_once()
        self.assertEqual(mock_imputer.transform.call_count, 3)
        
        # Verify imputation dictionary
        self.assertEqual(impute_dict, {"feature1": 0.5})

    @patch('src.cursus.steps.scripts.xgboost_train.RiskTableMappingProcessor')
    def test_fit_and_apply_risk_tables(self, mock_processor_class):
        """Test risk table fitting and application."""
        # Create mock processor
        mock_processor = MagicMock()
        mock_processor.transform.side_effect = lambda series: series.map({
            "A": 0.1, "B": 0.2, "C": 0.3, "X": 0.4, "Y": 0.5
        }).fillna(0.0)
        mock_processor.get_risk_tables.return_value = {"A": 0.1, "B": 0.2, "C": 0.3}
        mock_processor_class.return_value = mock_processor
        
        # Test data
        train_df = self.sample_data.copy()
        val_df = self.sample_data.copy()
        test_df = self.sample_data.copy()
        
        result = fit_and_apply_risk_tables(
            self.sample_config, train_df, val_df, test_df
        )
        
        train_result, val_result, test_result, risk_tables = result
        
        # Verify processors were created for each categorical variable
        self.assertEqual(mock_processor_class.call_count, 2)  # 2 categorical variables
        
        # Verify risk tables structure
        self.assertIn("category1", risk_tables)
        self.assertIn("category2", risk_tables)

    @patch('src.cursus.steps.scripts.xgboost_train.xgb.DMatrix')
    def test_prepare_dmatrices(self, mock_dmatrix):
        """Test DMatrix preparation."""
        # Create mock DMatrix instances
        mock_dtrain = MagicMock()
        mock_dval = MagicMock()
        mock_dmatrix.side_effect = [mock_dtrain, mock_dval]
        
        # Prepare test data with numerical values for categorical columns
        # (simulating the output after risk table mapping)
        train_df = self.sample_data.copy()
        val_df = self.sample_data.copy()
        
        # Convert categorical columns to numerical (as would happen after risk table mapping)
        train_df["category1"] = train_df["category1"].map({"A": 0.1, "B": 0.2, "C": 0.3})
        train_df["category2"] = train_df["category2"].map({"X": 0.4, "Y": 0.5})
        val_df["category1"] = val_df["category1"].map({"A": 0.1, "B": 0.2, "C": 0.3})
        val_df["category2"] = val_df["category2"].map({"X": 0.4, "Y": 0.5})
        
        result = prepare_dmatrices(self.sample_config, train_df, val_df)
        dtrain, dval, feature_columns = result
        
        # Verify feature columns
        expected_features = ["feature1", "feature2", "feature3", "category1", "category2"]
        self.assertEqual(feature_columns, expected_features)
        
        # Verify DMatrix creation
        self.assertEqual(mock_dmatrix.call_count, 2)
        
        # Verify feature names were set
        mock_dtrain.feature_names = expected_features
        mock_dval.feature_names = expected_features

    def test_prepare_dmatrices_nan_values(self):
        """Test DMatrix preparation fails with NaN values."""
        # Create data with NaN values
        train_df = self.sample_data.copy()
        val_df = self.sample_data.copy()
        
        # Convert categorical columns to numerical first (as would happen after risk table mapping)
        train_df["category1"] = train_df["category1"].map({"A": 0.1, "B": 0.2, "C": 0.3})
        train_df["category2"] = train_df["category2"].map({"X": 0.4, "Y": 0.5})
        val_df["category1"] = val_df["category1"].map({"A": 0.1, "B": 0.2, "C": 0.3})
        val_df["category2"] = val_df["category2"].map({"X": 0.4, "Y": 0.5})
        
        # Now add NaN values
        train_df.loc[0, "feature1"] = np.nan
        
        with self.assertRaises(ValueError) as context:
            prepare_dmatrices(self.sample_config, train_df, val_df)
        
        self.assertIn("Training data contains NaN or inf values", str(context.exception))

    @patch('src.cursus.steps.scripts.xgboost_train.xgb.train')
    def test_train_model_binary(self, mock_xgb_train):
        """Test binary classification model training."""
        # Create mock DMatrix objects
        mock_dtrain = MagicMock()
        mock_dval = MagicMock()
        mock_dtrain.get_label.return_value = np.array([0, 1, 0, 1])
        mock_dval.get_label.return_value = np.array([1, 0, 1, 0])
        
        # Mock trained model
        mock_model = MagicMock()
        mock_xgb_train.return_value = mock_model
        
        result = train_model(self.sample_config, mock_dtrain, mock_dval)
        
        # Verify xgb.train was called
        mock_xgb_train.assert_called_once()
        
        # Verify parameters
        call_args = mock_xgb_train.call_args
        params = call_args[1]['params']
        self.assertEqual(params['objective'], 'binary:logistic')
        self.assertEqual(params['eta'], 0.1)
        self.assertEqual(params['max_depth'], 6)
        
        # Verify result
        self.assertEqual(result, mock_model)

    @patch('src.cursus.steps.scripts.xgboost_train.xgb.train')
    def test_train_model_multiclass(self, mock_xgb_train):
        """Test multiclass model training."""
        # Update config for multiclass
        multiclass_config = self.sample_config.copy()
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
        params = call_args[1]['params']
        self.assertEqual(params['objective'], 'multi:softprob')
        self.assertEqual(params['num_class'], 3)

    @patch('src.cursus.steps.scripts.xgboost_train.json.dump')
    @patch('src.cursus.steps.scripts.xgboost_train.pkl.dump')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.makedirs')
    def test_save_artifacts(self, mock_makedirs, mock_file_open, mock_pkl_dump, mock_json_dump):
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
            mock_model, risk_tables, impute_dict, model_path, 
            feature_columns, self.sample_config
        )
        
        # Verify directory creation
        mock_makedirs.assert_called_once_with(model_path, exist_ok=True)
        
        # Verify model saving
        mock_model.save_model.assert_called_once()
        
        # Verify pickle dumps (risk tables and imputation dict)
        self.assertEqual(mock_pkl_dump.call_count, 2)
        
        # Verify JSON dumps (feature importance and hyperparameters)
        self.assertEqual(mock_json_dump.call_count, 2)


class TestXGBoostTrainMain(unittest.TestCase):
    """Tests for the main function of the XGBoost training script."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create directory structure
        self.data_dir = self.temp_dir / "data"
        self.config_dir = self.temp_dir / "config"
        self.model_dir = self.temp_dir / "model"
        self.output_dir = self.temp_dir / "output"
        
        for dir_path in [self.data_dir, self.config_dir, self.model_dir, self.output_dir]:
            dir_path.mkdir(parents=True)
        
        # Create subdirectories for train/val/test
        for split in ["train", "val", "test"]:
            (self.data_dir / split).mkdir()
        
        # Create sample configuration
        self.sample_config = {
            "tab_field_list": ["feature1", "feature2"],
            "cat_field_list": ["category1"],
            "label_name": "target",
            "is_binary": True,
            "num_classes": 2,
            "eta": 0.1,
            "max_depth": 3,
            "num_round": 10,
            "early_stopping_rounds": 5
        }
        
        # Save configuration
        config_path = self.config_dir / "hyperparameters.json"
        with open(config_path, "w") as f:
            json.dump(self.sample_config, f)
        
        # Create sample datasets
        np.random.seed(42)
        for split in ["train", "val", "test"]:
            data = pd.DataFrame({
                "feature1": np.random.normal(0, 1, 50),
                "feature2": np.random.normal(1, 2, 50),
                "category1": np.random.choice(["A", "B"], 50),
                "target": np.random.choice([0, 1], 50),
                "id": range(50)
            })
            data.to_csv(self.data_dir / split / "data.csv", index=False)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    @patch('src.cursus.steps.scripts.xgboost_train.xgb.train')
    @patch('src.cursus.steps.scripts.xgboost_train.RiskTableMappingProcessor')
    @patch('src.cursus.steps.scripts.xgboost_train.NumericalVariableImputationProcessor')
    def test_main_success(self, mock_imputer_class, mock_processor_class, mock_xgb_train):
        """Test successful main function execution."""
        # Setup mocks
        mock_imputer = MagicMock()
        mock_imputer.transform.side_effect = lambda df: df
        mock_imputer.get_params.return_value = {"imputation_dict": {}}
        mock_imputer_class.return_value = mock_imputer
        
        mock_processor = MagicMock()
        mock_processor.transform.side_effect = lambda series: series.map({"A": 0.1, "B": 0.2}).fillna(0.0)
        mock_processor.get_risk_tables.return_value = {"A": 0.1, "B": 0.2}
        mock_processor_class.return_value = mock_processor
        
        mock_model = MagicMock()
        mock_model.get_fscore.return_value = {"feature1": 10}
        mock_model.predict.return_value = np.array([0.3, 0.7, 0.4, 0.6])
        mock_xgb_train.return_value = mock_model
        
        # Prepare input parameters
        input_paths = {
            "data_dir": str(self.data_dir),
            "config_dir": str(self.config_dir)
        }
        output_paths = {
            "model_dir": str(self.model_dir),
            "output_dir": str(self.output_dir)
        }
        environ_vars = {}
        args = argparse.Namespace()
        
        # Execute main function
        try:
            main(input_paths, output_paths, environ_vars, args)
            success = True
        except Exception as e:
            success = False
            error = str(e)
        
        # Verify success
        self.assertTrue(success, f"Main function failed with error: {error if not success else 'None'}")
        
        # Verify model training was called
        mock_xgb_train.assert_called_once()
        
        # Verify model artifacts were saved
        self.assertTrue((Path(self.model_dir) / "xgboost_model.bst").exists() or 
                       mock_model.save_model.called)

    def test_main_missing_config(self):
        """Test main function with missing configuration file."""
        # Remove config file
        (self.config_dir / "hyperparameters.json").unlink()
        
        input_paths = {
            "data_dir": str(self.data_dir),
            "config_dir": str(self.config_dir)
        }
        output_paths = {
            "model_dir": str(self.model_dir),
            "output_dir": str(self.output_dir)
        }
        environ_vars = {}
        args = argparse.Namespace()
        
        with self.assertRaises(FileNotFoundError):
            main(input_paths, output_paths, environ_vars, args)

    def test_main_missing_data(self):
        """Test main function with missing data files."""
        # Remove training data
        shutil.rmtree(self.data_dir / "train")
        
        input_paths = {
            "data_dir": str(self.data_dir),
            "config_dir": str(self.config_dir)
        }
        output_paths = {
            "model_dir": str(self.model_dir),
            "output_dir": str(self.output_dir)
        }
        environ_vars = {}
        args = argparse.Namespace()
        
        with self.assertRaises(FileNotFoundError):
            main(input_paths, output_paths, environ_vars, args)

    def test_main_invalid_config(self):
        """Test main function with invalid configuration."""
        # Create invalid config (missing required keys)
        invalid_config = {"tab_field_list": ["feature1"]}
        config_path = self.config_dir / "hyperparameters.json"
        with open(config_path, "w") as f:
            json.dump(invalid_config, f)
        
        input_paths = {
            "data_dir": str(self.data_dir),
            "config_dir": str(self.config_dir)
        }
        output_paths = {
            "model_dir": str(self.model_dir),
            "output_dir": str(self.output_dir)
        }
        environ_vars = {}
        args = argparse.Namespace()
        
        with self.assertRaises(ValueError):
            main(input_paths, output_paths, environ_vars, args)


class TestXGBoostTrainIntegration(unittest.TestCase):
    """Integration tests for the XGBoost training script."""

    def setUp(self):
        """Set up test fixtures for integration tests."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create directory structure
        self.data_dir = self.temp_dir / "data"
        self.config_dir = self.temp_dir / "config"
        self.model_dir = self.temp_dir / "model"
        self.output_dir = self.temp_dir / "output"
        
        for dir_path in [self.data_dir, self.config_dir, self.model_dir, self.output_dir]:
            dir_path.mkdir(parents=True)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_script_imports_successfully(self):
        """Test that the script can be imported without errors."""
        try:
            import src.cursus.steps.scripts.xgboost_training
            success = True
        except Exception as e:
            success = False
            error = str(e)
        
        self.assertTrue(success, f"Script import failed: {error if not success else 'None'}")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

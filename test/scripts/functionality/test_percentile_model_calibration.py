import pytest
from unittest.mock import patch, MagicMock, mock_open
import os
import tempfile
import shutil
import json
from pathlib import Path
import numpy as np
import pandas as pd
import argparse

# Import the functions to be tested
from cursus.steps.scripts.percentile_model_calibration import (
    get_calibrated_score_map,
    find_first_data_file,
    load_calibration_dictionary,
    main,
)


class TestGetCalibratedScoreMap:
    """Tests for the get_calibrated_score_map function."""

    @pytest.fixture
    def setup_test_data(self):
        """Set up test data for calibration."""
        np.random.seed(42)
        n_samples = 100
        
        # Create test dataframe with scores
        df = pd.DataFrame({
            "prob_class_1": np.random.uniform(0, 1, n_samples),
            "weight": np.random.uniform(0.5, 1.5, n_samples)
        })
        
        # Simple calibration dictionary for testing
        calibration_dict = {
            0.1: 0.9,
            0.2: 0.8,
            0.3: 0.7,
            0.4: 0.6,
            0.5: 0.5,
            0.6: 0.4,
            0.7: 0.3,
            0.8: 0.2,
            0.9: 0.1
        }
        
        return df, calibration_dict

    def test_get_calibrated_score_map_basic(self, setup_test_data):
        """Test basic functionality of get_calibrated_score_map."""
        df, calibration_dict = setup_test_data
        
        score_map = get_calibrated_score_map(
            df=df,
            score_field="prob_class_1",
            calibration_dictionary=calibration_dict,
            weight_field=None
        )
        
        # Check that we get a list of tuples
        assert isinstance(score_map, list)
        assert len(score_map) > 0
        
        # Check that all entries are tuples with two elements
        for entry in score_map:
            assert isinstance(entry, tuple)
            assert len(entry) == 2
            assert isinstance(entry[0], (int, float))
            assert isinstance(entry[1], (int, float))
        
        # Check boundary conditions
        assert score_map[0] == (0.0, 0.0)
        assert score_map[-1] == (1.0, 1.0)
        
        # Check that scores are sorted
        scores = [entry[0] for entry in score_map]
        assert scores == sorted(scores)

    def test_get_calibrated_score_map_with_weights(self, setup_test_data):
        """Test get_calibrated_score_map with weight field."""
        df, calibration_dict = setup_test_data
        
        score_map = get_calibrated_score_map(
            df=df,
            score_field="prob_class_1",
            calibration_dictionary=calibration_dict,
            weight_field="weight"
        )
        
        # Should still return valid score map
        assert isinstance(score_map, list)
        assert len(score_map) > 0
        assert score_map[0] == (0.0, 0.0)
        assert score_map[-1] == (1.0, 1.0)

    def test_get_calibrated_score_map_empty_calibration_dict(self, setup_test_data):
        """Test get_calibrated_score_map with empty calibration dictionary."""
        df, _ = setup_test_data
        
        score_map = get_calibrated_score_map(
            df=df,
            score_field="prob_class_1",
            calibration_dictionary={},
            weight_field=None
        )
        
        # Should only have boundary points
        assert len(score_map) == 2
        assert score_map[0] == (0.0, 0.0)
        assert score_map[1] == (1.0, 1.0)

    def test_get_calibrated_score_map_missing_score_field(self, setup_test_data):
        """Test get_calibrated_score_map with missing score field."""
        df, calibration_dict = setup_test_data
        
        with pytest.raises(KeyError):
            get_calibrated_score_map(
                df=df,
                score_field="nonexistent_field",
                calibration_dictionary=calibration_dict,
                weight_field=None
            )

    def test_get_calibrated_score_map_missing_weight_field(self, setup_test_data):
        """Test get_calibrated_score_map with missing weight field."""
        df, calibration_dict = setup_test_data
        
        with pytest.raises(KeyError):
            get_calibrated_score_map(
                df=df,
                score_field="prob_class_1",
                calibration_dictionary=calibration_dict,
                weight_field="nonexistent_weight"
            )


class TestFindFirstDataFile:
    """Tests for the find_first_data_file function."""

    @pytest.fixture
    def setup_temp_dir(self):
        """Set up temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_find_first_data_file_csv(self, setup_temp_dir):
        """Test finding CSV data file."""
        temp_dir = setup_temp_dir
        
        # Create test files
        (temp_dir / "data.csv").write_text("col1,col2\n1,2\n")
        (temp_dir / "other.txt").write_text("not a data file")
        
        result = find_first_data_file(str(temp_dir))
        assert result.endswith("data.csv")
        assert str(temp_dir) in result

    def test_find_first_data_file_parquet(self, setup_temp_dir):
        """Test finding Parquet data file."""
        temp_dir = setup_temp_dir
        
        # Create a dummy parquet file (just touch it for the test)
        (temp_dir / "data.parquet").touch()
        (temp_dir / "other.txt").write_text("not a data file")
        
        result = find_first_data_file(str(temp_dir))
        assert result.endswith("data.parquet")

    def test_find_first_data_file_json(self, setup_temp_dir):
        """Test finding JSON data file."""
        temp_dir = setup_temp_dir
        
        # Create a dummy JSON file
        (temp_dir / "data.json").write_text('{"key": "value"}')
        (temp_dir / "other.txt").write_text("not a data file")
        
        result = find_first_data_file(str(temp_dir))
        assert result.endswith("data.json")

    def test_find_first_data_file_sorted_order(self, setup_temp_dir):
        """Test that files are found in sorted order."""
        temp_dir = setup_temp_dir
        
        # Create multiple data files
        (temp_dir / "z_data.csv").write_text("col1,col2\n1,2\n")
        (temp_dir / "a_data.csv").write_text("col1,col2\n3,4\n")
        
        result = find_first_data_file(str(temp_dir))
        assert result.endswith("a_data.csv")

    def test_find_first_data_file_no_files(self, setup_temp_dir):
        """Test finding data file when none exist."""
        temp_dir = setup_temp_dir
        
        # Create non-data files
        (temp_dir / "readme.txt").write_text("not a data file")
        
        with pytest.raises(FileNotFoundError) as exc_info:
            find_first_data_file(str(temp_dir))
        
        assert "No supported data file" in str(exc_info.value)

    def test_find_first_data_file_no_directory(self):
        """Test finding data file when directory doesn't exist."""
        with pytest.raises(FileNotFoundError) as exc_info:
            find_first_data_file("/nonexistent/directory")
        
        assert "Directory does not exist" in str(exc_info.value)


class TestLoadCalibrationDictionary:
    """Tests for the load_calibration_dictionary function."""

    @pytest.fixture
    def setup_temp_dir(self):
        """Set up temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_load_calibration_dictionary_external_config(self, setup_temp_dir):
        """Test loading calibration dictionary from external config."""
        temp_dir = setup_temp_dir
        config_dir = temp_dir / "calibration"
        config_dir.mkdir()
        
        # Create external calibration dictionary
        external_dict = {"0.1": 0.9, "0.2": 0.8, "0.3": 0.7}
        config_file = config_dir / "standard_calibration_dictionary.json"
        with open(config_file, 'w') as f:
            json.dump(external_dict, f)
        
        input_paths = {"calibration_config": str(config_dir)}
        
        result = load_calibration_dictionary(input_paths)
        
        # Should load external dictionary with float keys
        expected = {0.1: 0.9, 0.2: 0.8, 0.3: 0.7}
        assert result == expected

    def test_load_calibration_dictionary_no_config_path(self, setup_temp_dir):
        """Test loading calibration dictionary with no config path."""
        input_paths = {}
        
        result = load_calibration_dictionary(input_paths)
        
        # Should return default dictionary
        assert isinstance(result, dict)
        assert len(result) > 0
        assert 0.001 in result  # Check for a known default key

    def test_load_calibration_dictionary_nonexistent_config_path(self, setup_temp_dir):
        """Test loading calibration dictionary with nonexistent config path."""
        input_paths = {"calibration_config": "/nonexistent/path"}
        
        result = load_calibration_dictionary(input_paths)
        
        # Should fallback to default dictionary
        assert isinstance(result, dict)
        assert len(result) > 0
        assert 0.001 in result

    def test_load_calibration_dictionary_missing_config_file(self, setup_temp_dir):
        """Test loading calibration dictionary with missing config file."""
        temp_dir = setup_temp_dir
        config_dir = temp_dir / "calibration"
        config_dir.mkdir()
        
        input_paths = {"calibration_config": str(config_dir)}
        
        result = load_calibration_dictionary(input_paths)
        
        # Should fallback to default dictionary
        assert isinstance(result, dict)
        assert len(result) > 0
        assert 0.001 in result

    def test_load_calibration_dictionary_invalid_json(self, setup_temp_dir):
        """Test loading calibration dictionary with invalid JSON."""
        temp_dir = setup_temp_dir
        config_dir = temp_dir / "calibration"
        config_dir.mkdir()
        
        # Create invalid JSON file
        config_file = config_dir / "standard_calibration_dictionary.json"
        config_file.write_text("invalid json content")
        
        input_paths = {"calibration_config": str(config_dir)}
        
        result = load_calibration_dictionary(input_paths)
        
        # Should fallback to default dictionary
        assert isinstance(result, dict)
        assert len(result) > 0
        assert 0.001 in result


class TestPercentileModelCalibrationMain:
    """Tests for the main function."""

    @pytest.fixture
    def setup_main_test(self):
        """Set up test fixtures for main function."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create test data
        np.random.seed(42)
        n_samples = 100
        df = pd.DataFrame({
            "prob_class_1": np.random.uniform(0, 1, n_samples),
            "other_score": np.random.uniform(0, 1, n_samples),
        })
        
        # Create input data file
        input_dir = temp_dir / "input"
        input_dir.mkdir(parents=True)
        df.to_csv(input_dir / "processed_data.csv", index=False)
        
        # Set up paths
        input_paths = {
            "evaluation_data": str(input_dir),
            "calibration_config": str(temp_dir / "calibration")
        }
        output_paths = {
            "calibration_output": str(temp_dir / "calibration"),
            "metrics_output": str(temp_dir / "metrics"),
            "calibrated_data": str(temp_dir / "calibrated_data")
        }
        environ_vars = {
            "N_BINS": "1000",
            "SCORE_FIELD": "prob_class_1",
            "ACCURACY": "1e-5"
        }
        
        yield temp_dir, input_paths, output_paths, environ_vars, df, n_samples
        shutil.rmtree(temp_dir)

    def test_main_successful_calibration(self, setup_main_test):
        """Test successful main function execution."""
        temp_dir, input_paths, output_paths, environ_vars, df, n_samples = setup_main_test
        
        # Create mock job args
        job_args = argparse.Namespace()
        job_args.job_type = "calibration"
        
        # Run main function
        result = main(input_paths, output_paths, environ_vars, job_args)
        
        # Check return value
        assert result["status"] == "success"
        assert result["calibration_method"] == "percentile_score_mapping"
        assert result["num_input_scores"] == n_samples
        assert "output_files" in result
        assert "config" in result
        
        # Check that output files were created
        calibration_dir = Path(output_paths["calibration_output"])
        metrics_dir = Path(output_paths["metrics_output"])
        calibrated_dir = Path(output_paths["calibrated_data"])
        
        assert calibration_dir.exists()
        assert metrics_dir.exists()
        assert calibrated_dir.exists()
        
        # Check specific output files
        assert (calibration_dir / "percentile_score.pkl").exists()
        assert (metrics_dir / "calibration_metrics.json").exists()
        assert (calibrated_dir / "calibrated_data.csv").exists()
        
        # Verify metrics file content
        with open(metrics_dir / "calibration_metrics.json", "r") as f:
            metrics = json.load(f)
        
        assert metrics["calibration_method"] == "percentile_score_mapping"
        assert metrics["num_input_scores"] == n_samples
        assert "score_statistics" in metrics
        assert "calibration_range" in metrics
        assert "config" in metrics

    def test_main_with_fallback_data_file(self, setup_main_test):
        """Test main function with fallback to first data file."""
        temp_dir, input_paths, output_paths, environ_vars, df, n_samples = setup_main_test
        
        # Remove the processed_data.csv file and create a different CSV
        input_dir = Path(input_paths["evaluation_data"])
        (input_dir / "processed_data.csv").unlink()
        df.to_csv(input_dir / "fallback_data.csv", index=False)
        
        # Create mock job args
        job_args = argparse.Namespace()
        job_args.job_type = "calibration"
        
        # Run main function
        result = main(input_paths, output_paths, environ_vars, job_args)
        
        # Should still succeed
        assert result["status"] == "success"
        assert result["num_input_scores"] == n_samples

    def test_main_missing_score_field(self, setup_main_test):
        """Test main function with missing score field."""
        temp_dir, input_paths, output_paths, environ_vars, df, n_samples = setup_main_test
        
        # Create data without the required score field
        input_dir = Path(input_paths["evaluation_data"])
        (input_dir / "processed_data.csv").unlink()
        
        df_no_score = pd.DataFrame({
            "other_field": np.random.uniform(0, 1, n_samples)
        })
        df_no_score.to_csv(input_dir / "data.csv", index=False)
        
        # Create mock job args
        job_args = argparse.Namespace()
        job_args.job_type = "calibration"
        
        # Run main function
        result = main(input_paths, output_paths, environ_vars, job_args)
        
        # Should fail with error
        assert result["status"] == "error"
        assert "Score field 'prob_class_1' not found" in result["error_message"]

    def test_main_missing_evaluation_data_path(self, setup_main_test):
        """Test main function with missing evaluation data path."""
        temp_dir, input_paths, output_paths, environ_vars, df, n_samples = setup_main_test
        
        # Remove evaluation_data from input_paths
        input_paths_missing = {k: v for k, v in input_paths.items() if k != "evaluation_data"}
        
        # Create mock job args
        job_args = argparse.Namespace()
        job_args.job_type = "calibration"
        
        # Run main function
        result = main(input_paths_missing, output_paths, environ_vars, job_args)
        
        # Should fail with error
        assert result["status"] == "error"
        assert "evaluation_data path not provided" in result["error_message"]

    def test_main_missing_output_paths(self, setup_main_test):
        """Test main function with missing output paths."""
        temp_dir, input_paths, output_paths, environ_vars, df, n_samples = setup_main_test
        
        # Remove required output path
        output_paths_missing = {k: v for k, v in output_paths.items() if k != "calibration_output"}
        
        # Create mock job args
        job_args = argparse.Namespace()
        job_args.job_type = "calibration"
        
        # Run main function
        result = main(input_paths, output_paths_missing, environ_vars, job_args)
        
        # Should fail with error
        assert result["status"] == "error"
        assert "calibration_output path not provided" in result["error_message"]

    def test_main_with_custom_score_field(self, setup_main_test):
        """Test main function with custom score field."""
        temp_dir, input_paths, output_paths, environ_vars, df, n_samples = setup_main_test
        
        # Update environment variables to use different score field
        environ_vars["SCORE_FIELD"] = "other_score"
        
        # Create mock job args
        job_args = argparse.Namespace()
        job_args.job_type = "calibration"
        
        # Run main function
        result = main(input_paths, output_paths, environ_vars, job_args)
        
        # Should succeed
        assert result["status"] == "success"
        assert result["config"]["score_field"] == "other_score"

    def test_main_with_external_calibration_config(self, setup_main_test):
        """Test main function with external calibration config."""
        temp_dir, input_paths, output_paths, environ_vars, df, n_samples = setup_main_test
        
        # Create external calibration config
        config_dir = Path(input_paths["calibration_config"])
        config_dir.mkdir(parents=True)
        
        external_dict = {"0.1": 0.9, "0.5": 0.5, "0.9": 0.1}
        config_file = config_dir / "standard_calibration_dictionary.json"
        with open(config_file, 'w') as f:
            json.dump(external_dict, f)
        
        # Create mock job args
        job_args = argparse.Namespace()
        job_args.job_type = "calibration"
        
        # Run main function
        result = main(input_paths, output_paths, environ_vars, job_args)
        
        # Should succeed and use external config
        assert result["status"] == "success"
        assert result["config"]["calibration_dict_size"] == len(external_dict)

    def test_main_no_job_args(self, setup_main_test):
        """Test main function without job args."""
        temp_dir, input_paths, output_paths, environ_vars, df, n_samples = setup_main_test
        
        # Run main function without job_args
        result = main(input_paths, output_paths, environ_vars, None)
        
        # Should succeed with default job_type
        assert result["status"] == "success"
        assert result["config"]["job_type"] == "calibration"

    def test_main_unsupported_file_format(self, setup_main_test):
        """Test main function with unsupported file format."""
        temp_dir, input_paths, output_paths, environ_vars, df, n_samples = setup_main_test
        
        # Remove CSV file and create unsupported format
        input_dir = Path(input_paths["evaluation_data"])
        (input_dir / "processed_data.csv").unlink()
        (input_dir / "data.xlsx").write_text("fake excel file")
        
        # Create mock job args
        job_args = argparse.Namespace()
        job_args.job_type = "calibration"
        
        # Run main function
        result = main(input_paths, output_paths, environ_vars, job_args)
        
        # Should fail with error
        assert result["status"] == "error"
        assert "No supported data file" in result["error_message"]


class TestPercentileModelCalibrationIntegration:
    """Integration tests for the percentile model calibration."""

    @pytest.fixture
    def setup_integration_test(self):
        """Set up integration test with realistic data."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create realistic test data with known distribution
        np.random.seed(42)
        n_samples = 1000
        
        # Create scores with some structure
        scores_low = np.random.beta(2, 8, n_samples // 2)  # Skewed towards low values
        scores_high = np.random.beta(8, 2, n_samples // 2)  # Skewed towards high values
        scores = np.concatenate([scores_low, scores_high])
        np.random.shuffle(scores)
        
        df = pd.DataFrame({
            "prob_class_1": scores,
            "id": range(n_samples),
        })
        
        # Create input data file
        input_dir = temp_dir / "input"
        input_dir.mkdir(parents=True)
        df.to_csv(input_dir / "processed_data.csv", index=False)
        
        # Set up paths
        input_paths = {
            "evaluation_data": str(input_dir),
            "calibration_config": str(temp_dir / "calibration")
        }
        output_paths = {
            "calibration_output": str(temp_dir / "calibration"),
            "metrics_output": str(temp_dir / "metrics"),
            "calibrated_data": str(temp_dir / "calibrated_data")
        }
        environ_vars = {
            "N_BINS": "100",
            "SCORE_FIELD": "prob_class_1",
            "ACCURACY": "1e-6"
        }
        
        yield temp_dir, input_paths, output_paths, environ_vars, df, n_samples
        shutil.rmtree(temp_dir)

    def test_integration_end_to_end(self, setup_integration_test):
        """Test complete end-to-end integration."""
        temp_dir, input_paths, output_paths, environ_vars, df, n_samples = setup_integration_test
        
        # Create mock job args
        job_args = argparse.Namespace()
        job_args.job_type = "calibration"
        
        # Run main function
        result = main(input_paths, output_paths, environ_vars, job_args)
        
        # Verify successful execution
        assert result["status"] == "success"
        assert result["num_input_scores"] == n_samples
        
        # Load and verify calibrated data
        calibrated_data_path = Path(output_paths["calibrated_data"]) / "calibrated_data.csv"
        df_calibrated = pd.read_csv(calibrated_data_path)
        
        # Check that calibrated data has expected structure
        assert len(df_calibrated) == n_samples
        assert "prob_class_1" in df_calibrated.columns
        assert "prob_class_1_percentile" in df_calibrated.columns
        assert "id" in df_calibrated.columns
        
        # Check that percentile scores are in valid range
        percentile_scores = df_calibrated["prob_class_1_percentile"]
        assert percentile_scores.min() >= 0.0
        assert percentile_scores.max() <= 1.0
        
        # Load and verify pickle file
        import pickle
        percentile_score_path = Path(output_paths["calibration_output"]) / "percentile_score.pkl"
        with open(percentile_score_path, "rb") as f:
            score_map = pickle.load(f)
        
        # Verify score map structure
        assert isinstance(score_map, list)
        assert len(score_map) > 2  # Should have more than just boundary points
        assert score_map[0] == (0.0, 0.0)
        assert score_map[-1] == (1.0, 1.0)
        
        # Verify metrics
        metrics_path = Path(output_paths["metrics_output"]) / "calibration_metrics.json"
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        
        # Check metrics structure and values
        assert metrics["calibration_method"] == "percentile_score_mapping"
        assert metrics["num_input_scores"] == n_samples
        assert metrics["num_calibration_points"] > 2
        
        # Check score statistics
        score_stats = metrics["score_statistics"]
        assert "min_score" in score_stats
        assert "max_score" in score_stats
        assert "mean_score" in score_stats
        assert "std_score" in score_stats
        assert 0.0 <= score_stats["min_score"] <= 1.0
        assert 0.0 <= score_stats["max_score"] <= 1.0
        assert score_stats["min_score"] <= score_stats["max_score"]
        
        # Check calibration range
        calib_range = metrics["calibration_range"]
        assert "min_percentile" in calib_range
        assert "max_percentile" in calib_range
        assert "min_score_threshold" in calib_range
        assert "max_score_threshold" in calib_range
        
        # Check config
        config = metrics["config"]
        assert config["n_bins"] == 100
        assert config["score_field"] == "prob_class_1"
        assert config["accuracy"] == 1e-6
        assert config["job_type"] == "calibration"
        assert config["calibration_dict_size"] > 0

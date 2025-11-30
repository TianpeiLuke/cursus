"""
Comprehensive test suite for percentile_model_calibration.py script.

This test suite follows pytest best practices and provides thorough coverage
of the percentile model calibration functionality including:
- File format detection and loading (CSV, TSV, Parquet)
- Format preservation during save operations
- Multi-task score field parsing and validation
- Calibration dictionary loading (external and built-in)
- Percentile score mapping computation
- Data file discovery and prioritization
- Calibrated data generation with percentile columns
- Metrics and report generation
- Exception handling and edge cases
"""

import pytest
from unittest.mock import patch, MagicMock, Mock, call, mock_open
import os
import sys
import tempfile
import shutil
import json
import pickle as pkl
from pathlib import Path
import numpy as np
import pandas as pd
import argparse

# Import the functions to be tested
from cursus.steps.scripts.percentile_model_calibration import (
    _detect_file_format,
    load_dataframe_with_format,
    save_dataframe_with_format,
    parse_score_fields,
    validate_score_fields,
    get_calibrated_score_map,
    find_first_data_file,
    load_calibration_dictionary,
    main,
)


class TestFileFormatDetection:
    """Tests for file format detection."""

    def test_detect_file_format_csv(self):
        """Test CSV format detection."""
        assert _detect_file_format("predictions.csv") == "csv"
        assert _detect_file_format("/path/to/predictions.csv") == "csv"
        assert _detect_file_format("DATA.CSV") == "csv"

    def test_detect_file_format_tsv(self):
        """Test TSV format detection."""
        assert _detect_file_format("predictions.tsv") == "tsv"
        assert _detect_file_format("/path/to/predictions.tsv") == "tsv"
        assert _detect_file_format("PREDICTIONS.TSV") == "tsv"

    def test_detect_file_format_parquet(self):
        """Test Parquet format detection."""
        assert _detect_file_format("predictions.parquet") == "parquet"
        assert _detect_file_format("/path/to/predictions.PARQUET") == "parquet"

    def test_detect_file_format_unsupported(self):
        """Test error with unsupported format."""
        with pytest.raises(RuntimeError, match="Unsupported file format"):
            _detect_file_format("predictions.json")

        with pytest.raises(RuntimeError, match="Unsupported file format"):
            _detect_file_format("predictions.txt")


class TestDataLoading:
    """Tests for data loading and saving with format preservation."""

    @pytest.fixture
    def setup_temp_dir(self):
        """Set up temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_calibration_data(self):
        """Create sample calibration dataframe."""
        np.random.seed(42)
        n_samples = 100
        return pd.DataFrame(
            {
                "id": range(n_samples),
                "label": np.random.randint(0, 2, n_samples),
                "prob_class_1": np.random.uniform(0.0, 1.0, n_samples),
                "raw_score": np.random.uniform(0.0, 1.0, n_samples),
            }
        )

    def test_load_dataframe_with_format_csv(
        self, setup_temp_dir, sample_calibration_data
    ):
        """Test loading dataframe from CSV with format detection."""
        temp_dir = setup_temp_dir
        csv_file = temp_dir / "data.csv"
        sample_calibration_data.to_csv(csv_file, index=False)

        df_loaded, detected_format = load_dataframe_with_format(str(csv_file))

        assert detected_format == "csv"
        pd.testing.assert_frame_equal(df_loaded, sample_calibration_data)

    def test_load_dataframe_with_format_tsv(
        self, setup_temp_dir, sample_calibration_data
    ):
        """Test loading dataframe from TSV with format detection."""
        temp_dir = setup_temp_dir
        tsv_file = temp_dir / "data.tsv"
        sample_calibration_data.to_csv(tsv_file, sep="\t", index=False)

        df_loaded, detected_format = load_dataframe_with_format(str(tsv_file))

        assert detected_format == "tsv"
        pd.testing.assert_frame_equal(df_loaded, sample_calibration_data)

    def test_load_dataframe_with_format_parquet(
        self, setup_temp_dir, sample_calibration_data
    ):
        """Test loading dataframe from Parquet with format detection."""
        temp_dir = setup_temp_dir
        parquet_file = temp_dir / "data.parquet"
        sample_calibration_data.to_parquet(parquet_file)

        df_loaded, detected_format = load_dataframe_with_format(str(parquet_file))

        assert detected_format == "parquet"
        pd.testing.assert_frame_equal(df_loaded, sample_calibration_data)

    def test_save_dataframe_with_format_csv(
        self, setup_temp_dir, sample_calibration_data
    ):
        """Test saving dataframe to CSV with format preservation."""
        temp_dir = setup_temp_dir
        output_path = temp_dir / "output"

        saved_path = save_dataframe_with_format(
            sample_calibration_data, str(output_path), "csv"
        )

        assert Path(saved_path).exists()
        assert saved_path.endswith(".csv")
        df_loaded = pd.read_csv(saved_path)
        pd.testing.assert_frame_equal(df_loaded, sample_calibration_data)

    def test_save_dataframe_with_format_tsv(
        self, setup_temp_dir, sample_calibration_data
    ):
        """Test saving dataframe to TSV with format preservation."""
        temp_dir = setup_temp_dir
        output_path = temp_dir / "output"

        saved_path = save_dataframe_with_format(
            sample_calibration_data, str(output_path), "tsv"
        )

        assert Path(saved_path).exists()
        assert saved_path.endswith(".tsv")
        df_loaded = pd.read_csv(saved_path, sep="\t")
        pd.testing.assert_frame_equal(df_loaded, sample_calibration_data)

    def test_save_dataframe_with_format_parquet(
        self, setup_temp_dir, sample_calibration_data
    ):
        """Test saving dataframe to Parquet with format preservation."""
        temp_dir = setup_temp_dir
        output_path = temp_dir / "output"

        saved_path = save_dataframe_with_format(
            sample_calibration_data, str(output_path), "parquet"
        )

        assert Path(saved_path).exists()
        assert saved_path.endswith(".parquet")
        df_loaded = pd.read_parquet(saved_path)
        pd.testing.assert_frame_equal(df_loaded, sample_calibration_data)

    def test_save_dataframe_with_format_unsupported(
        self, setup_temp_dir, sample_calibration_data
    ):
        """Test error with unsupported output format."""
        temp_dir = setup_temp_dir
        output_path = temp_dir / "output"

        with pytest.raises(RuntimeError, match="Unsupported output format"):
            save_dataframe_with_format(
                sample_calibration_data, str(output_path), "json"
            )


class TestScoreFieldParsing:
    """Tests for score field parsing functionality."""

    def test_parse_score_fields_multi_task(self):
        """Test parsing multiple score fields from SCORE_FIELDS."""
        environ_vars = {"SCORE_FIELDS": "task_0_prob,task_1_prob,task_2_prob"}

        score_fields = parse_score_fields(environ_vars)

        assert score_fields == ["task_0_prob", "task_1_prob", "task_2_prob"]
        assert len(score_fields) == 3

    def test_parse_score_fields_single_task_fallback(self):
        """Test fallback to SCORE_FIELD for single task."""
        environ_vars = {"SCORE_FIELD": "prob_class_1"}

        score_fields = parse_score_fields(environ_vars)

        assert score_fields == ["prob_class_1"]
        assert len(score_fields) == 1

    def test_parse_score_fields_default(self):
        """Test default score field when neither is provided."""
        environ_vars = {}

        score_fields = parse_score_fields(environ_vars)

        assert score_fields == ["prob_class_1"]

    def test_parse_score_fields_with_spaces(self):
        """Test parsing score fields with extra spaces."""
        environ_vars = {"SCORE_FIELDS": " task_0_prob , task_1_prob , task_2_prob "}

        score_fields = parse_score_fields(environ_vars)

        assert score_fields == ["task_0_prob", "task_1_prob", "task_2_prob"]

    def test_parse_score_fields_empty_string(self):
        """Test empty SCORE_FIELDS falls back to SCORE_FIELD."""
        environ_vars = {"SCORE_FIELDS": "", "SCORE_FIELD": "custom_score"}

        score_fields = parse_score_fields(environ_vars)

        assert score_fields == ["custom_score"]

    def test_validate_score_fields_all_valid(self):
        """Test validation when all score fields exist."""
        df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "task_0_prob": [0.1, 0.2, 0.3],
                "task_1_prob": [0.4, 0.5, 0.6],
            }
        )
        score_fields = ["task_0_prob", "task_1_prob"]

        valid_fields, missing_fields = validate_score_fields(df, score_fields)

        assert valid_fields == ["task_0_prob", "task_1_prob"]
        assert missing_fields == []

    def test_validate_score_fields_some_missing(self):
        """Test validation when some score fields are missing."""
        df = pd.DataFrame({"id": [1, 2, 3], "task_0_prob": [0.1, 0.2, 0.3]})
        score_fields = ["task_0_prob", "task_1_prob", "task_2_prob"]

        valid_fields, missing_fields = validate_score_fields(df, score_fields)

        assert valid_fields == ["task_0_prob"]
        assert set(missing_fields) == {"task_1_prob", "task_2_prob"}

    def test_validate_score_fields_all_missing(self):
        """Test validation when all score fields are missing."""
        df = pd.DataFrame({"id": [1, 2, 3], "label": [0, 1, 0]})
        score_fields = ["task_0_prob", "task_1_prob"]

        valid_fields, missing_fields = validate_score_fields(df, score_fields)

        assert valid_fields == []
        assert set(missing_fields) == {"task_0_prob", "task_1_prob"}


class TestCalibratedScoreMapping:
    """Tests for calibrated score map computation."""

    @pytest.fixture
    def sample_calibration_dict(self):
        """Create sample calibration dictionary."""
        return {
            0.001: 0.995,
            0.01: 0.95,
            0.1: 0.8,
            0.2: 0.7,
            0.3: 0.6,
            0.5: 0.5,
            0.7: 0.3,
            0.9: 0.1,
        }

    @pytest.fixture
    def sample_score_data(self):
        """Create sample score data for calibration."""
        np.random.seed(42)
        n_samples = 1000
        return pd.DataFrame({"raw_scores": np.random.uniform(0.0, 1.0, n_samples)})

    def test_get_calibrated_score_map_basic(
        self, sample_score_data, sample_calibration_dict
    ):
        """Test basic calibrated score map generation."""
        score_map = get_calibrated_score_map(
            df=sample_score_data,
            score_field="raw_scores",
            calibration_dictionary=sample_calibration_dict,
            weight_field=None,
        )

        # Check score map structure
        assert isinstance(score_map, list)
        assert len(score_map) > 0
        assert all(isinstance(item, tuple) and len(item) == 2 for item in score_map)

        # Check boundary points
        assert score_map[0] == (0.0, 0.0)
        assert score_map[-1] == (1.0, 1.0)

        # Check score map is sorted by score threshold
        thresholds = [item[0] for item in score_map]
        assert thresholds == sorted(thresholds)

    def test_get_calibrated_score_map_with_weights(
        self, sample_score_data, sample_calibration_dict
    ):
        """Test calibrated score map with sample weights."""
        # Add weight column
        np.random.seed(42)
        sample_score_data["weights"] = np.random.uniform(
            0.5, 2.0, len(sample_score_data)
        )

        score_map = get_calibrated_score_map(
            df=sample_score_data,
            score_field="raw_scores",
            calibration_dictionary=sample_calibration_dict,
            weight_field="weights",
        )

        # Check score map structure
        assert isinstance(score_map, list)
        assert len(score_map) > 0
        assert score_map[0] == (0.0, 0.0)
        assert score_map[-1] == (1.0, 1.0)

    def test_get_calibrated_score_map_extreme_scores(self, sample_calibration_dict):
        """Test calibration with extreme score distributions."""
        # All high scores
        df_high = pd.DataFrame({"raw_scores": np.random.uniform(0.9, 1.0, 100)})
        score_map_high = get_calibrated_score_map(
            df=df_high,
            score_field="raw_scores",
            calibration_dictionary=sample_calibration_dict,
        )
        assert len(score_map_high) > 0

        # All low scores
        df_low = pd.DataFrame({"raw_scores": np.random.uniform(0.0, 0.1, 100)})
        score_map_low = get_calibrated_score_map(
            df=df_low,
            score_field="raw_scores",
            calibration_dictionary=sample_calibration_dict,
        )
        assert len(score_map_low) > 0

    def test_get_calibrated_score_map_monotonic(
        self, sample_score_data, sample_calibration_dict
    ):
        """Test that calibrated percentiles are monotonically increasing."""
        score_map = get_calibrated_score_map(
            df=sample_score_data,
            score_field="raw_scores",
            calibration_dictionary=sample_calibration_dict,
        )

        # Extract percentiles (second element of each tuple)
        percentiles = [item[1] for item in score_map]

        # Check monotonicity (percentiles should be non-decreasing)
        for i in range(len(percentiles) - 1):
            assert percentiles[i] <= percentiles[i + 1], (
                f"Percentiles not monotonic at index {i}: {percentiles[i]} > {percentiles[i + 1]}"
            )


class TestDataFileDiscovery:
    """Tests for data file discovery functionality."""

    @pytest.fixture
    def setup_temp_dir(self):
        """Set up temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_find_first_data_file_eval_predictions(self, setup_temp_dir):
        """Test finding eval_predictions.csv (highest priority)."""
        temp_dir = setup_temp_dir
        (temp_dir / "eval_predictions.csv").touch()
        (temp_dir / "predictions.csv").touch()

        found_file = find_first_data_file(str(temp_dir))

        assert found_file == str(temp_dir / "eval_predictions.csv")

    def test_find_first_data_file_eval_predictions_with_comparison(
        self, setup_temp_dir
    ):
        """Test finding eval_predictions_with_comparison.csv."""
        temp_dir = setup_temp_dir
        (temp_dir / "eval_predictions_with_comparison.csv").touch()
        (temp_dir / "predictions.csv").touch()

        found_file = find_first_data_file(str(temp_dir))

        assert found_file == str(temp_dir / "eval_predictions_with_comparison.csv")

    def test_find_first_data_file_predictions_csv(self, setup_temp_dir):
        """Test finding predictions.csv."""
        temp_dir = setup_temp_dir
        (temp_dir / "predictions.csv").touch()

        found_file = find_first_data_file(str(temp_dir))

        assert found_file == str(temp_dir / "predictions.csv")

    def test_find_first_data_file_predictions_parquet(self, setup_temp_dir):
        """Test finding predictions.parquet."""
        temp_dir = setup_temp_dir
        (temp_dir / "predictions.parquet").touch()

        found_file = find_first_data_file(str(temp_dir))

        assert found_file == str(temp_dir / "predictions.parquet")

    def test_find_first_data_file_priority_order(self, setup_temp_dir):
        """Test file priority order when multiple files exist."""
        temp_dir = setup_temp_dir
        # Create multiple files
        (temp_dir / "eval_predictions.csv").touch()
        (temp_dir / "eval_predictions_with_comparison.csv").touch()
        (temp_dir / "predictions.csv").touch()
        (temp_dir / "predictions.parquet").touch()

        found_file = find_first_data_file(str(temp_dir))

        # Should find eval_predictions.csv (highest priority)
        assert found_file == str(temp_dir / "eval_predictions.csv")

    def test_find_first_data_file_alphabetical_fallback(self, setup_temp_dir):
        """Test alphabetical fallback when no priority files exist."""
        temp_dir = setup_temp_dir
        (temp_dir / "zebra.csv").touch()
        (temp_dir / "alpha.csv").touch()

        found_file = find_first_data_file(str(temp_dir))

        # Should find alpha.csv (alphabetically first)
        assert found_file == str(temp_dir / "alpha.csv")

    def test_find_first_data_file_no_file_found(self, setup_temp_dir):
        """Test error when no supported data file is found."""
        temp_dir = setup_temp_dir

        with pytest.raises(FileNotFoundError, match="No supported data file"):
            find_first_data_file(str(temp_dir))

    def test_find_first_data_file_directory_not_exist(self):
        """Test error when directory doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Directory does not exist"):
            find_first_data_file("/nonexistent/directory")


class TestCalibrationDictionaryLoading:
    """Tests for calibration dictionary loading."""

    @pytest.fixture
    def setup_temp_dir(self):
        """Set up temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_external_calibration_dict(self):
        """Create sample external calibration dictionary."""
        return {
            "0.01": 0.95,
            "0.1": 0.8,
            "0.5": 0.5,
            "0.9": 0.1,
        }

    def test_load_calibration_dictionary_external(
        self, setup_temp_dir, sample_external_calibration_dict
    ):
        """Test loading calibration dictionary from external config."""
        temp_dir = setup_temp_dir
        config_dir = temp_dir / "calibration_config"
        config_dir.mkdir()

        # Save external calibration dict
        config_file = config_dir / "standard_calibration_dictionary.json"
        with open(config_file, "w") as f:
            json.dump(sample_external_calibration_dict, f)

        input_paths = {"calibration_config": str(config_dir)}
        calibration_dict = load_calibration_dictionary(input_paths)

        # Check loaded dictionary (keys should be floats)
        assert isinstance(calibration_dict, dict)
        assert len(calibration_dict) == 4
        assert all(isinstance(k, float) for k in calibration_dict.keys())
        assert calibration_dict[0.01] == 0.95
        assert calibration_dict[0.5] == 0.5

    def test_load_calibration_dictionary_builtin_default(self):
        """Test fallback to built-in default calibration dictionary."""
        input_paths = {}

        calibration_dict = load_calibration_dictionary(input_paths)

        # Check built-in default dictionary
        assert isinstance(calibration_dict, dict)
        assert len(calibration_dict) > 0
        # Check some known values from default
        assert 0.001 in calibration_dict
        assert 0.5 in calibration_dict
        assert 0.999 in calibration_dict

    def test_load_calibration_dictionary_missing_file(self, setup_temp_dir):
        """Test fallback to default when config file doesn't exist."""
        temp_dir = setup_temp_dir
        config_dir = temp_dir / "calibration_config"
        config_dir.mkdir()

        input_paths = {"calibration_config": str(config_dir)}
        calibration_dict = load_calibration_dictionary(input_paths)

        # Should fall back to built-in default
        assert isinstance(calibration_dict, dict)
        assert len(calibration_dict) > 0

    def test_load_calibration_dictionary_invalid_json(self, setup_temp_dir):
        """Test fallback to default when JSON is invalid."""
        temp_dir = setup_temp_dir
        config_dir = temp_dir / "calibration_config"
        config_dir.mkdir()

        # Save invalid JSON
        config_file = config_dir / "standard_calibration_dictionary.json"
        with open(config_file, "w") as f:
            f.write("{ invalid json }")

        input_paths = {"calibration_config": str(config_dir)}
        calibration_dict = load_calibration_dictionary(input_paths)

        # Should fall back to built-in default
        assert isinstance(calibration_dict, dict)
        assert len(calibration_dict) > 0


class TestMainFunction:
    """Tests for main function integration."""

    @pytest.fixture
    def setup_integration_test(self):
        """Set up integration test environment."""
        temp_dir = Path(tempfile.mkdtemp())

        # Create input directory with test data
        input_dir = temp_dir / "eval_data"
        input_dir.mkdir(parents=True)

        # Create realistic calibration test data
        np.random.seed(42)
        n_samples = 200
        df = pd.DataFrame(
            {
                "id": range(n_samples),
                "label": np.random.randint(0, 2, n_samples),
                "prob_class_1": np.random.uniform(0.0, 1.0, n_samples),
            }
        )
        df.to_csv(input_dir / "eval_predictions.csv", index=False)

        # Set up paths
        input_paths = {"evaluation_data": str(input_dir)}
        output_paths = {
            "calibration_output": str(temp_dir / "calibration"),
            "metrics_output": str(temp_dir / "metrics"),
            "calibrated_data": str(temp_dir / "calibrated_data"),
        }
        environ_vars = {
            "N_BINS": "1000",
            "SCORE_FIELD": "prob_class_1",
            "ACCURACY": "1e-5",
        }

        yield temp_dir, input_paths, output_paths, environ_vars
        shutil.rmtree(temp_dir)

    def test_main_single_task_success(self, setup_integration_test):
        """Test successful single-task calibration."""
        temp_dir, input_paths, output_paths, environ_vars = setup_integration_test

        job_args = argparse.Namespace(job_type="calibration")

        # Call main function
        result = main(input_paths, output_paths, environ_vars, job_args)

        # Check result status
        assert result["status"] == "success"
        assert result["mode"] == "single-task"
        assert result["num_tasks"] == 1
        assert "prob_class_1" in result["score_fields"]

        # Check output files created
        calibration_dir = Path(output_paths["calibration_output"])
        assert (calibration_dir / "percentile_score_prob_class_1.pkl").exists()
        assert (
            calibration_dir / "percentile_score.pkl"
        ).exists()  # Backward compatibility

        metrics_dir = Path(output_paths["metrics_output"])
        assert (metrics_dir / "calibration_metrics.json").exists()

        calibrated_data_dir = Path(output_paths["calibrated_data"])
        assert (calibrated_data_dir / "calibrated_data.csv").exists()

        # Verify metrics structure
        with open(metrics_dir / "calibration_metrics.json", "r") as f:
            metrics = json.load(f)

        assert metrics["calibration_method"] == "percentile_score_mapping"
        assert metrics["mode"] == "single-task"
        assert metrics["num_tasks"] == 1
        assert "prob_class_1" in metrics["per_task_metrics"]

    def test_main_multi_task_success(self, setup_integration_test):
        """Test successful multi-task calibration."""
        temp_dir, input_paths, output_paths, environ_vars = setup_integration_test

        # Add multiple score columns to input data
        input_dir = Path(input_paths["evaluation_data"])
        df = pd.read_csv(input_dir / "eval_predictions.csv")
        np.random.seed(43)
        df["task_0_prob"] = np.random.uniform(0.0, 1.0, len(df))
        df["task_1_prob"] = np.random.uniform(0.0, 1.0, len(df))
        df.to_csv(input_dir / "eval_predictions.csv", index=False)

        # Set multi-task environment variables
        environ_vars["SCORE_FIELDS"] = "task_0_prob,task_1_prob"

        job_args = argparse.Namespace(job_type="calibration")

        # Call main function
        result = main(input_paths, output_paths, environ_vars, job_args)

        # Check result status
        assert result["status"] == "success"
        assert result["mode"] == "multi-task"
        assert result["num_tasks"] == 2
        assert "task_0_prob" in result["score_fields"]
        assert "task_1_prob" in result["score_fields"]

        # Check per-task calibration files
        calibration_dir = Path(output_paths["calibration_output"])
        assert (calibration_dir / "percentile_score_task_0_prob.pkl").exists()
        assert (calibration_dir / "percentile_score_task_1_prob.pkl").exists()

        # Verify calibrated data has percentile columns for each task
        calibrated_data_path = (
            Path(output_paths["calibrated_data"]) / "calibrated_data.csv"
        )
        df_calibrated = pd.read_csv(calibrated_data_path)
        assert "task_0_prob_percentile" in df_calibrated.columns
        assert "task_1_prob_percentile" in df_calibrated.columns

    def test_main_format_preservation_tsv(self, setup_integration_test):
        """Test that TSV format is not auto-discovered by find_first_data_file.

        Note: TSV is supported by load_dataframe_with_format and save_dataframe_with_format,
        but find_first_data_file only looks for .csv, .parquet, and .json files.
        This test verifies that TSV files are NOT found during auto-discovery.
        """
        temp_dir, input_paths, output_paths, environ_vars = setup_integration_test

        # Change input to TSV format (should not be found)
        input_dir = Path(input_paths["evaluation_data"])
        df = pd.read_csv(input_dir / "eval_predictions.csv")
        (input_dir / "eval_predictions.csv").unlink()
        df.to_csv(input_dir / "eval_predictions.tsv", sep="\t", index=False)

        job_args = argparse.Namespace(job_type="calibration")

        # Call main function - should fail because TSV is not supported in find_first_data_file
        result = main(input_paths, output_paths, environ_vars, job_args)

        # Should return error status because no supported file was found
        assert result["status"] == "error"
        assert "No supported data file" in result["error_message"]

    def test_main_format_preservation_parquet(self, setup_integration_test):
        """Test format preservation with Parquet input."""
        temp_dir, input_paths, output_paths, environ_vars = setup_integration_test

        # Change input to Parquet format
        input_dir = Path(input_paths["evaluation_data"])
        df = pd.read_csv(input_dir / "eval_predictions.csv")
        (input_dir / "eval_predictions.csv").unlink()
        df.to_parquet(input_dir / "eval_predictions.parquet")

        job_args = argparse.Namespace(job_type="calibration")

        # Call main function
        result = main(input_paths, output_paths, environ_vars, job_args)

        # Check output is also Parquet
        calibrated_data_dir = Path(output_paths["calibrated_data"])
        assert (calibrated_data_dir / "calibrated_data.parquet").exists()
        assert not (calibrated_data_dir / "calibrated_data.csv").exists()

    def test_main_with_external_calibration_dict(self, setup_integration_test):
        """Test calibration with external calibration dictionary."""
        temp_dir, input_paths, output_paths, environ_vars = setup_integration_test

        # Create external calibration dict
        config_dir = temp_dir / "calibration_config"
        config_dir.mkdir()
        external_dict = {"0.1": 0.9, "0.5": 0.5, "0.9": 0.1}
        with open(config_dir / "standard_calibration_dictionary.json", "w") as f:
            json.dump(external_dict, f)

        input_paths["calibration_config"] = str(config_dir)

        job_args = argparse.Namespace(job_type="calibration")

        # Call main function
        result = main(input_paths, output_paths, environ_vars, job_args)

        # Should succeed with external dictionary
        assert result["status"] == "success"

    def test_main_missing_score_field(self, setup_integration_test):
        """Test error when score field doesn't exist in data."""
        temp_dir, input_paths, output_paths, environ_vars = setup_integration_test

        # Set non-existent score field
        environ_vars["SCORE_FIELD"] = "nonexistent_field"

        job_args = argparse.Namespace(job_type="calibration")

        # Call main function - should handle gracefully
        result = main(input_paths, output_paths, environ_vars, job_args)

        # Should return error status
        assert result["status"] == "error"

    def test_main_calibrated_data_with_nan_values(self, setup_integration_test):
        """Test calibration handles NaN values correctly."""
        temp_dir, input_paths, output_paths, environ_vars = setup_integration_test

        # Add NaN values to input data
        input_dir = Path(input_paths["evaluation_data"])
        df = pd.read_csv(input_dir / "eval_predictions.csv")
        # Set some scores to NaN
        df.loc[0:10, "prob_class_1"] = np.nan
        df.to_csv(input_dir / "eval_predictions.csv", index=False)

        job_args = argparse.Namespace(job_type="calibration")

        # Call main function
        result = main(input_paths, output_paths, environ_vars, job_args)

        # Should succeed
        assert result["status"] == "success"

        # Verify calibrated data has NaN preserved in calibrated column
        calibrated_data_path = (
            Path(output_paths["calibrated_data"]) / "calibrated_data.csv"
        )
        df_calibrated = pd.read_csv(calibrated_data_path)
        assert "prob_class_1_percentile" in df_calibrated.columns
        # First 11 rows should have NaN in calibrated column
        assert df_calibrated["prob_class_1_percentile"].iloc[0:11].isna().all()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def setup_temp_dir(self):
        """Set up temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_calibration_with_constant_scores(self):
        """Test calibration when all scores are constant."""
        df = pd.DataFrame({"raw_scores": np.full(100, 0.5)})
        calibration_dict = {0.1: 0.9, 0.5: 0.5, 0.9: 0.1}

        score_map = get_calibrated_score_map(
            df=df,
            score_field="raw_scores",
            calibration_dictionary=calibration_dict,
        )

        # Should handle constant scores without error
        assert isinstance(score_map, list)
        assert len(score_map) > 0

    def test_calibration_with_small_dataset(self):
        """Test calibration with very small dataset."""
        np.random.seed(42)
        df = pd.DataFrame({"raw_scores": np.random.uniform(0, 1, 10)})
        calibration_dict = {0.1: 0.9, 0.5: 0.5, 0.9: 0.1}

        score_map = get_calibrated_score_map(
            df=df,
            score_field="raw_scores",
            calibration_dictionary=calibration_dict,
        )

        # Should handle small dataset
        assert isinstance(score_map, list)
        assert len(score_map) > 0

    def test_calibration_with_all_zeros(self):
        """Test calibration when all scores are zero."""
        df = pd.DataFrame({"raw_scores": np.zeros(100)})
        calibration_dict = {0.1: 0.9, 0.5: 0.5, 0.9: 0.1}

        score_map = get_calibrated_score_map(
            df=df,
            score_field="raw_scores",
            calibration_dictionary=calibration_dict,
        )

        # Should handle all zeros
        assert isinstance(score_map, list)
        assert len(score_map) > 0

    def test_calibration_with_all_ones(self):
        """Test calibration when all scores are one."""
        df = pd.DataFrame({"raw_scores": np.ones(100)})
        calibration_dict = {0.1: 0.9, 0.5: 0.5, 0.9: 0.1}

        score_map = get_calibrated_score_map(
            df=df,
            score_field="raw_scores",
            calibration_dictionary=calibration_dict,
        )

        # Should handle all ones
        assert isinstance(score_map, list)
        assert len(score_map) > 0

    def test_score_field_validation_empty_dataframe(self):
        """Test validation with empty dataframe."""
        df = pd.DataFrame()
        score_fields = ["prob_class_1"]

        valid_fields, missing_fields = validate_score_fields(df, score_fields)

        assert valid_fields == []
        assert missing_fields == ["prob_class_1"]

    def test_parse_score_fields_with_empty_values(self):
        """Test parsing score fields with empty comma-separated values."""
        environ_vars = {"SCORE_FIELDS": "task_0_prob,,task_1_prob,"}

        score_fields = parse_score_fields(environ_vars)

        # Should filter out empty strings
        assert score_fields == ["task_0_prob", "task_1_prob"]
        assert len(score_fields) == 2

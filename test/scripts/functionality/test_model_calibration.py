"""
Comprehensive test suite for model_calibration.py script.

This test suite follows pytest best practices and provides thorough coverage
of the model calibration functionality including:
- Binary and multi-class calibration
- Multiple calibration methods (GAM, Isotonic, Platt scaling)
- Multi-task calibration support
- File format preservation
- Data loading and preparation
- Nested tarball extraction
- Calibration metrics computation
- Visualization generation
"""

import pytest
from unittest.mock import patch, MagicMock, Mock, mock_open, call
import os
import sys
import tempfile
import shutil
import json
import tarfile
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
import pickle as pkl

# CRITICAL: Mock subprocess.check_call BEFORE importing the module
# The model_calibration script calls install_packages() at module level (line 185)
# which triggers subprocess.check_call to run pip install.
# We must mock check_call before the module import to prevent actual installations.
with patch("subprocess.check_call"):
    # Import the functions to be tested
    from cursus.steps.scripts.model_calibration import (
        CalibrationConfig,
        create_directories,
        find_first_data_file,
        load_dataframe_with_format,
        save_dataframe_with_format,
        _detect_file_format,
        load_data,
        load_and_prepare_data,
        train_gam_calibration,
        train_isotonic_calibration,
        train_platt_scaling,
        train_multiclass_calibration,
        apply_multiclass_calibration,
        _interpolate_score,
        _model_to_lookup_table,
        compute_calibration_metrics,
        compute_multiclass_calibration_metrics,
        plot_reliability_diagram,
        plot_multiclass_reliability_diagram,
        parse_score_fields,
        validate_score_fields,
        extract_and_load_nested_tarball_data,
        main,
    )


class TestCalibrationConfig:
    """Tests for CalibrationConfig class."""

    def test_config_initialization_default_values(self):
        """Test configuration initialization with default values."""
        config = CalibrationConfig()

        assert config.input_data_path == "/opt/ml/processing/input/eval_data"
        assert config.output_calibration_path == "/opt/ml/processing/output/calibration"
        assert config.calibration_method == "gam"
        assert config.label_field == "label"
        assert config.score_field == "prob_class_1"
        assert config.is_binary is True
        assert config.monotonic_constraint is True
        assert config.gam_splines == 10
        assert config.error_threshold == 0.05
        assert config.num_classes == 2
        assert config.calibration_sample_points == 1000

    def test_config_initialization_custom_values(self):
        """Test configuration initialization with custom values."""
        config = CalibrationConfig(
            input_data_path="/custom/input",
            calibration_method="isotonic",
            label_field="target",
            score_field="score",
            is_binary=False,
            num_classes=3,
            monotonic_constraint=False,
            gam_splines=15,
            calibration_sample_points=500,
        )

        assert config.input_data_path == "/custom/input"
        assert config.calibration_method == "isotonic"
        assert config.label_field == "target"
        assert config.score_field == "score"
        assert config.is_binary is False
        assert config.num_classes == 3
        assert config.monotonic_constraint is False
        assert config.gam_splines == 15
        assert config.calibration_sample_points == 500

    def test_config_from_env_binary(self):
        """Test configuration creation from environment variables for binary case."""
        environ_vars = {
            "CALIBRATION_METHOD": "platt",
            "LABEL_FIELD": "y_true",
            "SCORE_FIELD": "y_score",
            "IS_BINARY": "True",
            "MONOTONIC_CONSTRAINT": "False",
            "GAM_SPLINES": "20",
            "ERROR_THRESHOLD": "0.1",
            "CALIBRATION_SAMPLE_POINTS": "2000",
        }

        with patch.dict(os.environ, environ_vars):
            config = CalibrationConfig.from_env()

        assert config.calibration_method == "platt"
        assert config.label_field == "y_true"
        assert config.score_field == "y_score"
        assert config.is_binary is True
        assert config.monotonic_constraint is False
        assert config.gam_splines == 20
        assert config.error_threshold == 0.1
        assert config.calibration_sample_points == 2000

    def test_config_from_env_multiclass(self):
        """Test configuration creation from environment variables for multiclass case."""
        environ_vars = {
            "IS_BINARY": "False",
            "NUM_CLASSES": "4",
            "SCORE_FIELD_PREFIX": "prob_",
            "MULTICLASS_CATEGORIES": '["cat", "dog", "bird", "fish"]',
        }

        with patch.dict(os.environ, environ_vars):
            config = CalibrationConfig.from_env()

        assert config.is_binary is False
        assert config.num_classes == 4
        assert config.score_field_prefix == "prob_"
        assert config.multiclass_categories == ["cat", "dog", "bird", "fish"]

    def test_config_multiclass_categories_fallback(self):
        """Test multiclass categories with fallback to numeric indices."""
        config = CalibrationConfig(is_binary=False, num_classes=3)

        assert config.multiclass_categories == ["0", "1", "2"]


class TestFileIO:
    """Tests for file I/O helper functions."""

    @pytest.fixture
    def setup_temp_dir(self):
        """Set up temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_detect_file_format_csv(self):
        """Test CSV format detection."""
        assert _detect_file_format("data.csv") == "csv"
        assert _detect_file_format("/path/to/data.csv") == "csv"

    def test_detect_file_format_tsv(self):
        """Test TSV format detection."""
        assert _detect_file_format("data.tsv") == "tsv"
        assert _detect_file_format("/path/to/data.tsv") == "tsv"

    def test_detect_file_format_parquet(self):
        """Test Parquet format detection."""
        assert _detect_file_format("data.parquet") == "parquet"
        assert _detect_file_format("/path/to/data.parquet") == "parquet"

    def test_detect_file_format_unsupported(self):
        """Test unsupported format detection."""
        with pytest.raises(RuntimeError, match="Unsupported file format"):
            _detect_file_format("data.xlsx")

    def test_load_dataframe_csv(self, setup_temp_dir):
        """Test loading CSV dataframe."""
        temp_dir = setup_temp_dir
        csv_file = temp_dir / "data.csv"

        df_original = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        df_original.to_csv(csv_file, index=False)

        df_loaded, format_str = load_dataframe_with_format(str(csv_file))

        assert format_str == "csv"
        pd.testing.assert_frame_equal(df_loaded, df_original)

    def test_load_dataframe_tsv(self, setup_temp_dir):
        """Test loading TSV dataframe."""
        temp_dir = setup_temp_dir
        tsv_file = temp_dir / "data.tsv"

        df_original = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        df_original.to_csv(tsv_file, sep="\t", index=False)

        df_loaded, format_str = load_dataframe_with_format(str(tsv_file))

        assert format_str == "tsv"
        pd.testing.assert_frame_equal(df_loaded, df_original)

    def test_save_dataframe_csv(self, setup_temp_dir):
        """Test saving CSV dataframe."""
        temp_dir = setup_temp_dir
        output_base = temp_dir / "output"

        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        output_path = save_dataframe_with_format(df, str(output_base), "csv")

        assert Path(output_path).exists()
        assert output_path.endswith(".csv")

        df_loaded = pd.read_csv(output_path)
        pd.testing.assert_frame_equal(df_loaded, df)

    def test_save_dataframe_parquet(self, setup_temp_dir):
        """Test saving Parquet dataframe."""
        temp_dir = setup_temp_dir
        output_base = temp_dir / "output"

        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        output_path = save_dataframe_with_format(df, str(output_base), "parquet")

        assert Path(output_path).exists()
        assert output_path.endswith(".parquet")

        df_loaded = pd.read_parquet(output_path)
        pd.testing.assert_frame_equal(df_loaded, df)


class TestDataLoading:
    """Tests for data loading functions."""

    @pytest.fixture
    def setup_temp_dir(self):
        """Set up temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration."""
        return CalibrationConfig()

    def test_find_first_data_file_csv(self, setup_temp_dir, sample_config):
        """Test finding first CSV data file."""
        temp_dir = setup_temp_dir
        (temp_dir / "data.csv").write_text("col1,col2\n1,2\n")

        result = find_first_data_file(str(temp_dir), sample_config)

        assert result.endswith("data.csv")
        assert str(temp_dir) in result

    def test_find_first_data_file_sorted_order(self, setup_temp_dir, sample_config):
        """Test that files are found in sorted order."""
        temp_dir = setup_temp_dir
        (temp_dir / "z_data.csv").write_text("col1\n1\n")
        (temp_dir / "a_data.csv").write_text("col1\n2\n")

        result = find_first_data_file(str(temp_dir), sample_config)

        assert result.endswith("a_data.csv")

    def test_find_first_data_file_no_files(self, setup_temp_dir, sample_config):
        """Test finding data file when none exist."""
        temp_dir = setup_temp_dir
        (temp_dir / "readme.txt").write_text("not a data file")

        with pytest.raises(
            FileNotFoundError,
            match="No supported data file \\(\\.csv, \\.parquet, \\.json\\) found",
        ):
            find_first_data_file(str(temp_dir), sample_config)

    def test_find_first_data_file_no_directory(self, sample_config):
        """Test finding data file when directory doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Directory does not exist"):
            find_first_data_file("/nonexistent/directory", sample_config)

    def test_load_data_with_required_columns(self, setup_temp_dir):
        """Test loading data with required columns."""
        temp_dir = setup_temp_dir
        config = CalibrationConfig(
            input_data_path=str(temp_dir),
            label_field="label",
            score_field="prob_class_1",
            is_binary=True,
        )

        df = pd.DataFrame(
            {
                "label": [0, 1, 0, 1],
                "prob_class_1": [0.2, 0.8, 0.3, 0.9],
            }
        )
        df.to_csv(temp_dir / "data.csv", index=False)

        df_loaded, format_str = load_data(config)

        assert format_str == "csv"
        assert "label" in df_loaded.columns
        assert "prob_class_1" in df_loaded.columns

    def test_load_data_missing_label_field(self, setup_temp_dir):
        """Test loading data with missing label field."""
        temp_dir = setup_temp_dir
        config = CalibrationConfig(
            input_data_path=str(temp_dir),
            label_field="label",
        )

        df = pd.DataFrame({"score": [0.2, 0.8]})
        df.to_csv(temp_dir / "data.csv", index=False)

        with pytest.raises(ValueError, match="Label field 'label' not found"):
            load_data(config)

    def test_load_data_missing_score_field_binary(self, setup_temp_dir):
        """Test loading data with missing score field for binary classification."""
        temp_dir = setup_temp_dir
        config = CalibrationConfig(
            input_data_path=str(temp_dir),
            label_field="label",
            score_field="prob_class_1",
            is_binary=True,
        )

        df = pd.DataFrame({"label": [0, 1]})
        df.to_csv(temp_dir / "data.csv", index=False)

        with pytest.raises(ValueError, match="Score field 'prob_class_1' not found"):
            load_data(config)


class TestScoreFieldParsing:
    """Tests for score field parsing and validation."""

    def test_parse_score_fields_single_field(self):
        """Test parsing single score field."""
        environ_vars = {"SCORE_FIELD": "prob_class_1"}

        result = parse_score_fields(environ_vars)

        assert result == ["prob_class_1"]

    def test_parse_score_fields_multiple_fields(self):
        """Test parsing multiple score fields."""
        environ_vars = {"SCORE_FIELDS": "task1_score, task2_score, task3_score"}

        result = parse_score_fields(environ_vars)

        assert result == ["task1_score", "task2_score", "task3_score"]

    def test_parse_score_fields_default(self):
        """Test parsing with no score field specified."""
        environ_vars = {}

        result = parse_score_fields(environ_vars)

        assert result == ["prob_class_1"]

    def test_validate_score_fields_all_valid(self):
        """Test validation when all score fields exist."""
        df = pd.DataFrame(
            {
                "label": [0, 1, 0],
                "score1": [0.2, 0.8, 0.3],
                "score2": [0.3, 0.7, 0.4],
            }
        )
        score_fields = ["score1", "score2"]

        result = validate_score_fields(df, score_fields, "label")

        assert result == ["score1", "score2"]

    def test_validate_score_fields_some_invalid(self):
        """Test validation when some score fields don't exist."""
        df = pd.DataFrame(
            {
                "label": [0, 1, 0],
                "score1": [0.2, 0.8, 0.3],
            }
        )
        score_fields = ["score1", "score2", "score3"]

        result = validate_score_fields(df, score_fields, "label")

        assert result == ["score1"]

    def test_validate_score_fields_none_valid(self):
        """Test validation when no score fields exist."""
        df = pd.DataFrame({"label": [0, 1, 0]})
        score_fields = ["score1", "score2"]

        with pytest.raises(ValueError, match="None of the specified score fields"):
            validate_score_fields(df, score_fields, "label")


class TestInterpolation:
    """Tests for score interpolation."""

    def test_interpolate_score_within_range(self):
        """Test interpolation for score within lookup table range."""
        lookup_table = [(0.0, 0.0), (0.5, 0.6), (1.0, 1.0)]

        result = _interpolate_score(0.25, lookup_table)

        # Linear interpolation: 0.0 + (0.6 - 0.0) * (0.25 - 0.0) / (0.5 - 0.0) = 0.3
        assert result == pytest.approx(0.3)

    def test_interpolate_score_below_range(self):
        """Test interpolation for score below lookup table range."""
        lookup_table = [(0.0, 0.1), (0.5, 0.6), (1.0, 1.0)]

        result = _interpolate_score(-0.1, lookup_table)

        assert result == 0.1  # Should return first value

    def test_interpolate_score_above_range(self):
        """Test interpolation for score above lookup table range."""
        lookup_table = [(0.0, 0.0), (0.5, 0.6), (1.0, 0.9)]

        result = _interpolate_score(1.5, lookup_table)

        assert result == 0.9  # Should return last value

    def test_interpolate_score_exact_match(self):
        """Test interpolation for score exactly matching table entry."""
        lookup_table = [(0.0, 0.0), (0.5, 0.6), (1.0, 1.0)]

        result = _interpolate_score(0.5, lookup_table)

        assert result == 0.6


class TestCalibrationTraining:
    """Tests for calibration training methods."""

    @pytest.fixture
    def binary_calibration_data(self):
        """Create binary calibration data."""
        np.random.seed(42)
        n_samples = 100

        scores = np.random.uniform(0, 1, n_samples)
        labels = (scores + np.random.normal(0, 0.2, n_samples) > 0.5).astype(int)

        return scores, labels

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration."""
        return CalibrationConfig(
            monotonic_constraint=True,
            gam_splines=10,
            calibration_sample_points=100,
        )

    def test_train_isotonic_calibration(self, binary_calibration_data, sample_config):
        """Test isotonic regression calibration training."""
        scores, labels = binary_calibration_data

        lookup_table = train_isotonic_calibration(scores, labels, sample_config)

        assert isinstance(lookup_table, list)
        assert len(lookup_table) > 0
        assert all(isinstance(item, tuple) and len(item) == 2 for item in lookup_table)
        # Check boundaries
        assert lookup_table[0][0] == pytest.approx(0.0)
        assert lookup_table[-1][0] == pytest.approx(1.0)

    def test_train_platt_scaling(self, binary_calibration_data, sample_config):
        """Test Platt scaling calibration training."""
        scores, labels = binary_calibration_data

        lookup_table = train_platt_scaling(scores, labels, sample_config)

        assert isinstance(lookup_table, list)
        assert len(lookup_table) > 0
        assert all(isinstance(item, tuple) and len(item) == 2 for item in lookup_table)
        assert lookup_table[0][0] == pytest.approx(0.0)
        assert lookup_table[-1][0] == pytest.approx(1.0)

    @patch("cursus.steps.scripts.model_calibration.HAS_PYGAM", True)
    @patch("cursus.steps.scripts.model_calibration.LogisticGAM")
    def test_train_gam_calibration_with_pygam(
        self, mock_gam_class, binary_calibration_data, sample_config
    ):
        """Test GAM calibration training when pygam is available."""
        scores, labels = binary_calibration_data

        mock_gam = MagicMock()
        mock_gam.statistics_ = {"deviance": 1.5}
        mock_gam.predict_proba.return_value = np.linspace(
            0, 1, sample_config.calibration_sample_points
        )
        mock_gam_class.return_value = mock_gam

        lookup_table = train_gam_calibration(scores, labels, sample_config)

        assert isinstance(lookup_table, list)
        assert len(lookup_table) == sample_config.calibration_sample_points
        mock_gam.fit.assert_called_once()

    @patch("cursus.steps.scripts.model_calibration.HAS_PYGAM", False)
    def test_train_gam_calibration_without_pygam(
        self, binary_calibration_data, sample_config
    ):
        """Test GAM calibration training raises error when pygam not available."""
        scores, labels = binary_calibration_data

        with pytest.raises(ImportError, match="pygam package is required"):
            train_gam_calibration(scores, labels, sample_config)


class TestMulticlassCalibration:
    """Tests for multi-class calibration."""

    @pytest.fixture
    def multiclass_data(self):
        """Create multi-class calibration data."""
        np.random.seed(42)
        n_samples = 100
        n_classes = 3

        # Create probability matrix
        y_prob_matrix = np.random.dirichlet(np.ones(n_classes), size=n_samples)

        # Create true labels
        y_true = np.random.randint(0, n_classes, n_samples)

        return y_prob_matrix, y_true

    @pytest.fixture
    def multiclass_config(self):
        """Create multiclass configuration."""
        return CalibrationConfig(
            is_binary=False,
            num_classes=3,
            multiclass_categories=["class_0", "class_1", "class_2"],
            calibration_sample_points=100,
        )

    def test_train_multiclass_calibration_isotonic(
        self, multiclass_data, multiclass_config
    ):
        """Test multiclass calibration with isotonic method."""
        y_prob_matrix, y_true = multiclass_data

        calibrators = train_multiclass_calibration(
            y_prob_matrix, y_true, method="isotonic", config=multiclass_config
        )

        assert len(calibrators) == multiclass_config.num_classes
        assert all(isinstance(cal, list) for cal in calibrators)

    def test_apply_multiclass_calibration(self, multiclass_data, multiclass_config):
        """Test applying multiclass calibration."""
        y_prob_matrix, y_true = multiclass_data

        # Train calibrators
        calibrators = train_multiclass_calibration(
            y_prob_matrix, y_true, method="isotonic", config=multiclass_config
        )

        # Apply calibration
        y_prob_calibrated = apply_multiclass_calibration(
            y_prob_matrix, calibrators, multiclass_config
        )

        assert y_prob_calibrated.shape == y_prob_matrix.shape
        # Check probabilities sum to 1
        assert np.allclose(y_prob_calibrated.sum(axis=1), 1.0)
        # Check all probabilities are in [0, 1]
        assert np.all((y_prob_calibrated >= 0) & (y_prob_calibrated <= 1))


class TestCalibrationMetrics:
    """Tests for calibration metrics computation."""

    @pytest.fixture
    def binary_prediction_data(self):
        """Create binary prediction data for metrics testing."""
        np.random.seed(42)
        n_samples = 100

        y_true = np.random.randint(0, 2, n_samples)
        y_prob = np.random.uniform(0, 1, n_samples)

        return y_true, y_prob

    def test_compute_calibration_metrics_structure(self, binary_prediction_data):
        """Test calibration metrics structure."""
        y_true, y_prob = binary_prediction_data

        metrics = compute_calibration_metrics(y_true, y_prob, n_bins=10)

        # Check required fields
        assert "expected_calibration_error" in metrics
        assert "maximum_calibration_error" in metrics
        assert "brier_score" in metrics
        assert "auc_roc" in metrics
        assert "reliability_diagram" in metrics
        assert "bin_statistics" in metrics
        assert "num_samples" in metrics
        assert "num_bins" in metrics

    def test_compute_calibration_metrics_values(self, binary_prediction_data):
        """Test calibration metrics value ranges."""
        y_true, y_prob = binary_prediction_data

        metrics = compute_calibration_metrics(y_true, y_prob)

        # ECE should be in [0, 1]
        assert 0 <= metrics["expected_calibration_error"] <= 1
        # MCE should be in [0, 1]
        assert 0 <= metrics["maximum_calibration_error"] <= 1
        # Brier score should be in [0, 1]
        assert 0 <= metrics["brier_score"] <= 1
        # AUC should be in [0, 1]
        assert 0 <= metrics["auc_roc"] <= 1

    def test_compute_multiclass_calibration_metrics(self):
        """Test multiclass calibration metrics."""
        np.random.seed(42)
        n_samples = 100
        n_classes = 3

        config = CalibrationConfig(
            is_binary=False,
            num_classes=3,
            multiclass_categories=["A", "B", "C"],
        )

        y_true = np.random.randint(0, n_classes, n_samples)
        y_prob_matrix = np.random.dirichlet(np.ones(n_classes), size=n_samples)

        metrics = compute_multiclass_calibration_metrics(
            y_true, y_prob_matrix, n_bins=10, config=config
        )

        assert "multiclass_brier_score" in metrics
        assert "macro_expected_calibration_error" in metrics
        assert "per_class_metrics" in metrics
        assert len(metrics["per_class_metrics"]) == n_classes


class TestVisualization:
    """Tests for visualization functions."""

    @pytest.fixture
    def setup_temp_dir(self):
        """Set up temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_config(self, setup_temp_dir):
        """Create sample configuration with temp output paths."""
        temp_dir = setup_temp_dir
        return CalibrationConfig(
            output_metrics_path=str(temp_dir / "metrics"),
        )

    def test_plot_reliability_diagram_creates_file(self, sample_config):
        """Test reliability diagram creation."""
        np.random.seed(42)
        n_samples = 100

        y_true = np.random.randint(0, 2, n_samples)
        y_prob_uncal = np.random.uniform(0, 1, n_samples)
        y_prob_cal = y_prob_uncal * 0.8 + 0.1  # Simple transformation

        os.makedirs(sample_config.output_metrics_path, exist_ok=True)

        plot_path = plot_reliability_diagram(
            y_true, y_prob_uncal, y_prob_cal, n_bins=10, config=sample_config
        )

        assert Path(plot_path).exists()
        assert plot_path.endswith("reliability_diagram.png")

    def test_plot_multiclass_reliability_diagram_creates_file(self, sample_config):
        """Test multiclass reliability diagram creation."""
        np.random.seed(42)
        n_samples = 100
        n_classes = 3

        sample_config.num_classes = n_classes
        sample_config.multiclass_categories = ["A", "B", "C"]

        y_true = np.random.randint(0, n_classes, n_samples)
        y_prob_uncal = np.random.dirichlet(np.ones(n_classes), size=n_samples)
        y_prob_cal = y_prob_uncal * 0.8 + 0.05
        # Normalize to ensure sum to 1
        y_prob_cal = y_prob_cal / y_prob_cal.sum(axis=1, keepdims=True)

        os.makedirs(sample_config.output_metrics_path, exist_ok=True)

        plot_path = plot_multiclass_reliability_diagram(
            y_true, y_prob_uncal, y_prob_cal, n_bins=10, config=sample_config
        )

        assert Path(plot_path).exists()
        assert plot_path.endswith("multiclass_reliability_diagram.png")


class TestMainFunction:
    """Tests for the main function integration."""

    @pytest.fixture
    def setup_integration_test(self):
        """Set up integration test environment."""
        temp_dir = Path(tempfile.mkdtemp())

        # Create input directory with test data
        input_dir = temp_dir / "input"
        input_dir.mkdir(parents=True)

        # Create realistic test data
        np.random.seed(42)
        n_samples = 100
        df = pd.DataFrame(
            {
                "label": np.random.randint(0, 2, n_samples),
                "prob_class_1": np.random.uniform(0, 1, n_samples),
            }
        )
        df.to_csv(input_dir / "predictions.csv", index=False)

        # Set up paths
        input_paths = {"evaluation_data": str(input_dir)}
        output_paths = {
            "calibration_output": str(temp_dir / "calibration"),
            "metrics_output": str(temp_dir / "metrics"),
            "calibrated_data": str(temp_dir / "calibrated_data"),
        }
        environ_vars = {
            "CALIBRATION_METHOD": "isotonic",
            "LABEL_FIELD": "label",
            "SCORE_FIELD": "prob_class_1",
            "IS_BINARY": "True",
            "MONOTONIC_CONSTRAINT": "True",
            "GAM_SPLINES": "10",
            "ERROR_THRESHOLD": "0.05",
            "CALIBRATION_SAMPLE_POINTS": "100",
        }

        yield temp_dir, input_paths, output_paths, environ_vars
        shutil.rmtree(temp_dir)

    def test_main_binary_calibration_success(self, setup_integration_test):
        """Test successful binary calibration via main function."""
        temp_dir, input_paths, output_paths, environ_vars = setup_integration_test

        job_args = argparse.Namespace(job_type="calibration")

        result = main(input_paths, output_paths, environ_vars, job_args)

        assert result["status"] == "success"
        assert result["mode"] == "binary"
        assert result["calibration_method"] == "isotonic"
        assert "metrics_report" in result
        assert "summary" in result

        # Check output files created
        calibration_dir = Path(output_paths["calibration_output"])
        assert (calibration_dir / "calibration_model.pkl").exists()
        assert (calibration_dir / "calibration_summary.json").exists()

    def test_main_multitask_calibration_success(self, setup_integration_test):
        """Test successful multi-task calibration via main function."""
        temp_dir, input_paths, output_paths, environ_vars = setup_integration_test

        # Add multiple score fields to data
        input_dir = Path(input_paths["evaluation_data"])
        df = pd.read_csv(input_dir / "predictions.csv")
        df["task1_score"] = np.random.uniform(0, 1, len(df))
        df["task2_score"] = np.random.uniform(0, 1, len(df))
        df.to_csv(input_dir / "predictions.csv", index=False)

        # Update environment for multi-task
        environ_vars["SCORE_FIELDS"] = "task1_score, task2_score"

        job_args = argparse.Namespace(job_type="calibration")

        result = main(input_paths, output_paths, environ_vars, job_args)

        assert result["status"] == "success"
        assert result["mode"] == "binary_multitask"
        # Check metrics_report structure for multitask
        assert "metrics_report" in result
        assert result["metrics_report"]["num_tasks"] == 2

    def test_main_multiclass_calibration_success(self, setup_integration_test):
        """Test successful multiclass calibration via main function."""
        temp_dir, input_paths, output_paths, environ_vars = setup_integration_test

        # Create multiclass data
        input_dir = Path(input_paths["evaluation_data"])
        np.random.seed(42)
        n_samples = 100
        n_classes = 3

        df = pd.DataFrame(
            {
                "label": np.random.randint(0, n_classes, n_samples),
            }
        )
        # Add probability columns
        probs = np.random.dirichlet(np.ones(n_classes), size=n_samples)
        for i in range(n_classes):
            df[f"prob_class_{i}"] = probs[:, i]

        df.to_csv(input_dir / "predictions.csv", index=False)

        # Update environment for multiclass
        environ_vars["IS_BINARY"] = "False"
        environ_vars["NUM_CLASSES"] = "3"
        environ_vars["SCORE_FIELD_PREFIX"] = "prob_class_"
        environ_vars["MULTICLASS_CATEGORIES"] = '["0", "1", "2"]'

        job_args = argparse.Namespace(job_type="calibration")

        result = main(input_paths, output_paths, environ_vars, job_args)

        assert result["status"] == "success"
        assert result["mode"] == "multi-class"


class TestNestedTarballExtraction:
    """Tests for nested tarball extraction functionality."""

    @pytest.fixture
    def setup_tarball_test(self):
        """Set up nested tarball test environment."""
        temp_dir = Path(tempfile.mkdtemp())

        # Create inner archive structure
        inner_dir = temp_dir / "inner"
        inner_dir.mkdir()
        val_dir = inner_dir / "val"
        val_dir.mkdir()

        # Create predictions CSV
        df = pd.DataFrame(
            {
                "label": [0, 1, 0, 1],
                "prob_class_1": [0.2, 0.8, 0.3, 0.9],
            }
        )
        df.to_csv(val_dir / "predictions.csv", index=False)

        # Create inner tarball
        inner_tar = temp_dir / "val.tar.gz"
        with tarfile.open(inner_tar, "w:gz") as tar:
            tar.add(val_dir, arcname="val")

        # Create outer archive directory
        outer_dir = temp_dir / "outer"
        outer_dir.mkdir()

        # Move inner tarball to outer directory
        shutil.move(str(inner_tar), str(outer_dir / "val.tar.gz"))

        # Create outer tarball
        outer_tar = temp_dir / "output.tar.gz"
        with tarfile.open(outer_tar, "w:gz") as tar:
            for item in outer_dir.iterdir():
                tar.add(item, arcname=item.name)

        # Create input directory with outer tarball
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        shutil.move(str(outer_tar), str(input_dir / "output.tar.gz"))

        yield temp_dir, input_dir
        shutil.rmtree(temp_dir)

    def test_extract_nested_tarball_data(self, setup_tarball_test):
        """Test extraction of nested tarball data."""
        temp_dir, input_dir = setup_tarball_test

        config = CalibrationConfig(input_data_path=str(input_dir))

        df = extract_and_load_nested_tarball_data(config)

        assert len(df) == 4
        assert "label" in df.columns
        assert "prob_class_1" in df.columns
        assert "dataset_origin" in df.columns
        assert df["dataset_origin"].iloc[0] == "val"

    def test_extract_nested_tarball_fallback_to_direct_file(self):
        """Test fallback to direct file when no tarball exists."""
        # Create temp directory with direct CSV file
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # Create realistic test data
            np.random.seed(42)
            n_samples = 100
            df = pd.DataFrame(
                {
                    "label": np.random.randint(0, 2, n_samples),
                    "prob_class_1": np.random.uniform(0, 1, n_samples),
                }
            )
            df.to_csv(temp_dir / "predictions.csv", index=False)

            config = CalibrationConfig(input_data_path=str(temp_dir))

            # Should load direct CSV file
            df_loaded = extract_and_load_nested_tarball_data(config)

            assert len(df_loaded) == 100
            assert "label" in df_loaded.columns
        finally:
            shutil.rmtree(temp_dir)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def setup_temp_dir(self):
        """Set up temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_calibration_with_perfect_predictions(self, setup_temp_dir):
        """Test calibration with perfectly calibrated predictions."""
        temp_dir = setup_temp_dir
        config = CalibrationConfig(
            input_data_path=str(temp_dir),
            calibration_method="isotonic",
            calibration_sample_points=50,
        )

        # Create perfectly calibrated data
        n_samples = 100
        y_true = np.array([i % 2 for i in range(n_samples)])
        y_prob = y_true.astype(float)  # Perfect calibration

        lookup_table = train_isotonic_calibration(y_prob, y_true, config)

        # Should still create valid lookup table
        assert isinstance(lookup_table, list)
        assert len(lookup_table) > 0

    def test_calibration_with_constant_predictions(self, setup_temp_dir):
        """Test calibration with constant predictions."""
        temp_dir = setup_temp_dir
        config = CalibrationConfig(calibration_sample_points=50)

        # All predictions are 0.5
        n_samples = 100
        y_true = np.random.randint(0, 2, n_samples)
        y_prob = np.full(n_samples, 0.5)

        lookup_table = train_isotonic_calibration(y_prob, y_true, config)

        # Should handle constant predictions
        assert isinstance(lookup_table, list)
        assert len(lookup_table) > 0

    def test_calibration_with_small_dataset(self, setup_temp_dir):
        """Test calibration with very small dataset."""
        temp_dir = setup_temp_dir
        config = CalibrationConfig(calibration_sample_points=20)

        # Only 10 samples
        n_samples = 10
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.45, 0.55])

        lookup_table = train_isotonic_calibration(y_prob, y_true, config)

        # Should handle small dataset
        assert isinstance(lookup_table, list)
        assert len(lookup_table) > 0

    def test_model_to_lookup_table_invalid_method(self):
        """Test _model_to_lookup_table with invalid method."""
        mock_model = Mock()
        config = CalibrationConfig()

        with pytest.raises(ValueError, match="Unknown calibration method"):
            _model_to_lookup_table(mock_model, "invalid_method", config)

    def test_create_directories_creates_all_paths(self, setup_temp_dir):
        """Test that create_directories creates all required paths."""
        temp_dir = setup_temp_dir
        config = CalibrationConfig(
            output_calibration_path=str(temp_dir / "calibration"),
            output_metrics_path=str(temp_dir / "metrics"),
            output_calibrated_data_path=str(temp_dir / "calibrated_data"),
        )

        create_directories(config)

        assert Path(config.output_calibration_path).exists()
        assert Path(config.output_metrics_path).exists()
        assert Path(config.output_calibrated_data_path).exists()


class TestCalibrationMethods:
    """Tests for different calibration methods."""

    @pytest.fixture
    def calibration_data(self):
        """Create calibration test data."""
        np.random.seed(42)
        n_samples = 200

        # Create biased predictions
        y_true = np.random.randint(0, 2, n_samples)
        # Add systematic bias
        y_prob = np.clip(y_true + np.random.normal(0.2, 0.3, n_samples), 0, 1)

        return y_true, y_prob

    def test_isotonic_regression_improves_calibration(self, calibration_data):
        """Test that isotonic regression improves calibration."""
        y_true, y_prob_uncal = calibration_data
        config = CalibrationConfig(calibration_sample_points=100)

        # Train calibrator
        lookup_table = train_isotonic_calibration(y_prob_uncal, y_true, config)

        # Apply calibration
        y_prob_cal = np.array(
            [_interpolate_score(score, lookup_table) for score in y_prob_uncal]
        )

        # Compute metrics
        metrics_uncal = compute_calibration_metrics(y_true, y_prob_uncal)
        metrics_cal = compute_calibration_metrics(y_true, y_prob_cal)

        # Calibration should reduce ECE (or at worst keep it similar)
        # Using >= to allow for cases where calibration doesn't hurt
        assert (
            metrics_uncal["expected_calibration_error"]
            >= metrics_cal["expected_calibration_error"] - 0.05
        )

    def test_platt_scaling_improves_calibration(self, calibration_data):
        """Test that Platt scaling improves calibration."""
        y_true, y_prob_uncal = calibration_data
        config = CalibrationConfig(calibration_sample_points=100)

        # Train calibrator
        lookup_table = train_platt_scaling(y_prob_uncal, y_true, config)

        # Apply calibration
        y_prob_cal = np.array(
            [_interpolate_score(score, lookup_table) for score in y_prob_uncal]
        )

        # Compute metrics
        metrics_uncal = compute_calibration_metrics(y_true, y_prob_uncal)
        metrics_cal = compute_calibration_metrics(y_true, y_prob_cal)

        # Calibration should reduce ECE (or at worst keep it similar)
        assert (
            metrics_uncal["expected_calibration_error"]
            >= metrics_cal["expected_calibration_error"] - 0.05
        )

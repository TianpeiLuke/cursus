import pytest
from unittest.mock import patch, MagicMock, mock_open
import os
import tempfile
import shutil
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Import the functions to be tested
from cursus.steps.scripts.model_calibration import (
    CalibrationConfig,
    create_directories,
    find_first_data_file,
    load_data,
    load_and_prepare_data,
    train_gam_calibration,
    train_isotonic_calibration,
    train_platt_scaling,
    train_multiclass_calibration,
    apply_multiclass_calibration,
    compute_calibration_metrics,
    compute_multiclass_calibration_metrics,
    plot_reliability_diagram,
    plot_multiclass_reliability_diagram,
    main,
)


class TestCalibrationConfig:
    """Tests for the CalibrationConfig class."""

    def test_calibration_config_defaults(self):
        """Test CalibrationConfig with default values."""
        config = CalibrationConfig()

        assert config.calibration_method == "gam"
        assert config.label_field == "label"
        assert config.score_field == "prob_class_1"
        assert config.is_binary is True
        assert config.monotonic_constraint is True
        assert config.gam_splines == 10
        assert config.error_threshold == 0.05
        assert config.num_classes == 2

    def test_calibration_config_custom_values(self):
        """Test CalibrationConfig with custom values."""
        config = CalibrationConfig(
            calibration_method="isotonic",
            label_field="target",
            score_field="prediction",
            is_binary=False,
            monotonic_constraint=False,
            gam_splines=20,
            error_threshold=0.1,
            num_classes=3,
            multiclass_categories=["A", "B", "C"],
        )

        assert config.calibration_method == "isotonic"
        assert config.label_field == "target"
        assert config.score_field == "prediction"
        assert config.is_binary is False
        assert config.monotonic_constraint is False
        assert config.gam_splines == 20
        assert config.error_threshold == 0.1
        assert config.num_classes == 3
        assert config.multiclass_categories == ["A", "B", "C"]

    @patch.dict(
        os.environ,
        {
            "CALIBRATION_METHOD": "platt",
            "LABEL_FIELD": "y_true",
            "SCORE_FIELD": "y_score",
            "IS_BINARY": "false",
            "MONOTONIC_CONSTRAINT": "false",
            "GAM_SPLINES": "15",
            "ERROR_THRESHOLD": "0.02",
            "NUM_CLASSES": "4",
            "MULTICLASS_CATEGORIES": '["class1", "class2", "class3", "class4"]',
        },
    )
    def test_calibration_config_from_env(self):
        """Test CalibrationConfig.from_env() with environment variables."""
        config = CalibrationConfig.from_env()

        assert config.calibration_method == "platt"
        assert config.label_field == "y_true"
        assert config.score_field == "y_score"
        assert config.is_binary is False
        assert config.monotonic_constraint is False
        assert config.gam_splines == 15
        assert config.error_threshold == 0.02
        assert config.num_classes == 4
        assert config.multiclass_categories == ["class1", "class2", "class3", "class4"]


class TestCalibrationHelpers:
    """Tests for calibration helper functions."""

    @pytest.fixture
    def setup_dirs(self):
        """Set up test fixtures."""
        temp_dir = Path(tempfile.mkdtemp())
        config = CalibrationConfig(
            input_data_path=str(temp_dir / "input"),
            output_calibration_path=str(temp_dir / "calibration"),
            output_metrics_path=str(temp_dir / "metrics"),
            output_calibrated_data_path=str(temp_dir / "calibrated"),
        )
        
        yield temp_dir, config
        shutil.rmtree(temp_dir)

    def test_create_directories(self, setup_dirs):
        """Test create_directories function."""
        temp_dir, config = setup_dirs
        
        create_directories(config)

        assert Path(config.output_calibration_path).exists()
        assert Path(config.output_metrics_path).exists()
        assert Path(config.output_calibrated_data_path).exists()

    def test_find_first_data_file_csv(self, setup_dirs):
        """Test finding CSV data file."""
        temp_dir, config = setup_dirs
        
        input_dir = Path(config.input_data_path)
        input_dir.mkdir(parents=True)

        # Create test files
        (input_dir / "data.csv").write_text("col1,col2\n1,2\n")
        (input_dir / "other.txt").write_text("not a data file")

        result = find_first_data_file(config=config)
        assert result.endswith("data.csv")

    def test_find_first_data_file_parquet(self, setup_dirs):
        """Test finding Parquet data file."""
        temp_dir, config = setup_dirs
        
        input_dir = Path(config.input_data_path)
        input_dir.mkdir(parents=True)

        # Create a dummy parquet file (just touch it for the test)
        (input_dir / "data.parquet").touch()

        result = find_first_data_file(config=config)
        assert result.endswith("data.parquet")

    def test_find_first_data_file_no_files(self, setup_dirs):
        """Test finding data file when none exist."""
        temp_dir, config = setup_dirs
        
        input_dir = Path(config.input_data_path)
        input_dir.mkdir(parents=True)

        with pytest.raises(FileNotFoundError):
            find_first_data_file(config=config)

    def test_find_first_data_file_no_directory(self, setup_dirs):
        """Test finding data file when directory doesn't exist."""
        temp_dir, config = setup_dirs
        
        with pytest.raises(FileNotFoundError):
            find_first_data_file(config=config)

    @patch("cursus.steps.scripts.model_calibration.pd.read_csv")
    @patch("cursus.steps.scripts.model_calibration.find_first_data_file")
    def test_load_data_csv(self, mock_find_file, mock_read_csv, setup_dirs):
        """Test loading CSV data."""
        temp_dir, config = setup_dirs
        
        mock_find_file.return_value = "/path/to/data.csv"
        mock_df = pd.DataFrame(
            {"label": [0, 1, 0, 1], "prob_class_1": [0.2, 0.8, 0.3, 0.9]}
        )
        mock_read_csv.return_value = mock_df

        result = load_data(config)

        mock_find_file.assert_called_once()
        mock_read_csv.assert_called_once_with("/path/to/data.csv")
        pd.testing.assert_frame_equal(result, mock_df)

    @patch("cursus.steps.scripts.model_calibration.pd.read_parquet")
    @patch("cursus.steps.scripts.model_calibration.find_first_data_file")
    def test_load_data_parquet(self, mock_find_file, mock_read_parquet, setup_dirs):
        """Test loading Parquet data."""
        temp_dir, config = setup_dirs
        
        mock_find_file.return_value = "/path/to/data.parquet"
        mock_df = pd.DataFrame(
            {"label": [0, 1, 0, 1], "prob_class_1": [0.2, 0.8, 0.3, 0.9]}
        )
        mock_read_parquet.return_value = mock_df

        result = load_data(config)

        mock_find_file.assert_called_once()
        mock_read_parquet.assert_called_once_with("/path/to/data.parquet")
        pd.testing.assert_frame_equal(result, mock_df)

    @patch("cursus.steps.scripts.model_calibration.find_first_data_file")
    def test_load_data_missing_label_field(self, mock_find_file, setup_dirs):
        """Test loading data with missing label field."""
        temp_dir, config = setup_dirs
        
        input_dir = Path(config.input_data_path)
        input_dir.mkdir(parents=True)
        csv_file = input_dir / "data.csv"

        # Create CSV without label field
        df = pd.DataFrame({"other_col": [1, 2, 3], "prob_class_1": [0.1, 0.5, 0.9]})
        df.to_csv(csv_file, index=False)

        mock_find_file.return_value = str(csv_file)

        with pytest.raises(ValueError) as exc_info:
            load_data(config)

        assert "Label field 'label' not found" in str(exc_info.value)

    @patch("cursus.steps.scripts.model_calibration.find_first_data_file")
    def test_load_data_missing_score_field_binary(self, mock_find_file, setup_dirs):
        """Test loading binary data with missing score field."""
        temp_dir, config = setup_dirs
        
        input_dir = Path(config.input_data_path)
        input_dir.mkdir(parents=True)
        csv_file = input_dir / "data.csv"

        # Create CSV without score field
        df = pd.DataFrame({"label": [0, 1, 0], "other_col": [0.1, 0.5, 0.9]})
        df.to_csv(csv_file, index=False)

        mock_find_file.return_value = str(csv_file)

        with pytest.raises(ValueError) as exc_info:
            load_data(config)

        assert "Score field 'prob_class_1' not found" in str(exc_info.value)


class TestCalibrationMethods:
    """Tests for calibration training methods."""

    @pytest.fixture
    def setup_data(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 100
        scores = np.random.uniform(0, 1, n_samples)
        labels = (scores > 0.5).astype(int)
        config = CalibrationConfig()
        
        return scores, labels, config

    def test_train_isotonic_calibration(self, setup_data):
        """Test training isotonic regression calibration."""
        scores, labels, config = setup_data
        
        calibrator = train_isotonic_calibration(scores, labels, config)

        # Test that we get a calibrator object
        assert calibrator is not None

        # Test that it can make predictions
        calibrated_scores = calibrator.transform(scores)
        assert len(calibrated_scores) == len(scores)
        assert np.all(calibrated_scores >= 0)
        assert np.all(calibrated_scores <= 1)

    def test_train_platt_scaling(self, setup_data):
        """Test training Platt scaling calibration."""
        scores, labels, config = setup_data
        
        calibrator = train_platt_scaling(scores, labels, config)

        # Test that we get a calibrator object
        assert calibrator is not None

        # Test that it can make predictions
        calibrated_scores = calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]
        assert len(calibrated_scores) == len(scores)
        assert np.all(calibrated_scores >= 0)
        assert np.all(calibrated_scores <= 1)

    @patch("cursus.steps.scripts.model_calibration.HAS_PYGAM", True)
    def test_train_gam_calibration_with_pygam(self, setup_data):
        """Test training GAM calibration when pygam is available."""
        scores, labels, config = setup_data
        
        mock_gam = MagicMock()
        mock_gam.statistics_ = {"deviance": 100.0}
        mock_s = MagicMock()

        # Mock both LogisticGAM and s function from pygam
        with patch(
            "cursus.steps.scripts.model_calibration.LogisticGAM",
            return_value=mock_gam,
            create=True,
        ) as mock_gam_class, patch(
            "cursus.steps.scripts.model_calibration.s", return_value=mock_s, create=True
        ):
            result = train_gam_calibration(scores, labels, config)

            assert result == mock_gam
            mock_gam.fit.assert_called_once()
            mock_gam_class.assert_called_once()

    @patch("cursus.steps.scripts.model_calibration.HAS_PYGAM", False)
    def test_train_gam_calibration_without_pygam(self, setup_data):
        """Test training GAM calibration when pygam is not available."""
        scores, labels, config = setup_data
        
        with pytest.raises(ImportError):
            train_gam_calibration(scores, labels, config)


class TestMulticlassCalibration:
    """Tests for multiclass calibration methods."""

    @pytest.fixture
    def setup_multiclass_data(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 100
        n_classes = 3
        y_prob_matrix = np.random.dirichlet([1, 1, 1], n_samples)
        y_true = np.random.choice(n_classes, n_samples)
        config = CalibrationConfig(
            is_binary=False,
            num_classes=n_classes,
            multiclass_categories=["class_0", "class_1", "class_2"],
        )
        
        return y_prob_matrix, y_true, config, n_samples, n_classes

    def test_train_multiclass_calibration_isotonic(self, setup_multiclass_data):
        """Test training multiclass calibration with isotonic regression."""
        y_prob_matrix, y_true, config, n_samples, n_classes = setup_multiclass_data
        
        calibrators = train_multiclass_calibration(
            y_prob_matrix, y_true, "isotonic", config
        )

        assert len(calibrators) == n_classes

        # Test that each calibrator can make predictions
        for calibrator in calibrators:
            test_scores = np.random.uniform(0, 1, 10)
            calibrated = calibrator.transform(test_scores)
            assert len(calibrated) == 10

    def test_train_multiclass_calibration_platt(self, setup_multiclass_data):
        """Test training multiclass calibration with Platt scaling."""
        y_prob_matrix, y_true, config, n_samples, n_classes = setup_multiclass_data
        
        calibrators = train_multiclass_calibration(
            y_prob_matrix, y_true, "platt", config
        )

        assert len(calibrators) == n_classes

        # Test that each calibrator can make predictions
        for calibrator in calibrators:
            test_scores = np.random.uniform(0, 1, 10).reshape(-1, 1)
            calibrated = calibrator.predict_proba(test_scores)[:, 1]
            assert len(calibrated) == 10

    def test_apply_multiclass_calibration(self, setup_multiclass_data):
        """Test applying multiclass calibration."""
        y_prob_matrix, y_true, config, n_samples, n_classes = setup_multiclass_data
        
        # Train calibrators first
        calibrators = train_multiclass_calibration(
            y_prob_matrix, y_true, "isotonic", config
        )

        # Apply calibration
        calibrated_probs = apply_multiclass_calibration(
            y_prob_matrix, calibrators, config
        )

        # Check output shape and properties
        assert calibrated_probs.shape == y_prob_matrix.shape

        # Check that probabilities sum to 1 (approximately)
        row_sums = calibrated_probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-10)

        # Check that all probabilities are between 0 and 1
        assert np.all(calibrated_probs >= 0)
        assert np.all(calibrated_probs <= 1)


class TestCalibrationMetrics:
    """Tests for calibration metrics computation."""

    @pytest.fixture
    def setup_metrics_data(self):
        """Set up test data."""
        np.random.seed(42)
        n_samples = 100
        y_true = np.random.choice([0, 1], n_samples)
        y_prob = np.random.uniform(0, 1, n_samples)
        
        return y_true, y_prob, n_samples

    def test_compute_calibration_metrics(self, setup_metrics_data):
        """Test computing calibration metrics."""
        y_true, y_prob, n_samples = setup_metrics_data
        
        metrics = compute_calibration_metrics(y_true, y_prob)

        # Check that all expected metrics are present
        expected_keys = [
            "expected_calibration_error",
            "maximum_calibration_error",
            "brier_score",
            "auc_roc",
            "reliability_diagram",
            "bin_statistics",
            "num_samples",
            "num_bins",
        ]

        for key in expected_keys:
            assert key in metrics

        # Check metric value ranges
        assert metrics["expected_calibration_error"] >= 0
        assert metrics["expected_calibration_error"] <= 1
        assert metrics["maximum_calibration_error"] >= 0
        assert metrics["maximum_calibration_error"] <= 1
        assert metrics["brier_score"] >= 0
        assert metrics["brier_score"] <= 1
        assert metrics["auc_roc"] >= 0
        assert metrics["auc_roc"] <= 1
        assert metrics["num_samples"] == n_samples

    def test_compute_multiclass_calibration_metrics(self, setup_metrics_data):
        """Test computing multiclass calibration metrics."""
        _, _, n_samples = setup_metrics_data
        
        n_classes = 3
        y_true = np.random.choice(n_classes, n_samples)
        y_prob_matrix = np.random.dirichlet([1, 1, 1], n_samples)

        config = CalibrationConfig(
            is_binary=False,
            num_classes=n_classes,
            multiclass_categories=["class_0", "class_1", "class_2"],
        )

        metrics = compute_multiclass_calibration_metrics(
            y_true, y_prob_matrix, config=config
        )

        # Check that all expected metrics are present
        expected_keys = [
            "multiclass_brier_score",
            "macro_expected_calibration_error",
            "macro_maximum_calibration_error",
            "maximum_calibration_error",
            "per_class_metrics",
            "num_samples",
            "num_bins",
            "num_classes",
        ]

        for key in expected_keys:
            assert key in metrics

        # Check per-class metrics
        assert len(metrics["per_class_metrics"]) == n_classes
        assert metrics["num_classes"] == n_classes
        assert metrics["num_samples"] == n_samples


class TestCalibrationVisualization:
    """Tests for calibration visualization functions."""

    @pytest.fixture
    def setup_viz_data(self):
        """Set up test data."""
        temp_dir = Path(tempfile.mkdtemp())
        config = CalibrationConfig(output_metrics_path=str(temp_dir))

        np.random.seed(42)
        n_samples = 100
        y_true = np.random.choice([0, 1], n_samples)
        y_prob_uncalibrated = np.random.uniform(0, 1, n_samples)
        y_prob_calibrated = np.random.uniform(0, 1, n_samples)
        
        yield temp_dir, config, y_true, y_prob_uncalibrated, y_prob_calibrated, n_samples
        shutil.rmtree(temp_dir)

    @patch("cursus.steps.scripts.model_calibration.plt")
    def test_plot_reliability_diagram(self, mock_plt, setup_viz_data):
        """Test plotting reliability diagram."""
        temp_dir, config, y_true, y_prob_uncalibrated, y_prob_calibrated, n_samples = setup_viz_data
        
        mock_fig = MagicMock()
        mock_plt.figure.return_value = mock_fig
        mock_plt.subplot2grid.return_value = MagicMock()

        result_path = plot_reliability_diagram(
            y_true,
            y_prob_uncalibrated,
            y_prob_calibrated,
            config=config,
        )

        expected_path = str(temp_dir / "reliability_diagram.png")
        assert result_path == expected_path

        # Verify that plotting functions were called
        mock_plt.figure.assert_called_once()
        mock_plt.savefig.assert_called_once_with(expected_path)
        mock_plt.close.assert_called_once_with(mock_fig)

    @patch("cursus.steps.scripts.model_calibration.plt")
    def test_plot_multiclass_reliability_diagram(self, mock_plt, setup_viz_data):
        """Test plotting multiclass reliability diagram."""
        temp_dir, _, _, _, _, n_samples = setup_viz_data
        
        n_classes = 3
        y_true = np.random.choice(n_classes, n_samples)
        y_prob_uncalibrated = np.random.dirichlet([1, 1, 1], n_samples)
        y_prob_calibrated = np.random.dirichlet([1, 1, 1], n_samples)

        config = CalibrationConfig(
            output_metrics_path=str(temp_dir),
            is_binary=False,
            num_classes=n_classes,
            multiclass_categories=["class_0", "class_1", "class_2"],
        )

        mock_fig = MagicMock()
        mock_axes = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        result_path = plot_multiclass_reliability_diagram(
            y_true, y_prob_uncalibrated, y_prob_calibrated, config=config
        )

        expected_path = str(temp_dir / "multiclass_reliability_diagram.png")
        assert result_path == expected_path

        # Verify that plotting functions were called
        mock_plt.subplots.assert_called_once()
        mock_plt.savefig.assert_called_once_with(expected_path)
        mock_plt.close.assert_called_once_with(mock_fig)


class TestCalibrationMain:
    """Tests for the main calibration function."""

    @pytest.fixture
    def setup_main_data(self):
        """Set up test fixtures."""
        temp_dir = Path(tempfile.mkdtemp())

        # Create test data
        np.random.seed(42)
        n_samples = 100
        df = pd.DataFrame(
            {
                "label": np.random.choice([0, 1], n_samples),
                "prob_class_1": np.random.uniform(0, 1, n_samples),
            }
        )

        # Create input data file
        input_dir = temp_dir / "input"
        input_dir.mkdir(parents=True)
        df.to_csv(input_dir / "data.csv", index=False)

        config = CalibrationConfig(
            input_data_path=str(input_dir),
            output_calibration_path=str(temp_dir / "calibration"),
            output_metrics_path=str(temp_dir / "metrics"),
            output_calibrated_data_path=str(temp_dir / "calibrated"),
            calibration_method="isotonic",
        )
        
        yield temp_dir, df, n_samples
        shutil.rmtree(temp_dir)

    @patch("cursus.steps.scripts.model_calibration.plot_reliability_diagram")
    @patch("cursus.steps.scripts.model_calibration.joblib.dump")
    def test_main_binary_calibration(self, mock_joblib_dump, mock_plot, setup_main_data):
        """Test main function for binary calibration."""
        temp_dir, df, n_samples = setup_main_data
        
        mock_plot.return_value = str(temp_dir / "plot.png")

        # Set up input and output paths
        input_paths = {"eval_data": str(temp_dir / "input")}
        output_paths = {
            "calibration": str(temp_dir / "calibration"),
            "metrics": str(temp_dir / "metrics"),
            "calibrated_data": str(temp_dir / "calibrated"),
        }
        environ_vars = {
            "CALIBRATION_METHOD": "isotonic",
            "LABEL_FIELD": "label",
            "SCORE_FIELD": "prob_class_1",
            "IS_BINARY": "True",
            "MONOTONIC_CONSTRAINT": "True",
            "GAM_SPLINES": "10",
            "ERROR_THRESHOLD": "0.05",
            "NUM_CLASSES": "2",
            "SCORE_FIELD_PREFIX": "prob_class_",
            "MULTICLASS_CATEGORIES": None,
        }

        # Run main function
        main(input_paths, output_paths, environ_vars)

        # Check that output files were created
        calibration_dir = Path(output_paths["calibration"])
        metrics_dir = Path(output_paths["metrics"])
        calibrated_dir = Path(output_paths["calibrated_data"])

        assert calibration_dir.exists()
        assert metrics_dir.exists()
        assert calibrated_dir.exists()

        # Check specific output files
        assert (calibration_dir / "calibration_summary.json").exists()
        assert (metrics_dir / "calibration_metrics.json").exists()
        assert (calibrated_dir / "calibrated_data.parquet").exists()

        # Verify joblib.dump was called to save the calibrator
        mock_joblib_dump.assert_called_once()

        # Verify plot function was called
        mock_plot.assert_called_once()

    @patch("cursus.steps.scripts.model_calibration.plot_multiclass_reliability_diagram")
    @patch("cursus.steps.scripts.model_calibration.joblib.dump")
    def test_main_multiclass_calibration(self, mock_joblib_dump, mock_plot, setup_main_data):
        """Test main function for multiclass calibration."""
        temp_dir, df, n_samples = setup_main_data
        
        # Create multiclass test data
        n_classes = 3
        multiclass_df = pd.DataFrame(
            {
                "label": np.random.choice(n_classes, n_samples),
                "prob_class_0": np.random.uniform(0, 1, n_samples),
                "prob_class_1": np.random.uniform(0, 1, n_samples),
                "prob_class_2": np.random.uniform(0, 1, n_samples),
            }
        )

        # Normalize probabilities to sum to 1
        prob_cols = ["prob_class_0", "prob_class_1", "prob_class_2"]
        multiclass_df[prob_cols] = multiclass_df[prob_cols].div(
            multiclass_df[prob_cols].sum(axis=1), axis=0
        )

        # Save multiclass data (remove existing binary data first)
        input_dir = Path(temp_dir / "input")
        # Remove existing binary data file
        existing_files = list(input_dir.glob("*.csv"))
        for f in existing_files:
            f.unlink()
        multiclass_df.to_csv(input_dir / "multiclass_data.csv", index=False)

        mock_plot.return_value = str(temp_dir / "plot.png")

        # Set up input and output paths
        input_paths = {"eval_data": str(input_dir)}
        output_paths = {
            "calibration": str(temp_dir / "calibration"),
            "metrics": str(temp_dir / "metrics"),
            "calibrated_data": str(temp_dir / "calibrated"),
        }
        environ_vars = {
            "CALIBRATION_METHOD": "isotonic",
            "LABEL_FIELD": "label",
            "SCORE_FIELD": "prob_class_1",
            "IS_BINARY": "False",
            "MONOTONIC_CONSTRAINT": "True",
            "GAM_SPLINES": "10",
            "ERROR_THRESHOLD": "0.05",
            "NUM_CLASSES": "3",
            "SCORE_FIELD_PREFIX": "prob_class_",
            "MULTICLASS_CATEGORIES": '["0", "1", "2"]',
        }

        # Run main function
        main(input_paths, output_paths, environ_vars)

        # Check that output files were created
        calibration_dir = Path(output_paths["calibration"])
        metrics_dir = Path(output_paths["metrics"])
        calibrated_dir = Path(output_paths["calibrated_data"])

        assert (calibration_dir / "calibration_summary.json").exists()
        assert (metrics_dir / "calibration_metrics.json").exists()
        assert (calibrated_dir / "calibrated_data.parquet").exists()

        # For multiclass, multiple calibrators should be saved
        assert mock_joblib_dump.call_count == n_classes

        # Verify plot function was called
        mock_plot.assert_called_once()

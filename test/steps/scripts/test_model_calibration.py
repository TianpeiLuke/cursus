import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import tempfile
import shutil
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Import the functions to be tested
from src.cursus.steps.scripts.model_calibration import (
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
    main
)


class TestCalibrationConfig(unittest.TestCase):
    """Tests for the CalibrationConfig class."""

    def test_calibration_config_defaults(self):
        """Test CalibrationConfig with default values."""
        config = CalibrationConfig()
        
        self.assertEqual(config.calibration_method, "gam")
        self.assertEqual(config.label_field, "label")
        self.assertEqual(config.score_field, "prob_class_1")
        self.assertTrue(config.is_binary)
        self.assertTrue(config.monotonic_constraint)
        self.assertEqual(config.gam_splines, 10)
        self.assertEqual(config.error_threshold, 0.05)
        self.assertEqual(config.num_classes, 2)

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
            multiclass_categories=["A", "B", "C"]
        )
        
        self.assertEqual(config.calibration_method, "isotonic")
        self.assertEqual(config.label_field, "target")
        self.assertEqual(config.score_field, "prediction")
        self.assertFalse(config.is_binary)
        self.assertFalse(config.monotonic_constraint)
        self.assertEqual(config.gam_splines, 20)
        self.assertEqual(config.error_threshold, 0.1)
        self.assertEqual(config.num_classes, 3)
        self.assertEqual(config.multiclass_categories, ["A", "B", "C"])

    @patch.dict(os.environ, {
        'CALIBRATION_METHOD': 'platt',
        'LABEL_FIELD': 'y_true',
        'SCORE_FIELD': 'y_score',
        'IS_BINARY': 'false',
        'MONOTONIC_CONSTRAINT': 'false',
        'GAM_SPLINES': '15',
        'ERROR_THRESHOLD': '0.02',
        'NUM_CLASSES': '4',
        'MULTICLASS_CATEGORIES': '["class1", "class2", "class3", "class4"]'
    })
    def test_calibration_config_from_env(self):
        """Test CalibrationConfig.from_env() with environment variables."""
        config = CalibrationConfig.from_env()
        
        self.assertEqual(config.calibration_method, "platt")
        self.assertEqual(config.label_field, "y_true")
        self.assertEqual(config.score_field, "y_score")
        self.assertFalse(config.is_binary)
        self.assertFalse(config.monotonic_constraint)
        self.assertEqual(config.gam_splines, 15)
        self.assertEqual(config.error_threshold, 0.02)
        self.assertEqual(config.num_classes, 4)
        self.assertEqual(config.multiclass_categories, ["class1", "class2", "class3", "class4"])


class TestCalibrationHelpers(unittest.TestCase):
    """Tests for calibration helper functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = CalibrationConfig(
            input_data_path=str(self.temp_dir / "input"),
            output_calibration_path=str(self.temp_dir / "calibration"),
            output_metrics_path=str(self.temp_dir / "metrics"),
            output_calibrated_data_path=str(self.temp_dir / "calibrated")
        )

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_create_directories(self):
        """Test create_directories function."""
        create_directories(self.config)
        
        self.assertTrue(Path(self.config.output_calibration_path).exists())
        self.assertTrue(Path(self.config.output_metrics_path).exists())
        self.assertTrue(Path(self.config.output_calibrated_data_path).exists())

    def test_find_first_data_file_csv(self):
        """Test finding CSV data file."""
        input_dir = Path(self.config.input_data_path)
        input_dir.mkdir(parents=True)
        
        # Create test files
        (input_dir / "data.csv").write_text("col1,col2\n1,2\n")
        (input_dir / "other.txt").write_text("not a data file")
        
        result = find_first_data_file(config=self.config)
        self.assertTrue(result.endswith("data.csv"))

    def test_find_first_data_file_parquet(self):
        """Test finding Parquet data file."""
        input_dir = Path(self.config.input_data_path)
        input_dir.mkdir(parents=True)
        
        # Create a dummy parquet file (just touch it for the test)
        (input_dir / "data.parquet").touch()
        
        result = find_first_data_file(config=self.config)
        self.assertTrue(result.endswith("data.parquet"))

    def test_find_first_data_file_no_files(self):
        """Test finding data file when none exist."""
        input_dir = Path(self.config.input_data_path)
        input_dir.mkdir(parents=True)
        
        with self.assertRaises(FileNotFoundError):
            find_first_data_file(config=self.config)

    def test_find_first_data_file_no_directory(self):
        """Test finding data file when directory doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            find_first_data_file(config=self.config)

    @patch('src.cursus.steps.scripts.model_calibration.pd.read_csv')
    @patch('src.cursus.steps.scripts.model_calibration.find_first_data_file')
    def test_load_data_csv(self, mock_find_file, mock_read_csv):
        """Test loading CSV data."""
        mock_find_file.return_value = "/path/to/data.csv"
        mock_df = pd.DataFrame({
            'label': [0, 1, 0, 1],
            'prob_class_1': [0.2, 0.8, 0.3, 0.9]
        })
        mock_read_csv.return_value = mock_df
        
        result = load_data(self.config)
        
        mock_find_file.assert_called_once()
        mock_read_csv.assert_called_once_with("/path/to/data.csv")
        pd.testing.assert_frame_equal(result, mock_df)

    @patch('src.cursus.steps.scripts.model_calibration.pd.read_parquet')
    @patch('src.cursus.steps.scripts.model_calibration.find_first_data_file')
    def test_load_data_parquet(self, mock_find_file, mock_read_parquet):
        """Test loading Parquet data."""
        mock_find_file.return_value = "/path/to/data.parquet"
        mock_df = pd.DataFrame({
            'label': [0, 1, 0, 1],
            'prob_class_1': [0.2, 0.8, 0.3, 0.9]
        })
        mock_read_parquet.return_value = mock_df
        
        result = load_data(self.config)
        
        mock_find_file.assert_called_once()
        mock_read_parquet.assert_called_once_with("/path/to/data.parquet")
        pd.testing.assert_frame_equal(result, mock_df)

    @patch('src.cursus.steps.scripts.model_calibration.find_first_data_file')
    def test_load_data_missing_label_field(self, mock_find_file):
        """Test loading data with missing label field."""
        input_dir = Path(self.config.input_data_path)
        input_dir.mkdir(parents=True)
        csv_file = input_dir / "data.csv"
        
        # Create CSV without label field
        df = pd.DataFrame({'other_col': [1, 2, 3], 'prob_class_1': [0.1, 0.5, 0.9]})
        df.to_csv(csv_file, index=False)
        
        mock_find_file.return_value = str(csv_file)
        
        with self.assertRaises(ValueError) as context:
            load_data(self.config)
        
        self.assertIn("Label field 'label' not found", str(context.exception))

    @patch('src.cursus.steps.scripts.model_calibration.find_first_data_file')
    def test_load_data_missing_score_field_binary(self, mock_find_file):
        """Test loading binary data with missing score field."""
        input_dir = Path(self.config.input_data_path)
        input_dir.mkdir(parents=True)
        csv_file = input_dir / "data.csv"
        
        # Create CSV without score field
        df = pd.DataFrame({'label': [0, 1, 0], 'other_col': [0.1, 0.5, 0.9]})
        df.to_csv(csv_file, index=False)
        
        mock_find_file.return_value = str(csv_file)
        
        with self.assertRaises(ValueError) as context:
            load_data(self.config)
        
        self.assertIn("Score field 'prob_class_1' not found", str(context.exception))


class TestCalibrationMethods(unittest.TestCase):
    """Tests for calibration training methods."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_samples = 100
        self.scores = np.random.uniform(0, 1, self.n_samples)
        self.labels = (self.scores > 0.5).astype(int)
        self.config = CalibrationConfig()

    def test_train_isotonic_calibration(self):
        """Test training isotonic regression calibration."""
        calibrator = train_isotonic_calibration(self.scores, self.labels, self.config)
        
        # Test that we get a calibrator object
        self.assertIsNotNone(calibrator)
        
        # Test that it can make predictions
        calibrated_scores = calibrator.transform(self.scores)
        self.assertEqual(len(calibrated_scores), len(self.scores))
        self.assertTrue(np.all(calibrated_scores >= 0))
        self.assertTrue(np.all(calibrated_scores <= 1))

    def test_train_platt_scaling(self):
        """Test training Platt scaling calibration."""
        calibrator = train_platt_scaling(self.scores, self.labels, self.config)
        
        # Test that we get a calibrator object
        self.assertIsNotNone(calibrator)
        
        # Test that it can make predictions
        calibrated_scores = calibrator.predict_proba(self.scores.reshape(-1, 1))[:, 1]
        self.assertEqual(len(calibrated_scores), len(self.scores))
        self.assertTrue(np.all(calibrated_scores >= 0))
        self.assertTrue(np.all(calibrated_scores <= 1))

    @patch('src.cursus.steps.scripts.model_calibration.HAS_PYGAM', True)
    def test_train_gam_calibration_with_pygam(self):
        """Test training GAM calibration when pygam is available."""
        mock_gam = MagicMock()
        mock_gam.statistics_ = {'deviance': 100.0}
        mock_s = MagicMock()
        
        # Mock both LogisticGAM and s function from pygam
        with patch('src.cursus.steps.scripts.model_calibration.LogisticGAM', return_value=mock_gam, create=True) as mock_gam_class, \
             patch('src.cursus.steps.scripts.model_calibration.s', return_value=mock_s, create=True):
            result = train_gam_calibration(self.scores, self.labels, self.config)
            
            self.assertEqual(result, mock_gam)
            mock_gam.fit.assert_called_once()
            mock_gam_class.assert_called_once()

    @patch('src.cursus.steps.scripts.model_calibration.HAS_PYGAM', False)
    def test_train_gam_calibration_without_pygam(self):
        """Test training GAM calibration when pygam is not available."""
        with self.assertRaises(ImportError):
            train_gam_calibration(self.scores, self.labels, self.config)


class TestMulticlassCalibration(unittest.TestCase):
    """Tests for multiclass calibration methods."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_samples = 100
        self.n_classes = 3
        self.y_prob_matrix = np.random.dirichlet([1, 1, 1], self.n_samples)
        self.y_true = np.random.choice(self.n_classes, self.n_samples)
        self.config = CalibrationConfig(
            is_binary=False,
            num_classes=self.n_classes,
            multiclass_categories=["class_0", "class_1", "class_2"]
        )

    def test_train_multiclass_calibration_isotonic(self):
        """Test training multiclass calibration with isotonic regression."""
        calibrators = train_multiclass_calibration(
            self.y_prob_matrix, self.y_true, "isotonic", self.config
        )
        
        self.assertEqual(len(calibrators), self.n_classes)
        
        # Test that each calibrator can make predictions
        for calibrator in calibrators:
            test_scores = np.random.uniform(0, 1, 10)
            calibrated = calibrator.transform(test_scores)
            self.assertEqual(len(calibrated), 10)

    def test_train_multiclass_calibration_platt(self):
        """Test training multiclass calibration with Platt scaling."""
        calibrators = train_multiclass_calibration(
            self.y_prob_matrix, self.y_true, "platt", self.config
        )
        
        self.assertEqual(len(calibrators), self.n_classes)
        
        # Test that each calibrator can make predictions
        for calibrator in calibrators:
            test_scores = np.random.uniform(0, 1, 10).reshape(-1, 1)
            calibrated = calibrator.predict_proba(test_scores)[:, 1]
            self.assertEqual(len(calibrated), 10)

    def test_apply_multiclass_calibration(self):
        """Test applying multiclass calibration."""
        # Train calibrators first
        calibrators = train_multiclass_calibration(
            self.y_prob_matrix, self.y_true, "isotonic", self.config
        )
        
        # Apply calibration
        calibrated_probs = apply_multiclass_calibration(
            self.y_prob_matrix, calibrators, self.config
        )
        
        # Check output shape and properties
        self.assertEqual(calibrated_probs.shape, self.y_prob_matrix.shape)
        
        # Check that probabilities sum to 1 (approximately)
        row_sums = calibrated_probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-10)
        
        # Check that all probabilities are between 0 and 1
        self.assertTrue(np.all(calibrated_probs >= 0))
        self.assertTrue(np.all(calibrated_probs <= 1))


class TestCalibrationMetrics(unittest.TestCase):
    """Tests for calibration metrics computation."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_samples = 100
        self.y_true = np.random.choice([0, 1], self.n_samples)
        self.y_prob = np.random.uniform(0, 1, self.n_samples)

    def test_compute_calibration_metrics(self):
        """Test computing calibration metrics."""
        metrics = compute_calibration_metrics(self.y_true, self.y_prob)
        
        # Check that all expected metrics are present
        expected_keys = [
            "expected_calibration_error",
            "maximum_calibration_error", 
            "brier_score",
            "auc_roc",
            "reliability_diagram",
            "bin_statistics",
            "num_samples",
            "num_bins"
        ]
        
        for key in expected_keys:
            self.assertIn(key, metrics)
        
        # Check metric value ranges
        self.assertGreaterEqual(metrics["expected_calibration_error"], 0)
        self.assertLessEqual(metrics["expected_calibration_error"], 1)
        self.assertGreaterEqual(metrics["maximum_calibration_error"], 0)
        self.assertLessEqual(metrics["maximum_calibration_error"], 1)
        self.assertGreaterEqual(metrics["brier_score"], 0)
        self.assertLessEqual(metrics["brier_score"], 1)
        self.assertGreaterEqual(metrics["auc_roc"], 0)
        self.assertLessEqual(metrics["auc_roc"], 1)
        self.assertEqual(metrics["num_samples"], self.n_samples)

    def test_compute_multiclass_calibration_metrics(self):
        """Test computing multiclass calibration metrics."""
        n_classes = 3
        y_true = np.random.choice(n_classes, self.n_samples)
        y_prob_matrix = np.random.dirichlet([1, 1, 1], self.n_samples)
        
        config = CalibrationConfig(
            is_binary=False,
            num_classes=n_classes,
            multiclass_categories=["class_0", "class_1", "class_2"]
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
            "num_classes"
        ]
        
        for key in expected_keys:
            self.assertIn(key, metrics)
        
        # Check per-class metrics
        self.assertEqual(len(metrics["per_class_metrics"]), n_classes)
        self.assertEqual(metrics["num_classes"], n_classes)
        self.assertEqual(metrics["num_samples"], self.n_samples)


class TestCalibrationVisualization(unittest.TestCase):
    """Tests for calibration visualization functions."""

    def setUp(self):
        """Set up test data."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = CalibrationConfig(
            output_metrics_path=str(self.temp_dir)
        )
        
        np.random.seed(42)
        self.n_samples = 100
        self.y_true = np.random.choice([0, 1], self.n_samples)
        self.y_prob_uncalibrated = np.random.uniform(0, 1, self.n_samples)
        self.y_prob_calibrated = np.random.uniform(0, 1, self.n_samples)

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    @patch('src.cursus.steps.scripts.model_calibration.plt')
    def test_plot_reliability_diagram(self, mock_plt):
        """Test plotting reliability diagram."""
        mock_fig = MagicMock()
        mock_plt.figure.return_value = mock_fig
        mock_plt.subplot2grid.return_value = MagicMock()
        
        result_path = plot_reliability_diagram(
            self.y_true, self.y_prob_uncalibrated, self.y_prob_calibrated, config=self.config
        )
        
        expected_path = str(self.temp_dir / "reliability_diagram.png")
        self.assertEqual(result_path, expected_path)
        
        # Verify that plotting functions were called
        mock_plt.figure.assert_called_once()
        mock_plt.savefig.assert_called_once_with(expected_path)
        mock_plt.close.assert_called_once_with(mock_fig)

    @patch('src.cursus.steps.scripts.model_calibration.plt')
    def test_plot_multiclass_reliability_diagram(self, mock_plt):
        """Test plotting multiclass reliability diagram."""
        n_classes = 3
        y_true = np.random.choice(n_classes, self.n_samples)
        y_prob_uncalibrated = np.random.dirichlet([1, 1, 1], self.n_samples)
        y_prob_calibrated = np.random.dirichlet([1, 1, 1], self.n_samples)
        
        config = CalibrationConfig(
            output_metrics_path=str(self.temp_dir),
            is_binary=False,
            num_classes=n_classes,
            multiclass_categories=["class_0", "class_1", "class_2"]
        )
        
        mock_fig = MagicMock()
        mock_axes = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        
        result_path = plot_multiclass_reliability_diagram(
            y_true, y_prob_uncalibrated, y_prob_calibrated, config=config
        )
        
        expected_path = str(self.temp_dir / "multiclass_reliability_diagram.png")
        self.assertEqual(result_path, expected_path)
        
        # Verify that plotting functions were called
        mock_plt.subplots.assert_called_once()
        mock_plt.savefig.assert_called_once_with(expected_path)
        mock_plt.close.assert_called_once_with(mock_fig)


class TestCalibrationMain(unittest.TestCase):
    """Tests for the main calibration function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create test data
        np.random.seed(42)
        self.n_samples = 100
        self.df = pd.DataFrame({
            'label': np.random.choice([0, 1], self.n_samples),
            'prob_class_1': np.random.uniform(0, 1, self.n_samples)
        })
        
        # Create input data file
        input_dir = self.temp_dir / "input"
        input_dir.mkdir(parents=True)
        self.df.to_csv(input_dir / "data.csv", index=False)
        
        self.config = CalibrationConfig(
            input_data_path=str(input_dir),
            output_calibration_path=str(self.temp_dir / "calibration"),
            output_metrics_path=str(self.temp_dir / "metrics"),
            output_calibrated_data_path=str(self.temp_dir / "calibrated"),
            calibration_method="isotonic"
        )

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    @patch('src.cursus.steps.scripts.model_calibration.plot_reliability_diagram')
    @patch('src.cursus.steps.scripts.model_calibration.joblib.dump')
    def test_main_binary_calibration(self, mock_joblib_dump, mock_plot):
        """Test main function for binary calibration."""
        mock_plot.return_value = str(self.temp_dir / "plot.png")
        
        # Set up input and output paths
        input_paths = {
            "eval_data": str(self.temp_dir / "input")
        }
        output_paths = {
            "calibration": str(self.temp_dir / "calibration"),
            "metrics": str(self.temp_dir / "metrics"),
            "calibrated_data": str(self.temp_dir / "calibrated")
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
            "MULTICLASS_CATEGORIES": None
        }
        
        # Run main function
        main(input_paths, output_paths, environ_vars)
        
        # Check that output files were created
        calibration_dir = Path(output_paths["calibration"])
        metrics_dir = Path(output_paths["metrics"])
        calibrated_dir = Path(output_paths["calibrated_data"])
        
        self.assertTrue(calibration_dir.exists())
        self.assertTrue(metrics_dir.exists())
        self.assertTrue(calibrated_dir.exists())
        
        # Check specific output files
        self.assertTrue((calibration_dir / "calibration_summary.json").exists())
        self.assertTrue((metrics_dir / "calibration_metrics.json").exists())
        self.assertTrue((calibrated_dir / "calibrated_data.parquet").exists())
        
        # Verify joblib.dump was called to save the calibrator
        mock_joblib_dump.assert_called_once()
        
        # Verify plot function was called
        mock_plot.assert_called_once()

    @patch('src.cursus.steps.scripts.model_calibration.plot_multiclass_reliability_diagram')
    @patch('src.cursus.steps.scripts.model_calibration.joblib.dump')
    def test_main_multiclass_calibration(self, mock_joblib_dump, mock_plot):
        """Test main function for multiclass calibration."""
        # Create multiclass test data
        n_classes = 3
        multiclass_df = pd.DataFrame({
            'label': np.random.choice(n_classes, self.n_samples),
            'prob_class_0': np.random.uniform(0, 1, self.n_samples),
            'prob_class_1': np.random.uniform(0, 1, self.n_samples),
            'prob_class_2': np.random.uniform(0, 1, self.n_samples)
        })
        
        # Normalize probabilities to sum to 1
        prob_cols = ['prob_class_0', 'prob_class_1', 'prob_class_2']
        multiclass_df[prob_cols] = multiclass_df[prob_cols].div(
            multiclass_df[prob_cols].sum(axis=1), axis=0
        )
        
        # Save multiclass data (remove existing binary data first)
        input_dir = Path(self.temp_dir / "input")
        # Remove existing binary data file
        existing_files = list(input_dir.glob("*.csv"))
        for f in existing_files:
            f.unlink()
        multiclass_df.to_csv(input_dir / "multiclass_data.csv", index=False)
        
        mock_plot.return_value = str(self.temp_dir / "plot.png")
        
        # Set up input and output paths
        input_paths = {
            "eval_data": str(input_dir)
        }
        output_paths = {
            "calibration": str(self.temp_dir / "calibration"),
            "metrics": str(self.temp_dir / "metrics"),
            "calibrated_data": str(self.temp_dir / "calibrated")
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
            "MULTICLASS_CATEGORIES": '["0", "1", "2"]'
        }
        
        # Run main function
        main(input_paths, output_paths, environ_vars)
        
        # Check that output files were created
        calibration_dir = Path(output_paths["calibration"])
        metrics_dir = Path(output_paths["metrics"])
        calibrated_dir = Path(output_paths["calibrated_data"])
        
        self.assertTrue((calibration_dir / "calibration_summary.json").exists())
        self.assertTrue((metrics_dir / "calibration_metrics.json").exists())
        self.assertTrue((calibrated_dir / "calibrated_data.parquet").exists())
        
        # For multiclass, multiple calibrators should be saved
        self.assertEqual(mock_joblib_dump.call_count, n_classes)
        
        # Verify plot function was called
        mock_plot.assert_called_once()


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

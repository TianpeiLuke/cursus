#!/usr/bin/env python
"""
Unit Tests for Model Calibration Script.

This file contains tests for the model_calibration.py script,
focusing on both binary and multi-class calibration functionality.
"""

import os
import unittest
import numpy as np
import pandas as pd
import tempfile
import shutil
import json
import joblib
import importlib
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

# Import the module to test
from src.pipeline_scripts import model_calibration


class TestModelCalibration(unittest.TestCase):
    """Test cases for model_calibration.py script."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create temporary directories for testing
        self.temp_dir = tempfile.mkdtemp()
        self.input_data_path = os.path.join(self.temp_dir, "input", "eval_data")
        self.output_calibration_path = os.path.join(self.temp_dir, "output", "calibration")
        self.output_metrics_path = os.path.join(self.temp_dir, "output", "metrics")
        self.output_calibrated_data_path = os.path.join(self.temp_dir, "output", "calibrated_data")
        
        # Create directory structure
        os.makedirs(self.input_data_path, exist_ok=True)
        os.makedirs(self.output_calibration_path, exist_ok=True)
        os.makedirs(self.output_metrics_path, exist_ok=True)
        os.makedirs(self.output_calibrated_data_path, exist_ok=True)
        
        # Create test configuration
        self.test_config = model_calibration.CalibrationConfig(
            input_data_path=self.input_data_path,
            output_calibration_path=self.output_calibration_path,
            output_metrics_path=self.output_metrics_path,
            output_calibrated_data_path=self.output_calibrated_data_path,
            calibration_method="isotonic",
            label_field="label",
            score_field="prob_class_1",
            is_binary=True,
            monotonic_constraint=True,
            gam_splines=10,
            error_threshold=0.05
        )
        
        # Create multiclass config for multiclass tests
        self.multiclass_config = model_calibration.CalibrationConfig(
            input_data_path=self.input_data_path,
            output_calibration_path=self.output_calibration_path,
            output_metrics_path=self.output_metrics_path,
            output_calibrated_data_path=self.output_calibrated_data_path,
            calibration_method="isotonic",
            label_field="label",
            score_field="prob_class_1",
            is_binary=False,
            num_classes=3,
            score_field_prefix="prob_class_"
        )
        
        # Generate sample data for testing
        self.binary_data = self._generate_binary_test_data()
        self.multiclass_data = self._generate_multiclass_test_data()
        
        # Patch CalibrationConfig.from_env to return our test config
        self.config_patch = patch.object(
            model_calibration.CalibrationConfig, 
            'from_env', 
            return_value=self.test_config
        )
        self.config_patch.start()

    def tearDown(self):
        """Clean up after each test."""
        # Stop all patches
        self.config_patch.stop()
        
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)

    def _generate_binary_test_data(self, n_samples=100, seed=42):
        """Generate synthetic binary classification data for testing."""
        np.random.seed(seed)
        df = pd.DataFrame({
            'label': np.random.choice([0, 1], size=n_samples),
            'prob_class_1': np.clip(np.random.beta(2, 5, size=n_samples) + np.random.normal(0, 0.1, size=n_samples), 0, 1)
        })
        # Make probabilities correlate with labels but not perfectly calibrated
        df.loc[df['label'] == 1, 'prob_class_1'] = df.loc[df['label'] == 1, 'prob_class_1'] * 1.3
        df['prob_class_1'] = np.clip(df['prob_class_1'], 0, 1)
        df['prob_class_0'] = 1 - df['prob_class_1']
        return df

    def _generate_multiclass_test_data(self, n_samples=100, n_classes=3, seed=42):
        """Generate synthetic multi-class classification data for testing."""
        np.random.seed(seed)
        
        # Generate labels
        labels = np.random.choice(range(n_classes), size=n_samples)
        
        # Generate raw scores (not perfectly calibrated)
        raw_scores = np.random.random((n_samples, n_classes))
        
        # Bias scores toward the correct class
        for i in range(n_samples):
            raw_scores[i, labels[i]] += 0.5
            
        # Convert to probabilities using softmax
        exp_scores = np.exp(raw_scores)
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        
        # Create DataFrame
        df = pd.DataFrame({'label': labels})
        
        # Add probability columns
        for i in range(n_classes):
            df[f'prob_class_{i}'] = probs[:, i]
            
        return df

    def _save_test_data(self, df, filename="test_data.csv"):
        """Save test data to the input directory."""
        filepath = os.path.join(self.input_data_path, filename)
        df.to_csv(filepath, index=False)
        return filepath

    def test_find_first_data_file(self):
        """Test the find_first_data_file function."""
        # Create test files
        self._save_test_data(self.binary_data, "data1.csv")
        self._save_test_data(self.binary_data, "data2.parquet")
        
        # Test finding CSV file
        found_file = model_calibration.find_first_data_file(self.input_data_path)
        self.assertTrue(found_file.endswith('data1.csv'))
        
        # Test handling directory that doesn't exist
        with self.assertRaises(FileNotFoundError):
            model_calibration.find_first_data_file(os.path.join(self.temp_dir, "nonexistent"))

    def test_create_directories(self):
        """Test the create_directories function."""
        # Remove output directories
        shutil.rmtree(self.output_calibration_path)
        shutil.rmtree(self.output_metrics_path)
        shutil.rmtree(self.output_calibrated_data_path)
        
        # Recreate using the function
        model_calibration.create_directories()
        
        # Check directories exist
        self.assertTrue(os.path.exists(self.output_calibration_path))
        self.assertTrue(os.path.exists(self.output_metrics_path))
        self.assertTrue(os.path.exists(self.output_calibrated_data_path))

    def test_load_data_binary(self):
        """Test loading binary classification data."""
        # Save test data
        self._save_test_data(self.binary_data, "binary_data.csv")
        
        # Load data
        df = model_calibration.load_data()
        
        # Check data is loaded correctly
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue('label' in df.columns)
        self.assertTrue('prob_class_1' in df.columns)
        self.assertEqual(len(df), len(self.binary_data))

    def test_load_data_missing_columns(self):
        """Test error handling when required columns are missing."""
        # Create data with missing columns
        bad_data = self.binary_data.drop(columns=['prob_class_1'])
        self._save_test_data(bad_data, "bad_data.csv")
        
        # Should raise ValueError due to missing score column
        with self.assertRaises(ValueError):
            model_calibration.load_data()

    def test_load_and_prepare_data_binary(self):
        """Test preparation of binary classification data."""
        # Save test data
        self._save_test_data(self.binary_data, "binary_data.csv")
        
        # Load and prepare
        df, y_true, y_prob, _ = model_calibration.load_and_prepare_data()
        
        # Check data is prepared correctly
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(y_true, np.ndarray)
        self.assertIsInstance(y_prob, np.ndarray)
        self.assertEqual(len(y_true), len(self.binary_data))
        self.assertEqual(len(y_prob), len(self.binary_data))

    def test_load_and_prepare_data_multiclass(self):
        """Test preparation of multi-class classification data."""
        # Save test data
        self._save_test_data(self.multiclass_data, "multiclass_data.csv")
        
        # Temporarily patch the from_env method to return multiclass config
        with patch.object(model_calibration.CalibrationConfig, 'from_env', return_value=self.multiclass_config):
            # Load and prepare with multiclass config
            df, y_true, _, y_prob_matrix = model_calibration.load_and_prepare_data()
        
        # Check data is prepared correctly
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(y_true, np.ndarray)
        self.assertIsInstance(y_prob_matrix, np.ndarray)
        self.assertEqual(len(y_true), len(self.multiclass_data))
        self.assertEqual(y_prob_matrix.shape, (len(self.multiclass_data), 3))

    def test_train_isotonic_calibration(self):
        """Test training isotonic regression calibration model."""
        # Generate sample data
        scores = np.random.random(100)
        labels = np.random.choice([0, 1], size=100)
        
        # Train calibration model
        calibrator = model_calibration.train_isotonic_calibration(scores, labels)
        
        # Check if calibrator is created correctly
        self.assertIsNotNone(calibrator)
        from sklearn.isotonic import IsotonicRegression
        self.assertIsInstance(calibrator, IsotonicRegression)

    def test_train_platt_scaling(self):
        """Test training Platt scaling (logistic regression) calibration model."""
        # Generate sample data
        scores = np.random.random(100)
        labels = np.random.choice([0, 1], size=100)
        
        # Train calibration model
        calibrator = model_calibration.train_platt_scaling(scores, labels)
        
        # Check if calibrator is created correctly
        self.assertIsNotNone(calibrator)
        from sklearn.linear_model import LogisticRegression
        self.assertIsInstance(calibrator, LogisticRegression)

    def test_compute_calibration_metrics(self):
        """Test computation of calibration metrics."""
        # Generate sample data
        y_true = np.random.choice([0, 1], size=100)
        y_prob = np.random.random(100)
        
        # Compute metrics
        metrics = model_calibration.compute_calibration_metrics(y_true, y_prob)
        
        # Check if metrics are computed correctly
        self.assertIsNotNone(metrics)
        self.assertIn('expected_calibration_error', metrics)
        self.assertIn('maximum_calibration_error', metrics)
        self.assertIn('brier_score', metrics)
        self.assertIn('auc_roc', metrics)
        
    def test_binary_calibration_workflow(self):
        """Test the complete binary calibration workflow."""
        # Save test data
        self._save_test_data(self.binary_data, "binary_data.csv")
        
        # Mock plot_reliability_diagram to avoid saving actual plots
        with patch.object(model_calibration, 'plot_reliability_diagram', return_value='mock_plot_path.png'):
            # Run main function
            model_calibration.main()
            
            # Check that output files are created
            self.assertTrue(os.path.exists(os.path.join(self.output_calibration_path, "calibration_model.joblib")))
            self.assertTrue(os.path.exists(os.path.join(self.output_metrics_path, "calibration_metrics.json")))
            self.assertTrue(os.path.exists(os.path.join(self.output_calibrated_data_path, "calibrated_data.parquet")))
            
            # Load and check metrics
            with open(os.path.join(self.output_metrics_path, "calibration_metrics.json"), 'r') as f:
                metrics = json.load(f)
                self.assertIn('calibrated', metrics)
                self.assertIn('uncalibrated', metrics)
                self.assertIn('improvement', metrics)
            
            # Check that calibrator was saved correctly
            calibrator = joblib.load(os.path.join(self.output_calibration_path, "calibration_model.joblib"))
            self.assertIsNotNone(calibrator)

    def test_multiclass_calibration_workflow(self):
        """Test the complete multi-class calibration workflow."""
        # Save test data
        self._save_test_data(self.multiclass_data, "multiclass_data.csv")
        
        # Temporarily patch the from_env method to return multiclass config
        with patch.object(model_calibration.CalibrationConfig, 'from_env', return_value=self.multiclass_config):
            # Mock plot functions to avoid saving actual plots
            with patch.object(model_calibration, 'plot_multiclass_reliability_diagram', return_value='mock_plot_path.png'):
                # Run main function with multiclass config
                model_calibration.main(self.multiclass_config)
            
            # Check that output files are created
            calibrator_dir = os.path.join(self.output_calibration_path, "calibration_models")
            self.assertTrue(os.path.exists(calibrator_dir))
            self.assertTrue(os.path.exists(os.path.join(self.output_metrics_path, "calibration_metrics.json")))
            self.assertTrue(os.path.exists(os.path.join(self.output_calibrated_data_path, "calibrated_data.parquet")))
            
            # Check that at least one calibrator per class was created
            calibrators = list(Path(calibrator_dir).glob("calibration_model_class_*.joblib"))
            self.assertGreaterEqual(len(calibrators), 3)
            
            # Load and check metrics
            with open(os.path.join(self.output_metrics_path, "calibration_metrics.json"), 'r') as f:
                metrics = json.load(f)
                self.assertEqual(metrics['mode'], 'multi-class')
                self.assertIn('num_classes', metrics)
                self.assertEqual(metrics['num_classes'], 3)

    def test_exception_handling(self):
        """Test exception handling in main function."""
        # Create invalid data that will cause an error
        invalid_df = pd.DataFrame({'wrong_column': [1, 2, 3]})
        self._save_test_data(invalid_df, "invalid_data.csv")
        
        # Mock sys.exit to prevent test from actually exiting
        with patch('sys.exit') as mock_exit:
            model_calibration.main()
            # Check that sys.exit was called with error code 1
            mock_exit.assert_called_once_with(1)


if __name__ == '__main__':
    unittest.main()

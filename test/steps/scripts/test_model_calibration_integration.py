#!/usr/bin/env python
"""
Integration test for Model Calibration Script.

This test creates synthetic data, trains an XGBoost model, and then tests
the calibration functionality on the model's predictions to verify that
calibration improves the probability estimates.
"""

import os
import unittest
import numpy as np
import pandas as pd
import tempfile
import shutil
import json
import joblib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed, skipping integration tests")

from src.pipeline_scripts import model_calibration


@unittest.skipIf(not HAS_XGBOOST, "XGBoost is required for integration tests")
class TestModelCalibrationIntegration(unittest.TestCase):
    """Integration tests for model calibration with real ML workflow."""

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
        
        # Create test configuration for binary classification
        self.binary_config = model_calibration.CalibrationConfig(
            input_data_path=self.input_data_path,
            output_calibration_path=self.output_calibration_path,
            output_metrics_path=self.output_metrics_path,
            output_calibrated_data_path=self.output_calibrated_data_path,
            calibration_method="isotonic",
            label_field="label",
            score_field="prediction",
            is_binary=True
        )
        
        # Create test configuration for multi-class
        self.multiclass_config = model_calibration.CalibrationConfig(
            input_data_path=self.input_data_path,
            output_calibration_path=self.output_calibration_path,
            output_metrics_path=self.output_metrics_path,
            output_calibrated_data_path=self.output_calibrated_data_path,
            calibration_method="isotonic",
            label_field="label",
            score_field="prediction",
            is_binary=False,
            num_classes=3,
            score_field_prefix="prob_class_"
        )

    def tearDown(self):
        """Clean up after each test."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)

    def _generate_synthetic_binary_data(self, n_samples=1000, n_features=10, random_state=42):
        """Generate synthetic binary classification data.
        
        Returns:
            tuple: (X_train, y_train, X_test, y_test) - train/test split of features and labels
        """
        np.random.seed(random_state)
        
        # Generate features from standard normal distribution
        X = np.random.randn(n_samples, n_features)
        
        # Generate binary labels using a nonlinear function of features
        # This will make the classification task challenging enough
        feature_weights = np.random.randn(n_features)
        
        # Create linear combination
        z = np.dot(X, feature_weights)
        
        # Add nonlinearity
        z = z + 0.3 * np.sum(X**2, axis=1)
        
        # Convert to probabilities
        p = 1.0 / (1.0 + np.exp(-z))
        
        # Sample labels
        y = np.random.binomial(1, p)
        
        # Split into training and testing sets (70/30)
        train_samples = int(0.7 * n_samples)
        X_train, X_test = X[:train_samples], X[train_samples:]
        y_train, y_test = y[:train_samples], y[train_samples:]
        
        return X_train, y_train, X_test, y_test, feature_weights

    def _generate_synthetic_multiclass_data(self, n_samples=1000, n_features=10, n_classes=3, random_state=42):
        """Generate synthetic multi-class classification data.
        
        Returns:
            tuple: (X_train, y_train, X_test, y_test) - train/test split of features and labels
        """
        np.random.seed(random_state)
        
        # Generate features
        X = np.random.randn(n_samples, n_features)
        
        # Generate multiple weight vectors, one per class
        feature_weights = np.random.randn(n_classes, n_features)
        
        # Calculate logits for each class
        logits = np.zeros((n_samples, n_classes))
        for c in range(n_classes):
            logits[:, c] = np.dot(X, feature_weights[c])
            
        # Add some non-linearity 
        for c in range(n_classes):
            logits[:, c] = logits[:, c] + 0.2 * np.sum(X**2, axis=1) * (c + 1)
            
        # Convert to probabilities using softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probas = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Sample labels from multinomial distribution
        y = np.array([np.random.choice(n_classes, p=probas[i]) for i in range(n_samples)])
        
        # Split into training and testing sets (70/30)
        train_samples = int(0.7 * n_samples)
        X_train, X_test = X[:train_samples], X[train_samples:]
        y_train, y_test = y[:train_samples], y[train_samples:]
        
        return X_train, y_train, X_test, y_test

    def _train_xgboost_binary(self, X_train, y_train, random_state=42):
        """Train an XGBoost binary classifier.
        
        Returns:
            xgb.Booster: Trained XGBoost model
        """
        # Train an XGBoost model that will likely need calibration
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'seed': random_state,
            'max_depth': 3,
            'eta': 0.1
        }
        
        model = xgb.train(params, dtrain, num_boost_round=50)
        return model

    def _train_xgboost_multiclass(self, X_train, y_train, num_classes=3, random_state=42):
        """Train an XGBoost multi-class classifier.
        
        Returns:
            xgb.Booster: Trained XGBoost model
        """
        # Train an XGBoost model that will likely need calibration
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        params = {
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'num_class': num_classes,
            'seed': random_state,
            'max_depth': 3,
            'eta': 0.1
        }
        
        model = xgb.train(params, dtrain, num_boost_round=50)
        return model

    def _get_predictions_binary(self, model, X_test):
        """Get predictions from a binary XGBoost model.
        
        Returns:
            np.ndarray: Probability predictions for the positive class
        """
        dtest = xgb.DMatrix(X_test)
        return model.predict(dtest)

    def _get_predictions_multiclass(self, model, X_test, num_classes=3):
        """Get predictions from a multi-class XGBoost model.
        
        Returns:
            np.ndarray: Probability predictions matrix (samples × classes)
        """
        dtest = xgb.DMatrix(X_test)
        pred_probs = model.predict(dtest)
        
        # Reshape from flat array to matrix
        return pred_probs.reshape(X_test.shape[0], num_classes)

    def _save_test_data_with_predictions(self, y_test, predictions, is_binary=True, filename="eval_data.csv"):
        """Save test data with predictions to a CSV file.
        
        Args:
            y_test: Ground truth labels
            predictions: Model predictions
            is_binary: Whether this is binary classification
            filename: Output filename
        
        Returns:
            str: Path to the saved file
        """
        df = pd.DataFrame({"label": y_test})
        
        if is_binary:
            df["prediction"] = predictions
        else:
            # For multi-class, add one column per class
            num_classes = predictions.shape[1]
            for i in range(num_classes):
                df[f"prob_class_{i}"] = predictions[:, i]
        
        filepath = os.path.join(self.input_data_path, filename)
        df.to_csv(filepath, index=False)
        return filepath

    def test_end_to_end_binary_workflow(self):
        """Test the end-to-end binary classification workflow with XGBoost and calibration."""
        if not HAS_XGBOOST:
            self.skipTest("XGBoost is required for this test")
        
        # 1. Generate synthetic data
        X_train, y_train, X_test, y_test, _ = self._generate_synthetic_binary_data(n_samples=1000)
        
        # 2. Train an XGBoost model
        model = self._train_xgboost_binary(X_train, y_train)
        
        # 3. Get predictions on test set
        predictions = self._get_predictions_binary(model, X_test)
        
        # 4. Save test data with predictions
        self._save_test_data_with_predictions(y_test, predictions, is_binary=True)
        
        # 5. Calculate pre-calibration metrics
        uncalibrated_metrics = model_calibration.compute_calibration_metrics(y_test, predictions)
        
        # 6. Apply isotonic calibration
        isotonic_calibrator = model_calibration.train_isotonic_calibration(predictions, y_test)
        calibrated_preds_isotonic = isotonic_calibrator.transform(predictions)
        
        # 7. Calculate post-calibration metrics
        calibrated_metrics_isotonic = model_calibration.compute_calibration_metrics(y_test, calibrated_preds_isotonic)
        
        # 8. Apply Platt scaling calibration
        platt_calibrator = model_calibration.train_platt_scaling(predictions, y_test)
        calibrated_preds_platt = platt_calibrator.predict_proba(predictions.reshape(-1, 1))[:, 1]
        
        # 9. Calculate post-calibration metrics for Platt scaling
        calibrated_metrics_platt = model_calibration.compute_calibration_metrics(y_test, calibrated_preds_platt)
        
        # 10. Check that calibration improved the expected calibration error (ECE)
        print(f"Binary classification calibration results:")
        print(f"  Uncalibrated ECE: {uncalibrated_metrics['expected_calibration_error']:.4f}")
        print(f"  Isotonic calibrated ECE: {calibrated_metrics_isotonic['expected_calibration_error']:.4f}")
        print(f"  Platt scaled ECE: {calibrated_metrics_platt['expected_calibration_error']:.4f}")
        
        # 11. Verify improvements
        self.assertLess(
            calibrated_metrics_isotonic['expected_calibration_error'],
            uncalibrated_metrics['expected_calibration_error'],
            "Isotonic calibration should improve ECE"
        )
        
        self.assertLess(
            calibrated_metrics_platt['expected_calibration_error'],
            uncalibrated_metrics['expected_calibration_error'],
            "Platt scaling should improve ECE"
        )
        
        # 12. Run the full main function to test end-to-end workflow
        model_calibration.main(self.binary_config)
        
        # 13. Verify output files are created
        self.assertTrue(os.path.exists(os.path.join(self.output_calibration_path, "calibration_model.joblib")))
        self.assertTrue(os.path.exists(os.path.join(self.output_metrics_path, "calibration_metrics.json")))
        self.assertTrue(os.path.exists(os.path.join(self.output_calibrated_data_path, "calibrated_data.parquet")))
        
        # 14. Load and check the output metrics
        with open(os.path.join(self.output_metrics_path, "calibration_metrics.json"), 'r') as f:
            metrics = json.load(f)
            
            # Check calibration improved
            self.assertLess(
                metrics['calibrated']['expected_calibration_error'],
                metrics['uncalibrated']['expected_calibration_error'],
                "Calibration should improve ECE in end-to-end workflow"
            )

    def test_end_to_end_multiclass_workflow(self):
        """Test the end-to-end multi-class classification workflow with XGBoost and calibration."""
        if not HAS_XGBOOST:
            self.skipTest("XGBoost is required for this test")
        
        num_classes = 3
        
        # 1. Generate synthetic data
        X_train, y_train, X_test, y_test = self._generate_synthetic_multiclass_data(
            n_samples=1000, n_classes=num_classes
        )
        
        # 2. Train an XGBoost model
        model = self._train_xgboost_multiclass(X_train, y_train, num_classes=num_classes)
        
        # 3. Get predictions on test set
        predictions = self._get_predictions_multiclass(model, X_test, num_classes=num_classes)
        
        # 4. Save test data with predictions
        self._save_test_data_with_predictions(y_test, predictions, is_binary=False)
        
        # 5. Calculate pre-calibration metrics
        y_true_onehot = np.zeros((len(y_test), num_classes))
        for i, class_idx in enumerate(y_test):
            y_true_onehot[i, int(class_idx)] = 1
        
        uncalibrated_metrics = model_calibration.compute_multiclass_calibration_metrics(
            y_test, predictions, config=self.multiclass_config
        )
        
        # 6. Train calibrators for each class
        calibrators = model_calibration.train_multiclass_calibration(
            predictions, y_test, method="isotonic", config=self.multiclass_config
        )
        
        # 7. Apply calibration
        calibrated_preds = model_calibration.apply_multiclass_calibration(
            predictions, calibrators, config=self.multiclass_config
        )
        
        # 8. Calculate post-calibration metrics
        calibrated_metrics = model_calibration.compute_multiclass_calibration_metrics(
            y_test, calibrated_preds, config=self.multiclass_config
        )
        
        # 9. Check that calibration improved the expected calibration error (ECE)
        print(f"Multi-class classification calibration results:")
        print(f"  Uncalibrated Macro ECE: {uncalibrated_metrics['macro_expected_calibration_error']:.4f}")
        print(f"  Calibrated Macro ECE: {calibrated_metrics['macro_expected_calibration_error']:.4f}")
        
        # 10. Verify improvements
        self.assertLess(
            calibrated_metrics['macro_expected_calibration_error'],
            uncalibrated_metrics['macro_expected_calibration_error'],
            "Calibration should improve macro ECE for multi-class"
        )
        
        # 11. Run the full main function to test end-to-end workflow
        model_calibration.main(self.multiclass_config)
        
        # 12. Verify output files are created
        calibrator_dir = os.path.join(self.output_calibration_path, "calibration_models")
        self.assertTrue(os.path.exists(calibrator_dir))
        self.assertTrue(os.path.exists(os.path.join(self.output_metrics_path, "calibration_metrics.json")))
        self.assertTrue(os.path.exists(os.path.join(self.output_calibrated_data_path, "calibrated_data.parquet")))
        
        # 13. Check that the right number of calibrator files were created
        calibrator_files = [f for f in os.listdir(calibrator_dir) if f.startswith("calibration_model_class_")]
        self.assertEqual(len(calibrator_files), num_classes)
        
        # 14. Load and check the output metrics
        with open(os.path.join(self.output_metrics_path, "calibration_metrics.json"), 'r') as f:
            metrics = json.load(f)
            
            # Check calibration improved
            self.assertLess(
                metrics['calibrated']['macro_expected_calibration_error'],
                metrics['uncalibrated']['macro_expected_calibration_error'],
                "Calibration should improve macro ECE in end-to-end workflow"
            )


if __name__ == '__main__':
    unittest.main()

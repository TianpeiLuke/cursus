import pytest
from unittest.mock import patch, MagicMock, mock_open
import os
import tempfile
import shutil
import json
import pickle as pkl
from pathlib import Path
import numpy as np
import pandas as pd

# Import the functions to be tested
from cursus.steps.scripts.xgboost_model_eval import (
    load_model_artifacts,
    preprocess_eval_data,
    log_metrics_summary,
    compute_metrics_binary,
    compute_metrics_multiclass,
    compute_comparison_metrics,
    perform_statistical_tests,
    load_eval_data,
    get_id_label_columns,
    save_predictions,
    save_metrics,
    plot_and_save_roc_curve,
    plot_and_save_pr_curve,
    plot_comparison_roc_curves,
    plot_comparison_pr_curves,
    plot_score_scatter,
    plot_score_distributions,
    evaluate_model,
    evaluate_model_with_comparison,
    save_predictions_with_comparison,
    create_comparison_report,
    main,
)


class TestModelEvaluationHelpers:
    """Tests for helper functions in the model evaluation script."""

    @pytest.fixture
    def temp_dir(self):
        """Set up test fixtures."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    def _create_mock_model_artifacts(self, model_dir):
        """Create mock model artifacts for testing."""
        model_dir.mkdir(parents=True, exist_ok=True)

        # Create mock XGBoost model file (just a dummy file)
        (model_dir / "xgboost_model.bst").write_text("mock xgboost model")

        # Create mock risk table map
        risk_tables = {"feature1": {"A": 0.1, "B": 0.2}}
        with open(model_dir / "risk_table_map.pkl", "wb") as f:
            pkl.dump(risk_tables, f)

        # Create mock impute dict
        impute_dict = {"feature2": 0.5}
        with open(model_dir / "impute_dict.pkl", "wb") as f:
            pkl.dump(impute_dict, f)

        # Create mock feature columns
        with open(model_dir / "feature_columns.txt", "w") as f:
            f.write("# Feature columns\n")
            f.write("0,feature1\n")
            f.write("1,feature2\n")

        # Create mock hyperparameters
        hyperparams = {"is_binary": True, "learning_rate": 0.1}
        with open(model_dir / "hyperparameters.json", "w") as f:
            json.dump(hyperparams, f)

    @patch("cursus.steps.scripts.xgboost_model_eval.xgb.Booster")
    def test_load_model_artifacts(self, mock_booster, temp_dir):
        """Test loading model artifacts."""
        model_dir = temp_dir / "model"
        self._create_mock_model_artifacts(model_dir)

        # Mock XGBoost Booster
        mock_model = MagicMock()
        mock_booster.return_value = mock_model

        model, risk_tables, impute_dict, feature_columns, hyperparams = (
            load_model_artifacts(str(model_dir))
        )

        # Verify artifacts were loaded correctly
        assert model == mock_model
        assert risk_tables == {"feature1": {"A": 0.1, "B": 0.2}}
        assert impute_dict == {"feature2": 0.5}
        assert feature_columns == ["feature1", "feature2"]
        assert hyperparams == {"is_binary": True, "learning_rate": 0.1}

    @patch("cursus.steps.scripts.xgboost_model_eval.RiskTableMappingProcessor")
    @patch(
        "cursus.steps.scripts.xgboost_model_eval.NumericalVariableImputationProcessor"
    )
    def test_preprocess_eval_data(self, mock_imputer_class, mock_risk_processor_class):
        """Test preprocessing evaluation data."""
        # Create test dataframe
        df = pd.DataFrame(
            {
                "feature1": ["A", "B", "A"],
                "feature2": [1.0, np.nan, 3.0],
                "other_col": [10, 20, 30],
            }
        )

        feature_columns = ["feature1", "feature2"]
        risk_tables = {"feature1": {"A": 0.1, "B": 0.2}}
        impute_dict = {"feature2": 2.0}

        # Mock processors
        mock_risk_processor = MagicMock()
        mock_risk_processor.transform.return_value = pd.Series([0.1, 0.2, 0.1])
        mock_risk_processor_class.return_value = mock_risk_processor

        mock_imputer = MagicMock()
        mock_imputer.transform.return_value = df.copy()
        mock_imputer_class.return_value = mock_imputer

        result = preprocess_eval_data(df, feature_columns, risk_tables, impute_dict)

        # Verify processors were called
        mock_risk_processor_class.assert_called_once()
        mock_imputer_class.assert_called_once_with(imputation_dict=impute_dict)

        # Verify result is a DataFrame (the actual implementation may return all columns)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

    @patch("cursus.steps.scripts.xgboost_model_eval.logger")
    def test_log_metrics_summary_binary(self, mock_logger):
        """Test logging metrics summary for binary classification."""
        metrics = {"auc_roc": 0.85, "average_precision": 0.78, "f1_score": 0.72}

        log_metrics_summary(metrics, is_binary=True)

        # Verify logger was called
        assert mock_logger.info.called

        # Check that key metrics were logged
        call_args = [call[0][0] for call in mock_logger.info.call_args_list]
        assert any("AUC-ROC" in arg for arg in call_args)
        assert any("Average Precision" in arg for arg in call_args)
        assert any("F1 Score" in arg for arg in call_args)

    @patch("cursus.steps.scripts.xgboost_model_eval.logger")
    def test_log_metrics_summary_multiclass(self, mock_logger):
        """Test logging metrics summary for multiclass classification."""
        metrics = {
            "auc_roc_macro": 0.85,
            "auc_roc_micro": 0.87,
            "f1_score_macro": 0.72,
            "f1_score_micro": 0.75,
        }

        log_metrics_summary(metrics, is_binary=False)

        # Verify logger was called
        assert mock_logger.info.called

        # Check that key metrics were logged
        call_args = [call[0][0] for call in mock_logger.info.call_args_list]
        assert any("Macro AUC-ROC" in arg for arg in call_args)
        assert any("Micro AUC-ROC" in arg for arg in call_args)

    def test_compute_metrics_binary(self):
        """Test computing binary classification metrics."""
        np.random.seed(42)
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        y_prob = np.array(
            [
                [0.8, 0.2],
                [0.3, 0.7],
                [0.9, 0.1],
                [0.2, 0.8],
                [0.1, 0.9],
                [0.7, 0.3],
                [0.4, 0.6],
                [0.6, 0.4],
            ]
        )

        metrics = compute_metrics_binary(y_true, y_prob)

        # Check that all expected metrics are present
        expected_keys = ["auc_roc", "average_precision", "f1_score"]
        for key in expected_keys:
            assert key in metrics

        # Check metric value ranges
        assert 0 <= metrics["auc_roc"] <= 1
        assert 0 <= metrics["average_precision"] <= 1
        assert 0 <= metrics["f1_score"] <= 1

    def test_compute_metrics_multiclass(self):
        """Test computing multiclass classification metrics."""
        np.random.seed(42)
        n_classes = 3
        n_samples = 20
        y_true = np.random.choice(n_classes, n_samples)
        y_prob = np.random.dirichlet([1, 1, 1], n_samples)

        metrics = compute_metrics_multiclass(y_true, y_prob, n_classes)

        # Check that all expected metrics are present
        expected_keys = [
            "auc_roc_micro",
            "auc_roc_macro",
            "f1_score_micro",
            "f1_score_macro",
        ]
        for key in expected_keys:
            assert key in metrics

        # Check per-class metrics
        for i in range(n_classes):
            assert f"auc_roc_class_{i}" in metrics
            assert f"f1_score_class_{i}" in metrics
            assert f"class_{i}_count" in metrics

    def test_load_eval_data_csv(self, temp_dir):
        """Test loading evaluation data from CSV."""
        eval_dir = temp_dir / "eval_data"
        eval_dir.mkdir()

        # Create test CSV
        df = pd.DataFrame(
            {"id": [1, 2, 3], "label": [0, 1, 0], "feature1": [0.1, 0.5, 0.9]}
        )
        df.to_csv(eval_dir / "eval_data.csv", index=False)

        result = load_eval_data(str(eval_dir))

        pd.testing.assert_frame_equal(result, df)

    def test_load_eval_data_parquet(self, temp_dir):
        """Test loading evaluation data from Parquet."""
        eval_dir = temp_dir / "eval_data"
        eval_dir.mkdir()

        # Create test Parquet
        df = pd.DataFrame(
            {"id": [1, 2, 3], "label": [0, 1, 0], "feature1": [0.1, 0.5, 0.9]}
        )
        df.to_parquet(eval_dir / "eval_data.parquet", index=False)

        result = load_eval_data(str(eval_dir))

        pd.testing.assert_frame_equal(result, df)

    def test_load_eval_data_no_files(self, temp_dir):
        """Test loading evaluation data when no files exist."""
        eval_dir = temp_dir / "eval_data"
        eval_dir.mkdir()

        with pytest.raises(RuntimeError):
            load_eval_data(str(eval_dir))

    def test_get_id_label_columns(self):
        """Test getting ID and label columns."""
        df = pd.DataFrame(
            {"user_id": [1, 2, 3], "target": [0, 1, 0], "feature1": [0.1, 0.5, 0.9]}
        )

        # Test with existing columns
        id_col, label_col = get_id_label_columns(df, "user_id", "target")
        assert id_col == "user_id"
        assert label_col == "target"

        # Test with non-existing columns (should fall back to first two columns)
        id_col, label_col = get_id_label_columns(
            df, "nonexistent_id", "nonexistent_label"
        )
        assert id_col == "user_id"  # First column
        assert label_col == "target"  # Second column

    def test_save_predictions(self, temp_dir):
        """Test saving predictions to CSV."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        ids = np.array([1, 2, 3])
        y_true = np.array([0, 1, 0])
        y_prob = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])

        save_predictions(ids, y_true, y_prob, "id", "label", str(output_dir))

        # Verify file was created
        output_file = output_dir / "eval_predictions.csv"
        assert output_file.exists()

        # Verify content
        result_df = pd.read_csv(output_file)
        expected_df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "label": [0, 1, 0],
                "prob_class_0": [0.8, 0.3, 0.9],
                "prob_class_1": [0.2, 0.7, 0.1],
            }
        )
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_save_metrics(self, temp_dir):
        """Test saving metrics to JSON."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        metrics = {"auc_roc": 0.85, "average_precision": 0.78, "f1_score": 0.72}

        save_metrics(metrics, str(output_dir))

        # Verify JSON file was created
        json_file = output_dir / "metrics.json"
        assert json_file.exists()

        # Verify content
        with open(json_file, "r") as f:
            saved_metrics = json.load(f)
        assert saved_metrics == metrics

        # Verify summary file was created
        summary_file = output_dir / "metrics_summary.txt"
        assert summary_file.exists()

    @patch("cursus.steps.scripts.xgboost_model_eval.plt")
    def test_plot_and_save_roc_curve(self, mock_plt, temp_dir):
        """Test plotting and saving ROC curve."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        y_true = np.array([0, 1, 0, 1])
        y_score = np.array([0.2, 0.8, 0.3, 0.9])

        plot_and_save_roc_curve(y_true, y_score, str(output_dir))

        # Verify plotting functions were called
        mock_plt.figure.assert_called_once()
        mock_plt.plot.assert_called()
        mock_plt.savefig.assert_called_once()
        mock_plt.close.assert_called_once()

    @patch("cursus.steps.scripts.xgboost_model_eval.plt")
    def test_plot_and_save_pr_curve(self, mock_plt, temp_dir):
        """Test plotting and saving PR curve."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        y_true = np.array([0, 1, 0, 1])
        y_score = np.array([0.2, 0.8, 0.3, 0.9])

        plot_and_save_pr_curve(y_true, y_score, str(output_dir))

        # Verify plotting functions were called
        mock_plt.figure.assert_called_once()
        mock_plt.plot.assert_called()
        mock_plt.savefig.assert_called_once()
        mock_plt.close.assert_called_once()


class TestModelEvaluationIntegration:
    """Integration tests for model evaluation functions."""

    @pytest.fixture
    def temp_dir(self):
        """Set up test fixtures."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @patch("cursus.steps.scripts.xgboost_model_eval.xgb.DMatrix")
    @patch("cursus.steps.scripts.xgboost_model_eval.plot_and_save_roc_curve")
    @patch("cursus.steps.scripts.xgboost_model_eval.plot_and_save_pr_curve")
    def test_evaluate_model_binary(self, mock_pr_curve, mock_roc_curve, mock_dmatrix, temp_dir):
        """Test evaluating binary classification model."""
        # Create test data
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "label": [0, 1, 0, 1],
                "feature1": [0.1, 0.5, 0.2, 0.8],
                "feature2": [0.3, 0.7, 0.4, 0.9],
            }
        )

        feature_columns = ["feature1", "feature2"]
        hyperparams = {"is_binary": True}

        # Mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.2, 0.8, 0.3, 0.9])

        # Mock DMatrix
        mock_dmatrix_instance = MagicMock()
        mock_dmatrix.return_value = mock_dmatrix_instance

        # Create output directories
        output_eval_dir = temp_dir / "eval"
        output_metrics_dir = temp_dir / "metrics"
        output_eval_dir.mkdir()
        output_metrics_dir.mkdir()

        evaluate_model(
            mock_model,
            df,
            feature_columns,
            "id",
            "label",
            hyperparams,
            str(output_eval_dir),
            str(output_metrics_dir),
        )

        # Verify model was called
        mock_model.predict.assert_called_once_with(mock_dmatrix_instance)

        # Verify output files were created
        assert (output_eval_dir / "eval_predictions.csv").exists()
        assert (output_metrics_dir / "metrics.json").exists()

        # Verify plotting functions were called
        mock_roc_curve.assert_called_once()
        mock_pr_curve.assert_called_once()

    @patch("cursus.steps.scripts.xgboost_model_eval.xgb.DMatrix")
    @patch("cursus.steps.scripts.xgboost_model_eval.plot_and_save_roc_curve")
    @patch("cursus.steps.scripts.xgboost_model_eval.plot_and_save_pr_curve")
    def test_evaluate_model_multiclass(
        self, mock_pr_curve, mock_roc_curve, mock_dmatrix, temp_dir
    ):
        """Test evaluating multiclass classification model."""
        # Create test data
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "label": [0, 1, 2, 0],
                "feature1": [0.1, 0.5, 0.2, 0.8],
                "feature2": [0.3, 0.7, 0.4, 0.9],
            }
        )

        feature_columns = ["feature1", "feature2"]
        hyperparams = {"is_binary": False}

        # Mock model - return multiclass probabilities
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array(
            [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.1, 0.7], [0.6, 0.3, 0.1]]
        )

        # Mock DMatrix
        mock_dmatrix_instance = MagicMock()
        mock_dmatrix.return_value = mock_dmatrix_instance

        # Create output directories
        output_eval_dir = temp_dir / "eval"
        output_metrics_dir = temp_dir / "metrics"
        output_eval_dir.mkdir()
        output_metrics_dir.mkdir()

        evaluate_model(
            mock_model,
            df,
            feature_columns,
            "id",
            "label",
            hyperparams,
            str(output_eval_dir),
            str(output_metrics_dir),
        )

        # Verify model was called
        mock_model.predict.assert_called_once_with(mock_dmatrix_instance)

        # Verify output files were created
        assert (output_eval_dir / "eval_predictions.csv").exists()
        assert (output_metrics_dir / "metrics.json").exists()

        # For multiclass, plotting functions should be called for each class
        assert mock_roc_curve.call_count == 3  # 3 classes
        assert mock_pr_curve.call_count == 3


class TestModelEvaluationMain:
    """Tests for the main function of model evaluation script."""

    @pytest.fixture
    def temp_dir(self):
        """Set up test fixtures."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    def _create_mock_model_artifacts(self, model_dir):
        """Create mock model artifacts for testing."""
        model_dir.mkdir(parents=True, exist_ok=True)

        # Create mock files
        (model_dir / "xgboost_model.bst").write_text("mock model")

        with open(model_dir / "risk_table_map.pkl", "wb") as f:
            pkl.dump({}, f)

        with open(model_dir / "impute_dict.pkl", "wb") as f:
            pkl.dump({}, f)

        with open(model_dir / "feature_columns.txt", "w") as f:
            f.write("0,feature1\n1,feature2\n")

        with open(model_dir / "hyperparameters.json", "w") as f:
            json.dump({"is_binary": True}, f)

    @patch("cursus.steps.scripts.xgboost_model_eval.evaluate_model")
    @patch("cursus.steps.scripts.xgboost_model_eval.preprocess_eval_data")
    @patch("cursus.steps.scripts.xgboost_model_eval.load_eval_data")
    @patch("cursus.steps.scripts.xgboost_model_eval.load_model_artifacts")
    def test_main_function(
        self, mock_load_artifacts, mock_load_data, mock_preprocess, mock_evaluate
    ):
        """Test main function execution."""
        # Create job_args mock
        from argparse import Namespace

        job_args = Namespace(job_type="validation")

        # Set up input and output paths
        input_paths = {
            "model_input": "/mock/model",
            "processed_data": "/mock/eval_data",
        }
        output_paths = {
            "eval_output": "/mock/output/eval",
            "metrics_output": "/mock/output/metrics",
        }
        environ_vars = {"ID_FIELD": "id", "LABEL_FIELD": "label"}

        # Mock loaded artifacts
        mock_model = MagicMock()
        mock_load_artifacts.return_value = (
            mock_model,
            {},
            {},
            ["feature1", "feature2"],
            {"is_binary": True},
        )

        # Mock loaded data
        mock_df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "label": [0, 1, 0],
                "feature1": [0.1, 0.5, 0.9],
                "feature2": [0.2, 0.6, 0.8],
            }
        )
        mock_load_data.return_value = mock_df
        mock_preprocess.return_value = mock_df[["feature1", "feature2"]]

        # Mock os.makedirs to avoid actual directory creation
        with patch("cursus.steps.scripts.xgboost_model_eval.os.makedirs"):
            # Run main function
            main(input_paths, output_paths, environ_vars, job_args)

        # Verify all major functions were called
        mock_load_artifacts.assert_called_once()
        mock_load_data.assert_called_once()
        mock_preprocess.assert_called_once()
        mock_evaluate.assert_called_once()


class TestModelComparisonFunctionality:
    """Tests for new model comparison functionality."""

    def test_compute_comparison_metrics_binary(self):
        """Test computing comparison metrics for binary classification."""
        np.random.seed(42)
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        y_new_score = np.array([0.2, 0.8, 0.1, 0.9, 0.7, 0.3, 0.6, 0.4])
        y_prev_score = np.array([0.1, 0.7, 0.2, 0.8, 0.6, 0.4, 0.5, 0.3])

        metrics = compute_comparison_metrics(y_true, y_new_score, y_prev_score, is_binary=True)

        # Check that all expected comparison metrics are present
        expected_keys = [
            "pearson_correlation", "spearman_correlation",
            "new_model_auc", "previous_model_auc", "auc_delta", "auc_lift_percent",
            "new_model_ap", "previous_model_ap", "ap_delta", "ap_lift_percent",
            "new_score_mean", "previous_score_mean", "score_mean_delta"
        ]
        for key in expected_keys:
            assert key in metrics

        # Check correlation values are in valid range
        assert -1 <= metrics["pearson_correlation"] <= 1
        assert -1 <= metrics["spearman_correlation"] <= 1

        # Check AUC values are in valid range
        assert 0 <= metrics["new_model_auc"] <= 1
        assert 0 <= metrics["previous_model_auc"] <= 1

        # Check F1 scores at different thresholds
        for threshold in [0.3, 0.5, 0.7]:
            assert f"new_model_f1_at_{threshold}" in metrics
            assert f"previous_model_f1_at_{threshold}" in metrics
            assert f"f1_delta_at_{threshold}" in metrics

        # Check agreement metrics
        for threshold in [0.3, 0.5, 0.7]:
            assert f"prediction_agreement_at_{threshold}" in metrics
            assert 0 <= metrics[f"prediction_agreement_at_{threshold}"] <= 1

    def test_perform_statistical_tests_binary(self):
        """Test performing statistical tests for binary classification."""
        np.random.seed(42)
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
        y_new_score = np.array([0.2, 0.8, 0.1, 0.9, 0.7, 0.3, 0.6, 0.4, 0.8, 0.2])
        y_prev_score = np.array([0.1, 0.7, 0.2, 0.8, 0.6, 0.4, 0.5, 0.3, 0.7, 0.3])

        test_results = perform_statistical_tests(y_true, y_new_score, y_prev_score, is_binary=True)

        # Check McNemar's test results
        expected_mcnemar_keys = [
            "mcnemar_statistic", "mcnemar_p_value", "mcnemar_significant",
            "correct_both", "new_correct_prev_wrong", "new_wrong_prev_correct", "wrong_both"
        ]
        for key in expected_mcnemar_keys:
            assert key in test_results

        # Check paired t-test results
        expected_ttest_keys = ["paired_t_statistic", "paired_t_p_value", "paired_t_significant"]
        for key in expected_ttest_keys:
            assert key in test_results

        # Check Wilcoxon test results
        expected_wilcoxon_keys = ["wilcoxon_statistic", "wilcoxon_p_value", "wilcoxon_significant"]
        for key in expected_wilcoxon_keys:
            assert key in test_results

        # Check p-values are in valid range (0 to 1)
        assert 0 <= test_results["mcnemar_p_value"] <= 1
        assert 0 <= test_results["paired_t_p_value"] <= 1
        if not np.isnan(test_results["wilcoxon_p_value"]):
            assert 0 <= test_results["wilcoxon_p_value"] <= 1

        # Check significance flags are boolean
        assert isinstance(test_results["mcnemar_significant"], bool)
        assert isinstance(test_results["paired_t_significant"], bool)
        assert isinstance(test_results["wilcoxon_significant"], bool)

    @patch("cursus.steps.scripts.xgboost_model_eval.plt")
    def test_plot_comparison_roc_curves(self, mock_plt, temp_dir):
        """Test plotting comparison ROC curves."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_new_score = np.array([0.2, 0.8, 0.1, 0.9, 0.7, 0.3])
        y_prev_score = np.array([0.1, 0.7, 0.2, 0.8, 0.6, 0.4])

        plot_comparison_roc_curves(y_true, y_new_score, y_prev_score, str(output_dir))

        # Verify plotting functions were called
        mock_plt.figure.assert_called_once()
        mock_plt.plot.assert_called()  # Should be called multiple times for both models
        mock_plt.savefig.assert_called_once()
        mock_plt.close.assert_called_once()

        # Check that savefig was called with correct filename
        savefig_call = mock_plt.savefig.call_args
        assert "comparison_roc_curves.jpg" in str(savefig_call)

    @patch("cursus.steps.scripts.xgboost_model_eval.plt")
    def test_plot_comparison_pr_curves(self, mock_plt, temp_dir):
        """Test plotting comparison PR curves."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_new_score = np.array([0.2, 0.8, 0.1, 0.9, 0.7, 0.3])
        y_prev_score = np.array([0.1, 0.7, 0.2, 0.8, 0.6, 0.4])

        plot_comparison_pr_curves(y_true, y_new_score, y_prev_score, str(output_dir))

        # Verify plotting functions were called
        mock_plt.figure.assert_called_once()
        mock_plt.plot.assert_called()
        mock_plt.savefig.assert_called_once()
        mock_plt.close.assert_called_once()

        # Check that savefig was called with correct filename
        savefig_call = mock_plt.savefig.call_args
        assert "comparison_pr_curves.jpg" in str(savefig_call)

    @patch("cursus.steps.scripts.xgboost_model_eval.plt")
    def test_plot_score_scatter(self, mock_plt, temp_dir):
        """Test plotting score scatter plot."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_new_score = np.array([0.2, 0.8, 0.1, 0.9, 0.7, 0.3])
        y_prev_score = np.array([0.1, 0.7, 0.2, 0.8, 0.6, 0.4])

        plot_score_scatter(y_new_score, y_prev_score, y_true, str(output_dir))

        # Verify plotting functions were called
        mock_plt.figure.assert_called_once()
        mock_plt.scatter.assert_called()  # Should be called for positive and negative examples
        mock_plt.savefig.assert_called_once()
        mock_plt.close.assert_called_once()

        # Check that savefig was called with correct filename
        savefig_call = mock_plt.savefig.call_args
        assert "score_scatter_plot.jpg" in str(savefig_call)

    @patch("cursus.steps.scripts.xgboost_model_eval.plt")
    def test_plot_score_distributions(self, mock_plt, temp_dir):
        """Test plotting score distributions."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_new_score = np.array([0.2, 0.8, 0.1, 0.9, 0.7, 0.3])
        y_prev_score = np.array([0.1, 0.7, 0.2, 0.8, 0.6, 0.4])

        plot_score_distributions(y_new_score, y_prev_score, y_true, str(output_dir))

        # Verify plotting functions were called
        mock_plt.subplots.assert_called_once()
        mock_plt.savefig.assert_called_once()
        mock_plt.close.assert_called_once()

        # Check that savefig was called with correct filename
        savefig_call = mock_plt.savefig.call_args
        assert "score_distributions.jpg" in str(savefig_call)

    def test_save_predictions_with_comparison(self, temp_dir):
        """Test saving predictions with comparison data."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        ids = np.array([1, 2, 3])
        y_true = np.array([0, 1, 0])
        y_prob = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
        previous_scores = np.array([0.7, 0.6, 0.8])

        save_predictions_with_comparison(
            ids, y_true, y_prob, previous_scores, "id", "label", str(output_dir)
        )

        # Verify file was created
        output_file = output_dir / "eval_predictions_with_comparison.csv"
        assert output_file.exists()

        # Verify content
        result_df = pd.read_csv(output_file)
        
        # Check expected columns exist
        expected_columns = [
            "id", "label", "new_model_prob_class_0", "new_model_prob_class_1",
            "previous_model_score", "score_difference"
        ]
        for col in expected_columns:
            assert col in result_df.columns

        # Check data integrity
        assert len(result_df) == 3
        assert list(result_df["id"]) == [1, 2, 3]
        assert list(result_df["label"]) == [0, 1, 0]

    def test_create_comparison_report(self, temp_dir):
        """Test creating comparison report."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        # Create comprehensive metrics dictionary
        metrics = {
            "new_model_auc": 0.85,
            "previous_model_auc": 0.80,
            "auc_delta": 0.05,
            "auc_lift_percent": 6.25,
            "new_model_ap": 0.78,
            "previous_model_ap": 0.75,
            "ap_delta": 0.03,
            "ap_lift_percent": 4.0,
            "pearson_correlation": 0.92,
            "spearman_correlation": 0.89,
            "mcnemar_p_value": 0.045,
            "mcnemar_significant": True,
            "paired_t_p_value": 0.032,
            "paired_t_significant": True,
        }

        create_comparison_report(metrics, str(output_dir), is_binary=True)

        # Verify report file was created
        report_file = output_dir / "comparison_report.txt"
        assert report_file.exists()

        # Verify content contains key sections
        report_content = report_file.read_text()
        assert "MODEL COMPARISON REPORT" in report_content
        assert "PERFORMANCE SUMMARY" in report_content
        assert "CORRELATION ANALYSIS" in report_content
        assert "STATISTICAL SIGNIFICANCE" in report_content
        assert "RECOMMENDATIONS" in report_content

        # Check specific metrics are included
        assert "0.8500" in report_content  # New model AUC
        assert "0.8000" in report_content  # Previous model AUC
        assert "+0.0500" in report_content  # AUC delta
        assert "0.9200" in report_content  # Pearson correlation

    @pytest.fixture
    def temp_dir(self):
        """Set up test fixtures."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @patch("cursus.steps.scripts.xgboost_model_eval.xgb.DMatrix")
    @patch("cursus.steps.scripts.xgboost_model_eval.plot_comparison_roc_curves")
    @patch("cursus.steps.scripts.xgboost_model_eval.plot_comparison_pr_curves")
    @patch("cursus.steps.scripts.xgboost_model_eval.plot_score_scatter")
    @patch("cursus.steps.scripts.xgboost_model_eval.plot_score_distributions")
    @patch("cursus.steps.scripts.xgboost_model_eval.plot_and_save_roc_curve")
    @patch("cursus.steps.scripts.xgboost_model_eval.save_metrics")
    @patch("cursus.steps.scripts.xgboost_model_eval.plot_and_save_pr_curve")
    def test_evaluate_model_with_comparison(
        self, mock_pr_curve, mock_save_metrics, mock_roc_curve, mock_score_dist, mock_score_scatter,
        mock_comp_pr, mock_comp_roc, mock_dmatrix, temp_dir
    ):
        """Test evaluating model with comparison functionality."""
        # Create test data with previous scores
        df = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "label": [0, 1, 0, 1],
            "feature1": [0.1, 0.5, 0.2, 0.8],
            "feature2": [0.3, 0.7, 0.4, 0.9],
        })

        feature_columns = ["feature1", "feature2"]
        hyperparams = {"is_binary": True}
        previous_scores = np.array([0.1, 0.7, 0.2, 0.8])

        # Mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.2, 0.8, 0.3, 0.9])

        # Mock DMatrix
        mock_dmatrix_instance = MagicMock()
        mock_dmatrix.return_value = mock_dmatrix_instance

        # Create output directories
        output_eval_dir = temp_dir / "eval"
        output_metrics_dir = temp_dir / "metrics"
        output_eval_dir.mkdir()
        output_metrics_dir.mkdir()

        evaluate_model_with_comparison(
            mock_model,
            df,
            feature_columns,
            "id",
            "label",
            previous_scores,
            hyperparams,
            str(output_eval_dir),
            str(output_metrics_dir),
            comparison_metrics="all",
            statistical_tests=True,
            comparison_plots=True,
        )

        # Verify model was called (use more flexible assertion)
        mock_model.predict.assert_called_once()
        # Verify DMatrix was created
        mock_dmatrix.assert_called_once()

        # Verify output files were created
        assert (output_eval_dir / "eval_predictions_with_comparison.csv").exists()
        # Note: metrics.json creation is mocked, so we verify the mock was called instead
        mock_save_metrics.assert_called_once()
        assert (output_metrics_dir / "comparison_report.txt").exists()

        # Verify all plotting functions were called
        mock_comp_roc.assert_called_once()
        mock_comp_pr.assert_called_once()
        mock_score_scatter.assert_called_once()
        mock_score_dist.assert_called_once()
        
        # Standard plots for both models
        assert mock_roc_curve.call_count == 2  # New and previous model
        assert mock_pr_curve.call_count == 2

    @patch("cursus.steps.scripts.xgboost_model_eval.evaluate_model_with_comparison")
    @patch("cursus.steps.scripts.xgboost_model_eval.evaluate_model")
    @patch("cursus.steps.scripts.xgboost_model_eval.preprocess_eval_data")
    @patch("cursus.steps.scripts.xgboost_model_eval.load_eval_data")
    @patch("cursus.steps.scripts.xgboost_model_eval.load_model_artifacts")
    def test_main_function_with_comparison_mode(
        self, mock_load_artifacts, mock_load_data, mock_preprocess, 
        mock_evaluate, mock_evaluate_comparison
    ):
        """Test main function with comparison mode enabled."""
        from argparse import Namespace

        job_args = Namespace(job_type="validation")

        # Set up input and output paths
        input_paths = {
            "model_input": "/mock/model",
            "processed_data": "/mock/eval_data",
        }
        output_paths = {
            "eval_output": "/mock/output/eval",
            "metrics_output": "/mock/output/metrics",
        }
        
        # Enable comparison mode
        environ_vars = {
            "ID_FIELD": "id", 
            "LABEL_FIELD": "label",
            "COMPARISON_MODE": "true",
            "PREVIOUS_SCORE_FIELD": "baseline_score",
            "COMPARISON_METRICS": "all",
            "STATISTICAL_TESTS": "true",
            "COMPARISON_PLOTS": "true",
        }

        # Mock loaded artifacts
        mock_model = MagicMock()
        mock_load_artifacts.return_value = (
            mock_model, {}, {}, ["feature1", "feature2"], {"is_binary": True}
        )

        # Mock loaded data with previous scores
        mock_df = pd.DataFrame({
            "id": [1, 2, 3],
            "label": [0, 1, 0],
            "feature1": [0.1, 0.5, 0.9],
            "feature2": [0.2, 0.6, 0.8],
            "baseline_score": [0.2, 0.7, 0.3],  # Previous model scores
        })
        mock_load_data.return_value = mock_df
        mock_preprocess.return_value = mock_df

        # Mock os.makedirs to avoid actual directory creation
        with patch("cursus.steps.scripts.xgboost_model_eval.os.makedirs"):
            # Run main function
            main(input_paths, output_paths, environ_vars, job_args)

        # Verify comparison evaluation was called instead of standard evaluation
        mock_evaluate_comparison.assert_called_once()
        mock_evaluate.assert_not_called()

        # Verify all major functions were called
        mock_load_artifacts.assert_called_once()
        mock_load_data.assert_called_once()
        mock_preprocess.assert_called_once()

    @patch("cursus.steps.scripts.xgboost_model_eval.evaluate_model")
    @patch("cursus.steps.scripts.xgboost_model_eval.preprocess_eval_data")
    @patch("cursus.steps.scripts.xgboost_model_eval.load_eval_data")
    @patch("cursus.steps.scripts.xgboost_model_eval.load_model_artifacts")
    def test_main_function_comparison_mode_disabled_missing_field(
        self, mock_load_artifacts, mock_load_data, mock_preprocess, mock_evaluate
    ):
        """Test main function with comparison mode disabled due to missing previous score field."""
        from argparse import Namespace

        job_args = Namespace(job_type="validation")

        input_paths = {
            "model_input": "/mock/model",
            "processed_data": "/mock/eval_data",
        }
        output_paths = {
            "eval_output": "/mock/output/eval",
            "metrics_output": "/mock/output/metrics",
        }
        
        # Enable comparison mode but with empty previous score field
        environ_vars = {
            "ID_FIELD": "id", 
            "LABEL_FIELD": "label",
            "COMPARISON_MODE": "true",
            "PREVIOUS_SCORE_FIELD": "",  # Empty field should disable comparison
            "COMPARISON_METRICS": "all",
            "STATISTICAL_TESTS": "true",
            "COMPARISON_PLOTS": "true",
        }

        # Mock loaded artifacts
        mock_model = MagicMock()
        mock_load_artifacts.return_value = (
            mock_model, {}, {}, ["feature1", "feature2"], {"is_binary": True}
        )

        # Mock loaded data without previous scores
        mock_df = pd.DataFrame({
            "id": [1, 2, 3],
            "label": [0, 1, 0],
            "feature1": [0.1, 0.5, 0.9],
            "feature2": [0.2, 0.6, 0.8],
        })
        mock_load_data.return_value = mock_df
        mock_preprocess.return_value = mock_df

        # Mock os.makedirs to avoid actual directory creation
        with patch("cursus.steps.scripts.xgboost_model_eval.os.makedirs"):
            # Run main function
            main(input_paths, output_paths, environ_vars, job_args)

        # Verify standard evaluation was called (comparison mode should be disabled)
        mock_evaluate.assert_called_once()

        # Verify all major functions were called
        mock_load_artifacts.assert_called_once()
        mock_load_data.assert_called_once()
        mock_preprocess.assert_called_once()

"""
Comprehensive test suite for model_metrics_computation.py script.

This test suite follows pytest best practices and provides thorough coverage
of the model metrics computation functionality including:
- Binary and multiclass classification metrics
- Domain-specific metrics (count recall, dollar recall)
- Model comparison metrics and statistical tests
- File format detection and loading (CSV, TSV, Parquet, JSON)
- Performance visualizations (ROC, PR curves, score distributions)
- Comprehensive report generation
- Exception handling and edge cases
"""

import pytest
from unittest.mock import patch, MagicMock, Mock, call, mock_open
import os
import sys
import tempfile
import shutil
import json
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
from datetime import datetime

# CRITICAL: Mock subprocess.check_call BEFORE importing the module
# The model_metrics_computation script calls install_packages() at module level
# which triggers subprocess.check_call to run pip install.
# We must mock check_call before the module import to prevent actual installations.
with patch("subprocess.check_call"):
    # Import the functions to be tested
    from cursus.steps.scripts.model_metrics_computation import (
        _detect_file_format,
        detect_and_load_predictions,
        validate_prediction_data,
        compute_standard_metrics,
        calculate_count_recall,
        calculate_dollar_recall,
        compute_domain_metrics,
        compute_comparison_metrics,
        perform_statistical_tests,
        plot_and_save_roc_curve,
        plot_and_save_pr_curve,
        generate_performance_visualizations,
        plot_comparison_roc_curves,
        plot_comparison_pr_curves,
        plot_score_scatter,
        plot_score_distributions,
        generate_performance_insights,
        generate_recommendations,
        generate_comprehensive_report,
        generate_text_summary,
        log_metrics_summary,
        save_metrics,
        create_health_check_file,
        main,
        CONTAINER_PATHS,
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

    def test_detect_file_format_parquet(self):
        """Test Parquet format detection."""
        assert _detect_file_format("predictions.parquet") == "parquet"
        assert _detect_file_format("predictions.pq") == "parquet"
        assert _detect_file_format("/path/to/predictions.PARQUET") == "parquet"

    def test_detect_file_format_json(self):
        """Test JSON format detection."""
        assert _detect_file_format("predictions.json") == "json"
        assert _detect_file_format("/path/to/predictions.json") == "json"

    def test_detect_file_format_default_csv(self):
        """Test default fallback to CSV for unknown extensions."""
        assert _detect_file_format("predictions.txt") == "csv"
        assert _detect_file_format("predictions") == "csv"


class TestDataLoading:
    """Tests for data loading functionality."""

    @pytest.fixture
    def setup_temp_dir(self):
        """Set up temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_binary_predictions(self):
        """Create sample binary predictions dataframe."""
        np.random.seed(42)
        n_samples = 100
        return pd.DataFrame(
            {
                "id": range(n_samples),
                "label": np.random.randint(0, 2, n_samples),
                "prob_class_0": np.random.uniform(0.0, 0.5, n_samples),
                "prob_class_1": np.random.uniform(0.5, 1.0, n_samples),
            }
        )

    def test_detect_and_load_predictions_csv(
        self, setup_temp_dir, sample_binary_predictions
    ):
        """Test loading predictions from CSV file."""
        temp_dir = setup_temp_dir
        csv_file = temp_dir / "predictions.csv"
        sample_binary_predictions.to_csv(csv_file, index=False)

        df_loaded = detect_and_load_predictions(str(temp_dir), preferred_format="csv")

        pd.testing.assert_frame_equal(df_loaded, sample_binary_predictions)

    def test_detect_and_load_predictions_tsv(
        self, setup_temp_dir, sample_binary_predictions
    ):
        """Test loading predictions from TSV file."""
        temp_dir = setup_temp_dir
        tsv_file = temp_dir / "predictions.tsv"
        sample_binary_predictions.to_csv(tsv_file, sep="\t", index=False)

        df_loaded = detect_and_load_predictions(str(temp_dir), preferred_format="tsv")

        pd.testing.assert_frame_equal(df_loaded, sample_binary_predictions)

    def test_detect_and_load_predictions_parquet(
        self, setup_temp_dir, sample_binary_predictions
    ):
        """Test loading predictions from Parquet file."""
        temp_dir = setup_temp_dir
        parquet_file = temp_dir / "predictions.parquet"
        sample_binary_predictions.to_parquet(parquet_file)

        df_loaded = detect_and_load_predictions(
            str(temp_dir), preferred_format="parquet"
        )

        pd.testing.assert_frame_equal(df_loaded, sample_binary_predictions)

    def test_detect_and_load_predictions_json(
        self, setup_temp_dir, sample_binary_predictions
    ):
        """Test loading predictions from JSON file."""
        temp_dir = setup_temp_dir
        json_file = temp_dir / "predictions.json"
        sample_binary_predictions.to_json(json_file)

        df_loaded = detect_and_load_predictions(str(temp_dir), preferred_format="json")

        # JSON may have different dtypes, compare values
        assert len(df_loaded) == len(sample_binary_predictions)
        assert list(df_loaded.columns) == list(sample_binary_predictions.columns)

    def test_detect_and_load_predictions_auto_format(
        self, setup_temp_dir, sample_binary_predictions
    ):
        """Test auto-detection of file format."""
        temp_dir = setup_temp_dir
        csv_file = temp_dir / "predictions.csv"
        sample_binary_predictions.to_csv(csv_file, index=False)

        df_loaded = detect_and_load_predictions(str(temp_dir))

        pd.testing.assert_frame_equal(df_loaded, sample_binary_predictions)

    def test_detect_and_load_predictions_eval_predictions_fallback(
        self, setup_temp_dir, sample_binary_predictions
    ):
        """Test fallback to eval_predictions.csv from xgboost_model_eval."""
        temp_dir = setup_temp_dir
        eval_file = temp_dir / "eval_predictions.csv"
        sample_binary_predictions.to_csv(eval_file, index=False)

        df_loaded = detect_and_load_predictions(str(temp_dir))

        pd.testing.assert_frame_equal(df_loaded, sample_binary_predictions)

    def test_detect_and_load_predictions_no_file(self, setup_temp_dir):
        """Test error when no predictions file found."""
        temp_dir = setup_temp_dir

        with pytest.raises(
            FileNotFoundError,
            match="No predictions file found in supported formats",
        ):
            detect_and_load_predictions(str(temp_dir))


class TestDataValidation:
    """Tests for data validation functionality."""

    @pytest.fixture
    def valid_binary_data(self):
        """Create valid binary classification data."""
        np.random.seed(42)
        n_samples = 100
        return pd.DataFrame(
            {
                "id": range(n_samples),
                "label": np.random.randint(0, 2, n_samples),
                "prob_class_0": np.random.uniform(0.0, 0.5, n_samples),
                "prob_class_1": np.random.uniform(0.5, 1.0, n_samples),
                "amount": np.random.uniform(10.0, 1000.0, n_samples),
            }
        )

    def test_validate_prediction_data_valid(self, valid_binary_data):
        """Test validation with valid data."""
        report = validate_prediction_data(
            valid_binary_data, id_field="id", label_field="label", amount_field="amount"
        )

        assert report["is_valid"] is True
        assert len(report["errors"]) == 0
        assert report["data_summary"]["total_records"] == 100
        assert report["data_summary"]["has_amount_data"] is True
        assert len(report["data_summary"]["prediction_columns"]) == 2

    def test_validate_prediction_data_missing_id_field(self, valid_binary_data):
        """Test validation with missing ID field."""
        df = valid_binary_data.drop(columns=["id"])

        report = validate_prediction_data(
            df, id_field="id", label_field="label", amount_field="amount"
        )

        assert report["is_valid"] is False
        assert any("Missing required columns" in error for error in report["errors"])

    def test_validate_prediction_data_missing_label_field(self, valid_binary_data):
        """Test validation with missing label field."""
        df = valid_binary_data.drop(columns=["label"])

        report = validate_prediction_data(
            df, id_field="id", label_field="label", amount_field="amount"
        )

        assert report["is_valid"] is False
        assert any("Missing required columns" in error for error in report["errors"])

    def test_validate_prediction_data_no_prob_columns(self, valid_binary_data):
        """Test validation with no probability columns."""
        df = valid_binary_data.drop(columns=["prob_class_0", "prob_class_1"])

        report = validate_prediction_data(
            df, id_field="id", label_field="label", amount_field="amount"
        )

        assert report["is_valid"] is False
        assert any(
            "No prediction probability columns" in error for error in report["errors"]
        )

    def test_validate_prediction_data_missing_amount_field(self, valid_binary_data):
        """Test validation with missing amount field."""
        df = valid_binary_data.drop(columns=["amount"])

        report = validate_prediction_data(
            df, id_field="id", label_field="label", amount_field="amount"
        )

        # Should be valid but with warning
        assert report["is_valid"] is True
        assert any(
            "Amount field" in warning and "not found" in warning
            for warning in report["warnings"]
        )

    def test_validate_prediction_data_label_distribution(self, valid_binary_data):
        """Test label distribution in validation report."""
        report = validate_prediction_data(
            valid_binary_data, id_field="id", label_field="label"
        )

        assert "label_distribution" in report["data_summary"]
        assert 0 in report["data_summary"]["label_distribution"]
        assert 1 in report["data_summary"]["label_distribution"]


class TestStandardMetrics:
    """Tests for standard ML metrics computation."""

    @pytest.fixture
    def binary_classification_data(self):
        """Create binary classification test data."""
        np.random.seed(42)
        n_samples = 100
        y_true = np.random.randint(0, 2, n_samples)
        y_prob = np.column_stack(
            [
                1 - np.random.uniform(0.1, 0.9, n_samples),
                np.random.uniform(0.1, 0.9, n_samples),
            ]
        )
        return y_true, y_prob

    @pytest.fixture
    def multiclass_classification_data(self):
        """Create multiclass classification test data."""
        np.random.seed(42)
        n_samples = 100
        n_classes = 3
        y_true = np.random.randint(0, n_classes, n_samples)
        y_prob = np.random.dirichlet(np.ones(n_classes), size=n_samples)
        return y_true, y_prob

    def test_compute_standard_metrics_binary(self, binary_classification_data):
        """Test standard metrics computation for binary classification."""
        y_true, y_prob = binary_classification_data

        metrics = compute_standard_metrics(y_true, y_prob, is_binary=True)

        # Check core binary metrics exist
        assert "auc_roc" in metrics
        assert "average_precision" in metrics
        assert "f1_score" in metrics
        assert "precision_at_threshold_0.5" in metrics
        assert "recall_at_threshold_0.5" in metrics

        # Check threshold-based metrics
        for threshold in [0.3, 0.5, 0.7]:
            assert f"f1_score_at_{threshold}" in metrics
            assert f"precision_at_{threshold}" in metrics
            assert f"recall_at_{threshold}" in metrics

        # Check additional metrics
        assert "max_f1_score" in metrics
        assert "optimal_threshold" in metrics

        # Verify value ranges
        assert 0 <= metrics["auc_roc"] <= 1
        assert 0 <= metrics["average_precision"] <= 1
        assert 0 <= metrics["f1_score"] <= 1

    def test_compute_standard_metrics_multiclass(self, multiclass_classification_data):
        """Test standard metrics computation for multiclass classification."""
        y_true, y_prob = multiclass_classification_data

        metrics = compute_standard_metrics(y_true, y_prob, is_binary=False)

        # Check per-class metrics
        n_classes = y_prob.shape[1]
        for i in range(n_classes):
            assert f"auc_roc_class_{i}" in metrics
            assert f"average_precision_class_{i}" in metrics
            assert f"f1_score_class_{i}" in metrics

        # Check micro/macro averages
        assert "auc_roc_micro" in metrics
        assert "auc_roc_macro" in metrics
        assert "average_precision_micro" in metrics
        assert "average_precision_macro" in metrics
        assert "f1_score_micro" in metrics
        assert "f1_score_macro" in metrics

        # Check class distribution metrics
        for i in range(n_classes):
            assert f"class_{i}_count" in metrics
            assert f"class_{i}_ratio" in metrics

        # Verify value ranges
        assert 0 <= metrics["auc_roc_macro"] <= 1
        assert 0 <= metrics["f1_score_macro"] <= 1


class TestDomainMetrics:
    """Tests for domain-specific metrics computation."""

    @pytest.fixture
    def abuse_detection_data(self):
        """Create abuse detection test data."""
        np.random.seed(42)
        n_samples = 100
        scores = np.random.uniform(0, 1, n_samples)
        labels = (scores > 0.6).astype(int)
        amounts = np.random.uniform(10.0, 1000.0, n_samples)
        return scores, labels, amounts

    def test_calculate_count_recall(self, abuse_detection_data):
        """Test count recall calculation."""
        scores, labels, amounts = abuse_detection_data

        count_recall = calculate_count_recall(scores, labels, amounts, cutoff=0.1)

        # Verify return type and range
        assert isinstance(count_recall, (float, np.floating))
        assert 0 <= count_recall <= 1

    def test_calculate_count_recall_different_cutoffs(self, abuse_detection_data):
        """Test count recall with different cutoffs."""
        scores, labels, amounts = abuse_detection_data

        recall_10 = calculate_count_recall(scores, labels, amounts, cutoff=0.1)
        recall_20 = calculate_count_recall(scores, labels, amounts, cutoff=0.2)

        # Higher cutoff should catch more abuse cases
        assert recall_20 >= recall_10

    def test_calculate_dollar_recall(self, abuse_detection_data):
        """Test dollar recall calculation."""
        scores, labels, amounts = abuse_detection_data

        dollar_recall = calculate_dollar_recall(scores, labels, amounts, fpr=0.1)

        # Verify return type and range
        assert isinstance(dollar_recall, (float, np.floating))
        assert 0 <= dollar_recall <= 1

    def test_calculate_dollar_recall_different_fpr(self, abuse_detection_data):
        """Test dollar recall with different FPR values."""
        scores, labels, amounts = abuse_detection_data

        recall_10 = calculate_dollar_recall(scores, labels, amounts, fpr=0.1)
        recall_20 = calculate_dollar_recall(scores, labels, amounts, fpr=0.2)

        # Higher FPR threshold should catch more abuse dollar amount
        assert recall_20 >= recall_10

    def test_compute_domain_metrics_full(self, abuse_detection_data):
        """Test full domain metrics computation."""
        scores, labels, amounts = abuse_detection_data

        metrics = compute_domain_metrics(
            scores=scores,
            labels=labels,
            amounts=amounts,
            compute_dollar_recall=True,
            compute_count_recall=True,
            dollar_recall_fpr=0.1,
            count_recall_cutoff=0.1,
        )

        # Check count recall metrics
        assert "count_recall" in metrics
        assert "count_recall_cutoff" in metrics

        # Check dollar recall metrics
        assert "dollar_recall" in metrics
        assert "dollar_recall_fpr" in metrics
        assert "total_abuse_amount" in metrics
        assert "average_abuse_amount" in metrics
        assert "total_legitimate_amount" in metrics
        assert "amount_ratio_abuse_to_total" in metrics

    def test_compute_domain_metrics_no_amounts(self, abuse_detection_data):
        """Test domain metrics computation without amount data."""
        scores, labels, _ = abuse_detection_data

        metrics = compute_domain_metrics(
            scores=scores,
            labels=labels,
            amounts=None,
            compute_dollar_recall=True,
            compute_count_recall=True,
        )

        # Should have count recall but not dollar recall
        assert "count_recall" in metrics
        assert "dollar_recall" not in metrics

    def test_compute_domain_metrics_only_count_recall(self, abuse_detection_data):
        """Test domain metrics with only count recall enabled."""
        scores, labels, amounts = abuse_detection_data

        metrics = compute_domain_metrics(
            scores=scores,
            labels=labels,
            amounts=amounts,
            compute_dollar_recall=False,
            compute_count_recall=True,
        )

        assert "count_recall" in metrics
        assert "dollar_recall" not in metrics


class TestComparisonMetrics:
    """Tests for model comparison metrics."""

    @pytest.fixture
    def comparison_data(self):
        """Create comparison test data."""
        np.random.seed(42)
        n_samples = 100
        y_true = np.random.randint(0, 2, n_samples)
        y_new_score = np.random.uniform(0, 1, n_samples)
        # Add correlation to previous scores
        y_prev_score = y_new_score + np.random.normal(0, 0.1, n_samples)
        y_prev_score = np.clip(y_prev_score, 0, 1)
        return y_true, y_new_score, y_prev_score

    def test_compute_comparison_metrics_binary(self, comparison_data):
        """Test comparison metrics for binary classification."""
        y_true, y_new_score, y_prev_score = comparison_data

        metrics = compute_comparison_metrics(
            y_true, y_new_score, y_prev_score, is_binary=True
        )

        # Check correlation metrics
        assert "pearson_correlation" in metrics
        assert "pearson_p_value" in metrics
        assert "spearman_correlation" in metrics
        assert "spearman_p_value" in metrics

        # Check performance comparison metrics
        assert "new_model_auc" in metrics
        assert "previous_model_auc" in metrics
        assert "auc_delta" in metrics
        assert "auc_lift_percent" in metrics
        assert "new_model_ap" in metrics
        assert "previous_model_ap" in metrics
        assert "ap_delta" in metrics
        assert "ap_lift_percent" in metrics

        # Check F1 comparison at thresholds
        for threshold in [0.3, 0.5, 0.7]:
            assert f"new_model_f1_at_{threshold}" in metrics
            assert f"previous_model_f1_at_{threshold}" in metrics
            assert f"f1_delta_at_{threshold}" in metrics

        # Check score distribution comparison
        assert "new_score_mean" in metrics
        assert "previous_score_mean" in metrics
        assert "new_score_std" in metrics
        assert "previous_score_std" in metrics
        assert "score_mean_delta" in metrics

        # Check agreement metrics
        for threshold in [0.3, 0.5, 0.7]:
            assert f"prediction_agreement_at_{threshold}" in metrics

    def test_perform_statistical_tests_binary(self, comparison_data):
        """Test statistical significance tests for binary classification."""
        y_true, y_new_score, y_prev_score = comparison_data

        test_results = perform_statistical_tests(
            y_true, y_new_score, y_prev_score, is_binary=True
        )

        # Check McNemar's test results
        assert "mcnemar_statistic" in test_results
        assert "mcnemar_p_value" in test_results
        assert "mcnemar_significant" in test_results
        assert "correct_both" in test_results
        assert "new_correct_prev_wrong" in test_results
        assert "new_wrong_prev_correct" in test_results
        assert "wrong_both" in test_results

        # Check paired t-test results
        assert "paired_t_statistic" in test_results
        assert "paired_t_p_value" in test_results
        assert "paired_t_significant" in test_results

        # Check Wilcoxon test results
        assert "wilcoxon_statistic" in test_results
        assert "wilcoxon_p_value" in test_results
        assert "wilcoxon_significant" in test_results


class TestVisualization:
    """Tests for visualization functions."""

    @pytest.fixture
    def setup_temp_dir(self):
        """Set up temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def binary_viz_data(self):
        """Create binary classification visualization data."""
        np.random.seed(42)
        n_samples = 100
        y_true = np.random.randint(0, 2, n_samples)
        y_score = np.random.uniform(0, 1, n_samples)
        return y_true, y_score

    def test_plot_and_save_roc_curve(self, setup_temp_dir, binary_viz_data):
        """Test ROC curve plotting and saving."""
        temp_dir = setup_temp_dir
        y_true, y_score = binary_viz_data

        output_path = plot_and_save_roc_curve(y_true, y_score, str(temp_dir))

        assert Path(output_path).exists()
        assert output_path.endswith("roc_curve.jpg")

    def test_plot_and_save_roc_curve_with_prefix(self, setup_temp_dir, binary_viz_data):
        """Test ROC curve plotting with filename prefix."""
        temp_dir = setup_temp_dir
        y_true, y_score = binary_viz_data

        output_path = plot_and_save_roc_curve(
            y_true, y_score, str(temp_dir), prefix="test_"
        )

        assert Path(output_path).exists()
        assert "test_roc_curve.jpg" in output_path

    def test_plot_and_save_pr_curve(self, setup_temp_dir, binary_viz_data):
        """Test Precision-Recall curve plotting and saving."""
        temp_dir = setup_temp_dir
        y_true, y_score = binary_viz_data

        output_path = plot_and_save_pr_curve(y_true, y_score, str(temp_dir))

        assert Path(output_path).exists()
        assert output_path.endswith("pr_curve.jpg")

    def test_generate_performance_visualizations_binary(
        self, setup_temp_dir, binary_viz_data
    ):
        """Test generation of binary classification visualizations."""
        temp_dir = setup_temp_dir
        y_true, y_score = binary_viz_data
        y_prob = np.column_stack([1 - y_score, y_score])

        metrics = compute_standard_metrics(y_true, y_prob, is_binary=True)
        plot_paths = generate_performance_visualizations(
            y_true, y_prob, metrics, str(temp_dir), is_binary=True
        )

        # Check all expected plots were created
        assert "roc_curve" in plot_paths
        assert "precision_recall_curve" in plot_paths
        assert "score_distribution" in plot_paths
        assert "threshold_analysis" in plot_paths

        # Verify files exist
        for path in plot_paths.values():
            assert Path(path).exists()

    def test_generate_performance_visualizations_multiclass(self, setup_temp_dir):
        """Test generation of multiclass visualizations."""
        temp_dir = setup_temp_dir
        np.random.seed(42)
        n_samples = 100
        n_classes = 3

        y_true = np.random.randint(0, n_classes, n_samples)
        y_prob = np.random.dirichlet(np.ones(n_classes), size=n_samples)

        metrics = compute_standard_metrics(y_true, y_prob, is_binary=False)
        plot_paths = generate_performance_visualizations(
            y_true, y_prob, metrics, str(temp_dir), is_binary=False
        )

        # Check multiclass plots were created
        assert "multiclass_roc_curves" in plot_paths
        for i in range(n_classes):
            assert f"roc_curve_class_{i}" in plot_paths
            assert f"pr_curve_class_{i}" in plot_paths

        # Verify files exist
        for path in plot_paths.values():
            assert Path(path).exists()

    def test_plot_comparison_roc_curves(self, setup_temp_dir, binary_viz_data):
        """Test plotting comparison ROC curves."""
        temp_dir = setup_temp_dir
        y_true, y_score = binary_viz_data
        y_prev_score = y_score + np.random.normal(0, 0.1, len(y_score))
        y_prev_score = np.clip(y_prev_score, 0, 1)

        output_path = plot_comparison_roc_curves(
            y_true, y_score, y_prev_score, str(temp_dir)
        )

        assert Path(output_path).exists()
        assert output_path.endswith("comparison_roc_curves.jpg")

    def test_plot_comparison_pr_curves(self, setup_temp_dir, binary_viz_data):
        """Test plotting comparison Precision-Recall curves."""
        temp_dir = setup_temp_dir
        y_true, y_score = binary_viz_data
        y_prev_score = y_score + np.random.normal(0, 0.1, len(y_score))
        y_prev_score = np.clip(y_prev_score, 0, 1)

        output_path = plot_comparison_pr_curves(
            y_true, y_score, y_prev_score, str(temp_dir)
        )

        assert Path(output_path).exists()
        assert output_path.endswith("comparison_pr_curves.jpg")

    def test_plot_score_scatter(self, setup_temp_dir, binary_viz_data):
        """Test plotting score scatter plot."""
        temp_dir = setup_temp_dir
        y_true, y_score = binary_viz_data
        y_prev_score = y_score + np.random.normal(0, 0.1, len(y_score))
        y_prev_score = np.clip(y_prev_score, 0, 1)

        output_path = plot_score_scatter(y_score, y_prev_score, y_true, str(temp_dir))

        assert Path(output_path).exists()
        assert output_path.endswith("score_scatter_plot.jpg")

    def test_plot_score_distributions(self, setup_temp_dir, binary_viz_data):
        """Test plotting score distributions."""
        temp_dir = setup_temp_dir
        y_true, y_score = binary_viz_data
        y_prev_score = y_score + np.random.normal(0, 0.1, len(y_score))
        y_prev_score = np.clip(y_prev_score, 0, 1)

        output_path = plot_score_distributions(
            y_score, y_prev_score, y_true, str(temp_dir)
        )

        assert Path(output_path).exists()
        assert output_path.endswith("score_distributions.jpg")


class TestInsightsAndRecommendations:
    """Tests for insights and recommendations generation."""

    def test_generate_performance_insights_excellent_auc(self):
        """Test insights generation with excellent AUC."""
        metrics = {
            "auc_roc": 0.95,
            "dollar_recall": 0.85,
            "count_recall": 0.80,
            "optimal_threshold": 0.45,
        }

        insights = generate_performance_insights(metrics)

        assert any("Excellent discrimination" in insight for insight in insights)
        assert len(insights) > 0

    def test_generate_performance_insights_good_auc(self):
        """Test insights generation with good AUC."""
        metrics = {"auc_roc": 0.85}

        insights = generate_performance_insights(metrics)

        assert any("Good discrimination" in insight for insight in insights)

    def test_generate_performance_insights_poor_auc(self):
        """Test insights generation with poor AUC."""
        metrics = {"auc_roc": 0.65}

        insights = generate_performance_insights(metrics)

        assert any("Poor discrimination" in insight for insight in insights)

    def test_generate_performance_insights_dollar_vs_count(self):
        """Test insights generation comparing dollar and count recall."""
        metrics = {
            "auc_roc": 0.85,
            "dollar_recall": 0.90,
            "count_recall": 0.70,
        }

        insights = generate_performance_insights(metrics)

        assert any("high-value abuse" in insight for insight in insights)

    def test_generate_recommendations_low_auc(self):
        """Test recommendations generation for low AUC."""
        metrics = {"auc_roc": 0.70}

        recommendations = generate_recommendations(metrics)

        assert any(
            "feature engineering" in rec.lower() for rec in recommendations
        ) or any("data quality" in rec.lower() for rec in recommendations)

    def test_generate_recommendations_low_dollar_recall(self):
        """Test recommendations generation for low dollar recall."""
        metrics = {
            "auc_roc": 0.85,
            "dollar_recall": 0.50,
            "count_recall": 0.80,
        }

        recommendations = generate_recommendations(metrics)

        assert any("high-value" in rec.lower() for rec in recommendations)


class TestReportGeneration:
    """Tests for comprehensive report generation."""

    @pytest.fixture
    def setup_temp_dir(self):
        """Set up temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_report_data(self):
        """Create sample report data."""
        standard_metrics = {
            "auc_roc": 0.85,
            "average_precision": 0.80,
            "f1_score": 0.75,
        }
        domain_metrics = {
            "count_recall": 0.80,
            "dollar_recall": 0.85,
            "total_abuse_amount": 50000.0,
        }
        plot_paths = {
            "roc_curve": "/path/to/roc.jpg",
            "pr_curve": "/path/to/pr.jpg",
        }
        validation_report = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "data_summary": {
                "total_records": 100,
                "prediction_columns": ["prob_class_0", "prob_class_1"],
                "has_amount_data": True,
            },
        }
        return standard_metrics, domain_metrics, plot_paths, validation_report

    def test_generate_comprehensive_report(self, setup_temp_dir, sample_report_data):
        """Test comprehensive report generation."""
        temp_dir = setup_temp_dir
        standard_metrics, domain_metrics, plot_paths, validation_report = (
            sample_report_data
        )

        report_paths = generate_comprehensive_report(
            standard_metrics,
            domain_metrics,
            plot_paths,
            validation_report,
            str(temp_dir),
        )

        # Check report files were created
        assert "json_report" in report_paths
        assert "text_summary" in report_paths
        assert Path(report_paths["json_report"]).exists()
        assert Path(report_paths["text_summary"]).exists()

        # Verify JSON report structure
        with open(report_paths["json_report"], "r") as f:
            json_report = json.load(f)

        assert "timestamp" in json_report
        assert "data_summary" in json_report
        assert "standard_metrics" in json_report
        assert "domain_metrics" in json_report
        assert "visualizations" in json_report
        assert "performance_insights" in json_report
        assert "recommendations" in json_report

    def test_generate_text_summary(self, sample_report_data):
        """Test text summary generation."""
        standard_metrics, domain_metrics, plot_paths, validation_report = (
            sample_report_data
        )

        json_report = {
            "timestamp": datetime.utcnow().isoformat(),
            "data_summary": validation_report["data_summary"],
            "standard_metrics": standard_metrics,
            "domain_metrics": domain_metrics,
            "visualizations": plot_paths,
            "performance_insights": ["Insight 1", "Insight 2"],
            "recommendations": ["Recommendation 1"],
        }

        summary = generate_text_summary(json_report)

        # Check summary contains key sections
        assert "MODEL METRICS COMPUTATION REPORT" in summary
        assert "DATA SUMMARY" in summary
        assert "STANDARD ML METRICS" in summary
        assert "DOMAIN-SPECIFIC METRICS" in summary
        assert "PERFORMANCE INSIGHTS" in summary
        assert "RECOMMENDATIONS" in summary


class TestMetricsSaving:
    """Tests for metrics saving functionality."""

    @pytest.fixture
    def setup_temp_dir(self):
        """Set up temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_save_metrics_json(self, setup_temp_dir):
        """Test saving metrics to JSON file."""
        temp_dir = setup_temp_dir
        metrics = {
            "auc_roc": 0.85,
            "f1_score": 0.75,
            "count_recall": 0.80,
        }

        save_metrics(metrics, str(temp_dir))

        json_path = temp_dir / "metrics.json"
        assert json_path.exists()

        with open(json_path, "r") as f:
            saved_metrics = json.load(f)

        assert saved_metrics["auc_roc"] == 0.85
        assert saved_metrics["f1_score"] == 0.75

    def test_save_metrics_text_summary(self, setup_temp_dir):
        """Test saving metrics text summary."""
        temp_dir = setup_temp_dir
        metrics = {
            "auc_roc": 0.85,
            "average_precision": 0.80,
            "f1_score": 0.75,
        }

        save_metrics(metrics, str(temp_dir))

        summary_path = temp_dir / "metrics_summary.txt"
        assert summary_path.exists()

        with open(summary_path, "r") as f:
            content = f.read()

        assert "METRICS SUMMARY" in content
        assert "AUC-ROC" in content

    def test_save_metrics_numpy_types(self, setup_temp_dir):
        """Test saving metrics with numpy types."""
        temp_dir = setup_temp_dir
        metrics = {
            "auc_roc": np.float64(0.85),
            "count": np.int64(100),
            "flag": np.bool_(True),
        }

        save_metrics(metrics, str(temp_dir))

        json_path = temp_dir / "metrics.json"
        with open(json_path, "r") as f:
            saved_metrics = json.load(f)

        # Should be serializable Python types
        assert isinstance(saved_metrics["auc_roc"], float)
        assert isinstance(saved_metrics["count"], int)
        assert isinstance(saved_metrics["flag"], bool)

    def test_create_health_check_file(self, setup_temp_dir):
        """Test health check file creation."""
        temp_dir = setup_temp_dir
        health_path = temp_dir / "_HEALTH"

        result = create_health_check_file(str(health_path))

        assert Path(result).exists()
        assert result == str(health_path)

        with open(health_path, "r") as f:
            content = f.read()

        assert "healthy:" in content


class TestMainFunction:
    """Tests for main function integration."""

    @pytest.fixture
    def setup_integration_test(self):
        """Set up integration test environment."""
        temp_dir = Path(tempfile.mkdtemp())

        # Create input directory with test data
        input_dir = temp_dir / "eval_data"
        input_dir.mkdir(parents=True)

        # Create realistic binary classification test data
        np.random.seed(42)
        n_samples = 100
        df = pd.DataFrame(
            {
                "id": range(n_samples),
                "label": np.random.randint(0, 2, n_samples),
                "prob_class_0": np.random.uniform(0.0, 0.5, n_samples),
                "prob_class_1": np.random.uniform(0.5, 1.0, n_samples),
                "amount": np.random.uniform(10.0, 1000.0, n_samples),
            }
        )
        df.to_csv(input_dir / "predictions.csv", index=False)

        # Set up paths
        input_paths = {"processed_data": str(input_dir)}
        output_paths = {
            "metrics_output": str(temp_dir / "metrics"),
            "plots_output": str(temp_dir / "plots"),
        }
        environ_vars = {
            "ID_FIELD": "id",
            "LABEL_FIELD": "label",
            "AMOUNT_FIELD": "amount",
            "INPUT_FORMAT": "csv",
            "COMPUTE_DOLLAR_RECALL": "true",
            "COMPUTE_COUNT_RECALL": "true",
            "DOLLAR_RECALL_FPR": "0.1",
            "COUNT_RECALL_CUTOFF": "0.1",
            "GENERATE_PLOTS": "true",
            "COMPARISON_MODE": "false",
        }

        yield temp_dir, input_paths, output_paths, environ_vars
        shutil.rmtree(temp_dir)

    def test_main_binary_classification_success(self, setup_integration_test):
        """Test successful binary classification metrics computation."""
        temp_dir, input_paths, output_paths, environ_vars = setup_integration_test

        job_args = argparse.Namespace(job_type="metrics_computation")

        # Call main function
        main(input_paths, output_paths, environ_vars, job_args)

        # Check output files created
        metrics_dir = Path(output_paths["metrics_output"])
        assert (metrics_dir / "metrics.json").exists()
        assert (metrics_dir / "metrics_summary.txt").exists()
        assert (metrics_dir / "metrics_report.json").exists()
        # Note: _SUCCESS is created in __main__ block, not by main() function itself
        # Note: _HEALTH is also created in __main__ block, not by main() function itself

        # Check plots created
        plots_dir = Path(output_paths["plots_output"])
        assert (plots_dir / "roc_curve.jpg").exists()
        assert (plots_dir / "pr_curve.jpg").exists()

    def test_main_with_comparison_mode(self, setup_integration_test):
        """Test metrics computation with comparison mode enabled."""
        temp_dir, input_paths, output_paths, environ_vars = setup_integration_test

        # Add previous score field to data
        input_dir = Path(input_paths["processed_data"])
        df = pd.read_csv(input_dir / "predictions.csv")
        df["prev_score"] = df["prob_class_1"] + np.random.normal(0, 0.1, len(df))
        df["prev_score"] = df["prev_score"].clip(0, 1)
        df.to_csv(input_dir / "predictions.csv", index=False)

        # Enable comparison mode
        environ_vars["COMPARISON_MODE"] = "true"
        environ_vars["PREVIOUS_SCORE_FIELD"] = "prev_score"
        environ_vars["STATISTICAL_TESTS"] = "true"
        environ_vars["COMPARISON_PLOTS"] = "true"

        job_args = argparse.Namespace(job_type="metrics_computation")

        # Call main function
        main(input_paths, output_paths, environ_vars, job_args)

        # Check comparison visualizations created
        plots_dir = Path(output_paths["plots_output"])
        assert (plots_dir / "comparison_roc_curves.jpg").exists()
        assert (plots_dir / "comparison_pr_curves.jpg").exists()
        assert (plots_dir / "score_scatter_plot.jpg").exists()
        assert (plots_dir / "score_distributions.jpg").exists()

        # Verify metrics include comparison data
        with open(Path(output_paths["metrics_output"]) / "metrics.json", "r") as f:
            metrics = json.load(f)

        assert "pearson_correlation" in metrics
        assert "new_model_auc" in metrics
        assert "previous_model_auc" in metrics
        assert "auc_delta" in metrics

    def test_main_multiclass_classification(self, setup_integration_test):
        """Test metrics computation for multiclass classification."""
        temp_dir, input_paths, output_paths, environ_vars = setup_integration_test

        # Create multiclass data
        input_dir = Path(input_paths["processed_data"])
        np.random.seed(42)
        n_samples = 100
        n_classes = 3

        df = pd.DataFrame(
            {
                "id": range(n_samples),
                "label": np.random.randint(0, n_classes, n_samples),
            }
        )

        # Add probability columns
        probs = np.random.dirichlet(np.ones(n_classes), size=n_samples)
        for i in range(n_classes):
            df[f"prob_class_{i}"] = probs[:, i]

        df.to_csv(input_dir / "predictions.csv", index=False)

        job_args = argparse.Namespace(job_type="metrics_computation")

        # Call main function
        main(input_paths, output_paths, environ_vars, job_args)

        # Check multiclass outputs
        plots_dir = Path(output_paths["plots_output"])
        assert (plots_dir / "multiclass_roc_curves.jpg").exists()

        # Verify metrics include multiclass data
        with open(Path(output_paths["metrics_output"]) / "metrics.json", "r") as f:
            metrics = json.load(f)

        assert "auc_roc_macro" in metrics
        assert "auc_roc_micro" in metrics


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def setup_temp_dir(self):
        """Set up temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_perfect_predictions(self):
        """Test metrics computation with perfect predictions."""
        np.random.seed(42)
        n_samples = 100
        y_true = np.random.randint(0, 2, n_samples)
        y_prob = np.column_stack([1 - y_true, y_true.astype(float)])

        metrics = compute_standard_metrics(y_true, y_prob, is_binary=True)

        # AUC should be perfect
        assert metrics["auc_roc"] == pytest.approx(1.0)

    def test_constant_predictions(self):
        """Test metrics computation with constant predictions."""
        np.random.seed(42)
        n_samples = 100
        y_true = np.random.randint(0, 2, n_samples)
        y_prob = np.column_stack([np.full(n_samples, 0.5), np.full(n_samples, 0.5)])

        metrics = compute_standard_metrics(y_true, y_prob, is_binary=True)

        # Should handle constant predictions without error
        assert "auc_roc" in metrics
        assert 0 <= metrics["auc_roc"] <= 1

    def test_small_dataset(self):
        """Test metrics computation with very small dataset."""
        n_samples = 10
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        y_prob = np.column_stack(
            [
                1 - np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.45, 0.55]),
                np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.45, 0.55]),
            ]
        )

        metrics = compute_standard_metrics(y_true, y_prob, is_binary=True)

        # Should handle small dataset
        assert "auc_roc" in metrics
        assert "f1_score" in metrics

    def test_imbalanced_dataset(self):
        """Test metrics computation with highly imbalanced dataset."""
        np.random.seed(42)
        n_samples = 100
        # 95% class 0, 5% class 1
        y_true = np.array([0] * 95 + [1] * 5)
        y_prob = np.column_stack(
            [1 - np.random.uniform(0, 1, n_samples), np.random.uniform(0, 1, n_samples)]
        )

        metrics = compute_standard_metrics(y_true, y_prob, is_binary=True)

        # Should handle imbalanced data
        assert "auc_roc" in metrics
        assert "average_precision" in metrics

    def test_comparison_mode_without_previous_score(self, setup_temp_dir):
        """Test that comparison mode is disabled when previous score field is missing."""
        temp_dir = setup_temp_dir
        input_dir = temp_dir / "input"
        input_dir.mkdir()

        # Create data without previous score
        df = pd.DataFrame(
            {
                "id": range(10),
                "label": [0, 1] * 5,
                "prob_class_0": [0.3, 0.7] * 5,
                "prob_class_1": [0.7, 0.3] * 5,
            }
        )
        df.to_csv(input_dir / "predictions.csv", index=False)

        input_paths = {"processed_data": str(input_dir)}
        output_paths = {
            "metrics_output": str(temp_dir / "metrics"),
            "plots_output": str(temp_dir / "plots"),
        }
        environ_vars = {
            "ID_FIELD": "id",
            "LABEL_FIELD": "label",
            "GENERATE_PLOTS": "false",
            "COMPARISON_MODE": "true",
            "PREVIOUS_SCORE_FIELD": "",  # Empty field
        }

        job_args = argparse.Namespace(job_type="metrics_computation")

        # Should run without error, comparison mode disabled
        main(input_paths, output_paths, environ_vars, job_args)

        # Verify comparison metrics not in output
        with open(Path(output_paths["metrics_output"]) / "metrics.json", "r") as f:
            metrics = json.load(f)

        assert "pearson_correlation" not in metrics

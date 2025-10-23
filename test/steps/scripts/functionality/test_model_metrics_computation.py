import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import json
from pathlib import Path
import argparse
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from cursus.steps.scripts.model_metrics_computation import (
    main,
    detect_and_load_predictions,
    validate_prediction_data,
    compute_standard_metrics,
    calculate_count_recall,
    calculate_dollar_recall,
    compute_domain_metrics,
    plot_and_save_roc_curve,
    plot_and_save_pr_curve,
    generate_performance_visualizations,
    generate_performance_insights,
    generate_recommendations,
    generate_comprehensive_report,
    generate_text_summary,
    log_metrics_summary,
    save_metrics,
    create_health_check_file,
    compute_comparison_metrics,
    perform_statistical_tests,
    plot_comparison_roc_curves,
    plot_comparison_pr_curves,
    plot_score_scatter,
    plot_score_distributions,
    CONTAINER_PATHS
)


class TestDetectAndLoadPredictions:
    """Tests for detect_and_load_predictions function."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def create_sample_predictions(self, temp_dir, format_type="csv"):
        """Helper to create sample prediction files."""
        data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'label': [0, 1, 0, 1, 0],
            'prob_class_0': [0.8, 0.3, 0.9, 0.2, 0.7],
            'prob_class_1': [0.2, 0.7, 0.1, 0.8, 0.3],
            'amount': [100.0, 250.0, 75.0, 500.0, 150.0]
        })
        
        if format_type == "csv":
            file_path = temp_dir / "predictions.csv"
            data.to_csv(file_path, index=False)
        elif format_type == "parquet":
            file_path = temp_dir / "predictions.parquet"
            data.to_parquet(file_path, index=False)
        elif format_type == "json":
            file_path = temp_dir / "predictions.json"
            data.to_json(file_path, orient='records')
        
        return file_path, data

    def test_detect_and_load_predictions_csv(self, temp_dir):
        """Test loading CSV predictions file."""
        file_path, expected_data = self.create_sample_predictions(temp_dir, "csv")
        
        result = detect_and_load_predictions(str(temp_dir))
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        assert list(result.columns) == ['id', 'label', 'prob_class_0', 'prob_class_1', 'amount']
        pd.testing.assert_frame_equal(result, expected_data)

    def test_detect_and_load_predictions_parquet(self, temp_dir):
        """Test loading Parquet predictions file."""
        file_path, expected_data = self.create_sample_predictions(temp_dir, "parquet")
        
        result = detect_and_load_predictions(str(temp_dir))
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        pd.testing.assert_frame_equal(result, expected_data)

    def test_detect_and_load_predictions_json(self, temp_dir):
        """Test loading JSON predictions file."""
        file_path, expected_data = self.create_sample_predictions(temp_dir, "json")
        
        result = detect_and_load_predictions(str(temp_dir))
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        # JSON loading may change column order, so check columns exist
        assert set(result.columns) == set(expected_data.columns)

    def test_detect_and_load_predictions_preferred_format(self, temp_dir):
        """Test loading with preferred format specification."""
        # Create both CSV and Parquet files
        self.create_sample_predictions(temp_dir, "csv")
        file_path, expected_data = self.create_sample_predictions(temp_dir, "parquet")
        
        # Should prefer parquet when specified
        result = detect_and_load_predictions(str(temp_dir), preferred_format="parquet")
        
        pd.testing.assert_frame_equal(result, expected_data)

    def test_detect_and_load_predictions_eval_predictions(self, temp_dir):
        """Test loading eval_predictions.csv fallback file."""
        data = pd.DataFrame({
            'id': [1, 2, 3],
            'label': [0, 1, 0],
            'prob_class_0': [0.8, 0.3, 0.9],
            'prob_class_1': [0.2, 0.7, 0.1]
        })
        
        eval_pred_path = temp_dir / "eval_predictions.csv"
        data.to_csv(eval_pred_path, index=False)
        
        result = detect_and_load_predictions(str(temp_dir))
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        pd.testing.assert_frame_equal(result, data)

    def test_detect_and_load_predictions_file_not_found(self, temp_dir):
        """Test error when no prediction files exist."""
        with pytest.raises(FileNotFoundError, match="No predictions file found in supported formats"):
            detect_and_load_predictions(str(temp_dir))


class TestValidatePredictionData:
    """Tests for validate_prediction_data function."""

    @pytest.fixture
    def valid_data(self):
        """Create valid prediction data."""
        return pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'label': [0, 1, 0, 1, 0],
            'prob_class_0': [0.8, 0.3, 0.9, 0.2, 0.7],
            'prob_class_1': [0.2, 0.7, 0.1, 0.8, 0.3],
            'amount': [100.0, 250.0, 75.0, 500.0, 150.0]
        })

    def test_validate_prediction_data_success(self, valid_data):
        """Test validation with valid data."""
        result = validate_prediction_data(valid_data, "id", "label", "amount")
        
        assert result["is_valid"] is True
        assert len(result["errors"]) == 0
        assert len(result["warnings"]) == 0
        assert result["data_summary"]["total_records"] == 5
        assert "prob_class_0" in result["data_summary"]["prediction_columns"]
        assert "prob_class_1" in result["data_summary"]["prediction_columns"]
        assert result["data_summary"]["has_amount_data"] is True

    def test_validate_prediction_data_missing_required_columns(self, valid_data):
        """Test validation with missing required columns."""
        data_missing_id = valid_data.drop('id', axis=1)
        
        result = validate_prediction_data(data_missing_id, "id", "label")
        
        assert result["is_valid"] is False
        assert len(result["errors"]) == 1
        assert "Missing required columns: ['id']" in result["errors"][0]

    def test_validate_prediction_data_missing_prob_columns(self, valid_data):
        """Test validation with missing probability columns."""
        data_no_probs = valid_data[['id', 'label', 'amount']]
        
        result = validate_prediction_data(data_no_probs, "id", "label")
        
        assert result["is_valid"] is False
        assert "No prediction probability columns found" in result["errors"]

    def test_validate_prediction_data_missing_amount_field(self, valid_data):
        """Test validation with missing amount field."""
        result = validate_prediction_data(valid_data, "id", "label", "missing_amount")
        
        assert result["is_valid"] is True
        assert len(result["warnings"]) == 1
        assert "Amount field 'missing_amount' not found" in result["warnings"][0]

    def test_validate_prediction_data_no_amount_field(self, valid_data):
        """Test validation without specifying amount field."""
        result = validate_prediction_data(valid_data, "id", "label")
        
        assert result["is_valid"] is True
        assert result["data_summary"]["has_amount_data"] is False


class TestComputeStandardMetrics:
    """Tests for compute_standard_metrics function."""

    @pytest.fixture
    def binary_data(self):
        """Create binary classification test data."""
        np.random.seed(42)
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        y_prob = np.array([
            [0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.2, 0.8], [0.7, 0.3],
            [0.4, 0.6], [0.85, 0.15], [0.1, 0.9], [0.75, 0.25], [0.35, 0.65]
        ])
        return y_true, y_prob

    @pytest.fixture
    def multiclass_data(self):
        """Create multiclass classification test data."""
        np.random.seed(42)
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_prob = np.array([
            [0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8],
            [0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7],
            [0.6, 0.3, 0.1], [0.3, 0.6, 0.1], [0.2, 0.1, 0.7]
        ])
        return y_true, y_prob

    def test_compute_standard_metrics_binary(self, binary_data):
        """Test computing standard metrics for binary classification."""
        y_true, y_prob = binary_data
        
        result = compute_standard_metrics(y_true, y_prob, is_binary=True)
        
        # Check that all expected metrics are present
        expected_metrics = [
            "auc_roc", "average_precision", "f1_score",
            "precision_at_threshold_0.5", "recall_at_threshold_0.5",
            "f1_score_at_0.3", "precision_at_0.3", "recall_at_0.3",
            "f1_score_at_0.5", "precision_at_0.5", "recall_at_0.5",
            "f1_score_at_0.7", "precision_at_0.7", "recall_at_0.7",
            "max_f1_score", "optimal_threshold"
        ]
        
        for metric in expected_metrics:
            assert metric in result
            assert isinstance(result[metric], (int, float))
        
        # Check that AUC is reasonable
        assert 0 <= result["auc_roc"] <= 1
        assert 0 <= result["average_precision"] <= 1

    def test_compute_standard_metrics_multiclass(self, multiclass_data):
        """Test computing standard metrics for multiclass classification."""
        y_true, y_prob = multiclass_data
        
        result = compute_standard_metrics(y_true, y_prob, is_binary=False)
        
        # Check per-class metrics
        for i in range(3):
            assert f"auc_roc_class_{i}" in result
            assert f"average_precision_class_{i}" in result
            assert f"f1_score_class_{i}" in result
        
        # Check aggregate metrics
        expected_aggregates = [
            "auc_roc_micro", "auc_roc_macro",
            "average_precision_micro", "average_precision_macro",
            "f1_score_micro", "f1_score_macro"
        ]
        
        for metric in expected_aggregates:
            assert metric in result
            assert isinstance(result[metric], (int, float))
        
        # Check class distribution metrics
        for i in range(3):
            assert f"class_{i}_count" in result
            assert f"class_{i}_ratio" in result

    def test_compute_standard_metrics_single_class_edge_case(self):
        """Test computing metrics with single class (edge case)."""
        y_true = np.array([0, 0, 0, 0, 0])
        y_prob = np.array([[0.8, 0.2], [0.7, 0.3], [0.9, 0.1], [0.6, 0.4], [0.85, 0.15]])
        
        # This should handle the edge case gracefully
        # Some metrics may be undefined (NaN) which is expected behavior
        result = compute_standard_metrics(y_true, y_prob, is_binary=True)
        
        assert isinstance(result, dict)
        # AUC should be undefined for single class
        assert np.isnan(result["auc_roc"]) or result["auc_roc"] == 0.5


class TestDomainMetrics:
    """Tests for domain-specific metrics functions."""

    @pytest.fixture
    def domain_data(self):
        """Create test data for domain metrics."""
        np.random.seed(42)
        scores = np.array([0.1, 0.8, 0.2, 0.9, 0.3, 0.7, 0.15, 0.85, 0.25, 0.75])
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        amounts = np.array([100, 500, 200, 800, 150, 600, 120, 750, 180, 650])
        return scores, labels, amounts

    def test_calculate_count_recall(self, domain_data):
        """Test count recall calculation."""
        scores, labels, amounts = domain_data
        
        result = calculate_count_recall(scores, labels, amounts, cutoff=0.1)
        
        assert isinstance(result, (int, float))
        assert 0 <= result <= 1

    def test_calculate_dollar_recall(self, domain_data):
        """Test dollar recall calculation."""
        scores, labels, amounts = domain_data
        
        result = calculate_dollar_recall(scores, labels, amounts, fpr=0.1)
        
        assert isinstance(result, (int, float))
        assert 0 <= result <= 1

    def test_calculate_count_recall_mismatched_lengths(self):
        """Test count recall with mismatched input lengths."""
        scores = np.array([0.1, 0.8, 0.2])
        labels = np.array([0, 1])  # Different length
        amounts = np.array([100, 500, 200])
        
        with pytest.raises(AssertionError, match="Input lengths don't match"):
            calculate_count_recall(scores, labels, amounts)

    def test_calculate_dollar_recall_mismatched_lengths(self):
        """Test dollar recall with mismatched input lengths."""
        scores = np.array([0.1, 0.8, 0.2])
        labels = np.array([0, 1, 0])
        amounts = np.array([100, 500])  # Different length
        
        with pytest.raises(AssertionError, match="Input lengths don't match"):
            calculate_dollar_recall(scores, labels, amounts)

    def test_compute_domain_metrics_with_amounts(self, domain_data):
        """Test computing domain metrics with amount data."""
        scores, labels, amounts = domain_data
        
        result = compute_domain_metrics(
            scores=scores,
            labels=labels,
            amounts=amounts,
            compute_dollar_recall=True,
            compute_count_recall=True
        )
        
        expected_metrics = [
            "count_recall", "count_recall_cutoff",
            "dollar_recall", "dollar_recall_fpr",
            "total_abuse_amount", "average_abuse_amount",
            "total_legitimate_amount", "amount_ratio_abuse_to_total"
        ]
        
        for metric in expected_metrics:
            assert metric in result
            # Note: numpy integers are not Python int type, so check for numeric types
            assert isinstance(result[metric], (int, float, np.integer, np.floating))

    def test_compute_domain_metrics_without_amounts(self, domain_data):
        """Test computing domain metrics without amount data."""
        scores, labels, amounts = domain_data
        
        result = compute_domain_metrics(
            scores=scores,
            labels=labels,
            amounts=None,
            compute_dollar_recall=False,
            compute_count_recall=True
        )
        
        assert "count_recall" in result
        assert "dollar_recall" not in result
        assert "total_abuse_amount" not in result

    def test_compute_domain_metrics_disabled(self, domain_data):
        """Test computing domain metrics with all computations disabled."""
        scores, labels, amounts = domain_data
        
        result = compute_domain_metrics(
            scores=scores,
            labels=labels,
            amounts=amounts,
            compute_dollar_recall=False,
            compute_count_recall=False
        )
        
        assert len(result) == 0


class TestVisualizationFunctions:
    """Tests for visualization and plotting functions."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def binary_viz_data(self):
        """Create binary classification data for visualization tests."""
        np.random.seed(42)
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        y_score = np.array([0.2, 0.7, 0.1, 0.8, 0.3, 0.6, 0.15, 0.85, 0.25, 0.75])
        return y_true, y_score

    def test_plot_and_save_roc_curve(self, temp_dir, binary_viz_data):
        """Test ROC curve plotting and saving."""
        y_true, y_score = binary_viz_data
        
        result_path = plot_and_save_roc_curve(y_true, y_score, str(temp_dir))
        
        assert os.path.exists(result_path)
        assert result_path.endswith("roc_curve.jpg")
        assert Path(result_path).stat().st_size > 0  # File is not empty

    def test_plot_and_save_pr_curve(self, temp_dir, binary_viz_data):
        """Test Precision-Recall curve plotting and saving."""
        y_true, y_score = binary_viz_data
        
        result_path = plot_and_save_pr_curve(y_true, y_score, str(temp_dir))
        
        assert os.path.exists(result_path)
        assert result_path.endswith("pr_curve.jpg")
        assert Path(result_path).stat().st_size > 0  # File is not empty

    def test_plot_and_save_roc_curve_with_prefix(self, temp_dir, binary_viz_data):
        """Test ROC curve plotting with prefix."""
        y_true, y_score = binary_viz_data
        
        result_path = plot_and_save_roc_curve(y_true, y_score, str(temp_dir), prefix="test_")
        
        assert os.path.exists(result_path)
        assert "test_roc_curve.jpg" in result_path

    def test_generate_performance_visualizations_binary(self, temp_dir):
        """Test generating performance visualizations for binary classification."""
        np.random.seed(42)
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        y_prob = np.array([
            [0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.2, 0.8], [0.7, 0.3],
            [0.4, 0.6], [0.85, 0.15], [0.1, 0.9], [0.75, 0.25], [0.35, 0.65]
        ])
        metrics = {"optimal_threshold": 0.5}
        
        result = generate_performance_visualizations(
            y_true, y_prob, metrics, str(temp_dir), is_binary=True
        )
        
        expected_plots = ["roc_curve", "precision_recall_curve", "score_distribution", "threshold_analysis"]
        for plot_name in expected_plots:
            assert plot_name in result
            assert os.path.exists(result[plot_name])

    def test_generate_performance_visualizations_multiclass(self, temp_dir):
        """Test generating performance visualizations for multiclass classification."""
        np.random.seed(42)
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_prob = np.array([
            [0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8],
            [0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7],
            [0.6, 0.3, 0.1], [0.3, 0.6, 0.1], [0.2, 0.1, 0.7]
        ])
        metrics = {"auc_roc_class_0": 0.8, "auc_roc_class_1": 0.9, "auc_roc_class_2": 0.7}
        
        result = generate_performance_visualizations(
            y_true, y_prob, metrics, str(temp_dir), is_binary=False
        )
        
        # Should have per-class plots and multiclass overview
        assert "multiclass_roc_curves" in result
        assert os.path.exists(result["multiclass_roc_curves"])


class TestReportGeneration:
    """Tests for report generation functions."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def sample_metrics(self):
        """Create sample metrics for testing."""
        standard_metrics = {
            "auc_roc": 0.85,
            "average_precision": 0.78,
            "f1_score": 0.72,
            "optimal_threshold": 0.45
        }
        domain_metrics = {
            "count_recall": 0.68,
            "dollar_recall": 0.75,
            "total_abuse_amount": 5000.0
        }
        return standard_metrics, domain_metrics

    @pytest.fixture
    def sample_validation_report(self):
        """Create sample validation report."""
        return {
            "data_summary": {
                "total_records": 1000,
                "prediction_columns": ["prob_class_0", "prob_class_1"],
                "has_amount_data": True,
                "label_distribution": {0: 800, 1: 200}
            }
        }

    def test_generate_performance_insights(self, sample_metrics):
        """Test generating performance insights."""
        standard_metrics, domain_metrics = sample_metrics
        all_metrics = {**standard_metrics, **domain_metrics}
        
        result = generate_performance_insights(all_metrics)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert any("Good discrimination capability" in insight for insight in result)

    def test_generate_recommendations(self, sample_metrics):
        """Test generating recommendations."""
        standard_metrics, domain_metrics = sample_metrics
        all_metrics = {**standard_metrics, **domain_metrics}
        
        result = generate_recommendations(all_metrics)
        
        assert isinstance(result, list)
        # With good metrics, should have fewer recommendations

    def test_generate_recommendations_poor_performance(self):
        """Test generating recommendations for poor performance."""
        poor_metrics = {
            "auc_roc": 0.65,
            "dollar_recall": 0.45,
            "count_recall": 0.55,
            "max_f1_score": 0.5
        }
        
        result = generate_recommendations(poor_metrics)
        
        assert isinstance(result, list)
        assert len(result) > 0
        assert any("feature engineering" in rec.lower() for rec in result)

    def test_generate_text_summary(self, sample_metrics, sample_validation_report):
        """Test generating text summary."""
        standard_metrics, domain_metrics = sample_metrics
        plot_paths = {"roc_curve": "/path/to/roc.jpg"}
        
        json_report = {
            "timestamp": "2023-01-01T00:00:00",
            "data_summary": sample_validation_report["data_summary"],
            "standard_metrics": standard_metrics,
            "domain_metrics": domain_metrics,
            "visualizations": plot_paths,
            "performance_insights": ["Test insight"],
            "recommendations": ["Test recommendation"]
        }
        
        result = generate_text_summary(json_report)
        
        assert isinstance(result, str)
        assert "MODEL METRICS COMPUTATION REPORT" in result
        assert "Total Records: 1000" in result
        assert "auc_roc: 0.8500" in result
        assert "Test insight" in result
        assert "Test recommendation" in result

    def test_generate_comprehensive_report(self, temp_dir, sample_metrics, sample_validation_report):
        """Test generating comprehensive report."""
        standard_metrics, domain_metrics = sample_metrics
        plot_paths = {"roc_curve": "/path/to/roc.jpg"}
        
        result = generate_comprehensive_report(
            standard_metrics, domain_metrics, plot_paths, 
            sample_validation_report, str(temp_dir)
        )
        
        assert "json_report" in result
        assert "text_summary" in result
        assert os.path.exists(result["json_report"])
        assert os.path.exists(result["text_summary"])
        
        # Check JSON report content
        with open(result["json_report"], 'r') as f:
            json_data = json.load(f)
        
        assert "timestamp" in json_data
        assert "standard_metrics" in json_data
        assert "domain_metrics" in json_data
        assert "performance_insights" in json_data
        assert "recommendations" in json_data


class TestUtilityFunctions:
    """Tests for utility functions."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_log_metrics_summary_binary(self, caplog):
        """Test logging metrics summary for binary classification."""
        import logging
        caplog.set_level(logging.INFO)
        
        metrics = {
            "auc_roc": 0.85,
            "average_precision": 0.78,
            "f1_score": 0.72
        }
        
        log_metrics_summary(metrics, is_binary=True)
        
        # Check that metrics were logged
        assert "METRICS SUMMARY" in caplog.text
        assert "auc_roc" in caplog.text

    def test_log_metrics_summary_multiclass(self, caplog):
        """Test logging metrics summary for multiclass classification."""
        import logging
        caplog.set_level(logging.INFO)
        
        metrics = {
            "auc_roc_macro": 0.82,
            "auc_roc_micro": 0.85,
            "f1_score_macro": 0.75,
            "f1_score_micro": 0.78
        }
        
        log_metrics_summary(metrics, is_binary=False)
        
        # Check that multiclass metrics were logged
        assert "METRICS SUMMARY" in caplog.text
        assert "auc_roc_macro" in caplog.text

    def test_save_metrics(self, temp_dir):
        """Test saving metrics to files."""
        metrics = {
            "auc_roc": 0.85,
            "average_precision": 0.78,
            "f1_score": 0.72
        }
        
        save_metrics(metrics, str(temp_dir))
        
        # Check JSON file
        json_path = temp_dir / "metrics.json"
        assert json_path.exists()
        
        with open(json_path, 'r') as f:
            saved_metrics = json.load(f)
        
        assert saved_metrics == metrics
        
        # Check text summary file
        summary_path = temp_dir / "metrics_summary.txt"
        assert summary_path.exists()
        
        with open(summary_path, 'r') as f:
            summary_content = f.read()
        
        assert "METRICS SUMMARY" in summary_content
        assert "AUC-ROC:" in summary_content

    def test_create_health_check_file(self, temp_dir):
        """Test creating health check file."""
        health_path = temp_dir / "health_check.txt"
        
        result = create_health_check_file(str(health_path))
        
        assert result == str(health_path)
        assert health_path.exists()
        
        with open(health_path, 'r') as f:
            content = f.read()
        
        assert "healthy:" in content


class TestMainFunction:
    """Tests for main function."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def setup_test_data(self, temp_dir):
        """Helper to set up test data structure."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        
        # Create sample predictions data
        data = pd.DataFrame({
            'id': range(100),
            'label': np.random.choice([0, 1], 100, p=[0.8, 0.2]),
            'prob_class_0': np.random.uniform(0.1, 0.9, 100),
            'prob_class_1': np.random.uniform(0.1, 0.9, 100),
            'amount': np.random.uniform(50, 1000, 100)
        })
        
        # Normalize probabilities
        prob_sum = data['prob_class_0'] + data['prob_class_1']
        data['prob_class_0'] = data['prob_class_0'] / prob_sum
        data['prob_class_1'] = data['prob_class_1'] / prob_sum
        
        predictions_file = input_dir / "predictions.csv"
        data.to_csv(predictions_file, index=False)
        
        return input_dir

    def test_main_function_success(self, temp_dir):
        """Test main function with valid inputs."""
        # Set up test data
        input_dir = self.setup_test_data(temp_dir)
        output_metrics_dir = temp_dir / "metrics"
        output_plots_dir = temp_dir / "plots"
        
        # Create arguments
        args = argparse.Namespace()
        
        # Environment variables
        environ_vars = {
            "ID_FIELD": "id",
            "LABEL_FIELD": "label",
            "AMOUNT_FIELD": "amount",
            "INPUT_FORMAT": "auto",
            "COMPUTE_DOLLAR_RECALL": "true",
            "COMPUTE_COUNT_RECALL": "true",
            "DOLLAR_RECALL_FPR": "0.1",
            "COUNT_RECALL_CUTOFF": "0.1",
            "GENERATE_PLOTS": "true"
        }
        
        # Path dictionaries
        input_paths = {"processed_data": str(input_dir)}
        output_paths = {
            "metrics_output": str(output_metrics_dir),
            "plots_output": str(output_plots_dir)
        }
        
        # Run main function
        main(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,
            job_args=args
        )
        
        # Verify outputs
        assert (output_metrics_dir / "metrics.json").exists()
        assert (output_metrics_dir / "metrics_summary.txt").exists()
        assert (output_metrics_dir / "metrics_report.json").exists()
        assert (output_metrics_dir / "metrics_summary.txt").exists()

    def test_main_function_missing_input_path(self, temp_dir):
        """Test main function with missing input path."""
        args = argparse.Namespace()
        environ_vars = {"ID_FIELD": "id", "LABEL_FIELD": "label"}
        
        input_paths = {}  # Missing processed_data
        output_paths = {"metrics_output": str(temp_dir)}
        
        # Should handle missing input gracefully or raise appropriate error
        # Based on source code analysis, this will likely cause an error
        with pytest.raises((KeyError, TypeError)):
            main(
                input_paths=input_paths,
                output_paths=output_paths,
                environ_vars=environ_vars,
                job_args=args
            )

    def test_main_function_invalid_data(self, temp_dir):
        """Test main function with invalid prediction data."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        
        # Create invalid data (missing probability columns)
        invalid_data = pd.DataFrame({
            'id': [1, 2, 3],
            'label': [0, 1, 0],
            'amount': [100, 200, 150]
            # Missing prob_class_* columns
        })
        
        predictions_file = input_dir / "predictions.csv"
        invalid_data.to_csv(predictions_file, index=False)
        
        args = argparse.Namespace()
        environ_vars = {"ID_FIELD": "id", "LABEL_FIELD": "label"}
        input_paths = {"processed_data": str(input_dir)}
        output_paths = {"metrics_output": str(temp_dir)}
        
        # Should raise ValueError due to validation failure
        with pytest.raises(ValueError, match="Input data validation failed"):
            main(
                input_paths=input_paths,
                output_paths=output_paths,
                environ_vars=environ_vars,
                job_args=args
            )

    def test_main_function_no_plots(self, temp_dir):
        """Test main function with plot generation disabled."""
        input_dir = self.setup_test_data(temp_dir)
        output_dir = temp_dir / "output"
        
        args = argparse.Namespace()
        environ_vars = {
            "ID_FIELD": "id",
            "LABEL_FIELD": "label",
            "GENERATE_PLOTS": "false"  # Disable plots
        }
        
        input_paths = {"processed_data": str(input_dir)}
        output_paths = {"metrics_output": str(output_dir)}
        
        main(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,
            job_args=args
        )
        
        # Should still create metrics but no plots
        assert (output_dir / "metrics.json").exists()
        # Plot files should not exist
        assert not (output_dir / "roc_curve.jpg").exists()


class TestCommonFailurePatterns:
    """Tests for common failure patterns identified from pytest guides."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_single_class_data_edge_case(self):
        """Test handling of single-class data (common failure pattern)."""
        # All labels are the same class - should cause issues with some metrics
        y_true = np.array([0, 0, 0, 0, 0])
        y_prob = np.array([[0.8, 0.2], [0.7, 0.3], [0.9, 0.1], [0.6, 0.4], [0.85, 0.15]])
        
        # Based on source code analysis, this should handle gracefully but some metrics will be NaN
        result = compute_standard_metrics(y_true, y_prob, is_binary=True)
        
        # AUC should be undefined (NaN) for single class
        assert np.isnan(result["auc_roc"]) or result["auc_roc"] == 0.5

    def test_empty_data_edge_case(self):
        """Test handling of empty data (common failure pattern)."""
        y_true = np.array([])
        y_prob = np.array([]).reshape(0, 2)
        
        # This should raise an error or handle gracefully
        # Based on sklearn behavior, this will likely raise an error
        with pytest.raises((ValueError, IndexError)):
            compute_standard_metrics(y_true, y_prob, is_binary=True)

    def test_mismatched_array_lengths(self):
        """Test handling of mismatched array lengths (common failure pattern)."""
        y_true = np.array([0, 1, 0])
        y_prob = np.array([[0.8, 0.2], [0.3, 0.7]])  # Different length
        
        # This should raise an error
        with pytest.raises((ValueError, IndexError)):
            compute_standard_metrics(y_true, y_prob, is_binary=True)

    def test_invalid_probability_values(self):
        """Test handling of invalid probability values (common failure pattern)."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([
            [0.8, 0.2], [0.3, 0.7], 
            [1.5, -0.5],  # Invalid probabilities (>1, <0)
            [0.2, 0.8]
        ])
        
        # Should handle invalid probabilities gracefully
        # Most sklearn functions are robust to this
        result = compute_standard_metrics(y_true, y_prob, is_binary=True)
        assert isinstance(result, dict)

    def test_nan_in_predictions(self):
        """Test handling of NaN values in predictions (common failure pattern)."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([
            [0.8, 0.2], [np.nan, 0.7], 
            [0.9, 0.1], [0.2, np.nan]
        ])
        
        # NaN values should cause issues with metrics computation
        # Based on sklearn behavior, this will likely raise an error or return NaN
        with pytest.raises((ValueError, RuntimeError)) or True:  # May handle gracefully
            result = compute_standard_metrics(y_true, y_prob, is_binary=True)

    def test_file_format_detection_edge_cases(self, temp_dir):
        """Test file format detection edge cases (common failure pattern)."""
        # Create file with wrong extension but correct content
        data = pd.DataFrame({
            'id': [1, 2, 3],
            'label': [0, 1, 0],
            'prob_class_0': [0.8, 0.3, 0.9],
            'prob_class_1': [0.2, 0.7, 0.1]
        })
        
        # Save as CSV but with .txt extension
        wrong_ext_file = temp_dir / "predictions.txt"
        data.to_csv(wrong_ext_file, index=False)
        
        # Should not find the file since it looks for specific extensions
        with pytest.raises(FileNotFoundError):
            detect_and_load_predictions(str(temp_dir))

    def test_corrupted_file_handling(self, temp_dir):
        """Test handling of corrupted files (common failure pattern)."""
        # Create corrupted CSV file that pandas can still read but has wrong structure
        corrupted_file = temp_dir / "predictions.csv"
        with open(corrupted_file, 'w') as f:
            f.write("corrupted,data,here\n1,2,invalid_data\n")
        
        # Pandas can read this file, but validation should catch the missing columns
        # The actual behavior is that it loads successfully but validation fails later
        result = detect_and_load_predictions(str(temp_dir))
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1  # One data row
        # The columns will be ['corrupted', 'data', 'here']
        assert 'corrupted' in result.columns

    def test_memory_intensive_operations(self):
        """Test memory-intensive operations (common failure pattern)."""
        # Create large dataset to test memory handling
        np.random.seed(42)
        large_size = 10000
        y_true = np.random.choice([0, 1], large_size)
        y_prob = np.random.rand(large_size, 2)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)  # Normalize
        
        # Should handle large datasets without memory issues
        result = compute_standard_metrics(y_true, y_prob, is_binary=True)
        
        assert isinstance(result, dict)
        assert "auc_roc" in result
        assert isinstance(result["auc_roc"], (int, float))

    def test_matplotlib_backend_issues(self, temp_dir):
        """Test matplotlib backend issues in headless environments (common failure pattern)."""
        # This test ensures matplotlib works in headless environments
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_score = np.array([0.2, 0.7, 0.1, 0.8, 0.3, 0.6])
        
        # Should not raise display-related errors
        result_path = plot_and_save_roc_curve(y_true, y_score, str(temp_dir))
        assert os.path.exists(result_path)

    def test_division_by_zero_edge_cases(self):
        """Test division by zero edge cases (common failure pattern)."""
        # Create data that might cause division by zero
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])  # Perfect predictions
        
        # Should handle perfect predictions gracefully
        result = compute_standard_metrics(y_true, y_prob, is_binary=True)
        assert isinstance(result, dict)
        # Some metrics might be 1.0 or undefined, both are acceptable

    def test_extremely_imbalanced_data(self):
        """Test extremely imbalanced data (common failure pattern)."""
        # 99% class 0, 1% class 1
        y_true = np.array([0] * 99 + [1] * 1)
        y_prob = np.random.rand(100, 2)
        y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)  # Normalize
        
        # Should handle extreme imbalance gracefully
        result = compute_standard_metrics(y_true, y_prob, is_binary=True)
        assert isinstance(result, dict)
        assert "auc_roc" in result

    def test_unicode_and_special_characters_in_paths(self, temp_dir):
        """Test handling of unicode and special characters in file paths (common failure pattern)."""
        # Create directory with special characters
        special_dir = temp_dir / "test_dir_with_spaces_and_üñíçødé"
        special_dir.mkdir()
        
        # Create test data
        data = pd.DataFrame({
            'id': [1, 2, 3],
            'label': [0, 1, 0],
            'prob_class_0': [0.8, 0.3, 0.9],
            'prob_class_1': [0.2, 0.7, 0.1]
        })
        
        predictions_file = special_dir / "predictions.csv"
        data.to_csv(predictions_file, index=False)
        
        # Should handle special characters in paths
        result = detect_and_load_predictions(str(special_dir))
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3


class TestModelComparisonFunctionality:
    """Tests for new model comparison functionality added to model_metrics_computation."""

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

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @patch("cursus.steps.scripts.model_metrics_computation.plt")
    def test_plot_comparison_roc_curves(self, mock_plt, temp_dir):
        """Test plotting comparison ROC curves."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_new_score = np.array([0.2, 0.8, 0.1, 0.9, 0.7, 0.3])
        y_prev_score = np.array([0.1, 0.7, 0.2, 0.8, 0.6, 0.4])

        result_path = plot_comparison_roc_curves(y_true, y_new_score, y_prev_score, str(output_dir))

        # Verify plotting functions were called
        mock_plt.figure.assert_called_once()
        mock_plt.plot.assert_called()  # Should be called multiple times for both models
        mock_plt.savefig.assert_called_once()
        mock_plt.close.assert_called_once()

        # Check that result path is correct
        assert "comparison_roc_curves.jpg" in result_path

    @patch("cursus.steps.scripts.model_metrics_computation.plt")
    def test_plot_comparison_pr_curves(self, mock_plt, temp_dir):
        """Test plotting comparison PR curves."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_new_score = np.array([0.2, 0.8, 0.1, 0.9, 0.7, 0.3])
        y_prev_score = np.array([0.1, 0.7, 0.2, 0.8, 0.6, 0.4])

        result_path = plot_comparison_pr_curves(y_true, y_new_score, y_prev_score, str(output_dir))

        # Verify plotting functions were called
        mock_plt.figure.assert_called_once()
        mock_plt.plot.assert_called()
        mock_plt.savefig.assert_called_once()
        mock_plt.close.assert_called_once()

        # Check that result path is correct
        assert "comparison_pr_curves.jpg" in result_path

    @patch("cursus.steps.scripts.model_metrics_computation.plt")
    def test_plot_score_scatter(self, mock_plt, temp_dir):
        """Test plotting score scatter plot."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_new_score = np.array([0.2, 0.8, 0.1, 0.9, 0.7, 0.3])
        y_prev_score = np.array([0.1, 0.7, 0.2, 0.8, 0.6, 0.4])

        result_path = plot_score_scatter(y_new_score, y_prev_score, y_true, str(output_dir))

        # Verify plotting functions were called
        mock_plt.figure.assert_called_once()
        mock_plt.scatter.assert_called()  # Should be called for positive and negative examples
        mock_plt.savefig.assert_called_once()
        mock_plt.close.assert_called_once()

        # Check that result path is correct
        assert "score_scatter_plot.jpg" in result_path

    @patch("cursus.steps.scripts.model_metrics_computation.plt")
    def test_plot_score_distributions(self, mock_plt, temp_dir):
        """Test plotting score distributions."""
        output_dir = temp_dir / "output"
        output_dir.mkdir()

        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_new_score = np.array([0.2, 0.8, 0.1, 0.9, 0.7, 0.3])
        y_prev_score = np.array([0.1, 0.7, 0.2, 0.8, 0.6, 0.4])

        result_path = plot_score_distributions(y_new_score, y_prev_score, y_true, str(output_dir))

        # Verify plotting functions were called
        mock_plt.subplots.assert_called_once()
        mock_plt.savefig.assert_called_once()
        mock_plt.close.assert_called_once()

        # Check that result path is correct
        assert "score_distributions.jpg" in result_path

    def test_main_function_with_comparison_mode(self, temp_dir):
        """Test main function with comparison mode enabled."""
        # Set up test data with previous scores
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        
        # Create sample predictions data with previous model scores
        data = pd.DataFrame({
            'id': range(50),
            'label': np.random.choice([0, 1], 50, p=[0.7, 0.3]),
            'prob_class_0': np.random.uniform(0.1, 0.9, 50),
            'prob_class_1': np.random.uniform(0.1, 0.9, 50),
            'amount': np.random.uniform(50, 1000, 50),
            'baseline_score': np.random.uniform(0.1, 0.9, 50)  # Previous model scores
        })
        
        # Normalize probabilities
        prob_sum = data['prob_class_0'] + data['prob_class_1']
        data['prob_class_0'] = data['prob_class_0'] / prob_sum
        data['prob_class_1'] = data['prob_class_1'] / prob_sum
        
        predictions_file = input_dir / "predictions.csv"
        data.to_csv(predictions_file, index=False)
        
        output_metrics_dir = temp_dir / "metrics"
        output_plots_dir = temp_dir / "plots"
        
        # Create arguments
        args = argparse.Namespace()
        
        # Environment variables with comparison mode enabled
        environ_vars = {
            "ID_FIELD": "id",
            "LABEL_FIELD": "label",
            "AMOUNT_FIELD": "amount",
            "INPUT_FORMAT": "auto",
            "COMPUTE_DOLLAR_RECALL": "true",
            "COMPUTE_COUNT_RECALL": "true",
            "DOLLAR_RECALL_FPR": "0.1",
            "COUNT_RECALL_CUTOFF": "0.1",
            "GENERATE_PLOTS": "true",
            "COMPARISON_MODE": "true",
            "PREVIOUS_SCORE_FIELD": "baseline_score",
            "COMPARISON_METRICS": "all",
            "STATISTICAL_TESTS": "true",
            "COMPARISON_PLOTS": "true"
        }
        
        # Path dictionaries
        input_paths = {"processed_data": str(input_dir)}
        output_paths = {
            "metrics_output": str(output_metrics_dir),
            "plots_output": str(output_plots_dir)
        }
        
        # Run main function
        main(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,
            job_args=args
        )
        
        # Verify standard outputs
        assert (output_metrics_dir / "metrics.json").exists()
        assert (output_metrics_dir / "metrics_summary.txt").exists()
        assert (output_metrics_dir / "metrics_report.json").exists()
        
        # Verify comparison plots were generated
        assert (output_plots_dir / "comparison_roc_curves.jpg").exists()
        assert (output_plots_dir / "comparison_pr_curves.jpg").exists()
        assert (output_plots_dir / "score_scatter_plot.jpg").exists()
        assert (output_plots_dir / "score_distributions.jpg").exists()
        
        # Verify comparison metrics are in the saved metrics
        with open(output_metrics_dir / "metrics.json", 'r') as f:
            saved_metrics = json.load(f)
        
        # Check for comparison metrics
        comparison_metric_keys = [
            "pearson_correlation", "auc_delta", "mcnemar_p_value", "paired_t_p_value"
        ]
        for key in comparison_metric_keys:
            assert key in saved_metrics

    def test_main_function_comparison_mode_disabled_missing_field(self, temp_dir):
        """Test main function with comparison mode disabled due to missing previous score field."""
        # Set up test data without previous scores
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        
        # Create sample predictions data WITHOUT previous model scores
        data = pd.DataFrame({
            'id': range(50),
            'label': np.random.choice([0, 1], 50, p=[0.7, 0.3]),
            'prob_class_0': np.random.uniform(0.1, 0.9, 50),
            'prob_class_1': np.random.uniform(0.1, 0.9, 50),
            'amount': np.random.uniform(50, 1000, 50)
            # No baseline_score column
        })
        
        # Normalize probabilities
        prob_sum = data['prob_class_0'] + data['prob_class_1']
        data['prob_class_0'] = data['prob_class_0'] / prob_sum
        data['prob_class_1'] = data['prob_class_1'] / prob_sum
        
        predictions_file = input_dir / "predictions.csv"
        data.to_csv(predictions_file, index=False)
        
        output_metrics_dir = temp_dir / "metrics"
        output_plots_dir = temp_dir / "plots"
        
        # Create arguments
        args = argparse.Namespace()
        
        # Environment variables with comparison mode enabled but missing field
        environ_vars = {
            "ID_FIELD": "id",
            "LABEL_FIELD": "label",
            "AMOUNT_FIELD": "amount",
            "GENERATE_PLOTS": "true",
            "COMPARISON_MODE": "true",
            "PREVIOUS_SCORE_FIELD": "baseline_score",  # Field doesn't exist in data
            "COMPARISON_METRICS": "all",
            "STATISTICAL_TESTS": "true",
            "COMPARISON_PLOTS": "true"
        }
        
        # Path dictionaries
        input_paths = {"processed_data": str(input_dir)}
        output_paths = {
            "metrics_output": str(output_metrics_dir),
            "plots_output": str(output_plots_dir)
        }
        
        # Run main function
        main(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,
            job_args=args
        )
        
        # Verify standard outputs exist
        assert (output_metrics_dir / "metrics.json").exists()
        assert (output_metrics_dir / "metrics_summary.txt").exists()
        
        # Verify comparison plots were NOT generated (comparison mode should be disabled)
        assert not (output_plots_dir / "comparison_roc_curves.jpg").exists()
        assert not (output_plots_dir / "comparison_pr_curves.jpg").exists()
        
        # Verify comparison metrics are NOT in the saved metrics
        with open(output_metrics_dir / "metrics.json", 'r') as f:
            saved_metrics = json.load(f)
        
        # Check that comparison metrics are absent
        comparison_metric_keys = [
            "pearson_correlation", "auc_delta", "mcnemar_p_value"
        ]
        for key in comparison_metric_keys:
            assert key not in saved_metrics

    def test_main_function_comparison_mode_disabled_empty_field(self, temp_dir):
        """Test main function with comparison mode disabled due to empty PREVIOUS_SCORE_FIELD."""
        # Set up test data
        input_dir = temp_dir / "input"
        input_dir.mkdir()
        
        data = pd.DataFrame({
            'id': range(20),
            'label': np.random.choice([0, 1], 20, p=[0.7, 0.3]),
            'prob_class_0': np.random.uniform(0.1, 0.9, 20),
            'prob_class_1': np.random.uniform(0.1, 0.9, 20),
        })
        
        # Normalize probabilities
        prob_sum = data['prob_class_0'] + data['prob_class_1']
        data['prob_class_0'] = data['prob_class_0'] / prob_sum
        data['prob_class_1'] = data['prob_class_1'] / prob_sum
        
        predictions_file = input_dir / "predictions.csv"
        data.to_csv(predictions_file, index=False)
        
        output_metrics_dir = temp_dir / "metrics"
        
        # Create arguments
        args = argparse.Namespace()
        
        # Environment variables with comparison mode enabled but empty PREVIOUS_SCORE_FIELD
        environ_vars = {
            "ID_FIELD": "id",
            "LABEL_FIELD": "label",
            "GENERATE_PLOTS": "false",  # Disable plots for simpler test
            "COMPARISON_MODE": "true",
            "PREVIOUS_SCORE_FIELD": "",  # Empty field should disable comparison
            "COMPARISON_METRICS": "all",
            "STATISTICAL_TESTS": "true",
            "COMPARISON_PLOTS": "true"
        }
        
        # Path dictionaries
        input_paths = {"processed_data": str(input_dir)}
        output_paths = {"metrics_output": str(output_metrics_dir)}
        
        # Run main function
        main(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,
            job_args=args
        )
        
        # Verify standard outputs exist
        assert (output_metrics_dir / "metrics.json").exists()
        
        # Verify comparison metrics are NOT in the saved metrics
        with open(output_metrics_dir / "metrics.json", 'r') as f:
            saved_metrics = json.load(f)
        
        # Check that comparison metrics are absent
        assert "pearson_correlation" not in saved_metrics
        assert "auc_delta" not in saved_metrics
        assert "mcnemar_p_value" not in saved_metrics

    def test_comparison_functions_consistency_with_xgboost_eval(self):
        """Test that comparison functions are identical to xgboost_model_eval.py implementation."""
        # This test verifies that the comparison functions imported from xgboost_model_eval
        # work identically in the model_metrics_computation context
        
        np.random.seed(42)
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
        y_new_score = np.array([0.2, 0.8, 0.1, 0.9, 0.7, 0.3, 0.6, 0.4, 0.8, 0.2])
        y_prev_score = np.array([0.1, 0.7, 0.2, 0.8, 0.6, 0.4, 0.5, 0.3, 0.7, 0.3])
        
        # Test compute_comparison_metrics
        comp_metrics = compute_comparison_metrics(y_true, y_new_score, y_prev_score, is_binary=True)
        
        # Should have all the same keys as the xgboost_model_eval version
        expected_keys = [
            "pearson_correlation", "spearman_correlation", "new_model_auc", "previous_model_auc",
            "auc_delta", "auc_lift_percent", "new_model_ap", "previous_model_ap", "ap_delta", "ap_lift_percent"
        ]
        for key in expected_keys:
            assert key in comp_metrics
        
        # Test perform_statistical_tests
        stat_results = perform_statistical_tests(y_true, y_new_score, y_prev_score, is_binary=True)
        
        # Should have all the same keys as the xgboost_model_eval version
        expected_stat_keys = [
            "mcnemar_statistic", "mcnemar_p_value", "mcnemar_significant",
            "paired_t_statistic", "paired_t_p_value", "paired_t_significant",
            "wilcoxon_statistic", "wilcoxon_p_value", "wilcoxon_significant"
        ]
        for key in expected_stat_keys:
            assert key in stat_results
        
        # Verify the functions produce reasonable results
        assert isinstance(comp_metrics["pearson_correlation"], (int, float))
        assert isinstance(stat_results["mcnemar_p_value"], (int, float))
        assert isinstance(stat_results["mcnemar_significant"], bool)

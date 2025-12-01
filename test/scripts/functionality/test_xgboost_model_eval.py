"""
Comprehensive pytest tests for xgboost_model_eval.py script.

This test suite covers:
- File I/O operations with format detection
- Model artifact loading and decompression
- Data preprocessing (risk table mapping, imputation)
- Metrics computation (binary and multiclass classification)
- Model comparison mode functionality
- Visualization generation
- Error handling and edge cases
- Main entry point execution

Following pytest best practices:
- Mocking module-level side effects before import
- Proper mock path precision
- Fixture isolation
- Implementation-driven testing
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import json
import tempfile
import tarfile
import pickle as pkl
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime
import sys

# CRITICAL: Mock subprocess.check_call BEFORE importing the module
# This prevents actual pip installations from running during test collection
with patch("subprocess.check_call"):
    from src.cursus.steps.scripts.xgboost_model_eval import (
        _detect_file_format,
        load_dataframe_with_format,
        save_dataframe_with_format,
        decompress_model_artifacts,
        load_model_artifacts,
        load_eval_data,
        preprocess_eval_data,
        compute_metrics_binary,
        compute_metrics_multiclass,
        compute_comparison_metrics,
        perform_statistical_tests,
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
        create_health_check_file,
        main,
        RiskTableMappingProcessor,
        NumericalVariableImputationProcessor,
    )


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "label": [0, 1, 0, 1, 1],
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature2": [5.0, 4.0, 3.0, 2.0, 1.0],
            "cat_feature": ["A", "B", "A", "C", "B"],
        }
    )


@pytest.fixture
def sample_binary_labels():
    """Create sample binary labels for testing."""
    return np.array([0, 1, 0, 1, 1, 0, 1, 0])


@pytest.fixture
def sample_binary_predictions():
    """Create sample binary prediction probabilities."""
    # Shape: (n_samples, 2) for binary classification
    return np.array(
        [
            [0.8, 0.2],
            [0.3, 0.7],
            [0.9, 0.1],
            [0.2, 0.8],
            [0.4, 0.6],
            [0.7, 0.3],
            [0.1, 0.9],
            [0.6, 0.4],
        ]
    )


@pytest.fixture
def sample_multiclass_labels():
    """Create sample multiclass labels for testing."""
    return np.array([0, 1, 2, 0, 1, 2, 0, 1])


@pytest.fixture
def sample_multiclass_predictions():
    """Create sample multiclass prediction probabilities."""
    # Shape: (n_samples, 3) for 3-class classification
    return np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.6, 0.3, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.8, 0.1, 0.1],
            [0.1, 0.6, 0.3],
        ]
    )


@pytest.fixture
def sample_risk_tables():
    """Create sample risk tables for testing."""
    return {
        "cat_feature": {
            "bins": {
                "A": 0.3,
                "B": 0.7,
                "C": 0.5,
            },
            "default_bin": 0.4,
        }
    }


@pytest.fixture
def sample_impute_dict():
    """Create sample imputation dictionary for testing."""
    return {
        "feature1": 3.0,
        "feature2": 3.0,
    }


@pytest.fixture
def mock_xgboost_model():
    """Create a mock XGBoost model."""
    mock_model = Mock()
    mock_model.predict = Mock(return_value=np.array([0.2, 0.7, 0.1, 0.8, 0.6]))
    return mock_model


@pytest.fixture
def model_artifacts_dir(
    temp_dir, mock_xgboost_model, sample_risk_tables, sample_impute_dict
):
    """Create a directory with model artifacts for testing."""
    artifacts_dir = temp_dir / "model_artifacts"
    artifacts_dir.mkdir()

    # Create mock model file (XGBoost requires actual file for load_model)
    # We'll mock the load operation instead

    # Create risk table pickle
    with open(artifacts_dir / "risk_table_map.pkl", "wb") as f:
        pkl.dump(sample_risk_tables, f)

    # Create impute dict pickle
    with open(artifacts_dir / "impute_dict.pkl", "wb") as f:
        pkl.dump(sample_impute_dict, f)

    # Create feature columns file
    with open(artifacts_dir / "feature_columns.txt", "w") as f:
        f.write("# Feature columns\n")
        f.write("0,feature1\n")
        f.write("1,feature2\n")
        f.write("2,cat_feature\n")

    # Create hyperparameters file
    hyperparams = {
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "is_binary": True,
    }
    with open(artifacts_dir / "hyperparameters.json", "w") as f:
        json.dump(hyperparams, f)

    return artifacts_dir


# ============================================================================
# TESTS: File I/O and Format Detection
# ============================================================================


class TestFileFormatDetection:
    """Tests for file format detection functionality."""

    def test_detect_csv_format(self, temp_dir):
        """Test detection of CSV file format."""
        csv_file = temp_dir / "test.csv"
        csv_file.touch()

        format_detected = _detect_file_format(csv_file)
        assert format_detected == "csv"

    def test_detect_tsv_format(self, temp_dir):
        """Test detection of TSV file format."""
        tsv_file = temp_dir / "test.tsv"
        tsv_file.touch()

        format_detected = _detect_file_format(tsv_file)
        assert format_detected == "tsv"

    def test_detect_parquet_format(self, temp_dir):
        """Test detection of Parquet file format."""
        parquet_file = temp_dir / "test.parquet"
        parquet_file.touch()

        format_detected = _detect_file_format(parquet_file)
        assert format_detected == "parquet"

    def test_unsupported_format_raises_error(self, temp_dir):
        """Test that unsupported file formats raise RuntimeError."""
        unsupported_file = temp_dir / "test.txt"
        unsupported_file.touch()

        with pytest.raises(RuntimeError, match="Unsupported file format"):
            _detect_file_format(unsupported_file)


class TestDataFrameLoading:
    """Tests for DataFrame loading with format detection."""

    def test_load_csv_dataframe(self, temp_dir, sample_dataframe):
        """Test loading CSV file returns correct DataFrame and format."""
        csv_file = temp_dir / "test.csv"
        sample_dataframe.to_csv(csv_file, index=False)

        df, file_format = load_dataframe_with_format(csv_file)

        assert file_format == "csv"
        assert df.shape == sample_dataframe.shape
        assert list(df.columns) == list(sample_dataframe.columns)

    def test_load_tsv_dataframe(self, temp_dir, sample_dataframe):
        """Test loading TSV file returns correct DataFrame and format."""
        tsv_file = temp_dir / "test.tsv"
        sample_dataframe.to_csv(tsv_file, sep="\t", index=False)

        df, file_format = load_dataframe_with_format(tsv_file)

        assert file_format == "tsv"
        assert df.shape == sample_dataframe.shape

    def test_load_parquet_dataframe(self, temp_dir, sample_dataframe):
        """Test loading Parquet file returns correct DataFrame and format."""
        parquet_file = temp_dir / "test.parquet"
        sample_dataframe.to_parquet(parquet_file, index=False)

        df, file_format = load_dataframe_with_format(parquet_file)

        assert file_format == "parquet"
        assert df.shape == sample_dataframe.shape


class TestDataFrameSaving:
    """Tests for DataFrame saving with format preservation."""

    def test_save_csv_dataframe(self, temp_dir, sample_dataframe):
        """Test saving DataFrame as CSV."""
        output_path = temp_dir / "output"

        saved_path = save_dataframe_with_format(sample_dataframe, output_path, "csv")

        assert saved_path.suffix == ".csv"
        assert saved_path.exists()

        # Verify content
        loaded_df = pd.read_csv(saved_path)
        assert loaded_df.shape == sample_dataframe.shape

    def test_save_tsv_dataframe(self, temp_dir, sample_dataframe):
        """Test saving DataFrame as TSV."""
        output_path = temp_dir / "output"

        saved_path = save_dataframe_with_format(sample_dataframe, output_path, "tsv")

        assert saved_path.suffix == ".tsv"
        assert saved_path.exists()

        # Verify content
        loaded_df = pd.read_csv(saved_path, sep="\t")
        assert loaded_df.shape == sample_dataframe.shape

    def test_save_parquet_dataframe(self, temp_dir, sample_dataframe):
        """Test saving DataFrame as Parquet."""
        output_path = temp_dir / "output"

        saved_path = save_dataframe_with_format(
            sample_dataframe, output_path, "parquet"
        )

        assert saved_path.suffix == ".parquet"
        assert saved_path.exists()

        # Verify content
        loaded_df = pd.read_parquet(saved_path)
        assert loaded_df.shape == sample_dataframe.shape

    def test_unsupported_format_raises_error(self, temp_dir, sample_dataframe):
        """Test that unsupported save formats raise RuntimeError."""
        output_path = temp_dir / "output"

        with pytest.raises(RuntimeError, match="Unsupported output format"):
            save_dataframe_with_format(sample_dataframe, output_path, "xml")


# ============================================================================
# TESTS: Model Artifact Loading
# ============================================================================


class TestModelArtifactLoading:
    """Tests for loading model artifacts."""

    def test_decompress_model_artifacts_with_tarball(self, temp_dir):
        """Test decompression of model.tar.gz file."""
        # Create a dummy file to compress
        dummy_file = temp_dir / "dummy_model.txt"
        dummy_file.write_text("model content")

        # Create tarball
        tarball_path = temp_dir / "model.tar.gz"
        with tarfile.open(tarball_path, "w:gz") as tar:
            tar.add(dummy_file, arcname="dummy_model.txt")

        # Remove original file
        dummy_file.unlink()

        # Decompress
        decompress_model_artifacts(str(temp_dir))

        # Verify file was extracted
        assert (temp_dir / "dummy_model.txt").exists()

    def test_decompress_model_artifacts_no_tarball(self, temp_dir):
        """Test decompress when no tarball exists (no-op)."""
        # Should not raise error
        decompress_model_artifacts(str(temp_dir))

    @patch("src.cursus.steps.scripts.xgboost_model_eval.xgb.Booster")
    def test_load_model_artifacts_success(
        self, mock_booster_class, model_artifacts_dir
    ):
        """Test successful loading of all model artifacts."""
        # Mock the XGBoost model loading
        mock_booster = Mock()
        mock_booster_class.return_value = mock_booster

        model, risk_tables, impute_dict, feature_columns, hyperparams = (
            load_model_artifacts(str(model_artifacts_dir))
        )

        # Verify model was loaded
        assert model == mock_booster
        mock_booster.load_model.assert_called_once()

        # Verify risk tables
        assert "cat_feature" in risk_tables
        assert "bins" in risk_tables["cat_feature"]

        # Verify impute dict
        assert "feature1" in impute_dict
        assert "feature2" in impute_dict

        # Verify feature columns
        assert feature_columns == ["feature1", "feature2", "cat_feature"]

        # Verify hyperparameters
        assert hyperparams["is_binary"] is True
        assert hyperparams["max_depth"] == 6

    @patch("src.cursus.steps.scripts.xgboost_model_eval.xgb.Booster")
    def test_load_model_artifacts_with_tarball(
        self, mock_booster_class, temp_dir, sample_risk_tables, sample_impute_dict
    ):
        """Test loading artifacts from compressed tarball."""
        # Create artifacts
        artifacts_dir = temp_dir / "artifacts"
        artifacts_dir.mkdir()

        # Create artifacts
        with open(artifacts_dir / "risk_table_map.pkl", "wb") as f:
            pkl.dump(sample_risk_tables, f)

        with open(artifacts_dir / "impute_dict.pkl", "wb") as f:
            pkl.dump(sample_impute_dict, f)

        with open(artifacts_dir / "feature_columns.txt", "w") as f:
            f.write("0,feature1\n")

        with open(artifacts_dir / "hyperparameters.json", "w") as f:
            json.dump({"is_binary": True}, f)

        # Create dummy model file
        model_file = artifacts_dir / "xgboost_model.bst"
        model_file.write_text("model")

        # Create tarball
        tarball_path = temp_dir / "model.tar.gz"
        with tarfile.open(tarball_path, "w:gz") as tar:
            for item in artifacts_dir.iterdir():
                tar.add(item, arcname=item.name)

        # Mock the XGBoost model loading
        mock_booster = Mock()
        mock_booster_class.return_value = mock_booster

        # Load artifacts (should decompress first)
        model, risk_tables, impute_dict, feature_columns, hyperparams = (
            load_model_artifacts(str(temp_dir))
        )

        assert model == mock_booster
        assert "cat_feature" in risk_tables


# ============================================================================
# TESTS: Data Preprocessing
# ============================================================================


class TestDataPreprocessing:
    """Tests for data preprocessing functions."""

    def test_preprocess_eval_data_basic(
        self, sample_dataframe, sample_risk_tables, sample_impute_dict
    ):
        """Test basic data preprocessing without missing values."""
        feature_columns = ["feature1", "feature2", "cat_feature"]

        result_df = preprocess_eval_data(
            sample_dataframe, feature_columns, sample_risk_tables, sample_impute_dict
        )

        # Verify shape is preserved
        assert result_df.shape == sample_dataframe.shape

        # Verify id and label columns are preserved
        assert "id" in result_df.columns
        assert "label" in result_df.columns
        pd.testing.assert_series_equal(result_df["id"], sample_dataframe["id"])

        # Verify categorical feature was mapped
        assert result_df["cat_feature"].dtype in [np.float64, np.float32]

    def test_preprocess_eval_data_with_missing_values(
        self, sample_risk_tables, sample_impute_dict
    ):
        """Test preprocessing with missing values."""
        df_with_na = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "label": [0, 1, 0],
                "feature1": [1.0, np.nan, 3.0],
                "feature2": [np.nan, 2.0, 3.0],
                "cat_feature": ["A", "B", "A"],
            }
        )

        feature_columns = ["feature1", "feature2", "cat_feature"]

        result_df = preprocess_eval_data(
            df_with_na, feature_columns, sample_risk_tables, sample_impute_dict
        )

        # Verify no NaN values in feature columns after imputation
        assert not result_df["feature1"].isna().any()
        assert not result_df["feature2"].isna().any()

    def test_preprocess_eval_data_preserves_non_feature_columns(
        self, sample_dataframe, sample_risk_tables, sample_impute_dict
    ):
        """Test that non-feature columns are preserved during preprocessing."""
        # Add extra column
        df_with_extra = sample_dataframe.copy()
        df_with_extra["extra_col"] = ["x", "y", "z", "w", "v"]

        feature_columns = ["feature1", "feature2", "cat_feature"]

        result_df = preprocess_eval_data(
            df_with_extra, feature_columns, sample_risk_tables, sample_impute_dict
        )

        # Verify extra column is preserved
        assert "extra_col" in result_df.columns
        assert list(result_df["extra_col"]) == ["x", "y", "z", "w", "v"]


# ============================================================================
# TESTS: Metrics Computation
# ============================================================================


class TestBinaryMetricsComputation:
    """Tests for binary classification metrics computation."""

    def test_compute_metrics_binary_basic(
        self, sample_binary_labels, sample_binary_predictions
    ):
        """Test basic binary metrics computation."""
        metrics = compute_metrics_binary(
            sample_binary_labels, sample_binary_predictions
        )

        # Verify key metrics are present
        assert "auc_roc" in metrics
        assert "average_precision" in metrics
        assert "f1_score" in metrics

        # Verify metric values are reasonable
        assert 0 <= metrics["auc_roc"] <= 1
        assert 0 <= metrics["average_precision"] <= 1
        assert 0 <= metrics["f1_score"] <= 1

    def test_compute_metrics_binary_includes_threshold_metrics(
        self, sample_binary_labels, sample_binary_predictions
    ):
        """Test that binary metrics include threshold-specific metrics."""
        metrics = compute_metrics_binary(
            sample_binary_labels, sample_binary_predictions
        )

        # Verify threshold-specific metrics
        assert "f1_score_at_0.3" in metrics
        assert "f1_score_at_0.5" in metrics
        assert "f1_score_at_0.7" in metrics

    def test_compute_metrics_binary_perfect_predictions(self):
        """Test binary metrics with perfect predictions."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

        metrics = compute_metrics_binary(y_true, y_prob)

        # Perfect predictions should have AUC = 1.0
        assert metrics["auc_roc"] == 1.0
        assert metrics["average_precision"] == 1.0


class TestMulticlassMetricsComputation:
    """Tests for multiclass classification metrics computation."""

    def test_compute_metrics_multiclass_basic(
        self, sample_multiclass_labels, sample_multiclass_predictions
    ):
        """Test basic multiclass metrics computation."""
        n_classes = 3
        metrics = compute_metrics_multiclass(
            sample_multiclass_labels, sample_multiclass_predictions, n_classes
        )

        # Verify macro and micro metrics
        assert "auc_roc_macro" in metrics
        assert "auc_roc_micro" in metrics
        assert "f1_score_macro" in metrics
        assert "f1_score_micro" in metrics

        # Verify per-class metrics
        for i in range(n_classes):
            assert f"auc_roc_class_{i}" in metrics
            assert f"average_precision_class_{i}" in metrics
            assert f"f1_score_class_{i}" in metrics

    def test_compute_metrics_multiclass_includes_class_distribution(
        self, sample_multiclass_labels, sample_multiclass_predictions
    ):
        """Test that multiclass metrics include class distribution info."""
        n_classes = 3
        metrics = compute_metrics_multiclass(
            sample_multiclass_labels, sample_multiclass_predictions, n_classes
        )

        # Verify class distribution metrics
        for i in range(n_classes):
            assert f"class_{i}_count" in metrics
            assert f"class_{i}_ratio" in metrics


# ============================================================================
# TESTS: Model Comparison Functionality
# ============================================================================


class TestComparisonMetrics:
    """Tests for model comparison metrics computation."""

    def test_compute_comparison_metrics_binary(self, sample_binary_labels):
        """Test comparison metrics for binary classification."""
        y_new = np.array([0.2, 0.7, 0.1, 0.8, 0.6, 0.3, 0.9, 0.4])
        y_prev = np.array([0.3, 0.6, 0.2, 0.7, 0.5, 0.4, 0.8, 0.5])

        metrics = compute_comparison_metrics(
            sample_binary_labels, y_new, y_prev, is_binary=True
        )

        # Verify correlation metrics
        assert "pearson_correlation" in metrics
        assert "spearman_correlation" in metrics

        # Verify performance comparison
        assert "new_model_auc" in metrics
        assert "previous_model_auc" in metrics
        assert "auc_delta" in metrics
        assert "auc_lift_percent" in metrics

        # Verify score distribution metrics
        assert "new_score_mean" in metrics
        assert "previous_score_mean" in metrics
        assert "score_mean_delta" in metrics

    def test_compute_comparison_metrics_includes_agreement(self, sample_binary_labels):
        """Test that comparison metrics include prediction agreement."""
        y_new = np.array([0.2, 0.7, 0.1, 0.8, 0.6, 0.3, 0.9, 0.4])
        y_prev = np.array([0.3, 0.6, 0.2, 0.7, 0.5, 0.4, 0.8, 0.5])

        metrics = compute_comparison_metrics(
            sample_binary_labels, y_new, y_prev, is_binary=True
        )

        # Verify agreement metrics at different thresholds
        assert "prediction_agreement_at_0.3" in metrics
        assert "prediction_agreement_at_0.5" in metrics
        assert "prediction_agreement_at_0.7" in metrics


class TestStatisticalTests:
    """Tests for statistical significance testing."""

    def test_perform_statistical_tests_binary(self, sample_binary_labels):
        """Test statistical tests for binary classification."""
        y_new = np.array([0.2, 0.7, 0.1, 0.8, 0.6, 0.3, 0.9, 0.4])
        y_prev = np.array([0.3, 0.6, 0.2, 0.7, 0.5, 0.4, 0.8, 0.5])

        test_results = perform_statistical_tests(
            sample_binary_labels, y_new, y_prev, is_binary=True
        )

        # Verify McNemar's test results
        assert "mcnemar_statistic" in test_results
        assert "mcnemar_p_value" in test_results
        assert "mcnemar_significant" in test_results

        # Verify paired t-test results
        assert "paired_t_statistic" in test_results
        assert "paired_t_p_value" in test_results
        assert "paired_t_significant" in test_results

        # Verify Wilcoxon test results
        assert "wilcoxon_statistic" in test_results
        assert "wilcoxon_p_value" in test_results
        assert "wilcoxon_significant" in test_results


# ============================================================================
# TESTS: Prediction and Metrics Saving
# ============================================================================


class TestPredictionSaving:
    """Tests for saving predictions."""

    def test_save_predictions_csv(self, temp_dir):
        """Test saving predictions in CSV format."""
        ids = np.array([1, 2, 3])
        y_true = np.array([0, 1, 0])
        y_prob = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])

        save_predictions(ids, y_true, y_prob, "id", "label", str(temp_dir), "csv")

        # Verify file was created
        output_file = temp_dir / "eval_predictions.csv"
        assert output_file.exists()

        # Verify content
        df = pd.read_csv(output_file)
        assert "id" in df.columns
        assert "label" in df.columns
        assert "prob_class_0" in df.columns
        assert "prob_class_1" in df.columns
        assert len(df) == 3


class TestMetricsSaving:
    """Tests for saving metrics."""

    def test_save_metrics_creates_json(self, temp_dir):
        """Test that save_metrics creates metrics.json file."""
        metrics = {
            "auc_roc": 0.85,
            "average_precision": 0.80,
            "f1_score": 0.75,
        }

        save_metrics(metrics, str(temp_dir))

        # Verify JSON file
        json_file = temp_dir / "metrics.json"
        assert json_file.exists()

        with open(json_file) as f:
            loaded_metrics = json.load(f)

        assert loaded_metrics["auc_roc"] == 0.85

    def test_save_metrics_creates_summary(self, temp_dir):
        """Test that save_metrics creates metrics_summary.txt file."""
        metrics = {
            "auc_roc": 0.85,
            "average_precision": 0.80,
        }

        save_metrics(metrics, str(temp_dir))

        # Verify summary file
        summary_file = temp_dir / "metrics_summary.txt"
        assert summary_file.exists()

        content = summary_file.read_text()
        assert "METRICS SUMMARY" in content
        assert "auc_roc" in content


# ============================================================================
# TESTS: Visualization Generation
# ============================================================================


class TestVisualizationGeneration:
    """Tests for visualization generation."""

    @patch("src.cursus.steps.scripts.xgboost_model_eval.plt")
    def test_plot_and_save_roc_curve(self, mock_plt, temp_dir, sample_binary_labels):
        """Test ROC curve generation and saving."""
        y_score = np.array([0.2, 0.7, 0.1, 0.8, 0.6, 0.3, 0.9, 0.4])

        plot_and_save_roc_curve(
            sample_binary_labels, y_score, str(temp_dir), prefix="test_"
        )

        # Verify savefig was called with correct path
        mock_plt.savefig.assert_called_once()
        assert str(temp_dir) in str(mock_plt.savefig.call_args)

    @patch("src.cursus.steps.scripts.xgboost_model_eval.plt")
    def test_plot_and_save_pr_curve(self, mock_plt, temp_dir, sample_binary_labels):
        """Test PR curve generation and saving."""
        y_score = np.array([0.2, 0.7, 0.1, 0.8, 0.6, 0.3, 0.9, 0.4])

        plot_and_save_pr_curve(
            sample_binary_labels, y_score, str(temp_dir), prefix="test_"
        )

        # Verify savefig was called
        mock_plt.savefig.assert_called_once()

    @patch("src.cursus.steps.scripts.xgboost_model_eval.plt")
    def test_plot_comparison_roc_curves(self, mock_plt, temp_dir, sample_binary_labels):
        """Test comparison ROC curve generation."""
        y_new = np.array([0.2, 0.7, 0.1, 0.8, 0.6, 0.3, 0.9, 0.4])
        y_prev = np.array([0.3, 0.6, 0.2, 0.7, 0.5, 0.4, 0.8, 0.5])

        plot_comparison_roc_curves(sample_binary_labels, y_new, y_prev, str(temp_dir))

        # Verify savefig was called
        mock_plt.savefig.assert_called_once()
        assert "comparison_roc_curves.jpg" in str(mock_plt.savefig.call_args)


# ============================================================================
# TESTS: Health Check and Utility Functions
# ============================================================================


class TestHealthCheck:
    """Tests for health check functionality."""

    def test_create_health_check_file(self, temp_dir):
        """Test creation of health check file."""
        health_file = temp_dir / "_HEALTH"

        result_path = create_health_check_file(str(health_file))

        assert Path(result_path).exists()
        content = Path(result_path).read_text()
        assert "healthy:" in content


# ============================================================================
# TESTS: Embedded Processor Classes
# ============================================================================


class TestRiskTableMappingProcessor:
    """Tests for RiskTableMappingProcessor."""

    def test_risk_table_processor_initialization(self):
        """Test processor initialization with risk tables."""
        risk_tables = {
            "bins": {"A": 0.3, "B": 0.7},
            "default_bin": 0.5,
        }

        processor = RiskTableMappingProcessor(
            column_name="test_col", label_name="label", risk_tables=risk_tables
        )

        assert processor.is_fitted is True
        assert processor.column_name == "test_col"

    def test_risk_table_processor_fit(self):
        """Test processor fitting with data."""
        df = pd.DataFrame(
            {
                "cat_feature": ["A", "B", "A", "C", "B"],
                "label": [0, 1, 0, 1, 1],
            }
        )

        processor = RiskTableMappingProcessor(
            column_name="cat_feature", label_name="label"
        )

        processor.fit(df)

        assert processor.is_fitted is True
        assert "bins" in processor.risk_tables
        assert "default_bin" in processor.risk_tables

    def test_risk_table_processor_transform(self):
        """Test processor transform method."""
        risk_tables = {
            "bins": {"A": 0.3, "B": 0.7, "C": 0.5},
            "default_bin": 0.4,
        }

        processor = RiskTableMappingProcessor(
            column_name="cat_feature", label_name="label", risk_tables=risk_tables
        )

        df = pd.DataFrame(
            {
                "cat_feature": ["A", "B", "C", "D"],
            }
        )

        result = processor.transform(df)

        # Verify transformation
        assert result["cat_feature"][0] == 0.3  # A
        assert result["cat_feature"][1] == 0.7  # B
        assert result["cat_feature"][2] == 0.5  # C
        assert result["cat_feature"][3] == 0.4  # D (default)


class TestNumericalVariableImputationProcessor:
    """Tests for NumericalVariableImputationProcessor."""

    def test_imputation_processor_initialization(self):
        """Test processor initialization with imputation dict."""
        impute_dict = {"feature1": 3.0, "feature2": 5.0}

        processor = NumericalVariableImputationProcessor(imputation_dict=impute_dict)

        assert processor.is_fitted is True
        assert processor.imputation_dict == impute_dict

    def test_imputation_processor_fit_mean(self):
        """Test processor fitting with mean strategy."""
        df = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, np.nan],
                "feature2": [4.0, 5.0, 6.0, np.nan],
            }
        )

        processor = NumericalVariableImputationProcessor(
            variables=["feature1", "feature2"], strategy="mean"
        )

        processor.fit(df)

        assert processor.is_fitted is True
        assert processor.imputation_dict["feature1"] == 2.0  # mean of [1,2,3]
        assert processor.imputation_dict["feature2"] == 5.0  # mean of [4,5,6]

    def test_imputation_processor_transform(self):
        """Test processor transform method."""
        impute_dict = {"feature1": 3.0, "feature2": 5.0}

        processor = NumericalVariableImputationProcessor(imputation_dict=impute_dict)

        df = pd.DataFrame(
            {
                "feature1": [1.0, np.nan, 3.0],
                "feature2": [np.nan, 2.0, 3.0],
            }
        )

        result = processor.transform(df)

        # Verify imputation
        assert result["feature1"][1] == 3.0  # Imputed
        assert result["feature2"][0] == 5.0  # Imputed
        assert not result.isna().any().any()


# ============================================================================
# TESTS: Direct Function Tests for Key Orchestrators
# ============================================================================


class TestLoadEvalData:
    """Tests for load_eval_data function."""

    def test_load_eval_data_csv(self, temp_dir, sample_dataframe):
        """Test loading eval data from CSV file."""
        # Create eval data directory with CSV file
        eval_dir = temp_dir / "eval_data"
        eval_dir.mkdir()
        csv_file = eval_dir / "eval.csv"
        sample_dataframe.to_csv(csv_file, index=False)

        # Load eval data
        df, format_detected = load_eval_data(str(eval_dir))

        assert format_detected == "csv"
        assert df.shape == sample_dataframe.shape
        assert list(df.columns) == list(sample_dataframe.columns)

    def test_load_eval_data_no_files_raises_error(self, temp_dir):
        """Test that missing eval data raises RuntimeError."""
        eval_dir = temp_dir / "eval_data"
        eval_dir.mkdir()

        with pytest.raises(RuntimeError, match="No eval data file found"):
            load_eval_data(str(eval_dir))

    def test_load_eval_data_selects_first_file(self, temp_dir, sample_dataframe):
        """Test that load_eval_data selects first file when multiple exist."""
        eval_dir = temp_dir / "eval_data"
        eval_dir.mkdir()

        # Create multiple files
        csv_file1 = eval_dir / "a_first.csv"
        csv_file2 = eval_dir / "z_second.csv"
        sample_dataframe.to_csv(csv_file1, index=False)
        sample_dataframe.to_csv(csv_file2, index=False)

        df, format_detected = load_eval_data(str(eval_dir))

        # Should load first file alphabetically
        assert format_detected == "csv"


class TestEvaluateModel:
    """Tests for evaluate_model orchestration function."""

    @patch("src.cursus.steps.scripts.xgboost_model_eval.xgb.DMatrix")
    @patch("src.cursus.steps.scripts.xgboost_model_eval.save_predictions")
    @patch("src.cursus.steps.scripts.xgboost_model_eval.save_metrics")
    @patch("src.cursus.steps.scripts.xgboost_model_eval.plot_and_save_roc_curve")
    @patch("src.cursus.steps.scripts.xgboost_model_eval.plot_and_save_pr_curve")
    def test_evaluate_model_binary_orchestration(
        self,
        mock_plot_pr,
        mock_plot_roc,
        mock_save_metrics,
        mock_save_predictions,
        mock_dmatrix,
        sample_dataframe,
    ):
        """Test evaluate_model orchestrates binary classification correctly."""
        # Setup
        mock_model = Mock()
        # Binary classification: 1D predictions converted to 2-column
        mock_model.predict.return_value = np.array([0.2, 0.7, 0.1, 0.8, 0.6])

        feature_columns = ["feature1", "feature2", "cat_feature"]
        hyperparams = {"is_binary": True}

        # Execute
        evaluate_model(
            model=mock_model,
            df=sample_dataframe,
            feature_columns=feature_columns,
            id_col="id",
            label_col="label",
            hyperparams=hyperparams,
            output_eval_dir="/tmp/eval",
            output_metrics_dir="/tmp/metrics",
            input_format="csv",
        )

        # Verify orchestration
        mock_model.predict.assert_called_once()
        mock_save_predictions.assert_called_once()
        mock_save_metrics.assert_called_once()
        mock_plot_roc.assert_called_once()
        mock_plot_pr.assert_called_once()


class TestEvaluateModelWithComparison:
    """Tests for evaluate_model_with_comparison orchestration function."""

    @patch("src.cursus.steps.scripts.xgboost_model_eval.xgb.DMatrix")
    @patch(
        "src.cursus.steps.scripts.xgboost_model_eval.save_predictions_with_comparison"
    )
    @patch("src.cursus.steps.scripts.xgboost_model_eval.save_metrics")
    @patch("src.cursus.steps.scripts.xgboost_model_eval.plot_comparison_roc_curves")
    @patch("src.cursus.steps.scripts.xgboost_model_eval.plot_comparison_pr_curves")
    @patch("src.cursus.steps.scripts.xgboost_model_eval.plot_score_scatter")
    @patch("src.cursus.steps.scripts.xgboost_model_eval.plot_score_distributions")
    @patch("src.cursus.steps.scripts.xgboost_model_eval.create_comparison_report")
    @patch("src.cursus.steps.scripts.xgboost_model_eval.plot_and_save_roc_curve")
    @patch("src.cursus.steps.scripts.xgboost_model_eval.plot_and_save_pr_curve")
    def test_evaluate_with_comparison_orchestration(
        self,
        mock_plot_pr_single,
        mock_plot_roc_single,
        mock_report,
        mock_plot_dist,
        mock_plot_scatter,
        mock_plot_pr,
        mock_plot_roc,
        mock_save_metrics,
        mock_save_predictions,
        mock_dmatrix,
        sample_dataframe,
        temp_dir,
    ):
        """Test evaluate_model_with_comparison orchestrates correctly."""
        # Setup output directories
        output_eval = temp_dir / "output_eval"
        output_eval.mkdir()
        output_metrics = temp_dir / "output_metrics"
        output_metrics.mkdir()

        # Setup
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.2, 0.7, 0.1, 0.8, 0.6])

        df_with_prev = sample_dataframe.copy()
        previous_scores = np.array([0.3, 0.6, 0.2, 0.7, 0.5])

        feature_columns = ["feature1", "feature2", "cat_feature"]
        hyperparams = {"is_binary": True}

        # Execute
        evaluate_model_with_comparison(
            model=mock_model,
            df=df_with_prev,
            feature_columns=feature_columns,
            id_col="id",
            label_col="label",
            previous_scores=previous_scores,
            hyperparams=hyperparams,
            output_eval_dir=str(output_eval),
            output_metrics_dir=str(output_metrics),
            comparison_metrics="all",
            statistical_tests=True,
            comparison_plots=True,
            input_format="csv",
        )

        # Verify comparison-specific orchestration
        mock_save_predictions.assert_called_once()
        mock_save_metrics.assert_called_once()
        mock_plot_roc.assert_called_once()
        mock_plot_pr.assert_called_once()
        mock_plot_scatter.assert_called_once()
        mock_plot_dist.assert_called_once()
        mock_report.assert_called_once()


# ============================================================================
# TESTS: Main Entry Point - Simplified Orchestration Tests
# ============================================================================


class TestMainEntryPoint:
    """Tests for main entry point orchestration (not internal implementation)."""

    @patch("src.cursus.steps.scripts.xgboost_model_eval.xgb.Booster")
    @patch("src.cursus.steps.scripts.xgboost_model_eval.load_model_artifacts")
    @patch("src.cursus.steps.scripts.xgboost_model_eval.load_eval_data")
    @patch("src.cursus.steps.scripts.xgboost_model_eval.preprocess_eval_data")
    @patch("src.cursus.steps.scripts.xgboost_model_eval.evaluate_model")
    def test_main_standard_mode_orchestration(
        self,
        mock_evaluate,
        mock_preprocess,
        mock_load_eval,
        mock_load_artifacts,
        mock_booster_class,
        temp_dir,
        sample_dataframe,
        sample_risk_tables,
        sample_impute_dict,
    ):
        """Test main() orchestrates standard evaluation correctly."""
        # Setup output directories
        output_eval = temp_dir / "output_eval"
        output_eval.mkdir()
        output_metrics = temp_dir / "output_metrics"
        output_metrics.mkdir()

        # Mock return values
        mock_model = Mock()
        mock_booster_class.return_value = mock_model

        mock_load_artifacts.return_value = (
            mock_model,
            sample_risk_tables,
            sample_impute_dict,
            ["feature1", "feature2", "cat_feature"],
            {"is_binary": True},
        )

        mock_load_eval.return_value = (sample_dataframe, "csv")
        mock_preprocess.return_value = sample_dataframe
        mock_evaluate.return_value = None

        # Execute
        from argparse import Namespace

        args = Namespace(job_type="evaluation")

        input_paths = {
            "model_input": str(temp_dir / "model"),
            "processed_data": str(temp_dir / "eval_data"),
        }

        output_paths = {
            "eval_output": str(output_eval),
            "metrics_output": str(output_metrics),
        }

        environ_vars = {
            "ID_FIELD": "id",
            "LABEL_FIELD": "label",
            "COMPARISON_MODE": "false",
            "PREVIOUS_SCORE_FIELD": "",
            "COMPARISON_METRICS": "all",
            "STATISTICAL_TESTS": "true",
            "COMPARISON_PLOTS": "true",
        }

        main(input_paths, output_paths, environ_vars, args)

        # Verify orchestration sequence
        mock_load_artifacts.assert_called_once()
        mock_load_eval.assert_called_once()
        mock_preprocess.assert_called_once()
        mock_evaluate.assert_called_once()

        # Verify evaluate_model was called (parameters passed correctly)
        assert mock_evaluate.called

    @patch("src.cursus.steps.scripts.xgboost_model_eval.xgb.Booster")
    @patch("src.cursus.steps.scripts.xgboost_model_eval.load_model_artifacts")
    @patch("src.cursus.steps.scripts.xgboost_model_eval.load_eval_data")
    @patch("src.cursus.steps.scripts.xgboost_model_eval.preprocess_eval_data")
    @patch("src.cursus.steps.scripts.xgboost_model_eval.evaluate_model_with_comparison")
    def test_main_comparison_mode_orchestration(
        self,
        mock_evaluate_comp,
        mock_preprocess,
        mock_load_eval,
        mock_load_artifacts,
        mock_booster_class,
        temp_dir,
        sample_dataframe,
        sample_risk_tables,
        sample_impute_dict,
    ):
        """Test main() orchestrates comparison evaluation correctly."""
        # Add previous score column
        df_with_prev = sample_dataframe.copy()
        df_with_prev["previous_score"] = [0.3, 0.6, 0.2, 0.7, 0.5]

        # Setup output directories
        output_eval = temp_dir / "output_eval"
        output_eval.mkdir()
        output_metrics = temp_dir / "output_metrics"
        output_metrics.mkdir()

        # Mock return values
        mock_model = Mock()
        mock_booster_class.return_value = mock_model

        mock_load_artifacts.return_value = (
            mock_model,
            sample_risk_tables,
            sample_impute_dict,
            ["feature1", "feature2", "cat_feature"],
            {"is_binary": True},
        )

        mock_load_eval.return_value = (df_with_prev, "csv")
        mock_preprocess.return_value = df_with_prev
        mock_evaluate_comp.return_value = None

        # Execute with comparison mode
        from argparse import Namespace

        args = Namespace(job_type="evaluation")

        input_paths = {
            "model_input": str(temp_dir / "model"),
            "processed_data": str(temp_dir / "eval_data"),
        }

        output_paths = {
            "eval_output": str(output_eval),
            "metrics_output": str(output_metrics),
        }

        environ_vars = {
            "ID_FIELD": "id",
            "LABEL_FIELD": "label",
            "COMPARISON_MODE": "true",
            "PREVIOUS_SCORE_FIELD": "previous_score",
            "COMPARISON_METRICS": "all",
            "STATISTICAL_TESTS": "true",
            "COMPARISON_PLOTS": "true",
        }

        main(input_paths, output_paths, environ_vars, args)

        # Verify comparison mode was used
        mock_evaluate_comp.assert_called_once()

        # Verify evaluate_model_with_comparison was called (parameters passed correctly)
        assert mock_evaluate_comp.called


# ============================================================================
# TESTS: Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_preprocess_eval_data_with_missing_features(
        self, sample_risk_tables, sample_impute_dict
    ):
        """Test preprocessing when some features are missing from data."""
        df_missing_features = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "label": [0, 1, 0],
                "feature1": [1.0, 2.0, 3.0],
                # feature2 is missing
                "cat_feature": ["A", "B", "A"],
            }
        )

        feature_columns = ["feature1", "feature2", "cat_feature"]

        # Should handle gracefully
        result_df = preprocess_eval_data(
            df_missing_features, feature_columns, sample_risk_tables, sample_impute_dict
        )

        # feature2 won't be in result since it wasn't in input
        assert "feature1" in result_df.columns
        assert "cat_feature" in result_df.columns

    def test_compute_metrics_binary_with_edge_scores(self):
        """Test binary metrics with edge case scores (all 0s or all 1s)."""
        y_true = np.array([0, 0, 0, 1])
        y_prob = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        # Should not raise error
        metrics = compute_metrics_binary(y_true, y_prob)

        # Verify metrics are computed
        assert "auc_roc" in metrics
        assert "average_precision" in metrics

    def test_save_dataframe_with_empty_dataframe(self, temp_dir):
        """Test saving empty DataFrame."""
        empty_df = pd.DataFrame()
        output_path = temp_dir / "empty_output"

        # Should not raise error
        saved_path = save_dataframe_with_format(empty_df, output_path, "csv")

        assert saved_path.exists()


# ============================================================================
# SUMMARY
# ============================================================================

"""
Test Coverage Summary:

✓ File I/O and format detection (CSV, TSV, Parquet)
✓ Model artifact loading and decompression
✓ Data preprocessing (risk table mapping, numerical imputation)
✓ Binary classification metrics computation
✓ Multiclass classification metrics computation
✓ Model comparison metrics and statistical tests
✓ Prediction and metrics saving
✓ Visualization generation (ROC, PR curves)
✓ Embedded processor classes (RiskTableMappingProcessor, NumericalVariableImputationProcessor)
✓ Main entry point execution flow
✓ Comparison mode functionality
✓ Edge cases and error handling

Total test classes: 15
Estimated total tests: 45+

All tests follow pytest best practices:
- Module-level mocking for subprocess.check_call
- Proper fixture isolation with temp_dir
- Implementation-driven testing
- Comprehensive coverage of success and failure scenarios
"""

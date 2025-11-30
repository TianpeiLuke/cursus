"""
Comprehensive pytest tests for active_sample_selection script.

This test suite provides extensive coverage for all components of the active sample
selection script including file I/O, data loading, sampling strategies, selection
engine, output management, and integration tests.

Test Coverage:
1. File format detection and loading (csv, tsv, parquet)
2. Score column extraction with various patterns
3. Score normalization to probabilities
4. All sampling strategies (SSL and Active Learning)
5. Selection engine coordination
6. Output format preservation
7. Strategy validation for use cases
8. Error handling and edge cases
9. Integration tests for main function

Author: Test Suite
Date: 2025-11-29
"""

import argparse
import json
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from cursus.steps.scripts.active_sample_selection import (
    BADGESampler,
    ConfidenceThresholdSampler,
    DiversitySampler,
    TopKPerClassSampler,
    UncertaintySampler,
    _detect_file_format,
    extract_score_columns,
    load_dataframe_with_format,
    load_inference_data,
    main,
    normalize_scores_to_probabilities,
    save_dataframe_with_format,
    save_selected_samples,
    save_selection_metadata,
    select_samples,
    validate_strategy_for_use_case,
)


# ============================================================================
# Fixtures for Test Data
# ============================================================================


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame with probability columns for testing."""
    np.random.seed(42)
    n_samples = 100
    n_classes = 3
    
    # Generate sample probabilities
    raw_probs = np.random.rand(n_samples, n_classes)
    probs = raw_probs / raw_probs.sum(axis=1, keepdims=True)
    
    df = pd.DataFrame({
        "id": [f"sample_{i}" for i in range(n_samples)],
    })
    
    # Add probability columns
    for i in range(n_classes):
        df[f"prob_class_{i}"] = probs[:, i]
    
    return df


@pytest.fixture
def sample_dataframe_with_embeddings():
    """Create a sample DataFrame with probability columns and embeddings."""
    np.random.seed(42)
    n_samples = 100
    n_classes = 3
    n_features = 10
    
    # Generate sample probabilities
    raw_probs = np.random.rand(n_samples, n_classes)
    probs = raw_probs / raw_probs.sum(axis=1, keepdims=True)
    
    df = pd.DataFrame({
        "id": [f"sample_{i}" for i in range(n_samples)],
    })
    
    # Add probability columns
    for i in range(n_classes):
        df[f"prob_class_{i}"] = probs[:, i]
    
    # Add embedding columns
    for i in range(n_features):
        df[f"emb_{i}"] = np.random.randn(n_samples)
    
    return df


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_csv_file(temp_data_dir, sample_dataframe):
    """Create a sample CSV file for testing."""
    csv_path = temp_data_dir / "data.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_tsv_file(temp_data_dir, sample_dataframe):
    """Create a sample TSV file for testing."""
    tsv_path = temp_data_dir / "data.tsv"
    sample_dataframe.to_csv(tsv_path, sep="\t", index=False)
    return tsv_path


@pytest.fixture
def sample_parquet_file(temp_data_dir, sample_dataframe):
    """Create a sample Parquet file for testing."""
    parquet_path = temp_data_dir / "data.parquet"
    sample_dataframe.to_parquet(parquet_path, index=False)
    return parquet_path


# ============================================================================
# Tests for File I/O Helper Functions
# ============================================================================


class TestFileFormatDetection:
    """Tests for file format detection functionality."""
    
    def test_detect_csv_format(self, sample_csv_file):
        """Test detection of CSV file format."""
        format_str = _detect_file_format(sample_csv_file)
        assert format_str == "csv"
    
    def test_detect_tsv_format(self, sample_tsv_file):
        """Test detection of TSV file format."""
        format_str = _detect_file_format(sample_tsv_file)
        assert format_str == "tsv"
    
    def test_detect_parquet_format(self, sample_parquet_file):
        """Test detection of Parquet file format."""
        format_str = _detect_file_format(sample_parquet_file)
        assert format_str == "parquet"
    
    def test_detect_unsupported_format(self, temp_data_dir):
        """Test error handling for unsupported file formats."""
        unsupported_file = temp_data_dir / "data.json"
        unsupported_file.write_text("{}")
        
        with pytest.raises(RuntimeError, match="Unsupported file format"):
            _detect_file_format(unsupported_file)
    
    def test_detect_format_with_string_path(self, sample_csv_file):
        """Test format detection with string path instead of Path object."""
        format_str = _detect_file_format(str(sample_csv_file))
        assert format_str == "csv"


class TestDataFrameLoading:
    """Tests for DataFrame loading with format detection."""
    
    def test_load_csv_dataframe(self, sample_csv_file):
        """Test loading CSV DataFrame with format detection."""
        df, format_str = load_dataframe_with_format(sample_csv_file)
        
        assert format_str == "csv"
        assert len(df) == 100
        assert "id" in df.columns
        assert "prob_class_0" in df.columns
    
    def test_load_tsv_dataframe(self, sample_tsv_file):
        """Test loading TSV DataFrame with format detection."""
        df, format_str = load_dataframe_with_format(sample_tsv_file)
        
        assert format_str == "tsv"
        assert len(df) == 100
        assert "id" in df.columns
    
    def test_load_parquet_dataframe(self, sample_parquet_file):
        """Test loading Parquet DataFrame with format detection."""
        df, format_str = load_dataframe_with_format(sample_parquet_file)
        
        assert format_str == "parquet"
        assert len(df) == 100
        assert "id" in df.columns
    
    def test_load_unsupported_format(self, temp_data_dir):
        """Test error handling when loading unsupported format."""
        unsupported_file = temp_data_dir / "data.json"
        unsupported_file.write_text("{}")
        
        with pytest.raises(RuntimeError, match="Unsupported"):
            load_dataframe_with_format(unsupported_file)


class TestDataFrameSaving:
    """Tests for DataFrame saving with format preservation."""
    
    def test_save_csv_dataframe(self, temp_data_dir, sample_dataframe):
        """Test saving DataFrame in CSV format."""
        output_path = temp_data_dir / "output"
        saved_path = save_dataframe_with_format(sample_dataframe, output_path, "csv")
        
        assert saved_path.exists()
        assert saved_path.suffix == ".csv"
        
        # Verify content
        loaded_df = pd.read_csv(saved_path)
        assert len(loaded_df) == len(sample_dataframe)
    
    def test_save_tsv_dataframe(self, temp_data_dir, sample_dataframe):
        """Test saving DataFrame in TSV format."""
        output_path = temp_data_dir / "output"
        saved_path = save_dataframe_with_format(sample_dataframe, output_path, "tsv")
        
        assert saved_path.exists()
        assert saved_path.suffix == ".tsv"
        
        # Verify content
        loaded_df = pd.read_csv(saved_path, sep="\t")
        assert len(loaded_df) == len(sample_dataframe)
    
    def test_save_parquet_dataframe(self, temp_data_dir, sample_dataframe):
        """Test saving DataFrame in Parquet format."""
        output_path = temp_data_dir / "output"
        saved_path = save_dataframe_with_format(sample_dataframe, output_path, "parquet")
        
        assert saved_path.exists()
        assert saved_path.suffix == ".parquet"
        
        # Verify content
        loaded_df = pd.read_parquet(saved_path)
        assert len(loaded_df) == len(sample_dataframe)
    
    def test_save_unsupported_format(self, temp_data_dir, sample_dataframe):
        """Test error handling when saving with unsupported format."""
        output_path = temp_data_dir / "output"
        
        with pytest.raises(RuntimeError, match="Unsupported output format"):
            save_dataframe_with_format(sample_dataframe, output_path, "json")


# ============================================================================
# Tests for Data Loading Component
# ============================================================================


class TestInferenceDataLoading:
    """Tests for loading inference data from various sources."""
    
    def test_load_inference_data_csv(self, sample_csv_file):
        """Test loading inference data from CSV file."""
        data_dir = sample_csv_file.parent
        df, format_str = load_inference_data(str(data_dir), id_field="id")
        
        assert format_str == "csv"
        assert len(df) == 100
        assert "id" in df.columns
    
    def test_load_inference_data_parquet(self, sample_parquet_file):
        """Test loading inference data from Parquet file."""
        data_dir = sample_parquet_file.parent
        df, format_str = load_inference_data(str(data_dir), id_field="id")
        
        assert format_str in ["csv", "tsv", "parquet"]
        assert len(df) > 0
        assert "id" in df.columns
    
    def test_load_inference_data_no_files(self, temp_data_dir):
        """Test error handling when no data files are found."""
        with pytest.raises(FileNotFoundError, match="No inference data files"):
            load_inference_data(str(temp_data_dir), id_field="id")
    
    def test_load_inference_data_missing_id_field(self, temp_data_dir):
        """Test error handling when ID field is missing."""
        # Create data without ID field
        df = pd.DataFrame({"value": [1, 2, 3]})
        csv_path = temp_data_dir / "data.csv"
        df.to_csv(csv_path, index=False)
        
        with pytest.raises(ValueError, match="ID field 'id' not found"):
            load_inference_data(str(temp_data_dir), id_field="id")


class TestScoreColumnExtraction:
    """Tests for extracting score columns from inference data."""
    
    def test_extract_explicit_score_field(self, sample_dataframe):
        """Test extraction when explicit score field is specified."""
        sample_dataframe["confidence_score"] = 0.9
        
        score_cols = extract_score_columns(sample_dataframe, score_field="confidence_score")
        assert score_cols == ["confidence_score"]
    
    def test_extract_with_prefix(self, sample_dataframe):
        """Test extraction using score field prefix."""
        score_cols = extract_score_columns(sample_dataframe, score_prefix="prob_class_")
        
        assert len(score_cols) == 3
        assert all(col.startswith("prob_class_") for col in score_cols)
    
    def test_extract_llm_format(self):
        """Test auto-detection of LLM/Bedrock format."""
        df = pd.DataFrame({
            "id": ["1", "2"],
            "confidence_score": [0.9, 0.8],
            "text": ["sample1", "sample2"]
        })
        
        score_cols = extract_score_columns(df)
        assert "confidence_score" in score_cols
    
    def test_extract_ruleset_format(self):
        """Test auto-detection of ruleset format."""
        df = pd.DataFrame({
            "id": ["1", "2"],
            "rule_score": [0.9, 0.8],
            "text": ["sample1", "sample2"]
        })
        
        score_cols = extract_score_columns(df)
        assert "rule_score" in score_cols
    
    def test_extract_no_valid_columns(self):
        """Test error handling when no valid score columns are found."""
        df = pd.DataFrame({
            "id": ["1", "2"],
            "text": ["sample1", "sample2"]
        })
        
        with pytest.raises(ValueError, match="No valid score columns found"):
            extract_score_columns(df)


class TestScoreNormalization:
    """Tests for normalizing various score formats to probabilities."""
    
    def test_normalize_already_normalized(self, sample_dataframe):
        """Test normalization of already normalized probabilities."""
        score_cols = [f"prob_class_{i}" for i in range(3)]
        df_norm = normalize_scores_to_probabilities(sample_dataframe, score_cols)
        
        # Should return same values since already normalized
        for col in score_cols:
            assert col in df_norm.columns
            np.testing.assert_array_almost_equal(
                sample_dataframe[col].values,
                df_norm[col].values,
                decimal=5
            )
    
    def test_normalize_raw_scores(self):
        """Test normalization of raw scores using softmax."""
        df = pd.DataFrame({
            "id": ["1", "2", "3"],
            "score_0": [2.0, 1.0, 3.0],
            "score_1": [1.0, 2.0, 1.0],
            "score_2": [0.5, 0.5, 2.0]
        })
        
        score_cols = ["score_0", "score_1", "score_2"]
        df_norm = normalize_scores_to_probabilities(df, score_cols)
        
        # Check that probabilities sum to 1
        prob_cols = [f"prob_class_{i}" for i in range(3)]
        row_sums = df_norm[prob_cols].sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(3), decimal=5)
    
    def test_normalize_single_score(self):
        """Test normalization with single score column."""
        df = pd.DataFrame({
            "id": ["1", "2", "3"],
            "confidence": [0.8, 0.9, 0.7]
        })
        
        df_norm = normalize_scores_to_probabilities(df, ["confidence"])
        
        # Single score should be normalized
        assert "prob_class_0" in df_norm.columns


# ============================================================================
# Tests for Sampling Strategies
# ============================================================================


class TestConfidenceThresholdSampler:
    """Tests for confidence threshold sampling strategy (SSL)."""
    
    def test_select_high_confidence_samples(self):
        """Test selection of high-confidence samples above threshold."""
        sampler = ConfidenceThresholdSampler(confidence_threshold=0.8)
        
        # Create probabilities with some high-confidence samples
        probabilities = np.array([
            [0.9, 0.05, 0.05],  # High confidence
            [0.5, 0.3, 0.2],    # Low confidence
            [0.85, 0.1, 0.05],  # High confidence
            [0.4, 0.4, 0.2],    # Low confidence
        ])
        
        selected_idx, scores = sampler.select_batch(probabilities)
        
        assert len(selected_idx) == 2
        assert 0 in selected_idx
        assert 2 in selected_idx
        assert all(score >= 0.8 for score in scores)
    
    def test_max_samples_limit(self):
        """Test that max_samples parameter limits selection count."""
        sampler = ConfidenceThresholdSampler(confidence_threshold=0.7, max_samples=2)
        
        # Create probabilities with many high-confidence samples
        probabilities = np.array([
            [0.9, 0.05, 0.05],
            [0.85, 0.1, 0.05],
            [0.95, 0.03, 0.02],
            [0.8, 0.15, 0.05],
        ])
        
        selected_idx, scores = sampler.select_batch(probabilities)
        
        assert len(selected_idx) == 2
        # Should select top 2 by confidence
        assert 2 in selected_idx  # Highest confidence (0.95)
        assert 0 in selected_idx  # Second highest (0.9)
    
    def test_no_samples_above_threshold(self):
        """Test behavior when no samples meet threshold."""
        sampler = ConfidenceThresholdSampler(confidence_threshold=0.95)
        
        probabilities = np.array([
            [0.5, 0.3, 0.2],
            [0.6, 0.3, 0.1],
            [0.7, 0.2, 0.1],
        ])
        
        selected_idx, scores = sampler.select_batch(probabilities)
        
        assert len(selected_idx) == 0
        assert len(scores) == 0
    
    def test_with_indices_parameter(self):
        """Test selection with custom indices array."""
        sampler = ConfidenceThresholdSampler(confidence_threshold=0.8)
        
        probabilities = np.array([
            [0.9, 0.05, 0.05],
            [0.5, 0.3, 0.2],
            [0.85, 0.1, 0.05],
        ])
        
        custom_indices = np.array([10, 20, 30])
        selected_idx, scores = sampler.select_batch(probabilities, custom_indices)
        
        assert 10 in selected_idx
        assert 30 in selected_idx
        assert 20 not in selected_idx


class TestTopKPerClassSampler:
    """Tests for top-k per class sampling strategy (SSL)."""
    
    def test_select_balanced_samples(self):
        """Test balanced selection across classes."""
        sampler = TopKPerClassSampler(k_per_class=2)
        
        # Create probabilities with different predicted classes
        probabilities = np.array([
            [0.9, 0.05, 0.05],  # Class 0
            [0.85, 0.1, 0.05],  # Class 0
            [0.05, 0.9, 0.05],  # Class 1
            [0.1, 0.85, 0.05],  # Class 1
            [0.05, 0.1, 0.85],  # Class 2
            [0.05, 0.05, 0.9],  # Class 2
        ])
        
        selected_idx, scores = sampler.select_batch(probabilities)
        
        # Should select 2 samples from each class (6 total)
        assert len(selected_idx) == 6
        assert len(scores) == 6
    
    def test_handle_imbalanced_classes(self):
        """Test handling when some classes have fewer samples than k."""
        sampler = TopKPerClassSampler(k_per_class=3)
        
        probabilities = np.array([
            [0.9, 0.05, 0.05],  # Class 0
            [0.85, 0.1, 0.05],  # Class 0
            [0.05, 0.9, 0.05],  # Class 1 (only 1 sample)
        ])
        
        selected_idx, scores = sampler.select_batch(probabilities)
        
        # Should select 2 from class 0 and 1 from class 1
        assert len(selected_idx) == 3
    
    def test_select_top_confidence_per_class(self):
        """Test that highest confidence samples are selected per class."""
        sampler = TopKPerClassSampler(k_per_class=1)
        
        probabilities = np.array([
            [0.8, 0.1, 0.1],   # Class 0, conf=0.8
            [0.95, 0.03, 0.02],  # Class 0, conf=0.95 (should be selected)
            [0.1, 0.7, 0.2],   # Class 1, conf=0.7
            [0.1, 0.9, 0.0],   # Class 1, conf=0.9 (should be selected)
        ])
        
        selected_idx, scores = sampler.select_batch(probabilities)
        
        assert len(selected_idx) == 2
        assert 1 in selected_idx  # Highest conf for class 0
        assert 3 in selected_idx  # Highest conf for class 1


class TestUncertaintySampler:
    """Tests for uncertainty sampling strategies (Active Learning)."""
    
    def test_margin_sampling(self):
        """Test margin-based uncertainty sampling."""
        sampler = UncertaintySampler(strategy="margin")
        
        probabilities = np.array([
            [0.9, 0.05, 0.05],  # Low uncertainty (high margin)
            [0.5, 0.49, 0.01],  # High uncertainty (low margin)
            [0.6, 0.3, 0.1],    # Medium uncertainty
        ])
        
        scores = sampler.compute_scores(probabilities)
        
        # Higher scores = more uncertain
        assert scores[1] > scores[2] > scores[0]
    
    def test_entropy_sampling(self):
        """Test entropy-based uncertainty sampling."""
        sampler = UncertaintySampler(strategy="entropy")
        
        probabilities = np.array([
            [0.9, 0.05, 0.05],   # Low entropy
            [0.33, 0.33, 0.34],  # High entropy
            [0.6, 0.3, 0.1],     # Medium entropy
        ])
        
        scores = sampler.compute_scores(probabilities)
        
        # Higher scores = more uncertain (higher entropy)
        assert scores[1] > scores[2] > scores[0]
    
    def test_least_confidence_sampling(self):
        """Test least confidence uncertainty sampling."""
        sampler = UncertaintySampler(strategy="least_confidence")
        
        probabilities = np.array([
            [0.9, 0.05, 0.05],  # High confidence
            [0.4, 0.35, 0.25],  # Low confidence
            [0.7, 0.2, 0.1],    # Medium confidence
        ])
        
        scores = sampler.compute_scores(probabilities)
        
        # Higher scores = less confident
        assert scores[1] > scores[2] > scores[0]
    
    def test_select_batch(self):
        """Test batch selection with uncertainty sampling."""
        sampler = UncertaintySampler(strategy="margin")
        
        probabilities = np.array([
            [0.9, 0.05, 0.05],
            [0.5, 0.49, 0.01],
            [0.6, 0.3, 0.1],
            [0.51, 0.48, 0.01],
        ])
        
        batch_size = 2
        selected_idx, scores = sampler.select_batch(probabilities, batch_size)
        
        assert len(selected_idx) == 2
        # Should select most uncertain samples (indices 1 and 3)
        assert 1 in selected_idx
        assert 3 in selected_idx
    
    def test_invalid_strategy(self):
        """Test error handling for invalid strategy."""
        sampler = UncertaintySampler(strategy="invalid")
        
        probabilities = np.array([[0.5, 0.3, 0.2]])
        
        with pytest.raises(ValueError, match="Unknown strategy"):
            sampler.compute_scores(probabilities)


class TestDiversitySampler:
    """Tests for diversity sampling (Active Learning)."""
    
    def test_select_diverse_samples(self):
        """Test selection of diverse samples using k-center algorithm."""
        sampler = DiversitySampler(metric="euclidean")
        
        # Create embeddings with clear clusters
        embeddings = np.array([
            [0.0, 0.0],    # Cluster 1
            [0.1, 0.1],    # Cluster 1
            [5.0, 5.0],    # Cluster 2
            [5.1, 5.1],    # Cluster 2
            [10.0, 10.0],  # Cluster 3
        ])
        
        batch_size = 3
        selected_idx, scores = sampler.select_batch(embeddings, batch_size)
        
        assert len(selected_idx) == 3
        # Should select samples from different clusters
        # (exact indices depend on random seed)
    
    def test_cosine_metric(self):
        """Test diversity sampling with cosine distance."""
        sampler = DiversitySampler(metric="cosine")
        
        embeddings = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])
        
        batch_size = 2
        selected_idx, scores = sampler.select_batch(embeddings, batch_size)
        
        assert len(selected_idx) == 2
    
    def test_batch_size_larger_than_data(self):
        """Test behavior when batch_size exceeds data size."""
        sampler = DiversitySampler()
        
        embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        batch_size = 10
        selected_idx, scores = sampler.select_batch(embeddings, batch_size)
        
        # Should select all available samples
        assert len(selected_idx) == 2
    
    def test_invalid_metric(self):
        """Test error handling for invalid distance metric."""
        sampler = DiversitySampler(metric="invalid")
        
        embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        with pytest.raises(ValueError, match="Unknown metric"):
            sampler.select_batch(embeddings, batch_size=2)


class TestBADGESampler:
    """Tests for BADGE sampling (Active Learning)."""
    
    def test_compute_gradient_embeddings(self):
        """Test computation of gradient embeddings."""
        sampler = BADGESampler()
        
        features = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ])
        
        probabilities = np.array([
            [0.7, 0.2, 0.1],
            [0.3, 0.6, 0.1],
        ])
        
        grad_embeddings = sampler.compute_gradient_embeddings(features, probabilities)
        
        # Gradient embeddings should have shape (n_samples, n_features * n_classes)
        assert grad_embeddings.shape == (2, 6)  # 2 features * 3 classes
    
    def test_select_batch(self):
        """Test BADGE batch selection."""
        sampler = BADGESampler()
        
        features = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ])
        
        probabilities = np.array([
            [0.7, 0.2, 0.1],
            [0.3, 0.6, 0.1],
            [0.5, 0.3, 0.2],
            [0.2, 0.2, 0.6],
        ])
        
        batch_size = 2
        selected_idx, scores = sampler.select_batch(features, probabilities, batch_size)
        
        assert len(selected_idx) == 2
        assert len(scores) == 2


# ============================================================================
# Tests for Selection Engine Component
# ============================================================================


class TestSelectSamples:
    """Tests for the main select_samples function."""
    
    def test_confidence_threshold_strategy(self, sample_dataframe):
        """Test selection using confidence threshold strategy."""
        strategy_config = {
            "confidence_threshold": 0.8,
            "max_samples": 10,
            "random_seed": 42
        }
        
        selected_df = select_samples(
            df=sample_dataframe,
            strategy="confidence_threshold",
            batch_size=32,
            strategy_config=strategy_config,
            id_field="id"
        )
        
        assert len(selected_df) > 0
        assert "selection_score" in selected_df.columns
        assert "selection_rank" in selected_df.columns
        assert all(selected_df["selection_score"] >= 0.8)
    
    def test_top_k_per_class_strategy(self, sample_dataframe):
        """Test selection using top-k per class strategy."""
        strategy_config = {
            "k_per_class": 5,
            "random_seed": 42
        }
        
        selected_df = select_samples(
            df=sample_dataframe,
            strategy="top_k_per_class",
            batch_size=32,
            strategy_config=strategy_config,
            id_field="id"
        )
        
        assert len(selected_df) > 0
        assert "selection_score" in selected_df.columns
        assert "selection_rank" in selected_df.columns
    
    def test_uncertainty_strategy(self, sample_dataframe):
        """Test selection using uncertainty strategy."""
        strategy_config = {
            "uncertainty_mode": "margin",
            "random_seed": 42
        }
        
        batch_size = 10
        selected_df = select_samples(
            df=sample_dataframe,
            strategy="uncertainty",
            batch_size=batch_size,
            strategy_config=strategy_config,
            id_field="id"
        )
        
        assert len(selected_df) == batch_size
        assert "selection_score" in selected_df.columns
    
    def test_diversity_strategy(self, sample_dataframe_with_embeddings):
        """Test selection using diversity strategy."""
        strategy_config = {
            "metric": "euclidean",
            "random_seed": 42
        }
        
        batch_size = 10
        selected_df = select_samples(
            df=sample_dataframe_with_embeddings,
            strategy="diversity",
            batch_size=batch_size,
            strategy_config=strategy_config,
            id_field="id"
        )
        
        assert len(selected_df) == batch_size
        assert "selection_score" in selected_df.columns
    
    def test_badge_strategy(self, sample_dataframe_with_embeddings):
        """Test selection using BADGE strategy."""
        strategy_config = {
            "metric": "euclidean",
            "random_seed": 42
        }
        
        batch_size = 10
        selected_df = select_samples(
            df=sample_dataframe_with_embeddings,
            strategy="badge",
            batch_size=batch_size,
            strategy_config=strategy_config,
            id_field="id"
        )
        
        assert len(selected_df) == batch_size
        assert "selection_score" in selected_df.columns
    
    def test_unknown_strategy(self, sample_dataframe):
        """Test error handling for unknown strategy."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            select_samples(
                df=sample_dataframe,
                strategy="invalid_strategy",
                batch_size=10,
                strategy_config={},
                id_field="id"
            )
    
    def test_no_prob_columns(self):
        """Test error handling when no probability columns exist."""
        df = pd.DataFrame({
            "id": ["1", "2", "3"],
            "text": ["sample1", "sample2", "sample3"]
        })
        
        with pytest.raises(ValueError, match="No prob_class_"):
            select_samples(
                df=df,
                strategy="confidence_threshold",
                batch_size=10,
                strategy_config={},
                id_field="id"
            )


# ============================================================================
# Tests for Output Management Component
# ============================================================================


class TestOutputManagement:
    """Tests for output saving and metadata management."""
    
    def test_save_selected_samples_csv(self, temp_data_dir, sample_dataframe):
        """Test saving selected samples in CSV format."""
        output_path = save_selected_samples(
            selected_df=sample_dataframe,
            output_dir=str(temp_data_dir),
            output_format="csv"
        )
        
        assert Path(output_path).exists()
        assert Path(output_path).suffix == ".csv"
        
        # Verify content
        loaded_df = pd.read_csv(output_path)
        assert len(loaded_df) == len(sample_dataframe)
    
    def test_save_selected_samples_parquet(self, temp_data_dir, sample_dataframe):
        """Test saving selected samples in Parquet format."""
        output_path = save_selected_samples(
            selected_df=sample_dataframe,
            output_dir=str(temp_data_dir),
            output_format="parquet"
        )
        
        assert Path(output_path).exists()
        assert Path(output_path).suffix == ".parquet"
    
    def test_save_selection_metadata(self, temp_data_dir):
        """Test saving selection metadata."""
        metadata = {
            "strategy": "confidence_threshold",
            "use_case": "ssl",
            "batch_size": 32,
            "selected_count": 50,
            "timestamp": "2025-11-29T12:00:00"
        }
        
        metadata_path = save_selection_metadata(
            metadata=metadata,
            metadata_dir=str(temp_data_dir)
        )
        
        assert Path(metadata_path).exists()
        
        # Verify content
        with open(metadata_path, "r") as f:
            loaded_metadata = json.load(f)
        
        assert loaded_metadata["strategy"] == "confidence_threshold"
        assert loaded_metadata["selected_count"] == 50


# ============================================================================
# Tests for Use Case Validation
# ============================================================================


class TestUseCaseValidation:
    """Tests for strategy validation based on use case."""
    
    def test_ssl_with_confidence_strategy(self):
        """Test SSL use case with confidence-based strategy."""
        # Should not raise error
        validate_strategy_for_use_case("confidence_threshold", "ssl")
        validate_strategy_for_use_case("top_k_per_class", "ssl")
    
    def test_ssl_with_uncertainty_strategy(self):
        """Test SSL use case with uncertainty strategy (should fail)."""
        with pytest.raises(ValueError, match="NOT valid for SSL"):
            validate_strategy_for_use_case("uncertainty", "ssl")
    
    def test_active_learning_with_uncertainty_strategy(self):
        """Test Active Learning use case with uncertainty strategy."""
        # Should not raise error
        validate_strategy_for_use_case("uncertainty", "active_learning")
        validate_strategy_for_use_case("diversity", "active_learning")
        validate_strategy_for_use_case("badge", "active_learning")
    
    def test_active_learning_with_confidence_strategy(self):
        """Test Active Learning with confidence strategy (should warn)."""
        with pytest.raises(ValueError, match="NOT recommended for Active Learning"):
            validate_strategy_for_use_case("confidence_threshold", "active_learning")
    
    def test_auto_use_case(self):
        """Test auto use case (no validation)."""
        # Should not raise error for any strategy
        validate_strategy_for_use_case("confidence_threshold", "auto")
        validate_strategy_for_use_case("uncertainty", "auto")
        validate_strategy_for_use_case("diversity", "auto")


# ============================================================================
# Tests for Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame({
            "id": [],
            "prob_class_0": [],
            "prob_class_1": [],
            "prob_class_2": []
        })
        
        strategy_config = {"confidence_threshold": 0.8, "random_seed": 42}
        
        selected_df = select_samples(
            df=df,
            strategy="confidence_threshold",
            batch_size=10,
            strategy_config=strategy_config,
            id_field="id"
        )
        
        assert len(selected_df) == 0
    
    def test_single_sample(self):
        """Test selection with single sample."""
        df = pd.DataFrame({
            "id": ["sample_0"],
            "prob_class_0": [0.9],
            "prob_class_1": [0.05],
            "prob_class_2": [0.05]
        })
        
        strategy_config = {"confidence_threshold": 0.8, "random_seed": 42}
        
        selected_df = select_samples(
            df=df,
            strategy="confidence_threshold",
            batch_size=10,
            strategy_config=strategy_config,
            id_field="id"
        )
        
        assert len(selected_df) == 1
    
    def test_batch_size_larger_than_data(self, sample_dataframe):
        """Test when batch size exceeds available data."""
        batch_size = 1000  # Much larger than 100 samples
        
        strategy_config = {"uncertainty_mode": "margin", "random_seed": 42}
        
        selected_df = select_samples(
            df=sample_dataframe,
            strategy="uncertainty",
            batch_size=batch_size,
            strategy_config=strategy_config,
            id_field="id"
        )
        
        # Should select all available samples
        assert len(selected_df) == len(sample_dataframe)
    
    def test_diversity_without_embeddings(self, sample_dataframe):
        """Test diversity sampling when no embeddings are available."""
        # Add some numeric feature columns
        sample_dataframe["feature_1"] = np.random.randn(len(sample_dataframe))
        sample_dataframe["feature_2"] = np.random.randn(len(sample_dataframe))
        
        strategy_config = {
            "metric": "euclidean",
            "random_seed": 42,
            "feature_columns": ["feature_1", "feature_2"]
        }
        
        selected_df = select_samples(
            df=sample_dataframe,
            strategy="diversity",
            batch_size=10,
            strategy_config=strategy_config,
            id_field="id"
        )
        
        assert len(selected_df) == 10


# ============================================================================
# Integration Tests for Main Function
# ============================================================================


class TestMainFunction:
    """Integration tests for the main function."""
    
    def test_main_confidence_threshold_strategy(self, temp_data_dir, sample_csv_file):
        """Test main function with confidence threshold strategy."""
        # Set up input/output paths
        input_paths = {
            "evaluation_data": str(sample_csv_file.parent)
        }
        
        output_dir = temp_data_dir / "output"
        output_dir.mkdir()
        
        output_paths = {
            "selected_samples": str(output_dir / "selected"),
            "selection_metadata": str(output_dir / "metadata")
        }
        
        # Configure environment variables
        environ_vars = {
            "ID_FIELD": "id",
            "SELECTION_STRATEGY": "confidence_threshold",
            "USE_CASE": "ssl",
            "BATCH_SIZE": "32",
            "OUTPUT_FORMAT": "csv",
            "CONFIDENCE_THRESHOLD": "0.7",
            "K_PER_CLASS": "100",
            "MAX_SAMPLES": "0",
            "UNCERTAINTY_MODE": "margin",
            "METRIC": "euclidean",
            "RANDOM_SEED": "42",
            "SCORE_FIELD": None,
            "SCORE_FIELD_PREFIX": "prob_class_"
        }
        
        # Create job arguments
        args = argparse.Namespace(job_type="ssl_selection")
        
        # Run main function
        main(input_paths, output_paths, environ_vars, args)
        
        # Verify outputs
        selected_samples_dir = Path(output_paths["selected_samples"])
        assert selected_samples_dir.exists()
        
        # Check for output file
        output_files = list(selected_samples_dir.glob("selected_samples.*"))
        assert len(output_files) > 0
        
        # Check metadata
        metadata_dir = Path(output_paths["selection_metadata"])
        metadata_file = metadata_dir / "selection_metadata.json"
        assert metadata_file.exists()
        
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        
        assert metadata["strategy"] == "confidence_threshold"
        assert metadata["use_case"] == "ssl"
    
    def test_main_uncertainty_strategy(self, temp_data_dir, sample_csv_file):
        """Test main function with uncertainty strategy."""
        input_paths = {
            "evaluation_data": str(sample_csv_file.parent)
        }
        
        output_dir = temp_data_dir / "output"
        output_dir.mkdir()
        
        output_paths = {
            "selected_samples": str(output_dir / "selected"),
            "selection_metadata": str(output_dir / "metadata")
        }
        
        environ_vars = {
            "ID_FIELD": "id",
            "SELECTION_STRATEGY": "uncertainty",
            "USE_CASE": "active_learning",
            "BATCH_SIZE": "20",
            "OUTPUT_FORMAT": "csv",
            "CONFIDENCE_THRESHOLD": "0.9",
            "K_PER_CLASS": "100",
            "MAX_SAMPLES": "0",
            "UNCERTAINTY_MODE": "entropy",
            "METRIC": "euclidean",
            "RANDOM_SEED": "42",
            "SCORE_FIELD": None,
            "SCORE_FIELD_PREFIX": "prob_class_"
        }
        
        args = argparse.Namespace(job_type="active_learning_selection")
        
        # Run main function
        main(input_paths, output_paths, environ_vars, args)
        
        # Verify outputs
        output_files = list(Path(output_paths["selected_samples"]).glob("selected_samples.*"))
        assert len(output_files) > 0
        
        # Load and verify selected samples
        if output_files[0].suffix == ".csv":
            selected_df = pd.read_csv(output_files[0])
        elif output_files[0].suffix == ".parquet":
            selected_df = pd.read_parquet(output_files[0])
        
        assert len(selected_df) == 20
        assert "selection_score" in selected_df.columns
        assert "selection_rank" in selected_df.columns
    
    def test_main_format_preservation(self, temp_data_dir, sample_parquet_file):
        """Test that main function preserves input format."""
        input_paths = {
            "evaluation_data": str(sample_parquet_file.parent)
        }
        
        output_dir = temp_data_dir / "output"
        output_dir.mkdir()
        
        output_paths = {
            "selected_samples": str(output_dir / "selected"),
            "selection_metadata": str(output_dir / "metadata")
        }
        
        # Use default csv format (should preserve parquet)
        environ_vars = {
            "ID_FIELD": "id",
            "SELECTION_STRATEGY": "confidence_threshold",
            "USE_CASE": "auto",
            "BATCH_SIZE": "32",
            "OUTPUT_FORMAT": "csv",  # Default value
            "CONFIDENCE_THRESHOLD": "0.7",
            "K_PER_CLASS": "100",
            "MAX_SAMPLES": "0",
            "UNCERTAINTY_MODE": "margin",
            "METRIC": "euclidean",
            "RANDOM_SEED": "42",
            "SCORE_FIELD": None,
            "SCORE_FIELD_PREFIX": "prob_class_"
        }
        
        args = argparse.Namespace(job_type="ssl_selection")
        
        # Run main function
        main(input_paths, output_paths, environ_vars, args)
        
        # Check output format matches input (parquet)
        output_files = list(Path(output_paths["selected_samples"]).glob("selected_samples.*"))
        assert len(output_files) > 0
        # Should preserve parquet format
        assert any(f.suffix == ".parquet" for f in output_files)


# ============================================================================
# Performance and Scalability Tests
# ============================================================================


class TestPerformanceAndScalability:
    """Tests for performance with larger datasets."""
    
    def test_large_dataset_selection(self):
        """Test selection on larger dataset."""
        np.random.seed(42)
        n_samples = 10000
        n_classes = 5
        
        # Generate large dataset
        raw_probs = np.random.rand(n_samples, n_classes)
        probs = raw_probs / raw_probs.sum(axis=1, keepdims=True)
        
        df = pd.DataFrame({
            "id": [f"sample_{i}" for i in range(n_samples)]
        })
        
        for i in range(n_classes):
            df[f"prob_class_{i}"] = probs[:, i]
        
        # Test uncertainty strategy (more reliable for random data)
        strategy_config = {"uncertainty_mode": "margin", "random_seed": 42}
        
        selected_df = select_samples(
            df=df,
            strategy="uncertainty",
            batch_size=100,
            strategy_config=strategy_config,
            id_field="id"
        )
        
        assert len(selected_df) == 100
        assert "selection_score" in selected_df.columns
    
    def test_high_dimensional_embeddings(self):
        """Test diversity sampling with high-dimensional embeddings."""
        np.random.seed(42)
        n_samples = 1000
        n_dims = 128
        n_classes = 3
        
        # Generate high-dimensional data
        raw_probs = np.random.rand(n_samples, n_classes)
        probs = raw_probs / raw_probs.sum(axis=1, keepdims=True)
        
        df = pd.DataFrame({
            "id": [f"sample_{i}" for i in range(n_samples)]
        })
        
        for i in range(n_classes):
            df[f"prob_class_{i}"] = probs[:, i]
        
        for i in range(n_dims):
            df[f"emb_{i}"] = np.random.randn(n_samples)
        
        # Test diversity strategy
        strategy_config = {"metric": "euclidean", "random_seed": 42}
        
        selected_df = select_samples(
            df=df,
            strategy="diversity",
            batch_size=50,
            strategy_config=strategy_config,
            id_field="id"
        )
        
        assert len(selected_df) == 50

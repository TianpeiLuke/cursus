#!/usr/bin/env python3
"""
Comprehensive tests for feature_selection.py script.

This test suite follows pytest best practices:
1. Mock subprocess.check_call BEFORE importing module (prevents actual pip installs)
2. Read source implementation to understand actual behavior
3. Test both training and inference modes
4. Test format preservation (CSV, TSV, Parquet)
5. Test parameter accumulator pattern (artifact copying)
6. Comprehensive coverage of all feature selection methods
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import json
import pickle as pkl
from pathlib import Path
import argparse
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# ============================================================================
# CRITICAL: Mock subprocess BEFORE importing module
# ============================================================================
# This prevents actual pip install from running during test collection
# Feature selection script installs xgboost at module level (line 130)
with patch("subprocess.check_call"):
    from cursus.steps.scripts.feature_selection import (
        main,
        load_preprocessed_data,
        load_single_split_data,
        save_selected_data,
        copy_existing_artifacts,
        load_selected_features,
        save_selection_results,
        variance_threshold_selection,
        correlation_based_selection,
        mutual_info_selection,
        chi2_selection,
        f_classif_selection,
        rfe_selection,
        feature_importance_selection,
        lasso_selection,
        permutation_importance_selection,
        combine_selection_results,
        apply_feature_selection_pipeline,
        _detect_file_format,
    )


class TestDetectFileFormat:
    """Tests for _detect_file_format function."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_detect_csv_format(self, temp_dir):
        """Test detecting CSV format."""
        split_dir = temp_dir / "train"
        split_dir.mkdir()

        # Create CSV file
        data = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        csv_file = split_dir / "train_processed_data.csv"
        data.to_csv(csv_file, index=False)

        file_path, fmt = _detect_file_format(split_dir, "train")

        assert file_path == csv_file
        assert fmt == "csv"

    def test_detect_tsv_format(self, temp_dir):
        """Test detecting TSV format."""
        split_dir = temp_dir / "train"
        split_dir.mkdir()

        # Create TSV file
        data = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        tsv_file = split_dir / "train_processed_data.tsv"
        data.to_csv(tsv_file, sep="\t", index=False)

        file_path, fmt = _detect_file_format(split_dir, "train")

        assert file_path == tsv_file
        assert fmt == "tsv"

    def test_detect_parquet_format(self, temp_dir):
        """Test detecting Parquet format."""
        split_dir = temp_dir / "train"
        split_dir.mkdir()

        # Create Parquet file
        data = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        parquet_file = split_dir / "train_processed_data.parquet"
        data.to_parquet(parquet_file, index=False)

        file_path, fmt = _detect_file_format(split_dir, "train")

        assert file_path == parquet_file
        assert fmt == "parquet"

    def test_detect_format_preference_order(self, temp_dir):
        """Test format detection prefers CSV > TSV > Parquet."""
        split_dir = temp_dir / "train"
        split_dir.mkdir()

        data = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

        # Create all three formats
        csv_file = split_dir / "train_processed_data.csv"
        tsv_file = split_dir / "train_processed_data.tsv"
        parquet_file = split_dir / "train_processed_data.parquet"

        data.to_csv(csv_file, index=False)
        data.to_csv(tsv_file, sep="\t", index=False)
        data.to_parquet(parquet_file, index=False)

        file_path, fmt = _detect_file_format(split_dir, "train")

        # Should prefer CSV
        assert file_path == csv_file
        assert fmt == "csv"

    def test_detect_format_file_not_found(self, temp_dir):
        """Test error when no format file is found."""
        split_dir = temp_dir / "train"
        split_dir.mkdir()

        with pytest.raises(RuntimeError, match="No processed data file found"):
            _detect_file_format(split_dir, "train")


class TestLoadPreprocessedData:
    """Tests for load_preprocessed_data function."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def setup_data(self, temp_dir, file_format="csv"):
        """Helper to set up training data structure."""
        input_dir = temp_dir / "input"

        # Create train, test, val splits with features for selection
        for split in ["train", "test", "val"]:
            split_dir = input_dir / split
            split_dir.mkdir(parents=True)

            # Create sample data with multiple features
            data = pd.DataFrame(
                {
                    "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                    "feature2": [2.0, 4.0, 6.0, 8.0, 10.0],
                    "feature3": [0.1, 0.2, 0.3, 0.4, 0.5],
                    "feature4": [5.0, 4.0, 3.0, 2.0, 1.0],
                    "target": [0, 1, 0, 1, 0],
                }
            )

            if file_format == "csv":
                data_file = split_dir / f"{split}_processed_data.csv"
                data.to_csv(data_file, index=False)
            elif file_format == "tsv":
                data_file = split_dir / f"{split}_processed_data.tsv"
                data.to_csv(data_file, sep="\t", index=False)
            elif file_format == "parquet":
                data_file = split_dir / f"{split}_processed_data.parquet"
                data.to_parquet(data_file, index=False)

        return input_dir

    def test_load_preprocessed_data_csv(self, temp_dir):
        """Test loading CSV format data."""
        input_dir = self.setup_data(temp_dir, file_format="csv")

        result = load_preprocessed_data(str(input_dir))

        # Should return all three splits plus format metadata
        assert isinstance(result, dict)
        assert "train" in result
        assert "test" in result
        assert "val" in result
        assert "_format" in result
        assert result["_format"] == "csv"

        # Check data structure
        for split_name, df in result.items():
            if split_name == "_format":
                continue
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 5
            assert "feature1" in df.columns
            assert "target" in df.columns

    def test_load_preprocessed_data_tsv(self, temp_dir):
        """Test loading TSV format data."""
        input_dir = self.setup_data(temp_dir, file_format="tsv")

        result = load_preprocessed_data(str(input_dir))

        assert result["_format"] == "tsv"
        assert isinstance(result["train"], pd.DataFrame)

    def test_load_preprocessed_data_parquet(self, temp_dir):
        """Test loading Parquet format data."""
        input_dir = self.setup_data(temp_dir, file_format="parquet")

        result = load_preprocessed_data(str(input_dir))

        assert result["_format"] == "parquet"
        assert isinstance(result["train"], pd.DataFrame)

    def test_load_preprocessed_data_missing_split(self, temp_dir):
        """Test loading data when a split is missing."""
        input_dir = temp_dir / "input"

        # Only create train split
        train_dir = input_dir / "train"
        train_dir.mkdir(parents=True)

        data = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "target": [0, 1, 0]}
        )
        csv_file = train_dir / "train_processed_data.csv"
        data.to_csv(csv_file, index=False)

        result = load_preprocessed_data(str(input_dir))

        # Should only have train split (and format)
        assert "train" in result
        assert "test" not in result
        assert "val" not in result

    def test_load_preprocessed_data_no_splits(self, temp_dir):
        """Test loading data when no splits exist."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="No valid data splits found"):
            load_preprocessed_data(str(input_dir))


class TestLoadSingleSplitData:
    """Tests for load_single_split_data function."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_load_single_split_validation(self, temp_dir):
        """Test loading validation split."""
        input_dir = temp_dir / "input"
        val_dir = input_dir / "validation"
        val_dir.mkdir(parents=True)

        # Create validation data
        data = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [4.0, 5.0, 6.0],
                "target": [0, 1, 0],
            }
        )

        data_file = val_dir / "validation_processed_data.csv"
        data.to_csv(data_file, index=False)

        result = load_single_split_data(str(input_dir), "validation")

        # Should return validation split plus format metadata
        assert isinstance(result, dict)
        assert "validation" in result
        assert "_format" in result
        assert result["_format"] == "csv"

        val_df = result["validation"]
        assert isinstance(val_df, pd.DataFrame)
        assert len(val_df) == 3

    def test_load_single_split_testing(self, temp_dir):
        """Test loading testing split."""
        input_dir = temp_dir / "input"
        test_dir = input_dir / "testing"
        test_dir.mkdir(parents=True)

        data = pd.DataFrame({"feature1": [1, 2], "target": [0, 1]})
        data_file = test_dir / "testing_processed_data.csv"
        data.to_csv(data_file, index=False)

        result = load_single_split_data(str(input_dir), "testing")

        assert "testing" in result
        assert len(result["testing"]) == 2

    def test_load_single_split_not_found(self, temp_dir):
        """Test loading split that doesn't exist."""
        input_dir = temp_dir / "input"
        nonexistent_dir = input_dir / "nonexistent"

        with pytest.raises(FileNotFoundError, match="Split directory not found"):
            load_single_split_data(str(input_dir), "nonexistent")


class TestSaveSelectedData:
    """Tests for save_selected_data function."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_save_selected_data_csv(self, temp_dir):
        """Test saving selected data in CSV format."""
        output_dir = temp_dir / "output"

        # Create test data with format metadata
        data_dict = {
            "train": pd.DataFrame(
                {
                    "feature1": [1, 2, 3],
                    "feature2": [4, 5, 6],
                    "feature3": [7, 8, 9],
                    "target": [0, 1, 0],
                }
            ),
            "_format": "csv",
        }

        selected_features = ["feature1", "feature3"]  # Select subset of features

        save_selected_data(data_dict, selected_features, "target", str(output_dir))

        # Check that file was created
        expected_file = output_dir / "train" / "train_processed_data.csv"
        assert expected_file.exists()

        # Check that only selected features + target were saved
        saved_data = pd.read_csv(expected_file)
        assert list(saved_data.columns) == ["feature1", "feature3", "target"]

    def test_save_selected_data_tsv(self, temp_dir):
        """Test saving selected data in TSV format."""
        output_dir = temp_dir / "output"

        data_dict = {
            "train": pd.DataFrame(
                {"feature1": [1, 2], "feature2": [3, 4], "target": [0, 1]}
            ),
            "_format": "tsv",
        }

        save_selected_data(data_dict, ["feature1"], "target", str(output_dir))

        expected_file = output_dir / "train" / "train_processed_data.tsv"
        assert expected_file.exists()

        # Verify TSV format
        saved_data = pd.read_csv(expected_file, sep="\t")
        assert list(saved_data.columns) == ["feature1", "target"]

    def test_save_selected_data_parquet(self, temp_dir):
        """Test saving selected data in Parquet format."""
        output_dir = temp_dir / "output"

        data_dict = {
            "train": pd.DataFrame(
                {"feature1": [1, 2], "feature2": [3, 4], "target": [0, 1]}
            ),
            "_format": "parquet",
        }

        save_selected_data(data_dict, ["feature2"], "target", str(output_dir))

        expected_file = output_dir / "train" / "train_processed_data.parquet"
        assert expected_file.exists()

        # Verify Parquet format
        saved_data = pd.read_parquet(expected_file)
        assert list(saved_data.columns) == ["feature2", "target"]

    def test_save_selected_data_multiple_splits(self, temp_dir):
        """Test saving multiple splits."""
        output_dir = temp_dir / "output"

        data_dict = {
            "train": pd.DataFrame({"f1": [1], "f2": [2], "target": [0]}),
            "test": pd.DataFrame({"f1": [3], "f2": [4], "target": [1]}),
            "val": pd.DataFrame({"f1": [5], "f2": [6], "target": [0]}),
            "_format": "csv",
        }

        save_selected_data(data_dict, ["f1"], "target", str(output_dir))

        # Check all splits were saved
        for split in ["train", "test", "val"]:
            expected_file = output_dir / split / f"{split}_processed_data.csv"
            assert expected_file.exists()

    def test_save_selected_data_missing_column(self, temp_dir):
        """Test error when selected column is missing."""
        output_dir = temp_dir / "output"

        data_dict = {
            "train": pd.DataFrame({"feature1": [1, 2], "target": [0, 1]}),
            "_format": "csv",
        }

        # Try to select non-existent feature
        with pytest.raises(ValueError, match="Missing columns"):
            save_selected_data(
                data_dict, ["feature1", "nonexistent"], "target", str(output_dir)
            )


class TestCopyExistingArtifacts:
    """Tests for copy_existing_artifacts function."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_copy_existing_artifacts_success(self, temp_dir):
        """Test copying existing artifacts successfully."""
        src_dir = temp_dir / "src"
        dst_dir = temp_dir / "dst"
        src_dir.mkdir()

        # Create some artifact files
        (src_dir / "artifact1.pkl").write_text("artifact1")
        (src_dir / "artifact2.json").write_text('{"key": "value"}')

        copy_existing_artifacts(str(src_dir), str(dst_dir))

        # Check that files were copied
        assert (dst_dir / "artifact1.pkl").exists()
        assert (dst_dir / "artifact2.json").exists()

    def test_copy_existing_artifacts_empty_source(self, temp_dir):
        """Test copying when source directory is empty."""
        src_dir = temp_dir / "src"
        dst_dir = temp_dir / "dst"
        src_dir.mkdir()

        copy_existing_artifacts(str(src_dir), str(dst_dir))

        # Destination should be created but empty
        assert dst_dir.exists()
        assert len(list(dst_dir.iterdir())) == 0

    def test_copy_existing_artifacts_source_not_exists(self, temp_dir):
        """Test copying when source directory doesn't exist."""
        src_dir = temp_dir / "nonexistent"
        dst_dir = temp_dir / "dst"

        # Should not raise error, just log and return
        copy_existing_artifacts(str(src_dir), str(dst_dir))

        # Destination should not be created
        assert not dst_dir.exists()

    def test_copy_existing_artifacts_none_source(self, temp_dir):
        """Test copying when source is None."""
        dst_dir = temp_dir / "dst"

        # Should not raise error
        copy_existing_artifacts(None, str(dst_dir))

        # Destination should not be created
        assert not dst_dir.exists()


class TestLoadSelectedFeatures:
    """Tests for load_selected_features function."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_load_selected_features_success(self, temp_dir):
        """Test loading selected features successfully."""
        artifacts_dir = temp_dir / "artifacts"
        artifacts_dir.mkdir()

        # Create selected_features.json file
        features_data = {"selected_features": ["feature1", "feature2", "feature3"]}
        features_file = artifacts_dir / "selected_features.json"

        with open(features_file, "w") as f:
            json.dump(features_data, f)

        result = load_selected_features(str(artifacts_dir))

        assert isinstance(result, list)
        assert len(result) == 3
        assert "feature1" in result
        assert "feature2" in result
        assert "feature3" in result

    def test_load_selected_features_file_not_found(self, temp_dir):
        """Test loading when file doesn't exist."""
        artifacts_dir = temp_dir / "artifacts"
        artifacts_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="Selected features file not found"):
            load_selected_features(str(artifacts_dir))

    def test_load_selected_features_empty_list(self, temp_dir):
        """Test loading empty features list."""
        artifacts_dir = temp_dir / "artifacts"
        artifacts_dir.mkdir()

        features_data = {"selected_features": []}
        features_file = artifacts_dir / "selected_features.json"

        with open(features_file, "w") as f:
            json.dump(features_data, f)

        result = load_selected_features(str(artifacts_dir))

        assert isinstance(result, list)
        assert len(result) == 0


class TestFeatureSelectionMethods:
    """Tests for individual feature selection methods."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing feature selection methods."""
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100) * 2,
                "feature3": np.random.randn(100) * 0.1,  # Low variance
                "feature4": np.random.randn(100),
                "feature5": np.random.randn(100),
            }
        )
        y = pd.Series(np.random.randint(0, 2, 100))
        return X, y

    def test_variance_threshold_selection(self, sample_data):
        """Test variance threshold feature selection."""
        X, y = sample_data

        result = variance_threshold_selection(X, threshold=0.05)

        assert "method" in result
        assert result["method"] == "variance_threshold"
        assert "selected_features" in result
        assert "scores" in result
        assert "n_selected" in result

        # Low variance feature (feature3) should be excluded
        assert "feature3" not in result["selected_features"]
        assert len(result["selected_features"]) < len(X.columns)

    def test_correlation_based_selection(self, sample_data):
        """Test correlation-based feature selection."""
        X, y = sample_data

        result = correlation_based_selection(X, y, threshold=0.9)

        assert result["method"] == "correlation_based"
        assert "selected_features" in result
        assert "removed_features" in result
        assert isinstance(result["selected_features"], list)

    def test_mutual_info_selection(self, sample_data):
        """Test mutual information feature selection."""
        X, y = sample_data

        result = mutual_info_selection(X, y, k=3, random_state=42)

        assert result["method"] == "mutual_information"
        assert "selected_features" in result
        assert result["n_selected"] <= 3
        assert "is_classification" in result

    def test_chi2_selection(self, sample_data):
        """Test chi-square feature selection."""
        X, y = sample_data

        result = chi2_selection(X, y, k=3)

        assert result["method"] == "chi2"
        assert "selected_features" in result
        # Chi2 requires non-negative values, so check if handled
        assert result["n_selected"] >= 0

    def test_f_classif_selection(self, sample_data):
        """Test F-test feature selection."""
        X, y = sample_data

        result = f_classif_selection(X, y, k=3)

        assert result["method"] == "f_test"
        assert "selected_features" in result
        assert result["n_selected"] <= 3

    def test_rfe_selection(self, sample_data):
        """Test RFE feature selection."""
        X, y = sample_data

        result = rfe_selection(X, y, estimator_type="rf", n_features=3, random_state=42)

        assert result["method"] == "rfe_rf"
        assert "selected_features" in result
        assert "rankings" in result
        assert result["n_selected"] <= 3

    def test_feature_importance_selection(self, sample_data):
        """Test feature importance-based selection."""
        X, y = sample_data

        result = feature_importance_selection(
            X, y, method="random_forest", n_features=3, random_state=42
        )

        assert result["method"] == "importance_random_forest"
        assert "selected_features" in result
        assert result["n_selected"] <= 3

    def test_lasso_selection(self, sample_data):
        """Test LASSO feature selection."""
        X, y = sample_data

        result = lasso_selection(X, y, alpha=0.01, random_state=42)

        assert result["method"] == "lasso"
        assert "selected_features" in result
        assert isinstance(result["selected_features"], list)

    def test_permutation_importance_selection(self, sample_data):
        """Test permutation importance feature selection."""
        X, y = sample_data

        result = permutation_importance_selection(
            X, y, estimator_type="rf", n_features=3, random_state=42
        )

        assert result["method"] == "permutation_rf"
        assert "selected_features" in result
        assert result["n_selected"] <= 3


class TestCombineSelectionResults:
    """Tests for combine_selection_results function."""

    @pytest.fixture
    def sample_method_results(self):
        """Create sample method results for testing."""
        return [
            {
                "method": "method1",
                "selected_features": ["feature1", "feature2", "feature3"],
                "scores": {"feature1": 0.9, "feature2": 0.8, "feature3": 0.7},
                "n_selected": 3,
            },
            {
                "method": "method2",
                "selected_features": ["feature1", "feature3", "feature4"],
                "scores": {"feature1": 0.95, "feature3": 0.85, "feature4": 0.75},
                "n_selected": 3,
            },
            {
                "method": "method3",
                "selected_features": ["feature2", "feature3", "feature5"],
                "scores": {"feature2": 0.88, "feature3": 0.82, "feature5": 0.72},
                "n_selected": 3,
            },
        ]

    def test_combine_selection_voting(self, sample_method_results):
        """Test voting combination strategy."""
        result = combine_selection_results(
            sample_method_results, combination_strategy="voting", final_k=3
        )

        assert "selected_features" in result
        assert "scores" in result
        assert "method_contributions" in result
        assert result["combination_strategy"] == "voting"

        # Features selected by multiple methods should be preferred
        assert "feature1" in result["selected_features"]
        assert "feature3" in result["selected_features"]

    def test_combine_selection_ranking(self, sample_method_results):
        """Test ranking combination strategy."""
        result = combine_selection_results(
            sample_method_results, combination_strategy="ranking", final_k=2
        )

        assert result["combination_strategy"] == "ranking"
        assert len(result["selected_features"]) <= 2

    def test_combine_selection_scoring(self, sample_method_results):
        """Test scoring combination strategy."""
        result = combine_selection_results(
            sample_method_results, combination_strategy="scoring", final_k=3
        )

        assert result["combination_strategy"] == "scoring"
        assert len(result["selected_features"]) <= 3

    def test_combine_selection_empty_results(self):
        """Test combining with no method results."""
        result = combine_selection_results([], combination_strategy="voting", final_k=5)

        assert result["selected_features"] == []
        assert result["scores"] == {}

    def test_combine_selection_method_contributions(self, sample_method_results):
        """Test that method contributions are calculated."""
        result = combine_selection_results(
            sample_method_results, combination_strategy="voting", final_k=3
        )

        # Check that contributions sum makes sense
        assert "method1" in result["method_contributions"]
        assert "method2" in result["method_contributions"]
        assert "method3" in result["method_contributions"]

        # Contributions should be between 0 and 1
        for contrib in result["method_contributions"].values():
            assert 0 <= contrib <= 1


class TestApplyFeatureSelectionPipeline:
    """Tests for apply_feature_selection_pipeline function."""

    @pytest.fixture
    def sample_splits(self):
        """Create sample data splits for testing."""
        np.random.seed(42)
        data = {
            "train": pd.DataFrame(
                {
                    "feature1": np.random.randn(100),
                    "feature2": np.random.randn(100) * 2,
                    "feature3": np.random.randn(100) * 0.1,
                    "feature4": np.random.randn(100),
                    "target": np.random.randint(0, 2, 100),
                }
            ),
            "val": pd.DataFrame(
                {
                    "feature1": np.random.randn(50),
                    "feature2": np.random.randn(50) * 2,
                    "feature3": np.random.randn(50) * 0.1,
                    "feature4": np.random.randn(50),
                    "target": np.random.randint(0, 2, 50),
                }
            ),
        }
        return data

    @pytest.fixture
    def method_configs(self):
        """Create method configurations for testing."""
        return {
            "variance": {"threshold": 0.05},
            "mutual_info": {"k": 3, "random_state": 42},
            "final_k": 3,
            "combination_strategy": "voting",
        }

    def test_apply_feature_selection_pipeline_basic(
        self, sample_splits, method_configs
    ):
        """Test basic feature selection pipeline."""
        methods = ["variance", "mutual_info"]

        result = apply_feature_selection_pipeline(
            sample_splits, "target", methods, method_configs
        )

        assert "selected_features" in result
        assert "method_results" in result
        assert "combined_result" in result
        assert "original_features" in result
        assert "target_variable" in result

        # Should have selected features
        assert len(result["selected_features"]) > 0
        assert len(result["selected_features"]) <= method_configs["final_k"]

    def test_apply_feature_selection_pipeline_no_training_data(self, method_configs):
        """Test error when training data is missing."""
        splits = {"val": pd.DataFrame({"f1": [1, 2], "target": [0, 1]})}

        with pytest.raises(ValueError, match="Training data not found"):
            apply_feature_selection_pipeline(
                splits, "target", ["variance"], method_configs
            )

    def test_apply_feature_selection_pipeline_missing_target(
        self, sample_splits, method_configs
    ):
        """Test error when target variable is missing."""
        # Remove target from train split
        sample_splits["train"] = sample_splits["train"].drop("target", axis=1)

        with pytest.raises(ValueError, match="Target variable .* not found"):
            apply_feature_selection_pipeline(
                sample_splits, "target", ["variance"], method_configs
            )


class TestSaveSelectionResults:
    """Tests for save_selection_results function."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def sample_selection_results(self):
        """Create sample selection results."""
        return {
            "selected_features": ["feature1", "feature2", "feature3"],
            "method_results": [
                {
                    "method": "variance",
                    "selected_features": ["feature1", "feature2"],
                    "scores": {"feature1": 0.9, "feature2": 0.8},
                    "n_selected": 2,
                    "processing_time": 0.5,
                }
            ],
            "combined_result": {
                "selected_features": ["feature1", "feature2", "feature3"],
                "scores": {"feature1": 0.9, "feature2": 0.8, "feature3": 0.7},
                "method_contributions": {"variance": 0.67},
                "combination_strategy": "voting",
            },
            "original_features": ["feature1", "feature2", "feature3", "feature4"],
            "target_variable": "target",
            "n_original_features": 4,
            "n_selected_features": 3,
        }

    def test_save_selection_results_files_created(
        self, temp_dir, sample_selection_results
    ):
        """Test that all result files are created."""
        save_selection_results(sample_selection_results, str(temp_dir))

        # Check files exist
        assert (temp_dir / "selected_features.json").exists()
        assert (temp_dir / "feature_scores.csv").exists()
        assert (temp_dir / "feature_selection_report.json").exists()

    def test_save_selection_results_selected_features_content(
        self, temp_dir, sample_selection_results
    ):
        """Test selected_features.json content."""
        save_selection_results(sample_selection_results, str(temp_dir))

        with open(temp_dir / "selected_features.json", "r") as f:
            data = json.load(f)

        assert "selected_features" in data
        assert "selection_metadata" in data
        assert "method_contributions" in data
        assert len(data["selected_features"]) == 3

    def test_save_selection_results_feature_scores_content(
        self, temp_dir, sample_selection_results
    ):
        """Test feature_scores.csv content."""
        save_selection_results(sample_selection_results, str(temp_dir))

        df = pd.read_csv(temp_dir / "feature_scores.csv")

        assert "feature_name" in df.columns
        assert "combined_score" in df.columns
        assert "selected" in df.columns
        assert len(df) == 4  # All original features

    def test_save_selection_results_report_content(
        self, temp_dir, sample_selection_results
    ):
        """Test feature_selection_report.json content."""
        save_selection_results(sample_selection_results, str(temp_dir))

        with open(temp_dir / "feature_selection_report.json", "r") as f:
            report = json.load(f)

        assert "selection_summary" in report
        assert "method_performance" in report
        assert "feature_statistics" in report


class TestMainFunction:
    """Tests for main function."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def setup_training_data(self, temp_dir):
        """Helper to set up training data structure."""
        input_dir = temp_dir / "input"

        # Create train, test, val splits
        for split in ["train", "test", "val"]:
            split_dir = input_dir / split
            split_dir.mkdir(parents=True)

            # Create sample data with features for selection
            data = pd.DataFrame(
                {
                    "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                    "feature2": [2.0, 4.0, 6.0, 8.0, 10.0],
                    "feature3": [0.1, 0.2, 0.3, 0.4, 0.5],
                    "feature4": [5.0, 4.0, 3.0, 2.0, 1.0],
                    "target": [0, 1, 0, 1, 0],
                }
            )

            data_file = split_dir / f"{split}_processed_data.csv"
            data.to_csv(data_file, index=False)

        return input_dir

    def test_main_training_job_type(self, temp_dir):
        """Test main function with training job type."""
        # Set up input data
        input_dir = self.setup_training_data(temp_dir)
        output_dir = temp_dir / "output"

        # Create arguments
        args = argparse.Namespace(job_type="training")

        # Environment variables
        environ_vars = {
            "LABEL_FIELD": "target",
            "FEATURE_SELECTION_METHODS": "variance,mutual_info",
            "N_FEATURES_TO_SELECT": "3",
            "COMBINATION_STRATEGY": "voting",
        }

        # Path dictionaries
        input_paths = {"input_data": str(input_dir)}
        output_paths = {"processed_data": str(output_dir)}

        # Run main function with subprocess mocked (already done at module level)
        main(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,
            job_args=args,
        )

        # Verify output files were created
        # Files are saved directly to output_dir/split/split_processed_data.csv
        for split in ["train", "test", "val"]:
            expected_file = output_dir / split / f"{split}_processed_data.csv"
            assert expected_file.exists()

        # Verify artifacts were saved
        # Artifacts are saved to output_dir/model_artifacts when model_artifacts_output not specified
        artifacts_dir = output_dir / "model_artifacts"
        assert (artifacts_dir / "selected_features.json").exists()
        assert (artifacts_dir / "feature_scores.csv").exists()
        assert (artifacts_dir / "feature_selection_report.json").exists()

    def test_main_validation_job_type(self, temp_dir):
        """Test main function with validation job type."""
        # Set up validation data
        input_dir = temp_dir / "input"
        val_dir = input_dir / "validation"
        val_dir.mkdir(parents=True)

        data = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [4.0, 5.0, 6.0],
                "feature3": [7.0, 8.0, 9.0],
                "target": [0, 1, 0],
            }
        )
        data_file = val_dir / "validation_processed_data.csv"
        data.to_csv(data_file, index=False)

        # Set up model artifacts with selected features
        artifacts_input_dir = input_dir / "model_artifacts"
        artifacts_input_dir.mkdir()

        features_data = {"selected_features": ["feature1", "feature2"]}
        with open(artifacts_input_dir / "selected_features.json", "w") as f:
            json.dump(features_data, f)

        output_dir = temp_dir / "output"

        # Create arguments
        args = argparse.Namespace(job_type="validation")

        # Environment variables
        environ_vars = {"LABEL_FIELD": "target"}

        # Path dictionaries
        input_paths = {
            "input_data": str(input_dir),
            "model_artifacts_input": str(artifacts_input_dir),
        }
        output_paths = {"processed_data": str(output_dir)}

        # Run main function
        main(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,
            job_args=args,
        )

        # Verify validation data was processed
        # Files are saved directly to output_dir/validation/validation_processed_data.csv
        expected_file = output_dir / "validation" / "validation_processed_data.csv"
        assert expected_file.exists()

        # Verify only selected features + target were saved
        saved_data = pd.read_csv(expected_file)
        assert list(saved_data.columns) == ["feature1", "feature2", "target"]

    def test_main_missing_label_field(self, temp_dir):
        """Test main function with missing label field."""
        input_dir = self.setup_training_data(temp_dir)
        output_dir = temp_dir / "output"

        args = argparse.Namespace(job_type="training")

        # Missing LABEL_FIELD
        environ_vars = {}

        input_paths = {"input_data": str(input_dir)}
        output_paths = {"processed_data": str(output_dir)}

        with pytest.raises(ValueError, match="LABEL_FIELD environment variable"):
            main(
                input_paths=input_paths,
                output_paths=output_paths,
                environ_vars=environ_vars,
                job_args=args,
            )

    def test_main_parameter_accumulator_pattern(self, temp_dir):
        """Test that main function copies existing artifacts (parameter accumulator pattern)."""
        input_dir = self.setup_training_data(temp_dir)
        output_dir = temp_dir / "output"

        # Set up existing artifacts from previous step
        artifacts_input_dir = input_dir / "model_artifacts"
        artifacts_input_dir.mkdir()
        (artifacts_input_dir / "previous_artifact.pkl").write_text("previous")

        args = argparse.Namespace(job_type="training")

        environ_vars = {
            "LABEL_FIELD": "target",
            "FEATURE_SELECTION_METHODS": "variance",
            "N_FEATURES_TO_SELECT": "2",
        }

        input_paths = {
            "input_data": str(input_dir),
            "model_artifacts_input": str(artifacts_input_dir),
        }
        output_paths = {"processed_data": str(output_dir)}

        # Run main function
        main(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,
            job_args=args,
        )

        # Verify existing artifact was copied
        artifacts_output_dir = output_dir / "model_artifacts"
        assert (artifacts_output_dir / "previous_artifact.pkl").exists()
        assert (
            artifacts_output_dir / "previous_artifact.pkl"
        ).read_text() == "previous"

        # Verify new artifacts were also created
        assert (artifacts_output_dir / "selected_features.json").exists()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_feature_selection_with_all_constant_features(self):
        """Test feature selection when all features have zero variance."""
        X = pd.DataFrame(
            {
                "feature1": [1, 1, 1, 1],
                "feature2": [2, 2, 2, 2],
                "feature3": [3, 3, 3, 3],
            }
        )
        y = pd.Series([0, 1, 0, 1])

        result = variance_threshold_selection(X, threshold=0.01)

        # Should return no features selected
        assert result["n_selected"] == 0
        assert len(result["selected_features"]) == 0

    def test_feature_selection_with_single_feature(self):
        """Test feature selection with only one feature."""
        X = pd.DataFrame({"feature1": [1, 2, 3, 4, 5]})
        y = pd.Series([0, 1, 0, 1, 0])

        result = mutual_info_selection(X, y, k=3, random_state=42)

        # Should select the single feature
        assert result["n_selected"] == 1
        assert "feature1" in result["selected_features"]

    def test_feature_selection_k_exceeds_feature_count(self):
        """Test when k exceeds number of features."""
        X = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        y = pd.Series([0, 1, 0])

        result = mutual_info_selection(X, y, k=10, random_state=42)

        # Should select all available features
        assert result["n_selected"] == 2

    def test_correlation_selection_with_perfect_correlation(self):
        """Test correlation-based selection with perfectly correlated features."""
        X = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [2, 4, 6, 8, 10],  # Perfect correlation with feature1
                "feature3": [5, 4, 3, 2, 1],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0])

        result = correlation_based_selection(X, y, threshold=0.95)

        # Should remove one of the perfectly correlated features
        assert result["n_selected"] < 3
        assert len(result["removed_features"]) > 0


class TestFormatPreservation:
    """Tests to ensure format preservation across different file types."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_format_preservation_csv_to_csv(self, temp_dir):
        """Test that CSV format is preserved."""
        # Create CSV input
        input_dir = temp_dir / "input"
        train_dir = input_dir / "train"
        train_dir.mkdir(parents=True)

        data = pd.DataFrame({"f1": [1, 2], "f2": [3, 4], "target": [0, 1]})
        data.to_csv(train_dir / "train_processed_data.csv", index=False)

        # Load and save
        splits = load_preprocessed_data(str(input_dir))
        output_dir = temp_dir / "output"
        save_selected_data(splits, ["f1"], "target", str(output_dir))

        # Verify CSV format preserved
        output_file = output_dir / "train" / "train_processed_data.csv"
        assert output_file.exists()
        assert output_file.suffix == ".csv"

    def test_format_preservation_parquet_to_parquet(self, temp_dir):
        """Test that Parquet format is preserved."""
        # Create Parquet input
        input_dir = temp_dir / "input"
        train_dir = input_dir / "train"
        train_dir.mkdir(parents=True)

        data = pd.DataFrame({"f1": [1, 2], "f2": [3, 4], "target": [0, 1]})
        data.to_parquet(train_dir / "train_processed_data.parquet", index=False)

        # Load and save
        splits = load_preprocessed_data(str(input_dir))
        output_dir = temp_dir / "output"
        save_selected_data(splits, ["f1"], "target", str(output_dir))

        # Verify Parquet format preserved
        output_file = output_dir / "train" / "train_processed_data.parquet"
        assert output_file.exists()
        assert output_file.suffix == ".parquet"


# ============================================================================
# SUMMARY
# ============================================================================
"""
Test Coverage Summary:
- ✓ File format detection (CSV, TSV, Parquet)
- ✓ Data loading (training and inference modes)
- ✓ Data saving with format preservation
- ✓ Artifact management (copy, load, save)
- ✓ All feature selection methods (9 methods)
- ✓ Combination strategies (voting, ranking, scoring)
- ✓ End-to-end pipeline
- ✓ Main function (training and validation modes)
- ✓ Edge cases (constant features, single feature, k > n)
- ✓ Parameter accumulator pattern
- ✓ Error handling
"""

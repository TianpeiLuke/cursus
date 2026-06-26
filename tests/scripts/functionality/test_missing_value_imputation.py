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
from sklearn.impute import SimpleImputer

from cursus.steps.scripts.missing_value_imputation import (
    main,
    internal_main,
    load_split_data,
    save_output_data,
    analyze_missing_values,
    validate_imputation_data,
    load_imputation_config,
    get_pandas_na_values,
    validate_text_fill_value,
    detect_column_type,
    ImputationStrategyManager,
    SimpleImputationEngine,
    save_imputation_artifacts,
    load_imputation_parameters,
    process_data,
    generate_imputation_report,
    calculate_imputation_quality_metrics,
    generate_imputation_recommendations,
    generate_imputation_text_summary,
    copy_existing_artifacts,
    _detect_file_format,
    IMPUTATION_PARAMS_FILENAME,
    IMPUTATION_SUMMARY_FILENAME,
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


class TestLoadSplitData:
    """Tests for load_split_data function."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def setup_training_data(self, temp_dir, file_format="csv"):
        """Helper to set up training data structure."""
        input_dir = temp_dir / "input"

        # Create train, test, val splits
        for split in ["train", "test", "val"]:
            split_dir = input_dir / split
            split_dir.mkdir(parents=True)

            # Create sample data with missing values
            data = pd.DataFrame(
                {
                    "feature1": [1.0, 2.0, np.nan, 4.0, 5.0],
                    "feature2": ["A", "B", None, "D", "E"],
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

    def test_load_split_data_training_job_type(self, temp_dir):
        """Test loading data for training job type."""
        input_dir = self.setup_training_data(temp_dir)

        result = load_split_data("training", str(input_dir))

        # Should return all three splits plus format metadata
        assert isinstance(result, dict)
        assert "train" in result
        assert "test" in result
        assert "val" in result
        assert "_format" in result  # Format metadata key

        # Check data structure (skip _format key)
        for split_name, df in result.items():
            if split_name == "_format":
                continue
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 5
            assert list(df.columns) == ["feature1", "feature2", "target"]

    def test_load_split_data_tsv_format(self, temp_dir):
        """Test loading TSV format data."""
        input_dir = self.setup_training_data(temp_dir, file_format="tsv")

        result = load_split_data("training", str(input_dir))

        assert result["_format"] == "tsv"
        assert isinstance(result["train"], pd.DataFrame)
        assert len(result["train"]) == 5

    def test_load_split_data_parquet_format(self, temp_dir):
        """Test loading Parquet format data."""
        input_dir = self.setup_training_data(temp_dir, file_format="parquet")

        result = load_split_data("training", str(input_dir))

        assert result["_format"] == "parquet"
        assert isinstance(result["train"], pd.DataFrame)
        assert len(result["train"]) == 5

    def test_load_split_data_validation_job_type(self, temp_dir):
        """Test loading data for validation job type."""
        input_dir = temp_dir / "input"
        val_dir = input_dir / "validation"
        val_dir.mkdir(parents=True)

        # Create validation data
        data = pd.DataFrame(
            {
                "feature1": [1.0, np.nan, 3.0],
                "feature2": ["A", None, "C"],
                "target": [0, 1, 0],
            }
        )

        data_file = val_dir / "validation_processed_data.csv"
        data.to_csv(data_file, index=False)

        result = load_split_data("validation", str(input_dir))

        # Should return validation split plus format metadata
        assert isinstance(result, dict)
        assert "validation" in result
        assert "_format" in result
        assert len(result) == 2  # validation + _format

        val_df = result["validation"]
        assert isinstance(val_df, pd.DataFrame)
        assert len(val_df) == 3

    def test_load_split_data_file_not_found(self, temp_dir):
        """Test loading data when file doesn't exist."""
        input_dir = temp_dir / "input"
        nonexistent_dir = input_dir / "nonexistent"
        nonexistent_dir.mkdir(parents=True)

        with pytest.raises(RuntimeError, match="No processed data file found"):
            load_split_data("nonexistent", str(input_dir))


class TestSaveOutputData:
    """Tests for save_output_data function."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_save_output_data_training_csv(self, temp_dir):
        """Test saving data for training job type in CSV format."""
        output_dir = temp_dir / "output"

        # Create test data with format metadata
        data_dict = {
            "train": pd.DataFrame({"feature1": [1, 2, 3], "target": [0, 1, 0]}),
            "test": pd.DataFrame({"feature1": [4, 5, 6], "target": [1, 0, 1]}),
            "val": pd.DataFrame({"feature1": [7, 8, 9], "target": [0, 1, 0]}),
            "_format": "csv",
        }

        save_output_data("training", str(output_dir), data_dict)

        # Check that all files were created in CSV format
        for split_name in ["train", "test", "val"]:
            expected_file = output_dir / split_name / f"{split_name}_processed_data.csv"
            assert expected_file.exists()

            # Check file content
            saved_data = pd.read_csv(expected_file)
            pd.testing.assert_frame_equal(saved_data, data_dict[split_name])

    def test_save_output_data_training_tsv(self, temp_dir):
        """Test saving data in TSV format."""
        output_dir = temp_dir / "output"

        data_dict = {
            "train": pd.DataFrame({"feature1": [1, 2, 3], "target": [0, 1, 0]}),
            "_format": "tsv",
        }

        save_output_data("training", str(output_dir), data_dict)

        expected_file = output_dir / "train" / "train_processed_data.tsv"
        assert expected_file.exists()

        # Verify TSV format
        saved_data = pd.read_csv(expected_file, sep="\t")
        pd.testing.assert_frame_equal(saved_data, data_dict["train"])

    def test_save_output_data_training_parquet(self, temp_dir):
        """Test saving data in Parquet format."""
        output_dir = temp_dir / "output"

        data_dict = {
            "train": pd.DataFrame({"feature1": [1, 2, 3], "target": [0, 1, 0]}),
            "_format": "parquet",
        }

        save_output_data("training", str(output_dir), data_dict)

        expected_file = output_dir / "train" / "train_processed_data.parquet"
        assert expected_file.exists()

        # Verify Parquet format
        saved_data = pd.read_parquet(expected_file)
        pd.testing.assert_frame_equal(saved_data, data_dict["train"])

    def test_save_output_data_validation(self, temp_dir):
        """Test saving data for validation job type."""
        output_dir = temp_dir / "output"

        data_dict = {
            "validation": pd.DataFrame({"feature1": [1, 2, 3], "target": [0, 1, 0]}),
            "_format": "csv",
        }

        save_output_data("validation", str(output_dir), data_dict)

        expected_file = output_dir / "validation" / "validation_processed_data.csv"
        assert expected_file.exists()

        saved_data = pd.read_csv(expected_file)
        pd.testing.assert_frame_equal(saved_data, data_dict["validation"])

    def test_save_output_data_unsupported_format(self, temp_dir):
        """Test error with unsupported format."""
        output_dir = temp_dir / "output"

        data_dict = {
            "train": pd.DataFrame({"feature1": [1, 2, 3], "target": [0, 1, 0]}),
            "_format": "json",  # Unsupported format
        }

        with pytest.raises(RuntimeError, match="Unsupported output format"):
            save_output_data("training", str(output_dir), data_dict)


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
        (src_dir / "artifact3.txt").write_text("text artifact")

        copy_existing_artifacts(str(src_dir), str(dst_dir))

        # Check that files were copied
        assert (dst_dir / "artifact1.pkl").exists()
        assert (dst_dir / "artifact2.json").exists()
        assert (dst_dir / "artifact3.txt").exists()

        # Verify content
        assert (dst_dir / "artifact1.pkl").read_text() == "artifact1"
        assert (dst_dir / "artifact2.json").read_text() == '{"key": "value"}'
        assert (dst_dir / "artifact3.txt").read_text() == "text artifact"

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

    def test_copy_existing_artifacts_skip_subdirectories(self, temp_dir):
        """Test that subdirectories are not copied."""
        src_dir = temp_dir / "src"
        dst_dir = temp_dir / "dst"
        src_dir.mkdir()

        # Create files and subdirectory
        (src_dir / "file1.txt").write_text("file1")
        subdir = src_dir / "subdir"
        subdir.mkdir()
        (subdir / "file2.txt").write_text("file2")

        copy_existing_artifacts(str(src_dir), str(dst_dir))

        # Only top-level files should be copied
        assert (dst_dir / "file1.txt").exists()
        assert not (dst_dir / "subdir").exists()


class TestAnalyzeMissingValues:
    """Tests for analyze_missing_values function."""

    def test_analyze_missing_values_with_missing_data(self):
        """Test analysis with missing values present."""
        df = pd.DataFrame(
            {
                "numeric_col": [1.0, 2.0, np.nan, 4.0, np.nan],
                "text_col": ["A", "B", None, "D", "E"],
                "complete_col": [1, 2, 3, 4, 5],
                "target": [0, 1, 0, 1, 0],
            }
        )

        result = analyze_missing_values(df)

        # Check structure
        assert isinstance(result, dict)
        assert "total_records" in result
        assert "columns_with_missing" in result
        assert "missing_patterns" in result
        assert "data_types" in result
        assert "imputation_recommendations" in result

        # Check values
        assert result["total_records"] == 5
        assert "numeric_col" in result["columns_with_missing"]
        assert "text_col" in result["columns_with_missing"]
        assert "complete_col" not in result["columns_with_missing"]

        # Check numeric column analysis
        numeric_analysis = result["columns_with_missing"]["numeric_col"]
        assert numeric_analysis["missing_count"] == 2
        assert numeric_analysis["missing_percentage"] == 40.0

        # Check recommendations
        assert "numeric_col" in result["imputation_recommendations"]
        assert "text_col" in result["imputation_recommendations"]

    def test_analyze_missing_values_no_missing_data(self):
        """Test analysis with no missing values."""
        df = pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["A", "B", "C", "D", "E"],
                "target": [0, 1, 0, 1, 0],
            }
        )

        result = analyze_missing_values(df)

        assert result["total_records"] == 5
        assert len(result["columns_with_missing"]) == 0
        assert result["missing_patterns"]["records_with_no_missing"] == 5
        assert result["missing_patterns"]["records_with_missing"] == 0

    def test_analyze_missing_values_empty_dataframe(self):
        """Test analysis with empty DataFrame."""
        df = pd.DataFrame()

        # Based on source code analysis, empty DataFrame causes NaN in max() operation
        # This should raise ValueError as per the actual implementation
        with pytest.raises(ValueError, match="cannot convert float NaN to integer"):
            analyze_missing_values(df)

    def test_analyze_missing_values_skewed_distribution(self):
        """Test recommendation for skewed distribution."""
        # Create skewed numerical data
        df = pd.DataFrame(
            {
                "skewed_col": [1, 1, 1, 1, 1, 2, 3, 4, 100, np.nan],
                "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            }
        )

        result = analyze_missing_values(df)

        # Should recommend median for skewed data
        assert result["imputation_recommendations"]["skewed_col"] == "median"


class TestValidateImputationData:
    """Tests for validate_imputation_data function."""

    def test_validate_imputation_data_success(self):
        """Test validation with valid data."""
        df = pd.DataFrame(
            {
                "feature1": [1.0, np.nan, 3.0],
                "feature2": ["A", None, "C"],
                "target": [0, 1, 0],
            }
        )

        result = validate_imputation_data(df, "target")

        assert result["is_valid"] is True
        assert "target" in result["excluded_columns"]
        assert "feature1" in result["imputable_columns"]
        assert "feature2" in result["imputable_columns"]

    def test_validate_imputation_data_no_missing_values(self):
        """Test validation when no missing values exist."""
        df = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": ["A", "B", "C"], "target": [0, 1, 0]}
        )

        result = validate_imputation_data(df, "target")

        assert result["is_valid"] is True
        assert len(result["imputable_columns"]) == 0
        assert len(result["warnings"]) > 0

    def test_validate_imputation_data_missing_label(self):
        """Test validation when label field is missing."""
        df = pd.DataFrame(
            {"feature1": [1.0, np.nan, 3.0], "feature2": ["A", None, "C"]}
        )

        result = validate_imputation_data(df, "target")

        assert result["is_valid"] is True
        assert any("not found" in warning for warning in result["warnings"])

    def test_validate_imputation_data_with_exclude_columns(self):
        """Test validation with excluded columns."""
        df = pd.DataFrame(
            {
                "feature1": [1.0, np.nan, 3.0],
                "feature2": ["A", None, "C"],
                "id_col": [1, 2, 3],
                "target": [0, 1, 0],
            }
        )

        result = validate_imputation_data(df, "target", exclude_columns=["id_col"])

        assert "id_col" in result["excluded_columns"]
        assert "target" in result["excluded_columns"]
        assert "id_col" not in result["imputable_columns"]


class TestLoadImputationConfig:
    """Tests for load_imputation_config function."""

    def test_load_imputation_config_defaults(self):
        """Test loading config with default values."""
        environ_vars = {}

        result = load_imputation_config(environ_vars)

        # Check default values
        assert result["default_numerical_strategy"] == "mean"
        assert result["default_categorical_strategy"] == "mode"
        assert result["default_text_strategy"] == "mode"
        assert result["numerical_constant_value"] == 0.0
        assert result["categorical_constant_value"] == "Unknown"
        assert result["text_constant_value"] == "Unknown"
        assert result["categorical_preserve_dtype"] is True
        assert result["auto_detect_categorical"] is True
        assert result["categorical_unique_ratio_threshold"] == 0.1
        assert result["validate_fill_values"] is True
        assert result["column_strategies"] == {}
        assert result["exclude_columns"] == []

    def test_load_imputation_config_custom_values(self):
        """Test loading config with custom values."""
        environ_vars = {
            "DEFAULT_NUMERICAL_STRATEGY": "median",
            "DEFAULT_CATEGORICAL_STRATEGY": "constant",
            "NUMERICAL_CONSTANT_VALUE": "999",
            "CATEGORICAL_CONSTANT_VALUE": "Missing",
            "CATEGORICAL_PRESERVE_DTYPE": "false",
            "AUTO_DETECT_CATEGORICAL": "false",
            "CATEGORICAL_UNIQUE_RATIO_THRESHOLD": "0.05",
            "VALIDATE_FILL_VALUES": "false",
            "EXCLUDE_COLUMNS": "id,timestamp,metadata",
        }

        result = load_imputation_config(environ_vars)

        assert result["default_numerical_strategy"] == "median"
        assert result["default_categorical_strategy"] == "constant"
        assert result["numerical_constant_value"] == 999.0
        assert result["categorical_constant_value"] == "Missing"
        assert result["categorical_preserve_dtype"] is False
        assert result["auto_detect_categorical"] is False
        assert result["categorical_unique_ratio_threshold"] == 0.05
        assert result["validate_fill_values"] is False
        assert result["exclude_columns"] == ["id", "timestamp", "metadata"]

    def test_load_imputation_config_column_strategies(self):
        """Test loading config with column-specific strategies."""
        environ_vars = {
            "COLUMN_STRATEGY_age": "median",
            "COLUMN_STRATEGY_income": "mean",
            "COLUMN_STRATEGY_category": "mode",
        }

        result = load_imputation_config(environ_vars)

        assert result["column_strategies"]["age"] == "median"
        assert result["column_strategies"]["income"] == "mean"
        assert result["column_strategies"]["category"] == "mode"


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_get_pandas_na_values(self):
        """Test getting pandas NA values."""
        na_values = get_pandas_na_values()

        assert isinstance(na_values, set)
        assert "N/A" in na_values
        assert "NULL" in na_values
        assert "NaN" in na_values
        assert "none" in na_values

    def test_validate_text_fill_value_valid(self):
        """Test validating valid text fill values."""
        assert validate_text_fill_value("Unknown") is True
        assert validate_text_fill_value("Missing") is True
        assert validate_text_fill_value("") is True
        assert validate_text_fill_value("Custom_Value") is True

    def test_validate_text_fill_value_invalid(self):
        """Test validating invalid text fill values."""
        assert validate_text_fill_value("N/A") is False
        assert validate_text_fill_value("NULL") is False
        assert validate_text_fill_value("NaN") is False
        assert validate_text_fill_value("none") is False

    def test_detect_column_type_numerical(self):
        """Test detecting numerical column type."""
        df = pd.DataFrame({"col": [1, 2, 3, np.nan]})
        config = {"auto_detect_categorical": True}

        result = detect_column_type(df, "col", config)
        assert result == "numerical"

    def test_detect_column_type_categorical(self):
        """Test detecting categorical column type."""
        df = pd.DataFrame({"col": pd.Categorical(["A", "B", "A", "B"])})
        config = {"auto_detect_categorical": True}

        result = detect_column_type(df, "col", config)
        assert result == "categorical"

    def test_detect_column_type_text_low_unique_ratio(self):
        """Test detecting categorical type based on unique ratio."""
        df = pd.DataFrame({"col": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"]})
        config = {
            "auto_detect_categorical": True,
            "categorical_unique_ratio_threshold": 0.3,
        }

        result = detect_column_type(df, "col", config)
        assert result == "categorical"

    def test_detect_column_type_text_high_unique_ratio(self):
        """Test detecting text type based on unique ratio."""
        df = pd.DataFrame({"col": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]})
        config = {
            "auto_detect_categorical": True,
            "categorical_unique_ratio_threshold": 0.3,
        }

        result = detect_column_type(df, "col", config)
        assert result == "text"

    def test_detect_column_type_auto_detect_disabled(self):
        """Test detecting type with auto-detection disabled."""
        df = pd.DataFrame({"col": ["A", "B", "A", "B"]})
        config = {"auto_detect_categorical": False}

        result = detect_column_type(df, "col", config)
        assert result == "text"


class TestImputationStrategyManager:
    """Tests for ImputationStrategyManager class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            "default_numerical_strategy": "mean",
            "default_categorical_strategy": "mode",
            "default_text_strategy": "mode",
            "numerical_constant_value": 0,
            "categorical_constant_value": "Unknown",
            "text_constant_value": "Unknown",
            "validate_fill_values": True,
            "auto_detect_categorical": True,
            "categorical_unique_ratio_threshold": 0.1,
            "column_strategies": {"age": "median", "category": "constant"},
        }

    @pytest.fixture
    def manager(self, config):
        """Create ImputationStrategyManager instance."""
        return ImputationStrategyManager(config)

    def test_init(self, config):
        """Test ImputationStrategyManager initialization."""
        manager = ImputationStrategyManager(config)
        assert manager.config == config
        assert isinstance(manager.pandas_na_values, set)

    def test_get_strategy_for_column_numerical(self, manager):
        """Test getting strategy for numerical column."""
        df = pd.DataFrame({"col": [1.0, 2.0, np.nan, 4.0]})

        imputer = manager.get_strategy_for_column(df, "col")

        assert isinstance(imputer, SimpleImputer)
        assert imputer.strategy == "mean"

    def test_get_strategy_for_column_categorical(self, manager):
        """Test getting strategy for categorical column."""
        df = pd.DataFrame({"col": pd.Categorical(["A", "B", "A", None])})

        imputer = manager.get_strategy_for_column(df, "col")

        assert isinstance(imputer, SimpleImputer)
        assert imputer.strategy == "most_frequent"

    def test_get_strategy_for_column_text(self, manager):
        """Test getting strategy for text column."""
        df = pd.DataFrame({"col": ["text1", "text2", "text3", None]})

        imputer = manager.get_strategy_for_column(df, "col")

        assert isinstance(imputer, SimpleImputer)
        assert imputer.strategy == "most_frequent"

    def test_get_strategy_for_column_explicit_config(self, manager):
        """Test getting strategy with explicit column configuration."""
        df = pd.DataFrame({"age": [25, 30, np.nan, 40]})

        imputer = manager.get_strategy_for_column(df, "age")

        assert isinstance(imputer, SimpleImputer)
        assert imputer.strategy == "median"

    def test_create_numerical_strategy_mean(self, manager):
        """Test creating numerical mean strategy."""
        imputer = manager._create_numerical_strategy("mean")
        assert imputer.strategy == "mean"

    def test_create_numerical_strategy_median(self, manager):
        """Test creating numerical median strategy."""
        imputer = manager._create_numerical_strategy("median")
        assert imputer.strategy == "median"

    def test_create_numerical_strategy_constant(self, manager):
        """Test creating numerical constant strategy."""
        imputer = manager._create_numerical_strategy("constant")
        assert imputer.strategy == "constant"
        assert imputer.fill_value == 0  # From config

    def test_create_numerical_strategy_unknown(self, manager):
        """Test creating numerical strategy with unknown name."""
        # Should fall back to mean and log warning
        imputer = manager._create_numerical_strategy("unknown_strategy")
        assert imputer.strategy == "mean"

    def test_create_categorical_strategy_mode(self, manager):
        """Test creating categorical mode strategy."""
        df = pd.DataFrame({"col": ["A", "B", "A", None]})
        imputer = manager._create_categorical_strategy(df, "col", "mode")
        assert imputer.strategy == "most_frequent"

    def test_create_categorical_strategy_constant(self, manager):
        """Test creating categorical constant strategy."""
        df = pd.DataFrame({"col": ["A", "B", "A", None]})
        imputer = manager._create_categorical_strategy(df, "col", "constant")
        assert imputer.strategy == "constant"
        assert imputer.fill_value == "Unknown"

    def test_create_categorical_strategy_unsafe_fill_value(self, manager):
        """Test creating categorical strategy with unsafe fill value."""
        # Configure with pandas NA value
        manager.config["categorical_constant_value"] = "N/A"
        df = pd.DataFrame({"col": ["A", "B", "A", None]})

        imputer = manager._create_categorical_strategy(df, "col", "constant")
        assert imputer.strategy == "constant"
        assert imputer.fill_value == "Missing"  # Should be replaced

    def test_create_text_strategy_mode(self, manager):
        """Test creating text mode strategy."""
        imputer = manager._create_text_strategy("mode")
        assert imputer.strategy == "most_frequent"

    def test_create_text_strategy_constant(self, manager):
        """Test creating text constant strategy."""
        imputer = manager._create_text_strategy("constant")
        assert imputer.strategy == "constant"
        assert imputer.fill_value == "Unknown"

    def test_create_text_strategy_empty(self, manager):
        """Test creating text empty strategy."""
        imputer = manager._create_text_strategy("empty")
        assert imputer.strategy == "constant"
        assert imputer.fill_value == ""

    def test_create_text_strategy_unsafe_fill_value(self, manager):
        """Test creating text strategy with unsafe fill value."""
        # Configure with pandas NA value
        manager.config["text_constant_value"] = "NULL"

        imputer = manager._create_text_strategy("constant")
        assert imputer.strategy == "constant"
        assert imputer.fill_value == "Unknown"  # Should be replaced


class TestSimpleImputationEngine:
    """Tests for SimpleImputationEngine class."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            "default_numerical_strategy": "mean",
            "default_categorical_strategy": "mode",
            "default_text_strategy": "mode",
            "exclude_columns": [],
        }

    @pytest.fixture
    def strategy_manager(self, config):
        """Create ImputationStrategyManager instance."""
        return ImputationStrategyManager(config)

    @pytest.fixture
    def engine(self, strategy_manager):
        """Create SimpleImputationEngine instance."""
        return SimpleImputationEngine(strategy_manager, "target")

    @pytest.fixture
    def sample_data(self):
        """Create sample data with missing values."""
        return pd.DataFrame(
            {
                "numeric_col": [1.0, 2.0, np.nan, 4.0, np.nan],
                "text_col": ["A", "B", None, "D", "E"],
                "complete_col": [1, 2, 3, 4, 5],
                "target": [0, 1, 0, 1, 0],
            }
        )

    def test_init(self, strategy_manager):
        """Test SimpleImputationEngine initialization."""
        engine = SimpleImputationEngine(strategy_manager, "target")
        assert engine.strategy_manager == strategy_manager
        assert engine.label_field == "target"
        assert engine.fitted_imputers == {}
        assert engine.imputation_statistics == {}

    def test_fit(self, engine, sample_data):
        """Test fitting imputation parameters."""
        engine.fit(sample_data)

        # Should fit imputers for columns with missing values (excluding target)
        assert "numeric_col" in engine.fitted_imputers
        assert "text_col" in engine.fitted_imputers
        assert "complete_col" not in engine.fitted_imputers  # No missing values
        assert "target" not in engine.fitted_imputers  # Excluded

        # Check imputation statistics
        assert "numeric_col" in engine.imputation_statistics
        assert "text_col" in engine.imputation_statistics

        numeric_stats = engine.imputation_statistics["numeric_col"]
        assert numeric_stats["missing_count_training"] == 2
        assert numeric_stats["missing_percentage_training"] == 40.0
        assert numeric_stats["strategy"] == "mean"

    def test_transform(self, engine, sample_data):
        """Test transforming data with fitted imputers."""
        # First fit the engine
        engine.fit(sample_data)

        # Then transform
        result = engine.transform(sample_data)

        # Check that missing values were imputed
        assert result["numeric_col"].isnull().sum() == 0
        assert result["complete_col"].isnull().sum() == 0  # Was already complete

        # Check that transformation log was created
        assert hasattr(engine, "last_transformation_log")
        assert "numeric_col" in engine.last_transformation_log
        assert "text_col" in engine.last_transformation_log

    def test_fit_transform(self, engine, sample_data):
        """Test fit_transform method."""
        result = engine.fit_transform(sample_data)

        # Should have both fitted imputers and transformed data
        assert len(engine.fitted_imputers) > 0
        assert result["numeric_col"].isnull().sum() == 0

    def test_transform_without_fit(self, engine, sample_data):
        """Test transforming without fitting first."""
        # Should work but not impute anything since no fitted imputers
        result = engine.transform(sample_data)

        # Data should be unchanged
        pd.testing.assert_frame_equal(result, sample_data)

    def test_transform_missing_column(self, engine, sample_data):
        """Test transforming data missing a column that was fitted."""
        # Fit on full data
        engine.fit(sample_data)

        # Transform data missing a column
        partial_data = sample_data.drop("text_col", axis=1)
        result = engine.transform(partial_data)

        # Should work without error, just skip missing column
        assert "text_col" not in result.columns
        assert result["numeric_col"].isnull().sum() == 0

    def test_get_imputation_summary(self, engine, sample_data):
        """Test getting imputation summary."""
        engine.fit(sample_data)
        engine.transform(sample_data)

        summary = engine.get_imputation_summary()

        assert "fitted_columns" in summary
        assert "imputation_statistics" in summary
        assert "last_transformation_log" in summary
        assert "total_imputers" in summary

        assert len(summary["fitted_columns"]) == 2  # numeric_col and text_col
        assert summary["total_imputers"] == 2


class TestFileOperations:
    """Tests for file operation functions."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def sample_engine(self):
        """Create sample imputation engine with fitted parameters."""
        config = {"default_numerical_strategy": "mean"}
        strategy_manager = ImputationStrategyManager(config)
        engine = SimpleImputationEngine(strategy_manager, "target")

        # Mock fitted imputers
        engine.fitted_imputers = {"col1": SimpleImputer(strategy="mean")}
        engine.imputation_statistics = {
            "col1": {
                "strategy": "mean",
                "missing_count_training": 5,
                "missing_percentage_training": 25.0,
                "data_type": "float64",
            }
        }

        return engine

    def test_save_imputation_artifacts(self, temp_dir, sample_engine):
        """Test saving imputation artifacts."""
        config = {"test_config": "value"}

        # Need to add statistics to the imputer for extraction
        sample_engine.fitted_imputers["col1"].statistics_ = np.array([2.5])

        save_imputation_artifacts(sample_engine, config, temp_dir)

        # Check that files were created
        params_file = temp_dir / IMPUTATION_PARAMS_FILENAME
        summary_file = temp_dir / IMPUTATION_SUMMARY_FILENAME

        assert params_file.exists()
        assert summary_file.exists()

        # Check params file content - now saves simple dict format
        with open(params_file, "rb") as f:
            params = pkl.load(f)

        # New format: simple dict mapping column names to values
        assert isinstance(params, dict)
        assert "col1" in params

    def test_load_imputation_parameters_success(self, temp_dir, sample_engine):
        """Test loading imputation parameters successfully."""
        config = {"test_config": "value"}

        # Need to add statistics to the imputer for extraction
        sample_engine.fitted_imputers["col1"].statistics_ = np.array([2.5])

        # First save parameters
        save_imputation_artifacts(sample_engine, config, temp_dir)

        # Then load them
        params_file = temp_dir / IMPUTATION_PARAMS_FILENAME
        result = load_imputation_parameters(params_file)

        # New format: simple dict mapping column names to values
        assert isinstance(result, dict)
        assert "col1" in result

    def test_load_imputation_parameters_file_not_found(self, temp_dir):
        """Test loading imputation parameters when file doesn't exist."""
        nonexistent_file = temp_dir / "nonexistent.pkl"

        with pytest.raises(
            FileNotFoundError, match="Imputation parameters file not found"
        ):
            load_imputation_parameters(nonexistent_file)


class TestProcessData:
    """Tests for process_data function."""

    @pytest.fixture
    def sample_data_dict(self):
        """Create sample data dictionary."""
        return {
            "train": pd.DataFrame(
                {
                    "feature1": [1.0, 2.0, np.nan, 4.0],
                    "feature2": ["A", "B", None, "D"],
                    "target": [0, 1, 0, 1],
                }
            ),
            "test": pd.DataFrame(
                {
                    "feature1": [5.0, np.nan, 7.0, 8.0],
                    "feature2": ["E", None, "G", "H"],
                    "target": [1, 0, 1, 0],
                }
            ),
        }

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            "default_numerical_strategy": "mean",
            "default_categorical_strategy": "mode",
            "exclude_columns": [],
        }

    def test_process_data_training_mode(self, sample_data_dict, config):
        """Test processing data in training mode."""
        transformed_data, engine = process_data(
            data_dict=sample_data_dict,
            label_field="target",
            job_type="training",
            imputation_config=config,
        )

        # Should return transformed data for all splits
        assert "train" in transformed_data
        assert "test" in transformed_data

        # Should have fitted imputers
        assert len(engine.fitted_imputers) > 0

        # Data should be imputed
        assert transformed_data["train"]["feature1"].isnull().sum() == 0
        assert transformed_data["test"]["feature1"].isnull().sum() == 0

    def test_process_data_inference_mode(self, sample_data_dict, config):
        """Test processing data in inference mode."""
        # Create mock imputation parameters - simple dict format
        imputation_parameters = {
            "feature1": 2.5,  # Simple column -> value mapping
            "feature2": "B",
        }

        # Only use validation data
        val_data = {"validation": sample_data_dict["train"]}

        transformed_data, engine = process_data(
            data_dict=val_data,
            label_field="target",
            job_type="validation",
            imputation_config=config,
            imputation_parameters=imputation_parameters,
        )

        # Should return transformed validation data
        assert "validation" in transformed_data
        assert len(transformed_data) > 0

        # Data should be imputed
        assert transformed_data["validation"]["feature1"].isnull().sum() == 0

    def test_process_data_inference_mode_no_parameters(self, sample_data_dict, config):
        """Test processing data in inference mode without parameters."""
        val_data = {"validation": sample_data_dict["train"]}

        with pytest.raises(ValueError, match="imputation_parameters must be provided"):
            process_data(
                data_dict=val_data,
                label_field="target",
                job_type="validation",
                imputation_config=config,
                imputation_parameters=None,
            )


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

            # Create sample data with missing values
            data = pd.DataFrame(
                {
                    "feature1": [1.0, 2.0, np.nan, 4.0, 5.0],
                    "feature2": ["A", "B", None, "D", "E"],
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
            "DEFAULT_NUMERICAL_STRATEGY": "mean",
            "DEFAULT_CATEGORICAL_STRATEGY": "mode",
        }

        # Path dictionaries - use correct key names
        input_paths = {"input_data": str(input_dir)}
        output_paths = {"processed_data": str(output_dir)}

        # Run main function
        result, engine = main(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,
            job_args=args,
        )

        # Verify results
        assert isinstance(result, dict)
        assert "train" in result
        assert "test" in result
        assert "val" in result

        # Check that imputation was applied (skip _format key)
        for split_name, df in result.items():
            if split_name == "_format":
                continue
            assert df["feature1"].isnull().sum() == 0

        # Check that artifacts were saved in model_artifacts subdirectory
        artifacts_dir = output_dir / "model_artifacts"
        params_file = artifacts_dir / IMPUTATION_PARAMS_FILENAME
        summary_file = artifacts_dir / IMPUTATION_SUMMARY_FILENAME
        assert params_file.exists()
        assert summary_file.exists()

    def test_main_missing_required_paths(self, temp_dir):
        """Test main function with missing required paths."""
        args = argparse.Namespace(job_type="training")
        environ_vars = {"LABEL_FIELD": "target"}

        # Missing input_data
        with pytest.raises(ValueError, match="Missing required input path: input_data"):
            main(
                input_paths={},
                output_paths={"processed_data": str(temp_dir)},
                environ_vars=environ_vars,
                job_args=args,
            )

        # Missing processed_data
        with pytest.raises(
            ValueError, match="Missing required output path: processed_data"
        ):
            main(
                input_paths={"input_data": str(temp_dir)},
                output_paths={},
                environ_vars=environ_vars,
                job_args=args,
            )

    def test_main_missing_job_args(self, temp_dir):
        """Test main function with missing job_args."""
        environ_vars = {"LABEL_FIELD": "target"}
        input_paths = {"input_data": str(temp_dir)}
        output_paths = {"processed_data": str(temp_dir)}

        # Missing job_args
        with pytest.raises(
            ValueError, match="job_args must contain job_type parameter"
        ):
            main(
                input_paths=input_paths,
                output_paths=output_paths,
                environ_vars=environ_vars,
                job_args=None,
            )

        # job_args without job_type
        args = argparse.Namespace()  # No job_type attribute
        with pytest.raises(
            ValueError, match="job_args must contain job_type parameter"
        ):
            main(
                input_paths=input_paths,
                output_paths=output_paths,
                environ_vars=environ_vars,
                job_args=args,
            )


class TestReportGeneration:
    """Tests for report generation functions."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def sample_imputation_summary(self):
        """Create sample imputation summary."""
        return {
            "fitted_columns": ["col1", "col2"],
            "imputation_statistics": {
                "col1": {
                    "strategy": "mean",
                    "missing_percentage_training": 25.0,
                    "data_type": "float64",
                },
                "col2": {
                    "strategy": "most_frequent",
                    "missing_percentage_training": 60.0,
                    "data_type": "object",
                },
            },
            "total_imputers": 2,
        }

    @pytest.fixture
    def sample_missing_analysis(self):
        """Create sample missing analysis."""
        return {
            "total_records": 100,
            "columns_with_missing": {"col1": {}, "col2": {}},
            "missing_patterns": {
                "records_with_missing": 40,
                "records_with_no_missing": 60,
            },
        }

    @pytest.fixture
    def sample_engine(self):
        """Create sample imputation engine for testing."""
        config = {"default_numerical_strategy": "mean", "exclude_columns": []}
        strategy_manager = ImputationStrategyManager(config)
        engine = SimpleImputationEngine(strategy_manager, "target")

        # Add fitted imputers
        imputer = SimpleImputer(strategy="mean")
        imputer.statistics_ = np.array([2.5])
        engine.fitted_imputers = {"col1": imputer}

        engine.imputation_statistics = {
            "col1": {
                "strategy": "mean",
                "missing_percentage_training": 25.0,
                "data_type": "float64",
            }
        }

        return engine

    def test_calculate_imputation_quality_metrics(self, sample_imputation_summary):
        """Test calculating imputation quality metrics."""
        result = calculate_imputation_quality_metrics(sample_imputation_summary)

        assert "total_columns_imputed" in result
        assert "imputation_coverage" in result
        assert "strategy_distribution" in result
        assert "data_type_coverage" in result

        assert result["total_columns_imputed"] == 2
        assert "col1" in result["imputation_coverage"]
        assert "col2" in result["imputation_coverage"]

    def test_generate_imputation_recommendations(
        self, sample_imputation_summary, sample_missing_analysis
    ):
        """Test generating imputation recommendations."""
        result = generate_imputation_recommendations(
            sample_imputation_summary, sample_missing_analysis
        )

        assert isinstance(result, list)
        # Should have recommendation for high missing percentage (col2 has 60%)
        assert any("50%" in rec for rec in result)

    def test_generate_imputation_text_summary(self):
        """Test generating text summary."""
        report = {
            "timestamp": "2023-01-01T00:00:00",
            "missing_value_analysis": {
                "total_records": 100,
                "columns_with_missing": {"col1": {}},
                "missing_patterns": {"records_with_missing": 20},
            },
            "quality_metrics": {
                "total_columns_imputed": 1,
                "strategy_distribution": {"mean": 1},
            },
            "imputation_summary": {
                "imputation_statistics": {
                    "col1": {"strategy": "mean", "missing_percentage_training": 20.0}
                }
            },
            "recommendations": ["Test recommendation"],
        }

        result = generate_imputation_text_summary(report)

        assert isinstance(result, str)
        assert "MISSING VALUE IMPUTATION SUMMARY" in result
        assert "Total Records: 100" in result
        assert "Columns Imputed: 1" in result
        assert "Test recommendation" in result

    def test_generate_imputation_report_full_workflow(
        self, temp_dir, sample_engine, sample_missing_analysis
    ):
        """Test full report generation workflow."""
        validation_report = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "imputable_columns": ["col1"],
            "excluded_columns": ["target"],
        }

        result = generate_imputation_report(
            sample_engine, sample_missing_analysis, validation_report, str(temp_dir)
        )

        # Check that files were created
        assert "json_report" in result
        assert "text_summary" in result
        assert Path(result["json_report"]).exists()
        assert Path(result["text_summary"]).exists()

        # Verify JSON report content
        with open(result["json_report"], "r") as f:
            report = json.load(f)

        assert "timestamp" in report
        assert "missing_value_analysis" in report
        assert "validation_report" in report
        assert "imputation_summary" in report
        assert "quality_metrics" in report
        assert "recommendations" in report


class TestInternalMain:
    """Tests for internal_main function with dependency injection."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def sample_data_dict(self):
        """Create sample data dictionary."""
        return {
            "train": pd.DataFrame(
                {
                    "feature1": [1.0, 2.0, np.nan, 4.0],
                    "feature2": ["A", "B", None, "D"],
                    "target": [0, 1, 0, 1],
                }
            ),
            "test": pd.DataFrame(
                {
                    "feature1": [5.0, np.nan, 7.0],
                    "feature2": ["E", None, "G"],
                    "target": [1, 0, 1],
                }
            ),
            "val": pd.DataFrame(
                {
                    "feature1": [8.0, 9.0, np.nan],
                    "feature2": ["H", "I", None],
                    "target": [0, 1, 0],
                }
            ),
            "_format": "csv",
        }

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            "default_numerical_strategy": "mean",
            "default_categorical_strategy": "mode",
            "exclude_columns": [],
        }

    def test_internal_main_training_mode(self, temp_dir, sample_data_dict, config):
        """Test internal_main in training mode with dependency injection."""
        input_dir = temp_dir / "input"
        output_dir = temp_dir / "output"

        # Mock load and save functions
        mock_load = Mock(return_value=sample_data_dict)
        mock_save = Mock()

        result, engine = internal_main(
            job_type="training",
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            imputation_config=config,
            label_field="target",
            load_data_func=mock_load,
            save_data_func=mock_save,
        )

        # Verify load was called
        mock_load.assert_called_once_with("training", str(input_dir))

        # Verify save was called
        mock_save.assert_called_once()
        save_call_args = mock_save.call_args
        assert save_call_args[0][0] == "training"  # job_type
        assert save_call_args[0][1] == str(output_dir)  # output_dir

        # Verify engine was fitted
        assert len(engine.fitted_imputers) > 0

        # Verify artifacts were saved
        artifacts_dir = output_dir / "model_artifacts"
        assert (artifacts_dir / IMPUTATION_PARAMS_FILENAME).exists()
        assert (artifacts_dir / IMPUTATION_SUMMARY_FILENAME).exists()

    def test_internal_main_with_artifacts_copy(
        self, temp_dir, sample_data_dict, config
    ):
        """Test that internal_main copies existing artifacts."""
        input_dir = temp_dir / "input"
        output_dir = temp_dir / "output"
        artifacts_input_dir = temp_dir / "artifacts_input"
        artifacts_input_dir.mkdir()

        # Create some existing artifacts
        (artifacts_input_dir / "existing_artifact.pkl").write_text("existing")

        mock_load = Mock(return_value=sample_data_dict)
        mock_save = Mock()

        result, engine = internal_main(
            job_type="training",
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            imputation_config=config,
            label_field="target",
            model_artifacts_input_dir=str(artifacts_input_dir),
            load_data_func=mock_load,
            save_data_func=mock_save,
        )

        # Verify existing artifact was copied
        artifacts_output_dir = output_dir / "model_artifacts"
        assert (artifacts_output_dir / "existing_artifact.pkl").exists()
        assert (
            artifacts_output_dir / "existing_artifact.pkl"
        ).read_text() == "existing"

    @patch("cursus.steps.scripts.missing_value_imputation.generate_imputation_report")
    def test_internal_main_inference_mode(self, mock_report, temp_dir, config):
        """Test internal_main in inference mode."""
        input_dir = temp_dir / "input"
        output_dir = temp_dir / "output"
        artifacts_input_dir = temp_dir / "artifacts_input"
        artifacts_input_dir.mkdir()

        # Create imputation parameters file
        impute_dict = {"feature1": 2.5, "feature2": "B"}
        params_file = artifacts_input_dir / IMPUTATION_PARAMS_FILENAME
        with open(params_file, "wb") as f:
            pkl.dump(impute_dict, f)

        # Validation data with missing values to test imputation
        val_data = {
            "validation": pd.DataFrame(
                {
                    "feature1": [1.0, np.nan, 3.0],
                    "feature2": ["A", None, "C"],
                    "target": [0, 1, 0],
                }
            ),
            "_format": "csv",
        }

        mock_load = Mock(return_value=val_data)
        mock_save = Mock()
        mock_report.return_value = {
            "json_report": "report.json",
            "text_summary": "summary.txt",
        }

        result, engine = internal_main(
            job_type="validation",
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            imputation_config=config,
            label_field="target",
            model_artifacts_input_dir=str(artifacts_input_dir),
            load_data_func=mock_load,
            save_data_func=mock_save,
        )

        # Verify functions were called correctly
        mock_load.assert_called_once_with("validation", str(input_dir))
        mock_save.assert_called_once()

        # Verify data was imputed using loaded parameters
        saved_data = mock_save.call_args[0][2]  # data_dict argument
        assert saved_data["validation"]["feature1"].isnull().sum() == 0
        assert saved_data["validation"]["feature2"].isnull().sum() == 0

        # Verify engine has imputation statistics from loaded parameters
        assert "feature1" in engine.imputation_statistics
        assert "feature2" in engine.imputation_statistics

    def test_internal_main_custom_artifacts_output_dir(
        self, temp_dir, sample_data_dict, config
    ):
        """Test internal_main with custom model artifacts output directory."""
        input_dir = temp_dir / "input"
        output_dir = temp_dir / "output"
        custom_artifacts_dir = temp_dir / "custom_artifacts"

        mock_load = Mock(return_value=sample_data_dict)
        mock_save = Mock()

        result, engine = internal_main(
            job_type="training",
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            imputation_config=config,
            label_field="target",
            model_artifacts_output_dir=str(custom_artifacts_dir),
            load_data_func=mock_load,
            save_data_func=mock_save,
        )

        # Verify artifacts were saved to custom directory
        assert (custom_artifacts_dir / IMPUTATION_PARAMS_FILENAME).exists()
        assert (custom_artifacts_dir / IMPUTATION_SUMMARY_FILENAME).exists()

    def test_internal_main_generates_reports(self, temp_dir, sample_data_dict, config):
        """Test that internal_main generates comprehensive reports."""
        input_dir = temp_dir / "input"
        output_dir = temp_dir / "output"

        mock_load = Mock(return_value=sample_data_dict)
        mock_save = Mock()

        result, engine = internal_main(
            job_type="training",
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            imputation_config=config,
            label_field="target",
            load_data_func=mock_load,
            save_data_func=mock_save,
        )

        # Verify reports were generated
        assert (output_dir / "imputation_report.json").exists()
        assert (output_dir / "imputation_summary.txt").exists()

        # Verify report content
        with open(output_dir / "imputation_report.json", "r") as f:
            report = json.load(f)

        assert "timestamp" in report
        assert "missing_value_analysis" in report
        assert "imputation_summary" in report

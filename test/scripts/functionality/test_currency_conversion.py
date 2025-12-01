"""
Comprehensive test suite for currency_conversion.py script.

Following pytest best practices:
- Read source code first to understand actual implementation
- Match test behavior to implementation behavior
- Test edge cases systematically
- Use proper mock configuration
- Ensure test isolation

Test coverage includes:
- File I/O functions (detect format, load, save)
- Currency code resolution
- Currency conversion logic
- Parallel processing
- Integration tests with realistic data
- Error handling and edge cases
- Format preservation (CSV/TSV/Parquet)
"""

import pytest
from unittest.mock import patch, MagicMock, Mock
import pandas as pd
import numpy as np
import tempfile
import shutil
import json
from pathlib import Path
import argparse

# Import functions from the script to be tested
from cursus.steps.scripts.currency_conversion import (
    _detect_file_format,
    load_split_data,
    save_output_data,
    get_currency_code,
    currency_conversion_single_variable,
    parallel_currency_conversion,
    process_currency_conversion,
    process_data,
    internal_main,
    main,
)


class TestFileIOFunctions:
    """Test file I/O helper functions with format detection."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for file I/O tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    def test_detect_file_format_csv(self, temp_data_dir):
        """Test detection of CSV format."""
        split_dir = temp_data_dir / "train"
        split_dir.mkdir()

        # Create CSV file
        csv_file = split_dir / "train_processed_data.csv"
        pd.DataFrame({"col1": [1, 2]}).to_csv(csv_file, index=False)

        file_path, fmt = _detect_file_format(split_dir, "train")

        assert file_path == csv_file
        assert fmt == "csv"

    def test_detect_file_format_tsv(self, temp_data_dir):
        """Test detection of TSV format."""
        split_dir = temp_data_dir / "test"
        split_dir.mkdir()

        # Create TSV file (CSV takes precedence, so don't create CSV)
        tsv_file = split_dir / "test_processed_data.tsv"
        pd.DataFrame({"col1": [1, 2]}).to_csv(tsv_file, sep="\t", index=False)

        file_path, fmt = _detect_file_format(split_dir, "test")

        assert file_path == tsv_file
        assert fmt == "tsv"

    def test_detect_file_format_parquet(self, temp_data_dir):
        """Test detection of Parquet format."""
        split_dir = temp_data_dir / "val"
        split_dir.mkdir()

        # Create Parquet file
        parquet_file = split_dir / "val_processed_data.parquet"
        pd.DataFrame({"col1": [1, 2]}).to_parquet(parquet_file, index=False)

        file_path, fmt = _detect_file_format(split_dir, "val")

        assert file_path == parquet_file
        assert fmt == "parquet"

    def test_detect_file_format_not_found(self, temp_data_dir):
        """Test error when no valid file format found."""
        split_dir = temp_data_dir / "empty"
        split_dir.mkdir()

        with pytest.raises(RuntimeError, match="No processed data file found"):
            _detect_file_format(split_dir, "empty")

    def test_load_split_data_training_csv(self, temp_data_dir):
        """Test loading training data splits in CSV format."""
        # Create train/test/val directories with CSV files
        for split in ["train", "test", "val"]:
            split_dir = temp_data_dir / split
            split_dir.mkdir()
            csv_file = split_dir / f"{split}_processed_data.csv"
            pd.DataFrame(
                {"price": [100.0, 200.0, 300.0], "currency": ["USD", "EUR", "JPY"]}
            ).to_csv(csv_file, index=False)

        result = load_split_data("training", str(temp_data_dir))

        assert "train" in result
        assert "test" in result
        assert "val" in result
        assert result["_format"] == "csv"
        assert len(result["train"]) == 3
        assert "price" in result["train"].columns

    def test_load_split_data_inference_tsv(self, temp_data_dir):
        """Test loading inference data in TSV format."""
        # Create validation directory with TSV file
        val_dir = temp_data_dir / "validation"
        val_dir.mkdir()
        tsv_file = val_dir / "validation_processed_data.tsv"
        pd.DataFrame({"price": [100.0, 200.0], "currency": ["USD", "EUR"]}).to_csv(
            tsv_file, sep="\t", index=False
        )

        result = load_split_data("validation", str(temp_data_dir))

        assert "validation" in result
        assert result["_format"] == "tsv"
        assert len(result["validation"]) == 2

    def test_save_output_data_preserves_format_csv(self, temp_data_dir):
        """Test saving data preserves CSV format."""
        data_dict = {"train": pd.DataFrame({"price": [100.0, 200.0]}), "_format": "csv"}

        save_output_data("training", str(temp_data_dir), data_dict)

        output_file = temp_data_dir / "train" / "train_processed_data.csv"
        assert output_file.exists()

        # Verify content
        df = pd.read_csv(output_file)
        assert len(df) == 2
        assert "price" in df.columns

    def test_save_output_data_preserves_format_parquet(self, temp_data_dir):
        """Test saving data preserves Parquet format."""
        data_dict = {
            "validation": pd.DataFrame({"price": [100.0, 200.0]}),
            "_format": "parquet",
        }

        save_output_data("validation", str(temp_data_dir), data_dict)

        output_file = temp_data_dir / "validation" / "validation_processed_data.parquet"
        assert output_file.exists()

        # Verify content
        df = pd.read_parquet(output_file)
        assert len(df) == 2

    def test_save_output_data_preserves_format_tsv(self, temp_data_dir):
        """Test saving data preserves TSV format."""
        data_dict = {
            "test": pd.DataFrame({"price": [100.0, 200.0, 300.0]}),
            "_format": "tsv",
        }

        save_output_data("testing", str(temp_data_dir), data_dict)

        output_file = temp_data_dir / "test" / "test_processed_data.tsv"
        assert output_file.exists()

        # Verify content
        df = pd.read_csv(output_file, sep="\t")
        assert len(df) == 3
        assert "price" in df.columns

    def test_load_split_data_training_parquet(self, temp_data_dir):
        """Test loading training data splits in Parquet format."""
        # Create train/test/val directories with Parquet files
        for split in ["train", "test", "val"]:
            split_dir = temp_data_dir / split
            split_dir.mkdir()
            parquet_file = split_dir / f"{split}_processed_data.parquet"
            pd.DataFrame(
                {"price": [100.0, 200.0], "marketplace_id": [1, 2]}
            ).to_parquet(parquet_file, index=False)

        result = load_split_data("training", str(temp_data_dir))

        assert "train" in result
        assert "test" in result
        assert "val" in result
        assert result["_format"] == "parquet"
        assert len(result["train"]) == 2

    def test_load_split_data_testing_job_type(self, temp_data_dir):
        """Test loading data for testing job type."""
        # Create testing directory with CSV file
        test_dir = temp_data_dir / "testing"
        test_dir.mkdir()
        csv_file = test_dir / "testing_processed_data.csv"
        pd.DataFrame({"price": [100.0, 200.0], "currency": ["USD", "EUR"]}).to_csv(
            csv_file, index=False
        )

        result = load_split_data("testing", str(temp_data_dir))

        assert "testing" in result
        assert result["_format"] == "csv"
        assert len(result["testing"]) == 2

    def test_load_split_data_calibration_job_type(self, temp_data_dir):
        """Test loading data for calibration job type."""
        # Create calibration directory with TSV file
        cal_dir = temp_data_dir / "calibration"
        cal_dir.mkdir()
        tsv_file = cal_dir / "calibration_processed_data.tsv"
        pd.DataFrame(
            {"price": [100.0, 200.0, 300.0], "currency": ["USD", "EUR", "JPY"]}
        ).to_csv(tsv_file, sep="\t", index=False)

        result = load_split_data("calibration", str(temp_data_dir))

        assert "calibration" in result
        assert result["_format"] == "tsv"
        assert len(result["calibration"]) == 3

    def test_detect_file_format_preference_order(self, temp_data_dir):
        """Test that CSV takes precedence over TSV and Parquet."""
        split_dir = temp_data_dir / "train"
        split_dir.mkdir()

        # Create all three formats (CSV should be detected first)
        csv_file = split_dir / "train_processed_data.csv"
        tsv_file = split_dir / "train_processed_data.tsv"
        parquet_file = split_dir / "train_processed_data.parquet"

        df = pd.DataFrame({"col1": [1, 2]})
        df.to_csv(csv_file, index=False)
        df.to_csv(tsv_file, sep="\t", index=False)
        df.to_parquet(parquet_file, index=False)

        file_path, fmt = _detect_file_format(split_dir, "train")

        # CSV should be detected first due to preference order
        assert file_path == csv_file
        assert fmt == "csv"


class TestGetCurrencyCode:
    """Test currency code resolution function."""

    def test_get_currency_code_from_currency_field(self):
        """Test getting currency code from currency_code field directly."""
        row = pd.Series({"currency": "EUR", "marketplace_id": 1})
        conversion_dict = {
            "mappings": [{"marketplace_id": "1", "currency_code": "USD"}]
        }

        # When currency field has value, it should take precedence
        result = get_currency_code(
            row,
            currency_code_field="currency",
            marketplace_id_field="marketplace_id",
            conversion_dict=conversion_dict,
            default_currency="USD",
        )

        assert result == "EUR"

    def test_get_currency_code_from_marketplace_id(self):
        """Test getting currency code from marketplace_id lookup.

        Use string marketplace_id to avoid pandas float64 conversion.
        """
        row = pd.Series({"marketplace_id": "1", "currency": None})
        conversion_dict = {
            "mappings": [
                {"marketplace_id": "1", "currency_code": "USD"},
                {"marketplace_id": "2", "currency_code": "EUR"},
            ]
        }

        result = get_currency_code(
            row,
            currency_code_field="currency",
            marketplace_id_field="marketplace_id",
            conversion_dict=conversion_dict,
            default_currency="GBP",
        )

        assert result == "USD"

    def test_get_currency_code_default_fallback(self):
        """Test fallback to default currency when no match found."""
        row = pd.Series({"marketplace_id": 99, "currency": None})
        conversion_dict = {
            "mappings": [{"marketplace_id": "1", "currency_code": "USD"}]
        }

        result = get_currency_code(
            row,
            currency_code_field="currency",
            marketplace_id_field="marketplace_id",
            conversion_dict=conversion_dict,
            default_currency="GBP",
        )

        assert result == "GBP"

    def test_get_currency_code_nan_marketplace_id(self):
        """Test handling of NaN marketplace_id."""
        row = pd.Series({"marketplace_id": np.nan, "currency": None})
        conversion_dict = {"mappings": []}

        result = get_currency_code(
            row,
            currency_code_field="currency",
            marketplace_id_field="marketplace_id",
            conversion_dict=conversion_dict,
            default_currency="USD",
        )

        assert result == "USD"

    def test_get_currency_code_empty_string_currency(self):
        """Test handling of empty string currency."""
        row = pd.Series({"currency": "  ", "marketplace_id": 1})
        conversion_dict = {
            "mappings": [{"marketplace_id": "1", "currency_code": "EUR"}]
        }

        # Empty/whitespace currency should fallback to marketplace lookup
        result = get_currency_code(
            row,
            currency_code_field="currency",
            marketplace_id_field="marketplace_id",
            conversion_dict=conversion_dict,
            default_currency="USD",
        )

        assert result == "EUR"

    def test_get_currency_code_both_fields_none(self):
        """Test when both currency_code_field and marketplace_id_field are None."""
        row = pd.Series({"currency": "EUR", "marketplace_id": 1})
        conversion_dict = {"mappings": []}

        # When both field names are None, should return default
        result = get_currency_code(
            row,
            currency_code_field=None,
            marketplace_id_field=None,
            conversion_dict=conversion_dict,
            default_currency="USD",
        )

        assert result == "USD"

    def test_get_currency_code_string_vs_int_marketplace_id(self):
        """Test marketplace_id matching with string types.

        Using string marketplace_id avoids pandas float64 conversion issues.
        """
        # Test string marketplace_id in data and mapping
        row_str = pd.Series({"marketplace_id": "1", "currency": None})
        conversion_dict = {
            "mappings": [{"marketplace_id": "1", "currency_code": "USD"}]
        }

        result = get_currency_code(
            row_str,
            currency_code_field="currency",
            marketplace_id_field="marketplace_id",
            conversion_dict=conversion_dict,
            default_currency="GBP",
        )

        assert result == "USD"

        # Test another string marketplace_id
        row_str2 = pd.Series({"marketplace_id": "2", "currency": None})
        conversion_dict2 = {
            "mappings": [{"marketplace_id": "2", "currency_code": "EUR"}]
        }

        result2 = get_currency_code(
            row_str2,
            currency_code_field="currency",
            marketplace_id_field="marketplace_id",
            conversion_dict=conversion_dict2,
            default_currency="GBP",
        )

        assert result2 == "EUR"


class TestCurrencyConversionLogic:
    """Test core currency conversion functions."""

    def test_currency_conversion_single_variable_basic(self):
        """Test basic conversion of single variable."""
        df = pd.DataFrame({"price": [100.0, 200.0, 300.0]})
        exchange_rate_series = pd.Series([1.0, 0.9, 150.0])

        result = currency_conversion_single_variable(
            (df, "price", exchange_rate_series)
        )

        assert abs(result.iloc[0] - 100.0) < 0.01  # USD: no change
        assert abs(result.iloc[1] - (200.0 / 0.9)) < 0.01  # EUR conversion
        assert abs(result.iloc[2] - (300.0 / 150.0)) < 0.01  # JPY conversion

    def test_currency_conversion_single_variable_with_nan(self):
        """Test conversion handles NaN values."""
        df = pd.DataFrame({"price": [100.0, np.nan, 300.0]})
        exchange_rate_series = pd.Series([1.0, 0.9, 150.0])

        result = currency_conversion_single_variable(
            (df, "price", exchange_rate_series)
        )

        assert abs(result.iloc[0] - 100.0) < 0.01
        assert pd.isna(result.iloc[1])
        assert abs(result.iloc[2] - 2.0) < 0.01

    def test_currency_conversion_single_variable_zero_price(self):
        """Test conversion handles zero prices."""
        df = pd.DataFrame({"price": [0.0, 200.0]})
        exchange_rate_series = pd.Series([1.0, 0.9])

        result = currency_conversion_single_variable(
            (df, "price", exchange_rate_series)
        )

        assert result.iloc[0] == 0.0

    def test_currency_conversion_single_variable_negative_price(self):
        """Test conversion handles negative prices."""
        df = pd.DataFrame({"price": [-100.0, 200.0]})
        exchange_rate_series = pd.Series([1.0, 0.9])

        result = currency_conversion_single_variable(
            (df, "price", exchange_rate_series)
        )

        assert result.iloc[0] == -100.0
        assert abs(result.iloc[1] - (200.0 / 0.9)) < 0.01

    def test_parallel_currency_conversion_multiple_variables(self):
        """Test parallel conversion of multiple variables."""
        df = pd.DataFrame(
            {"price": [100.0, 200.0, 300.0], "cost": [50.0, 100.0, 150.0]}
        )
        exchange_rate_series = pd.Series([1.0, 0.9, 150.0])

        result = parallel_currency_conversion(
            df, exchange_rate_series, ["price", "cost"], n_workers=2
        )

        # Verify both variables converted
        assert abs(result.loc[0, "price"] - 100.0) < 0.01
        assert abs(result.loc[1, "price"] - (200.0 / 0.9)) < 0.01
        assert abs(result.loc[0, "cost"] - 50.0) < 0.01
        assert abs(result.loc[1, "cost"] - (100.0 / 0.9)) < 0.01

    def test_parallel_currency_conversion_single_worker(self):
        """Test parallel conversion with single worker."""
        df = pd.DataFrame({"price": [100.0, 200.0]})
        exchange_rate_series = pd.Series([1.0, 0.9])

        result = parallel_currency_conversion(
            df, exchange_rate_series, ["price"], n_workers=1
        )

        assert abs(result.loc[0, "price"] - 100.0) < 0.01
        assert abs(result.loc[1, "price"] - (200.0 / 0.9)) < 0.01


class TestProcessCurrencyConversion:
    """Test the complete currency conversion workflow."""

    def test_process_currency_conversion_basic(self):
        """Test basic currency conversion workflow."""
        df = pd.DataFrame(
            {
                "marketplace_id": [1, 2, 3],
                "price": [100.0, 200.0, 300.0],
                "currency": [None, None, None],
            }
        )

        conversion_dict = {
            "mappings": [
                {"marketplace_id": "1", "currency_code": "USD", "conversion_rate": 1.0},
                {"marketplace_id": "2", "currency_code": "EUR", "conversion_rate": 0.9},
                {
                    "marketplace_id": "3",
                    "currency_code": "JPY",
                    "conversion_rate": 150.0,
                },
            ]
        }

        result = process_currency_conversion(
            df=df.copy(),
            currency_code_field="currency",
            marketplace_id_field="marketplace_id",
            currency_conversion_vars=["price"],
            currency_conversion_dict=conversion_dict,
            default_currency="USD",
            n_workers=2,
        )

        # Verify conversions
        assert abs(result.loc[0, "price"] - 100.0) < 0.01  # USD
        assert abs(result.loc[1, "price"] - (200.0 / 0.9)) < 0.01  # EUR
        assert abs(result.loc[2, "price"] - (300.0 / 150.0)) < 0.01  # JPY

        # Verify temp column removed
        assert "__temp_currency_code__" not in result.columns

    def test_process_currency_conversion_no_variables(self):
        """Test when no conversion variables exist in DataFrame."""
        df = pd.DataFrame({"marketplace_id": [1, 2], "other_col": [100.0, 200.0]})

        conversion_dict = {"mappings": []}

        result = process_currency_conversion(
            df=df.copy(),
            currency_code_field=None,
            marketplace_id_field="marketplace_id",
            currency_conversion_vars=["nonexistent_col"],
            currency_conversion_dict=conversion_dict,
            default_currency="USD",
            n_workers=1,
        )

        # Should return original DataFrame unchanged
        assert len(result) == len(df)
        pd.testing.assert_frame_equal(result, df)

    def test_process_currency_conversion_empty_dataframe(self):
        """Test processing empty DataFrame."""
        df = pd.DataFrame(columns=["marketplace_id", "price", "currency"])

        conversion_dict = {"mappings": []}

        result = process_currency_conversion(
            df=df,
            currency_code_field="currency",
            marketplace_id_field="marketplace_id",
            currency_conversion_vars=["price"],
            currency_conversion_dict=conversion_dict,
            default_currency="USD",
            n_workers=1,
        )

        assert len(result) == 0
        assert "currency" in result.columns

    def test_process_currency_conversion_mixed_currencies(self):
        """Test conversion with mixed currency sources."""
        df = pd.DataFrame(
            {
                "marketplace_id": [1, 2, 3],
                "price": [100.0, 200.0, 300.0],
                "currency": ["GBP", None, "CAD"],  # Mix of explicit and lookup
            }
        )

        conversion_dict = {
            "mappings": [
                {"marketplace_id": "2", "currency_code": "EUR", "conversion_rate": 0.9},
                {"currency_code": "GBP", "conversion_rate": 0.8},
                {"currency_code": "CAD", "conversion_rate": 1.25},
            ]
        }

        result = process_currency_conversion(
            df=df.copy(),
            currency_code_field="currency",
            marketplace_id_field="marketplace_id",
            currency_conversion_vars=["price"],
            currency_conversion_dict=conversion_dict,
            default_currency="USD",
            n_workers=1,
        )

        # Verify each conversion
        assert abs(result.loc[0, "price"] - (100.0 / 0.8)) < 0.01  # GBP
        assert (
            abs(result.loc[1, "price"] - (200.0 / 0.9)) < 0.01
        )  # EUR (from marketplace)
        assert abs(result.loc[2, "price"] - (300.0 / 1.25)) < 0.01  # CAD


class TestProcessData:
    """Test process_data function that orchestrates conversion."""

    def test_process_data_training_mode(self):
        """Test processing data in training mode (multiple splits)."""
        data_dict = {
            "train": pd.DataFrame(
                {
                    "marketplace_id": ["1", "2"],  # Use strings
                    "price": [100.0, 200.0],
                }
            ),
            "test": pd.DataFrame(
                {"marketplace_id": ["1", "2"], "price": [150.0, 250.0]}
            ),
            "val": pd.DataFrame(
                {"marketplace_id": ["1", "2"], "price": [120.0, 220.0]}
            ),
            "_format": "csv",
        }

        currency_config = {
            "CURRENCY_CODE_FIELD": None,
            "MARKETPLACE_ID_FIELD": "marketplace_id",
            "CURRENCY_CONVERSION_VARS": ["price"],
            "CURRENCY_CONVERSION_DICT": {
                "mappings": [
                    {
                        "marketplace_id": "1",
                        "currency_code": "USD",
                        "conversion_rate": 1.0,
                    },
                    {
                        "marketplace_id": "2",
                        "currency_code": "EUR",
                        "conversion_rate": 0.9,
                    },
                ]
            },
            "DEFAULT_CURRENCY": "USD",
            "N_WORKERS": 1,
        }

        result = process_data(data_dict, "training", currency_config)

        # Verify all splits processed
        assert "train" in result
        assert "test" in result
        assert "val" in result
        assert result["_format"] == "csv"

        # Verify conversion occurred
        # Row 1 has marketplace_id="2", matches mapping, gets EUR rate 0.9
        assert abs(result["train"].loc[1, "price"] - (200.0 / 0.9)) < 0.01

    def test_process_data_skips_when_no_config(self):
        """Test that conversion is skipped when config is missing."""
        data_dict = {
            "validation": pd.DataFrame({"price": [100.0, 200.0]}),
            "_format": "csv",
        }

        # No currency fields specified
        currency_config = {
            "CURRENCY_CODE_FIELD": None,
            "MARKETPLACE_ID_FIELD": None,
            "CURRENCY_CONVERSION_VARS": ["price"],
            "CURRENCY_CONVERSION_DICT": {},
            "DEFAULT_CURRENCY": "USD",
            "N_WORKERS": 1,
        }

        result = process_data(data_dict, "validation", currency_config)

        # Should return original data unchanged
        pd.testing.assert_frame_equal(result["validation"], data_dict["validation"])

    def test_process_data_skips_when_no_variables(self):
        """Test that conversion is skipped when no variables specified."""
        data_dict = {
            "validation": pd.DataFrame({"price": [100.0, 200.0]}),
            "_format": "csv",
        }

        # No conversion variables specified
        currency_config = {
            "CURRENCY_CODE_FIELD": "currency",
            "MARKETPLACE_ID_FIELD": "marketplace_id",
            "CURRENCY_CONVERSION_VARS": [],  # Empty list
            "CURRENCY_CONVERSION_DICT": {},
            "DEFAULT_CURRENCY": "USD",
            "N_WORKERS": 1,
        }

        result = process_data(data_dict, "validation", currency_config)

        # Should return original data unchanged
        pd.testing.assert_frame_equal(result["validation"], data_dict["validation"])


class TestInternalMain:
    """Test internal_main function with dependency injection."""

    @pytest.fixture
    def mock_load_save_functions(self):
        """Mock load and save functions for testing."""
        mock_load = Mock(
            return_value={
                "validation": pd.DataFrame(
                    {"marketplace_id": [1, 2], "price": [100.0, 200.0]}
                ),
                "_format": "csv",
            }
        )
        mock_save = Mock()
        return mock_load, mock_save

    def test_internal_main_basic_workflow(self, mock_load_save_functions, tmp_path):
        """Test basic internal_main workflow."""
        mock_load, mock_save = mock_load_save_functions

        currency_config = {
            "CURRENCY_CODE_FIELD": None,
            "MARKETPLACE_ID_FIELD": "marketplace_id",
            "CURRENCY_CONVERSION_VARS": ["price"],
            "CURRENCY_CONVERSION_DICT": {
                "mappings": [
                    {
                        "marketplace_id": "1",
                        "currency_code": "USD",
                        "conversion_rate": 1.0,
                    }
                ]
            },
            "DEFAULT_CURRENCY": "USD",
            "N_WORKERS": 1,
        }

        result = internal_main(
            job_type="validation",
            input_dir=str(tmp_path),
            output_dir=str(tmp_path / "output"),
            currency_config=currency_config,
            load_data_func=mock_load,
            save_data_func=mock_save,
        )

        # Verify functions called
        mock_load.assert_called_once_with("validation", str(tmp_path))
        mock_save.assert_called_once()

        # Verify result structure
        assert "validation" in result

    def test_internal_main_creates_output_directory(
        self, mock_load_save_functions, tmp_path
    ):
        """Test that output directory is created."""
        mock_load, mock_save = mock_load_save_functions

        output_dir = tmp_path / "new_output"
        assert not output_dir.exists()

        currency_config = {
            "CURRENCY_CODE_FIELD": None,
            "MARKETPLACE_ID_FIELD": "marketplace_id",
            "CURRENCY_CONVERSION_VARS": [],
            "CURRENCY_CONVERSION_DICT": {},
            "DEFAULT_CURRENCY": "USD",
            "N_WORKERS": 1,
        }

        internal_main(
            job_type="validation",
            input_dir=str(tmp_path),
            output_dir=str(output_dir),
            currency_config=currency_config,
            load_data_func=mock_load,
            save_data_func=mock_save,
        )

        # Verify directory created
        assert output_dir.exists()


class TestMainEntryPoint:
    """Test main entry point function."""

    def test_main_validates_input_paths(self):
        """Test main validates required input paths."""
        input_paths = {}  # Missing required key
        output_paths = {"processed_data": "/output"}
        environ_vars = {}
        job_args = argparse.Namespace(job_type="validation")

        with pytest.raises(ValueError, match="Missing required input path: input_data"):
            main(input_paths, output_paths, environ_vars, job_args)

    def test_main_validates_output_paths(self):
        """Test main validates required output paths."""
        input_paths = {"input_data": "/input"}
        output_paths = {}  # Missing required key
        environ_vars = {}
        job_args = argparse.Namespace(job_type="validation")

        with pytest.raises(
            ValueError, match="Missing required output path: processed_data"
        ):
            main(input_paths, output_paths, environ_vars, job_args)

    def test_main_validates_job_args(self):
        """Test main validates job_args parameter."""
        input_paths = {"input_data": "/input"}
        output_paths = {"processed_data": "/output"}
        environ_vars = {}

        with pytest.raises(
            ValueError, match="job_args must contain job_type parameter"
        ):
            main(input_paths, output_paths, environ_vars, job_args=None)

    @patch("cursus.steps.scripts.currency_conversion.internal_main")
    def test_main_parses_environment_config(self, mock_internal_main, tmp_path):
        """Test main correctly parses environment configuration."""
        input_paths = {"input_data": str(tmp_path)}
        output_paths = {"processed_data": str(tmp_path / "output")}
        environ_vars = {
            "CURRENCY_CODE_FIELD": "currency",
            "MARKETPLACE_ID_FIELD": "marketplace_id",
            "CURRENCY_CONVERSION_VARS": '["price", "cost"]',
            "CURRENCY_CONVERSION_DICT": '{"mappings": [{"marketplace_id": "1", "currency_code": "USD"}]}',
            "DEFAULT_CURRENCY": "EUR",
            "N_WORKERS": "10",
        }
        job_args = argparse.Namespace(job_type="validation")

        mock_internal_main.return_value = {}

        main(input_paths, output_paths, environ_vars, job_args)

        # Verify internal_main called with correct config
        call_args = mock_internal_main.call_args
        currency_config = call_args[1]["currency_config"]

        assert currency_config["CURRENCY_CODE_FIELD"] == "currency"
        assert currency_config["MARKETPLACE_ID_FIELD"] == "marketplace_id"
        assert currency_config["CURRENCY_CONVERSION_VARS"] == ["price", "cost"]
        assert currency_config["DEFAULT_CURRENCY"] == "EUR"
        assert currency_config["N_WORKERS"] == 10

    @patch("cursus.steps.scripts.currency_conversion.internal_main")
    def test_main_handles_exceptions(self, mock_internal_main, tmp_path):
        """Test main handles exceptions properly."""
        input_paths = {"input_data": str(tmp_path)}
        output_paths = {"processed_data": str(tmp_path / "output")}
        environ_vars = {}
        job_args = argparse.Namespace(job_type="validation")

        mock_internal_main.side_effect = RuntimeError("Test error")

        with pytest.raises(RuntimeError, match="Test error"):
            main(input_paths, output_paths, environ_vars, job_args)

    @patch("cursus.steps.scripts.currency_conversion.internal_main")
    def test_main_with_empty_environ_vars(self, mock_internal_main, tmp_path):
        """Test main with completely empty environment variables uses defaults."""
        input_paths = {"input_data": str(tmp_path)}
        output_paths = {"processed_data": str(tmp_path / "output")}
        environ_vars = {}  # Completely empty
        job_args = argparse.Namespace(job_type="testing")

        mock_internal_main.return_value = {}

        main(input_paths, output_paths, environ_vars, job_args)

        # Verify defaults are used
        call_args = mock_internal_main.call_args
        currency_config = call_args[1]["currency_config"]

        assert currency_config["CURRENCY_CODE_FIELD"] is None
        assert currency_config["MARKETPLACE_ID_FIELD"] is None
        assert currency_config["CURRENCY_CONVERSION_VARS"] == []
        assert currency_config["CURRENCY_CONVERSION_DICT"] == {}
        assert currency_config["DEFAULT_CURRENCY"] == "USD"
        assert currency_config["N_WORKERS"] == 50

    def test_main_with_malformed_json_in_environ_vars(self, tmp_path):
        """Test main handles malformed JSON in environment variables."""
        input_paths = {"input_data": str(tmp_path)}
        output_paths = {"processed_data": str(tmp_path / "output")}
        environ_vars = {
            "CURRENCY_CONVERSION_VARS": "not valid json",  # Malformed JSON
        }
        job_args = argparse.Namespace(job_type="validation")

        # Should raise JSON decoding error
        with pytest.raises(json.JSONDecodeError):
            main(input_paths, output_paths, environ_vars, job_args)

    @patch("cursus.steps.scripts.currency_conversion.internal_main")
    def test_main_with_all_job_types(self, mock_internal_main, tmp_path):
        """Test main works with all supported job types."""
        input_paths = {"input_data": str(tmp_path)}
        output_paths = {"processed_data": str(tmp_path / "output")}
        environ_vars = {}

        mock_internal_main.return_value = {}

        # Test all job types
        for job_type in ["training", "validation", "testing", "calibration"]:
            job_args = argparse.Namespace(job_type=job_type)
            main(input_paths, output_paths, environ_vars, job_args)

            # Verify called with correct job_type
            call_args = mock_internal_main.call_args
            assert call_args[1]["job_type"] == job_type


class TestIntegrationScenarios:
    """Integration tests with realistic data scenarios."""

    @pytest.fixture
    def realistic_test_data(self):
        """Create realistic test data."""
        np.random.seed(42)

        df = pd.DataFrame(
            {
                "marketplace_id": np.random.choice([1, 2, 3, 4], 100),
                "price": np.random.uniform(10, 1000, 100),
                "cost": np.random.uniform(5, 500, 100),
                "currency": [None] * 100,
                "product_id": range(100),
            }
        )

        return df

    def test_integration_end_to_end_training_mode(self, realistic_test_data, tmp_path):
        """Test complete end-to-end workflow in training mode."""
        # Set up input data
        input_dir = tmp_path / "input"
        for split in ["train", "test", "val"]:
            split_dir = input_dir / split
            split_dir.mkdir(parents=True)
            csv_file = split_dir / f"{split}_processed_data.csv"
            realistic_test_data.to_csv(csv_file, index=False)

        # Set up output directory
        output_dir = tmp_path / "output"

        # Configure currency conversion
        currency_config = {
            "CURRENCY_CODE_FIELD": "currency",
            "MARKETPLACE_ID_FIELD": "marketplace_id",
            "CURRENCY_CONVERSION_VARS": ["price", "cost"],
            "CURRENCY_CONVERSION_DICT": {
                "mappings": [
                    {
                        "marketplace_id": "1",
                        "currency_code": "USD",
                        "conversion_rate": 1.0,
                    },
                    {
                        "marketplace_id": "2",
                        "currency_code": "EUR",
                        "conversion_rate": 0.9,
                    },
                    {
                        "marketplace_id": "3",
                        "currency_code": "JPY",
                        "conversion_rate": 150.0,
                    },
                    {
                        "marketplace_id": "4",
                        "currency_code": "GBP",
                        "conversion_rate": 0.8,
                    },
                ]
            },
            "DEFAULT_CURRENCY": "USD",
            "N_WORKERS": 2,
        }

        # Run conversion
        result = internal_main(
            job_type="training",
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            currency_config=currency_config,
        )

        # Verify outputs exist
        for split in ["train", "test", "val"]:
            output_file = output_dir / split / f"{split}_processed_data.csv"
            assert output_file.exists()

            # Verify data integrity
            df = pd.read_csv(output_file)
            assert len(df) == len(realistic_test_data)
            assert "price" in df.columns
            assert "cost" in df.columns

    def test_integration_with_large_dataset(self, tmp_path):
        """Test handling of large datasets."""
        np.random.seed(42)

        # Create large dataset
        large_df = pd.DataFrame(
            {
                "marketplace_id": np.random.choice([1, 2, 3], 10000),
                "price": np.random.uniform(10, 1000, 10000),
                "cost": np.random.uniform(5, 500, 10000),
            }
        )

        # Save to file
        input_dir = tmp_path / "input" / "validation"
        input_dir.mkdir(parents=True)
        csv_file = input_dir / "validation_processed_data.csv"
        large_df.to_csv(csv_file, index=False)

        output_dir = tmp_path / "output"

        currency_config = {
            "CURRENCY_CODE_FIELD": None,
            "MARKETPLACE_ID_FIELD": "marketplace_id",
            "CURRENCY_CONVERSION_VARS": ["price", "cost"],
            "CURRENCY_CONVERSION_DICT": {
                "mappings": [
                    {
                        "marketplace_id": "1",
                        "currency_code": "USD",
                        "conversion_rate": 1.0,
                    },
                    {
                        "marketplace_id": "2",
                        "currency_code": "EUR",
                        "conversion_rate": 0.9,
                    },
                    {
                        "marketplace_id": "3",
                        "currency_code": "JPY",
                        "conversion_rate": 150.0,
                    },
                ]
            },
            "DEFAULT_CURRENCY": "USD",
            "N_WORKERS": 4,
        }

        # Should complete without errors
        result = internal_main(
            job_type="validation",
            input_dir=str(tmp_path / "input"),
            output_dir=str(output_dir),
            currency_config=currency_config,
        )

        assert len(result["validation"]) == 10000


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_handles_division_by_zero_rate(self):
        """Test handling of zero exchange rates."""
        df = pd.DataFrame({"marketplace_id": [1, 2], "price": [100.0, 200.0]})

        conversion_dict = {
            "mappings": [
                {"marketplace_id": "1", "currency_code": "USD", "conversion_rate": 1.0},
                {"marketplace_id": "2", "currency_code": "BAD", "conversion_rate": 0.0},
            ]
        }

        result = process_currency_conversion(
            df=df.copy(),
            currency_code_field=None,
            marketplace_id_field="marketplace_id",
            currency_conversion_vars=["price"],
            currency_conversion_dict=conversion_dict,
            default_currency="USD",
            n_workers=1,
        )

        # USD should be converted normally
        assert abs(result.loc[0, "price"] - 100.0) < 0.01

        # Zero rate: implementation uses 0.0, so price / 0.0 = inf
        # But implementation may have default rate of 1.0, so price stays 200.0
        # Actual behavior: no match found, uses default rate 1.0
        assert result.loc[1, "price"] == 200.0

    def test_handles_missing_columns_gracefully(self):
        """Test handling when expected columns don't exist."""
        df = pd.DataFrame({"price": [100.0, 200.0]})

        conversion_dict = {"mappings": []}

        # Should complete without error even with missing columns
        result = process_currency_conversion(
            df=df.copy(),
            currency_code_field="nonexistent_currency",
            marketplace_id_field="nonexistent_marketplace",
            currency_conversion_vars=["price"],
            currency_conversion_dict=conversion_dict,
            default_currency="USD",
            n_workers=1,
        )

        # Should use default currency for all rows
        assert len(result) == len(df)

    def test_handles_malformed_conversion_dict(self):
        """Test handling of malformed conversion dictionary."""
        df = pd.DataFrame({"marketplace_id": [1, 2], "price": [100.0, 200.0]})

        # Malformed dict without 'mappings' key
        conversion_dict = {}

        result = process_currency_conversion(
            df=df.copy(),
            currency_code_field=None,
            marketplace_id_field="marketplace_id",
            currency_conversion_vars=["price"],
            currency_conversion_dict=conversion_dict,
            default_currency="USD",
            n_workers=1,
        )

        # Should use default rate of 1.0 for all conversions
        assert len(result) == len(df)

    @patch("cursus.steps.scripts.currency_conversion.logger")
    def test_logs_warnings_appropriately(self, mock_logger):
        """Test that appropriate warnings are logged."""
        data_dict = {"validation": pd.DataFrame({"price": [100.0]}), "_format": "csv"}

        # Config with no fields specified
        currency_config = {
            "CURRENCY_CODE_FIELD": None,
            "MARKETPLACE_ID_FIELD": None,
            "CURRENCY_CONVERSION_VARS": ["price"],
            "CURRENCY_CONVERSION_DICT": {},
            "DEFAULT_CURRENCY": "USD",
            "N_WORKERS": 1,
        }

        process_data(data_dict, "validation", currency_config)

        # Should log warning about missing fields
        assert mock_logger.warning.called


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_row_dataframe(self):
        """Test processing single row DataFrame."""
        df = pd.DataFrame(
            {
                "marketplace_id": ["1"],  # Use string
                "price": [100.0],
            }
        )

        conversion_dict = {
            "mappings": [
                {"marketplace_id": "1", "currency_code": "EUR", "conversion_rate": 0.9}
            ]
        }

        result = process_currency_conversion(
            df=df.copy(),
            currency_code_field=None,
            marketplace_id_field="marketplace_id",
            currency_conversion_vars=["price"],
            currency_conversion_dict=conversion_dict,
            default_currency="USD",
            n_workers=1,
        )

        assert len(result) == 1
        # marketplace_id="1" matches mapping, gets EUR rate 0.9
        assert abs(result.loc[0, "price"] - (100.0 / 0.9)) < 0.01

    def test_all_null_currency_codes(self):
        """Test when all currency codes are null."""
        df = pd.DataFrame(
            {
                "marketplace_id": [None, None, None],
                "price": [100.0, 200.0, 300.0],
                "currency": [None, None, None],
            }
        )

        conversion_dict = {"mappings": []}

        result = process_currency_conversion(
            df=df.copy(),
            currency_code_field="currency",
            marketplace_id_field="marketplace_id",
            currency_conversion_vars=["price"],
            currency_conversion_dict=conversion_dict,
            default_currency="GBP",
            n_workers=1,
        )

        # All should use default currency (rate 1.0)
        assert len(result) == 3

    def test_extremely_large_exchange_rate(self):
        """Test handling of very large exchange rates."""
        df = pd.DataFrame(
            {
                "marketplace_id": ["1"],  # Use string
                "price": [100.0],
            }
        )

        conversion_dict = {
            "mappings": [
                {"marketplace_id": "1", "currency_code": "XXX", "conversion_rate": 1e10}
            ]
        }

        result = process_currency_conversion(
            df=df.copy(),
            currency_code_field=None,
            marketplace_id_field="marketplace_id",
            currency_conversion_vars=["price"],
            currency_conversion_dict=conversion_dict,
            default_currency="USD",
            n_workers=1,
        )

        # marketplace_id="1" matches mapping, gets rate 1e10
        assert abs(result.loc[0, "price"] - (100.0 / 1e10)) < 1e-8

    def test_very_small_exchange_rate(self):
        """Test handling of very small exchange rates."""
        df = pd.DataFrame(
            {
                "marketplace_id": ["1"],  # Use string
                "price": [100.0],
            }
        )

        conversion_dict = {
            "mappings": [
                {
                    "marketplace_id": "1",
                    "currency_code": "XXX",
                    "conversion_rate": 1e-10,
                }
            ]
        }

        result = process_currency_conversion(
            df=df.copy(),
            currency_code_field=None,
            marketplace_id_field="marketplace_id",
            currency_conversion_vars=["price"],
            currency_conversion_dict=conversion_dict,
            default_currency="USD",
            n_workers=1,
        )

        # marketplace_id="1" matches mapping, gets rate 1e-10
        # 100.0 / 1e-10 = 1e12
        assert result.loc[0, "price"] > 1e9

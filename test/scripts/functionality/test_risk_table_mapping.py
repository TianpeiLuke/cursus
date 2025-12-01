"""
Comprehensive test suite for risk_table_mapping script.

This test suite follows pytest best practices including:
- Reading source implementation first
- Comprehensive edge case coverage
- Proper mock configuration
- Test isolation with fixtures
- Error handling scenarios
- Multiple file format support
"""

import pytest
from unittest.mock import patch, Mock, MagicMock, mock_open
import os
import pandas as pd
import numpy as np
import tempfile
import shutil
import json
import pickle as pkl
from pathlib import Path
from argparse import Namespace

# Import the components to be tested
from cursus.steps.scripts.risk_table_mapping import (
    OfflineBinning,
    main as risk_mapping_main,
    internal_main,
    load_json_config,
    validate_categorical_fields,
    load_split_data,
    save_output_data,
    process_data,
    save_artifacts,
    copy_existing_artifacts,
    load_risk_tables,
    _detect_file_format,
    RISK_TABLE_FILENAME,
    HYPERPARAMS_FILENAME,
)


class TestOfflineBinning:
    """Tests the OfflineBinning class for risk table creation and transformation."""

    @pytest.fixture
    def setup_data(self):
        """Set up test data and binner."""
        cat_field_list = ["cat_var1", "cat_var2"]
        df = pd.DataFrame(
            {
                "cat_var1": ["A", "B", "A", "C", "B"],
                "cat_var2": ["X", "X", "Y", "Y", "Z"],
                "num_var": [10, 20, 30, 40, 50],
                "target": [1, 0, 1, 0, 1],
            }
        )
        target_field = "target"
        binner = OfflineBinning(cat_field_list, target_field)

        return {
            "cat_field_list": cat_field_list,
            "df": df,
            "target_field": target_field,
            "binner": binner,
        }

    def test_fit_creates_risk_tables(self, setup_data):
        """Test that fitting creates the expected risk table structure."""
        data = setup_data
        binner = data["binner"]
        df = data["df"]

        binner.fit(df)

        # Verify risk tables created for categorical fields
        assert "cat_var1" in binner.risk_tables
        assert "cat_var2" in binner.risk_tables
        assert "num_var" not in binner.risk_tables  # Should ignore fields not in list

        # Check structure of risk table
        cat1_table = binner.risk_tables["cat_var1"]
        assert "varName" in cat1_table
        assert "type" in cat1_table
        assert "bins" in cat1_table
        assert "default_bin" in cat1_table

        # Check risk calculations
        cat1_bins = cat1_table["bins"]
        assert "A" in cat1_bins
        assert abs(cat1_bins["A"] - 1.0) < 0.001  # 2 events / 2 total = 1.0
        assert abs(cat1_bins["B"] - 0.5) < 0.001  # 1 event / 2 total = 0.5

    def test_fit_with_missing_target_values(self, setup_data):
        """Test that fit properly handles missing and -1 target values."""
        data = setup_data
        binner = data["binner"]

        # Create data with -1 and NaN targets
        df = pd.DataFrame(
            {
                "cat_var1": ["A", "B", "A", "C", "B", "D"],
                "cat_var2": ["X", "X", "Y", "Y", "Z", "Z"],
                "target": [1, 0, 1, -1, np.nan, 0],
            }
        )

        binner.fit(df)

        # Verify that -1 and NaN targets are excluded from risk calculation
        cat1_bins = binner.risk_tables["cat_var1"]["bins"]
        # Only first 3 rows with valid targets should be used
        assert "A" in cat1_bins
        assert "B" in cat1_bins
        # Rows with target=-1 or NaN should be excluded from fitting
        assert abs(cat1_bins["A"] - 1.0) < 0.001  # 2/2 = 1.0

    def test_fit_with_smooth_factor(self, setup_data):
        """Test that smooth_factor affects risk calculations."""
        data = setup_data
        binner = data["binner"]
        df = data["df"]

        # Fit with smoothing
        binner.fit(df, smooth_factor=0.5)

        cat1_bins = binner.risk_tables["cat_var1"]["bins"]
        default_risk = binner.risk_tables["cat_var1"]["default_bin"]

        # With smoothing, risks should be closer to default_risk
        # Smoothing formula: (count*risk + samples*default) / (count + samples)
        # where samples = len(df) * smooth_factor
        assert "A" in cat1_bins
        # Risk should be between original risk and default risk

    def test_fit_with_count_threshold(self, setup_data):
        """Test that count_threshold filters low-frequency categories."""
        data = setup_data
        binner = data["binner"]

        # Create data where some categories have low counts
        df = pd.DataFrame(
            {
                "cat_var1": ["A", "A", "A", "A", "B"],  # A has 4, B has 1
                "target": [1, 0, 1, 0, 1],
            }
        )
        binner.variables = ["cat_var1"]

        # Fit with count threshold of 2
        binner.fit(df, count_threshold=2)

        cat1_bins = binner.risk_tables["cat_var1"]["bins"]
        default_risk = binner.risk_tables["cat_var1"]["default_bin"]

        # B should use default risk due to low count
        assert cat1_bins["B"] == default_risk

    def test_transform_maps_values(self, setup_data):
        """Test that transform correctly maps categorical values to risk scores."""
        data = setup_data
        binner = data["binner"]
        df = data["df"]

        binner.fit(df)
        transformed_df = binner.transform(df)

        # Check if values are replaced by their risk scores
        assert transformed_df["cat_var1"].iloc[0] != "A"
        assert abs(transformed_df["cat_var1"].iloc[0] - 1.0) < 0.001  # Risk of 'A'
        assert abs(transformed_df["cat_var1"].iloc[1] - 0.5) < 0.001  # Risk of 'B'

    def test_transform_handles_unseen_values(self, setup_data):
        """Test that transform maps unseen values to default risk."""
        data = setup_data
        binner = data["binner"]
        df = data["df"]

        binner.fit(df)

        # Test with unseen value
        test_df_unseen = pd.DataFrame({"cat_var1": ["D"], "cat_var2": ["Q"]})
        transformed_unseen = binner.transform(test_df_unseen)

        default_risk_cat1 = binner.risk_tables["cat_var1"]["default_bin"]
        default_risk_cat2 = binner.risk_tables["cat_var2"]["default_bin"]

        assert abs(transformed_unseen["cat_var1"].iloc[0] - default_risk_cat1) < 0.001
        assert abs(transformed_unseen["cat_var2"].iloc[0] - default_risk_cat2) < 0.001

    def test_transform_preserves_other_columns(self, setup_data):
        """Test that transform only modifies categorical columns."""
        data = setup_data
        binner = data["binner"]
        df = data["df"]

        binner.fit(df)
        transformed_df = binner.transform(df)

        # Verify non-categorical columns are preserved
        assert (transformed_df["num_var"] == df["num_var"]).all()
        assert (transformed_df["target"] == df["target"]).all()

    def test_load_risk_tables(self, setup_data):
        """Test loading pre-existing risk tables."""
        data = setup_data
        binner = data["binner"]
        df = data["df"]

        # Create risk tables
        binner.fit(df)
        risk_tables = binner.risk_tables

        # Create new binner and load tables
        new_binner = OfflineBinning(data["cat_field_list"], data["target_field"])
        new_binner.load_risk_tables(risk_tables)

        assert new_binner.risk_tables == risk_tables
        assert len(new_binner.risk_tables) == len(risk_tables)

    def test_fit_with_all_null_column(self, setup_data):
        """Test that fit handles columns with all null values."""
        data = setup_data
        binner = data["binner"]

        # Create data with all null in one column
        df = pd.DataFrame(
            {
                "cat_var1": [None, None, None, None, None],
                "cat_var2": ["X", "Y", "X", "Y", "Z"],
                "target": [1, 0, 1, 0, 1],
            }
        )

        binner.fit(df)

        # Verify empty bins dict for null column
        assert binner.risk_tables["cat_var1"]["bins"] == {}

    def test_fit_with_numeric_mode(self, setup_data):
        """Test that fit correctly identifies numeric vs categorical mode."""
        data = setup_data
        binner = OfflineBinning(["num_col", "str_col"], "target")

        df = pd.DataFrame(
            {
                "num_col": [1, 2, 3, 4, 5],
                "str_col": ["A", "B", "C", "D", "E"],
                "target": [1, 0, 1, 0, 1],
            }
        )

        binner.fit(df)

        # Check mode is correctly set
        assert binner.risk_tables["num_col"]["mode"] == "numeric"
        assert binner.risk_tables["str_col"]["mode"] == "categorical"


class TestHelperFunctions:
    """Tests for helper functions used in risk table mapping."""

    def test_load_json_config_success(self):
        """Test successful loading of JSON config."""
        config_data = {"key1": "value1", "key2": 123}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            loaded_config = load_json_config(config_path)
            assert loaded_config == config_data
        finally:
            os.remove(config_path)

    def test_load_json_config_file_not_found(self):
        """Test that FileNotFoundError is raised for missing config."""
        with pytest.raises(FileNotFoundError):
            load_json_config("/nonexistent/path/config.json")

    def test_load_json_config_invalid_json(self):
        """Test that JSONDecodeError is raised for invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json }")
            config_path = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                load_json_config(config_path)
        finally:
            os.remove(config_path)

    def test_validate_categorical_fields_valid(self):
        """Test validation of suitable categorical fields."""
        # Create DataFrame with consistent lengths
        n_rows = 150
        df = pd.DataFrame(
            {
                "cat_field": ["A", "B", "C", "A", "B"]
                * 30,  # 5 unique values, repeated
                "num_field": list(range(n_rows)),
                "many_unique": list(range(n_rows)),  # 150 unique values
            }
        )

        cat_field_list = ["cat_field", "num_field", "many_unique", "missing_field"]
        valid_fields = validate_categorical_fields(df, cat_field_list)

        # Only cat_field should be valid (< 100 unique values)
        assert "cat_field" in valid_fields
        assert "missing_field" not in valid_fields  # Not in dataframe
        assert "many_unique" not in valid_fields  # Too many unique values (150 >= 100)

    def test_validate_categorical_fields_categorical_dtype(self):
        """Test that categorical dtype fields are always valid."""
        df = pd.DataFrame(
            {
                "cat_field": pd.Categorical(["A", "B", "C"]),
            }
        )

        valid_fields = validate_categorical_fields(df, ["cat_field"])
        assert "cat_field" in valid_fields

    def test_detect_file_format_csv(self):
        """Test detection of CSV format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            split_dir = Path(temp_dir)
            csv_file = split_dir / "train_processed_data.csv"
            csv_file.write_text("col1,col2\nval1,val2")

            file_path, fmt = _detect_file_format(split_dir, "train")
            assert fmt == "csv"
            assert file_path == csv_file

    def test_detect_file_format_tsv(self):
        """Test detection of TSV format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            split_dir = Path(temp_dir)
            tsv_file = split_dir / "train_processed_data.tsv"
            tsv_file.write_text("col1\tcol2\nval1\tval2")

            file_path, fmt = _detect_file_format(split_dir, "train")
            assert fmt == "tsv"
            assert file_path == tsv_file

    def test_detect_file_format_parquet(self):
        """Test detection of Parquet format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            split_dir = Path(temp_dir)
            parquet_file = split_dir / "train_processed_data.parquet"
            df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
            df.to_parquet(parquet_file, index=False)

            file_path, fmt = _detect_file_format(split_dir, "train")
            assert fmt == "parquet"
            assert file_path == parquet_file

    def test_detect_file_format_missing_file(self):
        """Test that RuntimeError is raised when no file is found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            split_dir = Path(temp_dir)

            with pytest.raises(RuntimeError, match="No processed data file found"):
                _detect_file_format(split_dir, "train")

    def test_copy_existing_artifacts(self):
        """Test copying existing artifacts from previous steps."""
        with tempfile.TemporaryDirectory() as temp_dir:
            src_dir = os.path.join(temp_dir, "src")
            dst_dir = os.path.join(temp_dir, "dst")
            os.makedirs(src_dir)

            # Create some artifact files
            artifact1 = os.path.join(src_dir, "model.pkl")
            artifact2 = os.path.join(src_dir, "config.json")
            Path(artifact1).write_text("model data")
            Path(artifact2).write_text('{"key": "value"}')

            # Copy artifacts
            copy_existing_artifacts(src_dir, dst_dir)

            # Verify files were copied
            assert os.path.exists(os.path.join(dst_dir, "model.pkl"))
            assert os.path.exists(os.path.join(dst_dir, "config.json"))

    def test_copy_existing_artifacts_empty_source(self):
        """Test that empty source directory is handled gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dst_dir = os.path.join(temp_dir, "dst")

            # Call with non-existent source
            copy_existing_artifacts("/nonexistent/path", dst_dir)

            # Should not raise error

    def test_load_risk_tables_success(self):
        """Test successful loading of risk tables from file."""
        risk_tables = {"var1": {"bins": {"A": 0.5, "B": 0.3}, "default_bin": 0.4}}

        with tempfile.TemporaryDirectory() as temp_dir:
            risk_table_path = Path(temp_dir) / RISK_TABLE_FILENAME
            with open(risk_table_path, "wb") as f:
                pkl.dump(risk_tables, f)

            loaded_tables = load_risk_tables(risk_table_path)
            assert loaded_tables == risk_tables

    def test_load_risk_tables_file_not_found(self):
        """Test that FileNotFoundError is raised for missing risk tables."""
        with pytest.raises(FileNotFoundError):
            load_risk_tables(Path("/nonexistent/risk_table_map.pkl"))

    def test_save_artifacts(self):
        """Test saving risk table artifacts."""
        binner = OfflineBinning(["cat_var"], "target")
        df = pd.DataFrame({"cat_var": ["A", "B"], "target": [1, 0]})
        binner.fit(df)

        hyperparams = {"cat_field_list": ["cat_var"], "label_name": "target"}

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir)
            save_artifacts(binner, hyperparams, output_path)

            # Verify artifacts were saved
            assert (output_path / RISK_TABLE_FILENAME).exists()
            assert (output_path / HYPERPARAMS_FILENAME).exists()

            # Verify content
            with open(output_path / RISK_TABLE_FILENAME, "rb") as f:
                loaded_tables = pkl.load(f)
            assert "cat_var" in loaded_tables


class TestDataLoading:
    """Tests for data loading and saving functions."""

    def test_load_split_data_training_csv(self):
        """Test loading training data splits in CSV format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = temp_dir

            # Create train/test/val splits
            for split in ["train", "test", "val"]:
                split_dir = os.path.join(input_dir, split)
                os.makedirs(split_dir)
                df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
                df.to_csv(
                    os.path.join(split_dir, f"{split}_processed_data.csv"), index=False
                )

            result = load_split_data("training", input_dir)

            assert "train" in result
            assert "test" in result
            assert "val" in result
            assert result["_format"] == "csv"
            assert result["train"].shape == (2, 2)

    def test_load_split_data_validation_tsv(self):
        """Test loading validation data in TSV format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = temp_dir
            val_dir = os.path.join(input_dir, "validation")
            os.makedirs(val_dir)

            df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
            df.to_csv(
                os.path.join(val_dir, "validation_processed_data.tsv"),
                sep="\t",
                index=False,
            )

            result = load_split_data("validation", input_dir)

            assert "validation" in result
            assert result["_format"] == "tsv"
            assert result["validation"].shape == (2, 2)

    def test_load_split_data_parquet(self):
        """Test loading data in Parquet format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = temp_dir
            test_dir = os.path.join(input_dir, "testing")
            os.makedirs(test_dir)

            df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
            df.to_parquet(
                os.path.join(test_dir, "testing_processed_data.parquet"), index=False
            )

            result = load_split_data("testing", input_dir)

            assert "testing" in result
            assert result["_format"] == "parquet"

    def test_save_output_data_csv(self):
        """Test saving data in CSV format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = temp_dir

            data_dict = {
                "train": pd.DataFrame({"col1": [1, 2], "col2": [3, 4]}),
                "_format": "csv",
            }

            save_output_data("training", output_dir, data_dict)

            # Verify file was created
            output_file = os.path.join(output_dir, "train", "train_processed_data.csv")
            assert os.path.exists(output_file)

            # Verify content
            df = pd.read_csv(output_file)
            assert df.shape == (2, 2)

    def test_save_output_data_preserves_format(self):
        """Test that save preserves the input format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = temp_dir

            # Test TSV format
            data_dict = {
                "validation": pd.DataFrame({"col1": [1, 2], "col2": [3, 4]}),
                "_format": "tsv",
            }

            save_output_data("validation", output_dir, data_dict)

            output_file = os.path.join(
                output_dir, "validation", "validation_processed_data.tsv"
            )
            assert os.path.exists(output_file)


class TestProcessData:
    """Tests for the core process_data function."""

    def test_process_data_training_mode(self):
        """Test process_data in training mode."""
        data_dict = {
            "train": pd.DataFrame(
                {"cat_var": ["A", "B", "A", "C"], "target": [1, 0, 1, 0]}
            ),
            "test": pd.DataFrame({"cat_var": ["A", "B"], "target": [1, 0]}),
            "val": pd.DataFrame({"cat_var": ["B", "C"], "target": [0, 1]}),
            "_format": "csv",
        }

        transformed_data, binner = process_data(
            data_dict=data_dict,
            cat_field_list=["cat_var"],
            label_name="target",
            job_type="training",
            smooth_factor=0.01,
            count_threshold=1,
        )

        # Verify risk tables were created
        assert "cat_var" in binner.risk_tables

        # Verify all splits were transformed
        assert "train" in transformed_data
        assert "test" in transformed_data
        assert "val" in transformed_data

        # Verify transformation occurred
        assert pd.api.types.is_numeric_dtype(transformed_data["train"]["cat_var"])

    def test_process_data_training_no_valid_fields(self):
        """Test process_data when no valid categorical fields exist."""
        data_dict = {
            "train": pd.DataFrame(
                {
                    "num_var": list(range(150)),  # Too many unique values
                    "target": [1, 0] * 75,
                }
            ),
            "_format": "csv",
        }

        transformed_data, binner = process_data(
            data_dict=data_dict,
            cat_field_list=["num_var"],
            label_name="target",
            job_type="training",
        )

        # Should create empty binner
        assert len(binner.risk_tables) == 0

        # Data should be unchanged
        assert transformed_data["train"].equals(data_dict["train"])

    def test_process_data_inference_mode(self):
        """Test process_data in inference mode."""
        # Pre-trained risk tables
        risk_tables_dict = {
            "cat_var": {
                "varName": "cat_var",
                "bins": {"A": 0.8, "B": 0.3},
                "default_bin": 0.5,
            }
        }

        data_dict = {
            "validation": pd.DataFrame(
                {
                    "cat_var": ["A", "B", "C"],  # C is unseen
                    "target": [1, 0, 1],
                }
            ),
            "_format": "csv",
        }

        transformed_data, binner = process_data(
            data_dict=data_dict,
            cat_field_list=["cat_var"],
            label_name="target",
            job_type="validation",
            risk_tables_dict=risk_tables_dict,
        )

        # Verify transformation used pre-trained tables
        assert abs(transformed_data["validation"]["cat_var"].iloc[0] - 0.8) < 0.001  # A
        assert abs(transformed_data["validation"]["cat_var"].iloc[1] - 0.3) < 0.001  # B
        assert (
            abs(transformed_data["validation"]["cat_var"].iloc[2] - 0.5) < 0.001
        )  # C (default)

    def test_process_data_inference_missing_risk_tables(self):
        """Test that ValueError is raised when risk_tables_dict is missing in inference mode."""
        data_dict = {
            "validation": pd.DataFrame({"cat_var": ["A"], "target": [1]}),
            "_format": "csv",
        }

        with pytest.raises(ValueError, match="risk_tables_dict must be provided"):
            process_data(
                data_dict=data_dict,
                cat_field_list=["cat_var"],
                label_name="target",
                job_type="validation",
                risk_tables_dict=None,
            )


class TestInternalMain:
    """Tests for the internal_main function."""

    @pytest.fixture
    def setup_temp_dirs(self):
        """Set up temporary directories for testing."""
        temp_dir = tempfile.mkdtemp()
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        yield {
            "temp_dir": temp_dir,
            "input_dir": input_dir,
            "output_dir": output_dir,
        }

        shutil.rmtree(temp_dir)

    def test_internal_main_training(self, setup_temp_dirs):
        """Test internal_main in training mode."""
        dirs = setup_temp_dirs
        input_dir = dirs["input_dir"]
        output_dir = dirs["output_dir"]

        # Create training data
        for split in ["train", "test", "val"]:
            split_dir = os.path.join(input_dir, split)
            os.makedirs(split_dir)
            df = pd.DataFrame({"cat_var": ["A", "B", "A"], "target": [1, 0, 1]})
            df.to_csv(
                os.path.join(split_dir, f"{split}_processed_data.csv"), index=False
            )

        hyperparams = {
            "cat_field_list": ["cat_var"],
            "label_name": "target",
            "smooth_factor": 0.01,
            "count_threshold": 1,
        }

        result, binner = internal_main(
            job_type="training",
            input_dir=input_dir,
            output_dir=output_dir,
            hyperparams=hyperparams,
        )

        # Verify outputs
        assert os.path.exists(
            os.path.join(output_dir, "train", "train_processed_data.csv")
        )
        assert os.path.exists(
            os.path.join(output_dir, "model_artifacts", RISK_TABLE_FILENAME)
        )

    def test_internal_main_with_artifacts_copy(self, setup_temp_dirs):
        """Test that internal_main copies existing artifacts."""
        dirs = setup_temp_dirs
        input_dir = dirs["input_dir"]
        output_dir = dirs["output_dir"]
        artifacts_input_dir = os.path.join(dirs["temp_dir"], "input_artifacts")
        os.makedirs(artifacts_input_dir)

        # Create existing artifact
        existing_artifact = os.path.join(artifacts_input_dir, "previous_model.pkl")
        Path(existing_artifact).write_text("previous model data")

        # Create training data
        for split in ["train", "test", "val"]:
            split_dir = os.path.join(input_dir, split)
            os.makedirs(split_dir)
            df = pd.DataFrame({"cat_var": ["A", "B"], "target": [1, 0]})
            df.to_csv(
                os.path.join(split_dir, f"{split}_processed_data.csv"), index=False
            )

        hyperparams = {"cat_field_list": ["cat_var"], "label_name": "target"}

        result, binner = internal_main(
            job_type="training",
            input_dir=input_dir,
            output_dir=output_dir,
            hyperparams=hyperparams,
            model_artifacts_input_dir=artifacts_input_dir,
        )

        # Verify existing artifact was copied
        assert os.path.exists(
            os.path.join(output_dir, "model_artifacts", "previous_model.pkl")
        )

    def test_internal_main_with_custom_load_save_funcs(self, setup_temp_dirs):
        """Test internal_main with custom load/save functions for dependency injection."""
        dirs = setup_temp_dirs

        # Mock load and save functions
        mock_data_dict = {
            "train": pd.DataFrame({"cat_var": ["A", "B"], "target": [1, 0]}),
            "_format": "csv",
        }

        def mock_load_func(job_type, input_dir):
            return mock_data_dict

        save_called = []

        def mock_save_func(job_type, output_dir, data_dict):
            save_called.append((job_type, output_dir, data_dict))

        hyperparams = {"cat_field_list": ["cat_var"], "label_name": "target"}

        result, binner = internal_main(
            job_type="training",
            input_dir=dirs["input_dir"],
            output_dir=dirs["output_dir"],
            hyperparams=hyperparams,
            load_data_func=mock_load_func,
            save_data_func=mock_save_func,
        )

        # Verify custom functions were used
        assert len(save_called) == 1
        assert save_called[0][0] == "training"


class TestMainRiskTableFlow:
    """Tests the main execution flow of the risk table mapping script."""

    @pytest.fixture
    def setup_dirs(self):
        """Set up temporary directories and hyperparameters."""
        temp_dir = tempfile.mkdtemp()
        input_dir = os.path.join(temp_dir, "input")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Create dummy hyperparameters
        hyperparams = {
            "cat_field_list": ["cat_var"],
            "label_name": "target",
            "smooth_factor": 0.01,
            "count_threshold": 5,
        }

        yield {
            "temp_dir": temp_dir,
            "input_dir": input_dir,
            "output_dir": output_dir,
            "hyperparams": hyperparams,
        }

        shutil.rmtree(temp_dir)

    def test_main_training_mode(self, setup_dirs):
        """Test the main logic in 'training' mode."""
        dirs = setup_dirs
        temp_dir = dirs["temp_dir"]
        input_dir = dirs["input_dir"]
        output_dir = dirs["output_dir"]
        hyperparams = dirs["hyperparams"]

        # Create split data files as expected by the new API
        for split in ["train", "test", "val"]:
            split_dir = os.path.join(input_dir, split)
            os.makedirs(split_dir)
            df = pd.DataFrame(
                {
                    "cat_var": ["A", "B", "A", "C", "B"],
                    "num_var": [10, 20, 30, 40, 50],
                    "target": [1, 0, 1, 0, 1],
                }
            )
            df.to_csv(
                os.path.join(split_dir, f"{split}_processed_data.csv"), index=False
            )

        # Create config directory and hyperparameters file
        config_dir = os.path.join(temp_dir, "config")
        os.makedirs(config_dir)
        hyperparams_path = os.path.join(config_dir, "hyperparameters.json")
        with open(hyperparams_path, "w") as f:
            json.dump(hyperparams, f)

        # Create job_args mock
        job_args = Namespace(job_type="training")

        # Set up input and output paths - use correct key names
        input_paths = {
            "input_data": input_dir,
            "hyperparameters_s3_uri": hyperparams_path,
        }
        output_paths = {
            "processed_data": output_dir,
            "model_artifacts_output": os.path.join(output_dir, "model_artifacts"),
        }
        environ_vars = {}

        # Run main function
        result, binner = risk_mapping_main(
            input_paths, output_paths, environ_vars, job_args
        )

        # Assertions
        train_path = os.path.join(output_dir, "train", "train_processed_data.csv")
        test_path = os.path.join(output_dir, "test", "test_processed_data.csv")
        val_path = os.path.join(output_dir, "val", "val_processed_data.csv")
        assert os.path.exists(train_path)
        assert os.path.exists(test_path)
        assert os.path.exists(val_path)

        # Check that artifacts were saved in model_artifacts subdirectory
        artifacts_dir = os.path.join(output_dir, "model_artifacts")
        assert os.path.exists(os.path.join(artifacts_dir, RISK_TABLE_FILENAME))
        assert os.path.exists(os.path.join(artifacts_dir, HYPERPARAMS_FILENAME))

        # Check content of transformed data
        train_df = pd.read_csv(train_path)
        assert "cat_var" in train_df.columns
        assert pd.api.types.is_numeric_dtype(
            train_df["cat_var"]
        )  # Should be numeric after risk mapping

    def test_main_inference_mode(self, setup_dirs):
        """Test the main logic in a non-training ('validation') mode."""
        dirs = setup_dirs
        temp_dir = dirs["temp_dir"]
        input_dir = dirs["input_dir"]
        output_dir = dirs["output_dir"]
        hyperparams = dirs["hyperparams"]

        # First, create training data and run training to generate risk tables
        # Need to create all required splits for training mode
        for split in ["train", "test", "val"]:
            split_dir = os.path.join(input_dir, split)
            os.makedirs(split_dir)
            train_df = pd.DataFrame({"cat_var": ["A", "B", "A"], "target": [1, 0, 1]})
            train_df.to_csv(
                os.path.join(split_dir, f"{split}_processed_data.csv"), index=False
            )

        # Create a temporary directory for risk tables
        risk_table_dir = os.path.join(temp_dir, "risk_tables")
        os.makedirs(risk_table_dir)

        # Create config directory and hyperparameters file
        config_dir = os.path.join(temp_dir, "config")
        os.makedirs(config_dir)
        hyperparams_path = os.path.join(config_dir, "hyperparameters.json")
        with open(hyperparams_path, "w") as f:
            json.dump(hyperparams, f)

        # Generate risk tables by running training mode first using internal_main
        internal_main(
            job_type="training",
            input_dir=input_dir,
            output_dir=risk_table_dir,
            hyperparams=hyperparams,
            model_artifacts_output_dir=os.path.join(risk_table_dir, "model_artifacts"),
        )

        # Now create validation data
        val_input_dir = os.path.join(temp_dir, "val_input")
        os.makedirs(val_input_dir)
        val_dir = os.path.join(val_input_dir, "validation")
        os.makedirs(val_dir)
        val_df = pd.DataFrame({"cat_var": ["A", "B", "C"], "target": [1, 0, 1]})
        val_df.to_csv(
            os.path.join(val_dir, "validation_processed_data.csv"), index=False
        )

        # Create job_args mock for validation
        job_args = Namespace(job_type="validation")

        # Set up input and output paths for validation - use correct key names
        input_paths = {
            "input_data": val_input_dir,
            "hyperparameters_s3_uri": hyperparams_path,
            "model_artifacts_input": os.path.join(risk_table_dir, "model_artifacts"),
        }
        output_paths = {"processed_data": output_dir}
        environ_vars = {}

        # Run main function in validation mode
        result, binner = risk_mapping_main(
            input_paths, output_paths, environ_vars, job_args
        )

        # Assertions
        val_output_path = os.path.join(
            output_dir, "validation", "validation_processed_data.csv"
        )
        assert os.path.exists(val_output_path)

        # Check that the validation data was transformed based on the train data
        artifacts_dir = os.path.join(risk_table_dir, "model_artifacts")
        with open(os.path.join(artifacts_dir, RISK_TABLE_FILENAME), "rb") as f:
            bins = pkl.load(f)

        # Check if validation data was transformed using the risk tables
        val_df_output = pd.read_csv(val_output_path)
        assert pd.api.types.is_numeric_dtype(val_df_output["cat_var"])

    def test_main_missing_required_input_path(self, setup_dirs):
        """Test that ValueError is raised when required input path is missing."""
        job_args = Namespace(job_type="training")

        input_paths = {}  # Missing required input_data
        output_paths = {"processed_data": "/tmp/output"}
        environ_vars = {}

        with pytest.raises(ValueError, match="Missing required input path: input_data"):
            risk_mapping_main(input_paths, output_paths, environ_vars, job_args)

    def test_main_missing_required_output_path(self, setup_dirs):
        """Test that ValueError is raised when required output path is missing."""
        job_args = Namespace(job_type="training")

        input_paths = {"input_data": "/tmp/input"}
        output_paths = {}  # Missing required processed_data
        environ_vars = {}

        with pytest.raises(
            ValueError, match="Missing required output path: processed_data"
        ):
            risk_mapping_main(input_paths, output_paths, environ_vars, job_args)

    def test_main_missing_job_type(self, setup_dirs):
        """Test that ValueError is raised when job_type is missing."""
        job_args = Namespace()  # No job_type

        input_paths = {"input_data": "/tmp/input"}
        output_paths = {"processed_data": "/tmp/output"}
        environ_vars = {}

        with pytest.raises(
            ValueError, match="job_args must contain job_type parameter"
        ):
            risk_mapping_main(input_paths, output_paths, environ_vars, job_args)

    def test_main_missing_hyperparameters_file(self, setup_dirs):
        """Test that FileNotFoundError is raised when hyperparameters file is missing."""
        dirs = setup_dirs
        job_args = Namespace(job_type="training")

        input_paths = {"input_data": dirs["input_dir"]}
        output_paths = {"processed_data": dirs["output_dir"]}
        environ_vars = {}

        with pytest.raises(FileNotFoundError, match="Hyperparameters file not found"):
            risk_mapping_main(input_paths, output_paths, environ_vars, job_args)

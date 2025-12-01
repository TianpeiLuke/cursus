# test/test_tabular_preprocess.py
import pytest
from unittest.mock import patch, MagicMock, Mock, call
import os
import pandas as pd
import numpy as np
import tempfile
import shutil
import gzip
import json
from pathlib import Path
from multiprocessing import cpu_count

# Import the functions to be tested from the updated script
from cursus.steps.scripts.tabular_preprocessing import (
    combine_shards,
    _read_file_to_df,
    _read_shard_wrapper,
    _batch_concat_dataframes,
    peek_json_format,
    load_signature_columns,
    main as preprocess_main,
)


class TestTabularPreprocessHelpers:
    """Unit tests for the helper functions in the preprocessing script."""

    @pytest.fixture
    def temp_dir(self):
        """Set up a temporary directory to act as a mock filesystem."""
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        yield temp_dir, temp_path
        shutil.rmtree(temp_dir)

    # --- Helper methods to create test files ---
    def _create_csv_shard(
        self, temp_path, filename, data, gzipped=False, delimiter=","
    ):
        """Helper to create a CSV/TSV shard file."""
        path = temp_path / filename
        df = pd.DataFrame(data)
        if gzipped:
            with gzip.open(path, "wt", newline="") as f:
                df.to_csv(f, index=False, sep=delimiter)
        else:
            df.to_csv(path, index=False, sep=delimiter)
        return path

    def _create_json_shard(self, temp_path, filename, data, lines=True, gzipped=False):
        """Helper to create a JSON shard file."""
        path = temp_path / filename
        open_func = gzip.open if gzipped else open
        mode = "wt"
        with open_func(path, mode) as f:
            if lines:
                if data:
                    for record in data:
                        f.write(json.dumps(record) + "\n")
            else:
                json.dump(data, f)
        return path

    def _create_parquet_shard(self, temp_path, filename, data):
        """Helper to create a Parquet shard file."""
        path = temp_path / filename
        df = pd.DataFrame(data)
        df.to_parquet(path, index=False)
        return path

    # --- Tests for file processing and utility functions ---
    def test_combine_shards_success(self, temp_dir):
        """Test combine_shards with multiple file formats."""
        temp_dir_path, temp_path = temp_dir

        self._create_csv_shard(temp_path, "part-00000.csv", [{"a": 1}])
        self._create_json_shard(temp_path, "part-00001.json", [{"a": 2}])
        combined_df = combine_shards(temp_dir_path)
        assert len(combined_df) == 2

    def test_combine_shards_no_files(self, temp_dir):
        """Test that combine_shards raises an error if no valid shards are found."""
        temp_dir_path, temp_path = temp_dir

        with pytest.raises(RuntimeError, match="No CSV/JSON/Parquet shards found"):
            combine_shards(temp_dir_path)

    def test_read_json_single_object(self, temp_dir):
        """Test reading a JSON file with a single top-level object."""
        temp_dir_path, temp_path = temp_dir

        json_path = self._create_json_shard(
            temp_path, "single.json", {"a": 1, "b": "c"}, lines=False
        )
        df = _read_file_to_df(json_path)
        assert df.shape == (1, 2)
        assert df.iloc[0]["b"] == "c"

    def test_peek_json_format_empty_file(self, temp_dir):
        """Test peek_json_format with an empty file raises an error."""
        temp_dir_path, temp_path = temp_dir

        empty_path = temp_path / "empty.json"
        empty_path.touch()
        with pytest.raises(RuntimeError, match="Empty file"):
            peek_json_format(empty_path)

    # --- NEW TESTS FOR PARALLEL PROCESSING ---
    
    def test_read_shard_wrapper_success(self, temp_dir):
        """Test _read_shard_wrapper successfully reads a shard."""
        temp_dir_path, temp_path = temp_dir
        
        # Create a test shard
        shard_path = self._create_csv_shard(
            temp_path, "part-00000.csv", [{"col1": 1, "col2": 2}]
        )
        
        # Test the wrapper function
        args = (shard_path, None, 0, 1)
        result_df = _read_shard_wrapper(args)
        
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 1
        assert "col1" in result_df.columns
        assert result_df.iloc[0]["col1"] == 1

    def test_read_shard_wrapper_with_signature_columns(self, temp_dir):
        """Test _read_shard_wrapper with signature columns."""
        temp_dir_path, temp_path = temp_dir
        
        # Create a test shard with custom columns
        shard_path = self._create_csv_shard(
            temp_path, "part-00000.csv", [{"a": 1, "b": 2}]
        )
        
        # Test with signature columns
        signature_columns = ["feature1", "feature2"]
        args = (shard_path, signature_columns, 0, 1)
        result_df = _read_shard_wrapper(args)
        
        assert isinstance(result_df, pd.DataFrame)
        assert list(result_df.columns) == signature_columns

    def test_read_shard_wrapper_error_handling(self, temp_dir):
        """Test _read_shard_wrapper error handling for invalid shard."""
        temp_dir_path, temp_path = temp_dir
        
        # Create path to non-existent shard
        invalid_path = temp_path / "nonexistent.csv"
        
        args = (invalid_path, None, 0, 1)
        with pytest.raises(RuntimeError, match="Failed to read shard"):
            _read_shard_wrapper(args)

    def test_batch_concat_dataframes_single_df(self):
        """Test _batch_concat_dataframes with single DataFrame."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = _batch_concat_dataframes([df])
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        pd.testing.assert_frame_equal(result, df)

    def test_batch_concat_dataframes_multiple_dfs(self):
        """Test _batch_concat_dataframes with multiple DataFrames."""
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [3, 4]})
        df3 = pd.DataFrame({"a": [5, 6]})
        
        result = _batch_concat_dataframes([df1, df2, df3], batch_size=2)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 6
        assert result["a"].tolist() == [1, 2, 3, 4, 5, 6]

    def test_batch_concat_dataframes_large_batch(self):
        """Test _batch_concat_dataframes with many DataFrames."""
        dfs = [pd.DataFrame({"a": [i]}) for i in range(20)]
        
        result = _batch_concat_dataframes(dfs, batch_size=5)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 20
        assert result["a"].tolist() == list(range(20))

    def test_batch_concat_dataframes_empty_list(self):
        """Test _batch_concat_dataframes with empty list raises error."""
        with pytest.raises(ValueError, match="No DataFrames to concatenate"):
            _batch_concat_dataframes([])

    def test_combine_shards_parallel_processing(self, temp_dir):
        """Test combine_shards uses parallel processing for multiple shards."""
        temp_dir_path, temp_path = temp_dir
        
        # Create multiple shards
        for i in range(5):
            self._create_csv_shard(
                temp_path, f"part-{i:05d}.csv", [{"col": i}]
            )
        
        # Test parallel processing
        result_df = combine_shards(temp_dir_path, max_workers=2)
        
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 5
        assert "col" in result_df.columns

    def test_combine_shards_sequential_fallback(self, temp_dir):
        """Test combine_shards falls back to sequential for single shard."""
        temp_dir_path, temp_path = temp_dir
        
        # Create single shard
        self._create_csv_shard(temp_path, "part-00000.csv", [{"col": 1}])
        
        # Should use sequential processing
        result_df = combine_shards(temp_dir_path, max_workers=1)
        
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 1

    def test_combine_shards_batch_size_parameter(self, temp_dir):
        """Test combine_shards respects batch_size parameter."""
        temp_dir_path, temp_path = temp_dir
        
        # Create multiple shards
        for i in range(10):
            self._create_csv_shard(
                temp_path, f"part-{i:05d}.csv", [{"col": i}]
            )
        
        # Test with custom batch size
        result_df = combine_shards(temp_dir_path, batch_size=3)
        
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 10

    def test_combine_shards_with_signature_columns(self, temp_dir):
        """Test combine_shards with signature columns."""
        temp_dir_path, temp_path = temp_dir
        
        # Create shards with generic column names
        self._create_csv_shard(temp_path, "part-00000.csv", [{"a": 1, "b": 2}])
        self._create_csv_shard(temp_path, "part-00001.csv", [{"a": 3, "b": 4}])
        
        # Provide signature columns
        signature_columns = ["feature1", "feature2"]
        result_df = combine_shards(temp_dir_path, signature_columns=signature_columns)
        
        assert isinstance(result_df, pd.DataFrame)
        assert list(result_df.columns) == signature_columns
        assert len(result_df) == 2

    def test_load_signature_columns_success(self, temp_dir):
        """Test load_signature_columns successfully loads column names."""
        temp_dir_path, temp_path = temp_dir
        
        # Create signature file
        signature_dir = temp_path / "signature"
        signature_dir.mkdir()
        signature_file = signature_dir / "signature"
        signature_file.write_text("col1, col2, col3")
        
        result = load_signature_columns(str(signature_dir))
        
        assert result == ["col1", "col2", "col3"]

    def test_load_signature_columns_no_directory(self, temp_dir):
        """Test load_signature_columns returns None for non-existent directory."""
        temp_dir_path, temp_path = temp_dir
        
        nonexistent_dir = temp_path / "nonexistent"
        result = load_signature_columns(str(nonexistent_dir))
        
        assert result is None

    def test_load_signature_columns_empty_directory(self, temp_dir):
        """Test load_signature_columns returns None for empty directory."""
        temp_dir_path, temp_path = temp_dir
        
        # Create empty signature directory
        signature_dir = temp_path / "signature"
        signature_dir.mkdir()
        
        result = load_signature_columns(str(signature_dir))
        
        assert result is None

    def test_combine_shards_signature_ignored_for_json(self, temp_dir):
        """Test that signature columns are ignored for JSON files."""
        temp_dir_path, temp_path = temp_dir
        
        # Create JSON shards with actual column names
        self._create_json_shard(
            temp_path, "part-00000.json", [{"actual_col1": 1, "actual_col2": 2}]
        )
        self._create_json_shard(
            temp_path, "part-00001.json", [{"actual_col1": 3, "actual_col2": 4}]
        )
        
        # Provide signature columns that should be ignored for JSON
        signature_columns = ["signature_col1", "signature_col2"]
        result_df = combine_shards(temp_dir_path, signature_columns=signature_columns)
        
        # Verify actual column names are used, not signature columns
        assert isinstance(result_df, pd.DataFrame)
        assert list(result_df.columns) == ["actual_col1", "actual_col2"]
        assert len(result_df) == 2

    def test_combine_shards_signature_ignored_for_parquet(self, temp_dir):
        """Test that signature columns are ignored for Parquet files."""
        temp_dir_path, temp_path = temp_dir
        
        # Create Parquet shards with actual column names
        self._create_parquet_shard(
            temp_path, "part-00000.parquet", [{"actual_col1": 1, "actual_col2": 2}]
        )
        self._create_parquet_shard(
            temp_path, "part-00001.parquet", [{"actual_col1": 3, "actual_col2": 4}]
        )
        
        # Provide signature columns that should be ignored for Parquet
        signature_columns = ["signature_col1", "signature_col2"]
        result_df = combine_shards(temp_dir_path, signature_columns=signature_columns)
        
        # Verify actual column names are used, not signature columns
        assert isinstance(result_df, pd.DataFrame)
        assert list(result_df.columns) == ["actual_col1", "actual_col2"]
        assert len(result_df) == 2


class TestMainFunction:
    """Tests for the main preprocessing logic."""

    @pytest.fixture
    def setup_dirs(self):
        """Set up test directories."""
        temp_dir = tempfile.mkdtemp()
        input_base_dir = os.path.join(temp_dir, "input")
        input_data_dir = os.path.join(input_base_dir, "data")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(input_data_dir, exist_ok=True)

        yield temp_dir, input_data_dir, output_dir
        shutil.rmtree(temp_dir)

    def test_main_training_split(self, setup_dirs):
        """Test the main logic for a 'training' job_type, verifying the three-way split."""
        temp_dir, input_data_dir, output_dir = setup_dirs

        df = pd.DataFrame(
            {
                "feature1": np.random.rand(100),
                "feature2__DOT__val": np.random.rand(100),
                "label": np.random.choice(["A", "B"], 100),
            }
        )
        df.to_csv(os.path.join(input_data_dir, "part-00000.csv"), index=False)

        # Create job_args mock
        from argparse import Namespace

        job_args = Namespace(job_type="training")

        # Set up input and output paths - match implementation expectations
        # Create empty signature directory to avoid file vs directory error
        signature_dir = os.path.join(temp_dir, "signature")
        os.makedirs(signature_dir, exist_ok=True)
        input_paths = {"DATA": input_data_dir, "SIGNATURE": signature_dir}
        output_paths = {"processed_data": output_dir}
        environ_vars = {
            "LABEL_FIELD": "label",
            "TRAIN_RATIO": "0.6",
            "TEST_VAL_RATIO": "0.5",
        }

        result = preprocess_main(input_paths, output_paths, environ_vars, job_args)

        # Verify outputs
        train_df = pd.read_csv(
            os.path.join(output_dir, "train", "train_processed_data.csv")
        )
        test_df = pd.read_csv(
            os.path.join(output_dir, "test", "test_processed_data.csv")
        )
        val_df = pd.read_csv(os.path.join(output_dir, "val", "val_processed_data.csv"))

        assert len(train_df) == 60
        assert len(test_df) == 20
        assert len(val_df) == 20
        assert "feature2.val" in train_df.columns
        assert pd.api.types.is_integer_dtype(train_df["label"])

        # Verify return value
        assert "train" in result
        assert "test" in result
        assert "val" in result

    def test_main_validation_mode(self, setup_dirs):
        """Test main logic for a non-training job_type, ensuring no split occurs."""
        temp_dir, input_data_dir, output_dir = setup_dirs

        df = pd.DataFrame({"feature1": range(100), "label": range(100)})
        df.to_csv(os.path.join(input_data_dir, "part-00000.csv"), index=False)

        # Create job_args mock
        from argparse import Namespace

        job_args = Namespace(job_type="validation")

        # Set up input and output paths - match implementation expectations
        # Create empty signature directory to avoid file vs directory error
        signature_dir = os.path.join(temp_dir, "signature")
        os.makedirs(signature_dir, exist_ok=True)
        input_paths = {"DATA": input_data_dir, "SIGNATURE": signature_dir}
        output_paths = {"processed_data": output_dir}
        environ_vars = {
            "LABEL_FIELD": "label",
            "TRAIN_RATIO": "0.8",
            "TEST_VAL_RATIO": "0.5",
        }

        result = preprocess_main(input_paths, output_paths, environ_vars, job_args)

        assert os.path.exists(os.path.join(output_dir, "validation"))
        assert not os.path.exists(os.path.join(output_dir, "train"))
        val_df = pd.read_csv(
            os.path.join(output_dir, "validation", "validation_processed_data.csv")
        )
        assert len(val_df) == 100

        # Verify return value
        assert "validation" in result
        assert len(result) == 1

    def test_main_label_not_found_error(self, setup_dirs):
        """Test that main raises a RuntimeError if the label field is not found."""
        temp_dir, input_data_dir, output_dir = setup_dirs

        df = pd.DataFrame({"feature1": [1, 2]})
        df.to_csv(os.path.join(input_data_dir, "part-00000.csv"), index=False)

        # Create job_args mock
        from argparse import Namespace

        job_args = Namespace(job_type="training")

        # Set up input and output paths - match implementation expectations
        # Create empty signature directory to avoid file vs directory error
        signature_dir = os.path.join(temp_dir, "signature")
        os.makedirs(signature_dir, exist_ok=True)
        input_paths = {"DATA": input_data_dir, "SIGNATURE": signature_dir}
        output_paths = {"processed_data": output_dir}
        environ_vars = {
            "LABEL_FIELD": "wrong_label",
            "TRAIN_RATIO": "0.8",
            "TEST_VAL_RATIO": "0.5",
        }

        with pytest.raises(RuntimeError, match="Label field 'wrong_label' not found"):
            preprocess_main(input_paths, output_paths, environ_vars, job_args)

    # --- NEW TESTS FOR OUTPUT FORMATS ---
    
    def test_main_parquet_output_format(self, setup_dirs):
        """Test main with Parquet output format."""
        temp_dir, input_data_dir, output_dir = setup_dirs

        df = pd.DataFrame({"feature1": range(50), "label": range(50)})
        df.to_csv(os.path.join(input_data_dir, "part-00000.csv"), index=False)

        from argparse import Namespace
        job_args = Namespace(job_type="validation")

        signature_dir = os.path.join(temp_dir, "signature")
        os.makedirs(signature_dir, exist_ok=True)
        input_paths = {"DATA": input_data_dir, "SIGNATURE": signature_dir}
        output_paths = {"processed_data": output_dir}
        environ_vars = {
            "LABEL_FIELD": "label",
            "TRAIN_RATIO": "0.8",
            "TEST_VAL_RATIO": "0.5",
            "OUTPUT_FORMAT": "parquet",
        }

        result = preprocess_main(input_paths, output_paths, environ_vars, job_args)

        # Verify Parquet file was created
        parquet_path = os.path.join(
            output_dir, "validation", "validation_processed_data.parquet"
        )
        assert os.path.exists(parquet_path)
        
        # Verify we can read the Parquet file
        val_df = pd.read_parquet(parquet_path)
        assert len(val_df) == 50

    def test_main_tsv_output_format(self, setup_dirs):
        """Test main with TSV output format."""
        temp_dir, input_data_dir, output_dir = setup_dirs

        df = pd.DataFrame({"feature1": range(50), "label": range(50)})
        df.to_csv(os.path.join(input_data_dir, "part-00000.csv"), index=False)

        from argparse import Namespace
        job_args = Namespace(job_type="validation")

        signature_dir = os.path.join(temp_dir, "signature")
        os.makedirs(signature_dir, exist_ok=True)
        input_paths = {"DATA": input_data_dir, "SIGNATURE": signature_dir}
        output_paths = {"processed_data": output_dir}
        environ_vars = {
            "LABEL_FIELD": "label",
            "TRAIN_RATIO": "0.8",
            "TEST_VAL_RATIO": "0.5",
            "OUTPUT_FORMAT": "tsv",
        }

        result = preprocess_main(input_paths, output_paths, environ_vars, job_args)

        # Verify TSV file was created
        tsv_path = os.path.join(
            output_dir, "validation", "validation_processed_data.tsv"
        )
        assert os.path.exists(tsv_path)
        
        # Verify we can read the TSV file
        val_df = pd.read_csv(tsv_path, sep="\t")
        assert len(val_df) == 50

    def test_main_invalid_output_format_defaults_to_csv(self, setup_dirs):
        """Test main with invalid output format defaults to CSV."""
        temp_dir, input_data_dir, output_dir = setup_dirs

        df = pd.DataFrame({"feature1": range(50), "label": range(50)})
        df.to_csv(os.path.join(input_data_dir, "part-00000.csv"), index=False)

        from argparse import Namespace
        job_args = Namespace(job_type="validation")

        signature_dir = os.path.join(temp_dir, "signature")
        os.makedirs(signature_dir, exist_ok=True)
        input_paths = {"DATA": input_data_dir, "SIGNATURE": signature_dir}
        output_paths = {"processed_data": output_dir}
        environ_vars = {
            "LABEL_FIELD": "label",
            "TRAIN_RATIO": "0.8",
            "TEST_VAL_RATIO": "0.5",
            "OUTPUT_FORMAT": "invalid_format",
        }

        result = preprocess_main(input_paths, output_paths, environ_vars, job_args)

        # Verify CSV file was created (default fallback)
        csv_path = os.path.join(
            output_dir, "validation", "validation_processed_data.csv"
        )
        assert os.path.exists(csv_path)

    def test_main_without_label_field(self, setup_dirs):
        """Test main without label field (feature engineering only)."""
        temp_dir, input_data_dir, output_dir = setup_dirs

        df = pd.DataFrame({"feature1": range(100), "feature2": range(100, 200)})
        df.to_csv(os.path.join(input_data_dir, "part-00000.csv"), index=False)

        from argparse import Namespace
        job_args = Namespace(job_type="validation")

        signature_dir = os.path.join(temp_dir, "signature")
        os.makedirs(signature_dir, exist_ok=True)
        input_paths = {"DATA": input_data_dir, "SIGNATURE": signature_dir}
        output_paths = {"processed_data": output_dir}
        environ_vars = {
            "LABEL_FIELD": None,  # No label field
            "TRAIN_RATIO": "0.8",
            "TEST_VAL_RATIO": "0.5",
        }

        result = preprocess_main(input_paths, output_paths, environ_vars, job_args)

        # Verify processing completed successfully without labels
        val_df = pd.read_csv(
            os.path.join(output_dir, "validation", "validation_processed_data.csv")
        )
        assert len(val_df) == 100
        assert "feature1" in val_df.columns
        assert "feature2" in val_df.columns

    def test_main_training_without_label_uses_random_split(self, setup_dirs):
        """Test main in training mode without labels uses random splits."""
        temp_dir, input_data_dir, output_dir = setup_dirs

        df = pd.DataFrame({"feature1": range(100), "feature2": range(100, 200)})
        df.to_csv(os.path.join(input_data_dir, "part-00000.csv"), index=False)

        from argparse import Namespace
        job_args = Namespace(job_type="training")

        signature_dir = os.path.join(temp_dir, "signature")
        os.makedirs(signature_dir, exist_ok=True)
        input_paths = {"DATA": input_data_dir, "SIGNATURE": signature_dir}
        output_paths = {"processed_data": output_dir}
        environ_vars = {
            "LABEL_FIELD": None,  # No label field
            "TRAIN_RATIO": "0.7",
            "TEST_VAL_RATIO": "0.5",
        }

        result = preprocess_main(input_paths, output_paths, environ_vars, job_args)

        # Verify splits were created
        train_df = pd.read_csv(
            os.path.join(output_dir, "train", "train_processed_data.csv")
        )
        test_df = pd.read_csv(
            os.path.join(output_dir, "test", "test_processed_data.csv")
        )
        val_df = pd.read_csv(
            os.path.join(output_dir, "val", "val_processed_data.csv")
        )

        # Verify split ratios (should use random split, not stratified)
        assert len(train_df) == 70
        assert len(test_df) == 15
        assert len(val_df) == 15

    def test_main_testing_mode(self, setup_dirs):
        """Test main logic for 'testing' job_type."""
        temp_dir, input_data_dir, output_dir = setup_dirs

        df = pd.DataFrame({"feature1": range(100), "label": range(100)})
        df.to_csv(os.path.join(input_data_dir, "part-00000.csv"), index=False)

        from argparse import Namespace
        job_args = Namespace(job_type="testing")

        signature_dir = os.path.join(temp_dir, "signature")
        os.makedirs(signature_dir, exist_ok=True)
        input_paths = {"DATA": input_data_dir, "SIGNATURE": signature_dir}
        output_paths = {"processed_data": output_dir}
        environ_vars = {
            "LABEL_FIELD": "label",
            "TRAIN_RATIO": "0.8",
            "TEST_VAL_RATIO": "0.5",
        }

        result = preprocess_main(input_paths, output_paths, environ_vars, job_args)

        # Verify testing output created
        assert os.path.exists(os.path.join(output_dir, "testing"))
        assert not os.path.exists(os.path.join(output_dir, "train"))
        testing_df = pd.read_csv(
            os.path.join(output_dir, "testing", "testing_processed_data.csv")
        )
        assert len(testing_df) == 100

        # Verify return value
        assert "testing" in result
        assert len(result) == 1

    def test_main_calibration_mode(self, setup_dirs):
        """Test main logic for 'calibration' job_type."""
        temp_dir, input_data_dir, output_dir = setup_dirs

        df = pd.DataFrame({"feature1": range(100), "label": range(100)})
        df.to_csv(os.path.join(input_data_dir, "part-00000.csv"), index=False)

        from argparse import Namespace
        job_args = Namespace(job_type="calibration")

        signature_dir = os.path.join(temp_dir, "signature")
        os.makedirs(signature_dir, exist_ok=True)
        input_paths = {"DATA": input_data_dir, "SIGNATURE": signature_dir}
        output_paths = {"processed_data": output_dir}
        environ_vars = {
            "LABEL_FIELD": "label",
            "TRAIN_RATIO": "0.8",
            "TEST_VAL_RATIO": "0.5",
        }

        result = preprocess_main(input_paths, output_paths, environ_vars, job_args)

        # Verify calibration output created
        assert os.path.exists(os.path.join(output_dir, "calibration"))
        assert not os.path.exists(os.path.join(output_dir, "train"))
        calibration_df = pd.read_csv(
            os.path.join(output_dir, "calibration", "calibration_processed_data.csv")
        )
        assert len(calibration_df) == 100

        # Verify return value
        assert "calibration" in result
        assert len(result) == 1

    def test_main_output_format_case_insensitive(self, setup_dirs):
        """Test main handles OUTPUT_FORMAT case-insensitively."""
        temp_dir, input_data_dir, output_dir = setup_dirs

        df = pd.DataFrame({"feature1": range(50), "label": range(50)})
        df.to_csv(os.path.join(input_data_dir, "part-00000.csv"), index=False)

        from argparse import Namespace
        job_args = Namespace(job_type="validation")

        signature_dir = os.path.join(temp_dir, "signature")
        os.makedirs(signature_dir, exist_ok=True)
        input_paths = {"DATA": input_data_dir, "SIGNATURE": signature_dir}
        output_paths = {"processed_data": output_dir}

        # Test uppercase CSV
        environ_vars = {
            "LABEL_FIELD": "label",
            "TRAIN_RATIO": "0.8",
            "TEST_VAL_RATIO": "0.5",
            "OUTPUT_FORMAT": "CSV",  # Uppercase
        }

        result = preprocess_main(input_paths, output_paths, environ_vars, job_args)

        # Verify CSV file was created
        csv_path = os.path.join(
            output_dir, "validation", "validation_processed_data.csv"
        )
        assert os.path.exists(csv_path)

        # Clean up for next test
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        # Test uppercase PARQUET
        environ_vars["OUTPUT_FORMAT"] = "PARQUET"
        result = preprocess_main(input_paths, output_paths, environ_vars, job_args)

        parquet_path = os.path.join(
            output_dir, "validation", "validation_processed_data.parquet"
        )
        assert os.path.exists(parquet_path)

        # Clean up for next test
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        # Test uppercase TSV
        environ_vars["OUTPUT_FORMAT"] = "TSV"
        result = preprocess_main(input_paths, output_paths, environ_vars, job_args)

        tsv_path = os.path.join(
            output_dir, "validation", "validation_processed_data.tsv"
        )
        assert os.path.exists(tsv_path)

    def test_main_with_custom_logger(self, setup_dirs):
        """Test main with custom logger parameter."""
        temp_dir, input_data_dir, output_dir = setup_dirs

        df = pd.DataFrame({"feature1": range(50), "label": range(50)})
        df.to_csv(os.path.join(input_data_dir, "part-00000.csv"), index=False)

        from argparse import Namespace
        job_args = Namespace(job_type="validation")

        signature_dir = os.path.join(temp_dir, "signature")
        os.makedirs(signature_dir, exist_ok=True)
        input_paths = {"DATA": input_data_dir, "SIGNATURE": signature_dir}
        output_paths = {"processed_data": output_dir}
        environ_vars = {
            "LABEL_FIELD": "label",
            "TRAIN_RATIO": "0.8",
            "TEST_VAL_RATIO": "0.5",
        }

        # Create a mock logger to capture log messages
        log_messages = []

        def custom_logger(msg):
            log_messages.append(msg)

        result = preprocess_main(
            input_paths, output_paths, environ_vars, job_args, logger=custom_logger
        )

        # Verify logger was used (should have captured some messages)
        assert len(log_messages) > 0
        # Verify at least some expected log messages were captured
        assert any("Combining data shards" in msg for msg in log_messages)
        assert any("Preprocessing complete" in msg for msg in log_messages)

        # Verify processing still completed successfully
        val_df = pd.read_csv(
            os.path.join(output_dir, "validation", "validation_processed_data.csv")
        )
        assert len(val_df) == 50


class TestPerformanceAndScalability:
    """Tests for performance and scalability features."""

    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for performance tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_combine_shards_handles_many_shards(self, temp_workspace):
        """Test combine_shards efficiently handles many shards."""
        # Create 50 shards
        for i in range(50):
            df = pd.DataFrame({"col": [i]})
            df.to_csv(temp_workspace / f"part-{i:05d}.csv", index=False)

        # Should complete efficiently with parallel processing
        result_df = combine_shards(str(temp_workspace))

        assert len(result_df) == 50
        assert "col" in result_df.columns

    def test_combine_shards_handles_large_individual_shards(self, temp_workspace):
        """Test combine_shards handles large individual shards."""
        # Create 5 shards with 1000 rows each
        for i in range(5):
            df = pd.DataFrame({"col": range(i * 1000, (i + 1) * 1000)})
            df.to_csv(temp_workspace / f"part-{i:05d}.csv", index=False)

        result_df = combine_shards(str(temp_workspace))

        assert len(result_df) == 5000
        assert result_df["col"].min() == 0
        assert result_df["col"].max() == 4999

    def test_batch_concatenation_memory_efficiency(self):
        """Test batch concatenation is memory efficient."""
        # Create many small DataFrames
        dfs = [pd.DataFrame({"col": [i]}) for i in range(100)]

        # Should handle efficiently with batching
        result = _batch_concat_dataframes(dfs, batch_size=10)

        assert len(result) == 100
        assert result["col"].tolist() == list(range(100))


class TestEdgeCasesAndErrorHandling:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for edge case tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_combine_shards_mixed_compressed_uncompressed(self, temp_workspace):
        """Test combine_shards handles mixed compressed and uncompressed files."""
        # Create uncompressed CSV
        df1 = pd.DataFrame({"col": [1, 2]})
        df1.to_csv(temp_workspace / "part-00000.csv", index=False)

        # Create compressed CSV
        df2 = pd.DataFrame({"col": [3, 4]})
        with gzip.open(temp_workspace / "part-00001.csv.gz", "wt") as f:
            df2.to_csv(f, index=False)

        result_df = combine_shards(str(temp_workspace))

        assert len(result_df) == 4
        assert result_df["col"].tolist() == [1, 2, 3, 4]

    def test_combine_shards_empty_shards(self, temp_workspace):
        """Test combine_shards handles shards with no data rows gracefully."""
        # Create CSV with headers but no data rows
        df_empty = pd.DataFrame({"col": []})
        df_empty.to_csv(temp_workspace / "part-00000.csv", index=False)

        # Create non-empty CSV
        df = pd.DataFrame({"col": [1, 2]})
        df.to_csv(temp_workspace / "part-00001.csv", index=False)

        result_df = combine_shards(str(temp_workspace))

        # Should concatenate both, resulting in 2 rows (0 from empty + 2 from non-empty)
        assert len(result_df) == 2
        assert "col" in result_df.columns

    def test_combine_shards_with_missing_values(self, temp_workspace):
        """Test combine_shards preserves missing values."""
        # Create shard with missing values
        df1 = pd.DataFrame({"col1": [1, np.nan, 3], "col2": [np.nan, 5, 6]})
        df1.to_csv(temp_workspace / "part-00000.csv", index=False)

        df2 = pd.DataFrame({"col1": [4, 5, np.nan], "col2": [7, np.nan, 9]})
        df2.to_csv(temp_workspace / "part-00001.csv", index=False)

        result_df = combine_shards(str(temp_workspace))

        # Verify missing values are preserved
        assert len(result_df) == 6
        assert result_df.isna().sum().sum() == 4  # Total NaN count

    def test_combine_shards_mismatched_columns(self, temp_workspace):
        """Test combine_shards handles mismatched columns across shards."""
        # Create shards with different columns
        df1 = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        df1.to_csv(temp_workspace / "part-00000.csv", index=False)

        df2 = pd.DataFrame({"col1": [5, 6], "col3": [7, 8]})
        df2.to_csv(temp_workspace / "part-00001.csv", index=False)

        result_df = combine_shards(str(temp_workspace))

        # Verify all columns are present with NaN for missing values
        assert len(result_df) == 4
        assert set(result_df.columns) == {"col1", "col2", "col3"}
        assert result_df["col2"].isna().sum() == 2  # Missing in second shard
        assert result_df["col3"].isna().sum() == 2  # Missing in first shard

    def test_read_file_to_df_tsv_detection(self, temp_workspace):
        """Test _read_file_to_df correctly detects TSV delimiter."""
        # Create TSV file
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        tsv_path = temp_workspace / "data.tsv"
        df.to_csv(tsv_path, sep="\t", index=False)

        result_df = _read_file_to_df(tsv_path)

        assert len(result_df) == 2
        assert list(result_df.columns) == ["col1", "col2"]

    def test_combine_shards_invalid_directory(self):
        """Test combine_shards raises error for invalid directory."""
        with pytest.raises(RuntimeError, match="does not exist"):
            combine_shards("/nonexistent/directory")

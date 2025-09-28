# test/test_tabular_preprocess.py
import pytest
from unittest.mock import patch, MagicMock
import os
import pandas as pd
import numpy as np
import tempfile
import shutil
import gzip
import json
from pathlib import Path

# Import the functions to be tested from the updated script
from cursus.steps.scripts.tabular_preprocessing import (
    combine_shards,
    _read_file_to_df,
    peek_json_format,
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
    def _create_csv_shard(self, temp_path, filename, data, gzipped=False, delimiter=","):
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

        # Set up input and output paths
        input_paths = {"data_input": input_data_dir}
        output_paths = {"data_output": output_dir}
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
        val_df = pd.read_csv(
            os.path.join(output_dir, "val", "val_processed_data.csv")
        )

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

        # Set up input and output paths
        input_paths = {"data_input": input_data_dir}
        output_paths = {"data_output": output_dir}
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

        # Set up input and output paths
        input_paths = {"data_input": input_data_dir}
        output_paths = {"data_output": output_dir}
        environ_vars = {
            "LABEL_FIELD": "wrong_label",
            "TRAIN_RATIO": "0.8",
            "TEST_VAL_RATIO": "0.5",
        }

        with pytest.raises(RuntimeError, match="Label field 'wrong_label' not found"):
            preprocess_main(input_paths, output_paths, environ_vars, job_args)

# test/test_tabular_preprocess.py
import unittest
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
from src.pipeline_scripts.tabular_preprocess import (
    combine_shards,
    _read_file_to_df,
    peek_json_format,
    main as preprocess_main,
)

class TestTabularPreprocessHelpers(unittest.TestCase):
    """Unit tests for the helper functions in the preprocessing script."""
    def setUp(self):
        """Set up a temporary directory to act as a mock filesystem."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Remove the temporary directory after tests are complete."""
        shutil.rmtree(self.temp_dir)

    # --- Helper methods to create test files ---
    def _create_csv_shard(self, filename, data, gzipped=False, delimiter=','):
        """Helper to create a CSV/TSV shard file."""
        path = self.temp_path / filename
        df = pd.DataFrame(data)
        if gzipped:
            with gzip.open(path, 'wt', newline='') as f:
                df.to_csv(f, index=False, sep=delimiter)
        else:
            df.to_csv(path, index=False, sep=delimiter)
        return path

    def _create_json_shard(self, filename, data, lines=True, gzipped=False):
        """Helper to create a JSON shard file."""
        path = self.temp_path / filename
        open_func = gzip.open if gzipped else open
        mode = 'wt'
        with open_func(path, mode) as f:
            if lines:
                if data:
                    for record in data:
                        f.write(json.dumps(record) + '\n')
            else:
                json.dump(data, f)
        return path

    def _create_parquet_shard(self, filename, data):
        """Helper to create a Parquet shard file."""
        path = self.temp_path / filename
        df = pd.DataFrame(data)
        df.to_parquet(path, index=False)
        return path

    # --- Tests for file processing and utility functions ---
    def test_combine_shards_success(self):
        """Test combine_shards with multiple file formats."""
        self._create_csv_shard("part-00000.csv", [{"a": 1}])
        self._create_json_shard("part-00001.json", [{"a": 2}])
        combined_df = combine_shards(self.temp_dir)
        self.assertEqual(len(combined_df), 2)

    def test_combine_shards_no_files(self):
        """Test that combine_shards raises an error if no valid shards are found."""
        with self.assertRaisesRegex(RuntimeError, "No CSV/JSON/Parquet shards found"):
            combine_shards(self.temp_dir)

    def test_read_json_single_object(self):
        """Test reading a JSON file with a single top-level object."""
        json_path = self._create_json_shard("single.json", {"a": 1, "b": "c"}, lines=False)
        df = _read_file_to_df(json_path)
        self.assertEqual(df.shape, (1, 2))
        self.assertEqual(df.iloc[0]['b'], 'c')

    def test_peek_json_format_empty_file(self):
        """Test peek_json_format with an empty file raises an error."""
        empty_path = self.temp_path / "empty.json"
        empty_path.touch()
        with self.assertRaisesRegex(RuntimeError, "Empty file"):
            peek_json_format(empty_path)


class TestMainFunction(unittest.TestCase):
    """Tests for the main preprocessing logic."""
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.input_base_dir = os.path.join(self.temp_dir, 'input')
        self.input_data_dir = os.path.join(self.input_base_dir, 'data')
        self.output_dir = os.path.join(self.temp_dir, 'output')
        os.makedirs(self.input_data_dir, exist_ok=True)
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_main_training_split(self):
        """Test the main logic for a 'training' job_type, verifying the three-way split."""
        df = pd.DataFrame({
            "feature1": np.random.rand(100),
            "feature2__DOT__val": np.random.rand(100),
            "label": np.random.choice(['A', 'B'], 100)
        })
        df.to_csv(os.path.join(self.input_data_dir, 'part-00000.csv'), index=False)
        
        preprocess_main(
            job_type='training',
            label_field='label',
            train_ratio=0.6,
            test_val_ratio=0.5,
            input_base_dir=self.input_base_dir,
            output_dir=self.output_dir
        )
        
        # Verify outputs
        train_df = pd.read_csv(os.path.join(self.output_dir, 'train', 'train_processed_data.csv'))
        test_df = pd.read_csv(os.path.join(self.output_dir, 'test', 'test_processed_data.csv'))
        val_df = pd.read_csv(os.path.join(self.output_dir, 'val', 'val_processed_data.csv'))

        self.assertEqual(len(train_df), 60)
        self.assertEqual(len(test_df), 20)
        self.assertEqual(len(val_df), 20)
        self.assertIn("feature2.val", train_df.columns)
        self.assertTrue(pd.api.types.is_integer_dtype(train_df['label']))

    def test_main_validation_mode(self):
        """Test main logic for a non-training job_type, ensuring no split occurs."""
        df = pd.DataFrame({"feature1": range(100), "label": range(100)})
        df.to_csv(os.path.join(self.input_data_dir, 'part-00000.csv'), index=False)
        
        preprocess_main(
            job_type='validation',
            label_field='label',
            train_ratio=0.8,
            test_val_ratio=0.5,
            input_base_dir=self.input_base_dir,
            output_dir=self.output_dir
        )
        
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'validation')))
        self.assertFalse(os.path.exists(os.path.join(self.output_dir, 'train')))
        val_df = pd.read_csv(os.path.join(self.output_dir, 'validation', 'validation_processed_data.csv'))
        self.assertEqual(len(val_df), 100)
    
    def test_main_label_not_found_error(self):
        """Test that main raises a RuntimeError if the label field is not found."""
        df = pd.DataFrame({"feature1": [1,2]})
        df.to_csv(os.path.join(self.input_data_dir, 'part-00000.csv'), index=False)

        with self.assertRaisesRegex(RuntimeError, "Label field 'wrong_label' not found"):
            preprocess_main(
                job_type='training',
                label_field='wrong_label',
                train_ratio=0.8,
                test_val_ratio=0.5,
                input_base_dir=self.input_base_dir,
                output_dir=self.output_dir
            )

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

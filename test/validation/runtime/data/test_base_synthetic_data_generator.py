"""Unit tests for BaseSyntheticDataGenerator."""

import unittest
from unittest.mock import Mock, patch, mock_open
import tempfile
import shutil
import pandas as pd
import numpy as np
import json
from pathlib import Path
from abc import ABC

from src.cursus.validation.runtime.data.base_synthetic_data_generator import BaseSyntheticDataGenerator


class ConcreteSyntheticDataGenerator(BaseSyntheticDataGenerator):
    """Concrete implementation for testing abstract base class."""
    
    def get_supported_scripts(self):
        return ["test_script", "pattern_*", "exact_match"]
    
    def generate_for_script(self, script_name, data_size="small", **kwargs):
        return {"output": f"/path/to/{script_name}_data.csv"}


class TestBaseSyntheticDataGenerator(unittest.TestCase):
    """Test cases for BaseSyntheticDataGenerator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.generator = ConcreteSyntheticDataGenerator(random_seed=42)
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_abstract_class_cannot_be_instantiated(self):
        """Test that BaseSyntheticDataGenerator cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            BaseSyntheticDataGenerator()
    
    def test_init_with_default_seed(self):
        """Test initialization with default random seed."""
        generator = ConcreteSyntheticDataGenerator()
        self.assertEqual(generator.random_seed, 42)
    
    def test_init_with_custom_seed(self):
        """Test initialization with custom random seed."""
        generator = ConcreteSyntheticDataGenerator(random_seed=123)
        self.assertEqual(generator.random_seed, 123)
    
    @patch('numpy.random.seed')
    def test_init_sets_numpy_seed(self, mock_seed):
        """Test that initialization sets numpy random seed."""
        ConcreteSyntheticDataGenerator(random_seed=456)
        mock_seed.assert_called_once_with(456)
    
    def test_supports_script_exact_match(self):
        """Test supports_script with exact match."""
        self.assertTrue(self.generator.supports_script("test_script"))
        self.assertTrue(self.generator.supports_script("exact_match"))
        self.assertFalse(self.generator.supports_script("nonexistent_script"))
    
    def test_supports_script_pattern_match(self):
        """Test supports_script with pattern matching."""
        self.assertTrue(self.generator.supports_script("pattern_preprocessing"))
        self.assertTrue(self.generator.supports_script("pattern_training"))
        self.assertTrue(self.generator.supports_script("pattern_"))
        self.assertFalse(self.generator.supports_script("other_pattern"))
    
    def test_supports_script_case_insensitive_partial_match(self):
        """Test supports_script with case-insensitive partial matching."""
        # Create generator with mixed case patterns
        class MixedCaseGenerator(BaseSyntheticDataGenerator):
            def get_supported_scripts(self):
                return ["XGBoost", "Tabular"]
            def generate_for_script(self, script_name, data_size="small", **kwargs):
                return {}
        
        generator = MixedCaseGenerator()
        self.assertTrue(generator.supports_script("xgboost_training"))
        self.assertTrue(generator.supports_script("tabular_preprocessing"))
        self.assertTrue(generator.supports_script("my_xgboost_script"))
        self.assertFalse(generator.supports_script("random_forest"))
    
    def test_get_data_size_config_small(self):
        """Test get_data_size_config for small size."""
        config = self.generator.get_data_size_config("small")
        expected = {"num_records": 100, "num_features": 5}
        self.assertEqual(config, expected)
    
    def test_get_data_size_config_medium(self):
        """Test get_data_size_config for medium size."""
        config = self.generator.get_data_size_config("medium")
        expected = {"num_records": 1000, "num_features": 10}
        self.assertEqual(config, expected)
    
    def test_get_data_size_config_large(self):
        """Test get_data_size_config for large size."""
        config = self.generator.get_data_size_config("large")
        expected = {"num_records": 10000, "num_features": 20}
        self.assertEqual(config, expected)
    
    def test_get_data_size_config_invalid_defaults_to_small(self):
        """Test get_data_size_config with invalid size defaults to small."""
        config = self.generator.get_data_size_config("invalid")
        expected = {"num_records": 100, "num_features": 5}
        self.assertEqual(config, expected)
    
    def test_save_dataframe_csv_format(self):
        """Test save_dataframe with CSV format."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        filename = "test_data.csv"
        
        result_path = self.generator.save_dataframe(df, filename, self.temp_dir)
        
        expected_path = str(Path(self.temp_dir) / filename)
        self.assertEqual(result_path, expected_path)
        self.assertTrue(Path(result_path).exists())
        
        # Verify content
        loaded_df = pd.read_csv(result_path)
        pd.testing.assert_frame_equal(df, loaded_df)
    
    def test_save_dataframe_json_format(self):
        """Test save_dataframe with JSON format."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        filename = "test_data.json"
        
        result_path = self.generator.save_dataframe(df, filename, self.temp_dir)
        
        expected_path = str(Path(self.temp_dir) / filename)
        self.assertEqual(result_path, expected_path)
        self.assertTrue(Path(result_path).exists())
        
        # Verify content
        with open(result_path) as f:
            data = json.load(f)
        self.assertEqual(len(data), 3)
        self.assertEqual(data[0]["col1"], 1)
        self.assertEqual(data[0]["col2"], "a")
    
    def test_save_dataframe_parquet_format(self):
        """Test save_dataframe with Parquet format."""
        # Check if parquet dependencies are available
        try:
            import pyarrow
            parquet_available = True
        except ImportError:
            try:
                import fastparquet
                parquet_available = True
            except ImportError:
                parquet_available = False
        
        if not parquet_available:
            self.skipTest("Parquet dependencies (pyarrow or fastparquet) not available")
        
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        filename = "test_data.parquet"
        
        result_path = self.generator.save_dataframe(df, filename, self.temp_dir)
        
        expected_path = str(Path(self.temp_dir) / filename)
        self.assertEqual(result_path, expected_path)
        self.assertTrue(Path(result_path).exists())
        
        # Verify content
        loaded_df = pd.read_parquet(result_path)
        pd.testing.assert_frame_equal(df, loaded_df)
    
    def test_save_dataframe_unknown_format_defaults_to_csv(self):
        """Test save_dataframe with unknown format defaults to CSV."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        filename = "test_data.unknown"
        
        result_path = self.generator.save_dataframe(df, filename, self.temp_dir)
        
        # Should save as CSV despite unknown extension
        self.assertTrue(Path(result_path).exists())
        loaded_df = pd.read_csv(result_path)
        pd.testing.assert_frame_equal(df, loaded_df)
    
    def test_save_dataframe_no_output_dir(self):
        """Test save_dataframe without output directory."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        filename = "test_data.csv"
        
        result_path = self.generator.save_dataframe(df, filename)
        
        # Should save in current directory
        self.assertEqual(result_path, filename)
        # Clean up
        if Path(filename).exists():
            Path(filename).unlink()
    
    def test_save_dataframe_creates_directories(self):
        """Test save_dataframe creates nested directories."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        filename = "nested/dir/test_data.csv"
        
        result_path = self.generator.save_dataframe(df, filename, self.temp_dir)
        
        expected_path = str(Path(self.temp_dir) / filename)
        self.assertEqual(result_path, expected_path)
        self.assertTrue(Path(result_path).exists())
        self.assertTrue(Path(result_path).parent.exists())
    
    def test_save_json_basic(self):
        """Test save_json with basic dictionary."""
        data = {"key1": "value1", "key2": 42, "key3": [1, 2, 3]}
        filename = "test_config.json"
        
        result_path = self.generator.save_json(data, filename, self.temp_dir)
        
        expected_path = str(Path(self.temp_dir) / filename)
        self.assertEqual(result_path, expected_path)
        self.assertTrue(Path(result_path).exists())
        
        # Verify content
        with open(result_path) as f:
            loaded_data = json.load(f)
        self.assertEqual(loaded_data, data)
    
    def test_save_json_with_datetime_serialization(self):
        """Test save_json handles datetime serialization."""
        from datetime import datetime
        data = {"timestamp": datetime.now(), "value": 123}
        filename = "test_datetime.json"
        
        result_path = self.generator.save_json(data, filename, self.temp_dir)
        
        # Should not raise exception and file should exist
        self.assertTrue(Path(result_path).exists())
        
        # Verify content can be loaded
        with open(result_path) as f:
            loaded_data = json.load(f)
        self.assertEqual(loaded_data["value"], 123)
        self.assertIsInstance(loaded_data["timestamp"], str)
    
    def test_save_json_creates_directories(self):
        """Test save_json creates nested directories."""
        data = {"test": "data"}
        filename = "nested/config/test.json"
        
        result_path = self.generator.save_json(data, filename, self.temp_dir)
        
        expected_path = str(Path(self.temp_dir) / filename)
        self.assertEqual(result_path, expected_path)
        self.assertTrue(Path(result_path).exists())
        self.assertTrue(Path(result_path).parent.exists())
    
    def test_save_json_no_output_dir(self):
        """Test save_json without output directory."""
        data = {"test": "data"}
        filename = "test.json"
        
        result_path = self.generator.save_json(data, filename)
        
        # Should save in current directory
        self.assertEqual(result_path, filename)
        # Clean up
        if Path(filename).exists():
            Path(filename).unlink()
    
    def test_generate_random_dataframe_int_columns(self):
        """Test generate_random_dataframe with integer columns."""
        columns = {"id": "int", "count": "int"}
        df = self.generator.generate_random_dataframe(10, columns)
        
        self.assertEqual(len(df), 10)
        self.assertEqual(list(df.columns), ["id", "count"])
        self.assertTrue(df["id"].dtype in [np.int32, np.int64])
        self.assertTrue(df["count"].dtype in [np.int32, np.int64])
        self.assertTrue(all(1 <= val <= 999 for val in df["id"]))
    
    def test_generate_random_dataframe_float_columns(self):
        """Test generate_random_dataframe with float columns."""
        columns = {"value": "float", "score": "float"}
        df = self.generator.generate_random_dataframe(5, columns)
        
        self.assertEqual(len(df), 5)
        self.assertTrue(df["value"].dtype in [np.float32, np.float64])
        self.assertTrue(df["score"].dtype in [np.float32, np.float64])
    
    def test_generate_random_dataframe_category_columns(self):
        """Test generate_random_dataframe with category columns."""
        columns = {"category": "category", "type": "category"}
        df = self.generator.generate_random_dataframe(20, columns)
        
        self.assertEqual(len(df), 20)
        valid_categories = {"A", "B", "C", "D", "E"}
        self.assertTrue(set(df["category"].unique()).issubset(valid_categories))
        self.assertTrue(set(df["type"].unique()).issubset(valid_categories))
    
    def test_generate_random_dataframe_bool_columns(self):
        """Test generate_random_dataframe with boolean columns."""
        columns = {"flag": "bool", "active": "bool"}
        df = self.generator.generate_random_dataframe(15, columns)
        
        self.assertEqual(len(df), 15)
        self.assertTrue(df["flag"].dtype == bool)
        self.assertTrue(df["active"].dtype == bool)
        self.assertTrue(set(df["flag"].unique()).issubset({True, False}))
    
    def test_generate_random_dataframe_datetime_columns(self):
        """Test generate_random_dataframe with datetime columns."""
        columns = {"created_at": "datetime", "updated_at": "datetime"}
        df = self.generator.generate_random_dataframe(8, columns)
        
        self.assertEqual(len(df), 8)
        # Datetime columns are stored as strings in YYYY-MM-DD format
        self.assertTrue(df["created_at"].dtype == object)
        # Verify date format
        import re
        date_pattern = r'\d{4}-\d{2}-\d{2}'
        self.assertTrue(all(re.match(date_pattern, date) for date in df["created_at"]))
    
    def test_generate_random_dataframe_unknown_type_defaults_to_string(self):
        """Test generate_random_dataframe with unknown column type."""
        columns = {"unknown_col": "unknown_type"}
        df = self.generator.generate_random_dataframe(5, columns)
        
        self.assertEqual(len(df), 5)
        self.assertTrue(df["unknown_col"].dtype == object)
        # Should generate string values
        self.assertTrue(all(isinstance(val, str) for val in df["unknown_col"]))
        self.assertTrue(all(val.startswith("value_") for val in df["unknown_col"]))
    
    def test_generate_random_dataframe_mixed_column_types(self):
        """Test generate_random_dataframe with mixed column types."""
        columns = {
            "id": "int",
            "name": "string",
            "score": "float",
            "category": "category",
            "active": "bool",
            "date": "datetime"
        }
        df = self.generator.generate_random_dataframe(12, columns)
        
        self.assertEqual(len(df), 12)
        self.assertEqual(len(df.columns), 6)
        
        # Verify each column type
        self.assertTrue(df["id"].dtype in [np.int32, np.int64])
        self.assertTrue(df["name"].dtype == object)
        self.assertTrue(df["score"].dtype in [np.float32, np.float64])
        self.assertTrue(df["category"].dtype == object)
        self.assertTrue(df["active"].dtype == bool)
        self.assertTrue(df["date"].dtype == object)
    
    def test_generate_random_dataframe_empty_columns(self):
        """Test generate_random_dataframe with empty columns dict."""
        df = self.generator.generate_random_dataframe(5, {})
        
        # When no columns are specified, pandas creates an empty DataFrame with 0 rows
        self.assertEqual(len(df), 0)
        self.assertEqual(len(df.columns), 0)
    
    def test_generate_random_dataframe_zero_records(self):
        """Test generate_random_dataframe with zero records."""
        columns = {"col1": "int", "col2": "float"}
        df = self.generator.generate_random_dataframe(0, columns)
        
        self.assertEqual(len(df), 0)
        self.assertEqual(list(df.columns), ["col1", "col2"])
    
    @patch('numpy.random.seed')
    def test_random_seed_reproducibility(self, mock_seed):
        """Test that random seed ensures reproducible results."""
        # Create two generators with same seed
        gen1 = ConcreteSyntheticDataGenerator(random_seed=123)
        gen2 = ConcreteSyntheticDataGenerator(random_seed=123)
        
        # Both should have called numpy.random.seed with 123
        self.assertEqual(mock_seed.call_count, 2)
        mock_seed.assert_called_with(123)
    
    def test_abstract_methods_must_be_implemented(self):
        """Test that abstract methods must be implemented in subclasses."""
        class IncompleteGenerator(BaseSyntheticDataGenerator):
            # Missing implementation of abstract methods
            pass
        
        with self.assertRaises(TypeError):
            IncompleteGenerator()
    
    def test_concrete_implementation_methods(self):
        """Test that concrete implementation methods work correctly."""
        # Test get_supported_scripts
        supported = self.generator.get_supported_scripts()
        self.assertEqual(supported, ["test_script", "pattern_*", "exact_match"])
        
        # Test generate_for_script
        result = self.generator.generate_for_script("test_script", "medium")
        self.assertEqual(result, {"output": "/path/to/test_script_data.csv"})
    
    def test_supports_script_with_empty_supported_list(self):
        """Test supports_script when no scripts are supported."""
        class EmptyGenerator(BaseSyntheticDataGenerator):
            def get_supported_scripts(self):
                return []
            def generate_for_script(self, script_name, data_size="small", **kwargs):
                return {}
        
        generator = EmptyGenerator()
        self.assertFalse(generator.supports_script("any_script"))
    
    def test_wildcard_pattern_edge_cases(self):
        """Test wildcard pattern matching edge cases."""
        class WildcardGenerator(BaseSyntheticDataGenerator):
            def get_supported_scripts(self):
                return ["prefix_*", "exact_pattern"]  # Use more specific patterns
            def generate_for_script(self, script_name, data_size="small", **kwargs):
                return {}
        
        generator = WildcardGenerator()
        
        # Test prefix wildcard
        self.assertTrue(generator.supports_script("prefix_anything"))
        self.assertTrue(generator.supports_script("prefix_"))
        
        # Test exact pattern
        self.assertTrue(generator.supports_script("exact_pattern"))
        
        # Test non-matching patterns
        self.assertFalse(generator.supports_script("other_anything"))
        self.assertFalse(generator.supports_script("different_script"))
        
        # Test universal wildcard separately
        class UniversalGenerator(BaseSyntheticDataGenerator):
            def get_supported_scripts(self):
                return ["*"]
            def generate_for_script(self, script_name, data_size="small", **kwargs):
                return {}
        
        universal_gen = UniversalGenerator()
        self.assertTrue(universal_gen.supports_script("anything"))
        self.assertTrue(universal_gen.supports_script("any_script_name"))
    
    def test_file_operations_error_handling(self):
        """Test error handling in file operations."""
        df = pd.DataFrame({"col1": [1, 2, 3]})
        
        # Test with invalid directory path
        with patch('pathlib.Path.mkdir', side_effect=OSError("Permission denied")):
            with self.assertRaises(OSError):
                self.generator.save_dataframe(df, "test.csv", "/invalid/path")


if __name__ == '__main__':
    unittest.main()

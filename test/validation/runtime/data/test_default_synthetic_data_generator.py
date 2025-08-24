"""Unit tests for DefaultSyntheticDataGenerator."""

import unittest
from unittest.mock import Mock, patch, mock_open
import tempfile
import shutil
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta

from src.cursus.validation.runtime.data.default_synthetic_data_generator import DefaultSyntheticDataGenerator
from src.cursus.validation.runtime.data.base_synthetic_data_generator import BaseSyntheticDataGenerator


class TestDefaultSyntheticDataGenerator(unittest.TestCase):
    """Test cases for DefaultSyntheticDataGenerator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.generator = DefaultSyntheticDataGenerator(random_seed=42)
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        # Clean up any temporary files created in current directory
        for pattern in ["temp_*.csv", "temp_*.json"]:
            import glob
            for file in glob.glob(pattern):
                try:
                    Path(file).unlink()
                except:
                    pass
    
    def test_inheritance_from_base_class(self):
        """Test that DefaultSyntheticDataGenerator inherits from BaseSyntheticDataGenerator."""
        self.assertIsInstance(self.generator, BaseSyntheticDataGenerator)
    
    def test_get_supported_scripts(self):
        """Test get_supported_scripts returns expected patterns."""
        supported = self.generator.get_supported_scripts()
        expected = [
            "currency*", "conversion*",
            "tabular*", "preprocessing*", 
            "xgboost*", "training*",
            "calibration*", "model_calibration*",
            "*"  # Fallback for any script
        ]
        self.assertEqual(supported, expected)
    
    def test_supports_currency_scripts(self):
        """Test that currency-related scripts are supported."""
        self.assertTrue(self.generator.supports_script("currency_conversion"))
        self.assertTrue(self.generator.supports_script("conversion_rates"))
        self.assertTrue(self.generator.supports_script("currency_preprocessing"))
    
    def test_supports_tabular_scripts(self):
        """Test that tabular-related scripts are supported."""
        self.assertTrue(self.generator.supports_script("tabular_preprocessing"))
        self.assertTrue(self.generator.supports_script("preprocessing_step"))
        self.assertTrue(self.generator.supports_script("tabular_feature_engineering"))
    
    def test_supports_xgboost_scripts(self):
        """Test that XGBoost-related scripts are supported."""
        self.assertTrue(self.generator.supports_script("xgboost_training"))
        self.assertTrue(self.generator.supports_script("training_pipeline"))
        self.assertTrue(self.generator.supports_script("xgboost_model"))
    
    def test_supports_calibration_scripts(self):
        """Test that calibration-related scripts are supported."""
        self.assertTrue(self.generator.supports_script("calibration_step"))
        self.assertTrue(self.generator.supports_script("model_calibration_pipeline"))
        self.assertTrue(self.generator.supports_script("calibration_analysis"))
    
    def test_supports_fallback_pattern(self):
        """Test that any script is supported due to fallback pattern."""
        self.assertTrue(self.generator.supports_script("random_script"))
        self.assertTrue(self.generator.supports_script("unknown_pipeline"))
        self.assertTrue(self.generator.supports_script("custom_analysis"))
    
    def test_generate_for_script_currency_routing(self):
        """Test generate_for_script routes currency scripts correctly."""
        with patch.object(self.generator, '_generate_currency_data') as mock_currency:
            mock_currency.return_value = {"input": "/path/to/currency.csv"}
            
            result = self.generator.generate_for_script("currency_conversion", "small")
            
            mock_currency.assert_called_once_with("small")
            self.assertEqual(result, {"input": "/path/to/currency.csv"})
    
    def test_generate_for_script_conversion_routing(self):
        """Test generate_for_script routes conversion scripts correctly."""
        with patch.object(self.generator, '_generate_currency_data') as mock_currency:
            mock_currency.return_value = {"input": "/path/to/conversion.csv"}
            
            result = self.generator.generate_for_script("conversion_rates", "medium")
            
            mock_currency.assert_called_once_with("medium")
            self.assertEqual(result, {"input": "/path/to/conversion.csv"})
    
    def test_generate_for_script_tabular_routing(self):
        """Test generate_for_script routes tabular scripts correctly."""
        with patch.object(self.generator, '_generate_tabular_data') as mock_tabular:
            mock_tabular.return_value = {"input": "/path/to/tabular.csv"}
            
            result = self.generator.generate_for_script("tabular_preprocessing", "large")
            
            mock_tabular.assert_called_once_with("large")
            self.assertEqual(result, {"input": "/path/to/tabular.csv"})
    
    def test_generate_for_script_preprocessing_routing(self):
        """Test generate_for_script routes preprocessing scripts correctly."""
        with patch.object(self.generator, '_generate_tabular_data') as mock_tabular:
            mock_tabular.return_value = {"input": "/path/to/preprocessing.csv"}
            
            result = self.generator.generate_for_script("preprocessing_step", "small")
            
            mock_tabular.assert_called_once_with("small")
            self.assertEqual(result, {"input": "/path/to/preprocessing.csv"})
    
    def test_generate_for_script_xgboost_routing(self):
        """Test generate_for_script routes XGBoost scripts correctly."""
        with patch.object(self.generator, '_generate_training_data') as mock_training:
            mock_training.return_value = {"input": "/path/to/training.csv"}
            
            result = self.generator.generate_for_script("xgboost_training", "medium")
            
            mock_training.assert_called_once_with("medium")
            self.assertEqual(result, {"input": "/path/to/training.csv"})
    
    def test_generate_for_script_training_routing(self):
        """Test generate_for_script routes training scripts correctly."""
        with patch.object(self.generator, '_generate_training_data') as mock_training:
            mock_training.return_value = {"input": "/path/to/training.csv"}
            
            result = self.generator.generate_for_script("training_pipeline", "large")
            
            mock_training.assert_called_once_with("large")
            self.assertEqual(result, {"input": "/path/to/training.csv"})
    
    def test_generate_for_script_calibration_routing(self):
        """Test generate_for_script routes calibration scripts correctly."""
        with patch.object(self.generator, '_generate_calibration_data') as mock_calibration:
            mock_calibration.return_value = {"input": "/path/to/calibration.csv"}
            
            result = self.generator.generate_for_script("calibration_step", "small")
            
            mock_calibration.assert_called_once_with("small")
            self.assertEqual(result, {"input": "/path/to/calibration.csv"})
    
    def test_generate_for_script_model_calibration_routing(self):
        """Test generate_for_script routes model_calibration scripts correctly."""
        with patch.object(self.generator, '_generate_calibration_data') as mock_calibration:
            mock_calibration.return_value = {"input": "/path/to/model_calibration.csv"}
            
            result = self.generator.generate_for_script("model_calibration_pipeline", "medium")
            
            mock_calibration.assert_called_once_with("medium")
            self.assertEqual(result, {"input": "/path/to/model_calibration.csv"})
    
    def test_generate_for_script_generic_routing(self):
        """Test generate_for_script routes unknown scripts to generic data."""
        with patch.object(self.generator, '_generate_generic_data') as mock_generic:
            mock_generic.return_value = {"input": "/path/to/generic.csv"}
            
            result = self.generator.generate_for_script("unknown_script", "large")
            
            mock_generic.assert_called_once_with("large")
            self.assertEqual(result, {"input": "/path/to/generic.csv"})
    
    def test_generate_currency_data_small(self):
        """Test _generate_currency_data with small data size."""
        result = self.generator._generate_currency_data("small")
        
        self.assertIn("input", result)
        file_path = Path(result["input"])
        self.assertTrue(file_path.exists())
        self.assertEqual(file_path.name, "temp_currency_data.csv")
        
        # Verify data content
        df = pd.read_csv(file_path)
        self.assertEqual(len(df), 100)  # small size
        expected_columns = ["from_currency", "to_currency", "amount", "date"]
        self.assertEqual(list(df.columns), expected_columns)
        
        # Verify currency values
        valid_currencies = {"USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CNY", "INR"}
        self.assertTrue(set(df["from_currency"].unique()).issubset(valid_currencies))
        self.assertTrue(set(df["to_currency"].unique()).issubset(valid_currencies))
        
        # Verify amounts are positive
        self.assertTrue(all(df["amount"] > 0))
        self.assertTrue(all(df["amount"] <= 10000))
        
        # Verify date format
        import re
        date_pattern = r'\d{4}-\d{2}-\d{2}'
        self.assertTrue(all(re.match(date_pattern, date) for date in df["date"]))
    
    def test_generate_currency_data_medium(self):
        """Test _generate_currency_data with medium data size."""
        result = self.generator._generate_currency_data("medium")
        
        file_path = Path(result["input"])
        df = pd.read_csv(file_path)
        self.assertEqual(len(df), 1000)  # medium size
    
    def test_generate_currency_data_large(self):
        """Test _generate_currency_data with large data size."""
        result = self.generator._generate_currency_data("large")
        
        file_path = Path(result["input"])
        df = pd.read_csv(file_path)
        self.assertEqual(len(df), 10000)  # large size
    
    def test_generate_currency_data_different_currencies(self):
        """Test that currency data generates different from/to currencies."""
        result = self.generator._generate_currency_data("small")
        
        file_path = Path(result["input"])
        df = pd.read_csv(file_path)
        
        # Verify that from_currency and to_currency are different for each row
        different_currencies = df["from_currency"] != df["to_currency"]
        self.assertTrue(all(different_currencies))
    
    def test_generate_tabular_data_small(self):
        """Test _generate_tabular_data with small data size."""
        result = self.generator._generate_tabular_data("small")
        
        self.assertIn("input", result)
        file_path = Path(result["input"])
        self.assertTrue(file_path.exists())
        self.assertEqual(file_path.name, "temp_tabular_data.csv")
        
        # Verify data content
        df = pd.read_csv(file_path)
        self.assertEqual(len(df), 500)  # small size for tabular
        expected_columns = ["id", "feature_1", "feature_2", "feature_3", "feature_4", "feature_5", "target"]
        self.assertEqual(list(df.columns), expected_columns)
        
        # Verify data types and ranges
        self.assertTrue(all(df["id"] == range(1, 501)))
        self.assertTrue(df["feature_3"].dtype == object)  # categorical
        self.assertTrue(set(df["feature_3"].unique()).issubset({"A", "B", "C"}))
        self.assertTrue(set(df["target"].unique()).issubset({0, 1}))
        
        # Verify missing values exist (approximately 5%)
        missing_count = df["feature_1"].isna().sum()
        self.assertGreater(missing_count, 0)
        self.assertLess(missing_count, len(df) * 0.1)  # Less than 10%
    
    def test_generate_tabular_data_medium(self):
        """Test _generate_tabular_data with medium data size."""
        result = self.generator._generate_tabular_data("medium")
        
        file_path = Path(result["input"])
        df = pd.read_csv(file_path)
        self.assertEqual(len(df), 5000)  # medium size
    
    def test_generate_tabular_data_large(self):
        """Test _generate_tabular_data with large data size."""
        result = self.generator._generate_tabular_data("large")
        
        file_path = Path(result["input"])
        df = pd.read_csv(file_path)
        self.assertEqual(len(df), 50000)  # large size
    
    def test_generate_training_data_small(self):
        """Test _generate_training_data with small data size."""
        result = self.generator._generate_training_data("small")
        
        self.assertIn("input", result)
        file_path = Path(result["input"])
        self.assertTrue(file_path.exists())
        self.assertEqual(file_path.name, "temp_training_data.csv")
        
        # Verify data content
        df = pd.read_csv(file_path)
        self.assertEqual(len(df), 1000)  # small size for training
        
        # Verify feature columns
        feature_columns = [f"feature_{i}" for i in range(10)]
        for col in feature_columns:
            self.assertIn(col, df.columns)
        
        # Verify additional columns
        self.assertIn("target", df.columns)
        self.assertIn("category", df.columns)
        
        # Verify target is binary
        self.assertTrue(set(df["target"].unique()).issubset({0, 1}))
        
        # Verify categorical values
        valid_categories = {"cat_A", "cat_B", "cat_C", "cat_D"}
        self.assertTrue(set(df["category"].unique()).issubset(valid_categories))
    
    def test_generate_training_data_medium(self):
        """Test _generate_training_data with medium data size."""
        result = self.generator._generate_training_data("medium")
        
        file_path = Path(result["input"])
        df = pd.read_csv(file_path)
        self.assertEqual(len(df), 10000)  # medium size
    
    def test_generate_training_data_large(self):
        """Test _generate_training_data with large data size."""
        result = self.generator._generate_training_data("large")
        
        file_path = Path(result["input"])
        df = pd.read_csv(file_path)
        self.assertEqual(len(df), 100000)  # large size
    
    def test_generate_training_data_nonlinear_relationship(self):
        """Test that training data has non-linear relationships."""
        result = self.generator._generate_training_data("small")
        
        file_path = Path(result["input"])
        df = pd.read_csv(file_path)
        
        # The target should have some relationship to features (not completely random)
        # We can't test the exact relationship due to randomness, but we can verify
        # that both target values (0 and 1) are present
        target_values = set(df["target"].unique())
        self.assertEqual(target_values, {0, 1})
        
        # Verify that the distribution isn't too skewed (should be somewhat balanced)
        target_counts = df["target"].value_counts()
        min_count = min(target_counts)
        max_count = max(target_counts)
        ratio = min_count / max_count
        self.assertGreater(ratio, 0.1)  # Not too imbalanced
    
    def test_generate_calibration_data_small(self):
        """Test _generate_calibration_data with small data size."""
        result = self.generator._generate_calibration_data("small")
        
        self.assertIn("input", result)
        file_path = Path(result["input"])
        self.assertTrue(file_path.exists())
        self.assertEqual(file_path.name, "temp_calibration_data.csv")
        
        # Verify data content
        df = pd.read_csv(file_path)
        self.assertEqual(len(df), 500)  # small size for calibration
        expected_columns = ["prediction", "actual", "segment", "timestamp"]
        self.assertEqual(list(df.columns), expected_columns)
        
        # Verify prediction values are between 0 and 1
        self.assertTrue(all(0 <= pred <= 1 for pred in df["prediction"]))
        
        # Verify actual values are binary
        self.assertTrue(set(df["actual"].unique()).issubset({0, 1}))
        
        # Verify segments
        valid_segments = {"seg_A", "seg_B", "seg_C"}
        self.assertTrue(set(df["segment"].unique()).issubset(valid_segments))
        
        # Verify timestamp format
        import re
        date_pattern = r'\d{4}-\d{2}-\d{2}'
        self.assertTrue(all(re.match(date_pattern, ts) for ts in df["timestamp"]))
    
    def test_generate_calibration_data_medium(self):
        """Test _generate_calibration_data with medium data size."""
        result = self.generator._generate_calibration_data("medium")
        
        file_path = Path(result["input"])
        df = pd.read_csv(file_path)
        self.assertEqual(len(df), 5000)  # medium size
    
    def test_generate_calibration_data_large(self):
        """Test _generate_calibration_data with large data size."""
        result = self.generator._generate_calibration_data("large")
        
        file_path = Path(result["input"])
        df = pd.read_csv(file_path)
        self.assertEqual(len(df), 50000)  # large size
    
    def test_generate_calibration_data_bias_relationship(self):
        """Test that calibration data has biased predictions for calibration to correct."""
        result = self.generator._generate_calibration_data("small")
        
        file_path = Path(result["input"])
        df = pd.read_csv(file_path)
        
        # The implementation creates biased predictions using power transformation
        # We can verify that predictions and actuals have some relationship
        # (not completely random)
        self.assertTrue(len(df["prediction"].unique()) > 10)  # Should have variety
        self.assertEqual(set(df["actual"].unique()), {0, 1})  # Binary actuals
    
    def test_generate_generic_data_small(self):
        """Test _generate_generic_data with small data size."""
        result = self.generator._generate_generic_data("small")
        
        self.assertIn("input", result)
        file_path = Path(result["input"])
        self.assertTrue(file_path.exists())
        self.assertEqual(file_path.name, "temp_generic_data.csv")
        
        # Verify data content
        df = pd.read_csv(file_path)
        self.assertEqual(len(df), 100)  # small size for generic
        expected_columns = ["id", "numeric_value", "integer_value", "category", "boolean", "timestamp"]
        self.assertEqual(list(df.columns), expected_columns)
        
        # Verify data types and values
        self.assertTrue(all(df["id"] == range(1, 101)))
        self.assertTrue(all(1 <= val <= 99 for val in df["integer_value"]))
        self.assertTrue(set(df["category"].unique()).issubset({"X", "Y", "Z"}))
        self.assertTrue(set(df["boolean"].unique()).issubset({True, False}))
        
        # Verify timestamp format
        import re
        date_pattern = r'\d{4}-\d{2}-\d{2}'
        self.assertTrue(all(re.match(date_pattern, ts) for ts in df["timestamp"]))
    
    def test_generate_generic_data_medium(self):
        """Test _generate_generic_data with medium data size."""
        result = self.generator._generate_generic_data("medium")
        
        file_path = Path(result["input"])
        df = pd.read_csv(file_path)
        self.assertEqual(len(df), 1000)  # medium size
    
    def test_generate_generic_data_large(self):
        """Test _generate_generic_data with large data size."""
        result = self.generator._generate_generic_data("large")
        
        file_path = Path(result["input"])
        df = pd.read_csv(file_path)
        self.assertEqual(len(df), 10000)  # large size
    
    def test_generate_generic_data_invalid_size_defaults_to_small(self):
        """Test _generate_generic_data with invalid size defaults to small."""
        result = self.generator._generate_generic_data("invalid_size")
        
        file_path = Path(result["input"])
        df = pd.read_csv(file_path)
        self.assertEqual(len(df), 100)  # Should default to small size
    
    @patch('numpy.random.seed')
    def test_random_seed_consistency(self, mock_seed):
        """Test that random seed is set consistently."""
        generator = DefaultSyntheticDataGenerator(random_seed=123)
        mock_seed.assert_called_with(123)
    
    def test_case_insensitive_script_matching(self):
        """Test that script matching is case insensitive."""
        # Test uppercase variations
        with patch.object(self.generator, '_generate_currency_data') as mock_currency:
            mock_currency.return_value = {"input": "/path/to/data.csv"}
            
            self.generator.generate_for_script("CURRENCY_conversion", "small")
            mock_currency.assert_called_once()
        
        # Test mixed case variations
        with patch.object(self.generator, '_generate_tabular_data') as mock_tabular:
            mock_tabular.return_value = {"input": "/path/to/data.csv"}
            
            self.generator.generate_for_script("Tabular_Preprocessing", "small")
            mock_tabular.assert_called_once()
    
    def test_multiple_keyword_matching(self):
        """Test scripts with multiple matching keywords."""
        # Script with both "currency" and "conversion" should match currency
        with patch.object(self.generator, '_generate_currency_data') as mock_currency:
            mock_currency.return_value = {"input": "/path/to/data.csv"}
            
            self.generator.generate_for_script("currency_conversion_preprocessing", "small")
            mock_currency.assert_called_once()
        
        # Script with both "xgboost" and "training" should match training
        with patch.object(self.generator, '_generate_training_data') as mock_training:
            mock_training.return_value = {"input": "/path/to/data.csv"}
            
            self.generator.generate_for_script("xgboost_training_pipeline", "small")
            mock_training.assert_called_once()
    
    def test_priority_order_in_script_matching(self):
        """Test that script matching follows priority order (first match wins)."""
        # "currency_training" should match currency (first in the if-elif chain)
        with patch.object(self.generator, '_generate_currency_data') as mock_currency:
            mock_currency.return_value = {"input": "/path/to/data.csv"}
            
            self.generator.generate_for_script("currency_training", "small")
            mock_currency.assert_called_once()
    
    def test_file_cleanup_on_multiple_calls(self):
        """Test that multiple calls don't interfere with each other."""
        # Generate currency data
        result1 = self.generator._generate_currency_data("small")
        file1 = Path(result1["input"])
        self.assertTrue(file1.exists())
        
        # Generate tabular data (should overwrite temp file)
        result2 = self.generator._generate_tabular_data("small")
        file2 = Path(result2["input"])
        self.assertTrue(file2.exists())
        
        # Files should be different
        self.assertNotEqual(result1["input"], result2["input"])
    
    def test_data_size_mapping_consistency(self):
        """Test that data size mapping is consistent across all generation methods."""
        # Test small size consistency
        currency_result = self.generator._generate_currency_data("small")
        currency_df = pd.read_csv(currency_result["input"])
        
        generic_result = self.generator._generate_generic_data("small")
        generic_df = pd.read_csv(generic_result["input"])
        
        # Currency uses 100 for small, generic also uses 100
        self.assertEqual(len(currency_df), 100)
        self.assertEqual(len(generic_df), 100)
        
        # Test medium size
        tabular_result = self.generator._generate_tabular_data("medium")
        tabular_df = pd.read_csv(tabular_result["input"])
        
        calibration_result = self.generator._generate_calibration_data("medium")
        calibration_df = pd.read_csv(calibration_result["input"])
        
        # Both should use medium size (though values differ by method)
        self.assertEqual(len(tabular_df), 5000)
        self.assertEqual(len(calibration_df), 5000)
    
    def test_integration_with_base_class_methods(self):
        """Test integration with inherited base class methods."""
        # Test that we can use base class methods
        config = self.generator.get_data_size_config("medium")
        self.assertEqual(config, {"num_records": 1000, "num_features": 10})
        
        # Test supports_script works with our patterns
        self.assertTrue(self.generator.supports_script("currency_test"))
        self.assertTrue(self.generator.supports_script("unknown_script"))  # Due to "*" pattern
    
    def test_error_handling_in_data_generation(self):
        """Test error handling in data generation methods."""
        # Test with invalid data size (should not crash)
        result = self.generator._generate_currency_data(None)
        self.assertIn("input", result)
        
        # File should still be created
        file_path = Path(result["input"])
        self.assertTrue(file_path.exists())
    
    def test_data_quality_validation(self):
        """Test that generated data meets quality expectations."""
        # Generate training data and validate quality
        result = self.generator._generate_training_data("small")
        df = pd.read_csv(result["input"])
        
        # Check for data quality issues
        # 1. No completely null columns
        for col in df.columns:
            self.assertGreater(df[col].notna().sum(), 0)
        
        # 2. Reasonable value ranges for numeric columns
        for i in range(10):  # feature_0 to feature_9
            feature_col = f"feature_{i}"
            if feature_col in df.columns:
                # Should have some variation (not all same value)
                self.assertGreater(df[feature_col].nunique(), 1)
        
        # 3. Target distribution is reasonable
        target_dist = df["target"].value_counts(normalize=True)
        # Neither class should be less than 10% of data
        self.assertTrue(all(prop >= 0.1 for prop in target_dist.values))
    
    def test_timestamp_generation_validity(self):
        """Test that generated timestamps are valid and within reasonable range."""
        result = self.generator._generate_calibration_data("small")
        df = pd.read_csv(result["input"])
        
        # Parse timestamps and verify they're within last year
        from datetime import datetime, timedelta
        now = datetime.now()
        one_year_ago = now - timedelta(days=365)
        
        for timestamp_str in df["timestamp"].sample(10):  # Check sample
            timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d")
            self.assertGreater(timestamp, one_year_ago)
            self.assertLessEqual(timestamp, now)
    
    def test_categorical_data_distribution(self):
        """Test that categorical data has reasonable distribution."""
        result = self.generator._generate_tabular_data("small")
        df = pd.read_csv(result["input"])
        
        # Check feature_3 (categorical column)
        category_counts = df["feature_3"].value_counts()
        
        # Should have all expected categories
        expected_categories = {"A", "B", "C"}
        self.assertEqual(set(category_counts.index), expected_categories)
        
        # No category should be completely absent or dominate too much
        total_count = len(df)
        for count in category_counts.values:
            proportion = count / total_count
            self.assertGreater(proportion, 0.1)  # At least 10%
            self.assertLess(proportion, 0.8)     # At most 80%


if __name__ == '__main__':
    unittest.main()

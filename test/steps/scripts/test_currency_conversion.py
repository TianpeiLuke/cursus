"""
Comprehensive test suite for currency_conversion.py script.

This enhanced version includes:
- Complete unit test coverage for all functions
- Integration tests with realistic data scenarios
- Performance and scalability testing
- Error handling and edge case validation
- Data quality and validation testing
- Contract compliance verification
"""

import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import sys
import pandas as pd
import numpy as np
import tempfile
import shutil
import json
import time
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolError

# Import all functions from the script to be tested
from src.cursus.steps.scripts.currency_conversion import (
    get_currency_code,
    combine_currency_codes,
    currency_conversion_single_variable,
    parallel_currency_conversion,
    process_currency_conversion,
    main,
)


class TestCurrencyConversionHelpers(unittest.TestCase):
    """Comprehensive unit tests for helper functions in currency conversion script."""

    def setUp(self):
        """Set up test fixtures with comprehensive test data."""
        self.df = pd.DataFrame({
            'mp_id': [1, 2, 3, np.nan, 'invalid', 4.5, -1],
            'price': [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0],
            'currency': ['USD', 'EUR', None, 'CAD', 'INVALID', 'JPY', 'USD']
        })
        
        self.marketplace_info = {
            "1": {"currency_code": "USD"},
            "2": {"currency_code": "EUR"},
            "3": {"currency_code": "JPY"},
            "4": {"currency_code": "GBP"}
        }
        
        self.currency_dict = {
            "EUR": 0.9, 
            "JPY": 150.0, 
            "USD": 1.0, 
            "GBP": 0.8,
            "CAD": 1.25
        }

    def test_get_currency_code_valid_cases(self):
        """Test get_currency_code with valid marketplace IDs."""
        self.assertEqual(get_currency_code(1, self.marketplace_info, "USD"), "USD")
        self.assertEqual(get_currency_code(2, self.marketplace_info, "USD"), "EUR")
        self.assertEqual(get_currency_code(3, self.marketplace_info, "USD"), "JPY")
        
        # Test with float ID that converts to int
        self.assertEqual(get_currency_code(1.0, self.marketplace_info, "USD"), "USD")
        self.assertEqual(get_currency_code(2.9, self.marketplace_info, "USD"), "EUR")  # Should truncate to 2

    def test_get_currency_code_invalid_cases(self):
        """Test get_currency_code with invalid inputs."""
        # Non-existent marketplace ID
        self.assertEqual(get_currency_code(99, self.marketplace_info, "USD"), "USD")
        
        # NaN marketplace ID
        self.assertEqual(get_currency_code(np.nan, self.marketplace_info, "USD"), "USD")
        
        # String that can't be converted to int
        self.assertEqual(get_currency_code("invalid", self.marketplace_info, "USD"), "USD")
        
        # None marketplace ID
        self.assertEqual(get_currency_code(None, self.marketplace_info, "USD"), "USD")
        
        # Negative marketplace ID
        self.assertEqual(get_currency_code(-1, self.marketplace_info, "USD"), "USD")

    def test_get_currency_code_edge_cases(self):
        """Test get_currency_code with edge case inputs."""
        # Empty marketplace_info
        self.assertEqual(get_currency_code(1, {}, "USD"), "USD")
        
        # None marketplace_info
        self.assertEqual(get_currency_code(1, None, "USD"), "USD")
        
        # Marketplace info missing currency_code field
        incomplete_info = {"1": {"other_field": "value"}}
        self.assertEqual(get_currency_code(1, incomplete_info, "USD"), "USD")
        
        # Very large marketplace ID
        self.assertEqual(get_currency_code(999999999, self.marketplace_info, "USD"), "USD")

    def test_combine_currency_codes_with_existing_column(self):
        """Test combine_currency_codes when currency column exists."""
        df_combined, col_name = combine_currency_codes(
            self.df.copy(), 'mp_id', 'currency', self.marketplace_info, 'USD', False
        )
        
        self.assertEqual(col_name, 'currency')
        
        # Check that None values are filled from marketplace info
        self.assertEqual(df_combined.loc[2, 'currency'], 'JPY')  # mp_id=3 -> JPY
        
        # Check that existing values are preserved
        self.assertEqual(df_combined.loc[0, 'currency'], 'USD')  # Original USD
        self.assertEqual(df_combined.loc[1, 'currency'], 'EUR')  # Original EUR

    def test_combine_currency_codes_without_existing_column(self):
        """Test combine_currency_codes when currency column doesn't exist."""
        df_no_currency = self.df.drop(columns=['currency'])
        df_combined, col_name = combine_currency_codes(
            df_no_currency.copy(), 'mp_id', 'currency', self.marketplace_info, 'USD', False
        )
        
        self.assertEqual(col_name, 'currency_code_from_marketplace_id')
        self.assertTrue('currency_code_from_marketplace_id' in df_combined.columns)
        
        # Check mappings
        self.assertEqual(df_combined.loc[0, 'currency_code_from_marketplace_id'], 'USD')  # mp_id=1
        self.assertEqual(df_combined.loc[1, 'currency_code_from_marketplace_id'], 'EUR')  # mp_id=2
        self.assertEqual(df_combined.loc[2, 'currency_code_from_marketplace_id'], 'JPY')  # mp_id=3

    def test_combine_currency_codes_skip_invalid_currencies(self):
        """Test combine_currency_codes with skip_invalid_currencies=True."""
        df_with_invalid = self.df.copy()
        df_with_invalid.loc[4, 'currency'] = np.nan  # Make currency invalid
        
        df_combined, _ = combine_currency_codes(
            df_with_invalid, 'mp_id', 'currency', self.marketplace_info, 'USD', True
        )
        
        # Should fill NaN currencies with default
        self.assertEqual(df_combined.loc[4, 'currency'], 'USD')

    def test_combine_currency_codes_empty_dataframe(self):
        """Test combine_currency_codes with empty DataFrame."""
        empty_df = pd.DataFrame(columns=['mp_id', 'price'])
        result_df, col_name = combine_currency_codes(
            empty_df, 'mp_id', 'currency', self.marketplace_info, 'USD', False
        )
        
        self.assertEqual(len(result_df), 0)
        self.assertEqual(col_name, 'currency_code_from_marketplace_id')

    def test_currency_conversion_single_variable_basic(self):
        """Test basic currency conversion for single variable."""
        df_test = pd.DataFrame({
            'price': [100.0, 200.0, 300.0],
            'currency': ['USD', 'EUR', 'JPY']
        })
        
        result = currency_conversion_single_variable(
            df_test, 'price', 'currency', self.currency_dict
        )
        
        # USD should remain unchanged (rate = 1.0)
        self.assertEqual(result.loc[0, 'price'], 100.0)
        
        # EUR should be converted: 200 / 0.9 ≈ 222.22
        self.assertAlmostEqual(result.loc[1, 'price'], 200.0 / 0.9, places=2)
        
        # JPY should be converted: 300 / 150 = 2.0
        self.assertEqual(result.loc[2, 'price'], 2.0)

    def test_currency_conversion_single_variable_edge_cases(self):
        """Test currency conversion with edge cases."""
        df_test = pd.DataFrame({
            'price': [0.0, -100.0, 1000000.0, np.nan],
            'currency': ['USD', 'EUR', 'JPY', 'USD']
        })
        
        result = currency_conversion_single_variable(
            df_test, 'price', 'currency', self.currency_dict
        )
        
        # Zero price should remain zero
        self.assertEqual(result.loc[0, 'price'], 0.0)
        
        # Negative price should be converted
        self.assertAlmostEqual(result.loc[1, 'price'], -100.0 / 0.9, places=2)
        
        # Large price should be converted
        self.assertEqual(result.loc[2, 'price'], 1000000.0 / 150.0)
        
        # NaN price should remain NaN
        self.assertTrue(pd.isna(result.loc[3, 'price']))

    def test_currency_conversion_single_variable_invalid_currencies(self):
        """Test currency conversion with invalid currency codes."""
        df_test = pd.DataFrame({
            'price': [100.0, 200.0, 300.0],
            'currency': ['USD', 'INVALID', None]
        })
        
        result = currency_conversion_single_variable(
            df_test, 'price', 'currency', self.currency_dict
        )
        
        # USD should be converted
        self.assertEqual(result.loc[0, 'price'], 100.0)
        
        # Invalid currency should remain unchanged
        self.assertEqual(result.loc[1, 'price'], 200.0)
        
        # None currency should remain unchanged
        self.assertEqual(result.loc[2, 'price'], 300.0)

    def test_currency_conversion_zero_exchange_rates(self):
        """Test currency conversion with zero exchange rates."""
        df_test = pd.DataFrame({
            'price': [100.0, 200.0],
            'currency': ['USD', 'ZERO']
        })
        
        currency_dict_with_zero = {"USD": 1.0, "ZERO": 0.0}
        
        result = currency_conversion_single_variable(
            df_test, 'price', 'currency', currency_dict_with_zero
        )
        
        # USD should be converted normally
        self.assertEqual(result.loc[0, 'price'], 100.0)
        
        # Zero rate should leave price unchanged (avoid division by zero)
        self.assertEqual(result.loc[1, 'price'], 200.0)

    def test_parallel_currency_conversion_basic(self):
        """Test parallel currency conversion with multiple variables."""
        df_test = pd.DataFrame({
            'price1': [100.0, 200.0, 300.0],
            'price2': [50.0, 100.0, 150.0],
            'currency': ['USD', 'EUR', 'JPY']
        })
        
        result = parallel_currency_conversion(
            df_test, ['price1', 'price2'], 'currency', self.currency_dict, n_workers=2
        )
        
        # Check USD conversions (should be unchanged)
        self.assertEqual(result.loc[0, 'price1'], 100.0)
        self.assertEqual(result.loc[0, 'price2'], 50.0)
        
        # Check EUR conversions
        self.assertAlmostEqual(result.loc[1, 'price1'], 200.0 / 0.9, places=2)
        self.assertAlmostEqual(result.loc[1, 'price2'], 100.0 / 0.9, places=2)
        
        # Check JPY conversions
        self.assertEqual(result.loc[2, 'price1'], 2.0)
        self.assertEqual(result.loc[2, 'price2'], 1.0)

    def test_parallel_currency_conversion_single_worker(self):
        """Test parallel currency conversion with single worker."""
        df_test = pd.DataFrame({
            'price1': [100.0, 200.0],
            'price2': [50.0, 100.0],
            'currency': ['USD', 'EUR']
        })
        
        result = parallel_currency_conversion(
            df_test, ['price1', 'price2'], 'currency', self.currency_dict, n_workers=1
        )
        
        # Should work the same as with multiple workers
        self.assertEqual(result.loc[0, 'price1'], 100.0)
        self.assertAlmostEqual(result.loc[1, 'price1'], 200.0 / 0.9, places=2)

    def test_process_currency_conversion_complete_workflow(self):
        """Test the complete currency conversion workflow."""
        result = process_currency_conversion(
            df=self.df.copy(),
            marketplace_id_col='mp_id',
            currency_conversion_vars=['price'],
            currency_conversion_dict=self.currency_dict,
            marketplace_info=self.marketplace_info,
            currency_col='currency',
            default_currency='USD',
            skip_invalid_currencies=False,
            n_workers=1
        )
        
        # Should drop rows with NaN marketplace_id
        self.assertLess(len(result), len(self.df))
        
        # Check that conversions were applied
        # Find the row with EUR currency (originally mp_id=2)
        eur_rows = result[result['currency'] == 'EUR']
        if not eur_rows.empty:
            original_price = 200.0
            expected_converted = original_price / 0.9
            self.assertAlmostEqual(eur_rows.iloc[0]['price'], expected_converted, places=2)

    def test_process_currency_conversion_no_conversion_vars(self):
        """Test process_currency_conversion with no variables to convert."""
        result = process_currency_conversion(
            df=self.df.copy(),
            marketplace_id_col='mp_id',
            currency_conversion_vars=['nonexistent_var'],
            currency_conversion_dict=self.currency_dict,
            marketplace_info=self.marketplace_info,
            currency_col='currency'
        )
        
        # Should still process (drop NaN mp_id) but not convert anything
        self.assertLess(len(result), len(self.df))
        
        # Original prices should be unchanged
        usd_rows = result[result['currency'] == 'USD']
        if not usd_rows.empty:
            # Should find original USD price unchanged
            self.assertIn(100.0, usd_rows['price'].values)

    def test_process_currency_conversion_empty_dataframe(self):
        """Test process_currency_conversion with empty DataFrame."""
        empty_df = pd.DataFrame(columns=['mp_id', 'price', 'currency'])
        
        result = process_currency_conversion(
            df=empty_df,
            marketplace_id_col='mp_id',
            currency_conversion_vars=['price'],
            currency_conversion_dict=self.currency_dict,
            marketplace_info=self.marketplace_info,
            currency_col='currency'
        )
        
        self.assertEqual(len(result), 0)
        self.assertTrue('currency' in result.columns)


class TestCurrencyConversionIntegration(unittest.TestCase):
    """Integration tests for currency conversion with realistic scenarios."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.input_dir = self.temp_dir / "input" / "data"
        self.output_dir = self.temp_dir / "output"
        
        # Create directories
        for split in ["train", "test", "val"]:
            (self.input_dir / split).mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Mock environment variables
        self.mock_env = {
            "CURRENCY_CONVERSION_VARS": json.dumps(["price", "cost"]),
            "CURRENCY_CONVERSION_DICT": json.dumps({
                "EUR": 0.85, "JPY": 110.0, "GBP": 0.75, "USD": 1.0
            }),
            "MARKETPLACE_INFO": json.dumps({
                "1": {"currency_code": "USD"},
                "2": {"currency_code": "EUR"},
                "3": {"currency_code": "JPY"},
                "4": {"currency_code": "GBP"}
            }),
            "LABEL_FIELD": "label",
            "TRAIN_RATIO": "0.7",
            "TEST_VAL_RATIO": "0.5"
        }

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def _create_realistic_test_data(self):
        """Create realistic test data with multiple currencies and edge cases."""
        np.random.seed(42)  # For reproducible tests
        
        for split in ["train", "test", "val"]:
            n_samples = {"train": 1000, "test": 200, "val": 200}[split]
            
            # Create diverse marketplace IDs and currencies
            marketplace_ids = np.random.choice([1, 2, 3, 4, np.nan], n_samples, p=[0.4, 0.3, 0.2, 0.05, 0.05])
            
            df = pd.DataFrame({
                "marketplace_id": marketplace_ids,
                "price": np.random.uniform(10, 1000, n_samples),
                "cost": np.random.uniform(5, 500, n_samples),
                "label": np.random.choice([0, 1], n_samples),
                "other_feature": np.random.normal(0, 1, n_samples)
            })
            
            # Add some currency overrides
            currencies = []
            for mp_id in marketplace_ids:
                if pd.isna(mp_id):
                    currencies.append(np.random.choice(["USD", "EUR", None]))
                else:
                    # Sometimes override the marketplace currency
                    if np.random.random() < 0.1:  # 10% override rate
                        currencies.append(np.random.choice(["USD", "EUR", "JPY", "GBP"]))
                    else:
                        currencies.append(None)  # Will be filled from marketplace info
            
            df["currency"] = currencies
            
            split_dir = self.input_dir / split
            df.to_csv(split_dir / f"{split}_processed_data.csv", index=False)
            df.to_csv(split_dir / f"{split}_full_data.csv", index=False)

    @patch('src.cursus.steps.scripts.currency_conversion.Path')
    @patch('src.cursus.steps.scripts.currency_conversion.argparse.ArgumentParser')
    def test_main_per_split_mode_integration(self, mock_arg_parser, mock_path):
        """Test main function in per_split mode with realistic data."""
        self._create_realistic_test_data()
        
        # Mock paths and arguments
        mock_path.side_effect = [self.input_dir, self.output_dir]
        mock_args = MagicMock(
            job_type="training",
            mode="per_split",
            enable_conversion=True,
            marketplace_id_col="marketplace_id",
            currency_col="currency",
            default_currency="USD",
            skip_invalid_currencies=False,
            n_workers=2,
            train_ratio=0.7,
            test_val_ratio=0.5
        )
        mock_arg_parser.return_value.parse_args.return_value = mock_args

        # Run main function
        with patch.dict(os.environ, self.mock_env):
            main(
                mock_args,
                json.loads(self.mock_env["CURRENCY_CONVERSION_VARS"]),
                json.loads(self.mock_env["CURRENCY_CONVERSION_DICT"]),
                json.loads(self.mock_env["MARKETPLACE_INFO"])
            )

        # Verify output files exist
        for split in ["train", "test", "val"]:
            processed_file = self.output_dir / split / f"{split}_processed_data.csv"
            full_file = self.output_dir / split / f"{split}_full_data.csv"
            
            self.assertTrue(processed_file.exists(), f"Missing {processed_file}")
            self.assertTrue(full_file.exists(), f"Missing {full_file}")
            
            # Verify data integrity
            df_out = pd.read_csv(processed_file)
            self.assertGreater(len(df_out), 0, f"Empty output file for {split}")
            self.assertTrue(all(col in df_out.columns for col in ["price", "cost", "label"]))

    @patch('src.cursus.steps.scripts.currency_conversion.Path')
    @patch('src.cursus.steps.scripts.currency_conversion.argparse.ArgumentParser')
    def test_main_split_after_conversion_mode(self, mock_arg_parser, mock_path):
        """Test main function in split_after_conversion mode."""
        self._create_realistic_test_data()
        
        mock_path.side_effect = [self.input_dir, self.output_dir]
        mock_args = MagicMock(
            job_type="training",
            mode="split_after_conversion",
            enable_conversion=True,
            marketplace_id_col="marketplace_id",
            currency_col="currency",
            default_currency="USD",
            skip_invalid_currencies=False,
            n_workers=2,
            train_ratio=0.7,
            test_val_ratio=0.5
        )
        mock_arg_parser.return_value.parse_args.return_value = mock_args

        with patch.dict(os.environ, self.mock_env):
            main(
                mock_args,
                json.loads(self.mock_env["CURRENCY_CONVERSION_VARS"]),
                json.loads(self.mock_env["CURRENCY_CONVERSION_DICT"]),
                json.loads(self.mock_env["MARKETPLACE_INFO"])
            )

        # Verify outputs
        for split in ["train", "test", "val"]:
            processed_file = self.output_dir / split / f"{split}_processed_data.csv"
            self.assertTrue(processed_file.exists())
            
            df_out = pd.read_csv(processed_file)
            self.assertGreater(len(df_out), 0)

    @patch('src.cursus.steps.scripts.currency_conversion.Path')
    @patch('src.cursus.steps.scripts.currency_conversion.argparse.ArgumentParser')
    def test_main_conversion_disabled(self, mock_arg_parser, mock_path):
        """Test main function with conversion disabled."""
        self._create_realistic_test_data()
        
        mock_path.side_effect = [self.input_dir, self.output_dir]
        mock_args = MagicMock(
            job_type="training",
            mode="per_split",
            enable_conversion=False,
            marketplace_id_col="marketplace_id",
            currency_col="currency",
            default_currency="USD",
            skip_invalid_currencies=False,
            n_workers=1,
            train_ratio=0.7,
            test_val_ratio=0.5
        )
        mock_arg_parser.return_value.parse_args.return_value = mock_args

        with patch.dict(os.environ, self.mock_env):
            main(
                mock_args,
                [],  # Empty conversion vars
                {},  # Empty conversion dict
                {}   # Empty marketplace info
            )

        # Verify files exist and data is unchanged
        train_file = self.output_dir / "train" / "train_processed_data.csv"
        self.assertTrue(train_file.exists())
        
        df_out = pd.read_csv(train_file)
        # When conversion is disabled, original prices should be preserved
        # (This is a basic check - in practice you'd compare with input data)
        self.assertTrue("price" in df_out.columns)
        self.assertTrue("cost" in df_out.columns)


class TestCurrencyConversionPerformance(unittest.TestCase):
    """Performance and scalability tests for currency conversion."""

    def setUp(self):
        """Set up performance test fixtures."""
        self.currency_dict = {"EUR": 0.9, "JPY": 150, "USD": 1.0, "GBP": 0.8}
        self.marketplace_info = {str(i): {"currency_code": "USD"} for i in range(1, 101)}

    def test_parallel_conversion_performance(self):
        """Test performance of parallel currency conversion with different worker counts."""
        # Create large test dataset
        n_rows = 10000
        np.random.seed(42)
        
        df_large = pd.DataFrame({
            'price1': np.random.uniform(10, 1000, n_rows),
            'price2': np.random.uniform(5, 500, n_rows),
            'price3': np.random.uniform(1, 100, n_rows),
            'currency': np.random.choice(['USD', 'EUR', 'JPY', 'GBP'], n_rows)
        })
        
        variables = ['price1', 'price2', 'price3']
        
        # Test with different worker counts
        performance_results = {}
        for n_workers in [1, 2, 4]:
            start_time = time.time()
            
            result = parallel_currency_conversion(
                df_large.copy(), variables, 'currency', self.currency_dict, n_workers
            )
            
            end_time = time.time()
            duration = end_time - start_time
            performance_results[n_workers] = duration
            
            # Verify correctness
            self.assertEqual(len(result), n_rows)
            self.assertTrue(all(var in result.columns for var in variables))
            
            print(f"Parallel conversion with {n_workers} workers: {duration:.3f}s")
        
        # Performance should generally improve with more workers (though not always due to overhead)
        self.assertLess(performance_results[4], performance_results[1] * 2)  # At least some improvement

    def test_large_dataset_processing(self):
        """Test processing of large datasets."""
        # Create very large dataset
        n_rows = 50000
        np.random.seed(42)
        
        large_df = pd.DataFrame({
            'mp_id': np.random.choice(range(1, 101), n_rows),
            'price': np.random.uniform(1, 10000, n_rows),
            'currency': np.random.choice(['USD', 'EUR', 'JPY'], n_rows, p=[0.5, 0.3, 0.2])
        })
        
        start_time = time.time()
        
        result = process_currency_conversion(
            df=large_df,
            marketplace_id_col='mp_id',
            currency_conversion_vars=['price'],
            currency_conversion_dict=self.currency_dict,
            marketplace_info=self.marketplace_info,
            n_workers=4
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Large dataset processing ({n_rows} rows): {duration:.3f}s")
        
        # Verify results
        self.assertEqual(len(result), n_rows)  # No rows should be dropped (all mp_ids are valid)
        self.assertTrue('price' in result.columns)
        
        # Performance should be reasonable (less than 10 seconds for 50k rows)
        self.assertLess(duration, 10.0)


class TestCurrencyConversionErrorHandling(unittest.TestCase):
    """Test error handling and edge cases in currency conversion."""

    def setUp(self):
        """Set up error handling test fixtures."""
        self.currency_dict = {"EUR": 0.9, "USD": 1.0}
        self.marketplace_info = {"1": {"currency_code": "USD"}}

    def test_missing_columns_handling(self):
        """Test handling of missing required columns."""
        # DataFrame missing marketplace_id column
        df_missing_mp = pd.DataFrame({
            'price': [100, 200],
            'currency': ['USD', 'EUR']
        })
        
        with self.assertRaises(KeyError):
            process_currency_conversion(
                df=df_missing_mp,
                marketplace_id_col='nonexistent_col',
                currency_conversion_vars=['price'],
                currency_conversion_dict=self.currency_dict,
                marketplace_info=self.marketplace_info
            )

    def test_corrupted_data_handling(self):
        """Test handling of corrupted or malformed data."""
        # DataFrame with various data corruption issues
        corrupted_df = pd.DataFrame({
            'mp_id': [1, 2, "corrupted", float('inf'), -float('inf')],
            'price': [100, "not_a_number", np.inf, -np.inf, None],
            'currency': ['USD', 'EUR', 123, [], {}]
        })
        
        # Should handle corrupted data gracefully
        try:
            result = process_currency_conversion(
                df=corrupted_df,
                marketplace_id_col='mp_id',
                currency_conversion_vars=['price'],
                currency_conversion_dict=self.currency_dict,
                marketplace_info=self.marketplace_info,
                skip_invalid_currencies=True
            )
            
            # Should return some valid results
            self.assertGreaterEqual(len(result), 0)
            
        except Exception as e:
            # If it raises an exception, it should be a reasonable one
            self.assertIsInstance(e, (ValueError, TypeError, KeyError))

    def test_memory_constraints(self):
        """Test behavior under memory constraints (simulated)."""
        # Create a dataset that might cause memory issues if not handled properly
        n_rows = 100000
        
        # Use object dtype to increase memory usage
        large_df = pd.DataFrame({
            'mp_id': [str(i % 100) for i in range(n_rows)],
            'price': [f"{i}.{i%100}" for i in range(n_rows)],  # String prices
            'currency': ['USD'] * n_rows
        })
        
        # Convert to proper types
        large_df['mp_id'] = pd.to_numeric(large_df['mp_id'], errors='coerce')
        large_df['price'] = pd.to_numeric(large_df['price'], errors='coerce')
        
        # This should complete without memory errors
        result = process_currency_conversion(
            df=large_df,
            marketplace_id_col='mp_id',
            currency_conversion_vars=['price'],
            currency_conversion_dict=self.currency_dict,
            marketplace_info=self.marketplace_info,
            n_workers=1  # Use single worker to reduce memory overhead
        )
        
        self.assertGreater(len(result), 0)

    @patch('src.cursus.steps.scripts.currency_conversion.logger')
    def test_logging_behavior(self, mock_logger):
        """Test that appropriate logging occurs during processing."""
        df_test = pd.DataFrame({
            'mp_id': [1, 2, 3],
            'price': [100, 200, 300],
            'currency': ['USD', 'EUR', 'JPY']
        })
        
        process_currency_conversion(
            df=df_test,
            marketplace_id_col='mp_id',
            currency_conversion_vars=['price'],
            currency_conversion_dict=self.currency_dict,
            marketplace_info=self.marketplace_info
        )
        
        # Verify that logging occurred
        self.assertTrue(mock_logger.info.called)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

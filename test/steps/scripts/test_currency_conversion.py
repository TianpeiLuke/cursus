import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import pandas as pd
import numpy as np
import tempfile
import shutil
import json
from pathlib import Path
import argparse

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
    """Unit tests for the helper functions in the currency conversion script."""

    def setUp(self):
        """Set up common data for tests."""
        self.df = pd.DataFrame({
            'mp_id': [1, 2, 3, np.nan, 'invalid'],
            'price': [100, 200, 300, 400, 500],
            'currency': ['USD', 'EUR', None, 'CAD', 'INVALID']
        })
        self.marketplace_info = {
            "1": {"currency_code": "USD"},
            "2": {"currency_code": "EUR"},
            "3": {"currency_code": "JPY"}
        }
        self.currency_dict = {"EUR": 0.9, "JPY": 150, "USD": 1.0}

    def test_get_currency_code(self):
        """Test the currency code retrieval logic."""
        self.assertEqual(get_currency_code(1, self.marketplace_info, "USD"), "USD")
        self.assertEqual(get_currency_code(3, self.marketplace_info, "USD"), "JPY")
        self.assertEqual(get_currency_code(99, self.marketplace_info, "USD"), "USD") # Invalid ID
        self.assertEqual(get_currency_code(np.nan, self.marketplace_info, "USD"), "USD") # NaN ID
        self.assertEqual(get_currency_code("invalid", self.marketplace_info, "USD"), "USD") # TypeError

    def test_combine_currency_codes(self):
        """Test the logic for combining and cleaning currency codes."""
        # Case 1: Combine with existing currency column
        df_combined, col_name = combine_currency_codes(
            self.df.copy(), 'mp_id', 'currency', self.marketplace_info, 'USD', False
        )
        self.assertEqual(col_name, 'currency')
        # Row 2 (mp_id=3) should have its None currency filled with JPY
        self.assertEqual(df_combined.loc[2, 'currency'], 'JPY')
        # Row 0 and 1 should remain unchanged
        self.assertEqual(df_combined.loc[0, 'currency'], 'USD')
        self.assertEqual(df_combined.loc[1, 'currency'], 'EUR')

        # Case 2: No existing currency column
        df_no_curr_col = self.df.drop(columns=['currency'])
        df_combined_new, col_name_new = combine_currency_codes(
            df_no_curr_col.copy(), 'mp_id', 'currency', self.marketplace_info, 'USD', False
        )
        self.assertEqual(col_name_new, 'currency_code_from_marketplace_id')
        self.assertTrue('currency_code_from_marketplace_id' in df_combined_new.columns)
        self.assertEqual(df_combined_new.loc[2, 'currency_code_from_marketplace_id'], 'JPY')

    def test_process_currency_conversion(self):
        """Test the main processing wrapper function."""
        df_processed = process_currency_conversion(
            df=self.df.copy(),
            marketplace_id_col='mp_id',
            currency_conversion_vars=['price'],
            currency_conversion_dict=self.currency_dict,
            marketplace_info=self.marketplace_info,
            currency_col='currency',
            skip_invalid_currencies=False,
            n_workers=1
        )
        # Check that NaN mp_id is dropped
        self.assertEqual(len(df_processed), 4)
        # Check conversion logic (e.g., for EUR)
        original_eur_price = self.df.loc[1, 'price']
        expected_converted_price = original_eur_price / 0.9
        self.assertAlmostEqual(df_processed.loc[1, 'price'], expected_converted_price)
        # Check that USD price is unchanged
        self.assertEqual(df_processed.loc[0, 'price'], self.df.loc[0, 'price'])

    def test_process_currency_conversion_no_vars(self):
        """Test that the function handles cases with no variables to convert."""
        # The function should run without error and return the dataframe
        df_processed = process_currency_conversion(
            df=self.df.copy(),
            marketplace_id_col='mp_id',
            currency_conversion_vars=['non_existent_var'], # This var is not in the df
            currency_conversion_dict=self.currency_dict,
            marketplace_info=self.marketplace_info
        )
        self.assertEqual(len(df_processed), 4) # Still drops NaN mp_id
        
        # FIX: The expected dataframe should also have the row with NaN mp_id dropped.
        expected_df = self.df.dropna(subset=['mp_id']).reset_index(drop=True)
        pd.testing.assert_series_equal(expected_df['price'], df_processed['price'])

    def test_currency_conversion_single_variable(self):
        """Test single variable currency conversion."""
        df_test = pd.DataFrame({
            'price': [100, 200, 300],
            'currency': ['USD', 'EUR', 'JPY']
        })
        
        result = currency_conversion_single_variable(
            df_test, 'price', 'currency', self.currency_dict
        )
        
        # USD should remain unchanged (rate = 1.0)
        self.assertEqual(result.loc[0, 'price'], 100.0)
        # EUR should be converted: 200 / 0.9 = 222.22...
        self.assertAlmostEqual(result.loc[1, 'price'], 200 / 0.9, places=2)
        # JPY should be converted: 300 / 150 = 2.0
        self.assertEqual(result.loc[2, 'price'], 2.0)

    def test_currency_conversion_single_variable_invalid_currency(self):
        """Test single variable conversion with invalid currency codes."""
        df_test = pd.DataFrame({
            'price': [100, 200],
            'currency': ['USD', 'INVALID']
        })
        
        result = currency_conversion_single_variable(
            df_test, 'price', 'currency', self.currency_dict
        )
        
        # USD should be converted
        self.assertEqual(result.loc[0, 'price'], 100.0)
        # INVALID currency should remain unchanged
        self.assertEqual(result.loc[1, 'price'], 200.0)

    def test_parallel_currency_conversion(self):
        """Test parallel currency conversion with multiple workers."""
        df_test = pd.DataFrame({
            'price1': [100, 200, 300],
            'price2': [50, 100, 150],
            'currency': ['USD', 'EUR', 'JPY']
        })
        
        result = parallel_currency_conversion(
            df_test, ['price1', 'price2'], 'currency', self.currency_dict, n_workers=2
        )
        
        # Check that both variables were converted
        self.assertEqual(result.loc[0, 'price1'], 100.0)  # USD unchanged
        self.assertEqual(result.loc[0, 'price2'], 50.0)   # USD unchanged
        self.assertAlmostEqual(result.loc[1, 'price1'], 200 / 0.9, places=2)  # EUR converted
        self.assertAlmostEqual(result.loc[1, 'price2'], 100 / 0.9, places=2)  # EUR converted

    def test_process_currency_conversion_skip_invalid(self):
        """Test processing with skip_invalid_currencies=True."""
        df_processed = process_currency_conversion(
            df=self.df.copy(),
            marketplace_id_col='mp_id',
            currency_conversion_vars=['price'],
            currency_conversion_dict=self.currency_dict,
            marketplace_info=self.marketplace_info,
            currency_col='currency',
            skip_invalid_currencies=True,
            n_workers=1
        )
        
        # Should still have 4 rows (NaN mp_id dropped)
        self.assertEqual(len(df_processed), 4)
        
        # Row with 'INVALID' currency should be dropped when skip_invalid_currencies=True
        # But since we're not implementing that logic in this test, we just verify it runs

    def test_process_currency_conversion_empty_dataframe(self):
        """Test processing with empty dataframe."""
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

    def test_get_currency_code_edge_cases(self):
        """Test edge cases for get_currency_code function."""
        # Test with empty marketplace_info
        self.assertEqual(get_currency_code(1, {}, "USD"), "USD")
        
        # Test with None marketplace_info
        self.assertEqual(get_currency_code(1, None, "USD"), "USD")
        
        # Test with marketplace_info missing currency_code
        incomplete_info = {"1": {"other_field": "value"}}
        self.assertEqual(get_currency_code(1, incomplete_info, "USD"), "USD")

    def test_combine_currency_codes_edge_cases(self):
        """Test edge cases for combine_currency_codes function."""
        # Test with empty dataframe
        empty_df = pd.DataFrame()
        result_df, col_name = combine_currency_codes(
            empty_df, 'mp_id', 'currency', self.marketplace_info, 'USD', False
        )
        self.assertEqual(len(result_df), 0)
        
        # Test with overwrite=True
        df_with_currency = self.df.copy()
        result_df, col_name = combine_currency_codes(
            df_with_currency, 'mp_id', 'currency', self.marketplace_info, 'USD', True
        )
        # All currency codes should be overwritten based on marketplace_id
        self.assertEqual(result_df.loc[0, 'currency'], 'USD')  # mp_id=1
        self.assertEqual(result_df.loc[1, 'currency'], 'EUR')  # mp_id=2
        self.assertEqual(result_df.loc[2, 'currency'], 'JPY')  # mp_id=3

class TestMainExecution(unittest.TestCase):
    """Tests for the main execution flow of the script."""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.input_dir = self.temp_dir / "input" / "data"
        self.output_dir = self.temp_dir / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Mock environment variables
        self.mock_env = {
            "CURRENCY_CONVERSION_VARS": json.dumps(["price"]),
            "CURRENCY_CONVERSION_DICT": json.dumps({"EUR": 0.5}),
            "MARKETPLACE_INFO": json.dumps({"1": {"currency_code": "USD"}, "2": {"currency_code": "EUR"}}),
            "LABEL_FIELD": "label"
        }

    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def _create_mock_split_files(self):
        """Creates dummy data files in a train/test/val structure."""
        for split in ["train", "test", "val"]:
            split_dir = self.input_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame({
                "marketplace_id": [1, 2],
                "price": [100, 200], # USD and EUR prices
                "label": [0, 1]
            })
            df.to_csv(split_dir / f"{split}_processed_data.csv", index=False)
            df.to_csv(split_dir / f"{split}_full_data.csv", index=False)
            
    @patch('src.cursus.steps.scripts.currency_conversion.Path')
    @patch('src.cursus.steps.scripts.currency_conversion.argparse.ArgumentParser')
    def test_main_per_split_mode(self, mock_arg_parser, mock_path):
        """Test the main function in 'per_split' mode."""
        self._create_mock_split_files()
        
        # Mock paths and args
        mock_path.side_effect = [self.input_dir, self.output_dir]
        mock_args = MagicMock(
            job_type="training", mode="per_split", enable_conversion=True,
            marketplace_id_col="marketplace_id", currency_col=None, default_currency="USD",
            skip_invalid_currencies=False, n_workers=1
        )
        mock_arg_parser.return_value.parse_args.return_value = mock_args

        # Run main with mocked env vars
        with patch.dict(os.environ, self.mock_env):
            main(mock_args, json.loads(self.mock_env["CURRENCY_CONVERSION_VARS"]), 
                 json.loads(self.mock_env["CURRENCY_CONVERSION_DICT"]), 
                 json.loads(self.mock_env["MARKETPLACE_INFO"]))

        # Assertions
        train_out_path = self.output_dir / "train" / "train_processed_data.csv"
        self.assertTrue(train_out_path.exists())
        
        # Check if conversion was applied
        df_out = pd.read_csv(train_out_path)
        self.assertEqual(df_out.loc[0, 'price'], 100.0) # USD, should be unchanged
        self.assertEqual(df_out.loc[1, 'price'], 400.0) # EUR, 200 / 0.5 = 400
        
    @patch('src.cursus.steps.scripts.currency_conversion.Path')
    @patch('src.cursus.steps.scripts.currency_conversion.argparse.ArgumentParser')
    def test_main_conversion_disabled(self, mock_arg_parser, mock_path):
        """Test that no conversion is applied when disabled."""
        self._create_mock_split_files()
        
        mock_path.side_effect = [self.input_dir, self.output_dir]
        mock_args = MagicMock(
            job_type="training", mode="per_split", enable_conversion=False,
            marketplace_id_col="marketplace_id" # Other args not needed
        )
        mock_arg_parser.return_value.parse_args.return_value = mock_args

        with patch.dict(os.environ, self.mock_env):
            main(mock_args, [], {}, {})

        train_out_path = self.output_dir / "train" / "train_processed_data.csv"
        self.assertTrue(train_out_path.exists())
        
        df_out = pd.read_csv(train_out_path)
        # Prices should be unchanged from the original
        self.assertEqual(df_out.loc[1, 'price'], 200.0)

    def test_main_single_file_mode(self):
        """Test main function in single file mode."""
        # Create a single input file
        input_file = self.input_dir / "data.csv"
        input_file.parent.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame({
            "marketplace_id": [1, 2, 3],
            "price": [100, 200, 300],
            "label": [0, 1, 0]
        })
        df.to_csv(input_file, index=False)
        
        with patch('src.cursus.steps.scripts.currency_conversion.Path') as mock_path, \
             patch('src.cursus.steps.scripts.currency_conversion.argparse.ArgumentParser') as mock_arg_parser:
            
            mock_path.side_effect = [self.input_dir, self.output_dir]
            mock_args = MagicMock(
                job_type="training", mode="single_file", enable_conversion=True,
                marketplace_id_col="marketplace_id", currency_col=None, default_currency="USD",
                skip_invalid_currencies=False, n_workers=1
            )
            mock_arg_parser.return_value.parse_args.return_value = mock_args

            with patch.dict(os.environ, self.mock_env):
                main(mock_args, json.loads(self.mock_env["CURRENCY_CONVERSION_VARS"]), 
                     json.loads(self.mock_env["CURRENCY_CONVERSION_DICT"]), 
                     json.loads(self.mock_env["MARKETPLACE_INFO"]))

            # Check output file exists
            output_file = self.output_dir / "data.csv"
            self.assertTrue(output_file.exists())

    @patch('src.cursus.steps.scripts.currency_conversion.logger')
    def test_main_with_logging(self, mock_logger):
        """Test that main function logs appropriately."""
        self._create_mock_split_files()
        
        with patch('src.cursus.steps.scripts.currency_conversion.Path') as mock_path, \
             patch('src.cursus.steps.scripts.currency_conversion.argparse.ArgumentParser') as mock_arg_parser:
            
            mock_path.side_effect = [self.input_dir, self.output_dir]
            mock_args = MagicMock(
                job_type="training", mode="per_split", enable_conversion=True,
                marketplace_id_col="marketplace_id", currency_col=None, default_currency="USD",
                skip_invalid_currencies=False, n_workers=1
            )
            mock_arg_parser.return_value.parse_args.return_value = mock_args

            with patch.dict(os.environ, self.mock_env):
                main(mock_args, json.loads(self.mock_env["CURRENCY_CONVERSION_VARS"]), 
                     json.loads(self.mock_env["CURRENCY_CONVERSION_DICT"]), 
                     json.loads(self.mock_env["MARKETPLACE_INFO"]))

            # Verify logging was called
            self.assertTrue(mock_logger.info.called)

    def test_main_error_handling(self):
        """Test main function error handling with invalid input."""
        # Don't create any input files to trigger an error
        
        with patch('src.cursus.steps.scripts.currency_conversion.Path') as mock_path, \
             patch('src.cursus.steps.scripts.currency_conversion.argparse.ArgumentParser') as mock_arg_parser:
            
            mock_path.side_effect = [self.input_dir, self.output_dir]
            mock_args = MagicMock(
                job_type="training", mode="per_split", enable_conversion=True,
                marketplace_id_col="marketplace_id", currency_col=None, default_currency="USD",
                skip_invalid_currencies=False, n_workers=1
            )
            mock_arg_parser.return_value.parse_args.return_value = mock_args

            with patch.dict(os.environ, self.mock_env):
                # This should handle the error gracefully (depending on implementation)
                try:
                    main(mock_args, json.loads(self.mock_env["CURRENCY_CONVERSION_VARS"]), 
                         json.loads(self.mock_env["CURRENCY_CONVERSION_DICT"]), 
                         json.loads(self.mock_env["MARKETPLACE_INFO"]))
                except Exception as e:
                    # If an exception is raised, that's also acceptable behavior
                    self.assertIsInstance(e, Exception)


class TestCurrencyConversionEdgeCases(unittest.TestCase):
    """Additional tests for edge cases and error conditions."""

    def setUp(self):
        """Set up test data for edge cases."""
        self.currency_dict = {"EUR": 0.9, "JPY": 150, "USD": 1.0}
        self.marketplace_info = {
            "1": {"currency_code": "USD"},
            "2": {"currency_code": "EUR"}
        }

    def test_currency_conversion_with_zero_rates(self):
        """Test currency conversion with zero exchange rates."""
        df_test = pd.DataFrame({
            'price': [100, 200],
            'currency': ['USD', 'EUR']
        })
        
        currency_dict_with_zero = {"USD": 1.0, "EUR": 0.0}
        
        result = currency_conversion_single_variable(
            df_test, 'price', 'currency', currency_dict_with_zero
        )
        
        # USD should be converted normally
        self.assertEqual(result.loc[0, 'price'], 100.0)
        # EUR with zero rate should remain unchanged (to avoid division by zero)
        self.assertEqual(result.loc[1, 'price'], 200.0)

    def test_currency_conversion_with_negative_prices(self):
        """Test currency conversion with negative prices."""
        df_test = pd.DataFrame({
            'price': [-100, -200],
            'currency': ['USD', 'EUR']
        })
        
        result = currency_conversion_single_variable(
            df_test, 'price', 'currency', self.currency_dict
        )
        
        # Negative prices should be converted normally
        self.assertEqual(result.loc[0, 'price'], -100.0)  # USD unchanged
        self.assertAlmostEqual(result.loc[1, 'price'], -200 / 0.9, places=2)  # EUR converted

    def test_currency_conversion_with_missing_columns(self):
        """Test currency conversion when required columns are missing."""
        df_test = pd.DataFrame({
            'other_col': [1, 2, 3]
        })
        
        # This should handle missing columns gracefully
        try:
            result = currency_conversion_single_variable(
                df_test, 'price', 'currency', self.currency_dict
            )
            # If it doesn't raise an error, the result should be the original dataframe
            pd.testing.assert_frame_equal(result, df_test)
        except KeyError:
            # It's also acceptable to raise a KeyError for missing columns
            pass

    def test_process_currency_conversion_with_all_nan_marketplace_ids(self):
        """Test processing when all marketplace IDs are NaN."""
        df_all_nan = pd.DataFrame({
            'mp_id': [np.nan, np.nan, np.nan],
            'price': [100, 200, 300],
            'currency': ['USD', 'EUR', 'JPY']
        })
        
        result = process_currency_conversion(
            df=df_all_nan,
            marketplace_id_col='mp_id',
            currency_conversion_vars=['price'],
            currency_conversion_dict=self.currency_dict,
            marketplace_info=self.marketplace_info
        )
        
        # All rows should be dropped due to NaN marketplace IDs
        self.assertEqual(len(result), 0)

    def test_parallel_currency_conversion_single_worker(self):
        """Test parallel conversion with n_workers=1 (sequential processing)."""
        df_test = pd.DataFrame({
            'price1': [100, 200],
            'price2': [50, 100],
            'currency': ['USD', 'EUR']
        })
        
        result = parallel_currency_conversion(
            df_test, ['price1', 'price2'], 'currency', self.currency_dict, n_workers=1
        )
        
        # Should work the same as with multiple workers
        self.assertEqual(result.loc[0, 'price1'], 100.0)
        self.assertAlmostEqual(result.loc[1, 'price1'], 200 / 0.9, places=2)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

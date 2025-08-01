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
from src.pipeline_scripts.currency_conversion import (
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
            
    @patch('src.pipeline_scripts.currency_conversion.Path')
    @patch('src.pipeline_scripts.currency_conversion.argparse.ArgumentParser')
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
        
    @patch('src.pipeline_scripts.currency_conversion.Path')
    @patch('src.pipeline_scripts.currency_conversion.argparse.ArgumentParser')
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

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
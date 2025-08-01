import unittest
from unittest.mock import patch
import os
import pandas as pd
import numpy as np
import tempfile
import shutil
import json
import pickle as pkl
from pathlib import Path


# Import the components to be tested
from src.pipeline_scripts.risk_table_mapping import (
    OfflineBinning,
    main as risk_mapping_main
)

class TestOfflineBinning(unittest.TestCase):
    """Tests the OfflineBinning class."""
    def setUp(self):
        self.cat_field_list = ['cat_var1', 'cat_var2']
        self.df = pd.DataFrame({
            'cat_var1': ['A', 'B', 'A', 'C', 'B'],
            'cat_var2': ['X', 'X', 'Y', 'Y', 'Z'],
            'num_var': [10, 20, 30, 40, 50],
            'target': [1, 0, 1, 0, 1]
        })
        self.target_field = 'target'
        self.binner = OfflineBinning(self.cat_field_list, self.target_field)

    def test_fit_creates_risk_tables(self):
        """Test that fitting creates the expected risk table structure."""
        self.binner.fit(self.df)
        self.assertIn('cat_var1', self.binner.risk_tables)
        self.assertIn('cat_var2', self.binner.risk_tables)
        self.assertNotIn('num_var', self.binner.risk_tables) # Should ignore numeric vars
        
        # Check content of a risk table
        cat1_bins = self.binner.risk_tables['cat_var1']['bins']
        self.assertIn('A', cat1_bins)
        self.assertAlmostEqual(cat1_bins['A'], 1.0) # 2 events / 2 total
        self.assertAlmostEqual(cat1_bins['B'], 0.5) # 1 event / 2 total

    def test_transform_maps_values(self):
        """Test that transform correctly maps categorical values to risk scores."""
        self.binner.fit(self.df)
        transformed_df = self.binner.transform(self.df)
        
        # Check if values are replaced by their risk scores
        self.assertNotEqual(transformed_df['cat_var1'].iloc[0], 'A')
        self.assertAlmostEqual(transformed_df['cat_var1'].iloc[0], 1.0) # Risk of 'A'
        self.assertAlmostEqual(transformed_df['cat_var1'].iloc[1], 0.5) # Risk of 'B'
        
        # Check that unseen values are mapped to the default risk
        test_df_unseen = pd.DataFrame({'cat_var1': ['D']})
        transformed_unseen = self.binner.transform(test_df_unseen)
        default_risk = self.binner.risk_tables['cat_var1']['default_bin']
        self.assertAlmostEqual(transformed_unseen['cat_var1'].iloc[0], default_risk)


class TestMainRiskTableFlow(unittest.TestCase):
    """Tests the main execution flow of the risk table mapping script."""
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.temp_dir, 'input')
        self.output_dir = os.path.join(self.temp_dir, 'output')
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Create dummy hyperparameters
        self.hyperparams = {
            "cat_field_list": ["cat_var"],
            "label_name": "target",
            "smooth_factor": 0.01,
            "count_threshold": 5
        }

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_main_training_mode(self):
        """Test the main logic in 'training' mode."""
        # Create split data files as expected by the new API
        for split in ['train', 'test', 'val']:
            split_dir = os.path.join(self.input_dir, split)
            os.makedirs(split_dir)
            df = pd.DataFrame({
                'cat_var': ['A', 'B', 'A', 'C', 'B'],
                'num_var': [10, 20, 30, 40, 50],
                'target': [1, 0, 1, 0, 1]
            })
            df.to_csv(os.path.join(split_dir, f'{split}_processed_data.csv'), index=False)

        # Run main function
        risk_mapping_main(
            job_type='training',
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            hyperparams=self.hyperparams
        )

        # Assertions
        train_path = os.path.join(self.output_dir, 'train', 'train_processed_data.csv')
        test_path = os.path.join(self.output_dir, 'test', 'test_processed_data.csv')
        val_path = os.path.join(self.output_dir, 'val', 'val_processed_data.csv')
        self.assertTrue(os.path.exists(train_path))
        self.assertTrue(os.path.exists(test_path))
        self.assertTrue(os.path.exists(val_path))

        # Check that artifacts were saved
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'bin_mapping.pkl')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'hyperparameters.json')))
        
        # Check content of transformed data
        train_df = pd.read_csv(train_path)
        self.assertIn('cat_var', train_df.columns)
        self.assertTrue(pd.api.types.is_numeric_dtype(train_df['cat_var'])) # Should be numeric after risk mapping

    def test_main_inference_mode(self):
        """Test the main logic in a non-training ('validation') mode."""
        # First, create training data and run training to generate risk tables
        # Need to create all required splits for training mode
        for split in ['train', 'test', 'val']:
            split_dir = os.path.join(self.input_dir, split)
            os.makedirs(split_dir)
            train_df = pd.DataFrame({
                'cat_var': ['A', 'B', 'A'],
                'target': [1, 0, 1]
            })
            train_df.to_csv(os.path.join(split_dir, f'{split}_processed_data.csv'), index=False)
        
        # Create a temporary directory for risk tables
        risk_table_dir = os.path.join(self.temp_dir, 'risk_tables')
        os.makedirs(risk_table_dir)
        
        # Generate risk tables by running training mode first
        risk_mapping_main(
            job_type='training',
            input_dir=self.input_dir,
            output_dir=risk_table_dir,
            hyperparams=self.hyperparams
        )
        
        # Now create validation data
        val_input_dir = os.path.join(self.temp_dir, 'val_input')
        os.makedirs(val_input_dir)
        val_dir = os.path.join(val_input_dir, 'validation')
        os.makedirs(val_dir)
        val_df = pd.DataFrame({
            'cat_var': ['A', 'B', 'C'],
            'target': [1, 0, 1]
        })
        val_df.to_csv(os.path.join(val_dir, 'validation_processed_data.csv'), index=False)
            
        # Run main function in validation mode
        risk_mapping_main(
            job_type='validation',
            input_dir=val_input_dir,
            output_dir=self.output_dir,
            hyperparams=self.hyperparams,
            risk_table_input_dir=risk_table_dir
        )
        
        # Assertions
        val_output_path = os.path.join(self.output_dir, 'validation', 'validation_processed_data.csv')
        self.assertTrue(os.path.exists(val_output_path))
        
        # Check that the validation data was transformed based on the train data
        with open(os.path.join(risk_table_dir, 'bin_mapping.pkl'), 'rb') as f:
            bins = pkl.load(f)
        
        # Check if validation data was transformed using the risk tables
        val_df_output = pd.read_csv(val_output_path)
        self.assertTrue(pd.api.types.is_numeric_dtype(val_df_output['cat_var']))

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

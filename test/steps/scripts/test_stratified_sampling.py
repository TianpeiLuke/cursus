#!/usr/bin/env python
"""
Test script for stratified_sampling.py implementation.
"""
import pandas as pd
import numpy as np
import tempfile
import os
import sys
from pathlib import Path
import argparse

# Add src to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from cursus.steps.scripts.stratified_sampling import main, StratifiedSampler

def create_test_data():
    """Create sample test data with class imbalance."""
    np.random.seed(42)
    
    # Create imbalanced dataset
    n_samples = 1000
    
    # Class 0: 70% of data
    class_0_size = int(0.7 * n_samples)
    class_0_data = {
        'feature1': np.random.normal(0, 1, class_0_size),
        'feature2': np.random.normal(2, 1.5, class_0_size),
        'label': [0] * class_0_size
    }
    
    # Class 1: 25% of data  
    class_1_size = int(0.25 * n_samples)
    class_1_data = {
        'feature1': np.random.normal(1, 1, class_1_size),
        'feature2': np.random.normal(-1, 1, class_1_size),
        'label': [1] * class_1_size
    }
    
    # Class 2: 5% of data (minority class)
    class_2_size = n_samples - class_0_size - class_1_size
    class_2_data = {
        'feature1': np.random.normal(-1, 0.5, class_2_size),
        'feature2': np.random.normal(0, 2, class_2_size),
        'label': [2] * class_2_size
    }
    
    # Combine all classes
    df = pd.DataFrame({
        'feature1': np.concatenate([class_0_data['feature1'], class_1_data['feature1'], class_2_data['feature1']]),
        'feature2': np.concatenate([class_0_data['feature2'], class_1_data['feature2'], class_2_data['feature2']]),
        'label': class_0_data['label'] + class_1_data['label'] + class_2_data['label']
    })
    
    return df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

def test_stratified_sampler():
    """Test the StratifiedSampler class directly."""
    print("Testing StratifiedSampler class...")
    
    # Create test data
    df = create_test_data()
    print(f"Original data shape: {df.shape}")
    print(f"Original class distribution:\n{df['label'].value_counts().sort_index()}")
    
    sampler = StratifiedSampler(random_state=42)
    
    # Test balanced allocation
    print("\n--- Testing Balanced Allocation ---")
    balanced_sample = sampler.sample(
        df=df,
        strata_column='label',
        target_size=300,
        strategy='balanced',
        min_samples_per_stratum=10
    )
    print(f"Balanced sample shape: {balanced_sample.shape}")
    print(f"Balanced class distribution:\n{balanced_sample['label'].value_counts().sort_index()}")
    
    # Test proportional with minimum
    print("\n--- Testing Proportional with Minimum ---")
    prop_sample = sampler.sample(
        df=df,
        strata_column='label',
        target_size=300,
        strategy='proportional_min',
        min_samples_per_stratum=20
    )
    print(f"Proportional sample shape: {prop_sample.shape}")
    print(f"Proportional class distribution:\n{prop_sample['label'].value_counts().sort_index()}")
    
    # Test optimal allocation
    print("\n--- Testing Optimal Allocation ---")
    optimal_sample = sampler.sample(
        df=df,
        strata_column='label',
        target_size=300,
        strategy='optimal',
        min_samples_per_stratum=10,
        variance_column='feature1'
    )
    print(f"Optimal sample shape: {optimal_sample.shape}")
    print(f"Optimal class distribution:\n{optimal_sample['label'].value_counts().sort_index()}")

def test_main_function():
    """Test the main function with simulated file structure."""
    print("\n" + "="*50)
    print("Testing main function with file I/O...")
    
    # Create test data
    df = create_test_data()
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create input structure (simulating tabular_preprocessing output)
        input_dir = temp_path / "input"
        train_dir = input_dir / "train"
        val_dir = input_dir / "val"
        test_dir = input_dir / "test"
        
        train_dir.mkdir(parents=True)
        val_dir.mkdir(parents=True)
        test_dir.mkdir(parents=True)
        
        # Split data into train/val/test
        train_df = df.iloc[:600].copy()
        val_df = df.iloc[600:800].copy()
        test_df = df.iloc[800:].copy()
        
        # Save data files
        train_df.to_csv(train_dir / "train_processed_data.csv", index=False)
        val_df.to_csv(val_dir / "val_processed_data.csv", index=False)
        test_df.to_csv(test_dir / "test_processed_data.csv", index=False)
        
        # Create output directory
        output_dir = temp_path / "output"
        
        # Test training job type
        print("\n--- Testing Training Job Type ---")
        
        # Create mock arguments
        args = argparse.Namespace(job_type="training")
        
        # Environment variables
        environ_vars = {
            "STRATA_COLUMN": "label",
            "SAMPLING_STRATEGY": "balanced",
            "TARGET_SAMPLE_SIZE": "150",
            "MIN_SAMPLES_PER_STRATUM": "10",
            "VARIANCE_COLUMN": "feature1",
            "RANDOM_STATE": "42"
        }
        
        # Path dictionaries
        input_paths = {"data_input": str(input_dir)}
        output_paths = {"data_output": str(output_dir)}
        
        # Run main function
        result = main(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,
            job_args=args,
            logger=print
        )
        
        print(f"Result splits: {list(result.keys())}")
        for split_name, split_df in result.items():
            print(f"{split_name}: {split_df.shape}, class distribution: {dict(split_df['label'].value_counts().sort_index())}")
        
        # Verify output files exist
        for split_name in ["train", "val", "test"]:
            output_file = output_dir / split_name / f"{split_name}_processed_data.csv"
            if output_file.exists():
                print(f"✓ Output file exists: {output_file}")
            else:
                print(f"✗ Output file missing: {output_file}")

def test_non_training_job_type():
    """Test non-training job types (validation, testing, calibration)."""
    print("\n--- Testing Non-Training Job Type (validation) ---")
    
    # Create test data
    df = create_test_data()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create input structure for validation job
        input_dir = temp_path / "input"
        val_dir = input_dir / "validation"
        val_dir.mkdir(parents=True)
        
        # Save validation data
        df.to_csv(val_dir / "validation_processed_data.csv", index=False)
        
        # Create output directory
        output_dir = temp_path / "output"
        
        # Create mock arguments for validation job
        args = argparse.Namespace(job_type="validation")
        
        # Environment variables
        environ_vars = {
            "STRATA_COLUMN": "label",
            "SAMPLING_STRATEGY": "proportional_min",
            "TARGET_SAMPLE_SIZE": "200",
            "MIN_SAMPLES_PER_STRATUM": "15",
            "RANDOM_STATE": "42"
        }
        
        # Path dictionaries
        input_paths = {"data_input": str(input_dir)}
        output_paths = {"data_output": str(output_dir)}
        
        # Run main function
        result = main(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,
            job_args=args,
            logger=print
        )
        
        print(f"Result splits: {list(result.keys())}")
        for split_name, split_df in result.items():
            print(f"{split_name}: {split_df.shape}, class distribution: {dict(split_df['label'].value_counts().sort_index())}")

if __name__ == "__main__":
    print("Running Stratified Sampling Tests")
    print("=" * 50)
    
    try:
        # Test the sampler class
        test_stratified_sampler()
        
        # Test the main function with training job type
        test_main_function()
        
        # Test non-training job type
        test_non_training_job_type()
        
        print("\n" + "="*50)
        print("All tests completed successfully! ✓")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
import argparse

from cursus.steps.scripts.stratified_sampling import main, StratifiedSampler


class TestStratifiedSampler:
    """Tests for the StratifiedSampler class."""

    @pytest.fixture
    def sample_data(self):
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

    def test_stratified_sampler_balanced(self, sample_data):
        """Test the StratifiedSampler with balanced allocation."""
        df = sample_data
        
        sampler = StratifiedSampler(random_state=42)
        
        # Test balanced allocation
        balanced_sample = sampler.sample(
            df=df,
            strata_column='label',
            target_size=300,
            strategy='balanced',
            min_samples_per_stratum=10
        )
        
        assert balanced_sample.shape[0] == 300
        
        # Check that all classes are represented
        class_counts = balanced_sample['label'].value_counts().sort_index()
        assert len(class_counts) == 3  # All 3 classes should be present
        
        # In balanced sampling, each class should have roughly equal representation
        # (accounting for minimum samples constraint)
        assert all(count >= 10 for count in class_counts.values)

    def test_stratified_sampler_proportional(self, sample_data):
        """Test the StratifiedSampler with proportional allocation."""
        df = sample_data
        
        sampler = StratifiedSampler(random_state=42)
        
        # Test proportional with minimum
        prop_sample = sampler.sample(
            df=df,
            strata_column='label',
            target_size=300,
            strategy='proportional_min',
            min_samples_per_stratum=20
        )
        
        assert prop_sample.shape[0] >= 300  # May be more due to minimum constraints
        assert prop_sample.shape[0] <= 320  # Allow some tolerance for minimum constraints
        
        # Check that all classes are represented with at least minimum samples
        class_counts = prop_sample['label'].value_counts().sort_index()
        assert len(class_counts) == 3  # All 3 classes should be present
        assert all(count >= 20 for count in class_counts.values)

    def test_stratified_sampler_optimal(self, sample_data):
        """Test the StratifiedSampler with optimal allocation."""
        df = sample_data
        
        sampler = StratifiedSampler(random_state=42)
        
        # Test optimal allocation
        optimal_sample = sampler.sample(
            df=df,
            strata_column='label',
            target_size=300,
            strategy='optimal',
            min_samples_per_stratum=10,
            variance_column='feature1'
        )
        
        assert optimal_sample.shape[0] >= 300  # May be more due to minimum constraints
        assert optimal_sample.shape[0] <= 320  # Allow some tolerance for minimum constraints
        
        # Check that all classes are represented
        class_counts = optimal_sample['label'].value_counts().sort_index()
        assert len(class_counts) == 3  # All 3 classes should be present
        assert all(count >= 10 for count in class_counts.values)

    def test_stratified_sampler_invalid_strategy(self, sample_data):
        """Test that invalid strategy raises an error."""
        df = sample_data
        
        sampler = StratifiedSampler(random_state=42)
        
        with pytest.raises(ValueError):
            sampler.sample(
                df=df,
                strata_column='label',
                target_size=300,
                strategy='invalid_strategy',
                min_samples_per_stratum=10
            )

    def test_stratified_sampler_missing_column(self, sample_data):
        """Test that missing strata column raises an error."""
        df = sample_data
        
        sampler = StratifiedSampler(random_state=42)
        
        with pytest.raises(KeyError):
            sampler.sample(
                df=df,
                strata_column='nonexistent_column',
                target_size=300,
                strategy='balanced',
                min_samples_per_stratum=10
            )


class TestStratifiedSamplingMain:
    """Tests for the main function of stratified sampling."""

    @pytest.fixture
    def sample_data(self):
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

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_main_training_job_type(self, sample_data, temp_dir):
        """Test the main function with training job type."""
        df = sample_data
        
        # Create input structure (simulating tabular_preprocessing output)
        input_dir = temp_dir / "input"
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
        output_dir = temp_dir / "output"
        
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
            job_args=args
        )
        
        # Verify results
        assert isinstance(result, dict)
        assert "train" in result
        assert "val" in result
        assert "test" in result
        
        # Check that all splits have the expected structure
        for split_name, split_df in result.items():
            assert isinstance(split_df, pd.DataFrame)
            assert 'label' in split_df.columns
            assert len(split_df) > 0
            
            # Check class distribution
            class_counts = split_df['label'].value_counts()
            assert len(class_counts) >= 1  # At least one class should be present
        
        # Verify output files exist
        for split_name in ["train", "val", "test"]:
            output_file = output_dir / split_name / f"{split_name}_processed_data.csv"
            assert output_file.exists(), f"Output file missing: {output_file}"

    def test_main_validation_job_type(self, sample_data, temp_dir):
        """Test the main function with validation job type."""
        df = sample_data
        
        # Create input structure for validation job
        input_dir = temp_dir / "input"
        val_dir = input_dir / "validation"
        val_dir.mkdir(parents=True)
        
        # Save validation data
        df.to_csv(val_dir / "validation_processed_data.csv", index=False)
        
        # Create output directory
        output_dir = temp_dir / "output"
        
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
            job_args=args
        )
        
        # Verify results
        assert isinstance(result, dict)
        assert "validation" in result
        assert len(result) == 1  # Only validation split should be present
        
        # Check validation split
        val_split = result["validation"]
        assert isinstance(val_split, pd.DataFrame)
        assert 'label' in val_split.columns
        assert len(val_split) > 0
        
        # Check that minimum samples per stratum is respected
        class_counts = val_split['label'].value_counts()
        assert all(count >= 15 for count in class_counts.values)

    def test_main_missing_strata_column(self, sample_data, temp_dir):
        """Test main function with missing strata column."""
        df = sample_data.drop('label', axis=1)  # Remove label column
        
        # Create input structure
        input_dir = temp_dir / "input"
        train_dir = input_dir / "train"
        train_dir.mkdir(parents=True)
        
        # Save data without label column
        df.to_csv(train_dir / "train_processed_data.csv", index=False)
        
        # Create output directory
        output_dir = temp_dir / "output"
        
        # Create mock arguments
        args = argparse.Namespace(job_type="training")
        
        # Environment variables
        environ_vars = {
            "STRATA_COLUMN": "label",  # This column doesn't exist
            "SAMPLING_STRATEGY": "balanced",
            "TARGET_SAMPLE_SIZE": "150",
            "MIN_SAMPLES_PER_STRATUM": "10",
            "RANDOM_STATE": "42"
        }
        
        # Path dictionaries
        input_paths = {"data_input": str(input_dir)}
        output_paths = {"data_output": str(output_dir)}
        
        # Run main function and expect an error
        with pytest.raises(RuntimeError):
            main(
                input_paths=input_paths,
                output_paths=output_paths,
                environ_vars=environ_vars,
                job_args=args
            )

    def test_main_invalid_sampling_strategy(self, sample_data, temp_dir):
        """Test main function with invalid sampling strategy."""
        df = sample_data
        
        # Create input structure
        input_dir = temp_dir / "input"
        train_dir = input_dir / "train"
        train_dir.mkdir(parents=True)
        
        # Save data
        df.to_csv(train_dir / "train_processed_data.csv", index=False)
        
        # Create output directory
        output_dir = temp_dir / "output"
        
        # Create mock arguments
        args = argparse.Namespace(job_type="training")
        
        # Environment variables with invalid strategy
        environ_vars = {
            "STRATA_COLUMN": "label",
            "SAMPLING_STRATEGY": "invalid_strategy",  # Invalid strategy
            "TARGET_SAMPLE_SIZE": "150",
            "MIN_SAMPLES_PER_STRATUM": "10",
            "RANDOM_STATE": "42"
        }
        
        # Path dictionaries
        input_paths = {"data_input": str(input_dir)}
        output_paths = {"data_output": str(output_dir)}
        
        # Run main function and expect an error
        with pytest.raises(RuntimeError):
            main(
                input_paths=input_paths,
                output_paths=output_paths,
                environ_vars=environ_vars,
                job_args=args
            )

    def test_main_missing_input_files(self, temp_dir):
        """Test main function with missing input files."""
        # Create empty input structure
        input_dir = temp_dir / "input"
        train_dir = input_dir / "train"
        train_dir.mkdir(parents=True)
        # Don't create any data files
        
        # Create output directory
        output_dir = temp_dir / "output"
        
        # Create mock arguments
        args = argparse.Namespace(job_type="training")
        
        # Environment variables
        environ_vars = {
            "STRATA_COLUMN": "label",
            "SAMPLING_STRATEGY": "balanced",
            "TARGET_SAMPLE_SIZE": "150",
            "MIN_SAMPLES_PER_STRATUM": "10",
            "RANDOM_STATE": "42"
        }
        
        # Path dictionaries
        input_paths = {"data_input": str(input_dir)}
        output_paths = {"data_output": str(output_dir)}
        
        # Run main function and expect an error
        with pytest.raises(RuntimeError):
            main(
                input_paths=input_paths,
                output_paths=output_paths,
                environ_vars=environ_vars,
                job_args=args
            )

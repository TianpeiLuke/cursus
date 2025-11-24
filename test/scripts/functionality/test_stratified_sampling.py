import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
import argparse
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from cursus.steps.scripts.stratified_sampling import (
    main,
    StratifiedSampler,
    _read_processed_data,
    _save_sampled_data,
)


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
            "feature1": np.random.normal(0, 1, class_0_size),
            "feature2": np.random.normal(2, 1.5, class_0_size),
            "variance_col": np.random.normal(1, 2, class_0_size),
            "label": [0] * class_0_size,
        }

        # Class 1: 25% of data
        class_1_size = int(0.25 * n_samples)
        class_1_data = {
            "feature1": np.random.normal(1, 1, class_1_size),
            "feature2": np.random.normal(-1, 1, class_1_size),
            "variance_col": np.random.normal(3, 1, class_1_size),
            "label": [1] * class_1_size,
        }

        # Class 2: 5% of data (minority class)
        class_2_size = n_samples - class_0_size - class_1_size
        class_2_data = {
            "feature1": np.random.normal(-1, 0.5, class_2_size),
            "feature2": np.random.normal(0, 2, class_2_size),
            "variance_col": np.random.normal(5, 0.5, class_2_size),
            "label": [2] * class_2_size,
        }

        # Combine all classes
        df = pd.DataFrame(
            {
                "feature1": np.concatenate(
                    [
                        class_0_data["feature1"],
                        class_1_data["feature1"],
                        class_2_data["feature1"],
                    ]
                ),
                "feature2": np.concatenate(
                    [
                        class_0_data["feature2"],
                        class_1_data["feature2"],
                        class_2_data["feature2"],
                    ]
                ),
                "variance_col": np.concatenate(
                    [
                        class_0_data["variance_col"],
                        class_1_data["variance_col"],
                        class_2_data["variance_col"],
                    ]
                ),
                "label": class_0_data["label"]
                + class_1_data["label"]
                + class_2_data["label"],
            }
        )

        return df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

    @pytest.fixture
    def sampler(self):
        """Create a StratifiedSampler instance."""
        return StratifiedSampler(random_state=42)

    def test_init(self):
        """Test StratifiedSampler initialization."""
        sampler = StratifiedSampler(random_state=123)
        assert sampler.random_state == 123
        assert "balanced" in sampler.strategies
        assert "proportional_min" in sampler.strategies
        assert "optimal" in sampler.strategies
        assert len(sampler.strategies) == 3

    def test_sample_balanced_strategy(self, sampler, sample_data):
        """Test balanced allocation strategy."""
        df = sample_data

        result = sampler.sample(
            df=df,
            strata_column="label",
            target_size=300,
            strategy="balanced",
            min_samples_per_stratum=10,
        )

        # Check basic properties
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 300  # May be less due to availability constraints
        assert len(result) > 0

        # Check that all classes are represented
        class_counts = result["label"].value_counts().sort_index()
        assert len(class_counts) == 3  # All 3 classes should be present

        # In balanced sampling, each class should have at least minimum samples
        assert all(count >= 10 for count in class_counts.values)

        # Check that result contains original columns
        assert set(result.columns) == set(df.columns)

    def test_sample_proportional_min_strategy(self, sampler, sample_data):
        """Test proportional allocation with minimum constraints."""
        df = sample_data

        result = sampler.sample(
            df=df,
            strata_column="label",
            target_size=300,
            strategy="proportional_min",
            min_samples_per_stratum=20,
        )

        # Check basic properties
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

        # Check that all classes are represented with at least minimum samples
        class_counts = result["label"].value_counts().sort_index()
        assert len(class_counts) == 3  # All 3 classes should be present
        assert all(count >= 20 for count in class_counts.values)

        # Check proportional representation (class 0 should have most samples)
        assert class_counts[0] >= class_counts[1] >= class_counts[2]

    def test_sample_optimal_strategy(self, sampler, sample_data):
        """Test optimal allocation (Neyman) strategy."""
        df = sample_data

        result = sampler.sample(
            df=df,
            strata_column="label",
            target_size=300,
            strategy="optimal",
            min_samples_per_stratum=10,
            variance_column="variance_col",
        )

        # Check basic properties
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

        # Check that all classes are represented
        class_counts = result["label"].value_counts().sort_index()
        assert len(class_counts) == 3  # All 3 classes should be present
        assert all(count >= 10 for count in class_counts.values)

    def test_sample_optimal_strategy_without_variance_column(
        self, sampler, sample_data
    ):
        """Test optimal allocation without variance column (should use default variance)."""
        df = sample_data

        result = sampler.sample(
            df=df,
            strata_column="label",
            target_size=300,
            strategy="optimal",
            min_samples_per_stratum=10,
            variance_column=None,
        )

        # Should still work with default variance
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

        class_counts = result["label"].value_counts().sort_index()
        assert len(class_counts) == 3

    def test_sample_optimal_strategy_missing_variance_column(
        self, sampler, sample_data
    ):
        """Test optimal allocation with non-existent variance column."""
        df = sample_data

        result = sampler.sample(
            df=df,
            strata_column="label",
            target_size=300,
            strategy="optimal",
            min_samples_per_stratum=10,
            variance_column="nonexistent_column",
        )

        # Should work with default variance when column doesn't exist
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_sample_invalid_strategy(self, sampler, sample_data):
        """Test that invalid strategy raises ValueError."""
        df = sample_data

        with pytest.raises(ValueError, match="Unknown strategy: invalid_strategy"):
            sampler.sample(
                df=df,
                strata_column="label",
                target_size=300,
                strategy="invalid_strategy",
                min_samples_per_stratum=10,
            )

    def test_sample_missing_strata_column(self, sampler, sample_data):
        """Test that missing strata column raises KeyError."""
        df = sample_data

        with pytest.raises(KeyError):
            sampler.sample(
                df=df,
                strata_column="nonexistent_column",
                target_size=300,
                strategy="balanced",
                min_samples_per_stratum=10,
            )

    def test_sample_target_size_larger_than_data(self, sampler):
        """Test sampling when target size is larger than available data."""
        # Create small dataset
        df = pd.DataFrame({"feature": [1, 2, 3, 4, 5], "label": [0, 0, 1, 1, 2]})

        result = sampler.sample(
            df=df,
            strata_column="label",
            target_size=100,  # Much larger than available data
            strategy="balanced",
            min_samples_per_stratum=1,
        )

        # Should return all available data
        assert len(result) == 5

    def test_sample_empty_dataframe(self, sampler):
        """Test sampling with empty DataFrame."""
        df = pd.DataFrame({"label": []})

        # Based on source code analysis, _balanced_allocation has division by zero when no strata exist
        # This should raise ZeroDivisionError as per the actual implementation
        with pytest.raises(ZeroDivisionError):
            sampler.sample(
                df=df,
                strata_column="label",
                target_size=100,
                strategy="balanced",
                min_samples_per_stratum=10,
            )

    def test_get_strata_info(self, sampler, sample_data):
        """Test _get_strata_info method."""
        df = sample_data

        strata_info = sampler._get_strata_info(df, "label", "variance_col")

        # Check structure
        assert isinstance(strata_info, dict)
        assert 0 in strata_info
        assert 1 in strata_info
        assert 2 in strata_info

        # Check each stratum info
        for stratum, info in strata_info.items():
            assert "size" in info
            assert "data" in info
            assert "variance" in info
            assert "std" in info
            assert info["size"] > 0
            assert isinstance(info["data"], pd.DataFrame)

    def test_get_strata_info_without_variance_column(self, sampler, sample_data):
        """Test _get_strata_info without variance column."""
        df = sample_data

        strata_info = sampler._get_strata_info(df, "label", None)

        # Should use default variance
        for stratum, info in strata_info.items():
            assert info["variance"] == 1.0
            assert info["std"] == 1.0

    def test_balanced_allocation(self, sampler, sample_data):
        """Test _balanced_allocation method."""
        df = sample_data
        strata_info = sampler._get_strata_info(df, "label")

        allocation = sampler._balanced_allocation(strata_info, 300, 10)

        # Check allocation structure
        assert isinstance(allocation, dict)
        assert 0 in allocation
        assert 1 in allocation
        assert 2 in allocation

        # Check minimum constraints
        assert all(count >= 10 for count in allocation.values())

        # Check total allocation is reasonable
        total_allocated = sum(allocation.values())
        assert (
            total_allocated <= 300 + len(strata_info) * 10
        )  # Allow for minimum constraints

    def test_proportional_with_min_allocation(self, sampler, sample_data):
        """Test _proportional_with_min method."""
        df = sample_data
        strata_info = sampler._get_strata_info(df, "label")

        allocation = sampler._proportional_with_min(strata_info, 300, 15)

        # Check allocation structure
        assert isinstance(allocation, dict)
        assert all(stratum in allocation for stratum in strata_info.keys())

        # Check minimum constraints
        assert all(count >= 15 for count in allocation.values())

        # Check proportional representation (class 0 should get most)
        assert allocation[0] >= allocation[1] >= allocation[2]

    def test_optimal_allocation(self, sampler, sample_data):
        """Test _optimal_allocation method."""
        df = sample_data
        strata_info = sampler._get_strata_info(df, "label", "variance_col")

        allocation = sampler._optimal_allocation(strata_info, 300, 10)

        # Check allocation structure
        assert isinstance(allocation, dict)
        assert all(stratum in allocation for stratum in strata_info.keys())

        # Check minimum constraints
        assert all(count >= 10 for count in allocation.values())

    def test_perform_sampling(self, sampler, sample_data):
        """Test _perform_sampling method."""
        df = sample_data
        allocation = {0: 50, 1: 30, 2: 20}

        result = sampler._perform_sampling(df, "label", allocation)

        # Check result structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 100  # Sum of allocation

        # Check class distribution matches allocation
        class_counts = result["label"].value_counts().sort_index()
        assert class_counts[0] == 50
        assert class_counts[1] == 30
        assert class_counts[2] == 20

    def test_perform_sampling_insufficient_data(self, sampler):
        """Test _perform_sampling when stratum has insufficient data."""
        df = pd.DataFrame(
            {
                "feature": [1, 2, 3],
                "label": [0, 0, 1],  # Only 2 samples for class 0, 1 for class 1
            }
        )
        allocation = {0: 5, 1: 3}  # Request more than available

        result = sampler._perform_sampling(df, "label", allocation)

        # Should return all available data
        assert len(result) == 3
        class_counts = result["label"].value_counts().sort_index()
        assert class_counts[0] == 2  # All available for class 0
        assert class_counts[1] == 1  # All available for class 1

    def test_perform_sampling_empty_allocation(self, sampler, sample_data):
        """Test _perform_sampling with empty allocation."""
        df = sample_data
        allocation = {}

        result = sampler._perform_sampling(df, "label", allocation)

        # Should return empty DataFrame
        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)

    def test_perform_sampling_zero_allocation(self, sampler, sample_data):
        """Test _perform_sampling with zero allocation for some strata."""
        df = sample_data
        allocation = {0: 0, 1: 10, 2: 5}  # Zero allocation for class 0

        result = sampler._perform_sampling(df, "label", allocation)

        # Should only include classes with non-zero allocation
        assert len(result) == 15
        class_counts = result["label"].value_counts().sort_index()
        assert 0 not in class_counts  # Class 0 should not be present
        assert class_counts[1] == 10
        assert class_counts[2] == 5


class TestFileIOHelpers:
    """Tests for file I/O helper functions."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_read_processed_data_success(self, temp_dir):
        """Test successful reading of processed data."""
        # Create test data structure
        input_dir = temp_dir / "input"
        split_dir = input_dir / "train"
        split_dir.mkdir(parents=True)

        # Create test data file
        test_data = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "label": [0, 1, 0]}
        )
        data_file = split_dir / "train_processed_data.csv"
        test_data.to_csv(data_file, index=False)

        # Test reading - returns tuple of (DataFrame, format)
        result, detected_format = _read_processed_data(str(input_dir), "train")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert list(result.columns) == ["feature1", "feature2", "label"]
        assert detected_format == "csv"
        pd.testing.assert_frame_equal(result, test_data)

    def test_read_processed_data_file_not_found(self, temp_dir):
        """Test reading when file doesn't exist."""
        input_dir = temp_dir / "input"
        split_dir = input_dir / "nonexistent"
        split_dir.mkdir(parents=True)

        with pytest.raises(RuntimeError, match="No processed data file found"):
            _read_processed_data(str(input_dir), "nonexistent")

    def test_save_sampled_data(self, temp_dir):
        """Test saving sampled data."""
        # Create test data
        test_data = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "label": [0, 1, 0]}
        )

        # Mock logger
        mock_logger = Mock()

        # Save data - need to specify format
        output_dir = temp_dir / "output"
        _save_sampled_data(test_data, str(output_dir), "validation", "csv", mock_logger)

        # Check file was created
        expected_file = output_dir / "validation" / "validation_processed_data.csv"
        assert expected_file.exists()

        # Check file content
        saved_data = pd.read_csv(expected_file)
        pd.testing.assert_frame_equal(saved_data, test_data)

        # Check logger was called
        mock_logger.assert_called_once()
        assert "validation_processed_data.csv" in mock_logger.call_args[0][0]
        assert "shape=(3, 3)" in mock_logger.call_args[0][0]


class TestMainFunction:
    """Tests for the main function."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 100),
                "feature2": np.random.normal(1, 1, 100),
                "label": np.random.choice([0, 1, 2], 100, p=[0.7, 0.25, 0.05]),
            }
        )

    def setup_input_data(self, temp_dir, sample_data, splits=None):
        """Helper to set up input data structure."""
        if splits is None:
            splits = ["train", "val", "test"]

        input_dir = temp_dir / "input"

        for i, split in enumerate(splits):
            split_dir = input_dir / split
            split_dir.mkdir(parents=True)

            # Create different data for each split
            start_idx = i * 30
            end_idx = start_idx + 30
            split_data = sample_data.iloc[start_idx:end_idx].copy()

            data_file = split_dir / f"{split}_processed_data.csv"
            split_data.to_csv(data_file, index=False)

        return input_dir

    def test_main_training_job_type(self, temp_dir, sample_data):
        """Test main function with training job type."""
        # Set up input data
        input_dir = self.setup_input_data(temp_dir, sample_data)
        output_dir = temp_dir / "output"

        # Create arguments
        args = argparse.Namespace(job_type="training")

        # Environment variables
        environ_vars = {
            "STRATA_COLUMN": "label",
            "SAMPLING_STRATEGY": "balanced",
            "TARGET_SAMPLE_SIZE": "50",
            "MIN_SAMPLES_PER_STRATUM": "5",
            "RANDOM_STATE": "42",
        }

        # Path dictionaries
        input_paths = {"input_data": str(input_dir)}
        output_paths = {"processed_data": str(output_dir)}

        # Mock logger
        mock_logger = Mock()

        # Run main function
        result = main(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,
            job_args=args,
            logger=mock_logger,
        )

        # Verify results
        assert isinstance(result, dict)
        assert "train" in result
        assert "val" in result
        assert "test" in result  # Test split copied unchanged

        # Check that all splits have the expected structure
        for split_name, split_df in result.items():
            assert isinstance(split_df, pd.DataFrame)
            assert "label" in split_df.columns
            assert len(split_df) > 0

        # Verify output files exist
        for split_name in ["train", "val", "test"]:
            output_file = output_dir / split_name / f"{split_name}_processed_data.csv"
            assert output_file.exists()

        # Check logger was called
        assert mock_logger.call_count > 0

    def test_main_validation_job_type(self, temp_dir, sample_data):
        """Test main function with validation job type."""
        # Set up input data for validation only
        input_dir = self.setup_input_data(temp_dir, sample_data, splits=["validation"])
        output_dir = temp_dir / "output"

        # Create arguments
        args = argparse.Namespace(job_type="validation")

        # Environment variables
        environ_vars = {
            "STRATA_COLUMN": "label",
            "SAMPLING_STRATEGY": "proportional_min",
            "TARGET_SAMPLE_SIZE": "40",
            "MIN_SAMPLES_PER_STRATUM": "3",
            "RANDOM_STATE": "42",
        }

        # Path dictionaries
        input_paths = {"input_data": str(input_dir)}
        output_paths = {"processed_data": str(output_dir)}
        # Path dictionaries
        input_paths = {"input_data": str(input_dir)}
        output_paths = {"processed_data": str(output_dir)}

        # Run main function
        result = main(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,
            job_args=args,
        )

        # Verify results
        assert isinstance(result, dict)
        assert "validation" in result
        assert len(result) == 1  # Only validation split should be present

        # Check validation split
        val_split = result["validation"]
        assert isinstance(val_split, pd.DataFrame)
        assert "label" in val_split.columns
        assert len(val_split) > 0

        # Verify output file exists
        output_file = output_dir / "validation" / "validation_processed_data.csv"
        assert output_file.exists()

    def test_main_testing_job_type(self, temp_dir, sample_data):
        """Test main function with testing job type."""
        # Set up input data for testing only
        input_dir = self.setup_input_data(temp_dir, sample_data, splits=["testing"])
        output_dir = temp_dir / "output"

        # Create arguments
        args = argparse.Namespace(job_type="testing")

        # Environment variables
        environ_vars = {
            "STRATA_COLUMN": "label",
            "SAMPLING_STRATEGY": "optimal",
            "TARGET_SAMPLE_SIZE": "35",
            "MIN_SAMPLES_PER_STRATUM": "2",
            "VARIANCE_COLUMN": "feature1",
            "RANDOM_STATE": "42",
        }

        # Path dictionaries
        input_paths = {"input_data": str(input_dir)}
        output_paths = {"processed_data": str(output_dir)}

        # Run main function
        result = main(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,
            job_args=args,
        )

        # Verify results
        assert isinstance(result, dict)
        assert "testing" in result
        assert len(result) == 1  # Only testing split should be present

    def test_main_calibration_job_type(self, temp_dir, sample_data):
        """Test main function with calibration job type."""
        # Set up input data for calibration only
        input_dir = self.setup_input_data(temp_dir, sample_data, splits=["calibration"])
        output_dir = temp_dir / "output"

        # Create arguments
        args = argparse.Namespace(job_type="calibration")

        # Environment variables
        environ_vars = {
            "STRATA_COLUMN": "label",
            "SAMPLING_STRATEGY": "balanced",
            "TARGET_SAMPLE_SIZE": "30",
            "MIN_SAMPLES_PER_STRATUM": "1",
            "RANDOM_STATE": "42",
        }

        # Path dictionaries
        input_paths = {"input_data": str(input_dir)}
        output_paths = {"processed_data": str(output_dir)}

        # Run main function
        result = main(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,
            job_args=args,
        )

        # Verify results
        assert isinstance(result, dict)
        assert "calibration" in result
        assert len(result) == 1  # Only calibration split should be present

    def test_main_missing_strata_column_env_var(self, temp_dir, sample_data):
        """Test main function with missing STRATA_COLUMN environment variable."""
        # Set up input data
        input_dir = self.setup_input_data(temp_dir, sample_data, splits=["train"])
        output_dir = temp_dir / "output"

        # Create arguments
        args = argparse.Namespace(job_type="training")

        # Environment variables without STRATA_COLUMN
        environ_vars = {
            "SAMPLING_STRATEGY": "balanced",
            "TARGET_SAMPLE_SIZE": "50",
            "MIN_SAMPLES_PER_STRATUM": "5",
            "RANDOM_STATE": "42",
        }

        # Path dictionaries
        input_paths = {"input_data": str(input_dir)}
        output_paths = {"processed_data": str(output_dir)}

        # Should raise RuntimeError
        with pytest.raises(
            RuntimeError, match="STRATA_COLUMN environment variable must be set"
        ):
            main(
                input_paths=input_paths,
                output_paths=output_paths,
                environ_vars=environ_vars,
                job_args=args,
            )

    def test_main_invalid_sampling_strategy(self, temp_dir, sample_data):
        """Test main function with invalid sampling strategy."""
        # Set up input data
        input_dir = self.setup_input_data(temp_dir, sample_data, splits=["train"])
        output_dir = temp_dir / "output"

        # Create arguments
        args = argparse.Namespace(job_type="training")

        # Environment variables with invalid strategy
        environ_vars = {
            "STRATA_COLUMN": "label",
            "SAMPLING_STRATEGY": "invalid_strategy",
            "TARGET_SAMPLE_SIZE": "50",
            "MIN_SAMPLES_PER_STRATUM": "5",
            "RANDOM_STATE": "42",
        }

        # Path dictionaries
        input_paths = {"input_data": str(input_dir)}
        output_paths = {"processed_data": str(output_dir)}

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="Invalid SAMPLING_STRATEGY"):
            main(
                input_paths=input_paths,
                output_paths=output_paths,
                environ_vars=environ_vars,
                job_args=args,
            )

    def test_main_missing_strata_column_in_data(self, temp_dir, sample_data):
        """Test main function when strata column is missing from data."""
        # Create data without label column
        data_without_label = sample_data.drop("label", axis=1)

        # Set up input data
        input_dir = temp_dir / "input"
        train_dir = input_dir / "train"
        train_dir.mkdir(parents=True)

        data_file = train_dir / "train_processed_data.csv"
        data_without_label.to_csv(data_file, index=False)

        output_dir = temp_dir / "output"

        # Create arguments
        args = argparse.Namespace(job_type="training")

        # Environment variables
        environ_vars = {
            "STRATA_COLUMN": "label",  # This column doesn't exist in data
            "SAMPLING_STRATEGY": "balanced",
            "TARGET_SAMPLE_SIZE": "50",
            "MIN_SAMPLES_PER_STRATUM": "5",
            "RANDOM_STATE": "42",
        }

        # Path dictionaries
        input_paths = {"input_data": str(input_dir)}
        output_paths = {"processed_data": str(output_dir)}

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="Strata column 'label' not found"):
            main(
                input_paths=input_paths,
                output_paths=output_paths,
                environ_vars=environ_vars,
                job_args=args,
            )

    def test_main_missing_input_files(self, temp_dir):
        """Test main function with missing input files."""
        # Create empty input structure
        input_dir = temp_dir / "input"
        train_dir = input_dir / "train"
        train_dir.mkdir(parents=True)
        # Don't create any data files

        output_dir = temp_dir / "output"

        # Create arguments
        args = argparse.Namespace(job_type="training")

        # Environment variables
        environ_vars = {
            "STRATA_COLUMN": "label",
            "SAMPLING_STRATEGY": "balanced",
            "TARGET_SAMPLE_SIZE": "50",
            "MIN_SAMPLES_PER_STRATUM": "5",
            "RANDOM_STATE": "42",
        }

        # Path dictionaries
        input_paths = {"input_data": str(input_dir)}
        output_paths = {"processed_data": str(output_dir)}

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="No processed data file found"):
            main(
                input_paths=input_paths,
                output_paths=output_paths,
                environ_vars=environ_vars,
                job_args=args,
            )

    def test_main_with_variance_column_warning(self, temp_dir, sample_data):
        """Test main function with optimal strategy and missing variance column."""
        # The sample_data fixture doesn't have variance_col, so we use it as-is
        # This test verifies the warning when VARIANCE_COLUMN is specified but doesn't exist
        data_without_variance = sample_data  # Already doesn't have variance_col

        # Set up input data with both train and val splits (required for training job_type)
        input_dir = self.setup_input_data(
            temp_dir, data_without_variance, splits=["train", "val"]
        )

        output_dir = temp_dir / "output"

        # Create arguments
        args = argparse.Namespace(job_type="training")

        # Environment variables with optimal strategy and variance column
        environ_vars = {
            "STRATA_COLUMN": "label",
            "SAMPLING_STRATEGY": "optimal",
            "TARGET_SAMPLE_SIZE": "50",
            "MIN_SAMPLES_PER_STRATUM": "5",
            "VARIANCE_COLUMN": "variance_col",  # This column doesn't exist
            "RANDOM_STATE": "42",
        }

        # Path dictionaries
        input_paths = {"input_data": str(input_dir)}
        output_paths = {"processed_data": str(output_dir)}

        # Mock logger to capture warning
        mock_logger = Mock()

        # Should work but log warning
        result = main(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,
            job_args=args,
            logger=mock_logger,
        )

        # Should still return results
        assert isinstance(result, dict)
        assert "train" in result

        # Check that warning was logged
        warning_logged = any(
            "WARNING" in str(call) and "variance_col" in str(call)
            for call in mock_logger.call_args_list
        )
        assert warning_logged

    def test_main_default_logger(self, temp_dir, sample_data):
        """Test main function with default logger (None)."""
        # Set up input data with both train and val splits (required for training job_type)
        input_dir = self.setup_input_data(
            temp_dir, sample_data, splits=["train", "val"]
        )
        output_dir = temp_dir / "output"

        # Create arguments
        args = argparse.Namespace(job_type="training")

        # Environment variables
        environ_vars = {
            "STRATA_COLUMN": "label",
            "SAMPLING_STRATEGY": "balanced",
            "TARGET_SAMPLE_SIZE": "30",
            "MIN_SAMPLES_PER_STRATUM": "3",
            "RANDOM_STATE": "42",
        }

        # Path dictionaries
        input_paths = {"input_data": str(input_dir)}
        output_paths = {"processed_data": str(output_dir)}

        # Run main function without logger (should use print)
        result = main(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,
            job_args=args,
            logger=None,  # Should default to print
        )

        # Should still work
        assert isinstance(result, dict)
        assert "train" in result

    def test_main_environment_variable_defaults(self, temp_dir, sample_data):
        """Test main function with default environment variable values."""
        # Set up input data with both train and val splits (required for training job_type)
        input_dir = self.setup_input_data(
            temp_dir, sample_data, splits=["train", "val"]
        )
        output_dir = temp_dir / "output"

        # Create arguments
        args = argparse.Namespace(job_type="training")

        # Environment variables with only required ones (others should use defaults)
        environ_vars = {
            "STRATA_COLUMN": "label"
            # All other variables should use defaults from implementation
        }

        # Path dictionaries
        input_paths = {"input_data": str(input_dir)}
        output_paths = {"processed_data": str(output_dir)}

        # Run main function
        result = main(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,
            job_args=args,
        )

        # Should work with defaults
        assert isinstance(result, dict)
        assert "train" in result

    def test_main_edge_case_small_target_size(self, temp_dir, sample_data):
        """Test main function with very small target size."""
        # Set up input data with both train and val splits (required for training job_type)
        input_dir = self.setup_input_data(
            temp_dir, sample_data, splits=["train", "val"]
        )
        output_dir = temp_dir / "output"

        # Create arguments
        args = argparse.Namespace(job_type="training")

        # Environment variables with very small target size
        environ_vars = {
            "STRATA_COLUMN": "label",
            "SAMPLING_STRATEGY": "balanced",
            "TARGET_SAMPLE_SIZE": "5",  # Very small
            "MIN_SAMPLES_PER_STRATUM": "1",
            "RANDOM_STATE": "42",
        }

        # Path dictionaries
        input_paths = {"input_data": str(input_dir)}
        output_paths = {"processed_data": str(output_dir)}

        # Run main function
        result = main(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,
            job_args=args,
        )

        # Should still work
        assert isinstance(result, dict)
        assert "train" in result
        assert len(result["train"]) > 0

    def test_main_training_copies_test_split_unchanged(self, temp_dir, sample_data):
        """Test that training job type copies test split unchanged."""
        # Set up input data with all splits
        input_dir = self.setup_input_data(temp_dir, sample_data)
        output_dir = temp_dir / "output"

        # Create arguments
        args = argparse.Namespace(job_type="training")

        # Environment variables
        environ_vars = {
            "STRATA_COLUMN": "label",
            "SAMPLING_STRATEGY": "balanced",
            "TARGET_SAMPLE_SIZE": "20",  # Small target size
            "MIN_SAMPLES_PER_STRATUM": "2",
            "RANDOM_STATE": "42",
        }

        # Path dictionaries
        input_paths = {"input_data": str(input_dir)}
        output_paths = {"processed_data": str(output_dir)}

        # Read original test data
        original_test = pd.read_csv(input_dir / "test" / "test_processed_data.csv")

        # Run main function
        result = main(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,
            job_args=args,
        )

        # Check that test split is unchanged
        assert "test" in result
        test_result = result["test"]

        # Test split should be identical to original
        pd.testing.assert_frame_equal(test_result, original_test)

        # Train and val should be sampled (smaller than original)
        assert len(result["train"]) <= 30  # Original train split size
        assert len(result["val"]) <= 30  # Original val split size

    def test_main_training_handles_missing_test_split(self, temp_dir, sample_data):
        """Test training job type when test split is missing."""
        # Set up input data without test split
        input_dir = self.setup_input_data(
            temp_dir, sample_data, splits=["train", "val"]
        )
        output_dir = temp_dir / "output"

        # Create arguments
        args = argparse.Namespace(job_type="training")

        # Environment variables
        environ_vars = {
            "STRATA_COLUMN": "label",
            "SAMPLING_STRATEGY": "balanced",
            "TARGET_SAMPLE_SIZE": "20",
            "MIN_SAMPLES_PER_STRATUM": "2",
            "RANDOM_STATE": "42",
        }

        # Path dictionaries
        input_paths = {"input_data": str(input_dir)}
        output_paths = {"processed_data": str(output_dir)}

        # Mock logger to capture warning
        mock_logger = Mock()

        # Run main function
        result = main(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,
            job_args=args,
            logger=mock_logger,
        )

        # Should still work without test split
        assert isinstance(result, dict)
        assert "train" in result
        assert "val" in result
        assert "test" not in result  # Test split should not be present

        # Check that warning was logged about missing test split
        warning_logged = any(
            "WARNING" in str(call) or "Could not copy test split" in str(call)
            for call in mock_logger.call_args_list
        )
        assert warning_logged

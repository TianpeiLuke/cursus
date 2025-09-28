#!/usr/bin/env python
import os
import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, Optional, Callable, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# --- Stratified Sampling Core Logic ---


class StratifiedSampler:
    """
    Stratified sampling implementation with three core allocation strategies:
    1. Balanced allocation - for class imbalance
    2. Proportional with minimum constraints - for causal analysis
    3. Optimal allocation (Neyman) - for variance optimization
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.strategies = {
            'balanced': self._balanced_allocation,
            'proportional_min': self._proportional_with_min,
            'optimal': self._optimal_allocation
        }
    
    def sample(
        self, 
        df: pd.DataFrame, 
        strata_column: str,
        target_size: int,
        strategy: str = 'balanced',
        min_samples_per_stratum: int = 10,
        variance_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Perform stratified sampling on a DataFrame.
        
        Args:
            df: Input DataFrame
            strata_column: Column name to stratify by
            target_size: Total desired sample size
            strategy: Sampling strategy ('balanced', 'proportional_min', 'optimal')
            min_samples_per_stratum: Minimum samples per stratum
            variance_column: Column for variance calculation (needed for optimal strategy)
            
        Returns:
            Sampled DataFrame
        """
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list(self.strategies.keys())}")
        
        # Get stratum information
        strata_info = self._get_strata_info(df, strata_column, variance_column)
        
        # Calculate allocation
        allocation = self.strategies[strategy](
            strata_info, target_size, min_samples_per_stratum
        )
        
        # Perform sampling
        return self._perform_sampling(df, strata_column, allocation)
    
    def _get_strata_info(self, df: pd.DataFrame, strata_column: str, variance_column: Optional[str] = None) -> Dict:
        """Extract stratum information from DataFrame."""
        strata_info = {}
        
        for stratum in df[strata_column].unique():
            stratum_df = df[df[strata_column] == stratum]
            info = {
                'size': len(stratum_df),
                'data': stratum_df
            }
            
            if variance_column and variance_column in df.columns:
                info['variance'] = stratum_df[variance_column].var()
                info['std'] = stratum_df[variance_column].std()
            else:
                info['variance'] = 1.0  # Default variance
                info['std'] = 1.0
            
            strata_info[stratum] = info
        
        return strata_info
    
    def _balanced_allocation(self, strata_info: Dict, target_size: int, min_samples: int) -> Dict[Any, int]:
        """
        Balanced allocation strategy - equal samples per stratum.
        Handles class imbalance by giving equal representation to all classes.
        """
        num_strata = len(strata_info)
        samples_per_stratum = max(min_samples, target_size // num_strata)
        
        allocation = {}
        total_allocated = 0
        
        for stratum, info in strata_info.items():
            # Don't exceed available samples in stratum
            allocated = min(samples_per_stratum, info['size'])
            allocation[stratum] = allocated
            total_allocated += allocated
        
        # Distribute remaining samples proportionally if we're under target
        remaining = target_size - total_allocated
        if remaining > 0:
            # Sort strata by available capacity (size - current allocation)
            available_capacity = {
                stratum: info['size'] - allocation[stratum] 
                for stratum, info in strata_info.items()
            }
            
            # Distribute remaining samples to strata with capacity
            strata_with_capacity = [s for s, cap in available_capacity.items() if cap > 0]
            if strata_with_capacity:
                extra_per_stratum = remaining // len(strata_with_capacity)
                for stratum in strata_with_capacity:
                    extra = min(extra_per_stratum, available_capacity[stratum])
                    allocation[stratum] += extra
        
        return allocation
    
    def _proportional_with_min(self, strata_info: Dict, target_size: int, min_samples: int) -> Dict[Any, int]:
        """
        Proportional allocation with minimum constraints.
        Maintains representativeness while ensuring adequate samples for causal inference.
        """
        total_population = sum(info['size'] for info in strata_info.values())
        allocation = {}
        
        # First pass: allocate proportionally
        for stratum, info in strata_info.items():
            proportion = info['size'] / total_population
            proportional_size = int(target_size * proportion)
            allocation[stratum] = max(min_samples, proportional_size)
        
        # Second pass: adjust if we exceeded target due to minimum constraints
        total_allocated = sum(allocation.values())
        if total_allocated > target_size:
            # Scale down while respecting minimums
            excess = total_allocated - target_size
            adjustable_strata = {
                stratum: allocation[stratum] - min_samples 
                for stratum in allocation 
                if allocation[stratum] > min_samples
            }
            
            if sum(adjustable_strata.values()) >= excess:
                # Proportionally reduce from adjustable strata
                total_adjustable = sum(adjustable_strata.values())
                for stratum, adjustable in adjustable_strata.items():
                    reduction = int(excess * adjustable / total_adjustable)
                    allocation[stratum] -= reduction
        
        # Ensure we don't exceed available samples in each stratum
        for stratum, info in strata_info.items():
            allocation[stratum] = min(allocation[stratum], info['size'])
        
        return allocation
    
    def _optimal_allocation(self, strata_info: Dict, target_size: int, min_samples: int) -> Dict[Any, int]:
        """
        Optimal allocation (Neyman) strategy.
        Minimizes sampling variance by allocating based on stratum size and variability.
        """
        # Calculate Neyman allocation: n_h = n * (N_h * S_h) / sum(N_i * S_i)
        numerators = {}
        total_numerator = 0
        
        for stratum, info in strata_info.items():
            numerator = info['size'] * info['std']
            numerators[stratum] = numerator
            total_numerator += numerator
        
        allocation = {}
        for stratum, numerator in numerators.items():
            if total_numerator > 0:
                optimal_size = int(target_size * numerator / total_numerator)
            else:
                optimal_size = target_size // len(strata_info)
            
            # Apply minimum constraint and don't exceed stratum size
            allocation[stratum] = min(
                max(min_samples, optimal_size),
                strata_info[stratum]['size']
            )
        
        return allocation
    
    def _perform_sampling(self, df: pd.DataFrame, strata_column: str, allocation: Dict[Any, int]) -> pd.DataFrame:
        """Perform the actual sampling based on allocation."""
        sampled_dfs = []
        
        for stratum, sample_size in allocation.items():
            if sample_size > 0:
                stratum_df = df[df[strata_column] == stratum]
                if len(stratum_df) >= sample_size:
                    sampled = stratum_df.sample(n=sample_size, random_state=self.random_state)
                else:
                    sampled = stratum_df  # Take all available if not enough
                sampled_dfs.append(sampled)
        
        if sampled_dfs:
            return pd.concat(sampled_dfs, ignore_index=True)
        else:
            return pd.DataFrame()


# --- File I/O Helper Functions ---


def _read_processed_data(input_dir: str, split_name: str) -> pd.DataFrame:
    """Read processed data from tabular_preprocessing output structure."""
    input_path = Path(input_dir)
    split_dir = input_path / split_name
    
    # Look for the processed data file
    processed_file = split_dir / f"{split_name}_processed_data.csv"
    
    if not processed_file.exists():
        raise RuntimeError(f"Processed data file not found: {processed_file}")
    
    return pd.read_csv(processed_file)


def _save_sampled_data(df: pd.DataFrame, output_dir: str, split_name: str, logger: Callable[[str], None]):
    """Save sampled data maintaining the same folder structure as tabular_preprocessing."""
    output_path = Path(output_dir)
    split_dir = output_path / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    
    # Save with same naming convention
    output_file = split_dir / f"{split_name}_processed_data.csv"
    df.to_csv(output_file, index=False)
    logger(f"[INFO] Saved {output_file} (shape={df.shape})")


# --- Main Processing Logic ---


def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Main logic for stratified sampling, following tabular_preprocessing format.

    Args:
        input_paths: Dictionary of input paths with logical names
        output_paths: Dictionary of output paths with logical names
        environ_vars: Dictionary of environment variables
        job_args: Command line arguments
        logger: Optional logger object (defaults to print if None)

    Returns:
        Dictionary of sampled DataFrames by split name
    """
    # Extract parameters from arguments and environment variables
    job_type = job_args.job_type
    strata_column = environ_vars.get("STRATA_COLUMN")
    sampling_strategy = environ_vars.get("SAMPLING_STRATEGY", "balanced")
    target_sample_size = int(environ_vars.get("TARGET_SAMPLE_SIZE", 1000))
    min_samples_per_stratum = int(environ_vars.get("MIN_SAMPLES_PER_STRATUM", 10))
    variance_column = environ_vars.get("VARIANCE_COLUMN")  # Optional for optimal strategy
    random_state = int(environ_vars.get("RANDOM_STATE", 42))
    
    # Extract paths
    input_data_dir = input_paths.get("data_input", "/opt/ml/processing/input/data")
    output_dir = output_paths.get("data_output", "/opt/ml/processing/output")
    
    # Use print function if no logger is provided
    log = logger or print
    
    # Validate required parameters
    if not strata_column:
        raise RuntimeError("STRATA_COLUMN environment variable must be set.")
    
    if sampling_strategy not in ['balanced', 'proportional_min', 'optimal']:
        raise RuntimeError(f"Invalid SAMPLING_STRATEGY: {sampling_strategy}. Must be one of: balanced, proportional_min, optimal")
    
    # Initialize sampler
    sampler = StratifiedSampler(random_state=random_state)
    
    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    log(f"[INFO] Starting stratified sampling with strategy: {sampling_strategy}")
    log(f"[INFO] Strata column: {strata_column}")
    log(f"[INFO] Target sample size: {target_sample_size}")
    log(f"[INFO] Min samples per stratum: {min_samples_per_stratum}")
    
    # Determine which splits to process based on job_type
    if job_type == "training":
        # For training job_type, process train and val splits (not test)
        splits_to_process = ["train", "val"]
        log("[INFO] Training job type detected - processing train and val splits only")
    else:
        # For other job types, process only that specific split
        splits_to_process = [job_type]
        log(f"[INFO] Non-training job type detected - processing {job_type} split only")
    
    sampled_splits = {}
    
    # Process each split
    for split_name in splits_to_process:
        try:
            log(f"[INFO] Processing {split_name} split...")
            
            # Read the processed data from tabular_preprocessing output
            df = _read_processed_data(input_data_dir, split_name)
            log(f"[INFO] Loaded {split_name} data with shape: {df.shape}")
            
            # Validate strata column exists
            if strata_column not in df.columns:
                raise RuntimeError(f"Strata column '{strata_column}' not found in {split_name} data. Available columns: {df.columns.tolist()}")
            
            # Check if variance column exists (for optimal strategy)
            if sampling_strategy == 'optimal' and variance_column and variance_column not in df.columns:
                log(f"[WARNING] Variance column '{variance_column}' not found. Using default variance for optimal allocation.")
                variance_column = None
            
            # Calculate target size for this split (could be different per split)
            split_target_size = min(target_sample_size, len(df))  # Don't exceed available data
            
            # Perform stratified sampling
            sampled_df = sampler.sample(
                df=df,
                strata_column=strata_column,
                target_size=split_target_size,
                strategy=sampling_strategy,
                min_samples_per_stratum=min_samples_per_stratum,
                variance_column=variance_column
            )
            
            log(f"[INFO] Sampled {split_name} data: {len(sampled_df)} rows from {len(df)} original rows")
            
            # Log stratum distribution
            strata_counts = sampled_df[strata_column].value_counts().sort_index()
            log(f"[INFO] {split_name} stratum distribution: {dict(strata_counts)}")
            
            # Save sampled data
            _save_sampled_data(sampled_df, output_dir, split_name, log)
            sampled_splits[split_name] = sampled_df
            
        except Exception as e:
            log(f"[ERROR] Failed to process {split_name} split: {str(e)}")
            raise
    
    # For training job_type, also copy test split unchanged (if it exists)
    if job_type == "training":
        try:
            test_df = _read_processed_data(input_data_dir, "test")
            log(f"[INFO] Copying test split unchanged (shape: {test_df.shape})")
            _save_sampled_data(test_df, output_dir, "test", log)
            sampled_splits["test"] = test_df
        except Exception as e:
            log(f"[WARNING] Could not copy test split: {str(e)}")
    
    log("[INFO] Stratified sampling complete.")
    return sampled_splits


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--job_type",
            type=str,
            required=True,
            choices=["training", "validation", "testing", "calibration"],
            help="One of ['training','validation','testing','calibration']",
        )
        args = parser.parse_args()

        # Read configuration from environment variables
        STRATA_COLUMN = os.environ.get("STRATA_COLUMN")
        if not STRATA_COLUMN:
            raise RuntimeError("STRATA_COLUMN environment variable must be set.")
        
        SAMPLING_STRATEGY = os.environ.get("SAMPLING_STRATEGY", "balanced")
        TARGET_SAMPLE_SIZE = int(os.environ.get("TARGET_SAMPLE_SIZE", 1000))
        MIN_SAMPLES_PER_STRATUM = int(os.environ.get("MIN_SAMPLES_PER_STRATUM", 10))
        VARIANCE_COLUMN = os.environ.get("VARIANCE_COLUMN")  # Optional
        RANDOM_STATE = int(os.environ.get("RANDOM_STATE", 42))

        # Define standard SageMaker paths - use contract-declared paths directly
        INPUT_DATA_DIR = "/opt/ml/processing/input/data"
        OUTPUT_DIR = "/opt/ml/processing/output"

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logger = logging.getLogger(__name__)

        # Log key parameters
        logger.info(f"Starting stratified sampling with parameters:")
        logger.info(f"  Job Type: {args.job_type}")
        logger.info(f"  Strata Column: {STRATA_COLUMN}")
        logger.info(f"  Sampling Strategy: {SAMPLING_STRATEGY}")
        logger.info(f"  Target Sample Size: {TARGET_SAMPLE_SIZE}")
        logger.info(f"  Min Samples Per Stratum: {MIN_SAMPLES_PER_STRATUM}")
        logger.info(f"  Variance Column: {VARIANCE_COLUMN}")
        logger.info(f"  Random State: {RANDOM_STATE}")
        logger.info(f"  Input Directory: {INPUT_DATA_DIR}")
        logger.info(f"  Output Directory: {OUTPUT_DIR}")

        # Set up path dictionaries
        input_paths = {"data_input": INPUT_DATA_DIR}
        output_paths = {"data_output": OUTPUT_DIR}

        # Environment variables dictionary
        environ_vars = {
            "STRATA_COLUMN": STRATA_COLUMN,
            "SAMPLING_STRATEGY": SAMPLING_STRATEGY,
            "TARGET_SAMPLE_SIZE": str(TARGET_SAMPLE_SIZE),
            "MIN_SAMPLES_PER_STRATUM": str(MIN_SAMPLES_PER_STRATUM),
            "VARIANCE_COLUMN": VARIANCE_COLUMN,
            "RANDOM_STATE": str(RANDOM_STATE),
        }

        # Execute the main processing logic
        result = main(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,
            job_args=args,
            logger=logger.info,
        )

        # Log completion summary
        splits_summary = ", ".join(
            [f"{name}: {df.shape}" for name, df in result.items()]
        )
        logger.info(f"Stratified sampling completed successfully. Splits: {splits_summary}")
        sys.exit(0)
        
    except Exception as e:
        logging.error(f"Error in stratified sampling script: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)

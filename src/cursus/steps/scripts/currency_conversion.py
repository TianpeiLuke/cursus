#!/usr/bin/env python
"""
Currency Conversion Processing Script

This script handles currency conversion for tabular data using exchange rates.
It supports both training mode (all splits) and inference mode (single split).
Follows the same pattern as feature_selection.py and missing_value_imputation.py for consistency.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import json
import pickle as pkl
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging
from multiprocessing import Pool, cpu_count

# Default paths (will be overridden by parameters in main function)
DEFAULT_INPUT_DIR = "/opt/ml/processing/input/data"
DEFAULT_OUTPUT_DIR = "/opt/ml/processing/output"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_split_data(job_type: str, input_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load data according to job_type, following feature_selection pattern.

    For 'training': Loads data from train, test, and val subdirectories
    For others: Loads single job_type split
    """
    input_path = Path(input_dir)

    if job_type == "training":
        # For training, we expect data in train/test/val subdirectories
        train_df = pd.read_csv(input_path / "train" / "train_processed_data.csv")
        test_df = pd.read_csv(input_path / "test" / "test_processed_data.csv")
        val_df = pd.read_csv(input_path / "val" / "val_processed_data.csv")
        logger.info(
            f"Loaded training data splits: train={train_df.shape}, test={test_df.shape}, val={val_df.shape}"
        )
        return {"train": train_df, "test": test_df, "val": val_df}
    else:
        # For other job types, we expect data in a single directory named after job_type
        df = pd.read_csv(input_path / job_type / f"{job_type}_processed_data.csv")
        logger.info(f"Loaded {job_type} data: {df.shape}")
        return {job_type: df}


def save_output_data(
    job_type: str, output_dir: str, data_dict: Dict[str, pd.DataFrame]
) -> None:
    """
    Save processed data according to job_type, following feature_selection pattern.

    For 'training': Saves data to train, test, and val subdirectories
    For others: Saves to single job_type directory
    """
    output_path = Path(output_dir)

    for split_name, df in data_dict.items():
        split_output_dir = output_path / split_name
        split_output_dir.mkdir(exist_ok=True, parents=True)

        output_file = split_output_dir / f"{split_name}_processed_data.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {split_name} data to {output_file}, shape: {df.shape}")


def get_currency_code(
    row: pd.Series,
    currency_code_field: Optional[str],
    marketplace_id_field: Optional[str],
    conversion_dict: Dict[str, Any],
    default_currency: str,
) -> str:
    """
    Get currency code for a given row based on available fields.

    Args:
        row: Data row
        currency_code_field: Name of column containing currency codes directly
        marketplace_id_field: Name of column containing marketplace IDs
        conversion_dict: Dictionary with currency conversion mappings
        default_currency: Default currency code to use when lookup fails

    Returns:
        Currency code for the row
    """
    # Check if we have currency_code field directly
    if currency_code_field and currency_code_field in row:
        currency = row[currency_code_field]
        if pd.notna(currency) and str(currency).strip():
            return str(currency).strip()

    # Otherwise use marketplace_id field
    if marketplace_id_field and marketplace_id_field in row:
        marketplace_id = row[marketplace_id_field]
        if pd.notna(marketplace_id):
            # Look up currency code by marketplace ID
            mappings = conversion_dict.get("mappings", [])
            for mapping in mappings:
                if str(mapping.get("marketplace_id")) == str(marketplace_id):
                    return mapping.get("currency_code", default_currency)

    # Default fallback
    return default_currency


def currency_conversion_single_variable(
    args: Tuple[pd.DataFrame, str, pd.Series],
) -> pd.Series:
    """Convert single variable's currency values."""
    df, variable, exchange_rate_series = args
    return df[variable] / exchange_rate_series.values


def parallel_currency_conversion(
    df: pd.DataFrame,
    exchange_rate_series: pd.Series,
    currency_conversion_vars: List[str],
    n_workers: int = 50,
) -> pd.DataFrame:
    """Perform parallel currency conversion on multiple variables."""
    processes = min(cpu_count(), len(currency_conversion_vars), n_workers)

    with Pool(processes=processes) as pool:
        results = pool.map(
            currency_conversion_single_variable,
            [
                (df[[var]], var, exchange_rate_series)
                for var in currency_conversion_vars
            ],
        )
        df[currency_conversion_vars] = pd.concat(results, axis=1)

    return df


def process_currency_conversion(
    df: pd.DataFrame,
    currency_code_field: Optional[str],
    marketplace_id_field: Optional[str],
    currency_conversion_vars: List[str],
    currency_conversion_dict: Dict[str, Any],
    default_currency: str = "USD",
    n_workers: int = 50,
) -> pd.DataFrame:
    """Process currency conversion for a DataFrame."""
    logger.info(f"Starting currency conversion on DataFrame with shape: {df.shape}")

    # Filter variables that exist in the DataFrame
    currency_conversion_vars = [
        var for var in currency_conversion_vars if var in df.columns
    ]

    if not currency_conversion_vars:
        logger.warning("No variables require currency conversion")
        return df

    # Get currency codes for each row
    df["__temp_currency_code__"] = df.apply(
        lambda row: get_currency_code(
            row,
            currency_code_field,
            marketplace_id_field,
            currency_conversion_dict,
            default_currency,
        ),
        axis=1,
    )

    # Create exchange rate series
    exchange_rates = []
    mappings = currency_conversion_dict.get("mappings", [])

    for currency_code in df["__temp_currency_code__"]:
        rate = 1.0  # Default no conversion
        for mapping in mappings:
            if mapping.get("currency_code") == currency_code:
                rate = mapping.get("conversion_rate", 1.0)
                break
        exchange_rates.append(rate)

    exchange_rate_series = pd.Series(exchange_rates, index=df.index)

    logger.info(f"Converting currencies for variables: {currency_conversion_vars}")
    df = parallel_currency_conversion(
        df,
        exchange_rate_series,
        currency_conversion_vars,
        n_workers,
    )

    # Clean up temporary column
    df = df.drop(columns=["__temp_currency_code__"])

    logger.info("Currency conversion completed")
    return df


def process_data(
    data_dict: Dict[str, pd.DataFrame],
    job_type: str,
    currency_config: Dict[str, Any],
) -> Dict[str, pd.DataFrame]:
    """
    Core data processing logic for currency conversion.

    Args:
        data_dict: Dictionary of dataframes keyed by split name
        job_type: Type of job (training, validation, testing, calibration)
        currency_config: Currency conversion configuration dictionary

    Returns:
        Dictionary of converted dataframes
    """
    # Extract configuration
    currency_code_field = currency_config.get("CURRENCY_CODE_FIELD")
    marketplace_id_field = currency_config.get("MARKETPLACE_ID_FIELD")
    currency_conversion_vars = currency_config.get("CURRENCY_CONVERSION_VARS", [])
    currency_conversion_dict = currency_config.get("CURRENCY_CONVERSION_DICT", {})
    default_currency = currency_config.get("DEFAULT_CURRENCY", "USD")
    n_workers = currency_config.get("N_WORKERS", 50)

    # Validate configuration
    if not currency_code_field and not marketplace_id_field:
        logger.warning(
            "Neither CURRENCY_CODE_FIELD nor MARKETPLACE_ID_FIELD specified. Skipping conversion."
        )
        return data_dict

    if not currency_conversion_vars:
        logger.warning(
            "No currency conversion variables specified. Skipping conversion."
        )
        return data_dict

    logger.info(f"Running currency conversion with job_type={job_type}")
    logger.info(f"Currency code field: {currency_code_field}")
    logger.info(f"Marketplace ID field: {marketplace_id_field}")
    logger.info(f"Variables to convert: {currency_conversion_vars}")
    logger.info(f"Default currency: {default_currency}")

    # Process all splits
    converted_data = {}
    for split_name, df in data_dict.items():
        logger.info(f"Processing {split_name} split with {len(df)} rows")
        df_converted = process_currency_conversion(
            df=df,
            currency_code_field=currency_code_field,
            marketplace_id_field=marketplace_id_field,
            currency_conversion_vars=currency_conversion_vars,
            currency_conversion_dict=currency_conversion_dict,
            default_currency=default_currency,
            n_workers=n_workers,
        )
        converted_data[split_name] = df_converted
        logger.info(f"Converted {split_name} data, shape: {df_converted.shape}")

    return converted_data


def internal_main(
    job_type: str,
    input_dir: str,
    output_dir: str,
    currency_config: Dict[str, Any],
    load_data_func=load_split_data,
    save_data_func=save_output_data,
) -> Dict[str, pd.DataFrame]:
    """
    Main logic for currency conversion, handling both training and inference modes.

    Args:
        job_type: Type of job (training, validation, testing, calibration)
        input_dir: Input directory for data
        output_dir: Output directory for processed data
        currency_config: Currency conversion configuration dictionary
        load_data_func: Function to load data (for dependency injection in tests)
        save_data_func: Function to save data (for dependency injection in tests)

    Returns:
        Dictionary of converted dataframes
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Using currency conversion configuration: {currency_config}")

    # Load data according to job type
    data_dict = load_data_func(job_type, input_dir)

    # Process the data
    converted_data = process_data(
        data_dict=data_dict,
        job_type=job_type,
        currency_config=currency_config,
    )

    # Save processed data
    save_data_func(job_type, output_dir, converted_data)

    logger.info("Currency conversion complete.")
    return converted_data


def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: Optional[argparse.Namespace] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Standardized main entry point for currency conversion script.

    Args:
        input_paths: Dictionary of input paths with logical names
            - "processed_data": Input data directory (from previous preprocessing step)
        output_paths: Dictionary of output paths with logical names
            - "processed_data": Output directory for converted data
        environ_vars: Dictionary of environment variables
        job_args: Command line arguments containing job_type

    Returns:
        Dictionary of converted dataframes
    """
    try:
        # Extract paths from input parameters - required keys must be present
        if "processed_data" not in input_paths:
            raise ValueError("Missing required input path: processed_data")
        if "processed_data" not in output_paths:
            raise ValueError("Missing required output path: processed_data")

        # Extract job_type from args
        if job_args is None or not hasattr(job_args, "job_type"):
            raise ValueError("job_args must contain job_type parameter")

        job_type = job_args.job_type
        input_dir = input_paths["processed_data"]
        output_dir = output_paths["processed_data"]

        # Log input/output paths for clarity
        logger.info(f"Input data directory: {input_dir}")
        logger.info(f"Output directory: {output_dir}")

        # Load currency conversion configuration from environment variables
        currency_config = {
            "CURRENCY_CODE_FIELD": environ_vars.get("CURRENCY_CODE_FIELD"),
            "MARKETPLACE_ID_FIELD": environ_vars.get("MARKETPLACE_ID_FIELD"),
            "CURRENCY_CONVERSION_VARS": json.loads(
                environ_vars.get("CURRENCY_CONVERSION_VARS", "[]")
            ),
            "CURRENCY_CONVERSION_DICT": json.loads(
                environ_vars.get("CURRENCY_CONVERSION_DICT", "{}")
            ),
            "DEFAULT_CURRENCY": environ_vars.get("DEFAULT_CURRENCY", "USD"),
            "N_WORKERS": int(environ_vars.get("N_WORKERS", "50")),
        }

        # Execute the internal main logic
        return internal_main(
            job_type=job_type,
            input_dir=input_dir,
            output_dir=output_dir,
            currency_config=currency_config,
        )

    except Exception as e:
        logger.error(f"Error in currency conversion: {str(e)}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--job_type",
            type=str,
            required=True,
            choices=["training", "validation", "testing", "calibration"],
            help="Type of job to perform",
        )
        args = parser.parse_args()

        # Define standard paths based on contract
        input_paths = {
            "processed_data": DEFAULT_INPUT_DIR,
        }

        output_paths = {
            "processed_data": DEFAULT_OUTPUT_DIR,
        }

        # Environment variables dictionary
        environ_vars = {
            "CURRENCY_CODE_FIELD": os.environ.get("CURRENCY_CODE_FIELD"),
            "MARKETPLACE_ID_FIELD": os.environ.get("MARKETPLACE_ID_FIELD"),
            "CURRENCY_CONVERSION_VARS": os.environ.get(
                "CURRENCY_CONVERSION_VARS", "[]"
            ),
            "CURRENCY_CONVERSION_DICT": os.environ.get(
                "CURRENCY_CONVERSION_DICT", "{}"
            ),
            "DEFAULT_CURRENCY": os.environ.get("DEFAULT_CURRENCY", "USD"),
            "N_WORKERS": os.environ.get("N_WORKERS", "50"),
        }

        # Execute the main function with standardized inputs
        result = main(input_paths, output_paths, environ_vars, args)

        logger.info(f"Currency conversion completed successfully")
        sys.exit(0)
    except FileNotFoundError as e:
        logger.error(f"File not found error: {str(e)}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Value error: {str(e)}")
        sys.exit(2)
    except Exception as e:
        logger.error(f"Error in currency conversion script: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(3)

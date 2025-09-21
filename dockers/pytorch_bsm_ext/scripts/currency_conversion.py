#!/usr/bin/env python
import os
import json
import argparse
import sys
import traceback
from pathlib import Path
from typing import Tuple, List, Dict, Any, Union, Optional
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
import logging
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_currency_code(
    marketplace_id: Union[int, float],
    marketplace_info: Dict[str, Dict[str, str]],
    default_currency: str,
) -> str:
    """Get currency code for a given marketplace ID."""
    try:
        if pd.isna(marketplace_id) or str(int(marketplace_id)) not in marketplace_info:
            return default_currency
        return marketplace_info[str(int(marketplace_id))]["currency_code"]
    except (ValueError, TypeError):
        return default_currency


def combine_currency_codes(
    df: pd.DataFrame,
    marketplace_id_col: str,
    currency_col: Optional[str],
    marketplace_info: Dict[str, Dict[str, str]],
    default_currency: str,
    skip_invalid_currencies: bool,
) -> Tuple[pd.DataFrame, str]:
    """Combine currency codes from marketplace ID and existing currency column."""
    df["currency_code_from_marketplace_id"] = df[marketplace_id_col].apply(
        lambda x: get_currency_code(x, marketplace_info, default_currency)
    )

    if currency_col and currency_col in df.columns:
        df[currency_col] = df[currency_col].combine_first(
            df["currency_code_from_marketplace_id"]
        )
        final_currency_col = currency_col
    else:
        final_currency_col = "currency_code_from_marketplace_id"

    # Handle invalid currencies
    if not skip_invalid_currencies:
        df = df.dropna(subset=[final_currency_col]).reset_index(drop=True)
    else:
        # Replace invalid currencies with default
        df[final_currency_col] = df[final_currency_col].fillna(default_currency)

    return df, final_currency_col


def currency_conversion_single_variable(
    args: Tuple[pd.DataFrame, str, pd.Series]
) -> pd.Series:
    """Convert single variable's currency values."""
    df, variable, exchange_rate_series = args
    return df[variable] / exchange_rate_series.values


def parallel_currency_conversion(
    df: pd.DataFrame,
    currency_col: str,
    currency_conversion_vars: List[str],
    currency_conversion_dict: Dict[str, float],
    n_workers: int = 50,
) -> pd.DataFrame:
    """Perform parallel currency conversion on multiple variables."""
    exchange_rate_series = df[currency_col].apply(
        lambda x: currency_conversion_dict.get(x, 1.0)
    )
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
    marketplace_id_col: str,
    currency_conversion_vars: List[str],
    currency_conversion_dict: Dict[str, float],
    marketplace_info: Dict[str, Dict[str, str]],
    currency_col: Optional[str] = None,
    default_currency: str = "USD",
    skip_invalid_currencies: bool = False,
    n_workers: int = 50,
) -> pd.DataFrame:
    """Process currency conversion."""
    # Drop rows with missing marketplace IDs
    df = df.dropna(subset=[marketplace_id_col]).reset_index(drop=True)

    # Get and combine currency codes
    df, final_currency_col = combine_currency_codes(
        df,
        marketplace_id_col,
        currency_col,
        marketplace_info,
        default_currency,
        skip_invalid_currencies,
    )

    # Filter variables that exist in the DataFrame
    currency_conversion_vars = [
        var for var in currency_conversion_vars if var in df.columns
    ]

    if currency_conversion_vars:
        logger.info(f"Converting currencies for variables: {currency_conversion_vars}")
        df = parallel_currency_conversion(
            df,
            final_currency_col,
            currency_conversion_vars,
            currency_conversion_dict,
            n_workers,
        )
        logger.info("Currency conversion completed")
    else:
        logger.warning("No variables require currency conversion")

    return df


def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
) -> Dict[str, pd.DataFrame]:
    """
    Main function to execute the currency conversion logic.
    Refactored for improved testability.

    Args:
        input_paths: Dictionary of input paths
        output_paths: Dictionary of output paths
        environ_vars: Dictionary of environment variables
        job_args: Command line arguments

    Returns:
        Dictionary of processed DataFrames by split name
    """
    job_type = job_args.job_type
    mode = job_args.mode

    # Parse input parameters from environment variables
    currency_vars = json.loads(environ_vars.get("CURRENCY_CONVERSION_VARS", "[]"))
    currency_dict = json.loads(environ_vars.get("CURRENCY_CONVERSION_DICT", "{}"))
    marketplace_info = json.loads(environ_vars.get("MARKETPLACE_INFO", "{}"))

    # Extract paths from parameters
    input_base = Path(input_paths.get("data_input", "/opt/ml/processing/input/data"))
    output_base = Path(output_paths.get("data_output", "/opt/ml/processing/output"))
    output_base.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running currency conversion in mode={mode}, job_type={job_type}")

    def apply_conversion(df: pd.DataFrame) -> pd.DataFrame:
        if job_args.enable_conversion:
            return process_currency_conversion(
                df=df,
                marketplace_id_col=job_args.marketplace_id_col,
                currency_conversion_vars=currency_vars,
                currency_conversion_dict=currency_dict,
                marketplace_info=marketplace_info,
                currency_col=job_args.currency_col,
                default_currency=job_args.default_currency,
                skip_invalid_currencies=job_args.skip_invalid_currencies,
                n_workers=job_args.n_workers,
            )
        else:
            logger.info("Conversion disabled—returning original DataFrame")
            return df

    if mode == "split_after_conversion" and job_type == "training":
        # gather all the processed shards from subfolders
        splits = ["train", "test", "val"]
        dfs = []
        for sp in splits:
            fpath = input_base / sp / f"{sp}_processed_data.csv"
            logger.info(f"  Reading split {sp} from {fpath}")
            dfs.append(pd.read_csv(fpath))
        df_all = pd.concat(dfs, ignore_index=True)
        df_conv = apply_conversion(df_all)

        # re-split
        label_field = environ_vars.get("LABEL_FIELD", "label")
        train_df, holdout_df = train_test_split(
            df_conv,
            train_size=job_args.train_ratio,
            random_state=42,
            stratify=df_conv[label_field],
        )
        test_df, val_df = train_test_split(
            holdout_df,
            test_size=job_args.test_val_ratio,
            random_state=42,
            stratify=holdout_df[label_field],
        )

        for split_name, split_df in [
            ("train", train_df),
            ("test", test_df),
            ("val", val_df),
        ]:
            out_dir = output_base / split_name
            out_dir.mkdir(parents=True, exist_ok=True)

            # processed
            proc = split_df.copy()
            proc_path = out_dir / f"{split_name}_processed_data.csv"
            proc.to_csv(proc_path, index=False)
            logger.info(f"Wrote converted processed: {proc_path} (shape={proc.shape})")

            # full (just alias here—but you could re-read your full_data.csv if needed)
            full = proc.copy()
            full_path = out_dir / f"{split_name}_full_data.csv"
            full.to_csv(full_path, index=False)
            logger.info(f"Wrote converted full: {full_path} (shape={full.shape})")

    else:
        # per_split mode OR non‐training split_after_conversion
        if job_type == "training":
            splits = ["train", "test", "val"]
        else:
            splits = [job_type]

        for sp in splits:
            in_dir = input_base / sp
            proc_in_path = in_dir / f"{sp}_processed_data.csv"
            full_in_path = in_dir / f"{sp}_full_data.csv"

            df_proc = pd.read_csv(proc_in_path)
            df_conv = apply_conversion(df_proc)

            out_dir = output_base / sp
            out_dir.mkdir(parents=True, exist_ok=True)

            # write processed
            proc_out = out_dir / f"{sp}_processed_data.csv"
            df_conv.to_csv(proc_out, index=False)
            logger.info(
                f"Converted processed for '{sp}' → {proc_out} (shape={df_conv.shape})"
            )

            # if you want full_data converted as well
            if full_in_path.exists():
                df_full = pd.read_csv(full_in_path)
                df_full_conv = apply_conversion(df_full)
                full_out = out_dir / f"{sp}_full_data.csv"
                df_full_conv.to_csv(full_out, index=False)
                logger.info(
                    f"Converted full for '{sp}' → {full_out} (shape={df_full_conv.shape})"
                )

    logger.info("Currency conversion step complete.")


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--job-type",
            type=str,
            required=True,
            choices=["training", "validation", "testing", "calibration"],
            help="One of ['training','validation','testing','calibration']",
        )
        parser.add_argument(
            "--mode",
            type=str,
            choices=["per_split", "split_after_conversion"],
            default="per_split",
            help=(
                "per_split: apply conversion separately on each existing split folder; "
                "split_after_conversion: combine all data, convert, then re-split"
            ),
        )
        parser.add_argument(
            "--train-ratio",
            type=float,
            default=float(os.environ.get("TRAIN_RATIO", 0.7)),
        )
        parser.add_argument(
            "--test-val-ratio",
            type=float,
            default=float(os.environ.get("TEST_VAL_RATIO", 0.5)),
        )
        parser.add_argument("--n-workers", type=int, default=50)
        parser.add_argument("--marketplace-id-col", required=True)
        parser.add_argument("--currency-col", default=None)
        parser.add_argument("--default-currency", default="USD")
        parser.add_argument("--skip-invalid-currencies", action="store_true")
        parser.add_argument(
            "--enable-conversion", type=lambda x: x.lower() == "true", default=True
        )
        args = parser.parse_args()

        # Standard SageMaker paths
        INPUT_PATH = "/opt/ml/processing/input/data"
        OUTPUT_PATH = "/opt/ml/processing/output"

        # Set up path dictionaries
        input_paths = {"data_input": INPUT_PATH}

        output_paths = {"data_output": OUTPUT_PATH}

        # Environment variables dictionary
        environ_vars = {
            "CURRENCY_CONVERSION_VARS": os.environ.get(
                "CURRENCY_CONVERSION_VARS", "[]"
            ),
            "CURRENCY_CONVERSION_DICT": os.environ.get(
                "CURRENCY_CONVERSION_DICT", "{}"
            ),
            "MARKETPLACE_INFO": os.environ.get("MARKETPLACE_INFO", "{}"),
            "LABEL_FIELD": os.environ.get("LABEL_FIELD", "label"),
            "TRAIN_RATIO": os.environ.get("TRAIN_RATIO", "0.7"),
            "TEST_VAL_RATIO": os.environ.get("TEST_VAL_RATIO", "0.5"),
        }

        # Execute the main function with standardized inputs
        result = main(input_paths, output_paths, environ_vars, args)

        logger.info("Currency conversion completed successfully")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error in currency conversion script: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

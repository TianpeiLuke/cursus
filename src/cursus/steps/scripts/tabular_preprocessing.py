#!/usr/bin/env python
import os
import gzip
import tempfile
import shutil
import csv
import json
import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import Dict, Optional, Callable, Any
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# --- Helper Functions ---


def _is_gzipped(path: str) -> bool:
    return path.lower().endswith(".gz")


def _detect_separator_from_sample(sample_lines: str) -> str:
    """Use csv.Sniffer to detect a delimiter, defaulting to comma."""
    try:
        dialect = csv.Sniffer().sniff(sample_lines)
        return dialect.delimiter
    except Exception:
        return ","


def peek_json_format(file_path: Path, open_func: Callable = open) -> str:
    """Check if the JSON file is in JSON Lines or regular format."""
    try:
        with open_func(str(file_path), "rt") as f:
            first_char = f.read(1)
            if not first_char:
                raise ValueError("Empty file")
            f.seek(0)
            first_line = f.readline().strip()
            try:
                json.loads(first_line)
                return "lines" if first_char != "[" else "regular"
            except json.JSONDecodeError:
                f.seek(0)
                json.loads(f.read())
                return "regular"
    except (json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(f"Error checking JSON format for {file_path}: {e}")


def _read_json_file(file_path: Path) -> pd.DataFrame:
    """Read a JSON or JSON Lines file into a DataFrame."""
    open_func = gzip.open if _is_gzipped(str(file_path)) else open
    fmt = peek_json_format(file_path, open_func)
    if fmt == "lines":
        return pd.read_json(str(file_path), lines=True, compression="infer")
    else:
        with open_func(str(file_path), "rt") as f:
            data = json.load(f)
        return pd.json_normalize(data if isinstance(data, list) else [data])


def _read_file_to_df(file_path: Path) -> pd.DataFrame:
    """Read a single file (CSV, TSV, JSON, Parquet) into a DataFrame."""
    suffix = file_path.suffix.lower()
    if suffix == ".gz":
        inner_ext = Path(file_path.stem).suffix.lower()
        if inner_ext in [".csv", ".tsv"]:
            with gzip.open(str(file_path), "rt") as f:
                sep = _detect_separator_from_sample(f.readline() + f.readline())
            return pd.read_csv(str(file_path), sep=sep, compression="gzip")
        elif inner_ext == ".json":
            return _read_json_file(file_path)
        elif inner_ext.endswith(".parquet"):
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                with gzip.open(str(file_path), "rb") as f_in, open(
                    tmp.name, "wb"
                ) as f_out:
                    shutil.copyfileobj(f_in, f_out)
                df = pd.read_parquet(tmp.name)
            os.unlink(tmp.name)
            return df
        else:
            raise ValueError(f"Unsupported gzipped file type: {file_path}")
    elif suffix in [".csv", ".tsv"]:
        with open(str(file_path), "rt") as f:
            sep = _detect_separator_from_sample(f.readline() + f.readline())
        return pd.read_csv(str(file_path), sep=sep)
    elif suffix == ".json":
        return _read_json_file(file_path)
    elif suffix.endswith(".parquet"):
        return pd.read_parquet(str(file_path))
    else:
        raise ValueError(f"Unsupported file type: {file_path}")


def combine_shards(input_dir: str) -> pd.DataFrame:
    """Detect and combine all supported data shards in a directory."""
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise RuntimeError(f"Input directory does not exist: {input_dir}")
    patterns = [
        "part-*.csv",
        "part-*.csv.gz",
        "part-*.json",
        "part-*.json.gz",
        "part-*.parquet",
        "part-*.snappy.parquet",
        "part-*.parquet.gz",
    ]
    all_shards = sorted([p for pat in patterns for p in input_path.glob(pat)])
    if not all_shards:
        raise RuntimeError(f"No CSV/JSON/Parquet shards found under {input_dir}")
    try:
        dfs = [_read_file_to_df(shard) for shard in all_shards]
        return pd.concat(dfs, axis=0, ignore_index=True)
    except Exception as e:
        raise RuntimeError(f"Failed to read or concatenate shards: {e}")


# --- Main Processing Logic ---


def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Main logic for preprocessing data, refactored for testability.

    Args:
        input_paths: Dictionary of input paths with logical names
        output_paths: Dictionary of output paths with logical names
        environ_vars: Dictionary of environment variables
        job_args: Command line arguments
        logger: Optional logger object (defaults to print if None)

    Returns:
        Dictionary of DataFrames by split name (e.g., 'train', 'test', 'val')
    """
    # Extract parameters from arguments and environment variables
    job_type = job_args.job_type
    label_field = environ_vars.get("LABEL_FIELD")
    train_ratio = float(environ_vars.get("TRAIN_RATIO", 0.7))
    test_val_ratio = float(environ_vars.get("TEST_VAL_RATIO", 0.5))

    # Extract paths
    input_data_dir = input_paths.get("data_input", "/opt/ml/processing/input/data")
    output_dir = output_paths.get("data_output", "/opt/ml/processing/output")
    # Use print function if no logger is provided
    log = logger or print

    # 1. Setup paths
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 2. Combine data shards
    log(f"[INFO] Combining data shards from {input_data_dir}…")
    df = combine_shards(input_data_dir)
    log(f"[INFO] Combined data shape: {df.shape}")

    # 3. Process columns and labels
    df.columns = [col.replace("__DOT__", ".") for col in df.columns]
    if label_field not in df.columns:
        raise RuntimeError(
            f"Label field '{label_field}' not found in columns: {df.columns.tolist()}"
        )

    if not pd.api.types.is_numeric_dtype(df[label_field]):
        unique_labels = sorted(df[label_field].dropna().unique())
        label_map = {val: idx for idx, val in enumerate(unique_labels)}
        df[label_field] = df[label_field].map(label_map)

    df[label_field] = pd.to_numeric(df[label_field], errors="coerce").astype("Int64")
    df.dropna(subset=[label_field], inplace=True)
    df[label_field] = df[label_field].astype(int)
    log(f"[INFO] Data shape after cleaning labels: {df.shape}")

    # 4. Split data if training, otherwise use the job_type as the single split
    if job_type == "training":
        train_df, holdout_df = train_test_split(
            df, train_size=train_ratio, random_state=42, stratify=df[label_field]
        )
        test_df, val_df = train_test_split(
            holdout_df,
            test_size=test_val_ratio,
            random_state=42,
            stratify=holdout_df[label_field],
        )
        splits = {"train": train_df, "test": test_df, "val": val_df}
    else:
        splits = {job_type: df}

    # 5. Save output files
    for split_name, split_df in splits.items():
        subfolder = output_path / split_name
        subfolder.mkdir(exist_ok=True)

        # Only output processed_data.csv
        proc_path = subfolder / f"{split_name}_processed_data.csv"
        split_df.to_csv(proc_path, index=False)
        log(f"[INFO] Saved {proc_path} (shape={split_df.shape})")

    log("[INFO] Preprocessing complete.")
    return splits


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
        LABEL_FIELD = os.environ.get("LABEL_FIELD")
        if not LABEL_FIELD:
            raise RuntimeError("LABEL_FIELD environment variable must be set.")
        TRAIN_RATIO = float(os.environ.get("TRAIN_RATIO", 0.7))
        TEST_VAL_RATIO = float(os.environ.get("TEST_VAL_RATIO", 0.5))

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
        logger.info(f"Starting tabular preprocessing with parameters:")
        logger.info(f"  Job Type: {args.job_type}")
        logger.info(f"  Label Field: {LABEL_FIELD}")
        logger.info(f"  Train Ratio: {TRAIN_RATIO}")
        logger.info(f"  Test/Val Ratio: {TEST_VAL_RATIO}")
        logger.info(f"  Input Directory: {INPUT_DATA_DIR}")
        logger.info(f"  Output Directory: {OUTPUT_DIR}")

        # Set up path dictionaries
        input_paths = {"data_input": INPUT_DATA_DIR}

        output_paths = {"data_output": OUTPUT_DIR}

        # Environment variables dictionary
        environ_vars = {
            "LABEL_FIELD": LABEL_FIELD,
            "TRAIN_RATIO": str(TRAIN_RATIO),
            "TEST_VAL_RATIO": str(TEST_VAL_RATIO),
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
        logger.info(f"Preprocessing completed successfully. Splits: {splits_summary}")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Error in preprocessing script: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)

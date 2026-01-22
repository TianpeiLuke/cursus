#!/usr/bin/env python
"""
Temporal Split Preprocessing Script

Comprehensive preprocessing with temporal splitting capabilities:
1. Temporal cutoff (date-based split for OOT test)
2. Customer-level random split (train/validation)
3. Ensures no customer leakage between train and OOT
4. Parallel processing for large datasets
5. Signature file support
6. Memory-efficient batch concatenation
7. Multiple output formats (CSV, TSV, Parquet)
"""

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
import gc
from pathlib import Path
from typing import Dict, Optional, Callable, Any, List
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split


# ============================================================================
# SHARED UTILITY FUNCTIONS (Used by both Batch and Streaming modes)
# ============================================================================


def load_signature_columns(signature_path: str) -> Optional[list]:
    """
    Load column names from signature file.

    Args:
        signature_path: Path to the signature file directory

    Returns:
        List of column names if signature file exists, None otherwise
    """
    signature_dir = Path(signature_path)
    if not signature_dir.exists():
        return None

    # Look for signature file in the directory
    signature_files = list(signature_dir.glob("*"))
    if not signature_files:
        return None

    # Use the first file found (typically named 'signature')
    signature_file = signature_files[0]

    try:
        with open(signature_file, "r") as f:
            content = f.read().strip()
            if content:
                # Split by comma and strip whitespace
                columns = [col.strip() for col in content.split(",")]
                return columns
    except Exception as e:
        raise RuntimeError(f"Error reading signature file {signature_file}: {e}")

    return None


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


def _read_file_to_df(
    file_path: Path, column_names: Optional[list] = None
) -> pd.DataFrame:
    """Read a single file (CSV, TSV, JSON, Parquet) into a DataFrame."""
    suffix = file_path.suffix.lower()
    if suffix == ".gz":
        inner_ext = Path(file_path.stem).suffix.lower()
        if inner_ext in [".csv", ".tsv"]:
            with gzip.open(str(file_path), "rt") as f:
                sep = _detect_separator_from_sample(f.readline() + f.readline())
            # Use column names from signature if provided for CSV/TSV files
            if column_names:
                return pd.read_csv(
                    str(file_path),
                    sep=sep,
                    compression="gzip",
                    names=column_names,
                    header=0,
                )
            else:
                return pd.read_csv(str(file_path), sep=sep, compression="gzip")
        elif inner_ext == ".json":
            return _read_json_file(file_path)
        elif inner_ext.endswith(".parquet"):
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                with (
                    gzip.open(str(file_path), "rb") as f_in,
                    open(tmp.name, "wb") as f_out,
                ):
                    shutil.copyfileobj(f_in, f_out)
                df = pd.read_parquet(tmp.name)
            os.unlink(tmp.name)
            return df
        else:
            raise ValueError(f"Unsupported gzipped file type: {file_path}")
    elif suffix in [".csv", ".tsv"]:
        with open(str(file_path), "rt") as f:
            sep = _detect_separator_from_sample(f.readline() + f.readline())
        # Use column names from signature if provided for CSV/TSV files
        if column_names:
            return pd.read_csv(str(file_path), sep=sep, names=column_names, header=0)
        else:
            return pd.read_csv(str(file_path), sep=sep)
    elif suffix == ".json":
        return _read_json_file(file_path)
    elif suffix.endswith(".parquet"):
        return pd.read_parquet(str(file_path))
    else:
        raise ValueError(f"Unsupported file type: {file_path}")


def _read_shard_wrapper(args: tuple) -> pd.DataFrame:
    """
    Wrapper function for parallel shard reading.

    Args:
        args: Tuple of (shard_path, signature_columns, shard_index, total_shards)

    Returns:
        DataFrame from the shard
    """
    shard_path, signature_columns, idx, total = args
    try:
        df = _read_file_to_df(shard_path, signature_columns)
        # Log progress (will be captured by parent process)
        print(
            f"[INFO] Processed shard {idx + 1}/{total}: {shard_path.name} ({df.shape[0]} rows)"
        )
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to read shard {shard_path.name}: {e}")


def _batch_concat_dataframes(dfs: list, batch_size: int = 10) -> pd.DataFrame:
    """
    Concatenate DataFrames in batches to minimize memory copies.

    Args:
        dfs: List of DataFrames to concatenate
        batch_size: Number of DataFrames to concatenate at once

    Returns:
        Single concatenated DataFrame
    """
    if not dfs:
        raise ValueError("No DataFrames to concatenate")

    if len(dfs) == 1:
        return dfs[0]

    # Process in batches to reduce intermediate copies
    while len(dfs) > 1:
        batch_results = []
        for i in range(0, len(dfs), batch_size):
            batch = dfs[i : i + batch_size]
            if len(batch) == 1:
                batch_results.append(batch[0])
            else:
                batch_results.append(pd.concat(batch, axis=0, ignore_index=True))
        dfs = batch_results

    return dfs[0]


def combine_shards(
    input_dir: str,
    signature_columns: Optional[list] = None,
    max_workers: Optional[int] = None,
    batch_size: int = 10,
) -> pd.DataFrame:
    """
    Detect and combine all supported data shards in a directory using parallel processing.

    Uses parallel shard reading and batch concatenation for improved performance.
    Memory-efficient approach avoids PyArrow's 2GB column limit error.

    Args:
        input_dir: Directory containing data shards
        signature_columns: Optional column names for CSV/TSV files
        max_workers: Maximum number of parallel workers (default: cpu_count)
        batch_size: Number of DataFrames to concatenate at once (default: 10)

    Returns:
        Combined DataFrame from all shards
    """
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

    total_shards = len(all_shards)
    print(f"[INFO] Found {total_shards} shards to process")

    try:
        # Determine optimal number of workers
        if max_workers is None:
            max_workers = min(cpu_count(), total_shards)

        print(f"[INFO] Using {max_workers} parallel workers for shard reading")

        # Prepare arguments for parallel processing
        shard_args = [
            (shard, signature_columns, i, total_shards)
            for i, shard in enumerate(all_shards)
        ]

        # Read shards in parallel
        if max_workers > 1 and total_shards > 1:
            with Pool(processes=max_workers) as pool:
                dataframes = pool.map(_read_shard_wrapper, shard_args)
        else:
            # Fall back to sequential processing for single shard or single worker
            print("[INFO] Using sequential processing (single worker or single shard)")
            dataframes = [_read_shard_wrapper(args) for args in shard_args]

        if not dataframes:
            raise RuntimeError("No data was loaded from any shards")

        # Log total rows before concatenation
        total_rows = sum(df.shape[0] for df in dataframes)
        print(f"[INFO] Loaded {total_rows} total rows from {total_shards} shards")

        # Concatenate using batch approach
        print(f"[INFO] Concatenating DataFrames with batch_size={batch_size}")
        result_df = _batch_concat_dataframes(dataframes, batch_size)

        # Verify final shape
        print(f"[INFO] Final combined shape: {result_df.shape}")

        return result_df

    except Exception as e:
        raise RuntimeError(f"Failed to read or concatenate shards: {e}")


def generate_main_task_label(
    df: pd.DataFrame,
    targets: list,
    main_task_index: int = 0,
    logger: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    """
    Generate main task label based on subtasks by taking the maximum value across subtasks.

    The main task label is set as the maximum value of all subtasks for each sample.
    For example, if targets=['is_abuse','is_abusive_dnr','is_abusive_pda','is_abusive_rr']
    and main_task_index=0, then 'is_abuse' will be set as the max of the other subtasks.

    Args:
        df: Input DataFrame
        targets: List of target column names (main task + subtasks)
        main_task_index: Index of the main task in the targets list (default: 0)
        logger: Optional logger function

    Returns:
        DataFrame with updated main task labels

    Example:
        >>> targets = ['is_abuse','is_abusive_dnr','is_abusive_pda','is_abusive_rr']
        >>> df = generate_main_task_label(df, targets, main_task_index=0)
        # 'is_abuse' will be set as max('is_abusive_dnr', 'is_abusive_pda', 'is_abusive_rr')
    """
    log = logger or print

    # Validate inputs
    if not targets:
        raise ValueError("targets list cannot be empty")
    if main_task_index < 0 or main_task_index >= len(targets):
        raise ValueError(
            f"main_task_index {main_task_index} is out of range for targets list of length {len(targets)}"
        )

    # Check if all target columns exist in the DataFrame
    missing_columns = [col for col in targets if col not in df.columns]
    if missing_columns:
        raise RuntimeError(
            f"Target columns not found in DataFrame: {missing_columns}. Available columns: {df.columns.tolist()}"
        )

    main_task = targets[main_task_index]

    # Get subtask columns (all targets except the main task)
    subtask_indices = list(range(len(targets)))
    subtask_indices.remove(main_task_index)
    subtasks = [targets[i] for i in subtask_indices]

    log(
        f"[INFO] Generating main task label for '{main_task}' based on subtasks: {subtasks}"
    )
    log(f"[INFO] Original {main_task} value counts:")
    log(f"[INFO] {df[main_task].value_counts().to_dict()}")

    # Store original values for comparison
    original_main_task = df[main_task].copy()

    # Generate main task label as max of subtasks
    if subtasks:
        df[main_task] = df[subtasks].max(axis=1)
        log(f"[INFO] Updated {main_task} value counts after taking max of subtasks:")
        log(f"[INFO] {df[main_task].value_counts().to_dict()}")

        # Log statistics about the change
        changed_samples = (original_main_task != df[main_task]).sum()
        total_samples = len(df)
        log(
            f"[INFO] Changed {changed_samples}/{total_samples} samples ({changed_samples / total_samples * 100:.2f}%)"
        )

        # Log detailed change statistics
        if changed_samples > 0:
            change_summary = pd.crosstab(
                original_main_task, df[main_task], margins=True
            )
            log(f"[INFO] Change summary (original vs new):")
            log(f"[INFO] \n{change_summary}")
    else:
        log(f"[WARNING] No subtasks found for main task '{main_task}', no changes made")

    return df


def temporal_customer_split(
    df: pd.DataFrame,
    date_column: str,
    group_id_column: str,
    split_date: str,
    train_ratio: float = 0.9,
    random_seed: int = 42,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Split data temporally and by group ID.

    Args:
        df: Input DataFrame
        date_column: Name of the date column
        group_id_column: Name of the group ID column
        split_date: Date string for temporal split (format: YYYY-MM-DD)
        train_ratio: Ratio of groups for training (rest go to validation)
        random_seed: Random seed for reproducibility
        logger: Optional logger function

    Returns:
        Dictionary with 'train', 'val', and 'oot' DataFrames
    """
    log = logger or print

    # Convert date column to datetime
    df[date_column] = pd.to_datetime(df[date_column])
    split_date_dt = pd.to_datetime(split_date)

    log(f"[INFO] Splitting data at date: {split_date}")
    log(f"[INFO] Original data shape: {df.shape}")

    # Temporal split: before split_date vs after
    pre_split_df = df[df[date_column] < split_date_dt].copy()
    post_split_df = df[df[date_column] >= split_date_dt].copy()

    log(f"[INFO] Pre-split data shape: {pre_split_df.shape}")
    log(f"[INFO] Post-split data shape (before filtering): {post_split_df.shape}")

    # Get unique groups from pre-split data
    group_ids = list(pre_split_df[group_id_column].unique())
    log(f"[INFO] Total unique groups in pre-split data: {len(group_ids)}")

    # Shuffle groups
    random.seed(random_seed)
    random.shuffle(group_ids)

    # Split groups into train and validation
    train_size = int(len(group_ids) * train_ratio)
    train_groups = group_ids[:train_size]
    val_groups = group_ids[train_size:]

    log(f"[INFO] Train groups: {len(train_groups)}")
    log(f"[INFO] Validation groups: {len(val_groups)}")

    # Create train and validation splits
    train_df = pre_split_df[pre_split_df[group_id_column].isin(train_groups)]
    val_df = pre_split_df[pre_split_df[group_id_column].isin(val_groups)]

    # Remove training groups from OOT to prevent leakage
    oot_df = post_split_df[~post_split_df[group_id_column].isin(train_groups)]

    log(f"[INFO] Final train shape: {train_df.shape}")
    log(f"[INFO] Final validation shape: {val_df.shape}")
    log(f"[INFO] Final OOT shape (after filtering): {oot_df.shape}")

    # Validate that we have data in all splits
    if train_df.empty:
        raise RuntimeError("Training data is empty after temporal split")
    if val_df.empty:
        raise RuntimeError("Validation data is empty after temporal split")
    if oot_df.empty:
        log("[WARNING] OOT (test) data is empty after temporal split and filtering")
        log("[WARNING] This could happen if:")
        log("[WARNING] 1. Split date is too recent (no data after split date)")
        log("[WARNING] 2. All post-split customers were in training set")
        log("[WARNING] Consider adjusting split_date or train_ratio")
        # Create a minimal empty DataFrame with same columns for compatibility
        oot_df = pd.DataFrame(columns=train_df.columns)

    return {"train": train_df, "val": val_df, "oot": oot_df}


# ============================================================================
# BATCH MODE FUNCTIONS
# ============================================================================


def process_batch_mode_temporal_split(
    input_data_dir: str,
    signature_columns: Optional[list],
    date_column: str,
    group_id_column: str,
    split_date: str,
    train_ratio: float,
    random_seed: int,
    targets_str: Optional[str],
    main_task_index: Optional[int],
    label_field: Optional[str],
    output_format: str,
    max_workers: int,
    batch_size: int,
    output_paths: Dict[str, str],
    log_func: Callable,
) -> Dict[str, pd.DataFrame]:
    """
    Process temporal split in batch mode.

    Loads all data into memory, applies temporal split logic, and saves outputs.

    Args:
        input_data_dir: Input data directory
        signature_columns: Optional signature columns
        date_column: Name of date column
        group_id_column: Name of group ID column
        split_date: Temporal split date
        train_ratio: Train/val ratio
        random_seed: Random seed
        targets_str: Optional targets string for multi-task
        main_task_index: Optional main task index
        label_field: Optional label field
        output_format: Output format
        max_workers: Max parallel workers
        batch_size: Batch concatenation size
        output_paths: Output path dictionary
        log_func: Logging function

    Returns:
        Dictionary with training_data and oot_data DataFrames
    """
    log_func("[BATCH] Starting batch mode temporal split preprocessing")

    # Combine data shards
    log_func(f"[BATCH] Combining data shards from {input_data_dir}…")
    df = combine_shards(
        input_data_dir,
        signature_columns=signature_columns,
        max_workers=max_workers,
        batch_size=batch_size,
    )
    log_func(f"[BATCH] Combined data shape: {df.shape}")

    # Process columns
    df.columns = [col.replace("__DOT__", ".") for col in df.columns]

    # Validate required columns exist
    if date_column not in df.columns:
        raise RuntimeError(
            f"Date column '{date_column}' not found in data. Available: {df.columns.tolist()}"
        )
    if group_id_column not in df.columns:
        raise RuntimeError(
            f"Group ID column '{group_id_column}' not found in data. Available: {df.columns.tolist()}"
        )

    # Main task label generation (if targets are provided)
    if targets_str and main_task_index is not None:
        try:
            # Parse targets from string
            if targets_str.startswith("[") and targets_str.endswith("]"):
                import ast

                targets = ast.literal_eval(targets_str)
            else:
                targets = [t.strip().strip("'\"") for t in targets_str.split(",")]

            log_func(f"[BATCH] Generating main task labels with targets: {targets}")
            log_func(f"[BATCH] Main task index: {main_task_index}")

            df = generate_main_task_label(
                df=df, targets=targets, main_task_index=main_task_index, logger=log_func
            )
        except Exception as e:
            log_func(f"[BATCH WARNING] Failed to generate main task labels: {e}")
            log_func("[BATCH WARNING] Continuing without main task label generation")

    # Optional label processing
    if label_field:
        if label_field not in df.columns:
            raise RuntimeError(
                f"Label field '{label_field}' not found in columns: {df.columns.tolist()}"
            )

        if not pd.api.types.is_numeric_dtype(df[label_field]):
            unique_labels = sorted(df[label_field].dropna().unique())
            label_map = {val: idx for idx, val in enumerate(unique_labels)}
            df[label_field] = df[label_field].map(label_map)

        df[label_field] = pd.to_numeric(df[label_field], errors="coerce").astype(
            "Int64"
        )
        df.dropna(subset=[label_field], inplace=True)
        df[label_field] = df[label_field].astype(int)
        log_func(f"[BATCH] Data shape after cleaning labels: {df.shape}")

    # Temporal split
    log_func(f"[BATCH] Performing temporal split at date: {split_date}")
    splits = temporal_customer_split(
        df=df,
        date_column=date_column,
        group_id_column=group_id_column,
        split_date=split_date,
        train_ratio=train_ratio,
        random_seed=random_seed,
        logger=log_func,
    )

    # Validate output format
    if output_format not in ["csv", "tsv", "parquet"]:
        log_func(
            f"[BATCH WARNING] Invalid OUTPUT_FORMAT '{output_format}', defaulting to CSV"
        )
        output_format = "csv"

    # Extract training data and OOT data
    training_data = pd.concat(
        [splits["train"], splits["val"]], axis=0, ignore_index=True
    )
    oot_data = splits["oot"]

    log_func(f"[BATCH] Training data shape (train + val): {training_data.shape}")
    log_func(f"[BATCH] OOT data shape: {oot_data.shape}")

    # Save outputs
    training_output_dir = output_paths.get(
        "training_data",
        output_paths.get("processed_data", "/opt/ml/processing/output/training_data"),
    )
    training_output_path = Path(training_output_dir)
    training_output_path.mkdir(parents=True, exist_ok=True)

    # Create split subdirectories
    for split_name in ["train", "val", "test"]:
        split_dir = training_output_path / split_name
        split_dir.mkdir(exist_ok=True)

    # Save splits
    for split_name, split_df in [
        ("train", splits["train"]),
        ("val", splits["val"]),
        ("test", splits["oot"]),
    ]:
        split_dir = training_output_path / split_name

        if output_format == "csv":
            proc_path = split_dir / f"{split_name}_processed_data.csv"
            split_df.to_csv(proc_path, index=False)
        elif output_format == "tsv":
            proc_path = split_dir / f"{split_name}_processed_data.tsv"
            split_df.to_csv(proc_path, sep="\t", index=False)
        elif output_format == "parquet":
            proc_path = split_dir / f"{split_name}_processed_data.parquet"
            split_df.to_parquet(proc_path, index=False)

        log_func(f"[BATCH] Saved {proc_path} (shape={split_df.shape})")

    # Save OOT data separately
    oot_output_dir = output_paths.get(
        "oot_data",
        output_paths.get("processed_data", "/opt/ml/processing/output/oot_data"),
    )
    oot_output_path = Path(oot_output_dir)
    oot_output_path.mkdir(parents=True, exist_ok=True)

    if output_format == "csv":
        oot_proc_path = oot_output_path / "oot_data.csv"
        oot_data.to_csv(oot_proc_path, index=False)
    elif output_format == "tsv":
        oot_proc_path = oot_output_path / "oot_data.tsv"
        oot_data.to_csv(oot_proc_path, sep="\t", index=False)
    elif output_format == "parquet":
        oot_proc_path = oot_output_path / "oot_data.parquet"
        oot_data.to_parquet(oot_proc_path, index=False)

    log_func(f"[BATCH] Saved OOT {oot_proc_path} (shape={oot_data.shape})")
    log_func("[BATCH] Temporal split preprocessing complete in batch mode")

    return {"training_data": training_data, "oot_data": oot_data}


# ============================================================================
# STREAMING MODE FUNCTIONS
# ============================================================================


def find_input_shards(input_dir: str, log_func: Callable) -> List[Path]:
    """Find all input shards in directory."""
    input_path = Path(input_dir)
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
        raise RuntimeError(f"No shards found in {input_dir}")

    log_func(f"[STREAMING] Found {len(all_shards)} input shards")
    return all_shards


def extract_shard_number(shard_path: Path) -> int:
    """
    Extract shard number from filename like part-00042.csv.

    Handles various formats:
    - part-00042.csv → 42
    - part-00042.csv.gz → 42
    - part-00042.parquet → 42
    - part-00042.snappy.parquet → 42

    Args:
        shard_path: Path to shard file

    Returns:
        Integer shard number

    Raises:
        ValueError: If shard number cannot be extracted

    Example:
        >>> extract_shard_number(Path("part-00042.csv"))
        42
        >>> extract_shard_number(Path("part-00001.csv.gz"))
        1
    """
    stem = shard_path.stem

    # Handle .gz compression
    if stem.endswith(".gz"):
        stem = Path(stem).stem

    # Extract number from part-XXXXX pattern
    import re

    match = re.search(r"part-(\d+)", stem)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(
            f"Cannot extract shard number from {shard_path.name}. "
            f"Expected format: part-XXXXX.ext"
        )


def collect_customer_allocation(
    all_shards: List[Path],
    signature_columns: Optional[list],
    date_column: str,
    group_id_column: str,
    split_date: str,
    train_ratio: float,
    random_seed: int,
    log_func: Callable,
) -> Dict[str, str]:
    """
    Pass 1: Scan pre-split data and build complete customer allocation map.

    Memory: O(unique_customers) - typically ~50MB for 1M customers
    Scans all shards to collect unique customer IDs from pre-split period,
    then randomly allocates them to train vs validation sets.

    Args:
        all_shards: List of all input shard paths
        signature_columns: Optional column names
        date_column: Name of date column for temporal filtering
        group_id_column: Name of customer/group ID column
        split_date: Date string for temporal cutoff (YYYY-MM-DD)
        train_ratio: Proportion of customers for training
        random_seed: Random seed for reproducibility
        log_func: Logging function

    Returns:
        Dictionary mapping customer_id to split assignment ("train" or "val")
        Example: {"customer_1": "train", "customer_2": "val", ...}
    """
    log_func("[PASS 1] Building customer→split mapping...")

    split_date_dt = pd.to_datetime(split_date)
    all_customers = set()

    for i, shard in enumerate(all_shards):
        try:
            # Read only date + group_id columns (very low memory)
            suffix = shard.suffix.lower()

            if suffix.endswith(".parquet") or suffix.endswith(".snappy.parquet"):
                df = pd.read_parquet(shard, columns=[date_column, group_id_column])
            else:
                # For CSV/JSON, read full but keep only needed columns
                df = _read_file_to_df(shard, signature_columns)
                df = df[[date_column, group_id_column]]

            # Convert date column
            df[date_column] = pd.to_datetime(df[date_column])

            # Filter to pre-split data only
            pre_split_df = df[df[date_column] < split_date_dt]

            # Collect unique customers
            all_customers.update(pre_split_df[group_id_column].unique())

            del df, pre_split_df
            gc.collect()

            if (i + 1) % 100 == 0:
                log_func(
                    f"[PASS 1] Processed {i + 1}/{len(all_shards)} shards, "
                    f"found {len(all_customers)} unique customers"
                )

        except Exception as e:
            log_func(f"[PASS 1 WARNING] Failed to read {shard.name}: {e}")
            continue

    if not all_customers:
        raise RuntimeError("No customers found in pre-split data")

    log_func(f"[PASS 1] Total unique customers in pre-split data: {len(all_customers)}")

    # Shuffle and allocate customers to train/val
    customer_list = list(all_customers)
    random.seed(random_seed)
    random.shuffle(customer_list)

    train_size = int(len(customer_list) * train_ratio)

    # Build dictionary mapping
    customer_split_map = {}
    for i, customer in enumerate(customer_list):
        customer_split_map[customer] = "train" if i < train_size else "val"

    # Memory usage estimate
    memory_mb = len(customer_split_map) * 50 / 1024 / 1024
    log_func(f"[PASS 1] Map size: ~{memory_mb:.2f} MB")
    log_func(f"[PASS 1] Allocated: {train_size} train ({train_ratio * 100:.1f}%)")
    log_func(f"[PASS 1] Allocated: {len(customer_list) - train_size} val")

    return customer_split_map


def write_single_shard(
    df: pd.DataFrame,
    output_dir: Path,
    shard_number: int,
    output_format: str = "csv",
) -> Path:
    """
    Write a single data shard in the specified format.

    Note: Kept for potential future use, but not currently used by streaming mode.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_format == "csv":
        shard_path = output_dir / f"part-{shard_number:05d}.csv"
        df.to_csv(shard_path, index=False)
    elif output_format == "tsv":
        shard_path = output_dir / f"part-{shard_number:05d}.tsv"
        df.to_csv(shard_path, sep="\t", index=False)
    elif output_format == "parquet":
        shard_path = output_dir / f"part-{shard_number:05d}.parquet"
        df.to_parquet(shard_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    return shard_path


def process_shard_temporal_split(args: tuple) -> Dict[str, int]:
    """
    Process single shard with global customer allocation map.
    Uses VECTORIZED operations for performance (not df.apply).

    This function runs in parallel across multiple workers.

    Args:
        args: Tuple of (shard_path, shard_num, customer_split_map, config)

    Returns:
        Statistics dict with row counts per split
    """
    shard_path, shard_num, customer_split_map, config = args

    # Extract config
    date_column = config["date_column"]
    group_id_column = config["group_id_column"]
    split_date = config["split_date"]
    output_base = config["output_base"]
    output_format = config["output_format"]
    signature_columns = config.get("signature_columns")
    targets = config.get("targets")
    main_task_index = config.get("main_task_index")
    label_field = config.get("label_field")

    # Read shard
    df = _read_file_to_df(shard_path, signature_columns)
    df.columns = [col.replace("__DOT__", ".") for col in df.columns]

    # Apply preprocessing (multi-task labels, label processing, etc.)
    if targets and main_task_index is not None:
        df = generate_main_task_label(df, targets, main_task_index, lambda x: None)

    if label_field and label_field in df.columns:
        if not pd.api.types.is_numeric_dtype(df[label_field]):
            unique_labels = sorted(df[label_field].dropna().unique())
            label_map = {val: idx for idx, val in enumerate(unique_labels)}
            df[label_field] = df[label_field].map(label_map)

        df[label_field] = pd.to_numeric(df[label_field], errors="coerce").astype(
            "Int64"
        )
        df.dropna(subset=[label_field], inplace=True)
        df[label_field] = df[label_field].astype(int)

    # Convert date column
    df[date_column] = pd.to_datetime(df[date_column])
    split_date_dt = pd.to_datetime(split_date)

    # ============================================
    # VECTORIZED TEMPORAL SPLIT ASSIGNMENT
    # ============================================

    # Step 1: Determine pre-split vs post-split (vectorized boolean)
    is_pre_split = df[date_column] < split_date_dt

    # Step 2: Map customer IDs to split assignments (vectorized)
    # customer_split_map.get() returns "train" or "val" or None
    customer_splits = df[group_id_column].map(customer_split_map)

    # Step 3: Assign final splits using vectorized conditions
    df["_split"] = None  # Default: will be filtered out

    # Pre-split data: use customer assignment ("train" or "val")
    df.loc[is_pre_split, "_split"] = customer_splits[is_pre_split]

    # Post-split data: assign "oot" only if customer is NOT in train set
    # (i.e., customer is in val set or unknown)
    is_post_split = ~is_pre_split
    is_not_train = customer_splits != "train"  # Val or None
    df.loc[is_post_split & is_not_train, "_split"] = "oot"

    # Step 4: Filter out None assignments (train customers in post-split period)
    initial_rows = len(df)
    df = df[df["_split"].notna()]
    filtered_rows = initial_rows - len(df)

    # Log filtering stats (will be captured by parent process)
    if filtered_rows > 0:
        print(
            f"[Shard {shard_num}] Filtered {filtered_rows} rows "
            f"({filtered_rows / initial_rows * 100:.1f}%)"
        )

    # Step 5: Write to split folders with preserved shard number
    stats = {}
    for split_name in ["train", "val", "oot"]:
        split_df = df[df["_split"] == split_name].drop("_split", axis=1)

        if len(split_df) > 0:
            split_dir = output_base / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            output_path = split_dir / f"part-{shard_num:05d}.{output_format}"

            if output_format == "csv":
                split_df.to_csv(output_path, index=False)
            elif output_format == "tsv":
                split_df.to_csv(output_path, sep="\t", index=False)
            elif output_format == "parquet":
                split_df.to_parquet(output_path, index=False)

            stats[split_name] = len(split_df)
            print(
                f"[Shard {shard_num}] Wrote {output_path.name} ({len(split_df)} rows)"
            )
        else:
            stats[split_name] = 0

    return stats


def process_streaming_temporal_split_parallel(
    all_shards: List[Path],
    training_output_path: Path,
    customer_split_map: Dict[str, str],
    config: Dict,
    max_workers: int,
    log_func: Callable,
) -> None:
    """
    Pass 2: Process all shards in parallel using customer map.

    Args:
        all_shards: List of all input shard paths
        training_output_path: Base output directory
        customer_split_map: Customer→split mapping from Pass 1
        config: Configuration dictionary
        max_workers: Number of parallel workers
        log_func: Logging function
    """
    log_func(f"[PASS 2] Processing {len(all_shards)} shards with {max_workers} workers")
    log_func("[PASS 2] Using PARALLEL processing with vectorized operations")

    # Prepare arguments for each shard
    shard_args = [
        (shard, extract_shard_number(shard), customer_split_map, config)
        for shard in all_shards
    ]

    # ✅ Process ALL shards in parallel
    with Pool(processes=max_workers) as pool:
        results = pool.map(process_shard_temporal_split, shard_args)

    # Aggregate statistics
    total_stats = {
        "train": sum(r.get("train", 0) for r in results),
        "val": sum(r.get("val", 0) for r in results),
        "oot": sum(r.get("oot", 0) for r in results),
    }

    # Count non-empty shards per split
    shard_counts = {
        "train": sum(1 for r in results if r.get("train", 0) > 0),
        "val": sum(1 for r in results if r.get("val", 0) > 0),
        "oot": sum(1 for r in results if r.get("oot", 0) > 0),
    }

    log_func(f"[PASS 2] Complete! Row distribution: {total_stats}")
    log_func(
        f"[PASS 2] Output shards - train={shard_counts['train']}, "
        f"val={shard_counts['val']}, oot={shard_counts['oot']}"
    )


def consolidate_shards_to_single_files(
    training_output_path: Path, output_format: str, log_func: Callable
) -> Dict[str, pd.DataFrame]:
    """Consolidate temporary shards into single files per split."""
    log_func("[STREAMING] Consolidating shards into single files per split...")

    result = {}

    for split_name in ["train", "val", "oot"]:
        split_dir = training_output_path / split_name
        if not split_dir.exists():
            log_func(
                f"[STREAMING WARNING] {split_name} directory does not exist, skipping"
            )
            result[split_name] = pd.DataFrame()
            continue

        # Find all shards for this split
        shard_files = sorted(split_dir.glob(f"part-*.{output_format}"))
        if not shard_files:
            log_func(
                f"[STREAMING WARNING] No {split_name} shards found, creating empty DataFrame"
            )
            result[split_name] = pd.DataFrame()
            continue

        log_func(f"[STREAMING] Consolidating {len(shard_files)} {split_name} shards...")

        # Read and concatenate all shards
        shard_dfs = []
        for shard_file in shard_files:
            if output_format == "csv":
                shard_df = pd.read_csv(shard_file)
            elif output_format == "tsv":
                shard_df = pd.read_csv(shard_file, sep="\t")
            elif output_format == "parquet":
                shard_df = pd.read_parquet(shard_file)
            shard_dfs.append(shard_df)

        # Concatenate all shards
        consolidated_df = pd.concat(shard_dfs, axis=0, ignore_index=True)
        del shard_dfs
        gc.collect()

        # Save as "test" for OOT split (for consistency with batch mode)
        save_name = "test" if split_name == "oot" else split_name

        # Write consolidated file
        if output_format == "csv":
            output_file = split_dir / f"{save_name}_processed_data.csv"
            consolidated_df.to_csv(output_file, index=False)
        elif output_format == "tsv":
            output_file = split_dir / f"{save_name}_processed_data.tsv"
            consolidated_df.to_csv(output_file, sep="\t", index=False)
        elif output_format == "parquet":
            output_file = split_dir / f"{save_name}_processed_data.parquet"
            consolidated_df.to_parquet(output_file, index=False)

        log_func(f"[STREAMING] Wrote {output_file} (shape={consolidated_df.shape})")

        result[split_name] = consolidated_df

        # Delete shard files
        for shard_file in shard_files:
            shard_file.unlink()
        log_func(f"[STREAMING] Cleaned up {len(shard_files)} temporary shards")

    return result


# ============================================================================
# MAIN PROCESSING LOGIC
# ============================================================================


def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Main logic for temporal split preprocessing with comprehensive features, refactored for testability.

    Args:
        input_paths: Dictionary of input paths with logical names
        output_paths: Dictionary of output paths with logical names
        environ_vars: Dictionary of environment variables
        job_args: Command line arguments
        logger: Optional logger object (defaults to print if None)

    Returns:
        Dictionary of DataFrames by split name (e.g., 'train', 'val', 'oot')
    """
    # Extract parameters from arguments and environment variables
    job_type = environ_vars.get("JOB_TYPE", "training")
    date_column = environ_vars.get("DATE_COLUMN")
    group_id_column = environ_vars.get("GROUP_ID_COLUMN")
    split_date = environ_vars.get("SPLIT_DATE")
    train_ratio = float(environ_vars.get("TRAIN_RATIO", 0.9))
    random_seed = int(environ_vars.get("RANDOM_SEED", 42))
    output_format = environ_vars.get("OUTPUT_FORMAT", "CSV").lower()
    max_workers_str = environ_vars.get("MAX_WORKERS", "4")
    if (
        max_workers_str
        and str(max_workers_str).lower() != "none"
        and str(max_workers_str).strip() != ""
    ):
        try:
            max_workers = int(max_workers_str)
        except ValueError as e:
            raise RuntimeError(
                f"Invalid MAX_WORKERS value: '{max_workers_str}'. Error: {e}"
            )
    else:
        max_workers = 4  # Default to 4 workers
    batch_size = int(environ_vars.get("BATCH_SIZE", 10))

    # Streaming mode parameters
    enable_true_streaming = (
        environ_vars.get("ENABLE_TRUE_STREAMING", "false").lower() == "true"
    )

    # Optional label processing (for compatibility with standard preprocessing)
    label_field_raw = environ_vars.get("LABEL_FIELD")
    if (
        label_field_raw
        and str(label_field_raw).lower() != "none"
        and str(label_field_raw).strip() != ""
    ):
        label_field = label_field_raw
    else:
        label_field = None

    # Main task label generation parameters
    targets_str = environ_vars.get("TARGETS")
    main_task_index = environ_vars.get("MAIN_TASK_INDEX")
    if (
        main_task_index is not None
        and str(main_task_index).lower() != "none"
        and str(main_task_index).strip() != ""
    ):
        main_task_index = int(main_task_index)
    else:
        main_task_index = None

    # Extract paths
    input_data_dir = input_paths["DATA"]
    input_signature_dir = input_paths.get("SIGNATURE")
    # Handle both old single output and new dual output formats
    if "training_data" in output_paths and "oot_data" in output_paths:
        # New dual output format
        output_dir = None  # Will be handled separately for each output
    else:
        # Legacy single output format
        output_dir = output_paths.get(
            "processed_data",
            output_paths.get("training_data", list(output_paths.values())[0]),
        )

    # Use print function if no logger is provided
    log = logger or print

    # Validate required temporal split parameters
    if not date_column:
        raise RuntimeError("DATE_COLUMN environment variable must be set.")
    if not group_id_column:
        raise RuntimeError("GROUP_ID_COLUMN environment variable must be set.")
    if not split_date:
        raise RuntimeError("SPLIT_DATE environment variable must be set.")

    # 1. Setup paths - handle legacy single output or new dual output
    if output_dir:
        # Legacy single output format
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        # New dual output format - paths will be created later
        output_path = None

    # 2. Load signature columns if available
    signature_columns = None
    if input_signature_dir:
        signature_columns = load_signature_columns(input_signature_dir)
        if signature_columns:
            log(f"[INFO] Loaded signature with {len(signature_columns)} columns")
        else:
            log("[INFO] No signature file found, using default column handling")
    else:
        log("[INFO] No signature directory provided, using default column handling")

    # ========================================================================
    # MODE ROUTING: Simple if/else switch between streaming and batch modes
    # ========================================================================

    if enable_true_streaming:
        # ==================================================================
        # STREAMING MODE
        # ==================================================================
        log("[INFO] Using TRUE STREAMING MODE for temporal split")

        # Find input shards
        all_shards = find_input_shards(input_data_dir, log)

        # Parse targets if needed
        targets = None
        if targets_str and main_task_index is not None:
            if targets_str.startswith("["):
                import ast

                targets = ast.literal_eval(targets_str)
            else:
                targets = [t.strip().strip("'\"") for t in targets_str.split(",")]
            log(f"[STREAMING] Multi-task targets: {targets}")

        # PASS 1: Collect customer allocation (now returns dictionary)
        log("[STREAMING] ===== STARTING PASS 1: Customer allocation =====")
        customer_split_map = collect_customer_allocation(
            all_shards,
            signature_columns,
            date_column,
            group_id_column,
            split_date,
            train_ratio,
            random_seed,
            log,
        )
        log("[STREAMING] ===== PASS 1 COMPLETE =====")
        log("")

        # PASS 2: Process shards in parallel
        training_output_dir = output_paths.get(
            "training_data",
            output_paths.get(
                "processed_data", "/opt/ml/processing/output/training_data"
            ),
        )
        training_output_path = Path(training_output_dir)
        training_output_path.mkdir(parents=True, exist_ok=True)

        # Build config dictionary for parallel processing
        config = {
            "date_column": date_column,
            "group_id_column": group_id_column,
            "split_date": split_date,
            "output_base": training_output_path,
            "output_format": output_format,
            "signature_columns": signature_columns,
            "targets": targets,
            "main_task_index": main_task_index,
            "label_field": label_field,
        }

        # Determine optimal number of workers
        if max_workers is None or max_workers == 0:
            max_workers = min(cpu_count(), len(all_shards))

        log(f"[STREAMING] Using {max_workers} parallel workers for Pass 2")

        # Use parallel processing
        log("[STREAMING] ===== STARTING PASS 2: Parallel processing =====")
        process_streaming_temporal_split_parallel(
            all_shards,
            training_output_path,
            customer_split_map,
            config,
            max_workers,
            log,
        )
        log("[STREAMING] ===== PASS 2 COMPLETE =====")

        # TRUE STREAMING MODE: Keep sharded output
        log("[STREAMING] Preserving sharded output (1:1 shard mapping)")
        log("[STREAMING] Output uses 1:1 shard mapping for PyTorch compatibility")

        # Don't consolidate - data stays in shards
        training_data = pd.DataFrame()
        oot_data = pd.DataFrame()

        # Save OOT data separately
        oot_output_dir = output_paths.get(
            "oot_data",
            output_paths.get("processed_data", "/opt/ml/processing/output/oot_data"),
        )
        oot_output_path = Path(oot_output_dir)
        oot_output_path.mkdir(parents=True, exist_ok=True)

        # True streaming mode: Copy OOT shards
        oot_source_dir = training_output_path / "oot"
        if oot_source_dir.exists():
            log(
                f"[STREAMING] Copying OOT shards from {oot_source_dir} to {oot_output_path}"
            )
            for shard_file in sorted(oot_source_dir.glob(f"part-*.{output_format}")):
                dest_file = oot_output_path / shard_file.name
                shutil.copy2(shard_file, dest_file)
                log(f"[STREAMING] Copied {shard_file.name}")
        else:
            log("[STREAMING WARNING] No OOT shards found to copy")

        log("[STREAMING] Temporal split preprocessing complete in streaming mode")
        return {"training_data": training_data, "oot_data": oot_data}

    # ========================================================================
    # BATCH MODE (existing code continues)
    # ========================================================================

    log("[INFO] Using BATCH MODE for temporal split")

    # 3. Combine data shards with advanced features
    log(f"[INFO] Combining data shards from {input_data_dir}…")
    df = combine_shards(
        input_data_dir,
        signature_columns=signature_columns,
        max_workers=max_workers,
        batch_size=batch_size,
    )
    log(f"[INFO] Combined data shape: {df.shape}")

    # 4. Process columns and labels (conditional based on label_field availability)
    df.columns = [col.replace("__DOT__", ".") for col in df.columns]

    # Validate required columns exist
    if date_column not in df.columns:
        raise RuntimeError(
            f"Date column '{date_column}' not found in data. Available: {df.columns.tolist()}"
        )
    if group_id_column not in df.columns:
        raise RuntimeError(
            f"Group ID column '{group_id_column}' not found in data. Available: {df.columns.tolist()}"
        )

    # Main task label generation (if targets are provided)
    if targets_str and main_task_index is not None:
        try:
            # Parse targets from string (assuming comma-separated or JSON format)
            if targets_str.startswith("[") and targets_str.endswith("]"):
                # JSON format: ['is_abuse','is_abusive_dnr','is_abusive_pda']
                import ast

                targets = ast.literal_eval(targets_str)
            else:
                # Comma-separated format: is_abuse,is_abusive_dnr,is_abusive_pda
                targets = [t.strip().strip("'\"") for t in targets_str.split(",")]

            log(f"[INFO] Generating main task labels with targets: {targets}")
            log(f"[INFO] Main task index: {main_task_index}")

            df = generate_main_task_label(
                df=df, targets=targets, main_task_index=main_task_index, logger=log
            )
        except Exception as e:
            log(f"[WARNING] Failed to generate main task labels: {e}")
            log("[WARNING] Continuing without main task label generation")
    else:
        log(
            "[INFO] No targets or main_task_index provided, skipping main task label generation"
        )

    # Optional label processing (for compatibility with standard preprocessing)
    if label_field:
        if label_field not in df.columns:
            raise RuntimeError(
                f"Label field '{label_field}' not found in columns: {df.columns.tolist()}"
            )

        if not pd.api.types.is_numeric_dtype(df[label_field]):
            unique_labels = sorted(df[label_field].dropna().unique())
            label_map = {val: idx for idx, val in enumerate(unique_labels)}
            df[label_field] = df[label_field].map(label_map)

        df[label_field] = pd.to_numeric(df[label_field], errors="coerce").astype(
            "Int64"
        )
        df.dropna(subset=[label_field], inplace=True)
        df[label_field] = df[label_field].astype(int)
        log(f"[INFO] Data shape after cleaning labels: {df.shape}")
    else:
        log("[INFO] No label field provided, skipping label processing")

    log(f"[INFO] Starting temporal split preprocessing")
    log(f"[INFO] Date column: {date_column}")
    log(f"[INFO] Group ID column: {group_id_column}")
    log(f"[INFO] Split date: {split_date}")
    log(f"[INFO] Train ratio: {train_ratio}")
    log(f"[INFO] Random seed: {random_seed}")
    log(f"[INFO] Output format: {output_format}")
    log(f"[INFO] Max workers: {max_workers if max_workers else 'auto'}")
    log(f"[INFO] Batch size: {batch_size}")

    # 5. Split data temporally - always create training and OOT splits
    splits = temporal_customer_split(
        df=df,
        date_column=date_column,
        group_id_column=group_id_column,
        split_date=split_date,
        train_ratio=train_ratio,
        random_seed=random_seed,
        logger=log,
    )

    # 6. Save output files to two separate output directories
    # Validate output format
    if output_format not in ["csv", "tsv", "parquet"]:
        log(f"[WARNING] Invalid OUTPUT_FORMAT '{output_format}', defaulting to CSV")
        output_format = "csv"

    # Extract training data (train + val) and OOT data
    training_data = pd.concat(
        [splits["train"], splits["val"]], axis=0, ignore_index=True
    )
    oot_data = splits["oot"]

    log(f"[INFO] Training data shape (train + val): {training_data.shape}")
    log(f"[INFO] OOT data shape: {oot_data.shape}")

    # Save training data to training_data output path
    training_output_dir = output_paths.get(
        "training_data",
        output_paths.get("processed_data", "/opt/ml/processing/output/training_data"),
    )
    training_output_path = Path(training_output_dir)
    training_output_path.mkdir(parents=True, exist_ok=True)

    # Create train/val/test subdirectories for lightgbmmt_training compatibility
    train_dir = training_output_path / "train"
    val_dir = training_output_path / "val"
    test_dir = training_output_path / "test"  # Add test directory for OOT data
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)

    # Save train, val, and test (OOT) splits for training
    for split_name, split_df in [
        ("train", splits["train"]),
        ("val", splits["val"]),
        ("test", splits["oot"]),
    ]:
        split_dir = training_output_path / split_name

        # Always save files, even if empty (downstream steps expect files to exist)
        # Use standard naming convention: {split_name}_processed_data.{ext}
        if output_format == "csv":
            proc_path = split_dir / f"{split_name}_processed_data.csv"
            split_df.to_csv(proc_path, index=False)
        elif output_format == "tsv":
            proc_path = split_dir / f"{split_name}_processed_data.tsv"
            split_df.to_csv(proc_path, sep="\t", index=False)
        elif output_format == "parquet":
            proc_path = split_dir / f"{split_name}_processed_data.parquet"
            split_df.to_parquet(proc_path, index=False)

        log(
            f"[INFO] Saved training {proc_path} (format={output_format}, shape={split_df.shape})"
        )
        if split_df.empty:
            log(
                f"[WARNING] {split_name} data is empty - this may cause issues in downstream steps"
            )

    # Save OOT data to oot_data output path
    oot_output_dir = output_paths.get(
        "oot_data",
        output_paths.get("processed_data", "/opt/ml/processing/output/oot_data"),
    )
    oot_output_path = Path(oot_output_dir)
    oot_output_path.mkdir(parents=True, exist_ok=True)

    # Determine file extension and save method based on output format
    if output_format == "csv":
        oot_proc_path = oot_output_path / "oot_data.csv"
        oot_data.to_csv(oot_proc_path, index=False)
    elif output_format == "tsv":
        oot_proc_path = oot_output_path / "oot_data.tsv"
        oot_data.to_csv(oot_proc_path, sep="\t", index=False)
    elif output_format == "parquet":
        oot_proc_path = oot_output_path / "oot_data.parquet"
        oot_data.to_parquet(oot_proc_path, index=False)

    log(
        f"[INFO] Saved OOT {oot_proc_path} (format={output_format}, shape={oot_data.shape})"
    )

    log("[INFO] Temporal split preprocessing complete.")
    return {"training_data": training_data, "oot_data": oot_data}


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

        # DEBUG: Print comprehensive debugging information
        print("=" * 80)
        print("DEBUG: Script execution information:")
        print(f"  Script path: {__file__}")
        print(f"  Working directory: {os.getcwd()}")
        print(f"  Python executable: {sys.executable}")
        print("DEBUG: Command line arguments received:")
        print(f"  sys.argv = {sys.argv}")
        print(f"  Total arguments: {len(sys.argv)}")
        print("DEBUG: Parsed arguments:")
        print(f"  args = {args}")
        print("DEBUG: SageMaker environment paths:")
        print(f"  SM_MODEL_DIR: {os.environ.get('SM_MODEL_DIR', 'Not set')}")
        print(
            f"  SM_OUTPUT_DATA_DIR: {os.environ.get('SM_OUTPUT_DATA_DIR', 'Not set')}"
        )
        print(
            f"  SM_CHANNEL_TRAINING: {os.environ.get('SM_CHANNEL_TRAINING', 'Not set')}"
        )
        print("DEBUG: Relevant environment variables:")
        for key, value in sorted(os.environ.items()):
            if any(
                keyword in key.upper()
                for keyword in [
                    "DATE",
                    "GROUP",
                    "SPLIT",
                    "TARGET",
                    "TRAIN",
                    "RANDOM",
                    "OUTPUT",
                    "LABEL",
                    "SM_",
                ]
            ):
                print(f"  {key} = {value}")
        print("=" * 80)

        # Read configuration from environment variables
        JOB_TYPE = os.environ.get(
            "JOB_TYPE", "training"
        )  # Default to training if not set
        DATE_COLUMN = os.environ.get("DATE_COLUMN")
        GROUP_ID_COLUMN = os.environ.get("GROUP_ID_COLUMN")
        SPLIT_DATE = os.environ.get("SPLIT_DATE")

        # Handle numeric conversions with better error handling
        try:
            TRAIN_RATIO = float(os.environ.get("TRAIN_RATIO", 0.9))
        except ValueError as e:
            raise RuntimeError(
                f"Invalid TRAIN_RATIO value: {os.environ.get('TRAIN_RATIO')}. Error: {e}"
            )

        try:
            RANDOM_SEED = int(os.environ.get("RANDOM_SEED", 42))
        except ValueError as e:
            raise RuntimeError(
                f"Invalid RANDOM_SEED value: {os.environ.get('RANDOM_SEED')}. Error: {e}"
            )

        OUTPUT_FORMAT = os.environ.get("OUTPUT_FORMAT", "CSV")

        # Advanced processing parameters
        MAX_WORKERS_RAW = os.environ.get("MAX_WORKERS", "4")
        try:
            if (
                MAX_WORKERS_RAW
                and str(MAX_WORKERS_RAW).lower() != "none"
                and str(MAX_WORKERS_RAW).strip() != ""
            ):
                MAX_WORKERS = int(MAX_WORKERS_RAW)
            else:
                MAX_WORKERS = 4  # Default to 4 workers
        except ValueError as e:
            raise RuntimeError(
                f"Invalid MAX_WORKERS value: '{MAX_WORKERS_RAW}'. Error: {e}"
            )

        try:
            BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 10))
        except ValueError as e:
            raise RuntimeError(
                f"Invalid BATCH_SIZE value: {os.environ.get('BATCH_SIZE')}. Error: {e}"
            )

        # Streaming mode parameters
        ENABLE_TRUE_STREAMING = (
            os.environ.get("ENABLE_TRUE_STREAMING", "false").lower() == "true"
        )

        # Optional label processing (for compatibility with standard preprocessing)
        LABEL_FIELD_RAW = os.environ.get("LABEL_FIELD")
        if (
            LABEL_FIELD_RAW
            and str(LABEL_FIELD_RAW).lower() != "none"
            and str(LABEL_FIELD_RAW).strip() != ""
        ):
            LABEL_FIELD = LABEL_FIELD_RAW
        else:
            LABEL_FIELD = None

        # Main task label generation parameters
        TARGETS_RAW = os.environ.get("TARGETS")
        if (
            TARGETS_RAW
            and str(TARGETS_RAW).lower() != "none"
            and str(TARGETS_RAW).strip() != ""
        ):
            TARGETS = TARGETS_RAW
        else:
            TARGETS = None
        MAIN_TASK_INDEX_RAW = os.environ.get("MAIN_TASK_INDEX")

        try:
            if (
                MAIN_TASK_INDEX_RAW is not None
                and MAIN_TASK_INDEX_RAW.lower() != "none"
                and MAIN_TASK_INDEX_RAW.strip() != ""
            ):
                MAIN_TASK_INDEX = int(MAIN_TASK_INDEX_RAW)
            else:
                MAIN_TASK_INDEX = None
        except ValueError as e:
            raise RuntimeError(
                f"Invalid MAIN_TASK_INDEX value: '{MAIN_TASK_INDEX_RAW}'. Error: {e}"
            )

        # DEBUG: Print the specific values we're looking for
        print("DEBUG: Specific environment variable values:")
        print(f"  JOB_TYPE = '{JOB_TYPE}'")
        print(f"  DATE_COLUMN = '{DATE_COLUMN}'")
        print(f"  GROUP_ID_COLUMN = '{GROUP_ID_COLUMN}'")
        print(f"  SPLIT_DATE = '{SPLIT_DATE}'")
        print(f"  TARGETS = '{TARGETS}'")
        print(f"  MAIN_TASK_INDEX = '{MAIN_TASK_INDEX}'")
        print("=" * 80)

        # Validate required parameters
        if not DATE_COLUMN:
            raise RuntimeError("DATE_COLUMN environment variable must be set.")
        if not GROUP_ID_COLUMN:
            raise RuntimeError("GROUP_ID_COLUMN environment variable must be set.")
        if not SPLIT_DATE:
            raise RuntimeError("SPLIT_DATE environment variable must be set.")

        # Define standard SageMaker paths as constants
        INPUT_DATA_DIR = "/opt/ml/processing/input/data"
        INPUT_SIGNATURE_DIR = "/opt/ml/processing/input/signature"
        OUTPUT_DIR = "/opt/ml/processing/output"

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logger = logging.getLogger(__name__)

        # Log key parameters
        logger.info(f"Starting temporal split preprocessing with parameters:")
        logger.info(f"  Job Type: {JOB_TYPE}")
        logger.info(f"  Date Column: {DATE_COLUMN}")
        logger.info(f"  Group ID Column: {GROUP_ID_COLUMN}")
        logger.info(f"  Split Date: {SPLIT_DATE}")
        logger.info(f"  Train Ratio: {TRAIN_RATIO}")
        logger.info(f"  Random Seed: {RANDOM_SEED}")
        logger.info(f"  Output Format: {OUTPUT_FORMAT}")
        logger.info(f"  Max Workers: {MAX_WORKERS if MAX_WORKERS else 'auto'}")
        logger.info(f"  Batch Size: {BATCH_SIZE}")
        logger.info(f"  Label Field: {LABEL_FIELD if LABEL_FIELD else 'Not specified'}")
        logger.info(f"  Targets: {TARGETS if TARGETS else 'Not specified'}")
        logger.info(
            f"  Main Task Index: {MAIN_TASK_INDEX if MAIN_TASK_INDEX else 'Not specified'}"
        )

        logger.info(f"  Input Directory: {INPUT_DATA_DIR}")
        logger.info(f"  Input Signature Directory: {INPUT_SIGNATURE_DIR}")
        logger.info(f"  Output Directory: {OUTPUT_DIR}")

        # Set up path dictionaries for dual outputs
        input_paths = {"DATA": INPUT_DATA_DIR, "SIGNATURE": INPUT_SIGNATURE_DIR}
        output_paths = {
            "training_data": "/opt/ml/processing/output/training_data",
            "oot_data": "/opt/ml/processing/output/oot_data",
        }

        # Environment variables dictionary
        environ_vars = {
            "JOB_TYPE": JOB_TYPE,
            "DATE_COLUMN": DATE_COLUMN,
            "GROUP_ID_COLUMN": GROUP_ID_COLUMN,
            "SPLIT_DATE": SPLIT_DATE,
            "TRAIN_RATIO": str(TRAIN_RATIO),
            "RANDOM_SEED": str(RANDOM_SEED),
            "OUTPUT_FORMAT": OUTPUT_FORMAT,
            "MAX_WORKERS": str(MAX_WORKERS),
            "BATCH_SIZE": str(BATCH_SIZE),
            "ENABLE_TRUE_STREAMING": str(ENABLE_TRUE_STREAMING).lower(),
            "LABEL_FIELD": LABEL_FIELD,
            "TARGETS": TARGETS,
            "MAIN_TASK_INDEX": MAIN_TASK_INDEX,
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
        logger.info(
            f"Temporal split preprocessing completed successfully. Splits: {splits_summary}"
        )
        sys.exit(0)
    except Exception as e:
        logging.error(f"Error in temporal split preprocessing script: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)

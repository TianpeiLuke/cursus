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
import random
from pathlib import Path
from typing import Dict, Optional, Callable, Any, List
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import train_test_split

# ============================================================================
# SHARED UTILITY FUNCTIONS (Used by both Batch and Streaming modes)
# ============================================================================


def optimize_dtypes(
    df: pd.DataFrame, log_func: Optional[Callable] = None
) -> pd.DataFrame:
    """
    Optimize DataFrame dtypes to reduce memory usage.

    Applies the following optimizations:
    - Downcast numeric types (int64->int32, float64->float32)
    - Convert object columns with low cardinality to category

    Args:
        df: Input DataFrame
        log_func: Optional logging function

    Returns:
        DataFrame with optimized dtypes
    """
    log = log_func or print
    initial_memory = df.memory_usage(deep=True).sum() / 1024**2

    # Downcast numeric columns
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    # Convert low-cardinality object columns to category
    for col in df.select_dtypes(include=["object"]).columns:
        num_unique = df[col].nunique()
        num_total = len(df[col])
        if num_unique / num_total < 0.5:  # Less than 50% unique values
            df[col] = df[col].astype("category")

    final_memory = df.memory_usage(deep=True).sum() / 1024**2
    reduction = (1 - final_memory / initial_memory) * 100
    log(
        f"[INFO] Memory optimization: {initial_memory:.2f} MB -> {final_memory:.2f} MB ({reduction:.1f}% reduction)"
    )

    return df


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


def process_label_column(
    df: pd.DataFrame, label_field: str, log_func: Callable
) -> pd.DataFrame:
    """
    Process label column: convert to numeric and handle missing values.

    Used by both batch and streaming modes.

    Args:
        df: DataFrame with label column
        label_field: Name of label column
        log_func: Logging function

    Returns:
        DataFrame with processed labels
    """
    if not pd.api.types.is_numeric_dtype(df[label_field]):
        unique_labels = sorted(df[label_field].dropna().unique())
        label_map = {val: idx for idx, val in enumerate(unique_labels)}
        df[label_field] = df[label_field].map(label_map)

    df[label_field] = pd.to_numeric(df[label_field], errors="coerce").astype("Int64")
    df.dropna(subset=[label_field], inplace=True)
    df[label_field] = df[label_field].astype(int)

    log_func(f"[INFO] Processed labels, shape after cleaning: {df.shape}")
    return df


def _is_gzipped(path: str) -> bool:
    """Check if file is gzipped."""
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


# ============================================================================
# NAMES3RISK CUSTOM PREPROCESSING (Used by both modes)
# ============================================================================


def detect_and_apply_names3risk_preprocessing(
    df: pd.DataFrame, log_func: Callable
) -> pd.DataFrame:
    """
    Auto-detect and apply Names3Risk-specific preprocessing.

    Applies transformations matching legacy train.py behavior (EXACT ORDER):
    1. Amazon email filtering: Remove amazon.com emails (FIRST - reduces volume)
    2. Create label from status field (F/I→1, N→0, filter invalid)
    3. Text concatenation: email|billing|customer|payment → 'text' field
    4. Sort by orderDate (before deduplication for consistency)
    5. Customer deduplication: Keep first occurrence by customerId

    Zero configuration required - smart defaults based on data shape.

    Args:
        df: Input DataFrame
        log_func: Logging function (e.g., print or logger.info)

    Returns:
        Preprocessed DataFrame
    """
    # 1. Amazon email filtering (FIRST - reduces data volume early, matches legacy)
    if "emailDomain" in df.columns:
        initial_count = len(df)
        df = df[~df["emailDomain"].str.lower().str.contains("amazon.", na=False)]
        filtered = initial_count - len(df)
        if filtered > 0:
            log_func(
                f"[INFO] Filtered {filtered} amazon.com emails ({filtered / initial_count * 100:.2f}%)"
            )

    # 2. Create label from status field (if exists and no label field)
    if "status" in df.columns and "label" not in df.columns:
        log_func("[INFO] Creating label from status field (F/I→1, N→0)...")
        initial_count = len(df)

        # Map status to label: F/I (fraud/investigation) → 1, N (normal) → 0, others → NaN
        status_map = {"F": 1, "I": 1, "N": 0}
        df["label"] = df["status"].map(status_map)

        # Filter out records with invalid status (label is NaN)
        df = df[df["label"].notna()]
        filtered = initial_count - len(df)

        if filtered > 0:
            log_func(
                f"[INFO] Filtered {filtered} records with invalid status ({filtered / initial_count * 100:.2f}%)"
            )

        log_func(f"[INFO] Label distribution: {df['label'].value_counts().to_dict()}")

    # 3. Text concatenation (if all 4 fields exist)
    text_fields = [
        "emailAddress",
        "billingAddressName",
        "customerName",
        "paymentAccountHolderName",
    ]

    if all(field in df.columns for field in text_fields):
        log_func("[INFO] Detected Names3Risk text fields, concatenating...")
        df["text"] = df[text_fields].fillna("[MISSING]").agg("|".join, axis=1)
        log_func(f"[INFO] Created 'text' field from {len(text_fields)} columns")

    # 4. Sort by orderDate (BEFORE deduplication to ensure "first" is chronologically first)
    if "orderDate" in df.columns:
        log_func("[INFO] Sorting by orderDate for temporal consistency...")
        df = df.sort_values("orderDate").reset_index(drop=True)

    # 5. Deduplication by customerId (if exists)
    if "customerId" in df.columns:
        initial_count = len(df)
        df = df.drop_duplicates(subset=["customerId"], keep="first")
        removed = initial_count - len(df)
        if removed > 0:
            log_func(
                f"[INFO] Removed {removed} duplicate customerIds ({removed / initial_count * 100:.2f}%)"
            )

    return df


def filter_to_numeric_features(df: pd.DataFrame, log_func: Callable) -> pd.DataFrame:
    """
    Filter DataFrame to numeric features only (Names3Risk requirement).

    Preserves: numeric columns + 'text' field + 'label' field

    Args:
        df: Input DataFrame
        log_func: Logging function

    Returns:
        Filtered DataFrame with only numeric features and preserved columns
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Always preserve these critical columns if they exist
    preserve_cols = []
    if "text" in df.columns:
        preserve_cols.append("text")
    if "label" in df.columns:
        preserve_cols.append("label")

    # Combine numeric columns with preserved columns
    keep_cols = list(set(numeric_cols + preserve_cols))

    if len(keep_cols) < len(df.columns):
        dropped_cols = set(df.columns) - set(keep_cols)
        log_func(
            f"[INFO] Filtering to numeric features only. Dropped {len(dropped_cols)} non-numeric columns"
        )
        log_func(
            f"[INFO] Kept {len(keep_cols)} columns: {len(numeric_cols)} numeric + {len(preserve_cols)} preserved"
        )
        df = df[keep_cols]

    return df


def get_temporal_column(df: pd.DataFrame) -> Optional[str]:
    """
    Identify temporal column for time-based splitting.

    Args:
        df: Input DataFrame

    Returns:
        Name of temporal column if found, None otherwise
    """
    if "transactionDate" in df.columns:
        return "transactionDate"
    elif "orderDate" in df.columns:
        return "orderDate"
    return None


# ============================================================================
# BATCH MODE FUNCTIONS
# ============================================================================


def _combine_shards_streaming(
    shard_args: list,
    max_workers: int,
    concat_batch_size: int,
    streaming_batch_size: int,
) -> pd.DataFrame:
    """
    Combine shards using streaming batch processing for memory efficiency.

    NOTE: Despite the name, this is NOT true streaming - it accumulates
    the full DataFrame by the end. Use for batch mode only.

    Instead of loading all shards into memory, processes them in batches,
    concatenating incrementally and freeing memory between batches.

    Memory usage: streaming_batch_size × avg_shard_size (much lower than loading all)

    Args:
        shard_args: List of shard arguments for _read_shard_wrapper
        max_workers: Number of parallel workers
        concat_batch_size: Batch size for DataFrame concatenation
        streaming_batch_size: Number of shards to process per streaming batch

    Returns:
        Combined DataFrame from all shards
    """
    total_shards = len(shard_args)
    result_df = None
    total_rows = 0

    # Process shards in streaming batches
    for batch_start in range(0, total_shards, streaming_batch_size):
        batch_end = min(batch_start + streaming_batch_size, total_shards)
        batch_args = shard_args[batch_start:batch_end]
        batch_num = (batch_start // streaming_batch_size) + 1
        total_batches = (
            total_shards + streaming_batch_size - 1
        ) // streaming_batch_size

        print(
            f"[INFO] Processing streaming batch {batch_num}/{total_batches} ({len(batch_args)} shards)"
        )

        # Read current batch of shards
        if max_workers > 1 and len(batch_args) > 1:
            with Pool(processes=max_workers) as pool:
                batch_dfs = pool.map(_read_shard_wrapper, batch_args)
        else:
            batch_dfs = [_read_shard_wrapper(args) for args in batch_args]

        # Concatenate batch
        batch_result = _batch_concat_dataframes(batch_dfs, concat_batch_size)
        batch_rows = batch_result.shape[0]
        total_rows += batch_rows

        print(f"[INFO] Batch {batch_num} combined: {batch_rows} rows")

        # Incrementally concatenate with result
        if result_df is None:
            result_df = batch_result
        else:
            result_df = pd.concat([result_df, batch_result], axis=0, ignore_index=True)

        # Free memory
        del batch_dfs, batch_result
        gc.collect()

    print(
        f"[INFO] Streaming complete: {total_rows} total rows from {total_shards} shards"
    )
    return result_df


def combine_shards(
    input_dir: str,
    signature_columns: Optional[list] = None,
    max_workers: Optional[int] = None,
    batch_size: int = 10,
    streaming_batch_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Detect and combine all supported data shards in a directory using parallel processing.

    Used by BATCH MODE only.

    Uses parallel shard reading and batch concatenation for improved performance.
    Memory-efficient approach avoids PyArrow's 2GB column limit error.

    Streaming Mode:
    When streaming_batch_size is set, processes shards in batches to avoid loading
    all DataFrames into memory simultaneously. This is the most memory-efficient mode.

    Args:
        input_dir: Directory containing data shards
        signature_columns: Optional column names for CSV/TSV files
        max_workers: Maximum number of parallel workers (default: cpu_count)
        batch_size: Number of DataFrames to concatenate at once (default: 10)
        streaming_batch_size: Number of shards to process per batch (enables streaming mode)
            - If None: Loads all shards into memory (original behavior)
            - If set: Processes shards in batches, concatenating incrementally
            - Recommended: 10-20 shards per batch for memory-constrained environments

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

        # STREAMING MODE: Process shards in batches to avoid loading all into memory
        if streaming_batch_size is not None and streaming_batch_size > 0:
            print(
                f"[INFO] Streaming mode enabled: processing {streaming_batch_size} shards per batch"
            )
            result_df = _combine_shards_streaming(
                shard_args, max_workers, batch_size, streaming_batch_size
            )
            print(f"[INFO] Final combined shape: {result_df.shape}")
            return result_df

        # ORIGINAL MODE: Load all shards then concatenate
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

        # Clear intermediate DataFrames to free memory
        del dataframes
        gc.collect()

        # Verify final shape
        print(f"[INFO] Final combined shape: {result_df.shape}")

        return result_df

    except Exception as e:
        raise RuntimeError(f"Failed to read or concatenate shards: {e}")


def process_batch_mode_preprocessing(
    input_data_dir: str,
    input_signature_dir: str,
    output_dir: str,
    signature_columns: Optional[list],
    job_type: str,
    label_field: Optional[str],
    train_ratio: float,
    test_val_ratio: float,
    output_format: str,
    max_workers: Optional[int],
    batch_size: int,
    streaming_batch_size: Optional[int],
    optimize_memory: bool,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Batch mode for Names3Risk tabular preprocessing.

    Loads full DataFrame into memory, applies Names3Risk-specific transformations,
    and splits data using TIME-BASED splits (not stratified).

    Args:
        input_data_dir: Directory containing input shards
        input_signature_dir: Directory containing signature file
        output_dir: Base output directory
        signature_columns: Optional column names from signature file
        job_type: "training", "validation", "testing", or "calibration"
        label_field: Name of label column (optional)
        train_ratio: Training set ratio (for training jobs)
        test_val_ratio: Test/val split ratio (for training jobs)
        output_format: "csv", "tsv", or "parquet"
        max_workers: Max parallel workers
        batch_size: Batch size for concatenation
        streaming_batch_size: Optional incremental loading batch size
        optimize_memory: Whether to optimize dtypes
        logger: Optional logging function

    Returns:
        Dictionary of DataFrames by split name
    """
    log = logger or print
    output_path = Path(output_dir)

    # Combine data shards
    log(f"[BATCH] Combining data shards from {input_data_dir}…")
    df = combine_shards(
        input_data_dir, signature_columns, max_workers, batch_size, streaming_batch_size
    )
    log(f"[BATCH] Combined data shape: {df.shape}")

    # Apply memory optimization if enabled
    if optimize_memory:
        df = optimize_dtypes(df, log)

    # Process columns
    df.columns = [col.replace("__DOT__", ".") for col in df.columns]

    # Apply Names3Risk-specific preprocessing
    df = detect_and_apply_names3risk_preprocessing(df, log)

    # Filter to numeric features only
    df = filter_to_numeric_features(df, log)

    # Process labels if provided
    if label_field:
        if label_field not in df.columns:
            raise RuntimeError(
                f"Label field '{label_field}' not found in columns: {df.columns.tolist()}"
            )
        df = process_label_column(df, label_field, log)
    else:
        log("[BATCH] No label field provided, skipping label processing")

    # Split data
    if job_type == "training":
        # Time-based split (Names3Risk requirement)
        temporal_col = get_temporal_column(df)

        if temporal_col:
            log(
                f"[BATCH] Using time-based split sorted by '{temporal_col}' (shuffle=False)"
            )
            df = df.sort_values(temporal_col).reset_index(drop=True)
        else:
            log("[BATCH WARNING] No temporal column found, using sequential split")

        # Time-based split with shuffle=False (oldest data first)
        train_df, holdout_df = train_test_split(
            df, train_size=train_ratio, shuffle=False, random_state=42
        )
        test_df, val_df = train_test_split(
            holdout_df, test_size=test_val_ratio, shuffle=False, random_state=42
        )

        log(
            f"[BATCH] Time-based split: train={len(train_df)} ({len(train_df) / len(df) * 100:.1f}%), "
            f"val={len(val_df)} ({len(val_df) / len(df) * 100:.1f}%), "
            f"test={len(test_df)} ({len(test_df) / len(df) * 100:.1f}%)"
        )

        splits = {"train": train_df, "test": test_df, "val": val_df}
    else:
        splits = {job_type: df}

    # Save output files
    for split_name, split_df in splits.items():
        subfolder = output_path / split_name
        subfolder.mkdir(exist_ok=True, parents=True)

        # Write based on output format
        if output_format == "csv":
            proc_path = subfolder / f"{split_name}_processed_data.csv"
            split_df.to_csv(proc_path, index=False)
        elif output_format == "tsv":
            proc_path = subfolder / f"{split_name}_processed_data.tsv"
            split_df.to_csv(proc_path, sep="\t", index=False)
        elif output_format == "parquet":
            proc_path = subfolder / f"{split_name}_processed_data.parquet"
            split_df.to_parquet(proc_path, index=False)

        log(
            f"[BATCH] Saved {proc_path} (format={output_format}, shape={split_df.shape})"
        )

    log("[BATCH] Preprocessing complete in batch mode")
    return splits


# ============================================================================
# STREAMING MODE FUNCTIONS
# ============================================================================


def write_single_shard(
    df: pd.DataFrame,
    output_dir: Path,
    shard_number: int,
    output_format: str = "csv",
) -> Path:
    """
    Write a single data shard in the specified format.

    Used by STREAMING MODE to write temporary shards.

    Args:
        df: DataFrame to write
        output_dir: Output directory
        shard_number: Shard index number
        output_format: "csv", "tsv", or "parquet"

    Returns:
        Path to the written shard file
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


def assign_temporal_splits(
    df: pd.DataFrame,
    train_ratio: float,
    test_val_ratio: float,
    temporal_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Assign rows to train/test/val splits based on temporal ordering.

    Uses cumulative position to assign splits, respecting temporal order.
    For Names3Risk: oldest data → train, newest data → test.

    Args:
        df: Input DataFrame (should already be temporally sorted)
        train_ratio: Proportion for training (e.g., 0.7)
        test_val_ratio: Proportion of non-train for test vs val (e.g., 0.5)
        temporal_col: Optional temporal column name for sorting

    Returns:
        DataFrame with added '_split' column
    """
    # Sort by temporal column if provided
    if temporal_col and temporal_col in df.columns:
        df = df.sort_values(temporal_col).reset_index(drop=True)

    # Calculate split boundaries based on cumulative position
    total_rows = len(df)
    train_cutoff = int(total_rows * train_ratio)
    val_cutoff = train_cutoff + int((total_rows - train_cutoff) * test_val_ratio)

    # Assign splits based on position
    def temporal_split(idx):
        if idx < train_cutoff:
            return "train"
        elif idx < val_cutoff:
            return "test"
        else:
            return "val"

    df["_split"] = df.index.map(temporal_split)
    return df


def write_splits_to_shards(
    df: pd.DataFrame,
    output_base: Path,
    split_counters: Dict[str, int],
    shard_size: int,
    output_format: str,
    log_func: Callable,
) -> None:
    """
    Write DataFrame to separate split directories based on '_split' column.

    Args:
        df: DataFrame with '_split' column
        output_base: Base output directory
        split_counters: Dictionary tracking shard numbers per split (modified in place)
        shard_size: Rows per shard
        output_format: "csv", "tsv", or "parquet"
        log_func: Logging function
    """
    for split_name in ["train", "test", "val"]:
        split_data = df[df["_split"] == split_name].drop("_split", axis=1)

        if len(split_data) == 0:
            continue

        split_dir = output_base / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        # Write in shards
        for i in range(0, len(split_data), shard_size):
            shard_df = split_data.iloc[i : i + shard_size]
            write_single_shard(
                shard_df, split_dir, split_counters[split_name], output_format
            )
            split_counters[split_name] += 1


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


def process_single_batch(
    shard_files: List[Path],
    signature_columns: Optional[list],
    batch_size: int,
    optimize_memory: bool,
    label_field: Optional[str],
    log_func: Callable,
) -> pd.DataFrame:
    """
    Process a single batch of shards with Names3Risk preprocessing.

    Args:
        shard_files: List of shard file paths
        signature_columns: Optional column names
        batch_size: Batch size for concatenation
        optimize_memory: Whether to optimize dtypes
        label_field: Name of label column (optional)
        log_func: Logging function

    Returns:
        Processed DataFrame from batch
    """
    # Read batch
    batch_dfs = []
    for shard in shard_files:
        df = _read_file_to_df(shard, signature_columns)
        batch_dfs.append(df)

    batch_df = _batch_concat_dataframes(batch_dfs, batch_size)
    del batch_dfs
    gc.collect()

    # Apply memory optimization if enabled
    if optimize_memory:
        batch_df = optimize_dtypes(batch_df, log_func)

    # Process columns
    batch_df.columns = [col.replace("__DOT__", ".") for col in batch_df.columns]

    # Apply Names3Risk-specific preprocessing
    batch_df = detect_and_apply_names3risk_preprocessing(batch_df, log_func)

    # Filter to numeric features only
    batch_df = filter_to_numeric_features(batch_df, log_func)

    # Process labels if provided
    if label_field:
        if label_field not in batch_df.columns:
            raise RuntimeError(f"Label field '{label_field}' not found in columns")
        batch_df = process_label_column(batch_df, label_field, log_func)

    return batch_df


def process_training_splits_streaming(
    all_shards: List[Path],
    output_path: Path,
    signature_columns: Optional[list],
    label_field: Optional[str],
    train_ratio: float,
    test_val_ratio: float,
    output_format: str,
    streaming_batch_size: int,
    shard_size: int,
    batch_size: int,
    optimize_memory: bool,
    log_func: Callable,
) -> None:
    """
    Process training data with time-based splits in streaming mode.

    Uses temporal ordering instead of random splits to match batch mode behavior.

    Args:
        all_shards: List of all input shard paths
        output_path: Base output directory
        signature_columns: Optional column names
        label_field: Name of label column (optional)
        train_ratio: Training set ratio
        test_val_ratio: Test/val split ratio
        output_format: Output format ("csv", "tsv", "parquet")
        streaming_batch_size: Number of shards per batch
        shard_size: Rows per output shard
        batch_size: Batch size for concatenation
        optimize_memory: Whether to optimize dtypes
        log_func: Logging function
    """
    log_func(
        "[STREAMING] Training mode: Using time-based split for Names3Risk compatibility"
    )

    split_counters = {"train": 0, "test": 0, "val": 0}

    # Detect temporal column from first shard
    first_shard_df = _read_file_to_df(all_shards[0], signature_columns)
    first_shard_df.columns = [
        col.replace("__DOT__", ".") for col in first_shard_df.columns
    ]
    temporal_col = get_temporal_column(first_shard_df)
    del first_shard_df
    gc.collect()

    if temporal_col:
        log_func(
            f"[STREAMING] Using temporal column '{temporal_col}' for time-based splits"
        )
    else:
        log_func(
            "[STREAMING WARNING] No temporal column found, using sequential position"
        )

    # Process all batches
    for batch_start in range(0, len(all_shards), streaming_batch_size):
        batch_end = min(batch_start + streaming_batch_size, len(all_shards))
        batch_shards = all_shards[batch_start:batch_end]
        batch_num = (batch_start // streaming_batch_size) + 1

        log_func(
            f"[STREAMING] Processing batch {batch_num} ({len(batch_shards)} shards)"
        )

        # Process batch
        batch_df = process_single_batch(
            batch_shards,
            signature_columns,
            batch_size,
            optimize_memory,
            label_field,
            log_func,
        )

        # Assign to splits and write
        batch_df = assign_temporal_splits(
            batch_df, train_ratio, test_val_ratio, temporal_col
        )
        write_splits_to_shards(
            batch_df, output_path, split_counters, shard_size, output_format, log_func
        )

        del batch_df
        gc.collect()

    log_func(
        f"[STREAMING] Complete: train={split_counters['train']}, "
        f"test={split_counters['test']}, val={split_counters['val']} shards"
    )


def process_single_split_streaming(
    all_shards: List[Path],
    output_path: Path,
    job_type: str,
    signature_columns: Optional[list],
    label_field: Optional[str],
    output_format: str,
    streaming_batch_size: int,
    shard_size: int,
    batch_size: int,
    optimize_memory: bool,
    log_func: Callable,
) -> None:
    """Process non-training data as single split in streaming mode."""
    log_func(f"[STREAMING] {job_type.capitalize()} mode: Single split")

    split_counter = 0
    split_dir = output_path / job_type
    split_dir.mkdir(parents=True, exist_ok=True)

    # Process all batches
    for batch_start in range(0, len(all_shards), streaming_batch_size):
        batch_end = min(batch_start + streaming_batch_size, len(all_shards))
        batch_shards = all_shards[batch_start:batch_end]
        batch_num = (batch_start // streaming_batch_size) + 1

        log_func(
            f"[STREAMING] Processing batch {batch_num} ({len(batch_shards)} shards)"
        )

        # Process batch
        batch_df = process_single_batch(
            batch_shards,
            signature_columns,
            batch_size,
            optimize_memory,
            label_field,
            log_func,
        )

        # Write shards
        for i in range(0, len(batch_df), shard_size):
            shard_df = batch_df.iloc[i : i + shard_size]
            write_single_shard(shard_df, split_dir, split_counter, output_format)
            split_counter += 1

        del batch_df
        gc.collect()

    log_func(f"[STREAMING] Complete: {split_counter} shards written to {job_type}")


def consolidate_shards_to_single_files(
    output_path: Path, job_type: str, output_format: str, log_func: Callable
) -> None:
    """Consolidate temporary shards into single files per split."""
    log_func("[STREAMING] Consolidating shards into single files per split...")

    if job_type == "training":
        # Consolidate train/test/val splits
        for split_name in ["train", "test", "val"]:
            _consolidate_single_split(
                output_path / split_name, split_name, output_format, log_func
            )
    else:
        # Consolidate single split
        _consolidate_single_split(
            output_path / job_type, job_type, output_format, log_func
        )

    log_func("[STREAMING] Output format now matches batch mode")


def _consolidate_single_split(
    split_dir: Path, split_name: str, output_format: str, log_func: Callable
) -> None:
    """Consolidate shards for a single split."""
    if not split_dir.exists():
        return

    # Find all shards for this split
    shard_files = sorted(split_dir.glob(f"part-*.{output_format}"))
    if not shard_files:
        return

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

    # Write consolidated file
    if output_format == "csv":
        output_file = split_dir / f"{split_name}_processed_data.csv"
        consolidated_df.to_csv(output_file, index=False)
    elif output_format == "tsv":
        output_file = split_dir / f"{split_name}_processed_data.tsv"
        consolidated_df.to_csv(output_file, sep="\t", index=False)
    elif output_format == "parquet":
        output_file = split_dir / f"{split_name}_processed_data.parquet"
        consolidated_df.to_parquet(output_file, index=False)

    log_func(f"[STREAMING] Wrote {output_file} (shape={consolidated_df.shape})")
    del consolidated_df
    gc.collect()

    # Delete shard files
    for shard_file in shard_files:
        shard_file.unlink()
    log_func(f"[STREAMING] Cleaned up {len(shard_files)} temporary shards")


def process_streaming_mode_preprocessing(
    input_dir: str,
    output_dir: str,
    signature_columns: Optional[list],
    job_type: str,
    label_field: Optional[str],
    train_ratio: float,
    test_val_ratio: float,
    output_format: str,
    streaming_batch_size: int,
    shard_size: int,
    max_workers: Optional[int],
    batch_size: int,
    optimize_memory: bool,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    True streaming mode for Names3Risk tabular preprocessing.

    Processes data in batches, never loading the full DataFrame into memory.
    For training jobs, uses TIME-BASED split (not random) to match batch mode.
    For non-training jobs, processes as a single split.

    Memory usage: Fixed at ~2GB per batch regardless of total data size.

    Args:
        input_dir: Directory containing input shards
        output_dir: Base output directory
        signature_columns: Optional column names from signature file
        job_type: "training", "validation", "testing", or "calibration"
        label_field: Name of label column (optional)
        train_ratio: Training set ratio (for training jobs)
        test_val_ratio: Test/val split ratio (for training jobs)
        output_format: "csv", "tsv", or "parquet"
        streaming_batch_size: Number of shards per batch
        shard_size: Rows per output shard
        max_workers: Max parallel workers
        batch_size: Batch size for concatenation
        optimize_memory: Whether to optimize dtypes
        logger: Optional logging function

    Returns:
        Empty dictionary (data was written incrementally)
    """
    log = logger or print

    log("[STREAMING] Starting true streaming mode preprocessing for Names3Risk")
    log(f"[STREAMING] Job type: {job_type}")
    log(f"[STREAMING] Streaming batch size: {streaming_batch_size}")

    # Setup
    output_path = Path(output_dir)
    random.seed(42)

    # Find input shards
    all_shards = find_input_shards(input_dir, log)

    # Process data based on job type
    if job_type == "training":
        process_training_splits_streaming(
            all_shards,
            output_path,
            signature_columns,
            label_field,
            train_ratio,
            test_val_ratio,
            output_format,
            streaming_batch_size,
            shard_size,
            batch_size,
            optimize_memory,
            log,
        )
    else:
        process_single_split_streaming(
            all_shards,
            output_path,
            job_type,
            signature_columns,
            label_field,
            output_format,
            streaming_batch_size,
            shard_size,
            batch_size,
            optimize_memory,
            log,
        )

    # Consolidate temporary shards into single files
    consolidate_shards_to_single_files(output_path, job_type, output_format, log)

    log("[STREAMING] Preprocessing complete in streaming mode")
    return {}


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
    Main logic for Names3Risk preprocessing data, refactored for testability.

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

    # Memory optimization parameters
    max_workers = int(environ_vars.get("MAX_WORKERS", 0)) or None  # 0 means auto
    batch_size = int(environ_vars.get("BATCH_SIZE", 5))  # Reduced from 10 to 5
    optimize_memory = environ_vars.get("OPTIMIZE_MEMORY", "true").lower() == "true"
    streaming_batch_size = (
        int(environ_vars.get("STREAMING_BATCH_SIZE", 0)) or None
    )  # 0 means disabled

    # Streaming mode parameters
    enable_true_streaming = (
        environ_vars.get("ENABLE_TRUE_STREAMING", "false").lower() == "true"
    )
    shard_size = int(environ_vars.get("SHARD_SIZE", 100000))

    # Extract paths
    input_data_dir = input_paths["DATA"]
    input_signature_dir = input_paths["SIGNATURE"]
    output_dir = output_paths["processed_data"]
    # Use print function if no logger is provided
    log = logger or print

    # Log memory optimization settings
    log("[INFO] Memory optimization settings:")
    log(f"  MAX_WORKERS: {max_workers if max_workers else 'auto'}")
    log(f"  BATCH_SIZE: {batch_size}")
    log(f"  OPTIMIZE_MEMORY: {optimize_memory}")
    log(
        f"  STREAMING_BATCH_SIZE: {streaming_batch_size if streaming_batch_size else 'disabled'}"
    )
    log(f"  ENABLE_TRUE_STREAMING: {enable_true_streaming}")

    # 1. Setup paths
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 2. Load signature columns if available
    signature_columns = load_signature_columns(input_signature_dir)
    if signature_columns:
        log(f"[INFO] Loaded signature with {len(signature_columns)} columns")
    else:
        log("[INFO] No signature file found, using default column handling")

    # 3. Get output format
    output_format = environ_vars.get("OUTPUT_FORMAT", "CSV").lower()
    if output_format not in ["csv", "tsv", "parquet"]:
        log(f"[WARNING] Invalid OUTPUT_FORMAT '{output_format}', defaulting to CSV")
        output_format = "csv"

    # 4. ROUTING: Choose between batch mode and streaming mode
    if enable_true_streaming:
        log("[INFO] Using TRUE STREAMING MODE (never loads full DataFrame)")
        return process_streaming_mode_preprocessing(
            input_dir=input_data_dir,
            output_dir=output_dir,
            signature_columns=signature_columns,
            job_type=job_type,
            label_field=label_field,
            train_ratio=train_ratio,
            test_val_ratio=test_val_ratio,
            output_format=output_format,
            streaming_batch_size=streaming_batch_size or 10,
            shard_size=shard_size,
            max_workers=max_workers,
            batch_size=batch_size,
            optimize_memory=optimize_memory,
            logger=log,
        )
    else:
        log("[INFO] Using BATCH MODE (loads full DataFrame)")
        return process_batch_mode_preprocessing(
            input_data_dir=input_data_dir,
            input_signature_dir=input_signature_dir,
            output_dir=output_dir,
            signature_columns=signature_columns,
            job_type=job_type,
            label_field=label_field,
            train_ratio=train_ratio,
            test_val_ratio=test_val_ratio,
            output_format=output_format,
            max_workers=max_workers,
            batch_size=batch_size,
            streaming_batch_size=streaming_batch_size,
            optimize_memory=optimize_memory,
            logger=log,
        )


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
        # LABEL_FIELD is now optional for all job types
        # The script will skip label processing if not provided
        TRAIN_RATIO = float(os.environ.get("TRAIN_RATIO", 0.7))
        TEST_VAL_RATIO = float(os.environ.get("TEST_VAL_RATIO", 0.5))

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
        logger.info("Starting Names3Risk tabular preprocessing with parameters:")
        logger.info(f"  Job Type: {args.job_type}")
        logger.info(f"  Label Field: {LABEL_FIELD if LABEL_FIELD else 'Not specified'}")
        logger.info(f"  Train Ratio: {TRAIN_RATIO}")
        logger.info(f"  Test/Val Ratio: {TEST_VAL_RATIO}")
        logger.info(f"  Input Directory: {INPUT_DATA_DIR}")
        logger.info(f"  Input Signature Directory: {INPUT_SIGNATURE_DIR}")
        logger.info(f"  Output Directory: {OUTPUT_DIR}")

        # Set up path dictionaries
        input_paths = {"DATA": INPUT_DATA_DIR, "SIGNATURE": INPUT_SIGNATURE_DIR}

        output_paths = {"processed_data": OUTPUT_DIR}

        # Environment variables dictionary
        environ_vars = {
            "LABEL_FIELD": LABEL_FIELD,
            "TRAIN_RATIO": str(TRAIN_RATIO),
            "TEST_VAL_RATIO": str(TEST_VAL_RATIO),
            "OUTPUT_FORMAT": os.environ.get("OUTPUT_FORMAT", "CSV"),
            "MAX_WORKERS": os.environ.get("MAX_WORKERS", "0"),
            "BATCH_SIZE": os.environ.get("BATCH_SIZE", "5"),
            "OPTIMIZE_MEMORY": os.environ.get("OPTIMIZE_MEMORY", "true"),
            "STREAMING_BATCH_SIZE": os.environ.get("STREAMING_BATCH_SIZE", "0"),
            "ENABLE_TRUE_STREAMING": os.environ.get("ENABLE_TRUE_STREAMING", "false"),
            "SHARD_SIZE": os.environ.get("SHARD_SIZE", "100000"),
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

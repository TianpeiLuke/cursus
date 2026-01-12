#!/usr/bin/env python
"""
Dummy Data Loading Processing Script

This script processes user-provided data instead of calling internal Cradle services.
It serves as a drop-in replacement for CradleDataLoadingStep by reading data from
an input channel, generating schema signatures and metadata, and outputting the
processed data in the same format as the original Cradle data loading step.
"""

import argparse
import csv
import json
import logging
import os
import shutil
import sys
import traceback
import gc
from pathlib import Path
from typing import Dict, Optional, List, Any, Union, Callable
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
import boto3
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Standard SageMaker paths
INPUT_DATA_DIR = "/opt/ml/processing/input/data"
SIGNATURE_OUTPUT_DIR = "/opt/ml/processing/output/signature"
METADATA_OUTPUT_DIR = "/opt/ml/processing/output/metadata"
DATA_OUTPUT_DIR = "/opt/ml/processing/output/data"


# ============================================================================
# SHARED UTILITY FUNCTIONS (Used by both Batch and Streaming modes)
# ============================================================================


def ensure_directory(directory: Path) -> bool:
    """Ensure a directory exists, creating it if necessary."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory ensured: {directory}")
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {directory}: {str(e)}", exc_info=True)
        return False


# --- Memory Optimization Functions ---


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


def _read_file_wrapper(args: tuple) -> pd.DataFrame:
    """
    Wrapper function for parallel file reading.

    Args:
        args: Tuple of (file_path, file_index, total_files)

    Returns:
        DataFrame from the file
    """
    file_path, idx, total = args
    try:
        file_format = detect_file_format(file_path)
        if file_format == "unknown":
            raise ValueError(f"Unknown file format for {file_path}")

        df = read_data_file(file_path, file_format)
        # Log progress
        logger.info(
            f"[INFO] Processed file {idx + 1}/{total}: {file_path.name} ({df.shape[0]} rows)"
        )
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to read file {file_path.name}: {e}")


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


def _combine_files_streaming(
    file_args: list,
    max_workers: int,
    concat_batch_size: int,
    streaming_batch_size: int,
) -> pd.DataFrame:
    """
    Combine files using streaming batch processing for memory efficiency.

    Instead of loading all files into memory, processes them in batches,
    concatenating incrementally and freeing memory between batches.

    Memory usage: streaming_batch_size × avg_file_size (much lower than loading all)

    Args:
        file_args: List of file arguments for _read_file_wrapper
        max_workers: Number of parallel workers
        concat_batch_size: Batch size for DataFrame concatenation
        streaming_batch_size: Number of files to process per streaming batch

    Returns:
        Combined DataFrame from all files
    """
    total_files = len(file_args)
    result_df = None
    total_rows = 0

    # Process files in streaming batches
    for batch_start in range(0, total_files, streaming_batch_size):
        batch_end = min(batch_start + streaming_batch_size, total_files)
        batch_args = file_args[batch_start:batch_end]
        batch_num = (batch_start // streaming_batch_size) + 1
        total_batches = (total_files + streaming_batch_size - 1) // streaming_batch_size

        logger.info(
            f"[INFO] Processing streaming batch {batch_num}/{total_batches} ({len(batch_args)} files)"
        )

        # Read current batch of files
        if max_workers > 1 and len(batch_args) > 1:
            with Pool(processes=max_workers) as pool:
                batch_dfs = pool.map(_read_file_wrapper, batch_args)
        else:
            batch_dfs = [_read_file_wrapper(args) for args in batch_args]

        # Concatenate batch
        batch_result = _batch_concat_dataframes(batch_dfs, concat_batch_size)
        batch_rows = batch_result.shape[0]
        total_rows += batch_rows

        logger.info(f"[INFO] Batch {batch_num} combined: {batch_rows} rows")

        # Incrementally concatenate with result
        if result_df is None:
            result_df = batch_result
        else:
            result_df = pd.concat([result_df, batch_result], axis=0, ignore_index=True)

        # Free memory
        del batch_dfs, batch_result
        gc.collect()

    logger.info(
        f"[INFO] Streaming complete: {total_rows} total rows from {total_files} files"
    )
    return result_df


def detect_file_format(file_path: Path) -> str:
    """
    Detect the format of a data file based on extension and content.

    Args:
        file_path: Path to the data file

    Returns:
        String indicating the format: 'csv', 'parquet', 'json', or 'unknown'
    """
    logger.info(f"Detecting format for file: {file_path}")

    # Check file extension first
    suffix = file_path.suffix.lower()
    if suffix in [".csv"]:
        return "csv"
    elif suffix in [".parquet", ".pq"]:
        return "parquet"
    elif suffix in [".json", ".jsonl"]:
        return "json"

    # If extension is unclear, try to read the file
    try:
        # Try CSV first
        pd.read_csv(file_path, nrows=1)
        return "csv"
    except:
        pass

    try:
        # Try Parquet
        pd.read_parquet(file_path)
        return "parquet"
    except:
        pass

    try:
        # Try JSON
        pd.read_json(file_path, lines=True, nrows=1)
        return "json"
    except:
        pass

    logger.warning(f"Could not detect format for file: {file_path}")
    return "unknown"


def read_data_file(file_path: Path, file_format: str) -> pd.DataFrame:
    """
    Read a data file based on its format.

    Args:
        file_path: Path to the data file
        file_format: Format of the file ('csv', 'parquet', 'json')

    Returns:
        DataFrame containing the data

    Raises:
        ValueError: If the format is unsupported
        Exception: If reading fails
    """
    logger.info(f"Reading {file_format} file: {file_path}")

    try:
        if file_format == "csv":
            df = pd.read_csv(file_path)
        elif file_format == "parquet":
            df = pd.read_parquet(file_path)
        elif file_format == "json":
            df = pd.read_json(file_path, lines=True)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        logger.info(f"Successfully read {len(df)} rows and {len(df.columns)} columns")
        return df

    except Exception as e:
        logger.error(f"Error reading {file_format} file {file_path}: {str(e)}")
        raise


def generate_schema_signature(df: pd.DataFrame) -> List[str]:
    """
    Generate a schema signature from a DataFrame.

    The schema signature is just a list of column names from the input data.

    Args:
        df: DataFrame to analyze

    Returns:
        List of column names
    """
    logger.info("Generating schema signature")

    # Simple signature - just the list of column names
    signature = list(df.columns)

    logger.info(f"Generated signature for {len(signature)} columns: {signature}")
    return signature


def generate_metadata(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate metadata information from a DataFrame.

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary containing metadata information
    """
    logger.info("Generating metadata")

    metadata = {
        "version": "1.0",
        "data_info": {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
        },
        "column_info": {},
    }

    for column in df.columns:
        col_info = {
            "data_type": str(df[column].dtype),
            "null_count": int(df[column].isnull().sum()),
            "memory_usage": int(df[column].memory_usage(deep=True)),
        }

        # Safe unique count - handle unhashable types (lists, dicts, etc.)
        try:
            col_info["unique_count"] = int(df[column].nunique())
        except TypeError:
            # Column contains unhashable types (lists, dicts from Parquet)
            logger.warning(
                f"Column '{column}' contains unhashable types, skipping unique count"
            )
            col_info["unique_count"] = None
            col_info["contains_complex_types"] = True

        # Add basic statistics for numeric columns
        if pd.api.types.is_numeric_dtype(df[column]):
            col_info.update(
                {
                    "min": float(df[column].min()) if not df[column].empty else None,
                    "max": float(df[column].max()) if not df[column].empty else None,
                    "mean": float(df[column].mean()) if not df[column].empty else None,
                    "std": float(df[column].std()) if not df[column].empty else None,
                }
            )

        metadata["column_info"][column] = col_info

    logger.info(f"Generated metadata for {len(metadata['column_info'])} columns")
    return metadata


def generate_mods_metadata(df: pd.DataFrame) -> List[List[str]]:
    """
    Generate MODS-compatible CSV metadata from a DataFrame.

    MODS metadata format is a simple CSV with 3 columns:
    - varname: Column name
    - iscategory: "true" if string/object/category type, "false" otherwise
    - datatype: pandas dtype as string

    This lightweight format can be generated from the first batch only,
    enabling true streaming mode without needing the full DataFrame.

    Args:
        df: DataFrame to analyze (typically first batch)

    Returns:
        List of lists representing CSV rows [header, row1, row2, ...]
    """
    logger.info("Generating MODS-compatible metadata")

    # Header row
    metadata = [["varname", "iscategory", "datatype"]]

    # Data rows - one per column
    for column in df.columns:
        dtype_str = str(df[column].dtype)

        # Determine if categorical based on dtype
        # String, object, and category types are considered categorical
        is_categorical = dtype_str in ["object", "string", "category"]
        is_category_str = "true" if is_categorical else "false"

        metadata.append([str(column), is_category_str, dtype_str])

    logger.info(f"Generated MODS metadata for {len(metadata) - 1} columns")
    return metadata


def find_data_files(input_dir: Path) -> List[Path]:
    """
    Find all data files in the input directory.

    Args:
        input_dir: Directory to search for data files

    Returns:
        List of paths to data files
    """
    logger.info(f"Searching for data files in: {input_dir}")

    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return []

    data_files = []
    supported_extensions = {".csv", ".parquet", ".pq", ".json", ".jsonl"}

    for file_path in input_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            data_files.append(file_path)
            logger.info(f"Found data file: {file_path}")

    logger.info(f"Found {len(data_files)} data files")
    return data_files


def combine_files(
    data_files: List[Path],
    max_workers: Optional[int] = None,
    batch_size: int = 10,
    streaming_batch_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Combine multiple data files using parallel processing and optional streaming.

    Uses parallel file reading and batch concatenation for improved performance.
    Memory-efficient approach with optional streaming mode.

    Streaming Mode:
    When streaming_batch_size is set, processes files in batches to avoid loading
    all DataFrames into memory simultaneously. This is the most memory-efficient mode.

    Args:
        data_files: List of data file paths
        max_workers: Maximum number of parallel workers (default: cpu_count)
        batch_size: Number of DataFrames to concatenate at once (default: 10)
        streaming_batch_size: Number of files to process per batch (enables streaming mode)
            - If None: Loads all files into memory (original behavior)
            - If set: Processes files in batches, concatenating incrementally
            - Recommended: 10-20 files per batch for memory-constrained environments

    Returns:
        Combined DataFrame from all files
    """
    if not data_files:
        raise ValueError("No data files found to process")

    total_files = len(data_files)
    logger.info(f"[INFO] Found {total_files} files to process")

    try:
        # Determine optimal number of workers
        if max_workers is None:
            max_workers = min(cpu_count(), total_files)

        logger.info(f"[INFO] Using {max_workers} parallel workers for file reading")

        # Prepare arguments for parallel processing
        file_args = [(file, i, total_files) for i, file in enumerate(data_files)]

        # STREAMING MODE: Process files in batches to avoid loading all into memory
        if streaming_batch_size is not None and streaming_batch_size > 0:
            logger.info(
                f"[INFO] Streaming mode enabled: processing {streaming_batch_size} files per batch"
            )
            result_df = _combine_files_streaming(
                file_args, max_workers, batch_size, streaming_batch_size
            )
            logger.info(f"[INFO] Final combined shape: {result_df.shape}")
            return result_df

        # ORIGINAL MODE: Load all files then concatenate
        # Read files in parallel
        if max_workers > 1 and total_files > 1:
            with Pool(processes=max_workers) as pool:
                dataframes = pool.map(_read_file_wrapper, file_args)
        else:
            # Fall back to sequential processing for single file or single worker
            logger.info(
                "[INFO] Using sequential processing (single worker or single file)"
            )
            dataframes = [_read_file_wrapper(args) for args in file_args]

        if not dataframes:
            raise RuntimeError("No data was loaded from any files")

        # Log total rows before concatenation
        total_rows = sum(df.shape[0] for df in dataframes)
        logger.info(f"[INFO] Loaded {total_rows} total rows from {total_files} files")

        # Concatenate using batch approach
        logger.info(f"[INFO] Concatenating DataFrames with batch_size={batch_size}")
        result_df = _batch_concat_dataframes(dataframes, batch_size)

        # Clear intermediate DataFrames to free memory
        del dataframes
        gc.collect()

        # Verify final shape
        logger.info(f"[INFO] Final combined shape: {result_df.shape}")

        return result_df

    except Exception as e:
        raise RuntimeError(f"Failed to read or concatenate files: {e}")


def process_data_files(data_files: List[Path]) -> pd.DataFrame:
    """
    DEPRECATED: Legacy function for backward compatibility.
    Use combine_files() instead for better performance and memory efficiency.

    Process multiple data files and combine them into a single DataFrame.

    Args:
        data_files: List of data file paths

    Returns:
        Combined DataFrame
    """
    logger.warning(
        "[WARNING] Using deprecated process_data_files(). Consider using combine_files() for better performance."
    )
    return combine_files(
        data_files, max_workers=1, batch_size=5, streaming_batch_size=None
    )


def write_signature_file(signature: List[str], output_dir: Path) -> Path:
    """
    Write the signature file to the output directory in CSV format.

    The signature file contains column names separated by commas, matching
    the format expected by tabular_preprocessing script.

    Args:
        signature: Schema signature list of column names
        output_dir: Output directory path

    Returns:
        Path to the written signature file
    """
    ensure_directory(output_dir)
    signature_file = output_dir / "signature"

    logger.info(f"Writing signature file: {signature_file}")

    try:
        # Write signature as comma-separated values (CSV format)
        with open(signature_file, "w") as f:
            f.write(",".join(signature))

        logger.info(
            f"Signature file written successfully with {len(signature)} columns"
        )
        return signature_file

    except Exception as e:
        logger.error(f"Error writing signature file: {str(e)}")
        raise


def write_metadata_file(
    metadata: Union[Dict[str, Any], List[List[str]]],
    output_dir: Path,
    format: str = "JSON",
) -> Path:
    """
    Write the metadata file to the output directory in specified format.

    Supports two formats:
    - JSON: Detailed metadata with statistics (requires Dict input)
    - MODS: Simple CSV with 3 columns (requires List[List[str]] input)

    Args:
        metadata: Metadata as Dict (JSON) or List[List[str]] (MODS CSV)
        output_dir: Output directory path
        format: Output format - "JSON" or "MODS" (default: "JSON")

    Returns:
        Path to the written metadata file

    Raises:
        ValueError: If format is unsupported or metadata type doesn't match format
    """
    ensure_directory(output_dir)
    metadata_file = output_dir / "metadata"

    logger.info(f"Writing metadata file in {format} format: {metadata_file}")

    try:
        if format == "MODS":
            # Write MODS CSV format
            if not isinstance(metadata, list):
                raise ValueError(
                    "MODS format requires metadata as List[List[str]], "
                    f"got {type(metadata)}"
                )

            import csv

            with open(metadata_file, "w", newline="") as f:
                writer = csv.writer(
                    f, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
                )
                writer.writerows(metadata)

            logger.info(f"MODS metadata file written with {len(metadata) - 1} columns")

        elif format == "JSON":
            # Write JSON format (original behavior)
            if not isinstance(metadata, dict):
                raise ValueError(
                    f"JSON format requires metadata as Dict, got {type(metadata)}"
                )

            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info("JSON metadata file written successfully")

        else:
            raise ValueError(
                f"Unsupported metadata format: {format}. Supported formats: JSON, MODS"
            )

        return metadata_file

    except Exception as e:
        logger.error(f"Error writing metadata file: {str(e)}")
        raise


def write_single_shard(
    df: pd.DataFrame, output_dir: Path, shard_index: int, output_format: str
) -> Path:
    """
    Write a single data shard in the specified format.

    Args:
        df: DataFrame to write
        output_dir: Output directory path
        shard_index: Index of the shard (for filename)
        output_format: Output format ('CSV', 'JSON', 'PARQUET')

    Returns:
        Path to the written shard file

    Raises:
        ValueError: If the format is unsupported
        Exception: If writing fails
    """
    # Map format to file extension
    format_extensions = {"CSV": "csv", "JSON": "json", "PARQUET": "parquet"}

    if output_format not in format_extensions:
        raise ValueError(
            f"Unsupported output format: {output_format}. "
            f"Supported formats: {list(format_extensions.keys())}"
        )

    extension = format_extensions[output_format]
    shard_filename = f"part-{shard_index:05d}.{extension}"
    shard_path = output_dir / shard_filename

    logger.info(f"Writing {output_format} shard: {shard_path}")

    try:
        if output_format == "CSV":
            df.to_csv(shard_path, index=False)
        elif output_format == "JSON":
            df.to_json(shard_path, orient="records", lines=True)
        elif output_format == "PARQUET":
            df.to_parquet(shard_path, index=False)

        logger.info(f"Successfully wrote {len(df)} rows to {shard_path}")
        return shard_path

    except Exception as e:
        logger.error(f"Error writing {output_format} shard {shard_path}: {str(e)}")
        raise


def write_data_shards(
    df: pd.DataFrame, output_dir: Path, shard_size: int, output_format: str
) -> List[Path]:
    """
    Write DataFrame as multiple data shards.

    Args:
        df: DataFrame to write
        output_dir: Output directory path
        shard_size: Number of rows per shard
        output_format: Output format ('CSV', 'JSON', 'PARQUET')

    Returns:
        List of paths to written shard files
    """
    ensure_directory(output_dir)

    written_files = []
    total_rows = len(df)

    logger.info(
        f"Writing {total_rows} rows as shards of size {shard_size} in {output_format} format"
    )

    if total_rows <= shard_size:
        # Single shard
        shard_file = write_single_shard(df, output_dir, 0, output_format)
        written_files.append(shard_file)
    else:
        # Multiple shards
        for i in range(0, total_rows, shard_size):
            shard_df = df.iloc[i : i + shard_size]
            shard_index = i // shard_size
            shard_file = write_single_shard(
                shard_df, output_dir, shard_index, output_format
            )
            written_files.append(shard_file)

    logger.info(f"Successfully wrote {len(written_files)} shard files")
    return written_files


def write_single_data_file(
    df: pd.DataFrame, output_dir: Path, output_format: str
) -> Path:
    """
    Write DataFrame as a single data file.

    Args:
        df: DataFrame to write
        output_dir: Output directory path
        output_format: Output format ('CSV', 'JSON', 'PARQUET')

    Returns:
        Path to the written data file

    Raises:
        ValueError: If the format is unsupported
        Exception: If writing fails
    """
    ensure_directory(output_dir)

    # Map format to file extension
    format_extensions = {"CSV": "csv", "JSON": "json", "PARQUET": "parquet"}

    if output_format not in format_extensions:
        raise ValueError(
            f"Unsupported output format: {output_format}. "
            f"Supported formats: {list(format_extensions.keys())}"
        )

    extension = format_extensions[output_format]
    data_filename = (
        f"part-00000.{extension}"  # Use part-* naming pattern for compatibility
    )
    data_path = output_dir / data_filename

    logger.info(f"Writing single {output_format} data file: {data_path}")

    try:
        if output_format == "CSV":
            df.to_csv(data_path, index=False)
        elif output_format == "JSON":
            df.to_json(data_path, orient="records", lines=True)
        elif output_format == "PARQUET":
            df.to_parquet(data_path, index=False)

        logger.info(f"Successfully wrote {len(df)} rows to {data_path}")
        return data_path

    except Exception as e:
        logger.error(f"Error writing {output_format} data file {data_path}: {str(e)}")
        raise


def write_data_output(
    df: pd.DataFrame,
    output_dir: Path,
    write_shards: bool = False,
    shard_size: int = 10000,
    output_format: str = "CSV",
) -> Union[Path, List[Path]]:
    """
    Write data output - either as shards or single file based on configuration.

    Args:
        df: Processed DataFrame
        output_dir: Output directory path
        write_shards: If True, write data as shards; if False, write single file
        shard_size: Number of rows per shard file
        output_format: Output format ('CSV', 'JSON', 'PARQUET')

    Returns:
        Path to single data file or list of shard file paths
    """
    if not write_shards:
        # Write single data file
        logger.info(f"Writing single data file: format={output_format}")
        return write_single_data_file(df, output_dir, output_format)

    # Write data shards
    logger.info(
        f"Writing data shards (enhanced mode): format={output_format}, shard_size={shard_size}"
    )
    return write_data_shards(df, output_dir, shard_size, output_format)


# ============================================================================
# STREAMING MODE FUNCTIONS
# ============================================================================


def process_first_batch_for_metadata(
    first_batch_files: List[Path],
    metadata_format: str,
    max_workers: Optional[int],
    batch_size: int,
) -> tuple:
    """
    Process first batch of files to generate signature and metadata.

    Extracts signature and metadata from the first batch only, enabling
    streaming mode to proceed without loading the full dataset.

    Args:
        first_batch_files: List of file paths in first batch
        metadata_format: "JSON" or "MODS"
        max_workers: Number of parallel workers
        batch_size: DataFrame concat batch size

    Returns:
        Tuple of (signature, metadata, first_batch_df)
    """
    logger.info(f"[STREAMING] Reading first batch: {len(first_batch_files)} files")

    # Read first batch
    first_batch_df = combine_files(
        first_batch_files,
        max_workers=max_workers,
        batch_size=batch_size,
        streaming_batch_size=None,  # Disable streaming within first batch
    )

    logger.info(f"[STREAMING] First batch shape: {first_batch_df.shape}")

    # Generate signature & metadata from first batch ONLY
    signature = generate_schema_signature(first_batch_df)

    if metadata_format == "MODS":
        metadata = generate_mods_metadata(first_batch_df)
        logger.info(
            "[STREAMING] Using MODS metadata format (lightweight, batch-compatible)"
        )
    else:
        # JSON format - warn that stats are from first batch only
        logger.warning(
            "[STREAMING] JSON metadata format in streaming mode: "
            "statistics are computed from first batch only"
        )
        metadata = generate_metadata(first_batch_df)

    return signature, metadata, first_batch_df


def write_batch_as_shards(
    df: pd.DataFrame,
    output_dir: Path,
    shard_counter: int,
    shard_size: int,
    output_format: str,
) -> tuple:
    """
    Write DataFrame batch as shards with continuous numbering.

    Args:
        df: DataFrame to write as shards
        output_dir: Output directory path
        shard_counter: Starting shard index number
        shard_size: Rows per shard
        output_format: Output data format

    Returns:
        Tuple of (written_shard_paths, updated_counter)
    """
    written_shards = []
    batch_rows = len(df)

    for i in range(0, batch_rows, shard_size):
        shard_df = df.iloc[i : i + shard_size]
        shard_path = write_single_shard(
            shard_df, output_dir, shard_counter, output_format
        )
        written_shards.append(shard_path)
        shard_counter += 1

    return written_shards, shard_counter


def process_remaining_batches(
    remaining_files: List[Path],
    data_output_dir: Path,
    shard_counter: int,
    streaming_batch_size: int,
    shard_size: int,
    output_format: str,
    max_workers: Optional[int],
    batch_size: int,
) -> tuple:
    """
    Stream and write remaining file batches.

    Processes remaining files in batches, writing shards incrementally
    without loading the full dataset into memory.

    Args:
        remaining_files: List of remaining file paths to process
        data_output_dir: Output directory for data shards
        shard_counter: Starting shard index number
        streaming_batch_size: Number of files per batch
        shard_size: Rows per shard
        output_format: Output data format
        max_workers: Number of parallel workers
        batch_size: DataFrame concat batch size

    Returns:
        Tuple of (all_written_shards, total_rows_processed, final_counter)
    """
    written_shards = []
    total_rows = 0

    logger.info(f"[STREAMING] Processing {len(remaining_files)} remaining files")

    for batch_start in range(0, len(remaining_files), streaming_batch_size):
        batch_end = min(batch_start + streaming_batch_size, len(remaining_files))
        batch_files = remaining_files[batch_start:batch_end]
        batch_num = (batch_start // streaming_batch_size) + 2  # +2 because first is #1

        logger.info(
            f"[STREAMING] Processing batch {batch_num}: {len(batch_files)} files"
        )

        # Read batch
        batch_df = combine_files(
            batch_files,
            max_workers=max_workers,
            batch_size=batch_size,
            streaming_batch_size=None,
        )

        # Write batch as shards
        batch_shards, shard_counter = write_batch_as_shards(
            batch_df, data_output_dir, shard_counter, shard_size, output_format
        )
        written_shards.extend(batch_shards)

        batch_rows = len(batch_df)
        total_rows += batch_rows

        logger.info(
            f"[STREAMING] Batch {batch_num} complete: "
            f"{batch_rows} rows, {shard_counter} total shards"
        )

        # Free memory
        del batch_df
        gc.collect()

    return written_shards, total_rows, shard_counter


def process_streaming_mode(
    data_files: List[Path],
    signature_output_dir: Path,
    metadata_output_dir: Path,
    data_output_dir: Path,
    metadata_format: str,
    streaming_batch_size: int,
    shard_size: int,
    output_format: str,
    max_workers: Optional[int],
    batch_size: int,
) -> Dict[str, Union[Path, List[Path]]]:
    """
    True streaming mode: Never loads full DataFrame into memory.

    Process data files in batches, generating outputs incrementally:
    1. First batch → signature & metadata (from first batch only)
    2. All batches → write shards incrementally
    3. Free memory after each batch

    Memory usage: ~1-2GB per batch (not dependent on total data size)
    Scales to: ANY data size (10GB, 100GB, 1TB+)

    Args:
        data_files: List of data file paths
        signature_output_dir: Directory for signature output
        metadata_output_dir: Directory for metadata output
        data_output_dir: Directory for data shard output
        metadata_format: "JSON" or "MODS"
        streaming_batch_size: Number of files per batch
        shard_size: Rows per output shard
        output_format: Output data format
        max_workers: Number of parallel workers
        batch_size: DataFrame concat batch size

    Returns:
        Dictionary of output file paths
    """
    logger.info(
        f"[STREAMING] Starting true streaming mode: "
        f"{len(data_files)} files in batches of {streaming_batch_size}"
    )

    total_files = len(data_files)
    shard_counter = 0
    written_shards = []
    total_rows_processed = 0

    # STEP 1: Process first batch for signature & metadata
    first_batch_size = min(streaming_batch_size, total_files)
    first_batch_files = data_files[:first_batch_size]

    signature, metadata, first_batch_df = process_first_batch_for_metadata(
        first_batch_files, metadata_format, max_workers, batch_size
    )

    # Write signature & metadata
    signature_file = write_signature_file(signature, signature_output_dir)
    metadata_file = write_metadata_file(
        metadata, metadata_output_dir, format=metadata_format
    )

    logger.info("[STREAMING] Signature and metadata written from first batch")

    # STEP 2: Write first batch shards
    first_shards, shard_counter = write_batch_as_shards(
        first_batch_df, data_output_dir, shard_counter, shard_size, output_format
    )
    written_shards.extend(first_shards)

    first_batch_rows = len(first_batch_df)
    total_rows_processed += first_batch_rows
    logger.info(
        f"[STREAMING] First batch complete: {len(written_shards)} shards written, "
        f"{first_batch_rows} rows"
    )

    # Free memory from first batch
    del first_batch_df
    gc.collect()

    # STEP 3: Stream remaining batches
    remaining_files = data_files[first_batch_size:]

    if remaining_files:
        remaining_shards, remaining_rows, shard_counter = process_remaining_batches(
            remaining_files,
            data_output_dir,
            shard_counter,
            streaming_batch_size,
            shard_size,
            output_format,
            max_workers,
            batch_size,
        )
        written_shards.extend(remaining_shards)
        total_rows_processed += remaining_rows

    logger.info(
        f"[STREAMING] Complete: {shard_counter} shards, "
        f"{total_rows_processed} total rows from {total_files} files"
    )

    return {
        "signature": signature_file,
        "metadata": metadata_file,
        "data": written_shards,
    }


# ============================================================================
# MAIN PROCESSING LOGIC
# ============================================================================


def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: Optional[argparse.Namespace] = None,
) -> Dict[str, Union[Path, List[Path]]]:
    """
    Main entry point for the Dummy Data Loading script.

    Args:
        input_paths: Dictionary of input paths with logical names
        output_paths: Dictionary of output paths with logical names
        environ_vars: Dictionary of environment variables
        job_args: Command line arguments (optional)

    Returns:
        Dictionary of output file paths
    """
    try:
        logger.info("Starting dummy data loading process")

        # Get configuration from environment variables
        write_shards = environ_vars.get("WRITE_DATA_SHARDS", "false").lower() == "true"
        shard_size = int(environ_vars.get("SHARD_SIZE", "10000"))
        output_format = environ_vars.get("OUTPUT_FORMAT", "CSV").upper()

        # Memory optimization parameters
        max_workers = int(environ_vars.get("MAX_WORKERS", 0)) or None  # 0 means auto
        batch_size = int(environ_vars.get("BATCH_SIZE", 5))
        optimize_memory = environ_vars.get("OPTIMIZE_MEMORY", "true").lower() == "true"
        streaming_batch_size = (
            int(environ_vars.get("STREAMING_BATCH_SIZE", 0)) or None
        )  # 0 means disabled

        # NEW: True streaming mode and metadata format
        enable_true_streaming = (
            environ_vars.get("ENABLE_TRUE_STREAMING", "false").lower() == "true"
        )
        metadata_format = environ_vars.get("METADATA_FORMAT", "JSON").upper()

        # Validate output format
        supported_formats = ["CSV", "JSON", "PARQUET"]
        if output_format not in supported_formats:
            raise ValueError(
                f"Invalid OUTPUT_FORMAT: {output_format}. "
                f"Supported formats: {supported_formats}"
            )

        # Validate metadata format
        supported_metadata_formats = ["JSON", "MODS"]
        if metadata_format not in supported_metadata_formats:
            raise ValueError(
                f"Invalid METADATA_FORMAT: {metadata_format}. "
                f"Supported formats: {supported_metadata_formats}"
            )

        logger.info(
            f"Configuration: WRITE_DATA_SHARDS={write_shards}, "
            f"SHARD_SIZE={shard_size}, OUTPUT_FORMAT={output_format}"
        )
        logger.info(f"Memory optimization settings:")
        logger.info(f"  MAX_WORKERS: {max_workers if max_workers else 'auto'}")
        logger.info(f"  BATCH_SIZE: {batch_size}")
        logger.info(f"  OPTIMIZE_MEMORY: {optimize_memory}")
        logger.info(
            f"  STREAMING_BATCH_SIZE: {streaming_batch_size if streaming_batch_size else 'disabled'}"
        )
        logger.info(f"  ENABLE_TRUE_STREAMING: {enable_true_streaming}")
        logger.info(f"  METADATA_FORMAT: {metadata_format}")

        # Get input and output directories
        input_data_dir = Path(input_paths["INPUT_DATA"])
        signature_output_dir = Path(output_paths["SIGNATURE"])
        metadata_output_dir = Path(output_paths["METADATA"])
        data_output_dir = Path(output_paths["DATA"])

        logger.info(f"Input data directory: {input_data_dir}")
        logger.info(f"Signature output directory: {signature_output_dir}")
        logger.info(f"Metadata output directory: {metadata_output_dir}")
        logger.info(f"Data output directory: {data_output_dir}")

        # Find data files
        data_files = find_data_files(input_data_dir)
        if not data_files:
            raise ValueError(f"No supported data files found in {input_data_dir}")

        # ROUTING: Choose between TRUE STREAMING MODE or BATCH MODE
        if enable_true_streaming:
            # TRUE STREAMING MODE: Never loads full DataFrame
            if not write_shards:
                logger.warning(
                    "[WARNING] ENABLE_TRUE_STREAMING=true requires WRITE_DATA_SHARDS=true. "
                    "Enabling shard writing automatically."
                )
                write_shards = True

            if streaming_batch_size is None:
                # Auto-set streaming batch size if not provided
                streaming_batch_size = 10
                logger.info(
                    f"[STREAMING] Auto-set STREAMING_BATCH_SIZE to {streaming_batch_size}"
                )

            logger.info("[STREAMING] Using TRUE STREAMING MODE")
            result = process_streaming_mode(
                data_files=data_files,
                signature_output_dir=signature_output_dir,
                metadata_output_dir=metadata_output_dir,
                data_output_dir=data_output_dir,
                metadata_format=metadata_format,
                streaming_batch_size=streaming_batch_size,
                shard_size=shard_size,
                output_format=output_format,
                max_workers=max_workers,
                batch_size=batch_size,
            )

        else:
            # BATCH MODE: Original behavior with optional memory optimizations
            logger.info("[BATCH] Using BATCH MODE")

            # Process all data files using optimized combine_files function
            logger.info(f"[INFO] Combining data files...")
            combined_df = combine_files(
                data_files, max_workers, batch_size, streaming_batch_size
            )
            logger.info(f"[INFO] Combined data shape: {combined_df.shape}")

            # Apply memory optimization if enabled
            if optimize_memory:
                combined_df = optimize_dtypes(combined_df, logger.info)

            # Generate signature and metadata
            signature = generate_schema_signature(combined_df)

            if metadata_format == "MODS":
                metadata = generate_mods_metadata(combined_df)
            else:
                metadata = generate_metadata(combined_df)

            # Write output files
            signature_file = write_signature_file(signature, signature_output_dir)
            metadata_file = write_metadata_file(
                metadata, metadata_output_dir, format=metadata_format
            )

            # Write data output (configurable: shards or single file)
            data_output = write_data_output(
                combined_df,
                data_output_dir,
                write_shards=write_shards,
                shard_size=shard_size,
                output_format=output_format,
            )

            result = {
                "signature": signature_file,
                "metadata": metadata_file,
                "data": data_output,
            }

        logger.info("Dummy data loading completed successfully")
        return result

    except Exception as e:
        logger.error(f"Error in dummy data loading: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        # Define input and output paths based on contract
        input_paths = {"INPUT_DATA": INPUT_DATA_DIR}

        output_paths = {
            "SIGNATURE": SIGNATURE_OUTPUT_DIR,
            "METADATA": METADATA_OUTPUT_DIR,
            "DATA": DATA_OUTPUT_DIR,
        }

        # Read environment variables from system
        environ_vars = {
            "WRITE_DATA_SHARDS": os.environ.get("WRITE_DATA_SHARDS", "false"),
            "SHARD_SIZE": os.environ.get("SHARD_SIZE", "10000"),
            "OUTPUT_FORMAT": os.environ.get("OUTPUT_FORMAT", "CSV"),
            "MAX_WORKERS": os.environ.get("MAX_WORKERS", "0"),
            "BATCH_SIZE": os.environ.get("BATCH_SIZE", "5"),
            "OPTIMIZE_MEMORY": os.environ.get("OPTIMIZE_MEMORY", "false"),
            "STREAMING_BATCH_SIZE": os.environ.get("STREAMING_BATCH_SIZE", "0"),
            "ENABLE_TRUE_STREAMING": os.environ.get("ENABLE_TRUE_STREAMING", "false"),
            "METADATA_FORMAT": os.environ.get("METADATA_FORMAT", "JSON"),
        }

        # Log configuration for debugging
        logger.info(f"Environment configuration:")
        for key, value in environ_vars.items():
            logger.info(f"  {key}={value}")

        # No command line arguments needed for this script
        args = None

        # Execute the main function
        result = main(input_paths, output_paths, environ_vars, args)

        logger.info(f"Dummy data loading completed successfully")
        logger.info(f"Output files: {result}")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Error in dummy data loading script: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

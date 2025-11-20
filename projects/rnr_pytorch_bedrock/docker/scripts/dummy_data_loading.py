#!/usr/bin/env python
"""
Dummy Data Loading Processing Script

This script processes user-provided data instead of calling internal Cradle services.
It serves as a drop-in replacement for CradleDataLoadingStep by reading data from
an input channel, generating schema signatures and metadata, and outputting the
processed data in the same format as the original Cradle data loading step.
"""

import argparse
import json
import logging
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Dict, Optional, List, Any, Union
import pandas as pd
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


def ensure_directory(directory: Path) -> bool:
    """Ensure a directory exists, creating it if necessary."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory ensured: {directory}")
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {directory}: {str(e)}", exc_info=True)
        return False


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


def process_data_files(data_files: List[Path]) -> pd.DataFrame:
    """
    Process multiple data files and combine them into a single DataFrame.

    Args:
        data_files: List of data file paths

    Returns:
        Combined DataFrame

    Raises:
        ValueError: If no valid data files found
        Exception: If processing fails
    """
    if not data_files:
        raise ValueError("No data files found to process")

    logger.info(f"Processing {len(data_files)} data files")

    combined_df = None

    for file_path in data_files:
        try:
            file_format = detect_file_format(file_path)
            if file_format == "unknown":
                logger.warning(f"Skipping file with unknown format: {file_path}")
                continue

            df = read_data_file(file_path, file_format)

            if combined_df is None:
                combined_df = df
            else:
                # Combine DataFrames (concatenate rows)
                combined_df = pd.concat([combined_df, df], ignore_index=True)

            logger.info(f"Processed file: {file_path} ({len(df)} rows)")

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            # Continue with other files instead of failing completely
            continue

    if combined_df is None:
        raise ValueError("No valid data could be processed from input files")

    logger.info(
        f"Combined data: {len(combined_df)} rows, {len(combined_df.columns)} columns"
    )
    return combined_df


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


def write_metadata_file(metadata: Dict[str, Any], output_dir: Path) -> Path:
    """
    Write the metadata file to the output directory.

    Args:
        metadata: Metadata dictionary
        output_dir: Output directory path

    Returns:
        Path to the written metadata file
    """
    ensure_directory(output_dir)
    metadata_file = output_dir / "metadata"

    logger.info(f"Writing metadata file: {metadata_file}")

    try:
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Metadata file written successfully")
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

        # Validate output format
        supported_formats = ["CSV", "JSON", "PARQUET"]
        if output_format not in supported_formats:
            raise ValueError(
                f"Invalid OUTPUT_FORMAT: {output_format}. "
                f"Supported formats: {supported_formats}"
            )

        logger.info(
            f"Configuration: WRITE_DATA_SHARDS={write_shards}, "
            f"SHARD_SIZE={shard_size}, OUTPUT_FORMAT={output_format}"
        )

        # Get input and output directories
        input_data_dir = Path(input_paths["INPUT_DATA"])
        signature_output_dir = Path(output_paths["SIGNATURE"])
        metadata_output_dir = Path(output_paths["METADATA"])
        data_output_dir = Path(output_paths["DATA"])

        logger.info(f"Input data directory: {input_data_dir}")
        logger.info(f"Signature output directory: {signature_output_dir}")
        logger.info(f"Metadata output directory: {metadata_output_dir}")
        logger.info(f"Data output directory: {data_output_dir}")

        # Find and process data files
        data_files = find_data_files(input_data_dir)
        if not data_files:
            raise ValueError(f"No supported data files found in {input_data_dir}")

        # Process all data files
        combined_df = process_data_files(data_files)

        # Generate signature and metadata
        signature = generate_schema_signature(combined_df)
        metadata = generate_metadata(combined_df)

        # Write output files
        signature_file = write_signature_file(signature, signature_output_dir)
        metadata_file = write_metadata_file(metadata, metadata_output_dir)

        # Write data output (configurable: shards or placeholder)
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

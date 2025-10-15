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
DATA_OUTPUT_DIR = "/opt/ml/processing/output/place_holder"


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
    if suffix in ['.csv']:
        return 'csv'
    elif suffix in ['.parquet', '.pq']:
        return 'parquet'
    elif suffix in ['.json', '.jsonl']:
        return 'json'
    
    # If extension is unclear, try to read the file
    try:
        # Try CSV first
        pd.read_csv(file_path, nrows=1)
        return 'csv'
    except:
        pass
    
    try:
        # Try Parquet
        pd.read_parquet(file_path)
        return 'parquet'
    except:
        pass
    
    try:
        # Try JSON
        pd.read_json(file_path, lines=True, nrows=1)
        return 'json'
    except:
        pass
    
    logger.warning(f"Could not detect format for file: {file_path}")
    return 'unknown'


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
        if file_format == 'csv':
            df = pd.read_csv(file_path)
        elif file_format == 'parquet':
            df = pd.read_parquet(file_path)
        elif file_format == 'json':
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
            "memory_usage_bytes": int(df.memory_usage(deep=True).sum())
        },
        "column_info": {}
    }
    
    for column in df.columns:
        col_info = {
            "data_type": str(df[column].dtype),
            "null_count": int(df[column].isnull().sum()),
            "unique_count": int(df[column].nunique()),
            "memory_usage": int(df[column].memory_usage(deep=True))
        }
        
        # Add basic statistics for numeric columns
        if pd.api.types.is_numeric_dtype(df[column]):
            col_info.update({
                "min": float(df[column].min()) if not df[column].empty else None,
                "max": float(df[column].max()) if not df[column].empty else None,
                "mean": float(df[column].mean()) if not df[column].empty else None,
                "std": float(df[column].std()) if not df[column].empty else None
            })
        
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
    supported_extensions = {'.csv', '.parquet', '.pq', '.json', '.jsonl'}
    
    for file_path in input_dir.rglob('*'):
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
            if file_format == 'unknown':
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
    
    logger.info(f"Combined data: {len(combined_df)} rows, {len(combined_df.columns)} columns")
    return combined_df


def write_signature_file(signature: List[str], output_dir: Path) -> Path:
    """
    Write the signature file to the output directory.
    
    Args:
        signature: Schema signature dictionary
        output_dir: Output directory path
        
    Returns:
        Path to the written signature file
    """
    ensure_directory(output_dir)
    signature_file = output_dir / "signature"
    
    logger.info(f"Writing signature file: {signature_file}")
    
    try:
        with open(signature_file, 'w') as f:
            json.dump(signature, f, indent=2)
        
        logger.info("Signature file written successfully")
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
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Metadata file written successfully")
        return metadata_file
        
    except Exception as e:
        logger.error(f"Error writing metadata file: {str(e)}")
        raise


def write_data_placeholder(output_dir: Path) -> Path:
    """
    Write a placeholder file for data output (similar to Cradle behavior).
    
    Args:
        output_dir: Output directory path
        
    Returns:
        Path to the written placeholder file
    """
    ensure_directory(output_dir)
    placeholder_file = output_dir / "data_processed"
    
    logger.info(f"Writing data placeholder file: {placeholder_file}")
    
    try:
        with open(placeholder_file, 'w') as f:
            f.write("Data processing completed successfully\n")
        
        logger.info("Data placeholder file written successfully")
        return placeholder_file
        
    except Exception as e:
        logger.error(f"Error writing data placeholder file: {str(e)}")
        raise


def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: Optional[argparse.Namespace] = None,
) -> Dict[str, Path]:
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
        data_placeholder = write_data_placeholder(data_output_dir)
        
        result = {
            "signature": signature_file,
            "metadata": metadata_file,
            "data": data_placeholder
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
            "DATA": DATA_OUTPUT_DIR
        }
        
        # Environment variables dictionary
        environ_vars = {}
        
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

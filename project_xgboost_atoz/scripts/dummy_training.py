#!/usr/bin/env python
"""
DummyTraining Processing Script

This script validates, unpacks a pretrained model.tar.gz file, adds a hyperparameters.json file 
inside it, then repacks it and outputs to the destination. It serves as a dummy training step 
that skips actual training and integrates with downstream MIMS packaging and payload steps.
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tarfile
import tempfile
import traceback
from pathlib import Path
from typing import Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Source directory paths with fallback support
# These paths align with SOURCE node approach where model and hyperparameters
# are embedded in the source directory structure
MODEL_SOURCE_PATH = "/opt/ml/code/models/model.tar.gz"
HYPERPARAMS_SOURCE_PATH = "/opt/ml/code/hyperparams/hyperparameters.json"
MODEL_OUTPUT_DIR = "/opt/ml/processing/output/model"


def validate_model(input_path: Path) -> bool:
    """
    Validate the model file format and structure.

    Args:
        input_path: Path to the input model.tar.gz file

    Returns:
        True if validation passes, False otherwise

    Raises:
        ValueError: If the file format is incorrect
        Exception: For other validation errors
    """
    logger.info(f"Validating model file: {input_path}")

    # Check file extension
    if not input_path.suffix == ".tar.gz" and not str(input_path).endswith(".tar.gz"):
        raise ValueError(
            f"Expected a .tar.gz file, but got: {input_path} (ERROR_CODE: INVALID_FORMAT)"
        )

    # Check if it's a valid tar archive
    if not tarfile.is_tarfile(input_path):
        raise ValueError(
            f"File is not a valid tar archive: {input_path} (ERROR_CODE: INVALID_ARCHIVE)"
        )

    # Additional validation could be performed here:
    # - Check for required files within the archive
    # - Verify file sizes and structures
    # - Validate model format-specific details

    logger.info("Model validation successful")
    return True


def ensure_directory(directory: Path) -> bool:
    """Ensure a directory exists, creating it if necessary."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory ensured: {directory}")
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {directory}: {str(e)}", exc_info=True)
        return False


def extract_tarfile(tar_path: Path, extract_path: Path) -> None:
    """Extract a tar file to the specified path."""
    logger.info(f"Extracting tar file: {tar_path} to {extract_path}")

    if not tar_path.exists():
        raise FileNotFoundError(f"Tar file not found: {tar_path}")

    ensure_directory(extract_path)

    try:
        with tarfile.open(tar_path, "r:*") as tar:
            logger.info(f"Tar file contents before extraction:")
            total_size = 0
            for member in tar.getmembers():
                size_mb = member.size / 1024 / 1024
                total_size += size_mb
                logger.info(f"  {member.name} ({size_mb:.2f}MB)")
            logger.info(f"Total size in tar: {total_size:.2f}MB")

            logger.info(f"Extracting to: {extract_path}")
            tar.extractall(path=extract_path)

        logger.info("Extraction completed")

    except Exception as e:
        logger.error(f"Error during tar extraction: {str(e)}", exc_info=True)
        raise


def create_tarfile(output_tar_path: Path, source_dir: Path) -> None:
    """Create a tar file from the contents of a directory."""
    logger.info(f"Creating tar file: {output_tar_path} from {source_dir}")

    ensure_directory(output_tar_path.parent)

    try:
        total_size = 0
        files_added = 0

        with tarfile.open(output_tar_path, "w:gz") as tar:
            for item in source_dir.rglob("*"):
                if item.is_file():
                    arcname = item.relative_to(source_dir)
                    size_mb = item.stat().st_size / 1024 / 1024
                    total_size += size_mb
                    files_added += 1
                    logger.info(f"Adding to tar: {arcname} ({size_mb:.2f}MB)")
                    tar.add(item, arcname=arcname)

        logger.info(f"Tar creation summary:")
        logger.info(f"  Files added: {files_added}")
        logger.info(f"  Total uncompressed size: {total_size:.2f}MB")

        if output_tar_path.exists():
            compressed_size = output_tar_path.stat().st_size / 1024 / 1024
            logger.info(f"  Compressed tar size: {compressed_size:.2f}MB")
            logger.info(f"  Compression ratio: {compressed_size/total_size:.2%}")

    except Exception as e:
        logger.error(f"Error creating tar file: {str(e)}", exc_info=True)
        raise


def copy_file(src: Path, dst: Path) -> None:
    """Copy a file and ensure the destination directory exists."""
    logger.info(f"Copying file: {src} to {dst}")

    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")

    ensure_directory(dst.parent)

    try:
        shutil.copy2(src, dst)
        logger.info(f"File copied successfully")
    except Exception as e:
        logger.error(f"Error copying file: {str(e)}", exc_info=True)
        raise


def process_model_with_hyperparameters(
    model_path: Path, hyperparams_path: Path, output_dir: Path
) -> Path:
    """
    Process the model.tar.gz by unpacking it, adding hyperparameters.json, and repacking it.

    Args:
        model_path: Path to the input model.tar.gz file
        hyperparams_path: Path to the hyperparameters.json file
        output_dir: Directory to save the processed model

    Returns:
        Path to the processed model.tar.gz

    Raises:
        FileNotFoundError: If input files don't exist
        Exception: For processing errors
    """
    logger.info(f"Processing model with hyperparameters")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Hyperparameters path: {hyperparams_path}")
    logger.info(f"Output directory: {output_dir}")

    # Validate inputs
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not hyperparams_path.exists():
        raise FileNotFoundError(f"Hyperparameters file not found: {hyperparams_path}")

    # Create a temporary working directory
    with tempfile.TemporaryDirectory() as temp_dir:
        working_dir = Path(temp_dir)
        logger.info(f"Created temporary working directory: {working_dir}")

        # Extract the model.tar.gz
        extract_tarfile(model_path, working_dir)

        # Copy hyperparameters.json to the working directory
        hyperparams_dest = working_dir / "hyperparameters.json"
        copy_file(hyperparams_path, hyperparams_dest)

        # Ensure output directory exists
        ensure_directory(output_dir)

        # Create the output model.tar.gz
        output_path = output_dir / "model.tar.gz"
        create_tarfile(output_path, working_dir)

        logger.info(f"Model processing complete. Output at: {output_path}")
        return output_path


def find_model_file(base_paths: list[str]) -> Optional[Path]:
    """
    Find model.tar.gz file in multiple possible locations.
    
    Args:
        base_paths: List of base paths to search
        
    Returns:
        Path to model file if found, None otherwise
    """
    for base_path in base_paths:
        model_path = Path(base_path) / "model.tar.gz"
        if model_path.exists():
            logger.info(f"Found model file at: {model_path}")
            return model_path
    return None


def find_hyperparams_file(base_paths: list[str]) -> Optional[Path]:
    """
    Find hyperparameters.json file in multiple possible locations.
    
    Args:
        base_paths: List of base paths to search
        
    Returns:
        Path to hyperparameters file if found, None otherwise
    """
    for base_path in base_paths:
        hyperparams_path = Path(base_path) / "hyperparameters.json"
        if hyperparams_path.exists():
            logger.info(f"Found hyperparameters file at: {hyperparams_path}")
            return hyperparams_path
    return None


def main(
    input_paths: Dict[str, str],  # Will be empty dict for SOURCE node
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: Optional[argparse.Namespace] = None,
) -> Path:
    """
    Main entry point for the DummyTraining script.
    
    Reads model and hyperparameters from source directory with fallback mechanisms.

    Args:
        input_paths: Dictionary of input paths with logical names (empty for SOURCE node)
        output_paths: Dictionary of output paths with logical names
        environ_vars: Dictionary of environment variables
        job_args: Command line arguments (optional)

    Returns:
        Path to the processed model.tar.gz output
    """
    try:
        # Define fallback search paths for model file
        model_search_paths = [
            "/opt/ml/code/models",           # Primary: source directory models folder
            "/opt/ml/code",                  # Fallback: source directory root
            "/opt/ml/processing/input/model", # Legacy: processing input (if somehow provided)
        ]
        
        # Define fallback search paths for hyperparameters file
        hyperparams_search_paths = [
            "/opt/ml/code/hyperparams",      # Primary: source directory hyperparams folder
            "/opt/ml/code",                  # Fallback: source directory root
            "/opt/ml/processing/input/config", # Legacy: processing input (if somehow provided)
        ]
        
        # Find model file with fallback
        model_path = find_model_file(model_search_paths)
        if not model_path:
            raise FileNotFoundError(
                f"Model file (model.tar.gz) not found in any of these locations: {model_search_paths}"
            )
        
        # Find hyperparameters file with fallback
        hyperparams_path = find_hyperparams_file(hyperparams_search_paths)
        if not hyperparams_path:
            raise FileNotFoundError(
                f"Hyperparameters file (hyperparameters.json) not found in any of these locations: {hyperparams_search_paths}"
            )
        
        # Get output directory
        output_dir = Path(output_paths["model_output"])
        
        logger.info(f"Using paths:")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Hyperparameters: {hyperparams_path}")
        logger.info(f"  Output: {output_dir}")
        
        # Process model with hyperparameters from source directory
        output_path = process_model_with_hyperparameters(
            model_path, hyperparams_path, output_dir
        )
        
        return output_path
    except FileNotFoundError as e:
        logger.error(f"Required file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in dummy training: {e}")
        raise


if __name__ == "__main__":
    try:
        # Empty input paths - consistent with SOURCE node contract
        input_paths = {}
        
        output_paths = {"model_output": MODEL_OUTPUT_DIR}
        
        # Environment variables dictionary
        environ_vars = {}
        
        # No command line arguments needed for this script
        args = None
        
        # Execute the main function
        result = main(input_paths, output_paths, environ_vars, args)
        
        logger.info(f"Dummy training completed successfully. Output model at: {result}")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error in dummy training script: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

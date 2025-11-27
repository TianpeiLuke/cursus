#!/usr/bin/env python
"""
DummyTraining Processing Script

This script validates, unpacks a pretrained model.tar.gz file, conditionally adds a
hyperparameters.json file inside it, then repacks it and outputs to the destination.
It serves as a dummy training step that skips actual training and integrates with
downstream MIMS packaging and payload steps.

Hyperparameters Handling:
    - If model.tar.gz already contains hyperparameters.json (e.g., from PyTorch/XGBoost training):
      * Keeps the original hyperparameters from the model
      * Ignores any hyperparameters provided via input channel

    - If model.tar.gz does NOT contain hyperparameters.json:
      * Requires hyperparameters.json from input channel
      * Injects it into the model archive

    - Fails only if BOTH conditions are true:
      * Model archive doesn't contain hyperparameters.json
      * No hyperparameters.json provided via input channel
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
            logger.info(f"  Compression ratio: {compressed_size / total_size:.2%}")

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
    model_path: Path, hyperparams_path: Optional[Path], output_dir: Path
) -> Path:
    """
    Process the model.tar.gz by unpacking it and conditionally adding hyperparameters.json.

    The hyperparameters.json file is only added if:
    1. It doesn't already exist in the model archive
    2. An input hyperparameters_path is provided

    Args:
        model_path: Path to the input model.tar.gz file
        hyperparams_path: Optional path to the hyperparameters.json file (None if not provided)
        output_dir: Directory to save the processed model

    Returns:
        Path to the processed model.tar.gz

    Raises:
        FileNotFoundError: If model doesn't exist, or if hyperparameters are missing from both model and input
        Exception: For processing errors
    """
    logger.info(f"Processing model with hyperparameters")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Hyperparameters path: {hyperparams_path}")
    logger.info(f"Output directory: {output_dir}")

    # Validate inputs
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Create a temporary working directory
    with tempfile.TemporaryDirectory() as temp_dir:
        working_dir = Path(temp_dir)
        logger.info(f"Created temporary working directory: {working_dir}")

        # Extract the model.tar.gz
        extract_tarfile(model_path, working_dir)

        # Check if hyperparameters.json already exists in the extracted model
        hyperparams_dest = working_dir / "hyperparameters.json"

        if hyperparams_dest.exists():
            logger.info("=" * 70)
            logger.info("HYPERPARAMETERS ALREADY IN MODEL")
            logger.info("=" * 70)
            logger.info(
                f"hyperparameters.json found in model archive at: {hyperparams_dest}"
            )

            if hyperparams_path:
                logger.info("Input hyperparameters provided but will be IGNORED")
                logger.info(f"  Input path: {hyperparams_path}")
                logger.info(
                    "  Reason: Model archive already contains hyperparameters.json"
                )
                logger.info("  Action: Keeping original hyperparameters from model")
            else:
                logger.info("No input hyperparameters provided (not needed)")

            logger.info("=" * 70)
        else:
            logger.info("=" * 70)
            logger.info("HYPERPARAMETERS NOT IN MODEL")
            logger.info("=" * 70)
            logger.info("hyperparameters.json NOT found in model archive")

            if hyperparams_path:
                logger.info(f"Injecting hyperparameters from input: {hyperparams_path}")
                copy_file(hyperparams_path, hyperparams_dest)
                logger.info("✓ Hyperparameters successfully added to model")
            else:
                logger.error(
                    "ERROR: No hyperparameters found in model AND no input provided"
                )
                raise FileNotFoundError(
                    "hyperparameters.json not found in model.tar.gz and no input hyperparameters provided. "
                    "Either the model must contain hyperparameters.json or it must be provided via input channel."
                )

            logger.info("=" * 70)

        # Ensure output directory exists
        ensure_directory(output_dir)

        # Create the output model.tar.gz
        output_path = output_dir / "model.tar.gz"
        create_tarfile(output_path, working_dir)

        logger.info(f"Model processing complete. Output at: {output_path}")
        return output_path


def find_model_file(input_paths: Dict[str, str]) -> Optional[Path]:
    """
    Find model.tar.gz file with fallback search.

    Priority:
    1. Pre-configured path from input_paths (either input channel or /opt/ml/code/models)
    2. Final fallback: model.tar.gz relative to script location

    Args:
        input_paths: Dictionary of input paths from container

    Returns:
        Path to model file if found, None otherwise
    """
    # Priority 1: Pre-configured path
    if "model_artifacts_input" in input_paths and input_paths["model_artifacts_input"]:
        model_path = Path(input_paths["model_artifacts_input"]) / "model.tar.gz"
        if model_path.exists():
            logger.info(f"Found model file: {model_path}")
            return model_path
        else:
            logger.warning(f"model.tar.gz not found at: {model_path}")

    # Priority 2: Final fallback - relative to script location
    script_dir = Path(__file__).parent
    code_fallback_path = script_dir / "model.tar.gz"
    if code_fallback_path.exists():
        logger.info(f"Found model file relative to script: {code_fallback_path}")
        return code_fallback_path

    return None


def find_hyperparams_file(input_paths: Dict[str, str]) -> Optional[Path]:
    """
    Find hyperparameters.json file at the specified path.

    The input_paths["hyperparameters_s3_uri"] is pre-configured in __main__ to point to either:
    - /opt/ml/processing/input/hyperparameters_s3_uri (if dependency injection provided)
    - /opt/ml/code/hyperparams/ (SOURCE fallback)

    Args:
        input_paths: Dictionary of input paths from container

    Returns:
        Path to hyperparameters file if found, None otherwise
    """
    if (
        "hyperparameters_s3_uri" in input_paths
        and input_paths["hyperparameters_s3_uri"]
    ):
        hparam_path = (
            Path(input_paths["hyperparameters_s3_uri"]) / "hyperparameters.json"
        )
        if hparam_path.exists():
            logger.info(f"Found hyperparameters file: {hparam_path}")
            return hparam_path
        else:
            logger.warning(f"hyperparameters.json not found at: {hparam_path}")

    return None


def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: Optional[argparse.Namespace] = None,
) -> Path:
    """
    Main entry point for the DummyTraining script.

    Reads model and hyperparameters with flexible input modes:
    - Mode 1 (INTERNAL): From input channels (model_artifacts_input, hyperparameters_s3_uri)
    - Mode 2 (SOURCE): From source directory (fallback)

    Args:
        input_paths: Dictionary of input paths with logical names
            - "model_artifacts_input": Optional path to model.tar.gz
            - "hyperparameters_s3_uri": Optional path to hyperparameters.json
        output_paths: Dictionary of output paths with logical names
            - "model_output": Output directory for processed model
        environ_vars: Dictionary of environment variables
        job_args: Command line arguments (optional)

    Returns:
        Path to the processed model.tar.gz output
    """
    try:
        logger.info("=" * 70)
        logger.info("DUMMY TRAINING - FLEXIBLE INPUT MODE")
        logger.info("=" * 70)
        logger.info(f"Input paths provided: {list(input_paths.keys())}")
        logger.info(f"Output paths: {list(output_paths.keys())}")
        logger.info("=" * 70)

        # Find model file (REQUIRED)
        model_path = find_model_file(input_paths)
        if not model_path:
            raise FileNotFoundError(
                f"Model file (model.tar.gz) not found at: "
                f"{input_paths.get('model_artifacts_input', 'No path provided')}/model.tar.gz"
            )

        # Find hyperparameters file (OPTIONAL - may be in model.tar.gz)
        hyperparams_path = find_hyperparams_file(input_paths)
        if not hyperparams_path:
            logger.info("=" * 70)
            logger.info("HYPERPARAMETERS INPUT NOT PROVIDED")
            logger.info("=" * 70)
            logger.info("hyperparameters.json not found in input paths")
            logger.info("Will check if hyperparameters.json exists in model.tar.gz")
            logger.info("=" * 70)

        # Get output directory
        output_dir = Path(output_paths["model_output"])

        logger.info("=" * 70)
        logger.info("RESOLVED PATHS:")
        logger.info(f"  Model: {model_path}")
        logger.info(
            f"  Hyperparameters: {hyperparams_path if hyperparams_path else 'None (will check in model)'}"
        )
        logger.info(f"  Output: {output_dir}")
        logger.info("=" * 70)

        # Process model with hyperparameters
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
        # Container path constants
        CONTAINER_PATHS = {
            "MODEL_OUTPUT": "/opt/ml/processing/output/model",
            "MODEL_ARTIFACTS_INPUT": "/opt/ml/processing/input/model_artifacts_input",
            "HYPERPARAMETERS_INPUT": "/opt/ml/processing/input/hyperparameters_s3_uri",
        }

        # Define input paths - always provide paths (either input channel or code directory)
        input_paths = {}

        # Model artifacts path: Always provided (either input channel or code directory)
        if os.path.exists(CONTAINER_PATHS["MODEL_ARTIFACTS_INPUT"]):
            input_paths["model_artifacts_input"] = CONTAINER_PATHS[
                "MODEL_ARTIFACTS_INPUT"
            ]
            logger.info(
                f"[Input Channel] Using model artifacts from: {CONTAINER_PATHS['MODEL_ARTIFACTS_INPUT']}"
            )
        else:
            input_paths["model_artifacts_input"] = "/opt/ml/code/models"
            logger.info(
                f"[SOURCE Fallback] Using model artifacts from: /opt/ml/code/models"
            )

        # Hyperparameters path: Always provided (either input channel or code directory)
        if os.path.exists(CONTAINER_PATHS["HYPERPARAMETERS_INPUT"]):
            input_paths["hyperparameters_s3_uri"] = CONTAINER_PATHS[
                "HYPERPARAMETERS_INPUT"
            ]
            logger.info(
                f"[Input Channel] Using hyperparameters from: {CONTAINER_PATHS['HYPERPARAMETERS_INPUT']}"
            )
        else:
            input_paths["hyperparameters_s3_uri"] = "/opt/ml/code/hyperparams"
            logger.info(
                f"[SOURCE Fallback] Using hyperparameters from: /opt/ml/code/hyperparams"
            )

        # Define output paths
        output_paths = {"model_output": CONTAINER_PATHS["MODEL_OUTPUT"]}

        # Environment variables dictionary (currently unused but kept for consistency)
        environ_vars = {}

        # No command line arguments needed for this script
        args = None

        logger.info(
            f"Starting dummy training with input mode: {'INTERNAL' if input_paths else 'SOURCE'}"
        )

        # Execute the main function
        result = main(input_paths, output_paths, environ_vars, args)

        logger.info(f"Dummy training completed successfully. Output model at: {result}")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error in dummy training script: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

import shutil
import tarfile
import argparse
import traceback
from pathlib import Path
import logging
import os
from typing import List, Dict, Optional, Any
import sys

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constants - default paths (will be overridden by parameters in main function)
DEFAULT_MODEL_PATH = "/opt/ml/processing/input/model"
DEFAULT_SCRIPT_PATH = "/opt/ml/processing/input/script"
DEFAULT_CALIBRATION_PATH = "/opt/ml/processing/input/calibration"
DEFAULT_OUTPUT_PATH = "/opt/ml/processing/output"
DEFAULT_WORKING_DIRECTORY = "/tmp/mims_packaging_directory"


def ensure_directory(directory: Path) -> bool:
    """Ensure a directory exists, creating it if necessary."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory ensured: {directory}")
        logger.debug(f"Directory permissions: {oct(directory.stat().st_mode)[-3:]}")
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {directory}: {str(e)}", exc_info=True)
        return False


def check_file_exists(path: Path, description: str) -> bool:
    """Check if a file exists and log its details."""
    exists = path.exists() and path.is_file()
    try:
        if exists:
            stats = path.stat()
            size_mb = stats.st_size / 1024 / 1024
            logger.info(f"{description}:")
            logger.info(f"  Path: {path}")
            logger.info(f"  Size: {size_mb:.2f}MB")
            logger.info(f"  Permissions: {oct(stats.st_mode)[-3:]}")
            logger.info(f"  Last modified: {stats.st_mtime}")
        else:
            logger.warning(f"{description} not found at {path}")
        return exists
    except Exception as e:
        logger.error(f"Error checking file {path}: {str(e)}", exc_info=True)
        return False


def list_directory_contents(path: Path, description: str) -> None:
    """List and log the contents of a directory."""
    logger.info(f"\n{'='*20} Contents of {description} {'='*20}")
    logger.info(f"Path: {path}")

    if not path.exists():
        logger.warning(f"Directory does not exist: {path}")
        return

    if not path.is_dir():
        logger.warning(f"Path exists but is not a directory: {path}")
        return

    try:
        total_size = 0
        file_count = 0
        dir_count = 0

        logger.info("\nDetailed contents:")
        for item in path.rglob("*"):
            indent = "  " * len(item.relative_to(path).parts)
            try:
                if item.is_file():
                    size_mb = item.stat().st_size / 1024 / 1024
                    total_size += size_mb
                    file_count += 1
                    logger.info(f"{indent}📄 {item.name} ({size_mb:.2f}MB)")
                elif item.is_dir():
                    dir_count += 1
                    logger.info(f"{indent}📁 {item.name}/")
            except Exception as e:
                logger.error(f"Error accessing {item}: {str(e)}")

        logger.info(f"\nSummary for {description}:")
        logger.info(f"  Total files: {file_count}")
        logger.info(f"  Total directories: {dir_count}")
        logger.info(f"  Total size: {total_size:.2f}MB")

    except Exception as e:
        logger.error(
            f"Error listing directory contents for {path}: {str(e)}", exc_info=True
        )


def copy_file_robust(src: Path, dst: Path) -> bool:
    """Copy a file and log the operation, ensuring destination directory exists."""
    logger.info(f"\nAttempting to copy file:")
    logger.info(f"  From: {src}")
    logger.info(f"  To: {dst}")

    if not check_file_exists(src, "Source file for copy"):
        logger.warning("Source file does not exist or is not a file. Skipping copy.")
        return False

    try:
        ensure_directory(dst.parent)
        shutil.copy2(src, dst)
        if check_file_exists(dst, "Destination file after copy"):
            logger.info("File copied successfully")
            return True
        else:
            logger.error("Failed to verify copied file")
            return False
    except Exception as e:
        logger.error(f"Error copying file: {str(e)}", exc_info=True)
        return False


def copy_scripts(src_dir: Path, dst_dir: Path) -> None:
    """Recursively copy scripts from source to destination."""
    logger.info(f"\n{'='*20} Copying Scripts {'='*20}")
    logger.info(f"From: {src_dir}")
    logger.info(f"To: {dst_dir}")

    list_directory_contents(src_dir, "Source scripts directory")

    if not src_dir.exists() or not src_dir.is_dir():
        logger.warning(
            "Source scripts directory does not exist or is not a directory. Skipping script copy."
        )
        return

    ensure_directory(dst_dir)

    files_copied = 0
    total_size_mb = 0

    for item in src_dir.rglob("*"):
        if item.is_file():
            relative_path = item.relative_to(src_dir)
            destination_file = dst_dir / relative_path
            if copy_file_robust(item, destination_file):
                files_copied += 1
                total_size_mb += destination_file.stat().st_size / 1024 / 1024

    logger.info(f"\nScript copying summary:")
    logger.info(f"  Files copied: {files_copied}")
    logger.info(f"  Total size: {total_size_mb:.2f}MB")

    list_directory_contents(dst_dir, "Destination scripts directory")


def extract_tarfile(tar_path: Path, extract_path: Path) -> None:
    """Extract a tar file to the specified path."""
    logger.info(f"\n{'='*20} Extracting Tar File {'='*20}")

    if not check_file_exists(tar_path, "Tar file to extract"):
        logger.error("Cannot extract. Tar file does not exist.")
        return

    ensure_directory(extract_path)

    try:
        with tarfile.open(tar_path, "r:*") as tar:
            logger.info(f"\nTar file contents before extraction:")
            total_size = 0
            for member in tar.getmembers():
                size_mb = member.size / 1024 / 1024
                total_size += size_mb
                logger.info(f"  {member.name} ({size_mb:.2f}MB)")
            logger.info(f"Total size in tar: {total_size:.2f}MB")

            logger.info(f"\nExtracting to: {extract_path}")
            tar.extractall(path=extract_path)

        logger.info("\nExtraction completed. Verifying extracted contents:")
        list_directory_contents(extract_path, "Extracted contents")

    except Exception as e:
        logger.error(f"Error during tar extraction: {str(e)}", exc_info=True)


def create_tarfile(output_tar_path: Path, source_dir: Path) -> None:
    """Create a tar file from the contents of a directory."""
    logger.info(f"\n{'='*20} Creating Tar File {'='*20}")
    logger.info(f"Output tar: {output_tar_path}")
    logger.info(f"Source directory: {source_dir}")

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

        logger.info(f"\nTar creation summary:")
        logger.info(f"  Files added: {files_added}")
        logger.info(f"  Total uncompressed size: {total_size:.2f}MB")

        if check_file_exists(output_tar_path, "Created tar file"):
            compressed_size = output_tar_path.stat().st_size / 1024 / 1024
            logger.info(f"  Compressed tar size: {compressed_size:.2f}MB")
            logger.info(f"  Compression ratio: {compressed_size/total_size:.2%}")

    except Exception as e:
        logger.error(f"Error creating tar file: {str(e)}", exc_info=True)


def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: Optional[argparse.Namespace] = None,
) -> Path:
    """
    Main entry point for the packaging script.

    Args:
        input_paths: Dictionary of input paths with logical names
        output_paths: Dictionary of output paths with logical names
        environ_vars: Dictionary of environment variables
        job_args: Command line arguments (optional)

    Returns:
        Path to the packaged model.tar.gz output
    """
    # Extract paths from input parameters - required keys must be present
    if "model_input" not in input_paths:
        raise ValueError("Missing required input path: model_input")
    if "inference_scripts_input" not in input_paths:
        raise ValueError("Missing required input path: inference_scripts_input")
    if "packaged_model" not in output_paths:
        raise ValueError("Missing required output path: packaged_model")

    model_path = Path(input_paths["model_input"])
    script_path = Path(input_paths["inference_scripts_input"])
    output_path = Path(output_paths["packaged_model"])
    
    # Optional calibration model input
    calibration_path = None
    if "calibration_model" in input_paths:
        calibration_path = Path(input_paths["calibration_model"])
    working_directory = Path(
        environ_vars.get("WORKING_DIRECTORY", DEFAULT_WORKING_DIRECTORY)
    )
    code_directory = working_directory / "code"

    logger.info("\n=== Starting MIMS packaging process ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(
        f"Available disk space: {shutil.disk_usage('/').free / (1024*1024*1024):.2f}GB"
    )

    logger.info(f"\nUsing paths:")
    logger.info(f"  Model path: {model_path}")
    logger.info(f"  Script path: {script_path}")
    logger.info(f"  Output path: {output_path}")
    logger.info(f"  Working directory: {working_directory}")
    if calibration_path:
        logger.info(f"  Calibration path: {calibration_path}")
    else:
        logger.info("  Calibration path: Not provided (optional)")

    try:
        # Ensure working and output directories exist
        ensure_directory(working_directory)
        ensure_directory(output_path)

        # Extract input model.tar.gz if it exists
        input_model_tar = model_path / "model.tar.gz"
        logger.info("\nChecking for input model.tar.gz...")

        if check_file_exists(input_model_tar, "Input model.tar.gz"):
            extract_tarfile(input_model_tar, working_directory)
        else:
            logger.info("No model.tar.gz found. Copying all files from model_path...")
            files_copied = 0
            total_size = 0
            for item in model_path.rglob("*"):
                if item.is_file():
                    dest_path = working_directory / item.relative_to(model_path)
                    if copy_file_robust(item, dest_path):
                        files_copied += 1
                        total_size += item.stat().st_size / 1024 / 1024
            logger.info(
                f"\nCopied {files_copied} files, total size: {total_size:.2f}MB"
            )

        # Handle optional calibration model
        if calibration_path and calibration_path.exists():
            logger.info("\n=== Processing Calibration Model ===")
            
            # Create calibration subdirectory to match inference script expectations
            calibration_directory = working_directory / "calibration"
            ensure_directory(calibration_directory)
            
            # The calibration_path should contain the calibration artifacts from model_calibration script
            # This includes: calibration_model.pkl (binary) or calibration_models/ (multi-class)
            # and calibration_summary.json
            # Copy calibration artifacts to working_directory/calibration/ to match inference expectations
            logger.info("Copying calibration artifacts to calibration subdirectory...")
            files_copied = 0
            total_size = 0
            for item in calibration_path.rglob("*"):
                if item.is_file():
                    dest_path = calibration_directory / item.relative_to(calibration_path)
                    if copy_file_robust(item, dest_path):
                        files_copied += 1
                        total_size += item.stat().st_size / 1024 / 1024
            
            logger.info(f"Copied {files_copied} calibration files, total size: {total_size:.2f}MB")
            list_directory_contents(calibration_directory, "Calibration directory")
        else:
            logger.info("\n=== No Calibration Model Provided ===")
            logger.info("Skipping calibration model processing (optional)")

        # Copy inference scripts to working_directory/code
        copy_scripts(script_path, code_directory)

        # Create the output model.tar.gz
        output_tar_file = output_path / "model.tar.gz"
        create_tarfile(output_tar_file, working_directory)

        # Final verification and summary
        logger.info("\n=== Final State and Summary ===")
        list_directory_contents(working_directory, "Working directory final content")
        list_directory_contents(output_path, "Output directory final content")

        logger.info("\n=== MIMS packaging completed successfully ===")

        return output_tar_file

    except Exception as e:
        logger.error(f"Error in packaging process: {str(e)}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    try:
        # Standard SageMaker paths - using contract logical names
        input_paths = {
            "model_input": DEFAULT_MODEL_PATH,
            "inference_scripts_input": DEFAULT_SCRIPT_PATH,
            "calibration_model": DEFAULT_CALIBRATION_PATH,
        }

        output_paths = {"packaged_model": DEFAULT_OUTPUT_PATH}

        # Environment variables dictionary
        environ_vars = {"WORKING_DIRECTORY": DEFAULT_WORKING_DIRECTORY}

        # No command line arguments needed for this script
        args = None

        # Execute the main function
        result = main(input_paths, output_paths, environ_vars, args)

        logger.info(f"Packaging completed successfully. Output model at: {result}")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An unexpected error occurred during packaging: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

#!/usr/bin/env python
"""
MIMS Payload Generation Processing Script

This script reads field information from hyperparameters extracted from model.tar.gz,
extracts configuration from environment variables,
and creates payload files for model inference.
"""
import json
import logging
import os
import tarfile
import tempfile
import argparse
import sys
import traceback
from pathlib import Path
from enum import Enum
from typing import List, Dict, Any, Union, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for environment variable names
ENV_CONTENT_TYPES = "CONTENT_TYPES"
ENV_DEFAULT_NUMERIC_VALUE = "DEFAULT_NUMERIC_VALUE"
ENV_DEFAULT_TEXT_VALUE = "DEFAULT_TEXT_VALUE"
ENV_SPECIAL_FIELD_PREFIX = "SPECIAL_FIELD_"

# Default paths (will be overridden by parameters in main function)
DEFAULT_MODEL_DIR = "/opt/ml/processing/input/model"
DEFAULT_OUTPUT_DIR = "/opt/ml/processing/output"
DEFAULT_WORKING_DIRECTORY = "/tmp/mims_payload_work"


class VariableType(str, Enum):
    """Type of variable in model input/output"""

    NUMERIC = "NUMERIC"
    TEXT = "TEXT"


def ensure_directory(directory_path) -> bool:
    """Ensure a directory exists, creating it if necessary."""
    try:
        if isinstance(directory_path, str):
            directory_path = Path(directory_path)
        directory_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory ensured: {directory_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {directory_path}: {str(e)}")
        return False


def create_model_variable_list(
    full_field_list: List[str],
    tab_field_list: List[str],
    cat_field_list: List[str],
    label_name: str = "label",
    id_name: str = "id",
) -> List[List[str]]:
    """
    Creates a list of [variable_name, variable_type] pairs.

    Args:
        full_field_list: List of all field names
        tab_field_list: List of numeric/tabular field names
        cat_field_list: List of categorical field names
        label_name: Name of the label column (default: "label")
        id_name: Name of the ID column (default: "id")

    Returns:
        List[List[str]]: List of [variable_name, type] pairs where type is 'NUMERIC' or 'TEXT'
    """
    model_var_list = []

    for field in full_field_list:
        # Skip label and id fields
        if field in [label_name, id_name]:
            continue

        # Determine field type
        if field in tab_field_list:
            field_type = "NUMERIC"
        elif field in cat_field_list:
            field_type = "TEXT"
        else:
            # For any fields not explicitly categorized, default to TEXT
            field_type = "TEXT"

        # Add [field_name, field_type] pair
        model_var_list.append([field, field_type])

    return model_var_list


def extract_hyperparameters_from_tarball(
    input_model_dir: Path, working_directory: Path
) -> Dict:
    """Extract and load hyperparameters from model artifacts"""
    # The builder step has been updated to use the directory as destination, not model.tar.gz
    # But we'll keep the name for backward compatibility and handle both cases
    input_model_path = input_model_dir / "model.tar.gz"
    logger.info(f"Looking for hyperparameters in model artifacts")

    # Create temporary directory for extraction
    ensure_directory(working_directory)

    hyperparams_path = None

    # First check if model.tar.gz exists and is a file (original case)
    if input_model_path.exists() and input_model_path.is_file():
        logger.info(f"Found model.tar.gz file at {input_model_path}")
        try:
            # Extract just the hyperparameters.json file from tarball
            with tarfile.open(input_model_path, "r:gz") as tar:
                # Check if hyperparameters.json exists in the tarball
                hyperparams_info = None
                for member in tar.getmembers():
                    if member.name == "hyperparameters.json":
                        hyperparams_info = member
                        break

                if not hyperparams_info:
                    # List contents for debugging
                    contents = [m.name for m in tar.getmembers()]
                    logger.error(
                        f"hyperparameters.json not found in tarball. Contents: {contents}"
                    )
                    # Don't raise error here, continue checking other locations
                else:
                    # Extract only the hyperparameters file
                    tar.extract(hyperparams_info, working_directory)
                    hyperparams_path = working_directory / "hyperparameters.json"
        except Exception as e:
            logger.warning(f"Error processing model.tar.gz as tarfile: {e}")
            # Continue to other methods

    # Next check if model.tar.gz exists but is a directory (the error case we're fixing)
    if (
        hyperparams_path is None
        and input_model_path.exists()
        and input_model_path.is_dir()
    ):
        logger.info(
            f"{input_model_path} is a directory, looking for hyperparameters.json inside"
        )
        direct_hyperparams_path = input_model_path / "hyperparameters.json"
        if direct_hyperparams_path.exists():
            logger.info(
                f"Found hyperparameters.json directly in the model.tar.gz directory"
            )
            hyperparams_path = direct_hyperparams_path

    # Finally check if hyperparameters.json exists directly in the input model directory
    if hyperparams_path is None:
        logger.info(f"Looking for hyperparameters.json directly in {input_model_dir}")
        direct_hyperparams_path = input_model_dir / "hyperparameters.json"
        if direct_hyperparams_path.exists():
            logger.info(
                f"Found hyperparameters.json directly in the input model directory"
            )
            hyperparams_path = direct_hyperparams_path

    # If we still haven't found it, search recursively
    if hyperparams_path is None:
        logger.info(
            f"Searching recursively for hyperparameters.json in {input_model_dir}"
        )
        for path in input_model_dir.rglob("hyperparameters.json"):
            hyperparams_path = path
            logger.info(f"Found hyperparameters.json at {hyperparams_path}")
            break

    # If still not found, raise error
    if hyperparams_path is None:
        logger.error(f"hyperparameters.json not found in any location")
        # List directory contents for debugging
        contents = [str(f) for f in input_model_dir.rglob("*") if f.is_file()]
        logger.error(f"Directory contents: {contents}")
        raise FileNotFoundError("hyperparameters.json not found in model artifacts")

    # Load the hyperparameters
    with open(hyperparams_path, "r") as f:
        hyperparams = json.load(f)

    # Copy to working directory if not already there
    if not str(hyperparams_path).startswith(str(working_directory)):
        import shutil

        dest_path = working_directory / "hyperparameters.json"
        shutil.copy2(hyperparams_path, dest_path)

    logger.info(f"Successfully loaded hyperparameters: {list(hyperparams.keys())}")
    return hyperparams


def get_environment_content_types(environ_vars: Dict[str, str]) -> List[str]:
    """Get content types from environment variables."""
    content_types_str = environ_vars.get(ENV_CONTENT_TYPES, "application/json")
    return [ct.strip() for ct in content_types_str.split(",")]


def get_environment_default_numeric_value(environ_vars: Dict[str, str]) -> float:
    """Get default numeric value from environment variables."""
    try:
        return float(environ_vars.get(ENV_DEFAULT_NUMERIC_VALUE, "0.0"))
    except ValueError:
        logger.warning(f"Invalid {ENV_DEFAULT_NUMERIC_VALUE}, using default 0.0")
        return 0.0


def get_environment_default_text_value(environ_vars: Dict[str, str]) -> str:
    """Get default text value from environment variables."""
    return environ_vars.get(ENV_DEFAULT_TEXT_VALUE, "DEFAULT_TEXT")


def get_environment_special_fields(environ_vars: Dict[str, str]) -> Dict[str, str]:
    """Get special field values from environment variables."""
    special_fields = {}
    for env_var, env_value in environ_vars.items():
        if env_var.startswith(ENV_SPECIAL_FIELD_PREFIX):
            field_name = env_var[len(ENV_SPECIAL_FIELD_PREFIX) :].lower()
            special_fields[field_name] = env_value
    return special_fields


def get_field_default_value(
    field_name: str,
    var_type: str,
    default_numeric_value: float,
    default_text_value: str,
    special_field_values: Dict[str, str],
) -> str:
    """Get default value for a field"""
    if var_type == "TEXT" or var_type == VariableType.TEXT:
        if field_name in special_field_values:
            template = special_field_values[field_name]
            try:
                return template.format(
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
            except KeyError as e:
                raise ValueError(
                    f"Invalid placeholder in template for field '{field_name}': {str(e)}"
                )
        return default_text_value
    elif var_type == "NUMERIC" or var_type == VariableType.NUMERIC:
        return str(default_numeric_value)
    else:
        raise ValueError(f"Unknown variable type: {var_type}")


def generate_csv_payload(
    input_vars,
    default_numeric_value: float,
    default_text_value: str,
    special_field_values: Dict[str, str],
) -> str:
    """
    Generate CSV format payload following the order in input_vars.

    Returns:
        Comma-separated string of values
    """
    values = []

    if isinstance(input_vars, dict):
        # Dictionary format
        for field_name, var_type in input_vars.items():
            values.append(
                get_field_default_value(
                    field_name,
                    var_type,
                    default_numeric_value,
                    default_text_value,
                    special_field_values,
                )
            )
    else:
        # List format
        for field_name, var_type in input_vars:
            values.append(
                get_field_default_value(
                    field_name,
                    var_type,
                    default_numeric_value,
                    default_text_value,
                    special_field_values,
                )
            )

    return ",".join(values)


def generate_json_payload(
    input_vars,
    default_numeric_value: float,
    default_text_value: str,
    special_field_values: Dict[str, str],
) -> str:
    """
    Generate JSON format payload using input_vars.

    Returns:
        JSON string with field names and values
    """
    payload = {}

    if isinstance(input_vars, dict):
        # Dictionary format
        for field_name, var_type in input_vars.items():
            payload[field_name] = get_field_default_value(
                field_name,
                var_type,
                default_numeric_value,
                default_text_value,
                special_field_values,
            )
    else:
        # List format
        for field_name, var_type in input_vars:
            payload[field_name] = get_field_default_value(
                field_name,
                var_type,
                default_numeric_value,
                default_text_value,
                special_field_values,
            )

    return json.dumps(payload)


def generate_sample_payloads(
    input_vars,
    content_types: List[str],
    default_numeric_value: float,
    default_text_value: str,
    special_field_values: Dict[str, str],
) -> List[Dict[str, Union[str, dict]]]:
    """
    Generate sample payloads for each content type.

    Returns:
        List of dictionaries containing content type and payload
    """
    payloads = []

    for content_type in content_types:
        payload_info = {"content_type": content_type, "payload": None}

        if content_type == "text/csv":
            payload_info["payload"] = generate_csv_payload(
                input_vars,
                default_numeric_value,
                default_text_value,
                special_field_values,
            )
        elif content_type == "application/json":
            payload_info["payload"] = generate_json_payload(
                input_vars,
                default_numeric_value,
                default_text_value,
                special_field_values,
            )
        else:
            raise ValueError(f"Unsupported content type: {content_type}")

        payloads.append(payload_info)

    return payloads


def save_payloads(
    output_dir: str,
    input_vars,
    content_types: List[str],
    default_numeric_value: float,
    default_text_value: str,
    special_field_values: Dict[str, str],
) -> List[str]:
    """
    Save payloads to files.

    Args:
        output_dir: Directory to save payload files
        input_vars: Source model inference input variable list
        content_types: List of content types to generate payloads for
        default_numeric_value: Default value for numeric fields
        default_text_value: Default value for text fields
        special_field_values: Dictionary of special field values

    Returns:
        List of paths to created payload files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_paths = []
    payloads = generate_sample_payloads(
        input_vars,
        content_types,
        default_numeric_value,
        default_text_value,
        special_field_values,
    )

    logger.info("===== GENERATED PAYLOAD SAMPLES =====")

    for i, payload_info in enumerate(payloads):
        content_type = payload_info["content_type"]
        payload = payload_info["payload"]

        # Determine file extension and name
        ext = ".csv" if content_type == "text/csv" else ".json"
        file_name = f"payload_{content_type.replace('/', '_')}_{i}{ext}"
        file_path = output_dir / file_name

        # Log the payload content
        logger.info(f"Content Type: {content_type}")
        logger.info(f"Payload Sample: {payload}")
        logger.info("---------------------------------")

        # Save payload
        with open(file_path, "w") as f:
            f.write(payload)

        file_paths.append(str(file_path))
        logger.info(f"Created payload file: {file_path}")

    logger.info("===================================")

    return file_paths


def create_payload_archive(payload_files: List[str], output_dir: Path = None) -> str:
    """
    Create a tar.gz archive containing only payload files (not metadata).

    Args:
        payload_files: List of paths to payload files
        output_dir: Output directory path (defaults to DEFAULT_OUTPUT_DIR)

    Returns:
        Path to the created archive
    """
    # Create archive in the output directory
    output_dir = output_dir or Path(DEFAULT_OUTPUT_DIR)
    archive_path = output_dir / "payload.tar.gz"

    # Ensure parent directory exists (but not the actual archive path)
    ensure_directory(archive_path.parent)

    # Log archive creation
    logger.info(f"Creating payload archive at: {archive_path}")
    logger.info(f"Including {len(payload_files)} payload files")

    try:
        total_size = 0
        files_added = 0

        with tarfile.open(str(archive_path), "w:gz") as tar:
            for file_path in payload_files:
                # Add file to archive with basename as name
                file_name = os.path.basename(file_path)
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                total_size += size_mb
                files_added += 1
                logger.info(f"Adding to tar: {file_name} ({size_mb:.2f}MB)")
                tar.add(file_path, arcname=file_name)

        logger.info(f"Tar creation summary:")
        logger.info(f"  Files added: {files_added}")
        logger.info(f"  Total uncompressed size: {total_size:.2f}MB")

        # Verify archive was created
        if archive_path.exists() and archive_path.is_file():
            compressed_size = archive_path.stat().st_size / (1024 * 1024)
            logger.info(f"Successfully created payload archive: {archive_path}")
            logger.info(f"  Compressed tar size: {compressed_size:.2f}MB")
            logger.info(f"  Compression ratio: {compressed_size/total_size:.2%}")
        else:
            logger.error(
                f"Archive creation failed - file does not exist: {archive_path}"
            )

        return str(archive_path)

    except Exception as e:
        logger.error(f"Error creating payload archive: {str(e)}", exc_info=True)
        raise


def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: Optional[argparse.Namespace] = None,
) -> str:
    """
    Main entry point for the MIMS payload generation script.

    Args:
        input_paths: Dictionary of input paths with logical names
        output_paths: Dictionary of output paths with logical names
        environ_vars: Dictionary of environment variables
        job_args: Command line arguments (optional)

    Returns:
        Path to the generated payload archive file
    """
    try:
        # Extract paths from input parameters - required keys must be present
        if "model_input" not in input_paths:
            raise ValueError("Missing required input path: model_input")
        if "output_dir" not in output_paths:
            raise ValueError("Missing required output path: output_dir")

        # Set up paths
        model_dir = Path(input_paths["model_input"])
        output_dir = Path(output_paths["output_dir"])
        working_directory = Path(
            environ_vars.get("WORKING_DIRECTORY", DEFAULT_WORKING_DIRECTORY)
        )
        payload_sample_dir = working_directory / "payload_sample"

        logger.info(f"\nUsing paths:")
        logger.info(f"  Model input directory: {model_dir}")
        logger.info(f"  Output directory: {output_dir}")
        logger.info(f"  Working directory: {working_directory}")
        logger.info(f"  Payload sample directory: {payload_sample_dir}")

        # Extract hyperparameters from model tarball
        hyperparams = extract_hyperparameters_from_tarball(model_dir, working_directory)

        # Extract field information from hyperparameters
        full_field_list = hyperparams.get("full_field_list", [])
        tab_field_list = hyperparams.get("tab_field_list", [])
        cat_field_list = hyperparams.get("cat_field_list", [])
        label_name = hyperparams.get("label_name", "label")
        id_name = hyperparams.get("id_name", "id")

        # Create variable list
        adjusted_full_field_list = tab_field_list + cat_field_list
        var_type_list = create_model_variable_list(
            adjusted_full_field_list,
            tab_field_list,
            cat_field_list,
            label_name,
            id_name,
        )

        # Get parameters from environment variables
        content_types = get_environment_content_types(environ_vars)
        default_numeric_value = get_environment_default_numeric_value(environ_vars)
        default_text_value = get_environment_default_text_value(environ_vars)
        special_field_values = get_environment_special_fields(environ_vars)

        # Extract pipeline name and version from hyperparams
        pipeline_name = hyperparams.get("pipeline_name", "default_pipeline")
        pipeline_version = hyperparams.get("pipeline_version", "1.0.0")
        model_objective = hyperparams.get("model_objective", None)

        # Ensure working and output directories exist
        ensure_directory(working_directory)
        ensure_directory(output_dir)
        ensure_directory(payload_sample_dir)

        # Generate and save payloads to the sample directory
        payload_file_paths = save_payloads(
            payload_sample_dir,
            var_type_list,
            content_types,
            default_numeric_value,
            default_text_value,
            special_field_values,
        )

        # Create tar.gz archive of only payload files (not metadata)
        archive_path = create_payload_archive(payload_file_paths, output_dir)

        # Log summary information about the payload generation
        logger.info(f"MIMS payload generation complete.")
        logger.info(f"Number of payload samples generated: {len(payload_file_paths)}")
        logger.info(f"Content types: {content_types}")
        logger.info(f"Payload files saved to: {payload_sample_dir}")
        logger.info(f"Payload archive saved to: {archive_path}")

        # Print information about input fields for better debugging
        logger.info(f"Input field information:")
        logger.info(f"  Total fields: {len(var_type_list)}")
        for field_name, field_type in var_type_list:
            logger.info(f"  - {field_name}: {field_type}")

        return archive_path

    except Exception as e:
        logger.error(f"Error in payload generation: {str(e)}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    try:
        # Standard SageMaker paths
        input_paths = {"model_input": DEFAULT_MODEL_DIR}

        output_paths = {"output_dir": DEFAULT_OUTPUT_DIR}

        # Environment variables dictionary
        environ_vars = {}
        for env_var in [
            ENV_CONTENT_TYPES,
            ENV_DEFAULT_NUMERIC_VALUE,
            ENV_DEFAULT_TEXT_VALUE,
        ]:
            if env_var in os.environ:
                environ_vars[env_var] = os.environ[env_var]

        # Also add special field variables
        for env_var, env_value in os.environ.items():
            if env_var.startswith(ENV_SPECIAL_FIELD_PREFIX):
                environ_vars[env_var] = env_value

        # Set working directory
        environ_vars["WORKING_DIRECTORY"] = DEFAULT_WORKING_DIRECTORY

        # No command line arguments needed for this script
        args = None

        # Execute the main function
        result = main(input_paths, output_paths, environ_vars, args)

        logger.info(f"Payload generation completed successfully. Output at: {result}")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error in payload generation script: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

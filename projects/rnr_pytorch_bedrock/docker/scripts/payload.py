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
import shutil
import tarfile
import tempfile
import argparse
import sys
import traceback
from pathlib import Path
from enum import Enum
from typing import List, Tuple, Dict, Any, Union, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for environment variable names
ENV_CONTENT_TYPES = "CONTENT_TYPES"
ENV_DEFAULT_NUMERIC_VALUE = "DEFAULT_NUMERIC_VALUE"
ENV_DEFAULT_TEXT_VALUE = "DEFAULT_TEXT_VALUE"
ENV_SPECIAL_FIELD_PREFIX = "SPECIAL_FIELD_"
ENV_FIELD_DEFAULTS = "FIELD_DEFAULTS"  # NEW: Unified field defaults

# Default paths (will be overridden by parameters in main function)
DEFAULT_MODEL_DIR = "/opt/ml/processing/input/model"
DEFAULT_CUSTOM_PAYLOAD_DIR = "/opt/ml/processing/input/custom_payload"
DEFAULT_OUTPUT_DIR = "/opt/ml/processing/output"
DEFAULT_WORKING_DIRECTORY = "/tmp/mims_payload_work"


class VariableType(str, Enum):
    """Type of variable in model input/output"""

    NUMERIC = "NUMERIC"
    TEXT = "TEXT"


# ===== Phase 3: Multi-Modal Support Functions =====


def detect_model_type(hyperparams: Dict) -> str:
    """
    Detect model type from hyperparameters.

    Detection logic:
    1. Check for trimodal indicators (primary_text_name + secondary_text_name)
    2. Check for bimodal indicators (text_name field)
    3. Default to tabular (traditional XGBoost/LightGBM)

    Args:
        hyperparams: Dictionary loaded from hyperparameters.json

    Returns:
        'trimodal', 'bimodal', or 'tabular'
    """
    model_class = hyperparams.get("model_class", "").lower()

    # Check for trimodal
    if "trimodal" in model_class or (
        "primary_text_name" in hyperparams and "secondary_text_name" in hyperparams
    ):
        logger.info("Detected trimodal model (dual text + tabular)")
        return "trimodal"

    # Check for bimodal
    if "multimodal" in model_class or "text_name" in hyperparams:
        logger.info("Detected bimodal model (text + tabular)")
        return "bimodal"

    # Default to tabular
    logger.info("Detected tabular model")
    return "tabular"


def get_field_defaults(environ_vars: Dict[str, str]) -> Dict[str, str]:
    """
    Load field default values from environment.

    Priority (highest to lowest):
    1. SPECIAL_FIELD_* prefix (per-field overrides, highest priority for backward compatibility)
    2. FIELD_DEFAULTS (JSON dict, base defaults)
    3. Empty dict (use auto-generated intelligent defaults)

    Args:
        environ_vars: Environment variables dictionary

    Returns:
        Dictionary mapping field names to default values
    """
    field_defaults = {}

    # First: Load from JSON dictionary (base defaults)
    if ENV_FIELD_DEFAULTS in environ_vars:
        try:
            field_defaults = json.loads(environ_vars[ENV_FIELD_DEFAULTS])
            logger.info(
                f"Loaded {len(field_defaults)} field defaults from {ENV_FIELD_DEFAULTS}"
            )
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse {ENV_FIELD_DEFAULTS}: {e}")

    # Second: Load from SPECIAL_FIELD_ prefix (overrides JSON for backward compatibility)
    for env_var, env_value in environ_vars.items():
        if env_var.startswith(ENV_SPECIAL_FIELD_PREFIX):
            field_name = env_var[len(ENV_SPECIAL_FIELD_PREFIX) :].lower()
            field_defaults[field_name] = env_value
            logger.debug(f"Added SPECIAL_FIELD override for '{field_name}'")

    return field_defaults


def generate_text_sample(
    field_name: str,
    field_defaults: Dict[str, str],
    default_text_value: str = "Sample text for inference testing",
) -> str:
    """
    Generate sample text for a text field with 3-tier priority.

    Priority (highest to lowest):
    1. User-provided value from field_defaults (exact or case-insensitive match)
    2. Intelligent default based on field name pattern
    3. Generic default from DEFAULT_TEXT_VALUE

    Args:
        field_name: Name of the text field
        field_defaults: User-provided field defaults dictionary
        default_text_value: Generic fallback default

    Returns:
        Sample text string for the field
    """
    # Priority 1: User-provided (exact match)
    if field_name in field_defaults:
        value = field_defaults[field_name]
        # Support template expansion (e.g., {timestamp})
        try:
            return value.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        except (KeyError, ValueError):
            return value

    # Case-insensitive fallback
    field_lower = field_name.lower()
    for key, value in field_defaults.items():
        if key.lower() == field_lower:
            try:
                return value.format(
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
            except (KeyError, ValueError):
                return value

    # Priority 2: Intelligent defaults based on field name
    if (
        "chat" in field_lower
        or "dialogue" in field_lower
        or "conversation" in field_lower
    ):
        return "Hello, I need help with my order. Can you assist me?"
    elif (
        "shiptrack" in field_lower
        or "event" in field_lower
        or "tracking" in field_lower
    ):
        return "Package shipped|In transit|Delivered"
    elif "description" in field_lower or "desc" in field_lower:
        return "Product description text for testing purposes"
    elif "comment" in field_lower or "note" in field_lower:
        return "Additional notes and comments for testing"
    elif "title" in field_lower or "subject" in field_lower:
        return "Sample title for testing"
    elif "message" in field_lower or "msg" in field_lower:
        return "Sample message content for testing"

    # Priority 3: Generic default
    return default_text_value


def load_custom_payload(
    custom_path: Path, content_type: str = "application/json"
) -> Optional[Dict]:
    """
    Load user-provided custom payload sample.

    Supports:
    - JSON file: Load and return as dict
    - CSV file: Load first row as dict
    - Parquet file: Load first row as dict
    - Directory: Search for JSON/CSV/Parquet files

    Args:
        custom_path: Path to custom payload file or directory
        content_type: Expected content type ('application/json' or 'text/csv')

    Returns:
        Dictionary with payload data if successful, None otherwise
    """
    if not custom_path.exists():
        logger.warning(f"Custom payload path not found: {custom_path}")
        return None

    try:
        # Handle directory: search for sample files
        if custom_path.is_dir():
            logger.info(f"Searching for payload samples in directory: {custom_path}")

            # Look for JSON files first (highest priority)
            json_files = list(custom_path.glob("*.json"))
            if json_files:
                logger.info(
                    f"Found {len(json_files)} JSON files, using first: {json_files[0]}"
                )
                with open(json_files[0], "r") as f:
                    return json.load(f)

            # Look for CSV files (second priority)
            csv_files = list(custom_path.glob("*.csv"))
            if csv_files:
                logger.info(
                    f"Found {len(csv_files)} CSV files, using first: {csv_files[0]}"
                )
                # Import pandas only when needed for CSV loading
                try:
                    import pandas as pd

                    df = pd.read_csv(csv_files[0])
                    if len(df) > 0:
                        return df.iloc[0].to_dict()
                    else:
                        logger.warning("CSV file is empty")
                        return None
                except ImportError:
                    logger.error("pandas is required for CSV loading but not available")
                    return None

            # Look for Parquet files (third priority)
            parquet_files = list(custom_path.glob("*.parquet"))
            if parquet_files:
                logger.info(
                    f"Found {len(parquet_files)} Parquet files, using first: {parquet_files[0]}"
                )
                # Import pandas only when needed for Parquet loading
                try:
                    import pandas as pd

                    df = pd.read_parquet(parquet_files[0])
                    if len(df) > 0:
                        return df.iloc[0].to_dict()
                    else:
                        logger.warning("Parquet file is empty")
                        return None
                except ImportError:
                    logger.error(
                        "pandas is required for Parquet loading but not available"
                    )
                    return None

            logger.warning("No JSON, CSV, or Parquet files found in directory")
            return None

        # Handle file: load based on extension
        elif custom_path.is_file():
            logger.info(f"Loading custom payload from file: {custom_path}")

            if custom_path.suffix == ".json":
                with open(custom_path, "r") as f:
                    payload = json.load(f)
                    logger.info(f"Loaded JSON payload with {len(payload)} fields")
                    return payload

            elif custom_path.suffix == ".csv":
                # Import pandas only when needed for CSV loading
                try:
                    import pandas as pd

                    df = pd.read_csv(custom_path)
                    if len(df) > 0:
                        payload = df.iloc[0].to_dict()
                        logger.info(f"Loaded CSV payload with {len(payload)} fields")
                        return payload
                    else:
                        logger.warning("CSV file is empty")
                        return None
                except ImportError:
                    logger.error("pandas is required for CSV loading but not available")
                    return None

            elif custom_path.suffix == ".parquet":
                # Import pandas only when needed for Parquet loading
                try:
                    import pandas as pd

                    df = pd.read_parquet(custom_path)
                    if len(df) > 0:
                        payload = df.iloc[0].to_dict()
                        logger.info(
                            f"Loaded Parquet payload with {len(payload)} fields"
                        )
                        return payload
                    else:
                        logger.warning("Parquet file is empty")
                        return None
                except ImportError:
                    logger.error(
                        "pandas is required for Parquet loading but not available"
                    )
                    return None

            else:
                logger.warning(f"Unsupported file extension: {custom_path.suffix}")
                return None

    except Exception as e:
        logger.error(f"Failed to load custom payload: {e}", exc_info=True)
        return None

    return None


# ===== End Phase 3 Functions =====


def get_required_fields_from_model(
    model_dir: Path, hyperparams: Dict, var_type_list: List[List[str]]
) -> Dict[str, Any]:
    """
    Get required fields using the SAME logic as inference handlers.

    This ensures validation matches what the actual inference handler expects.

    Priority:
    1. feature_columns.txt (if exists) - for XGBoost/LightGBM models
    2. hyperparameters.json - for PyTorch models or fallback

    Args:
        model_dir: Directory containing model artifacts
        hyperparams: Model hyperparameters from hyperparameters.json
        var_type_list: List of [field_name, field_type] pairs

    Returns:
        Dictionary with:
            - tabular_fields: List of required tabular feature names
            - id_field: Optional ID field name
            - text_fields: Dict of text field names by type
            - model_type: 'tabular', 'bimodal', or 'trimodal'
            - field_order: Ordered list of all fields (for CSV)
            - source: 'feature_columns.txt' or 'hyperparameters.json'
    """
    required = {
        "tabular_fields": [],
        "id_field": None,
        "text_fields": {},
        "model_type": "tabular",
        "field_order": [],
        "source": None,
    }

    # Try to load feature_columns.txt (XGBoost/LightGBM)
    feature_columns_file = model_dir / "feature_columns.txt"

    # If not found directly, try to extract from model.tar.gz
    if not feature_columns_file.exists():
        logger.debug("feature_columns.txt not found directly, checking model.tar.gz")
        model_tarball = model_dir / "model.tar.gz"

        if model_tarball.exists() and model_tarball.is_file():
            logger.info("Attempting to extract feature_columns.txt from model.tar.gz")
            try:
                with tarfile.open(model_tarball, "r:gz") as tar:
                    # Look for feature_columns.txt in the tarball
                    for member in tar.getmembers():
                        if member.name == "feature_columns.txt" or member.name.endswith(
                            "/feature_columns.txt"
                        ):
                            # Extract to model_dir
                            tar.extract(member, model_dir)
                            # Handle case where file is in a subdirectory in the tarball
                            if "/" in member.name:
                                extracted_path = model_dir / member.name
                                # Move to root of model_dir if needed
                                if extracted_path != feature_columns_file:
                                    shutil.move(
                                        str(extracted_path), str(feature_columns_file)
                                    )
                                    # Clean up empty subdirectory if created
                                    try:
                                        extracted_path.parent.rmdir()
                                    except:
                                        pass
                            logger.info(
                                f"Successfully extracted feature_columns.txt from tarball"
                            )
                            break
            except Exception as e:
                logger.warning(
                    f"Failed to extract feature_columns.txt from tarball: {e}"
                )

    # Now check if we have feature_columns.txt
    if feature_columns_file.exists():
        logger.info(
            "Using feature_columns.txt as source of truth (XGBoost/LightGBM model)"
        )
        required["source"] = "feature_columns.txt"

        # Read ordered features from feature_columns.txt
        with open(feature_columns_file, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                try:
                    idx, column = line.strip().split(",")
                    required["tabular_fields"].append(column)
                except ValueError:
                    continue

        required["field_order"] = required["tabular_fields"].copy()
        logger.info(
            f"Loaded {len(required['tabular_fields'])} features from feature_columns.txt"
        )

    else:
        # Use hyperparameters (PyTorch models)
        logger.info("Using hyperparameters.json as source of truth (PyTorch model)")
        required["source"] = "hyperparameters.json"

        # Detect model type
        model_type = detect_model_type(hyperparams)
        required["model_type"] = model_type

        # Build field order: ID -> text fields -> tabular fields
        field_order = []

        # ID field
        id_name = hyperparams.get("id_name")
        if id_name:
            required["id_field"] = id_name
            field_order.append(id_name)

        # Text fields based on model type
        if model_type == "bimodal":
            text_name = hyperparams.get("text_name")
            if text_name:
                required["text_fields"]["text_name"] = text_name
                field_order.append(text_name)

        elif model_type == "trimodal":
            primary_text = hyperparams.get("primary_text_name")
            secondary_text = hyperparams.get("secondary_text_name")
            if primary_text:
                required["text_fields"]["primary_text_name"] = primary_text
                field_order.append(primary_text)
            if secondary_text:
                required["text_fields"]["secondary_text_name"] = secondary_text
                field_order.append(secondary_text)

        # Tabular fields from var_type_list
        for field_name, _ in var_type_list:
            required["tabular_fields"].append(field_name)
            field_order.append(field_name)

        required["field_order"] = field_order

    return required


def validate_payload_completeness(
    payload: Dict,
    hyperparams: Dict,
    var_type_list: List[List[str]],
    model_dir: Optional[Path] = None,
) -> Tuple[bool, List[str]]:
    """
    Validate payload contains all required fields for any model type.

    Uses the SAME source of truth as inference handlers:
    - feature_columns.txt for XGBoost/LightGBM
    - hyperparameters.json for PyTorch

    Args:
        payload: Generated payload dictionary
        hyperparams: Model hyperparameters
        var_type_list: List of [field_name, field_type] pairs
        model_dir: Optional model directory to check for feature_columns.txt

    Returns:
        (is_valid, missing_fields)
    """
    required_fields = set()

    # If model_dir provided, use same logic as inference handlers
    if model_dir and model_dir.exists():
        required = get_required_fields_from_model(model_dir, hyperparams, var_type_list)

        # Add all required fields
        if required["id_field"]:
            required_fields.add(required["id_field"])

        for text_field in required["text_fields"].values():
            required_fields.add(text_field)

        for tabular_field in required["tabular_fields"]:
            required_fields.add(tabular_field)

    else:
        # Fallback to hyperparameters-only validation (backward compatibility)
        model_type = detect_model_type(hyperparams)

        # ID field (optional but should be present if in hyperparams)
        id_name = hyperparams.get("id_name")
        if id_name:
            required_fields.add(id_name)

        # Text fields based on model type
        if model_type == "bimodal":
            text_name = hyperparams.get("text_name")
            if text_name:
                required_fields.add(text_name)

        elif model_type == "trimodal":
            primary_text_name = hyperparams.get("primary_text_name")
            secondary_text_name = hyperparams.get("secondary_text_name")

            if primary_text_name:
                required_fields.add(primary_text_name)
            if secondary_text_name:
                required_fields.add(secondary_text_name)

        # Tabular fields (all model types)
        for field_name, _ in var_type_list:
            required_fields.add(field_name)

    # Validate completeness
    payload_fields = set(payload.keys())
    missing = required_fields - payload_fields
    extra = payload_fields - required_fields

    if missing:
        logger.warning(f"Missing required fields: {missing}")
    if extra:
        logger.info(f"Extra fields in payload: {extra}")

    return (len(missing) == 0, list(missing))


def log_payload_field_mapping(
    payload: Dict, hyperparams: Dict, var_type_list: List[List[str]]
) -> None:
    """
    Log comprehensive field mapping for payload validation and debugging.

    Args:
        payload: Generated payload dictionary
        hyperparams: Model hyperparameters
        var_type_list: List of [field_name, field_type] pairs
    """
    logger.info("=== PAYLOAD FIELD MAPPING ===")

    # Detect model type
    model_type = detect_model_type(hyperparams)
    logger.info(f"Model type: {model_type}")

    # ID field (common to all types)
    id_name = hyperparams.get("id_name")
    if id_name:
        logger.info(f"  ID field: {id_name} = {payload.get(id_name)}")

    # Text fields - handle based on model type
    if model_type == "tabular":
        logger.info("  No text fields (tabular-only model)")

    elif model_type == "bimodal":
        text_name = hyperparams.get("text_name")
        if text_name:
            text_value = str(payload.get(text_name, ""))
            # Truncate long text for logging
            text_preview = (
                text_value[:50] + "..." if len(text_value) > 50 else text_value
            )
            logger.info(f"  Text field: {text_name} = {text_preview}")

    elif model_type == "trimodal":
        primary_text_name = hyperparams.get("primary_text_name")
        secondary_text_name = hyperparams.get("secondary_text_name")

        if primary_text_name:
            primary_value = str(payload.get(primary_text_name, ""))
            primary_preview = (
                primary_value[:50] + "..." if len(primary_value) > 50 else primary_value
            )
            logger.info(
                f"  Primary text field: {primary_text_name} = {primary_preview}"
            )

        if secondary_text_name:
            secondary_value = str(payload.get(secondary_text_name, ""))
            secondary_preview = (
                secondary_value[:50] + "..."
                if len(secondary_value) > 50
                else secondary_value
            )
            logger.info(
                f"  Secondary text field: {secondary_text_name} = {secondary_preview}"
            )

    # Tabular fields (common to all types)
    logger.info(f"  Tabular fields: {len(var_type_list)} fields")
    for field_name, field_type in var_type_list:
        field_value = payload.get(field_name, "MISSING")
        logger.info(f"    {field_name} ({field_type}) = {field_value}")

    logger.info("=" * 40)


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


def generate_csv_payload(
    input_vars,
    default_numeric_value: float,
    default_text_value: str,
    hyperparams: Optional[Dict] = None,
    field_defaults: Optional[Dict[str, str]] = None,
    model_dir: Optional[Path] = None,
) -> str:
    """
    Generate CSV format payload with multi-modal support.

    CRITICAL: For XGBoost/LightGBM models, field order MUST match feature_columns.txt
    for inference to work correctly. This function uses get_required_fields_from_model()
    to ensure correct ordering.

    Handles:
    - Tabular: Only numeric/categorical fields
    - Bimodal: text_name + numeric/categorical fields
    - Trimodal: primary_text_name + secondary_text_name + numeric/categorical fields

    Args:
        input_vars: List of [field_name, var_type] pairs for tabular features
        default_numeric_value: Default for numeric fields
        default_text_value: Generic default for text fields
        hyperparams: Full hyperparameters dict from model (for multi-modal detection)
        field_defaults: User-provided field defaults dictionary
        model_dir: Model directory to check for feature_columns.txt (CRITICAL for correct ordering)

    Returns:
        Comma-separated string of values (no header) in CORRECT field order
    """
    # Use field_defaults directly (already includes SPECIAL_FIELD_* for backward compat)
    field_defaults = field_defaults or {}

    # Build field name -> type mapping for quick lookup
    field_type_map = {}
    if isinstance(input_vars, dict):
        field_type_map = input_vars
    else:
        field_type_map = {name: vtype for name, vtype in input_vars}

    # Get correct field order from model (critical for XGBoost/LightGBM)
    if model_dir and hyperparams:
        required_info = get_required_fields_from_model(
            model_dir,
            hyperparams,
            list(field_type_map.items())
            if not isinstance(input_vars, dict)
            else [(k, v) for k, v in input_vars.items()],
        )
        field_order = required_info["field_order"]

        # Generate values in CORRECT order
        values = []
        for field_name in field_order:
            # Determine field type
            if field_name in field_type_map:
                var_type = field_type_map[field_name]
                if var_type in ["TEXT", VariableType.TEXT]:
                    values.append(
                        generate_text_sample(
                            field_name, field_defaults, default_text_value
                        )
                    )
                else:
                    values.append(str(default_numeric_value))
            else:
                # This is a text field (ID, text_name, primary_text_name, secondary_text_name)
                # Use intelligent default based on field name
                if field_name == hyperparams.get("id_name"):
                    values.append(
                        generate_text_sample(
                            field_name,
                            field_defaults,
                            f"TEST_ID_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        )
                    )
                else:
                    values.append(
                        generate_text_sample(
                            field_name, field_defaults, default_text_value
                        )
                    )

        return ",".join(values)

    # Fallback to old logic if model_dir not provided (backward compatibility)
    values = []

    # Add multi-modal text fields if hyperparams provided
    if hyperparams:
        model_type = detect_model_type(hyperparams)

        # Add ID field if present
        id_name = hyperparams.get("id_name")
        if id_name:
            values.append(
                generate_text_sample(
                    id_name,
                    field_defaults,
                    f"TEST_ID_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                )
            )

        # Add text fields based on model type
        if model_type == "bimodal":
            text_name = hyperparams.get("text_name")
            if text_name:
                values.append(
                    generate_text_sample(text_name, field_defaults, default_text_value)
                )

        elif model_type == "trimodal":
            primary_text_name = hyperparams.get("primary_text_name")
            secondary_text_name = hyperparams.get("secondary_text_name")

            if primary_text_name:
                values.append(
                    generate_text_sample(
                        primary_text_name, field_defaults, default_text_value
                    )
                )

            if secondary_text_name:
                values.append(
                    generate_text_sample(
                        secondary_text_name, field_defaults, default_text_value
                    )
                )

    # Add tabular fields
    if isinstance(input_vars, dict):
        # Dictionary format
        for field_name, var_type in input_vars.items():
            if var_type in ["TEXT", VariableType.TEXT]:
                values.append(
                    generate_text_sample(field_name, field_defaults, default_text_value)
                )
            else:
                values.append(str(default_numeric_value))
    else:
        # List format
        for field_name, var_type in input_vars:
            if var_type in ["TEXT", VariableType.TEXT]:
                values.append(
                    generate_text_sample(field_name, field_defaults, default_text_value)
                )
            else:
                values.append(str(default_numeric_value))

    return ",".join(values)


def generate_json_payload(
    input_vars,
    default_numeric_value: float,
    default_text_value: str,
    hyperparams: Optional[Dict] = None,
    field_defaults: Optional[Dict[str, str]] = None,
) -> str:
    """
    Generate JSON format payload with multi-modal support.

    Handles:
    - Tabular: Only numeric/categorical fields
    - Bimodal: text_name + numeric/categorical fields
    - Trimodal: primary_text_name + secondary_text_name + numeric/categorical fields

    Args:
        input_vars: List of [field_name, var_type] pairs for tabular features
        default_numeric_value: Default for numeric fields
        default_text_value: Generic default for text fields
        hyperparams: Full hyperparameters dict from model (for multi-modal detection)
        field_defaults: User-provided field defaults dictionary

    Returns:
        JSON string with complete payload
    """
    payload = {}

    # Use field_defaults directly (already includes SPECIAL_FIELD_* for backward compat)
    field_defaults = field_defaults or {}

    # Add multi-modal text fields if hyperparams provided
    if hyperparams:
        model_type = detect_model_type(hyperparams)

        # Add ID field if present
        id_name = hyperparams.get("id_name")
        if id_name:
            payload[id_name] = generate_text_sample(
                id_name,
                field_defaults,
                f"TEST_ID_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            )

        # Add text fields based on model type
        if model_type == "bimodal":
            text_name = hyperparams.get("text_name")
            if text_name:
                payload[text_name] = generate_text_sample(
                    text_name, field_defaults, default_text_value
                )
                logger.info(f"Added bimodal text field: {text_name}")

        elif model_type == "trimodal":
            primary_text_name = hyperparams.get("primary_text_name")
            secondary_text_name = hyperparams.get("secondary_text_name")

            if primary_text_name:
                payload[primary_text_name] = generate_text_sample(
                    primary_text_name, field_defaults, default_text_value
                )
                logger.info(f"Added primary text field: {primary_text_name}")

            if secondary_text_name:
                payload[secondary_text_name] = generate_text_sample(
                    secondary_text_name, field_defaults, default_text_value
                )
                logger.info(f"Added secondary text field: {secondary_text_name}")

    # Add tabular fields
    if isinstance(input_vars, dict):
        # Dictionary format
        for field_name, var_type in input_vars.items():
            if var_type in ["TEXT", VariableType.TEXT]:
                # For categorical TEXT fields, use generate_text_sample
                payload[field_name] = generate_text_sample(
                    field_name, field_defaults, default_text_value
                )
            else:
                payload[field_name] = str(default_numeric_value)
    else:
        # List format
        for field_name, var_type in input_vars:
            if var_type in ["TEXT", VariableType.TEXT]:
                # For categorical TEXT fields, use generate_text_sample
                payload[field_name] = generate_text_sample(
                    field_name, field_defaults, default_text_value
                )
            else:
                payload[field_name] = str(default_numeric_value)

    return json.dumps(payload)


def generate_sample_payloads(
    input_vars,
    content_types: List[str],
    default_numeric_value: float,
    default_text_value: str,
    hyperparams: Optional[Dict] = None,
    field_defaults: Optional[Dict[str, str]] = None,
    model_dir: Optional[Path] = None,
) -> List[Dict[str, Union[str, dict]]]:
    """
    Generate sample payloads for each content type with multi-modal support.

    Args:
        input_vars: List of [field_name, var_type] pairs for tabular features
        content_types: List of content types to generate
        default_numeric_value: Default for numeric fields
        default_text_value: Generic default for text fields
        hyperparams: Full hyperparameters dict (for multi-modal detection)
        field_defaults: User-provided field defaults dictionary
        model_dir: Model directory for correct CSV field ordering (CRITICAL for XGBoost/LightGBM)

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
                hyperparams,
                field_defaults,
                model_dir,  # CRITICAL: Pass model_dir for correct field ordering
            )
        elif content_type == "application/json":
            payload_info["payload"] = generate_json_payload(
                input_vars,
                default_numeric_value,
                default_text_value,
                hyperparams,
                field_defaults,
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
    hyperparams: Optional[Dict] = None,
    field_defaults: Optional[Dict[str, str]] = None,
    model_dir: Optional[Path] = None,
) -> List[str]:
    """
    Save payloads to files with multi-modal support.

    Args:
        output_dir: Directory to save payload files
        input_vars: Source model inference input variable list
        content_types: List of content types to generate payloads for
        default_numeric_value: Default value for numeric fields
        default_text_value: Default value for text fields
        hyperparams: Full hyperparameters dict (for multi-modal detection)
        field_defaults: User-provided field defaults dictionary
        model_dir: Model directory for correct CSV field ordering (CRITICAL for XGBoost/LightGBM)

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
        hyperparams,
        field_defaults,
        model_dir,  # CRITICAL: Pass model_dir for correct field ordering
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
            logger.info(f"  Compression ratio: {compressed_size / total_size:.2%}")
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

        # Load field defaults (unified approach, includes SPECIAL_FIELD_* for backward compat)
        field_defaults = get_field_defaults(environ_vars)
        logger.info(f"Loaded {len(field_defaults)} field defaults")

        # Ensure working and output directories exist
        ensure_directory(working_directory)
        ensure_directory(output_dir)
        ensure_directory(payload_sample_dir)

        # NEW: Check for custom payload input (optional)
        custom_payload_input_path = Path(
            input_paths.get(
                "custom_payload_input", "/opt/ml/processing/input/custom_payload"
            )
        )
        custom_payload = None

        if custom_payload_input_path.exists():
            logger.info(f"Found custom payload input at: {custom_payload_input_path}")
            custom_payload = load_custom_payload(
                custom_payload_input_path,
                content_types[0] if content_types else "application/json",
            )

        # Generate payloads
        if custom_payload:
            # Use custom payload directly
            logger.info("Using user-provided custom payload sample")

            # Validate custom payload has all required fields
            is_valid, missing_fields = validate_payload_completeness(
                custom_payload, hyperparams, var_type_list, model_dir
            )

            if not is_valid:
                model_type = detect_model_type(hyperparams)
                error_msg = (
                    f"Custom payload validation FAILED!\n"
                    f"Missing required fields: {missing_fields}\n"
                    f"Model type: {model_type}\n"
                    f"Please ensure your custom payload includes ALL required fields for inference."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.info(
                "✓ Custom payload validation PASSED - all required fields present"
            )

            # Log field mapping for debugging
            log_payload_field_mapping(custom_payload, hyperparams, var_type_list)

            # Get correct field order for CSV generation (critical for XGBoost/LightGBM)
            required_info = get_required_fields_from_model(
                model_dir, hyperparams, var_type_list
            )
            field_order = required_info["field_order"]

            # Save custom payload to files for each content type
            payload_file_paths = []
            for i, content_type in enumerate(content_types):
                ext = ".csv" if content_type == "text/csv" else ".json"
                file_name = f"payload_{content_type.replace('/', '_')}_{i}{ext}"
                file_path = payload_sample_dir / file_name

                if content_type == "application/json":
                    with open(file_path, "w") as f:
                        json.dump(custom_payload, f)
                else:
                    # For CSV, use correct field order from model (critical for XGBoost/LightGBM)
                    # Extract values in the order defined by feature_columns.txt or hyperparameters
                    ordered_values = []
                    for field in field_order:
                        if field in custom_payload:
                            ordered_values.append(str(custom_payload[field]))
                        else:
                            logger.warning(
                                f"Field '{field}' missing in custom payload for CSV generation"
                            )
                            ordered_values.append(
                                ""
                            )  # Use empty string for missing fields

                    with open(file_path, "w") as f:
                        f.write(",".join(ordered_values))

                payload_file_paths.append(str(file_path))
                logger.info(f"Saved custom payload to: {file_path}")
        else:
            # Generate from hyperparameters with multi-modal support
            model_type = detect_model_type(hyperparams)
            logger.info(f"Generating payload for {model_type} model")

            # Generate and save payloads to the sample directory (with multi-modal support)
            payload_file_paths = save_payloads(
                payload_sample_dir,
                var_type_list,
                content_types,
                default_numeric_value,
                default_text_value,
                hyperparams,  # Pass hyperparams for multi-modal detection
                field_defaults,  # Pass field defaults (includes SPECIAL_FIELD_* for backward compat)
                model_dir,  # CRITICAL: Pass model_dir for correct CSV field ordering
            )

            # Validate and log the first generated payload for verification
            if payload_file_paths and content_types:
                # Load the first JSON payload for validation
                first_json_file = None
                for file_path in payload_file_paths:
                    if file_path.endswith(".json"):
                        first_json_file = file_path
                        break

                if first_json_file:
                    with open(first_json_file, "r") as f:
                        first_payload = json.load(f)

                    # Validate completeness with model_dir to use feature_columns.txt if available
                    is_valid, missing_fields = validate_payload_completeness(
                        first_payload, hyperparams, var_type_list, model_dir
                    )

                    if is_valid:
                        logger.info(
                            "✓ Payload validation PASSED - all required fields present"
                        )
                    else:
                        logger.warning(
                            f"✗ Payload validation WARNING - missing fields: {missing_fields}"
                        )

                    # Log field mapping for debugging
                    log_payload_field_mapping(first_payload, hyperparams, var_type_list)

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
        input_paths = {
            "model_input": DEFAULT_MODEL_DIR,
            "custom_payload_input": DEFAULT_CUSTOM_PAYLOAD_DIR,
        }

        output_paths = {"output_dir": DEFAULT_OUTPUT_DIR}

        # Environment variables dictionary
        environ_vars = {}
        for env_var in [
            ENV_CONTENT_TYPES,
            ENV_DEFAULT_NUMERIC_VALUE,
            ENV_DEFAULT_TEXT_VALUE,
            ENV_FIELD_DEFAULTS,  # NEW: Unified field defaults
        ]:
            if env_var in os.environ:
                environ_vars[env_var] = os.environ[env_var]

        # Also add special field variables (backward compatibility)
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

"""
Bedrock Batch Processing Script

Self-contained script that provides AWS Bedrock batch inference capabilities.
Maintains identical input/output interface to bedrock_processing.py while providing
cost-efficient batch processing for large datasets with automatic fallback to real-time processing.

Integrates with Bedrock Prompt Template Generation step outputs and supports
template-driven response processing with dynamic Pydantic model creation.
"""

import os
import json
import sys

from subprocess import check_call
import logging

# ============================================================================
# PACKAGE INSTALLATION CONFIGURATION
# ============================================================================

# Control which PyPI source to use via environment variable
# Set USE_SECURE_PYPI=true to use secure CodeArtifact PyPI
# Set USE_SECURE_PYPI=false or leave unset to use public PyPI
USE_SECURE_PYPI = os.environ.get("USE_SECURE_PYPI", "false").lower() == "true"

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_secure_pypi_access_token() -> str:
    """
    Get CodeArtifact access token for secure PyPI.

    Returns:
        str: Authorization token for CodeArtifact

    Raises:
        Exception: If token retrieval fails
    """
    # Local import to avoid loading boto3 before package installation
    import boto3

    try:
        os.environ["AWS_STS_REGIONAL_ENDPOINTS"] = "regional"
        sts = boto3.client("sts", region_name="us-east-1")
        caller_identity = sts.get_caller_identity()
        assumed_role_object = sts.assume_role(
            RoleArn="arn:aws:iam::675292366480:role/SecurePyPIReadRole_"
            + caller_identity["Account"],
            RoleSessionName="SecurePypiReadRole",
        )
        credentials = assumed_role_object["Credentials"]
        code_artifact_client = boto3.client(
            "codeartifact",
            aws_access_key_id=credentials["AccessKeyId"],
            aws_secret_access_key=credentials["SecretAccessKey"],
            aws_session_token=credentials["SessionToken"],
            region_name="us-west-2",
        )
        token = code_artifact_client.get_authorization_token(
            domain="amazon", domainOwner="149122183214"
        )["authorizationToken"]

        logger.info("Successfully retrieved secure PyPI access token")
        return token

    except Exception as e:
        logger.error(f"Failed to retrieve secure PyPI access token: {e}")
        raise


def install_packages_from_public_pypi(packages: list) -> None:
    """
    Install packages from standard public PyPI.

    Args:
        packages: List of package specifications (e.g., ["pandas==1.5.0", "numpy"])
    """
    logger.info(f"Installing {len(packages)} packages from public PyPI")
    logger.info(f"Packages: {packages}")

    try:
        check_call([sys.executable, "-m", "pip", "install", "--upgrade", *packages])
        logger.info("✓ Successfully installed packages from public PyPI")
    except Exception as e:
        logger.error(f"✗ Failed to install packages from public PyPI: {e}")
        raise


def install_packages_from_secure_pypi(packages: list) -> None:
    """
    Install packages from secure CodeArtifact PyPI.

    Args:
        packages: List of package specifications (e.g., ["pandas==1.5.0", "numpy"])
    """
    logger.info(f"Installing {len(packages)} packages from secure PyPI")
    logger.info(f"Packages: {packages}")

    try:
        token = _get_secure_pypi_access_token()
        index_url = f"https://aws:{token}@amazon-149122183214.d.codeartifact.us-west-2.amazonaws.com/pypi/secure-pypi/simple/"

        check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--index-url",
                index_url,
                *packages,
            ]
        )

        logger.info("✓ Successfully installed packages from secure PyPI")
    except Exception as e:
        logger.error(f"✗ Failed to install packages from secure PyPI: {e}")
        raise


def install_packages(packages: list, use_secure: bool = USE_SECURE_PYPI) -> None:
    """
    Install packages from PyPI source based on configuration.

    This is the main installation function that delegates to either public or
    secure PyPI based on the USE_SECURE_PYPI environment variable.

    Args:
        packages: List of package specifications (e.g., ["pandas==1.5.0", "numpy"])
        use_secure: If True, use secure CodeArtifact PyPI; if False, use public PyPI.
                   Defaults to USE_SECURE_PYPI environment variable.

    Environment Variables:
        USE_SECURE_PYPI: Set to "true" to use secure PyPI, "false" for public PyPI

    Example:
        # Install from public PyPI (default)
        install_packages(["pandas==1.5.0", "numpy"])

        # Install from secure PyPI
        os.environ["USE_SECURE_PYPI"] = "true"
        install_packages(["pandas==1.5.0", "numpy"])
    """
    logger.info("=" * 70)
    logger.info("PACKAGE INSTALLATION")
    logger.info("=" * 70)
    logger.info(f"PyPI Source: {'SECURE (CodeArtifact)' if use_secure else 'PUBLIC'}")
    logger.info(
        f"Environment Variable USE_SECURE_PYPI: {os.environ.get('USE_SECURE_PYPI', 'not set')}"
    )
    logger.info(f"Number of packages: {len(packages)}")
    logger.info("=" * 70)

    try:
        if use_secure:
            install_packages_from_secure_pypi(packages)
        else:
            install_packages_from_public_pypi(packages)

        logger.info("=" * 70)
        logger.info("✓ PACKAGE INSTALLATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)

    except Exception as e:
        logger.error("=" * 70)
        logger.error("✗ PACKAGE INSTALLATION FAILED")
        logger.error("=" * 70)
        raise


# ============================================================================
# INSTALL REQUIRED PACKAGES
# ============================================================================

# Define required packages for this script
required_packages = [
    "pydantic==2.11.2",
    "tenacity==8.5.0",
    "boto3>=1.35.0",
    "botocore>=1.35.0",
]

# Install packages using unified installation function
install_packages(required_packages)

print("***********************Package Installation Complete*********************")

import argparse
import pandas as pd
import boto3
import traceback
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
import logging
from datetime import datetime
import tempfile
import gzip
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, ValidationError, create_model, Field
from tenacity import retry, stop_after_attempt, wait_exponential
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Unicode quote mappings for safe normalization
# CRITICAL: All fancy quotes map to ASCII apostrophe (') to align with output spec
# This prevents creating new JSON delimiters on the error path
UNICODE_DOUBLE_QUOTES = {
    "\u201c": "'",  # " → ' (Left double quotation mark)
    "\u201d": "'",  # " → ' (Right double quotation mark)
    "\u201e": "'",  # „ → ' (Double low-9 quotation mark)
    "\u201f": "'",  # ‟ → ' (Double high-reversed-9 quotation mark)
}

UNICODE_SINGLE_QUOTES = {
    "\u2018": "'",  # ' → ' (Left single quotation mark)
    "\u2019": "'",  # ' → ' (Right single quotation mark)
    "\u201a": "'",  # ‚ → ' (Single low-9 quotation mark)
    "\u201b": "'",  # ‛ → ' (Single high-reversed-9 quotation mark)
}

# Pattern to match German-style quoted text: „something"
# Matches „ followed by any text, then any closing quote (", ", or ")
GERMAN_OPEN_QUOTE_PATTERN = re.compile(r'„([^""\u201c\u201d]*)["\u201c\u201d]')


def normalize_unicode_quotes(text: str) -> str:
    """
    Normalize Unicode quotation marks to ASCII equivalents.

    CRITICAL: Only replaces Unicode quotes (U+201C, U+201D, etc.),
    NEVER touches ASCII double quotes (U+0022) which are structural in JSON.

    Args:
        text: Text containing Unicode quotes

    Returns:
        Text with Unicode quotes normalized to ASCII equivalents
    """
    for bad, repl in UNICODE_DOUBLE_QUOTES.items():
        text = text.replace(bad, repl)
    for bad, repl in UNICODE_SINGLE_QUOTES.items():
        text = text.replace(bad, repl)
    return text


def repair_json(text: str) -> str:
    """
    Repair Unicode/fancy quotes in JSON responses.

    This is a FOCUSED repair function that ONLY handles quote-related issues:
    1. Specifically fixes German quote pattern: „text" → \"text\"
    2. Normalizes other Unicode quotes to ASCII equivalents

    CRITICAL: Does NOT touch ASCII double quotes (") which are structural in JSON.
    CRITICAL: Does NOT attempt generic comma/whitespace fixes which can break valid JSON.

    Based on production analysis of 378,878 records:
    - 100% of parse errors are due to Unicode quotes (German „text" pattern)
    - No other JSON syntax errors observed
    - Generic repairs are unnecessary and risky

    Args:
        text: Raw JSON string that may contain Unicode quotes

    Returns:
        Repaired JSON string with Unicode quotes fixed
    """
    # STEP 1: Fix specific German quote pattern „name" → \"name\"
    # This is the primary cause of parse errors (100% of 341 errors in 378K records)
    # Example: "text „Lutz Koch" more" becomes "text \"Lutz Koch\" more"
    text = GERMAN_OPEN_QUOTE_PATTERN.sub(r'\\"\1\\"', text)

    # STEP 2: Normalize remaining Unicode quotes to ASCII equivalents
    # This handles any other fancy quotes that may appear
    text = normalize_unicode_quotes(text)

    return text


def extract_json_candidate(response_text: str) -> str:
    """
    Extract the first complete JSON object using intelligent brace counting.

    This function properly handles assistant prefilling and finds the first
    structurally complete JSON object by tracking brace balance, accounting
    for braces inside strings.

    Args:
        response_text: Raw response text from LLM

    Returns:
        Extracted JSON substring, or original text if no valid object found
    """
    start = response_text.find("{")
    if start == -1:
        return response_text.strip()

    brace_count = 0
    in_string = False
    escape_next = False

    for i in range(start, len(response_text)):
        char = response_text[i]

        # Handle escape sequences
        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        # Track string boundaries (braces inside strings don't count)
        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        # Count braces only outside strings
        if not in_string:
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                # Found first complete JSON object when count returns to 0
                if brace_count == 0:
                    return response_text[start : i + 1]

    # Fallback: no complete object found, return from first brace onwards
    return response_text[start:].strip()


# Container path constants
CONTAINER_PATHS = {
    "INPUT_DATA_DIR": "/opt/ml/processing/input/data",
    "INPUT_TEMPLATES_DIR": "/opt/ml/processing/input/templates",
    "INPUT_SCHEMA_DIR": "/opt/ml/processing/input/schema",
    "OUTPUT_DATA_DIR": "/opt/ml/processing/output/data",
    "OUTPUT_SUMMARY_DIR": "/opt/ml/processing/output/summary",
}


# ============================================================================
# FILE I/O HELPER FUNCTIONS WITH FORMAT PRESERVATION
# ============================================================================


def _detect_file_format(file_path: Path) -> str:
    """
    Detect the format of a data file based on its extension.

    Args:
        file_path: Path to the file

    Returns:
        Format string: 'csv', 'tsv', or 'parquet'
    """
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return "csv"
    elif suffix == ".tsv":
        return "tsv"
    elif suffix == ".parquet":
        return "parquet"
    else:
        raise RuntimeError(f"Unsupported file format: {suffix}")


def load_dataframe_with_format(file_path: Path) -> tuple:
    """
    Load DataFrame and detect its format.

    Args:
        file_path: Path to the file

    Returns:
        Tuple of (DataFrame, format_string)
    """
    detected_format = _detect_file_format(file_path)

    if detected_format == "csv":
        df = pd.read_csv(file_path)
    elif detected_format == "tsv":
        df = pd.read_csv(file_path, sep="\t")
    elif detected_format == "parquet":
        df = pd.read_parquet(file_path)
    else:
        raise RuntimeError(f"Unsupported format: {detected_format}")

    return df, detected_format


def save_dataframe_with_format(
    df: pd.DataFrame, output_path: Path, format_str: str
) -> Path:
    """
    Save DataFrame in specified format.

    Args:
        df: DataFrame to save
        output_path: Base output path (without extension)
        format_str: Format to save in ('csv', 'tsv', or 'parquet')

    Returns:
        Path to saved file
    """
    if format_str == "csv":
        file_path = output_path.with_suffix(".csv")
        df.to_csv(file_path, index=False)
    elif format_str == "tsv":
        file_path = output_path.with_suffix(".tsv")
        df.to_csv(file_path, sep="\t", index=False)
    elif format_str == "parquet":
        file_path = output_path.with_suffix(".parquet")
        df.to_parquet(file_path, index=False)
    else:
        raise RuntimeError(f"Unsupported output format: {format_str}")

    return file_path


class BedrockProcessor:
    """
    Base Bedrock processor with template-driven response processing.
    Integrates with Bedrock Prompt Template Generation step outputs.
    Supports both sequential and concurrent processing modes.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bedrock_client = None
        self.response_model_class = None
        self.effective_model_id = config["primary_model_id"]
        self.inference_profile_info = {}
        self.validation_schema = config.get("validation_schema", {})

        # Thread-local storage for concurrent processing
        self.thread_local = threading.local()

        # Rate limiting for concurrent requests
        self.max_concurrent_workers = config.get("max_concurrent_workers", 5)
        self.rate_limit_per_second = config.get("rate_limit_per_second", 10)
        self.concurrency_mode = config.get(
            "concurrency_mode", "sequential"
        )  # sequential, concurrent

        # Rate limiting state
        self.request_semaphore = threading.Semaphore(self.max_concurrent_workers)
        self.last_request_times = {}
        self.time_lock = threading.Lock()

        # Input truncation configuration
        self.max_input_field_length = config.get("max_input_field_length", 300000)
        self.truncation_enabled = config.get("truncation_enabled", True)
        self.log_truncations = config.get("log_truncations", True)

        # Truncation tracking
        self.truncation_stats = {
            "total_truncations": 0,
            "truncated_records": 0,
            "truncated_fields": {},
        }

        self._initialize_bedrock_client()
        self._configure_inference_profile()
        self._create_response_model_from_schema()

    def _initialize_bedrock_client(self):
        """Initialize Bedrock client."""
        region_name = self.config.get("region_name", "us-east-1")
        self.bedrock_client = boto3.client("bedrock-runtime", region_name=region_name)
        logger.info(f"Initialized Bedrock client for region: {region_name}")

    def _get_thread_local_bedrock_client(self):
        """Get thread-local Bedrock client for concurrent processing."""
        if not hasattr(self.thread_local, "bedrock_client"):
            region_name = self.config.get("region_name", "us-east-1")
            self.thread_local.bedrock_client = boto3.client(
                "bedrock-runtime", region_name=region_name
            )
        return self.thread_local.bedrock_client

    def _enforce_rate_limit(self):
        """Enforce rate limiting between requests for concurrent processing."""
        if self.concurrency_mode == "sequential":
            return  # No rate limiting needed for sequential processing

        with self.time_lock:
            current_time = time.time()
            min_interval = 1.0 / self.rate_limit_per_second

            thread_id = threading.current_thread().ident
            if thread_id in self.last_request_times:
                elapsed = current_time - self.last_request_times[thread_id]
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)

            self.last_request_times[thread_id] = time.time()

    def _configure_inference_profile(self):
        """Configure inference profile settings based on model and environment."""
        model_id = self.config["primary_model_id"]
        inference_profile_arn = self.config.get("inference_profile_arn")

        # Check if model requires inference profile
        inference_profile_required = json.loads(
            self.config.get("inference_profile_required_models", "[]")
        )

        if inference_profile_arn:
            # Use provided ARN
            self.effective_model_id = inference_profile_arn
            self.inference_profile_info = {
                "arn": inference_profile_arn,
                "method": "arn",
            }
            logger.info(f"Using inference profile ARN: {inference_profile_arn}")

        elif model_id in inference_profile_required:
            # Auto-configure for known models
            if model_id == "anthropic.claude-sonnet-4-20250514-v1:0":
                # Use global profile ID for Claude 4
                self.effective_model_id = (
                    "global.anthropic.claude-sonnet-4-20250514-v1:0"
                )
                self.inference_profile_info = {
                    "profile_id": "global.anthropic.claude-sonnet-4-20250514-v1:0",
                    "original_model_id": model_id,
                    "method": "profile_id",
                }
                logger.info(
                    f"Auto-configured to use inference profile ID: {self.effective_model_id}"
                )

            elif "claude-4" in model_id or "claude-sonnet-4" in model_id:
                logger.warning(
                    f"Model {model_id} may require an inference profile. Consider setting BEDROCK_INFERENCE_PROFILE_ARN."
                )

        # If model already starts with 'global.', it's already a profile ID
        if model_id.startswith("global."):
            self.inference_profile_info = {
                "profile_id": model_id,
                "method": "profile_id",
            }
            logger.info(f"Using provided inference profile ID: {model_id}")

    def _create_response_model_from_schema(self):
        """Create Pydantic response model from validation schema."""
        if not self.validation_schema:
            logger.warning("No validation schema provided, using basic JSON parsing")
            return

        try:
            # Extract schema properties
            properties = self.validation_schema.get("properties", {})
            required_fields = self.validation_schema.get("required", [])
            processing_config = self.validation_schema.get("processing_config", {})

            if not properties:
                logger.warning("No properties found in validation schema")
                return

            # Create Pydantic fields dynamically
            fields = {}
            for field_name, field_schema in properties.items():
                field_type = self._convert_json_schema_type_to_python(field_schema)
                description = field_schema.get("description", f"The {field_name} value")

                if field_name in required_fields:
                    fields[field_name] = (
                        field_type,
                        Field(..., description=description),
                    )
                else:
                    fields[field_name] = (
                        Optional[field_type],
                        Field(None, description=description),
                    )

            # Create dynamic Pydantic model
            model_name = processing_config.get("response_model_name", "BedrockResponse")
            self.response_model_class = create_model(model_name, **fields)

            logger.info(
                f"Created dynamic Pydantic model '{model_name}' with fields: {list(fields.keys())}"
            )

        except Exception as e:
            logger.error(f"Failed to create Pydantic model from schema: {e}")
            self.response_model_class = None

    def _convert_json_schema_type_to_python(self, field_schema: Dict[str, Any]) -> type:
        """Convert JSON schema field definition to Python type with support for nested objects."""
        field_type = field_schema.get("type", "string")

        if field_type == "string":
            if "enum" in field_schema:
                # For enum fields, use str type for simplicity
                # Pydantic will still validate against the schema's enum constraint
                # Note: Could use Literal but requires unpacking: Literal[*values]
                return str
            return str
        elif field_type == "number":
            return float
        elif field_type == "integer":
            return int
        elif field_type == "boolean":
            return bool
        elif field_type == "array":
            # Check if array has items schema for typed arrays
            items_schema = field_schema.get("items", {})
            if items_schema.get("type") == "string":
                from typing import List

                return List[str]
            elif items_schema.get("type") == "number":
                from typing import List

                return List[float]
            elif items_schema.get("type") == "integer":
                from typing import List

                return List[int]
            elif items_schema.get("type") == "object":
                # Array of nested objects
                from typing import List

                nested_model = self._create_nested_model_from_schema(items_schema)
                return List[nested_model]
            else:
                return list  # Generic list fallback
        elif field_type == "object":
            # Create nested Pydantic model for object types
            return self._create_nested_model_from_schema(field_schema)
        else:
            return str  # Default fallback

    def _create_nested_model_from_schema(self, object_schema: Dict[str, Any]) -> type:
        """
        Create a nested Pydantic model from JSON schema object definition.

        Args:
            object_schema: JSON schema for an object type with properties

        Returns:
            Dynamically created Pydantic model class
        """
        properties = object_schema.get("properties", {})
        required_fields = object_schema.get("required", [])

        if not properties:
            # Return generic dict if no properties defined
            return dict

        # Build fields for nested model
        nested_fields = {}
        for prop_name, prop_schema in properties.items():
            prop_type = self._convert_json_schema_type_to_python(prop_schema)
            prop_description = prop_schema.get("description", f"The {prop_name} value")

            if prop_name in required_fields:
                nested_fields[prop_name] = (
                    prop_type,
                    Field(..., description=prop_description),
                )
            else:
                nested_fields[prop_name] = (
                    Optional[prop_type],
                    Field(None, description=prop_description),
                )

        # Create dynamic nested model with unique name
        model_name = object_schema.get("title", f"NestedModel_{id(object_schema)}")
        return create_model(model_name, **nested_fields)

    def _truncate_field_value(
        self, value: str, field_name: str, max_length: int
    ) -> tuple[str, bool]:
        """
        Truncate field value to maximum length, preserving as much original content as possible.

        Args:
            value: Field value to potentially truncate
            field_name: Name of the field (for logging)
            max_length: Maximum length in characters

        Returns:
            Tuple of (truncated_value, was_truncated)
        """
        if len(value) <= max_length:
            return value, False

        # Truncation needed
        truncation_marker = "\n... [TRUNCATED DUE TO LENGTH]"
        keep_length = max_length - len(truncation_marker)

        if keep_length <= 0:
            # Edge case: max_length is too small
            logger.warning(
                f"Field '{field_name}' truncation limit ({max_length}) is too small. "
                f"Using minimum truncation."
            )
            return value[: max(100, max_length)] + truncation_marker, True

        truncated = value[:keep_length] + truncation_marker

        # Log truncation
        if self.log_truncations:
            logger.info(
                f"Truncated field '{field_name}': {len(value)} → {len(truncated)} chars "
                f"({len(value) - len(truncated)} chars removed)"
            )

        # Track truncation stats
        self.truncation_stats["total_truncations"] += 1
        if field_name not in self.truncation_stats["truncated_fields"]:
            self.truncation_stats["truncated_fields"][field_name] = 0
        self.truncation_stats["truncated_fields"][field_name] += 1

        return truncated, True

    def _truncate_input_data(
        self, row_data: Dict[str, Any]
    ) -> tuple[Dict[str, Any], List[str]]:
        """
        Truncate input fields if they exceed maximum length.

        Args:
            row_data: Dictionary containing row data

        Returns:
            Tuple of (truncated_row_data, list_of_truncated_field_names)
        """
        if not self.truncation_enabled:
            return row_data, []

        truncated_data = row_data.copy()
        truncated_fields = []

        for field_name, value in row_data.items():
            # Only truncate string fields
            if isinstance(value, str) and len(value) > self.max_input_field_length:
                truncated_value, was_truncated = self._truncate_field_value(
                    value, field_name, self.max_input_field_length
                )
                truncated_data[field_name] = truncated_value
                if was_truncated:
                    truncated_fields.append(field_name)

        return truncated_data, truncated_fields

    def _format_prompt(self, row_data: Dict[str, Any]) -> str:
        """Format prompt using template placeholders and DataFrame row data with input truncation."""
        # Apply truncation if enabled
        truncated_data, truncated_fields = self._truncate_input_data(row_data)

        if truncated_fields:
            # Track that this record had truncation
            self.truncation_stats["truncated_records"] += 1
            if self.log_truncations:
                logger.info(
                    f"Record had {len(truncated_fields)} truncated fields: {truncated_fields}"
                )

        # Use input_placeholders from template configuration (preferred method)
        placeholders = self.config.get("input_placeholders", [])

        # Fallback to regex extraction if input_placeholders not available
        if not placeholders:
            placeholders = re.findall(r"\{(\w+)\}", self.config["user_prompt_template"])

        # Start with the template
        formatted_prompt = self.config["user_prompt_template"]

        # Replace each placeholder with its value using string replacement
        # This avoids issues with curly braces in JSON examples being interpreted as placeholders
        for placeholder in placeholders:
            placeholder_pattern = "{" + placeholder + "}"
            if placeholder in truncated_data:
                # Convert value to string and replace
                value = (
                    str(truncated_data[placeholder])
                    if truncated_data[placeholder] is not None
                    else ""
                )
                formatted_prompt = formatted_prompt.replace(placeholder_pattern, value)
            else:
                # Log warning for missing placeholder data
                logger.warning(
                    f"Placeholder '{placeholder}' not found in row data. Available columns: {list(truncated_data.keys())}"
                )
                formatted_prompt = formatted_prompt.replace(
                    placeholder_pattern, f"[Missing: {placeholder}]"
                )

        return formatted_prompt

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _invoke_bedrock(self, prompt: str) -> Dict[str, Any]:
        """Invoke Bedrock with intelligent fallback strategy and retry logic."""
        # Enforce rate limiting for concurrent processing
        if self.concurrency_mode == "concurrent":
            self._enforce_rate_limit()

        # Use thread-local client for concurrent processing, main client for sequential
        if self.concurrency_mode == "concurrent":
            client = self._get_thread_local_bedrock_client()
        else:
            client = self.bedrock_client

        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": int(self.config["max_tokens"]),
            "temperature": float(self.config["temperature"]),
            "top_p": float(self.config["top_p"]),
            "messages": [
                {"role": "user", "content": prompt},
                {
                    "role": "assistant",
                    "content": "{",
                },  # Force JSON output via prefilling
            ],
        }

        if self.config.get("system_prompt"):
            request_body["system"] = self.config["system_prompt"]

        # Try primary model/profile first
        try:
            response = client.invoke_model(
                modelId=self.effective_model_id,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json",
            )
            return json.loads(response["body"].read())

        except Exception as e:
            # Fallback to on-demand model if inference profile fails
            fallback_model = self.config.get("fallback_model_id")
            if fallback_model and "ValidationException" in str(e):
                logger.warning(
                    f"Inference profile failed, falling back to: {fallback_model}"
                )
                try:
                    response = client.invoke_model(
                        modelId=fallback_model,
                        body=json.dumps(request_body),
                        contentType="application/json",
                        accept="application/json",
                    )
                    return json.loads(response["body"].read())
                except Exception as fallback_error:
                    logger.error(f"Fallback model also failed: {fallback_error}")
                    raise fallback_error
            else:
                raise e

    def _parse_response_with_pydantic(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse Bedrock response using Pydantic model validation with focused quote repair.

        Uses a two-step approach:
        1. Extract JSON candidate from response (between first { and last })
        2. Try parse → repair (quote-only) → retry

        The repair function ONLY handles Unicode quotes, preserving JSON structure.
        """
        if "content" in response and len(response["content"]) > 0:
            response_text = response["content"][0].get("text", "")
        else:
            raise ValueError("No content in Bedrock response")

        try:
            if self.response_model_class:
                # STEP 0: Handle assistant prefilling BEFORE extraction (CRITICAL)
                # Prepend { BEFORE extraction to avoid grabbing nested objects
                if not response_text.strip().startswith("{"):
                    response_text = "{" + response_text
                    logger.debug("Prepended opening brace from assistant prefilling")

                # STEP 1: Extract JSON with smart brace counting
                complete_json = extract_json_candidate(response_text)

                # STEP 1: Try parsing as-is
                try:
                    validated_response = self.response_model_class.model_validate_json(
                        complete_json
                    )
                    # Success on first attempt
                    result = validated_response.model_dump()
                    result["parse_status"] = "success"
                    result["validation_passed"] = True
                    return result

                except (ValidationError, json.JSONDecodeError) as first_error:
                    # STEP 2: Repair with focused quote-only repair and retry
                    logger.warning(
                        f"Initial JSON parsing failed, attempting focused quote repair: {first_error}"
                    )
                    repaired_json = repair_json(complete_json)

                    try:
                        validated_response = (
                            self.response_model_class.model_validate_json(repaired_json)
                        )
                        logger.info("JSON quote repair successful")

                        # Success after repair
                        result = validated_response.model_dump()
                        result["parse_status"] = "success"
                        result["validation_passed"] = True
                        return result

                    except (ValidationError, json.JSONDecodeError) as second_error:
                        # Both attempts failed - log for debugging
                        logger.error(
                            f"JSON repair failed. Original error: {first_error}"
                        )
                        logger.error(f"Repair attempt error: {second_error}")
                        logger.error(
                            f"Original JSON (first 500 chars): {complete_json[:500]}"
                        )
                        logger.error(
                            f"Repaired JSON (first 500 chars): {repaired_json[:500]}"
                        )
                        raise second_error

            else:
                # Fallback: No Pydantic model, use basic JSON parsing
                complete_json = extract_json_candidate(response_text)
                parsed_json = json.loads(complete_json)
                parsed_json["parse_status"] = "json_only"
                parsed_json["validation_passed"] = False
                return parsed_json

        except ValidationError as e:
            logger.error(f"Pydantic validation failed: {e}")
            return {
                "raw_response": response_text,
                "validation_error": str(e),
                "parse_status": "validation_failed",
                "validation_passed": False,
            }
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            return {
                "raw_response": response_text,
                "json_error": str(e),
                "parse_status": "json_failed",
                "validation_passed": False,
            }

    def process_single_case(self, row_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single case through Bedrock using template placeholders.

        Args:
            row_data: Dictionary containing all row data from DataFrame

        Returns:
            Dictionary with analysis results and metadata
        """
        try:
            # Format prompt using template placeholders
            prompt = self._format_prompt(row_data)

            # Invoke Bedrock
            response = self._invoke_bedrock(prompt)

            # Parse response with Pydantic validation
            parsed_result = self._parse_response_with_pydantic(response)

            # Add processing metadata
            result = {
                **parsed_result,
                "processing_status": "success",
                "error_message": None,
                "model_info": {
                    "effective_model_id": self.effective_model_id,
                    "inference_profile_info": self.inference_profile_info,
                },
            }

            return result

        except Exception as e:
            logger.error(f"Error processing case: {str(e)}")

            # Return structured error response
            error_result = {
                "processing_status": "error",
                "error_message": str(e),
                "raw_response": None,
                "parse_status": "error",
                "validation_passed": False,
                "model_info": {
                    "effective_model_id": self.effective_model_id,
                    "inference_profile_info": self.inference_profile_info,
                },
            }

            # Add default values for expected fields if Pydantic model is available
            if self.response_model_class:
                try:
                    default_fields = self.response_model_class.model_fields.keys()
                    for field in default_fields:
                        if field not in error_result:
                            error_result[field] = None
                except Exception:
                    pass

            return error_result

    def process_batch(
        self,
        df: pd.DataFrame,
        batch_size: Optional[int] = None,
        save_intermediate: bool = True,
        output_dir: Optional[Path] = None,
        input_filename: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Process a batch of data through Bedrock using template placeholders.
        Uses sequential processing mode.

        Args:
            df: Input DataFrame
            batch_size: Number of cases to process in each batch
            save_intermediate: Whether to save intermediate results

        Returns:
            DataFrame with analysis results
        """
        batch_size = batch_size or self.config.get("batch_size", 10)
        results = []
        total_batches = (len(df) + batch_size - 1) // batch_size

        output_prefix = self.config["output_column_prefix"]

        # Extract placeholders from template to validate DataFrame columns
        # Use input_placeholders from config if available, otherwise use regex
        if self.config.get("input_placeholders"):
            placeholders = self.config["input_placeholders"]
            logger.info(f"Using input_placeholders from template: {placeholders}")
        else:
            placeholders = re.findall(r"\{(\w+)\}", self.config["user_prompt_template"])
            logger.info(
                f"Using regex fallback for placeholder extraction: {placeholders}"
            )

        # Log available columns
        logger.info(f"Available DataFrame columns: {list(df.columns)}")
        logger.info("Sequential processing mode")

        # Check for missing placeholders
        missing_placeholders = [p for p in placeholders if p not in df.columns]
        if missing_placeholders:
            logger.warning(
                f"Missing DataFrame columns for placeholders: {missing_placeholders}"
            )

        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i : i + batch_size].copy()
            batch_num = i // batch_size + 1

            logger.info(
                f"Processing batch {batch_num}/{total_batches} ({len(batch_df)} records)"
            )

            batch_results = []
            for idx, row in batch_df.iterrows():
                # Convert row to dictionary for template processing
                row_data = row.to_dict()

                # Process single case using template placeholders
                result = self.process_single_case(row_data)

                # Add original row data
                result_row = row_data.copy()

                # Add Bedrock results with prefix
                for key, value in result.items():
                    if key not in ["processing_status", "error_message", "model_info"]:
                        result_row[f"{output_prefix}{key}"] = value

                # Add processing metadata
                result_row[f"{output_prefix}status"] = result["processing_status"]
                if result.get("error_message"):
                    result_row[f"{output_prefix}error"] = result["error_message"]

                batch_results.append(result_row)

            results.extend(batch_results)

            # Save intermediate results
            if save_intermediate:
                intermediate_df = pd.DataFrame(batch_results)
                output_dir_path = (
                    output_dir
                    if output_dir is not None
                    else Path(CONTAINER_PATHS["OUTPUT_DATA_DIR"])
                )
                output_dir_path.mkdir(parents=True, exist_ok=True)

                # Use filename-based naming to prevent collisions
                if input_filename:
                    base_name = Path(input_filename).stem
                    intermediate_file = (
                        output_dir_path
                        / f"{base_name}_batch_{batch_num:04d}_results.parquet"
                    )
                else:
                    # Fallback for backward compatibility
                    intermediate_file = (
                        output_dir_path / f"batch_{batch_num:04d}_results.parquet"
                    )

                intermediate_df.to_parquet(intermediate_file, index=False)
                logger.info(f"Saved intermediate results to {intermediate_file}")

        results_df = pd.DataFrame(results)
        logger.info(f"Completed sequential processing {len(results_df)} records")

        return results_df


class BedrockBatchProcessor(BedrockProcessor):
    """
    Bedrock batch processor extending BedrockProcessor with batch inference capabilities.
    Maintains full compatibility while adding cost-efficient batch processing.

    Uses cursus framework patterns for S3 path management via environment variables
    set by the step builder using _get_base_output_path() and Join().
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Batch-specific configuration
        self.batch_mode = config.get("batch_mode", "auto")  # auto, batch, realtime
        self.batch_threshold = config.get("batch_threshold", 1000)
        self.batch_role_arn = config.get("batch_role_arn")
        self.batch_timeout_hours = config.get("batch_timeout_hours", 24)

        # AWS Bedrock batch inference limits (per model)
        # Reference: https://docs.aws.amazon.com/general/latest/gr/bedrock.html
        self.max_records_per_job = config.get(
            "max_records_per_job", 45000
        )  # Conservative limit (AWS max: 50,000)
        self.max_concurrent_batch_jobs = config.get(
            "max_concurrent_batch_jobs", 20
        )  # AWS limit

        # Extract S3 paths from config (populated by main() from environ_vars)
        self.batch_input_s3_path = config.get("batch_input_s3_path")
        self.batch_output_s3_path = config.get("batch_output_s3_path")

        # Parse bucket and prefix from the full S3 paths
        if self.batch_input_s3_path:
            self.input_bucket, self.input_prefix = self._parse_s3_path(
                self.batch_input_s3_path
            )
            logger.info(f"Parsed input bucket: {self.input_bucket}")
            logger.info(f"Parsed input prefix: {self.input_prefix}")
        else:
            self.input_bucket, self.input_prefix = None, None
            logger.warning("No BEDROCK_BATCH_INPUT_S3_PATH found in environment")

        if self.batch_output_s3_path:
            self.output_bucket, self.output_prefix = self._parse_s3_path(
                self.batch_output_s3_path
            )
            logger.info(f"Parsed output bucket: {self.output_bucket}")
            logger.info(f"Parsed output prefix: {self.output_prefix}")
        else:
            self.output_bucket, self.output_prefix = None, None
            logger.warning("No BEDROCK_BATCH_OUTPUT_S3_PATH found in environment")

        # Initialize S3 client for batch operations
        self.s3_client = boto3.client(
            "s3", region_name=config.get("region_name", "us-east-1")
        )

        # Initialize Bedrock batch client for batch inference operations
        # Note: bedrock-runtime is for real-time inference, bedrock is for batch operations
        self.bedrock_batch_client = boto3.client(
            "bedrock", region_name=config.get("region_name", "us-east-1")
        )

        logger.info(
            f"Initialized batch processor - mode: {self.batch_mode}, threshold: {self.batch_threshold}"
        )
        if self.batch_input_s3_path:
            logger.info(f"Batch input S3 path: {self.batch_input_s3_path}")
        if self.batch_output_s3_path:
            logger.info(f"Batch output S3 path: {self.batch_output_s3_path}")

    def _parse_s3_path(self, s3_path: str) -> tuple[str, str]:
        """Parse S3 path into bucket and prefix components."""
        if isinstance(s3_path, str) and s3_path.startswith("s3://"):
            path_parts = s3_path.replace("s3://", "").split("/")
            bucket = path_parts[0]
            prefix = "/".join(path_parts[1:]) if len(path_parts) > 1 else ""
            return bucket, prefix
        else:
            # Handle SageMaker Join objects or other dynamic paths
            # These will be resolved at runtime by SageMaker
            logger.warning(f"S3 path is not a standard string: {s3_path}")
            return str(s3_path), ""

    def should_use_batch_processing(self, df: pd.DataFrame) -> bool:
        """Determine whether to use batch or real-time processing."""
        if self.batch_mode == "realtime":
            return False
        elif self.batch_mode == "batch":
            return True
        else:  # auto mode
            return (
                len(df) >= self.batch_threshold
                and self.batch_role_arn is not None
                and self.input_bucket is not None
                and self.output_bucket is not None
            )

    def convert_df_to_jsonl(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert DataFrame to Bedrock batch JSONL format using existing template logic."""
        jsonl_records = []

        logger.info(
            f"Converting {len(df)} records to JSONL format for batch processing"
        )

        for pos, (_, row) in enumerate(df.iterrows()):
            # Use parent class method to format prompt with template placeholders
            # This ensures identical template processing as real-time mode
            row_data = row.to_dict()
            prompt = self._format_prompt(row_data)

            # Create Bedrock batch inference record
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": int(self.config["max_tokens"]),
                "temperature": float(self.config["temperature"]),
                "top_p": float(self.config["top_p"]),
                "messages": [
                    {"role": "user", "content": prompt},
                    {
                        "role": "assistant",
                        "content": "{",
                    },  # Force JSON output via prefilling
                ],
            }

            if self.config.get("system_prompt"):
                request_body["system"] = self.config["system_prompt"]

            record = {"recordId": f"record_{pos}", "modelInput": request_body}

            jsonl_records.append(record)

        logger.info(f"Created {len(jsonl_records)} JSONL records for batch processing")
        return jsonl_records

    def upload_jsonl_to_s3(self, jsonl_records: List[Dict[str, Any]]) -> str:
        """Upload JSONL data to S3 using multipart upload for large files."""
        if not self.input_bucket:
            raise RuntimeError("No input S3 bucket configured for batch processing")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Use framework-provided input path with timestamp
        if self.input_prefix:
            s3_key = f"{self.input_prefix}/input_{timestamp}.jsonl"
        else:
            s3_key = f"input_{timestamp}.jsonl"

        # Convert to JSONL format
        jsonl_content = "\n".join([json.dumps(record) for record in jsonl_records])
        content_bytes = jsonl_content.encode("utf-8")
        content_size_mb = len(content_bytes) / (1024 * 1024)

        logger.info(
            f"Preparing to upload {len(jsonl_records)} records ({content_size_mb:.2f} MB)"
        )

        # Use multipart upload for files larger than 100MB
        if len(content_bytes) > 100 * 1024 * 1024:  # 100MB threshold
            logger.info(
                f"Using multipart upload for large file ({content_size_mb:.2f} MB)"
            )
            self._upload_large_file_multipart(content_bytes, self.input_bucket, s3_key)
        else:
            # Small files use regular put_object
            logger.info(
                f"Using standard upload for small file ({content_size_mb:.2f} MB)"
            )
            self.s3_client.put_object(
                Bucket=self.input_bucket,
                Key=s3_key,
                Body=content_bytes,
                ContentType="application/jsonl",
            )

        s3_uri = f"s3://{self.input_bucket}/{s3_key}"
        logger.info(f"Successfully uploaded batch input to: {s3_uri}")
        return s3_uri

    def _upload_large_file_multipart(self, content_bytes: bytes, bucket: str, key: str):
        """
        Upload large file using S3 multipart upload.

        S3 multipart upload allows files larger than 5GB to be uploaded by splitting
        them into smaller parts (minimum 5MB per part, except last part).

        Args:
            content_bytes: File content as bytes
            bucket: S3 bucket name
            key: S3 object key
        """
        import io

        part_size = 100 * 1024 * 1024  # 100MB chunks (well above 5MB minimum)
        total_size_mb = len(content_bytes) / (1024 * 1024)
        estimated_parts = (len(content_bytes) + part_size - 1) // part_size

        logger.info(f"Initiating multipart upload for {total_size_mb:.2f} MB file")
        logger.info(
            f"Estimated parts: {estimated_parts} (part size: {part_size / (1024 * 1024):.0f} MB)"
        )

        # Initiate multipart upload
        try:
            mpu = self.s3_client.create_multipart_upload(
                Bucket=bucket, Key=key, ContentType="application/jsonl"
            )
            mpu_id = mpu["UploadId"]
            logger.info(f"Multipart upload initiated with ID: {mpu_id}")

        except Exception as e:
            logger.error(f"Failed to initiate multipart upload: {e}")
            raise RuntimeError(f"Failed to initiate multipart upload: {e}")

        parts = []
        file_obj = io.BytesIO(content_bytes)
        part_number = 1

        try:
            # Upload parts
            while True:
                chunk = file_obj.read(part_size)
                if not chunk:
                    break

                chunk_size_mb = len(chunk) / (1024 * 1024)
                logger.info(
                    f"Uploading part {part_number}/{estimated_parts} ({chunk_size_mb:.2f} MB)..."
                )

                try:
                    part_response = self.s3_client.upload_part(
                        Bucket=bucket,
                        Key=key,
                        PartNumber=part_number,
                        UploadId=mpu_id,
                        Body=chunk,
                    )

                    parts.append(
                        {"PartNumber": part_number, "ETag": part_response["ETag"]}
                    )

                    logger.info(
                        f"✓ Part {part_number} uploaded successfully (ETag: {part_response['ETag']})"
                    )
                    part_number += 1

                except Exception as e:
                    logger.error(f"Failed to upload part {part_number}: {e}")
                    raise

            # Complete multipart upload
            logger.info(f"Completing multipart upload with {len(parts)} parts...")
            self.s3_client.complete_multipart_upload(
                Bucket=bucket,
                Key=key,
                UploadId=mpu_id,
                MultipartUpload={"Parts": parts},
            )

            logger.info(
                f"✓ Multipart upload completed successfully ({len(parts)} parts)"
            )

        except Exception as e:
            # Abort multipart upload on error to avoid orphaned parts
            logger.error(f"Multipart upload failed: {e}")
            logger.info(f"Aborting multipart upload {mpu_id}...")

            try:
                self.s3_client.abort_multipart_upload(
                    Bucket=bucket, Key=key, UploadId=mpu_id
                )
                logger.info("Multipart upload aborted successfully")
            except Exception as abort_error:
                logger.error(f"Failed to abort multipart upload: {abort_error}")

            raise RuntimeError(f"Multipart upload failed: {e}")

    def _validate_job_name(self, job_name: str) -> None:
        """
        Validate job name against AWS Bedrock naming requirements.

        AWS Bedrock job names must match: [a-zA-Z0-9]{1,63}(-*[a-zA-Z0-9\+\-\.]){0,63}

        Allowed characters:
        - Alphanumeric: a-z, A-Z, 0-9
        - Hyphens: -
        - Plus signs: +
        - Dots: .

        Not allowed:
        - Underscores: _
        - Other special characters

        Args:
            job_name: The job name to validate

        Raises:
            ValueError: If job name doesn't match AWS requirements
        """
        # AWS Bedrock job name pattern
        pattern = r"^[a-zA-Z0-9]{1,63}(-*[a-zA-Z0-9\+\-\.]){0,63}$"

        if not re.match(pattern, job_name):
            raise ValueError(
                f"Invalid job name '{job_name}'. "
                f"AWS Bedrock job names must match pattern: [a-zA-Z0-9]{{1,63}}(-*[a-zA-Z0-9\\+\\-\\.]{{0,63}}. "
                f"Allowed: alphanumeric, hyphens, plus signs, dots. "
                f"Not allowed: underscores or other special characters."
            )

        if len(job_name) > 126:  # 63 + 63 max length
            raise ValueError(
                f"Job name '{job_name}' is too long ({len(job_name)} chars). "
                f"Maximum length is 126 characters."
            )

    def create_batch_job(self, input_s3_uri: str) -> str:
        """Create Bedrock batch inference job using framework-provided output path."""
        if not self.output_bucket:
            raise RuntimeError("No output S3 bucket configured for batch processing")

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_name = f"cursus-bedrock-batch-{timestamp}"

        # Validate job name against AWS requirements
        self._validate_job_name(job_name)

        # Use framework-provided output path (timestamp only, no redundant batch-output/)
        if self.output_prefix:
            output_s3_uri = (
                f"s3://{self.output_bucket}/{self.output_prefix}/{timestamp}/"
            )
        else:
            output_s3_uri = f"s3://{self.output_bucket}/{timestamp}/"

        response = self.bedrock_batch_client.create_model_invocation_job(
            jobName=job_name,
            roleArn=self.batch_role_arn,
            modelId=self.effective_model_id,
            inputDataConfig={
                "s3InputDataConfig": {"s3Uri": input_s3_uri, "s3InputFormat": "JSONL"}
            },
            outputDataConfig={"s3OutputDataConfig": {"s3Uri": output_s3_uri}},
            timeoutDurationInHours=self.batch_timeout_hours,
        )

        job_arn = response["jobArn"]
        logger.info(f"Created batch job: {job_name} (ARN: {job_arn})")
        logger.info(f"Output will be written to: {output_s3_uri}")

        return job_arn

    def monitor_batch_job(self, job_arn: str) -> Dict[str, Any]:
        """Monitor batch job until completion with exponential backoff."""
        logger.info(f"Monitoring batch job: {job_arn}")

        start_time = time.time()
        check_count = 0

        while True:
            response = self.bedrock_batch_client.get_model_invocation_job(
                jobIdentifier=job_arn
            )
            status = response["status"]

            elapsed_time = time.time() - start_time
            logger.info(
                f"Job status: {status} (elapsed: {elapsed_time / 60:.1f} minutes)"
            )

            if status == "Completed":
                logger.info("Batch job completed successfully")
                return response
            elif status in ["Failed", "Stopping", "Stopped"]:
                error_msg = f"Batch job failed with status: {status}"
                if "failureMessage" in response:
                    error_msg += f". Error: {response['failureMessage']}"
                raise RuntimeError(error_msg)

            # Exponential backoff with maximum wait time
            check_count += 1
            wait_time = min(60, 10 * (1.2**check_count))
            time.sleep(wait_time)

    def download_batch_results(
        self, job_response: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Download and parse batch job results from S3."""
        output_config = job_response["outputDataConfig"]["s3OutputDataConfig"]
        output_s3_uri = output_config["s3Uri"]

        # Parse S3 URI
        bucket = output_s3_uri.replace("s3://", "").split("/")[0]
        prefix = "/".join(output_s3_uri.replace("s3://", "").split("/")[1:])

        logger.info(f"Downloading batch results from: {output_s3_uri}")

        # List objects in output location
        response = self.s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

        if "Contents" not in response:
            raise RuntimeError(f"No output files found at {output_s3_uri}")

        # Log all objects found for debugging
        logger.info(f"Found {len(response['Contents'])} objects in S3 location")
        for obj in response["Contents"]:
            logger.info(f"  S3 object: s3://{bucket}/{obj['Key']}")

        # Download and parse results (Bedrock outputs with .jsonl.out extension)
        all_results = []
        for obj in response["Contents"]:
            if obj["Key"].endswith(".jsonl.out"):
                logger.info(f"Downloading result file: s3://{bucket}/{obj['Key']}")

                # Download JSONL file
                response = self.s3_client.get_object(Bucket=bucket, Key=obj["Key"])
                content = response["Body"].read().decode("utf-8")

                # Parse JSONL
                for line in content.strip().split("\n"):
                    if line.strip():
                        result = json.loads(line)
                        all_results.append(result)

        logger.info(f"Downloaded {len(all_results)} batch results")
        return all_results

    def convert_batch_results_to_df(
        self, batch_results: List[Dict[str, Any]], original_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Convert batch results back to DataFrame format maintaining exact compatibility."""
        processed_rows = []
        output_prefix = self.config["output_column_prefix"]

        logger.info(
            f"Converting {len(batch_results)} batch results back to DataFrame format"
        )

        # Create mapping from recordId to original row index
        record_id_to_idx = {}
        for result in batch_results:
            record_id = result["recordId"]
            idx = int(record_id.split("_")[1])
            record_id_to_idx[record_id] = idx

        # Process each result
        for result in batch_results:
            record_id = result["recordId"]
            idx = record_id_to_idx[record_id]

            # Get original row data and preserve index
            original_row = original_df.iloc[idx].to_dict()
            original_row["_original_index"] = original_df.index[idx]

            try:
                # Parse response using parent class method (same as real-time processing)
                if "modelOutput" in result:
                    parsed_result = self._parse_response_with_pydantic(
                        result["modelOutput"]
                    )

                    # Add LLM results with prefix (same as parent class)
                    for key, value in parsed_result.items():
                        if key not in [
                            "processing_status",
                            "error_message",
                            "model_info",
                        ]:
                            original_row[f"{output_prefix}{key}"] = value

                    # Add processing metadata
                    original_row[f"{output_prefix}status"] = parsed_result.get(
                        "processing_status", "success"
                    )
                    if parsed_result.get("error_message"):
                        original_row[f"{output_prefix}error"] = parsed_result[
                            "error_message"
                        ]

                else:
                    # Handle error case
                    original_row[f"{output_prefix}status"] = "error"
                    original_row[f"{output_prefix}error"] = result.get(
                        "error", "Unknown error"
                    )

            except Exception as e:
                logger.error(f"Error processing result for record {record_id}: {e}")
                original_row[f"{output_prefix}status"] = "error"
                original_row[f"{output_prefix}error"] = str(e)

            processed_rows.append(original_row)

        # Convert to DataFrame and restore original index
        result_df = pd.DataFrame(processed_rows)

        # Set the original index and sort to maintain input order
        if "_original_index" in result_df.columns:
            result_df = result_df.set_index("_original_index")
            result_df.index.name = None  # Remove index name for clean output
            result_df = result_df.sort_index()
        logger.info(
            f"Converted batch results to DataFrame with {len(result_df)} records"
        )

        return result_df

    def _estimate_jsonl_size(self, jsonl_records: List[Dict[str, Any]]) -> int:
        """
        Estimate JSONL file size in bytes.

        Args:
            jsonl_records: List of JSONL records

        Returns:
            Estimated size in bytes
        """
        # Convert first few records to estimate average size
        sample_size = min(100, len(jsonl_records))
        sample_records = jsonl_records[:sample_size]

        # Serialize sample records
        sample_bytes = 0
        for record in sample_records:
            record_json = json.dumps(record)
            sample_bytes += len(record_json.encode("utf-8")) + 1  # +1 for newline

        # Estimate total size based on average
        avg_record_size = sample_bytes / sample_size
        estimated_total = int(avg_record_size * len(jsonl_records))

        return estimated_total

    def _split_dataframe_for_batch(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Split DataFrame into chunks that comply with AWS Bedrock batch limits.

        AWS Limits:
        - Record count: 50,000 records per file (conservative: 45,000)
        - File size: 1GB (1,073,741,824 bytes) per file (conservative: 900MB)

        Args:
            df: Input DataFrame

        Returns:
            List of DataFrame chunks
        """
        MAX_FILE_SIZE = 900 * 1024 * 1024  # 900MB conservative limit (AWS: 1GB)

        total_records = len(df)
        logger.info(f"Splitting {total_records} records with dual limits:")
        logger.info(f"  Max records per job: {self.max_records_per_job}")
        logger.info(f"  Max file size: {MAX_FILE_SIZE / (1024 * 1024):.0f}MB")

        chunks = []
        current_start = 0
        chunk_num = 1

        while current_start < total_records:
            # Start with max record count
            current_end = min(current_start + self.max_records_per_job, total_records)
            chunk = df.iloc[current_start:current_end].copy()

            # Convert to JSONL and check size
            jsonl_records = self.convert_df_to_jsonl(chunk)
            estimated_size = self._estimate_jsonl_size(jsonl_records)

            # If too large, split further by reducing record count
            while estimated_size > MAX_FILE_SIZE and len(chunk) > 1:
                # Reduce chunk size by proportion
                size_ratio = MAX_FILE_SIZE / estimated_size
                new_size = max(
                    1, int(len(chunk) * size_ratio * 0.9)
                )  # 90% safety margin

                logger.warning(
                    f"Chunk too large ({estimated_size / (1024 * 1024):.1f}MB > "
                    f"{MAX_FILE_SIZE / (1024 * 1024):.0f}MB), "
                    f"reducing from {len(chunk)} to {new_size} records"
                )

                current_end = current_start + new_size
                chunk = df.iloc[current_start:current_end].copy()
                jsonl_records = self.convert_df_to_jsonl(chunk)
                estimated_size = self._estimate_jsonl_size(jsonl_records)

            chunks.append(chunk)
            logger.info(
                f"Chunk {chunk_num}: {len(chunk)} records "
                f"(rows {current_start}-{current_end - 1}), "
                f"estimated size: {estimated_size / (1024 * 1024):.1f}MB"
            )

            current_start = current_end
            chunk_num += 1

        logger.info(f"Split into {len(chunks)} chunks total")
        return chunks

    def _monitor_multiple_batch_jobs(self, job_arns: List[str]) -> List[Dict[str, Any]]:
        """
        Monitor multiple batch jobs in parallel until all complete.

        Args:
            job_arns: List of job ARNs to monitor

        Returns:
            List of job responses in the same order as job_arns
        """
        logger.info(f"Monitoring {len(job_arns)} batch jobs in parallel")

        job_statuses = {arn: "Submitted" for arn in job_arns}
        job_responses = {arn: None for arn in job_arns}
        start_time = time.time()
        check_count = 0

        while True:
            # Check all jobs
            all_completed = True
            for job_arn in job_arns:
                if job_statuses[job_arn] in ["Completed", "Failed", "Stopped"]:
                    continue

                try:
                    response = self.bedrock_batch_client.get_model_invocation_job(
                        jobIdentifier=job_arn
                    )
                    status = response["status"]
                    job_statuses[job_arn] = status
                    job_responses[job_arn] = response

                    if status not in ["Completed", "Failed", "Stopping", "Stopped"]:
                        all_completed = False

                except Exception as e:
                    logger.error(f"Error checking job {job_arn}: {e}")
                    all_completed = False

            # Log progress
            elapsed_time = time.time() - start_time
            completed_count = sum(1 for s in job_statuses.values() if s == "Completed")
            failed_count = sum(
                1 for s in job_statuses.values() if s in ["Failed", "Stopped"]
            )
            in_progress_count = len(job_arns) - completed_count - failed_count

            logger.info(
                f"Job status (elapsed: {elapsed_time / 60:.1f} min): "
                f"Completed={completed_count}, InProgress={in_progress_count}, Failed={failed_count}"
            )

            # Check if all completed or failed
            if all_completed:
                # Check for failures
                failed_jobs = [
                    arn
                    for arn, status in job_statuses.items()
                    if status in ["Failed", "Stopped"]
                ]
                if failed_jobs:
                    error_msg = f"{len(failed_jobs)} batch jobs failed: {failed_jobs}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

                logger.info(f"All {len(job_arns)} batch jobs completed successfully")
                # Return responses in same order as input
                return [job_responses[arn] for arn in job_arns]

            # Exponential backoff
            check_count += 1
            wait_time = min(60, 10 * (1.2**check_count))
            time.sleep(wait_time)

    def process_batch_inference(
        self,
        df: pd.DataFrame,
        batch_size: Optional[int] = None,
        save_intermediate: bool = True,
        output_dir: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        Process DataFrame using Bedrock batch inference with multi-job support.

        Automatically splits large datasets into multiple batch jobs to comply with
        AWS Bedrock limits (50,000 records per file). Jobs are processed in parallel
        for optimal performance.

        Args:
            df: Input DataFrame
            batch_size: Unused (kept for interface compatibility)
            save_intermediate: Whether to save intermediate results
            output_dir: Output directory for intermediate results

        Returns:
            DataFrame with batch inference results
        """
        logger.info(f"Starting batch inference processing for {len(df)} records")

        try:
            # Check if we need multi-job processing
            if len(df) <= self.max_records_per_job:
                logger.info(f"Using single batch job for {len(df)} records")
                return self._process_single_batch_job(df)
            else:
                logger.info(f"Using multi-job batch processing for {len(df)} records")
                return self._process_multi_batch_jobs(df)

        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            logger.info("Falling back to real-time processing...")
            # Fallback to parent class real-time processing
            return super().process_batch(df, batch_size, save_intermediate, output_dir)

    def _process_single_batch_job(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process a single batch job (original implementation)."""
        logger.info(f"Processing single batch job with {len(df)} records")

        # 1. Convert DataFrame to JSONL format
        logger.info("Converting DataFrame to JSONL format...")
        jsonl_records = self.convert_df_to_jsonl(df)

        # 2. Upload to S3
        logger.info("Uploading data to S3...")
        input_s3_uri = self.upload_jsonl_to_s3(jsonl_records)

        # 3. Create batch job
        logger.info("Creating batch inference job...")
        job_arn = self.create_batch_job(input_s3_uri)

        # 4. Monitor job completion
        logger.info("Monitoring job completion...")
        job_response = self.monitor_batch_job(job_arn)

        # 5. Download results
        logger.info("Downloading batch results...")
        batch_results = self.download_batch_results(job_response)

        # 6. Convert back to DataFrame
        logger.info("Converting results back to DataFrame...")
        result_df = self.convert_batch_results_to_df(batch_results, df)

        logger.info(
            f"Single batch job completed successfully for {len(result_df)} records"
        )
        return result_df

    def _process_multi_batch_jobs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process multiple batch jobs in parallel for large datasets.

        Workflow:
        1. Split DataFrame into chunks (≤45K records each)
        2. Convert each chunk to JSONL and upload to S3
        3. Create batch jobs for all chunks
        4. Monitor all jobs in parallel
        5. Download and merge results

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with merged results from all batch jobs
        """
        logger.info(f"Starting multi-job batch processing for {len(df)} records")

        # Step 1: Split DataFrame into chunks
        chunks = self._split_dataframe_for_batch(df)
        logger.info(f"Split data into {len(chunks)} chunks")

        # Step 2 & 3: Upload chunks and create batch jobs
        job_arns = []
        chunk_info = []  # Track which chunk each job processes

        for i, chunk in enumerate(chunks):
            logger.info(
                f"Processing chunk {i + 1}/{len(chunks)} ({len(chunk)} records)"
            )

            # Convert to JSONL
            jsonl_records = self.convert_df_to_jsonl(chunk)

            # Upload to S3
            input_s3_uri = self.upload_jsonl_to_s3(jsonl_records)

            # Create batch job
            job_arn = self.create_batch_job(input_s3_uri)
            job_arns.append(job_arn)
            chunk_info.append(
                {
                    "chunk_index": i,
                    "chunk_size": len(chunk),
                    "start_row": chunk.index[0],
                    "end_row": chunk.index[-1],
                    "job_arn": job_arn,
                }
            )

            logger.info(f"Created job {i + 1}/{len(chunks)}: {job_arn}")

        logger.info(f"Successfully created {len(job_arns)} batch jobs")

        # Step 4: Monitor all jobs in parallel
        logger.info("Monitoring all batch jobs...")
        job_responses = self._monitor_multiple_batch_jobs(job_arns)

        # Step 5: Download and merge results
        logger.info("Downloading and merging results from all jobs...")
        all_results = []

        for i, (job_response, chunk) in enumerate(zip(job_responses, chunks)):
            logger.info(f"Downloading results for job {i + 1}/{len(job_responses)}")

            # Download results
            batch_results = self.download_batch_results(job_response)

            # Convert to DataFrame
            chunk_result_df = self.convert_batch_results_to_df(batch_results, chunk)
            all_results.append(chunk_result_df)

            logger.info(f"Processed {len(chunk_result_df)} results from job {i + 1}")

        # Merge all results
        logger.info("Merging results from all batch jobs...")
        result_df = pd.concat(all_results, ignore_index=False)

        # Sort by original index to maintain order
        result_df = result_df.sort_index()

        logger.info(
            f"Multi-job batch inference completed successfully: "
            f"{len(result_df)} total records from {len(job_arns)} jobs"
        )

        return result_df

    def process_batch(
        self,
        df: pd.DataFrame,
        batch_size: Optional[int] = None,
        save_intermediate: bool = True,
        output_dir: Optional[Path] = None,
        input_filename: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Main processing method with automatic batch/real-time selection.
        Maintains exact same interface as parent class.
        """
        if self.should_use_batch_processing(df):
            logger.info(f"Using batch processing for {len(df)} records")
            return self.process_batch_inference(
                df, batch_size, save_intermediate, output_dir
            )
        else:
            logger.info(f"Using real-time processing for {len(df)} records")
            return super().process_batch(
                df, batch_size, save_intermediate, output_dir, input_filename
            )


def load_prompt_templates(
    templates_path: str, log: Callable[[str], None]
) -> Dict[str, Any]:
    """
    Load prompt templates from Bedrock Prompt Template Generation step output.

    Expected file structure from Template Generation step:
    - prompts.json: JSON file containing system_prompt, user_prompt_template, and input_placeholders

    Args:
        templates_path: Path to templates directory from Template Generation step
        log: Logger function

    Returns:
        Dictionary with 'system_prompt', 'user_prompt_template', and 'input_placeholders' keys
    """
    templates = {}
    templates_dir = Path(templates_path)

    if not templates_dir.exists():
        raise ValueError(f"Templates directory not found: {templates_path}")

    # Load prompts.json (standard output from Template Generation step)
    prompts_file = templates_dir / "prompts.json"
    if prompts_file.exists():
        try:
            with open(prompts_file, "r", encoding="utf-8") as f:
                json_templates = json.load(f)

            if "system_prompt" in json_templates:
                templates["system_prompt"] = json_templates["system_prompt"]
                log(f"Loaded system prompt from {prompts_file}")

            if "user_prompt_template" in json_templates:
                templates["user_prompt_template"] = json_templates[
                    "user_prompt_template"
                ]
                log(f"Loaded user prompt template from {prompts_file}")

            if "input_placeholders" in json_templates:
                templates["input_placeholders"] = json_templates["input_placeholders"]
                log(
                    f"Loaded input placeholders from {prompts_file}: {json_templates['input_placeholders']}"
                )
            else:
                log("No input_placeholders found in template, will use regex fallback")

        except Exception as e:
            raise ValueError(f"Failed to load templates from {prompts_file}: {e}")
    else:
        raise ValueError(f"Required prompts.json not found in {templates_path}")

    return templates


def load_validation_schema(
    schema_path: str, log: Callable[[str], None]
) -> Dict[str, Any]:
    """
    Load validation schema from Bedrock Prompt Template Generation step output.

    Expected file structure from Template Generation step:
    - validation_schema_*.json: Enhanced validation schema with processing metadata

    Args:
        schema_path: Path to schema directory from Template Generation step
        log: Logger function

    Returns:
        Dictionary containing the validation schema
    """
    schema_dir = Path(schema_path)

    if not schema_dir.exists():
        raise ValueError(f"Schema directory not found: {schema_path}")

    # Look for validation schema files
    schema_files = list(schema_dir.glob("validation_schema_*.json"))
    if not schema_files:
        raise ValueError(f"No validation schema files found in {schema_path}")

    # Use the most recent schema file
    schema_file = sorted(schema_files)[-1]

    try:
        with open(schema_file, "r", encoding="utf-8") as f:
            schema = json.load(f)

        log(f"Loaded validation schema from {schema_file}")

        # Validate schema structure
        required_sections = ["properties", "required"]
        for section in required_sections:
            if section not in schema:
                raise ValueError(
                    f"Missing required section '{section}' in validation schema"
                )

        return schema

    except Exception as e:
        raise ValueError(f"Failed to load validation schema from {schema_file}: {e}")


def load_data_file(file_path: Path, log: Callable[[str], None]) -> pd.DataFrame:
    """
    Load data file with comprehensive format support matching TabularPreprocessing outputs.

    Supports all formats that TabularPreprocessing can output:
    - CSV files (.csv)
    - Parquet files (.parquet)
    - Compressed versions (.csv.gz, .parquet.gz)
    - Various naming patterns from TabularPreprocessing
    """
    try:
        file_str = str(file_path)

        if file_str.endswith(".csv.gz"):
            log(f"Loading compressed CSV file: {file_path}")
            return pd.read_csv(file_path, compression="gzip")
        elif file_str.endswith(".csv"):
            log(f"Loading CSV file: {file_path}")
            return pd.read_csv(file_path)
        elif file_str.endswith(".parquet.gz"):
            log(f"Loading compressed Parquet file: {file_path}")
            # Decompress first, then read parquet
            with gzip.open(file_path, "rb") as f_in:
                with tempfile.NamedTemporaryFile(
                    suffix=".parquet", delete=False
                ) as f_out:
                    f_out.write(f_in.read())
                    temp_path = f_out.name
            try:
                df = pd.read_parquet(temp_path)
                return df
            finally:
                os.unlink(temp_path)
        elif file_str.endswith(".parquet"):
            log(f"Loading Parquet file: {file_path}")
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

    except Exception as e:
        raise RuntimeError(f"Failed to load data file {file_path}: {e}")


def process_split_directory(
    split_name: str,
    split_input_path: Path,
    split_output_path: Path,
    processor: BedrockBatchProcessor,
    config: Dict[str, Any],
    log: Callable[[str], None],
) -> Dict[str, Any]:
    """
    Process a single split directory (train, val, or test).

    Args:
        split_name: Name of the split (train, val, test)
        split_input_path: Path to input split directory
        split_output_path: Path to output split directory
        processor: BedrockBatchProcessor instance
        config: Processing configuration
        log: Logger function

    Returns:
        Dictionary with processing statistics for this split
    """
    # Create output directory for this split
    split_output_path.mkdir(parents=True, exist_ok=True)

    # Find input files in this split directory with comprehensive format support
    input_files = (
        list(split_input_path.glob("*.csv"))
        + list(split_input_path.glob("*.parquet"))
        + list(split_input_path.glob("*.csv.gz"))
        + list(split_input_path.glob("*.parquet.gz"))
    )

    if not input_files:
        log(f"No input files found in {split_input_path}")
        return {
            "split_name": split_name,
            "total_files": 0,
            "total_records": 0,
            "successful_records": 0,
            "failed_records": 0,
            "validation_passed_records": 0,
            "files_processed": [],
            "batch_processing_used": False,
        }

    log(f"Processing {split_name} split with {len(input_files)} files")

    split_results = []
    split_stats = {
        "split_name": split_name,
        "total_files": len(input_files),
        "total_records": 0,
        "successful_records": 0,
        "failed_records": 0,
        "validation_passed_records": 0,
        "files_processed": [],
        "batch_processing_used": False,
    }

    for input_file in input_files:
        log(f"Processing {split_name} file: {input_file}")

        # Load data with format detection
        df, input_format = load_dataframe_with_format(input_file)
        log(f"Detected input format: {input_format}")

        # Process batch (automatically selects batch vs real-time)
        result_df = processor.process_batch(
            df, save_intermediate=False, input_filename=input_file.name
        )  # No intermediate saves for splits

        # Track batch processing usage
        batch_used = processor.should_use_batch_processing(df)
        if batch_used:
            split_stats["batch_processing_used"] = True

        # Update statistics
        split_stats["total_records"] += len(df)

        # Check if result DataFrame is empty
        if len(result_df) == 0:
            log(f"Warning: No results returned for {input_file.name}")
            split_stats["files_processed"].append(
                {
                    "filename": input_file.name,
                    "records": len(df),
                    "successful": 0,
                    "failed": 0,
                    "validation_passed": 0,
                    "success_rate": 0,
                    "validation_rate": 0,
                    "batch_processing_used": batch_used,
                    "warning": "Empty result DataFrame",
                }
            )
            continue

        # Check for required status column
        status_col = f"{config['output_column_prefix']}status"
        if status_col not in result_df.columns:
            log(
                f"Warning: Column '{status_col}' not found in results for {input_file.name}"
            )
            log(f"Available columns: {list(result_df.columns)}")
            split_stats["files_processed"].append(
                {
                    "filename": input_file.name,
                    "records": len(df),
                    "successful": 0,
                    "failed": 0,
                    "validation_passed": 0,
                    "success_rate": 0,
                    "validation_rate": 0,
                    "batch_processing_used": batch_used,
                    "warning": f"Missing column: {status_col}",
                }
            )
            continue

        success_count = len(result_df[result_df[status_col] == "success"])
        failed_count = len(result_df[result_df[status_col] == "error"])

        # Safe check for validation_passed column
        validation_col = f"{config['output_column_prefix']}validation_passed"
        if validation_col in result_df.columns:
            validation_passed_count = len(result_df[result_df[validation_col] == True])
        else:
            validation_passed_count = 0

        split_stats["successful_records"] += success_count
        split_stats["failed_records"] += failed_count
        split_stats["validation_passed_records"] += validation_passed_count
        split_stats["files_processed"].append(
            {
                "filename": input_file.name,
                "records": len(df),
                "successful": success_count,
                "failed": failed_count,
                "validation_passed": validation_passed_count,
                "success_rate": success_count / len(df) if len(df) > 0 else 0,
                "validation_rate": validation_passed_count / len(df)
                if len(df) > 0
                else 0,
                "batch_processing_used": batch_used,
            }
        )

        # Filter out error records if configured
        if config.get("skip_error_records", False):
            original_count = len(result_df)
            result_df = result_df[result_df[status_col] != "error"].copy()
            skipped_count = original_count - len(result_df)
            if skipped_count > 0:
                log(
                    f"Skipped {skipped_count} error records from output for {input_file.name}"
                )

        # Save results with simple channel-based naming
        output_base = split_output_path / f"{split_name}_processed_data"

        # Save in same format as input
        saved_file = save_dataframe_with_format(result_df, output_base, input_format)

        split_results.append(result_df)
        log(f"Saved {split_name} results (format={input_format}): {saved_file}")

    # Calculate split-level statistics
    split_stats["success_rate"] = (
        split_stats["successful_records"] / split_stats["total_records"]
        if split_stats["total_records"] > 0
        else 0
    )
    split_stats["validation_rate"] = (
        split_stats["validation_passed_records"] / split_stats["total_records"]
        if split_stats["total_records"] > 0
        else 0
    )

    log(
        f"Completed {split_name} split: {split_stats['total_records']} records, "
        f"{split_stats['success_rate']:.2%} success rate, batch processing: {split_stats['batch_processing_used']}"
    )

    return split_stats


def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Main logic for Bedrock batch processing with identical interface to bedrock_processing.py.

    Args:
        input_paths: Dictionary of input paths with logical names
        output_paths: Dictionary of output paths with logical names
        environ_vars: Dictionary of environment variables
        job_args: Command line arguments
        logger: Optional logger object (defaults to print if None)

    Returns:
        Dictionary containing processing results and statistics
    """
    # Use print function if no logger is provided
    log = logger or print

    try:
        # Get job_type from arguments
        job_type = job_args.job_type
        log(f"Processing with job_type: {job_type}")

        # Load prompt templates from Template Generation step (REQUIRED)
        if "prompt_templates" not in input_paths:
            raise ValueError(
                "prompt_templates input is required for Bedrock Processing"
            )

        templates = load_prompt_templates(input_paths["prompt_templates"], log)
        log(
            f"Loaded templates: system_prompt={bool(templates.get('system_prompt'))}, user_prompt_template={bool(templates.get('user_prompt_template'))}"
        )

        # Load validation schema from Template Generation step (REQUIRED)
        if "validation_schema" not in input_paths:
            raise ValueError(
                "validation_schema input is required for Bedrock Processing"
            )

        validation_schema = load_validation_schema(
            input_paths["validation_schema"], log
        )
        log(
            f"Loaded validation schema with {len(validation_schema.get('properties', {}))} properties"
        )

        # Build configuration with template integration + batch settings
        config = {
            # Standard Bedrock configuration (same as bedrock_processing.py)
            "primary_model_id": environ_vars.get("BEDROCK_PRIMARY_MODEL_ID"),
            "fallback_model_id": environ_vars.get("BEDROCK_FALLBACK_MODEL_ID", ""),
            "inference_profile_arn": environ_vars.get("BEDROCK_INFERENCE_PROFILE_ARN"),
            "inference_profile_required_models": environ_vars.get(
                "BEDROCK_INFERENCE_PROFILE_REQUIRED_MODELS", "[]"
            ),
            "region_name": environ_vars.get("AWS_DEFAULT_REGION", "us-east-1"),
            # Templates from Template Generation step (required)
            "system_prompt": templates.get("system_prompt"),
            "user_prompt_template": templates.get(
                "user_prompt_template", "Analyze: {input_data}"
            ),
            "input_placeholders": templates.get("input_placeholders", []),
            # Validation schema for response processing
            "validation_schema": validation_schema,
            # API configuration
            "max_tokens": int(environ_vars.get("BEDROCK_MAX_TOKENS", "32768")),
            "temperature": float(environ_vars.get("BEDROCK_TEMPERATURE", "1.0")),
            "top_p": float(environ_vars.get("BEDROCK_TOP_P", "0.999")),
            "max_retries": int(environ_vars.get("BEDROCK_MAX_RETRIES", "3")),
            # Processing configuration
            "batch_size": int(environ_vars.get("BEDROCK_BATCH_SIZE", "10")),
            "output_column_prefix": environ_vars.get(
                "BEDROCK_OUTPUT_COLUMN_PREFIX", "llm_"
            ),
            "skip_error_records": environ_vars.get(
                "BEDROCK_SKIP_ERROR_RECORDS", "false"
            ).lower()
            == "true",
            # Concurrency configuration (inherited from parent)
            "max_concurrent_workers": int(
                environ_vars.get("BEDROCK_MAX_CONCURRENT_WORKERS", "5")
            ),
            "rate_limit_per_second": int(
                environ_vars.get("BEDROCK_RATE_LIMIT_PER_SECOND", "10")
            ),
            "concurrency_mode": environ_vars.get(
                "BEDROCK_CONCURRENCY_MODE", "sequential"
            ),
            # Batch-specific configuration
            "batch_mode": environ_vars.get("BEDROCK_BATCH_MODE", "auto"),
            "batch_threshold": int(environ_vars.get("BEDROCK_BATCH_THRESHOLD", "1000")),
            "batch_role_arn": environ_vars.get("BEDROCK_BATCH_ROLE_ARN"),
            "batch_timeout_hours": int(
                environ_vars.get("BEDROCK_BATCH_TIMEOUT_HOURS", "24")
            ),
            # Batch-specific S3 paths (from step builder via environ_vars)
            "batch_input_s3_path": environ_vars.get("BEDROCK_BATCH_INPUT_S3_PATH"),
            "batch_output_s3_path": environ_vars.get("BEDROCK_BATCH_OUTPUT_S3_PATH"),
            # AWS Bedrock batch limits (configurable)
            "max_records_per_job": int(
                environ_vars.get("BEDROCK_MAX_RECORDS_PER_JOB", "45000")
            ),
            "max_concurrent_batch_jobs": int(
                environ_vars.get("BEDROCK_MAX_CONCURRENT_BATCH_JOBS", "20")
            ),
            # Input truncation configuration
            "max_input_field_length": int(
                environ_vars.get("BEDROCK_MAX_INPUT_FIELD_LENGTH", "400000")
            ),
            "truncation_enabled": environ_vars.get(
                "BEDROCK_TRUNCATION_ENABLED", "true"
            ).lower()
            == "true",
            "log_truncations": environ_vars.get(
                "BEDROCK_LOG_TRUNCATIONS", "true"
            ).lower()
            == "true",
        }

        # Initialize batch processor (extends BedrockProcessor)
        processor = BedrockBatchProcessor(config)

        # Load input data and process using identical logic to bedrock_processing.py
        input_path = Path(input_paths["input_data"])
        output_path = Path(output_paths["processed_data"])
        summary_path = Path(output_paths["analysis_summary"])

        # Create output directories
        output_path.mkdir(parents=True, exist_ok=True)
        summary_path.mkdir(parents=True, exist_ok=True)

        # Initialize processing statistics (same structure as bedrock_processing.py)
        processing_stats = {
            "job_type": job_type,
            "total_files": 0,
            "total_records": 0,
            "successful_records": 0,
            "failed_records": 0,
            "validation_passed_records": 0,
            "files_processed": [],
            "splits_processed": [],
            "model_info": processor.inference_profile_info,
            "effective_model_id": processor.effective_model_id,
            "template_integration": {
                "system_prompt_loaded": bool(templates.get("system_prompt")),
                "user_prompt_template_loaded": bool(
                    templates.get("user_prompt_template")
                ),
                "validation_schema_loaded": bool(validation_schema),
                "pydantic_model_created": processor.response_model_class is not None,
            },
            # Batch-specific statistics
            "batch_processing_used": False,
            "batch_job_info": None,
        }

        # Handle different job types with comprehensive format support (identical logic to bedrock_processing.py)
        if job_type == "training":
            # Training job type: expect train/val/test subdirectories from TabularPreprocessing
            log(
                "Training job type detected - looking for train/val/test subdirectories"
            )

            expected_splits = ["train", "val", "test"]
            splits_found = []

            for split_name in expected_splits:
                split_input_path = input_path / split_name
                if split_input_path.exists() and split_input_path.is_dir():
                    splits_found.append(split_name)
                    log(f"Found {split_name} split directory")

            if not splits_found:
                # Fallback: treat as single dataset if no splits found (TabularPreprocessing fallback mode)
                log(
                    "No train/val/test subdirectories found, treating as single dataset"
                )

                # Support all TabularPreprocessing output formats
                input_files = (
                    list(input_path.glob("*.csv"))
                    + list(input_path.glob("*.parquet"))
                    + list(input_path.glob("*.csv.gz"))
                    + list(input_path.glob("*.parquet.gz"))
                    + list(input_path.glob("*_processed_data.csv"))
                    + list(input_path.glob("*_processed_data.parquet"))
                )

                if not input_files:
                    raise ValueError(
                        f"No supported input files found in {input_path}. "
                        f"Expected formats: .csv, .parquet, .csv.gz, .parquet.gz"
                    )

                log(
                    f"Found {len(input_files)} input files: {[f.name for f in input_files]}"
                )

                # Process each file with batch processing capability
                for input_file in input_files:
                    log(f"Processing file: {input_file}")

                    # Load data with format detection
                    df, input_format = load_dataframe_with_format(input_file)
                    log(f"Detected input format: {input_format}")

                    # Process batch (automatically selects batch vs real-time based on size)
                    result_df = processor.process_batch(
                        df,
                        save_intermediate=True,
                        output_dir=output_path,
                        input_filename=input_file.name,
                    )

                    # Track batch processing usage for statistics
                    batch_used = processor.should_use_batch_processing(df)
                    if batch_used:
                        processing_stats["batch_processing_used"] = True
                        log(f"Used batch processing for {len(df)} records")
                    else:
                        log(f"Used real-time processing for {len(df)} records")

                    # Update statistics (same logic as bedrock_processing.py)
                    processing_stats["total_records"] += len(df)

                    status_col = f"{config['output_column_prefix']}status"
                    success_count = len(result_df[result_df[status_col] == "success"])
                    failed_count = len(result_df[result_df[status_col] == "error"])

                    # Safe check for validation_passed column
                    validation_col = (
                        f"{config['output_column_prefix']}validation_passed"
                    )
                    if validation_col in result_df.columns:
                        validation_passed_count = len(
                            result_df[result_df[validation_col] == True]
                        )
                    else:
                        validation_passed_count = 0

                    processing_stats["successful_records"] += success_count
                    processing_stats["failed_records"] += failed_count
                    processing_stats["validation_passed_records"] += (
                        validation_passed_count
                    )
                    processing_stats["files_processed"].append(
                        {
                            "filename": input_file.name,
                            "records": len(df),
                            "successful": success_count,
                            "failed": failed_count,
                            "validation_passed": validation_passed_count,
                            "success_rate": success_count / len(df)
                            if len(df) > 0
                            else 0,
                            "validation_rate": validation_passed_count / len(df)
                            if len(df) > 0
                            else 0,
                            "batch_processing_used": batch_used,
                        }
                    )

                    # Filter out error records if configured
                    if config.get("skip_error_records", False):
                        original_count = len(result_df)
                        result_df = result_df[result_df[status_col] != "error"].copy()
                        skipped_count = original_count - len(result_df)
                        if skipped_count > 0:
                            log(
                                f"Skipped {skipped_count} error records from output for {input_file.name}"
                            )

                    # Save results with simple channel-based naming
                    output_base = output_path / f"{job_type}_processed_data"

                    saved_file = save_dataframe_with_format(
                        result_df, output_base, input_format
                    )
                    log(f"Saved results (format={input_format}): {saved_file}")

                processing_stats["total_files"] = len(input_files)
            else:
                # Process each split separately with comprehensive format support
                log(f"Processing {len(splits_found)} splits: {splits_found}")

                for split_name in splits_found:
                    split_input_path = input_path / split_name
                    split_output_path = output_path / split_name

                    # Process split with batch processing capability
                    split_stats = process_split_directory(
                        split_name,
                        split_input_path,
                        split_output_path,
                        processor,
                        config,
                        log,
                    )

                    # Aggregate statistics
                    processing_stats["total_files"] += split_stats["total_files"]
                    processing_stats["total_records"] += split_stats["total_records"]
                    processing_stats["successful_records"] += split_stats[
                        "successful_records"
                    ]
                    processing_stats["failed_records"] += split_stats["failed_records"]
                    processing_stats["validation_passed_records"] += split_stats[
                        "validation_passed_records"
                    ]
                    processing_stats["files_processed"].extend(
                        split_stats["files_processed"]
                    )
                    processing_stats["splits_processed"].append(split_stats)

                    # Track batch processing usage across splits
                    if split_stats.get("batch_processing_used", False):
                        processing_stats["batch_processing_used"] = True

        else:
            # Non-training job types: expect single dataset with comprehensive format support
            log(
                f"Non-training job type ({job_type}) detected - processing single dataset"
            )

            # Support all TabularPreprocessing output formats for non-training jobs
            input_files = (
                list(input_path.glob("*.csv"))
                + list(input_path.glob("*.parquet"))
                + list(input_path.glob("*.csv.gz"))
                + list(input_path.glob("*.parquet.gz"))
                + list(input_path.glob(f"*{job_type}*.csv"))
                + list(input_path.glob(f"*{job_type}*.parquet"))
                + list(input_path.glob("*_processed_data.csv"))
                + list(input_path.glob("*_processed_data.parquet"))
            )

            if not input_files:
                raise ValueError(
                    f"No supported input files found in {input_path} for job_type '{job_type}'. "
                    f"Expected formats: .csv, .parquet, .csv.gz, .parquet.gz"
                )

            log(
                f"Found {len(input_files)} input files: {[f.name for f in input_files]}"
            )
            processing_stats["total_files"] = len(input_files)

            for input_file in input_files:
                log(f"Processing file: {input_file}")

                # Load data with format detection
                df, input_format = load_dataframe_with_format(input_file)
                log(f"Detected input format: {input_format}")

                # Process batch (automatically selects batch vs real-time based on size)
                result_df = processor.process_batch(
                    df,
                    save_intermediate=True,
                    output_dir=output_path,
                    input_filename=input_file.name,
                )

                # Track batch processing usage
                batch_used = processor.should_use_batch_processing(df)
                if batch_used:
                    processing_stats["batch_processing_used"] = True
                    log(f"Used batch processing for {len(df)} records")
                else:
                    log(f"Used real-time processing for {len(df)} records")

                # Update statistics (same logic as bedrock_processing.py)
                processing_stats["total_records"] += len(df)

                status_col = f"{config['output_column_prefix']}status"
                success_count = len(result_df[result_df[status_col] == "success"])
                failed_count = len(result_df[result_df[status_col] == "error"])

                # Safe check for validation_passed column
                validation_col = f"{config['output_column_prefix']}validation_passed"
                if validation_col in result_df.columns:
                    validation_passed_count = len(
                        result_df[result_df[validation_col] == True]
                    )
                else:
                    validation_passed_count = 0

                processing_stats["successful_records"] += success_count
                processing_stats["failed_records"] += failed_count
                processing_stats["validation_passed_records"] += validation_passed_count
                processing_stats["files_processed"].append(
                    {
                        "filename": input_file.name,
                        "records": len(df),
                        "successful": success_count,
                        "failed": failed_count,
                        "validation_passed": validation_passed_count,
                        "success_rate": success_count / len(df) if len(df) > 0 else 0,
                        "validation_rate": validation_passed_count / len(df)
                        if len(df) > 0
                        else 0,
                        "batch_processing_used": batch_used,
                    }
                )

                # Filter out error records if configured
                if config.get("skip_error_records", False):
                    original_count = len(result_df)
                    result_df = result_df[result_df[status_col] != "error"].copy()
                    skipped_count = original_count - len(result_df)
                    if skipped_count > 0:
                        log(
                            f"Skipped {skipped_count} error records from output for {input_file.name}"
                        )

                # Save results with simple channel-based naming
                output_base = output_path / f"{job_type}_processed_data"

                saved_file = save_dataframe_with_format(
                    result_df, output_base, input_format
                )
                log(f"Saved results (format={input_format}): {saved_file}")

        # Calculate overall statistics
        processing_stats["overall_success_rate"] = (
            processing_stats["successful_records"] / processing_stats["total_records"]
            if processing_stats["total_records"] > 0
            else 0
        )
        processing_stats["overall_validation_rate"] = (
            processing_stats["validation_passed_records"]
            / processing_stats["total_records"]
            if processing_stats["total_records"] > 0
            else 0
        )
        processing_stats["processing_timestamp"] = datetime.now().isoformat()

        # Add truncation statistics
        processing_stats["truncation_stats"] = {
            "truncation_enabled": config["truncation_enabled"],
            "max_input_field_length": config["max_input_field_length"],
            "total_truncations": processor.truncation_stats["total_truncations"],
            "truncated_records": processor.truncation_stats["truncated_records"],
            "truncated_fields": processor.truncation_stats["truncated_fields"],
            "truncation_rate": processor.truncation_stats["truncated_records"]
            / processing_stats["total_records"]
            if processing_stats["total_records"] > 0
            else 0,
        }

        # Save processing summary
        summary_file = (
            summary_path
            / f"processing_summary_{job_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(summary_file, "w") as f:
            json.dump(processing_stats, f, indent=2, default=str)

        log(f"Processing completed successfully for job_type: {job_type}")
        log(f"Total records: {processing_stats['total_records']}")
        log(f"Success rate: {processing_stats['overall_success_rate']:.2%}")
        log(f"Validation rate: {processing_stats['overall_validation_rate']:.2%}")
        log(f"Model used: {processing_stats['effective_model_id']}")
        log(f"Batch processing used: {processing_stats['batch_processing_used']}")

        # Log truncation statistics
        if processing_stats["truncation_stats"]["truncation_enabled"]:
            log(
                f"Truncation enabled: max_input_field_length={processing_stats['truncation_stats']['max_input_field_length']}"
            )
            log(
                f"Truncated records: {processing_stats['truncation_stats']['truncated_records']} ({processing_stats['truncation_stats']['truncation_rate']:.2%})"
            )
            log(
                f"Total truncations: {processing_stats['truncation_stats']['total_truncations']}"
            )
            if processing_stats["truncation_stats"]["truncated_fields"]:
                log(
                    f"Truncated fields: {processing_stats['truncation_stats']['truncated_fields']}"
                )

        if job_type == "training" and processing_stats["splits_processed"]:
            log("Split-level statistics:")
            for split_stats in processing_stats["splits_processed"]:
                log(
                    f"  {split_stats['split_name']}: {split_stats['total_records']} records, "
                    f"{split_stats['success_rate']:.2%} success rate"
                )

        return processing_stats

    except Exception as e:
        log(f"Processing failed: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        # Argument parser (identical to bedrock_processing.py)
        parser = argparse.ArgumentParser(
            description="Bedrock batch processing script with template integration"
        )
        parser.add_argument(
            "--job_type",
            type=str,
            required=True,
            choices=["training", "validation", "testing", "calibration"],
            help="One of ['training','validation','testing','calibration'] - determines processing behavior and output naming",
        )
        parser.add_argument(
            "--batch-size", type=int, default=10, help="Batch size for processing"
        )
        parser.add_argument(
            "--max-retries",
            type=int,
            default=3,
            help="Maximum retries for Bedrock calls",
        )

        args = parser.parse_args()

        # Set up path dictionaries matching the container paths (identical to bedrock_processing.py)
        input_paths = {
            "input_data": CONTAINER_PATHS["INPUT_DATA_DIR"],
            "prompt_templates": CONTAINER_PATHS["INPUT_TEMPLATES_DIR"],
            "validation_schema": CONTAINER_PATHS["INPUT_SCHEMA_DIR"],
        }

        output_paths = {
            "processed_data": CONTAINER_PATHS["OUTPUT_DATA_DIR"],
            "analysis_summary": CONTAINER_PATHS["OUTPUT_SUMMARY_DIR"],
        }

        # Environment variables dictionary (extends bedrock_processing.py with batch settings)
        environ_vars = {
            # Standard Bedrock configuration (same as bedrock_processing.py)
            "BEDROCK_PRIMARY_MODEL_ID": os.environ.get("BEDROCK_PRIMARY_MODEL_ID"),
            "BEDROCK_FALLBACK_MODEL_ID": os.environ.get(
                "BEDROCK_FALLBACK_MODEL_ID", ""
            ),
            "BEDROCK_INFERENCE_PROFILE_ARN": os.environ.get(
                "BEDROCK_INFERENCE_PROFILE_ARN"
            ),
            "BEDROCK_INFERENCE_PROFILE_REQUIRED_MODELS": os.environ.get(
                "BEDROCK_INFERENCE_PROFILE_REQUIRED_MODELS", "[]"
            ),
            "AWS_DEFAULT_REGION": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            "BEDROCK_MAX_TOKENS": os.environ.get("BEDROCK_MAX_TOKENS", "32768"),
            "BEDROCK_TEMPERATURE": os.environ.get("BEDROCK_TEMPERATURE", "1.0"),
            "BEDROCK_TOP_P": os.environ.get("BEDROCK_TOP_P", "0.999"),
            "BEDROCK_BATCH_SIZE": os.environ.get("BEDROCK_BATCH_SIZE", "10"),
            "BEDROCK_MAX_RETRIES": os.environ.get("BEDROCK_MAX_RETRIES", "3"),
            "BEDROCK_OUTPUT_COLUMN_PREFIX": os.environ.get(
                "BEDROCK_OUTPUT_COLUMN_PREFIX", "llm_"
            ),
            "BEDROCK_SKIP_ERROR_RECORDS": os.environ.get(
                "BEDROCK_SKIP_ERROR_RECORDS", "false"
            ),
            "BEDROCK_MAX_CONCURRENT_WORKERS": os.environ.get(
                "BEDROCK_MAX_CONCURRENT_WORKERS", "5"
            ),
            "BEDROCK_RATE_LIMIT_PER_SECOND": os.environ.get(
                "BEDROCK_RATE_LIMIT_PER_SECOND", "10"
            ),
            "BEDROCK_CONCURRENCY_MODE": os.environ.get(
                "BEDROCK_CONCURRENCY_MODE", "sequential"
            ),
            # Batch-specific configuration
            "BEDROCK_BATCH_MODE": os.environ.get("BEDROCK_BATCH_MODE", "auto"),
            "BEDROCK_BATCH_THRESHOLD": os.environ.get(
                "BEDROCK_BATCH_THRESHOLD", "1000"
            ),
            "BEDROCK_BATCH_ROLE_ARN": os.environ.get("BEDROCK_BATCH_ROLE_ARN"),
            "BEDROCK_BATCH_INPUT_S3_PATH": os.environ.get(
                "BEDROCK_BATCH_INPUT_S3_PATH"
            ),
            "BEDROCK_BATCH_OUTPUT_S3_PATH": os.environ.get(
                "BEDROCK_BATCH_OUTPUT_S3_PATH"
            ),
            "BEDROCK_BATCH_TIMEOUT_HOURS": os.environ.get(
                "BEDROCK_BATCH_TIMEOUT_HOURS", "24"
            ),
            # AWS Bedrock batch limits (configurable)
            "BEDROCK_MAX_RECORDS_PER_JOB": os.environ.get(
                "BEDROCK_MAX_RECORDS_PER_JOB", "45000"
            ),
            "BEDROCK_MAX_CONCURRENT_BATCH_JOBS": os.environ.get(
                "BEDROCK_MAX_CONCURRENT_BATCH_JOBS", "20"
            ),
            # Input truncation configuration
            "BEDROCK_MAX_INPUT_FIELD_LENGTH": os.environ.get(
                "BEDROCK_MAX_INPUT_FIELD_LENGTH", "400000"
            ),
            "BEDROCK_TRUNCATION_ENABLED": os.environ.get(
                "BEDROCK_TRUNCATION_ENABLED", "true"
            ),
            "BEDROCK_LOG_TRUNCATIONS": os.environ.get(
                "BEDROCK_LOG_TRUNCATIONS", "true"
            ),
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
        logger.info(
            f"Bedrock batch processing completed successfully. Results: {result}"
        )
        sys.exit(0)

    except Exception as e:
        logger.error(f"Error in Bedrock batch processing script: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

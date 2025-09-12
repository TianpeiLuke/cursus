"""
Name generator utilities for pipeline naming.

This module provides utilities for generating pipeline names with consistent formats
that comply with SageMaker naming constraints.
"""

import random
import string
import logging
import re

logger = logging.getLogger(__name__)

# SageMaker pipeline name constraint pattern
# Must match: [a-zA-Z0-9](-*[a-zA-Z0-9]){0,255}
PIPELINE_NAME_PATTERN = r"^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,255}$"


def generate_random_word(length: int = 4) -> str:
    """
    Generate a random word of specified length.

    Args:
        length: Length of the random word

    Returns:
        Random string of specified length
    """
    # Using uppercase letters for better readability in names
    return "".join(random.choices(string.ascii_uppercase, k=length))


def validate_pipeline_name(name: str) -> bool:
    """
    Validate that a pipeline name conforms to SageMaker constraints.

    Args:
        name: The pipeline name to validate

    Returns:
        True if the name is valid, False otherwise
    """
    # Check length constraint (SageMaker has a limit of 255 characters)
    if len(name) > 255 or len(name) == 0:
        return False

    # Check pattern constraint
    return bool(re.match(PIPELINE_NAME_PATTERN, name))


def sanitize_pipeline_name(name: str) -> str:
    """
    Sanitize a pipeline name to conform to SageMaker constraints.

    This function:
    1. Replaces dots with hyphens
    2. Replaces underscores with hyphens
    3. Removes any other special characters
    4. Ensures the name starts with an alphanumeric character
    5. Ensures the name ends with an alphanumeric character

    Args:
        name: The pipeline name to sanitize

    Returns:
        A sanitized version of the name that conforms to SageMaker constraints
    """
    # Replace dots and underscores with hyphens
    sanitized = name.replace(".", "-").replace("_", "-")

    # Remove any other special characters
    sanitized = re.sub(r"[^a-zA-Z0-9-]", "", sanitized)

    # Ensure the name starts with an alphanumeric character
    if sanitized and not re.match(r"^[a-zA-Z0-9]", sanitized):
        sanitized = "p" + sanitized

    # Replace multiple consecutive hyphens with a single hyphen
    sanitized = re.sub(r"-+", "-", sanitized)

    # Truncate if the name is too long (SageMaker has a 255 character limit)
    if len(sanitized) > 255:
        sanitized = sanitized[:255]

    # Ensure the name ends with an alphanumeric character (remove trailing hyphens)
    sanitized = sanitized.rstrip("-")

    # If we removed all characters and the original wasn't empty, provide a default
    if not sanitized and name:
        sanitized = "pipeline"

    # Check if the sanitized name is different from the original
    if sanitized != name:
        logger.info(
            f"Pipeline name '{name}' sanitized to '{sanitized}' to conform to SageMaker constraints"
        )

    return sanitized


def generate_pipeline_name(base_name: str, version: str = "1.0") -> str:
    """
    Generate a valid pipeline name with the format:
    {base_name}-{random_word}-{version}-pipeline

    This function ensures the generated name conforms to SageMaker constraints
    by sanitizing it before returning.

    Args:
        base_name: Base name for the pipeline
        version: Version string to include in the name

    Returns:
        A string with the generated pipeline name that passes SageMaker validation
    """
    # Generate random 4-letter word
    random_word = generate_random_word(4)

    # Combine all parts
    name = f"{base_name}-{version}-pipeline"  # f"{base_name}-{random_word}-{version}-pipeline"

    # Sanitize the name to ensure it conforms to SageMaker constraints
    return sanitize_pipeline_name(name)

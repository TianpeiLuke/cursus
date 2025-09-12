"""
Shared constants and enums for config field management.

This module contains constants and enums used throughout the config_field_manager
package, implementing the Type-Safe Specifications principle.
"""

from enum import Enum, auto
from typing import Set, List, Dict, Any

# Special fields that should always be kept in specific sections
SPECIAL_FIELDS_TO_KEEP_SPECIFIC: Set[str] = {
    "image_uri",
    "script_name",
    "output_path",
    "input_path",
    "model_path",
    "hyperparameters",
    "instance_type",
    "job_name_prefix",
    "output_schema",
}

# Patterns that indicate a field is likely non-static
NON_STATIC_FIELD_PATTERNS: Set[str] = {
    "_names",
    "input_",
    "output_",
    "_specific",
    # Modified to be more specific and avoid matching processing_instance_count
    "batch_count",
    "item_count",
    "record_count",
    "instance_type_count",
    "_path",
    "_uri",
}

# Fields that should be excluded from non-static detection
NON_STATIC_FIELD_EXCEPTIONS: Set[str] = {"processing_instance_count"}


class CategoryType(Enum):
    """
    Enumeration of field category types for the simplified structure.

    Implementing the Type-Safe Specifications principle by using an enum
    instead of string literals.
    """

    SHARED = auto()  # Fields shared across all configs
    SPECIFIC = auto()  # Fields specific to certain configs


class MergeDirection(Enum):
    """
    Enumeration of merge directions.

    Specifies the direction to resolve conflicts when merging fields.
    """

    PREFER_SOURCE = auto()  # Use source value in case of conflict
    PREFER_TARGET = auto()  # Use target value in case of conflict
    ERROR_ON_CONFLICT = auto()  # Raise an error on conflict


class SerializationMode(Enum):
    """
    Enumeration of serialization modes.

    Controls the behavior of the serializer with respect to type metadata.
    """

    PRESERVE_TYPES = auto()  # Preserve type information in serialized output
    SIMPLE_JSON = auto()  # Convert to plain JSON without type information
    CUSTOM_FIELDS = auto()  # Only preserve types for certain fields


# Mapping from data structure types to their serialized names
TYPE_MAPPING: Dict[str, str] = {
    "dict": "dict",
    "list": "list",
    "tuple": "tuple",
    "set": "set",
    "frozenset": "frozenset",
    "BaseModel": "pydantic_model",
    "Enum": "enum",
    "datetime": "datetime",
    "Path": "path",
}

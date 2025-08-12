---
tags:
  - code
  - core
  - config_constants
  - enums
  - type_definitions
keywords:
  - configuration constants
  - category types
  - merge directions
  - serialization modes
  - special field patterns
  - type mappings
  - field categorization rules
  - type-safe specifications
topics:
  - configuration constants
  - enumeration definitions
  - field categorization rules
  - type mapping definitions
language: python
date of note: 2025-08-12
---

# Configuration Constants

## Overview

The Configuration Constants module defines shared constants, enums, and type mappings used throughout the configuration field management system. It implements the **Type-Safe Specifications** principle by using enums instead of string literals and provides a **Single Source of Truth** for categorization rules and patterns.

## Core Enumerations

### CategoryType

Defines the field category types for the simplified configuration structure:

```python
class CategoryType(Enum):
    """
    Enumeration of field category types for the simplified structure.
    
    Implementing the Type-Safe Specifications principle by using an enum
    instead of string literals.
    """
    SHARED = auto()  # Fields shared across all configs
    SPECIFIC = auto() # Fields specific to certain configs
```

**Usage:**
- `SHARED`: Fields that have identical values across all configurations and are not special fields
- `SPECIFIC`: Fields that have different values, are special fields, or appear in only some configurations

### MergeDirection

Specifies conflict resolution strategies when merging configurations:

```python
class MergeDirection(Enum):
    """
    Enumeration of merge directions.
    
    Specifies the direction to resolve conflicts when merging fields.
    """
    PREFER_SOURCE = auto()     # Use source value in case of conflict
    PREFER_TARGET = auto()     # Use target value in case of conflict
    ERROR_ON_CONFLICT = auto() # Raise an error on conflict
```

**Usage Examples:**
- `PREFER_SOURCE`: When merging A into B, use A's values for conflicting fields
- `PREFER_TARGET`: When merging A into B, keep B's values for conflicting fields  
- `ERROR_ON_CONFLICT`: Raise an exception when conflicting values are encountered

### SerializationMode

Controls type preservation behavior during serialization:

```python
class SerializationMode(Enum):
    """
    Enumeration of serialization modes.
    
    Controls the behavior of the serializer with respect to type metadata.
    """
    PRESERVE_TYPES = auto()    # Preserve type information in serialized output
    SIMPLE_JSON = auto()       # Convert to plain JSON without type information
    CUSTOM_FIELDS = auto()     # Only preserve types for certain fields
```

**Usage Examples:**
- `PRESERVE_TYPES`: Full type metadata preservation for complex objects
- `SIMPLE_JSON`: Lightweight serialization without type information
- `CUSTOM_FIELDS`: Selective type preservation based on field characteristics

## Field Categorization Rules

### Special Fields

Fields that should always be kept in specific sections regardless of other characteristics:

```python
SPECIAL_FIELDS_TO_KEEP_SPECIFIC: Set[str] = {
    "image_uri", 
    "script_name",
    "output_path", 
    "input_path",
    "model_path",
    "hyperparameters",
    "instance_type",
    "job_name_prefix",
    "output_schema"
}
```

**Rationale:**
- These fields typically contain step-specific values
- They often represent complex objects or runtime-dependent values
- They are essential for step identity and cannot be shared

### Non-Static Field Patterns

Patterns that indicate a field is likely to change at runtime:

```python
NON_STATIC_FIELD_PATTERNS: Set[str] = {
    "_names", 
    "input_", 
    "output_", 
    "_specific", 
    "batch_count",
    "item_count",
    "record_count",
    "instance_type_count", 
    "_path",
    "_uri"
}
```

**Pattern Explanations:**
- `_names`: Dynamic naming fields
- `input_`, `output_`: I/O related fields that vary by step
- `_specific`: Fields marked as step-specific
- `*_count`: Runtime-dependent count fields
- `_path`, `_uri`: File paths and URIs that vary by environment

### Non-Static Field Exceptions

Fields that should be considered static despite matching non-static patterns:

```python
NON_STATIC_FIELD_EXCEPTIONS: Set[str] = {
    "processing_instance_count"
}
```

**Rationale:**
- `processing_instance_count`: Although it contains "count", this field typically has the same value across all processing steps in a pipeline

## Type Mapping Definitions

Mapping from Python data structure types to their serialized representations:

```python
TYPE_MAPPING: Dict[str, str] = {
    "dict": "dict",
    "list": "list",
    "tuple": "tuple",
    "set": "set",
    "frozenset": "frozenset",
    "BaseModel": "pydantic_model",
    "Enum": "enum",
    "datetime": "datetime",
    "Path": "path"
}
```

**Usage:**
- Used by the Type-Aware Serializer to preserve type information
- Enables proper reconstruction of complex types during deserialization
- Provides consistent type identification across the system

## Usage Examples

### Using Category Types

```python
from src.cursus.core.config_fields.constants import CategoryType

def categorize_field(field_name: str, field_info: dict) -> CategoryType:
    """Example categorization function using type-safe enums."""
    if field_info['is_special']:
        return CategoryType.SPECIFIC
    elif field_info['has_multiple_values']:
        return CategoryType.SPECIFIC
    else:
        return CategoryType.SHARED

# Type-safe comparison
if category == CategoryType.SHARED:
    place_in_shared_section(field_name, field_value)
elif category == CategoryType.SPECIFIC:
    place_in_specific_section(field_name, field_value)
```

### Using Merge Directions

```python
from src.cursus.core.config_fields.constants import MergeDirection

def merge_configurations(source: dict, target: dict, 
                        direction: MergeDirection = MergeDirection.PREFER_SOURCE) -> dict:
    """Example merge function with configurable conflict resolution."""
    result = target.copy()
    
    for key, source_value in source.items():
        if key in result and result[key] != source_value:
            # Handle conflict based on direction
            if direction == MergeDirection.PREFER_SOURCE:
                result[key] = source_value
            elif direction == MergeDirection.PREFER_TARGET:
                pass  # Keep target value
            elif direction == MergeDirection.ERROR_ON_CONFLICT:
                raise ValueError(f"Conflict on key {key}")
        else:
            result[key] = source_value
    
    return result
```

### Using Serialization Modes

```python
from src.cursus.core.config_fields.constants import SerializationMode, TYPE_MAPPING

def serialize_with_mode(value: Any, mode: SerializationMode) -> Any:
    """Example serialization with mode-dependent behavior."""
    if isinstance(value, datetime):
        if mode == SerializationMode.PRESERVE_TYPES:
            return {
                "__type_info__": TYPE_MAPPING["datetime"],
                "value": value.isoformat()
            }
        else:
            return value.isoformat()
    
    # Handle other types...
    return value
```

### Checking Field Patterns

```python
from src.cursus.core.config_fields.constants import (
    SPECIAL_FIELDS_TO_KEEP_SPECIFIC,
    NON_STATIC_FIELD_PATTERNS,
    NON_STATIC_FIELD_EXCEPTIONS
)

def is_field_static(field_name: str) -> bool:
    """Determine if a field is likely static based on patterns."""
    # Check exceptions first
    if field_name in NON_STATIC_FIELD_EXCEPTIONS:
        return True
    
    # Check special fields
    if field_name in SPECIAL_FIELDS_TO_KEEP_SPECIFIC:
        return False
    
    # Check non-static patterns
    if any(pattern in field_name for pattern in NON_STATIC_FIELD_PATTERNS):
        return False
    
    # Default to static
    return True

# Usage examples
print(is_field_static("processing_instance_count"))  # True (exception)
print(is_field_static("image_uri"))                  # False (special field)
print(is_field_static("input_path"))                 # False (matches pattern)
print(is_field_static("role"))                       # True (default)
```

## Design Principles

### 1. Type-Safe Specifications

All categorization decisions use enums instead of string literals:

```python
# Type-safe ✓
category = CategoryType.SHARED
if category == CategoryType.SHARED:
    # Handle shared field

# Not type-safe ✗
category = "shared"
if category == "shared":  # Prone to typos
    # Handle shared field
```

### 2. Single Source of Truth

All categorization rules and patterns are defined in one place:

```python
# All components reference the same constants
from .constants import SPECIAL_FIELDS_TO_KEEP_SPECIFIC

# In categorizer
if field_name in SPECIAL_FIELDS_TO_KEEP_SPECIFIC:
    return CategoryType.SPECIFIC

# In merger
if field in SPECIAL_FIELDS_TO_KEEP_SPECIFIC:
    self.logger.warning(f"Special field '{field}' found in shared section")
```

### 3. Explicit Over Implicit

Rules and patterns are explicitly defined rather than embedded in logic:

```python
# Explicit pattern definition ✓
NON_STATIC_FIELD_PATTERNS = {"_path", "_uri", "input_", "output_"}

def is_non_static(field_name: str) -> bool:
    return any(pattern in field_name for pattern in NON_STATIC_FIELD_PATTERNS)

# Implicit pattern logic ✗
def is_non_static(field_name: str) -> bool:
    return ("_path" in field_name or "_uri" in field_name or 
            field_name.startswith("input_") or field_name.startswith("output_"))
```

## Extensibility

The constants module is designed for easy extension:

### Adding New Category Types

```python
class CategoryType(Enum):
    SHARED = auto()
    SPECIFIC = auto()
    # New category types can be added here
    CONDITIONAL = auto()  # Fields that are shared under certain conditions
    TEMPLATE = auto()     # Fields that are templates for other fields
```

### Adding New Field Patterns

```python
# Extend existing patterns
NON_STATIC_FIELD_PATTERNS.update({
    "_dynamic",
    "_runtime",
    "_generated"
})

# Add new pattern categories
TEMPLATE_FIELD_PATTERNS: Set[str] = {
    "_template",
    "_pattern",
    "_format"
}
```

### Adding New Type Mappings

```python
# Extend type mappings for new data types
TYPE_MAPPING.update({
    "numpy.ndarray": "numpy_array",
    "pandas.DataFrame": "pandas_dataframe",
    "torch.Tensor": "torch_tensor"
})
```

## Validation and Consistency

The constants module supports validation of configuration consistency:

### Pattern Validation

```python
def validate_field_patterns() -> List[str]:
    """Validate that field patterns don't conflict."""
    issues = []
    
    # Check for conflicts between special fields and exceptions
    conflicts = SPECIAL_FIELDS_TO_KEEP_SPECIFIC.intersection(NON_STATIC_FIELD_EXCEPTIONS)
    if conflicts:
        issues.append(f"Fields in both special and exception lists: {conflicts}")
    
    return issues
```

### Enum Completeness

```python
def validate_enum_usage() -> List[str]:
    """Validate that all enum values are handled in code."""
    issues = []
    
    # Check that all CategoryType values are handled
    handled_categories = {CategoryType.SHARED, CategoryType.SPECIFIC}
    all_categories = set(CategoryType)
    
    unhandled = all_categories - handled_categories
    if unhandled:
        issues.append(f"Unhandled category types: {unhandled}")
    
    return issues
```

## Integration Points

### Configuration Field Categorizer

The categorizer uses constants for rule-based decisions:

```python
from .constants import CategoryType, SPECIAL_FIELDS_TO_KEEP_SPECIFIC, NON_STATIC_FIELD_PATTERNS

def _categorize_field(self, field_name: str) -> CategoryType:
    if field_name in SPECIAL_FIELDS_TO_KEEP_SPECIFIC:
        return CategoryType.SPECIFIC
    
    if any(pattern in field_name for pattern in NON_STATIC_FIELD_PATTERNS):
        return CategoryType.SPECIFIC
    
    return CategoryType.SHARED
```

### Configuration Merger

The merger uses constants for validation and conflict resolution:

```python
from .constants import MergeDirection, SPECIAL_FIELDS_TO_KEEP_SPECIFIC

def merge_with_direction(self, source: dict, target: dict, 
                        direction: MergeDirection) -> dict:
    # Use enum-based direction handling
    if direction == MergeDirection.PREFER_SOURCE:
        result[key] = source_value
    elif direction == MergeDirection.ERROR_ON_CONFLICT:
        raise ValueError(f"Conflict on key {key}")
```

### Type-Aware Serializer

The serializer uses constants for type mapping and mode control:

```python
from .constants import SerializationMode, TYPE_MAPPING

def serialize(self, val: Any) -> Any:
    if isinstance(val, datetime):
        if self.mode == SerializationMode.PRESERVE_TYPES:
            return {
                self.TYPE_INFO_FIELD: TYPE_MAPPING["datetime"],
                "value": val.isoformat()
            }
        return val.isoformat()
```

## Related Documentation

- [Configuration Field Categorizer](config_field_categorizer.md): Uses constants for categorization rules
- [Configuration Merger](config_merger.md): Uses constants for merge directions and validation
- [Type-Aware Configuration Serializer](type_aware_config_serializer.md): Uses constants for type mapping and modes
- [Configuration Fields Overview](README.md): System overview and integration

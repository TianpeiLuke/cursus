---
tags:
  - code
  - core
  - config_fields
  - constants
  - enums
keywords:
  - configuration constants
  - field categorization
  - merge directions
  - serialization modes
  - type mappings
topics:
  - configuration constants
  - enumeration types
  - field classification
language: python
date of note: 2025-09-07
---

# Configuration Fields Constants

## Overview

The `constants.py` module contains shared constants and enumerations used throughout the config field management system. It implements the **Type-Safe Specifications** principle by providing enum-based type definitions instead of string literals, ensuring compile-time type safety and preventing runtime errors.

## Purpose

This module provides centralized definitions for:
- Field categorization constants and patterns
- Type-safe enumerations for configuration operations
- Merge direction specifications for conflict resolution
- Serialization mode controls for type preservation
- Type mapping definitions for serialization support

## Field Classification Constants

### Special Fields Set

Fields that should always be kept in specific sections regardless of their distribution:

```python
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
    "output_schema"
}
```

These fields represent step-specific configuration that should never be shared, even if they have identical values across multiple configurations.

### Non-Static Field Patterns

Patterns that indicate a field is likely to change at runtime and should be treated as non-static:

```python
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
    "_uri"
}
```

Fields matching these patterns are considered dynamic and are typically placed in specific sections during categorization.

### Non-Static Field Exceptions

Fields that should be excluded from non-static detection despite matching patterns:

```python
# Fields that should be excluded from non-static detection
NON_STATIC_FIELD_EXCEPTIONS: Set[str] = {
    "processing_instance_count"
}
```

This allows for fine-tuned control over field classification, handling special cases where pattern matching would produce incorrect results.

## Core Enumerations

### CategoryType Enum

Defines the simplified field categorization structure:

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

#### Usage Examples

```python
from src.cursus.core.config_fields.constants import CategoryType

# Type-safe field categorization
def categorize_field(field_name: str, field_info: dict) -> CategoryType:
    """Categorize a field based on its characteristics."""
    
    if field_name in SPECIAL_FIELDS_TO_KEEP_SPECIFIC:
        return CategoryType.SPECIFIC
    
    if len(field_info['sources']) > 1 and len(field_info['values']) == 1:
        return CategoryType.SHARED
    
    return CategoryType.SPECIFIC

# Safe comparison without string literals
category = categorize_field("processing_instance_count", field_info)
if category == CategoryType.SHARED:
    print("Field should be placed in shared section")
elif category == CategoryType.SPECIFIC:
    print("Field should be placed in specific section")
```

### MergeDirection Enum

Specifies conflict resolution strategies when merging configuration fields:

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

#### Usage Examples

```python
from src.cursus.core.config_fields.constants import MergeDirection

def merge_configurations(source: dict, target: dict, 
                        direction: MergeDirection = MergeDirection.PREFER_SOURCE) -> dict:
    """Merge two configuration dictionaries with conflict resolution."""
    
    result = target.copy()
    
    for key, source_value in source.items():
        if key not in result:
            # Key only in source, add it
            result[key] = source_value
        else:
            target_value = result[key]
            
            if source_value != target_value:
                # Handle conflict based on direction
                if direction == MergeDirection.PREFER_SOURCE:
                    result[key] = source_value
                    print(f"Conflict on {key}: using source value {source_value}")
                elif direction == MergeDirection.PREFER_TARGET:
                    # Keep target value (no change needed)
                    print(f"Conflict on {key}: keeping target value {target_value}")
                elif direction == MergeDirection.ERROR_ON_CONFLICT:
                    raise ValueError(f"Conflict on key {key}: source={source_value}, target={target_value}")
    
    return result

# Usage examples
source_config = {"field1": "value1", "field2": "value2"}
target_config = {"field1": "different_value", "field3": "value3"}

# Prefer source values in conflicts
merged = merge_configurations(source_config, target_config, MergeDirection.PREFER_SOURCE)
# Result: {"field1": "value1", "field2": "value2", "field3": "value3"}

# Prefer target values in conflicts
merged = merge_configurations(source_config, target_config, MergeDirection.PREFER_TARGET)
# Result: {"field1": "different_value", "field2": "value2", "field3": "value3"}

# Error on conflicts
try:
    merged = merge_configurations(source_config, target_config, MergeDirection.ERROR_ON_CONFLICT)
except ValueError as e:
    print(f"Merge failed: {e}")
```

### SerializationMode Enum

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

#### Usage Examples

```python
from src.cursus.core.config_fields.constants import SerializationMode
from src.cursus.core.config_fields.type_aware_config_serializer import TypeAwareConfigSerializer

# Create serializers with different modes
preserve_serializer = TypeAwareConfigSerializer(mode=SerializationMode.PRESERVE_TYPES)
simple_serializer = TypeAwareConfigSerializer(mode=SerializationMode.SIMPLE_JSON)
custom_serializer = TypeAwareConfigSerializer(mode=SerializationMode.CUSTOM_FIELDS)

# Example configuration object
from datetime import datetime
from enum import Enum

class ProcessingMode(Enum):
    BATCH = "batch"
    STREAMING = "streaming"

config = {
    "created_at": datetime.now(),
    "processing_mode": ProcessingMode.BATCH,
    "batch_size": 32,
    "learning_rate": 0.001
}

# Serialize with type preservation
preserved = preserve_serializer.serialize(config)
print("With type preservation:")
print(preserved)
# Output includes type metadata for datetime and enum

# Serialize as simple JSON
simple = simple_serializer.serialize(config)
print("Simple JSON:")
print(simple)
# Output: plain JSON without type metadata

# Serialize with custom field handling
custom = custom_serializer.serialize(config)
print("Custom field handling:")
print(custom)
# Output: selective type preservation based on field types
```

## Type Mapping Dictionary

Mapping from Python data structure types to their serialized representation names:

```python
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
    "Path": "path"
}
```

### Usage in Serialization

The type mapping is used by the type-aware serializer to preserve type information:

```python
from src.cursus.core.config_fields.constants import TYPE_MAPPING

def serialize_with_type_info(value: Any) -> dict:
    """Serialize a value with type information."""
    
    value_type = type(value).__name__
    
    if value_type in TYPE_MAPPING:
        return {
            "__type_info__": TYPE_MAPPING[value_type],
            "value": serialize_value(value)
        }
    
    # Handle special cases
    if hasattr(value, '__module__') and hasattr(value, '__name__'):
        # Custom class
        return {
            "__type_info__": "custom_class",
            "__module__": value.__module__,
            "__class__": value.__name__,
            "value": serialize_value(value)
        }
    
    # Default serialization
    return serialize_value(value)

# Example usage
from datetime import datetime
from pathlib import Path

# Serialize datetime
dt_serialized = serialize_with_type_info(datetime.now())
# Result: {"__type_info__": "datetime", "value": "2025-09-07T08:21:00.000000"}

# Serialize Path
path_serialized = serialize_with_type_info(Path("/tmp/data"))
# Result: {"__type_info__": "path", "value": "/tmp/data"}

# Serialize list
list_serialized = serialize_with_type_info([1, 2, 3])
# Result: {"__type_info__": "list", "value": [1, 2, 3]}
```

## Field Classification Logic

### Special Field Detection

The constants are used in field classification logic:

```python
from src.cursus.core.config_fields.constants import (
    SPECIAL_FIELDS_TO_KEEP_SPECIFIC,
    NON_STATIC_FIELD_PATTERNS,
    NON_STATIC_FIELD_EXCEPTIONS
)

def is_special_field(field_name: str, field_value: Any) -> bool:
    """Determine if a field should be treated as special."""
    
    # Check against known special fields
    if field_name in SPECIAL_FIELDS_TO_KEEP_SPECIFIC:
        return True
    
    # Check for complex nested structures
    if isinstance(field_value, dict) and any(isinstance(v, (dict, list)) for v in field_value.values()):
        return True
    
    # Check for Pydantic models
    if hasattr(field_value, '__class__') and hasattr(field_value.__class__, 'model_fields'):
        return True
    
    return False

def is_static_field(field_name: str, field_value: Any) -> bool:
    """Determine if a field is likely static based on name and value patterns."""
    
    # Fields in the exceptions list are considered static
    if field_name in NON_STATIC_FIELD_EXCEPTIONS:
        return True
    
    # Special fields are never static
    if field_name in SPECIAL_FIELDS_TO_KEEP_SPECIFIC:
        return False
    
    # Check name patterns that suggest non-static fields
    for pattern in NON_STATIC_FIELD_PATTERNS:
        if pattern in field_name:
            return False
    
    # Check complex values
    if isinstance(field_value, dict) and len(field_value) > 3:
        return False
    if isinstance(field_value, list) and len(field_value) > 5:
        return False
    
    # Default to static
    return True

# Example usage
print(is_special_field("image_uri", "123456789012.dkr.ecr.us-west-2.amazonaws.com/my-image:latest"))
# Output: True

print(is_static_field("processing_instance_count", 1))
# Output: True (due to exception)

print(is_static_field("input_data_path", "/opt/ml/processing/input/data"))
# Output: False (matches "input_" pattern)
```

## Integration with Configuration Components

### Configuration Field Categorizer Integration

The categorizer uses these constants for field classification:

```python
from src.cursus.core.config_fields.constants import CategoryType, SPECIAL_FIELDS_TO_KEEP_SPECIFIC

class ConfigFieldCategorizer:
    def _categorize_field(self, field_name: str) -> CategoryType:
        """Categorize a field using the constants."""
        
        # Rule 1: Special fields always go to specific sections
        if field_name in SPECIAL_FIELDS_TO_KEEP_SPECIFIC:
            return CategoryType.SPECIFIC
        
        # Additional categorization logic...
        
        return CategoryType.SHARED
```

### Configuration Merger Integration

The merger uses merge directions for conflict resolution:

```python
from src.cursus.core.config_fields.constants import MergeDirection

class ConfigMerger:
    @classmethod
    def merge_with_direction(cls, source: dict, target: dict, 
                           direction: MergeDirection = MergeDirection.PREFER_SOURCE) -> dict:
        """Merge configurations using the specified direction."""
        # Implementation uses the enum for type-safe operations
        pass
```

### Type-Aware Serializer Integration

The serializer uses serialization modes and type mappings:

```python
from src.cursus.core.config_fields.constants import SerializationMode, TYPE_MAPPING

class TypeAwareConfigSerializer:
    def __init__(self, mode: SerializationMode = SerializationMode.PRESERVE_TYPES):
        self.mode = mode
        self.type_mapping = TYPE_MAPPING
    
    def serialize(self, value: Any) -> Any:
        """Serialize using the configured mode and type mappings."""
        # Implementation uses the constants for consistent behavior
        pass
```

## Design Rationale

### Why Enums Instead of String Constants?

The module uses enums instead of string constants for several reasons:

1. **Type Safety**: Enums provide compile-time type checking
2. **IDE Support**: Better autocomplete and refactoring support
3. **Error Prevention**: Prevents typos in string literals
4. **Extensibility**: Easy to add new values without breaking existing code

### Why Sets for Field Patterns?

Sets are used for field patterns because:

1. **Performance**: O(1) lookup time for membership testing
2. **Uniqueness**: Automatically prevents duplicate patterns
3. **Clarity**: Clear intent that these are collections of unique items

### Why Centralized Constants?

Centralized constants provide:

1. **Single Source of Truth**: All components use the same definitions
2. **Maintainability**: Changes only need to be made in one place
3. **Consistency**: Ensures consistent behavior across the system
4. **Documentation**: Clear documentation of all constants in one place

## Usage Best Practices

### Import Patterns

```python
# Import specific constants and enums
from src.cursus.core.config_fields.constants import (
    CategoryType,
    MergeDirection,
    SerializationMode,
    SPECIAL_FIELDS_TO_KEEP_SPECIFIC
)

# Use type hints for better code documentation
def process_field(field_name: str, category: CategoryType) -> None:
    """Process a field based on its category."""
    if category == CategoryType.SHARED:
        handle_shared_field(field_name)
    elif category == CategoryType.SPECIFIC:
        handle_specific_field(field_name)
```

### Enum Comparison

```python
# Correct: Use enum values for comparison
if merge_direction == MergeDirection.PREFER_SOURCE:
    use_source_value()

# Incorrect: Don't use string literals
# if merge_direction == "prefer_source":  # This will fail
```

### Pattern Matching

```python
# Efficient pattern checking using sets
def matches_non_static_pattern(field_name: str) -> bool:
    """Check if field name matches any non-static pattern."""
    return any(pattern in field_name for pattern in NON_STATIC_FIELD_PATTERNS)

# Exception handling
def is_non_static_exception(field_name: str) -> bool:
    """Check if field is an exception to non-static rules."""
    return field_name in NON_STATIC_FIELD_EXCEPTIONS
```

## Testing Support

The constants module supports comprehensive testing:

```python
import unittest
from src.cursus.core.config_fields.constants import (
    CategoryType, MergeDirection, SerializationMode,
    SPECIAL_FIELDS_TO_KEEP_SPECIFIC, NON_STATIC_FIELD_PATTERNS
)

class TestConstants(unittest.TestCase):
    
    def test_category_type_enum(self):
        """Test CategoryType enum values."""
        self.assertIsInstance(CategoryType.SHARED, CategoryType)
        self.assertIsInstance(CategoryType.SPECIFIC, CategoryType)
        self.assertNotEqual(CategoryType.SHARED, CategoryType.SPECIFIC)
    
    def test_merge_direction_enum(self):
        """Test MergeDirection enum values."""
        self.assertIsInstance(MergeDirection.PREFER_SOURCE, MergeDirection)
        self.assertIsInstance(MergeDirection.PREFER_TARGET, MergeDirection)
        self.assertIsInstance(MergeDirection.ERROR_ON_CONFLICT, MergeDirection)
    
    def test_special_fields_set(self):
        """Test special fields set contains expected values."""
        self.assertIn("image_uri", SPECIAL_FIELDS_TO_KEEP_SPECIFIC)
        self.assertIn("hyperparameters", SPECIAL_FIELDS_TO_KEEP_SPECIFIC)
        self.assertIsInstance(SPECIAL_FIELDS_TO_KEEP_SPECIFIC, set)
    
    def test_non_static_patterns(self):
        """Test non-static field patterns."""
        self.assertIn("_path", NON_STATIC_FIELD_PATTERNS)
        self.assertIn("input_", NON_STATIC_FIELD_PATTERNS)
        self.assertIsInstance(NON_STATIC_FIELD_PATTERNS, set)
    
    def test_serialization_mode_enum(self):
        """Test SerializationMode enum values."""
        self.assertIsInstance(SerializationMode.PRESERVE_TYPES, SerializationMode)
        self.assertIsInstance(SerializationMode.SIMPLE_JSON, SerializationMode)
        self.assertIsInstance(SerializationMode.CUSTOM_FIELDS, SerializationMode)
```

## Performance Considerations

### Set-Based Lookups

The use of sets for field patterns provides O(1) lookup performance:

```python
# Efficient membership testing
if field_name in SPECIAL_FIELDS_TO_KEEP_SPECIFIC:  # O(1)
    handle_special_field(field_name)

# Pattern matching with early termination
def matches_pattern(field_name: str) -> bool:
    for pattern in NON_STATIC_FIELD_PATTERNS:  # Short-circuits on first match
        if pattern in field_name:
            return True
    return False
```

### Enum Performance

Enums provide efficient comparison operations:

```python
# Enum comparisons are optimized
if category == CategoryType.SHARED:  # Fast identity comparison
    process_shared_field()
```

## Related Documentation

### Core Dependencies
- [Configuration Field Categorizer](config_field_categorizer.md): Uses constants for field classification
- [Configuration Merger](../config_field/config_merger.md): Uses merge directions for conflict resolution
- [Type-Aware Configuration Serializer](type_aware_config_serializer.md): Uses serialization modes and type mappings

### Integration Points
- [Configuration Class Detector](config_class_detector.md): May use constants for field detection patterns
- [Configuration Class Store](config_class_store.md): May use constants for registration patterns

### Base Classes
- [Base Enums](../base/enums.md): Related enumeration definitions for the core system
- [Configuration Base](../base/config_base.md): Uses constants for field categorization

### System Overview
- [Configuration Fields Overview](README.md): System overview and integration

---
tags:
  - code
  - core
  - config_fields
  - constants
  - enums
keywords:
  - CategoryType
  - MergeDirection
  - SerializationMode
  - SPECIAL_FIELDS_TO_KEEP_SPECIFIC
  - NON_STATIC_FIELD_PATTERNS
  - TYPE_MAPPING
topics:
  - configuration management
  - type safety
  - field categorization
language: python
date of note: 2025-09-07
---

# Configuration Fields Constants

Shared constants and enums for config field management, implementing the Type-Safe Specifications principle.

## Overview

The `constants` module contains constants and enums used throughout the config_field_manager package, implementing the Type-Safe Specifications principle. This module provides centralized definitions for field patterns, categorization types, merge behaviors, and serialization modes used across the configuration management system.

The module defines several key enumerations that ensure type safety and consistency across the system, including field category types, merge directions for conflict resolution, and serialization modes for controlling type preservation behavior.

## Classes and Methods

### Enums
- [`CategoryType`](#categorytype) - Enumeration of field category types
- [`MergeDirection`](#mergedirection) - Enumeration of merge directions
- [`SerializationMode`](#serializationmode) - Enumeration of serialization modes

### Constants
- [`SPECIAL_FIELDS_TO_KEEP_SPECIFIC`](#special_fields_to_keep_specific) - Set of special fields that should always be kept in specific sections
- [`NON_STATIC_FIELD_PATTERNS`](#non_static_field_patterns) - Set of patterns indicating non-static fields
- [`NON_STATIC_FIELD_EXCEPTIONS`](#non_static_field_exceptions) - Set of fields excluded from non-static detection
- [`TYPE_MAPPING`](#type_mapping) - Mapping from data structure types to serialized names

## API Reference

### CategoryType

_class_ cursus.core.config_fields.constants.CategoryType

Enumeration of field category types for the simplified structure. Implementing the Type-Safe Specifications principle by using an enum instead of string literals.

**Values:**
- **SHARED** – Fields shared across all configs
- **SPECIFIC** – Fields specific to certain configs

```python
from cursus.core.config_fields.constants import CategoryType

# Use in categorization logic
if category == CategoryType.SHARED:
    print("Field is shared")
elif category == CategoryType.SPECIFIC:
    print("Field is specific")
```

### MergeDirection

_class_ cursus.core.config_fields.constants.MergeDirection

Enumeration of merge directions. Specifies the direction to resolve conflicts when merging fields.

**Values:**
- **PREFER_SOURCE** – Use source value in case of conflict
- **PREFER_TARGET** – Use target value in case of conflict
- **ERROR_ON_CONFLICT** – Raise an error on conflict

```python
from cursus.core.config_fields.constants import MergeDirection

# Use in merge operations
result = ConfigMerger.merge_with_direction(
    source_dict, 
    target_dict, 
    MergeDirection.PREFER_SOURCE
)
```

### SerializationMode

_class_ cursus.core.config_fields.constants.SerializationMode

Enumeration of serialization modes. Controls the behavior of the serializer with respect to type metadata.

**Values:**
- **PRESERVE_TYPES** – Preserve type information in serialized output
- **SIMPLE_JSON** – Convert to plain JSON without type information
- **CUSTOM_FIELDS** – Only preserve types for certain fields

```python
from cursus.core.config_fields.constants import SerializationMode
from cursus.core.config_fields.type_aware_config_serializer import TypeAwareConfigSerializer

# Create serializer with specific mode
serializer = TypeAwareConfigSerializer(mode=SerializationMode.PRESERVE_TYPES)
```

### SPECIAL_FIELDS_TO_KEEP_SPECIFIC

Set of special fields that should always be kept in specific sections.

**Type:** _Set[str]_

**Value:** {"image_uri", "script_name", "output_path", "input_path", "model_path", "hyperparameters", "instance_type", "job_name_prefix", "output_schema"}

```python
from cursus.core.config_fields.constants import SPECIAL_FIELDS_TO_KEEP_SPECIFIC

# Check if field is special
if field_name in SPECIAL_FIELDS_TO_KEEP_SPECIFIC:
    print(f"Field {field_name} should be kept in specific section")
```

### NON_STATIC_FIELD_PATTERNS

Set of patterns that indicate a field is likely non-static.

**Type:** _Set[str]_

**Value:** {"_names", "input_", "output_", "_specific", "batch_count", "item_count", "record_count", "instance_type_count", "_path", "_uri"}

```python
from cursus.core.config_fields.constants import NON_STATIC_FIELD_PATTERNS

# Check if field matches non-static patterns
is_non_static = any(pattern in field_name for pattern in NON_STATIC_FIELD_PATTERNS)
```

### NON_STATIC_FIELD_EXCEPTIONS

Set of fields that should be excluded from non-static detection.

**Type:** _Set[str]_

**Value:** {"processing_instance_count"}

```python
from cursus.core.config_fields.constants import NON_STATIC_FIELD_EXCEPTIONS

# Check if field is exception to non-static rules
if field_name in NON_STATIC_FIELD_EXCEPTIONS:
    print(f"Field {field_name} is treated as static despite matching patterns")
```

### TYPE_MAPPING

Mapping from data structure types to their serialized names.

**Type:** _Dict[str, str]_

**Value:** {"dict": "dict", "list": "list", "tuple": "tuple", "set": "set", "frozenset": "frozenset", "BaseModel": "pydantic_model", "Enum": "enum", "datetime": "datetime", "Path": "path"}

```python
from cursus.core.config_fields.constants import TYPE_MAPPING

# Use in serialization
type_info = TYPE_MAPPING.get("datetime", "unknown")
serialized_data = {
    "__type_info__": type_info,
    "value": datetime_obj.isoformat()
}
```

## Related Documentation

- [Configuration Field Categorizer](config_field_categorizer.md) - Uses CategoryType and field patterns for categorization
- [Configuration Merger](config_merger.md) - Uses MergeDirection for conflict resolution
- [Type-Aware Configuration Serializer](type_aware_config_serializer.md) - Uses SerializationMode and TYPE_MAPPING
- [Configuration Fields Overview](README.md) - System overview and integration

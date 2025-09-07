---
tags:
  - code
  - core
  - config_fields
  - serialization
  - type_preservation
keywords:
  - TypeAwareConfigSerializer
  - serialize_config
  - deserialize_config
  - type preservation
  - circular reference handling
  - three-tier pattern
topics:
  - configuration management
  - serialization
  - type safety
language: python
date of note: 2025-09-07
---

# Type-Aware Configuration Serializer

Type-aware serializer for configuration objects that preserves type information during serialization, allowing for proper reconstruction of objects during deserialization.

## Overview

The `type_aware_config_serializer` module provides a serializer that preserves type information during serialization, implementing the Type-Safe Specifications principle. The serializer maintains type information during serialization and uses it for correct instantiation during deserialization.

The module handles complex types including Pydantic models, enums, datetime objects, and nested data structures. It includes advanced circular reference detection and supports the three-tier configuration pattern by intelligently handling essential, system, and derived fields during serialization and deserialization.

## Classes and Methods

### Classes
- [`TypeAwareConfigSerializer`](#typeawareconfigserializer) - Main serializer for handling complex types with type information

### Functions
- [`serialize_config`](#serialize_config) - Serialize a single config object with default settings
- [`deserialize_config`](#deserialize_config) - Deserialize a single config object with default settings

## API Reference

### TypeAwareConfigSerializer

_class_ cursus.core.config_fields.type_aware_config_serializer.TypeAwareConfigSerializer(_config_classes=None_, _mode=SerializationMode.PRESERVE_TYPES_)

Handles serialization and deserialization of complex types with type information. Maintains type information during serialization and uses it for correct instantiation during deserialization, implementing the Type-Safe Specifications principle.

**Parameters:**
- **config_classes** (_Optional[Dict[str, Type]]_) – Optional dictionary mapping class names to class objects
- **mode** (_SerializationMode_) – Serialization mode controlling type preservation behavior

**Class Constants:**
- **MODEL_TYPE_FIELD** (_str_) – Field name for model type ("__model_type__")
- **MODEL_MODULE_FIELD** (_str_) – Field name for model module ("__model_module__")
- **TYPE_INFO_FIELD** (_str_) – Field name for type information ("__type_info__")

```python
from cursus.core.config_fields.type_aware_config_serializer import TypeAwareConfigSerializer
from cursus.core.config_fields.constants import SerializationMode

# Create serializer with type preservation
serializer = TypeAwareConfigSerializer(mode=SerializationMode.PRESERVE_TYPES)

# Create serializer with specific config classes
config_classes = {"MyConfig": MyConfigClass}
serializer = TypeAwareConfigSerializer(config_classes=config_classes)
```

#### serialize

serialize(_val_)

Serialize a value with type information when needed. For configuration objects following the three-tier pattern, this method includes Tier 1 fields (essential user inputs), Tier 2 fields (system inputs with defaults) that aren't None, and Tier 3 fields (derived) via model_dump() if they need to be preserved.

**Parameters:**
- **val** (_Any_) – The value to serialize

**Returns:**
- **Any** – Serialized value suitable for JSON

```python
# Serialize a configuration object
config_obj = MyPipelineConfig(region="us-east-1", instance_type="ml.m5.large")
serialized = serializer.serialize(config_obj)

# Result includes type metadata and field values
print(serialized)
# {
#   "__model_type__": "MyPipelineConfig",
#   "__model_module__": "my.module.path",
#   "region": "us-east-1",
#   "instance_type": "ml.m5.large",
#   "_metadata": {...}
# }
```

#### deserialize

deserialize(_field_data_, _field_name=None_, _expected_type=None_)

Deserialize data with proper type handling.

**Parameters:**
- **field_data** (_Any_) – The serialized data
- **field_name** (_Optional[str]_) – Optional name of the field (for logging)
- **expected_type** (_Optional[Type]_) – Optional expected type

**Returns:**
- **Any** – Deserialized value

```python
# Deserialize back to object
deserialized_obj = serializer.deserialize(serialized)

# Object is properly reconstructed with correct type
assert isinstance(deserialized_obj, MyPipelineConfig)
assert deserialized_obj.region == "us-east-1"
```

#### generate_step_name

generate_step_name(_config_)

Generate a step name for a config, including job type and other distinguishing attributes. This implements job type variant handling by creating distinct step names for different job types (e.g., "CradleDataLoading_training"), which is essential for proper dependency resolution and pipeline variant creation.

**Parameters:**
- **config** (_Any_) – The configuration object

**Returns:**
- **str** – Generated step name with job type and other variants included

```python
# Generate step name with job type variants
training_config = CradleDataLoadConfig(job_type="training")
step_name = serializer.generate_step_name(training_config)
print(step_name)  # "CradleDataLoading_training"

calibration_config = CradleDataLoadConfig(job_type="calibration")
step_name = serializer.generate_step_name(calibration_config)
print(step_name)  # "CradleDataLoading_calibration"
```

### serialize_config

serialize_config(_config_)

Serialize a single config object with default settings. Preserves job type variant information in the step name, ensuring proper dependency resolution between job type variants (training, calibration, etc.).

**Parameters:**
- **config** (_Any_) – Configuration object to serialize

**Returns:**
- **Dict[str, Any]** – Serialized configuration with proper metadata including step name

```python
from cursus.core.config_fields.type_aware_config_serializer import serialize_config

# Simple serialization with default settings
config = MyPipelineConfig(region="us-west-2")
serialized = serialize_config(config)

# Includes metadata with step name
assert "_metadata" in serialized
assert "step_name" in serialized["_metadata"]
```

### deserialize_config

deserialize_config(_data_, _expected_type=None_)

Deserialize a single config object with default settings.

**Parameters:**
- **data** (_Dict[str, Any]_) – Serialized configuration data
- **expected_type** (_Optional[Type]_) – Optional expected type

**Returns:**
- **Any** – Configuration object

```python
from cursus.core.config_fields.type_aware_config_serializer import deserialize_config

# Simple deserialization with default settings
config_obj = deserialize_config(serialized_data)

# With expected type for validation
config_obj = deserialize_config(serialized_data, expected_type=MyPipelineConfig)
```

## Related Documentation

- [Circular Reference Tracker](circular_reference_tracker.md) - Used for advanced circular reference detection
- [Configuration Class Store](config_class_store.md) - Provides class registry for deserialization
- [Configuration Constants](constants.md) - Defines SerializationMode and TYPE_MAPPING
- [Configuration Merger](config_merger.md) - Uses serializer for merge operations
- [Configuration Fields Overview](README.md) - System overview and integration

---
tags:
  - code
  - core
  - type_aware_serializer
  - serialization
  - type_preservation
keywords:
  - type-aware serialization
  - configuration serialization
  - type preservation
  - circular reference handling
  - pydantic model serialization
  - three-tier pattern
  - step name generation
  - deserialization
topics:
  - type-aware serialization
  - configuration object handling
  - circular reference detection
  - step name generation
language: python
date of note: 2025-08-12
---

# Type-Aware Configuration Serializer

## Overview

The `TypeAwareConfigSerializer` handles serialization and deserialization of complex configuration objects while preserving type information. It implements the **Type-Safe Specifications** principle by maintaining type metadata during serialization and using it for correct instantiation during deserialization.

## Class Definition

```python
class TypeAwareConfigSerializer:
    """
    Handles serialization and deserialization of complex types with type information.
    
    Maintains type information during serialization and uses it for correct
    instantiation during deserialization, implementing the Type-Safe Specifications principle.
    """
    
    # Constants for metadata fields - following Single Source of Truth principle
    MODEL_TYPE_FIELD = "__model_type__"
    MODEL_MODULE_FIELD = "__model_module__"
    TYPE_INFO_FIELD = "__type_info__"
    
    def __init__(self, config_classes: Optional[Dict[str, Type]] = None, 
                 mode: SerializationMode = SerializationMode.PRESERVE_TYPES):
        """
        Initialize with optional config classes.
        
        Args:
            config_classes: Optional dictionary mapping class names to class objects
            mode: Serialization mode controlling type preservation behavior
        """
```

## Key Design Principles

### 1. Type-Safe Specifications

The serializer preserves type information using standardized metadata fields:

```python
# Constants for metadata fields - following Single Source of Truth principle
MODEL_TYPE_FIELD = "__model_type__"
MODEL_MODULE_FIELD = "__model_module__"
TYPE_INFO_FIELD = "__type_info__"
```

### 2. Three-Tier Pattern Support

The serializer understands the three-tier configuration pattern and handles each tier appropriately:

```python
# Check if the object has a categorize_fields method (three-tier pattern)
if hasattr(val, 'categorize_fields') and callable(getattr(val, 'categorize_fields')):
    # Get field categories
    categories = val.categorize_fields()
    
    # Add fields from Tier 1 and Tier 2, but not Tier 3 (derived)
    for tier in ['essential', 'system']:
        for field_name in categories.get(tier, []):
            field_value = getattr(val, field_name, None)
            # Skip None values for system fields (Tier 2)
            if tier == 'system' and field_value is None:
                continue
            result[field_name] = self.serialize(field_value)
```

### 3. Circular Reference Detection

The serializer includes advanced circular reference detection and handling:

```python
# Use the CircularReferenceTracker for advanced circular reference detection
self.ref_tracker = CircularReferenceTracker(max_depth=100)
```

## Serialization Modes

The serializer supports different serialization modes:

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

## Core Functionality

### Serialization

The serializer handles various data types with appropriate type preservation:

```python
def serialize(self, val: Any) -> Any:
    """
    Serialize a value with type information when needed.
    
    For configuration objects following the three-tier pattern, this method:
    1. Includes Tier 1 fields (essential user inputs) 
    2. Includes Tier 2 fields (system inputs with defaults) that aren't None
    3. Includes Tier 3 fields (derived) via model_dump() if they need to be preserved
    """
```

#### Basic Type Handling

```python
# Handle None
if val is None:
    return None
    
# Handle basic types that don't need special handling
if isinstance(val, (str, int, float, bool)):
    return val
```

#### DateTime Serialization

```python
# Handle datetime
if isinstance(val, datetime):
    if self.mode == SerializationMode.PRESERVE_TYPES:
        return {
            self.TYPE_INFO_FIELD: TYPE_MAPPING["datetime"],
            "value": val.isoformat()
        }
    return val.isoformat()
```

#### Enum Serialization

```python
# Handle Enum
if isinstance(val, Enum):
    if self.mode == SerializationMode.PRESERVE_TYPES:
        return {
            self.TYPE_INFO_FIELD: TYPE_MAPPING["Enum"],
            "enum_class": f"{val.__class__.__module__}.{val.__class__.__name__}",
            "value": val.value
        }
    return val.value
```

#### Pydantic Model Serialization

```python
# Handle Pydantic models
if isinstance(val, BaseModel):
    # Get class details
    cls = val.__class__
    module_name = cls.__module__
    cls_name = cls.__name__
    
    # Create serialized dict with type metadata
    result = {
        self.MODEL_TYPE_FIELD: cls_name,
        self.MODEL_MODULE_FIELD: module_name,
    }
    
    # Check if the object has a categorize_fields method (three-tier pattern)
    if hasattr(val, 'categorize_fields') and callable(getattr(val, 'categorize_fields')):
        # Handle three-tier pattern serialization
        categories = val.categorize_fields()
        
        # Add fields from Tier 1 and Tier 2, but not Tier 3 (derived)
        for tier in ['essential', 'system']:
            for field_name in categories.get(tier, []):
                field_value = getattr(val, field_name, None)
                # Skip None values for system fields (Tier 2)
                if tier == 'system' and field_value is None:
                    continue
                result[field_name] = self.serialize(field_value)
    else:
        # Fall back to standard serialization for non-three-tier models
        for k, v in val.model_dump().items():
            result[k] = self.serialize(v)
```

### Deserialization

The deserializer reconstructs objects with proper type handling:

```python
def deserialize(self, field_data: Any, field_name: Optional[str] = None, 
                expected_type: Optional[Type] = None) -> Any:
    """
    Deserialize data with proper type handling.
    
    Args:
        field_data: The serialized data
        field_name: Optional name of the field (for logging)
        expected_type: Optional expected type
        
    Returns:
        Deserialized value
    """
```

#### Type Information Processing

```python
# Handle type-info dict - from preserved types
if isinstance(field_data, dict) and self.TYPE_INFO_FIELD in field_data:
    type_info = field_data[self.TYPE_INFO_FIELD]
    value = field_data.get("value")
    
    # Handle each preserved type
    if type_info == TYPE_MAPPING["datetime"]:
        return datetime.fromisoformat(value)
        
    elif type_info == TYPE_MAPPING["Enum"]:
        enum_class_path = field_data.get("enum_class")
        try:
            module_name, class_name = enum_class_path.rsplit(".", 1)
            module = __import__(module_name, fromlist=[class_name])
            enum_class = getattr(module, class_name)
            return enum_class(field_data.get("value"))
        except (ImportError, AttributeError, ValueError) as e:
            self.logger.warning(f"Failed to deserialize enum: {str(e)}")
            return field_data.get("value")  # Fall back to raw value
```

#### Model Deserialization

```python
def _deserialize_model(self, field_data: Dict[str, Any], expected_type: Optional[Type] = None) -> Any:
    """
    Deserialize a model instance.
    
    For three-tier model configurations, this method:
    1. Identifies essential (Tier 1) and system (Tier 2) fields
    2. Passes only these fields to the constructor 
    3. Allows derived fields (Tier 3) to be computed during initialization
    """
    # Get the actual class to use
    type_name = field_data.get(self.MODEL_TYPE_FIELD)
    module_name = field_data.get(self.MODEL_MODULE_FIELD)
    actual_class = self._get_class_by_name(type_name, module_name)
    
    # Remove metadata fields
    filtered_data = {k: v for k, v in field_data.items() 
                   if k not in (self.MODEL_TYPE_FIELD, self.MODEL_MODULE_FIELD)}
                   
    # Recursively deserialize nested models
    for k, v in list(filtered_data.items()):
        # Get nested field type if available
        nested_type = None
        if hasattr(actual_class, 'model_fields') and k in actual_class.model_fields:
            nested_type = actual_class.model_fields[k].annotation
            
        filtered_data[k] = self.deserialize(v, k, nested_type)
        
    # For three-tier pattern classes, only pass fields that are in model_fields (Tier 1 & 2)
    if hasattr(actual_class, 'model_fields'):
        init_kwargs = {
            k: v for k, v in filtered_data.items() 
            if k in actual_class.model_fields and not k.startswith('_')
        }
    else:
        init_kwargs = filtered_data
    
    # Try to use model_validate with strict=False first (more lenient)
    try:
        if hasattr(actual_class, 'model_validate'):
            # Pydantic v2 style
            result = actual_class.model_validate(init_kwargs, strict=False)
            return result
        # Fall back to direct instantiation
        result = actual_class(**init_kwargs)
        return result
    except Exception as e:
        self.logger.error(f"Failed to instantiate {actual_class.__name__}: {str(e)}")
        # Try with model_construct as a last resort (bypass validation)
        if hasattr(actual_class, 'model_construct'):
            result = actual_class.model_construct(**init_kwargs)
            return result
        # Return as plain dict if all instantiation attempts fail
        return filtered_data
```

## Circular Reference Handling

The serializer includes sophisticated circular reference detection:

### Detection During Serialization

```python
# Get object id to detect circular references during serialization
obj_id = id(val)

# Check if this object is already being serialized (circular reference)
if hasattr(self, '_serializing_ids') and obj_id in self._serializing_ids:
    self.logger.warning(f"Circular reference detected during serialization of {cls_name}")
    # Return a minimal representation with type info but no fields
    return {
        self.MODEL_TYPE_FIELD: cls_name,
        self.MODEL_MODULE_FIELD: module_name,
        "_circular_ref": True,
        "_ref_message": "Circular reference detected - fields omitted"
    }

# Mark this object as being serialized
if not hasattr(self, '_serializing_ids'):
    self._serializing_ids = set()
self._serializing_ids.add(obj_id)
```

### Advanced Tracking During Deserialization

```python
# Use the tracker to check for circular references
context = {}
if expected_type:
    try:
        context['expected_type'] = expected_type.__name__
    except (AttributeError, TypeError):
        context['expected_type'] = str(expected_type)

is_circular, error = self.ref_tracker.enter_object(field_data, field_name, context)

if is_circular:
    # Create enhanced placeholder for circular references
    circular_ref_dict = {
        "__circular_ref__": True,
        "field_name": field_name,
        "error": error
    }
    
    # Special handling for specific model types
    if model_type == "DataSourceConfig":
        circular_ref_dict["data_source_name"] = "CIRCULAR_REF"
        circular_ref_dict["data_source_type"] = "MDS"
    
    # Try to create a stub object using model_construct to bypass validation
    if expected_type and hasattr(expected_type, 'model_construct'):
        try:
            return expected_type.model_construct(**circular_ref_dict)
        except Exception as e:
            self.logger.warning(f"Failed to create stub for circular reference: {str(e)}")
    
    return circular_ref_dict
```

## Step Name Generation

The serializer provides intelligent step name generation with job type variant support:

```python
def generate_step_name(self, config: Any) -> str:
    """
    Generate a step name for a config, including job type and other distinguishing attributes.
    
    This implements the job type variant handling described in the July 4, 2025 solution document.
    It creates distinct step names for different job types (e.g., "CradleDataLoading_training"),
    which is essential for proper dependency resolution and pipeline variant creation.
    """
    # First check for step_name_override - highest priority
    if hasattr(config, "step_name_override") and config.step_name_override:
        step_name_override = getattr(config, "step_name_override")
        if step_name_override != config.__class__.__name__:
            return step_name_override
        
    # Get class name
    class_name = config.__class__.__name__
    
    # Try to look up the step name from the registry (primary source of truth)
    base_step = None
    try:
        from ...steps.registry.step_names import CONFIG_STEP_REGISTRY            
        if class_name in CONFIG_STEP_REGISTRY:
            base_step = CONFIG_STEP_REGISTRY[class_name]
    except (ImportError, AttributeError, ModuleNotFoundError):
        pass  # Registry not available
        
    if not base_step:
        try:
            # Fall back to the old behavior if not in registry
            from ..base.config_base import BasePipelineConfig                
            base_step = BasePipelineConfig.get_step_name(class_name)
        except (ImportError, AttributeError, ModuleNotFoundError):
            # If neither registry nor BasePipelineConfig is available, use a simple fallback
            base_step = self._generate_step_name_fallback(class_name)
    
    step_name = base_step
    
    # Append distinguishing attributes - essential for job type variants
    for attr in ("job_type", "data_type", "mode"):
        if hasattr(config, attr):
            val = getattr(config, attr)
            if val is not None:
                step_name = f"{step_name}_{val}"
                
    return step_name
```

### Fallback Step Name Generation

```python
def _generate_step_name_fallback(self, class_name: str) -> str:
    """
    Fallback method to generate step names when registry is not available.
    """
    # Simple conversion: remove "Config" suffix and convert to step name format
    if class_name.endswith("Config"):
        base_name = class_name[:-6]  # Remove "Config"
    else:
        base_name = class_name
        
    # Convert CamelCase to step name format
    return base_name
```

## Class Resolution

The serializer provides flexible class resolution for deserialization:

```python
def _get_class_by_name(self, class_name: str, module_name: Optional[str] = None) -> Optional[Type]:
    """
    Get a class by name, from config_classes or by importing.
    
    Args:
        class_name: Name of the class
        module_name: Optional module to import from
        
    Returns:
        Class or None if not found
    """
    # First check registered classes
    if class_name in self.config_classes:
        return self.config_classes[class_name]
        
    # Try to import from module if provided
    if module_name:
        try:
            self.logger.debug(f"Attempting to import {class_name} from {module_name}")
            module = __import__(module_name, fromlist=[class_name])
            if hasattr(module, class_name):
                return getattr(module, class_name)
        except ImportError as e:
            self.logger.warning(f"Failed to import {class_name} from {module_name}: {str(e)}")
    
    self.logger.warning(f"Class {class_name} not found")
    return None
```

## Convenience Functions

The module provides convenience functions for common operations:

### Configuration Serialization

```python
def serialize_config(config: Any) -> Dict[str, Any]:
    """
    Serialize a single config object with default settings.

    Preserves job type variant information in the step name, ensuring proper
    dependency resolution between job type variants (training, calibration, etc.).
    """
    serializer = TypeAwareConfigSerializer()
    result = serializer.serialize(config)

    # If serialization resulted in a non-dict, wrap it in a dictionary
    if not isinstance(result, dict):
        step_name = serializer.generate_step_name(config) if hasattr(config, "__class__") else "unknown"
        model_type = config.__class__.__name__ if hasattr(config, "__class__") else "unknown"
        model_module = config.__class__.__module__ if hasattr(config, "__class__") else "unknown"
        
        return {
            "__model_type__": model_type,
            "__model_module__": model_module,
            "_metadata": {
                "step_name": step_name,
                "config_type": model_type,
                "serialization_note": "Object could not be fully serialized"
            },
            "value": result
        }

    # Ensure metadata with proper step name is present
    if "_metadata" not in result:
        step_name = serializer.generate_step_name(config)
        result["_metadata"] = {
            "step_name": step_name,
            "config_type": config.__class__.__name__,
        }

    return result
```

### Configuration Deserialization

```python
def deserialize_config(data: Dict[str, Any], expected_type: Optional[Type] = None) -> Any:
    """
    Deserialize a single config object with default settings.
    
    Args:
        data: Serialized configuration data
        expected_type: Optional expected type
        
    Returns:
        Configuration object
    """
    serializer = TypeAwareConfigSerializer()
    return serializer.deserialize(data, expected_type=expected_type)
```

## Usage Examples

### Basic Serialization and Deserialization

```python
from src.cursus.core.config_fields.type_aware_config_serializer import TypeAwareConfigSerializer

# Create serializer
serializer = TypeAwareConfigSerializer()

# Serialize a configuration object
config = MyPipelineConfig(
    pipeline_name="test-pipeline",
    processing_instance_count=2,
    hyperparameters={"max_depth": 6, "eta": 0.3}
)

serialized = serializer.serialize(config)
print(f"Serialized: {serialized}")

# Deserialize back to object
deserialized = serializer.deserialize(serialized, expected_type=MyPipelineConfig)
print(f"Deserialized: {deserialized}")
```

### Using Different Serialization Modes

```python
from src.cursus.core.config_fields.constants import SerializationMode

# Preserve all type information
serializer_preserve = TypeAwareConfigSerializer(mode=SerializationMode.PRESERVE_TYPES)
serialized_with_types = serializer_preserve.serialize(config)

# Simple JSON without type metadata
serializer_simple = TypeAwareConfigSerializer(mode=SerializationMode.SIMPLE_JSON)
serialized_simple = serializer_simple.serialize(config)

print(f"With types: {serialized_with_types}")
print(f"Simple JSON: {serialized_simple}")
```

### Step Name Generation

```python
# Generate step names with job type variants
training_config = MyTrainingConfig(job_type="training", data_type="tabular")
calibration_config = MyTrainingConfig(job_type="calibration", data_type="tabular")

serializer = TypeAwareConfigSerializer()

training_step_name = serializer.generate_step_name(training_config)
calibration_step_name = serializer.generate_step_name(calibration_config)

print(f"Training step: {training_step_name}")      # "MyTraining_training_tabular"
print(f"Calibration step: {calibration_step_name}") # "MyTraining_calibration_tabular"
```

### Handling Complex Data Types

```python
from datetime import datetime
from pathlib import Path
from enum import Enum

class ProcessingMode(Enum):
    BATCH = "batch"
    STREAMING = "streaming"

# Configuration with complex types
config = ComplexConfig(
    created_at=datetime.now(),
    output_path=Path("/tmp/output"),
    processing_mode=ProcessingMode.BATCH,
    nested_config=NestedConfig(value="test")
)

# Serialize with type preservation
serializer = TypeAwareConfigSerializer(mode=SerializationMode.PRESERVE_TYPES)
serialized = serializer.serialize(config)

# All complex types are preserved with metadata
print(f"Serialized complex config: {serialized}")

# Deserialize back to original types
deserialized = serializer.deserialize(serialized, expected_type=ComplexConfig)
assert isinstance(deserialized.created_at, datetime)
assert isinstance(deserialized.output_path, Path)
assert isinstance(deserialized.processing_mode, ProcessingMode)
```

### Convenience Functions

```python
from src.cursus.core.config_fields.type_aware_config_serializer import serialize_config, deserialize_config

# Use convenience functions
serialized = serialize_config(config)
deserialized = deserialize_config(serialized, MyPipelineConfig)

print(f"Step name from metadata: {serialized['_metadata']['step_name']}")
```

## Error Handling

The serializer provides comprehensive error handling:

### Serialization Errors

```python
try:
    serialized = serializer.serialize(config)
except Exception as e:
    # Serialization errors are caught and logged
    # Returns a dict with error info but preserves type information
    return {
        self.MODEL_TYPE_FIELD: cls_name,
        self.MODEL_MODULE_FIELD: module_name,
        "_error": str(e),
        "_serialization_error": True
    }
```

### Deserialization Errors

```python
# Multiple fallback strategies for deserialization failures
try:
    if hasattr(actual_class, 'model_validate'):
        result = actual_class.model_validate(init_kwargs, strict=False)
        return result
    result = actual_class(**init_kwargs)
    return result
except Exception as e:
    self.logger.error(f"Failed to instantiate {actual_class.__name__}: {str(e)}")
    try:
        # Try with model_construct as a last resort (bypass validation)
        if hasattr(actual_class, 'model_construct'):
            result = actual_class.model_construct(**init_kwargs)
            return result
    except Exception as e2:
        self.logger.error(f"Failed to use model_construct: {str(e2)}")
    
    # Return as plain dict if all instantiation attempts fail
    return filtered_data
```

### Circular Reference Errors

```python
if is_circular:
    # Log the detailed error message
    self.logger.warning(error)
    
    # Create enhanced placeholder for circular references
    circular_ref_dict = {
        "__circular_ref__": True,
        "field_name": field_name,
        "error": error
    }
    
    # Try to create a stub object using model_construct to bypass validation
    if expected_type and hasattr(expected_type, 'model_construct'):
        try:
            return expected_type.model_construct(**circular_ref_dict)
        except Exception as e:
            self.logger.warning(f"Failed to create stub for circular reference: {str(e)}")
    
    return circular_ref_dict
```

## Performance Considerations

### Memory Efficiency

- Object IDs are used for circular reference detection during serialization
- Circular reference tracking is scoped to individual operations
- Type information is only preserved when necessary based on serialization mode

### Processing Efficiency

- Early type checking to avoid unnecessary processing
- Fallback strategies for class resolution
- Optimized handling of basic types without metadata overhead

## Integration with Other Components

### Configuration Field Categorizer

The serializer is used by the categorizer for consistent data handling:

```python
from .type_aware_config_serializer import serialize_config

serialized = serialize_config(config)
```

### Configuration Merger

The merger uses the serializer for type-aware operations:

```python
self.serializer = TypeAwareConfigSerializer()
serializer = TypeAwareConfigSerializer(config_classes=config_classes)
result["shared"][field] = serializer.deserialize(value)
```

### Circular Reference Tracker

The serializer integrates with the circular reference tracker:

```python
self.ref_tracker = CircularReferenceTracker(max_depth=100)
is_circular, error = self.ref_tracker.enter_object(field_data, field_name, context)
```

## Related Documentation

- [Configuration Field Categorizer](config_field_categorizer.md): Uses serializer for data handling
- [Configuration Merger](config_merger.md): Integrates with serializer for type-aware merging
- [Configuration Constants](config_constants.md): Defines serialization modes and type mappings
- [Circular Reference Tracker](circular_reference_tracker.md): Provides circular reference detection
- [Configuration Fields Overview](README.md): System overview and integration

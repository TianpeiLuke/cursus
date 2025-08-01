# Config Field Manager Guide

## Overview

The `config_field_manager` package provides a robust system for managing configuration objects throughout their lifecycle:

- Categorizing fields across multiple configurations
- Serializing and deserializing configurations with type awareness
- Merging and saving multiple configuration objects to a single JSON file
- Loading configurations from previously saved JSON files

This guide explains the architecture, components, and usage patterns to help you effectively work with the system.

## Architecture

The system follows a modular, class-based architecture that implements several key design principles:

### Single Source of Truth
Field categorization logic is centralized in the `ConfigFieldCategorizer` class, eliminating redundancy and conflicts in categorization decisions.

### Declarative Over Imperative
The system uses a rule-based approach with explicit categorization rules instead of complex procedural logic.

### Type-Safe Specifications
Enum types and proper class structures ensure type safety throughout the system, catching errors at definition time rather than runtime.

### Explicit Over Implicit
All categorization decisions are explicit with clear logging, making the code self-documenting.

## Core Components

### ConfigClassStore
A registry for configuration classes that enables type-aware serialization and deserialization. Used as a decorator to register classes.

### TypeAwareConfigSerializer
Handles serialization and deserialization of configuration objects, preserving type information for proper reconstruction.

### ConfigFieldCategorizer
Analyzes fields across configurations and categorizes them based on explicit rules.

### ConfigMerger
Combines multiple configuration objects into a unified structure based on categorization results.

## Simplified Field Structure

The system uses a simplified structure with just two categories:

```
- shared      # Fields with identical values across all configs
- specific    # Fields unique to specific configs or with different values
```

This flattened structure (compared to the previous nested processing structure) provides:

1. More intuitive understanding of where fields belong
2. Clearer rules for field categorization
3. Simplified loading and saving logic
4. Easier debugging and maintenance

## Field Categorization Rules

The system categorizes fields using these rules (in order of precedence):

1. **Field is special** → Place in `specific`
   - Special fields include those in the `SPECIAL_FIELDS_TO_KEEP_SPECIFIC` list
   - Pydantic models are considered special fields
   - Complex nested structures are considered special fields

2. **Field appears only in one config** → Place in `specific`
   - If a field exists in only one configuration instance

3. **Field has different values across configs** → Place in `specific`
   - If a field has the same name but different values across configs

4. **Field is non-static** → Place in `specific`
   - Fields identified as non-static (runtime values, input/output fields)

5. **Field has identical value across all configs** → Place in `shared`
   - If a field has the same value across all configs and is static

6. **Default case** → Place in `specific`
   - When in doubt, place in specific to ensure proper functioning

## API Reference

### Registering Configuration Classes

```python
from src.config_field_manager import ConfigClassStore, register_config_class

# Option 1: Using ConfigClassStore directly as a decorator
@ConfigClassStore.register
class MyConfig:
    def __init__(self, **kwargs):
        self.field1 = "default_value"
        self.field2 = 123
        for key, value in kwargs.items():
            setattr(self, key, value)

# Option 2: Using the convenience function as a decorator
@register_config_class
class AnotherConfig:
    def __init__(self, **kwargs):
        self.field1 = "default_value"
        self.field2 = 123
        for key, value in kwargs.items():
            setattr(self, key, value)
```

### Serializing and Deserializing Configurations

```python
from src.config_field_manager import serialize_config, deserialize_config

# Create a config object
config = MyConfig(field1="custom_value", field3="extra_field")

# Serialize to a dict (with type metadata)
serialized = serialize_config(config)
print(serialized)
# {
#   "__model_type__": "MyConfig",
#   "__model_module__": "your_module",
#   "field1": "custom_value",
#   "field2": 123,
#   "field3": "extra_field"
# }

# Deserialize back to an object (or dict if class not found)
deserialized = deserialize_config(serialized)
```

### Merging and Saving Configurations

```python
from src.config_field_manager import merge_and_save_configs

# Create multiple config objects
config1 = MyConfig(step_name_override="step1", shared_field="shared_value")
config2 = AnotherConfig(step_name_override="step2", shared_field="shared_value")

# Merge and save to a file
merged = merge_and_save_configs([config1, config2], "output.json")
```

The output file will have this structure:

```json
{
  "metadata": {
    "created_at": "2025-07-17T12:34:56.789012",
    "config_types": {
      "step1": "MyConfig",
      "step2": "AnotherConfig"
    }
  },
  "configuration": {
    "shared": {
      "shared_field": "shared_value"
    },
    "specific": {
      "step1": {
        "field1": "default_value",
        "field2": 123
      },
      "step2": {
        "field1": "default_value",
        "field2": 123
      }
    }
  }
}
```

### Loading Configurations

```python
from src.config_field_manager import load_configs

# Load configs from a file
loaded_configs = load_configs("output.json")

# Access shared fields
shared_value = loaded_configs["shared"]["shared_field"]

# Access specific fields
step1_field1 = loaded_configs["specific"]["step1"]["field1"]
step2_field1 = loaded_configs["specific"]["step2"]["field1"]
```

## Job Type Variants

The system preserves job type variants in step names, which is critical for dependency resolution and pipeline variant creation. When a config has `job_type`, `data_type`, or `mode` attributes, they're automatically appended to the step name.

```python
# This config with job_type "training" and data_type "feature"
config = TrainingConfig(
    step_name_override="training_step",
    job_type="training",
    data_type="feature"
)

# Will produce a step name like "training_step_training_feature" in the output
```

This enables:
1. Different step names for job type variants
2. Proper dependency resolution between steps of the same job type
3. Pipeline variant creation (training-only, calibration-only, etc.)
4. Semantic keyword matching for step specifications

## Special Field Handling

Certain fields are always categorized as specific to ensure proper functionality:

- `image_uri`
- `script_name`
- `output_path`
- `input_path`
- `model_path`
- `hyperparameters`
- `instance_type`
- `job_name_prefix`

Additionally, the system automatically identifies and treats as special:
- Pydantic models
- Complex nested structures (nested dictionaries, lists)

## Common Patterns and Best Practices

### 1. Register All Configuration Classes

Always register your configuration classes with `ConfigClassStore` to ensure proper type handling:

```python
@ConfigClassStore.register
class MyConfig:
    # Your config implementation
```

### 2. Use Descriptive Step Name Overrides

Set descriptive `step_name_override` values to make your configs easy to identify:

```python
config = MyConfig(step_name_override="data_preprocessing")
```

### 3. Leverage Job Type Variants

Use `job_type`, `data_type`, and `mode` attributes to create step variants:

```python
training_config = MyConfig(job_type="training")
evaluation_config = MyConfig(job_type="evaluation")
```

### 4. Centralize Common Fields

Put common fields in a base class to ensure consistent field names and types:

```python
@ConfigClassStore.register
class BaseConfig:
    def __init__(self, **kwargs):
        self.pipeline_name = "default-pipeline"
        self.bucket = "default-bucket"
        # Apply overrides from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

@ConfigClassStore.register
class SpecificConfig(BaseConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.specific_field = "specific_value"
        # Apply overrides from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
```

### 5. Handle Complex Nested Types

For nested configuration objects, register each class type:

```python
@ConfigClassStore.register
class NestedConfig:
    def __init__(self, **kwargs):
        self.nested_field = "nested_value"
        # Apply overrides
        for key, value in kwargs.items():
            setattr(self, key, value)

@ConfigClassStore.register
class MainConfig:
    def __init__(self, **kwargs):
        self.nested_config = NestedConfig()
        # Apply overrides
        for key, value in kwargs.items():
            setattr(self, key, value)
```

## Migration Guide

### From Old to New System

If you're migrating from the old system in `src.pipeline_steps.utils`, follow these steps:

1. **Update Imports**
   ```python
   # Old approach
   from src.pipeline_steps.utils import merge_and_save_configs, load_configs

   # New approach
   from src.config_field_manager import merge_and_save_configs, load_configs
   ```

2. **Register Your Config Classes**
   ```python
   from src.config_field_manager import register_config_class

   @register_config_class
   class MyConfig:
       # Your config class
   ```

3. **Update Field Expectations**
   - The new system uses a simplified structure with just `shared` and `specific` sections
   - The nested `processing` section with `processing_shared` and `processing_specific` is no longer used
   - Ensure your code handles the new structure correctly when accessing fields

4. **Review Special Fields**
   - Check if you have any special fields that should always be categorized as specific
   - If needed, you can extend the `SPECIAL_FIELDS_TO_KEEP_SPECIFIC` list in `src.config_field_manager.constants`

## Troubleshooting

### Common Issues and Solutions

1. **"Class not found" during deserialization**
   - Ensure the class is registered with `ConfigClassStore` before deserializing
   - Check that the class name in the serialized data matches exactly with your registered class

2. **Fields appearing in wrong section**
   - Check the field values across configs - different values will put fields in specific sections
   - Special fields are always placed in specific sections, check if the field is in the special fields list
   - If a field is non-static (e.g., contains "input_", "output_", "_path"), it will go to specific sections

3. **Job type variants not working**
   - Ensure your config class has the proper attributes (`job_type`, `data_type`, or `mode`)
   - Verify that these attributes have non-None values

4. **Missing configuration after loading**
   - Check that the output file structure is correct with shared/specific sections
   - Ensure all required config classes are registered with `ConfigClassStore`

## Conclusion

The `config_field_manager` package provides a robust system for managing configuration fields with clear rules and strong type safety. By following the patterns and practices in this guide, you can effectively leverage the system to create, manipulate, and share configuration objects throughout your pipelines.

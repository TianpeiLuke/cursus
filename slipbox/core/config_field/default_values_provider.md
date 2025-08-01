# DefaultValuesProvider Documentation

## Overview

The `DefaultValuesProvider` is a key component of the three-tier configuration architecture that manages all system inputs (Tier 2). This class provides standardized default values for configuration fields that don't require direct user input but may need administrative customization.

## Purpose

The `DefaultValuesProvider` serves several crucial purposes in the configuration system:

1. Provide sensible default values for Tier 2 (system inputs) fields
2. Ensure consistent settings across pipeline configurations
3. Centralize default value management for easier administration
4. Support conditional defaults based on other configuration values
5. Maintain backward compatibility with existing pipeline configurations

## Implementation

The `DefaultValuesProvider` is implemented as a class with static methods and a central registry of default values:

```python
class DefaultValuesProvider:
    """Provides default values for system inputs (Tier 2)"""
    
    # Default values for system inputs, organized by category
    DEFAULT_VALUES = {
        # Base Model Hyperparameters
        "metric_choices": lambda config: ['f1_score', 'auroc'] if getattr(config, 'is_binary', True) else ['accuracy', 'f1_score'],
        "device": -1,
        "header": "true",
        "batch_size": 32,
        # ... other defaults
    }
    
    @classmethod
    def apply_defaults(cls, config, override_values=None, logger=None):
        """Apply default values to a configuration object"""
        # Implementation...
```

## Key Features

### 1. Default Value Types

The `DefaultValuesProvider` supports two types of default values:

#### Static Values

Simple, fixed values that are applied directly:

```python
"device": -1,
"header": "true",
"batch_size": 32
```

#### Dynamic Values (Lambda Functions)

Values that depend on other configuration settings:

```python
"metric_choices": lambda config: ['f1_score', 'auroc'] if getattr(config, 'is_binary', True) else ['accuracy', 'f1_score']
```

These lambda functions:
- Take the configuration object as input
- Can access any of its attributes
- Return the appropriate default value based on configuration context

### 2. Default Application Process

When defaults are applied to a configuration:

1. The system checks if the field already has a non-None value
2. If not, it applies the default (evaluating lambdas if necessary)
3. The value is set on the configuration object
4. The system tracks which defaults were applied

### 3. Override Mechanism

Administrators can override default values by providing an `override_values` dictionary:

```python
custom_defaults = {
    "processing_instance_type_large": "ml.m5.12xlarge",
    "processing_volume_size": 1000
}

DefaultValuesProvider.apply_defaults(config, override_values=custom_defaults)
```

### 4. Configuration Type Awareness

The provider includes a lookup table for determining appropriate values based on configuration type:

```python
CONFIG_TYPE_ENTRY_POINTS = {
    "TabularPreprocessingConfig": "tabular_preprocess.py",
    "ModelCalibrationConfig": "model_calibration.py",
    # ... other mappings
}
```

## Default Categories

The `DefaultValuesProvider` organizes defaults into logical categories:

1. **Base Model Hyperparameters**
   - `metric_choices`, `device`, `batch_size`, etc.

2. **Framework Settings**
   - `py_version`, `processing_framework_version`

3. **Processing Resources**
   - `processing_instance_type_large`, `processing_volume_size`, etc.

4. **Training Resources**
   - `training_instance_count`, `training_volume_size`, etc.

5. **Inference Resources**
   - `inference_instance_type`

6. **Processing Entry Points**
   - `processing_entry_point`, `training_entry_point`, etc.

7. **Calibration Settings**
   - `calibration_method`, `score_field`, etc.

8. **Model Evaluation Settings**
   - `use_large_processing_instance`, `eval_metric_choices`

9. **Payload Configuration**
   - `max_acceptable_error_rate`, `default_numeric_value`, etc.

10. **Integration Settings**
    - `source_model_inference_content_types`, `source_model_inference_response_types`

## Usage

The `DefaultValuesProvider` is designed to be used after creating configuration objects from essential inputs (Tier 1):

```python
# Create config object with essential inputs
config = ModelHyperparameters(
    tab_field_list=["field1", "field2", "field3"],
    is_binary=True
)

# Apply defaults
DefaultValuesProvider.apply_defaults(config)

# Config now has all Tier 2 fields populated with defaults
assert config.device == -1
assert config.batch_size == 32
assert config.metric_choices == ['f1_score', 'auroc']
```

For multiple configurations:

```python
# Apply defaults to multiple configurations
configs = [model_config, pipeline_config, evaluation_config]
DefaultValuesProvider.apply_defaults_to_multiple(configs)
```

## Design Principles

The `DefaultValuesProvider` implements several key design principles:

### 1. Single Source of Truth

All default values are centralized in one location, ensuring consistency and making updates easier.

### 2. Declarative Over Imperative

Defaults are defined declaratively rather than being scattered throughout procedural code.

### 3. Explicit Over Implicit

All default values and their application logic are explicit and transparent.

### 4. Maintainability and Extensibility

Adding new defaults requires minimal code changes, and the system adapts to new configuration fields automatically.

## Integration with Three-Tier Architecture

Within the three-tier architecture, the `DefaultValuesProvider` implements Tier 2 (System Inputs). The typical workflow is:

1. Collect essential user inputs (Tier 1)
2. Create config objects from essential inputs
3. **Apply system defaults (Tier 2)** using `DefaultValuesProvider`
4. Derive dependent fields (Tier 3) using `FieldDerivationEngine`
5. Generate final configuration

## Error Handling and Logging

The `DefaultValuesProvider` includes comprehensive error handling and logging:

- Gracefully handles failures in callable defaults
- Provides informative log messages about applied defaults
- Tracks which defaults were applied to each configuration

## Future Enhancements

1. **Environment-Specific Defaults**: Support for different defaults based on environment (dev, test, prod)
2. **Version-Specific Defaults**: Default values that adapt based on framework or system versions
3. **Default Profiles**: Predefined sets of defaults for different use cases
4. **Default Suggestions**: AI-assisted recommendations for optimal default values

## Implementation Status

The `DefaultValuesProvider` has been fully implemented as part of Phase 1 of the Essential Inputs Implementation Plan. It includes default values for all identified Tier 2 fields and provides a complete API for applying and customizing defaults.

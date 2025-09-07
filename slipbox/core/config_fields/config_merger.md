---
tags:
  - code
  - core
  - config_fields
  - configuration_merger
  - serialization
keywords:
  - ConfigMerger
  - merge_and_save_configs
  - load_configs
  - configuration merging
  - single source of truth
  - field categorization
topics:
  - configuration management
  - configuration merging
  - serialization
language: python
date of note: 2025-09-07
---

# Configuration Merger

Configuration merger for combining and saving multiple configuration objects, implementing the Single Source of Truth principle.

## Overview

The `config_merger` module provides a merger that combines configuration objects according to their field categorization, implementing the Single Source of Truth principle. The merger uses categorization results to produce properly structured output files with shared and specific sections.

The module implements the Explicit Over Implicit principle by clearly defining merge behavior and providing detailed verification of merged results. It combines multiple configuration objects into a unified output structure that separates shared fields (common across all configurations) from specific fields (unique to individual configurations).

## Classes and Methods

### Classes
- [`ConfigMerger`](#configmerger) - Main merger for combining multiple configuration objects

### Functions
- [`merge_and_save_configs`](#merge_and_save_configs) - Convenience function to merge configs and save to file
- [`load_configs`](#load_configs) - Convenience function to load configs from file

## API Reference

### ConfigMerger

_class_ cursus.core.config_fields.config_merger.ConfigMerger(_config_list_, _processing_step_config_base_class=None_)

Merger for combining multiple configuration objects into a unified output. Uses categorization results to produce properly structured output files. Implements the Explicit Over Implicit principle by clearly defining merge behavior.

**Parameters:**
- **config_list** (_List[Any]_) – List of configuration objects to merge
- **processing_step_config_base_class** (_Optional[type]_) – Optional base class for processing steps

```python
from cursus.core.config_fields.config_merger import ConfigMerger

# Create merger with list of config objects
configs = [training_config, processing_config, evaluation_config]
merger = ConfigMerger(configs)

# Merge configurations
merged_result = merger.merge()
```

#### merge

merge()

Merge configurations according to simplified categorization rules.

**Returns:**
- **Dict[str, Any]** – Merged configuration structure with 'shared' and 'specific' sections

```python
# Merge configurations into structured output
merged_config = merger.merge()

# Access shared fields
shared_fields = merged_config['shared']
print(f"Shared fields: {list(shared_fields.keys())}")

# Access specific fields by step
specific_fields = merged_config['specific']
for step_name, fields in specific_fields.items():
    print(f"Step {step_name}: {list(fields.keys())}")
```

#### save

save(_output_file_)

Save merged configuration to a file.

**Parameters:**
- **output_file** (_str_) – Path to output file

**Returns:**
- **Dict[str, Any]** – Merged configuration

```python
# Save merged configuration to file
merged_config = merger.save("output/merged_pipeline_config.json")

# File will contain:
# {
#   "metadata": {
#     "created_at": "2025-09-07T09:23:00",
#     "config_types": {
#       "TrainingStep": "TrainingStepConfig",
#       "ProcessingStep": "ProcessingStepConfig"
#     }
#   },
#   "configuration": {
#     "shared": { ... },
#     "specific": { ... }
#   }
# }
```

#### load

load(_input_file_, _config_classes=None_)

Load a merged configuration from a file. Supports the simplified structure with just shared and specific sections.

**Parameters:**
- **input_file** (_str_) – Path to input file
- **config_classes** (_Optional[Dict[str, type]]_) – Optional mapping of class names to class objects

**Returns:**
- **Dict[str, Any]** – Loaded configuration in the simplified structure

```python
# Load configuration from file
loaded_config = ConfigMerger.load("input/pipeline_config.json")

# Access loaded data
shared_fields = loaded_config['shared']
specific_fields = loaded_config['specific']
```

#### merge_with_direction

merge_with_direction(_source_, _target_, _direction=MergeDirection.PREFER_SOURCE_)

Merge two dictionaries with a specified merge direction.

**Parameters:**
- **source** (_Dict[str, Any]_) – Source dictionary
- **target** (_Dict[str, Any]_) – Target dictionary
- **direction** (_MergeDirection_) – Merge direction for conflict resolution

**Returns:**
- **Dict[str, Any]** – Merged dictionary

```python
from cursus.core.config_fields.constants import MergeDirection

# Merge with conflict resolution
result = ConfigMerger.merge_with_direction(
    source_dict,
    target_dict,
    MergeDirection.PREFER_SOURCE
)

# Handle conflicts by raising errors
try:
    result = ConfigMerger.merge_with_direction(
        source_dict,
        target_dict,
        MergeDirection.ERROR_ON_CONFLICT
    )
except ValueError as e:
    print(f"Merge conflict: {e}")
```

### merge_and_save_configs

merge_and_save_configs(_config_list_, _output_file_, _processing_step_config_base_class=None_)

Convenience function to merge configs and save to file.

**Parameters:**
- **config_list** (_List[Any]_) – List of configuration objects to merge
- **output_file** (_str_) – Path to output file
- **processing_step_config_base_class** (_Optional[type]_) – Optional base class for processing steps

**Returns:**
- **Dict[str, Any]** – Merged configuration

```python
from cursus.core.config_fields.config_merger import merge_and_save_configs

# Simple one-line merge and save
configs = [config1, config2, config3]
merged_result = merge_and_save_configs(configs, "output/pipeline.json")
```

### load_configs

load_configs(_input_file_, _config_classes=None_)

Convenience function to load configs from file.

**Parameters:**
- **input_file** (_str_) – Path to input file
- **config_classes** (_Optional[Dict[str, type]]_) – Optional mapping of class names to class objects

**Returns:**
- **Dict[str, Any]** – Loaded configuration

```python
from cursus.core.config_fields.config_merger import load_configs

# Simple one-line load
loaded_config = load_configs("input/pipeline_config.json")
```

## Related Documentation

- [Configuration Field Categorizer](config_field_categorizer.md) - Provides categorization logic used by merger
- [Type-Aware Configuration Serializer](type_aware_config_serializer.md) - Used for serialization during merge operations
- [Configuration Constants](constants.md) - Defines MergeDirection enum and field patterns
- [Configuration Fields Overview](README.md) - System overview and integration

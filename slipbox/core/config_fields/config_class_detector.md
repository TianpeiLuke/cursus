---
tags:
  - code
  - core
  - config_fields
  - class_detection
  - configuration
keywords:
  - ConfigClassDetector
  - detect_config_classes_from_json
  - JSON configuration
  - class discovery
  - configuration management
  - dynamic loading
topics:
  - configuration management
  - class detection
  - JSON processing
language: python
date of note: 2025-09-07
---

# Configuration Class Detector

Utility module for detecting required configuration classes from JSON configuration files, implementing efficient class loading by analyzing configuration files to determine which configuration classes are required, rather than loading all possible classes.

## Overview

The `config_class_detector` module provides utilities for detecting required configuration classes from JSON configuration files. This module implements the Type Detection and Validation principle by analyzing configuration files to determine which configuration classes are required, rather than loading all possible classes. This approach improves performance and reduces memory usage by only loading the classes that are actually needed for a specific configuration.

The module works by parsing JSON configuration files and extracting class names from metadata sections and model type fields, then loading only those specific classes from the available configuration class registry.

## Classes and Methods

### Classes
- [`ConfigClassDetector`](#configclassdetector) - Main utility class for detecting configuration classes

### Functions
- [`detect_config_classes_from_json`](#detect_config_classes_from_json) - Convenience function for class detection

## API Reference

### ConfigClassDetector

_class_ cursus.core.config_fields.config_class_detector.ConfigClassDetector

Utility class for detecting required configuration classes from JSON files. This class implements the Type Detection and Validation principle by analyzing configuration files to determine which configuration classes are required, rather than loading all possible classes.

**Class Constants:**
- **MODEL_TYPE_FIELD** (_str_) – JSON field name for model type ("__model_type__")
- **METADATA_FIELD** (_str_) – JSON field name for metadata ("metadata")
- **CONFIG_TYPES_FIELD** (_str_) – JSON field name for config types ("config_types")
- **CONFIGURATION_FIELD** (_str_) – JSON field name for configuration ("configuration")
- **SPECIFIC_FIELD** (_str_) – JSON field name for specific configs ("specific")
- **ESSENTIAL_CLASSES** (_List[str]_) – List of essential base classes (["BasePipelineConfig", "ProcessingStepConfigBase"])

#### detect_from_json

detect_from_json(_config_path_)

Detect required config classes from a configuration JSON file.

**Parameters:**
- **config_path** (_str_) – Path to the configuration JSON file

**Returns:**
- **Dict[str, Type]** – Dictionary mapping config class names to config classes

```python
from cursus.core.config_fields.config_class_detector import ConfigClassDetector

# Detect required classes from configuration file
required_classes = ConfigClassDetector.detect_from_json("config/pipeline_config.json")

print(f"Detected {len(required_classes)} required classes:")
for class_name, class_type in required_classes.items():
    print(f"  {class_name}: {class_type}")
```

#### from_config_store

from_config_store(_config_path_)

Alternative implementation that uses only ConfigClassStore for future compatibility.

**Parameters:**
- **config_path** (_str_) – Path to the configuration JSON file

**Returns:**
- **Dict[str, Type]** – Dictionary mapping config class names to config classes

```python
# Future-oriented implementation using ConfigClassStore
config_classes = ConfigClassDetector.from_config_store("config/pipeline_config.json")
```

### detect_config_classes_from_json

detect_config_classes_from_json(_config_path_)

Detect required config classes from a configuration JSON file. This helper function analyzes the configuration file to determine which configuration classes are actually used, rather than loading all possible classes.

**Parameters:**
- **config_path** (_str_) – Path to the configuration JSON file

**Returns:**
- **Dict[str, Type]** – Dictionary mapping config class names to config classes

```python
from cursus.core.config_fields.config_class_detector import detect_config_classes_from_json

# Simplified interface
config_classes = detect_config_classes_from_json("config/my_pipeline.json")

# Use with type-aware serializer
from cursus.core.config_fields.type_aware_config_serializer import TypeAwareConfigSerializer
serializer = TypeAwareConfigSerializer(config_classes=config_classes)
```

## Related Documentation

- [Configuration Class Store](config_class_store.md) - Centralized registry for configuration classes
- [Type-Aware Configuration Serializer](type_aware_config_serializer.md) - Uses detected classes for serialization
- [Configuration Constants](constants.md) - Defines field names and patterns used by the detector
- [Configuration Merger](config_merger.md) - May use detector for class loading
- [Configuration Fields Overview](README.md) - System overview and integration

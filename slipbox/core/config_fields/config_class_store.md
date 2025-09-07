---
tags:
  - code
  - core
  - config_fields
  - class_registry
  - configuration
keywords:
  - ConfigClassStore
  - build_complete_config_classes
  - class registration
  - configuration registry
  - single source of truth
topics:
  - configuration management
  - class registry
  - serialization
language: python
date of note: 2025-09-07
---

# Configuration Class Store

Centralized store for configuration classes used by serialization and deserialization, following the Single Source of Truth principle.

## Overview

The `config_class_store` module provides a centralized registry of configuration classes that can be easily extended. It implements the Single Source of Truth principle by providing a single place to register and retrieve config classes used throughout the system for serialization and deserialization operations.

The registry maintains a centralized mapping of class names to class objects, enabling dynamic class loading and ensuring consistent access to configuration classes across the entire system. Classes can be registered using decorators or direct method calls, and the registry provides methods for retrieving individual classes or all registered classes.

## Classes and Methods

### Classes
- [`ConfigClassStore`](#configclassstore) - Registry of configuration classes for serialization and deserialization

### Functions
- [`build_complete_config_classes`](#build_complete_config_classes) - Build complete mapping of config classes from all sources

## API Reference

### ConfigClassStore

_class_ cursus.core.config_fields.config_class_store.ConfigClassStore

Registry of configuration classes for serialization and deserialization. Maintains a centralized registry of config classes that can be easily extended. Implements the Single Source of Truth principle by providing a single place to register and retrieve config classes.

#### register

register(_config_class=None_)

Register a config class. Can be used as a decorator or called directly with a class.

**Parameters:**
- **config_class** (_Optional[Type[T]]_) – Optional class to register directly

**Returns:**
- **Callable[[Type[T]], Type[T]]** – Decorator function that registers the class or the class itself if provided

```python
from cursus.core.config_fields.config_class_store import ConfigClassStore

# Use as decorator
@ConfigClassStore.register
class MyConfig(BasePipelineConfig):
    pass

# Use as direct function call
ConfigClassStore.register(MyConfig)
```

#### get_class

get_class(_class_name_)

Get a registered class by name.

**Parameters:**
- **class_name** (_str_) – Name of the class

**Returns:**
- **Optional[Type]** – The class or None if not found

```python
# Retrieve a registered class
config_class = ConfigClassStore.get_class("MyConfig")
if config_class:
    instance = config_class()
```

#### get_all_classes

get_all_classes()

Get all registered classes.

**Returns:**
- **Dict[str, Type]** – Mapping of class names to classes

```python
# Get all registered classes
all_classes = ConfigClassStore.get_all_classes()
print(f"Registered classes: {list(all_classes.keys())}")
```

#### register_many

register_many(_*config_classes_)

Register multiple config classes at once.

**Parameters:**
- ***config_classes** (_Type_) – Classes to register

```python
# Register multiple classes at once
ConfigClassStore.register_many(ConfigA, ConfigB, ConfigC)
```

#### clear

clear()

Clear the registry. This is useful for testing or when you need to reset the registry.

```python
# Clear all registered classes (typically for testing)
ConfigClassStore.clear()
```

#### registered_names

registered_names()

Get all registered class names.

**Returns:**
- **Set[str]** – Set of registered class names

```python
# Get names of all registered classes
class_names = ConfigClassStore.registered_names()
print(f"Registered class names: {class_names}")
```

### build_complete_config_classes

build_complete_config_classes()

Build a complete mapping of config classes from all available sources. This function scans for all available config classes in the system, including those from third-party modules, and registers them.

**Returns:**
- **Dict[str, Type]** – Mapping of class names to class objects

```python
from cursus.core.config_fields.config_class_store import build_complete_config_classes

# Get complete mapping of all available config classes
all_config_classes = build_complete_config_classes()

# Use with serializer
from cursus.core.config_fields.type_aware_config_serializer import TypeAwareConfigSerializer
serializer = TypeAwareConfigSerializer(config_classes=all_config_classes)
```

## Related Documentation

- [Configuration Class Detector](config_class_detector.md) - Uses the store to detect required classes
- [Type-Aware Configuration Serializer](type_aware_config_serializer.md) - Primary consumer of the class store
- [Configuration Merger](config_merger.md) - Uses the store for class loading during merge operations
- [Configuration Fields Overview](README.md) - System overview and integration

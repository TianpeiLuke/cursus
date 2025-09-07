---
tags:
  - code
  - core
  - config_fields
  - class_registry
  - configuration
keywords:
  - config class store
  - class registration
  - configuration registry
  - class management
  - dynamic registration
topics:
  - configuration management
  - class registry
  - registration system
language: python
date of note: 2025-09-07
---

# Config Class Store

## Overview

The `ConfigClassStore` provides a centralized registry system for configuration classes. It enables dynamic registration, retrieval, and management of configuration classes used throughout the system.

## Core Components

### ConfigClassStore Class

A singleton-like class that maintains a global registry of configuration classes.

#### Key Methods

- `register(config_class: Optional[Type[T]] = None) -> Callable[[Type[T]], Type[T]]`: Registers a configuration class (can be used as decorator)
- `get_class(class_name: str) -> Optional[Type]`: Retrieves a registered class by name
- `get_all_classes() -> Dict[str, Type]`: Returns all registered classes
- `register_many(*config_classes: Type) -> None`: Registers multiple classes at once
- `clear() -> None`: Clears all registered classes
- `registered_names() -> Set[str]`: Returns names of all registered classes

### Utility Functions

- `build_complete_config_classes() -> Dict[str, Type]`: Builds a complete mapping of all configuration classes

## Usage Patterns

### Class Registration

```python
from cursus.core.config_fields.config_class_store import ConfigClassStore

# Register as decorator
@ConfigClassStore.register
class MyConfig(BaseModel):
    field1: str
    field2: int

# Register explicitly
ConfigClassStore.register(MyConfig)

# Register multiple classes
ConfigClassStore.register_many(Config1, Config2, Config3)
```

### Class Retrieval

```python
# Get specific class
config_class = ConfigClassStore.get_class("MyConfig")

# Get all registered classes
all_classes = ConfigClassStore.get_all_classes()

# Check registered names
names = ConfigClassStore.registered_names()
```

## Implementation Details

### Registration Strategy

1. **Decorator Pattern**: Supports both decorator and explicit registration
2. **Name-based Lookup**: Uses class names as registry keys
3. **Type Safety**: Maintains type information for registered classes
4. **Global Registry**: Provides system-wide access to configuration classes

### Memory Management

- **Singleton Pattern**: Ensures single registry instance
- **Clear Functionality**: Allows registry cleanup for testing
- **Efficient Lookup**: Fast name-based class retrieval

## Integration Points

### With Config Detection

```python
# Used by config class detector
detected_classes = ConfigClassDetector.from_config_store("config.json")
```

### With Serialization

```python
# Used by type-aware serializer
serializer = TypeAwareConfigSerializer(
    config_classes=ConfigClassStore.get_all_classes()
)
```

## Dependencies

- **Type System**: For maintaining type information
- **BaseModel**: For configuration class validation

## Related Components

- [`config_class_detector.md`](config_class_detector.md): Class detection system
- [`type_aware_config_serializer.md`](type_aware_config_serializer.md): Type-aware serialization
- [`config_merger.md`](config_merger.md): Configuration merging functionality

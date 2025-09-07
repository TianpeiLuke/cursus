---
tags:
  - code
  - core
  - config_fields
  - class_registry
  - singleton
keywords:
  - configuration class store
  - class registry
  - singleton pattern
  - class registration
  - centralized management
topics:
  - class registry
  - configuration management
  - singleton design pattern
language: python
date of note: 2025-09-07
---

# Configuration Class Store

## Overview

The `ConfigClassStore` provides a centralized registry for configuration classes used by serialization and deserialization components. It implements the **Single Source of Truth** principle by maintaining a single registry where all configuration classes can be registered and retrieved throughout the system.

## Purpose

This module implements centralized class management by:
- Providing a single registry for all configuration classes
- Supporting both decorator and direct registration patterns
- Enabling dynamic class discovery and retrieval
- Maintaining type safety with generic type variables
- Offering comprehensive registry management operations

## Class Definition

```python
class ConfigClassStore:
    """
    Registry of configuration classes for serialization and deserialization.
    
    Maintains a centralized registry of config classes that can be easily extended.
    Implements the Single Source of Truth principle by providing a single place
    to register and retrieve config classes.
    """
    
    # Single registry instance - implementing Single Source of Truth
    _registry: Dict[str, Type] = {}
    _logger = logging.getLogger(__name__)
```

## Core Registration Methods

### Decorator-Based Registration

The primary registration method supports both decorator and direct usage patterns:

```python
@classmethod
def register(cls, config_class: Optional[Type[T]] = None) -> Callable[[Type[T]], Type[T]]:
    """
    Register a config class.
    
    Can be used as a decorator:
    
    @ConfigClassStore.register
    class MyConfig(BasePipelineConfig):
        ...
    
    Args:
        config_class: Optional class to register directly
        
    Returns:
        Decorator function that registers the class or the class itself if provided
    """
    def _register(cls_to_register: Type[T]) -> Type[T]:
        cls_name = cls_to_register.__name__
        if cls_name in ConfigClassStore._registry and ConfigClassStore._registry[cls_name] != cls_to_register:
            cls._logger.warning(
                f"Class {cls_name} is already registered and is being overwritten. "
                f"This may cause issues if the classes are different."
            )
        ConfigClassStore._registry[cls_name] = cls_to_register
        cls._logger.debug(f"Registered class: {cls_name}")
        return cls_to_register
        
    if config_class is not None:
        # Used directly as a function
        return _register(config_class)
        
    # Used as a decorator
    return _register
```

### Bulk Registration

For registering multiple classes simultaneously:

```python
@classmethod
def register_many(cls, *config_classes: Type) -> None:
    """
    Register multiple config classes at once.
    
    Args:
        *config_classes: Classes to register
    """
    for config_class in config_classes:
        cls.register(config_class)
```

## Class Retrieval Methods

### Single Class Retrieval

```python
@classmethod
def get_class(cls, class_name: str) -> Optional[Type]:
    """
    Get a registered class by name.
    
    Args:
        class_name: Name of the class
        
    Returns:
        The class or None if not found
    """
    class_obj = cls._registry.get(class_name)
    if class_obj is None:
        cls._logger.debug(f"Class not found in registry: {class_name}")
    return class_obj
```

### Complete Registry Access

```python
@classmethod
def get_all_classes(cls) -> Dict[str, Type]:
    """
    Get all registered classes.
    
    Returns:
        dict: Mapping of class names to classes
    """
    return cls._registry.copy()
```

### Registry Metadata

```python
@classmethod
def registered_names(cls) -> Set[str]:
    """
    Get all registered class names.
    
    Returns:
        set: Set of registered class names
    """
    return set(cls._registry.keys())
```

## Registry Management

### Registry Clearing

```python
@classmethod
def clear(cls) -> None:
    """
    Clear the registry.
    
    This is useful for testing or when you need to reset the registry.
    """
    cls._registry.clear()
    cls._logger.debug("Cleared config class registry")
```

## Complete Class Building Function

The module provides a utility function for building complete class mappings:

```python
def build_complete_config_classes() -> Dict[str, Type]:
    """
    Build a complete mapping of config classes from all available sources.
    
    This function scans for all available config classes in the system,
    including those from third-party modules, and registers them.
    
    Returns:
        dict: Mapping of class names to class objects
    """
    # Start with registered classes
    config_classes = ConfigClassStore.get_all_classes()
    
    # TODO: Add logic to scan for classes in pipeline_steps, etc.
    # This is a placeholder for future implementation
    
    return config_classes
```

## Usage Patterns

### Decorator Registration

The most common usage pattern is as a class decorator:

```python
from src.cursus.core.config_fields.config_class_store import ConfigClassStore

@ConfigClassStore.register
class MyPipelineConfig(BasePipelineConfig):
    """Custom pipeline configuration."""
    
    custom_field: str = Field(description="Custom configuration field")
    processing_mode: str = Field(default="batch", description="Processing mode")

@ConfigClassStore.register
class MyTrainingConfig(BasePipelineConfig):
    """Custom training configuration."""
    
    learning_rate: float = Field(default=0.001, description="Learning rate")
    batch_size: int = Field(default=32, description="Training batch size")

# Classes are automatically registered when the module is imported
print(f"Registered classes: {ConfigClassStore.registered_names()}")
# Output: {'MyPipelineConfig', 'MyTrainingConfig', ...}
```

### Direct Registration

For dynamic registration scenarios:

```python
# Register a class directly
ConfigClassStore.register(MyDynamicConfig)

# Register multiple classes at once
ConfigClassStore.register_many(
    ConfigA,
    ConfigB,
    ConfigC
)

# Verify registration
if ConfigClassStore.get_class("MyDynamicConfig"):
    print("MyDynamicConfig successfully registered")
```

### Class Retrieval and Usage

```python
# Retrieve a specific class
config_class = ConfigClassStore.get_class("MyPipelineConfig")
if config_class:
    # Create an instance
    config_instance = config_class(
        author="user",
        bucket="my-bucket",
        role="arn:aws:iam::123456789012:role/MyRole",
        region="NA",
        service_name="my-service",
        pipeline_version="1.0.0"
    )
    print(f"Created instance: {config_instance}")

# Get all registered classes
all_classes = ConfigClassStore.get_all_classes()
print(f"Total registered classes: {len(all_classes)}")

for class_name, class_type in all_classes.items():
    print(f"  {class_name}: {class_type}")
```

### Integration with Type-Aware Serializer

The store integrates seamlessly with the type-aware serializer:

```python
from src.cursus.core.config_fields.type_aware_config_serializer import TypeAwareConfigSerializer

# Create serializer with all registered classes
all_classes = ConfigClassStore.get_all_classes()
serializer = TypeAwareConfigSerializer(config_classes=all_classes)

# Serialize a configuration
config = MyPipelineConfig(...)
serialized = serializer.serialize(config)

# Deserialize back to the correct type
deserialized = serializer.deserialize(serialized, expected_type=MyPipelineConfig)
assert isinstance(deserialized, MyPipelineConfig)
```

## Advanced Usage Examples

### Conditional Registration

```python
# Register classes conditionally
def register_config_classes():
    """Register configuration classes based on environment."""
    
    # Always register base classes
    ConfigClassStore.register(BasePipelineConfig)
    
    # Register specific classes based on availability
    try:
        from ..steps.configs.xgboost_training_config import XGBoostTrainingConfig
        ConfigClassStore.register(XGBoostTrainingConfig)
        print("Registered XGBoost configuration")
    except ImportError:
        print("XGBoost configuration not available")
    
    try:
        from ..steps.configs.pytorch_training_config import PyTorchTrainingConfig
        ConfigClassStore.register(PyTorchTrainingConfig)
        print("Registered PyTorch configuration")
    except ImportError:
        print("PyTorch configuration not available")

# Call during module initialization
register_config_classes()
```

### Registry Inspection and Debugging

```python
def inspect_registry():
    """Inspect the current state of the configuration class registry."""
    
    all_classes = ConfigClassStore.get_all_classes()
    registered_names = ConfigClassStore.registered_names()
    
    print(f"Configuration Class Registry Status:")
    print(f"  Total registered classes: {len(all_classes)}")
    print(f"  Registered names: {sorted(registered_names)}")
    
    # Group by module
    by_module = {}
    for class_name, class_type in all_classes.items():
        module_name = class_type.__module__
        if module_name not in by_module:
            by_module[module_name] = []
        by_module[module_name].append(class_name)
    
    print(f"\nClasses by module:")
    for module_name, class_names in sorted(by_module.items()):
        print(f"  {module_name}:")
        for class_name in sorted(class_names):
            print(f"    - {class_name}")

# Usage
inspect_registry()
```

### Testing Support

```python
import unittest
from src.cursus.core.config_fields.config_class_store import ConfigClassStore

class TestConfigClassStore(unittest.TestCase):
    
    def setUp(self):
        """Clear registry before each test."""
        ConfigClassStore.clear()
    
    def tearDown(self):
        """Clear registry after each test."""
        ConfigClassStore.clear()
    
    def test_registration_and_retrieval(self):
        """Test basic registration and retrieval."""
        
        @ConfigClassStore.register
        class TestConfig:
            pass
        
        # Verify registration
        self.assertIn("TestConfig", ConfigClassStore.registered_names())
        
        # Verify retrieval
        retrieved_class = ConfigClassStore.get_class("TestConfig")
        self.assertEqual(retrieved_class, TestConfig)
    
    def test_bulk_registration(self):
        """Test bulk registration functionality."""
        
        class ConfigA:
            pass
        
        class ConfigB:
            pass
        
        # Register multiple classes
        ConfigClassStore.register_many(ConfigA, ConfigB)
        
        # Verify both are registered
        self.assertIn("ConfigA", ConfigClassStore.registered_names())
        self.assertIn("ConfigB", ConfigClassStore.registered_names())
        
        # Verify retrieval
        self.assertEqual(ConfigClassStore.get_class("ConfigA"), ConfigA)
        self.assertEqual(ConfigClassStore.get_class("ConfigB"), ConfigB)
    
    def test_registry_clearing(self):
        """Test registry clearing functionality."""
        
        @ConfigClassStore.register
        class TempConfig:
            pass
        
        # Verify registration
        self.assertIn("TempConfig", ConfigClassStore.registered_names())
        
        # Clear registry
        ConfigClassStore.clear()
        
        # Verify clearing
        self.assertEqual(len(ConfigClassStore.registered_names()), 0)
        self.assertIsNone(ConfigClassStore.get_class("TempConfig"))
```

## Error Handling and Robustness

### Duplicate Registration Handling

The store handles duplicate registrations gracefully with warnings:

```python
def _register(cls_to_register: Type[T]) -> Type[T]:
    cls_name = cls_to_register.__name__
    if cls_name in ConfigClassStore._registry and ConfigClassStore._registry[cls_name] != cls_to_register:
        cls._logger.warning(
            f"Class {cls_name} is already registered and is being overwritten. "
            f"This may cause issues if the classes are different."
        )
    ConfigClassStore._registry[cls_name] = cls_to_register
    cls._logger.debug(f"Registered class: {cls_name}")
    return cls_to_register
```

### Missing Class Handling

When retrieving non-existent classes:

```python
@classmethod
def get_class(cls, class_name: str) -> Optional[Type]:
    """Get a registered class by name."""
    class_obj = cls._registry.get(class_name)
    if class_obj is None:
        cls._logger.debug(f"Class not found in registry: {class_name}")
    return class_obj
```

### Safe Registry Access

The registry provides safe access patterns:

```python
# Safe class retrieval with fallback
config_class = ConfigClassStore.get_class("MyConfig")
if config_class is None:
    # Fallback to default configuration
    config_class = BasePipelineConfig

# Safe iteration over registry
all_classes = ConfigClassStore.get_all_classes()  # Returns a copy
for class_name, class_type in all_classes.items():
    try:
        # Process class safely
        process_config_class(class_type)
    except Exception as e:
        logger.warning(f"Error processing {class_name}: {e}")
```

## Design Patterns and Principles

### Singleton Registry Pattern

The store implements a singleton registry pattern:

```python
# Single registry instance - implementing Single Source of Truth
_registry: Dict[str, Type] = {}
```

This ensures all components access the same registry instance.

### Type Safety with Generics

The store uses generic type variables for type safety:

```python
# Type variable for generic class registration
T = TypeVar('T')

@classmethod
def register(cls, config_class: Optional[Type[T]] = None) -> Callable[[Type[T]], Type[T]]:
```

This provides compile-time type checking and IDE support.

### Decorator Pattern

The registration method supports the decorator pattern for clean class definition:

```python
@ConfigClassStore.register
class MyConfig(BasePipelineConfig):
    """Configuration class with automatic registration."""
    pass
```

## Performance Considerations

### Memory Efficiency

- **Class References Only**: Stores only class references, not instances
- **Lazy Loading**: Classes are registered when modules are imported
- **Copy on Access**: `get_all_classes()` returns a copy to prevent modification

### Access Performance

- **O(1) Lookup**: Dictionary-based storage provides constant-time access
- **Minimal Overhead**: Registration adds minimal runtime overhead
- **Efficient Iteration**: Set-based name access for efficient iteration

## Integration Points

### ConfigClassDetector Integration

The detector uses the store for class retrieval:

```python
# Get classes from ConfigClassStore
required_classes = {}
for class_name in required_class_names:
    class_type = ConfigClassStore.get_class(class_name)
    if class_type:
        required_classes[class_name] = class_type
```

### Type-Aware Serializer Integration

The serializer uses the store for class resolution:

```python
# Create serializer with registered classes
all_classes = ConfigClassStore.get_all_classes()
serializer = TypeAwareConfigSerializer(config_classes=all_classes)
```

### Build Function Integration

The `build_complete_config_classes()` function starts with registered classes:

```python
def build_complete_config_classes() -> Dict[str, Type]:
    """Build complete class mapping starting with registered classes."""
    # Start with registered classes
    config_classes = ConfigClassStore.get_all_classes()
    
    # Add additional classes from other sources
    # ...
    
    return config_classes
```

## Future Enhancements

### Planned Features

The module includes placeholders for future enhancements:

```python
def build_complete_config_classes() -> Dict[str, Type]:
    """Build a complete mapping of config classes from all available sources."""
    # Start with registered classes
    config_classes = ConfigClassStore.get_all_classes()
    
    # TODO: Add logic to scan for classes in pipeline_steps, etc.
    # This is a placeholder for future implementation
    
    return config_classes
```

### Potential Extensions

1. **Automatic Discovery**: Scan modules for configuration classes
2. **Namespace Support**: Support for namespaced class registration
3. **Version Management**: Track class versions and compatibility
4. **Plugin System**: Support for plugin-based class registration

## Related Documentation

### Core Dependencies
- [Configuration Class Detector](config_class_detector.md): Uses the store for class retrieval
- [Type-Aware Configuration Serializer](type_aware_config_serializer.md): Integrates with store for class resolution
- [Configuration Constants](constants.md): May define registration patterns

### Base Classes
- [Configuration Base](../base/config_base.md): Base class that may be registered in the store
- [Hyperparameters Base](../base/hyperparameters_base.md): Base hyperparameters class

### Integration Points
- [Configuration Merger](../config_field/config_merger.md): May use store for class management
- [Configuration Field Categorizer](../config_field/config_field_categorizer.md): May use store for class analysis

### System Overview
- [Configuration Fields Overview](README.md): System overview and integration

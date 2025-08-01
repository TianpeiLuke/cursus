---
tags:
  - design
  - implementation
  - configuration
  - registry
keywords:
  - config registry
  - configuration classes
  - centralized registration
  - class discovery
  - type resolution
  - single source of truth
  - decorator registration
topics:
  - configuration management
  - registry pattern
  - class resolution
  - dynamic imports
language: python
date of note: 2025-07-31
---

# Config Registry

## Overview

The Config Registry provides a centralized registration system for configuration classes, implementing the Single Source of Truth design principle. This component ensures that configuration classes are easily discoverable, accessible, and consistently used throughout the system.

## Purpose

The purpose of the Config Registry is to:

1. **Provide a central registry** for all configuration classes
2. **Enable class discovery** without direct imports
3. **Support decorators** for easy class registration
4. **Ensure consistent access** to configuration classes
5. **Eliminate redundant class lookups** and references

## Key Components

### 1. Registry Class

The core registry implementation:

```python
class ConfigRegistry:
    """
    Registry of configuration classes for serialization and deserialization.
    
    Maintains a centralized registry of config classes that can be easily extended.
    """
    
    # Single registry instance - implementing Single Source of Truth
    _registry = {}
    
    @classmethod
    def register(cls, config_class):
        """
        Register a config class.
        
        Can be used as a decorator:
        
        @ConfigRegistry.register
        class MyConfig(BasePipelineConfig):
            ...
        
        Args:
            config_class: The class to register
            
        Returns:
            The registered class (for decorator usage)
        """
        cls._registry[config_class.__name__] = config_class
        return config_class
        
    @classmethod
    def get_class(cls, class_name):
        """
        Get a registered class by name.
        
        Args:
            class_name: Name of the class
            
        Returns:
            The class or None if not found
        """
        return cls._registry.get(class_name)
        
    @classmethod
    def get_all_classes(cls):
        """
        Get all registered classes.
        
        Returns:
            dict: Mapping of class names to classes
        """
        return cls._registry.copy()
        
    @classmethod
    def register_many(cls, *config_classes):
        """
        Register multiple config classes at once.
        
        Args:
            *config_classes: Classes to register
        """
        for config_class in config_classes:
            cls.register(config_class)
```

### 2. Decorator for Registration

The registry provides a decorator for easy class registration:

```python
# Alias for decorator usage
register_config_class = ConfigRegistry.register

@register_config_class
class CradleDataLoadConfig(BasePipelineConfig):
    job_type: Optional[str] = None
    region: str
    data_source_type: str
```

### 3. Class Resolution Methods

Methods to find and retrieve registered classes:

```python
def get_config_class_by_name(class_name):
    """
    Get a config class by name.
    
    First checks the registry, then tries to import.
    
    Args:
        class_name: Name of the class
        
    Returns:
        The class or None if not found
    """
    # Check registry first
    cls = ConfigRegistry.get_class(class_name)
    if cls:
        return cls
        
    # Try to import from common locations
    locations = [
        f"src.pipeline_steps.config_{class_name.lower().replace('config', '')}",
        "src.pipeline_steps.config_base",
        "src.pipeline_steps.config_processing_step_base"
    ]
    
    for location in locations:
        try:
            module = importlib.import_module(location)
            if hasattr(module, class_name):
                cls = getattr(module, class_name)
                # Register for future use
                ConfigRegistry.register(cls)
                return cls
        except (ImportError, AttributeError):
            pass
            
    return None
```

### 4. Automatic Discovery

The registry includes methods for automatic discovery of configuration classes:

```python
def discover_and_register_config_classes():
    """
    Discover and register all config classes in the project.
    
    Walks through all Python files in src/pipeline_steps and registers
    classes that inherit from BasePipelineConfig.
    """
    import inspect
    from src.pipeline_steps.config_base import BasePipelineConfig
    
    # Base path for pipeline steps
    base_path = Path(__file__).parent.parent / "src" / "pipeline_steps"
    
    # Find all Python files
    for py_file in base_path.glob("**/*.py"):
        module_name = f"src.pipeline_steps.{py_file.stem}"
        
        try:
            # Import the module
            module = importlib.import_module(module_name)
            
            # Find all classes that inherit from BasePipelineConfig
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BasePipelineConfig) and 
                    obj != BasePipelineConfig):
                    ConfigRegistry.register(obj)
                    logger.debug(f"Auto-registered {name} from {module_name}")
        except (ImportError, AttributeError) as e:
            logger.debug(f"Failed to import {module_name}: {str(e)}")
```

### 5. Integration with Config Class Store

Integration with the ConfigClassStore for backward compatibility:

```python
# ConfigClassStore is an alias for ConfigRegistry with same interface
ConfigClassStore = ConfigRegistry

# Register existing classes from ConfigClassStore
def sync_with_config_class_store():
    """
    Synchronize with ConfigClassStore for backward compatibility.
    """
    for class_name, cls in ConfigClassStore.get_all_classes().items():
        ConfigRegistry.register(cls)
```

## Usage Examples

### 1. Registering a Class

```python
# Using decorator
from src.config_field_manager import register_config_class

@register_config_class
class MyCustomConfig(BasePipelineConfig):
    field1: str
    field2: int = 42
```

### 2. Retrieving a Class

```python
# Direct retrieval
from src.config_field_manager import ConfigRegistry

config_class = ConfigRegistry.get_class("XGBoostTrainingConfig")
config = config_class(hyperparameters={"max_depth": 6})
```

### 3. Getting All Registered Classes

```python
from src.config_field_manager import ConfigRegistry

# Get all classes for config loading
config_classes = ConfigRegistry.get_all_classes()
loaded_configs = load_configs("config.json", config_classes)
```

### 4. Mass Registration

```python
from src.config_field_manager import ConfigRegistry

# Register multiple classes
ConfigRegistry.register_many(
    CradleDataLoadConfig,
    TabularPreprocessingConfig,
    XGBoostTrainingConfig
)
```

## Benefits

1. **Centralized Management**: Single source of truth for configuration classes
2. **Reduced Dependencies**: Modules can access classes without direct imports
3. **Easy Registration**: Decorator syntax makes registration simple
4. **Automatic Discovery**: Can automatically find and register classes
5. **Consistent Access**: Standardized access to configuration classes
6. **Better Testing**: Easier to mock and replace classes in tests

## Implementation Details

### 1. Storage Mechanism

The registry uses a class-level dictionary for storage:

```python
# Single registry instance - implementing Single Source of Truth
_registry = {}
```

This ensures a single instance is shared across the application.

### 2. Registration Process

The registration process is simple:

```python
@classmethod
def register(cls, config_class):
    cls._registry[config_class.__name__] = config_class
    return config_class
```

The class name is used as the key, and the actual class object as the value.

### 3. Discovery Implementation

The discovery mechanism uses Python's inspection capabilities:

```python
import inspect
from pathlib import Path

# Find all classes that inherit from BasePipelineConfig
for name, obj in inspect.getmembers(module):
    if (inspect.isclass(obj) and 
        issubclass(obj, BasePipelineConfig) and 
        obj != BasePipelineConfig):
        ConfigRegistry.register(obj)
```

### 4. Backward Compatibility

For backward compatibility, ConfigClassStore is provided as an alias:

```python
# Legacy name
ConfigClassStore = ConfigRegistry

# Legacy decorator
register_config_class = ConfigRegistry.register
```

## Error Handling

1. **Missing Classes**: `get_class` returns None if class is not found
2. **Import Errors**: Discovery gracefully handles import errors and continues
3. **Duplicate Registration**: Later registrations overwrite earlier ones
4. **Synchronization**: Methods ensure consistency between different registries

## Future Improvements

1. **Namespace Support**: Add support for namespaces to avoid name collisions
2. **Dependency Management**: Track dependencies between configuration classes
3. **Versioning**: Add versioning support for backward compatibility
4. **Registry Events**: Add events for registration and retrieval
5. **Type Checking**: Enhanced type checking during registration

## References

- [Type-Aware Serializer](./type_aware_serializer.md)
- [Config Merger](./config_merger.md)
- [Simplified Config Field Categorization](./simplified_config_field_categorization.md)

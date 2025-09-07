---
tags:
  - code
  - registry
  - hyperparameter_registry
  - model_hyperparameters
  - model_types
keywords:
  - hyperparameter registry
  - model hyperparameters
  - model types
  - hyperparameter classes
  - registry lookup
topics:
  - hyperparameter registry
  - model configuration
  - hyperparameter management
language: python
date of note: 2024-12-07
---

# Hyperparameter Registry

Central registry for all hyperparameter classes that serves as the single source of truth for hyperparameter configuration across the system.

## Overview

The hyperparameter registry module provides a centralized system for managing hyperparameter classes in the pipeline system. It maintains a comprehensive registry that maps model types to their corresponding hyperparameter classes, enabling automatic resolution and validation of hyperparameter configurations.

The registry includes support for base hyperparameter classes, model-specific hyperparameter implementations, module path resolution for dynamic imports, and comprehensive validation and lookup functions. This centralized approach ensures consistency in hyperparameter management across different model types and pipeline configurations.

## Classes and Methods

### Registry Data
- [`HYPERPARAMETER_REGISTRY`](#hyperparameter_registry) - Core hyperparameter registry mapping classes to metadata

### Helper Functions
- [`get_all_hyperparameter_classes`](#get_all_hyperparameter_classes) - Get all registered hyperparameter class names
- [`get_hyperparameter_class_by_model_type`](#get_hyperparameter_class_by_model_type) - Find hyperparameter class for specific model type
- [`get_module_path`](#get_module_path) - Get module path for hyperparameter class
- [`get_all_hyperparameter_info`](#get_all_hyperparameter_info) - Get complete information for all registered classes
- [`validate_hyperparameter_class`](#validate_hyperparameter_class) - Validate hyperparameter class exists in registry

## API Reference

### HYPERPARAMETER_REGISTRY

_dict_ HYPERPARAMETER_REGISTRY

Core hyperparameter registry that maps hyperparameter class names to their metadata including module paths, model types, and descriptions.

**Structure:**
```python
{
    "class_name": {
        "class_name": str,        # Hyperparameter class name
        "module_path": str,       # Python module path for import
        "model_type": str,        # Associated model type (or None for base)
        "description": str        # Human-readable description
    }
}
```

```python
from cursus.registry.hyperparameter_registry import HYPERPARAMETER_REGISTRY

# Access registry directly
print("Available hyperparameter classes:")
for class_name, info in HYPERPARAMETER_REGISTRY.items():
    print(f"  {class_name}: {info['description']}")
    print(f"    Model type: {info['model_type']}")
    print(f"    Module: {info['module_path']}")

# Check specific class
xgb_info = HYPERPARAMETER_REGISTRY["XGBoostHyperparameters"]
print(f"XGBoost hyperparameters: {xgb_info}")
```

### get_all_hyperparameter_classes

get_all_hyperparameter_classes()

Get all registered hyperparameter class names from the registry.

**Returns:**
- **List[str]** – List of all registered hyperparameter class names.

```python
from cursus.registry.hyperparameter_registry import get_all_hyperparameter_classes

# Get all hyperparameter classes
all_classes = get_all_hyperparameter_classes()
print(f"Available hyperparameter classes ({len(all_classes)}):")
for class_name in all_classes:
    print(f"  - {class_name}")

# Use in validation
def is_valid_hyperparameter_class(class_name):
    return class_name in get_all_hyperparameter_classes()

print(f"XGBoostHyperparameters valid: {is_valid_hyperparameter_class('XGBoostHyperparameters')}")
```

### get_hyperparameter_class_by_model_type

get_hyperparameter_class_by_model_type(_model_type_)

Find a hyperparameter class for a specific model type.

**Parameters:**
- **model_type** (_str_) – Model type to find hyperparameter class for (e.g., "xgboost", "pytorch").

**Returns:**
- **Optional[str]** – Hyperparameter class name if found, None otherwise.

```python
from cursus.registry.hyperparameter_registry import get_hyperparameter_class_by_model_type

# Find hyperparameter classes by model type
model_types = ["xgboost", "pytorch", "sklearn", "unknown"]

for model_type in model_types:
    hyperparameter_class = get_hyperparameter_class_by_model_type(model_type)
    if hyperparameter_class:
        print(f"✓ {model_type} → {hyperparameter_class}")
    else:
        print(f"✗ {model_type}: No hyperparameter class found")

# Use in pipeline configuration
def get_hyperparameters_for_model(model_type):
    class_name = get_hyperparameter_class_by_model_type(model_type)
    if class_name:
        module_path = get_module_path(class_name)
        # Dynamic import logic would go here
        return f"Found {class_name} at {module_path}"
    return f"No hyperparameters found for {model_type}"

print(get_hyperparameters_for_model("xgboost"))
```

### get_module_path

get_module_path(_class_name_)

Get the module path for a hyperparameter class to enable dynamic imports.

**Parameters:**
- **class_name** (_str_) – Hyperparameter class name to get module path for.

**Returns:**
- **Optional[str]** – Module path if class exists in registry, None otherwise.

```python
from cursus.registry.hyperparameter_registry import get_module_path

# Get module paths for dynamic imports
hyperparameter_classes = ["XGBoostHyperparameters", "BSMModelHyperparameters", "UnknownClass"]

for class_name in hyperparameter_classes:
    module_path = get_module_path(class_name)
    if module_path:
        print(f"✓ {class_name} → {module_path}")
        
        # Example dynamic import (would need proper error handling)
        try:
            # module = importlib.import_module(module_path)
            # hyperparameter_class = getattr(module, class_name)
            print(f"    Ready for import from {module_path}")
        except Exception as e:
            print(f"    Import would fail: {e}")
    else:
        print(f"✗ {class_name}: Not found in registry")
```

### get_all_hyperparameter_info

get_all_hyperparameter_info()

Get complete information for all registered hyperparameter classes.

**Returns:**
- **Dict[str, Dict[str, str]]** – Complete registry information with all metadata.

```python
from cursus.registry.hyperparameter_registry import get_all_hyperparameter_info

# Get complete hyperparameter information
all_info = get_all_hyperparameter_info()

print("Complete Hyperparameter Registry:")
print("=" * 40)

for class_name, info in all_info.items():
    print(f"\n{class_name}:")
    print(f"  Description: {info['description']}")
    print(f"  Model Type: {info['model_type'] or 'Base/Generic'}")
    print(f"  Module Path: {info['module_path']}")

# Analyze registry statistics
model_types = {}
for info in all_info.values():
    model_type = info['model_type'] or 'Base'
    model_types[model_type] = model_types.get(model_type, 0) + 1

print(f"\nRegistry Statistics:")
print(f"Total classes: {len(all_info)}")
print(f"Model type distribution: {model_types}")
```

### validate_hyperparameter_class

validate_hyperparameter_class(_class_name_)

Validate that a hyperparameter class exists in the registry.

**Parameters:**
- **class_name** (_str_) – Hyperparameter class name to validate.

**Returns:**
- **bool** – True if class exists in registry, False otherwise.

```python
from cursus.registry.hyperparameter_registry import validate_hyperparameter_class

# Validate hyperparameter classes
test_classes = [
    "XGBoostHyperparameters",      # Should be valid
    "BSMModelHyperparameters",     # Should be valid
    "ModelHyperparameters",        # Should be valid (base class)
    "InvalidHyperparameters",      # Should be invalid
    "UnknownClass"                 # Should be invalid
]

print("Hyperparameter Class Validation:")
print("=" * 35)

for class_name in test_classes:
    is_valid = validate_hyperparameter_class(class_name)
    status = "✓" if is_valid else "✗"
    print(f"{status} {class_name}")

# Use in configuration validation
def validate_pipeline_config(config):
    """Validate pipeline configuration includes valid hyperparameter class."""
    hyperparameter_class = getattr(config, 'hyperparameter_class', None)
    
    if not hyperparameter_class:
        return False, "No hyperparameter class specified"
    
    if not validate_hyperparameter_class(hyperparameter_class):
        available = get_all_hyperparameter_classes()
        return False, f"Invalid hyperparameter class '{hyperparameter_class}'. Available: {available}"
    
    return True, "Configuration valid"

# Example usage
class MockConfig:
    hyperparameter_class = "XGBoostHyperparameters"

valid, message = validate_pipeline_config(MockConfig())
print(f"\nConfig validation: {message}")
```

## Usage Examples

### Complete Hyperparameter Registry Workflow

```python
from cursus.registry.hyperparameter_registry import (
    get_all_hyperparameter_classes,
    get_hyperparameter_class_by_model_type,
    get_module_path,
    validate_hyperparameter_class,
    get_all_hyperparameter_info
)

# Get overview of available hyperparameter classes
print("Hyperparameter Registry Overview:")
print("=" * 40)

all_classes = get_all_hyperparameter_classes()
print(f"Total registered classes: {len(all_classes)}")

# Group by model type
all_info = get_all_hyperparameter_info()
by_model_type = {}
for class_name, info in all_info.items():
    model_type = info['model_type'] or 'Base'
    if model_type not in by_model_type:
        by_model_type[model_type] = []
    by_model_type[model_type].append(class_name)

print("\nClasses by Model Type:")
for model_type, classes in by_model_type.items():
    print(f"  {model_type}: {classes}")

# Test model type resolution
print("\nModel Type Resolution:")
test_models = ["xgboost", "pytorch", "tensorflow", "sklearn"]
for model_type in test_models:
    hyperparameter_class = get_hyperparameter_class_by_model_type(model_type)
    if hyperparameter_class:
        module_path = get_module_path(hyperparameter_class)
        print(f"✓ {model_type} → {hyperparameter_class} ({module_path})")
    else:
        print(f"✗ {model_type}: No specific hyperparameter class")
```

### Dynamic Hyperparameter Class Loading

```python
import importlib
from cursus.registry.hyperparameter_registry import (
    get_hyperparameter_class_by_model_type,
    get_module_path,
    validate_hyperparameter_class
)

def load_hyperparameter_class(model_type):
    """Dynamically load hyperparameter class for a model type."""
    
    # Find hyperparameter class for model type
    class_name = get_hyperparameter_class_by_model_type(model_type)
    if not class_name:
        print(f"No hyperparameter class found for model type: {model_type}")
        return None
    
    # Validate class exists in registry
    if not validate_hyperparameter_class(class_name):
        print(f"Hyperparameter class not valid: {class_name}")
        return None
    
    # Get module path for import
    module_path = get_module_path(class_name)
    if not module_path:
        print(f"No module path found for class: {class_name}")
        return None
    
    try:
        # Dynamic import
        print(f"Loading {class_name} from {module_path}")
        module = importlib.import_module(module_path)
        hyperparameter_class = getattr(module, class_name)
        
        print(f"✓ Successfully loaded {class_name}")
        return hyperparameter_class
        
    except ImportError as e:
        print(f"✗ Failed to import module {module_path}: {e}")
        return None
    except AttributeError as e:
        print(f"✗ Class {class_name} not found in module {module_path}: {e}")
        return None

# Test dynamic loading
test_models = ["xgboost", "pytorch"]
for model_type in test_models:
    print(f"\nTesting {model_type}:")
    hyperparameter_class = load_hyperparameter_class(model_type)
    if hyperparameter_class:
        print(f"  Class: {hyperparameter_class}")
        print(f"  MRO: {[cls.__name__ for cls in hyperparameter_class.__mro__]}")
```

### Registry Extension and Validation

```python
from cursus.registry.hyperparameter_registry import (
    HYPERPARAMETER_REGISTRY,
    validate_hyperparameter_class,
    get_all_hyperparameter_classes
)

def register_custom_hyperparameter_class(class_name, module_path, model_type, description):
    """Register a custom hyperparameter class (for demonstration)."""
    
    # Validate inputs
    if not class_name or not module_path:
        raise ValueError("Class name and module path are required")
    
    if class_name in HYPERPARAMETER_REGISTRY:
        print(f"Warning: {class_name} already exists in registry")
        return False
    
    # Add to registry (in practice, this would be done through proper channels)
    new_entry = {
        "class_name": class_name,
        "module_path": module_path,
        "model_type": model_type,
        "description": description
    }
    
    print(f"Would register: {class_name} → {new_entry}")
    return True

def validate_registry_integrity():
    """Validate the integrity of the hyperparameter registry."""
    
    print("Registry Integrity Check:")
    print("=" * 25)
    
    all_info = get_all_hyperparameter_info()
    issues = []
    
    for class_name, info in all_info.items():
        # Check required fields
        required_fields = ['class_name', 'module_path', 'description']
        for field in required_fields:
            if not info.get(field):
                issues.append(f"{class_name}: Missing {field}")
        
        # Check class name consistency
        if info.get('class_name') != class_name:
            issues.append(f"{class_name}: Inconsistent class name in metadata")
        
        # Check module path format
        module_path = info.get('module_path', '')
        if module_path and not module_path.replace('_', '').replace('.', '').isalnum():
            issues.append(f"{class_name}: Invalid module path format: {module_path}")
    
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  ✗ {issue}")
    else:
        print("✓ Registry integrity check passed")
    
    return len(issues) == 0

# Test registry operations
print("Testing Custom Registration:")
register_custom_hyperparameter_class(
    "CustomModelHyperparameters",
    "src.custom.hyperparameters_custom",
    "custom_model",
    "Hyperparameters for custom model implementation"
)

print("\nValidating Registry:")
validate_registry_integrity()

# Show registry statistics
all_classes = get_all_hyperparameter_classes()
print(f"\nRegistry contains {len(all_classes)} hyperparameter classes")
```

### Integration with Pipeline Configuration

```python
from cursus.registry.hyperparameter_registry import (
    get_hyperparameter_class_by_model_type,
    validate_hyperparameter_class,
    get_all_hyperparameter_info
)

class PipelineConfigValidator:
    """Validator for pipeline configurations using hyperparameter registry."""
    
    def __init__(self):
        self.registry_info = get_all_hyperparameter_info()
    
    def validate_model_config(self, model_type, hyperparameter_class=None):
        """Validate model configuration against registry."""
        
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        # If no hyperparameter class specified, try to find one
        if not hyperparameter_class:
            suggested_class = get_hyperparameter_class_by_model_type(model_type)
            if suggested_class:
                results['suggestions'].append(f"Consider using {suggested_class} for {model_type}")
            else:
                results['warnings'].append(f"No specific hyperparameter class found for {model_type}")
            return results
        
        # Validate specified hyperparameter class
        if not validate_hyperparameter_class(hyperparameter_class):
            results['valid'] = False
            results['errors'].append(f"Invalid hyperparameter class: {hyperparameter_class}")
            
            # Suggest alternatives
            available = list(self.registry_info.keys())
            similar = [cls for cls in available if model_type.lower() in cls.lower()]
            if similar:
                results['suggestions'].extend(f"Consider: {cls}" for cls in similar)
            
            return results
        
        # Check if hyperparameter class matches model type
        class_info = self.registry_info[hyperparameter_class]
        class_model_type = class_info.get('model_type')
        
        if class_model_type and class_model_type != model_type:
            results['warnings'].append(
                f"Hyperparameter class {hyperparameter_class} is designed for {class_model_type}, "
                f"but you're using it with {model_type}"
            )
        
        return results
    
    def get_recommendations(self, model_type):
        """Get hyperparameter recommendations for a model type."""
        
        # Direct match
        direct_match = get_hyperparameter_class_by_model_type(model_type)
        if direct_match:
            return {
                'primary': direct_match,
                'alternatives': [],
                'description': self.registry_info[direct_match]['description']
            }
        
        # Find alternatives
        alternatives = []
        for class_name, info in self.registry_info.items():
            if (info.get('model_type') and 
                model_type.lower() in info['model_type'].lower()):
                alternatives.append(class_name)
        
        return {
            'primary': None,
            'alternatives': alternatives,
            'description': f"No direct match for {model_type}"
        }

# Test pipeline configuration validation
validator = PipelineConfigValidator()

test_configs = [
    ("xgboost", "XGBoostHyperparameters"),      # Should be valid
    ("pytorch", "BSMModelHyperparameters"),     # Should be valid
    ("xgboost", "BSMModelHyperparameters"),     # Mismatch warning
    ("sklearn", None),                          # No hyperparameter class
    ("unknown", "InvalidClass")                 # Invalid class
]

print("Pipeline Configuration Validation:")
print("=" * 40)

for model_type, hyperparameter_class in test_configs:
    print(f"\nTesting: {model_type} with {hyperparameter_class}")
    
    results = validator.validate_model_config(model_type, hyperparameter_class)
    
    status = "✓" if results['valid'] else "✗"
    print(f"{status} Valid: {results['valid']}")
    
    for error in results['errors']:
        print(f"  Error: {error}")
    for warning in results['warnings']:
        print(f"  Warning: {warning}")
    for suggestion in results['suggestions']:
        print(f"  Suggestion: {suggestion}")
    
    # Get recommendations
    recommendations = validator.get_recommendations(model_type)
    if recommendations['primary']:
        print(f"  Recommended: {recommendations['primary']}")
    elif recommendations['alternatives']:
        print(f"  Alternatives: {recommendations['alternatives']}")
```

## Related Components

- **[Registry Module](__init__.md)** - Main registry module that exports hyperparameter functions
- **[Step Names](step_names.md)** - Step names registry that may reference hyperparameter classes
- **[Core Base](../core/base/hyperparameters_base.md)** - Base hyperparameter classes
- **[Validation Utils](validation_utils.md)** - Validation utilities that may use hyperparameter registry

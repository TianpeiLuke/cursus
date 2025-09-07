---
tags:
  - code
  - registry
  - hybrid
  - utils
  - utilities
  - conversion
  - validation
keywords:
  - load_registry_module
  - get_step_names_from_module
  - from_legacy_format
  - to_legacy_format
  - convert_registry_dict
  - validate_registry_type
  - validate_step_name
  - validate_workspace_id
  - validate_registry_data
  - format_registry_error
  - RegistryLoadError
topics:
  - registry utilities
  - data conversion
  - validation
  - error handling
language: python
date of note: 2024-12-07
---

# Registry Hybrid Utils

Streamlined utility functions for hybrid registry system without over-engineering, replacing complex utility classes with simple, focused functions.

## Overview

The hybrid utils module provides essential utility functions for the hybrid registry system, featuring simple loading functions for registry modules without complex class hierarchies, conversion functions for legacy format compatibility with optimized field mapping, validation functions using direct validation with enum support, and generic error formatting with template-based messages for consistent error handling.

This module replaces complex utility classes with straightforward functions that focus on essential functionality. The implementation emphasizes simplicity and performance while maintaining full compatibility with legacy registry formats and providing comprehensive validation capabilities.

## Classes and Methods

### Exceptions
- [`RegistryLoadError`](#registryloaderror) - Error loading registry from file

### Loading Functions
- [`load_registry_module`](#load_registry_module) - Load registry module from file
- [`get_step_names_from_module`](#get_step_names_from_module) - Extract STEP_NAMES from loaded module

### Conversion Functions
- [`from_legacy_format`](#from_legacy_format) - Convert legacy STEP_NAMES format to StepDefinition
- [`to_legacy_format`](#to_legacy_format) - Convert StepDefinition to legacy STEP_NAMES format
- [`convert_registry_dict`](#convert_registry_dict) - Convert complete registry dictionary to StepDefinition objects

### Validation Functions
- [`validate_registry_type`](#validate_registry_type) - Validate registry type using enum values
- [`validate_step_name`](#validate_step_name) - Validate step name format
- [`validate_workspace_id`](#validate_workspace_id) - Validate workspace ID format
- [`validate_registry_data`](#validate_registry_data) - Validate registry data using direct validation functions

### Error Formatting Functions
- [`format_registry_error`](#format_registry_error) - Generic error formatter using templates
- [`format_step_not_found_error`](#format_step_not_found_error) - Format step not found error messages
- [`format_registry_load_error`](#format_registry_load_error) - Format registry loading error messages
- [`format_validation_error`](#format_validation_error) - Format validation error messages

## API Reference

### RegistryLoadError

_class_ cursus.registry.hybrid.utils.RegistryLoadError

Error loading registry from file, inheriting from RegistryError.

```python
from cursus.registry.hybrid.utils import RegistryLoadError

try:
    module = load_registry_module("invalid_path.py")
except RegistryLoadError as e:
    print(f"Registry loading failed: {e}")
```

### load_registry_module

load_registry_module(_file_path_)

Loads registry module from file using importlib with proper error handling.

**Parameters:**
- **file_path** (_str_) – Path to the registry file to load.

**Returns:**
- **Any** – Loaded module object with registry definitions.

**Raises:**
- **RegistryLoadError** – If module loading fails or file doesn't exist.

```python
from cursus.registry.hybrid.utils import load_registry_module

# Load registry module
try:
    module = load_registry_module("/path/to/workspace_registry.py")
    print("Registry loaded successfully")
except RegistryLoadError as e:
    print(f"Failed to load registry: {e}")
```

### get_step_names_from_module

get_step_names_from_module(_module_)

Extracts STEP_NAMES dictionary from loaded registry module.

**Parameters:**
- **module** (_Any_) – Loaded registry module object.

**Returns:**
- **Dict[str, Dict[str, Any]]** – STEP_NAMES dictionary from the module, or empty dict if not found.

```python
from cursus.registry.hybrid.utils import load_registry_module, get_step_names_from_module

# Extract step names from module
module = load_registry_module("/path/to/registry.py")
step_names = get_step_names_from_module(module)
print(f"Found {len(step_names)} steps")
```

### from_legacy_format

from_legacy_format(_step_name_, _step_info_, _registry_type="core"_, _workspace_id=None_)

Converts legacy STEP_NAMES format to StepDefinition object with proper type mapping.

**Parameters:**
- **step_name** (_str_) – Name of the step to convert.
- **step_info** (_Dict[str, Any]_) – Legacy step information dictionary.
- **registry_type** (_str_) – Type of registry ("core", "workspace", or "override").
- **workspace_id** (_Optional[str]_) – Workspace identifier for workspace steps.

**Returns:**
- **StepDefinition** – Converted StepDefinition object.

```python
from cursus.registry.hybrid.utils import from_legacy_format

# Convert legacy format
legacy_step = {
    "config_class": "XGBoostTrainingConfig",
    "builder_step_name": "XGBoostTrainingStepBuilder",
    "spec_type": "XGBoostTraining",
    "sagemaker_step_type": "Training"
}

step_def = from_legacy_format(
    step_name="xgboost_training",
    step_info=legacy_step,
    registry_type="core"
)
```

### to_legacy_format

to_legacy_format(_definition_)

Converts StepDefinition to legacy STEP_NAMES format using optimized field mapping.

**Parameters:**
- **definition** (_StepDefinition_) – StepDefinition object to convert.

**Returns:**
- **Dict[str, Any]** – Legacy format dictionary compatible with existing systems.

```python
from cursus.registry.hybrid.utils import to_legacy_format
from cursus.registry.hybrid.models import StepDefinition, RegistryType

# Convert to legacy format
step_def = StepDefinition(
    name="custom_step",
    registry_type=RegistryType.WORKSPACE,
    config_class="CustomStepConfig"
)

legacy_dict = to_legacy_format(step_def)
print(legacy_dict)  # {"config_class": "CustomStepConfig", ...}
```

### convert_registry_dict

convert_registry_dict(_registry_dict_, _registry_type="core"_, _workspace_id=None_)

Converts a complete registry dictionary to StepDefinition objects for batch processing.

**Parameters:**
- **registry_dict** (_Dict[str, Dict[str, Any]]_) – Dictionary of step_name -> step_info.
- **registry_type** (_str_) – Type of registry for all steps.
- **workspace_id** (_Optional[str]_) – Workspace identifier for workspace registries.

**Returns:**
- **Dict[str, StepDefinition]** – Dictionary of step_name -> StepDefinition.

```python
from cursus.registry.hybrid.utils import convert_registry_dict

# Convert entire registry
legacy_registry = {
    "step1": {"config_class": "Step1Config"},
    "step2": {"config_class": "Step2Config"}
}

step_definitions = convert_registry_dict(
    registry_dict=legacy_registry,
    registry_type="workspace",
    workspace_id="my_workspace"
)
```

### validate_registry_type

validate_registry_type(_registry_type_)

Validates registry type using enum values for type safety.

**Parameters:**
- **registry_type** (_str_) – Registry type to validate.

**Returns:**
- **str** – Validated registry type value.

**Raises:**
- **ValueError** – If registry type is not valid.

```python
from cursus.registry.hybrid.utils import validate_registry_type

# Validate registry type
try:
    validated_type = validate_registry_type("workspace")
    print(f"Valid type: {validated_type}")
except ValueError as e:
    print(f"Invalid type: {e}")
```

### validate_step_name

validate_step_name(_step_name_)

Validates step name format ensuring alphanumeric characters with underscores and hyphens.

**Parameters:**
- **step_name** (_str_) – Step name to validate.

**Returns:**
- **str** – Validated and stripped step name.

**Raises:**
- **ValueError** – If step name is empty or contains invalid characters.

```python
from cursus.registry.hybrid.utils import validate_step_name

# Validate step name
try:
    valid_name = validate_step_name("my_custom_step")
    print(f"Valid name: {valid_name}")
except ValueError as e:
    print(f"Invalid name: {e}")
```

### validate_workspace_id

validate_workspace_id(_workspace_id_)

Validates workspace ID format using the same rules as step names.

**Parameters:**
- **workspace_id** (_Optional[str]_) – Workspace ID to validate.

**Returns:**
- **Optional[str]** – Validated workspace ID or None if input is None.

**Raises:**
- **ValueError** – If workspace ID format is invalid.

```python
from cursus.registry.hybrid.utils import validate_workspace_id

# Validate workspace ID
try:
    valid_id = validate_workspace_id("my_workspace")
    print(f"Valid workspace ID: {valid_id}")
except ValueError as e:
    print(f"Invalid workspace ID: {e}")
```

### validate_registry_data

validate_registry_data(_registry_type_, _step_name_, _workspace_id=None_)

Validates complete registry data using direct validation functions.

**Parameters:**
- **registry_type** (_str_) – Registry type to validate.
- **step_name** (_str_) – Step name to validate.
- **workspace_id** (_Optional[str]_) – Optional workspace ID to validate.

**Returns:**
- **bool** – True if all validation passes.

**Raises:**
- **ValueError** – If any validation fails.

```python
from cursus.registry.hybrid.utils import validate_registry_data

# Validate complete registry data
try:
    is_valid = validate_registry_data(
        registry_type="workspace",
        step_name="my_step",
        workspace_id="my_workspace"
    )
    print(f"Data is valid: {is_valid}")
except ValueError as e:
    print(f"Validation failed: {e}")
```

### format_registry_error

format_registry_error(_error_type_, _**kwargs_)

Generic error formatter using templates for consistent error messages.

**Parameters:**
- **error_type** (_str_) – Type of error to format (step_not_found, registry_load, validation, etc.).
- **kwargs** – Template variables for error message formatting.

**Returns:**
- **str** – Formatted error message.

```python
from cursus.registry.hybrid.utils import format_registry_error

# Format step not found error
error_msg = format_registry_error(
    'step_not_found',
    step_name='missing_step',
    workspace_context='my_workspace',
    available_steps=['step1', 'step2']
)
print(error_msg)
```

### format_step_not_found_error

format_step_not_found_error(_step_name_, _workspace_context=None_, _available_steps=None_)

Formats step not found error messages using generic formatter for backward compatibility.

**Parameters:**
- **step_name** (_str_) – Name of the step that was not found.
- **workspace_context** (_Optional[str]_) – Workspace context for the error.
- **available_steps** (_Optional[List[str]]_) – List of available steps for suggestions.

**Returns:**
- **str** – Formatted error message with context and suggestions.

```python
from cursus.registry.hybrid.utils import format_step_not_found_error

# Format step not found error
error_msg = format_step_not_found_error(
    step_name="missing_step",
    workspace_context="my_workspace",
    available_steps=["step1", "step2", "step3"]
)
```

### format_registry_load_error

format_registry_load_error(_registry_path_, _error_details_)

Formats registry loading error messages using generic formatter for backward compatibility.

**Parameters:**
- **registry_path** (_str_) – Path to the registry file that failed to load.
- **error_details** (_str_) – Detailed error information.

**Returns:**
- **str** – Formatted error message with path and details.

```python
from cursus.registry.hybrid.utils import format_registry_load_error

# Format registry load error
error_msg = format_registry_load_error(
    registry_path="/path/to/registry.py",
    error_details="File not found"
)
```

### format_validation_error

format_validation_error(_component_name_, _validation_issues_)

Formats validation error messages using generic formatter for backward compatibility.

**Parameters:**
- **component_name** (_str_) – Name of the component that failed validation.
- **validation_issues** (_List[str]_) – List of validation issues found.

**Returns:**
- **str** – Formatted error message with numbered issue list.

```python
from cursus.registry.hybrid.utils import format_validation_error

# Format validation error
error_msg = format_validation_error(
    component_name="step_definition",
    validation_issues=["Missing config_class", "Invalid step_name format"]
)
```

## Constants and Templates

### LEGACY_FIELDS

List of field names used for legacy format conversion optimization:

```python
LEGACY_FIELDS = [
    'config_class',
    'builder_step_name', 
    'spec_type',
    'sagemaker_step_type',
    'description',
    'framework',
    'job_types',
]
```

### ERROR_TEMPLATES

Dictionary of error message templates for consistent formatting:

```python
ERROR_TEMPLATES = {
    'step_not_found': "Step '{step_name}' not found{context}{suggestions}",
    'registry_load': "Failed to load registry from '{registry_path}': {error_details}",
    'validation': "Validation failed for '{component_name}':{issues}",
    'workspace_not_found': "Workspace '{workspace_id}' not found{suggestions}",
    'conflict_detected': "Step name conflict detected for '{step_name}'{context}",
    'invalid_registry_type': "Invalid registry type '{registry_type}'{suggestions}",
}
```

## Usage Examples

### Complete Registry Loading and Conversion

```python
from cursus.registry.hybrid.utils import (
    load_registry_module,
    get_step_names_from_module,
    convert_registry_dict,
    validate_registry_data
)

# Load and convert registry
try:
    # Load registry module
    module = load_registry_module("/path/to/workspace_registry.py")
    
    # Extract step names
    step_names = get_step_names_from_module(module)
    
    # Convert to StepDefinition objects
    step_definitions = convert_registry_dict(
        registry_dict=step_names,
        registry_type="workspace",
        workspace_id="my_workspace"
    )
    
    # Validate each step
    for step_name, step_def in step_definitions.items():
        validate_registry_data(
            registry_type=step_def.registry_type,
            step_name=step_name,
            workspace_id=step_def.workspace_id
        )
    
    print(f"Successfully loaded and validated {len(step_definitions)} steps")
    
except Exception as e:
    print(f"Registry processing failed: {e}")
```

### Legacy Format Conversion

```python
from cursus.registry.hybrid.utils import from_legacy_format, to_legacy_format

# Convert from legacy to modern format
legacy_step = {
    "config_class": "CustomProcessingConfig",
    "builder_step_name": "CustomProcessingStepBuilder",
    "spec_type": "CustomProcessing",
    "sagemaker_step_type": "Processing",
    "description": "Custom data processing step",
    "framework": "pandas",
    "job_types": ["processing", "transformation"]
}

# Convert to StepDefinition
step_def = from_legacy_format(
    step_name="custom_processing",
    step_info=legacy_step,
    registry_type="workspace",
    workspace_id="data_team"
)

# Convert back to legacy format
converted_back = to_legacy_format(step_def)
print("Conversion successful:", converted_back == legacy_step)
```

### Error Handling and Formatting

```python
from cursus.registry.hybrid.utils import (
    format_registry_error,
    format_step_not_found_error,
    validate_step_name
)

# Handle validation errors with formatted messages
def safe_validate_step(step_name: str, available_steps: list) -> bool:
    try:
        validate_step_name(step_name)
        return True
    except ValueError:
        error_msg = format_step_not_found_error(
            step_name=step_name,
            available_steps=available_steps
        )
        print(f"Validation error: {error_msg}")
        return False

# Use formatted error messages
result = safe_validate_step("invalid step name!", ["step1", "step2"])
```

### Batch Validation

```python
from cursus.registry.hybrid.utils import validate_registry_data, format_validation_error

def validate_registry_batch(registry_data: dict) -> bool:
    """Validate multiple registry entries with detailed error reporting."""
    validation_issues = []
    
    for step_name, step_info in registry_data.items():
        try:
            validate_registry_data(
                registry_type=step_info.get('registry_type', 'core'),
                step_name=step_name,
                workspace_id=step_info.get('workspace_id')
            )
        except ValueError as e:
            validation_issues.append(f"{step_name}: {e}")
    
    if validation_issues:
        error_msg = format_validation_error(
            component_name="registry_batch",
            validation_issues=validation_issues
        )
        print(error_msg)
        return False
    
    return True
```

## Performance Considerations

- **Field Mapping**: Uses optimized LEGACY_FIELDS list for conversion
- **Direct Validation**: Avoids complex validation classes for better performance
- **Template Caching**: Error templates are pre-defined for fast formatting
- **Minimal Dependencies**: Uses only essential imports for faster loading
- **Batch Operations**: Supports batch conversion and validation for efficiency

## Error Handling Patterns

The module provides consistent error handling through:

1. **Specific Exceptions**: RegistryLoadError for loading failures
2. **Template-based Messages**: Consistent error message formatting
3. **Context Information**: Detailed error context with suggestions
4. **Backward Compatibility**: Legacy error formatting functions
5. **Validation Chains**: Comprehensive validation with early failure detection

## Related Documentation

- [Registry Hybrid Models](models.md) - Data models used by utility functions
- [Registry Hybrid Manager](manager.md) - UnifiedRegistryManager using these utilities
- [Registry Hybrid Setup](setup.md) - Setup utilities for workspace initialization
- [Registry Exceptions](../exceptions.md) - Base exception classes
- [Registry Builder Registry](../builder_registry.md) - Step builder registry integration

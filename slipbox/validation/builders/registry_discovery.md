---
tags:
  - code
  - test
  - builders
  - registry
  - discovery
  - automation
keywords:
  - registry discovery
  - step builder discovery
  - automatic detection
  - module loading
  - builder class loading
  - registry integration
  - dynamic import
topics:
  - registry-based discovery
  - automated testing infrastructure
  - dynamic module loading
language: python
date of note: 2025-08-19
---

# Registry-Based Step Builder Discovery

## Overview

The `RegistryStepDiscovery` class provides comprehensive utilities for automatically discovering step builders and their paths using the central registry. This module makes tests adaptive to changes in the step builder ecosystem by providing dynamic discovery, loading, and validation capabilities for all registered step builders.

## Purpose

The registry discovery system serves several critical functions:

1. **Automatic Discovery**: Dynamically discovers all available step builders from the central registry
2. **Adaptive Testing**: Makes tests automatically adapt to changes in the step builder ecosystem
3. **Dynamic Loading**: Provides utilities for dynamically loading builder classes at runtime
4. **Validation Support**: Validates builder availability and loadability
5. **Type-Based Organization**: Organizes builders by SageMaker step types
6. **Error Handling**: Provides comprehensive error handling for discovery failures

## Core Architecture

### RegistryStepDiscovery Class

```python
class RegistryStepDiscovery:
    """
    Registry-based step builder discovery utility.
    
    This class provides methods to automatically discover step builders
    and their module paths using the central registry, making tests
    adaptive to changes in the step builder ecosystem.
    """
```

**Key Features:**
- Registry-based discovery
- Dynamic module loading
- Type-based organization
- Comprehensive validation
- Error handling and reporting

### Exclusion Management

```python
EXCLUDED_FROM_TESTING = {
    "Processing",  # Base processing step - no concrete implementation
    "Base"         # Base step - abstract
}
```

**Exclusion Strategy:**
- Filters out abstract/base steps
- Focuses on concrete implementations
- Maintains testing relevance
- Prevents testing of non-instantiable classes

## Discovery Methods

### Step Type-Based Discovery

#### `get_steps_by_sagemaker_type()`

Retrieves all step names for a specific SageMaker step type:

```python
@staticmethod
def get_steps_by_sagemaker_type(sagemaker_step_type: str, exclude_abstract: bool = True) -> List[str]:
    """
    Get all step names for a specific SageMaker step type from registry.
    
    Args:
        sagemaker_step_type: The SageMaker step type (e.g., 'Training', 'Transform', 'CreateModel')
        exclude_abstract: Whether to exclude abstract/base steps from the results
        
    Returns:
        List of step names that match the specified SageMaker step type
    """
```

**Supported Step Types:**
- **Training**: ML model training steps
- **Transform**: Batch transform operations
- **CreateModel**: Model creation steps
- **Processing**: Data processing steps
- **RegisterModel**: Model registration steps

#### `get_testable_steps_by_sagemaker_type()`

Retrieves only testable (non-abstract) steps for a specific type:

```python
@staticmethod
def get_testable_steps_by_sagemaker_type(sagemaker_step_type: str) -> List[str]:
    """
    Get all testable step names for a specific SageMaker step type from registry.
    Excludes abstract/base steps that shouldn't be tested.
    """
```

**Benefits:**
- Filters out abstract steps automatically
- Focuses on concrete implementations
- Reduces test noise
- Improves test reliability

### Builder Class Loading

#### `get_builder_class_path()`

Determines the module path and class name for a step builder:

```python
@staticmethod
def get_builder_class_path(step_name: str) -> Tuple[str, str]:
    """
    Get the module path and class name for a step builder from registry.
    
    Args:
        step_name: The step name from the registry
        
    Returns:
        Tuple of (module_path, class_name)
    """
```

**Path Resolution:**
- Uses registry information for class names
- Employs dynamic file discovery for module names
- Handles naming convention variations
- Provides comprehensive error reporting

#### `load_builder_class()`

Dynamically loads a step builder class:

```python
@staticmethod
def load_builder_class(step_name: str) -> Type[StepBuilderBase]:
    """
    Dynamically load a step builder class from registry information.
    
    Args:
        step_name: The step name from the registry
        
    Returns:
        The loaded step builder class
    """
```

**Loading Process:**
1. Resolve module path and class name
2. Import the module dynamically
3. Extract the class from the module
4. Validate class inheritance
5. Return the loaded class

### Batch Operations

#### `get_all_builder_classes_by_type()`

Loads all builder classes for a specific SageMaker step type:

```python
@staticmethod
def get_all_builder_classes_by_type(sagemaker_step_type: str) -> Dict[str, Type[StepBuilderBase]]:
    """
    Get all builder classes for a specific SageMaker step type.
    
    Args:
        sagemaker_step_type: The SageMaker step type
        
    Returns:
        Dictionary mapping step names to their builder classes
    """
```

**Batch Loading Features:**
- Loads multiple builders simultaneously
- Handles individual loading failures gracefully
- Provides comprehensive error reporting
- Returns successful loads only

## Validation and Reporting

### Builder Availability Validation

#### `validate_step_builder_availability()`

Validates that a step builder is available and loadable:

```python
@staticmethod
def validate_step_builder_availability(step_name: str) -> Dict[str, Any]:
    """
    Validate that a step builder is available and can be loaded.
    
    Args:
        step_name: The step name to validate
        
    Returns:
        Dictionary containing validation results
    """
```

**Validation Checks:**
- Registry presence verification
- Module existence validation
- Class existence validation
- Loadability testing
- Comprehensive error reporting

**Validation Result Structure:**
```python
{
    "step_name": str,
    "in_registry": bool,
    "module_exists": bool,
    "class_exists": bool,
    "loadable": bool,
    "error": Optional[str],
    "builder_class": Optional[Type[StepBuilderBase]]
}
```

### Discovery Reporting

#### `generate_discovery_report()`

Generates a comprehensive discovery status report:

```python
@staticmethod
def generate_discovery_report() -> Dict[str, Any]:
    """
    Generate a comprehensive report of step builder discovery status.
    
    Returns:
        Dictionary containing discovery report
    """
```

**Report Contents:**
- Total step count
- SageMaker step types
- Step type counts
- Availability summary
- Detailed step information
- Error analysis

## Dynamic Module Resolution

### File System Discovery

#### `_find_builder_module_name()`

Dynamically finds the correct module name by scanning the file system:

```python
@staticmethod
def _find_builder_module_name(step_name: str) -> str:
    """
    Dynamically find the correct module name for a step builder by scanning the file system.
    """
```

**Discovery Strategies:**
1. **Direct Lowercase Match**: Exact lowercase matching
2. **Camel-to-Snake Conversion**: Standard naming convention conversion
3. **Partial Matching**: Component-based matching for compound names
4. **Fuzzy Matching**: Pattern-based matching for known variations

**Naming Convention Handling:**
- CamelCase to snake_case conversion
- Abbreviation handling (PyTorch, XGBoost)
- Compound name resolution
- Special character normalization

### Naming Convention Support

#### `_camel_to_snake()`

Converts CamelCase to snake_case:

```python
@staticmethod
def _camel_to_snake(name: str) -> str:
    """
    Convert CamelCase to snake_case.
    """
```

**Conversion Features:**
- Handles complex CamelCase patterns
- Preserves acronyms appropriately
- Supports digit boundaries
- Maintains readability

## Convenience Functions

### Type-Specific Convenience Functions

The module provides convenience functions for common step types:

```python
def get_training_steps_from_registry() -> List[str]:
    """Get all testable training step names from registry."""

def get_transform_steps_from_registry() -> List[str]:
    """Get all testable transform step names from registry."""

def get_createmodel_steps_from_registry() -> List[str]:
    """Get all testable createmodel step names from registry."""

def get_processing_steps_from_registry() -> List[str]:
    """Get all testable processing step names from registry."""
```

### General Convenience Functions

```python
def get_builder_class_path(step_name: str) -> Tuple[str, str]:
    """Get builder class path for a step name."""

def load_builder_class(step_name: str) -> Type[StepBuilderBase]:
    """Load a builder class by step name."""
```

## Integration Points

### With Central Registry

Direct integration with the central step registry:

```python
from ...steps.registry.step_names import STEP_NAMES, get_steps_by_sagemaker_type
```

**Registry Integration:**
- Uses STEP_NAMES for step information
- Leverages get_steps_by_sagemaker_type for type-based queries
- Maintains consistency with central registry
- Adapts automatically to registry changes

### With Step Builder Base

Integration with the step builder base class:

```python
from ...core.base.builder_base import StepBuilderBase
```

**Base Class Integration:**
- Validates inheritance from StepBuilderBase
- Ensures type safety
- Supports polymorphic behavior
- Maintains interface consistency

## Usage Scenarios

### Automated Test Discovery

For automatically discovering all testable builders:

```python
# Discover all training step builders
training_builders = RegistryStepDiscovery.get_all_builder_classes_by_type("Training")

# Test each discovered builder
for step_name, builder_class in training_builders.items():
    tester = UniversalStepBuilderTest(builder_class)
    results = tester.run_all_tests()
```

### Dynamic Test Generation

For generating tests based on registry contents:

```python
# Generate tests for all step types
all_step_types = RegistryStepDiscovery.get_all_sagemaker_step_types()

for step_type in all_step_types:
    builders = RegistryStepDiscovery.get_all_builder_classes_by_type(step_type)
    generate_tests_for_builders(step_type, builders)
```

### Validation and Reporting

For validating builder availability:

```python
# Generate comprehensive discovery report
report = RegistryStepDiscovery.generate_discovery_report()

# Validate specific builder
validation = RegistryStepDiscovery.validate_step_builder_availability("XGBoostTraining")
```

### CI/CD Integration

For automated testing in CI/CD pipelines:

```python
# Discover and test all available builders
all_step_types = RegistryStepDiscovery.get_all_sagemaker_step_types()

for step_type in all_step_types:
    testable_steps = RegistryStepDiscovery.get_testable_steps_by_sagemaker_type(step_type)
    
    for step_name in testable_steps:
        try:
            builder_class = RegistryStepDiscovery.load_builder_class(step_name)
            run_comprehensive_tests(builder_class)
        except Exception as e:
            log_test_failure(step_name, e)
```

## Error Handling

### Comprehensive Error Management

The discovery system provides comprehensive error handling:

1. **Registry Errors**: Missing steps, invalid registry entries
2. **Import Errors**: Module loading failures, missing dependencies
3. **Class Errors**: Missing classes, invalid inheritance
4. **File System Errors**: Missing files, permission issues

### Error Reporting

Detailed error information for troubleshooting:

- **Error Context**: Where the error occurred
- **Error Details**: Specific error messages
- **Suggested Actions**: Recommendations for resolution
- **Fallback Options**: Alternative approaches when available

## Benefits

### For Test Automation

1. **Adaptive Testing**: Tests automatically adapt to registry changes
2. **Comprehensive Coverage**: Ensures all builders are tested
3. **Reduced Maintenance**: Eliminates manual test registration
4. **Error Detection**: Identifies missing or broken builders

### for Development

1. **Dynamic Discovery**: No need to manually register tests
2. **Immediate Feedback**: New builders are automatically tested
3. **Validation Support**: Validates builder availability during development
4. **Debugging Support**: Comprehensive error reporting for troubleshooting

### For CI/CD

1. **Automated Testing**: Complete automation of builder testing
2. **Failure Isolation**: Individual builder failures don't stop entire test suite
3. **Comprehensive Reporting**: Detailed reports for analysis
4. **Scalable Architecture**: Handles growing number of builders

## Future Enhancements

The registry discovery system is designed to support future improvements:

1. **Enhanced Matching**: More sophisticated name matching algorithms
2. **Caching**: Performance optimization through intelligent caching
3. **Parallel Loading**: Concurrent builder loading for improved performance
4. **Custom Filters**: User-defined filtering criteria for discovery
5. **Integration APIs**: Enhanced integration with external testing tools

## Conclusion

The `RegistryStepDiscovery` class provides essential infrastructure for automated step builder discovery and testing. By leveraging the central registry and providing comprehensive discovery, loading, and validation capabilities, it enables truly adaptive testing that automatically evolves with the step builder ecosystem.

The system's robust error handling, comprehensive reporting, and flexible discovery strategies ensure reliable operation even as the codebase grows and evolves, making it an essential component of the universal testing infrastructure.

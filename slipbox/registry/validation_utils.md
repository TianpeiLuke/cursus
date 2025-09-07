---
tags:
  - code
  - registry
  - validation_utils
  - step_validation
  - standardization
keywords:
  - validation utilities
  - step validation
  - standardization
  - PascalCase
  - naming conventions
topics:
  - validation utilities
  - step standardization
  - naming validation
language: python
date of note: 2024-12-07
---

# Validation Utils

Simple validation utilities for step definition standardization that provide lightweight validation for new step creation following a simplified, performance-optimized approach.

## Overview

The validation utilities module provides essential validation functionality for step definition standardization without over-engineering. It focuses on preventing naming violations during new step creation with optimized performance (target: <1ms per validation) and minimal redundancy (15-20% vs 30-35% in original design).

The module includes core validation patterns using regex for PascalCase and SageMaker step types, auto-correction functionality for common naming violations, performance tracking and caching for optimization, comprehensive error reporting with helpful suggestions, and flexible validation modes (warn, strict, auto_correct).

## Classes and Methods

### Core Validation Functions
- [`validate_new_step_definition`](#validate_new_step_definition) - Validate new step definition with essential checks
- [`auto_correct_step_definition`](#auto_correct_step_definition) - Auto-correct step definition with regex-based fixes
- [`get_validation_errors_with_suggestions`](#get_validation_errors_with_suggestions) - Get validation errors with helpful suggestions
- [`register_step_with_validation`](#register_step_with_validation) - Register step with standardization validation

### Utility Functions
- [`to_pascal_case`](#to_pascal_case) - Convert text to PascalCase using regex patterns
- [`create_validation_report`](#create_validation_report) - Create comprehensive validation report
- [`get_performance_metrics`](#get_performance_metrics) - Get performance metrics for validation operations
- [`get_validation_status`](#get_validation_status) - Get current validation system status

### Constants
- [`PASCAL_CASE_PATTERN`](#pascal_case_pattern) - Regex pattern for PascalCase validation
- [`VALID_SAGEMAKER_TYPES`](#valid_sagemaker_types) - Set of valid SageMaker step types

## API Reference

### validate_new_step_definition

validate_new_step_definition(_step_data_)

Validate new step definition with essential checks only. This function provides core validation logic optimized for <1ms performance, focusing on preventing naming violations during new step creation.

**Parameters:**
- **step_data** (_Dict[str, Any]_) – Dictionary containing step definition data with keys like 'name', 'config_class', 'builder_step_name', 'sagemaker_step_type'.

**Returns:**
- **List[str]** – List of error messages (empty if validation passes).

```python
from cursus.registry.validation_utils import validate_new_step_definition

# Test step definition validation
step_data = {
    'name': 'MyCustomStep',
    'config_class': 'MyCustomStepConfig',
    'builder_step_name': 'MyCustomStepStepBuilder',
    'sagemaker_step_type': 'Processing'
}

errors = validate_new_step_definition(step_data)
if errors:
    print("Validation errors:")
    for error in errors:
        print(f"  - {error}")
else:
    print("✓ Step definition is valid")

# Test invalid step definition
invalid_step = {
    'name': 'my_custom_step',  # Should be PascalCase
    'config_class': 'MyCustomStepConfiguration',  # Should end with 'Config'
    'builder_step_name': 'MyCustomStepBuilder',  # Should end with 'StepBuilder'
    'sagemaker_step_type': 'InvalidType'  # Invalid SageMaker type
}

errors = validate_new_step_definition(invalid_step)
print(f"Found {len(errors)} validation errors")
```

### auto_correct_step_definition

auto_correct_step_definition(_step_data_)

Auto-correct step definition with simple regex-based fixes. This function applies automatic corrections for common naming violations using regex patterns.

**Parameters:**
- **step_data** (_Dict[str, Any]_) – Dictionary containing step definition data to correct.

**Returns:**
- **Dict[str, Any]** – Corrected step data dictionary.

```python
from cursus.registry.validation_utils import auto_correct_step_definition

# Auto-correct invalid step definition
invalid_step = {
    'name': 'my_custom_step',
    'config_class': 'MyCustomStepConfiguration',
    'builder_step_name': 'MyCustomStepBuilder'
}

corrected_step = auto_correct_step_definition(invalid_step)
print("Auto-corrections applied:")
for key, value in corrected_step.items():
    if value != invalid_step.get(key):
        print(f"  {key}: '{invalid_step.get(key)}' → '{value}'")

# Verify corrections
errors = validate_new_step_definition(corrected_step)
print(f"Errors after correction: {len(errors)}")
```

### get_validation_errors_with_suggestions

get_validation_errors_with_suggestions(_step_data_)

Get validation errors with helpful suggestions and examples. This function provides clear error messages with examples for better developer experience.

**Parameters:**
- **step_data** (_Dict[str, Any]_) – Dictionary containing step definition data to validate.

**Returns:**
- **List[str]** – List of detailed error messages with suggestions and examples.

```python
from cursus.registry.validation_utils import get_validation_errors_with_suggestions

# Get detailed validation errors
step_data = {
    'name': 'my_step',
    'config_class': 'MyStepConfiguration',
    'builder_step_name': 'MyStepBuilder',
    'sagemaker_step_type': 'InvalidType'
}

detailed_errors = get_validation_errors_with_suggestions(step_data)
print("Detailed validation report:")
for error in detailed_errors:
    print(f"  {error}")

# Example output:
# ❌ Step name 'my_step' must be PascalCase. Example: 'MyStep' (suggested correction)
# ❌ Config class 'MyStepConfiguration' must end with 'Config'. Example: 'MyStepConfig' (suggested correction)
# 💡 PascalCase examples: 'CradleDataLoading', 'XGBoostTraining', 'PyTorchModel'
```

### register_step_with_validation

register_step_with_validation(_step_name_, _step_data_, _existing_steps_, _mode="warn"_)

Register step with simple standardization validation. This function provides minimal registry integration with flexible validation modes.

**Parameters:**
- **step_name** (_str_) – Name of the step to register.
- **step_data** (_Dict[str, Any]_) – Step definition data.
- **existing_steps** (_Dict[str, Any]_) – Dictionary of existing steps for duplicate checking.
- **mode** (_str_) – Validation mode ("warn", "strict", "auto_correct"). Defaults to "warn".

**Returns:**
- **List[str]** – List of warnings/messages from the registration process.

**Raises:**
- **ValueError** – If validation fails in strict mode.

```python
from cursus.registry.validation_utils import register_step_with_validation

# Existing steps registry
existing_steps = {
    'XGBoostTraining': {'config_class': 'XGBoostTrainingConfig'},
    'CradleDataLoading': {'config_class': 'CradleDataLoadingConfig'}
}

# Register new step with validation
step_data = {
    'config_class': 'MyCustomStepConfig',
    'builder_step_name': 'MyCustomStepStepBuilder',
    'sagemaker_step_type': 'Processing'
}

# Test different validation modes
modes = ["warn", "auto_correct", "strict"]
for mode in modes:
    try:
        warnings = register_step_with_validation(
            "MyCustomStep", 
            step_data, 
            existing_steps, 
            mode=mode
        )
        print(f"{mode.upper()} mode - Warnings: {len(warnings)}")
        for warning in warnings:
            print(f"  {warning}")
    except ValueError as e:
        print(f"{mode.upper()} mode - Error: {e}")
```

### to_pascal_case

to_pascal_case(_text_)

Convert text to PascalCase using simple regex patterns. This utility function is optimized with LRU cache for performance.

**Parameters:**
- **text** (_str_) – Input text to convert.

**Returns:**
- **str** – PascalCase version of the text.

```python
from cursus.registry.validation_utils import to_pascal_case

# Test various text formats
test_cases = [
    "my_custom_step",           # snake_case
    "my-custom-step",           # kebab-case
    "my custom step",           # space-separated
    "myCustomStep",             # camelCase
    "MyCustomStep",             # already PascalCase
    "xgboost_training_step"     # complex snake_case
]

print("PascalCase conversions:")
for text in test_cases:
    pascal = to_pascal_case(text)
    print(f"  '{text}' → '{pascal}'")

# Performance test with caching
import time
start = time.perf_counter()
for _ in range(1000):
    to_pascal_case("my_custom_step")  # Should hit cache after first call
end = time.perf_counter()
print(f"1000 cached conversions: {(end - start) * 1000:.2f}ms")
```

### create_validation_report

create_validation_report(_step_name_, _step_data_, _validation_mode="warn"_)

Create a comprehensive validation report for a step definition with detailed analysis and suggestions.

**Parameters:**
- **step_name** (_str_) – Name of the step to validate.
- **step_data** (_Dict[str, Any]_) – Step definition data.
- **validation_mode** (_str_) – Validation mode used. Defaults to "warn".

**Returns:**
- **Dict[str, Any]** – Dictionary containing comprehensive validation report.

```python
from cursus.registry.validation_utils import create_validation_report

# Create validation report
step_data = {
    'config_class': 'MyStepConfiguration',
    'builder_step_name': 'MyStepBuilder',
    'sagemaker_step_type': 'Processing'
}

report = create_validation_report("my_step", step_data, "warn")

print("Validation Report:")
print(f"  Step Name: {report['step_name']}")
print(f"  Valid: {report['is_valid']}")
print(f"  Error Count: {report['error_count']}")
print(f"  Corrections Available: {report['corrections_available']}")

if report['errors']:
    print("  Errors:")
    for error in report['errors']:
        print(f"    - {error}")

if report['suggested_corrections']:
    print("  Suggested Corrections:")
    for field, correction in report['suggested_corrections'].items():
        print(f"    {field}: '{correction['original']}' → '{correction['corrected']}'")
```

### get_performance_metrics

get_performance_metrics()

Get performance metrics for validation operations including timing statistics and cache performance.

**Returns:**
- **Dict[str, Any]** – Dictionary with performance statistics.

```python
from cursus.registry.validation_utils import get_performance_metrics, validate_new_step_definition

# Perform some validations to generate metrics
test_steps = [
    {'name': 'TestStep1', 'config_class': 'TestStep1Config'},
    {'name': 'test_step_2', 'config_class': 'TestStep2Config'},
    {'name': 'TestStep3', 'config_class': 'TestStep3Configuration'}
]

for step in test_steps:
    validate_new_step_definition(step)

# Get performance metrics
metrics = get_performance_metrics()
print("Performance Metrics:")
print(f"  Total Validations: {metrics['total_validations']}")
print(f"  Average Time: {metrics['average_time_ms']:.3f}ms")
print(f"  Target Met: {metrics['target_met']} (target: {metrics['performance_target']})")
print(f"  Cache Hit Rate: {metrics['cache_stats']['hit_rate']:.2%}")
print(f"  Cache Size: {metrics['cache_stats']['cache_size']}/{metrics['cache_stats']['max_size']}")
```

### get_validation_status

get_validation_status()

Get current validation system status with performance metrics and system information.

**Returns:**
- **Dict[str, Any]** – Dictionary with validation system information.

```python
from cursus.registry.validation_utils import get_validation_status

# Get validation system status
status = get_validation_status()

print("Validation System Status:")
print(f"  Available: {status['validation_available']}")
print(f"  Implementation: {status['implementation_approach']}")
print(f"  Redundancy Level: {status['redundancy_level']}")
print(f"  Supported Modes: {status['supported_modes']}")

print("\nAvailable Functions:")
for func in status['validation_functions']:
    print(f"  - {func}")

print(f"\nCurrent Performance:")
perf = status['current_performance']
print(f"  Average Time: {perf['average_time_ms']:.3f}ms")
print(f"  Target Met: {perf['target_met']}")
print(f"  Total Validations: {perf['total_validations']}")
print(f"  Cache Hit Rate: {perf['cache_hit_rate']:.2%}")
```

### PASCAL_CASE_PATTERN

_re.Pattern_ PASCAL_CASE_PATTERN

Compiled regex pattern for validating PascalCase naming convention.

**Pattern:** `^[A-Z][a-zA-Z0-9]*$`

```python
from cursus.registry.validation_utils import PASCAL_CASE_PATTERN

# Test PascalCase validation
test_names = ["MyStep", "myStep", "my_step", "MyStep123", "MYSTEP"]

for name in test_names:
    is_pascal = bool(PASCAL_CASE_PATTERN.match(name))
    status = "✓" if is_pascal else "✗"
    print(f"{status} '{name}' is PascalCase: {is_pascal}")
```

### VALID_SAGEMAKER_TYPES

_set_ VALID_SAGEMAKER_TYPES

Set of valid SageMaker step types for validation.

**Valid Types:** Processing, Training, Transform, CreateModel, RegisterModel, Base, Utility, Lambda, CradleDataLoading, MimsModelRegistrationProcessing

```python
from cursus.registry.validation_utils import VALID_SAGEMAKER_TYPES

# Check SageMaker type validity
test_types = ["Processing", "Training", "InvalidType", "Custom"]

for step_type in test_types:
    is_valid = step_type in VALID_SAGEMAKER_TYPES
    status = "✓" if is_valid else "✗"
    print(f"{status} '{step_type}' is valid SageMaker type: {is_valid}")

print(f"\nAll valid SageMaker types: {sorted(VALID_SAGEMAKER_TYPES)}")
```

## Usage Examples

### Complete Step Validation Workflow

```python
from cursus.registry.validation_utils import (
    validate_new_step_definition,
    auto_correct_step_definition,
    get_validation_errors_with_suggestions,
    register_step_with_validation,
    create_validation_report
)

# Define a step with various issues
problematic_step = {
    'name': 'my_custom_analysis_step',      # Should be PascalCase
    'config_class': 'MyCustomAnalysisConfiguration',  # Should end with 'Config'
    'builder_step_name': 'MyCustomAnalysisBuilder',   # Should end with 'StepBuilder'
    'sagemaker_step_type': 'CustomProcessing',        # Invalid SageMaker type
    'description': 'Custom analysis step for data processing'
}

print("Step Validation Workflow")
print("=" * 30)

# 1. Basic validation
print("1. Basic Validation:")
errors = validate_new_step_definition(problematic_step)
print(f"   Found {len(errors)} errors")

# 2. Detailed validation with suggestions
print("\n2. Detailed Validation:")
detailed_errors = get_validation_errors_with_suggestions(problematic_step)
for error in detailed_errors:
    print(f"   {error}")

# 3. Auto-correction
print("\n3. Auto-Correction:")
corrected_step = auto_correct_step_definition(problematic_step)
print("   Applied corrections:")
for key, value in corrected_step.items():
    original = problematic_step.get(key, '')
    if value != original:
        print(f"     {key}: '{original}' → '{value}'")

# 4. Validation after correction
print("\n4. Validation After Correction:")
final_errors = validate_new_step_definition(corrected_step)
print(f"   Remaining errors: {len(final_errors)}")

# 5. Comprehensive report
print("\n5. Comprehensive Report:")
report = create_validation_report("MyCustomAnalysisStep", problematic_step)
print(f"   Valid: {report['is_valid']}")
print(f"   Error Count: {report['error_count']}")
print(f"   Corrections Available: {report['corrections_available']}")
```

### Performance Testing and Optimization

```python
import time
from cursus.registry.validation_utils import (
    validate_new_step_definition,
    get_performance_metrics,
    reset_performance_metrics,
    to_pascal_case
)

# Reset metrics for clean test
reset_performance_metrics()

# Performance test with various step definitions
test_steps = [
    {'name': f'TestStep{i}', 'config_class': f'TestStep{i}Config'} 
    for i in range(100)
]

print("Performance Testing")
print("=" * 20)

# Test validation performance
start_time = time.perf_counter()
for step in test_steps:
    validate_new_step_definition(step)
end_time = time.perf_counter()

total_time_ms = (end_time - start_time) * 1000
avg_time_ms = total_time_ms / len(test_steps)

print(f"Validated {len(test_steps)} steps in {total_time_ms:.2f}ms")
print(f"Average time per validation: {avg_time_ms:.3f}ms")
print(f"Target (<1ms): {'✓ MET' if avg_time_ms < 1.0 else '✗ EXCEEDED'}")

# Test caching performance
print("\nCaching Performance:")
cache_test_strings = ["my_test_step"] * 1000

start_time = time.perf_counter()
for text in cache_test_strings:
    to_pascal_case(text)  # Should hit cache after first call
end_time = time.perf_counter()

cache_time_ms = (end_time - start_time) * 1000
print(f"1000 cached conversions: {cache_time_ms:.2f}ms")

# Get detailed metrics
metrics = get_performance_metrics()
print(f"\nDetailed Metrics:")
print(f"  Total Validations: {metrics['total_validations']}")
print(f"  Average Time: {metrics['average_time_ms']:.3f}ms")
print(f"  Cache Hit Rate: {metrics['cache_stats']['hit_rate']:.2%}")
```

### Integration with Registry System

```python
from cursus.registry.validation_utils import register_step_with_validation

# Simulate existing registry
existing_registry = {
    'XGBoostTraining': {
        'config_class': 'XGBoostTrainingConfig',
        'builder_step_name': 'XGBoostTrainingStepBuilder',
        'sagemaker_step_type': 'Training'
    },
    'CradleDataLoading': {
        'config_class': 'CradleDataLoadingConfig',
        'builder_step_name': 'CradleDataLoadingStepBuilder',
        'sagemaker_step_type': 'CradleDataLoading'
    }
}

# Test step registration with different validation modes
new_steps = [
    ('ValidStep', {
        'config_class': 'ValidStepConfig',
        'builder_step_name': 'ValidStepStepBuilder',
        'sagemaker_step_type': 'Processing'
    }),
    ('invalid_step', {  # Invalid name
        'config_class': 'InvalidStepConfiguration',  # Wrong suffix
        'builder_step_name': 'InvalidStepBuilder',   # Wrong suffix
        'sagemaker_step_type': 'InvalidType'         # Invalid type
    }),
    ('XGBoostTraining', {  # Duplicate name
        'config_class': 'XGBoostTrainingConfig',
        'builder_step_name': 'XGBoostTrainingStepBuilder',
        'sagemaker_step_type': 'Training'
    })
]

validation_modes = ['warn', 'auto_correct', 'strict']

for mode in validation_modes:
    print(f"\nTesting {mode.upper()} mode:")
    print("-" * 20)
    
    for step_name, step_data in new_steps:
        try:
            warnings = register_step_with_validation(
                step_name, step_data, existing_registry, mode=mode
            )
            
            print(f"✓ {step_name}: Registered with {len(warnings)} warnings")
            for warning in warnings[:2]:  # Show first 2 warnings
                print(f"    {warning}")
            if len(warnings) > 2:
                print(f"    ... and {len(warnings) - 2} more warnings")
                
        except ValueError as e:
            print(f"✗ {step_name}: {e}")
```

## Performance Considerations

The validation utilities are designed for optimal performance with several key optimizations:

### Performance Targets
- **Validation Speed**: <1ms per validation operation
- **Memory Usage**: Minimal memory footprint with LRU caching
- **Cache Efficiency**: High hit rates for repeated operations
- **Redundancy Level**: 15-20% (vs 30-35% in original design)

### Optimization Strategies
- **Regex Compilation**: Pre-compiled patterns for fast matching
- **LRU Caching**: Cached PascalCase conversions for repeated strings
- **Minimal Validation**: Only essential checks to prevent over-engineering
- **Performance Tracking**: Built-in metrics to monitor performance

## Best Practices

### Validation Integration
1. **Early Validation**: Validate step definitions during creation, not runtime
2. **Mode Selection**: Use "warn" for development, "strict" for production
3. **Auto-Correction**: Use "auto_correct" for automated fixes during development
4. **Performance Monitoring**: Regularly check metrics to ensure targets are met

### Error Handling
1. **Clear Messages**: Provide specific error messages with examples
2. **Suggestions**: Include corrected examples in error messages
3. **Graceful Degradation**: Handle validation failures appropriately
4. **Logging**: Use appropriate log levels for different validation outcomes

### Performance Optimization
1. **Cache Utilization**: Leverage caching for repeated operations
2. **Batch Validation**: Validate multiple steps efficiently
3. **Metrics Monitoring**: Track performance metrics regularly
4. **Resource Management**: Reset metrics periodically to prevent memory leaks

## Related Components

- **[Registry Module](__init__.md)** - Main registry module that uses validation utilities
- **[Step Names](step_names.md)** - Step names registry that integrates with validation
- **[Builder Registry](builder_registry.md)** - Builder registry that uses validation for registration
- **[Exceptions](exceptions.md)** - Registry exceptions that may be raised during validation

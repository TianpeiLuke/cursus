---
tags:
  - code
  - test
  - builders
  - factory
  - variant_system
  - automation
keywords:
  - test factory
  - variant system
  - step type detection
  - automatic selection
  - test variant creation
  - factory pattern
topics:
  - factory-based testing
  - variant management
  - automated test selection
language: python
date of note: 2025-08-19
---

# Universal Step Builder Test Factory

## Overview

The `UniversalStepBuilderTestFactory` class provides a factory pattern implementation for creating appropriate universal step builder test variants based on step type detection. This factory automatically selects the most suitable test variant for any given step builder, enabling intelligent and adaptive testing across the entire step builder ecosystem.

## Purpose

The test factory serves several critical functions:

1. **Automatic Variant Selection**: Intelligently selects appropriate test variants based on step type
2. **Extensible Architecture**: Provides a framework for registering new test variants
3. **Centralized Management**: Centralizes test variant creation and management
4. **Adaptive Testing**: Enables tests to automatically adapt to different step types
5. **Fallback Support**: Provides fallback options when specific variants are unavailable
6. **Discovery Integration**: Integrates with step info detection for intelligent selection

## Core Architecture

### Factory Pattern Implementation

```python
class UniversalStepBuilderTestFactory:
    """Factory for creating appropriate test variants based on step type."""
```

**Key Components:**
- **Variant Registry**: Central registry of available test variants
- **Step Detection**: Integration with step info detection system
- **Selection Logic**: Intelligent variant selection algorithms
- **Fallback Mechanism**: Generic variant fallback for unsupported types
- **Registration System**: Dynamic variant registration capabilities

### Variant Registry

```python
VARIANT_MAP = {}
```

**Registry Features:**
- Dynamic variant registration
- Step type to variant class mapping
- Runtime variant availability checking
- Extensible variant system

## Core Methods

### Factory Creation

#### `create_tester()`

Main factory method for creating appropriate test variants:

```python
@classmethod
def create_tester(
    cls, 
    builder_class: Type[StepBuilderBase], 
    **kwargs
) -> 'UniversalStepBuilderTestBase':
    """
    Create appropriate tester variant for the builder class.
    
    Args:
        builder_class: The step builder class to test
        **kwargs: Additional arguments to pass to the tester
        
    Returns:
        Appropriate test variant instance
    """
```

**Creation Process:**
1. **Variant Initialization**: Initialize built-in variants if needed
2. **Step Detection**: Detect step information using `StepInfoDetector`
3. **Variant Selection**: Select appropriate variant based on step type
4. **Instance Creation**: Create and return variant instance with provided arguments

**Benefits:**
- Automatic variant selection
- Seamless integration with step detection
- Flexible parameter passing
- Consistent interface across variants

### Variant Selection

#### `_select_variant()`

Intelligent variant selection based on step type and information:

```python
@classmethod
def _select_variant(
    cls, 
    sagemaker_step_type: Optional[str], 
    step_info: Dict[str, Any]
) -> Type['UniversalStepBuilderTestBase']:
    """
    Select appropriate test variant based on step type and info.
    """
```

**Selection Strategy:**
1. **Specific Variants**: Check for step type-specific variants
2. **Custom Patterns**: Handle custom step patterns
3. **Fallback Mechanism**: Use generic variant as fallback

**Selection Hierarchy:**
- **Primary**: SageMaker step type-specific variants
- **Secondary**: Custom step pattern variants
- **Fallback**: Generic test variant

### Variant Management

#### `register_variant()`

Registers new test variants for specific step types:

```python
@classmethod
def register_variant(
    cls, 
    step_type: str, 
    variant_class: Type['UniversalStepBuilderTestBase']
) -> None:
    """
    Register a new test variant for a step type.
    """
```

**Registration Features:**
- Dynamic variant registration
- Runtime variant addition
- Step type association
- Variant class validation

#### `get_available_variants()`

Retrieves all available test variants:

```python
@classmethod
def get_available_variants(cls) -> Dict[str, Type['UniversalStepBuilderTestBase']]:
    """
    Get all available test variants.
    """
```

**Variant Information:**
- Complete variant registry
- Step type mappings
- Variant class references
- Availability status

#### `supports_step_type()`

Checks if a step type is supported by registered variants:

```python
@classmethod
def supports_step_type(cls, step_type: str) -> bool:
    """
    Check if a step type is supported by registered variants.
    """
```

**Support Validation:**
- Step type support checking
- Variant availability verification
- Registration status validation
- Capability assessment

## Variant Initialization

### `_initialize_variants()`

Initializes built-in test variants:

```python
@classmethod
def _initialize_variants(cls):
    """Initialize built-in variants."""
```

**Initialization Process:**
1. **Availability Check**: Check if variants are already initialized
2. **Import Variants**: Import available variant implementations
3. **Registration**: Register variants with the factory
4. **Error Handling**: Graceful handling of unavailable variants

**Built-in Variants:**
- **Processing Variant**: `ProcessingStepBuilderTest`
- **Training Variant**: Future implementation
- **Transform Variant**: Future implementation
- **CreateModel Variant**: Future implementation

## Integration Points

### With Step Info Detection

Direct integration with step information detection:

```python
from .step_info_detector import StepInfoDetector
```

**Detection Integration:**
- Automatic step type detection
- Comprehensive step information gathering
- Framework detection
- Pattern recognition

### With Variant System

Integration with the variant testing system:

```python
from .variants.processing_test import ProcessingStepBuilderTest
from .generic_test import GenericStepBuilderTest
```

**Variant Integration:**
- Dynamic variant loading
- Fallback variant support
- Consistent variant interface
- Extensible variant system

### With Step Builder Base

Validation of step builder base class compliance:

```python
from ...core.base.builder_base import StepBuilderBase
```

**Base Class Integration:**
- Type validation
- Interface compliance
- Inheritance verification
- Polymorphic support

## Variant System Architecture

### Variant Types

#### Specific Variants
- **Processing**: Specialized for Processing step builders
- **Training**: Specialized for Training step builders (future)
- **Transform**: Specialized for Transform step builders (future)
- **CreateModel**: Specialized for CreateModel step builders (future)

#### Pattern Variants
- **Custom**: For custom step patterns
- **Generic**: Universal fallback variant

#### Framework Variants
- **XGBoost**: Framework-specific validation (future)
- **PyTorch**: Framework-specific validation (future)
- **Scikit-learn**: Framework-specific validation (future)

### Variant Selection Logic

```python
# Selection priority order:
1. Specific SageMaker step type variants
2. Custom step pattern variants  
3. Framework-specific variants (future)
4. Generic fallback variant
```

## Usage Scenarios

### Automatic Testing

For automatic test variant selection:

```python
# Factory automatically selects appropriate variant
tester = UniversalStepBuilderTestFactory.create_tester(MyStepBuilder)
results = tester.run_all_tests()
```

### Batch Testing

For testing multiple builders with automatic variant selection:

```python
builders = discover_all_step_builders()
for builder_class in builders:
    tester = UniversalStepBuilderTestFactory.create_tester(builder_class)
    results = tester.run_all_tests()
    process_results(builder_class, results)
```

### Variant Registration

For registering custom test variants:

```python
# Register custom variant
UniversalStepBuilderTestFactory.register_variant(
    "CustomStepType", 
    CustomStepBuilderTest
)

# Use factory with custom variant
tester = UniversalStepBuilderTestFactory.create_tester(CustomStepBuilder)
```

### Variant Discovery

For discovering available variants:

```python
# Get all available variants
variants = UniversalStepBuilderTestFactory.get_available_variants()
print(f"Available variants: {list(variants.keys())}")

# Check specific step type support
is_supported = UniversalStepBuilderTestFactory.supports_step_type("Processing")
```

### CI/CD Integration

For automated testing in CI/CD pipelines:

```python
def test_all_builders():
    builders = discover_all_step_builders()
    failed_tests = []
    
    for builder_class in builders:
        try:
            tester = UniversalStepBuilderTestFactory.create_tester(
                builder_class, 
                verbose=False
            )
            results = tester.run_all_tests()
            
            if not all(result['passed'] for result in results.values()):
                failed_tests.append(builder_class.__name__)
                
        except Exception as e:
            failed_tests.append(f"{builder_class.__name__}: {e}")
    
    return failed_tests
```

## Benefits

### For Development

1. **Automatic Selection**: No need to manually choose test variants
2. **Intelligent Testing**: Tests automatically adapt to step types
3. **Consistent Interface**: Uniform testing experience across all builders
4. **Easy Integration**: Simple factory interface for all testing needs

### For Testing Infrastructure

1. **Extensible System**: Easy to add new variants for new step types
2. **Centralized Management**: Single point for variant management
3. **Fallback Support**: Reliable testing even for unknown step types
4. **Dynamic Registration**: Runtime variant registration capabilities

### For Quality Assurance

1. **Comprehensive Coverage**: Ensures appropriate testing for all step types
2. **Specialized Validation**: Step type-specific validation patterns
3. **Consistent Standards**: Uniform quality standards across variants
4. **Automated Quality**: Automated quality assessment through appropriate variants

### For System Scalability

1. **Growing Ecosystem**: Supports growing number of step types
2. **Framework Adaptation**: Adapts to new frameworks and patterns
3. **Variant Evolution**: Supports evolution of testing requirements
4. **Performance Optimization**: Optimized testing through specialized variants

## Error Handling

### Comprehensive Error Management

The factory provides comprehensive error handling:

1. **Import Failures**: Graceful handling of unavailable variants
2. **Registration Errors**: Validation of variant registration
3. **Selection Failures**: Fallback mechanisms for selection issues
4. **Creation Errors**: Error handling during variant instantiation

### Fallback Mechanisms

Robust fallback strategies:

1. **Generic Variant**: Universal fallback for unsupported types
2. **Graceful Degradation**: Continued operation with reduced functionality
3. **Error Recovery**: Recovery strategies for common failure scenarios
4. **Diagnostic Information**: Detailed error information for troubleshooting

## Future Enhancements

The test factory is designed to support future improvements:

1. **Additional Variants**: Support for new step types and frameworks
2. **Smart Selection**: Machine learning-based variant selection
3. **Performance Optimization**: Caching and optimization strategies
4. **Custom Patterns**: User-defined variant selection patterns
5. **Integration APIs**: Enhanced integration with external testing tools

## Extensibility

### Adding New Variants

To add a new test variant:

```python
# 1. Create variant class
class NewStepBuilderTest(UniversalStepBuilderTestBase):
    # Implement variant-specific testing logic
    pass

# 2. Register variant
UniversalStepBuilderTestFactory.register_variant(
    "NewStepType", 
    NewStepBuilderTest
)

# 3. Factory automatically uses new variant
tester = UniversalStepBuilderTestFactory.create_tester(NewStepBuilder)
```

### Custom Selection Logic

For custom variant selection logic:

```python
# Override _select_variant method
class CustomTestFactory(UniversalStepBuilderTestFactory):
    @classmethod
    def _select_variant(cls, sagemaker_step_type, step_info):
        # Custom selection logic
        if custom_condition(step_info):
            return CustomVariant
        return super()._select_variant(sagemaker_step_type, step_info)
```

## Conclusion

The `UniversalStepBuilderTestFactory` provides essential infrastructure for intelligent and adaptive testing of step builders. By automatically selecting appropriate test variants based on step type detection, it enables comprehensive testing while maintaining simplicity and consistency.

The factory's extensible architecture ensures it can grow with the step builder ecosystem, supporting new step types, frameworks, and testing patterns as they emerge. This makes it a crucial component of the universal testing infrastructure, enabling reliable and scalable testing across the entire system.

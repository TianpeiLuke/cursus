# Enhanced Universal Step Builder Tester System

## Overview

The Enhanced Universal Step Builder Tester System is a comprehensive, extensible framework for testing SageMaker step builders. It automatically detects step types, selects appropriate test variants, and provides step type-specific validation patterns.

## Key Features

### 1. Automatic Step Type Detection
- **StepInfoDetector**: Analyzes builder classes to detect SageMaker step type, framework, and test patterns
- **Registry Integration**: Uses the STEP_NAMES registry to map builders to step types
- **Framework Detection**: Identifies ML frameworks (XGBoost, PyTorch, TensorFlow, etc.)

### 2. Step Type-Specific Mock Factory
- **StepTypeMockFactory**: Creates appropriate mock objects based on step type
- **Framework-Aware Mocking**: Generates framework-specific configurations
- **Dependency Resolution**: Provides expected dependencies for each step type

### 3. Variant-Based Testing Architecture
- **Abstract Base Class**: `UniversalStepBuilderTestBase` with step type-specific abstract methods
- **Specialized Variants**: Step type-specific test implementations
- **Factory Pattern**: Automatic variant selection based on detected step information

### 4. Extensible Design
- **Plugin Architecture**: Easy addition of new step type variants
- **Registration System**: Dynamic variant registration
- **Fallback Support**: Generic variant for unsupported step types

## Architecture

```
UniversalStepBuilderTestFactory
├── StepInfoDetector (detects step information)
├── StepTypeMockFactory (creates step-specific mocks)
└── Variants/
    ├── ProcessingStepBuilderTest
    ├── TrainingStepBuilderTest (future)
    ├── TransformStepBuilderTest (future)
    └── GenericStepBuilderTest (fallback)
```

## Usage

### Basic Usage

```python
from cursus.validation.builders import UniversalStepBuilderTestFactory

# Automatic variant selection and testing
tester = UniversalStepBuilderTestFactory.create_tester(
    MyStepBuilderClass, 
    verbose=True
)
results = tester.run_all_tests()
```

### Using the Example Function

```python
from cursus.validation.builders.example_usage import test_step_builder

# Test any step builder with automatic variant selection
results = test_step_builder(MyStepBuilderClass, verbose=True)
```

### Manual Variant Selection

```python
from cursus.validation.builders.variants.processing_test import ProcessingStepBuilderTest

# Use specific variant directly
tester = ProcessingStepBuilderTest(MyProcessingStepBuilder, verbose=True)
results = tester.run_all_tests()
```

## Components

### StepInfoDetector

Analyzes builder classes to extract:
- Builder class name and step name mapping
- SageMaker step type from registry
- Framework detection (XGBoost, PyTorch, etc.)
- Test pattern identification
- Custom step detection

### StepTypeMockFactory

Creates step type-specific mocks:
- **Processing Steps**: ProcessingInput, ProcessingOutput, framework-specific processors
- **Training Steps**: TrainingInput, framework-specific estimators
- **Transform Steps**: TransformInput, Transformer objects
- **CreateModel Steps**: Model objects with appropriate configurations

### Test Variants

#### ProcessingStepBuilderTest
Specialized tests for Processing steps:
- Processor creation validation
- ProcessingInput/Output handling
- Environment variable configuration
- Property files setup
- Framework-specific validation

#### GenericStepBuilderTest
Fallback variant providing:
- Basic step creation tests
- Configuration validation
- Dependency handling
- Mock factory functionality tests

## Step Type Support

### Currently Implemented
- **Processing**: Full support with framework-aware testing
- **Generic**: Fallback support for all step types

### Planned Implementation
- **Training**: XGBoost, PyTorch, TensorFlow variants
- **Transform**: Batch transform step testing
- **CreateModel**: Model creation and registration
- **Custom**: Custom step implementations

## Extension Guide

### Adding New Variants

1. **Create Variant Class**:
```python
from ..base_test import UniversalStepBuilderTestBase

class MyStepTypeTest(UniversalStepBuilderTestBase):
    def get_step_type_specific_tests(self) -> List[str]:
        return ["test_my_specific_feature"]
    
    def _configure_step_type_mocks(self) -> None:
        # Configure step-specific mocks
        pass
    
    def _validate_step_type_requirements(self) -> Dict[str, Any]:
        # Validate step-specific requirements
        return {}
```

2. **Register Variant**:
```python
from cursus.validation.builders import UniversalStepBuilderTestFactory

UniversalStepBuilderTestFactory.register_variant(
    "MyStepType", 
    MyStepTypeTest
)
```

### Extending Mock Factory

Add step type-specific mock creation:
```python
# In StepTypeMockFactory
def _create_mysteptype_mocks(self) -> Dict[str, Any]:
    return {
        'my_step_object': MagicMock(),
        'my_step_config': SimpleNamespace()
    }
```

## Design Principles

### 1. Step Type Awareness
Each SageMaker step type has unique characteristics:
- Different input/output patterns
- Framework-specific configurations
- Unique validation requirements

### 2. Framework Sensitivity
ML frameworks have different:
- Configuration parameters
- Container images
- Dependency patterns

### 3. Extensibility
The system supports:
- Easy addition of new variants
- Custom step type handling
- Framework-specific extensions

### 4. Automatic Detection
Minimizes manual configuration:
- Automatic step type detection
- Framework identification
- Appropriate variant selection

## Testing Patterns

### Processing Steps
- **Tabular Preprocessing**: DATA dependency, environment variables
- **Model Evaluation**: Multiple inputs (model + data), evaluation metrics
- **Feature Engineering**: Input/output transformations

### Training Steps
- **XGBoost Training**: Hyperparameters, framework version, input paths
- **PyTorch Training**: Custom training scripts, distributed training
- **TensorFlow Training**: TensorFlow-specific configurations

### Transform Steps
- **Batch Transform**: Model input, transform configuration
- **Real-time Inference**: Endpoint configuration

### CreateModel Steps
- **Model Registration**: Model artifacts, container configuration
- **Multi-model Endpoints**: Multiple model support

## Benefits

### For Developers
- **Reduced Boilerplate**: Automatic mock creation and configuration
- **Step Type Guidance**: Clear patterns for each step type
- **Comprehensive Testing**: All aspects of step builders validated

### For System Quality
- **Consistent Testing**: Standardized patterns across step types
- **Framework Compliance**: Ensures proper framework usage
- **Dependency Validation**: Verifies correct dependency handling

### For Maintenance
- **Extensible Architecture**: Easy addition of new step types
- **Centralized Logic**: Common patterns in base classes
- **Clear Separation**: Step type-specific logic isolated in variants

## Future Enhancements

### Planned Features
1. **Training Step Variants**: XGBoost, PyTorch, TensorFlow
2. **Transform Step Variants**: Batch and real-time inference
3. **CreateModel Step Variants**: Model registration patterns
4. **Custom Step Support**: Cradle data loading, MIMS registration
5. **Integration Testing**: End-to-end pipeline validation
6. **Performance Testing**: Step execution performance metrics

### Advanced Capabilities
1. **LLM-Assisted Analysis**: Automated code quality assessment
2. **Compliance Checking**: Standardization rule validation
3. **Dependency Graph Analysis**: Complex dependency pattern validation
4. **Configuration Optimization**: Automatic configuration tuning

## Migration Guide

### From Original Universal Tester

```python
# Old approach
from cursus.validation.builders import UniversalStepBuilderTester
tester = UniversalStepBuilderTester(MyBuilder)

# New approach
from cursus.validation.builders import UniversalStepBuilderTestFactory
tester = UniversalStepBuilderTestFactory.create_tester(MyBuilder)
```

### Benefits of Migration
- **Automatic Variant Selection**: No manual configuration needed
- **Step Type-Specific Testing**: More comprehensive validation
- **Framework Awareness**: Better mock creation and validation
- **Extensible Architecture**: Future-proof design

## Conclusion

The Enhanced Universal Step Builder Tester System provides a robust, extensible framework for testing SageMaker step builders. By combining automatic detection, step type-specific variants, and comprehensive validation patterns, it significantly improves the quality and maintainability of step builder testing.

The system's design aligns with the motivation to create different variants for each SageMaker step type, recognizing that step builders' implementations are highly dependent on the corresponding SageMaker step definitions and available methods.

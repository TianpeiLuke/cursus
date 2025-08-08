---
tags:
  - design
  - validation
  - universal_tester
  - sagemaker_steps
  - architecture
keywords:
  - universal step builder test
  - SageMaker step types
  - step builder validation
  - variant testing
  - hierarchical testing
  - step type classification
  - processing steps
  - training steps
  - transform steps
topics:
  - universal testing framework
  - SageMaker step validation
  - step builder architecture
  - test automation
language: python
date of note: 2025-01-08
---

# SageMaker Step Type Universal Tester Design

## Overview

This document presents a comprehensive design for enhancing the universal step builder test system with SageMaker step type-specific variants. The motivation is to create specialized testing frameworks for different SageMaker step types, as each type has unique implementation requirements and validation needs based on their corresponding SageMaker Step definitions.

## Problem Statement

The current universal tester provides general validation but lacks the specificity needed for different SageMaker step types. Each SageMaker step type (Processing, Training, Transform, CreateModel, etc.) has distinct:

- **API Requirements**: Different method signatures and parameters
- **Object Creation Patterns**: Specific SageMaker objects (Processor, Estimator, Transformer, Model)
- **Input/Output Handling**: Unique input/output object types and validation rules
- **Configuration Needs**: Step type-specific configuration parameters
- **Validation Rules**: Different compliance requirements and best practices

Additionally, the testing framework should leverage existing standardized step builder implementations as reference examples to ensure consistency and validate against proven patterns. Different step builders may require different levels of testing based on their complexity and whether they use custom packages or implement custom step types.

## Current State Analysis

### Existing Universal Test Framework

The current `UniversalStepBuilderTest` in `src/cursus/validation/builders/universal_test.py` provides:

- **Multi-level Architecture**: Interface, Specification, Path Mapping, Integration tests
- **SageMaker Integration**: Basic step type validation via `SageMakerStepTypeValidator`
- **Scoring System**: Quality assessment with weighted test results
- **Basic Step Type Tests**: Limited tests for Processing, Training, Transform, CreateModel, RegisterModel

### Step Registry Integration

The `src/cursus/steps/registry/step_names.py` contains comprehensive mapping:

```python
STEP_NAMES = {
    "TabularPreprocessing": {
        "sagemaker_step_type": "Processing",
        # ... other fields
    },
    "XGBoostTraining": {
        "sagemaker_step_type": "Training",
        # ... other fields
    },
    # ... more mappings
}
```

### SageMaker Step Types

From SageMaker documentation, the main step types include:

- **ProcessingStep**: Data processing jobs
- **TrainingStep**: Model training jobs  
- **TransformStep**: Batch transform jobs
- **CreateModelStep**: Model creation
- **TuningStep**: Hyperparameter tuning
- **LambdaStep**: AWS Lambda functions
- **CallbackStep**: Manual approval workflows
- **ConditionStep**: Conditional branching
- **FailStep**: Explicit failure handling
- **EMRStep**: EMR cluster jobs
- **AutoMLStep**: AutoML jobs
- **NotebookJobStep**: Notebook execution

## Proposed Architecture

### 1. Hierarchical Universal Tester System

```
UniversalStepBuilderTest (Base)
├── ProcessingStepBuilderTest (Variant)
├── TrainingStepBuilderTest (Variant)
├── TransformStepBuilderTest (Variant)
├── CreateModelStepBuilderTest (Variant)
├── TuningStepBuilderTest (Variant)
├── LambdaStepBuilderTest (Variant)
├── CallbackStepBuilderTest (Variant)
├── ConditionStepBuilderTest (Variant)
├── FailStepBuilderTest (Variant)
├── EMRStepBuilderTest (Variant)
├── AutoMLStepBuilderTest (Variant)
└── NotebookJobStepBuilderTest (Variant)
```

### 2. Design Principles

1. **Inheritance-Based Variants**: Each SageMaker step type gets specialized tester class
2. **Automatic Detection**: System detects appropriate variant from `sagemaker_step_type` field
3. **Step Type-Specific Validations**: Each variant implements additional tests for its requirements
4. **Extensible Framework**: Easy addition of new step types or enhancement of existing ones
5. **Backward Compatibility**: Existing tests continue working while gaining enhanced capabilities

### 3. Core Components

#### A. Enhanced Base Universal Tester

```python
class UniversalStepBuilderTest:
    def __init__(self, builder_class, **kwargs):
        self.builder_class = builder_class
        self.sagemaker_step_type = self._detect_sagemaker_step_type()
        self.variant_tester = self._create_variant_tester(**kwargs)
    
    def _detect_sagemaker_step_type(self):
        """Use step registry to determine SageMaker step type"""
        step_name = self._extract_step_name_from_builder_class()
        return get_sagemaker_step_type(step_name)
    
    def _create_variant_tester(self, **kwargs):
        """Factory method to create appropriate variant tester"""
        variant_class = STEP_TYPE_VARIANT_MAP.get(
            self.sagemaker_step_type, 
            self.__class__
        )
        return variant_class(self.builder_class, **kwargs)
    
    def run_all_tests(self):
        """Delegate to variant tester"""
        return self.variant_tester.run_all_tests()
```

#### B. Step Type Variant Registry

The step type variant registry is implemented in `src/cursus/steps/registry/step_type_test_variants.py` and provides:

**Core Registry Components**:
- `STEP_TYPE_VARIANT_MAP`: Maps step types to their test variant classes
- `STEP_TYPE_REQUIREMENTS`: Defines requirements for each step type
- `StepTypeRequirements`: Dataclass for step type specifications
- Registration and lookup functions for variant management

**Key Features**:
```python
from cursus.steps.registry.step_type_test_variants import (
    STEP_TYPE_VARIANT_MAP,
    STEP_TYPE_REQUIREMENTS,
    register_step_type_variant,
    get_step_type_variant,
    get_step_type_requirements
)

# Example usage
variant_class = get_step_type_variant("Processing")
requirements = get_step_type_requirements("Processing")
```

**Step Type Requirements Structure**:
Each step type has comprehensive requirements including:
- Required and optional methods
- Required attributes
- SageMaker step class mapping
- SageMaker objects used
- Validation rules and constraints

The registry supports all 12 SageMaker step types with detailed specifications for each variant's validation requirements.

## Reference Examples and Tiered Testing Strategy

### Reference Step Builder Examples

The universal tester variants leverage existing standardized step builder implementations as reference examples to ensure consistency and validate against proven patterns:

#### Processing Step Examples
- **`builder_tabular_preprocessing_step.py`**: Standard processing step without custom package dependencies
  - Uses SKLearnProcessor with standard framework
  - Implements specification-driven input/output handling
  - Follows contract-based environment variable setup
- **`builder_package_step.py`**: Standard processing step for model packaging
  - Uses SKLearnProcessor with local inference scripts handling
  - Implements standard ProcessingInput/ProcessingOutput patterns
- **`builder_model_eval_step_xgboost.py`**: Processing step with custom package dependencies
  - Uses XGBoostProcessor with custom framework version
  - Requires custom package-specific environment variables
  - Demonstrates advanced processor configuration

#### Training Step Examples
- **`builder_training_step_xgboost.py`**: Standard training step implementation
  - Uses XGBoost estimator with proper configuration
  - Implements TrainingInput channel management
  - Handles hyperparameter file generation and upload
  - Demonstrates output path management for model artifacts

#### Custom Step Examples
- **`builder_data_load_step_cradle.py`**: Custom CradleDataLoadingStep
  - Inherits from custom step class, not standard SageMaker steps
  - Requires basic interface validation only
  - Should skip advanced SageMaker integration tests
- **`builder_registration_step.py`**: Custom MimsModelRegistrationProcessingStep
  - Uses custom step implementation
  - Requires basic interface validation only
  - Should skip advanced SageMaker integration tests

### Tiered Testing Strategy

The universal tester implements a tiered approach where different step builders receive different levels of testing based on their complexity and type:

#### Level 1: Universal Interface Tests (All Variants)
All step builder variants share common Level 1 tests:
- Interface compliance validation
- Basic method signature checks
- Configuration validation
- Registry integration checks
- Step name generation validation

#### Level 2+: Variant-Specific Tests (Based on Examples)

**ProcessingStepBuilderTest Patterns:**

*Standard Processing Pattern* (following `tabular_preprocessing_step`, `package_step`):
```python
def test_standard_processing_pattern(self):
    # Validate SKLearnProcessor creation
    # Test ProcessingInput/ProcessingOutput handling
    # Verify environment variable setup
    # Check job arguments construction
```

*Custom Package Processing Pattern* (following `model_eval_step_xgboost`):
```python
def test_custom_package_processing_pattern(self):
    # Validate XGBoostProcessor creation
    # Test custom framework version handling
    # Verify package-specific environment variables
    # Check advanced processor configuration
```

**TrainingStepBuilderTest Patterns:**

*Standard Training Pattern* (following `training_step_xgboost`):
```python
def test_standard_training_pattern(self):
    # Validate XGBoost estimator creation
    # Test TrainingInput channel creation
    # Verify hyperparameter file handling
    # Check output path management
```

**Custom Step Detection:**

*Basic Interface Only* (for CradleDataLoadingStep, MimsModelRegistrationProcessingStep):
```python
def test_custom_step_basic_interface(self):
    # Only Level 1 tests (interface compliance)
    # Skip advanced SageMaker integration tests
    # Focus on configuration validation
    # Check basic method presence
```

### Example-Driven Validation Registry

```python
STEP_TYPE_EXAMPLES = {
    "Processing": {
        "standard_patterns": [
            "builder_tabular_preprocessing_step",
            "builder_package_step"
        ],
        "custom_package_patterns": [
            "builder_model_eval_step_xgboost"
        ],
        "custom_step_patterns": [
            "builder_data_load_step_cradle"
        ]
    },
    "Training": {
        "standard_patterns": [
            "builder_training_step_xgboost"
        ]
    }
}

CUSTOM_STEP_DETECTION = {
    "CradleDataLoadingStep": "basic_interface_only",
    "MimsModelRegistrationProcessingStep": "basic_interface_only"
}
```

### Pattern-Based Test Selection

The universal tester automatically selects appropriate test patterns based on:

1. **Step Type Detection**: Uses `sagemaker_step_type` from step registry
2. **Custom Step Detection**: Identifies custom step classes that inherit from non-standard SageMaker steps
3. **Package Dependency Analysis**: Detects custom package requirements (XGBoost, PyTorch, etc.)
4. **Reference Pattern Matching**: Compares implementation patterns against reference examples

This ensures that:
- Standard step builders get comprehensive testing based on proven patterns
- Custom package step builders get appropriate framework-specific validation
- Custom step implementations get basic interface validation without breaking on advanced tests
- The testing framework learns from existing standardized solutions

## Specific Variant Implementations

### 1. ProcessingStepBuilderTest

**Purpose**: Validate step builders that create SageMaker ProcessingStep instances.

**Specific Tests**:
- **Processor Creation**: Validate processor instance creation and configuration
- **Input/Output Handling**: Test ProcessingInput/ProcessingOutput objects
- **Job Arguments**: Validate command-line arguments and script parameters
- **Property Files**: Test property file configuration for outputs
- **Code Handling**: Validate script/code path handling and upload
- **Resource Configuration**: Test instance types, volumes, and networking

```python
class ProcessingStepBuilderTest(UniversalStepBuilderTest):
    def run_step_type_specific_tests(self):
        results = super().run_step_type_specific_tests()
        
        results.update({
            "test_processor_creation": self._test_processor_creation(),
            "test_processing_inputs_outputs": self._test_processing_inputs_outputs(),
            "test_processing_job_arguments": self._test_processing_job_arguments(),
            "test_property_files": self._test_property_files(),
            "test_processing_code_handling": self._test_processing_code_handling(),
            "test_resource_configuration": self._test_resource_configuration(),
        })
        return results
```

### 2. TrainingStepBuilderTest

**Purpose**: Validate step builders that create SageMaker TrainingStep instances.

**Specific Tests**:
- **Estimator Creation**: Validate estimator instance creation and framework setup
- **Training Inputs**: Test TrainingInput objects and data channels
- **Hyperparameters**: Validate hyperparameter handling and tuning configuration
- **Metric Definitions**: Test custom metric definitions and monitoring
- **Checkpointing**: Validate checkpoint configuration and model persistence
- **Distributed Training**: Test multi-instance and multi-GPU configurations

### 3. TransformStepBuilderTest

**Purpose**: Validate step builders that create SageMaker TransformStep instances.

**Specific Tests**:
- **Transformer Creation**: Validate transformer instance creation from models
- **Transform Inputs**: Test TransformInput objects and data sources
- **Batch Strategy**: Validate batching strategies (SingleRecord, MultiRecord)
- **Output Assembly**: Test output assembly methods (Line, None)
- **Model Integration**: Validate integration with CreateModelStep outputs

### 4. CreateModelStepBuilderTest

**Purpose**: Validate step builders that create SageMaker CreateModelStep instances.

**Specific Tests**:
- **Model Creation**: Validate model instance creation and configuration
- **Container Definitions**: Test container configurations and image URIs
- **Model Data**: Validate model artifact handling and S3 paths
- **Inference Code**: Test inference script handling and dependencies
- **Multi-Container Models**: Validate pipeline model configurations

## Enhanced Test Categories

### Level 1: Interface Compliance (Weight: 1.0)
- **Base Requirements**: Basic inheritance and method implementation
- **Step Type Interface**: Step type-specific interface requirements
- **Method Signatures**: Validate required method signatures match expectations

### Level 2: Specification Alignment (Weight: 1.5)
- **Base Alignment**: Specification and contract usage validation
- **Step Type Specifications**: Step type-specific specification validation
- **Parameter Mapping**: Validate parameter mapping between specs and implementations

### Level 3: SageMaker Integration (Weight: 2.0)
- **Object Creation**: Step type-specific SageMaker object creation
- **Parameter Validation**: Step type-specific parameter validation
- **Input/Output Handling**: Step type-specific input/output object handling
- **Resource Configuration**: Validate compute resources and configurations

### Level 4: Pipeline Integration (Weight: 2.5)
- **Dependency Resolution**: Validate step dependencies and execution order
- **Property Path Validation**: Step type-specific property path validation
- **Step Creation**: Validate actual SageMaker step creation
- **Pipeline Compatibility**: Test integration with SageMaker Pipelines

## Implementation Details

### Registry-Based Pattern Detection

The enhanced registry provides comprehensive pattern detection and test selection:

```python
from cursus.steps.registry.step_type_test_variants import (
    get_test_pattern_for_builder,
    should_run_advanced_tests,
    get_reference_examples_for_pattern,
    detect_framework_from_processor
)

# Example usage in universal tester
class UniversalStepBuilderTest:
    def __init__(self, builder_class, **kwargs):
        self.builder_class = builder_class
        self.step_type = self._detect_sagemaker_step_type()
        self.test_pattern = get_test_pattern_for_builder(
            builder_class.__name__, 
            self.step_type
        )
        self.run_advanced_tests = should_run_advanced_tests(
            builder_class.__name__, 
            self.step_type
        )
        
    def select_test_suite(self):
        """Select appropriate test suite based on pattern detection"""
        if self.test_pattern == "custom_step":
            return self._get_basic_interface_tests()
        elif self.test_pattern == "custom_package":
            return self._get_custom_package_tests()
        else:
            return self._get_standard_tests()
```

### Example-Driven Validation

The system uses reference examples to validate implementation patterns:

```python
# Get reference examples for validation
reference_examples = get_reference_examples_for_pattern("Processing", "standard")
# Returns: ["builder_tabular_preprocessing_step", "builder_package_step"]

# Use examples to validate method signatures and patterns
def validate_against_reference_examples(self, builder_class):
    """Validate builder follows patterns from reference examples"""
    for example_name in self.reference_examples:
        example_class = self._load_example_class(example_name)
        self._compare_method_signatures(builder_class, example_class)
        self._validate_implementation_patterns(builder_class, example_class)
```

### Framework Detection and Custom Package Handling

The registry automatically detects frameworks and custom packages:

```python
# Detect framework from processor usage
def detect_builder_framework(self, builder_class):
    """Detect framework used by step builder"""
    if hasattr(builder_class, '_create_processor'):
        # Analyze processor creation method
        processor_type = self._extract_processor_type(builder_class)
        framework = detect_framework_from_processor(processor_type)
        return framework
    return None

# Apply framework-specific tests
def apply_framework_tests(self, framework):
    """Apply framework-specific validation tests"""
    if framework == "xgboost":
        self._test_xgboost_processor_config()
        self._test_xgboost_environment_variables()
    elif framework == "sklearn":
        self._test_sklearn_processor_config()
        self._test_sklearn_script_handling()
```

## Implementation Strategy

### Phase 1: Enhanced Registry Implementation ✅
1. **Registry Enhancement**: Enhanced step type registry with reference examples and pattern detection
2. **Pattern Detection Functions**: Implemented helper functions for test pattern selection
3. **Custom Step Detection**: Added custom step detection and test level determination
4. **Framework Detection**: Added framework detection based on processor types

### Phase 2: Core Framework Enhancement
1. **Base Class Enhancement**: Modify `UniversalStepBuilderTest` with variant detection
2. **Factory Pattern**: Implement factory pattern for variant creation using registry
3. **Base Variant Class**: Create abstract base class for all variants
4. **Pattern-Based Test Selection**: Integrate registry functions for automatic test selection

### Phase 3: Primary Variants Implementation
1. **ProcessingStepBuilderTest**: Implement comprehensive processing step validation
   - Standard pattern tests (following tabular_preprocessing_step, package_step)
   - Custom package pattern tests (following model_eval_step_xgboost)
   - Custom step pattern tests (basic interface only)
2. **TrainingStepBuilderTest**: Implement training step validation with estimator handling
   - Standard pattern tests (following training_step_xgboost)
   - Hyperparameter handling validation
   - TrainingInput channel management tests

### Phase 4: Advanced Variants Implementation
1. **TransformStepBuilderTest**: Implement transform step validation with transformer handling
2. **CreateModelStepBuilderTest**: Implement model creation step validation
3. **Remaining Standard Variants**: Implement TuningStep, LambdaStep, CallbackStep variants
4. **Specialized Variants**: Implement EMRStep, AutoMLStep, NotebookJobStep variants

### Phase 5: Integration & Optimization
1. **CI/CD Integration**: Integrate enhanced testing with existing pipelines
2. **Performance Optimization**: Optimize test execution and resource usage
3. **Documentation**: Create comprehensive documentation and examples
4. **Monitoring**: Add test result monitoring and quality metrics

## Benefits

### 1. Type-Specific Validation
- Each SageMaker step type receives appropriate validation
- Catches step type-specific implementation errors early
- Ensures compliance with SageMaker API requirements

### 2. Enhanced Quality Assurance
- More comprehensive test coverage for each step type
- Better quality metrics specific to step type requirements
- Improved confidence in step builder implementations

### 3. Developer Experience
- Clear feedback on step type-specific issues
- Better error messages and debugging information
- Reduced time to identify and fix implementation problems

### 4. Maintainability
- Clear separation of concerns with inheritance hierarchy
- Easy to add new step types or enhance existing ones
- Centralized step type requirements and validation logic

### 5. Extensibility
- Framework easily accommodates new SageMaker step types
- Variant-specific enhancements don't affect other step types
- Supports custom step type implementations

## Migration Strategy

### 1. Backward Compatibility
- Existing tests continue to work without modification
- Gradual migration to enhanced variants
- Fallback to base universal tester for unknown step types

### 2. Incremental Adoption
- Teams can adopt enhanced variants incrementally
- No breaking changes to existing test infrastructure
- Optional enhanced validation features

### 3. Documentation and Training
- Comprehensive migration guide for development teams
- Examples and best practices for each step type variant
- Training materials for enhanced testing capabilities

## Future Enhancements

### 1. Dynamic Variant Loading
- Plugin-based architecture for custom step type variants
- Runtime discovery of new step type implementations
- Support for third-party step type extensions

### 2. Advanced Analytics
- Step type-specific quality metrics and trends
- Performance benchmarking across step types
- Automated quality improvement recommendations

### 3. Integration Enhancements
- IDE integration for real-time validation feedback
- Automated test generation based on step specifications
- Integration with SageMaker Studio for enhanced development experience

## Conclusion

The proposed SageMaker step type universal tester design provides a comprehensive solution for validating step builders across all SageMaker step types. By implementing specialized variants for each step type, we achieve:

- **Higher Quality**: More thorough validation specific to each step type's requirements
- **Better Developer Experience**: Clear, actionable feedback for step type-specific issues
- **Improved Maintainability**: Well-organized, extensible architecture
- **Future-Proof Design**: Easy accommodation of new SageMaker step types

This design maintains backward compatibility while providing significant enhancements to the testing framework, ensuring robust validation of step builder implementations across the entire SageMaker ecosystem.

## References

- [Universal Step Builder Test](universal_step_builder_test.md) - Current universal testing framework
- [Universal Step Builder Test Scoring](universal_step_builder_test_scoring.md) - Test scoring and quality metrics system

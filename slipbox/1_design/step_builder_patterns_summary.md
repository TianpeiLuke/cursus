---
tags:
  - design
  - step_builders
  - patterns
  - summary
  - universal_tester
keywords:
  - step builder patterns
  - pattern summary
  - universal tester design
  - SageMaker step types
  - testing framework
topics:
  - step builder architecture
  - pattern analysis
  - testing strategy
  - framework design
language: python
date of note: 2025-01-08
---

# Step Builder Patterns Summary

## Related Documents

### Pattern Analysis Documents
- [Processing Step Builder Patterns](processing_step_builder_patterns.md) - Detailed analysis of Processing step implementations
- [Training Step Builder Patterns](training_step_builder_patterns.md) - Detailed analysis of Training step implementations
- [CreateModel Step Builder Patterns](createmodel_step_builder_patterns.md) - Detailed analysis of CreateModel step implementations
- [Transform Step Builder Patterns](transform_step_builder_patterns.md) - Detailed analysis of Transform step implementations

### Enhanced Universal Tester Design
- [Enhanced Universal Step Builder Tester Design](enhanced_universal_step_builder_tester_design.md) - Comprehensive design for step type-aware testing

### Existing Universal Tester Documents
- [Universal Step Builder Test](universal_step_builder_test.md) - Current universal tester design and implementation
- [Universal Step Builder Test Scoring](universal_step_builder_test_scoring.md) - Scoring system for universal tester

### Related Design Documents
- [SageMaker Step Type Classification Design](sagemaker_step_type_classification_design.md) - Step type classification system
- [Step Builder Registry Design](step_builder_registry_design.md) - Step builder registry architecture
- [Step Builder](step_builder.md) - Core step builder design principles
- [Step Specification](step_specification.md) - Step specification system design
- [Specification Driven Design](specification_driven_design.md) - Specification-driven architecture

## Overview

This document provides a comprehensive summary of step builder patterns across all SageMaker step types in the cursus framework. It synthesizes the analysis from individual pattern documents to provide guidance for implementing the enhanced universal tester with step type-specific variants.

## SageMaker Step Type Classification

Based on the analysis of existing step builders, the following SageMaker step types are implemented:

### 1. Processing Steps (Most Common)
**SageMaker Step Type**: `Processing`
**Implementations**: 9 step builders
- TabularPreprocessing, Package, CurrencyConversion, RiskTableMapping
- ModelCalibration, XGBoostModelEval, Payload, DummyTraining
- Custom: CradleDataLoading (custom step type)

**Key Characteristics**:
- Use ProcessingStep with various processors (SKLearn, XGBoost)
- Support job type variants (training, validation, testing, calibration)
- Specification-driven input/output handling
- Environment variable-based configuration
- Command-line argument patterns

### 2. Training Steps
**SageMaker Step Type**: `Training`
**Implementations**: 2 step builders
- PyTorchTraining, XGBoostTraining

**Key Characteristics**:
- Use TrainingStep with framework-specific estimators
- Framework-specific hyperparameter handling (direct vs file-based)
- Different data channel strategies (single vs multiple channels)
- Model artifact and evaluation output handling

### 3. CreateModel Steps
**SageMaker Step Type**: `CreateModel`
**Implementations**: 2 step builders
- XGBoostModel, PyTorchModel

**Key Characteristics**:
- Use CreateModelStep with framework-specific model classes
- Automatic container image URI generation
- Model data processing from training step outputs
- Inference configuration through environment variables

### 4. Transform Steps
**SageMaker Step Type**: `Transform`
**Implementations**: 1 step builder
- BatchTransform (with job type variants)

**Key Characteristics**:
- Use TransformStep with SageMaker Transformer
- Model integration with CreateModelStep outputs
- Advanced input processing options (filtering, assembly)
- Job type support similar to Processing steps

### 5. Custom Steps
**SageMaker Step Type**: Custom types
**Implementations**: 2 step builders
- Registration (MimsModelRegistrationProcessing)
- CradleDataLoading (CradleDataLoadingStep)

**Key Characteristics**:
- Use custom step classes, not standard SageMaker steps
- Basic interface validation only
- Skip advanced SageMaker integration tests

## Universal Patterns Across All Step Types

### 1. Base Architecture Pattern
All step builders follow this consistent structure:

```python
@register_builder()
class StepBuilder(StepBuilderBase):
    def __init__(self, config, sagemaker_session=None, role=None, notebook_root=None, 
                 registry_manager=None, dependency_resolver=None):
        # Load appropriate specification
        spec = self._load_specification(config)
        super().__init__(config=config, spec=spec, ...)
        
    def validate_configuration(self) -> None:
        # Validate step-specific configuration
        
    def _create_step_object(self) -> StepObject:
        # Create step-specific object (processor, estimator, model, transformer)
        
    def _get_environment_variables(self) -> Dict[str, str]:
        # Build environment variables
        
    def _get_inputs(self, inputs) -> InputObjects:
        # Create step-specific input objects
        
    def _get_outputs(self, outputs) -> OutputObjects:
        # Handle step-specific outputs
        
    def create_step(self, **kwargs) -> SageMakerStep:
        # Orchestrate step creation
```

### 2. Specification-Driven Design Pattern
All modern step builders use specifications:

```python
# Specification loading patterns
if not self.spec:
    raise ValueError("Step specification is required")
    
if not self.contract:
    raise ValueError("Script contract is required for path mapping")

# Input processing using specifications
for _, dependency_spec in self.spec.dependencies.items():
    logical_name = dependency_spec.logical_name
    # Process dependency...

# Output processing using specifications  
for _, output_spec in self.spec.outputs.items():
    logical_name = output_spec.logical_name
    # Process output...
```

### 3. Dependency Resolution Pattern
All step builders support dependency extraction:

```python
def create_step(self, **kwargs):
    # Handle inputs from dependencies and explicit inputs
    inputs = {}
    if dependencies:
        extracted_inputs = self.extract_inputs_from_dependencies(dependencies)
        inputs.update(extracted_inputs)
    inputs.update(inputs_raw)
    
    # Process inputs using step-specific methods
    step_inputs = self._get_inputs(inputs)
```

### 4. Configuration Validation Pattern
All step builders validate their configurations:

```python
def validate_configuration(self) -> None:
    # Common validation pattern
    required_attrs = ['attr1', 'attr2', 'attr3']
    
    for attr in required_attrs:
        if not hasattr(self.config, attr) or getattr(self.config, attr) is None:
            raise ValueError(f"Missing required attribute: {attr}")
    
    # Step-specific validation
    self._validate_step_specific_config()
```

### 5. Step Creation Orchestration Pattern
All step builders follow similar orchestration:

```python
def create_step(self, **kwargs) -> SageMakerStep:
    # Extract and process inputs
    inputs = self._process_inputs(kwargs)
    outputs = kwargs.get('outputs', {})
    dependencies = kwargs.get('dependencies', [])
    
    # Create step-specific objects
    step_object = self._create_step_object()
    step_inputs = self._get_inputs(inputs)
    step_outputs = self._get_outputs(outputs)
    
    # Get standardized step name
    step_name = self._get_step_name()
    
    # Create the SageMaker step
    step = SageMakerStepClass(
        name=step_name,
        step_object=step_object,
        inputs=step_inputs,
        outputs=step_outputs,  # May be None for some step types
        depends_on=dependencies,
        cache_config=self._get_cache_config(enable_caching)
    )
    
    # Attach specification for future reference
    setattr(step, '_spec', self.spec)
    
    return step
```

## Step Type-Specific Patterns

### Processing Step Patterns
- **Processor Creation**: SKLearnProcessor vs XGBoostProcessor
- **Job Type Support**: Multi-job-type specifications
- **Environment Variables**: Complex JSON serialization
- **Job Arguments**: Various argument patterns
- **Special Input Handling**: Local path overrides (Package step)

### Training Step Patterns
- **Estimator Creation**: Framework-specific estimators
- **Hyperparameter Handling**: Direct vs file-based approaches
- **Data Channels**: Single vs multiple channel strategies
- **Output Paths**: Single output path for model and evaluation artifacts
- **Metric Definitions**: Framework-specific metric patterns

### CreateModel Step Patterns
- **Model Creation**: Framework-specific model classes
- **Image URI Generation**: Automatic container image selection
- **Model Data Processing**: Integration with training step outputs
- **Step Arguments**: model.create() argument handling
- **Output Handling**: Automatic ModelName property provision

### Transform Step Patterns
- **Transformer Creation**: SageMaker Transformer configuration
- **Input Processing**: TransformInput with filtering options
- **Model Integration**: Integration with CreateModelStep outputs
- **Output Assembly**: Various assembly and join strategies
- **Job Type Support**: Similar to Processing steps

## Testing Strategy by Step Type

### Level 1: Universal Interface Tests (All Step Types)
```python
class UniversalStepBuilderTest:
    def test_interface_compliance(self):
        # Test basic interface requirements
        
    def test_configuration_validation(self):
        # Test configuration validation
        
    def test_step_name_generation(self):
        # Test standardized step naming
        
    def test_specification_integration(self):
        # Test specification usage
```

### Level 2: Step Type-Specific Tests

#### Processing Step Tests
```python
class ProcessingStepBuilderTest(UniversalStepBuilderTest):
    def test_processor_creation(self):
        # Test SKLearnProcessor/XGBoostProcessor creation
        
    def test_processing_inputs_outputs(self):
        # Test ProcessingInput/ProcessingOutput objects
        
    def test_environment_variables(self):
        # Test environment variable construction
        
    def test_job_arguments(self):
        # Test command-line argument handling
        
    def test_job_type_variants(self):
        # Test different job type behaviors
```

#### Training Step Tests
```python
class TrainingStepBuilderTest(UniversalStepBuilderTest):
    def test_estimator_creation(self):
        # Test framework-specific estimator creation
        
    def test_hyperparameter_handling(self):
        # Test hyperparameter processing
        
    def test_training_inputs(self):
        # Test TrainingInput object creation
        
    def test_data_channel_strategy(self):
        # Test single vs multiple channel strategies
        
    def test_output_path_handling(self):
        # Test output path generation
```

#### CreateModel Step Tests
```python
class CreateModelStepBuilderTest(UniversalStepBuilderTest):
    def test_model_creation(self):
        # Test framework-specific model creation
        
    def test_image_uri_generation(self):
        # Test container image URI generation
        
    def test_model_data_processing(self):
        # Test model data input handling
        
    def test_step_arguments(self):
        # Test model.create() arguments
```

#### Transform Step Tests
```python
class TransformStepBuilderTest(UniversalStepBuilderTest):
    def test_transformer_creation(self):
        # Test Transformer object creation
        
    def test_transform_input_processing(self):
        # Test TransformInput configuration
        
    def test_model_integration(self):
        # Test model name extraction from dependencies
        
    def test_input_processing_options(self):
        # Test filtering and assembly options
```

## Framework Detection and Pattern Matching

### Automatic Framework Detection
```python
def detect_framework_from_step_builder(builder_class):
    """Detect framework based on step builder implementation."""
    
    # Check processor type for Processing steps
    if hasattr(builder_class, '_create_processor'):
        processor_method = getattr(builder_class, '_create_processor')
        if 'XGBoostProcessor' in str(processor_method):
            return 'xgboost'
        elif 'SKLearnProcessor' in str(processor_method):
            return 'sklearn'
    
    # Check estimator type for Training steps
    if hasattr(builder_class, '_create_estimator'):
        estimator_method = getattr(builder_class, '_create_estimator')
        if 'PyTorch' in str(estimator_method):
            return 'pytorch'
        elif 'XGBoost' in str(estimator_method):
            return 'xgboost'
    
    # Check model type for CreateModel steps
    if hasattr(builder_class, '_create_model'):
        model_method = getattr(builder_class, '_create_model')
        if 'PyTorchModel' in str(model_method):
            return 'pytorch'
        elif 'XGBoostModel' in str(model_method):
            return 'xgboost'
    
    return 'generic'
```

### Pattern Classification
```python
def classify_step_builder_pattern(builder_class, step_type):
    """Classify step builder into testing patterns."""
    
    builder_name = builder_class.__name__
    
    # Custom step detection
    if any(custom_type in builder_name for custom_type in ['Cradle', 'Mims']):
        return 'custom_step'
    
    # Framework-specific pattern detection
    framework = detect_framework_from_step_builder(builder_class)
    if framework in ['xgboost', 'pytorch']:
        return 'custom_package'
    
    # Standard pattern
    return 'standard'
```

## Implementation Recommendations

### 1. Enhanced Universal Tester Architecture
```python
class UniversalStepBuilderTest:
    def __init__(self, builder_class, **kwargs):
        self.builder_class = builder_class
        self.step_type = self._detect_sagemaker_step_type()
        self.pattern = self._classify_pattern()
        self.variant_tester = self._create_variant_tester(**kwargs)
    
    def _create_variant_tester(self, **kwargs):
        """Factory method to create appropriate variant tester."""
        variant_class = STEP_TYPE_VARIANT_MAP.get(self.step_type, self.__class__)
        return variant_class(self.builder_class, **kwargs)
```

### 2. Step Type Variant Registry
```python
STEP_TYPE_VARIANT_MAP = {
    'Processing': ProcessingStepBuilderTest,
    'Training': TrainingStepBuilderTest,
    'CreateModel': CreateModelStepBuilderTest,
    'Transform': TransformStepBuilderTest,
    'CradleDataLoading': CustomStepBuilderTest,
    'MimsModelRegistrationProcessing': CustomStepBuilderTest,
}
```

### 3. Pattern-Based Test Selection
```python
def select_test_suite(self, pattern, step_type):
    """Select appropriate test suite based on pattern and step type."""
    
    if pattern == 'custom_step':
        return self._get_basic_interface_tests()
    elif pattern == 'custom_package':
        return self._get_framework_specific_tests(step_type)
    else:
        return self._get_standard_tests(step_type)
```

## Key Insights for Universal Tester Design

### 1. Hierarchical Testing Strategy
- **Level 1**: Universal interface tests for all step builders
- **Level 2**: Step type-specific tests based on SageMaker step type
- **Level 3**: Pattern-specific tests based on implementation patterns
- **Level 4**: Framework-specific tests for custom packages

### 2. Specification-Driven Validation
- All modern step builders use specifications
- Validate specification compliance and contract integration
- Test input/output mapping through specifications

### 3. Framework-Specific Considerations
- Different frameworks have different patterns (PyTorch vs XGBoost)
- Custom package steps need framework-specific validation
- Standard steps use common patterns

### 4. Custom Step Handling
- Custom steps need basic interface validation only
- Skip advanced SageMaker integration tests for custom steps
- Focus on configuration validation and basic functionality

### 5. Dependency Integration Testing
- All step builders support dependency resolution
- Test both explicit inputs and dependency extraction
- Validate proper integration between step types

This comprehensive pattern analysis provides the foundation for implementing a robust, step type-aware universal testing framework that can validate step builders across all SageMaker step types while accommodating the unique patterns and requirements of each type.

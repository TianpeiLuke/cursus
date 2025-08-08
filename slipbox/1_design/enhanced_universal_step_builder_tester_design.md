---
tags:
  - design
  - universal_tester
  - step_builders
  - testing_framework
  - sagemaker_steps
keywords:
  - universal tester design
  - step type variants
  - SageMaker step types
  - testing architecture
  - pattern-based testing
topics:
  - testing framework design
  - step builder validation
  - universal tester enhancement
  - step type classification
language: python
date of note: 2025-01-08
---

# Enhanced Universal Step Builder Tester Design

## Related Documents

### Pattern Analysis Documents
- [Processing Step Builder Patterns](processing_step_builder_patterns.md) - Analysis of Processing step implementations
- [Training Step Builder Patterns](training_step_builder_patterns.md) - Analysis of Training step implementations  
- [CreateModel Step Builder Patterns](createmodel_step_builder_patterns.md) - Analysis of CreateModel step implementations
- [Transform Step Builder Patterns](transform_step_builder_patterns.md) - Analysis of Transform step implementations
- [Step Builder Patterns Summary](step_builder_patterns_summary.md) - Comprehensive summary of all step builder patterns

### Existing Universal Tester Documents
- [Universal Step Builder Test](universal_step_builder_test.md) - Current universal tester design and implementation
- [Universal Step Builder Test Scoring](universal_step_builder_test_scoring.md) - Scoring system for universal tester

### Related Design Documents
- [SageMaker Step Type Classification Design](sagemaker_step_type_classification_design.md) - Step type classification system
- [Step Builder Registry Design](step_builder_registry_design.md) - Step builder registry architecture
- [Step Builder](step_builder.md) - Core step builder design principles
- [Step Specification](step_specification.md) - Step specification system design
- [Specification Driven Design](specification_driven_design.md) - Specification-driven architecture
- [Dependency Resolver](dependency_resolver.md) - Dependency resolution system
- [Registry Manager](registry_manager.md) - Registry management system
- [Validation Engine](validation_engine.md) - Validation framework design

### Configuration and Contract Documents
- [Config Field Categorization](config_field_categorization.md) - Configuration field classification
- [Script Contract](script_contract.md) - Script contract specifications
- [Step Contract](step_contract.md) - Step contract definitions
- [Environment Variable Contract Enforcement](environment_variable_contract_enforcement.md) - Environment variable contracts

### Implementation Improvement Documents
- [Job Type Variant Handling](job_type_variant_handling.md) - Job type variant implementation
- [Training Step Improvements](training_step_improvements.md) - Training step enhancements
- [PyTorch Training Step Improvements](pytorch_training_step_improvements.md) - PyTorch-specific improvements
- [Packaging Step Improvements](packaging_step_improvements.md) - Package step enhancements

## Overview

This document presents a comprehensive design for enhancing the universal step builder tester to support step type-specific variants. The design is based on the comprehensive pattern analysis of existing step builders and addresses the need for different testing approaches based on SageMaker step types and implementation patterns.

## Design Motivation

The current universal tester treats all step builders uniformly, but our pattern analysis reveals significant differences between step types:

1. **Processing Steps**: Complex environment variables, job arguments, multiple job types
2. **Training Steps**: Framework-specific estimators, hyperparameter handling, data channels
3. **CreateModel Steps**: Model creation, image URI generation, inference configuration
4. **Transform Steps**: Transformer configuration, model integration, input processing
5. **Custom Steps**: Basic interface validation, skip SageMaker-specific tests

The enhanced design provides step type-aware testing while maintaining the universal interface.

## Architecture Overview

```
Enhanced Universal Tester
├── UniversalStepBuilderTester (Base)
│   ├── Interface Validation
│   ├── Configuration Validation
│   ├── Step Name Generation
│   └── Specification Integration
├── Step Type Variants
│   ├── ProcessingStepTester
│   ├── TrainingStepTester
│   ├── CreateModelStepTester
│   ├── TransformStepTester
│   └── CustomStepTester
└── Pattern Detection & Classification
    ├── SageMaker Step Type Detection
    ├── Framework Detection
    └── Pattern Classification
```

## Core Design Components

### 1. Enhanced Universal Tester Base Class

```python
class UniversalStepBuilderTester:
    """
    Enhanced universal tester with step type-specific variant support.
    
    This class provides the base testing framework and automatically delegates
    to appropriate step type-specific testers based on detected patterns.
    """
    
    def __init__(self, builder_class, **kwargs):
        """
        Initialize the universal tester with automatic variant detection.
        
        Args:
            builder_class: The step builder class to test
            **kwargs: Additional configuration for testing
        """
        self.builder_class = builder_class
        self.builder_name = builder_class.__name__
        
        # Detect step characteristics
        self.sagemaker_step_type = self._detect_sagemaker_step_type()
        self.framework = self._detect_framework()
        self.pattern = self._classify_pattern()
        
        # Create appropriate variant tester
        self.variant_tester = self._create_variant_tester(**kwargs)
        
        # Initialize base testing components
        self._initialize_base_components(**kwargs)
    
    def _detect_sagemaker_step_type(self) -> str:
        """
        Detect the SageMaker step type based on step builder implementation.
        
        Returns:
            SageMaker step type string
        """
        # Check for custom step types first
        if 'Cradle' in self.builder_name:
            return 'CradleDataLoading'
        elif 'Mims' in self.builder_name or 'Registration' in self.builder_name:
            return 'MimsModelRegistrationProcessing'
        
        # Check for standard SageMaker step types
        if hasattr(self.builder_class, '_create_processor'):
            return 'Processing'
        elif hasattr(self.builder_class, '_create_estimator'):
            return 'Training'
        elif hasattr(self.builder_class, '_create_model'):
            return 'CreateModel'
        elif hasattr(self.builder_class, '_create_transformer'):
            return 'Transform'
        
        # Fallback: analyze create_step method return type
        return self._analyze_create_step_return_type()
    
    def _detect_framework(self) -> str:
        """
        Detect the framework based on step builder implementation.
        
        Returns:
            Framework name string
        """
        # Check method implementations for framework indicators
        for method_name in ['_create_processor', '_create_estimator', '_create_model']:
            if hasattr(self.builder_class, method_name):
                method = getattr(self.builder_class, method_name)
                method_source = inspect.getsource(method) if hasattr(inspect, 'getsource') else str(method)
                
                if 'XGBoost' in method_source:
                    return 'xgboost'
                elif 'PyTorch' in method_source:
                    return 'pytorch'
                elif 'SKLearn' in method_source:
                    return 'sklearn'
        
        # Check builder name for framework indicators
        if 'XGBoost' in self.builder_name:
            return 'xgboost'
        elif 'PyTorch' in self.builder_name:
            return 'pytorch'
        
        return 'generic'
    
    def _classify_pattern(self) -> str:
        """
        Classify the step builder into testing patterns.
        
        Returns:
            Pattern classification string
        """
        # Custom step patterns
        if self.sagemaker_step_type in ['CradleDataLoading', 'MimsModelRegistrationProcessing']:
            return 'custom_step'
        
        # Framework-specific patterns
        if self.framework in ['xgboost', 'pytorch']:
            return 'framework_specific'
        
        # Standard patterns
        return 'standard'
    
    def _create_variant_tester(self, **kwargs):
        """
        Factory method to create appropriate step type-specific tester.
        
        Returns:
            Step type-specific tester instance
        """
        variant_class = STEP_TYPE_VARIANT_MAP.get(self.sagemaker_step_type, self.__class__)
        
        if variant_class == self.__class__:
            # No specific variant, use base functionality
            return None
        
        return variant_class(self.builder_class, **kwargs)
    
    def run_all_tests(self) -> TestResults:
        """
        Run all appropriate tests for the step builder.
        
        Returns:
            Comprehensive test results
        """
        results = TestResults()
        
        # Run universal interface tests
        results.add_section("Universal Interface Tests", self._run_universal_tests())
        
        # Run step type-specific tests if variant tester exists
        if self.variant_tester:
            results.add_section("Step Type-Specific Tests", self.variant_tester.run_variant_tests())
        
        # Run pattern-specific tests
        results.add_section("Pattern-Specific Tests", self._run_pattern_tests())
        
        return results
    
    def _run_universal_tests(self) -> List[TestResult]:
        """Run tests that apply to all step builders."""
        tests = [
            self.test_interface_compliance,
            self.test_configuration_validation,
            self.test_step_name_generation,
            self.test_specification_integration,
            self.test_dependency_resolution,
        ]
        
        return [test() for test in tests]
    
    def _run_pattern_tests(self) -> List[TestResult]:
        """Run tests specific to the detected pattern."""
        if self.pattern == 'custom_step':
            return self._run_custom_step_tests()
        elif self.pattern == 'framework_specific':
            return self._run_framework_specific_tests()
        else:
            return self._run_standard_pattern_tests()
```

### 2. Step Type-Specific Variant Testers

#### Processing Step Tester
```python
class ProcessingStepTester(UniversalStepBuilderTester):
    """
    Specialized tester for Processing step builders.
    
    Tests Processing-specific functionality including processor creation,
    environment variables, job arguments, and job type variants.
    """
    
    def run_variant_tests(self) -> List[TestResult]:
        """Run Processing step-specific tests."""
        tests = [
            self.test_processor_creation,
            self.test_processing_inputs_outputs,
            self.test_environment_variables,
            self.test_job_arguments,
            self.test_job_type_variants,
            self.test_script_path_handling,
        ]
        
        return [test() for test in tests]
    
    def test_processor_creation(self) -> TestResult:
        """Test that the processor is created correctly."""
        try:
            builder = self._create_test_builder()
            processor = builder._create_processor()
            
            # Validate processor type
            expected_types = ['SKLearnProcessor', 'XGBoostProcessor']
            processor_type = type(processor).__name__
            
            if processor_type not in expected_types:
                return TestResult.failure(f"Unexpected processor type: {processor_type}")
            
            # Validate processor configuration
            required_attrs = ['role', 'instance_type', 'instance_count', 'volume_size_in_gb']
            for attr in required_attrs:
                if not hasattr(processor, attr):
                    return TestResult.failure(f"Processor missing required attribute: {attr}")
            
            return TestResult.success("Processor creation validated")
            
        except Exception as e:
            return TestResult.error(f"Processor creation failed: {e}")
    
    def test_processing_inputs_outputs(self) -> TestResult:
        """Test ProcessingInput and ProcessingOutput object creation."""
        try:
            builder = self._create_test_builder()
            
            # Test input creation
            test_inputs = {'input_data': 's3://test-bucket/input/'}
            processing_inputs = builder._get_inputs(test_inputs)
            
            if not isinstance(processing_inputs, list):
                return TestResult.failure("_get_inputs should return a list")
            
            for proc_input in processing_inputs:
                if not hasattr(proc_input, 'source') or not hasattr(proc_input, 'destination'):
                    return TestResult.failure("ProcessingInput missing required attributes")
            
            # Test output creation
            test_outputs = {'output_data': 's3://test-bucket/output/'}
            processing_outputs = builder._get_outputs(test_outputs)
            
            if not isinstance(processing_outputs, list):
                return TestResult.failure("_get_outputs should return a list")
            
            for proc_output in processing_outputs:
                if not hasattr(proc_output, 'source') or not hasattr(proc_output, 'destination'):
                    return TestResult.failure("ProcessingOutput missing required attributes")
            
            return TestResult.success("Processing inputs/outputs validated")
            
        except Exception as e:
            return TestResult.error(f"Processing inputs/outputs test failed: {e}")
    
    def test_environment_variables(self) -> TestResult:
        """Test environment variable construction."""
        try:
            builder = self._create_test_builder()
            env_vars = builder._get_environment_variables()
            
            if not isinstance(env_vars, dict):
                return TestResult.failure("Environment variables should be a dictionary")
            
            # Check for common environment variables
            expected_vars = ['SAGEMAKER_PROGRAM', 'SAGEMAKER_SUBMIT_DIRECTORY']
            for var in expected_vars:
                if var not in env_vars:
                    return TestResult.warning(f"Missing common environment variable: {var}")
            
            # Validate JSON serialization for complex variables
            for key, value in env_vars.items():
                if isinstance(value, (dict, list)):
                    try:
                        json.dumps(value)
                    except (TypeError, ValueError):
                        return TestResult.failure(f"Environment variable {key} not JSON serializable")
            
            return TestResult.success("Environment variables validated")
            
        except Exception as e:
            return TestResult.error(f"Environment variables test failed: {e}")
    
    def test_job_arguments(self) -> TestResult:
        """Test job argument construction."""
        try:
            builder = self._create_test_builder()
            
            if hasattr(builder, '_get_job_arguments'):
                job_args = builder._get_job_arguments()
                
                if job_args is not None:
                    if not isinstance(job_args, list):
                        return TestResult.failure("Job arguments should be a list or None")
                    
                    # Validate argument format
                    for i, arg in enumerate(job_args):
                        if not isinstance(arg, str):
                            return TestResult.failure(f"Job argument {i} should be a string")
            
            return TestResult.success("Job arguments validated")
            
        except Exception as e:
            return TestResult.error(f"Job arguments test failed: {e}")
    
    def test_job_type_variants(self) -> TestResult:
        """Test different job type behaviors if supported."""
        try:
            if not hasattr(self.builder_class, 'job_type'):
                return TestResult.skip("Step builder does not support job types")
            
            job_types = ['training', 'validation', 'testing', 'calibration']
            results = []
            
            for job_type in job_types:
                try:
                    config = self._create_test_config()
                    config.job_type = job_type
                    builder = self.builder_class(config)
                    
                    # Test that builder can be created with different job types
                    builder.validate_configuration()
                    results.append(f"{job_type}: OK")
                    
                except Exception as e:
                    results.append(f"{job_type}: {str(e)}")
            
            return TestResult.success(f"Job type variants tested: {', '.join(results)}")
            
        except Exception as e:
            return TestResult.error(f"Job type variants test failed: {e}")
```

#### Training Step Tester
```python
class TrainingStepTester(UniversalStepBuilderTester):
    """
    Specialized tester for Training step builders.
    
    Tests Training-specific functionality including estimator creation,
    hyperparameter handling, training inputs, and output path handling.
    """
    
    def run_variant_tests(self) -> List[TestResult]:
        """Run Training step-specific tests."""
        tests = [
            self.test_estimator_creation,
            self.test_hyperparameter_handling,
            self.test_training_inputs,
            self.test_data_channel_strategy,
            self.test_output_path_handling,
            self.test_metric_definitions,
        ]
        
        return [test() for test in tests]
    
    def test_estimator_creation(self) -> TestResult:
        """Test that the estimator is created correctly."""
        try:
            builder = self._create_test_builder()
            estimator = builder._create_estimator()
            
            # Validate estimator type
            expected_types = ['PyTorch', 'XGBoost', 'TensorFlow', 'SKLearn']
            estimator_type = type(estimator).__name__
            
            if not any(expected_type in estimator_type for expected_type in expected_types):
                return TestResult.failure(f"Unexpected estimator type: {estimator_type}")
            
            # Validate estimator configuration
            required_attrs = ['role', 'instance_type', 'instance_count']
            for attr in required_attrs:
                if not hasattr(estimator, attr):
                    return TestResult.failure(f"Estimator missing required attribute: {attr}")
            
            return TestResult.success("Estimator creation validated")
            
        except Exception as e:
            return TestResult.error(f"Estimator creation failed: {e}")
    
    def test_hyperparameter_handling(self) -> TestResult:
        """Test hyperparameter processing."""
        try:
            builder = self._create_test_builder()
            
            # Test with hyperparameters object
            if hasattr(builder.config, 'hyperparameters') and builder.config.hyperparameters:
                estimator = builder._create_estimator()
                
                if hasattr(estimator, 'hyperparameters'):
                    hyperparams = estimator.hyperparameters
                    
                    if not isinstance(hyperparams, dict):
                        return TestResult.failure("Hyperparameters should be converted to dict")
                    
                    # Check for framework-specific hyperparameter handling
                    if self.framework == 'xgboost':
                        # XGBoost may use file-based hyperparameters
                        if 'hyperparameters_s3_uri' in hyperparams:
                            return TestResult.success("XGBoost file-based hyperparameters detected")
                    
                    return TestResult.success("Hyperparameter handling validated")
            
            return TestResult.skip("No hyperparameters to test")
            
        except Exception as e:
            return TestResult.error(f"Hyperparameter handling test failed: {e}")
    
    def test_training_inputs(self) -> TestResult:
        """Test TrainingInput object creation."""
        try:
            builder = self._create_test_builder()
            
            # Test input creation
            test_inputs = {'input_path': 's3://test-bucket/data/'}
            training_inputs = builder._get_inputs(test_inputs)
            
            if not isinstance(training_inputs, dict):
                return TestResult.failure("_get_inputs should return a dict for training steps")
            
            # Validate TrainingInput objects
            for channel_name, training_input in training_inputs.items():
                if not hasattr(training_input, 's3_data'):
                    return TestResult.failure(f"TrainingInput for {channel_name} missing s3_data")
            
            # Check data channel strategy
            if self.framework == 'pytorch':
                # PyTorch typically uses single 'data' channel
                if 'data' not in training_inputs:
                    return TestResult.warning("PyTorch step missing 'data' channel")
            elif self.framework == 'xgboost':
                # XGBoost typically uses multiple channels
                expected_channels = ['train', 'validation']
                for channel in expected_channels:
                    if channel not in training_inputs:
                        return TestResult.warning(f"XGBoost step missing '{channel}' channel")
            
            return TestResult.success("Training inputs validated")
            
        except Exception as e:
            return TestResult.error(f"Training inputs test failed: {e}")
    
    def test_output_path_handling(self) -> TestResult:
        """Test output path generation."""
        try:
            builder = self._create_test_builder()
            
            # Test output path generation
            test_outputs = {}
            output_path = builder._get_outputs(test_outputs)
            
            if not isinstance(output_path, str):
                return TestResult.failure("Training step should return string output path")
            
            if not output_path.startswith('s3://'):
                return TestResult.failure("Output path should be S3 URI")
            
            return TestResult.success("Output path handling validated")
            
        except Exception as e:
            return TestResult.error(f"Output path handling test failed: {e}")
```

#### CreateModel Step Tester
```python
class CreateModelStepTester(UniversalStepBuilderTester):
    """
    Specialized tester for CreateModel step builders.
    
    Tests CreateModel-specific functionality including model creation,
    image URI generation, and model data processing.
    """
    
    def run_variant_tests(self) -> List[TestResult]:
        """Run CreateModel step-specific tests."""
        tests = [
            self.test_model_creation,
            self.test_image_uri_generation,
            self.test_model_data_processing,
            self.test_step_arguments,
        ]
        
        return [test() for test in tests]
    
    def test_model_creation(self) -> TestResult:
        """Test that the model is created correctly."""
        try:
            builder = self._create_test_builder()
            
            # Test model creation with mock model data
            test_model_data = 's3://test-bucket/model/model.tar.gz'
            model = builder._create_model(test_model_data)
            
            # Validate model type
            expected_types = ['XGBoostModel', 'PyTorchModel', 'TensorFlowModel']
            model_type = type(model).__name__
            
            if not any(expected_type in model_type for expected_type in expected_types):
                return TestResult.failure(f"Unexpected model type: {model_type}")
            
            # Validate model configuration
            required_attrs = ['model_data', 'role', 'entry_point']
            for attr in required_attrs:
                if not hasattr(model, attr):
                    return TestResult.failure(f"Model missing required attribute: {attr}")
            
            return TestResult.success("Model creation validated")
            
        except Exception as e:
            return TestResult.error(f"Model creation failed: {e}")
    
    def test_image_uri_generation(self) -> TestResult:
        """Test container image URI generation."""
        try:
            builder = self._create_test_builder()
            
            if hasattr(builder, '_get_image_uri'):
                image_uri = builder._get_image_uri()
                
                if not isinstance(image_uri, str):
                    return TestResult.failure("Image URI should be a string")
                
                if not image_uri.startswith('246618743249.dkr.ecr'):
                    return TestResult.failure("Image URI should be ECR URI")
                
                # Check framework-specific image URI patterns
                if self.framework == 'xgboost' and 'xgboost' not in image_uri:
                    return TestResult.failure("XGBoost image URI should contain 'xgboost'")
                elif self.framework == 'pytorch' and 'pytorch' not in image_uri:
                    return TestResult.failure("PyTorch image URI should contain 'pytorch'")
                
                return TestResult.success("Image URI generation validated")
            
            return TestResult.skip("No image URI generation method found")
            
        except Exception as e:
            return TestResult.error(f"Image URI generation test failed: {e}")
    
    def test_model_data_processing(self) -> TestResult:
        """Test model data input processing."""
        try:
            builder = self._create_test_builder()
            
            # Test model data processing
            test_inputs = {'model_data': 's3://test-bucket/model/model.tar.gz'}
            processed_inputs = builder._get_inputs(test_inputs)
            
            if not isinstance(processed_inputs, dict):
                return TestResult.failure("_get_inputs should return a dict")
            
            if 'model_data' not in processed_inputs:
                return TestResult.failure("Processed inputs should contain model_data")
            
            return TestResult.success("Model data processing validated")
            
        except Exception as e:
            return TestResult.error(f"Model data processing test failed: {e}")
```

#### Transform Step Tester
```python
class TransformStepTester(UniversalStepBuilderTester):
    """
    Specialized tester for Transform step builders.
    
    Tests Transform-specific functionality including transformer creation,
    transform input processing, and model integration.
    """
    
    def run_variant_tests(self) -> List[TestResult]:
        """Run Transform step-specific tests."""
        tests = [
            self.test_transformer_creation,
            self.test_transform_input_processing,
            self.test_model_integration,
            self.test_input_processing_options,
        ]
        
        return [test() for test in tests]
    
    def test_transformer_creation(self) -> TestResult:
        """Test that the transformer is created correctly."""
        try:
            builder = self._create_test_builder()
            
            # Test transformer creation with mock model name
            test_model_name = 'test-model-name'
            transformer = builder._create_transformer(test_model_name)
            
            # Validate transformer type
            if type(transformer).__name__ != 'Transformer':
                return TestResult.failure(f"Expected Transformer, got {type(transformer).__name__}")
            
            # Validate transformer configuration
            required_attrs = ['model_name', 'instance_type', 'instance_count']
            for attr in required_attrs:
                if not hasattr(transformer, attr):
                    return TestResult.failure(f"Transformer missing required attribute: {attr}")
            
            return TestResult.success("Transformer creation validated")
            
        except Exception as e:
            return TestResult.error(f"Transformer creation failed: {e}")
    
    def test_transform_input_processing(self) -> TestResult:
        """Test TransformInput processing."""
        try:
            builder = self._create_test_builder()
            
            # Test transform input processing
            test_inputs = {
                'model_name': 'test-model',
                'processed_data': 's3://test-bucket/data/'
            }
            transform_input, model_name = builder._get_inputs(test_inputs)
            
            # Validate TransformInput
            if not hasattr(transform_input, 'data'):
                return TestResult.failure("TransformInput missing data attribute")
            
            # Validate model name extraction
            if model_name != 'test-model':
                return TestResult.failure("Model name not extracted correctly")
            
            return TestResult.success("Transform input processing validated")
            
        except Exception as e:
            return TestResult.error(f"Transform input processing test failed: {e}")
```

#### Custom Step Tester
```python
class CustomStepTester(UniversalStepBuilderTester):
    """
    Specialized tester for custom step builders.
    
    Provides basic validation for custom steps that don't follow
    standard SageMaker patterns.
    """
    
    def run_variant_tests(self) -> List[TestResult]:
        """Run custom step-specific tests."""
        tests = [
            self.test_basic_interface,
            self.test_configuration_handling,
        ]
        
        return [test() for test in tests]
    
    def test_basic_interface(self) -> TestResult:
        """Test basic interface compliance for custom steps."""
        try:
            builder = self._create_test_builder()
            
            # Test that create_step method exists and is callable
            if not hasattr(builder, 'create_step'):
                return TestResult.failure("Custom step missing create_step method")
            
            if not callable(getattr(builder, 'create_step')):
                return TestResult.failure("create_step is not callable")
            
            return TestResult.success("Basic interface validated")
            
        except Exception as e:
            return TestResult.error(f"Basic interface test failed: {e}")
    
    def test_configuration_handling(self) -> TestResult:
        """Test configuration handling for custom steps."""
        try:
            builder = self._create_test_builder()
            
            # Test configuration validation if method exists
            if hasattr(builder, 'validate_configuration'):
                builder.validate_configuration()
            
            return TestResult.success("Configuration handling validated")
            
        except Exception as e:
            return TestResult.error(f"Configuration handling test failed: {e}")
```

### 3. Step Type Variant Registry

```python
# Registry mapping SageMaker step types to their specific testers
STEP_TYPE_VARIANT_MAP = {
    'Processing': ProcessingStepTester,
    'Training': TrainingStepTester,
    'CreateModel': CreateModelStepTester,
    'Transform': TransformStepTester,
    'CradleDataLoading': CustomStepTester,
    'MimsModelRegistrationProcessing': CustomStepTester,
}

# Framework-specific test configurations
FRAMEWORK_TEST_CONFIGS = {
    'xgboost': {
        'expected_processor_types': ['XGBoostProcessor'],
        'expected_estimator_types': ['XGBoost'],
        'expected_model_types': ['XGBoostModel'],
        'hyperparameter_handling': 'file_based',
        'data_channel_strategy': 'multiple',
    },
    'pytorch': {
        'expected_processor_types': ['SKLearnProcessor'],
        'expected_estimator_types': ['PyTorch'],
        'expected_model_types': ['PyTorchModel'],
        'hyperparameter_handling': 'direct',
        'data_channel_strategy': 'single',
    },
    'sklearn': {
        'expected_processor_types': ['SKLearnProcessor'],
        'expected_estimator_types': ['SKLearn'],
        'expected_model_types': ['SKLearnModel'],
        'hyperparameter_handling': 'direct',
        'data_channel_strategy': 'multiple',
    },
}
```

### 4. Test Results and Reporting

```python
class TestResult:
    """Represents the result of a single test."""
    
    def __init__(self, status: str, message: str, details: Optional[Dict] = None):
        self.status = status  # 'success', 'failure', 'error', 'skip', 'warning'
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now()
    
    @classmethod
    def success(cls, message: str, **details) -> 'TestResult':
        return cls('success', message, details)
    
    @classmethod
    def failure(cls, message: str, **details) -> 'TestResult':
        return cls('failure', message, details)
    
    @classmethod
    def error(cls, message: str, **details) -> 'TestResult':
        return cls('error', message, details)
    
    @classmethod
    def skip(cls, message: str, **details) -> 'TestResult':
        return cls('skip', message, details)
    
    @classmethod
    def warning(cls, message: str, **details) -> 'TestResult':
        return cls('warning', message, details)

class TestResults:
    """Aggregates multiple test results with reporting capabilities."""
    
    def __init__(self):
        self.sections = {}
        self.start_time = datetime.now()
        self.end_time = None
    
    def add_section(self, section_name: str, results: List[TestResult]):
        self.sections[section_name] = results
    
    def finalize(self):
        self.end_time = datetime.now()
    
    def get_summary(self) -> Dict[str, int]:
        """Get summary statistics of test results."""
        summary = {'success': 0, 'failure': 0, 'error': 0, 'skip': 0, 'warning': 0}
        
        for results in self.sections.values():
            for result in results:
                summary[result.status] += 1
        
        return summary
    
    def generate_report(self) -> str:
        """Generate a comprehensive test report."""
        report = []
        report.append("=" * 80)
        report.append("ENHANCED UNIVERSAL STEP BUILDER TEST REPORT")
        report.append("=" * 80)
        
        # Summary
        summary = self.get_summary()
        report.append(f"\nSUMMARY:")
        report.append(f"  Total Tests: {sum(summary.values())}")
        report.append(f"  Successes: {summary['success']}")
        report.append(f"  Failures: {summary['failure']}")
        report.append(f"  Errors: {summary['error']}")
        report.append(f"  Warnings: {summary['warning']}")
        report.append(f"  Skipped: {summary['skip']}")
        
        # Detailed results by section
        for section_name, results in self.sections.items():
            report.append(f"\n{section_name.upper()}:")
            report.append("-" * len(section_name))
            
            for result in results:
                status_symbol = {
                    'success': '✓',
                    'failure': '✗',
                    'error': '⚠',
                    'skip': '○',
                    'warning': '!'
                }.get(result.status, '?')
                
                report.append(f"  {status_symbol} {result.message}")
                
                if result.details:
                    for key, value in result.details.items():
                        report.append(f"    {key}: {value}")
        
        # Execution time
        if self.end_time:
            duration = self.end_time - self.start_time
            report.append(f"\nExecution Time: {duration.total_seconds():.2f} seconds")
        
        return "\n".join(report)
```

### 5. Usage Examples

```python
# Example 1: Test a Processing step builder
def test_processing_step_builder():
    from src.cursus.steps.builders.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
    
    # Create enhanced universal tester
    tester = UniversalStepBuilderTester(TabularPreprocessingStepBuilder)
    
    # Run all tests
    results = tester.run_all_tests()
    
    # Generate report
    print(results.generate_report())
    
    # Check if all tests passed
    summary = results.get_summary()
    if summary['failure'] == 0 and summary['error'] == 0:
        print("All tests passed!")
    else:
        print(f"Tests failed: {summary['failure']} failures, {summary['error']} errors")

# Example 2: Test a Training step builder
def test_training_step_builder():
    from src.cursus.steps.builders.builder_training_step_pytorch import PyTorchTrainingStepBuilder
    
    # Create enhanced universal tester with custom configuration
    tester = UniversalStepBuilderTester(
        PyTorchTrainingStepBuilder,
        mock_sagemaker_session=True,
        test_role='arn:aws:iam::123456789012:role/SageMakerRole'
    )
    
    # Run all tests
    results = tester.run_all_tests()
    
    # Access specific test results
    training_tests = results.sections.get('Step Type-Specific Tests', [])
    for test_result in training_tests:
        if test_result.status == 'failure':
            print(f"Training test failed: {test_result.message}")

# Example 3: Batch test multiple step builders
def test_all_step_builders():
    from src.cursus.steps.registry.builder_registry import get_all_registered_builders
    
    all_results = {}
    
    for builder_name, builder_class in get_all_registered_builders().items():
        print(f"Testing {builder_name}...")
        
        try:
            tester = UniversalStepBuilderTester(builder_class)
            results = tester.run_all_tests()
            all_results[builder_name] = results
            
            summary = results.get_summary()
            print(f"  {summary['success']} passed, {summary['failure']} failed, {summary['error']} errors")
            
        except Exception as e:
            print(f"  Failed to test {builder_name}: {e}")
    
    # Generate combined report
    generate_combined_report(all_results)
```

## Implementation Strategy

### Phase 1: Core Framework
1. **Base Universal Tester**: Implement the enhanced base class with detection logic
2. **Test Result System**: Implement TestResult and TestResults classes
3. **Pattern Detection**: Implement SageMaker step type and framework detection
4. **Variant Registry**: Set up the step type variant mapping system

### Phase 2: Step Type Variants
1. **Processing Step Tester**: Implement comprehensive Processing step testing
2. **Training Step Tester**: Implement Training step-specific validation
3. **CreateModel Step Tester**: Implement CreateModel step testing
4. **Transform Step Tester**: Implement Transform step validation
5. **Custom Step Tester**: Implement basic custom step validation

### Phase 3: Advanced Features
1. **Framework-Specific Testing**: Implement framework-aware test configurations
2. **Specification Integration**: Deep integration with step specifications
3. **Dependency Testing**: Advanced dependency resolution testing
4. **Performance Testing**: Add performance and resource usage testing

### Phase 4: Integration and Tooling
1. **CLI Integration**: Command-line interface for running tests
2. **CI/CD Integration**: Integration with continuous integration systems
3. **Reporting Tools**: Advanced reporting and visualization
4. **Documentation**: Comprehensive documentation and examples

## Benefits of Enhanced Design

### 1. Comprehensive Coverage
- **Step Type Awareness**: Different testing strategies for different step types
- **Framework Specificity**: Framework-aware testing for better validation
- **Pattern Recognition**: Automatic detection of implementation patterns

### 2. Maintainability
- **Modular Design**: Each step type has its own specialized tester
- **Extensible Architecture**: Easy to add new step types and frameworks
- **Clear Separation**: Universal tests vs step-specific tests

### 3. Quality Assurance
- **Deeper Validation**: Step type-specific validation catches more issues
- **Framework Compliance**: Ensures proper framework usage patterns
- **Specification Adherence**: Validates compliance with step specifications

### 4. Developer Experience
- **Automatic Detection**: No manual configuration of test types
- **Comprehensive Reports**: Detailed reporting with actionable insights
- **Easy Integration**: Simple API for testing individual or multiple builders

## Migration Path

### From Current Universal Tester
1. **Backward Compatibility**: Enhanced tester maintains existing interface
2. **Gradual Migration**: Can be introduced alongside existing tester
3. **Progressive Enhancement**: Add step type variants incrementally
4. **Configuration Migration**: Existing test configurations remain valid

### Integration with Existing Systems
1. **Registry Integration**: Leverages existing builder registry system
2. **Specification System**: Integrates with existing step specifications
3. **Configuration System**: Works with existing configuration classes
4. **Testing Infrastructure**: Builds on existing testing patterns

## Conclusion

The enhanced universal step builder tester design provides a comprehensive, step type-aware testing framework that addresses the diverse patterns and requirements across different SageMaker step types. By combining universal interface testing with step type-specific validation, this design ensures robust quality assurance while maintaining flexibility and extensibility.

The hierarchical testing approach (universal → step type → pattern → framework) provides comprehensive coverage while avoiding unnecessary complexity for simpler step builders. The automatic detection and classification system makes the framework easy to use while providing powerful customization capabilities for advanced scenarios.

This design serves as the foundation for implementing a robust, maintainable, and comprehensive testing framework that can evolve with the cursus step builder ecosystem while ensuring high quality and reliability across all step implementations.

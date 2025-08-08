---
tags:
  - design
  - step_builders
  - processing_steps
  - patterns
  - sagemaker
keywords:
  - processing step patterns
  - SKLearnProcessor
  - XGBoostProcessor
  - specification driven
  - contract based
  - environment variables
topics:
  - step builder patterns
  - processing step implementation
  - SageMaker processing
  - step builder architecture
language: python
date of note: 2025-01-08
---

# Processing Step Builder Patterns

## Overview

This document analyzes the common patterns found in Processing step builder implementations in the cursus framework. Processing steps are the most common step type, with implementations including TabularPreprocessing, Package, CurrencyConversion, RiskTableMapping, ModelCalibration, XGBoostModelEval, Payload, and DummyTraining steps.

## SageMaker Step Type Classification

All Processing step builders create **ProcessingStep** instances using various processor types:
- **SKLearnProcessor**: Standard processing (TabularPreprocessing, Package, CurrencyConversion, etc.)
- **XGBoostProcessor**: XGBoost-specific processing (XGBoostModelEval)
- **Custom processors**: Framework-specific processing needs

## Common Implementation Patterns

### 1. Base Architecture Pattern

All Processing step builders follow this consistent architecture:

```python
@register_builder()
class ProcessingStepBuilder(StepBuilderBase):
    def __init__(self, config, sagemaker_session=None, role=None, notebook_root=None, 
                 registry_manager=None, dependency_resolver=None):
        # Specification loading based on job_type (if applicable)
        spec = self._load_specification_by_job_type(config.job_type)
        super().__init__(config=config, spec=spec, ...)
        
    def validate_configuration(self) -> None:
        # Validate required processing configuration
        
    def _create_processor(self) -> Processor:
        # Create appropriate processor (SKLearn, XGBoost, etc.)
        
    def _get_environment_variables(self) -> Dict[str, str]:
        # Build environment variables for processing job
        
    def _get_inputs(self, inputs) -> List[ProcessingInput]:
        # Create ProcessingInput objects using specification
        
    def _get_outputs(self, outputs) -> List[ProcessingOutput]:
        # Create ProcessingOutput objects using specification
        
    def _get_job_arguments(self) -> List[str]:
        # Build command-line arguments for processing script
        
    def create_step(self, **kwargs) -> ProcessingStep:
        # Orchestrate step creation
```

### 2. Job Type-Based Specification Loading Pattern

Many Processing steps support multiple job types (training, validation, testing, calibration):

```python
# Pattern found in TabularPreprocessing, CurrencyConversion
def __init__(self, config, ...):
    job_type = config.job_type.lower()
    
    if job_type == "training" and TRAINING_SPEC is not None:
        spec = TRAINING_SPEC
    elif job_type == "calibration" and CALIBRATION_SPEC is not None:
        spec = CALIBRATION_SPEC
    # ... other job types
    else:
        # Try dynamic import
        try:
            module_path = f"..pipeline_step_specs.{step_name}_{job_type}_spec"
            module = importlib.import_module(module_path, package=__package__)
            spec = getattr(module, f"{STEP_NAME}_{job_type.upper()}_SPEC")
        except (ImportError, AttributeError):
            raise ValueError(f"No specification found for job type: {job_type}")
```

### 3. Processor Creation Patterns

#### Standard SKLearnProcessor Pattern
```python
def _create_processor(self) -> SKLearnProcessor:
    instance_type = (self.config.processing_instance_type_large 
                    if self.config.use_large_processing_instance 
                    else self.config.processing_instance_type_small)
    
    return SKLearnProcessor(
        framework_version=self.config.processing_framework_version,
        role=self.role,
        instance_type=instance_type,
        instance_count=self.config.processing_instance_count,
        volume_size_in_gb=self.config.processing_volume_size,
        base_job_name=self._generate_job_name(),
        sagemaker_session=self.session,
        env=self._get_environment_variables(),
    )
```

#### Custom Framework Processor Pattern (XGBoost)
```python
def _create_processor(self) -> XGBoostProcessor:
    return XGBoostProcessor(
        framework_version=self.config.framework_version,
        py_version=self.config.py_version,
        role=self.role,
        instance_type=self.config.processing_instance_type,
        instance_count=self.config.processing_instance_count,
        volume_size_in_gb=self.config.processing_volume_size,
        base_job_name=self._generate_job_name(),
        sagemaker_session=self.session,
        env=self._get_environment_variables(),
    )
```

### 4. Environment Variables Pattern

Processing steps use environment variables to pass configuration to scripts:

```python
def _get_environment_variables(self) -> Dict[str, str]:
    # Get base environment variables from contract
    env_vars = super()._get_environment_variables()
    
    # Add step-specific environment variables
    if hasattr(self.config, 'label_name'):
        env_vars["LABEL_FIELD"] = self.config.label_name
    
    # Add list/dict configurations as JSON
    if hasattr(self.config, 'categorical_columns') and self.config.categorical_columns:
        env_vars["CATEGORICAL_COLUMNS"] = ",".join(self.config.categorical_columns)
    
    # Add complex configurations as JSON
    if hasattr(self.config, 'currency_conversion_dict'):
        env_vars["CURRENCY_CONVERSION_DICT"] = json.dumps(self.config.currency_conversion_dict)
    
    return env_vars
```

### 5. Specification-Driven Input/Output Pattern

All modern Processing steps use specifications to define inputs and outputs:

```python
def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
    if not self.spec or not self.contract:
        raise ValueError("Step specification and contract are required")
        
    processing_inputs = []
    
    for _, dependency_spec in self.spec.dependencies.items():
        logical_name = dependency_spec.logical_name
        
        # Skip optional inputs not provided
        if not dependency_spec.required and logical_name not in inputs:
            continue
            
        # Validate required inputs
        if dependency_spec.required and logical_name not in inputs:
            raise ValueError(f"Required input '{logical_name}' not provided")
        
        # Get container path from contract
        container_path = self.contract.expected_input_paths[logical_name]
        
        processing_inputs.append(ProcessingInput(
            input_name=logical_name,
            source=inputs[logical_name],
            destination=container_path
        ))
        
    return processing_inputs

def _get_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
    if not self.spec or not self.contract:
        raise ValueError("Step specification and contract are required")
        
    processing_outputs = []
    
    for _, output_spec in self.spec.outputs.items():
        logical_name = output_spec.logical_name
        container_path = self.contract.expected_output_paths[logical_name]
        
        # Use provided destination or generate default
        destination = outputs.get(logical_name) or self._generate_output_path(logical_name)
        
        processing_outputs.append(ProcessingOutput(
            output_name=logical_name,
            source=container_path,
            destination=destination
        ))
        
    return processing_outputs
```

### 6. Job Arguments Pattern

Processing steps pass arguments to their scripts in different ways:

#### Simple Job Type Pattern
```python
def _get_job_arguments(self) -> List[str]:
    # Simple job type argument
    return ["--job_type", self.config.job_type]
```

#### Complex Arguments Pattern
```python
def _get_job_arguments(self) -> List[str]:
    args = [
        "--job-type", self.config.job_type,
        "--mode", self.config.mode,
        "--marketplace-id-col", self.config.marketplace_id_col,
        "--enable-conversion", str(self.config.enable_currency_conversion).lower()
    ]
    
    # Add optional arguments
    if hasattr(self.config, "currency_col") and self.config.currency_col:
        args.extend(["--currency-col", self.config.currency_col])
        
    return args
```

#### No Arguments Pattern
```python
def _get_job_arguments(self) -> Optional[List[str]]:
    # Some steps use only environment variables
    return None
```

### 7. Special Input Handling Patterns

#### Local Path Override Pattern (Package Step)
```python
def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
    # Special handling for inference scripts - always use local path
    inference_scripts_key = "inference_scripts_input"
    inference_scripts_path = self.config.source_dir or "inference"
    
    # Override any dependency-provided value with local path
    if inference_scripts_key in inputs:
        self.log_info("Overriding dependency value with local path: %s", inference_scripts_path)
        del inputs[inference_scripts_key]  # Remove from working inputs
    
    # Add local path input
    processing_inputs.append(ProcessingInput(
        input_name=inference_scripts_key,
        source=inference_scripts_path,
        destination=self.contract.expected_input_paths[inference_scripts_key]
    ))
    
    # Process remaining inputs normally...
```

### 8. Step Creation Pattern

All Processing steps follow this orchestration pattern:

```python
def create_step(self, **kwargs) -> ProcessingStep:
    # Extract parameters
    inputs_raw = kwargs.get('inputs', {})
    outputs = kwargs.get('outputs', {})
    dependencies = kwargs.get('dependencies', [])
    enable_caching = kwargs.get('enable_caching', True)
    
    # Handle inputs from dependencies and explicit inputs
    inputs = {}
    if dependencies:
        extracted_inputs = self.extract_inputs_from_dependencies(dependencies)
        inputs.update(extracted_inputs)
    inputs.update(inputs_raw)
    
    # Create components
    processor = self._create_processor()
    proc_inputs = self._get_inputs(inputs)
    proc_outputs = self._get_outputs(outputs)
    job_args = self._get_job_arguments()
    
    # Get standardized step name
    step_name = self._get_step_name()
    
    # Create step
    step = ProcessingStep(
        name=step_name,
        processor=processor,
        inputs=proc_inputs,
        outputs=proc_outputs,
        code=self.config.get_script_path(),
        job_arguments=job_args,
        depends_on=dependencies,
        cache_config=self._get_cache_config(enable_caching)
    )
    
    # Attach specification for future reference
    setattr(step, '_spec', self.spec)
    
    return step
```

## Configuration Validation Patterns

### Standard Processing Configuration
```python
def validate_configuration(self) -> None:
    required_attrs = [
        'processing_instance_count', 'processing_volume_size',
        'processing_instance_type_large', 'processing_instance_type_small',
        'processing_framework_version', 'use_large_processing_instance'
    ]
    
    for attr in required_attrs:
        if not hasattr(self.config, attr) or getattr(self.config, attr) is None:
            raise ValueError(f"Missing required attribute: {attr}")
```

### Job Type Validation
```python
def validate_configuration(self) -> None:
    # Validate job type
    if self.config.job_type not in ["training", "validation", "testing", "calibration"]:
        raise ValueError(f"Invalid job_type: {self.config.job_type}")
```

### Step-Specific Validation
```python
def validate_configuration(self) -> None:
    # Currency conversion specific validation
    if self.config.enable_currency_conversion:
        if not self.config.marketplace_id_col:
            raise ValueError("marketplace_id_col required when conversion enabled")
        if not self.config.currency_conversion_var_list:
            raise ValueError("currency_conversion_var_list cannot be empty")
```

## Key Differences Between Processing Step Types

### 1. By Processor Type
- **SKLearnProcessor**: Most common, used for general data processing
- **XGBoostProcessor**: Used for XGBoost-specific processing with custom framework versions

### 2. By Job Type Support
- **Multi-job-type**: TabularPreprocessing, CurrencyConversion (training/validation/testing/calibration)
- **Single-purpose**: Package, Payload, ModelCalibration (specific function)

### 3. By Input Complexity
- **Simple inputs**: Single data input (RiskTableMapping)
- **Complex inputs**: Multiple data sources, metadata, signatures (TabularPreprocessing)
- **Special handling**: Local path overrides (Package step)

### 4. By Environment Variable Usage
- **Simple env vars**: Basic configuration (Package)
- **Complex env vars**: JSON serialized objects, lists (CurrencyConversion)
- **Script-driven**: Minimal env vars, script handles logic (some steps)

## Best Practices Identified

1. **Specification-Driven Design**: All modern steps use specifications for input/output definitions
2. **Contract-Based Path Mapping**: Container paths come from script contracts
3. **Standardized Job Naming**: Use `_generate_job_name()` for consistent naming
4. **Environment Variable Patterns**: Use env vars for configuration, arguments for runtime parameters
5. **Dependency Resolution**: Support both explicit inputs and dependency extraction
6. **Error Handling**: Comprehensive validation with clear error messages
7. **Logging**: Detailed logging for debugging and monitoring
8. **Specification Attachment**: Attach specs to steps for future reference

## Testing Implications

Processing step builders should be tested for:

1. **Processor Creation**: Correct processor type and configuration
2. **Input/Output Handling**: ProcessingInput/ProcessingOutput object creation
3. **Environment Variables**: Proper variable construction and JSON serialization
4. **Job Arguments**: Correct argument formatting and optional parameter handling
5. **Specification Compliance**: Adherence to step specifications
6. **Contract Integration**: Proper use of script contracts for path mapping
7. **Job Type Variants**: Different behavior for different job types
8. **Error Conditions**: Proper handling of missing inputs, invalid configurations

This pattern analysis provides the foundation for creating comprehensive, type-specific validation in the universal tester framework.

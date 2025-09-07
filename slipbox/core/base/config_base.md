---
tags:
  - code
  - base
  - config_base
  - pipeline_configuration
  - self_contained_derivation
keywords:
  - BasePipelineConfig
  - pipeline configuration
  - self-contained derivation
  - three-tier design
  - field categorization
  - Pydantic models
  - configuration management
topics:
  - pipeline configuration
  - configuration management
  - three-tier design
language: python
date of note: 2024-12-07
---

# Config Base

Base Pipeline Configuration with Self-Contained Derivation Logic that implements the base configuration class for pipeline steps using a self-contained design where each configuration class is responsible for its own field derivations.

## Overview

The `BasePipelineConfig` class provides the foundational configuration system for pipeline steps, implementing a sophisticated three-tier design pattern that categorizes fields by their purpose and derivation logic. This self-contained approach ensures that each configuration class manages its own field derivations through private fields and read-only properties.

The three-tier design includes Essential User Inputs (Tier 1) that are required fields users must explicitly provide, System Inputs with Defaults (Tier 2) that have reasonable defaults but can be overridden, and Derived Fields (Tier 3) that are calculated from other fields and stored in private attributes with public read-only properties for access.

The system supports advanced features including automatic field categorization and validation, workspace-aware step registry integration, script contract integration for implementation validation, comprehensive configuration serialization with derived fields, and flexible configuration inheritance patterns for derived classes.

## Classes and Methods

### Classes
- [`BasePipelineConfig`](#basepipelineconfig) - Base configuration class with three-tier design and self-contained derivation logic

## API Reference

### BasePipelineConfig

_class_ cursus.core.base.config_base.BasePipelineConfig(_author_, _bucket_, _role_, _region_, _service_name_, _pipeline_version_, _model_class="xgboost"_, _current_date=None_, _framework_version="2.1.0"_, _py_version="py310"_, _source_dir=None_)

Base configuration with shared pipeline attributes and self-contained derivation logic. This class implements the three-tier design pattern for organizing configuration fields and provides comprehensive validation and derivation capabilities.

**Parameters:**
- **author** (_str_) – Author or owner of the pipeline. Required essential user input.
- **bucket** (_str_) – S3 bucket name for pipeline artifacts and data. Required essential user input.
- **role** (_str_) – IAM role for pipeline execution. Required essential user input.
- **region** (_str_) – Custom region code (NA, EU, FE) for internal logic. Required essential user input.
- **service_name** (_str_) – Service name for the pipeline. Required essential user input.
- **pipeline_version** (_str_) – Version string for the SageMaker Pipeline. Required essential user input.
- **model_class** (_str_) – Model class (e.g., XGBoost, PyTorch). Defaults to "xgboost".
- **current_date** (_Optional[str]_) – Current date for versioning or pathing. Defaults to current date.
- **framework_version** (_str_) – Framework version (e.g., PyTorch). Defaults to "2.1.0".
- **py_version** (_str_) – Python version. Defaults to "py310".
- **source_dir** (_Optional[str]_) – Common source directory for scripts. Defaults to None.

```python
from cursus.core.base.config_base import BasePipelineConfig

# Create base configuration with essential inputs
base_config = BasePipelineConfig(
    author="data-scientist",
    bucket="ml-pipeline-artifacts",
    role="arn:aws:iam::123456789012:role/SageMakerRole",
    region="NA",
    service_name="fraud-detection",
    pipeline_version="1.0.0",
    model_class="xgboost",
    source_dir="src/scripts"
)

print(f"Pipeline name: {base_config.pipeline_name}")
print(f"AWS region: {base_config.aws_region}")
print(f"S3 location: {base_config.pipeline_s3_loc}")
```

### Properties (Derived Fields - Tier 3)

#### aws_region

_property_ aws_region

Get AWS region based on region code. This derived property maps the custom region code to the corresponding AWS region.

**Returns:**
- **str** – AWS region string (e.g., "us-east-1", "eu-west-1", "us-west-2").

```python
# Region mapping examples
config_na = BasePipelineConfig(region="NA", ...)  # -> "us-east-1"
config_eu = BasePipelineConfig(region="EU", ...)  # -> "eu-west-1"
config_fe = BasePipelineConfig(region="FE", ...)  # -> "us-west-2"

print(f"NA maps to: {config_na.aws_region}")
```

#### pipeline_name

_property_ pipeline_name

Get pipeline name derived from author, service_name, model_class, and region. This property creates a standardized pipeline naming convention.

**Returns:**
- **str** – Pipeline name in format "{author}-{service_name}-{model_class}-{region}".

```python
# Pipeline name derivation
config = BasePipelineConfig(
    author="john-doe",
    service_name="recommendation",
    model_class="pytorch",
    region="NA",
    ...
)
print(config.pipeline_name)  # Output: "john-doe-recommendation-pytorch-NA"
```

#### pipeline_description

_property_ pipeline_description

Get pipeline description derived from service_name, model_class, and region. This property creates a human-readable description for the pipeline.

**Returns:**
- **str** – Pipeline description in format "{service_name} {model_class} Model {region}".

```python
print(config.pipeline_description)  # Output: "recommendation pytorch Model NA"
```

#### pipeline_s3_loc

_property_ pipeline_s3_loc

Get S3 location for pipeline artifacts. This property constructs the complete S3 path for storing pipeline artifacts and outputs.

**Returns:**
- **str** – S3 URI in format "s3://{bucket}/MODS/{pipeline_name}_{pipeline_version}".

```python
print(config.pipeline_s3_loc)  
# Output: "s3://ml-pipeline-artifacts/MODS/john-doe-recommendation-pytorch-NA_1.0.0"
```

#### script_contract

_property_ script_contract

Property accessor for script contract. This property provides access to the script contract associated with this configuration, if available.

**Returns:**
- **Optional[ScriptContract]** – Script contract instance or None if not available.

```python
# Access script contract
contract = config.script_contract
if contract:
    print(f"Contract available: {contract.script_path}")
else:
    print("No script contract defined")
```

### Methods

#### categorize_fields

categorize_fields()

Categorize all fields into three tiers based on their characteristics and purpose. This method provides insight into the configuration structure and field organization.

**Returns:**
- **Dict[str, List[str]]** – Dictionary with keys 'essential', 'system', and 'derived' mapping to lists of field names.

```python
# Analyze field categorization
categories = config.categorize_fields()

print("Essential fields:", categories['essential'])
print("System fields:", categories['system'])
print("Derived fields:", categories['derived'])

# Output:
# Essential fields: ['author', 'bucket', 'role', 'region', 'service_name', 'pipeline_version']
# System fields: ['model_class', 'current_date', 'framework_version', 'py_version', 'source_dir']
# Derived fields: ['aws_region', 'pipeline_name', 'pipeline_description', 'pipeline_s3_loc']
```

#### print_config

print_config()

Print complete configuration information organized by tiers. This method provides a comprehensive view of the configuration with fields organized by their tier classification.

```python
# Print organized configuration
config.print_config()

# Output:
# ===== CONFIGURATION =====
# Class: BasePipelineConfig
# 
# ----- Essential User Inputs (Tier 1) -----
# Author: data-scientist
# Bucket: ml-pipeline-artifacts
# ...
# 
# ----- System Inputs with Defaults (Tier 2) -----
# Model_Class: xgboost
# Framework_Version: 2.1.0
# ...
# 
# ----- Derived Fields (Tier 3) -----
# Aws_Region: us-east-1
# Pipeline_Name: data-scientist-fraud-detection-xgboost-NA
# ...
```

#### get_public_init_fields

get_public_init_fields()

Get a dictionary of public fields suitable for initializing a child config. This method extracts all user-provided and system fields that should be propagated to derived configuration classes.

**Returns:**
- **Dict[str, Any]** – Dictionary of field names to values for child initialization.

```python
# Get fields for child configuration
init_fields = config.get_public_init_fields()
print("Fields for child config:", list(init_fields.keys()))

# Use for creating derived configuration
child_config = DerivedConfig(**init_fields, additional_param="value")
```

#### get_script_contract

get_script_contract()

Get script contract for this configuration. This method attempts to load the appropriate script contract based on the configuration class and job type.

**Returns:**
- **Optional[ScriptContract]** – Script contract instance or None if not available.

```python
# Get script contract
contract = config.get_script_contract()
if contract:
    print(f"Script path: {contract.script_path}")
    print(f"Expected inputs: {list(contract.expected_input_paths.keys())}")
    print(f"Expected outputs: {list(contract.expected_output_paths.keys())}")
```

#### get_script_path

get_script_path(_default_path=None_)

Get script path, preferring contract-defined path if available. This method provides a unified way to access script paths with fallback logic.

**Parameters:**
- **default_path** (_Optional[str]_) – Default script path to use if not found in contract.

**Returns:**
- **str** – Script path from contract, configuration, or default.

```python
# Get script path with fallback
script_path = config.get_script_path("default_script.py")
print(f"Script path: {script_path}")
```

#### model_dump

model_dump(_**kwargs_)

Override model_dump to include derived properties. This method ensures that derived fields are included in serialization output.

**Parameters:**
- ****kwargs** – Additional arguments passed to the parent model_dump method.

**Returns:**
- **Dict[str, Any]** – Dictionary representation including derived properties.

```python
# Serialize configuration including derived fields
config_dict = config.model_dump()
print("Serialized config keys:", list(config_dict.keys()))
# Includes both input fields and derived properties
```

### Class Methods

#### from_base_config

_classmethod_ from_base_config(_base_config_, _**kwargs_)

Create a new configuration instance from a base configuration. This method enables configuration inheritance and specialization patterns.

**Parameters:**
- **base_config** (_BasePipelineConfig_) – Parent BasePipelineConfig instance.
- ****kwargs** – Additional arguments specific to the derived class.

**Returns:**
- **BasePipelineConfig** – New instance of the derived class initialized with parent fields and additional kwargs.

```python
# Create derived configuration from base
class ProcessingConfig(BasePipelineConfig):
    processing_instance_type: str = "ml.m5.large"
    processing_instance_count: int = 1

# Inherit from base configuration
processing_config = ProcessingConfig.from_base_config(
    base_config,
    processing_instance_type="ml.m5.xlarge",
    processing_instance_count=2
)

print(f"Inherited pipeline name: {processing_config.pipeline_name}")
print(f"Processing instance type: {processing_config.processing_instance_type}")
```

#### get_step_name

_classmethod_ get_step_name(_config_class_name_)

Get the step name for a configuration class. This method provides mapping from configuration class names to step names using the step registry.

**Parameters:**
- **config_class_name** (_str_) – Name of the configuration class.

**Returns:**
- **str** – Corresponding step name from the registry.

```python
# Get step name from config class
step_name = BasePipelineConfig.get_step_name("ProcessingConfig")
print(f"Step name: {step_name}")  # Output: "Processing"
```

#### get_config_class_name

_classmethod_ get_config_class_name(_step_name_)

Get the configuration class name from a step name. This method provides reverse mapping from step names to configuration class names.

**Parameters:**
- **step_name** (_str_) – Name of the step.

**Returns:**
- **str** – Corresponding configuration class name.

```python
# Get config class name from step name
config_class = BasePipelineConfig.get_config_class_name("Processing")
print(f"Config class: {config_class}")  # Output: "ProcessingConfig"
```

## Usage Examples

### Basic Configuration Creation
```python
from cursus.core.base.config_base import BasePipelineConfig

# Create configuration with essential inputs
config = BasePipelineConfig(
    author="ml-engineer",
    bucket="company-ml-artifacts",
    role="arn:aws:iam::123456789012:role/SageMakerExecutionRole",
    region="NA",
    service_name="customer-churn",
    pipeline_version="2.1.0",
    model_class="pytorch",
    framework_version="1.12.0",
    py_version="py39"
)

# Access derived properties
print(f"Pipeline: {config.pipeline_name}")
print(f"Description: {config.pipeline_description}")
print(f"S3 Location: {config.pipeline_s3_loc}")
print(f"AWS Region: {config.aws_region}")
```

### Configuration Analysis and Debugging
```python
# Analyze configuration structure
def analyze_config(config):
    """Analyze configuration field organization."""
    categories = config.categorize_fields()
    
    print(f"Configuration Analysis for {config.__class__.__name__}:")
    print(f"  Essential fields: {len(categories['essential'])}")
    print(f"  System fields: {len(categories['system'])}")
    print(f"  Derived fields: {len(categories['derived'])}")
    
    # Show field details
    for category, fields in categories.items():
        print(f"\n{category.title()} Fields:")
        for field in sorted(fields):
            try:
                value = getattr(config, field)
                print(f"  {field}: {value}")
            except Exception as e:
                print(f"  {field}: <Error: {e}>")

# Analyze the configuration
analyze_config(config)

# Print organized view
config.print_config()
```

### Configuration Inheritance Pattern
```python
# Create specialized configuration class
class TrainingConfig(BasePipelineConfig):
    """Training-specific configuration."""
    
    # Additional training-specific fields
    instance_type: str = Field(default="ml.m5.large", description="Training instance type")
    instance_count: int = Field(default=1, description="Number of training instances")
    max_runtime_seconds: int = Field(default=3600, description="Maximum training runtime")
    
    # Override derived field if needed
    @property
    def training_job_name(self) -> str:
        """Get training job name."""
        return f"{self.pipeline_name}-training-{self.current_date}"

# Create training config from base config
training_config = TrainingConfig.from_base_config(
    config,
    instance_type="ml.m5.xlarge",
    instance_count=2,
    max_runtime_seconds=7200
)

print(f"Training job name: {training_config.training_job_name}")
print(f"Instance configuration: {training_config.instance_count}x {training_config.instance_type}")

# Verify inheritance
assert training_config.pipeline_name == config.pipeline_name
assert training_config.aws_region == config.aws_region
```

### Script Contract Integration
```python
# Configuration with script contract
class ProcessingConfig(BasePipelineConfig):
    """Processing configuration with script contract."""
    
    processing_instance_type: str = "ml.m5.large"
    
    def get_script_contract(self):
        """Override to provide specific contract."""
        from cursus.steps.contracts.processing_contract import PROCESSING_CONTRACT
        return PROCESSING_CONTRACT

# Create processing config
processing_config = ProcessingConfig(
    author="data-engineer",
    bucket="processing-artifacts",
    role="arn:aws:iam::123456789012:role/ProcessingRole",
    region="EU",
    service_name="data-pipeline",
    pipeline_version="1.0.0"
)

# Access script contract
contract = processing_config.script_contract
if contract:
    print(f"Script path: {contract.script_path}")
    print(f"Expected inputs: {list(contract.expected_input_paths.keys())}")
    
# Get script path with fallback
script_path = processing_config.get_script_path("fallback_script.py")
print(f"Final script path: {script_path}")
```

### Configuration Serialization
```python
# Serialize configuration including derived fields
config_dict = config.model_dump()

print("Serialized configuration:")
for key, value in sorted(config_dict.items()):
    print(f"  {key}: {value}")

# Save to JSON
import json
with open("pipeline_config.json", "w") as f:
    json.dump(config_dict, f, indent=2)

# Load and recreate configuration
with open("pipeline_config.json", "r") as f:
    loaded_dict = json.load(f)

# Remove derived fields before recreating (they'll be recalculated)
derived_fields = ["aws_region", "pipeline_name", "pipeline_description", "pipeline_s3_loc"]
for field in derived_fields:
    loaded_dict.pop(field, None)

# Recreate configuration
recreated_config = BasePipelineConfig(**loaded_dict)
assert recreated_config.pipeline_name == config.pipeline_name
```

## Related Documentation

- [Specification Base](specification_base.md) - Step specifications that work with configuration classes
- [Builder Base](builder_base.md) - Step builders that use configuration classes
- [Contract Base](contract_base.md) - Script contracts integrated with configurations
- [Hyperparameters Base](hyperparameters_base.md) - Hyperparameter configurations extending base config
- [Base Enums](enums.md) - Enumerations used in configuration validation

---
tags:
  - entry_point
  - code
  - core
  - compiler
  - pipeline_compiler
keywords:
  - pipeline compiler
  - DAG compiler
  - dynamic template
  - config resolution
  - validation
  - name generation
topics:
  - pipeline compilation
  - DAG to pipeline conversion
  - dynamic template generation
  - configuration resolution
language: python
date of note: 2024-12-07
---

# Pipeline Compiler

This module provides a comprehensive pipeline compilation system for converting PipelineDAG structures directly into executable SageMaker pipelines through intelligent configuration resolution and dynamic template generation.

## Overview

The Pipeline Compiler bridges the gap between abstract pipeline definitions (DAGs) and concrete SageMaker pipeline implementations by:

1. **Intelligent Configuration Resolution**: Automatically matching DAG nodes to configuration instances using multiple strategies
2. **Dynamic Template Generation**: Creating pipeline templates on-the-fly without manual template coding
3. **Comprehensive Validation**: Ensuring DAG-config compatibility before pipeline generation
4. **Pipeline Name Generation**: Rule-based pipeline naming with versioning support
5. **Exception Handling**: Detailed error reporting and recovery mechanisms

## Quick Start

### Simple Usage (Standard Pipeline)

```python
from src.pipeline_dag.base_dag import PipelineDAG
from src.pipeline_api import dag_to_pipeline_template

# Create a DAG
dag = PipelineDAG()
dag.add_node("data_load")
dag.add_node("preprocess") 
dag.add_node("train")
dag.add_edge("data_load", "preprocess")
dag.add_edge("preprocess", "train")

# Convert to pipeline
pipeline = dag_to_pipeline_template(
    dag=dag,
    config_path="configs/my_pipeline.json",
    sagemaker_session=session,
    role="arn:aws:iam::123456789012:role/SageMakerRole"
)

# Deploy and run
pipeline.upsert()
execution = pipeline.start()
```

### MODS Integration

```python
from src.pipeline_api.mods_dag_compiler import compile_mods_dag_to_pipeline, MODSPipelineDAGCompiler

# Using the simple function (automatically extracts MODS metadata from base config)
pipeline = compile_mods_dag_to_pipeline(
    dag=dag,
    config_path="configs/my_pipeline.json", 
    sagemaker_session=session,
    role="arn:aws:iam::123456789012:role/SageMakerRole"
)

# Using the advanced API
mods_compiler = MODSPipelineDAGCompiler(
    config_path="configs/my_pipeline.json",
    sagemaker_session=session,
    role=role
)
pipeline = mods_compiler.compile(dag)
```

### Advanced Usage with Validation

```python
from src.pipeline_api.dag_compiler import PipelineDAGCompiler

# Create converter for more control
converter = PipelineDAGConverter(
    config_path="configs/my_pipeline.json",
    sagemaker_session=session,
    role=role
)

# Validate DAG compatibility first
validation_result = converter.validate_dag_compatibility(dag)
if not validation_result.is_valid:
    print("Validation failed:")
    print(validation_result.detailed_report())
    exit(1)

# Preview resolution before conversion
preview = converter.preview_resolution(dag)
print("Resolution Preview:")
print(preview.display())

# Convert with detailed reporting
pipeline, report = converter.convert_with_report(dag)
print("Conversion Report:")
print(report.detailed_report())
```

## Core Components

### 1. [DAG Compiler](dag_compiler.md)

The main entry point providing pipeline compilation from DAG structures:
- **`PipelineDAGCompiler`**: Advanced API with validation and debugging capabilities
- **DAG to Pipeline Conversion**: Intelligent conversion of DAG nodes to pipeline steps
- **Template Integration**: Seamless integration with the pipeline template system
- **Error Handling**: Comprehensive error reporting and recovery mechanisms

### 2. [Dynamic Template](dynamic_template.md)

A dynamic implementation of `PipelineTemplateBase` that creates templates on-the-fly:
- **Auto-Detection**: Automatically detects required configuration classes
- **Abstract Method Implementation**: Implements template abstract methods using intelligent resolution
- **Validation Integration**: Provides validation and preview capabilities
- **Runtime Generation**: Creates pipeline templates without manual coding

### 3. [Config Resolver](config_resolver.md)

Intelligent configuration matching engine using multiple resolution strategies:
- **Direct Name Matching**: Exact node name to configuration identifier matching
- **Job Type Matching**: Resolution based on `job_type` attributes in configurations
- **Semantic Matching**: Uses synonyms and similarity algorithms for flexible matching
- **Pattern Matching**: Regex patterns for step type identification
- **Confidence Scoring**: Assigns confidence scores to resolution matches

### 4. [Validation Engine](validation.md)

Comprehensive validation system for pipeline compilation:
- **Configuration Validation**: Detects missing and invalid configurations
- **DAG Structure Validation**: Validates DAG topology and node relationships
- **Builder Resolution Validation**: Ensures all step builders can be resolved
- **Dependency Validation**: Validates step dependencies and connections
- **Preview Capabilities**: Provides detailed preview of resolution results

### 5. [Name Generator](name_generator.md)

Rule-based pipeline name generation system:
- **Versioning Support**: Automatic version-based naming
- **Naming Rules**: Consistent naming conventions across pipelines
- **Conflict Resolution**: Handles naming conflicts and duplicates
- **Metadata Integration**: Incorporates pipeline metadata into names

### 6. [Exception Handling](exceptions.md)

Comprehensive exception system for detailed error reporting:
- **Configuration Errors**: Specific exceptions for configuration issues
- **Validation Errors**: Detailed validation failure reporting
- **Resolution Errors**: Configuration resolution failure handling
- **Recovery Mechanisms**: Error recovery and fallback strategies

## Resolution Strategies

The config resolver uses multiple strategies in order of preference:

### 1. Direct Name Matching (Confidence: 1.0)
```python
# Exact key match in configuration file
dag.add_node("data_load_step")  # → matches config key "data_load_step"

# Metadata mapping from config_types (if available)
# metadata.config_types: {"training_job": "XGBoostTrainingConfig"}
dag.add_node("training_job")  # → XGBoostTrainingConfig

# Case-insensitive fallback
dag.add_node("Data_Load_Step")  # → matches "data_load_step"
```

### 2. Enhanced Job Type Matching (Confidence: 0.8-0.9)
```python
# Parsed node name with job type: ConfigType_JobType
dag.add_node("XGBoostTraining_training")  # → XGBoostTrainingConfig(job_type="training")

# Job type keywords with config type matching
dag.add_node("training_model")  # → matches configs with job_type="training"
```

### 3. Traditional Job Type Matching (Confidence: 0.7-1.0)
```python
# Node contains job type keywords, matches config job_type attribute
dag.add_node("model_train")  # → XGBoostTrainingConfig(job_type="training")
dag.add_node("eval_step")    # → XGBoostModelEvalConfig(job_type="evaluation")
```

### 4. Semantic Matching (Confidence: 0.5-0.8)
```python
# Semantic similarity using predefined mappings
dag.add_node("data_preprocessing")  # → TabularPreprocessingConfig
dag.add_node("model_fit")          # → XGBoostTrainingConfig (training synonym)
dag.add_node("model_test")         # → XGBoostModelEvalConfig (evaluation synonym)
```

### 5. Pattern Matching (Confidence: 0.6-0.9)
```python
# Regex patterns for step type detection
dag.add_node("data_load_job")      # → matches r'.*data_load.*' → CradleDataLoading
dag.add_node("preprocess_data")    # → matches r'.*preprocess.*' → TabularPreprocessing
dag.add_node("train_xgboost")      # → matches r'.*train.*' → XGBoostTraining
dag.add_node("eval_model")         # → matches r'.*eval.*' → XGBoostModelEval
```

### Job Type Keywords
The resolver recognizes these job type patterns:
- **Training**: `training`, `train`
- **Calibration**: `calibration`, `calib`
- **Evaluation**: `evaluation`, `eval`, `test`
- **Inference**: `inference`, `infer`, `predict`
- **Validation**: `validation`, `valid`

### Step Type Patterns
Pattern matching uses these regex patterns:
- **Data Loading**: `.*data_load.*` → `CradleDataLoading`
- **Preprocessing**: `.*preprocess.*` → `TabularPreprocessing`
- **Training**: `.*train.*` → `XGBoostTraining`, `PyTorchTraining`, `DummyTraining`
- **Evaluation**: `.*eval.*` → `XGBoostModelEval`
- **Model**: `.*model.*` → `XGBoostModel`, `PyTorchModel`
- **Calibration**: `.*calibrat.*` → `ModelCalibration`
- **Packaging**: `.*packag.*` → `MIMSPackaging`
- **Payload**: `.*payload.*` → `MIMSPayload`
- **Registration**: `.*regist.*` → `ModelRegistration`
- **Transform**: `.*transform.*` → `BatchTransform`
- **Currency**: `.*currency.*` → `CurrencyConversion`
- **Risk**: `.*risk.*` → `RiskTableMapping`
- **Hyperparameter**: `.*hyperparam.*` → `HyperparameterPrep`

## Configuration File Requirements

The configuration file uses a structured format created by the `merge_and_save_configs` function. The file contains two main sections:

### File Structure
```json
{
  "metadata": {
    "created_at": "2024-12-07T10:30:00.000000",
    "config_types": {
      "data_load_step": "CradleDataLoadConfig",
      "preprocess_step": "TabularPreprocessingConfig", 
      "train_step": "XGBoostTrainingConfig"
    },
    "field_sources": {
      "pipeline_name": ["data_load_step", "preprocess_step", "train_step"],
      "pipeline_version": ["data_load_step", "preprocess_step", "train_step"]
    }
  },
  "configuration": {
    "shared": {
      "pipeline_name": "my_pipeline",
      "pipeline_version": "1.0",
      "pipeline_s3_loc": "s3://bucket/pipelines/"
    },
    "specific": {
      "data_load_step": {
        "job_type": "data_loading",
        "input_path": "s3://bucket/data/",
        "output_path": "s3://bucket/processed/"
      },
      "preprocess_step": {
        "job_type": "preprocessing",
        "features": ["col1", "col2", "col3"]
      },
      "train_step": {
        "job_type": "training",
        "hyperparameters": {
          "max_depth": 6,
          "eta": 0.3
        }
      }
    }
  }
}
```

### Configuration Sections

#### 1. Metadata Section
- **`created_at`**: Timestamp when the configuration was created
- **`config_types`**: Maps step names to their configuration class names (used for resolution)
- **`field_sources`**: Tracks which fields come from which configuration instances

#### 2. Configuration Section
- **`shared`**: Fields that have identical values across all configuration instances
- **`specific`**: Fields that are unique to specific steps or have different values across steps

### Field Categorization Rules

The system automatically categorizes fields based on these rules:

1. **Special Fields** → Always placed in `specific`
   - Fields in `SPECIAL_FIELDS_TO_KEEP_SPECIFIC` list
   - Pydantic models and complex nested structures
   - Runtime values and input/output fields

2. **Single-Instance Fields** → Placed in `specific`
   - Fields that exist in only one configuration instance

3. **Different Values** → Placed in `specific`
   - Fields with the same name but different values across configurations

4. **Identical Values** → Placed in `shared`
   - Fields with identical values across all configurations (and not special)

### Creating Configuration Files

Use the `merge_and_save_configs` function to create properly formatted configuration files:

```python
from src.cursus.steps.configs import merge_and_save_configs
from src.cursus.steps.configs.config_cradle_data_loading_step import CradleDataLoadConfig
from src.cursus.steps.configs.config_xgboost_training_step import XGBoostTrainingConfig

# Create configuration instances
configs = [
    CradleDataLoadConfig(
        pipeline_name="my_pipeline",
        job_type="data_loading",
        input_path="s3://bucket/data/"
    ),
    XGBoostTrainingConfig(
        pipeline_name="my_pipeline", 
        job_type="training",
        hyperparameters={"max_depth": 6}
    )
]

# Save to file
merge_and_save_configs(configs, "config.json")
```

### Loading Configuration Files

The system uses `load_configs` to reconstruct configuration instances:

```python
from src.cursus.steps.configs import load_configs, build_complete_config_classes

# Build config class registry
config_classes = build_complete_config_classes()

# Load configurations
loaded_configs = load_configs("config.json", config_classes)
# Returns: {"data_load_step": CradleDataLoadConfig(...), "train_step": XGBoostTrainingConfig(...)}
```

### Resolution Enhancement

The `metadata.config_types` mapping enables precise resolution:
- Direct step name to configuration class mapping
- Eliminates ambiguity in configuration resolution
- Supports the enhanced job type matching strategy

## Error Handling

The API provides detailed error information through custom exception classes:

### Exception Classes

#### ConfigurationError
Raised when configuration-related errors occur:
```python
from src.cursus.core.compiler.exceptions import ConfigurationError

try:
    pipeline, report = dag_compiler.compile_with_report(dag)
except ConfigurationError as e:
    print(f"Configuration issue: {e}")
    print(f"Missing configs: {e.missing_configs}")
    print(f"Available configs: {e.available_configs}")
```

#### RegistryError
Raised when step builder registry errors occur:
```python
from src.cursus.registry.exceptions import RegistryError

try:
    pipeline, report = dag_compiler.compile_with_report(dag)
except RegistryError as e:
    print(f"Registry issue: {e}")
    print(f"Unresolvable types: {e.unresolvable_types}")
    print(f"Available builders: {e.available_builders}")
```

#### ValidationError
Raised when DAG-config validation fails:
```python
from src.cursus.core.compiler.exceptions import ValidationError

try:
    validation = dag_compiler.validate_dag_compatibility(dag)
    if not validation.is_valid:
        # Handle validation failure
        pass
except ValidationError as e:
    print(f"Validation issue: {e}")
    for category, errors in e.validation_errors.items():
        print(f"  {category}: {errors}")
```

#### AmbiguityError
Raised when multiple configurations could match a DAG node:
```python
from src.cursus.core.compiler.exceptions import AmbiguityError

try:
    config_map = resolver.resolve_config_map(dag.nodes, available_configs)
except AmbiguityError as e:
    print(f"Ambiguity issue: {e}")
    print(f"Node: {e.node_name}")
    for candidate in e.candidates:
        config_type = candidate.get('config_type', 'Unknown')
        confidence = candidate.get('confidence', 0.0)
        job_type = candidate.get('job_type', 'N/A')
        print(f"  - {config_type} (job_type='{job_type}', confidence={confidence:.2f})")
```

#### ResolutionError
Raised when DAG node resolution fails:
```python
from src.cursus.core.compiler.exceptions import ResolutionError

try:
    config_map = resolver.resolve_config_map(dag.nodes, available_configs)
except ResolutionError as e:
    print(f"Resolution issue: {e}")
    print(f"Failed nodes: {e.failed_nodes}")
    for suggestion in e.suggestions:
        print(f"  - {suggestion}")
```

### Complete Error Handling Example

Based on actual pipeline catalog patterns:

```python
import logging
from pathlib import Path
from src.cursus.core.compiler.dag_compiler import PipelineDAGCompiler
from src.cursus.core.compiler.exceptions import (
    ConfigurationError, ValidationError, AmbiguityError, ResolutionError
)
from src.cursus.registry.exceptions import RegistryError

logger = logging.getLogger(__name__)

def create_pipeline_with_error_handling(config_path: str, dag, session, role):
    """
    Create pipeline with comprehensive error handling.
    """
    try:
        # Create compiler
        dag_compiler = PipelineDAGCompiler(
            config_path=config_path,
            sagemaker_session=session,
            role=role
        )
        
        # Validate DAG compatibility first
        validation = dag_compiler.validate_dag_compatibility(dag)
        if not validation.is_valid:
            logger.warning(f"DAG validation failed: {validation.summary()}")
            if validation.missing_configs:
                logger.warning(f"Missing configs: {validation.missing_configs}")
            if validation.unresolvable_builders:
                logger.warning(f"Unresolvable builders: {validation.unresolvable_builders}")
            if validation.config_errors:
                logger.warning(f"Config errors: {validation.config_errors}")
            if validation.dependency_issues:
                logger.warning(f"Dependency issues: {validation.dependency_issues}")
        
        # Preview resolution for debugging
        try:
            preview = dag_compiler.preview_resolution(dag)
            logger.info("DAG node resolution preview:")
            for node, config_type in preview.node_config_map.items():
                confidence = preview.resolution_confidence.get(node, 0.0)
                logger.info(f"  {node} → {config_type} (confidence: {confidence:.2f})")
        except Exception as e:
            logger.warning(f"Preview resolution failed: {e}")
        
        # Compile the DAG
        pipeline, report = dag_compiler.compile_with_report(dag=dag)
        
        logger.info(f"Pipeline '{pipeline.name}' created successfully")
        logger.info(f"Average resolution confidence: {report.avg_confidence:.2f}")
        
        return pipeline, report, dag_compiler
        
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        if e.missing_configs:
            logger.error(f"Missing configurations: {e.missing_configs}")
        if e.available_configs:
            logger.error(f"Available configurations: {e.available_configs}")
        raise
        
    except RegistryError as e:
        logger.error(f"Registry error: {e}")
        if e.unresolvable_types:
            logger.error(f"Unresolvable step types: {e.unresolvable_types}")
        if e.available_builders:
            logger.error(f"Available builders: {e.available_builders}")
        raise
        
    except AmbiguityError as e:
        logger.error(f"Ambiguity error: {e}")
        logger.error(f"Multiple configurations match node '{e.node_name}'")
        for candidate in e.candidates:
            if isinstance(candidate, dict):
                config_type = candidate.get('config_type', 'Unknown')
                confidence = candidate.get('confidence', 0.0)
                job_type = candidate.get('job_type', 'N/A')
                logger.error(f"  - {config_type} (job_type='{job_type}', confidence={confidence:.2f})")
        raise
        
    except ResolutionError as e:
        logger.error(f"Resolution error: {e}")
        if e.failed_nodes:
            logger.error(f"Failed to resolve nodes: {e.failed_nodes}")
        for suggestion in e.suggestions:
            logger.error(f"  Suggestion: {suggestion}")
        raise
        
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        for category, errors in e.validation_errors.items():
            logger.error(f"  {category}:")
            for error in errors:
                logger.error(f"    - {error}")
        raise
        
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        # Check for default config file
        config_dir = Path.cwd().parent / "pipeline_config"
        default_config = config_dir / "config.json"
        if default_config.exists():
            logger.info(f"Consider using default config: {default_config}")
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error creating pipeline: {e}")
        raise
```

### Error Recovery Patterns

#### Configuration File Issues
```python
# Handle missing configuration files
try:
    pipeline = create_pipeline(config_path)
except FileNotFoundError:
    # Try fallback config
    fallback_config = "configs/default.json"
    if os.path.exists(fallback_config):
        logger.info(f"Using fallback config: {fallback_config}")
        pipeline = create_pipeline(fallback_config)
    else:
        raise FileNotFoundError("No configuration file found")
```

#### Resolution Failures
```python
# Handle resolution failures with suggestions
try:
    config_map = resolver.resolve_config_map(dag.nodes, configs)
except ResolutionError as e:
    logger.error(f"Resolution failed for nodes: {e.failed_nodes}")
    # Apply suggestions or use manual mapping
    for suggestion in e.suggestions:
        logger.info(f"Suggestion: {suggestion}")
```

#### Registry Issues
```python
# Handle registry loading issues
try:
    pipeline = create_pipeline(config_path)
except RegistryError as e:
    logger.error(f"Registry error: {e.unresolvable_types}")
    # Check if custom builders need registration
    logger.info("Consider registering custom step builders")
```

## Supported Step Types

The API supports all pipeline step types based on the actual configuration classes in `src/cursus/steps/configs/`:

### Core Step Types
- **Data Loading**: `CradleDataLoadingStep` (from `config_cradle_data_loading_step.py`)
- **Preprocessing**: `TabularPreprocessingStep` (from `config_tabular_preprocessing_step.py`)
- **Training Steps**:
  - `XGBoostTrainingStep` (from `config_xgboost_training_step.py`)
  - `PyTorchTrainingStep` (from `config_pytorch_training_step.py`)
  - `DummyTrainingStep` (from `config_dummy_training_step.py`)

### Model Operations
- **Model Steps**:
  - `XGBoostModelStep` (from `config_xgboost_model_step.py`)
  - `PyTorchModelStep` (from `config_pytorch_model_step.py`)
- **Evaluation**: `XGBoostModelEvalStep` (from `config_xgboost_model_eval_step.py`)
- **Calibration**: `ModelCalibrationStep` (from `config_model_calibration_step.py`)

### Deployment & Operations
- **Packaging**: `PackageStep` (from `config_package_step.py`)
- **Payload**: `PayloadStep` (from `config_payload_step.py`)
- **Registration**: `RegistrationStep` (from `config_registration_step.py`)
- **Transform**: `BatchTransformStep` (from `config_batch_transform_step.py`)

### Utility Steps
- **Currency**: `CurrencyConversionStep` (from `config_currency_conversion_step.py`)
- **Risk Mapping**: `RiskTableMappingStep` (from `config_risk_table_mapping_step.py`)

### Base Classes
- **Processing Base**: `ProcessingStepBase` (from `config_processing_step_base.py`)

### Configuration File Naming Pattern
All step configuration files follow the pattern: `config_{step_name}_step.py`

### Step Type Resolution
The config resolver maps these configuration classes to step types by:
1. Removing the `Config` suffix from class names
2. Removing the `Step` suffix if present
3. Applying special case mappings (e.g., `CradleDataLoad` → `CradleDataLoading`)

## Best Practices

### 1. Naming Conventions
Use the registered step names from the step registry for optimal resolution:

#### Recommended: Use Step Name + Job Type Pattern
When job_type is available, use the pattern `step_name + _ + job_type`:
```python
# Use step_name + _ + job_type for highest confidence matching
dag.add_node("CradleDataLoading_data_loading")    # → CradleDataLoadingConfig(job_type="data_loading")
dag.add_node("XGBoostTraining_training")          # → XGBoostTrainingConfig(job_type="training")
dag.add_node("XGBoostModelEval_evaluation")       # → XGBoostModelEvalConfig(job_type="evaluation")
dag.add_node("ModelCalibration_calibration")      # → ModelCalibrationConfig(job_type="calibration")
dag.add_node("Package_packaging")                 # → PackageConfig(job_type="packaging")
dag.add_node("Payload_payload")                   # → PayloadConfig(job_type="payload")
dag.add_node("Registration_registration")         # → RegistrationConfig(job_type="registration")
```

#### Alternative: Use Registered Step Names Only
For steps without specific job types or when job type is not critical:
```python
# Use exact registered step names for high confidence matching
dag.add_node("CradleDataLoading")     # → CradleDataLoadingConfig (confidence: 1.0)
dag.add_node("XGBoostTraining")       # → XGBoostTrainingConfig (confidence: 1.0)
dag.add_node("XGBoostModelEval")      # → XGBoostModelEvalConfig (confidence: 1.0)
dag.add_node("ModelCalibration")      # → ModelCalibrationConfig (confidence: 1.0)
dag.add_node("Package")               # → PackageConfig (confidence: 1.0)
dag.add_node("Payload")               # → PayloadConfig (confidence: 1.0)
dag.add_node("Registration")          # → RegistrationConfig (confidence: 1.0)
```

#### Available Registered Step Names
Based on the step registry (`src/cursus/registry/step_names_original.py`):

**Data Loading Steps:**
- `CradleDataLoading` → CradleDataLoadingConfig

**Processing Steps:**
- `TabularPreprocessing` → TabularPreprocessingConfig
- `RiskTableMapping` → RiskTableMappingConfig
- `CurrencyConversion` → CurrencyConversionConfig

**Training Steps:**
- `XGBoostTraining` → XGBoostTrainingConfig
- `PyTorchTraining` → PyTorchTrainingConfig
- `DummyTraining` → DummyTrainingConfig

**Evaluation Steps:**
- `XGBoostModelEval` → XGBoostModelEvalConfig

**Model Steps:**
- `XGBoostModel` → XGBoostModelConfig
- `PyTorchModel` → PyTorchModelConfig

**Model Processing Steps:**
- `ModelCalibration` → ModelCalibrationConfig

**Deployment Steps:**
- `Package` → PackageConfig
- `Payload` → PayloadConfig
- `Registration` → RegistrationConfig

**Transform Steps:**
- `BatchTransform` → BatchTransformStepConfig

**Utility Steps:**
- `HyperparameterPrep` → HyperparameterPrepConfig

#### Alternative: Descriptive Names with Hints
If you prefer descriptive names, include step type hints:
```python
dag.add_node("data_load_cradle")      # → CradleDataLoadingConfig (via pattern matching)
dag.add_node("xgb_training_step")     # → XGBoostTrainingConfig (via semantic matching)
dag.add_node("model_evaluation")      # → XGBoostModelEvalConfig (via pattern matching)
```

### 2. Job Type Attributes
Use `job_type` in configurations to improve matching accuracy:
```json
{
  "training_step": {
    "class": "XGBoostTrainingConfig",
    "job_type": "training"
  },
  "calibration_step": {
    "class": "ModelCalibrationConfig", 
    "job_type": "calibration"
  },
  "evaluation_step": {
    "class": "XGBoostModelEvalConfig",
    "job_type": "evaluation"
  }
}
```

### 3. Metadata Configuration Types
Leverage the `metadata.config_types` mapping for explicit resolution:
```json
{
  "metadata": {
    "config_types": {
      "my_training_job": "XGBoostTrainingConfig",
      "my_eval_job": "XGBoostModelEvalConfig",
      "my_package_job": "PackageConfig"
    }
  }
}
```

Then use these names in your DAG:
```python
dag.add_node("my_training_job")   # → XGBoostTrainingConfig (confidence: 1.0)
dag.add_node("my_eval_job")       # → XGBoostModelEvalConfig (confidence: 1.0)
dag.add_node("my_package_job")    # → PackageConfig (confidence: 1.0)
```

### 4. Validation First
Always validate before conversion in production:
```python
validation_result = dag_compiler.validate_dag_compatibility(dag)
if not validation_result.is_valid:
    logger.warning(f"DAG validation failed: {validation_result.summary()}")
    if validation_result.missing_configs:
        logger.warning(f"Missing configs: {validation_result.missing_configs}")
    if validation_result.unresolvable_builders:
        logger.warning(f"Unresolvable builders: {validation_result.unresolvable_builders}")
```

### 5. Preview Resolution
Use preview to understand how nodes will be resolved:
```python
preview = dag_compiler.preview_resolution(dag)
logger.info("DAG node resolution preview:")
for node, config_type in preview.node_config_map.items():
    confidence = preview.resolution_confidence.get(node, 0.0)
    logger.info(f"  {node} → {config_type} (confidence: {confidence:.2f})")
```

### 6. Configuration File Organization
Structure your configuration files using the proper format:
```python
from src.cursus.steps.configs import merge_and_save_configs
from src.cursus.steps.configs.config_xgboost_training_step import XGBoostTrainingConfig
from src.cursus.steps.configs.config_package_step import PackageConfig

# Create configs with consistent naming
configs = [
    XGBoostTrainingConfig(
        pipeline_name="my_pipeline",
        job_type="training"
    ),
    PackageConfig(
        pipeline_name="my_pipeline",
        job_type="packaging"
    )
]

# Save with proper structure
merge_and_save_configs(configs, "config.json")
```

## Extending the API

### Custom Config Resolver
```python
from src.pipeline_api.config_resolver import StepConfigResolver

class CustomResolver(StepConfigResolver):
    def _semantic_matching(self, node_name, configs):
        # Custom semantic matching logic
        return super()._semantic_matching(node_name, configs)

converter = PipelineDAGConverter(
    config_path="config.json",
    config_resolver=CustomResolver()
)
```

### Custom Step Builders
```python
from src.pipeline_registry.builder_registry import register_global_builder

register_global_builder("CustomStep", CustomStepBuilder)
```

## Troubleshooting

### Common Issues

1. **"No configuration found for node"**
   - Check node naming matches config identifiers
   - Verify config file contains required configurations
   - Use preview to see resolution candidates

2. **"No step builder found for config type"**
   - Ensure config class follows naming conventions
   - Check if custom builders need registration
   - Verify config class extends `BasePipelineConfig`

3. **"Multiple configurations match with similar confidence"**
   - Use more specific node names
   - Add `job_type` attributes to configs
   - Adjust confidence threshold in resolver

### Debug Mode
Enable detailed logging:
```python
import logging
logging.getLogger('src.pipeline_api').setLevel(logging.DEBUG)
```

## API Reference

See individual module documentation for detailed API reference:

### Core Modules
- [DAG Compiler](dag_compiler.md) - Main pipeline compilation functions and PipelineDAGCompiler class
- [Dynamic Template](dynamic_template.md) - Dynamic template implementation for runtime pipeline generation
- [Config Resolver](config_resolver.md) - Configuration resolution strategies and matching algorithms
- [Validation](validation.md) - Validation engine and preview capabilities
- [Name Generator](name_generator.md) - Rule-based pipeline name generation system
- [Exceptions](exceptions.md) - Custom exception classes for detailed error reporting

### Related Documentation
- [Pipeline Assembler](../assembler/README.md) - Pipeline assembly system that works with compiled templates
- [Pipeline Template Base](../assembler/pipeline_template_base.md) - Base class extended by dynamic templates
- [Configuration Fields](../config_fields/README.md) - Configuration field management system
- [Pipeline Dependencies](../deps/README.md) - Dependency resolution system integration

### Example Usage
- [Example Usage Script](example_usage.py) - Practical examples of compiler usage patterns

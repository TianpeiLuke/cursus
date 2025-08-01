---
tags:
  - entry_point
  - code
  - pipeline_api
  - documentation
  - overview
keywords:
  - pipeline API
  - DAG
  - template converter
  - MODS integration
  - documentation
topics:
  - pipeline API
  - usage examples
  - architecture
language: python
date of note: 2025-07-31
---

# Pipeline API - DAG to Template Converter and MODS Integration

This module provides a high-level API for converting PipelineDAG structures directly into executable SageMaker pipelines without requiring custom template classes.

## Overview

The Pipeline API bridges the gap between abstract pipeline definitions (DAGs) and concrete SageMaker pipeline implementations by:

1. **Intelligent Resolution**: Automatically matching DAG nodes to configuration instances using multiple strategies
2. **Dynamic Templates**: Creating pipeline templates on-the-fly without manual template coding
3. **Comprehensive Validation**: Ensuring DAG-config compatibility before pipeline generation
4. **Detailed Reporting**: Providing insights into the conversion process

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

### 1. DAG Compiler (`dag_compiler.py`)

The main entry point providing two interfaces:

- **`compile_dag_to_pipeline()`**: Simple one-call conversion
- **`PipelineDAGCompiler`**: Advanced API with validation and debugging

### 2. MODS DAG Compiler (`mods_dag_compiler.py`)

Extended compiler that integrates with the MODS system:
- Automatically extracts MODS metadata (author, version, description) from base config
- Applies the `MODSTemplate` decorator properly to avoid metaclass conflicts
- Provides both simple and advanced APIs consistent with standard compiler

### 3. Dynamic Template (`dynamic_template.py`)

A dynamic implementation of `PipelineTemplateBase` that:
- Auto-detects required configuration classes
- Implements abstract methods using intelligent resolution
- Provides validation and preview capabilities

### 4. Config Resolver (`config_resolver.py`)

Intelligent matching engine using multiple strategies:
- **Direct Name Matching**: Exact node name to config identifier
- **Job Type Matching**: Based on `job_type` attributes
- **Semantic Matching**: Using synonyms and similarity
- **Pattern Matching**: Regex patterns for step types

### 5. Builder Registry (`builder_registry.py`)

Centralized registry mapping configuration types to step builders:
- Pre-registered builders for all standard step types
- Support for custom builder registration
- Validation and statistics

### 6. Validation Engine (`validation.py`)

Comprehensive validation including:
- Missing configurations detection
- Unresolvable step builders
- Configuration validation errors
- Dependency resolution issues

## Resolution Strategies

The config resolver uses multiple strategies in order of preference:

### 1. Direct Name Matching (Confidence: 1.0)
```python
# DAG node "data_load_step" matches config identifier "data_load_step"
dag.add_node("data_load_step")  # → CradleDataLoadConfig("data_load_step")
```

### 2. Job Type Matching (Confidence: 0.7-1.0)
```python
# Node "training_job" matches config with job_type="training"
dag.add_node("training_job")  # → XGBoostTrainingConfig(job_type="training")
```

### 3. Semantic Matching (Confidence: 0.5-0.8)
```python
# Node "data_preprocessing" matches TabularPreprocessingConfig
dag.add_node("data_preprocessing")  # → TabularPreprocessingConfig
```

### 4. Pattern Matching (Confidence: 0.6-0.9)
```python
# Node "model_train_xgb" matches XGBoost training pattern
dag.add_node("model_train_xgb")  # → XGBoostTrainingConfig
```

## Configuration File Requirements

Your configuration file should contain instances of pipeline step configurations:

```json
{
  "data_load": {
    "class": "CradleDataLoadConfig",
    "job_type": "data_loading",
    "input_path": "s3://bucket/data/",
    "output_path": "s3://bucket/processed/"
  },
  "preprocess": {
    "class": "TabularPreprocessingConfig", 
    "job_type": "preprocessing",
    "features": ["col1", "col2", "col3"]
  },
  "train": {
    "class": "XGBoostTrainingConfig",
    "job_type": "training",
    "hyperparameters": {
      "max_depth": 6,
      "eta": 0.3
    }
  }
}
```

## Error Handling

The API provides detailed error information:

```python
from src.pipeline_api.exceptions import ConfigurationError, ValidationError
from src.pipeline_registry.exceptions import RegistryError

try:
    pipeline = dag_to_pipeline_template(dag, config_path)
except ConfigurationError as e:
    print(f"Configuration issue: {e}")
    print(f"Missing configs: {e.missing_configs}")
    print(f"Available configs: {e.available_configs}")
    
except RegistryError as e:
    print(f"Registry issue: {e}")
    print(f"Unresolvable types: {e.unresolvable_types}")
    print(f"Available builders: {e.available_builders}")
    
except ValidationError as e:
    print(f"Validation issue: {e}")
    for category, errors in e.validation_errors.items():
        print(f"  {category}: {errors}")
```

## Supported Step Types

The API supports all standard pipeline step types:

- **Data Loading**: `CradleDataLoading`
- **Preprocessing**: `TabularPreprocessing`
- **Training**: `XGBoostTraining`, `PyTorchTraining`, `DummyTraining`
- **Model Operations**: `XGBoostModel`, `PyTorchModel`
- **Evaluation**: `XGBoostModelEval`
- **Calibration**: `ModelCalibration`
- **Deployment**: `MIMSPackaging`, `MIMSPayload`, `ModelRegistration`
- **Transform**: `BatchTransform`
- **Utilities**: `CurrencyConversion`, `RiskTableMapping`, `HyperparameterPrep`

## Best Practices

### 1. Naming Conventions
Use descriptive node names that hint at the step type:
```python
dag.add_node("data_load_cradle")      # → CradleDataLoadingConfig
dag.add_node("xgb_training")          # → XGBoostTrainingConfig  
dag.add_node("model_evaluation")      # → XGBoostModelEvalConfig
```

### 2. Job Type Attributes
Use `job_type` in configurations to improve matching:
```json
{
  "train_model": {
    "class": "XGBoostTrainingConfig",
    "job_type": "training"
  }
}
```

### 3. Validation First
Always validate before conversion in production:
```python
validation_result = converter.validate_dag_compatibility(dag)
if not validation_result.is_valid:
    # Handle validation errors
    pass
```

### 4. Preview Resolution
Use preview to understand how nodes will be resolved:
```python
preview = converter.preview_resolution(dag)
print(preview.display())
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
- `dag_compiler.py` - Main conversion functions
- `mods_dag_compiler.py` - MODS-integrated compiler functions
- `dynamic_template.py` - Dynamic template implementation  
- `config_resolver.py` - Configuration resolution strategies
- `builder_registry.py` - Step builder registry
- `validation.py` - Validation and preview classes
- `exceptions.py` - Custom exception classes

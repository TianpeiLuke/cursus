# Pipeline Examples

## Overview

This document provides an overview of the example pipelines built using the `PipelineTemplateBase` and `PipelineAssembler` classes. These examples demonstrate how to use the specification-driven dependency resolution system to create various types of SageMaker pipelines with automatic step connection.

## Available Examples

### 1. XGBoost End-to-End Pipeline

**Source**: `src/pipeline_builder/template_pipeline_xgboost_end_to_end.py`

**Description**: Complete end-to-end XGBoost pipeline that performs:
1. Data Loading (from Cradle)
2. Tabular Preprocessing
3. XGBoost Model Training
4. Model Evaluation
5. Model Registration (MIMS)
6. Calibration Data Processing

**Key Features**:
- Extends `PipelineTemplateBase` for consistent structure
- Uses specification-driven dependency resolution
- Demonstrates execution document support
- Shows handling of different data types (training and calibration)

### 2. XGBoost Simple Pipeline

**Source**: `src/pipeline_builder/template_pipeline_xgboost_simple.py`

**Description**: Simplified XGBoost pipeline for quick experimentation:
1. Data Loading (from Cradle)
2. Tabular Preprocessing
3. XGBoost Model Training

**Key Features**:
- Minimal pipeline for rapid iteration
- Reduced configuration requirements
- Shows core specification connections
- Suitable for experimentation environments

### 3. PyTorch End-to-End Pipeline

**Source**: `src/pipeline_builder/template_pipeline_pytorch_end_to_end.py`

**Description**: Complete end-to-end PyTorch pipeline that performs:
1. Data Loading (from Cradle)
2. Tabular Preprocessing
3. PyTorch Model Training
4. Model Creation
5. Model Registration (MIMS)
6. Calibration Data Processing

**Key Features**:
- Shows specification-driven connections for PyTorch models
- Demonstrates integration with MIMS packaging and registration
- Handles deep learning model artifacts
- Supports proper hyperparameter configuration

### 4. PyTorch Model Registration Pipeline

**Source**: `src/pipeline_builder/template_pipeline_pytorch_model_registration.py`

**Description**: Pipeline focused on PyTorch model registration:
1. Model Creation (using an existing model artifact)
2. MIMS Packaging
3. MIMS Payload Generation
4. MIMS Registration

**Key Features**:
- Specialized for model registration workflows
- Accepts external model artifacts
- Shows how to validate model paths
- Demonstrates integration with MIMS components

### 5. XGBoost Data Loading and Preprocessing Pipeline

**Source**: `src/pipeline_builder/template_pipeline_xgboost_dataload_preprocess.py`

**Description**: Pipeline focused on data preparation steps:
1. Data Loading (from Cradle)
2. Tabular Preprocessing

**Key Features**:
- Data preparation-only pipeline
- Shows modular pipeline design
- Can be used for data exploration
- Demonstrates preprocessing configuration options

### 6. XGBoost Training and Evaluation Pipeline

**Source**: `src/pipeline_builder/template_pipeline_xgboost_train_evaluate_e2e.py`

**Description**: Pipeline focused on training and evaluation:
1. XGBoost Model Training
2. Model Evaluation
3. Model Registration (MIMS)

**Key Features**:
- Uses pre-processed data from S3
- Focuses on model performance and evaluation
- Shows evaluation metric handling
- Demonstrates model quality gates

### 7. XGBoost Training and Evaluation (No Registration)

**Source**: `src/pipeline_builder/template_pipeline_xgboost_train_evaluate_no_registration.py`

**Description**: Pipeline for training and evaluation without registration:
1. XGBoost Model Training
2. Model Evaluation

**Key Features**:
- Suitable for experimentation and testing
- No model registration steps
- Simplified for iteration speed
- Shows basic evaluation configuration

### 8. Cradle-Only Pipeline

**Source**: `src/pipeline_builder/template_pipeline_cradle_only.py`

**Description**: Pipeline focused on Cradle data operations:
1. Cradle Data Loading
2. Optional basic transformations

**Key Features**:
- Specialized for data exploration workflows
- Shows Cradle integration details
- Demonstrates data source configuration
- Includes execution document generation for Cradle requests

## Implementation Approach

### Abstract Base Class Extension

All examples extend the `PipelineTemplateBase` abstract base class:

```python
class XGBoostEndToEndTemplate(PipelineTemplateBase):
    # Define the configuration classes expected in the config file
    CONFIG_CLASSES = {
        'Base': BasePipelineConfig,
        'DataLoading': CradleDataLoadingConfig,
        'Preprocessing': TabularPreprocessingConfig,
        'Training': XGBoostTrainingConfig,
        'ModelEvaluation': XGBoostModelEvalConfig,
        'ModelRegistration': MIMSRegistrationConfig,
    }
    
    def _validate_configuration(self) -> None:
        # Custom validation logic
        pass
    
    def _create_pipeline_dag(self) -> PipelineDAG:
        # Create and return the pipeline DAG
        pass
    
    def _create_config_map(self) -> Dict[str, BasePipelineConfig]:
        # Map step names to configurations
        pass
    
    def _create_step_builder_map(self) -> Dict[str, Type[StepBuilderBase]]:
        # Map step types to builder classes
        pass
```

### Configuration File Approach

All examples use JSON configuration files to parameterize the pipeline:

```json
{
  "Base": {
    "pipeline_name": "xgboost-training-pipeline",
    "pipeline_s3_loc": "s3://my-bucket/pipelines/xgboost"
  },
  "DataLoading": {
    "cradle_endpoint": "https://cradle-endpoint.example.com",
    "source_type": "EDX",
    "query_params": {
      "start_date": "2023-01-01",
      "end_date": "2023-06-30"
    }
  },
  "Preprocessing": {
    "processing_instance_type": "ml.m5.xlarge",
    "processing_instance_count": 1,
    "categorical_columns": ["category_1", "category_2"],
    "numerical_columns": ["feature_1", "feature_2"]
  }
}
```

### Factory Methods for Creation

All examples provide factory methods for pipeline creation:

```python
@classmethod
def create_pipeline(cls, config_path: str, session=None, role=None) -> Pipeline:
    """Factory method to create and return a pipeline instance."""
    return cls.build_with_context(config_path, sagemaker_session=session, role=role)
```

## Usage Examples

### Basic Template Usage

```python
from src.pipeline_builder.template_pipeline_xgboost_end_to_end import XGBoostEndToEndTemplate

# Create pipeline with context management
pipeline = XGBoostEndToEndTemplate.build_with_context(
    config_path="configs/xgboost_config.json",
    sagemaker_session=sagemaker_session,
    role=execution_role
)

# Execute the pipeline
pipeline.upsert()
execution = pipeline.start()
```

### Advanced Template Usage with Execution Document

```python
from src.pipeline_builder.template_pipeline_xgboost_end_to_end import XGBoostEndToEndTemplate

# Create template instance
template = XGBoostEndToEndTemplate(
    config_path="configs/xgboost_config.json",
    sagemaker_session=sagemaker_session,
    role=execution_role
)

# Generate the pipeline
pipeline = template.generate_pipeline()

# Create an execution document template
execution_doc = {
    "execution": {
        "name": "XGBoost Training Pipeline",
        "steps": []
    }
}

# Fill execution document with pipeline metadata
filled_doc = template.fill_execution_document(execution_doc)

# Execute the pipeline
pipeline.upsert()
execution = pipeline.start()
```

## Benefits of the Specification-Driven Approach

These examples demonstrate the benefits of using the specification-driven dependency resolution system:

1. **Automatic Step Connection**: Dependencies between steps are automatically resolved based on specifications.
2. **Semantic Matching**: Inputs and outputs are matched based on semantic similarity, not just exact names.
3. **Type Compatibility**: The system ensures that connected steps have compatible input/output types.
4. **Configuration-Driven**: Pipelines are configured through JSON files, making them easy to modify.
5. **Declarative Definition**: Pipeline structure is defined declaratively through the DAG.
6. **Modular Design**: Pipelines can be composed of reusable components.
7. **Context Isolation**: Multiple pipelines can run in isolated contexts.
8. **Thread Safety**: Components can be used safely in multi-threaded environments.

## Related Documentation

### Pipeline Building
- [Pipeline Template Base](pipeline_template_base.md): Core abstract class for templates
- [Pipeline Assembler](pipeline_assembler.md): Assembles steps using specifications
- [Template Implementation](template_implementation.md): Implementation details
- [Pipeline Builder Overview](README.md): Introduction to the template-based system

### Pipeline Structure
- [Pipeline DAG Overview](../pipeline_dag/README.md): DAG-based pipeline structure
- [Base Pipeline DAG](../pipeline_dag/base_dag.md): Core DAG implementation
- [Enhanced Pipeline DAG](../pipeline_dag/enhanced_dag.md): Advanced DAG with port-level dependency
- [Edge Types](../pipeline_dag/edge_types.md): Types of edges in the DAG

### Dependency System
- [Dependency Resolver](../pipeline_deps/dependency_resolver.md): Resolves dependencies
- [Registry Manager](../pipeline_deps/registry_manager.md): Manages specifications
- [Base Specifications](../pipeline_deps/base_specifications.md): Core specifications
- [Property Reference](../pipeline_deps/property_reference.md): Runtime property bridge

### Pipeline Components
- [Pipeline Steps](../pipeline_steps/README.md): Available steps for pipelines
- [Script Contracts](../pipeline_script_contracts/README.md): Script validation

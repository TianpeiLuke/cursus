---
tags:
  - code
  - core
  - compiler
  - dynamic_template
  - pipeline_template
keywords:
  - DynamicPipelineTemplate
  - pipeline template
  - dynamic compilation
  - DAG to pipeline
  - configuration resolution
  - execution document
topics:
  - pipeline generation
  - dynamic compilation
  - template system
language: python
date of note: 2025-09-07
---

# Dynamic Pipeline Template

Dynamic implementation of PipelineTemplateBase that can work with any PipelineDAG structure without requiring custom template classes.

## Overview

The `dynamic_template` module provides a dynamic implementation of PipelineTemplateBase that can work with any PipelineDAG structure without requiring custom template classes. The template automatically implements the abstract methods of PipelineTemplateBase by using intelligent resolution mechanisms to map DAG nodes to configurations and step builders.

This module enables automatic pipeline generation from DAG structures by dynamically detecting required configuration classes, resolving DAG nodes to configurations, mapping configurations to step builders, and handling execution document generation with pipeline metadata.

## Classes and Methods

### Classes
- [`DynamicPipelineTemplate`](#dynamicpipelinetemplate) - Dynamic pipeline template that works with any PipelineDAG

## API Reference

### DynamicPipelineTemplate

_class_ cursus.core.compiler.dynamic_template.DynamicPipelineTemplate(_dag_, _config_path_, _config_resolver=None_, _builder_registry=None_, _skip_validation=False_, _**kwargs_)

Dynamic pipeline template that works with any PipelineDAG. This template automatically implements the abstract methods of PipelineTemplateBase by using intelligent resolution mechanisms to map DAG nodes to configurations and step builders.

**Parameters:**
- **dag** (_PipelineDAG_) – PipelineDAG instance defining pipeline structure
- **config_path** (_str_) – Path to configuration file
- **config_resolver** (_Optional[StepConfigResolver]_) – Custom config resolver (optional)
- **builder_registry** (_Optional[StepBuilderRegistry]_) – Custom builder registry (optional)
- **skip_validation** (_bool_) – Skip validation for testing purposes (default: False)
- ****kwargs** – Additional arguments for base template

**Class Attributes:**
- **CONFIG_CLASSES** (_Dict[str, Type[BasePipelineConfig]]_) – Dynamically populated configuration classes

```python
from cursus.core.compiler.dynamic_template import DynamicPipelineTemplate
from cursus.api.dag.base_dag import PipelineDAG

# Create a DAG
dag = PipelineDAG()
dag.add_node("data_load")
dag.add_node("training")
dag.add_edge("data_load", "training")

# Create dynamic template
template = DynamicPipelineTemplate(
    dag=dag,
    config_path="configs/pipeline.json",
    sagemaker_session=session,
    role="arn:aws:iam::123456789012:role/SageMakerRole"
)

# Generate pipeline
pipeline = template.generate_pipeline()
```

#### get_resolution_preview

get_resolution_preview()

Get a preview of how DAG nodes will be resolved.

**Returns:**
- **Dict[str, Any]** – Dictionary with resolution preview information

```python
# Preview resolution before pipeline generation
preview = template.get_resolution_preview()

print(f"Nodes to resolve: {preview['nodes']}")
for node, resolution in preview['resolutions'].items():
    print(f"Node '{node}' -> {resolution['config_type']} "
          f"(confidence: {resolution['confidence']:.2f})")
```

#### validate_before_build

validate_before_build()

Validate the configuration before building the pipeline.

**Returns:**
- **bool** – True if validation passes, False otherwise

```python
# Validate before building
if template.validate_before_build():
    print("Template validation passed")
    pipeline = template.generate_pipeline()
else:
    print("Template validation failed")
```

#### get_step_dependencies

get_step_dependencies()

Get the dependencies for each step based on the DAG.

**Returns:**
- **Dict[str, List[str]]** – Dictionary mapping step names to their dependencies

```python
# Get step dependencies
dependencies = template.get_step_dependencies()
for step, deps in dependencies.items():
    print(f"Step '{step}' depends on: {deps}")
```

#### get_execution_order

get_execution_order()

Get the topological execution order of steps.

**Returns:**
- **List[str]** – List of step names in execution order

```python
# Get execution order
execution_order = template.get_execution_order()
print(f"Execution order: {execution_order}")
```

#### fill_execution_document

fill_execution_document(_execution_document_)

Fill in the execution document with pipeline metadata. This method populates the execution document with Cradle data loading requests (if present in the pipeline) and registration configurations (if present in the pipeline).

**Parameters:**
- **execution_document** (_Dict[str, Any]_) – Execution document to fill

**Returns:**
- **Dict[str, Any]** – Updated execution document

```python
# Execution document template
execution_doc = {
    "PIPELINE_STEP_CONFIGS": {
        "data_load_step": {
            "STEP_TYPE": ["PROCESSING_STEP", "CradleDataLoading"],
            "STEP_CONFIG": {}
        },
        "registration_step": {
            "STEP_TYPE": ["PROCESSING_STEP", "ModelRegistration"],
            "STEP_CONFIG": {}
        }
    }
}

# Generate pipeline first to populate metadata
pipeline = template.generate_pipeline()

# Fill execution document
filled_doc = template.fill_execution_document(execution_doc)
print(f"Filled execution document: {filled_doc}")
```

#### get_builder_registry_stats

get_builder_registry_stats()

Get statistics about the builder registry.

**Returns:**
- **Dict[str, Any]** – Dictionary with registry statistics

```python
# Get registry statistics
stats = template.get_builder_registry_stats()
print(f"Registry stats: {stats}")
```

## Dynamic Resolution Process

The template follows a systematic process for dynamic resolution:

### 1. Configuration Class Detection
Automatically detects required configuration classes from the configuration file by analyzing:
- Config type metadata in the configuration file
- Model type information in configuration entries
- Essential base classes needed for all pipelines

### 2. DAG Node Resolution
Maps DAG node names to configuration instances using multiple strategies:
- Direct name matching with exact match
- Metadata mapping from config_types
- Job type + config type matching with pattern recognition
- Semantic similarity matching
- Pattern-based matching

### 3. Step Builder Mapping
Maps configuration types to their corresponding step builder classes using the StepBuilderRegistry.

### 4. Validation
Performs comprehensive validation including:
- All DAG nodes have matching configurations
- All configurations have corresponding step builders
- Configuration-specific validation passes
- Dependency resolution is possible

### 5. Pipeline Generation
Generates the complete SageMaker Pipeline with proper step dependencies and parameter handling.

## Execution Document Support

The template provides comprehensive execution document support:

### Cradle Data Loading
Automatically populates Cradle data loading configurations in execution documents for steps that use CradleDataLoadConfig.

### Model Registration
Handles model registration step configurations including:
- Image URI retrieval for inference
- Environment variable configuration
- Load testing information
- Model metadata

### Pipeline Metadata
Stores and manages pipeline metadata including:
- Cradle loading requests
- Registration configurations
- Model names and properties
- Step dependencies and execution order

## Related Documentation

- [DAG Compiler](dag_compiler.md) - Uses DynamicPipelineTemplate for pipeline generation
- [Configuration Resolver](config_resolver.md) - Used for resolving DAG nodes to configurations
- [Validation](validation.md) - Validation engine used by the template
- [Compiler Overview](README.md) - System overview and integration

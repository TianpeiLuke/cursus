---
tags:
  - code
  - pipeline_api
  - template
  - dynamic_generation
keywords:
  - template
  - dynamic
  - pipeline
  - DAG
  - adapter
topics:
  - pipeline API
  - dynamic templates
language: python
date of note: 2025-07-31
---

# Dynamic Pipeline Template

## Purpose

The `DynamicPipelineTemplate` is a flexible implementation of `PipelineTemplateBase` that can adapt to any PipelineDAG structure without requiring custom template classes. It implements the abstract methods of `PipelineTemplateBase` using intelligent resolution mechanisms to map DAG nodes to configurations and step builders.

## Core Problem Solved

Traditionally, each pipeline pattern would require a custom template class that hardcodes:
1. The specific DAG structure
2. The mapping between DAG nodes and configurations
3. The mapping between configurations and step builders

The `DynamicPipelineTemplate` removes this limitation by providing a universal template that can work with any DAG structure, dynamically resolving the appropriate configurations and builders.

## Implementation Details

The template implements the abstract methods of `PipelineTemplateBase`:

### 1. `_detect_config_classes()`
Automatically detects required configuration classes from the configuration file based on:
- Config type metadata in the configuration file
- Model type information in configuration entries
- Essential base classes needed for all pipelines

### 2. `_create_pipeline_dag()`
Simply returns the provided DAG.

### 3. `_create_config_map()`
Uses `StepConfigResolver` to intelligently map DAG node names to configuration instances from the loaded config file.

### 4. `_create_step_builder_map()`
Uses `StepBuilderRegistry` to map configuration types to step builder classes.

### 5. `_validate_configuration()`
Performs comprehensive validation including:
- DAG node-to-configuration mapping
- Configuration-to-builder mapping
- Configuration-specific validation
- Dependency resolution

## Usage

The `DynamicPipelineTemplate` is typically used through the `PipelineDAGCompiler`, but can be used directly:

```python
from src.pipeline_api.dynamic_template import DynamicPipelineTemplate

# Create a DAG
dag = create_pipeline_dag()

# Create template
template = DynamicPipelineTemplate(
    dag=dag,
    config_path="configs/my_pipeline.json",
    sagemaker_session=session,
    role=role
)

# Generate pipeline
pipeline = template.generate_pipeline()
```

## Key Features

### 1. Auto-detection of Config Classes
The template automatically detects which configuration classes are needed based on the provided configuration file, without requiring manual specification.

### 2. Intelligent Resolution
It uses intelligent resolution mechanisms to map DAG nodes to configurations and configurations to step builders.

### 3. Comprehensive Validation
It provides comprehensive validation to ensure compatibility between DAG structure, configurations, and step builders.

### 4. Preview and Debugging
It provides methods for previewing how DAG nodes will be resolved to configurations and builders.

### 5. Execution Document Support
It supports filling execution documents with metadata from the pipeline, especially for Cradle data loading and model registration steps.

## Integration with MODS

The `DynamicPipelineTemplate` class serves as the base for MODS integration through the `MODSPipelineDAGCompiler`, which decorates this class with the `MODSTemplate` decorator before instantiation.

This allows dynamic pipeline generation while still maintaining compatibility with the MODS system for execution document filling and other MODS-specific features.

---
tags:
  - code
  - pipeline_api
  - mods_integration
  - dag_compiler
keywords:
  - MODS
  - DAG
  - compiler
  - pipeline
  - template
topics:
  - pipeline API
  - MODS integration
language: python
date of note: 2025-07-31
---

# MODS DAG Compiler

## Purpose

The `MODSPipelineDAGCompiler` extends the standard `PipelineDAGCompiler` to enable MODS integration with dynamically generated pipelines. It solves the metaclass conflict issue that occurs when trying to apply the `MODSTemplate` decorator to an instance of `DynamicPipelineTemplate`.

## Core Problem Solved

When using the standard DAG compiler approach:

```python
dag = create_xgboost_pipeline_dag()
dag_compiler = PipelineDAGCompiler(
    config_path=config_path,
    sagemaker_session=pipeline_session,
    role=role
)
pipeline_template_builder = dag_compiler.create_template(dag)

# This fails with a metaclass conflict
MODSTemplate(author=base_config.author, description=base_config.pipeline_description, version=base_config.pipeline_version)(pipeline_template_builder)
```

The error occurs because:

1. `pipeline_template_builder` is already an *instance* of `DynamicPipelineTemplate`
2. `DynamicPipelineTemplate` inherits from `PipelineTemplateBase` which uses the `ABCMeta` metaclass
3. Trying to apply the `MODSTemplate` decorator to this instance creates a metaclass conflict

## Solution

The `MODSPipelineDAGCompiler` solves this by:

1. Applying the `MODSTemplate` decorator to the `DynamicPipelineTemplate` *class* (not an instance)
2. Then instantiating the decorated class with the provided parameters

```python
# Within create_template method:
# Decorate the DynamicPipelineTemplate class with MODSTemplate
MODSDecoratedTemplate = MODSTemplate(
    author=author,
    version=version,
    description=description
)(DynamicPipelineTemplate)

# Create dynamic template from the decorated class
template = MODSDecoratedTemplate(
    dag=dag,
    config_path=self.config_path,
    config_resolver=self.config_resolver,
    builder_registry=self.builder_registry,
    sagemaker_session=self.sagemaker_session,
    role=self.role,
    **template_kwargs
)
```

## Usage

```python
from src.pipeline_api.mods_dag_compiler import compile_mods_dag_to_pipeline, MODSPipelineDAGCompiler

# Method 1: Using convenience function
dag = create_xgboost_pipeline_dag()
pipeline = compile_mods_dag_to_pipeline(
    dag=dag,
    config_path=config_path,
    sagemaker_session=pipeline_session,
    role=role
)

# Method 2: Using compiler class
dag = create_xgboost_pipeline_dag()
mods_compiler = MODSPipelineDAGCompiler(
    config_path=config_path,
    sagemaker_session=pipeline_session,
    role=role
)
pipeline = mods_compiler.compile(dag)
```

## Key Features

1. **Automatic metadata extraction**: Extracts author, version, and description from the base config
2. **Metaclass conflict resolution**: Properly applies the MODSTemplate decorator to avoid conflicts
3. **API consistency**: Provides the same interface as the standard DAG compiler

## Implementation Details

The compiler automatically:
- Loads the base configuration from the specified config path
- Extracts MODS metadata (author, version, description) from the base config
- Decorates the `DynamicPipelineTemplate` class with `MODSTemplate` using this metadata
- Instantiates the decorated class with the provided parameters

This approach avoids the metaclass conflict by applying the decorator to the class definition rather than an already instantiated object.

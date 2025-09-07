---
tags:
  - code
  - assembler
  - pipeline_template_base
  - template_pattern
  - component_lifecycle
keywords:
  - PipelineTemplateBase
  - template pattern
  - component lifecycle
  - configuration loading
  - dependency injection
  - pipeline generation
  - abstract base class
topics:
  - pipeline templates
  - template pattern
  - component management
language: python
date of note: 2024-12-07
---

# Pipeline Template Base

Abstract base class for all pipeline templates, providing consistent structure, component lifecycle management, and standardized pipeline generation patterns.

## Overview

The `PipelineTemplateBase` class serves as the foundation for all pipeline templates in the system, implementing the template method pattern to ensure consistent pipeline generation workflows. This abstract base class enforces best practices, manages component lifecycles, and provides a standardized approach for creating pipeline templates.

The template follows a structured pipeline generation process: load configurations from file, initialize component dependencies (registry manager and dependency resolver), create the DAG structure along with config and step builder mappings, and use the PipelineAssembler to assemble the final pipeline. This approach reduces code duplication across different pipeline templates while enforcing architectural consistency.

The class supports advanced features including dependency injection for component management, thread-local component instances for multi-threaded environments, scoped dependency resolution contexts with automatic cleanup, execution document integration for pipeline metadata, and comprehensive configuration validation with detailed error reporting.

## Classes and Methods

### Classes
- [`PipelineTemplateBase`](#pipelinetemplatebase) - Abstract base class for pipeline templates with lifecycle management

### Class Methods
- [`create_with_components`](#create_with_components) - Factory method for creating template with managed components
- [`build_with_context`](#build_with_context) - Build pipeline with scoped dependency resolution context
- [`build_in_thread`](#build_in_thread) - Build pipeline using thread-local component instances

## API Reference

### PipelineTemplateBase

_class_ cursus.core.assembler.pipeline_template_base.PipelineTemplateBase(_config_path_, _sagemaker_session=None_, _role=None_, _notebook_root=None_, _registry_manager=None_, _dependency_resolver=None_)

Abstract base class for all pipeline templates. This class provides a consistent structure and common functionality for all pipeline templates, enforcing best practices and ensuring proper component lifecycle management.

**Parameters:**
- **config_path** (_str_) – Path to configuration file containing pipeline step configurations.
- **sagemaker_session** (_Optional[PipelineSession]_) – SageMaker session for pipeline execution. Defaults to None.
- **role** (_Optional[str]_) – IAM role for pipeline execution permissions. Defaults to None.
- **notebook_root** (_Optional[Path]_) – Root directory of notebook environment. Defaults to current working directory.
- **registry_manager** (_Optional[RegistryManager]_) – Optional registry manager for dependency injection and component management.
- **dependency_resolver** (_Optional[UnifiedDependencyResolver]_) – Optional dependency resolver for specification-based matching.

```python
from cursus.core.assembler.pipeline_template_base import PipelineTemplateBase

class MyPipelineTemplate(PipelineTemplateBase):
    CONFIG_CLASSES = {
        'PreprocessingConfig': PreprocessingConfig,
        'TrainingConfig': TrainingConfig
    }
    
    def _validate_configuration(self):
        # Implement validation logic
        pass
    
    def _create_pipeline_dag(self):
        # Implement DAG creation
        pass

# Create template instance
template = MyPipelineTemplate(
    config_path="config.json",
    role="arn:aws:iam::123456789012:role/SageMakerRole"
)
```

#### generate_pipeline

generate_pipeline()

Generate the SageMaker Pipeline. This method coordinates the pipeline generation process by creating the DAG, config map, and step builder map, then using PipelineAssembler to generate the final pipeline.

**Returns:**
- **Pipeline** – SageMaker Pipeline object with all steps properly configured and connected.

```python
pipeline = template.generate_pipeline()
print(f"Generated pipeline: {pipeline.name}")
```

#### fill_execution_document

fill_execution_document(_execution_document_)

Fill in the execution document with pipeline metadata. This method can be overridden by subclasses to fill in execution documents with step-specific metadata from the pipeline.

**Parameters:**
- **execution_document** (_Dict[str, Any]_) – Execution document to fill with pipeline metadata.

**Returns:**
- **Dict[str, Any]** – Updated execution document with pipeline-specific metadata.

```python
execution_doc = {"pipeline_id": "12345"}
updated_doc = template.fill_execution_document(execution_doc)
```

#### create_with_components

_classmethod_ create_with_components(_config_path_, _context_name=None_, _**kwargs_)

Create template with managed dependency components. This factory method creates a template with properly configured dependency resolution components from the factory module.

**Parameters:**
- **config_path** (_str_) – Path to configuration file containing pipeline configurations.
- **context_name** (_Optional[str]_) – Optional context name for registry isolation and component management.
- ****kwargs** – Additional arguments to pass to constructor including session, role, and notebook_root.

**Returns:**
- **PipelineTemplateBase** – Template instance with managed dependency components.

```python
template = MyPipelineTemplate.create_with_components(
    config_path="config.json",
    context_name="experiment-1",
    role="arn:aws:iam::123456789012:role/SageMakerRole"
)
```

#### build_with_context

_classmethod_ build_with_context(_config_path_, _**kwargs_)

Build pipeline with scoped dependency resolution context. This method creates a template with a dependency resolution context that ensures proper cleanup of resources after pipeline generation.

**Parameters:**
- **config_path** (_str_) – Path to configuration file containing pipeline configurations.
- ****kwargs** – Additional arguments to pass to constructor including session, role, and notebook_root.

**Returns:**
- **Pipeline** – Generated pipeline with automatic resource cleanup.

```python
pipeline = MyPipelineTemplate.build_with_context(
    config_path="config.json",
    role="arn:aws:iam::123456789012:role/SageMakerRole"
)
```

#### build_in_thread

_classmethod_ build_in_thread(_config_path_, _**kwargs_)

Build pipeline using thread-local component instances. This method creates a template with thread-local component instances, ensuring thread safety in multi-threaded environments.

**Parameters:**
- **config_path** (_str_) – Path to configuration file containing pipeline configurations.
- ****kwargs** – Additional arguments to pass to constructor including session, role, and notebook_root.

**Returns:**
- **Pipeline** – Generated pipeline with thread-local component isolation.

```python
import threading

def build_pipeline_in_thread():
    pipeline = MyPipelineTemplate.build_in_thread(
        config_path="config.json",
        role="arn:aws:iam::123456789012:role/SageMakerRole"
    )
    return pipeline

thread = threading.Thread(target=build_pipeline_in_thread)
thread.start()
```

### Abstract Methods

The following methods must be implemented by subclasses:

#### _validate_configuration

_validate_configuration()

Perform lightweight validation of configuration structure and essential parameters. This method focuses on validating presence/absence of required configurations, basic parameter validation, and non-dependency related concerns.

```python
def _validate_configuration(self):
    # Find preprocessing configs
    tp_configs = [cfg for name, cfg in self.configs.items() 
                 if isinstance(cfg, PreprocessingConfig)]
    
    if len(tp_configs) < 2:
        raise ValueError("Expected at least two PreprocessingConfig instances")
```

#### _create_pipeline_dag

_create_pipeline_dag()

Create the DAG structure for the pipeline. This method should be implemented by subclasses to define the pipeline's DAG structure.

**Returns:**
- **PipelineDAG** – PipelineDAG instance defining the pipeline structure.

#### _create_config_map

_create_config_map()

Create a mapping from step names to config instances. This method should be implemented by subclasses to map step names to their respective configurations.

**Returns:**
- **Dict[str, BasePipelineConfig]** – Dictionary mapping step names to configurations.

#### _create_step_builder_map

_create_step_builder_map()

Create a mapping from step types to builder classes. This method should be implemented by subclasses to map step types to their builder classes.

**Returns:**
- **Dict[str, Type[StepBuilderBase]]** – Dictionary mapping step types to builder classes.

## Related Documentation

- [Pipeline Assembler](pipeline_assembler.md) - Component-based pipeline assembly system used by templates
- [Config Field Manager](../config_fields/README.md) - Configuration management system for pipeline steps
- [Dependency Resolver](../deps/dependency_resolver.md) - Specification-based dependency resolution system
- [Step Builder Base](../base/step_builder_base.md) - Base class for step builders used in templates
- [Pipeline DAG](../../api/dag/base_dag.md) - DAG structure definition for pipeline templates

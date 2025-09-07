---
tags:
  - code
  - assembler
  - pipeline_assembler
  - dag_compilation
  - step_building
keywords:
  - PipelineAssembler
  - DAG compilation
  - step builders
  - dependency resolution
  - specification matching
  - SageMaker Pipeline
  - pipeline assembly
topics:
  - pipeline assembly
  - DAG compilation
  - dependency resolution
language: python
date of note: 2024-12-07
---

# Pipeline Assembler

Component-based pipeline assembly system that builds SageMaker Pipelines from DAG structures and step builders using specification-based dependency resolution.

## Overview

The `PipelineAssembler` class implements a sophisticated pipeline assembly system that leverages specification-based dependency resolution to intelligently connect pipeline steps. This approach simplifies pipeline construction by automatically matching step inputs to outputs based on their specifications, reducing the need for manual wiring.

The assembler follows a structured process: initialize step builders for all DAG nodes, determine build order through topological sorting, propagate messages between steps using the dependency resolver, instantiate steps in topological order with proper input/output handling, and create the final SageMaker Pipeline. This component-based architecture allows for flexible and modular pipeline definitions where each step manages its own configuration and dependencies.

The system supports advanced features including Cradle data loading integration, specification-based output generation, runtime property reference creation, and comprehensive error handling with detailed logging throughout the assembly process.

## Classes and Methods

### Classes
- [`PipelineAssembler`](#pipelineassembler) - Main pipeline assembly orchestrator with specification-based dependency resolution

### Class Methods
- [`create_with_components`](#create_with_components) - Factory method for creating assembler with managed components

## API Reference

### PipelineAssembler

_class_ cursus.core.assembler.pipeline_assembler.PipelineAssembler(_dag_, _config_map_, _step_builder_map_, _sagemaker_session=None_, _role=None_, _pipeline_parameters=None_, _notebook_root=None_, _registry_manager=None_, _dependency_resolver=None_)

Assembles pipeline steps using a DAG and step builders with specification-based dependency resolution. This class implements a component-based approach to building SageMaker Pipelines, leveraging the specification-based dependency resolution system to simplify the code and improve maintainability.

**Parameters:**
- **dag** (_PipelineDAG_) – PipelineDAG instance defining the pipeline structure with nodes and edges.
- **config_map** (_Dict[str, BasePipelineConfig]_) – Mapping from step name to config instance for each pipeline step.
- **step_builder_map** (_Dict[str, Type[StepBuilderBase]]_) – Mapping from step type to StepBuilderBase subclass for step creation.
- **sagemaker_session** (_Optional[PipelineSession]_) – SageMaker session to use for creating the pipeline. Defaults to None.
- **role** (_Optional[str]_) – IAM role to use for the pipeline execution. Defaults to None.
- **pipeline_parameters** (_Optional[List[ParameterString]]_) – List of pipeline parameters for parameterized execution. Defaults to empty list.
- **notebook_root** (_Optional[Path]_) – Root directory of the notebook environment. Defaults to current working directory.
- **registry_manager** (_Optional[RegistryManager]_) – Optional registry manager for dependency injection and component management.
- **dependency_resolver** (_Optional[UnifiedDependencyResolver]_) – Optional dependency resolver for specification-based matching.

```python
from cursus.core.assembler.pipeline_assembler import PipelineAssembler
from cursus.api.dag.base_dag import PipelineDAG

# Create DAG and configuration
dag = PipelineDAG()
dag.add_node("preprocessing")
dag.add_node("training")
dag.add_edge("preprocessing", "training")

config_map = {
    "preprocessing": preprocessing_config,
    "training": training_config
}

step_builder_map = {
    "Processing": ProcessingStepBuilder,
    "Training": TrainingStepBuilder
}

# Create assembler
assembler = PipelineAssembler(
    dag=dag,
    config_map=config_map,
    step_builder_map=step_builder_map,
    role="arn:aws:iam::123456789012:role/SageMakerRole"
)
```

#### generate_pipeline

generate_pipeline(_pipeline_name_)

Build and return a SageMaker Pipeline object. This method builds the pipeline by propagating messages between steps using specification-based matching, instantiating steps in topological order, and creating the pipeline with the instantiated steps.

**Parameters:**
- **pipeline_name** (_str_) – Name of the pipeline to be created.

**Returns:**
- **Pipeline** – SageMaker Pipeline object with all steps properly connected and configured.

```python
pipeline = assembler.generate_pipeline("my-ml-pipeline")
print(f"Created pipeline with {len(pipeline.steps)} steps")
```

#### create_with_components

_classmethod_ create_with_components(_dag_, _config_map_, _step_builder_map_, _context_name=None_, _**kwargs_)

Create pipeline assembler with managed components. This factory method creates a pipeline assembler with properly configured dependency components from the factory module.

**Parameters:**
- **dag** (_PipelineDAG_) – PipelineDAG instance defining the pipeline structure.
- **config_map** (_Dict[str, BasePipelineConfig]_) – Mapping from step name to config instance.
- **step_builder_map** (_Dict[str, Type[StepBuilderBase]]_) – Mapping from step type to StepBuilderBase subclass.
- **context_name** (_Optional[str]_) – Optional context name for registry isolation and component management.
- ****kwargs** – Additional arguments to pass to the constructor.

**Returns:**
- **PipelineAssembler** – Configured PipelineAssembler instance with managed dependency components.

```python
assembler = PipelineAssembler.create_with_components(
    dag=dag,
    config_map=config_map,
    step_builder_map=step_builder_map,
    context_name="ml-experiment-1",
    role="arn:aws:iam::123456789012:role/SageMakerRole"
)
```

## Related Documentation

- [Pipeline Template Base](pipeline_template_base.md) - Base class for pipeline templates using the assembler
- [DAG Compiler](../compiler/dag_compiler.md) - DAG compilation and pipeline generation utilities
- [Config Resolver](../compiler/config_resolver.md) - Configuration resolution for DAG nodes
- [Dependency Resolver](../deps/dependency_resolver.md) - Specification-based dependency resolution system
- [Step Builder Base](../base/step_builder_base.md) - Base class for step builders

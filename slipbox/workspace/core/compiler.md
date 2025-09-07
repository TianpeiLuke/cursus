---
tags:
  - code
  - workspace
  - compiler
  - dag
  - pipeline
keywords:
  - WorkspaceDAGCompiler
  - workspace compilation
  - DAG compiler
  - pipeline assembly
  - SageMaker pipeline
topics:
  - workspace management
  - pipeline compilation
  - DAG processing
language: python
date of note: 2024-12-07
---

# Workspace DAG Compiler

Workspace-aware DAG compiler for compiling workspace DAGs to executable SageMaker pipelines.

## Overview

The `WorkspaceDAGCompiler` extends the core `PipelineDAGCompiler` to provide workspace-aware compilation capabilities. This module enables the compilation of workspace DAGs to executable SageMaker pipelines with full workspace component support, cross-workspace dependency management, and comprehensive validation.

The compiler integrates with the consolidated workspace management system, providing enhanced component resolution, compilation preview capabilities, and detailed reporting. It supports Phase 1 integration with the WorkspaceManager and provides specialized functionality for workspace-specific compilation scenarios.

Key features include workspace component validation, compilation feasibility analysis, detailed compilation reporting, and integration with the workspace registry and discovery systems.

## Classes and Methods

### Classes
- [`WorkspaceDAGCompiler`](#workspacedagcompiler) - DAG compiler with workspace component support

### Methods
- [`compile_workspace_dag`](#compile_workspace_dag) - Compile workspace DAG to executable pipeline
- [`preview_workspace_resolution`](#preview_workspace_resolution) - Preview workspace DAG resolution
- [`validate_workspace_components`](#validate_workspace_components) - Validate component availability
- [`generate_compilation_report`](#generate_compilation_report) - Generate comprehensive compilation report
- [`compile_with_detailed_report`](#compile_with_detailed_report) - Compile with detailed reporting
- [`get_workspace_summary`](#get_workspace_summary) - Get compiler capabilities summary

### Class Methods
- [`from_workspace_config`](#from_workspace_config) - Create compiler from workspace configuration
- [`from_workspace_dag`](#from_workspace_dag) - Create compiler from workspace DAG

## API Reference

### WorkspaceDAGCompiler

_class_ cursus.workspace.core.compiler.WorkspaceDAGCompiler(_workspace_root_, _workspace_manager=None_, _sagemaker_session=None_, _role=None_, _**kwargs_)

DAG compiler with workspace component support that extends PipelineDAGCompiler.

**Parameters:**
- **workspace_root** (_str_) – Root path of the workspace directory.
- **workspace_manager** (_Optional[WorkspaceManager]_) – Optional consolidated WorkspaceManager instance for Phase 1 integration.
- **sagemaker_session** (_Optional[PipelineSession]_) – Optional SageMaker session for pipeline execution.
- **role** (_Optional[str]_) – Optional IAM role ARN for SageMaker operations.
- ****kwargs** – Additional arguments passed to parent PipelineDAGCompiler constructor.

```python
from cursus.workspace.core.compiler import WorkspaceDAGCompiler
from sagemaker.workflow.pipeline_context import PipelineSession

# Basic initialization
compiler = WorkspaceDAGCompiler(
    workspace_root="/path/to/workspace",
    role="arn:aws:iam::123456789012:role/SageMakerRole"
)

# With custom session and manager
session = PipelineSession()
compiler = WorkspaceDAGCompiler(
    workspace_root="/path/to/workspace",
    sagemaker_session=session,
    role="arn:aws:iam::123456789012:role/SageMakerRole"
)
```

#### compile_workspace_dag

compile_workspace_dag(_workspace_dag_, _config=None_)

Compile workspace DAG to executable SageMaker pipeline.

**Parameters:**
- **workspace_dag** (_WorkspaceAwareDAG_) – WorkspaceAwareDAG instance to compile.
- **config** (_Optional[Dict[str, Any]]_) – Optional additional configuration dictionary.

**Returns:**
- **Tuple[Pipeline, Dict]** – Tuple containing the compiled SageMaker Pipeline and compilation metadata dictionary.

```python
from cursus.api.dag.workspace_dag import WorkspaceAwareDAG

# Create and compile workspace DAG
workspace_dag = WorkspaceAwareDAG(workspace_root="/path/to/workspace")
workspace_dag.add_workspace_step("data_prep", "alice", {...})
workspace_dag.add_workspace_step("training", "bob", {...})

pipeline, metadata = compiler.compile_workspace_dag(workspace_dag)
print(f"Compiled pipeline with {metadata['step_count']} steps")
```

#### preview_workspace_resolution

preview_workspace_resolution(_workspace_dag_)

Preview how workspace DAG will be resolved to components without actual compilation.

**Parameters:**
- **workspace_dag** (_WorkspaceAwareDAG_) – WorkspaceAwareDAG instance to preview.

**Returns:**
- **Dict[str, Any]** – Preview information dictionary containing component resolution, validation results, and compilation feasibility analysis.

```python
# Preview compilation feasibility
preview = compiler.preview_workspace_resolution(workspace_dag)

print("DAG Summary:", preview['dag_summary'])
print("Can compile:", preview['compilation_feasibility']['can_compile'])
if preview['compilation_feasibility']['blocking_issues']:
    print("Issues:", preview['compilation_feasibility']['blocking_issues'])
```

#### validate_workspace_components

validate_workspace_components(_workspace_dag_)

Validate workspace component availability for compilation.

**Parameters:**
- **workspace_dag** (_WorkspaceAwareDAG_) – WorkspaceAwareDAG instance to validate.

**Returns:**
- **Dict[str, Any]** – Validation result dictionary with component availability status and compilation readiness.

```python
# Validate all required components
validation = compiler.validate_workspace_components(workspace_dag)

if validation['compilation_ready']:
    print("All components available for compilation")
else:
    print("Missing components:", validation.get('missing_components', []))
```

#### generate_compilation_report

generate_compilation_report(_workspace_dag_)

Generate comprehensive compilation report for workspace DAG analysis.

**Parameters:**
- **workspace_dag** (_WorkspaceAwareDAG_) – WorkspaceAwareDAG instance to analyze.

**Returns:**
- **Dict[str, Any]** – Comprehensive report including DAG analysis, complexity metrics, validation results, and resource estimates.

```python
# Generate detailed analysis report
report = compiler.generate_compilation_report(workspace_dag)

print("Complexity:", report['complexity_analysis'])
print("Recommendations:", report['recommendations'])
print("Estimated time:", report['estimated_resources']['compilation_time_seconds'])
```

#### compile_with_detailed_report

compile_with_detailed_report(_workspace_dag_, _config=None_)

Compile workspace DAG with comprehensive detailed compilation report.

**Parameters:**
- **workspace_dag** (_WorkspaceAwareDAG_) – WorkspaceAwareDAG instance to compile.
- **config** (_Optional[Dict[str, Any]]_) – Optional additional configuration dictionary.

**Returns:**
- **Tuple[Pipeline, Dict[str, Any]]** – Tuple containing compiled Pipeline and detailed compilation report.

```python
# Compile with comprehensive reporting
try:
    pipeline, detailed_report = compiler.compile_with_detailed_report(workspace_dag)
    
    print("Compilation successful!")
    print("Total time:", detailed_report['compilation_summary']['total_time'])
    print("Steps compiled:", detailed_report['compilation_summary']['steps_compiled'])
    
except ValueError as e:
    print(f"Compilation failed: {e}")
```

#### get_workspace_summary

get_workspace_summary()

Get summary of workspace compiler capabilities and current state.

**Returns:**
- **Dict[str, Any]** – Summary dictionary containing workspace root, registry information, and compiler capabilities.

```python
# Get compiler status and capabilities
summary = compiler.get_workspace_summary()

print("Workspace root:", summary['workspace_root'])
print("Registry summary:", summary['registry_summary'])
print("Capabilities:", summary['compiler_capabilities'])
```

### from_workspace_config

from_workspace_config(_workspace_config_, _sagemaker_session=None_, _role=None_, _**kwargs_)

Create WorkspaceDAGCompiler instance from workspace configuration.

**Parameters:**
- **workspace_config** (_WorkspacePipelineDefinition_) – WorkspacePipelineDefinition instance containing workspace configuration.
- **sagemaker_session** (_Optional[PipelineSession]_) – Optional SageMaker session for pipeline operations.
- **role** (_Optional[str]_) – Optional IAM role ARN for SageMaker operations.
- ****kwargs** – Additional arguments passed to constructor.

**Returns:**
- **WorkspaceDAGCompiler** – Configured WorkspaceDAGCompiler instance.

```python
from cursus.workspace.core.config import WorkspacePipelineDefinition

# Create from workspace configuration
config = WorkspacePipelineDefinition(
    workspace_root="/path/to/workspace",
    pipeline_name="my_pipeline",
    steps=[...]
)

compiler = WorkspaceDAGCompiler.from_workspace_config(
    config,
    role="arn:aws:iam::123456789012:role/SageMakerRole"
)
```

### from_workspace_dag

from_workspace_dag(_workspace_dag_, _sagemaker_session=None_, _role=None_, _**kwargs_)

Create WorkspaceDAGCompiler instance from workspace DAG.

**Parameters:**
- **workspace_dag** (_WorkspaceAwareDAG_) – WorkspaceAwareDAG instance to create compiler for.
- **sagemaker_session** (_Optional[PipelineSession]_) – Optional SageMaker session for pipeline operations.
- **role** (_Optional[str]_) – Optional IAM role ARN for SageMaker operations.
- ****kwargs** – Additional arguments passed to constructor.

**Returns:**
- **WorkspaceDAGCompiler** – Configured WorkspaceDAGCompiler instance.

```python
# Create compiler directly from DAG
compiler = WorkspaceDAGCompiler.from_workspace_dag(
    workspace_dag,
    role="arn:aws:iam::123456789012:role/SageMakerRole"
)
```

## Related Documentation

- [Workspace Pipeline Assembler](assembler.md) - Pipeline assembly from workspace components
- [Workspace Manager](manager.md) - Consolidated workspace management system
- [Workspace Component Registry](registry.md) - Component discovery and validation
- [DAG Compiler](../../core/compiler/dag_compiler.md) - Base DAG compilation functionality
- [Workspace Configuration](config.md) - Workspace pipeline configuration models

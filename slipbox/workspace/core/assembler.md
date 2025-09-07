---
tags:
  - code
  - workspace
  - core
  - assembler
  - pipeline
keywords:
  - WorkspacePipelineAssembler
  - workspace components
  - pipeline assembly
  - workspace registry
  - component resolution
topics:
  - workspace assembler
  - pipeline assembly
  - workspace components
  - component resolution
language: python
date of note: 2024-12-07
---

# Workspace Pipeline Assembler

Workspace-aware pipeline assembler that extends the core PipelineAssembler to support workspace components while maintaining full backward compatibility with existing functionality.

## Overview

The Workspace Pipeline Assembler module provides enhanced pipeline assembly capabilities with workspace component support. It extends the core PipelineAssembler to discover and resolve components from workspace environments, enabling collaborative pipeline development across isolated developer workspaces.

The module integrates with the consolidated workspace management system, providing component registry functionality, workspace validation, and enhanced assembly capabilities. It supports workspace-specific configurations, cross-workspace dependency resolution, and comprehensive validation of workspace components.

## Classes and Methods

### Classes
- [`WorkspacePipelineAssembler`](#workspacepipelineassembler) - Pipeline assembler with workspace component support

## API Reference

### WorkspacePipelineAssembler

_class_ cursus.workspace.core.assembler.WorkspacePipelineAssembler(_workspace_root_, _workspace_manager=None_, _dag=None_, _config_map=None_, _step_builder_map=None_, _sagemaker_session=None_, _role=None_, _pipeline_parameters=None_, _notebook_root=None_, _**kwargs_)

Pipeline assembler with workspace component support, extending the core PipelineAssembler with workspace-aware functionality.

**Parameters:**
- **workspace_root** (_str_) – Root path of the workspace
- **workspace_manager** (_Optional[WorkspaceManager]_) – Optional consolidated WorkspaceManager instance
- **dag** (_Optional[PipelineDAG]_) – Optional PipelineDAG instance
- **config_map** (_Optional[Dict[str, BasePipelineConfig]]_) – Optional mapping from step name to config instance
- **step_builder_map** (_Optional[Dict[str, Type[StepBuilderBase]]]_) – Optional mapping from step type to builder class
- **sagemaker_session** (_Optional[PipelineSession]_) – Optional SageMaker session
- **role** (_Optional[str]_) – Optional IAM role
- **pipeline_parameters** (_Optional[List]_) – Optional pipeline parameters
- **notebook_root** (_Optional[Path]_) – Optional notebook root directory
- **kwargs** – Additional arguments passed to parent constructor

```python
from cursus.workspace.core.assembler import WorkspacePipelineAssembler
from cursus.workspace.core.manager import WorkspaceManager

# Initialize workspace assembler
workspace_manager = WorkspaceManager("/workspaces")
assembler = WorkspacePipelineAssembler(
    workspace_root="/workspaces",
    workspace_manager=workspace_manager,
    role="arn:aws:iam::123456789012:role/SageMakerRole"
)

print(f"Assembler initialized for workspace: {assembler.workspace_root}")
```

#### assemble_workspace_pipeline

assemble_workspace_pipeline(_workspace_config_)

Assemble pipeline from workspace configuration with component resolution and validation.

**Parameters:**
- **workspace_config** (_WorkspacePipelineDefinition_) – Workspace pipeline configuration

**Returns:**
- **Pipeline** – Assembled SageMaker Pipeline

```python
from cursus.workspace.core.config import WorkspacePipelineDefinition

# Load workspace configuration
workspace_config = WorkspacePipelineDefinition.from_yaml_file("pipeline_config.yaml")

# Assemble pipeline
pipeline = assembler.assemble_workspace_pipeline(workspace_config)

print(f"Assembled pipeline: {pipeline.name}")
print(f"Pipeline steps: {len(pipeline.steps)}")
```

#### validate_workspace_components

validate_workspace_components(_workspace_config_)

Validate workspace component availability and compatibility before assembly.

**Parameters:**
- **workspace_config** (_WorkspacePipelineDefinition_) – Workspace pipeline configuration

**Returns:**
- **Dict[str, Any]** – Validation result dictionary with detailed information

```python
# Validate workspace components
validation_result = assembler.validate_workspace_components(workspace_config)

print(f"Overall validation: {'PASSED' if validation_result['overall_valid'] else 'FAILED'}")
print(f"Component availability: {'PASSED' if validation_result['valid'] else 'FAILED'}")
print(f"Workspace validation: {'PASSED' if validation_result['workspace_valid'] else 'FAILED'}")

if not validation_result['overall_valid']:
    print("Validation errors:")
    for error in validation_result.get('errors', []):
        print(f"  - {error}")
```

#### preview_workspace_assembly

preview_workspace_assembly(_workspace_config_)

Preview workspace assembly without actually building the pipeline, useful for validation and planning.

**Parameters:**
- **workspace_config** (_WorkspacePipelineDefinition_) – Workspace pipeline configuration

**Returns:**
- **Dict[str, Any]** – Preview information dictionary with assembly plan

```python
# Preview assembly before building
preview = assembler.preview_workspace_assembly(workspace_config)

print(f"Pipeline: {preview['workspace_config']['pipeline_name']}")
print(f"Steps: {preview['workspace_config']['step_count']}")
print(f"Developers: {preview['workspace_config']['developers']}")

# Check component resolution
for step_key, resolution in preview['component_resolution'].items():
    status = "✓" if resolution['builder_available'] and resolution['config_available'] else "✗"
    print(f"  {status} {step_key}: {resolution['step_type']}")

# Check assembly plan
if preview['assembly_plan']['dag_valid']:
    print(f"Build order: {preview['assembly_plan']['build_order']}")
else:
    print(f"DAG validation failed: {preview['assembly_plan']['error']}")
```

#### get_workspace_summary

get_workspace_summary()

Get comprehensive summary of workspace components and assembly status.

**Returns:**
- **Dict[str, Any]** – Summary information including registry and assembly status

```python
# Get workspace summary
summary = assembler.get_workspace_summary()

print(f"Workspace root: {summary['workspace_root']}")
print(f"Registry summary: {summary['registry_summary']}")
print(f"Assembly status:")
print(f"  DAG nodes: {summary['assembly_status']['dag_nodes']}")
print(f"  DAG edges: {summary['assembly_status']['dag_edges']}")
print(f"  Configs: {summary['assembly_status']['config_count']}")
print(f"  Builders: {summary['assembly_status']['builder_count']}")
print(f"  Step instances: {summary['assembly_status']['step_instances']}")
```

#### from_workspace_config

from_workspace_config(_workspace_config_, _sagemaker_session=None_, _role=None_, _**kwargs_)

Create assembler from workspace configuration with automatic component resolution.

**Parameters:**
- **workspace_config** (_WorkspacePipelineDefinition_) – Workspace pipeline configuration
- **sagemaker_session** (_Optional[PipelineSession]_) – Optional SageMaker session
- **role** (_Optional[str]_) – Optional IAM role
- **kwargs** – Additional arguments

**Returns:**
- **WorkspacePipelineAssembler** – Configured assembler instance

```python
# Create assembler from configuration
assembler = WorkspacePipelineAssembler.from_workspace_config(
    workspace_config=workspace_config,
    role="arn:aws:iam::123456789012:role/SageMakerRole"
)

# Assemble pipeline
pipeline = assembler.assemble_workspace_pipeline(workspace_config)
```

#### from_workspace_config_file

from_workspace_config_file(_config_file_path_, _sagemaker_session=None_, _role=None_, _**kwargs_)

Create assembler from workspace configuration file (JSON or YAML).

**Parameters:**
- **config_file_path** (_str_) – Path to workspace configuration file
- **sagemaker_session** (_Optional[PipelineSession]_) – Optional SageMaker session
- **role** (_Optional[str]_) – Optional IAM role
- **kwargs** – Additional arguments

**Returns:**
- **WorkspacePipelineAssembler** – Configured assembler instance

```python
# Create assembler from configuration file
assembler = WorkspacePipelineAssembler.from_workspace_config_file(
    config_file_path="/workspaces/pipeline_config.yaml",
    role="arn:aws:iam::123456789012:role/SageMakerRole"
)

# Load and assemble pipeline
workspace_config = WorkspacePipelineDefinition.from_yaml_file("/workspaces/pipeline_config.yaml")
pipeline = assembler.assemble_workspace_pipeline(workspace_config)
```

## Usage Examples

### Complete Workspace Pipeline Assembly

```python
from cursus.workspace.core.assembler import WorkspacePipelineAssembler
from cursus.workspace.core.config import WorkspacePipelineDefinition, WorkspaceStepDefinition
from cursus.workspace.core.manager import WorkspaceManager

def assemble_ml_pipeline():
    """Complete example of workspace pipeline assembly."""
    
    # Initialize workspace manager and assembler
    workspace_manager = WorkspaceManager("/workspaces")
    assembler = WorkspacePipelineAssembler(
        workspace_root="/workspaces",
        workspace_manager=workspace_manager,
        role="arn:aws:iam::123456789012:role/SageMakerRole"
    )
    
    # Create workspace pipeline configuration
    workspace_config = WorkspacePipelineDefinition(
        pipeline_name="ml_training_pipeline",
        workspace_root="/workspaces",
        steps=[
            WorkspaceStepDefinition(
                step_name="data_preprocessing",
                step_type="preprocessing",
                developer_id="data_team",
                dependencies=[],
                config_data={
                    "input_path": "s3://bucket/raw-data/",
                    "output_path": "s3://bucket/processed-data/",
                    "preprocessing_params": {
                        "normalize": True,
                        "remove_outliers": True
                    }
                }
            ),
            WorkspaceStepDefinition(
                step_name="feature_engineering",
                step_type="feature_engineering",
                developer_id="ml_team",
                dependencies=["data_preprocessing"],
                config_data={
                    "input_path": "s3://bucket/processed-data/",
                    "output_path": "s3://bucket/features/",
                    "feature_params": {
                        "create_interactions": True,
                        "polynomial_features": 2
                    }
                }
            ),
            WorkspaceStepDefinition(
                step_name="model_training",
                step_type="training",
                developer_id="ml_team",
                dependencies=["feature_engineering"],
                config_data={
                    "input_path": "s3://bucket/features/",
                    "model_path": "s3://bucket/models/",
                    "hyperparameters": {
                        "learning_rate": 0.01,
                        "max_depth": 6,
                        "n_estimators": 100
                    }
                }
            ),
            WorkspaceStepDefinition(
                step_name="model_evaluation",
                step_type="evaluation",
                developer_id="validation_team",
                dependencies=["model_training"],
                config_data={
                    "model_path": "s3://bucket/models/",
                    "test_data_path": "s3://bucket/test-data/",
                    "metrics_path": "s3://bucket/metrics/",
                    "evaluation_metrics": ["accuracy", "precision", "recall", "f1"]
                }
            )
        ]
    )
    
    print("Starting workspace pipeline assembly...")
    
    # 1. Preview assembly
    print("\n1. Previewing assembly...")
    preview = assembler.preview_workspace_assembly(workspace_config)
    
    print(f"Pipeline: {preview['workspace_config']['pipeline_name']}")
    print(f"Steps: {preview['workspace_config']['step_count']}")
    print(f"Developers: {preview['workspace_config']['developers']}")
    
    # Check component resolution
    print("\nComponent Resolution:")
    for step_key, resolution in preview['component_resolution'].items():
        builder_status = "✓" if resolution['builder_available'] else "✗"
        config_status = "✓" if resolution['config_available'] else "✗"
        print(f"  {step_key}:")
        print(f"    Builder: {builder_status} {resolution.get('builder_class', 'Not found')}")
        print(f"    Config: {config_status} {resolution.get('config_class', 'Not found')}")
    
    # Check assembly plan
    if preview['assembly_plan']['dag_valid']:
        print(f"\nBuild Order: {preview['assembly_plan']['build_order']}")
    else:
        print(f"\nDAG Error: {preview['assembly_plan']['error']}")
        return None
    
    # 2. Validate components
    print("\n2. Validating workspace components...")
    validation_result = assembler.validate_workspace_components(workspace_config)
    
    print(f"Overall validation: {'PASSED' if validation_result['overall_valid'] else 'FAILED'}")
    print(f"Component availability: {'PASSED' if validation_result['valid'] else 'FAILED'}")
    print(f"Workspace validation: {'PASSED' if validation_result['workspace_valid'] else 'FAILED'}")
    
    if not validation_result['overall_valid']:
        print("Validation errors found - cannot proceed with assembly")
        return None
    
    # 3. Assemble pipeline
    print("\n3. Assembling pipeline...")
    try:
        pipeline = assembler.assemble_workspace_pipeline(workspace_config)
        print(f"✓ Pipeline assembled successfully: {pipeline.name}")
        print(f"✓ Pipeline contains {len(pipeline.steps)} steps")
        return pipeline
    except Exception as e:
        print(f"✗ Pipeline assembly failed: {e}")
        return None

# Run the assembly
pipeline = assemble_ml_pipeline()
```

### Workspace Component Validation

```python
from cursus.workspace.core.assembler import WorkspacePipelineAssembler

def validate_workspace_components_detailed():
    """Detailed workspace component validation example."""
    
    assembler = WorkspacePipelineAssembler(
        workspace_root="/workspaces",
        role="arn:aws:iam::123456789012:role/SageMakerRole"
    )
    
    # Load configuration
    workspace_config = WorkspacePipelineDefinition.from_yaml_file("pipeline_config.yaml")
    
    # Perform detailed validation
    validation_result = assembler.validate_workspace_components(workspace_config)
    
    print("=== Workspace Component Validation Report ===")
    
    # Overall status
    print(f"\nOverall Status: {'✓ PASSED' if validation_result['overall_valid'] else '✗ FAILED'}")
    
    # Component availability
    print(f"\nComponent Availability:")
    print(f"  Valid: {'✓' if validation_result['valid'] else '✗'}")
    if 'missing_components' in validation_result:
        print(f"  Missing components: {len(validation_result['missing_components'])}")
        for component in validation_result['missing_components']:
            print(f"    - {component}")
    
    # Workspace-specific validation
    workspace_validation = validation_result.get('workspace_validation', {})
    
    print(f"\nWorkspace Validation:")
    
    # Dependency validation
    dep_validation = workspace_validation.get('dependency_validation', {})
    print(f"  Dependencies: {'✓' if dep_validation.get('valid') else '✗'}")
    if not dep_validation.get('valid'):
        for error in dep_validation.get('errors', []):
            print(f"    - {error}")
    
    # Developer consistency
    dev_validation = workspace_validation.get('developer_consistency', {})
    print(f"  Developer Consistency: {'✓' if dev_validation.get('valid') else '✗'}")
    
    dev_stats = dev_validation.get('developer_stats', {})
    if dev_stats:
        print(f"  Developer Statistics:")
        for dev_id, stats in dev_stats.items():
            print(f"    {dev_id}: {stats['step_count']} steps, types: {stats['step_types']}")
    
    # Step type consistency
    step_validation = workspace_validation.get('step_type_consistency', {})
    print(f"  Step Type Consistency: {'✓' if step_validation.get('valid') else '✗'}")
    
    step_stats = step_validation.get('step_type_stats', {})
    if step_stats:
        print(f"  Step Type Statistics:")
        for step_type, instances in step_stats.items():
            print(f"    {step_type}: {len(instances)} instances")
    
    # Warnings
    all_warnings = []
    for validation_type in ['dependency_validation', 'developer_consistency', 'step_type_consistency']:
        warnings = workspace_validation.get(validation_type, {}).get('warnings', [])
        all_warnings.extend(warnings)
    
    if all_warnings:
        print(f"\nWarnings:")
        for warning in all_warnings:
            print(f"  ⚠ {warning}")
    
    return validation_result

# Run detailed validation
validation_result = validate_workspace_components_detailed()
```

### Assembly Preview and Planning

```python
def preview_and_plan_assembly():
    """Preview assembly and create execution plan."""
    
    assembler = WorkspacePipelineAssembler(
        workspace_root="/workspaces",
        role="arn:aws:iam::123456789012:role/SageMakerRole"
    )
    
    workspace_config = WorkspacePipelineDefinition.from_yaml_file("pipeline_config.yaml")
    
    print("=== Assembly Preview and Planning ===")
    
    # Get preview
    preview = assembler.preview_workspace_assembly(workspace_config)
    
    # Pipeline overview
    config_info = preview['workspace_config']
    print(f"\nPipeline Overview:")
    print(f"  Name: {config_info['pipeline_name']}")
    print(f"  Steps: {config_info['step_count']}")
    print(f"  Developers: {len(config_info['developers'])}")
    print(f"  Developer list: {', '.join(config_info['developers'])}")
    
    # Component resolution analysis
    print(f"\nComponent Resolution Analysis:")
    resolution_stats = {
        'total_steps': 0,
        'builder_available': 0,
        'config_available': 0,
        'fully_resolved': 0
    }
    
    for step_key, resolution in preview['component_resolution'].items():
        resolution_stats['total_steps'] += 1
        if resolution['builder_available']:
            resolution_stats['builder_available'] += 1
        if resolution['config_available']:
            resolution_stats['config_available'] += 1
        if resolution['builder_available'] and resolution['config_available']:
            resolution_stats['fully_resolved'] += 1
        
        # Show detailed resolution info
        builder_status = "✓" if resolution['builder_available'] else "✗"
        config_status = "✓" if resolution['config_available'] else "✗"
        
        print(f"  {step_key} ({resolution['step_type']}):")
        print(f"    Builder: {builder_status} {resolution.get('builder_class', 'Not found')}")
        print(f"    Config:  {config_status} {resolution.get('config_class', 'Not found')}")
        if resolution['dependencies']:
            print(f"    Dependencies: {', '.join(resolution['dependencies'])}")
    
    # Resolution summary
    print(f"\nResolution Summary:")
    print(f"  Total steps: {resolution_stats['total_steps']}")
    print(f"  Builders available: {resolution_stats['builder_available']}/{resolution_stats['total_steps']}")
    print(f"  Configs available: {resolution_stats['config_available']}/{resolution_stats['total_steps']}")
    print(f"  Fully resolved: {resolution_stats['fully_resolved']}/{resolution_stats['total_steps']}")
    
    # Assembly plan
    assembly_plan = preview['assembly_plan']
    print(f"\nAssembly Plan:")
    if assembly_plan['dag_valid']:
        print(f"  DAG Status: ✓ Valid")
        print(f"  Build Order: {' → '.join(assembly_plan['build_order'])}")
        print(f"  Total Steps: {assembly_plan['total_steps']}")
    else:
        print(f"  DAG Status: ✗ Invalid")
        print(f"  Error: {assembly_plan['error']}")
    
    # Validation results
    validation_results = preview.get('validation_results', {})
    if validation_results:
        print(f"\nValidation Results:")
        print(f"  Overall Valid: {'✓' if validation_results.get('overall_valid') else '✗'}")
        print(f"  Component Valid: {'✓' if validation_results.get('valid') else '✗'}")
        print(f"  Workspace Valid: {'✓' if validation_results.get('workspace_valid') else '✗'}")
    
    # Readiness assessment
    ready_for_assembly = (
        assembly_plan['dag_valid'] and
        resolution_stats['fully_resolved'] == resolution_stats['total_steps'] and
        validation_results.get('overall_valid', False)
    )
    
    print(f"\nReadiness Assessment:")
    print(f"  Ready for Assembly: {'✓ YES' if ready_for_assembly else '✗ NO'}")
    
    if not ready_for_assembly:
        print(f"  Issues to resolve:")
        if not assembly_plan['dag_valid']:
            print(f"    - Fix DAG validation errors")
        if resolution_stats['fully_resolved'] < resolution_stats['total_steps']:
            print(f"    - Resolve missing components")
        if not validation_results.get('overall_valid', False):
            print(f"    - Fix validation errors")
    
    return preview

# Run preview and planning
preview = preview_and_plan_assembly()
```

### Error Handling and Recovery

```python
def robust_pipeline_assembly():
    """Robust pipeline assembly with error handling and recovery."""
    
    assembler = WorkspacePipelineAssembler(
        workspace_root="/workspaces",
        role="arn:aws:iam::123456789012:role/SageMakerRole"
    )
    
    try:
        # Load configuration with error handling
        try:
            workspace_config = WorkspacePipelineDefinition.from_yaml_file("pipeline_config.yaml")
        except Exception as e:
            print(f"✗ Failed to load configuration: {e}")
            return None
        
        # Preview assembly first
        print("Previewing assembly...")
        preview = assembler.preview_workspace_assembly(workspace_config)
        
        if 'error' in preview:
            print(f"✗ Preview failed: {preview['error']}")
            return None
        
        # Check if assembly is feasible
        if not preview['assembly_plan']['dag_valid']:
            print(f"✗ DAG validation failed: {preview['assembly_plan']['error']}")
            print("Attempting to fix dependency issues...")
            
            # Here you could implement automatic dependency resolution
            # For now, we'll just report the issue
            return None
        
        # Validate components with retry logic
        print("Validating components...")
        max_retries = 3
        validation_result = None
        
        for attempt in range(max_retries):
            try:
                validation_result = assembler.validate_workspace_components(workspace_config)
                break
            except Exception as e:
                print(f"Validation attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    print("✗ All validation attempts failed")
                    return None
        
        if not validation_result['overall_valid']:
            print("✗ Component validation failed")
            
            # Attempt to provide helpful error messages
            if 'missing_components' in validation_result:
                print("Missing components:")
                for component in validation_result['missing_components']:
                    print(f"  - {component}")
                    print(f"    Suggestion: Check if component exists in workspace")
            
            return None
        
        # Attempt assembly with error recovery
        print("Assembling pipeline...")
        try:
            pipeline = assembler.assemble_workspace_pipeline(workspace_config)
            print(f"✓ Pipeline assembled successfully: {pipeline.name}")
            
            # Verify pipeline structure
            if len(pipeline.steps) != len(workspace_config.steps):
                print(f"⚠ Warning: Expected {len(workspace_config.steps)} steps, got {len(pipeline.steps)}")
            
            return pipeline
            
        except Exception as e:
            print(f"✗ Assembly failed: {e}")
            
            # Attempt to provide recovery suggestions
            print("Recovery suggestions:")
            print("  1. Check that all required components are available")
            print("  2. Verify workspace configuration syntax")
            print("  3. Ensure all dependencies are correctly specified")
            print("  4. Check SageMaker permissions and role configuration")
            
            return None
    
    except Exception as e:
        print(f"✗ Unexpected error during assembly: {e}")
        return None

# Run robust assembly
pipeline = robust_pipeline_assembly()
```

## Integration Points

### Core Assembler Integration
The workspace assembler extends the core PipelineAssembler, maintaining full backward compatibility while adding workspace-aware functionality.

### Workspace Manager Integration
Integrates with the consolidated WorkspaceManager and its specialized managers for component discovery, validation, and lifecycle management.

### Registry Integration
Uses the WorkspaceComponentRegistry for component discovery and resolution across workspace boundaries.

### DAG Integration
Supports both standard PipelineDAG and WorkspaceAwareDAG for cross-workspace dependency management.

## Related Documentation

- [Workspace Core Manager](manager.md) - Core workspace management system
- [Workspace Configuration](config.md) - Workspace configuration management
- [Workspace Registry](registry.md) - Component registry and discovery
- [Core Assembler](../../core/assembler/README.md) - Base pipeline assembler functionality
- [Main Workspace Documentation](../README.md) - Overview of workspace system

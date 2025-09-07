---
tags:
  - code
  - workspace
  - integration
  - staging
  - promotion
keywords:
  - WorkspaceIntegrationManager
  - StagedComponent
  - IntegrationPipeline
  - component promotion
  - integration staging
topics:
  - workspace management
  - integration staging
  - component promotion
language: python
date of note: 2024-12-07
---

# Workspace Integration Manager

Integration staging coordination and management for workspace components with promotion and rollback capabilities.

## Overview

The Workspace Integration Manager provides comprehensive integration staging, component promotion, and cross-workspace integration validation. This module manages the integration staging process, component approval workflows, and cross-workspace integration coordination.

The integration system supports component staging for review, validation of integration readiness, promotion to production environments, and rollback capabilities for failed integrations. It provides comprehensive tracking of component status throughout the integration lifecycle.

Key features include integration staging coordination, component promotion workflows, cross-workspace integration validation, integration readiness assessment, and rollback and recovery capabilities.

## Classes and Methods

### Classes
- [`StagedComponent`](#stagedcomponent) - Represents a staged component with metadata and status tracking
- [`IntegrationPipeline`](#integrationpipeline) - Represents an integration pipeline with multiple components
- [`WorkspaceIntegrationManager`](#workspaceintegrationmanager) - Integration staging coordination and management

### Enums
- [`IntegrationStage`](#integrationstage) - Integration stage enumeration
- [`ComponentStatus`](#componentstatus) - Component status enumeration

### Methods
- [`stage_for_integration`](#stage_for_integration) - Stage component for integration
- [`validate_integration_readiness`](#validate_integration_readiness) - Validate integration readiness
- [`promote_to_production`](#promote_to_production) - Promote component to production
- [`rollback_integration`](#rollback_integration) - Rollback component integration
- [`get_integration_summary`](#get_integration_summary) - Get integration activities summary
- [`get_statistics`](#get_statistics) - Get integration management statistics

## API Reference

### IntegrationStage

_enum_ cursus.workspace.core.integration.IntegrationStage

Integration stage enumeration for component lifecycle management.

**Values:**
- **DEVELOPMENT** – Component in development stage
- **STAGING** – Component in staging for review
- **INTEGRATION** – Component in integration testing
- **PRODUCTION** – Component deployed to production

```python
from cursus.workspace.core.integration import IntegrationStage

# Use integration stages
current_stage = IntegrationStage.STAGING
print("Current stage:", current_stage.value)
```

### ComponentStatus

_enum_ cursus.workspace.core.integration.ComponentStatus

Component status enumeration for tracking component state.

**Values:**
- **PENDING** – Component pending review
- **STAGED** – Component staged for integration
- **APPROVED** – Component approved for promotion
- **REJECTED** – Component rejected from integration
- **PROMOTED** – Component promoted to production
- **ROLLED_BACK** – Component rolled back from production

```python
from cursus.workspace.core.integration import ComponentStatus

# Check component status
if component.status == ComponentStatus.APPROVED:
    print("Component ready for promotion")
```

### StagedComponent

_class_ cursus.workspace.core.integration.StagedComponent(_component_id_, _source_workspace_, _component_type_, _stage="staging"_, _metadata=None_)

Represents a staged component with metadata, status tracking, and approval history.

**Parameters:**
- **component_id** (_str_) – Component identifier.
- **source_workspace** (_str_) – Source workspace identifier.
- **component_type** (_str_) – Type of component (builder, script, etc.).
- **stage** (_str_) – Current integration stage, defaults to "staging".
- **metadata** (_Optional[Dict[str, Any]]_) – Additional component metadata dictionary.

```python
from cursus.workspace.core.integration import StagedComponent

# Create staged component
staged_comp = StagedComponent(
    component_id="data_preprocessing",
    source_workspace="alice",
    component_type="builders",
    stage="integration",
    metadata={
        "staging_path": "/path/to/staging",
        "original_files": [...]
    }
)

print("Component status:", staged_comp.status.value)
print("Staged at:", staged_comp.staged_at)
```

#### to_dict

to_dict()

Convert staged component to dictionary representation.

**Returns:**
- **Dict[str, Any]** – Dictionary containing all component information including status, metadata, and approval history.

```python
# Convert to dictionary for serialization
component_dict = staged_comp.to_dict()

print("Component ID:", component_dict['component_id'])
print("Status:", component_dict['status'])
print("Approval history:", component_dict['approval_history'])
```

### IntegrationPipeline

_class_ cursus.workspace.core.integration.IntegrationPipeline(_pipeline_id_, _components_)

Represents an integration pipeline containing multiple staged components.

**Parameters:**
- **pipeline_id** (_str_) – Pipeline identifier.
- **components** (_List[StagedComponent]_) – List of staged components in the pipeline.

```python
from cursus.workspace.core.integration import IntegrationPipeline, StagedComponent

# Create integration pipeline
components = [
    StagedComponent("data_prep", "alice", "builders"),
    StagedComponent("training", "bob", "builders")
]

pipeline = IntegrationPipeline("ml_pipeline_v1", components)
print("Pipeline created at:", pipeline.created_at)
```

#### to_dict

to_dict()

Convert integration pipeline to dictionary representation.

**Returns:**
- **Dict[str, Any]** – Dictionary containing pipeline information and all component details.

```python
# Convert pipeline to dictionary
pipeline_dict = pipeline.to_dict()

print("Pipeline ID:", pipeline_dict['pipeline_id'])
print("Component count:", len(pipeline_dict['components']))
print("Status:", pipeline_dict['status'])
```

### WorkspaceIntegrationManager

_class_ cursus.workspace.core.integration.WorkspaceIntegrationManager(_workspace_manager_)

Integration staging coordination and management with component promotion and rollback capabilities.

**Parameters:**
- **workspace_manager** (_WorkspaceManager_) – Parent WorkspaceManager instance for integration.

```python
from cursus.workspace.core.integration import WorkspaceIntegrationManager
from cursus.workspace.core.manager import WorkspaceManager

# Create integration manager
workspace_manager = WorkspaceManager("/path/to/workspace")
integration_manager = WorkspaceIntegrationManager(workspace_manager)

# Get integration summary
summary = integration_manager.get_integration_summary()
print("Staged components:", summary['staged_components'])
```

#### stage_for_integration

stage_for_integration(_component_id_, _source_workspace_, _target_stage="integration"_)

Stage component for integration with validation and file copying.

**Parameters:**
- **component_id** (_str_) – Component identifier to stage.
- **source_workspace** (_str_) – Source workspace identifier.
- **target_stage** (_str_) – Target staging area, defaults to "integration".

**Returns:**
- **Dict[str, Any]** – Staging result with success status, staging path, and any issues encountered.

```python
# Stage component for integration
staging_result = integration_manager.stage_for_integration(
    component_id="data_preprocessing",
    source_workspace="alice",
    target_stage="integration"
)

if staging_result['success']:
    print("Component staged at:", staging_result['staging_path'])
    print("Staged at:", staging_result['staged_at'])
else:
    print("Staging issues:", staging_result['issues'])
```

#### validate_integration_readiness

validate_integration_readiness(_staged_components_)

Validate integration readiness for staged components with comprehensive checks.

**Parameters:**
- **staged_components** (_List[str]_) – List of staged component identifiers to validate.

**Returns:**
- **Dict[str, Any]** – Integration readiness validation results with overall status, component results, and recommendations.

```python
# Validate integration readiness
staged_components = ["alice:data_preprocessing", "bob:model_training"]
validation = integration_manager.validate_integration_readiness(staged_components)

if validation['overall_ready']:
    print("All components ready for integration")
else:
    print("Integration issues:", validation['integration_issues'])
    print("Recommendations:", validation['recommendations'])

# Check individual component results
for comp_id, result in validation['component_results'].items():
    print(f"{comp_id}: {'READY' if result['ready'] else 'NOT READY'}")
```

#### promote_to_production

promote_to_production(_component_id_)

Promote component to production environment with validation.

**Parameters:**
- **component_id** (_str_) – Component identifier to promote.

**Returns:**
- **Dict[str, Any]** – Promotion result with success status, production path, and promotion timestamp.

```python
# Promote component to production
promotion_result = integration_manager.promote_to_production("data_preprocessing")

if promotion_result['success']:
    print("Component promoted at:", promotion_result['promoted_at'])
    print("Production path:", promotion_result['production_path'])
else:
    print("Promotion issues:", promotion_result['issues'])
```

#### rollback_integration

rollback_integration(_component_id_)

Rollback component integration with production cleanup.

**Parameters:**
- **component_id** (_str_) – Component identifier to rollback.

**Returns:**
- **Dict[str, Any]** – Rollback result with success status and rollback timestamp.

```python
# Rollback component integration
rollback_result = integration_manager.rollback_integration("data_preprocessing")

if rollback_result['success']:
    print("Component rolled back at:", rollback_result['rolled_back_at'])
else:
    print("Rollback issues:", rollback_result['issues'])
```

#### get_integration_summary

get_integration_summary()

Get summary of integration activities and component status distribution.

**Returns:**
- **Dict[str, Any]** – Summary including staged components count, status distribution, and recent activities.

```python
# Get integration activity summary
summary = integration_manager.get_integration_summary()

print("Staged components:", summary['staged_components'])
print("Integration pipelines:", summary['integration_pipelines'])
print("Status distribution:", summary['component_status_distribution'])

# Review recent activities
for activity in summary['recent_activities']:
    print(f"{activity['component_id']}: {activity['activity']['action']}")
```

#### get_statistics

get_statistics()

Get comprehensive integration management statistics.

**Returns:**
- **Dict[str, Any]** – Statistics including operation metrics, component statistics, and success rates.

```python
# Get comprehensive integration statistics
stats = integration_manager.get_statistics()

print("Total staged components:", stats['integration_operations']['total_staged_components'])
print("Promotion success rate:", stats['integration_operations']['promotion_success_rate'])

# Component statistics
print("By type:", stats['component_statistics']['by_type'])
print("By status:", stats['component_statistics']['by_status'])
print("By workspace:", stats['component_statistics']['by_workspace'])
```

## Integration Workflow

### Component Staging Process

```python
# Complete component staging workflow
def stage_component_workflow(integration_manager, component_id, source_workspace):
    # 1. Stage component
    staging_result = integration_manager.stage_for_integration(
        component_id, source_workspace
    )
    
    if not staging_result['success']:
        print("Staging failed:", staging_result['issues'])
        return False
    
    # 2. Validate readiness
    validation = integration_manager.validate_integration_readiness([
        f"{source_workspace}:{component_id}"
    ])
    
    if not validation['overall_ready']:
        print("Component not ready:", validation['integration_issues'])
        return False
    
    # 3. Promote to production
    promotion = integration_manager.promote_to_production(component_id)
    
    if promotion['success']:
        print("Component successfully promoted to production")
        return True
    else:
        print("Promotion failed:", promotion['issues'])
        return False
```

### Integration Pipeline Management

```python
# Manage integration pipeline
def manage_integration_pipeline(integration_manager, components):
    # Stage all components
    staged_components = []
    for comp_id, workspace in components:
        result = integration_manager.stage_for_integration(comp_id, workspace)
        if result['success']:
            staged_components.append(f"{workspace}:{comp_id}")
    
    # Validate pipeline readiness
    validation = integration_manager.validate_integration_readiness(staged_components)
    
    if validation['overall_ready']:
        # Promote all components
        for comp_id, workspace in components:
            integration_manager.promote_to_production(comp_id)
        print("Pipeline successfully integrated")
    else:
        print("Pipeline validation failed:", validation['integration_issues'])
        
        # Rollback if needed
        for comp_id, workspace in components:
            integration_manager.rollback_integration(comp_id)
```

### Error Handling and Recovery

```python
# Integration with error handling
def safe_integration_workflow(integration_manager, component_id, source_workspace):
    try:
        # Stage component
        staging_result = integration_manager.stage_for_integration(
            component_id, source_workspace
        )
        
        if not staging_result['success']:
            raise ValueError(f"Staging failed: {staging_result['issues']}")
        
        # Validate and promote
        validation = integration_manager.validate_integration_readiness([
            f"{source_workspace}:{component_id}"
        ])
        
        if validation['overall_ready']:
            promotion = integration_manager.promote_to_production(component_id)
            if not promotion['success']:
                raise ValueError(f"Promotion failed: {promotion['issues']}")
        else:
            raise ValueError(f"Validation failed: {validation['integration_issues']}")
            
    except Exception as e:
        print(f"Integration failed: {e}")
        
        # Attempt rollback
        try:
            rollback_result = integration_manager.rollback_integration(component_id)
            if rollback_result['success']:
                print("Successfully rolled back component")
            else:
                print("Rollback also failed:", rollback_result['issues'])
        except Exception as rollback_error:
            print(f"Rollback error: {rollback_error}")
```

## Related Documentation

- [Workspace Manager](manager.md) - Consolidated workspace management system
- [Workspace Discovery Manager](discovery.md) - Component discovery and validation
- [Workspace Configuration](config.md) - Pipeline and step configuration models
- [Workspace API](../api.md) - High-level workspace API interface
- [Workspace Validation](../validation/README.md) - Component validation framework

---
tags:
  - code
  - registry
  - hybrid
  - models
  - pydantic
  - validation
keywords:
  - StepDefinition
  - ResolutionContext
  - StepResolutionResult
  - RegistryValidationResult
  - ConflictAnalysis
  - RegistryType
  - ResolutionMode
  - ResolutionStrategy
  - ConflictType
topics:
  - registry management
  - data models
  - conflict resolution
  - validation
language: python
date of note: 2024-12-07
---

# Registry Hybrid Models

Essential Pydantic data models for the Phase 3 simplified hybrid registry system with optimized validation using enums.

## Overview

The hybrid models module provides core data structures for the hybrid registry system, featuring type-safe data models using Pydantic V2 with enum validation, conflict resolution metadata for intelligent step resolution, registry validation results with comprehensive error tracking, resolution context management for workspace-aware operations, and legacy format compatibility for backward compatibility.

This module serves as the foundation for all registry operations, providing structured data models that ensure type safety and validation consistency across the hybrid registry system. The models support advanced conflict resolution scenarios and maintain compatibility with legacy registry formats.

## Classes and Methods

### Classes
- [`StepDefinition`](#stepdefinition) - Enhanced step definition with registry metadata and conflict resolution capabilities
- [`ResolutionContext`](#resolutioncontext) - Context for step resolution and conflict resolution operations
- [`StepResolutionResult`](#stepresolutionresult) - Result of step conflict resolution with comprehensive metadata
- [`RegistryValidationResult`](#registryvalidationresult) - Results of registry validation with comprehensive error tracking
- [`ConflictAnalysis`](#conflictanalysis) - Analysis of step name conflicts with resolution recommendations

### Enumerations
- [`RegistryType`](#registrytype) - Registry type enumeration for automatic validation
- [`ResolutionMode`](#resolutionmode) - Resolution mode enumeration for automatic validation
- [`ResolutionStrategy`](#resolutionstrategy) - Resolution strategy enumeration for automatic validation
- [`ConflictType`](#conflicttype) - Conflict type enumeration for automatic validation

## API Reference

### StepDefinition

_class_ cursus.registry.hybrid.models.StepDefinition(_name_, _registry_type_, _config_class=None_, _spec_type=None_, _sagemaker_step_type=None_, _builder_step_name=None_, _description=None_, _framework=None_, _job_types=[]_, _workspace_id=None_, _override_source=None_, _priority=100_, _compatibility_tags=[]_, _framework_version=None_, _environment_tags=[]_, _conflict_resolution_strategy=ResolutionStrategy.WORKSPACE_PRIORITY_, _metadata={}_ )

Enhanced step definition with registry metadata and conflict resolution capabilities using Pydantic V2 validation.

**Parameters:**
- **name** (_str_) – Step name identifier (required, minimum length 1).
- **registry_type** (_RegistryType_) – Registry type using enum validation (CORE, WORKSPACE, or OVERRIDE).
- **config_class** (_Optional[str]_) – Configuration class name for the step.
- **spec_type** (_Optional[str]_) – Specification type identifier.
- **sagemaker_step_type** (_Optional[str]_) – SageMaker step type for pipeline integration.
- **builder_step_name** (_Optional[str]_) – Builder class name for step construction.
- **description** (_Optional[str]_) – Human-readable description of the step.
- **framework** (_Optional[str]_) – Framework used by the step (e.g., "xgboost", "pytorch").
- **job_types** (_List[str]_) – List of supported job types for the step.
- **workspace_id** (_Optional[str]_) – Workspace identifier for workspace-specific registrations.
- **override_source** (_Optional[str]_) – Source of override for tracking purposes.
- **priority** (_int_) – Resolution priority where lower values indicate higher priority (default: 100).
- **compatibility_tags** (_List[str]_) – Tags for smart resolution and compatibility checking.
- **framework_version** (_Optional[str]_) – Framework version for compatibility validation.
- **environment_tags** (_List[str]_) – Environment compatibility tags for context-aware resolution.
- **conflict_resolution_strategy** (_ResolutionStrategy_) – Strategy for resolving conflicts with other definitions.
- **metadata** (_Dict[str, Any]_) – Additional metadata dictionary for extensibility.

```python
from cursus.registry.hybrid.models import StepDefinition, RegistryType, ResolutionStrategy

# Create core step definition
step_def = StepDefinition(
    name="xgboost_training",
    registry_type=RegistryType.CORE,
    config_class="XGBoostTrainingConfig",
    framework="xgboost",
    job_types=["training"],
    priority=100,
    conflict_resolution_strategy=ResolutionStrategy.CORE_FALLBACK
)
```

#### to_legacy_format

to_legacy_format()

Converts the step definition to legacy STEP_NAMES format for backward compatibility.

**Returns:**
- **Dict[str, Any]** – Dictionary in legacy format compatible with existing registry systems.

```python
# Convert to legacy format
legacy_dict = step_def.to_legacy_format()
print(legacy_dict)
```

### ResolutionContext

_class_ cursus.registry.hybrid.models.ResolutionContext(_workspace_id=None_, _preferred_framework=None_, _environment_tags=[]_, _resolution_mode=ResolutionMode.AUTOMATIC_, _resolution_strategy=ResolutionStrategy.WORKSPACE_PRIORITY_)

Context for step resolution and conflict resolution operations with workspace awareness.

**Parameters:**
- **workspace_id** (_Optional[str]_) – Current workspace context for resolution operations.
- **preferred_framework** (_Optional[str]_) – Preferred framework for intelligent resolution.
- **environment_tags** (_List[str]_) – Current environment tags for context-aware matching.
- **resolution_mode** (_ResolutionMode_) – Resolution mode using enum validation (AUTOMATIC, INTERACTIVE, or STRICT).
- **resolution_strategy** (_ResolutionStrategy_) – Strategy for conflict resolution using enum validation.

```python
from cursus.registry.hybrid.models import ResolutionContext, ResolutionMode, ResolutionStrategy

# Create resolution context
context = ResolutionContext(
    workspace_id="ml_project_v2",
    preferred_framework="xgboost",
    environment_tags=["production", "gpu"],
    resolution_mode=ResolutionMode.AUTOMATIC,
    resolution_strategy=ResolutionStrategy.WORKSPACE_PRIORITY
)
```

### StepResolutionResult

_class_ cursus.registry.hybrid.models.StepResolutionResult(_step_name_, _resolved_, _source_registry_, _resolution_strategy_, _selected_definition=None_, _reason=None_, _conflicting_definitions=[]_, _workspace_id=None_, _conflict_detected=False_, _conflict_analysis=None_, _errors=[]_, _warnings=[]_, _resolution_metadata={}_ )

Result of step conflict resolution with comprehensive metadata and analysis.

**Parameters:**
- **step_name** (_str_) – Name of the step being resolved.
- **resolved** (_bool_) – Whether resolution was successful.
- **source_registry** (_str_) – Source registry of the resolved step.
- **resolution_strategy** (_str_) – Strategy used for resolution.
- **selected_definition** (_Optional[StepDefinition]_) – The selected step definition if resolution succeeded.
- **reason** (_Optional[str]_) – Human-readable reason for the resolution result.
- **conflicting_definitions** (_List[StepDefinition]_) – List of conflicting definitions found during resolution.
- **workspace_id** (_Optional[str]_) – Workspace context for the resolution.
- **conflict_detected** (_bool_) – Whether conflicts were detected during resolution.
- **conflict_analysis** (_Optional[ConflictAnalysis]_) – Detailed analysis of conflicts if any were found.
- **errors** (_List[str]_) – List of resolution errors encountered.
- **warnings** (_List[str]_) – List of resolution warnings generated.
- **resolution_metadata** (_Dict[str, Any]_) – Additional metadata about the resolution process.

```python
# Process resolution result
if result.resolved:
    print(f"Successfully resolved {result.step_name}")
    print(f"Selected from: {result.source_registry}")
    if result.conflict_detected:
        print(f"Conflicts resolved using: {result.resolution_strategy}")
else:
    print(f"Failed to resolve {result.step_name}: {result.reason}")
```

#### get_resolution_summary

get_resolution_summary()

Gets a summary of the resolution result for reporting and logging.

**Returns:**
- **Dict[str, Any]** – Summary dictionary containing key resolution information.

```python
summary = result.get_resolution_summary()
print(f"Step: {summary['step_name']}, Resolved: {summary['resolved']}")
```

### RegistryValidationResult

_class_ cursus.registry.hybrid.models.RegistryValidationResult(_is_valid_, _registry_type="unknown"_, _issues=[]_, _errors=[]_, _warnings=[]_, _step_count=0_)

Results of registry validation with comprehensive error tracking and reporting.

**Parameters:**
- **is_valid** (_bool_) – Whether the registry validation passed.
- **registry_type** (_str_) – Type of registry that was validated.
- **issues** (_List[str]_) – List of validation issues found.
- **errors** (_List[str]_) – List of validation errors encountered.
- **warnings** (_List[str]_) – List of validation warnings generated.
- **step_count** (_int_) – Number of steps validated (must be ≥ 0).

```python
# Check validation results
if validation_result.is_valid:
    print(f"Registry validation passed for {validation_result.step_count} steps")
else:
    print(f"Validation failed with {len(validation_result.errors)} errors")
    for error in validation_result.errors:
        print(f"Error: {error}")
```

#### get_validation_summary

get_validation_summary()

Gets a summary of validation results for reporting purposes.

**Returns:**
- **Dict[str, Any]** – Summary dictionary containing validation statistics.

```python
summary = validation_result.get_validation_summary()
print(f"Valid: {summary['valid']}, Issues: {summary['issue_count']}")
```

### ConflictAnalysis

_class_ cursus.registry.hybrid.models.ConflictAnalysis(_step_name_, _conflicting_sources_, _resolution_strategy_, _conflicting_definitions=[]_, _resolution_strategies=[]_, _recommended_strategy=None_, _impact_assessment=None_, _workspace_context=None_, _conflict_type="name_conflict"_)

Analysis of step name conflicts with resolution recommendations and impact assessment.

**Parameters:**
- **step_name** (_str_) – Name of the conflicting step (minimum length 1).
- **conflicting_sources** (_List[str]_) – Sources of conflicting definitions.
- **resolution_strategy** (_str_) – Strategy used for resolution.
- **conflicting_definitions** (_List[StepDefinition]_) – List of conflicting step definitions.
- **resolution_strategies** (_List[str]_) – Available resolution strategies.
- **recommended_strategy** (_Optional[str]_) – Recommended resolution strategy.
- **impact_assessment** (_Optional[str]_) – Assessment of conflict impact.
- **workspace_context** (_Optional[str]_) – Workspace context for resolution.
- **conflict_type** (_str_) – Type of conflict identified (default: "name_conflict").

```python
# Analyze conflict details
if conflict_analysis:
    summary = conflict_analysis.get_conflict_summary()
    print(f"Conflict in {summary['step_name']}: {summary['conflict_type']}")
    print(f"Involved workspaces: {summary['involved_workspaces']}")
```

#### get_conflict_summary

get_conflict_summary()

Gets a summary of the conflict for reporting and analysis.

**Returns:**
- **Dict[str, Any]** – Summary dictionary containing conflict details and statistics.

```python
summary = conflict_analysis.get_conflict_summary()
print(f"Frameworks involved: {summary['frameworks']}")
print(f"Resolution strategies: {summary['strategy_count']}")
```

### RegistryType

Registry type enumeration for automatic validation of registry categories.

**Values:**
- **CORE** – Core registry type for framework-provided steps
- **WORKSPACE** – Workspace registry type for project-specific steps
- **OVERRIDE** – Override registry type for temporary modifications

```python
from cursus.registry.hybrid.models import RegistryType

# Use in step definition
registry_type = RegistryType.WORKSPACE
```

### ResolutionMode

Resolution mode enumeration for automatic validation of resolution behavior.

**Values:**
- **AUTOMATIC** – Automatic resolution mode using configured strategies
- **INTERACTIVE** – Interactive resolution mode requiring user input
- **STRICT** – Strict resolution mode with no fallbacks

```python
from cursus.registry.hybrid.models import ResolutionMode

# Configure resolution mode
mode = ResolutionMode.AUTOMATIC
```

### ResolutionStrategy

Resolution strategy enumeration for automatic validation of conflict resolution approaches.

**Values:**
- **WORKSPACE_PRIORITY** – Prioritize workspace definitions over core definitions
- **FRAMEWORK_MATCH** – Match by framework compatibility and version
- **ENVIRONMENT_MATCH** – Match by environment tags and context
- **MANUAL** – Manual resolution required for conflicts
- **HIGHEST_PRIORITY** – Use definition with highest priority value
- **CORE_FALLBACK** – Fallback to core registry when conflicts occur

```python
from cursus.registry.hybrid.models import ResolutionStrategy

# Set resolution strategy
strategy = ResolutionStrategy.WORKSPACE_PRIORITY
```

### ConflictType

Conflict type enumeration for automatic validation of conflict categories.

**Values:**
- **NAME_CONFLICT** – Name conflict between step definitions
- **FRAMEWORK_CONFLICT** – Framework compatibility conflict
- **VERSION_CONFLICT** – Version compatibility conflict
- **WORKSPACE_CONFLICT** – Workspace context conflict

```python
from cursus.registry.hybrid.models import ConflictType

# Identify conflict type
conflict_type = ConflictType.NAME_CONFLICT
```

## Related Documentation

- [Registry Hybrid Manager](manager.md) - UnifiedRegistryManager implementation using these models
- [Registry Hybrid Utils](utils.md) - Utility functions for model conversion and validation
- [Registry Hybrid Setup](setup.md) - Setup utilities for hybrid registry initialization
- [Registry Builder Registry](../builder_registry.md) - Step builder registry integration
- [Registry Exceptions](../exceptions.md) - Registry-specific exception handling

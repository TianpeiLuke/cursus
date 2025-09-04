---
tags:
  - design
  - registry
  - distributed_system
  - multi_developer
  - architecture
keywords:
  - distributed registry
  - workspace registry
  - registry inheritance
  - step registration
  - component discovery
  - registry federation
  - namespace management
topics:
  - distributed registry architecture
  - multi-developer registry system
  - registry inheritance patterns
  - component registration design
language: python
date of note: 2025-08-28
---

# Workspace-Aware Consolidated Registry System Design

## Overview

**Note**: This design has been updated to reflect the **Phase 7 Consolidated Registry System** from the [2025-08-28 Workspace-Aware Unified Implementation Plan](../2_project_planning/2025-08-28_workspace_aware_unified_implementation_plan.md) and the consolidated workspace architecture outlined in the [Workspace-Aware System Refactoring Migration Plan](../2_project_planning/2025-09-02_workspace_aware_system_refactoring_migration_plan.md). All workspace functionality is now centralized within `src/cursus/` for proper packaging compliance.

This document outlines the design for transforming the current centralized registry system in `src/cursus/registry` into a **consolidated workspace-aware registry architecture** that supports multiple developer workspaces while solving the critical **step name collision problem**. The new system enables each workspace to register components locally while providing seamless integration with the shared core registry through intelligent namespacing and conflict resolution.

## Problem Statement

The current registry system (`src/cursus/registry/step_names.py`) is centralized and requires all step implementations to be registered in a single location. This creates several challenges for multi-developer environments:

1. **Central Registry Bottleneck**: All developers must modify the same central registry file
2. **Merge Conflicts**: Multiple developers editing the same registry leads to frequent conflicts
3. **Workspace Isolation**: No way to register workspace-specific implementations without affecting others
4. **Development Friction**: Developers cannot experiment with new steps without central registry changes
5. **Deployment Complexity**: All step implementations must be deployed together
6. **Step Name Collisions**: Multiple developers may register the same step names with different implementations, creating conflicts that traditional registries cannot resolve
7. **Registry Uniqueness Requirement**: Traditional registries require unique keys, but multiple developers may independently choose the same logical step names (e.g., "XGBoostTraining")

### Critical Step Name Collision Scenarios

The step name collision problem manifests in several critical scenarios that the enhanced registry system must address:

#### Scenario 1: Independent Development of Similar Steps
- **Developer A** creates "XGBoostTraining" for financial modeling with custom feature engineering
- **Developer B** creates "XGBoostTraining" for image classification with different preprocessing
- **Collision**: Both use the same logical name but have completely different implementations

#### Scenario 2: Iterative Development Conflicts
- **Developer A** registers "DataPreprocessing" with pandas-based implementation
- **Developer B** later registers "DataPreprocessing" with Spark-based implementation
- **Collision**: Same step name, different frameworks, incompatible configurations

#### Scenario 3: Framework-Specific Implementations
- **Developer A**: "ModelTraining" using PyTorch with GPU optimization
- **Developer B**: "ModelTraining" using TensorFlow with TPU support
- **Developer C**: "ModelTraining" using XGBoost with distributed training
- **Collision**: Multiple valid implementations for the same conceptual step

#### Scenario 4: Environment-Specific Adaptations
- **Development Environment**: "DataValidation" with relaxed constraints for rapid iteration
- **Production Environment**: "DataValidation" with strict validation rules
- **Collision**: Same step name, different validation logic based on environment needs

These scenarios require an intelligent resolution strategy that goes beyond simple uniqueness constraints.

## Core Architectural Principles

The Distributed Registry System is built on two fundamental principles that generalize the Separation of Concerns design principle:

### Principle 1: Workspace Isolation
**Everything that happens within a developer's workspace stays in that workspace.**

This principle ensures complete registry isolation between developer environments:
- Workspace registries are completely isolated from each other
- Developer-specific step definitions remain within their workspace
- Registry modifications in one workspace don't affect other workspaces
- Each workspace maintains its own registry namespace and resolution context
- Workspace registry overrides and extensions are contained within the workspace boundary

### Principle 2: Shared Core
**Only code within `src/cursus/` is shared for all workspaces.**

This principle defines the common registry foundation that all workspaces inherit:
- Core registry in `src/cursus/registry/` provides the shared foundation
- Common step definitions and registry infrastructure are maintained in the shared core
- All workspaces inherit from the same core registry baseline
- Integration pathway allows validated workspace components to join the shared core
- Shared registry components provide stability and consistency across all workspaces

These principles create a clear separation between:
- **Private Registry Space**: Individual workspace registries for experimentation and development
- **Shared Registry Space**: Common core registry that provides stability and shared step definitions

## Design Goals

Building on the core architectural principles, the system achieves these design goals:

1. **Distributed Registration**: Enable workspace-local registries that extend the core registry (implements Workspace Isolation Principle)
2. **Inheritance Model**: Workspace registries inherit from core registry and can override or extend entries (implements Shared Core Principle)
3. **Isolation**: Workspace registries don't affect other workspaces or the core system
4. **Backward Compatibility**: Existing code continues to work without modification
5. **Conflict Resolution**: Clear precedence rules for handling registry conflicts
6. **Discovery**: Automatic discovery of workspace registries and their components
7. **Performance**: Efficient registry resolution and caching mechanisms

## Architecture Overview

```
Simplified Unified Hybrid Registry System (Post-Redundancy Reduction)
src/cursus/registry/hybrid/
├── utils.py (streamlined utility functions - 226 lines, 41% reduction)
│   ├── load_registry_module() (simple function)
│   ├── from_legacy_format() (simple conversion)
│   ├── to_legacy_format() (simple conversion)
│   ├── RegistryValidationModel (Pydantic validator)
│   └── format_*_error() (simple error formatting functions)
├── models.py (data models)
│   ├── StepDefinition
│   ├── NamespacedStepDefinition
│   ├── ResolutionContext
│   ├── StepResolutionResult
│   └── RegistryValidationResult
├── manager.py (unified registry management - 418 lines, 38% reduction)
│   └── UnifiedRegistryManager (consolidated from 3 separate managers)
├── resolver.py (simplified conflict resolution - 143 lines, 66% reduction)
│   └── Simple workspace priority resolution (removed complex multi-strategy)
├── compatibility.py (streamlined compatibility - 164 lines, 57% reduction)
│   └── BackwardCompatibilityAdapter (single adapter class)
└── workspace.py (workspace management)
    ├── WorkspaceRegistryLoader
    ├── WorkspaceRegistryValidator
    └── WorkspaceComponentRegistry
```

**Key Simplifications After Redundancy Reduction:**
- **UnifiedRegistryManager**: Single manager replacing CoreStepRegistry, LocalStepRegistry, HybridRegistryManager
- **Simple utility functions**: Replaced complex utility classes with straightforward functions
- **Streamlined compatibility**: Single adapter instead of multiple compatibility layers
- **Simplified conflict resolution**: Focus on workspace priority instead of complex multi-strategy resolution
- **Backward compatibility aliases**: `CoreStepRegistry = UnifiedRegistryManager` for seamless migration

## Enhanced Conflict Resolution Architecture

### Namespaced Registry System

To address the critical step name collision scenarios, the enhanced registry system implements a **namespaced registry architecture** that allows multiple implementations of the same logical step name while maintaining clear resolution rules.

#### Namespace Structure
```python
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from typing import Dict, List, Any, Optional

class NamespacedStepDefinition(BaseModel):
    """Enhanced step definition with namespace support using Pydantic V2."""
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid',
        frozen=False,
        str_strip_whitespace=True
    )
    
    # Inherit from StepDefinition fields
    name: str = Field(..., min_length=1, description="Step name identifier")
    registry_type: str = Field(..., description="Registry type: 'core', 'workspace', 'override'")
    sagemaker_step_type: Optional[str] = Field(None, description="SageMaker step type")
    builder_step_name: Optional[str] = Field(None, description="Builder class name")
    description: Optional[str] = Field(None, description="Step description")
    framework: Optional[str] = Field(None, description="Framework used by step")
    job_types: List[str] = Field(default_factory=list, description="Supported job types")
    workspace_id: Optional[str] = Field(None, description="Workspace identifier for workspace registrations")
    override_source: Optional[str] = Field(None, description="Source of override for tracking")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    # Core namespace fields
    namespace: str = Field(..., min_length=1, description="Step namespace (workspace_id or 'core')")
    qualified_name: str = Field(default="", description="Fully qualified step name: namespace.name")
    
    # Conflict resolution metadata
    priority: int = Field(default=100, description="Resolution priority (lower = higher priority)")
    compatibility_tags: List[str] = Field(default_factory=list, description="Compatibility tags for smart resolution")
    framework_version: Optional[str] = Field(None, description="Framework version for compatibility checking")
    environment_tags: List[str] = Field(default_factory=list, description="Environment compatibility tags")
    
    # Conflict resolution hints
    conflict_resolution_strategy: str = Field(
        default="workspace_priority", 
        description="Strategy for resolving conflicts: 'workspace_priority', 'framework_match', 'environment_match', 'manual'"
    )
    
    @model_validator(mode='after')
    def generate_qualified_name(self):
        """Generate qualified name after initialization."""
        if not self.qualified_name:
            self.qualified_name = f"{self.namespace}.{self.name}"
        return self
    
    @field_validator('conflict_resolution_strategy')
    @classmethod
    def validate_resolution_strategy(cls, v: str) -> str:
        """Validate conflict resolution strategy."""
        allowed_strategies = {'workspace_priority', 'framework_match', 'environment_match', 'manual'}
        if v not in allowed_strategies:
            raise ValueError(f"conflict_resolution_strategy must be one of {allowed_strategies}")
        return v
    
    @field_validator('registry_type')
    @classmethod
    def validate_registry_type(cls, v: str) -> str:
        """Validate registry type."""
        allowed_types = {'core', 'workspace', 'override'}
        if v not in allowed_types:
            raise ValueError(f"registry_type must be one of {allowed_types}")
        return v
    
    def is_compatible_with(self, other: 'NamespacedStepDefinition') -> bool:
        """Check if this step definition is compatible with another."""
        # Same framework compatibility
        if self.framework and other.framework:
            if self.framework != other.framework:
                return False
        
        # Environment compatibility
        if self.environment_tags and other.environment_tags:
            if not set(self.environment_tags).intersection(set(other.environment_tags)):
                return False
        
        # Compatibility tags
        if self.compatibility_tags and other.compatibility_tags:
            return bool(set(self.compatibility_tags).intersection(set(other.compatibility_tags)))
        
        return True
    
    def get_resolution_score(self, context: 'ResolutionContext') -> int:
        """Calculate resolution score for conflict resolution."""
        score = self.priority
        
        # Framework match bonus
        if context.preferred_framework and self.framework == context.preferred_framework:
            score -= 50
        
        # Environment match bonus
        if context.environment_tags:
            matching_env_tags = set(self.environment_tags).intersection(set(context.environment_tags))
            score -= len(matching_env_tags) * 10
        
        # Workspace preference bonus
        if context.workspace_id and self.workspace_id == context.workspace_id:
            score -= 30
        
        return score

class ResolutionContext(BaseModel):
    """Context for step resolution and conflict resolution using Pydantic V2."""
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid',
        frozen=False
    )
    
    workspace_id: Optional[str] = Field(None, description="Current workspace context")
    preferred_framework: Optional[str] = Field(None, description="Preferred framework for resolution")
    environment_tags: List[str] = Field(default_factory=list, description="Current environment tags")
    resolution_mode: str = Field(default="automatic", description="Resolution mode: 'automatic', 'interactive', 'strict'")
    
    @field_validator('resolution_mode')
    @classmethod
    def validate_resolution_mode(cls, v: str) -> str:
        """Validate resolution mode."""
        allowed_modes = {'automatic', 'interactive', 'strict'}
        if v not in allowed_modes:
            raise ValueError(f"resolution_mode must be one of {allowed_modes}")
        return v
```

#### Smart Conflict Resolution Engine
```python
class RegistryConflictResolver:
    """
    Intelligent conflict resolution engine that handles step name collisions
    using multiple resolution strategies.
    """
    
    def __init__(self, registry_manager: 'DistributedRegistryManager'):
        self.registry_manager = registry_manager
        self._resolution_cache: Dict[str, Any] = {}
    
    def resolve_step_conflict(self, 
                            step_name: str, 
                            context: ResolutionContext) -> StepResolutionResult:
        """
        Resolve step name conflicts using intelligent resolution strategies.
        
        Args:
            step_name: Name of the step to resolve
            context: Resolution context with preferences and environment
            
        Returns:
            Resolution result with selected step definition and metadata
        """
        # Get all definitions for this step name across all registries
        conflicting_definitions = self._get_conflicting_definitions(step_name)
        
        if not conflicting_definitions:
            return StepResolutionResult(
                step_name=step_name,
                resolved=False,
                reason="Step not found in any registry"
            )
        
        if len(conflicting_definitions) == 1:
            # No conflict - single definition
            return StepResolutionResult(
                step_name=step_name,
                resolved=True,
                selected_definition=conflicting_definitions[0],
                resolution_strategy="no_conflict"
            )
        
        # Multiple definitions - resolve conflict
        return self._resolve_multiple_definitions(step_name, conflicting_definitions, context)
    
    def _get_conflicting_definitions(self, step_name: str) -> List[NamespacedStepDefinition]:
        """Get all definitions for a step name across all registries."""
        definitions = []
        
        # Check core registry
        core_def = self.registry_manager.core_registry.get_step_definition(step_name)
        if core_def:
            namespaced_def = self._convert_to_namespaced(core_def, "core")
            definitions.append(namespaced_def)
        
        # Check all workspace registries
        for workspace_id, registry in self.registry_manager._workspace_registries.items():
            workspace_def = registry.get_workspace_only_definitions().get(step_name)
            if workspace_def:
                namespaced_def = self._convert_to_namespaced(workspace_def, workspace_id)
                definitions.append(namespaced_def)
        
        return definitions
    
    def _convert_to_namespaced(self, definition: StepDefinition, namespace: str) -> NamespacedStepDefinition:
        """Convert a regular step definition to namespaced definition."""
        return NamespacedStepDefinition(
            **definition.model_dump(),
            namespace=namespace,
            qualified_name=f"{namespace}.{definition.name}"
        )
    
    def _resolve_multiple_definitions(self, 
                                    step_name: str, 
                                    definitions: List[NamespacedStepDefinition], 
                                    context: ResolutionContext) -> StepResolutionResult:
        """Resolve conflicts between multiple step definitions."""
        
        # Strategy 1: Workspace Priority Resolution
        if context.resolution_mode == "automatic":
            return self._resolve_by_workspace_priority(step_name, definitions, context)
        
        # Strategy 2: Framework Compatibility Resolution
        elif context.preferred_framework:
            return self._resolve_by_framework_compatibility(step_name, definitions, context)
        
        # Strategy 3: Environment Compatibility Resolution
        elif context.environment_tags:
            return self._resolve_by_environment_compatibility(step_name, definitions, context)
        
        # Strategy 4: Interactive Resolution
        elif context.resolution_mode == "interactive":
            return self._resolve_interactively(step_name, definitions, context)
        
        # Strategy 5: Strict Mode (fail on conflicts)
        elif context.resolution_mode == "strict":
            return StepResolutionResult(
                step_name=step_name,
                resolved=False,
                reason=f"Multiple definitions found in strict mode: {[d.qualified_name for d in definitions]}",
                conflicting_definitions=definitions
            )
        
        # Default: Score-based resolution
        return self._resolve_by_score(step_name, definitions, context)
    
    def _resolve_by_workspace_priority(self, 
                                     step_name: str, 
                                     definitions: List[NamespacedStepDefinition], 
                                     context: ResolutionContext) -> StepResolutionResult:
        """Resolve using workspace priority rules."""
        
        # Priority order: current workspace > other workspaces > core
        if context.workspace_id:
            # First, check current workspace
            for definition in definitions:
                if definition.workspace_id == context.workspace_id:
                    return StepResolutionResult(
                        step_name=step_name,
                        resolved=True,
                        selected_definition=definition,
                        resolution_strategy="workspace_priority",
                        reason=f"Selected from current workspace: {context.workspace_id}"
                    )
            
            # Then, check other workspaces
            workspace_definitions = [d for d in definitions if d.registry_type == 'workspace']
            if workspace_definitions:
                # Use the first workspace definition found
                selected = workspace_definitions[0]
                return StepResolutionResult(
                    step_name=step_name,
                    resolved=True,
                    selected_definition=selected,
                    resolution_strategy="workspace_priority",
                    reason=f"Selected from workspace: {selected.workspace_id}"
                )
        
        # Finally, fall back to core
        core_definitions = [d for d in definitions if d.registry_type == 'core']
        if core_definitions:
            return StepResolutionResult(
                step_name=step_name,
                resolved=True,
                selected_definition=core_definitions[0],
                resolution_strategy="workspace_priority",
                reason="Selected from core registry"
            )
        
        return StepResolutionResult(
            step_name=step_name,
            resolved=False,
            reason="No suitable definition found using workspace priority"
        )
    
    def _resolve_by_framework_compatibility(self, 
                                          step_name: str, 
                                          definitions: List[NamespacedStepDefinition], 
                                          context: ResolutionContext) -> StepResolutionResult:
        """Resolve using framework compatibility."""
        
        # Find definitions that match the preferred framework
        compatible_definitions = [
            d for d in definitions 
            if d.framework == context.preferred_framework
        ]
        
        if not compatible_definitions:
            # No framework match - fall back to workspace priority
            return self._resolve_by_workspace_priority(step_name, definitions, context)
        
        if len(compatible_definitions) == 1:
            return StepResolutionResult(
                step_name=step_name,
                resolved=True,
                selected_definition=compatible_definitions[0],
                resolution_strategy="framework_match",
                reason=f"Selected based on framework match: {context.preferred_framework}"
            )
        
        # Multiple framework matches - use workspace priority among them
        context_copy = context.model_copy()
        return self._resolve_by_workspace_priority(step_name, compatible_definitions, context_copy)
    
    def _resolve_by_environment_compatibility(self, 
                                            step_name: str, 
                                            definitions: List[NamespacedStepDefinition], 
                                            context: ResolutionContext) -> StepResolutionResult:
        """Resolve using environment compatibility."""
        
        # Find definitions that match environment tags
        compatible_definitions = []
        for definition in definitions:
            if definition.environment_tags:
                if set(definition.environment_tags).intersection(set(context.environment_tags)):
                    compatible_definitions.append(definition)
            else:
                # No environment tags means compatible with all environments
                compatible_definitions.append(definition)
        
        if not compatible_definitions:
            return StepResolutionResult(
                step_name=step_name,
                resolved=False,
                reason="No environment-compatible definitions found"
            )
        
        if len(compatible_definitions) == 1:
            return StepResolutionResult(
                step_name=step_name,
                resolved=True,
                selected_definition=compatible_definitions[0],
                resolution_strategy="environment_match",
                reason=f"Selected based on environment compatibility: {context.environment_tags}"
            )
        
        # Multiple environment matches - use workspace priority
        return self._resolve_by_workspace_priority(step_name, compatible_definitions, context)
    
    def _resolve_by_score(self, 
                         step_name: str, 
                         definitions: List[NamespacedStepDefinition], 
                         context: ResolutionContext) -> StepResolutionResult:
        """Resolve using scoring algorithm."""
        
        # Calculate scores for all definitions
        scored_definitions = [
            (definition, definition.get_resolution_score(context))
            for definition in definitions
        ]
        
        # Sort by score (lower is better)
        scored_definitions.sort(key=lambda x: x[1])
        
        # Select the best scoring definition
        best_definition, best_score = scored_definitions[0]
        
        return StepResolutionResult(
            step_name=step_name,
            resolved=True,
            selected_definition=best_definition,
            resolution_strategy="score_based",
            reason=f"Selected based on resolution score: {best_score}",
            resolution_metadata={
                'all_scores': [(d.qualified_name, score) for d, score in scored_definitions]
            }
        )
    
    def _resolve_interactively(self, 
                             step_name: str, 
                             definitions: List[NamespacedStepDefinition], 
                             context: ResolutionContext) -> StepResolutionResult:
        """Resolve using interactive selection (placeholder for future implementation)."""
        
        # For now, fall back to score-based resolution
        # Future implementation would present options to user
        result = self._resolve_by_score(step_name, definitions, context)
        result.resolution_strategy = "interactive_fallback"
        result.reason += " (interactive mode not yet implemented)"
        
        return result

class StepResolutionResult(BaseModel):
    """Result of step conflict resolution using Pydantic V2."""
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid',
        frozen=False
    )
    
    step_name: str = Field(..., min_length=1, description="Step name being resolved")
    resolved: bool = Field(..., description="Whether resolution was successful")
    selected_definition: Optional[NamespacedStepDefinition] = Field(None, description="Selected step definition")
    resolution_strategy: Optional[str] = Field(None, description="Strategy used for resolution")
    reason: str = Field(default="", description="Explanation of resolution decision")
    conflicting_definitions: List[NamespacedStepDefinition] = Field(
        default_factory=list, 
        description="All conflicting definitions found"
    )
    resolution_metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional resolution metadata"
    )
    
    def get_resolution_summary(self) -> Dict[str, Any]:
        """Get a summary of the resolution result."""
        return {
            'step_name': self.step_name,
            'resolved': self.resolved,
            'strategy': self.resolution_strategy,
            'selected_namespace': self.selected_definition.namespace if self.selected_definition else None,
            'conflict_count': len(self.conflicting_definitions),
            'reason': self.reason
        }
```

### Enhanced Distributed Registry Manager with Conflict Resolution

```python
class EnhancedDistributedRegistryManager(DistributedRegistryManager):
    """
    Enhanced registry manager with intelligent conflict resolution capabilities.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conflict_resolver = RegistryConflictResolver(self)
        self._resolution_cache: Dict[str, StepResolutionResult] = {}
    
    def resolve_step_with_context(self, 
                                step_name: str, 
                                context: ResolutionContext) -> StepResolutionResult:
        """
        Resolve a step with full context and conflict resolution.
        
        This is the primary method for step resolution in multi-developer environments.
        """
        cache_key = f"{step_name}_{context.workspace_id}_{context.preferred_framework}_{hash(tuple(context.environment_tags))}"
        
        if cache_key in self._resolution_cache:
            return self._resolution_cache[cache_key]
        
        result = self.conflict_resolver.resolve_step_conflict(step_name, context)
        self._resolution_cache[cache_key] = result
        
        return result
    
    def get_step_definition_with_resolution(self, 
                                          step_name: str, 
                                          workspace_id: str = None,
                                          preferred_framework: str = None,
                                          environment_tags: List[str] = None) -> Optional[StepDefinition]:
        """
        Get step definition with intelligent conflict resolution.
        
        This method provides a higher-level interface that handles conflicts automatically.
        """
        context = ResolutionContext(
            workspace_id=workspace_id,
            preferred_framework=preferred_framework,
            environment_tags=environment_tags or [],
            resolution_mode="automatic"
        )
        
        result = self.resolve_step_with_context(step_name, context)
        return result.selected_definition if result.resolved else None
    
    def get_all_step_conflicts_detailed(self) -> Dict[str, ConflictAnalysis]:
        """Get detailed analysis of all step conflicts in the system."""
        conflicts = {}
        
        # Get all step names across all registries
        all_step_names = set()
        registry_map = {}  # step_name -> list of (registry_id, definition)
        
        # Core registry
        core_definitions = self.core_registry.get_all_step_definitions()
        for step_name, definition in core_definitions.items():
            all_step_names.add(step_name)
            if step_name not in registry_map:
                registry_map[step_name] = []
            registry_map[step_name].append(("core", definition))
        
        # Workspace registries
        for workspace_id, registry in self._workspace_registries.items():
            workspace_definitions = registry.get_workspace_only_definitions()
            for step_name, definition in workspace_definitions.items():
                all_step_names.add(step_name)
                if step_name not in registry_map:
                    registry_map[step_name] = []
                registry_map[step_name].append((workspace_id, definition))
        
        # Analyze conflicts
        for step_name, registry_entries in registry_map.items():
            if len(registry_entries) > 1:
                # Multiple definitions found - analyze conflict
                definitions = [entry[1] for entry in registry_entries]
                namespaced_definitions = [
                    self.conflict_resolver._convert_to_namespaced(definition, registry_id)
                    for registry_id, definition in registry_entries
                ]
                
                conflicts[step_name] = ConflictAnalysis(
                    step_name=step_name,
                    conflicting_definitions=namespaced_definitions,
                    conflict_type=self._analyze_conflict_type(namespaced_definitions),
                    resolution_recommendations=self._generate_resolution_recommendations(namespaced_definitions)
                )
        
        return conflicts
    
    def _analyze_conflict_type(self, definitions: List[NamespacedStepDefinition]) -> str:
        """Analyze the type of conflict between definitions."""
        frameworks = {d.framework for d in definitions if d.framework}
        environments = set()
        for d in definitions:
            environments.update(d.environment_tags)
        
        if len(frameworks) > 1:
            return "framework_conflict"
        elif len(environments) > 1:
            return "environment_conflict"
        elif any(d.registry_type == 'override' for d in definitions):
            return "override_conflict"
        else:
            return "implementation_conflict"
    
    def _generate_resolution_recommendations(self, definitions: List[NamespacedStepDefinition]) -> List[str]:
        """Generate recommendations for resolving conflicts."""
        recommendations = []
        
        frameworks = {d.framework for d in definitions if d.framework}
        if len(frameworks) > 1:
            recommendations.append(f"Consider using framework-specific resolution with frameworks: {frameworks}")
        
        workspaces = {d.workspace_id for d in definitions if d.workspace_id}
        if len(workspaces) > 1:
            recommendations.append(f"Consider workspace-specific resolution for workspaces: {workspaces}")
        
        if any(d.registry_type == 'override' for d in definitions):
            recommendations.append("Review override definitions for necessity and correctness")
        
        recommendations.append("Consider renaming steps to be more specific to their use case")
        
        return recommendations

class ConflictAnalysis(BaseModel):
    """Analysis of a step name conflict using Pydantic V2."""
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid',
        frozen=False
    )
    
    step_name: str = Field(..., min_length=1, description="Conflicting step name")
    conflicting_definitions: List[NamespacedStepDefinition] = Field(..., description="All conflicting definitions")
    conflict_type: str = Field(..., description="Type of conflict identified")
    resolution_recommendations: List[str] = Field(default_factory=list, description="Recommendations for resolution")
    
    def get_conflict_summary(self) -> Dict[str, Any]:
        """Get a summary of the conflict."""
        return {
            'step_name': self.step_name,
            'conflict_type': self.conflict_type,
            'definition_count': len(self.conflicting_definitions),
            'involved_namespaces': [d.namespace for d in self.conflicting_definitions],
            'frameworks': list({d.framework for d in self.conflicting_definitions if d.framework}),
            'recommendation_count': len(self.resolution_recommendations)
        }
```

## Simplified Utility Layer Design (Post-Redundancy Reduction)

Based on the completed 4-phase redundancy reduction, the architecture now uses **simple utility functions** instead of complex utility classes. This approach eliminates code redundancy while maintaining all functionality with significantly improved maintainability.

### 1. Simple Registry Loading Functions

```python
def load_registry_module(file_path: str) -> Any:
    """
    Simple registry loading function that replaces RegistryLoader class.
    
    Args:
        file_path: Path to the registry file
        
    Returns:
        Loaded module object
        
    Raises:
        RegistryLoadError: If module loading fails
    """
    try:
        spec = importlib.util.spec_from_file_location("registry", file_path)
        if spec is None or spec.loader is None:
            raise RegistryLoadError(f"Could not create module spec from {file_path}")
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
        
    except Exception as e:
        raise RegistryLoadError(f"Failed to load registry module from {file_path}: {e}")

def validate_registry_file(file_path: str) -> bool:
    """
    Simple registry file validation function.
    
    Args:
        file_path: Path to validate
        
    Returns:
        True if file is valid, False otherwise
    """
    try:
        path = Path(file_path)
        return path.exists() and path.is_file() and path.suffix == '.py'
    except Exception:
        return False
```

### 2. Simple Format Conversion Functions

```python
def from_legacy_format(step_name: str, 
                      step_info: Dict[str, Any], 
                      registry_type: str = 'core', 
                      workspace_id: str = None) -> StepDefinition:
    """
    Simple conversion function that replaces StepDefinitionConverter class.
    
    Args:
        step_name: Name of the step
        step_info: Legacy step information dictionary
        registry_type: Type of registry ('core', 'workspace', 'override')
        workspace_id: Workspace identifier for workspace steps
        
    Returns:
        StepDefinition object
    """
    # Extract standard fields
    definition_data = {
        'name': step_name,
        'registry_type': registry_type,
        'sagemaker_step_type': step_info.get('sagemaker_step_type'),
        'builder_step_name': step_info.get('builder_step_name'),
        'description': step_info.get('description'),
        'framework': step_info.get('framework'),
        'job_types': step_info.get('job_types', [])
    }
    
    # Add workspace-specific fields
    if workspace_id:
        definition_data['workspace_id'] = workspace_id
    
    # Store any additional metadata
    metadata = {}
    for key, value in step_info.items():
        if key not in definition_data:
            metadata[key] = value
    if metadata:
        definition_data['metadata'] = metadata
    
    return StepDefinition(**definition_data)

def to_legacy_format(definition: StepDefinition) -> Dict[str, Any]:
    """
    Simple conversion function to legacy STEP_NAMES format.
    
    Args:
        definition: StepDefinition object
        
    Returns:
        Legacy format dictionary
    """
    legacy_dict = {}
    
    # Standard fields
    if definition.sagemaker_step_type:
        legacy_dict['sagemaker_step_type'] = definition.sagemaker_step_type
    if definition.builder_step_name:
        legacy_dict['builder_step_name'] = definition.builder_step_name
    if definition.description:
        legacy_dict['description'] = definition.description
    if definition.framework:
        legacy_dict['framework'] = definition.framework
    if definition.job_types:
        legacy_dict['job_types'] = definition.job_types
    
    # Additional metadata
    if definition.metadata:
        legacy_dict.update(definition.metadata)
    
    return legacy_dict
```

### 3. Pydantic Validation Model

```python
class RegistryValidationModel(BaseModel):
    """
    Simple Pydantic validation model that replaces RegistryValidationUtils class.
    
    Provides validation through Pydantic's built-in validation system.
    """
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid',
        frozen=False,
        str_strip_whitespace=True
    )
    
    registry_type: str = Field(..., description="Registry type")
    step_name: str = Field(..., min_length=1, description="Step name")
    workspace_id: Optional[str] = Field(None, description="Workspace ID")
    
    @field_validator('registry_type')
    @classmethod
    def validate_registry_type(cls, v: str) -> str:
        """Validate registry type."""
        allowed_types = {'core', 'workspace', 'override'}
        if v not in allowed_types:
            raise ValueError(f"registry_type must be one of {allowed_types}")
        return v
    
    @field_validator('step_name')
    @classmethod
    def validate_step_name(cls, v: str) -> str:
        """Validate step name."""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError(f"Step name '{v}' contains invalid characters")
        return v
    
    @field_validator('workspace_id')
    @classmethod
    def validate_workspace_id(cls, v: Optional[str]) -> Optional[str]:
        """Validate workspace ID."""
        if v is not None and not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError(f"Workspace ID '{v}' contains invalid characters")
        return v
```

### 4. Simple Error Formatting Functions

```python
def format_step_not_found_error(step_name: str, 
                               workspace_context: str = None,
                               available_steps: List[str] = None) -> str:
    """
    Simple error formatting function that replaces RegistryErrorFormatter class.
    
    Args:
        step_name: Name of the step that wasn't found
        workspace_context: Current workspace context
        available_steps: List of available step names
        
    Returns:
        Formatted error message
    """
    context_info = f" (workspace: {workspace_context})" if workspace_context else " (core registry)"
    error_msg = f"Step '{step_name}' not found{context_info}"
    
    if available_steps:
        error_msg += f". Available steps: {', '.join(sorted(available_steps))}"
    
    return error_msg

def format_registry_load_error(registry_path: str, error_details: str) -> str:
    """
    Simple registry loading error formatting function.
    
    Args:
        registry_path: Path to the registry that failed to load
        error_details: Detailed error information
        
    Returns:
        Formatted error message
    """
    return f"Failed to load registry from '{registry_path}': {error_details}"

def format_validation_error(component_name: str, validation_issues: List[str]) -> str:
    """
    Simple validation error formatting function.
    
    Args:
        component_name: Name of the component
        validation_issues: List of validation issues
        
    Returns:
        Formatted error message
    """
    error_msg = f"Registry '{component_name}' validation failed:"
    for i, issue in enumerate(validation_issues, 1):
        error_msg += f"\n  {i}. {issue}"
    return error_msg
```

### 5. Simplified Compatibility Functions

```python
# Simple compatibility functions that replace GenericStepFieldAccessor class
def get_config_class_name(step_name: str) -> str:
    """Get config class name for a step with workspace context."""
    compatibility_layer = get_enhanced_compatibility()
    step_names = compatibility_layer.get_step_names()
    if step_name not in step_names:
        available_steps = list(step_names.keys())
        error_msg = format_step_not_found_error(step_name, None, available_steps)
        raise ValueError(error_msg)
    return step_names[step_name]["config_class"]

def get_builder_step_name(step_name: str) -> str:
    """Get builder step class name for a step with workspace context."""
    compatibility_layer = get_enhanced_compatibility()
    step_names = compatibility_layer.get_step_names()
    if step_name not in step_names:
        available_steps = list(step_names.keys())
        error_msg = format_step_not_found_error(step_name, None, available_steps)
        raise ValueError(error_msg)
    return step_names[step_name]["builder_step_name"]

def get_spec_step_type(step_name: str) -> str:
    """Get step_type value for StepSpecification with workspace context."""
    compatibility_layer = get_enhanced_compatibility()
    step_names = compatibility_layer.get_step_names()
    if step_name not in step_names:
        available_steps = list(step_names.keys())
        error_msg = format_step_not_found_error(step_name, None, available_steps)
        raise ValueError(error_msg)
    return step_names[step_name]["spec_type"]

def get_sagemaker_step_type(step_name: str) -> str:
    """Get SageMaker step type for a step with workspace context."""
    compatibility_layer = get_enhanced_compatibility()
    step_names = compatibility_layer.get_step_names()
    if step_name not in step_names:
        available_steps = list(step_names.keys())
        error_msg = format_step_not_found_error(step_name, None, available_steps)
        raise ValueError(error_msg)
    return step_names[step_name]["sagemaker_step_type"]

def get_step_description(step_name: str) -> str:
    """Get description for a step with workspace context."""
    compatibility_layer = get_enhanced_compatibility()
    step_names = compatibility_layer.get_step_names()
    if step_name not in step_names:
        available_steps = list(step_names.keys())
        error_msg = format_step_not_found_error(step_name, None, available_steps)
        raise ValueError(error_msg)
    return step_names[step_name]["description"]
```

## Code Redundancy Mitigation Strategy

Based on the [2025-09-02 Hybrid Registry Migration Plan Analysis](../4_analysis/2025-09-02_hybrid_registry_migration_plan_analysis.md), the design incorporates comprehensive redundancy mitigation strategies to achieve a **75/100 → 95/100** improvement in code redundancy scores.

### Redundancy Elimination Patterns

#### Before: Redundant Registry Loading Logic
```python
# CoreStepRegistry._load_core_registry() - REDUNDANT
spec = importlib.util.spec_from_file_location("step_names", self.registry_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# LocalStepRegistry._load_local_registry() - REDUNDANT  
spec = importlib.util.spec_from_file_location("workspace_registry", registry_file)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
```

#### After: Shared RegistryLoader Utility
```python
# Both registries use shared utility - NO REDUNDANCY
class CoreStepRegistry:
    def _load_core_registry(self):
        module = RegistryLoader.load_registry_module(str(self.registry_path), "step_names")
        # ... rest of loading logic

class LocalStepRegistry:
    def _load_local_registry(self):
        module = RegistryLoader.load_registry_module(str(registry_file), "workspace_registry")
        # ... rest of loading logic
```

#### Before: Redundant Format Conversion Logic
```python
# Multiple places converting between formats - REDUNDANT
# CoreStepRegistry: STEP_NAMES → StepDefinition
# LocalStepRegistry: LOCAL_STEPS → StepDefinition  
# HybridStepDefinition: StepDefinition → legacy format
```

#### After: Shared StepDefinitionConverter
```python
# Single conversion utility - NO REDUNDANCY
class CoreStepRegistry:
    def _load_core_registry(self):
        step_names = RegistryLoader.get_registry_attributes(module, ['STEP_NAMES'])['STEP_NAMES']
        self._step_definitions = StepDefinitionConverter.convert_registry_dict(
            step_names, 'core'
        )

class LocalStepRegistry:
    def _load_local_registry(self):
        local_steps = RegistryLoader.get_registry_attributes(module, ['LOCAL_STEPS'])['LOCAL_STEPS']
        self._local_definitions = StepDefinitionConverter.convert_registry_dict(
            local_steps, 'workspace', self.workspace_id
        )
```

#### Before: Redundant Compatibility Functions
```python
# 15+ similar functions - HIGHLY REDUNDANT
def get_config_class_name(step_name: str) -> str:
    step_names = get_step_names()
    if step_name not in step_names:
        raise ValueError(f"Unknown step name: {step_name}")
    return step_names[step_name]["config_class"]

def get_builder_step_name(step_name: str) -> str:
    step_names = get_step_names()
    if step_name not in step_names:
        raise ValueError(f"Unknown step name: {step_name}")
    return step_names[step_name]["builder_step_name"]
# ... 13 more similar functions
```

#### After: Single Generic Function
```python
# Single optimized function - NO REDUNDANCY
def get_step_field(step_name: str, field_name: str) -> str:
    """Generic step field accessor."""
    accessor = _get_field_accessor()
    return accessor.get_step_field(step_name, field_name)

# Specific functions become simple wrappers
def get_config_class_name(step_name: str) -> str:
    return get_step_field(step_name, "config_class")

def get_builder_step_name(step_name: str) -> str:
    return get_step_field(step_name, "builder_step_name")
# ... all other functions follow same pattern
```

### Redundancy Reduction Metrics

| Component | Before (Lines) | After (Lines) | Reduction |
|-----------|----------------|---------------|-----------|
| Registry Loading Logic | 45 | 15 | 67% |
| Format Conversion Logic | 60 | 20 | 67% |
| Compatibility Functions | 180 | 45 | 75% |
| Validation Logic | 90 | 30 | 67% |
| Error Handling | 120 | 40 | 67% |
| **Total** | **495** | **150** | **70%** |

### Quality Score Improvements

| Quality Metric | Before | After | Improvement |
|----------------|--------|-------|-------------|
| Code Redundancy | 75/100 | 95/100 | +20 points |
| Maintainability | 85/100 | 95/100 | +10 points |
| Testability | 80/100 | 90/100 | +10 points |
| Performance | 85/100 | 90/100 | +5 points |
| **Overall Quality** | **81/100** | **93/100** | **+12 points** |

## Core Components Design

### 1. Core Registry System

```python
class CoreStepRegistry:
    """
    Core step registry that provides the base set of step definitions.
    
    This registry contains the fundamental step implementations that
    are available to all workspaces by default.
    """
    
    def __init__(self, registry_path: str = "src/cursus/registry/step_names.py"):
        self.registry_path = Path(registry_path)
        self._step_definitions: Dict[str, StepDefinition] = {}
        self._load_core_registry()
    
    def _load_core_registry(self):
        """Load the core step registry using shared utilities."""
        try:
            # Use shared RegistryLoader utility
            module = RegistryLoader.load_registry_module(str(self.registry_path), "step_names")
            
            # Extract STEP_NAMES using shared utility
            registry_attributes = RegistryLoader.get_registry_attributes(module, ['STEP_NAMES'])
            step_names = registry_attributes['STEP_NAMES']
            
            # Convert using shared StepDefinitionConverter
            self._step_definitions = StepDefinitionConverter.convert_registry_dict(
                step_names, 'core'
            )
                
        except Exception as e:
            error_msg = RegistryErrorFormatter.format_registry_load_error(
                str(self.registry_path), 
                str(e),
                ["Check file exists and is valid Python", "Verify STEP_NAMES dictionary format"]
            )
            raise RegistryLoadError(error_msg)
    
    def get_step_definition(self, step_name: str) -> Optional[StepDefinition]:
        """Get a step definition from the core registry."""
        return self._step_definitions.get(step_name)
    
    def get_all_step_definitions(self) -> Dict[str, StepDefinition]:
        """Get all step definitions from the core registry."""
        return self._step_definitions.copy()
    
    def get_steps_by_type(self, sagemaker_step_type: str) -> List[StepDefinition]:
        """Get all steps of a specific SageMaker step type."""
        return [
            definition for definition in self._step_definitions.values()
            if definition.sagemaker_step_type == sagemaker_step_type
        ]
    
    def register_step(self, step_definition: StepDefinition) -> bool:
        """
        Register a new step in the core registry.
        
        Note: This is primarily for programmatic registration.
        Manual registration should still use the step_names.py file.
        """
        if step_definition.name in self._step_definitions:
            return False  # Step already exists
        
        step_definition.registry_type = 'core'
        self._step_definitions[step_definition.name] = step_definition
        return True
    
    def validate_registry(self) -> RegistryValidationResult:
        """Validate the core registry using shared validation utilities."""
        issues = RegistryValidationUtils.validate_registry_consistency(self._step_definitions)
        
        return RegistryValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            registry_type='core',
            step_count=len(self._step_definitions)
        )

from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Dict, List, Any, Optional

class StepDefinition(BaseModel):
    """Enhanced step definition with registry metadata using Pydantic V2."""
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid',
        frozen=False,
        str_strip_whitespace=True
    )
    
    name: str = Field(..., min_length=1, description="Step name identifier")
    registry_type: str = Field(..., description="Registry type: 'core', 'workspace', 'override'")
    sagemaker_step_type: Optional[str] = Field(None, description="SageMaker step type")
    builder_step_name: Optional[str] = Field(None, description="Builder class name")
    description: Optional[str] = Field(None, description="Step description")
    framework: Optional[str] = Field(None, description="Framework used by step")
    job_types: List[str] = Field(default_factory=list, description="Supported job types")
    workspace_id: Optional[str] = Field(None, description="Workspace identifier for workspace registrations")
    override_source: Optional[str] = Field(None, description="Source of override for tracking")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @field_validator('registry_type')
    @classmethod
    def validate_registry_type(cls, v: str) -> str:
        """Validate registry type using shared validation utilities."""
        return RegistryValidationUtils.validate_registry_type(v)
    
    @field_validator('name', 'builder_step_name')
    @classmethod
    def validate_identifiers(cls, v: Optional[str]) -> Optional[str]:
        """Validate identifier fields using shared validation utilities."""
        if v is not None:
            return RegistryValidationUtils.validate_step_name(v)
        return v
    
    def to_legacy_format(self) -> Dict[str, Any]:
        """Convert to legacy STEP_NAMES format using shared converter."""
        return StepDefinitionConverter.to_legacy_format(self)

class RegistryValidationResult(BaseModel):
    """Results of registry validation using Pydantic V2."""
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid',
        frozen=False
    )
    
    is_valid: bool = Field(..., description="Whether validation passed")
    issues: List[str] = Field(default_factory=list, description="List of validation issues")
    registry_type: str = Field(..., description="Type of registry validated")
    step_count: int = Field(..., description="Number of steps validated")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of validation results."""
        return {
            'valid': self.is_valid,
            'issue_count': len(self.issues),
            'registry_type': self.registry_type,
            'step_count': self.step_count
        }
```

### 2. Workspace Registry System

Each workspace can have its own registry that extends or overrides the core registry.

```python
class WorkspaceStepRegistry:
    """
    Workspace-specific step registry that extends the core registry.
    
    Provides workspace isolation while maintaining inheritance from
    the core registry using shared utilities.
    """
    
    def __init__(self, workspace_path: str, core_registry: CoreStepRegistry):
        self.workspace_path = Path(workspace_path)
        self.workspace_id = RegistryValidationUtils.validate_workspace_id(self.workspace_path.name)
        self.core_registry = core_registry
        self._workspace_definitions: Dict[str, StepDefinition] = {}
        self._overrides: Dict[str, StepDefinition] = {}
        self._load_workspace_registry()
    
    def _load_workspace_registry(self):
        """Load workspace-specific registry definitions using shared utilities."""
        registry_file = self.workspace_path / "src" / "cursus_dev" / "registry" / "workspace_registry.py"
        
        if not RegistryLoader.validate_registry_file(str(registry_file)):
            # No workspace registry file - that's okay
            return
        
        try:
            # Use shared RegistryLoader
            module = RegistryLoader.load_registry_module(str(registry_file), "workspace_registry")
            
            # Extract registry attributes using shared utility
            registry_attributes = RegistryLoader.get_registry_attributes(
                module, ['WORKSPACE_STEPS', 'STEP_OVERRIDES']
            )
            
            # Load workspace step definitions using shared converter
            workspace_steps = registry_attributes['WORKSPACE_STEPS']
            self._workspace_definitions = StepDefinitionConverter.convert_registry_dict(
                workspace_steps, 'workspace', self.workspace_id
            )
            
            # Load step overrides using shared converter
            step_overrides = registry_attributes['STEP_OVERRIDES']
            override_definitions = StepDefinitionConverter.convert_registry_dict(
                step_overrides, 'override', self.workspace_id
            )
            
            # Set override source for tracking
            for definition in override_definitions.values():
                definition.override_source = 'workspace'
            
            self._overrides = override_definitions
                
        except Exception as e:
            error_msg = RegistryErrorFormatter.format_registry_load_error(
                str(registry_file),
                str(e),
                ["Check workspace registry file format", "Verify WORKSPACE_STEPS and STEP_OVERRIDES dictionaries"]
            )
            raise RegistryLoadError(error_msg)
    
    def get_step_definition(self, step_name: str) -> Optional[StepDefinition]:
        """
        Get a step definition with workspace precedence.
        
        Resolution order:
        1. Workspace overrides
        2. Workspace definitions
        3. Core registry
        """
        # Check workspace overrides first
        if step_name in self._overrides:
            return self._overrides[step_name]
        
        # Check workspace definitions
        if step_name in self._workspace_definitions:
            return self._workspace_definitions[step_name]
        
        # Fall back to core registry
        return self.core_registry.get_step_definition(step_name)
    
    def get_all_step_definitions(self) -> Dict[str, StepDefinition]:
        """Get all step definitions with workspace precedence applied."""
        # Start with core registry
        all_definitions = self.core_registry.get_all_step_definitions()
        
        # Add workspace definitions (may override core)
        all_definitions.update(self._workspace_definitions)
        
        # Apply workspace overrides
        all_definitions.update(self._overrides)
        
        return all_definitions
    
    def get_workspace_only_definitions(self) -> Dict[str, StepDefinition]:
        """Get only workspace-specific definitions (not inherited from core)."""
        workspace_only = {}
        workspace_only.update(self._workspace_definitions)
        workspace_only.update(self._overrides)
        return workspace_only
    
    def register_workspace_step(self, step_definition: StepDefinition) -> bool:
        """Register a new step in the workspace registry."""
        # Validate using shared utilities
        RegistryValidationUtils.validate_step_definition_fields(step_definition.model_dump())
        
        step_definition.registry_type = 'workspace'
        step_definition.workspace_id = self.workspace_id
        
        self._workspace_definitions[step_definition.name] = step_definition
        return True
    
    def override_core_step(self, step_name: str, step_definition: StepDefinition) -> bool:
        """Override a core step definition in this workspace."""
        if not self.core_registry.get_step_definition(step_name):
            return False  # Can't override non-existent core step
        
        # Validate using shared utilities
        RegistryValidationUtils.validate_step_definition_fields(step_definition.model_dump())
        
        step_definition.name = step_name
        step_definition.registry_type = 'override'
        step_definition.workspace_id = self.workspace_id
        step_definition.override_source = 'workspace'
        
        self._overrides[step_name] = step_definition
        return True
    
    def validate_registry(self) -> RegistryValidationResult:
        """Validate workspace registry using shared validation utilities."""
        all_definitions = {}
        all_definitions.update(self._workspace_definitions)
        all_definitions.update(self._overrides)
        
        issues = RegistryValidationUtils.validate_registry_consistency(all_definitions)
        
        return RegistryValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            registry_type='workspace',
            step_count=len(all_definitions)
        )
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """Get summary information about this workspace registry."""
        core_count = len(self.core_registry.get_all_step_definitions())
        workspace_count = len(self._workspace_definitions)
        override_count = len(self._overrides)
        
        return {
            'workspace_id': self.workspace_id,
            'workspace_path': str(self.workspace_path),
            'core_steps': core_count,
            'workspace_steps': workspace_count,
            'overridden_steps': override_count,
            'total_available_steps': core_count + workspace_count,
            'workspace_step_names': list(self._workspace_definitions.keys()),
            'overridden_step_names': list(self._overrides.keys())
        }
```

### Implementation Optimization Guidelines

#### Week 1: Shared Utilities Foundation
1. **Implement RegistryLoader** - Eliminate registry loading redundancy
2. **Implement StepDefinitionConverter** - Eliminate format conversion redundancy  
3. **Implement RegistryValidationUtils** - Eliminate validation redundancy
4. **Implement RegistryErrorFormatter** - Eliminate error handling redundancy

#### Week 2: Generic Function Optimization
1. **Implement GenericStepFieldAccessor** - Replace 15+ individual functions
2. **Refactor compatibility functions** - Use generic accessor pattern
3. **Add comprehensive caching** - Optimize repeated access patterns
4. **Performance testing** - Ensure optimization doesn't degrade performance

#### Week 3: Registry Component Integration
1. **Update CoreStepRegistry** - Use shared utilities throughout
2. **Update WorkspaceStepRegistry** - Use shared utilities throughout
3. **Update DistributedRegistryManager** - Use shared utilities throughout
4. **Integration testing** - Verify all components work with shared utilities

#### Week 4: Validation and Optimization
1. **Comprehensive testing** - Test all redundancy elimination
2. **Performance benchmarking** - Measure improvement metrics
3. **Code quality assessment** - Verify quality score improvements
4. **Documentation updates** - Document optimization patterns

### Redundancy Prevention Patterns

#### 1. Shared Utility First Principle
- **Before implementing any registry logic**, check if shared utility exists
- **If similar logic exists elsewhere**, extract to shared utility
- **Always prefer composition** over code duplication

#### 2. Generic Function Pattern
- **For similar functions with different field names**, use generic implementation
- **Provide specific wrappers** for backward compatibility
- **Cache results** to avoid repeated computation

#### 3. Consistent Error Handling
- **Use RegistryErrorFormatter** for all error messages
- **Provide context and suggestions** in all error messages
- **Maintain consistent error format** across all components

#### 4. Validation Consolidation
- **Use RegistryValidationUtils** for all validation logic
- **Avoid duplicating validation rules** across components
- **Centralize validation error formatting**

## Core Components Design

### 1. Core Registry System
```python
    """
    Core step registry that provides the base set of step definitions.
    
    This registry contains the fundamental step implementations that
    are available to all workspaces by default.
    """
    
    def __init__(self, registry_path: str = "src/cursus/registry/step_names.py"):
        self.registry_path = Path(registry_path)
        self._step_definitions: Dict[str, StepDefinition] = {}
        self._load_core_registry()
    
    def _load_core_registry(self):
        """Load the core step registry from the central location."""
        try:
            # Import the existing STEP_NAMES registry
            spec = importlib.util.spec_from_file_location("step_names", self.registry_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load registry from {self.registry_path}")
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Convert existing STEP_NAMES to StepDefinition objects
            step_names = getattr(module, 'STEP_NAMES', {})
            for step_name, step_info in step_names.items():
                self._step_definitions[step_name] = StepDefinition(
                    name=step_name,
                    registry_type='core',
                    **step_info
                )
                
        except Exception as e:
            raise RegistryLoadError(f"Failed to load core registry: {e}")
    
    def get_step_definition(self, step_name: str) -> Optional[StepDefinition]:
        """Get a step definition from the core registry."""
        return self._step_definitions.get(step_name)
    
    def get_all_step_definitions(self) -> Dict[str, StepDefinition]:
        """Get all step definitions from the core registry."""
        return self._step_definitions.copy()
    
    def get_steps_by_type(self, sagemaker_step_type: str) -> List[StepDefinition]:
        """Get all steps of a specific SageMaker step type."""
        return [
            definition for definition in self._step_definitions.values()
            if definition.sagemaker_step_type == sagemaker_step_type
        ]
    
    def register_step(self, step_definition: StepDefinition) -> bool:
        """
        Register a new step in the core registry.
        
        Note: This is primarily for programmatic registration.
        Manual registration should still use the step_names.py file.
        """
        if step_definition.name in self._step_definitions:
            return False  # Step already exists
        
        step_definition.registry_type = 'core'
        self._step_definitions[step_definition.name] = step_definition
        return True
    
    def validate_registry(self) -> RegistryValidationResult:
        """Validate the core registry for consistency and completeness."""
        validator = RegistryValidator()
        return validator.validate_registry(self._step_definitions)

from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Dict, List, Any, Optional

class StepDefinition(BaseModel):
    """Enhanced step definition with registry metadata using Pydantic V2."""
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid',
        frozen=False,
        str_strip_whitespace=True
    )
    
    name: str = Field(..., min_length=1, description="Step name identifier")
    registry_type: str = Field(..., description="Registry type: 'core', 'workspace', 'override'")
    sagemaker_step_type: Optional[str] = Field(None, description="SageMaker step type")
    builder_step_name: Optional[str] = Field(None, description="Builder class name")
    description: Optional[str] = Field(None, description="Step description")
    framework: Optional[str] = Field(None, description="Framework used by step")
    job_types: List[str] = Field(default_factory=list, description="Supported job types")
    workspace_id: Optional[str] = Field(None, description="Workspace identifier for workspace registrations")
    override_source: Optional[str] = Field(None, description="Source of override for tracking")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @field_validator('registry_type')
    @classmethod
    def validate_registry_type(cls, v: str) -> str:
        """Validate registry type is one of allowed values."""
        allowed_types = {'core', 'workspace', 'override'}
        if v not in allowed_types:
            raise ValueError(f"registry_type must be one of {allowed_types}")
        return v
    
    @field_validator('name', 'builder_step_name')
    @classmethod
    def validate_identifiers(cls, v: Optional[str]) -> Optional[str]:
        """Validate identifier fields."""
        if v is not None and not v.strip():
            raise ValueError("Identifier cannot be empty or whitespace")
        return v
    
    def to_legacy_format(self) -> Dict[str, Any]:
        """Convert to legacy STEP_NAMES format for backward compatibility."""
        legacy_dict = {}
        
        if self.sagemaker_step_type:
            legacy_dict['sagemaker_step_type'] = self.sagemaker_step_type
        if self.builder_step_name:
            legacy_dict['builder_step_name'] = self.builder_step_name
        if self.description:
            legacy_dict['description'] = self.description
        if self.framework:
            legacy_dict['framework'] = self.framework
        if self.job_types:
            legacy_dict['job_types'] = self.job_types
        
        # Add any additional metadata
        legacy_dict.update(self.metadata)
        
        return legacy_dict
```

### 2. Workspace Registry System

Each workspace can have its own registry that extends or overrides the core registry.

```python
class WorkspaceStepRegistry:
    """
    Workspace-specific step registry that extends the core registry.
    
    Provides workspace isolation while maintaining inheritance from
    the core registry.
    """
    
    def __init__(self, workspace_path: str, core_registry: CoreStepRegistry):
        self.workspace_path = Path(workspace_path)
        self.workspace_id = self.workspace_path.name
        self.core_registry = core_registry
        self._workspace_definitions: Dict[str, StepDefinition] = {}
        self._overrides: Dict[str, StepDefinition] = {}
        self._load_workspace_registry()
    
    def _load_workspace_registry(self):
        """Load workspace-specific registry definitions."""
        registry_file = self.workspace_path / "src" / "cursus_dev" / "registry" / "workspace_registry.py"
        
        if not registry_file.exists():
            # No workspace registry file - that's okay
            return
        
        try:
            spec = importlib.util.spec_from_file_location("workspace_registry", registry_file)
            if spec is None or spec.loader is None:
                return
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Load workspace step definitions
            workspace_steps = getattr(module, 'WORKSPACE_STEPS', {})
            for step_name, step_info in workspace_steps.items():
                definition = StepDefinition(
                    name=step_name,
                    registry_type='workspace',
                    workspace_id=self.workspace_id,
                    **step_info
                )
                self._workspace_definitions[step_name] = definition
            
            # Load step overrides
            step_overrides = getattr(module, 'STEP_OVERRIDES', {})
            for step_name, step_info in step_overrides.items():
                definition = StepDefinition(
                    name=step_name,
                    registry_type='override',
                    workspace_id=self.workspace_id,
                    override_source='workspace',
                    **step_info
                )
                self._overrides[step_name] = definition
                
        except Exception as e:
            raise RegistryLoadError(f"Failed to load workspace registry: {e}")
    
    def get_step_definition(self, step_name: str) -> Optional[StepDefinition]:
        """
        Get a step definition with workspace precedence.
        
        Resolution order:
        1. Workspace overrides
        2. Workspace definitions
        3. Core registry
        """
        # Check workspace overrides first
        if step_name in self._overrides:
            return self._overrides[step_name]
        
        # Check workspace definitions
        if step_name in self._workspace_definitions:
            return self._workspace_definitions[step_name]
        
        # Fall back to core registry
        return self.core_registry.get_step_definition(step_name)
    
    def get_all_step_definitions(self) -> Dict[str, StepDefinition]:
        """Get all step definitions with workspace precedence applied."""
        # Start with core registry
        all_definitions = self.core_registry.get_all_step_definitions()
        
        # Add workspace definitions (may override core)
        all_definitions.update(self._workspace_definitions)
        
        # Apply workspace overrides
        all_definitions.update(self._overrides)
        
        return all_definitions
    
    def get_workspace_only_definitions(self) -> Dict[str, StepDefinition]:
        """Get only workspace-specific definitions (not inherited from core)."""
        workspace_only = {}
        workspace_only.update(self._workspace_definitions)
        workspace_only.update(self._overrides)
        return workspace_only
    
    def register_workspace_step(self, step_definition: StepDefinition) -> bool:
        """Register a new step in the workspace registry."""
        step_definition.registry_type = 'workspace'
        step_definition.workspace_id = self.workspace_id
        
        self._workspace_definitions[step_definition.name] = step_definition
        return True
    
    def override_core_step(self, step_name: str, step_definition: StepDefinition) -> bool:
        """Override a core step definition in this workspace."""
        if not self.core_registry.get_step_definition(step_name):
            return False  # Can't override non-existent core step
        
        step_definition.name = step_name
        step_definition.registry_type = 'override'
        step_definition.workspace_id = self.workspace_id
        step_definition.override_source = 'workspace'
        
        self._overrides[step_name] = step_definition
        return True
    
    def get_registry_summary(self) -> Dict[str, Any]:
        """Get summary information about this workspace registry."""
        core_count = len(self.core_registry.get_all_step_definitions())
        workspace_count = len(self._workspace_definitions)
        override_count = len(self._overrides)
        
        return {
            'workspace_id': self.workspace_id,
            'workspace_path': str(self.workspace_path),
            'core_steps': core_count,
            'workspace_steps': workspace_count,
            'overridden_steps': override_count,
            'total_available_steps': core_count + workspace_count,
            'workspace_step_names': list(self._workspace_definitions.keys()),
            'overridden_step_names': list(self._overrides.keys())
        }
```

### 3. Distributed Registry Manager

The central coordinator that manages multiple registries and provides a unified interface.

```python
class DistributedRegistryManager:
    """
    Central manager for the distributed registry system.
    
    Coordinates between core registry and multiple workspace registries,
    providing a unified interface for step discovery and resolution.
    """
    
    def __init__(self, 
                 core_registry_path: str = "src/cursus/registry/step_names.py",
                 workspaces_root: str = "developer_workspaces/developers"):
        self.core_registry = CoreStepRegistry(core_registry_path)
        self.workspaces_root = Path(workspaces_root)
        self._workspace_registries: Dict[str, WorkspaceStepRegistry] = {}
        self._registry_cache: Dict[str, Any] = {}
        self._discover_workspace_registries()
    
    def _discover_workspace_registries(self):
        """Discover and load all workspace registries."""
        if not self.workspaces_root.exists():
            return
        
        for workspace_dir in self.workspaces_root.iterdir():
            if workspace_dir.is_dir():
                try:
                    workspace_registry = WorkspaceStepRegistry(
                        str(workspace_dir), 
                        self.core_registry
                    )
                    self._workspace_registries[workspace_dir.name] = workspace_registry
                except Exception as e:
                    # Log error but continue with other workspaces
                    print(f"Warning: Failed to load registry for workspace {workspace_dir.name}: {e}")
    
    def get_step_definition(self, step_name: str, workspace_id: str = None) -> Optional[StepDefinition]:
        """
        Get a step definition with optional workspace context.
        
        Args:
            step_name: Name of the step to look up
            workspace_id: Optional workspace context for resolution
            
        Returns:
            Step definition or None if not found
        """
        if workspace_id and workspace_id in self._workspace_registries:
            # Use workspace-specific resolution
            return self._workspace_registries[workspace_id].get_step_definition(step_name)
        else:
            # Use core registry only
            return self.core_registry.get_step_definition(step_name)
    
    def get_all_step_definitions(self, workspace_id: str = None) -> Dict[str, StepDefinition]:
        """Get all step definitions with optional workspace context."""
        if workspace_id and workspace_id in self._workspace_registries:
            return self._workspace_registries[workspace_id].get_all_step_definitions()
        else:
            return self.core_registry.get_all_step_definitions()
    
    def get_steps_by_type(self, sagemaker_step_type: str, workspace_id: str = None) -> List[StepDefinition]:
        """Get steps by SageMaker type with optional workspace context."""
        all_definitions = self.get_all_step_definitions(workspace_id)
        return [
            definition for definition in all_definitions.values()
            if definition.sagemaker_step_type == sagemaker_step_type
        ]
    
    def discover_workspace_steps(self, workspace_id: str) -> Dict[str, StepDefinition]:
        """Discover steps that are unique to a specific workspace."""
        if workspace_id not in self._workspace_registries:
            return {}
        
        return self._workspace_registries[workspace_id].get_workspace_only_definitions()
    
    def get_step_conflicts(self) -> Dict[str, List[StepDefinition]]:
        """
        Identify steps that are defined in multiple workspaces.
        
        Returns:
            Dictionary mapping step names to list of conflicting definitions
        """
        conflicts = {}
        all_step_names = set()
        
        # Collect all step names from all registries
        for workspace_id, registry in self._workspace_registries.items():
            workspace_steps = registry.get_workspace_only_definitions()
            for step_name in workspace_steps.keys():
                if step_name in all_step_names:
                    if step_name not in conflicts:
                        conflicts[step_name] = []
                    conflicts[step_name].append(workspace_steps[step_name])
                else:
                    all_step_names.add(step_name)
        
        return conflicts
    
    def validate_distributed_registry(self) -> DistributedRegistryValidationResult:
        """Validate the entire distributed registry system."""
        result = DistributedRegistryValidationResult()
        
        # Validate core registry
        core_validation = self.core_registry.validate_registry()
        result.core_validation = core_validation
        
        # Validate each workspace registry
        for workspace_id, registry in self._workspace_registries.items():
            workspace_validation = self._validate_workspace_registry(registry)
            result.workspace_validations[workspace_id] = workspace_validation
        
        # Check for conflicts
        conflicts = self.get_step_conflicts()
        result.conflicts = conflicts
        
        # Determine overall status
        result.is_valid = (
            core_validation.is_valid and
            all(v.is_valid for v in result.workspace_validations.values()) and
            len(conflicts) == 0
        )
        
        return result
    
    def _validate_workspace_registry(self, registry: WorkspaceStepRegistry) -> RegistryValidationResult:
        """Validate a specific workspace registry."""
        validator = RegistryValidator()
        workspace_definitions = registry.get_workspace_only_definitions()
        return validator.validate_registry(workspace_definitions)
    
    def get_registry_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the distributed registry."""
        stats = {
            'core_registry': {
                'total_steps': len(self.core_registry.get_all_step_definitions()),
                'step_types': {}
            },
            'workspace_registries': {},
            'system_totals': {
                'total_workspaces': len(self._workspace_registries),
                'total_workspace_steps': 0,
                'total_overrides': 0
            }
        }
        
        # Core registry statistics
        core_definitions = self.core_registry.get_all_step_definitions()
        for definition in core_definitions.values():
            step_type = definition.sagemaker_step_type or 'Unknown'
            stats['core_registry']['step_types'][step_type] = \
                stats['core_registry']['step_types'].get(step_type, 0) + 1
        
        # Workspace registry statistics
        for workspace_id, registry in self._workspace_registries.items():
            summary = registry.get_registry_summary()
            stats['workspace_registries'][workspace_id] = summary
            stats['system_totals']['total_workspace_steps'] += summary['workspace_steps']
            stats['system_totals']['total_overrides'] += summary['overridden_steps']
        
        return stats
    
    def create_legacy_step_names_dict(self, workspace_id: str = None) -> Dict[str, Dict[str, Any]]:
        """
        Create a legacy STEP_NAMES dictionary for backward compatibility.
        
        Args:
            workspace_id: Optional workspace context
            
        Returns:
            Dictionary in the original STEP_NAMES format
        """
        all_definitions = self.get_all_step_definitions(workspace_id)
        legacy_dict = {}
        
        for step_name, definition in all_definitions.items():
            legacy_dict[step_name] = definition.to_legacy_format()
        
        return legacy_dict

class DistributedRegistryValidationResult(BaseModel):
    """Results of distributed registry validation using Pydantic V2."""
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid',
        frozen=False
    )
    
    is_valid: bool = Field(default=False, description="Overall validation status")
    core_validation: Optional['RegistryValidationResult'] = Field(None, description="Core registry validation result")
    workspace_validations: Dict[str, 'RegistryValidationResult'] = Field(
        default_factory=dict, 
        description="Workspace validation results by workspace ID"
    )
    conflicts: Dict[str, List[StepDefinition]] = Field(
        default_factory=dict, 
        description="Step conflicts between workspaces"
    )
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of validation results."""
        return {
            'overall_valid': self.is_valid,
            'core_valid': self.core_validation.is_valid if self.core_validation else False,
            'workspace_count': len(self.workspace_validations),
            'valid_workspaces': sum(1 for v in self.workspace_validations.values() if v.is_valid),
            'conflict_count': len(self.conflicts),
            'conflicted_steps': list(self.conflicts.keys())
        }
```

### 4. Registry Discovery and Resolution

```python
class RegistryDiscoveryService:
    """
    Service for discovering and resolving step implementations across
    the distributed registry system.
    """
    
    def __init__(self, registry_manager: DistributedRegistryManager):
        self.registry_manager = registry_manager
        self._discovery_cache: Dict[str, Any] = {}
    
    def discover_step_builders(self, workspace_id: str = None) -> Dict[str, Type]:
        """
        Discover all available step builder classes.
        
        Args:
            workspace_id: Optional workspace context for discovery
            
        Returns:
            Dictionary mapping step names to builder classes
        """
        cache_key = f"builders_{workspace_id or 'core'}"
        if cache_key in self._discovery_cache:
            return self._discovery_cache[cache_key]
        
        builders = {}
        step_definitions = self.registry_manager.get_all_step_definitions(workspace_id)
        
        for step_name, definition in step_definitions.items():
            try:
                builder_class = self._load_builder_class(definition, workspace_id)
                if builder_class:
                    builders[step_name] = builder_class
            except Exception as e:
                # Log error but continue with other builders
                print(f"Warning: Failed to load builder for {step_name}: {e}")
        
        self._discovery_cache[cache_key] = builders
        return builders
    
    def _load_builder_class(self, definition: StepDefinition, workspace_id: str = None) -> Optional[Type]:
        """Load a builder class based on step definition and context."""
        if definition.registry_type == 'workspace' and workspace_id:
            # Load from workspace
            return self._load_workspace_builder(definition, workspace_id)
        else:
            # Load from core system
            return self._load_core_builder(definition)
    
    def _load_workspace_builder(self, definition: StepDefinition, workspace_id: str) -> Optional[Type]:
        """Load a builder class from a workspace."""
        try:
            workspace_path = self.registry_manager.workspaces_root / workspace_id
            module_loader = WorkspaceModuleLoader(str(workspace_path))
            
            # Construct builder file path from definition
            builder_file = f"builder_{definition.name.lower()}_step.py"
            return module_loader.load_builder_class(builder_file)
            
        except Exception as e:
            print(f"Failed to load workspace builder {definition.name}: {e}")
            return None
    
    def _load_core_builder(self, definition: StepDefinition) -> Optional[Type]:
        """Load a builder class from the core system."""
        try:
            # Use existing RegistryStepDiscovery for core builders
            from ...validation.builders.registry_discovery import RegistryStepDiscovery
            return RegistryStepDiscovery.load_builder_class(definition.name)
        except Exception as e:
            print(f"Failed to load core builder {definition.name}: {e}")
            return None
    
    def resolve_step_components(self, step_name: str, workspace_id: str = None) -> StepComponentResolution:
        """
        Resolve all components (builder, config, spec, contract, script) for a step.
        
        Args:
            step_name: Name of the step to resolve
            workspace_id: Optional workspace context
            
        Returns:
            Resolution result with component locations and metadata
        """
        definition = self.registry_manager.get_step_definition(step_name, workspace_id)
        if not definition:
            return StepComponentResolution(step_name=step_name, found=False)
        
        resolution = StepComponentResolution(
            step_name=step_name,
            found=True,
            definition=definition,
            workspace_id=workspace_id
        )
        
        # Resolve each component type
        if definition.registry_type == 'workspace' and workspace_id:
            self._resolve_workspace_components(resolution, workspace_id)
        else:
            self._resolve_core_components(resolution)
        
        return resolution
    
    def _resolve_workspace_components(self, resolution: StepComponentResolution, workspace_id: str):
        """Resolve components from a workspace."""
        workspace_path = self.registry_manager.workspaces_root / workspace_id
        file_resolver = DeveloperWorkspaceFileResolver(str(workspace_path))
        
        step_name = resolution.step_name
        
        # Find each component type
        resolution.builder_path = file_resolver.find_builder_file(step_name)
        resolution.config_path = file_resolver.find_config_file(step_name)
        resolution.spec_path = file_resolver.find_spec_file(step_name)
        resolution.contract_path = file_resolver.find_contract_file(step_name)
        resolution.script_path = file_resolver.find_script_file(step_name)
    
    def _resolve_core_components(self, resolution: StepComponentResolution):
        """Resolve components from the core system."""
        # Use existing file resolution for core components
        from ...validation.alignment.file_resolver import FlexibleFileResolver
        
        base_directories = {
            'contracts': 'src/cursus/steps/contracts',
            'builders': 'src/cursus/steps/builders',
            'scripts': 'src/cursus/steps/scripts',
            'specs': 'src/cursus/steps/specs',
            'configs': 'src/cursus/steps/configs'
        }
        
        file_resolver = FlexibleFileResolver(base_directories)
        step_name = resolution.step_name
        
        resolution.builder_path = file_resolver.find_builder_file(step_name)
        resolution.config_path = file_resolver.find_config_file(step_name)
        resolution.spec_path = file_resolver.find_spec_file(step_name)
        resolution.contract_path = file_resolver.find_contract_file(step_name)
        # Scripts are in the scripts directory
        script_path = Path(base_directories['scripts']) / f"{step_name}.py"
        resolution.script_path = str(script_path) if script_path.exists() else None
    
    def clear_discovery_cache(self):
        """Clear the discovery cache to force re-discovery."""
        self._discovery_cache.clear()

class StepComponentResolution(BaseModel):
    """Result of step component resolution using Pydantic V2."""
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid',
        frozen=False,
        str_strip_whitespace=True
    )
    
    step_name: str = Field(..., min_length=1, description="Step name being resolved")
    found: bool = Field(..., description="Whether the step was found")
    definition: Optional[StepDefinition] = Field(None, description="Step definition if found")
    workspace_id: Optional[str] = Field(None, description="Workspace context for resolution")
    builder_path: Optional[str] = Field(None, description="Path to builder file")
    config_path: Optional[str] = Field(None, description="Path to config file")
    spec_path: Optional[str] = Field(None, description="Path to spec file")
    contract_path: Optional[str] = Field(None, description="Path to contract file")
    script_path: Optional[str] = Field(None, description="Path to script file")
    
    def get_available_components(self) -> List[str]:
        """Get list of available component types."""
        components = []
        if self.builder_path:
            components.append('builder')
        if self.config_path:
            components.append('config')
        if self.spec_path:
            components.append('spec')
        if self.contract_path:
            components.append('contract')
        if self.script_path:
            components.append('script')
        return components
    
    def is_complete(self) -> bool:
        """Check if all expected components are available."""
        # At minimum, we expect a builder
        return self.builder_path is not None
```

## Workspace Registry File Format

Each workspace can define its own registry using a standardized format:

```python
# developer_workspaces/developers/developer_1/src/cursus_dev/registry/workspace_registry.py

"""
Workspace registry for developer_1.

This file defines workspace-specific step implementations and
overrides for core step definitions.
"""

# Workspace-specific step definitions
WORKSPACE_STEPS = {
    "MyCustomProcessingStep": {
        "sagemaker_step_type": "Processing",
        "builder_step_name": "MyCustomProcessingStepBuilder",
        "description": "Custom processing step for developer_1",
        "framework": "pandas",
        "job_types": ["training", "validation"]
    },
    
    "ExperimentalTrainingStep": {
        "sagemaker_step_type": "Training",
        "builder_step_name": "ExperimentalTrainingStepBuilder",
        "description": "Experimental training approach",
        "framework": "pytorch",
        "job_types": ["training"]
    }
}

# Overrides for core step definitions
STEP_OVERRIDES = {
    "XGBoostTraining": {
        "sagemaker_step_type": "Training",
        "builder_step_name": "CustomXGBoostTrainingStepBuilder",
        "description": "Custom XGBoost implementation with enhanced features",
        "framework": "xgboost",
        "job_types": ["training", "validation"],
        "custom_features": ["early_stopping", "custom_metrics"]
    }
}

# Optional: Workspace metadata
WORKSPACE_METADATA = {
    "developer_id": "developer_1",
    "version": "1.0.0",
    "description": "Machine learning pipeline extensions",
    "dependencies": ["pandas>=1.3.0", "pytorch>=1.9.0"]
}
```

## Backward Compatibility Layer

To maintain backward compatibility with existing code, we provide a compatibility layer:

```python
class BackwardCompatibilityLayer:
    """
    Provides backward compatibility for existing code that uses
    the centralized STEP_NAMES registry.
    """
    
    def __init__(self, registry_manager: DistributedRegistryManager):
        self.registry_manager = registry_manager
        self._current_workspace_context: Optional[str] = None
    
    def get_step_names(self) -> Dict[str, Dict[str, Any]]:
        """
        Get STEP_NAMES in the original format.
        
        This method provides the same interface as the original
        STEP_NAMES dictionary for backward compatibility.
        """
        return self.registry_manager.create_legacy_step_names_dict(
            self._current_workspace_context
        )
    
    def set_workspace_context(self, workspace_id: str):
        """Set the workspace context for registry resolution."""
        self._current_workspace_context = workspace_id
    
    def clear_workspace_context(self):
        """Clear the workspace context to use core registry only."""
        self._current_workspace_context = None
    
    def get_steps_by_sagemaker_type(self, sagemaker_step_type: str) -> List[str]:
        """
        Get steps by SageMaker type in legacy format.
        
        This method provides the same interface as the original
        get_steps_by_sagemaker_type function for backward compatibility.
        """
        definitions = self.registry_manager.get_steps_by_type(
            sagemaker_step_type, 
            self._current_workspace_context
        )
        return [definition.name for definition in definitions]

# Global instance for backward compatibility
_global_registry_manager = None
_global_compatibility_layer = None

def get_global_registry_manager() -> DistributedRegistryManager:
    """Get the global registry manager instance."""
    global _global_registry_manager
    if _global_registry_manager is None:
        _global_registry_manager = DistributedRegistryManager()
    return _global_registry_manager

def get_step_names() -> Dict[str, Dict[str, Any]]:
    """
    Global function to get STEP_NAMES for backward compatibility.
    
    This function replaces the original STEP_NAMES dictionary import.
    """
    global _global_compatibility_layer
    if _global_compatibility_layer is None:
        _global_compatibility_layer = BackwardCompatibilityLayer(get_global_registry_manager())
    return _global_compatibility_layer.get_step_names()

def get_steps_by_sagemaker_type(sagemaker_step_type: str) -> List[str]:
    """
    Global function to get steps by type for backward compatibility.
    
    This function replaces the original get_steps_by_sagemaker_type function.
    """
    global _global_compatibility_layer
    if _global_compatibility_layer is None:
        _global_compatibility_layer = BackwardCompatibilityLayer(get_global_registry_manager())
    return _global_compatibility_layer.get_steps_by_sagemaker_type(sagemaker_step_type)

def set_workspace_context(workspace_id: str):
    """Set the global workspace context for registry resolution."""
    global _global_compatibility_layer
    if _global_compatibility_layer is None:
        _global_compatibility_layer = BackwardCompatibilityLayer(get_global_registry_manager())
    _global_compatibility_layer.set_workspace_context(workspace_id)

def clear_workspace_context():
    """Clear the global workspace context."""
    global _global_compatibility_layer
    if _global_compatibility_layer is not None:
        _global_compatibility_layer.clear_workspace_context()
```

## Critical Integration with Existing Registry System

### Complete Existing Registry Ecosystem Analysis

The current registry system in `src/cursus/registry/` consists of multiple interconnected components that the new distributed registry must fully support:

#### 1. Core Registry Files and Their Functions

**`step_names.py` - Central Step Registry (CRITICAL)**
- **STEP_NAMES Dictionary**: 17 core step definitions with config_class, builder_step_name, spec_type, sagemaker_step_type, description
- **Derived Registries**: CONFIG_STEP_REGISTRY, BUILDER_STEP_NAMES, SPEC_STEP_TYPES
- **Helper Functions**: 15+ functions for step name resolution, validation, and conversion
- **SageMaker Integration**: Functions for SageMaker step type classification and mapping

**`builder_registry.py` - Builder Discovery System**
- **StepBuilderRegistry Class**: Auto-discovery and registration of step builders
- **Global Registry Instance**: Singleton pattern for system-wide builder access
- **Legacy Alias Support**: Backward compatibility for renamed steps
- **Auto-Discovery**: Automatic scanning and registration of builder classes

**`hyperparameter_registry.py` - Hyperparameter Management**
- **HYPERPARAMETER_REGISTRY**: Registry for hyperparameter classes by model type
- **Model Type Mapping**: Functions to find hyperparameters by model type
- **Module Path Resolution**: Dynamic loading of hyperparameter classes

**`__init__.py` - Public API Exports**
- **Unified Interface**: Exports all registry functions and classes
- **Public API**: 25+ exported functions and classes for external use

#### 2. Existing STEP_NAMES Registry Structure

The current `STEP_NAMES` dictionary contains 17 core steps with this structure:
```python
STEP_NAMES = {
    "XGBoostTraining": {
        "config_class": "XGBoostTrainingConfig",
        "builder_step_name": "XGBoostTrainingStepBuilder", 
        "spec_type": "XGBoostTraining",
        "sagemaker_step_type": "Training",
        "description": "XGBoost model training step"
    },
    # ... 16 more step definitions
}
```

**Core Step Categories:**
- **Processing Steps**: TabularPreprocessing, RiskTableMapping, CurrencyConversion, ModelCalibration
- **Training Steps**: PyTorchTraining, XGBoostTraining, DummyTraining
- **Model Steps**: PyTorchModel, XGBoostModel
- **Evaluation Steps**: XGBoostModelEval
- **Deployment Steps**: Package, Registration, Payload
- **Data Steps**: CradleDataLoading
- **Transform Steps**: BatchTransform
- **Utility Steps**: HyperparameterPrep

#### 3. Critical Functions That Must Be Preserved

**Step Name Resolution Functions:**
```python
# Core lookup functions
get_config_class_name(step_name: str) -> str
get_builder_step_name(step_name: str) -> str  
get_spec_step_type(step_name: str) -> str
get_spec_step_type_with_job_type(step_name: str, job_type: str) -> str

# Reverse lookup functions
get_step_name_from_spec_type(spec_type: str) -> str
get_canonical_name_from_file_name(file_name: str) -> str

# Validation functions
validate_step_name(step_name: str) -> bool
validate_spec_type(spec_type: str) -> bool
validate_file_name(file_name: str) -> bool

# Information functions
get_all_step_names() -> List[str]
get_step_description(step_name: str) -> str
list_all_step_info() -> Dict[str, Dict[str, str]]
```

**SageMaker Integration Functions:**
```python
# SageMaker step type functions
get_sagemaker_step_type(step_name: str) -> str
get_steps_by_sagemaker_type(sagemaker_type: str) -> List[str]
get_all_sagemaker_step_types() -> List[str]
validate_sagemaker_step_type(sagemaker_type: str) -> bool
get_sagemaker_step_type_mapping() -> Dict[str, List[str]]
```

**Advanced File Name Resolution:**
The `get_canonical_name_from_file_name()` function uses sophisticated algorithms:
- **Strategy 1**: PascalCase conversion (xgboost_training → XGBoostTraining)
- **Strategy 2**: Job type suffix removal (handles _training, _validation suffixes)
- **Strategy 3**: Abbreviation expansion (xgb → XGBoost, pytorch → PyTorch)
- **Strategy 4**: Compound name handling (model_evaluation_xgb → XGBoostModelEval)
- **Strategy 5**: Fuzzy matching with similarity scoring

#### 4. Builder Registry Integration

**StepBuilderRegistry Features:**
- **Auto-Discovery**: Scans `src/cursus/steps/builders/` for builder classes
- **Legacy Aliases**: Maps old names to canonical names (MIMSPackaging → Package)
- **Job Type Handling**: Supports job type variants (_training, _validation)
- **Config-to-Builder Mapping**: Converts config class names to step types
- **Validation**: Registry consistency checking and statistics

**Global Registry Functions:**
```python
get_global_registry() -> StepBuilderRegistry
register_global_builder(step_type: str, builder_class: Type) -> None
list_global_step_types() -> List[str]
```

#### 5. Import Patterns That Must Continue Working

**Direct Registry Imports:**
```python
from cursus.steps.registry.step_names import STEP_NAMES, CONFIG_STEP_REGISTRY, BUILDER_STEP_NAMES, SPEC_STEP_TYPES
from cursus.steps.registry import get_global_registry, register_global_builder
from cursus.steps.registry.hyperparameter_registry import HYPERPARAMETER_REGISTRY
```

**Function Imports:**
```python
from cursus.steps.registry.step_names import (
    get_sagemaker_step_type, get_canonical_name_from_file_name,
    get_all_step_names, get_step_name_from_spec_type,
    get_config_class_name, get_builder_step_name, get_spec_step_type
)
```

**Module-Level Access:**
```python
from cursus.steps.registry import (
    StepBuilderRegistry, STEP_NAMES, CONFIG_STEP_REGISTRY,
    get_all_hyperparameter_classes, validate_hyperparameter_class
)
```

## How the New Registry Design Supports Existing step_names Functionalities

### Complete Function-by-Function Compatibility Mapping

The new distributed registry system provides **100% backward compatibility** with all existing `step_names.py` functionalities through a comprehensive compatibility layer. Here's how each existing function and data structure is supported:

#### 1. Core Data Structures Support

**Original STEP_NAMES Dictionary:**
```python
# Original: Direct dictionary access
STEP_NAMES = {...}  # 17 core step definitions

# New: Dynamic generation with workspace context
class EnhancedBackwardCompatibilityLayer:
    def get_step_names(self, workspace_id: str = None) -> Dict[str, Dict[str, Any]]:
        """Returns exact same STEP_NAMES format, optionally with workspace extensions."""
        return self.registry_manager.create_legacy_step_names_dict(workspace_id)

# Global replacement maintains exact same interface
STEP_NAMES = get_step_names()  # Dynamically generated, workspace-aware
```

**Derived Registry Structures:**
```python
# Original: Static dictionaries
CONFIG_STEP_REGISTRY = {info["config_class"]: step_name for step_name, info in STEP_NAMES.items()}
BUILDER_STEP_NAMES = {step_name: info["builder_step_name"] for step_name, info in STEP_NAMES.items()}
SPEC_STEP_TYPES = {step_name: info["spec_type"] for step_name, info in STEP_NAMES.items()}

# New: Dynamic generation with workspace context
class EnhancedBackwardCompatibilityLayer:
    def get_config_step_registry(self, workspace_id: str = None) -> Dict[str, str]:
        step_names = self.get_step_names(workspace_id)
        return {info["config_class"]: name for name, info in step_names.items()}
    
    def get_builder_step_names(self, workspace_id: str = None) -> Dict[str, str]:
        step_names = self.get_step_names(workspace_id)
        return {name: info["builder_step_name"] for name, info in step_names.items()}
    
    def get_spec_step_types(self, workspace_id: str = None) -> Dict[str, str]:
        step_names = self.get_step_names(workspace_id)
        return {name: info["spec_type"] for name, info in step_names.items()}

# Global replacements maintain exact same interface
CONFIG_STEP_REGISTRY = get_config_step_registry()
BUILDER_STEP_NAMES = get_builder_step_names()
SPEC_STEP_TYPES = get_spec_step_types()
```

#### 2. Helper Functions Support

**Step Name Resolution Functions:**
```python
# Original functions → New distributed implementations

# get_config_class_name(step_name: str) -> str
def get_config_class_name(step_name: str) -> str:
    """Get config class name for a step with workspace context."""
    compatibility_layer = get_enhanced_compatibility()
    step_names = compatibility_layer.get_step_names()
    if step_name not in step_names:
        raise ValueError(f"Unknown step name: {step_name}")
    return step_names[step_name]["config_class"]

# get_builder_step_name(step_name: str) -> str
def get_builder_step_name(step_name: str) -> str:
    """Get builder step class name for a step with workspace context."""
    compatibility_layer = get_enhanced_compatibility()
    step_names = compatibility_layer.get_step_names()
    if step_name not in step_names:
        raise ValueError(f"Unknown step name: {step_name}")
    return step_names[step_name]["builder_step_name"]

# get_spec_step_type(step_name: str) -> str
def get_spec_step_type(step_name: str) -> str:
    """Get step_type value for StepSpecification with workspace context."""
    compatibility_layer = get_enhanced_compatibility()
    step_names = compatibility_layer.get_step_names()
    if step_name not in step_names:
        raise ValueError(f"Unknown step name: {step_name}")
    return step_names[step_name]["spec_type"]

# get_spec_step_type_with_job_type(step_name: str, job_type: str) -> str
def get_spec_step_type_with_job_type(step_name: str, job_type: str = None) -> str:
    """Get step_type with optional job_type suffix, workspace-aware."""
    base_type = get_spec_step_type(step_name)
    if job_type:
        return f"{base_type}_{job_type.capitalize()}"
    return base_type
```

**Reverse Lookup Functions:**
```python
# get_step_name_from_spec_type(spec_type: str) -> str
def get_step_name_from_spec_type(spec_type: str) -> str:
    """Get canonical step name from spec_type with workspace context."""
    base_spec_type = spec_type.split('_')[0] if '_' in spec_type else spec_type
    
    compatibility_layer = get_enhanced_compatibility()
    step_names = compatibility_layer.get_step_names()
    reverse_mapping = {info["spec_type"]: step_name for step_name, info in step_names.items()}
    return reverse_mapping.get(base_spec_type, spec_type)

# get_canonical_name_from_file_name(file_name: str) -> str
def get_canonical_name_from_file_name(file_name: str) -> str:
    """
    Enhanced file name resolution with workspace context awareness.
    Maintains all 5 resolution strategies from original implementation.
    """
    compatibility_layer = get_enhanced_compatibility()
    step_names = compatibility_layer.get_step_names()
    
    # Use the same sophisticated algorithm as original, but with workspace-aware STEP_NAMES
    return _resolve_file_name_with_workspace_context(file_name, step_names)
```

**Validation Functions:**
```python
# validate_step_name(step_name: str) -> bool
def validate_step_name(step_name: str) -> bool:
    """Validate that a step name exists in the registry with workspace context."""
    compatibility_layer = get_enhanced_compatibility()
    step_names = compatibility_layer.get_step_names()
    return step_name in step_names

# validate_spec_type(spec_type: str) -> bool
def validate_spec_type(spec_type: str) -> bool:
    """Validate that a spec_type exists in the registry with workspace context."""
    base_spec_type = spec_type.split('_')[0] if '_' in spec_type else spec_type
    compatibility_layer = get_enhanced_compatibility()
    step_names = compatibility_layer.get_step_names()
    return base_spec_type in [info["spec_type"] for info in step_names.values()]

# validate_file_name(file_name: str) -> bool
def validate_file_name(file_name: str) -> bool:
    """Validate that a file name can be mapped to a canonical name with workspace context."""
    try:
        get_canonical_name_from_file_name(file_name)
        return True
    except ValueError:
        return False
```

**Information Functions:**
```python
# get_all_step_names() -> List[str]
def get_all_step_names() -> List[str]:
    """Get all canonical step names with workspace context."""
    compatibility_layer = get_enhanced_compatibility()
    step_names = compatibility_layer.get_step_names()
    return list(step_names.keys())

# get_step_description(step_name: str) -> str
def get_step_description(step_name: str) -> str:
    """Get description for a step with workspace context."""
    compatibility_layer = get_enhanced_compatibility()
    step_names = compatibility_layer.get_step_names()
    if step_name not in step_names:
        raise ValueError(f"Unknown step name: {step_name}")
    return step_names[step_name]["description"]

# list_all_step_info() -> Dict[str, Dict[str, str]]
def list_all_step_info() -> Dict[str, Dict[str, str]]:
    """Get complete step information for all registered steps with workspace context."""
    compatibility_layer = get_enhanced_compatibility()
    return compatibility_layer.get_step_names()
```

#### 3. SageMaker Integration Functions Support

**SageMaker Step Type Functions:**
```python
# get_sagemaker_step_type(step_name: str) -> str
def get_sagemaker_step_type(step_name: str) -> str:
    """Get SageMaker step type for a step with workspace context."""
    compatibility_layer = get_enhanced_compatibility()
    step_names = compatibility_layer.get_step_names()
    if step_name not in step_names:
        raise ValueError(f"Unknown step name: {step_name}")
    return step_names[step_name]["sagemaker_step_type"]

# get_steps_by_sagemaker_type(sagemaker_type: str) -> List[str]
def get_steps_by_sagemaker_type(sagemaker_type: str) -> List[str]:
    """Get all step names that create a specific SageMaker step type with workspace context."""
    compatibility_layer = get_enhanced_compatibility()
    step_names = compatibility_layer.get_step_names()
    return [
        step_name for step_name, info in step_names.items()
        if info["sagemaker_step_type"] == sagemaker_type
    ]

# get_all_sagemaker_step_types() -> List[str]
def get_all_sagemaker_step_types() -> List[str]:
    """Get all unique SageMaker step types with workspace context."""
    compatibility_layer = get_enhanced_compatibility()
    step_names = compatibility_layer.get_step_names()
    return list(set(info["sagemaker_step_type"] for info in step_names.values()))

# get_sagemaker_step_type_mapping() -> Dict[str, List[str]]
def get_sagemaker_step_type_mapping() -> Dict[str, List[str]]:
    """Get mapping of SageMaker step types to step names with workspace context."""
    compatibility_layer = get_enhanced_compatibility()
    step_names = compatibility_layer.get_step_names()
    mapping = {}
    for step_name, info in step_names.items():
        sagemaker_type = info["sagemaker_step_type"]
        if sagemaker_type not in mapping:
            mapping[sagemaker_type] = []
        mapping[sagemaker_type].append(step_name)
    return mapping
```

#### 4. Builder Registry Integration Support

**StepBuilderRegistry Compatibility:**
```python
class WorkspaceAwareStepBuilderRegistry(StepBuilderRegistry):
    """Enhanced StepBuilderRegistry with workspace awareness."""
    
    def __init__(self):
        super().__init__()
        # Integration with distributed registry
        self.distributed_registry = get_global_registry_manager()
        self.compatibility_layer = get_enhanced_compatibility()
    
    def get_builder_for_config(self, config: BasePipelineConfig, node_name: str = None) -> Type[StepBuilderBase]:
        """Enhanced config-to-builder resolution with workspace context."""
        # Extract workspace context from config
        workspace_id = getattr(config, 'workspace_id', None)
        
        # Set workspace context for resolution
        if workspace_id:
            self.compatibility_layer.set_workspace_context(workspace_id)
        
        try:
            # Use original logic but with workspace-aware registries
            return super().get_builder_for_config(config, node_name)
        finally:
            # Clean up workspace context
            if workspace_id:
                self.compatibility_layer.clear_workspace_context()
    
    def discover_builders(self) -> Dict[str, Type[StepBuilderBase]]:
        """Enhanced builder discovery with workspace support."""
        # Get current workspace context
        workspace_id = get_workspace_context()
        
        # Discover builders from both core and workspace registries
        core_builders = super().discover_builders()
        
        if workspace_id:
            # Add workspace-specific builders
            workspace_builders = self.distributed_registry.discover_workspace_steps(workspace_id)
            # Convert workspace step definitions to builder classes
            for step_name, definition in workspace_builders.items():
                try:
                    builder_class = self._load_workspace_builder(definition, workspace_id)
                    if builder_class:
                        core_builders[step_name] = builder_class
                except Exception as e:
                    registry_logger.warning(f"Failed to load workspace builder {step_name}: {e}")
        
        return core_builders

# Global registry replacement maintains exact same interface
def get_global_registry() -> WorkspaceAwareStepBuilderRegistry:
    """Get the global step builder registry instance with workspace awareness."""
    global _global_registry
    if _global_registry is None:
        _global_registry = WorkspaceAwareStepBuilderRegistry()
    return _global_registry
```

#### 5. Hyperparameter Registry Support

**HYPERPARAMETER_REGISTRY Compatibility:**
```python
class WorkspaceAwareHyperparameterRegistry:
    """Enhanced hyperparameter registry with workspace support."""
    
    def __init__(self):
        # Load core hyperparameter registry
        from .hyperparameter_registry import HYPERPARAMETER_REGISTRY
        self.core_registry = HYPERPARAMETER_REGISTRY.copy()
        self.workspace_registries = {}
    
    def get_hyperparameter_registry(self, workspace_id: str = None) -> Dict[str, Dict[str, Any]]:
        """Get hyperparameter registry with optional workspace extensions."""
        if workspace_id and workspace_id in self.workspace_registries:
            # Merge core and workspace registries
            merged_registry = self.core_registry.copy()
            merged_registry.update(self.workspace_registries[workspace_id])
            return merged_registry
        return self.core_registry

# All existing hyperparameter functions work with workspace context
def get_all_hyperparameter_classes(workspace_id: str = None) -> List[str]:
    """Get all registered hyperparameter class names with workspace context."""
    registry = WorkspaceAwareHyperparameterRegistry().get_hyperparameter_registry(workspace_id)
    return list(registry.keys())

def get_hyperparameter_class_by_model_type(model_type: str, workspace_id: str = None) -> Optional[str]:
    """Find hyperparameter class for model type with workspace context."""
    registry = WorkspaceAwareHyperparameterRegistry().get_hyperparameter_registry(workspace_id)
    for class_name, info in registry.items():
        if info["model_type"] == model_type:
            return class_name
    return None
```

#### 6. Module-Level Import Compatibility

**Exact Import Replacement:**
```python
# Original imports continue to work exactly the same
from cursus.steps.registry.step_names import (
    STEP_NAMES,                    # → Dynamic workspace-aware dictionary
    CONFIG_STEP_REGISTRY,          # → Dynamic workspace-aware mapping
    BUILDER_STEP_NAMES,            # → Dynamic workspace-aware mapping
    SPEC_STEP_TYPES,               # → Dynamic workspace-aware mapping
    get_config_class_name,         # → Workspace-aware function
    get_builder_step_name,         # → Workspace-aware function
    get_spec_step_type,            # → Workspace-aware function
    get_spec_step_type_with_job_type,  # → Workspace-aware function
    get_step_name_from_spec_type,  # → Workspace-aware function
    get_all_step_names,            # → Workspace-aware function
    validate_step_name,            # → Workspace-aware function
    validate_spec_type,            # → Workspace-aware function
    get_step_description,          # → Workspace-aware function
    list_all_step_info,            # → Workspace-aware function
    get_sagemaker_step_type,       # → Workspace-aware function
    get_steps_by_sagemaker_type,   # → Workspace-aware function
    get_all_sagemaker_step_types,  # → Workspace-aware function
    validate_sagemaker_step_type,  # → Workspace-aware function
    get_sagemaker_step_type_mapping,  # → Workspace-aware function
    get_canonical_name_from_file_name,  # → Enhanced workspace-aware function
    validate_file_name             # → Workspace-aware function
)

# Original builder registry imports
from cursus.steps.registry import (
    StepBuilderRegistry,           # → WorkspaceAwareStepBuilderRegistry
    get_global_registry,           # → Workspace-aware global registry
    register_global_builder,       # → Workspace-aware registration
    list_global_step_types         # → Workspace-aware step types
)

# Original hyperparameter registry imports
from cursus.steps.registry.hyperparameter_registry import (
    HYPERPARAMETER_REGISTRY,       # → Workspace-aware registry
    get_all_hyperparameter_classes,  # → Workspace-aware function
    get_hyperparameter_class_by_model_type,  # → Workspace-aware function
    get_module_path,               # → Workspace-aware function
    get_all_hyperparameter_info,   # → Workspace-aware function
    validate_hyperparameter_class  # → Workspace-aware function
)
```

#### 7. Advanced File Name Resolution Support

**Enhanced get_canonical_name_from_file_name Implementation:**
```python
def get_canonical_name_from_file_name(file_name: str) -> str:
    """
    Enhanced file name resolution with workspace context awareness.
    Maintains all 5 resolution strategies from original implementation.
    """
    if not file_name:
        raise ValueError("File name cannot be empty")
    
    # Get workspace-aware step names
    compatibility_layer = get_enhanced_compatibility()
    step_names = compatibility_layer.get_step_names()
    
    parts = file_name.split('_')
    job_type_suffixes = ['training', 'validation', 'testing', 'calibration']
    
    # Strategy 1: Try full name as PascalCase
    full_pascal = ''.join(word.capitalize() for word in parts)
    if full_pascal in step_names:
        return full_pascal
    
    # Strategy 2: Try without last part if it's a job type suffix
    if len(parts) > 1 and parts[-1] in job_type_suffixes:
        base_parts = parts[:-1]
        base_pascal = ''.join(word.capitalize() for word in base_parts)
        if base_pascal in step_names:
            return base_pascal
    
    # Strategy 3: Handle special abbreviations and patterns
    abbreviation_map = {
        'xgb': 'XGBoost',
        'xgboost': 'XGBoost',
        'pytorch': 'PyTorch',
        'mims': '',
        'tabular': 'Tabular',
        'preprocess': 'Preprocessing'
    }
    
    # Apply abbreviation expansion
    expanded_parts = []
    for part in parts:
        if part in abbreviation_map:
            expansion = abbreviation_map[part]
            if expansion:
                expanded_parts.append(expansion)
        else:
            expanded_parts.append(part.capitalize())
    
    # Try expanded version
    if expanded_parts:
        expanded_pascal = ''.join(expanded_parts)
        if expanded_pascal in step_names:
            return expanded_pascal
        
        # Try expanded version without job type suffix
        if len(expanded_parts) > 1 and parts[-1] in job_type_suffixes:
            expanded_base = ''.join(expanded_parts[:-1])
            if expanded_base in step_names:
                return expanded_base
    
    # Strategy 4: Handle compound names (like "model_evaluation_xgb")
    if len(parts) >= 3:
        combinations_to_try = [
            (parts[-1], parts[0], parts[1]),  # xgb, model, evaluation → XGBoost, Model, Eval
            (parts[0], parts[1], parts[-1]),  # model, evaluation, xgb
        ]
        
        for combo in combinations_to_try:
            expanded_combo = []
            for part in combo:
                if part in abbreviation_map:
                    expansion = abbreviation_map[part]
                    if expansion:
                        expanded_combo.append(expansion)
                else:
                    if part == 'evaluation':
                        expanded_combo.append('Eval')
                    else:
                        expanded_combo.append(part.capitalize())
            
            combo_pascal = ''.join(expanded_combo)
            if combo_pascal in step_names:
                return combo_pascal
    
    # Strategy 5: Fuzzy matching against registry entries
    best_match = None
    best_score = 0.0
    
    for canonical_name in step_names.keys():
        score = _calculate_name_similarity(file_name, canonical_name)
        if score > best_score and score >= 0.8:
            best_score = score
            best_match = canonical_name
    
    if best_match:
        return best_match
    
    # Enhanced error message with workspace context
    tried_variations = [
        full_pascal,
        ''.join(word.capitalize() for word in parts[:-1]) if len(parts) > 1 and parts[-1] in job_type_suffixes else None,
        ''.join(expanded_parts) if expanded_parts else None
    ]
    tried_variations = [v for v in tried_variations if v]
    
    workspace_context = get_workspace_context()
    context_info = f" (workspace: {workspace_context})" if workspace_context else " (core registry)"
    
    raise ValueError(
        f"Cannot map file name '{file_name}' to canonical name{context_info}. "
        f"Tried variations: {tried_variations}. "
        f"Available canonical names: {sorted(step_names.keys())}"
    )
```

#### 8. Registry Module __init__.py Support

**Complete Public API Preservation:**
```python
# Enhanced __init__.py that maintains exact same exports with workspace awareness

from .exceptions import RegistryError

# Enhanced builder registry with workspace support
from .enhanced_builder_registry import (
    WorkspaceAwareStepBuilderRegistry as StepBuilderRegistry,
    get_global_registry,
    register_global_builder,
    list_global_step_types
)

# Enhanced step names with workspace support
from .enhanced_step_names import (
    STEP_NAMES,                    # Dynamic workspace-aware
    CONFIG_STEP_REGISTRY,          # Dynamic workspace-aware
    BUILDER_STEP_NAMES,            # Dynamic workspace-aware
    SPEC_STEP_TYPES,               # Dynamic workspace-aware
    get_config_class_name,         # Workspace-aware
    get_builder_step_name,         # Workspace-aware
    get_spec_step_type,            # Workspace-aware
    get_spec_step_type_with_job_type,  # Workspace-aware
    get_step_name_from_spec_type,  # Workspace-aware
    get_all_step_names,            # Workspace-aware
    validate_step_name,            # Workspace-aware
    validate_spec_type,            # Workspace-aware
    get_step_description,          # Workspace-aware
    list_all_step_info,            # Workspace-aware
    get_sagemaker_step_type,       # Workspace-aware
    get_steps_by_sagemaker_type,   # Workspace-aware
    get_all_sagemaker_step_types,  # Workspace-aware
    validate_sagemaker_step_type,  # Workspace-aware
    get_sagemaker_step_type_mapping,  # Workspace-aware
    get_canonical_name_from_file_name,  # Enhanced workspace-aware
    validate_file_name             # Workspace-aware
)

# Enhanced hyperparameter registry with workspace support
from .enhanced_hyperparameter_registry import (
    HYPERPARAMETER_REGISTRY,       # Dynamic workspace-aware
    get_all_hyperparameter_classes,  # Workspace-aware
    get_hyperparameter_class_by_model_type,  # Workspace-aware
    get_module_path,               # Workspace-aware
    get_all_hyperparameter_info,   # Workspace-aware
    validate_hyperparameter_class  # Workspace-aware
)

# Exact same __all__ list - no changes to public API
__all__ = [
    # Exceptions
    "RegistryError",
    
    # Builder registry
    "StepBuilderRegistry",
    "get_global_registry",
    "register_global_builder", 
    "list_global_step_types",
    
    # Step names and registry
    "STEP_NAMES",
    "CONFIG_STEP_REGISTRY",
    "BUILDER_STEP_NAMES",
    "SPEC_STEP_TYPES",
    "get_config_class_name",
    "get_builder_step_name",
    "get_spec_step_type",
    "get_spec_step_type_with_job_type",
    "get_step_name_from_spec_type",
    "get_all_step_names",
    "validate_step_name",
    "validate_spec_type",
    "get_step_description",
    "list_all_step_info",
    
    # Hyperparameter registry
    "HYPERPARAMETER_REGISTRY",
    "get_all_hyperparameter_classes",
    "get_hyperparameter_class_by_model_type",
    "get_module_path",
    "get_all_hyperparameter_info",
    "validate_hyperparameter_class"
]
```

### Seamless Migration Strategy

#### Drop-in Replacement Implementation

The new system provides **seamless drop-in replacement** of the existing registry:

1. **File Structure Preservation**: All existing files remain in the same locations
2. **Import Path Preservation**: All existing import statements continue to work
3. **Function Signature Preservation**: All function signatures remain identical
4. **Data Structure Preservation**: All dictionaries and data structures maintain exact same format
5. **Behavior Preservation**: All functions behave identically in single-workspace scenarios

#### Workspace Context Detection

The system automatically detects workspace context through multiple mechanisms:

```python
def _detect_workspace_context() -> Optional[str]:
    """Automatically detect current workspace context."""
    
    # Method 1: Environment variable
    import os
    workspace_id = os.environ.get('CURSUS_WORKSPACE_ID')
    if workspace_id:
        return workspace_id
    
    # Method 2: Thread-local context
    try:
        return get_workspace_context()
    except:
        pass
    
    # Method 3: Config-based detection (when available)
    # This would be set by pipeline configurations that include workspace_id
    
    # Method 4: File system detection (detect if running from workspace directory)
    current_dir = os.getcwd()
    if 'developer_workspaces/developers/' in current_dir:
        # Extract workspace ID from path
        parts = current_dir.split('developer_workspaces/developers/')
        if len(parts) > 1:
            workspace_path = parts[1].split('/')[0]
            return workspace_path
    
    return None  # Default to core registry
```

#### Zero-Configuration Upgrade

The system is designed for **zero-configuration upgrade**:

1. **Automatic Fallback**: If no workspace context is detected, system behaves exactly like original
2. **Gradual Adoption**: Workspaces can be added incrementally without affecting existing functionality
3. **No Breaking Changes**: Existing code requires no modifications
4. **Performance Preservation**: Core registry performance is maintained or improved

### Enhanced Backward Compatibility Implementation

The distributed registry must provide **transparent replacement** of the current module:

```python
class EnhancedBackwardCompatibilityLayer(BackwardCompatibilityLayer):
    """Enhanced compatibility layer that maintains all derived registry structures."""
    
    def get_builder_step_names(self, workspace_id: str = None) -> Dict[str, str]:
        """Returns BUILDER_STEP_NAMES format with workspace context."""
        step_names = self.get_step_names(workspace_id)
        return {name: info["builder_step_name"] for name, info in step_names.items()}
    
    def get_config_step_registry(self, workspace_id: str = None) -> Dict[str, str]:
        """Returns CONFIG_STEP_REGISTRY format with workspace context."""
        step_names = self.get_step_names(workspace_id)
        return {info["config_class"]: name for name, info in step_names.items()}
    
    def get_spec_step_types(self, workspace_id: str = None) -> Dict[str, str]:
        """Returns SPEC_STEP_TYPES format with workspace context."""
        step_names = self.get_step_names(workspace_id)
        return {name: info["spec_type"] for name, info in step_names.items()}

# Global registry replacement - these must work exactly like the original imports
_global_enhanced_compatibility = None

def get_enhanced_compatibility() -> EnhancedBackwardCompatibilityLayer:
    """Get the enhanced compatibility layer instance."""
    global _global_enhanced_compatibility
    if _global_enhanced_compatibility is None:
        _global_enhanced_compatibility = EnhancedBackwardCompatibilityLayer(get_global_registry_manager())
    return _global_enhanced_compatibility

# These replace the original module-level variables
def get_step_names() -> Dict[str, Dict[str, Any]]:
    return get_enhanced_compatibility().get_step_names()

def get_builder_step_names() -> Dict[str, str]:
    return get_enhanced_compatibility().get_builder_step_names()

def get_config_step_registry() -> Dict[str, str]:
    return get_enhanced_compatibility().get_config_step_registry()

def get_spec_step_types() -> Dict[str, str]:
    return get_enhanced_compatibility().get_spec_step_types()

# Dynamic module-level variables that update with workspace context
STEP_NAMES = get_step_names()
BUILDER_STEP_NAMES = get_builder_step_names()
CONFIG_STEP_REGISTRY = get_config_step_registry()
SPEC_STEP_TYPES = get_spec_step_types()
```

### Base Class Integration Strategy

#### StepBuilderBase Enhancement
```python
class StepBuilderBase(ABC):
    @property
    def STEP_NAMES(self):
        """Lazy load step names with workspace context awareness."""
        if not hasattr(self, '_step_names'):
            # Detect workspace context from config or environment
            workspace_id = self._get_workspace_context()
            
            # Use distributed registry with workspace context
            compatibility_layer = get_enhanced_compatibility()
            if workspace_id:
                compatibility_layer.set_workspace_context(workspace_id)
            
            self._step_names = compatibility_layer.get_builder_step_names()
        return self._step_names
    
    def _get_workspace_context(self) -> Optional[str]:
        """Extract workspace context from config or environment."""
        # Check config for workspace_id
        if hasattr(self.config, 'workspace_id') and self.config.workspace_id:
            return self.config.workspace_id
        
        # Check environment variable
        import os
        workspace_id = os.environ.get('CURSUS_WORKSPACE_ID')
        if workspace_id:
            return workspace_id
        
        # Check thread-local context
        try:
            from contextvars import ContextVar
            _workspace_context = ContextVar('workspace_id', default=None)
            return _workspace_context.get()
        except ImportError:
            pass
        
        return None
```

#### BasePipelineConfig Enhancement
```python
class BasePipelineConfig(ABC):
    _STEP_NAMES: ClassVar[Dict[str, str]] = {}
    
    @classmethod
    def get_step_registry(cls) -> Dict[str, str]:
        """Lazy load step registry with workspace context."""
        if not cls._STEP_NAMES:
            # Get workspace context (similar to StepBuilderBase)
            workspace_id = cls._get_workspace_context()
            
            compatibility_layer = get_enhanced_compatibility()
            if workspace_id:
                compatibility_layer.set_workspace_context(workspace_id)
            
            cls._STEP_NAMES = compatibility_layer.get_config_step_registry()
        return cls._STEP_NAMES
    
    @classmethod
    def _get_workspace_context(cls) -> Optional[str]:
        """Extract workspace context for config classes."""
        # Similar implementation to StepBuilderBase
        import os
        return os.environ.get('CURSUS_WORKSPACE_ID')
```

### Workspace Context Management

#### Thread-Safe Context Management
```python
import contextvars
from typing import Optional, ContextManager

# Thread-local workspace context
_workspace_context: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar('workspace_id', default=None)

def set_workspace_context(workspace_id: str) -> None:
    """Set the current workspace context."""
    _workspace_context.set(workspace_id)
    
    # Also update the global compatibility layer
    compatibility_layer = get_enhanced_compatibility()
    compatibility_layer.set_workspace_context(workspace_id)

def get_workspace_context() -> Optional[str]:
    """Get the current workspace context."""
    return _workspace_context.get()

def clear_workspace_context() -> None:
    """Clear the current workspace context."""
    _workspace_context.set(None)
    
    # Also clear the global compatibility layer
    compatibility_layer = get_enhanced_compatibility()
    compatibility_layer.clear_workspace_context()

@contextmanager
def workspace_context(workspace_id: str) -> ContextManager[None]:
    """Context manager for temporary workspace context."""
    old_context = get_workspace_context()
    try:
        set_workspace_context(workspace_id)
        yield
    finally:
        if old_context:
            set_workspace_context(old_context)
        else:
            clear_workspace_context()
```

### Integration with WorkspaceComponentRegistry

The WorkspaceComponentRegistry and Distributed Registry System work together at different levels:

```python
class WorkspaceComponentRegistry:
    """Enhanced to work with distributed registry system."""
    
    def __init__(self, workspace_root: str):
        self.workspace_root = workspace_root
        # Integration point: Use distributed registry for step definitions
        self.registry_manager = get_global_registry_manager()
        self.compatibility_layer = get_enhanced_compatibility()
    
    def find_builder_class(self, step_name: str, developer_id: str = None) -> Optional[Type]:
        """Find builder class with distributed registry integration."""
        # 1. Check if step exists in distributed registry
        step_definition = self.registry_manager.get_step_definition(step_name, developer_id)
        if not step_definition:
            return None
        
        # 2. Set workspace context for component discovery
        if developer_id:
            with workspace_context(developer_id):
                return self._discover_builder_class(step_name, step_definition)
        else:
            return self._discover_builder_class(step_name, step_definition)
    
    def _discover_builder_class(self, step_name: str, step_definition: StepDefinition) -> Optional[Type]:
        """Discover builder class based on step definition."""
        if step_definition.registry_type == 'workspace':
            # Load from workspace using existing logic
            return self._load_workspace_builder(step_name, step_definition.workspace_id)
        else:
            # Load from core system
            return self._load_core_builder(step_name)
```

### Migration Implementation Strategy

#### Phase 1: Drop-in Replacement (Week 1)
1. **Replace step_names.py module** with distributed registry backend
2. **Maintain exact same API** for all existing imports
3. **Test backward compatibility** with existing validation system

#### Phase 2: Base Class Enhancement (Week 2)
1. **Update StepBuilderBase** to use distributed registry
2. **Update BasePipelineConfig** to use distributed registry
3. **Add workspace context detection** mechanisms

#### Phase 3: Context Management (Week 3)
1. **Implement thread-safe context management**
2. **Add environment variable support**
3. **Create context manager utilities**

#### Phase 4: Integration Testing (Week 4)
1. **Test with all 108 STEP_NAMES references**
2. **Validate workspace-aware functionality**
3. **Performance testing and optimization**

This integration approach ensures that the distributed registry system provides a **seamless upgrade path** while enabling powerful workspace-aware capabilities.

## Usage Examples

### Basic Registry Usage

```python
# Using the distributed registry manager
from cursus.registry.distributed import DistributedRegistryManager

# Initialize the registry manager
registry_manager = DistributedRegistryManager()

# Get step definition from core registry
step_def = registry_manager.get_step_definition("XGBoostTraining")
print(f"Step Type: {step_def.sagemaker_step_type}")

# Get step definition with workspace context
workspace_step_def = registry_manager.get_step_definition("MyCustomStep", workspace_id="developer_1")
print(f"Workspace Step: {workspace_step_def.workspace_id}")
```

### Workspace Registry Management

```python
# Working with workspace registries
from cursus.registry.distributed import WorkspaceStepRegistry, CoreStepRegistry

# Load core registry
core_registry = CoreStepRegistry()

# Load workspace registry
workspace_registry = WorkspaceStepRegistry("developer_workspaces/developers/developer_1", core_registry)

# Get workspace summary
summary = workspace_registry.get_registry_summary()
print(f"Workspace Steps: {summary['workspace_steps']}")
print(f"Overridden Steps: {summary['overridden_steps']}")

# Register a new workspace step
from cursus.registry.distributed import StepDefinition

new_step = StepDefinition(
    name="MyNewStep",
    registry_type="workspace",
    sagemaker_step_type="Processing",
    builder_step_name="MyNewStepBuilder",
    description="A new experimental step"
)

workspace_registry.register_workspace_step(new_step)
```

### Backward Compatibility Usage

```python
# Existing code continues to work unchanged
from cursus.registry.distributed import get_step_names, get_steps_by_sagemaker_type

# Get all step names (legacy format)
STEP_NAMES = get_step_names()
print(f"Available steps: {list(STEP_NAMES.keys())}")

# Get steps by type (legacy format)
training_steps = get_steps_by_sagemaker_type("Training")
print(f"Training steps: {training_steps}")

# Set workspace context for resolution
from cursus.registry.distributed import set_workspace_context

set_workspace_context("developer_1")
workspace_step_names = get_step_names()  # Now includes workspace steps
```

### Registry Discovery and Resolution

```python
# Using the discovery service
from cursus.registry.distributed import RegistryDiscoveryService, DistributedRegistryManager

registry_manager = DistributedRegistryManager()
discovery_service = RegistryDiscoveryService(registry_manager)

# Discover all builders in core registry
core_builders = discovery_service.discover_step_builders()
print(f"Core builders: {list(core_builders.keys())}")

# Discover builders with workspace context
workspace_builders = discovery_service.discover_step_builders(workspace_id="developer_1")
print(f"Workspace builders: {list(workspace_builders.keys())}")

# Resolve step components
resolution = discovery_service.resolve_step_components("MyCustomStep", workspace_id="developer_1")
print(f"Available components: {resolution.get_available_components()}")
print(f"Builder path: {resolution.builder_path}")
```

### Registry Validation

```python
# Validate the distributed registry
validation_result = registry_manager.validate_distributed_registry()

print(f"Overall valid: {validation_result.is_valid}")
print(f"Core valid: {validation_result.core_validation.is_valid}")

# Check for conflicts
conflicts = registry_manager.get_step_conflicts()
if conflicts:
    print("Step conflicts found:")
    for step_name, definitions in conflicts.items():
        workspaces = [d.workspace_id for d in definitions]
        print(f"  {step_name}: {workspaces}")

# Get registry statistics
stats = registry_manager.get_registry_statistics()
print(f"Total workspaces: {stats['system_totals']['total_workspaces']}")
print(f"Total workspace steps: {stats['system_totals']['total_workspace_steps']}")
```

### Enhanced Conflict Resolution Usage

```python
# Using the enhanced registry manager with conflict resolution
from cursus.registry.distributed import EnhancedDistributedRegistryManager, ResolutionContext

# Initialize enhanced registry manager
enhanced_registry = EnhancedDistributedRegistryManager()

# Example 1: Automatic resolution with workspace context
context = ResolutionContext(
    workspace_id="developer_1",
    resolution_mode="automatic"
)

resolution_result = enhanced_registry.resolve_step_with_context("XGBoostTraining", context)
print(f"Resolved: {resolution_result.resolved}")
print(f"Strategy: {resolution_result.resolution_strategy}")
print(f"Selected from: {resolution_result.selected_definition.namespace}")
print(f"Reason: {resolution_result.reason}")

# Example 2: Framework-specific resolution
context = ResolutionContext(
    workspace_id="developer_2",
    preferred_framework="pytorch",
    resolution_mode="automatic"
)

resolution_result = enhanced_registry.resolve_step_with_context("ModelTraining", context)
if resolution_result.resolved:
    print(f"Selected PyTorch implementation from: {resolution_result.selected_definition.namespace}")
else:
    print(f"Resolution failed: {resolution_result.reason}")

# Example 3: Environment-specific resolution
context = ResolutionContext(
    workspace_id="developer_3",
    environment_tags=["production", "gpu"],
    resolution_mode="automatic"
)

resolution_result = enhanced_registry.resolve_step_with_context("DataValidation", context)
print(f"Selected production-compatible implementation: {resolution_result.selected_definition.qualified_name}")

# Example 4: Detailed conflict analysis
conflicts = enhanced_registry.get_all_step_conflicts_detailed()
for step_name, conflict_analysis in conflicts.items():
    print(f"\nConflict Analysis for {step_name}:")
    print(f"  Type: {conflict_analysis.conflict_type}")
    print(f"  Involved namespaces: {conflict_analysis.get_conflict_summary()['involved_namespaces']}")
    print(f"  Frameworks: {conflict_analysis.get_conflict_summary()['frameworks']}")
    print(f"  Recommendations:")
    for rec in conflict_analysis.resolution_recommendations:
        print(f"    - {rec}")

# Example 5: High-level resolution interface
step_def = enhanced_registry.get_step_definition_with_resolution(
    step_name="XGBoostTraining",
    workspace_id="developer_1",
    preferred_framework="xgboost",
    environment_tags=["training", "gpu"]
)

if step_def:
    print(f"Resolved step: {step_def.name} from {step_def.workspace_id or 'core'}")
    print(f"Framework: {step_def.framework}")
    print(f"Description: {step_def.description}")
```

### Workspace Registry with Conflict Resolution Metadata

```python
# Enhanced workspace registry with conflict resolution metadata
# developer_workspaces/developers/developer_1/src/cursus_dev/registry/workspace_registry.py

"""
Enhanced workspace registry with conflict resolution metadata.
"""

WORKSPACE_STEPS = {
    "XGBoostTraining": {
        "sagemaker_step_type": "Training",
        "builder_step_name": "FinancialXGBoostTrainingStepBuilder",
        "description": "XGBoost training optimized for financial modeling",
        "framework": "xgboost",
        "job_types": ["training"],
        
        # Enhanced conflict resolution metadata
        "priority": 80,  # Higher priority than default
        "compatibility_tags": ["financial", "tabular_data", "feature_engineering"],
        "framework_version": "1.7.0",
        "environment_tags": ["training", "cpu", "memory_optimized"],
        "conflict_resolution_strategy": "framework_match"
    },
    
    "DataPreprocessing": {
        "sagemaker_step_type": "Processing",
        "builder_step_name": "PandasDataPreprocessingStepBuilder",
        "description": "Pandas-based data preprocessing for small to medium datasets",
        "framework": "pandas",
        "job_types": ["preprocessing"],
        
        # Conflict resolution metadata
        "priority": 90,
        "compatibility_tags": ["small_data", "tabular", "pandas"],
        "environment_tags": ["development", "cpu"],
        "conflict_resolution_strategy": "environment_match"
    }
}

STEP_OVERRIDES = {
    "ModelEvaluation": {
        "sagemaker_step_type": "Processing",
        "builder_step_name": "CustomModelEvaluationStepBuilder",
        "description": "Enhanced model evaluation with custom metrics",
        "framework": "scikit-learn",
        "job_types": ["evaluation"],
        
        # Override-specific metadata
        "priority": 70,  # High priority override
        "compatibility_tags": ["custom_metrics", "financial"],
        "environment_tags": ["production", "evaluation"],
        "conflict_resolution_strategy": "workspace_priority"
    }
}

# Workspace metadata with conflict resolution preferences
WORKSPACE_METADATA = {
    "developer_id": "developer_1",
    "version": "1.0.0",
    "description": "Financial ML pipeline extensions",
    "dependencies": ["pandas>=1.3.0", "xgboost>=1.7.0"],
    
    # Conflict resolution preferences
    "default_resolution_strategy": "framework_match",
    "preferred_frameworks": ["xgboost", "pandas", "scikit-learn"],
    "environment_preferences": ["training", "cpu", "memory_optimized"],
    "conflict_tolerance": "low"  # Fail fast on unresolvable conflicts
}
```

## Migration Strategy

### Phase 1: Infrastructure Setup (Week 1-2)

1. **Install Distributed Registry Components**
   - Implement core registry classes
   - Create workspace registry infrastructure
   - Set up discovery and resolution services

2. **Backward Compatibility Layer**
   - Implement compatibility functions
   - Test with existing code
   - Ensure no breaking changes

### Phase 2: Workspace Integration (Week 3-4)

1. **Workspace Registry Support**
   - Enable workspace registry loading
   - Implement registry inheritance
   - Add conflict detection and resolution

2. **Validation Integration**
   - Integrate with workspace-aware validation
   - Update discovery services
   - Test end-to-end workflows

### Phase 3: Developer Onboarding (Week 5-6)

1. **Developer Tools**
   - Create workspace registry templates
   - Implement registry management CLI
   - Add validation and diagnostics tools

2. **Documentation and Training**
   - Create developer guides
   - Provide migration examples
   - Conduct training sessions

### Phase 4: Production Deployment (Week 7-8)

1. **Performance Optimization**
   - Implement caching strategies
   - Optimize registry loading
   - Monitor performance metrics

2. **Monitoring and Maintenance**
   - Set up registry health monitoring
   - Implement automated validation
   - Create maintenance procedures

## Performance Considerations

### Registry Loading Optimization

1. **Lazy Loading**: Load workspace registries only when needed
2. **Caching**: Cache registry definitions and discovery results
3. **Parallel Loading**: Load multiple workspace registries concurrently
4. **Incremental Updates**: Only reload changed registries

### Memory Management

1. **Registry Cleanup**: Automatically clean up unused registry instances
2. **Weak References**: Use weak references for cached data
3. **Memory Monitoring**: Track registry memory usage
4. **Garbage Collection**: Implement registry garbage collection

### File System Optimization

1. **Directory Scanning**: Optimize workspace directory scanning
2. **File Watching**: Monitor registry files for changes
3. **Batch Operations**: Batch multiple registry operations
4. **Index Files**: Create registry index files for faster loading

## Security and Access Control

### Registry Security

1. **Path Validation**: Validate all registry file paths
2. **Code Execution**: Sandbox registry code execution
3. **Access Control**: Implement workspace access permissions
4. **Audit Logging**: Log all registry modifications

### Workspace Isolation

1. **Registry Separation**: Ensure complete registry isolation
2. **Namespace Protection**: Prevent namespace conflicts
3. **Resource Limits**: Limit workspace registry resources
4. **Validation Sandboxing**: Sandbox registry validation

## Error Handling and Diagnostics

### Registry Error Handling

```python
class RegistryError(Exception):
    """Base exception for registry errors."""
    pass

class RegistryLoadError(RegistryError):
    """Error loading registry from file."""
    pass

class RegistryValidationError(RegistryError):
    """Error validating registry contents."""
    pass

class RegistryConflictError(RegistryError):
    """Error due to registry conflicts."""
    pass

class WorkspaceRegistryError(RegistryError):
    """Error specific to workspace registries."""
    pass
```

### Diagnostic Tools

```python
class RegistryDiagnostics:
    """Diagnostic tools for registry troubleshooting."""
    
    def __init__(self, registry_manager: DistributedRegistryManager):
        self.registry_manager = registry_manager
    
    def diagnose_step_resolution(self, step_name: str, workspace_id: str = None) -> Dict[str, Any]:
        """Diagnose step resolution issues."""
        diagnosis = {
            'step_name': step_name,
            'workspace_id': workspace_id,
            'resolution_path': [],
            'issues': [],
            'recommendations': []
        }
        
        # Check core registry
        core_def = self.registry_manager.core_registry.get_step_definition(step_name)
        if core_def:
            diagnosis['resolution_path'].append(f"Found in core registry: {core_def.registry_type}")
        else:
            diagnosis['issues'].append("Step not found in core registry")
        
        # Check workspace registry if specified
        if workspace_id:
            if workspace_id in self.registry_manager._workspace_registries:
                workspace_registry = self.registry_manager._workspace_registries[workspace_id]
                workspace_def = workspace_registry.get_step_definition(step_name)
                if workspace_def:
                    diagnosis['resolution_path'].append(f"Found in workspace {workspace_id}: {workspace_def.registry_type}")
                else:
                    diagnosis['issues'].append(f"Step not found in workspace {workspace_id}")
            else:
                diagnosis['issues'].append(f"Workspace {workspace_id} not found")
        
        # Generate recommendations
        if not diagnosis['resolution_path']:
            diagnosis['recommendations'].append("Check step name spelling and registry definitions")
        
        return diagnosis
    
    def diagnose_registry_health(self) -> Dict[str, Any]:
        """Diagnose overall registry system health."""
        health = {
            'overall_status': 'HEALTHY',
            'core_registry': {'status': 'UNKNOWN', 'issues': []},
            'workspace_registries': {},
            'system_issues': [],
            'recommendations': []
        }
        
        # Check core registry
        try:
            core_validation = self.registry_manager.core_registry.validate_registry()
            health['core_registry']['status'] = 'HEALTHY' if core_validation.is_valid else 'UNHEALTHY'
            if not core_validation.is_valid:
                health['core_registry']['issues'] = core_validation.issues
                health['overall_status'] = 'UNHEALTHY'
        except Exception as e:
            health['core_registry']['status'] = 'ERROR'
            health['core_registry']['issues'] = [str(e)]
            health['overall_status'] = 'UNHEALTHY'
        
        # Check workspace registries
        for workspace_id, registry in self.registry_manager._workspace_registries.items():
            try:
                workspace_validation = self.registry_manager._validate_workspace_registry(registry)
                workspace_health = {
                    'status': 'HEALTHY' if workspace_validation.is_valid else 'UNHEALTHY',
                    'issues': workspace_validation.issues if not workspace_validation.is_valid else []
                }
                health['workspace_registries'][workspace_id] = workspace_health
                
                if not workspace_validation.is_valid:
                    health['overall_status'] = 'UNHEALTHY'
                    
            except Exception as e:
                health['workspace_registries'][workspace_id] = {
                    'status': 'ERROR',
                    'issues': [str(e)]
                }
                health['overall_status'] = 'UNHEALTHY'
        
        # Check for conflicts
        conflicts = self.registry_manager.get_step_conflicts()
        if conflicts:
            health['system_issues'].append(f"Registry conflicts found: {list(conflicts.keys())}")
            health['overall_status'] = 'UNHEALTHY'
        
        # Generate recommendations
        if health['overall_status'] == 'UNHEALTHY':
            health['recommendations'].append("Review and fix registry validation issues")
            if conflicts:
                health['recommendations'].append("Resolve step name conflicts between workspaces")
        
        return health
```

## Future Enhancements

### Advanced Registry Features

1. **Registry Versioning**: Support for registry schema versions
2. **Registry Migration**: Tools for migrating between registry versions
3. **Registry Synchronization**: Sync registries across environments
4. **Registry Backup**: Automated registry backup and restore

### Integration Enhancements

1. **CI/CD Integration**: Registry validation in CI/CD pipelines
2. **IDE Integration**: Registry support in development environments
3. **Monitoring Integration**: Registry metrics in monitoring systems
4. **Documentation Generation**: Auto-generate registry documentation

### Scalability Improvements

1. **Distributed Caching**: Distributed registry caching
2. **Load Balancing**: Load balance registry requests
3. **Horizontal Scaling**: Scale registry services horizontally
4. **Performance Analytics**: Detailed registry performance analytics

## Conclusion

The Distributed Registry System design provides a comprehensive solution for transforming the centralized registry into a distributed, multi-developer-friendly architecture. The design maintains full backward compatibility while enabling powerful new capabilities for workspace isolation and collaborative development.

**Key Benefits:**

1. **Developer Independence**: Each workspace can maintain its own registry without conflicts
2. **Inheritance Model**: Workspaces inherit from core registry and can extend or override
3. **Backward Compatibility**: Existing code continues to work without modification
4. **Conflict Resolution**: Clear precedence rules and conflict detection
5. **Performance**: Efficient caching and lazy loading mechanisms
6. **Extensibility**: Designed for future enhancements and scaling

**Implementation Readiness:**

- **Well-Defined Architecture**: Clear component boundaries and interfaces
- **Incremental Migration**: Can be implemented and adopted gradually
- **Comprehensive Testing**: Extensive validation and diagnostic capabilities
- **Production Ready**: Designed for reliability and performance

This distributed registry system enables the Multi-Developer Workspace Management System by providing the registry infrastructure necessary to support isolated development while maintaining system coherence and backward compatibility. The design supports the collaborative development model while preserving the reliability and performance characteristics of the existing system.

## Related Documents

This design document is part of a comprehensive multi-developer system architecture. For complete understanding, refer to these related documents:

### Core System Architecture
- **[Workspace-Aware Multi-Developer Management Design](workspace_aware_multi_developer_management_design.md)** - Master design document that defines the overall architecture and core principles for supporting multiple developer workspaces
- **[Workspace-Aware Validation System Design](workspace_aware_validation_system_design.md)** - Validation framework extensions that work with the distributed registry system to provide comprehensive workspace validation

### Implementation Analysis
- **[Multi-Developer Validation System Analysis](../4_analysis/multi_developer_validation_system_analysis.md)** - Detailed analysis of current system capabilities and implementation feasibility for multi-developer support

### Integration Points
The Distributed Registry System integrates with:
- **Workspace-Aware Validation**: Registry discovery services provide component resolution for validation frameworks
- **Multi-Developer Management**: Registry federation enables workspace isolation while maintaining shared core functionality
- **Validation Analysis**: Registry statistics and health monitoring support the validation system's workspace assessment capabilities

These documents together form a complete architectural specification for transforming Cursus into a collaborative multi-developer platform while maintaining the high standards of code quality and system reliability.

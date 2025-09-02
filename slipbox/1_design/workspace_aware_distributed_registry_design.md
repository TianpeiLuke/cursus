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

# Workspace-Aware Registry System Design

## Overview

**Note**: This design has been updated to reflect the consolidated workspace architecture outlined in the [Workspace-Aware System Refactoring Migration Plan](../2_project_planning/2025-09-02_workspace_aware_system_refactoring_migration_plan.md). All workspace functionality is now centralized within `src/cursus/` for proper packaging compliance.

This document outlines the design for transforming the current centralized registry system in `src/cursus/steps/registry` into a workspace-aware registry architecture that supports multiple developer workspaces. The new system enables each workspace to maintain its own registry while inheriting from a common core registry, providing isolation and extensibility without breaking existing functionality.

## Problem Statement

The current registry system (`src/cursus/steps/registry/step_names.py`) is centralized and requires all step implementations to be registered in a single location. This creates several challenges for multi-developer environments:

1. **Central Registry Bottleneck**: All developers must modify the same central registry file
2. **Merge Conflicts**: Multiple developers editing the same registry leads to frequent conflicts
3. **Workspace Isolation**: No way to register workspace-specific implementations without affecting others
4. **Development Friction**: Developers cannot experiment with new steps without central registry changes
5. **Deployment Complexity**: All step implementations must be deployed together

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
- Core registry in `src/cursus/steps/registry/` provides the shared foundation
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
Distributed Registry System
├── Core Registry/
│   ├── CoreStepRegistry (base registry)
│   ├── StepRegistrationManager
│   └── RegistryValidator
├── Workspace Registries/
│   ├── WorkspaceStepRegistry
│   ├── WorkspaceRegistryLoader
│   └── WorkspaceRegistryValidator
├── Registry Federation/
│   ├── DistributedRegistryManager
│   ├── RegistryInheritanceResolver
│   └── RegistryConflictResolver
└── Discovery and Resolution/
    ├── RegistryDiscoveryService
    ├── ComponentResolver
    └── RegistryCache
```

## Core Components Design

### 1. Core Registry System

The core registry maintains the base set of step definitions that all workspaces inherit from.

```python
class CoreStepRegistry:
    """
    Core step registry that provides the base set of step definitions.
    
    This registry contains the fundamental step implementations that
    are available to all workspaces by default.
    """
    
    def __init__(self, registry_path: str = "src/cursus/steps/registry/step_names.py"):
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
                 core_registry_path: str = "src/cursus/steps/registry/step_names.py",
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

## Critical Integration with Existing System

### STEP_NAMES Integration Analysis

Based on comprehensive analysis of the existing system, there are **232+ references** to step_names throughout the codebase, requiring careful integration:

#### 1. Base Class Dependencies (CRITICAL)
- **`StepBuilderBase.STEP_NAMES` property**: Uses `BUILDER_STEP_NAMES` from registry with lazy loading
- **`BasePipelineConfig._STEP_NAMES`**: Uses `CONFIG_STEP_REGISTRY` for step mapping with lazy loading

#### 2. System Components Using Registry
- **Validation System (108+ references)**: Alignment validation, builder testing, config analysis, dependency validation
- **Core System Components**: Pipeline assembler, compiler validation, workspace registry
- **Step Specifications (40+ files)**: All step specs import registry functions for step_type assignment

#### 3. Derived Registry Structures (MUST MAINTAIN)
The current `step_names.py` creates several derived registries that are critical for backward compatibility:
```python
CONFIG_STEP_REGISTRY = {info["config_class"]: step_name for step_name, info in STEP_NAMES.items()}
BUILDER_STEP_NAMES = {step_name: info["builder_step_name"] for step_name, info in STEP_NAMES.items()}
SPEC_STEP_TYPES = {step_name: info["spec_type"] for step_name, info in STEP_NAMES.items()}
```

#### 4. Import Patterns That Must Continue Working
```python
# Direct registry imports
from cursus.steps.registry.step_names import STEP_NAMES, CONFIG_STEP_REGISTRY, BUILDER_STEP_NAMES

# Function imports  
from cursus.steps.registry.step_names import (
    get_sagemaker_step_type, get_canonical_name_from_file_name,
    get_all_step_names, get_step_name_from_spec_type
)
```

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

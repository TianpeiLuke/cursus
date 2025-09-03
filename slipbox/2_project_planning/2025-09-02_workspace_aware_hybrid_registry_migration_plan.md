---
tags:
  - project
  - planning
  - migration
  - registry_system
  - hybrid_architecture
  - multi_developer
  - implementation_plan
keywords:
  - hybrid registry migration
  - centralized to distributed registry
  - multi-developer registry system
  - workspace-aware registry
  - registry isolation
  - backward compatibility
  - step name collision resolution
topics:
  - registry system migration
  - hybrid architecture implementation
  - multi-developer collaboration
  - workspace isolation strategy
  - registry federation design
language: python
date of note: 2025-09-02
---

# Comprehensive Hybrid Registry Migration Plan

## Executive Summary

This document provides a comprehensive plan to migrate our current single centralized registry system (`src/cursus/steps/registry`) into a hybrid system supporting multiple developers. Each developer will own their local registry in `developer_workspace/developers/developer_k` while maintaining access to the central shared registry in `cursus/steps/registry`. This enables isolated local development with customized steps while preserving shared common functionality.

## Current System Analysis

### Existing Centralized Registry Architecture

**Core Registry Location**: `src/cursus/steps/registry/`
- **`step_names.py`**: Central STEP_NAMES dictionary with 17 core step definitions
- **`builder_registry.py`**: StepBuilderRegistry with auto-discovery and global instance
- **`hyperparameter_registry.py`**: HYPERPARAMETER_REGISTRY for model-specific hyperparameters
- **`__init__.py`**: Public API exports (25+ functions and classes)

**Current STEP_NAMES Structure**:
```python
STEP_NAMES = {
    "XGBoostTraining": {
        "config_class": "XGBoostTrainingConfig",
        "builder_step_name": "XGBoostTrainingStepBuilder", 
        "spec_type": "XGBoostTraining",
        "sagemaker_step_type": "Training",
        "description": "XGBoost model training step"
    },
    # ... 16 more core step definitions
}
```

**Derived Registries**:
- `CONFIG_STEP_REGISTRY`: Config class → step name mapping
- `BUILDER_STEP_NAMES`: Step name → builder class mapping
- `SPEC_STEP_TYPES`: Step name → spec type mapping

**Critical Dependencies**:
- **232+ references** across codebase to step_names functions
- **Base class integration**: StepBuilderBase and BasePipelineConfig use lazy loading
- **Validation system**: 108+ references in alignment and builder testing
- **Core system**: Pipeline assembler, compiler, and workspace components

### Developer Workspace Structure

**Current Developer Workspace**: `developer_workspaces/developers/developer_k/`
```
developer_k/
├── src/cursus_dev/steps/
│   ├── builders/     # Custom step builders
│   ├── configs/      # Custom configurations
│   ├── contracts/    # Custom script contracts
│   ├── scripts/      # Custom processing scripts
│   └── specs/        # Custom specifications
├── test/             # Developer tests
├── validation_reports/
└── README.md
```

**Missing Components**:
- **Local registry system**: No workspace-specific registry
- **Registry integration**: No connection to central registry
- **Step name management**: No local step name definitions
- **Conflict resolution**: No mechanism for handling step name collisions

## Problem Statement

### Critical Challenges

1. **Registry Bottleneck**: All developers must modify the same central `step_names.py`
2. **Merge Conflicts**: Multiple developers editing central registry creates conflicts
3. **Development Friction**: Cannot experiment with new steps without central changes
4. **Step Name Collisions**: Multiple developers may use same step names with different implementations
5. **Workspace Isolation**: No way to register workspace-specific implementations
6. **Deployment Complexity**: All step implementations must be deployed together

### Step Name Collision Scenarios

**Scenario 1: Independent Development**
- Developer A: "XGBoostTraining" for financial modeling
- Developer B: "XGBoostTraining" for image classification
- **Collision**: Same name, different implementations

**Scenario 2: Framework Variations**
- Developer A: "ModelTraining" using PyTorch
- Developer B: "ModelTraining" using TensorFlow
- Developer C: "ModelTraining" using XGBoost
- **Collision**: Multiple valid implementations for same concept

**Scenario 3: Environment-Specific Adaptations**
- Development: "DataValidation" with relaxed constraints
- Production: "DataValidation" with strict validation rules
- **Collision**: Same name, different validation logic

## Hybrid Registry System Design

### Core Architectural Principles

**Principle 1: Workspace Isolation**
- Everything within a developer's workspace stays in that workspace
- Local registry modifications don't affect other workspaces
- Complete development environment isolation

**Principle 2: Shared Core Foundation**
- Central registry in `src/cursus/steps/registry/` provides shared foundation
- All workspaces inherit from the same core registry baseline
- Common step definitions maintained centrally

**Principle 3: Intelligent Conflict Resolution**
- Smart resolution strategies for step name collisions
- Context-aware step selection based on workspace, framework, environment
- Clear precedence rules and fallback mechanisms

### Hybrid Architecture Overview

```
Hybrid Registry System
├── Central Shared Registry/
│   ├── src/cursus/steps/registry/step_names.py (ENHANCED)
│   ├── CoreStepRegistry (maintains 17 core steps)
│   └── SharedRegistryManager
├── Local Developer Registries/
│   ├── developer_k/src/cursus_dev/registry/
│   │   ├── workspace_registry.py (NEW)
│   │   └── local_step_names.py (NEW)
│   └── LocalRegistryManager
├── Hybrid Registry Federation/
│   ├── HybridRegistryManager (CENTRAL COORDINATOR)
│   ├── RegistryInheritanceResolver
│   └── IntelligentConflictResolver
└── Compatibility Layer/
    ├── BackwardCompatibilityAdapter
    ├── ContextAwareRegistryProxy
    └── LegacyAPIPreservation
```

## Detailed Migration Strategy

### Phase 1: Foundation Infrastructure (Weeks 1-2)

#### 1.1 Create Hybrid Registry Core Components

**Deliverable**: Core hybrid registry infrastructure

**Implementation Tasks**:

1. **Create Enhanced Step Definition Model**
```python
# File: src/cursus/steps/registry/hybrid/models.py
from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Dict, List, Any, Optional

class HybridStepDefinition(BaseModel):
    """Enhanced step definition with workspace and conflict resolution metadata."""
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid',
        frozen=False,
        str_strip_whitespace=True
    )
    
    # Core step information
    name: str = Field(..., min_length=1, description="Step name identifier")
    config_class: str = Field(..., min_length=1, description="Configuration class name")
    builder_step_name: str = Field(..., min_length=1, description="Builder class name")
    spec_type: str = Field(..., min_length=1, description="Specification type")
    sagemaker_step_type: str = Field(..., min_length=1, description="SageMaker step type")
    description: str = Field(..., min_length=1, description="Step description")
    
    # Registry metadata
    registry_type: str = Field(..., description="Registry type: 'core', 'workspace', 'override'")
    workspace_id: Optional[str] = Field(None, description="Workspace identifier for workspace registrations")
    
    # Conflict resolution metadata
    priority: int = Field(default=100, description="Resolution priority (lower = higher priority)")
    framework: Optional[str] = Field(None, description="Framework used by step")
    environment_tags: List[str] = Field(default_factory=list, description="Environment compatibility tags")
    compatibility_tags: List[str] = Field(default_factory=list, description="Compatibility tags for smart resolution")
    
    # Conflict resolution strategy
    conflict_resolution_strategy: str = Field(
        default="workspace_priority", 
        description="Strategy: 'workspace_priority', 'framework_match', 'environment_match'"
    )
    
    @field_validator('registry_type')
    @classmethod
    def validate_registry_type(cls, v: str) -> str:
        allowed_types = {'core', 'workspace', 'override'}
        if v not in allowed_types:
            raise ValueError(f"registry_type must be one of {allowed_types}")
        return v
    
    def to_legacy_format(self) -> Dict[str, Any]:
        """Convert to legacy STEP_NAMES format for backward compatibility."""
        return {
            'config_class': self.config_class,
            'builder_step_name': self.builder_step_name,
            'spec_type': self.spec_type,
            'sagemaker_step_type': self.sagemaker_step_type,
            'description': self.description
        }
```

2. **Create Core Registry Manager**
```python
# File: src/cursus/steps/registry/hybrid/core_registry.py
class CoreStepRegistry:
    """Enhanced core registry that maintains the shared foundation."""
    
    def __init__(self, registry_path: str = "src/cursus/steps/registry/step_names.py"):
        self.registry_path = Path(registry_path)
        self._step_definitions: Dict[str, HybridStepDefinition] = {}
        self._load_core_registry()
    
    def _load_core_registry(self):
        """Load and convert existing STEP_NAMES to hybrid format."""
        try:
            # Import existing STEP_NAMES
            spec = importlib.util.spec_from_file_location("step_names", self.registry_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            step_names = getattr(module, 'STEP_NAMES', {})
            for step_name, step_info in step_names.items():
                self._step_definitions[step_name] = HybridStepDefinition(
                    name=step_name,
                    registry_type='core',
                    **step_info
                )
        except Exception as e:
            raise RegistryLoadError(f"Failed to load core registry: {e}")
    
    def get_step_definition(self, step_name: str) -> Optional[HybridStepDefinition]:
        """Get step definition from core registry."""
        return self._step_definitions.get(step_name)
    
    def get_all_step_definitions(self) -> Dict[str, HybridStepDefinition]:
        """Get all core step definitions."""
        return self._step_definitions.copy()
```

3. **Create Local Registry Manager**
```python
# File: src/cursus/steps/registry/hybrid/local_registry.py
class LocalStepRegistry:
    """Local workspace registry that extends core registry."""
    
    def __init__(self, workspace_path: str, core_registry: CoreStepRegistry):
        self.workspace_path = Path(workspace_path)
        self.workspace_id = self.workspace_path.name
        self.core_registry = core_registry
        self._local_definitions: Dict[str, HybridStepDefinition] = {}
        self._overrides: Dict[str, HybridStepDefinition] = {}
        self._load_local_registry()
    
    def _load_local_registry(self):
        """Load workspace-specific registry definitions."""
        registry_file = self.workspace_path / "src" / "cursus_dev" / "registry" / "workspace_registry.py"
        
        if not registry_file.exists():
            return  # No local registry - that's okay
        
        try:
            spec = importlib.util.spec_from_file_location("workspace_registry", registry_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Load local step definitions
            local_steps = getattr(module, 'LOCAL_STEPS', {})
            for step_name, step_info in local_steps.items():
                definition = HybridStepDefinition(
                    name=step_name,
                    registry_type='workspace',
                    workspace_id=self.workspace_id,
                    **step_info
                )
                self._local_definitions[step_name] = definition
            
            # Load step overrides
            step_overrides = getattr(module, 'STEP_OVERRIDES', {})
            for step_name, step_info in step_overrides.items():
                definition = HybridStepDefinition(
                    name=step_name,
                    registry_type='override',
                    workspace_id=self.workspace_id,
                    **step_info
                )
                self._overrides[step_name] = definition
                
        except Exception as e:
            raise RegistryLoadError(f"Failed to load local registry: {e}")
    
    def get_step_definition(self, step_name: str) -> Optional[HybridStepDefinition]:
        """Get step definition with workspace precedence."""
        # Resolution order: overrides → local → core
        if step_name in self._overrides:
            return self._overrides[step_name]
        if step_name in self._local_definitions:
            return self._local_definitions[step_name]
        return self.core_registry.get_step_definition(step_name)
    
    def get_all_step_definitions(self) -> Dict[str, HybridStepDefinition]:
        """Get all step definitions with workspace precedence applied."""
        all_definitions = self.core_registry.get_all_step_definitions()
        all_definitions.update(self._local_definitions)
        all_definitions.update(self._overrides)
        return all_definitions
    
    def get_local_only_definitions(self) -> Dict[str, HybridStepDefinition]:
        """Get only workspace-specific definitions."""
        local_only = {}
        local_only.update(self._local_definitions)
        local_only.update(self._overrides)
        return local_only
```

#### 1.2 Create Intelligent Conflict Resolution System

**Deliverable**: Smart conflict resolution for step name collisions

**Implementation Tasks**:

1. **Resolution Context Model**
```python
# File: src/cursus/steps/registry/hybrid/resolution.py
class ResolutionContext(BaseModel):
    """Context for intelligent step resolution."""
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid',
        frozen=False
    )
    
    workspace_id: Optional[str] = Field(None, description="Current workspace context")
    preferred_framework: Optional[str] = Field(None, description="Preferred framework")
    environment_tags: List[str] = Field(default_factory=list, description="Environment tags")
    resolution_mode: str = Field(default="automatic", description="Resolution mode")
    
    @field_validator('resolution_mode')
    @classmethod
    def validate_resolution_mode(cls, v: str) -> str:
        allowed_modes = {'automatic', 'interactive', 'strict'}
        if v not in allowed_modes:
            raise ValueError(f"resolution_mode must be one of {allowed_modes}")
        return v

class StepResolutionResult(BaseModel):
    """Result of step conflict resolution."""
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid',
        frozen=False
    )
    
    step_name: str = Field(..., description="Step name being resolved")
    resolved: bool = Field(..., description="Whether resolution was successful")
    selected_definition: Optional[HybridStepDefinition] = Field(None, description="Selected step definition")
    resolution_strategy: Optional[str] = Field(None, description="Strategy used")
    reason: str = Field(default="", description="Resolution explanation")
    conflicting_definitions: List[HybridStepDefinition] = Field(default_factory=list, description="All conflicts found")
```

2. **Intelligent Conflict Resolver**
```python
class IntelligentConflictResolver:
    """Advanced conflict resolution engine for step name collisions."""
    
    def __init__(self, registry_manager: 'HybridRegistryManager'):
        self.registry_manager = registry_manager
        self._resolution_cache: Dict[str, StepResolutionResult] = {}
    
    def resolve_step_conflict(self, step_name: str, context: ResolutionContext) -> StepResolutionResult:
        """Resolve step name conflicts using intelligent strategies."""
        # Get all definitions for this step name
        conflicting_definitions = self._get_conflicting_definitions(step_name)
        
        if not conflicting_definitions:
            return StepResolutionResult(
                step_name=step_name,
                resolved=False,
                reason="Step not found in any registry"
            )
        
        if len(conflicting_definitions) == 1:
            return StepResolutionResult(
                step_name=step_name,
                resolved=True,
                selected_definition=conflicting_definitions[0],
                resolution_strategy="no_conflict"
            )
        
        # Multiple definitions - resolve conflict
        return self._resolve_multiple_definitions(step_name, conflicting_definitions, context)
    
    def _resolve_multiple_definitions(self, step_name: str, definitions: List[HybridStepDefinition], 
                                    context: ResolutionContext) -> StepResolutionResult:
        """Resolve conflicts using multiple strategies."""
        
        # Strategy 1: Workspace Priority Resolution
        if context.workspace_id:
            for definition in definitions:
                if definition.workspace_id == context.workspace_id:
                    return StepResolutionResult(
                        step_name=step_name,
                        resolved=True,
                        selected_definition=definition,
                        resolution_strategy="workspace_priority",
                        reason=f"Selected from current workspace: {context.workspace_id}"
                    )
        
        # Strategy 2: Framework Compatibility Resolution
        if context.preferred_framework:
            compatible_definitions = [
                d for d in definitions 
                if d.framework == context.preferred_framework
            ]
            if len(compatible_definitions) == 1:
                return StepResolutionResult(
                    step_name=step_name,
                    resolved=True,
                    selected_definition=compatible_definitions[0],
                    resolution_strategy="framework_match",
                    reason=f"Selected based on framework: {context.preferred_framework}"
                )
        
        # Strategy 3: Environment Compatibility Resolution
        if context.environment_tags:
            compatible_definitions = []
            for definition in definitions:
                if definition.environment_tags:
                    if set(definition.environment_tags).intersection(set(context.environment_tags)):
                        compatible_definitions.append(definition)
                else:
                    compatible_definitions.append(definition)  # No tags = compatible with all
            
            if len(compatible_definitions) == 1:
                return StepResolutionResult(
                    step_name=step_name,
                    resolved=True,
                    selected_definition=compatible_definitions[0],
                    resolution_strategy="environment_match",
                    reason=f"Selected based on environment: {context.environment_tags}"
                )
        
        # Strategy 4: Priority-Based Resolution
        definitions.sort(key=lambda d: d.priority)
        return StepResolutionResult(
            step_name=step_name,
            resolved=True,
            selected_definition=definitions[0],
            resolution_strategy="priority_based",
            reason=f"Selected based on priority: {definitions[0].priority}"
        )
```

#### 1.3 Create Hybrid Registry Manager

**Deliverable**: Central coordinator for hybrid registry system

**Implementation Tasks**:

```python
# File: src/cursus/steps/registry/hybrid/manager.py
class HybridRegistryManager:
    """Central coordinator for hybrid registry system."""
    
    def __init__(self, 
                 core_registry_path: str = "src/cursus/steps/registry/step_names.py",
                 workspaces_root: str = "developer_workspaces/developers"):
        self.core_registry = CoreStepRegistry(core_registry_path)
        self.workspaces_root = Path(workspaces_root)
        self._local_registries: Dict[str, LocalStepRegistry] = {}
        self.conflict_resolver = IntelligentConflictResolver(self)
        self._registry_cache: Dict[str, Any] = {}
        self._discover_local_registries()
    
    def _discover_local_registries(self):
        """Discover and load all local workspace registries."""
        if not self.workspaces_root.exists():
            return
        
        for workspace_dir in self.workspaces_root.iterdir():
            if workspace_dir.is_dir():
                try:
                    local_registry = LocalStepRegistry(str(workspace_dir), self.core_registry)
                    self._local_registries[workspace_dir.name] = local_registry
                except Exception as e:
                    print(f"Warning: Failed to load registry for workspace {workspace_dir.name}: {e}")
    
    def get_step_definition_with_resolution(self, 
                                          step_name: str, 
                                          workspace_id: str = None,
                                          preferred_framework: str = None,
                                          environment_tags: List[str] = None) -> Optional[HybridStepDefinition]:
        """Get step definition with intelligent conflict resolution."""
        context = ResolutionContext(
            workspace_id=workspace_id,
            preferred_framework=preferred_framework,
            environment_tags=environment_tags or [],
            resolution_mode="automatic"
        )
        
        result = self.conflict_resolver.resolve_step_conflict(step_name, context)
        return result.selected_definition if result.resolved else None
    
    def get_step_definition(self, step_name: str, workspace_id: str = None) -> Optional[HybridStepDefinition]:
        """Get step definition with optional workspace context (simplified interface)."""
        if workspace_id and workspace_id in self._local_registries:
            return self._local_registries[workspace_id].get_step_definition(step_name)
        else:
            return self.core_registry.get_step_definition(step_name)
    
    def get_all_step_definitions(self, workspace_id: str = None) -> Dict[str, HybridStepDefinition]:
        """Get all step definitions with optional workspace context."""
        if workspace_id and workspace_id in self._local_registries:
            return self._local_registries[workspace_id].get_all_step_definitions()
        else:
            return self.core_registry.get_all_step_definitions()
    
    def create_legacy_step_names_dict(self, workspace_id: str = None) -> Dict[str, Dict[str, Any]]:
        """Create legacy STEP_NAMES dictionary for backward compatibility."""
        all_definitions = self.get_all_step_definitions(workspace_id)
        legacy_dict = {}
        
        for step_name, definition in all_definitions.items():
            legacy_dict[step_name] = definition.to_legacy_format()
        
        return legacy_dict
    
    def get_step_conflicts(self) -> Dict[str, List[HybridStepDefinition]]:
        """Identify steps defined in multiple registries."""
        conflicts = {}
        all_step_names = set()
        
        # Collect all step names from all registries
        for workspace_id, registry in self._local_registries.items():
            local_steps = registry.get_local_only_definitions()
            for step_name in local_steps.keys():
                if step_name in all_step_names:
                    if step_name not in conflicts:
                        conflicts[step_name] = []
                    conflicts[step_name].append(local_steps[step_name])
                else:
                    all_step_names.add(step_name)
        
        return conflicts
```

### Phase 2: Backward Compatibility Layer (Weeks 3-4)

#### 2.1 Create Enhanced Compatibility Layer

**Deliverable**: Seamless backward compatibility for all existing code

**Implementation Tasks**:

1. **Enhanced Backward Compatibility Adapter**
```python
# File: src/cursus/steps/registry/hybrid/compatibility.py
class EnhancedBackwardCompatibilityLayer:
    """Comprehensive compatibility layer maintaining all derived registry structures."""
    
    def __init__(self, registry_manager: HybridRegistryManager):
        self.registry_manager = registry_manager
        self._current_workspace_context: Optional[str] = None
    
    def get_step_names(self, workspace_id: str = None) -> Dict[str, Dict[str, Any]]:
        """Get STEP_NAMES in original format with workspace context."""
        effective_workspace = workspace_id or self._current_workspace_context
        return self.registry_manager.create_legacy_step_names_dict(effective_workspace)
    
    def get_builder_step_names(self, workspace_id: str = None) -> Dict[str, str]:
        """Get BUILDER_STEP_NAMES format with workspace context."""
        step_names = self.get_step_names(workspace_id)
        return {name: info["builder_step_name"] for name, info in step_names.items()}
    
    def get_config_step_registry(self, workspace_id: str = None) -> Dict[str, str]:
        """Get CONFIG_STEP_REGISTRY format with workspace context."""
        step_names = self.get_step_names(workspace_id)
        return {info["config_class"]: name for name, info in step_names.items()}
    
    def get_spec_step_types(self, workspace_id: str = None) -> Dict[str, str]:
        """Get SPEC_STEP_TYPES format with workspace context."""
        step_names = self.get_step_names(workspace_id)
        return {name: info["spec_type"] for name, info in step_names.items()}
    
    def set_workspace_context(self, workspace_id: str):
        """Set workspace context for registry resolution."""
        self._current_workspace_context = workspace_id
    
    def clear_workspace_context(self):
        """Clear workspace context."""
        self._current_workspace_context = None
```

2. **Context-Aware Registry Proxy**
```python
# File: src/cursus/steps/registry/hybrid/proxy.py
import contextvars
from typing import Optional, ContextManager
from contextlib import contextmanager

# Thread-local workspace context
_workspace_context: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar('workspace_id', default=None)

def set_workspace_context(workspace_id: str) -> None:
    """Set current workspace context."""
    _workspace_context.set(workspace_id)
    
    # Update global compatibility layer
    compatibility_layer = get_enhanced_compatibility()
    compatibility_layer.set_workspace_context(workspace_id)

def get_workspace_context() -> Optional[str]:
    """Get current workspace context."""
    return _workspace_context.get()

def clear_workspace_context() -> None:
    """Clear current workspace context."""
    _workspace_context.set(None)
    
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

# Global instances for backward compatibility
_global_registry_manager = None
_global_compatibility_layer = None

def get_global_registry_manager() -> HybridRegistryManager:
    """Get global registry manager instance."""
    global _global_registry_manager
    if _global_registry_manager is None:
        _global_registry_manager = HybridRegistryManager()
    return _global_registry_manager

def get_enhanced_compatibility() -> EnhancedBackwardCompatibilityLayer:
    """Get enhanced compatibility layer instance."""
    global _global_compatibility_layer
    if _global_compatibility_layer is None:
        _global_compatibility_layer = EnhancedBackwardCompatibilityLayer(get_global_registry_manager())
    return _global_compatibility_layer
```

#### 2.2 Create Drop-in Replacement Functions

**Deliverable**: Exact API preservation for all existing functions

**Implementation Tasks**:

```python
# File: src/cursus/steps/registry/hybrid/legacy_api.py
"""Drop-in replacement functions that maintain exact API compatibility."""

def get_step_names() -> Dict[str, Dict[str, Any]]:
    """Global function to get STEP_NAMES for backward compatibility."""
    return get_enhanced_compatibility().get_step_names()

def get_builder_step_names() -> Dict[str, str]:
    """Global function to get BUILDER_STEP_NAMES for backward compatibility."""
    return get_enhanced_compatibility().get_builder_step_names()

def get_config_step_registry() -> Dict[str, str]:
    """Global function to get CONFIG_STEP_REGISTRY for backward compatibility."""
    return get_enhanced_compatibility().get_config_step_registry()

def get_spec_step_types() -> Dict[str, str]:
    """Global function to get SPEC_STEP_TYPES for backward compatibility."""
    return get_enhanced_compatibility().get_spec_step_types()

# All existing helper functions with workspace awareness
def get_config_class_name(step_name: str) -> str:
    """Get config class name with workspace context."""
    step_names = get_step_names()
    if step_name not in step_names:
        raise ValueError(f"Unknown step name: {step_name}")
    return step_names[step_name]["config_class"]

def get_builder_step_name(step_name: str) -> str:
    """Get builder step class name with workspace context."""
    step_names = get_step_names()
    if step_name not in step_names:
        raise ValueError(f"Unknown step name: {step_name}")
    return step_names[step_name]["builder_step_name"]

def get_spec_step_type(step_name: str) -> str:
    """Get step_type value for StepSpecification with workspace context."""
    step_names = get_step_names()
    if step_name not in step_names:
        raise ValueError(f"Unknown step name: {step_name}")
    return step_names[step_name]["spec_type"]

def get_spec_step_type_with_job_type(step_name: str, job_type: str = None) -> str:
    """Get step_type with optional job_type suffix, workspace-aware."""
    base_type = get_spec_step_type(step_name)
    if job_type:
        return f"{base_type}_{job_type.capitalize()}"
    return base_type

def get_step_name_from_spec_type(spec_type: str) -> str:
    """Get canonical step name from spec_type with workspace context."""
    base_spec_type = spec_type.split('_')[0] if '_' in spec_type else spec_type
    step_names = get_step_names()
    reverse_mapping = {info["spec_type"]: step_name for step_name, info in step_names.items()}
    return reverse_mapping.get(base_spec_type, spec_type)

def get_all_step_names() -> List[str]:
    """Get all canonical step names with workspace context."""
    step_names = get_step_names()
    return list(step_names.keys())

def validate_step_name(step_name: str) -> bool:
    """Validate step name exists with workspace context."""
    step_names = get_step_names()
    return step_name in step_names

def validate_spec_type(spec_type: str) -> bool:
    """Validate spec_type exists with workspace context."""
    base_spec_type = spec_type.split('_')[0] if '_' in spec_type else spec_type
    step_names = get_step_names()
    return base_spec_type in [info["spec_type"] for info in step_names.values()]

def get_step_description(step_name: str) -> str:
    """Get step description with workspace context."""
    step_names = get_step_names()
    if step_name not in step_names:
        raise ValueError(f"Unknown step name: {step_name}")
    return step_names[step_name]["description"]

def list_all_step_info() -> Dict[str, Dict[str, str]]:
    """Get complete step information with workspace context."""
    return get_step_names()

def get_sagemaker_step_type(step_name: str) -> str:
    """Get SageMaker step type with workspace context."""
    step_names = get_step_names()
    if step_name not in step_names:
        raise ValueError(f"Unknown step name: {step_name}")
    return step_names[step_name]["sagemaker_step_type"]

def get_steps_by_sagemaker_type(sagemaker_type: str) -> List[str]:
    """Get steps by SageMaker type with workspace context."""
    step_names = get_step_names()
    return [
        step_name for step_name, info in step_names.items()
        if info["sagemaker_step_type"] == sagemaker_type
    ]

def get_all_sagemaker_step_types() -> List[str]:
    """Get all SageMaker step types with workspace context."""
    step_names = get_step_names()
    return list(set(info["sagemaker_step_type"] for info in step_names.values()))

def validate_sagemaker_step_type(sagemaker_type: str) -> bool:
    """Validate SageMaker step type with workspace context."""
    valid_types = {"Processing", "Training", "Transform", "CreateModel", "RegisterModel", "Base", "Utility"}
    return sagemaker_type in valid_types

def get_sagemaker_step_type_mapping() -> Dict[str, List[str]]:
    """Get SageMaker step type mapping with workspace context."""
    step_names = get_step_names()
    mapping = {}
    for step_name, info in step_names.items():
        sagemaker_type = info["sagemaker_step_type"]
        if sagemaker_type not in mapping:
            mapping[sagemaker_type] = []
        mapping[sagemaker_type].append(step_name)
    return mapping

def get_canonical_name_from_file_name(file_name: str) -> str:
    """Enhanced file name resolution with workspace context awareness."""
    if not file_name:
        raise ValueError("File name cannot be empty")
    
    # Get workspace-aware step names
    step_names = get_step_names()
    
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

def _calculate_name_similarity(file_name: str, canonical_name: str) -> float:
    """Calculate similarity score between file name and canonical name."""
    file_lower = file_name.lower().replace('_', '')
    canonical_lower = canonical_name.lower()
    
    if file_lower == canonical_lower:
        return 1.0
    
    if file_lower in canonical_lower:
        return 0.9
    
    file_parts = file_name.lower().split('_')
    matches = sum(1 for part in file_parts if part in canonical_lower)
    
    if matches == len(file_parts):
        return 0.85
    elif matches >= len(file_parts) * 0.8:
        return 0.8
    else:
        return matches / len(file_parts) * 0.7

def validate_file_name(file_name: str) -> bool:
    """Validate file name can be mapped with workspace context."""
    try:
        get_canonical_name_from_file_name(file_name)
        return True
    except ValueError:
        return False

# Dynamic module-level variables that update with workspace context
STEP_NAMES = get_step_names()
BUILDER_STEP_NAMES = get_builder_step_names()
CONFIG_STEP_REGISTRY = get_config_step_registry()
SPEC_STEP_TYPES = get_spec_step_types()
```

### Phase 3: Local Registry Infrastructure (Weeks 5-6)

#### 3.1 Create Local Registry Templates

**Deliverable**: Standardized local registry format for developer workspaces

**Implementation Tasks**:

1. **Workspace Registry Template**
```python
# Template: developer_workspaces/developers/developer_k/src/cursus_dev/registry/workspace_registry.py
"""
Local registry for developer_k workspace.

This file defines workspace-specific step implementations and
overrides for core step definitions.
"""

# Local step definitions (new steps specific to this workspace)
LOCAL_STEPS = {
    "MyCustomProcessingStep": {
        "config_class": "MyCustomProcessingConfig",
        "builder_step_name": "MyCustomProcessingStepBuilder",
        "spec_type": "MyCustomProcessing",
        "sagemaker_step_type": "Processing",
        "description": "Custom processing step for developer_k",
        
        # Conflict resolution metadata
        "framework": "pandas",
        "environment_tags": ["development", "cpu"],
        "compatibility_tags": ["custom", "experimental"],
        "priority": 90,
        "conflict_resolution_strategy": "workspace_priority"
    },
    
    "ExperimentalTrainingStep": {
        "config_class": "ExperimentalTrainingConfig",
        "builder_step_name": "ExperimentalTrainingStepBuilder",
        "spec_type": "ExperimentalTraining",
        "sagemaker_step_type": "Training",
        "description": "Experimental training approach",
        
        # Conflict resolution metadata
        "framework": "pytorch",
        "environment_tags": ["development", "gpu"],
        "compatibility_tags": ["experimental", "research"],
        "priority": 95,
        "conflict_resolution_strategy": "framework_match"
    }
}

# Step overrides (override core step definitions for this workspace)
STEP_OVERRIDES = {
    "XGBoostTraining": {
        "config_class": "CustomXGBoostTrainingConfig",
        "builder_step_name": "CustomXGBoostTrainingStepBuilder",
        "spec_type": "CustomXGBoostTraining",
        "sagemaker_step_type": "Training",
        "description": "Custom XGBoost implementation with enhanced features",
        
        # Conflict resolution metadata
        "framework": "xgboost",
        "environment_tags": ["development", "custom"],
        "compatibility_tags": ["enhanced", "custom_metrics"],
        "priority": 80,  # Higher priority than default
        "conflict_resolution_strategy": "workspace_priority"
    }
}

# Workspace metadata
WORKSPACE_METADATA = {
    "developer_id": "developer_k",
    "version": "1.0.0",
    "description": "Custom ML pipeline extensions",
    "dependencies": ["pandas>=1.3.0", "pytorch>=1.9.0"],
    
    # Conflict resolution preferences
    "default_resolution_strategy": "workspace_priority",
    "preferred_frameworks": ["pytorch", "pandas", "xgboost"],
    "environment_preferences": ["development", "cpu"],
    "conflict_tolerance": "medium"
}
```

2. **Registry Initialization Script**
```python
# File: src/cursus/steps/registry/hybrid/init_workspace_registry.py
def create_workspace_registry(workspace_path: str, developer_id: str, template: str = "standard"):
    """Create workspace registry structure for a developer."""
    workspace_dir = Path(workspace_path)
    registry_dir = workspace_dir / "src" / "cursus_dev" / "registry"
    
    # Create registry directory
    registry_dir.mkdir(parents=True, exist_ok=True)
    
    # Create __init__.py
    init_file = registry_dir / "__init__.py"
    init_file.write_text('"""Local registry for workspace."""\n')
    
    # Create workspace_registry.py from template
    registry_file = registry_dir / "workspace_registry.py"
    template_content = _get_registry_template(developer_id, template)
    registry_file.write_text(template_content)
    
    return str(registry_file)

def _get_registry_template(developer_id: str, template: str) -> str:
    """Get registry template content."""
    if template == "standard":
        return f'''"""
Local registry for {developer_id} workspace.
"""

# Local step definitions
LOCAL_STEPS = {{
    # Add your custom steps here
    # Example:
    # "MyCustomStep": {{
    #     "config_class": "MyCustomStepConfig",
    #     "builder_step_name": "MyCustomStepBuilder",
    #     "spec_type": "MyCustomStep",
    #     "sagemaker_step_type": "Processing",
    #     "description": "My custom processing step",
    #     "framework": "pandas",
    #     "environment_tags": ["development"],
    #     "priority": 90
    # }}
}}

# Step overrides
STEP_OVERRIDES = {{
    # Override core steps here
    # Example:
    # "XGBoostTraining": {{
    #     "config_class": "CustomXGBoostTrainingConfig",
    #     "builder_step_name": "CustomXGBoostTrainingStepBuilder",
    #     "spec_type": "CustomXGBoostTraining",
    #     "sagemaker_step_type": "Training",
    #     "description": "Custom XGBoost with enhanced features",
    #     "framework": "xgboost",
    #     "priority": 80
    # }}
}}

# Workspace metadata
WORKSPACE_METADATA = {{
    "developer_id": "{developer_id}",
    "version": "1.0.0",
    "description": "Custom ML pipeline extensions",
    "dependencies": [],
    "default_resolution_strategy": "workspace_priority",
    "preferred_frameworks": [],
    "environment_preferences": ["development"]
}}
'''
    else:
        raise ValueError(f"Unknown template: {template}")
```

#### 3.2 Create Registry Management CLI

**Deliverable**: CLI tools for managing local registries

**Implementation Tasks**:

```python
# File: src/cursus/cli/registry_cli.py
import click
from pathlib import Path
from ..steps.registry.hybrid import (
    HybridRegistryManager, 
    create_workspace_registry,
    get_global_registry_manager
)

@click.group(name='registry')
def registry_cli():
    """Registry management commands."""
    pass

@registry_cli.command('init-workspace')
@click.argument('developer_id')
@click.option('--workspace-path', help='Workspace path (default: developer_workspaces/developers/{developer_id})')
@click.option('--template', default='standard', help='Registry template to use')
def init_workspace_registry(developer_id: str, workspace_path: str, template: str):
    """Initialize local registry for a developer workspace."""
    if not workspace_path:
        workspace_path = f"developer_workspaces/developers/{developer_id}"
    
    try:
        registry_file = create_workspace_registry(workspace_path, developer_id, template)
        click.echo(f"✅ Created workspace registry: {registry_file}")
        click.echo(f"📝 Edit the registry file to add your custom steps")
    except Exception as e:
        click.echo(f"❌ Failed to create workspace registry: {e}")

@registry_cli.command('list-steps')
@click.option('--workspace', help='Workspace ID to list steps from')
@click.option('--conflicts-only', is_flag=True, help='Show only conflicting steps')
@click.option('--format', type=click.Choice(['table', 'json']), default='table')
def list_steps(workspace: str, conflicts_only: bool, format: str):
    """List steps in registry with optional workspace context."""
    registry_manager = get_global_registry_manager()
    
    if conflicts_only:
        conflicts = registry_manager.get_step_conflicts()
        if format == 'json':
            import json
            click.echo(json.dumps(conflicts, indent=2))
        else:
            if conflicts:
                click.echo("Step Name Conflicts:")
                for step_name, definitions in conflicts.items():
                    workspaces = [d.workspace_id for d in definitions]
                    click.echo(f"  {step_name}: {', '.join(workspaces)}")
            else:
                click.echo("No step name conflicts found.")
    else:
        step_definitions = registry_manager.get_all_step_definitions(workspace)
        if format == 'json':
            import json
            legacy_dict = {name: defn.to_legacy_format() for name, defn in step_definitions.items()}
            click.echo(json.dumps(legacy_dict, indent=2))
        else:
            click.echo(f"Steps in {'workspace ' + workspace if workspace else 'core registry'}:")
            for step_name, definition in step_definitions.items():
                source = f"({definition.workspace_id})" if definition.workspace_id else "(core)"
                click.echo(f"  {step_name}: {definition.description} {source}")

@registry_cli.command('resolve-step')
@click.argument('step_name')
@click.option('--workspace', help='Workspace context')
@click.option('--framework', help='Preferred framework')
@click.option('--environment', help='Environment tags (comma-separated)')
def resolve_step(step_name: str, workspace: str, framework: str, environment: str):
    """Resolve step with conflict resolution."""
    registry_manager = get_global_registry_manager()
    
    environment_tags = environment.split(',') if environment else []
    
    definition = registry_manager.get_step_definition_with_resolution(
        step_name=step_name,
        workspace_id=workspace,
        preferred_framework=framework,
        environment_tags=environment_tags
    )
    
    if definition:
        source = f"workspace '{definition.workspace_id}'" if definition.workspace_id else "core registry"
        click.echo(f"✅ Resolved '{step_name}' from {source}")
        click.echo(f"   Config: {definition.config_class}")
        click.echo(f"   Builder: {definition.builder_step_name}")
        click.echo(f"   Framework: {definition.framework or 'N/A'}")
        click.echo(f"   Description: {definition.description}")
    else:
        click.echo(f"❌ Could not resolve step '{step_name}'")

@registry_cli.command('validate-registry')
@click.option('--workspace', help='Validate specific workspace registry')
@click.option('--check-conflicts', is_flag=True, help='Check for step name conflicts')
def validate_registry(workspace: str, check_conflicts: bool):
    """Validate registry consistency and conflicts."""
    registry_manager = get_global_registry_manager()
    
    if workspace:
        # Validate specific workspace
        if workspace in registry_manager._local_registries:
            local_registry = registry_manager._local_registries[workspace]
            # Perform validation
            click.echo(f"✅ Workspace '{workspace}' registry is valid")
        else:
            click.echo(f"❌ Workspace '{workspace}' not found")
    else:
        # Validate all registries
        click.echo("Validating all registries...")
        
        # Check core registry
        try:
            core_definitions = registry_manager.core_registry.get_all_step_definitions()
            click.echo(f"✅ Core registry: {len(core_definitions)} steps")
        except Exception as e:
            click.echo(f"❌ Core registry error: {e}")
        
        # Check workspace registries
        for workspace_id, local_registry in registry_manager._local_registries.items():
            try:
                local_definitions = local_registry.get_local_only_definitions()
                click.echo(f"✅ Workspace '{workspace_id}': {len(local_definitions)} local steps")
            except Exception as e:
                click.echo(f"❌ Workspace '{workspace_id}' error: {e}")
    
    if check_conflicts:
        conflicts = registry_manager.get_step_conflicts()
        if conflicts:
            click.echo("\n⚠️  Step Name Conflicts Found:")
            for step_name, definitions in conflicts.items():
                workspaces = [d.workspace_id for d in definitions]
                click.echo(f"  {step_name}: {', '.join(workspaces)}")
        else:
            click.echo("\n✅ No step name conflicts found")
```

### Phase 4: Base Class Integration (Weeks 7-8)

#### 4.1 Enhance StepBuilderBase Integration

**Deliverable**: Workspace-aware base class integration

**Implementation Tasks**:

```python
# File: src/cursus/core/base/builder_base.py (ENHANCED)
class StepBuilderBase(ABC):
    @property
    def STEP_NAMES(self):
        """Lazy load step names with workspace context awareness."""
        if not hasattr(self, '_step_names'):
            # Detect workspace context from config or environment
            workspace_id = self._get_workspace_context()
            
            # Use hybrid registry with workspace context
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
            return get_workspace_context()
        except:
            pass
        
        return None
```

#### 4.2 Enhance BasePipelineConfig Integration

**Implementation Tasks**:

```python
# File: src/cursus/core/base/config_base.py (ENHANCED)
class BasePipelineConfig(ABC):
    _STEP_NAMES: ClassVar[Dict[str, str]] = {}
    
    @classmethod
    def get_step_registry(cls) -> Dict[str, str]:
        """Lazy load step registry with workspace context."""
        if not cls._STEP_NAMES:
            # Get workspace context
            workspace_id = cls._get_workspace_context()
            
            compatibility_layer = get_enhanced_compatibility()
            if workspace_id:
                compatibility_layer.set_workspace_context(workspace_id)
            
            cls._STEP_NAMES = compatibility_layer.get_config_step_registry()
        return cls._STEP_NAMES
    
    @classmethod
    def _get_workspace_context(cls) -> Optional[str]:
        """Extract workspace context for config classes."""
        import os
        return os.environ.get('CURSUS_WORKSPACE_ID')
```

#### 4.3 Enhance Builder Registry Integration

**Implementation Tasks**:

```python
# File: src/cursus/steps/registry/builder_registry.py (ENHANCED)
class WorkspaceAwareStepBuilderRegistry(StepBuilderRegistry):
    """Enhanced StepBuilderRegistry with workspace awareness."""
    
    def __init__(self):
        super().__init__()
        # Integration with hybrid registry
        self.hybrid_registry = get_global_registry_manager()
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
            workspace_definitions = self.hybrid_registry.get_all_step_definitions(workspace_id)
            for step_name, definition in workspace_definitions.items():
                if definition.registry_type in ['workspace', 'override']:
                    try:
                        builder_class = self._load_workspace_builder(definition, workspace_id)
                        if builder_class:
                            core_builders[step_name] = builder_class
                    except Exception as e:
                        registry_logger.warning(f"Failed to load workspace builder {step_name}: {e}")
        
        return core_builders

# Global registry replacement maintains exact same interface
def get_global_registry() -> WorkspaceAwareStepBuilderRegistry:
    """Get global step builder registry instance with workspace awareness."""
    global _global_registry
    if _global_registry is None:
        _global_registry = WorkspaceAwareStepBuilderRegistry()
    return _global_registry
```

### Phase 5: Drop-in Registry Replacement (Weeks 9-10)

#### 5.1 Replace step_names.py Module

**Deliverable**: Seamless replacement of existing step_names.py with hybrid backend

**Implementation Tasks**:

1. **Create Enhanced step_names.py Replacement**
```python
# File: src/cursus/steps/registry/step_names.py (REPLACED)
"""
Enhanced step names registry with hybrid backend support.
Maintains 100% backward compatibility while adding workspace awareness.
"""

# Import hybrid registry components
from .hybrid.legacy_api import *
from .hybrid.proxy import (
    set_workspace_context, 
    get_workspace_context, 
    clear_workspace_context,
    workspace_context
)

# Re-export all original functions and variables for backward compatibility
# These now use the hybrid registry backend transparently
__all__ = [
    # Core registry data structures
    'STEP_NAMES',
    'CONFIG_STEP_REGISTRY', 
    'BUILDER_STEP_NAMES',
    'SPEC_STEP_TYPES',
    
    # Helper functions
    'get_config_class_name',
    'get_builder_step_name',
    'get_spec_step_type',
    'get_spec_step_type_with_job_type',
    'get_step_name_from_spec_type',
    'get_all_step_names',
    'validate_step_name',
    'validate_spec_type',
    'get_step_description',
    'list_all_step_info',
    
    # SageMaker integration functions
    'get_sagemaker_step_type',
    'get_steps_by_sagemaker_type',
    'get_all_sagemaker_step_types',
    'validate_sagemaker_step_type',
    'get_sagemaker_step_type_mapping',
    
    # Advanced functions
    'get_canonical_name_from_file_name',
    'validate_file_name',
    
    # Workspace context management (NEW)
    'set_workspace_context',
    'get_workspace_context',
    'clear_workspace_context',
    'workspace_context'
]
```

2. **Update Registry __init__.py**
```python
# File: src/cursus/steps/registry/__init__.py (ENHANCED)
"""
Enhanced Pipeline Registry Module with hybrid registry support.
Maintains backward compatibility while adding workspace awareness.
"""

from .exceptions import RegistryError

# Enhanced builder registry with workspace support
from .builder_registry import (
    WorkspaceAwareStepBuilderRegistry as StepBuilderRegistry,
    get_global_registry,
    register_global_builder,
    list_global_step_types
)

# Enhanced step names with workspace support (now using hybrid backend)
from .step_names import (
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
    validate_file_name,            # Workspace-aware
    # NEW: Workspace context management
    set_workspace_context,
    get_workspace_context,
    clear_workspace_context,
    workspace_context
)

# Hyperparameter registry (unchanged for now)
from .hyperparameter_registry import (
    HYPERPARAMETER_REGISTRY,
    get_all_hyperparameter_classes,
    get_hyperparameter_class_by_model_type,
    get_module_path,
    get_all_hyperparameter_info,
    validate_hyperparameter_class
)

# Exact same __all__ list with additions for workspace context
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
    "get_sagemaker_step_type",
    "get_steps_by_sagemaker_type",
    "get_all_sagemaker_step_types",
    "validate_sagemaker_step_type",
    "get_sagemaker_step_type_mapping",
    "get_canonical_name_from_file_name",
    "validate_file_name",
    
    # Hyperparameter registry
    "HYPERPARAMETER_REGISTRY",
    "get_all_hyperparameter_classes",
    "get_hyperparameter_class_by_model_type",
    "get_module_path",
    "get_all_hyperparameter_info",
    "validate_hyperparameter_class",
    
    # NEW: Workspace context management
    "set_workspace_context",
    "get_workspace_context", 
    "clear_workspace_context",
    "workspace_context"
]
```

#### 5.2 Initialize Developer Workspace Registries

**Deliverable**: Set up local registries for existing developer workspaces

**Implementation Tasks**:

```bash
# Initialize registries for existing workspaces
python -m cursus.cli.registry init-workspace developer_1
python -m cursus.cli.registry init-workspace developer_2  
python -m cursus.cli.registry init-workspace developer_3

# Validate registry setup
python -m cursus.cli.registry validate-registry --check-conflicts
```

### Phase 6: Integration and Testing (Weeks 11-12)

#### 6.1 Comprehensive Backward Compatibility Testing

**Deliverable**: Validation that all existing code continues to work

**Implementation Tasks**:

1. **Create Compatibility Test Suite**
```python
# File: test/registry/test_hybrid_compatibility.py
import pytest
from src.cursus.steps.registry import (
    STEP_NAMES, CONFIG_STEP_REGISTRY, BUILDER_STEP_NAMES, SPEC_STEP_TYPES,
    get_config_class_name, get_builder_step_name, get_spec_step_type,
    get_all_step_names, validate_step_name, get_canonical_name_from_file_name
)

class TestHybridRegistryCompatibility:
    """Test that hybrid registry maintains backward compatibility."""
    
    def test_step_names_structure(self):
        """Test STEP_NAMES maintains original structure."""
        assert isinstance(STEP_NAMES, dict)
        assert "XGBoostTraining" in STEP_NAMES
        assert "config_class" in STEP_NAMES["XGBoostTraining"]
        assert "builder_step_name" in STEP_NAMES["XGBoostTraining"]
        assert "spec_type" in STEP_NAMES["XGBoostTraining"]
        assert "sagemaker_step_type" in STEP_NAMES["XGBoostTraining"]
        assert "description" in STEP_NAMES["XGBoostTraining"]
    
    def test_derived_registries(self):
        """Test derived registries maintain original structure."""
        assert isinstance(CONFIG_STEP_REGISTRY, dict)
        assert isinstance(BUILDER_STEP_NAMES, dict)
        assert isinstance(SPEC_STEP_TYPES, dict)
        
        # Test specific mappings
        assert CONFIG_STEP_REGISTRY["XGBoostTrainingConfig"] == "XGBoostTraining"
        assert BUILDER_STEP_NAMES["XGBoostTraining"] == "XGBoostTrainingStepBuilder"
        assert SPEC_STEP_TYPES["XGBoostTraining"] == "XGBoostTraining"
    
    def test_helper_functions(self):
        """Test all helper functions work unchanged."""
        # Test basic functions
        assert get_config_class_name("XGBoostTraining") == "XGBoostTrainingConfig"
        assert get_builder_step_name("XGBoostTraining") == "XGBoostTrainingStepBuilder"
        assert get_spec_step_type("XGBoostTraining") == "XGBoostTraining"
        
        # Test validation functions
        assert validate_step_name("XGBoostTraining") == True
        assert validate_step_name("NonExistentStep") == False
        
        # Test file name resolution
        assert get_canonical_name_from_file_name("xgboost_training") == "XGBoostTraining"
    
    def test_workspace_context_isolation(self):
        """Test workspace context doesn't affect other workspaces."""
        from src.cursus.steps.registry import set_workspace_context, clear_workspace_context
        
        # Test without workspace context
        original_steps = set(get_all_step_names())
        
        # Set workspace context
        set_workspace_context("developer_1")
        workspace_steps = set(get_all_step_names())
        
        # Clear context
        clear_workspace_context()
        restored_steps = set(get_all_step_names())
        
        # Original steps should be restored
        assert original_steps == restored_steps
```

2. **Integration Testing with Existing Components**
```python
# File: test/registry/test_base_class_integration.py
class TestBaseClassIntegration:
    """Test base class integration with hybrid registry."""
    
    def test_step_builder_base_integration(self):
        """Test StepBuilderBase works with hybrid registry."""
        from src.cursus.core.base.builder_base import StepBuilderBase
        from src.cursus.steps.configs.xgboost_training_config import XGBoostTrainingConfig
        
        # Create mock builder
        class TestBuilder(StepBuilderBase):
            def __init__(self, config):
                self.config = config
        
        # Test STEP_NAMES property
        config = XGBoostTrainingConfig()
        builder = TestBuilder(config)
        step_names = builder.STEP_NAMES
        
        assert isinstance(step_names, dict)
        assert "XGBoostTraining" in step_names
    
    def test_config_base_integration(self):
        """Test BasePipelineConfig works with hybrid registry."""
        from src.cursus.core.base.config_base import BasePipelineConfig
        
        # Test step registry access
        registry = BasePipelineConfig._get_step_registry()
        assert isinstance(registry, dict)
        assert "XGBoostTrainingConfig" in registry
```

#### 6.2 Performance and Load Testing

**Deliverable**: Ensure hybrid registry meets performance requirements

**Implementation Tasks**:

1. **Performance Benchmark Suite**
```python
# File: test/registry/test_hybrid_performance.py
import time
import pytest
from src.cursus.steps.registry import get_all_step_names, get_config_class_name

class TestHybridRegistryPerformance:
    """Test hybrid registry performance."""
    
    def test_registry_access_performance(self):
        """Test registry access is within acceptable limits."""
        # Benchmark core registry access
        start_time = time.time()
        for _ in range(1000):
            step_names = get_all_step_names()
        core_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert core_time < 1.0  # 1 second for 1000 accesses
    
    def test_workspace_context_switching_performance(self):
        """Test workspace context switching overhead."""
        from src.cursus.steps.registry import set_workspace_context, clear_workspace_context
        
        start_time = time.time()
        for i in range(100):
            set_workspace_context(f"developer_{i % 3 + 1}")
            get_config_class_name("XGBoostTraining")
            clear_workspace_context()
        context_time = time.time() - start_time
        
        # Context switching should be fast
        assert context_time < 0.5  # 0.5 seconds for 100 context switches
    
    def test_conflict_resolution_performance(self):
        """Test conflict resolution performance."""
        from src.cursus.steps.registry.hybrid import get_global_registry_manager
        
        registry_manager = get_global_registry_manager()
        
        start_time = time.time()
        for _ in range(100):
            definition = registry_manager.get_step_definition_with_resolution(
                "XGBoostTraining",
                workspace_id="developer_1",
                preferred_framework="xgboost"
            )
        resolution_time = time.time() - start_time
        
        # Conflict resolution should be efficient
        assert resolution_time < 1.0  # 1 second for 100 resolutions
```

### Phase 7: Developer Experience Enhancement (Weeks 13-14)

#### 7.1 Create Developer Onboarding Tools

**Deliverable**: Streamlined developer onboarding for hybrid registry

**Implementation Tasks**:

1. **Developer Setup Script**
```python
# File: src/cursus/cli/developer_setup.py
import click
from pathlib import Path
from ..steps.registry.hybrid import create_workspace_registry, get_global_registry_manager

@click.command('setup-developer')
@click.argument('developer_id')
@click.option('--workspace-path', help='Custom workspace path')
@click.option('--template', default='standard', help='Registry template')
@click.option('--copy-from', help='Copy registry from existing developer')
def setup_developer(developer_id: str, workspace_path: str, template: str, copy_from: str):
    """Set up complete developer environment with hybrid registry."""
    
    if not workspace_path:
        workspace_path = f"developer_workspaces/developers/{developer_id}"
    
    workspace_dir = Path(workspace_path)
    
    try:
        # Create workspace directory structure
        workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # Create basic workspace structure
        (workspace_dir / "src" / "cursus_dev" / "steps" / "builders").mkdir(parents=True, exist_ok=True)
        (workspace_dir / "src" / "cursus_dev" / "steps" / "configs").mkdir(parents=True, exist_ok=True)
        (workspace_dir / "src" / "cursus_dev" / "steps" / "contracts").mkdir(parents=True, exist_ok=True)
        (workspace_dir / "src" / "cursus_dev" / "steps" / "scripts").mkdir(parents=True, exist_ok=True)
        (workspace_dir / "src" / "cursus_dev" / "steps" / "specs").mkdir(parents=True, exist_ok=True)
        (workspace_dir / "test").mkdir(parents=True, exist_ok=True)
        (workspace_dir / "validation_reports").mkdir(parents=True, exist_ok=True)
        
        # Create workspace registry
        if copy_from:
            registry_file = _copy_registry_from_developer(workspace_path, developer_id, copy_from)
        else:
            registry_file = create_workspace_registry(workspace_path, developer_id, template)
        
        # Create README
        readme_file = workspace_dir / "README.md"
        readme_content = f"""# Developer {developer_id} Workspace

This workspace contains custom step implementations for developer {developer_id}.

## Registry

Local registry: `{registry_file}`

## Usage

1. Add custom steps to the local registry
2. Implement step builders, configs, contracts, specs, and scripts
3. Use workspace context for testing:

```python
from cursus.steps.registry import set_workspace_context
set_workspace_context("{developer_id}")
# Your code here
```

## CLI Commands

```bash
# List steps in this workspace
python -m cursus.cli.registry list-steps --workspace {developer_id}

# Resolve step conflicts
python -m cursus.cli.registry resolve-step MyStep --workspace {developer_id}

# Validate registry
python -m cursus.cli.registry validate-registry --workspace {developer_id}
```
"""
        readme_file.write_text(readme_content)
        
        click.echo(f"✅ Developer workspace created: {workspace_path}")
        click.echo(f"📝 Registry file: {registry_file}")
        click.echo(f"📖 Documentation: {readme_file}")
        click.echo(f"\n🚀 Next steps:")
        click.echo(f"   1. Edit {registry_file} to add your custom steps")
        click.echo(f"   2. Implement your step components in src/cursus_dev/steps/")
        click.echo(f"   3. Test with: python -m cursus.cli.registry validate-registry --workspace {developer_id}")
        
    except Exception as e:
        click.echo(f"❌ Failed to create developer workspace: {e}")

def _copy_registry_from_developer(workspace_path: str, developer_id: str, source_developer: str) -> str:
    """Copy registry from existing developer."""
    source_path = f"developer_workspaces/developers/{source_developer}/src/cursus_dev/registry/workspace_registry.py"
    
    if not Path(source_path).exists():
        raise ValueError(f"Source developer {source_developer} has no registry")
    
    # Read source registry
    with open(source_path, 'r') as f:
        content = f.read()
    
    # Replace developer ID in content
    content = content.replace(f'"{source_developer}"', f'"{developer_id}"')
    content = content.replace(f"'{source_developer}'", f"'{developer_id}'")
    
    # Write to new location
    target_path = f"{workspace_path}/src/cursus_dev/registry/workspace_registry.py"
    Path(target_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(target_path, 'w') as f:
        f.write(content)
    
    return target_path
```

#### 7.2 Create Developer Documentation

**Deliverable**: Comprehensive documentation for hybrid registry usage

**Implementation Tasks**:

1. **Developer Guide for Hybrid Registry**
```markdown
# File: slipbox/0_developer_guide/hybrid_registry_guide.md
# Hybrid Registry Developer Guide

## Overview

The hybrid registry system allows each developer to maintain their own local registry while accessing shared core steps. This enables isolated development with customized steps while preserving common functionality.

## Local Registry Structure

Each developer workspace has a local registry at:
```
developer_workspaces/developers/{developer_id}/src/cursus_dev/registry/workspace_registry.py
```

### Registry Format

```python
# Local step definitions (new steps)
LOCAL_STEPS = {
    "MyCustomStep": {
        "config_class": "MyCustomStepConfig",
        "builder_step_name": "MyCustomStepBuilder",
        "spec_type": "MyCustomStep",
        "sagemaker_step_type": "Processing",
        "description": "My custom processing step",
        
        # Conflict resolution metadata
        "framework": "pandas",
        "environment_tags": ["development"],
        "priority": 90
    }
}

# Step overrides (override core steps)
STEP_OVERRIDES = {
    "XGBoostTraining": {
        "config_class": "CustomXGBoostTrainingConfig",
        "builder_step_name": "CustomXGBoostTrainingStepBuilder",
        "spec_type": "CustomXGBoostTraining", 
        "sagemaker_step_type": "Training",
        "description": "Custom XGBoost with enhanced features",
        "framework": "xgboost",
        "priority": 80
    }
}
```

## Usage Patterns

### Basic Usage (No Workspace Context)
```python
# Works exactly like before - uses core registry
from cursus.steps.registry import STEP_NAMES, get_config_class_name

step_names = STEP_NAMES  # Core registry only
config_class = get_config_class_name("XGBoostTraining")  # Core implementation
```

### Workspace-Aware Usage
```python
# Set workspace context for local registry access
from cursus.steps.registry import set_workspace_context, get_config_class_name

set_workspace_context("developer_1")
config_class = get_config_class_name("XGBoostTraining")  # May use local override
```

### Context Manager Usage
```python
# Temporary workspace context
from cursus.steps.registry import workspace_context, get_config_class_name

with workspace_context("developer_1"):
    config_class = get_config_class_name("MyCustomStep")  # Local step
# Context automatically cleared
```

## Conflict Resolution

### Resolution Strategies

1. **Workspace Priority**: Current workspace steps override others
2. **Framework Match**: Steps matching preferred framework selected
3. **Environment Match**: Steps matching environment tags selected
4. **Priority Based**: Lower priority number = higher precedence

### Advanced Resolution
```python
from cursus.steps.registry.hybrid import get_global_registry_manager

registry_manager = get_global_registry_manager()

# Intelligent resolution with context
definition = registry_manager.get_step_definition_with_resolution(
    step_name="XGBoostTraining",
    workspace_id="developer_1", 
    preferred_framework="xgboost",
    environment_tags=["development", "gpu"]
)
```

## CLI Commands

### Registry Management
```bash
# Initialize workspace registry
python -m cursus.cli.registry init-workspace developer_1

# List steps with workspace context
python -m cursus.cli.registry list-steps --workspace developer_1

# Check for conflicts
python -m cursus.cli.registry list-steps --conflicts-only

# Resolve specific step
python -m cursus.cli.registry resolve-step XGBoostTraining --workspace developer_1 --framework xgboost

# Validate registry
python -m cursus.cli.registry validate-registry --workspace developer_1 --check-conflicts
```

### Developer Setup
```bash
# Complete developer setup
python -m cursus.cli.developer setup-developer developer_1

# Copy from existing developer
python -m cursus.cli.developer setup-developer developer_2 --copy-from developer_1
```

## Best Practices

### 1. Step Naming
- Use descriptive, unique names for custom steps
- Include framework or domain in name to avoid conflicts
- Example: "FinancialXGBoostTraining" instead of "XGBoostTraining"

### 2. Conflict Resolution Metadata
- Always specify framework for framework-specific steps
- Use environment_tags for environment-specific implementations
- Set appropriate priority levels (lower = higher priority)

### 3. Registry Organization
- Group related steps in LOCAL_STEPS
- Use STEP_OVERRIDES sparingly, only when necessary
- Document why overrides are needed

### 4. Testing
- Test steps in workspace context
- Validate registry before committing changes
- Check for conflicts with other developers

## Migration from Central Registry

### For Existing Developers
1. Initialize workspace registry: `python -m cursus.cli.registry init-workspace {your_id}`
2. Move custom steps from central registry to local registry
3. Update step implementations to use workspace context
4. Test with workspace context enabled

### For New Developers
1. Set up workspace: `python -m cursus.cli.developer setup-developer {your_id}`
2. Define custom steps in local registry
3. Implement step components
4. Test and validate
```

### Phase 8: Production Deployment (Weeks 15-16)

#### 8.1 Production Rollout Strategy

**Deliverable**: Safe production deployment of hybrid registry

**Implementation Tasks**:

1. **Feature Flag Implementation**
```python
# File: src/cursus/steps/registry/hybrid/feature_flags.py
import os
from typing import Optional

class HybridRegistryFeatureFlags:
    """Feature flags for gradual hybrid registry rollout."""
    
    @staticmethod
    def is_hybrid_registry_enabled() -> bool:
        """Check if hybrid registry is enabled."""
        return os.environ.get('CURSUS_HYBRID_REGISTRY', 'false').lower() == 'true'
    
    @staticmethod
    def is_workspace_context_enabled() -> bool:
        """Check if workspace context is enabled."""
        return os.environ.get('CURSUS_WORKSPACE_CONTEXT', 'false').lower() == 'true'
    
    @staticmethod
    def get_fallback_mode() -> str:
        """Get fallback mode for registry failures."""
        return os.environ.get('CURSUS_REGISTRY_FALLBACK', 'core_only')

# Enhanced compatibility layer with feature flags
class SafeBackwardCompatibilityLayer(EnhancedBackwardCompatibilityLayer):
    """Safe compatibility layer with feature flag support."""
    
    def get_step_names(self, workspace_id: str = None) -> Dict[str, Dict[str, Any]]:
        """Get step names with feature flag protection."""
        try:
            if HybridRegistryFeatureFlags.is_hybrid_registry_enabled():
                return super().get_step_names(workspace_id)
            else:
                # Fallback to original implementation
                return self._get_original_step_names()
        except Exception as e:
            if HybridRegistryFeatureFlags.get_fallback_mode() == 'core_only':
                return self._get_original_step_names()
            else:
                raise e
    
    def _get_original_step_names(self) -> Dict[str, Dict[str, Any]]:
        """Fallback to original STEP_NAMES implementation."""
        # Import original step_names if hybrid fails
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "original_step_names", 
            "src/cursus/steps/registry/step_names_original.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, 'STEP_NAMES', {})
```

2. **Gradual Rollout Plan**
```bash
# Phase 8.1: Enable hybrid registry in development
export CURSUS_HYBRID_REGISTRY=true
export CURSUS_WORKSPACE_CONTEXT=false

# Phase 8.2: Enable workspace context for testing
export CURSUS_WORKSPACE_CONTEXT=true

# Phase 8.3: Full production deployment
# Remove feature flags after validation
```

#### 8.2 Monitoring and Diagnostics

**Deliverable**: Production monitoring for hybrid registry

**Implementation Tasks**:

1. **Registry Health Monitoring**
```python
# File: src/cursus/steps/registry/hybrid/monitoring.py
class RegistryHealthMonitor:
    """Monitor hybrid registry health and performance."""
    
    def __init__(self, registry_manager: HybridRegistryManager):
        self.registry_manager = registry_manager
        self.metrics = {
            'registry_access_count': 0,
            'conflict_resolution_count': 0,
            'workspace_context_switches': 0,
            'errors': []
        }
    
    def check_registry_health(self) -> Dict[str, Any]:
        """Comprehensive registry health check."""
        health_report = {
            'overall_status': 'HEALTHY',
            'core_registry': self._check_core_registry(),
            'local_registries': self._check_local_registries(),
            'conflicts': self._check_conflicts(),
            'performance': self._check_performance(),
            'recommendations': []
        }
        
        # Determine overall status
        if any(not status['healthy'] for status in health_report.values() if isinstance(status, dict) and 'healthy' in status):
            health_report['overall_status'] = 'UNHEALTHY'
        
        return health_report
    
    def _check_core_registry(self) -> Dict[str, Any]:
        """Check core registry health."""
        try:
            core_definitions = self.registry_manager.core_registry.get_all_step_definitions()
            return {
                'healthy': True,
                'step_count': len(core_definitions),
                'issues': []
            }
        except Exception as e:
            return {
                'healthy': False,
                'step_count': 0,
                'issues': [str(e)]
            }
    
    def _check_local_registries(self) -> Dict[str, Any]:
        """Check local registry health."""
        local_status = {
            'healthy': True,
            'total_workspaces': len(self.registry_manager._local_registries),
            'healthy_workspaces': 0,
            'issues': []
        }
        
        for workspace_id, local_registry in self.registry_manager._local_registries.items():
            try:
                local_definitions = local_registry.get_local_only_definitions()
                local_status['healthy_workspaces'] += 1
            except Exception as e:
                local_status['healthy'] = False
                local_status['issues'].append(f"Workspace {workspace_id}: {e}")
        
        return local_status
    
    def _check_conflicts(self) -> Dict[str, Any]:
        """Check for step name conflicts."""
        conflicts = self.registry_manager.get_step_conflicts()
        return {
            'healthy': len(conflicts) == 0,
            'conflict_count': len(conflicts),
            'conflicted_steps': list(conflicts.keys())
        }
    
    def _check_performance(self) -> Dict[str, Any]:
        """Check registry performance metrics."""
        return {
            'healthy': True,
            'metrics': self.metrics.copy()
        }
```

## Developer Workflow Examples

### Example 1: Adding a New Custom Step

```python
# 1. Set up workspace registry (one-time)
python -m cursus.cli.developer setup-developer john_doe

# 2. Edit workspace registry
# File: developer_workspaces/developers/john_doe/src/cursus_dev/registry/workspace_registry.py
LOCAL_STEPS = {
    "FinancialDataPreprocessing": {
        "config_class": "FinancialDataPreprocessingConfig",
        "builder_step_name": "FinancialDataPreprocessingStepBuilder",
        "spec_type": "FinancialDataPreprocessing",
        "sagemaker_step_type": "Processing",
        "description": "Financial data preprocessing with domain-specific transformations",
        "framework": "pandas",
        "environment_tags": ["development", "financial"],
        "priority": 90
    }
}

# 3. Implement step components
# File: developer_workspaces/developers/john_doe/src/cursus_dev/steps/configs/financial_data_preprocessing_config.py
from cursus.core.base.config_base import BasePipelineConfig

class FinancialDataPreprocessingConfig(BasePipelineConfig):
    # Custom config implementation
    pass

# 4. Test with workspace context
from cursus.steps.registry import set_workspace_context, get_config_class_name

set_workspace_context("john_doe")
config_class = get_config_class_name("FinancialDataPreprocessing")  # Uses local registry
```

### Example 2: Overriding Core Step Implementation

```python
# 1. Override core step in workspace registry
# File: developer_workspaces/developers/jane_smith/src/cursus_dev/registry/workspace_registry.py
STEP_OVERRIDES = {
    "XGBoostTraining": {
        "config_class": "EnhancedXGBoostTrainingConfig",
        "builder_step_name": "EnhancedXGBoostTrainingStepBuilder",
        "spec_type": "EnhancedXGBoostTraining",
        "sagemaker_step_type": "Training",
        "description": "XGBoost training with custom hyperparameter optimization",
        "framework": "xgboost",
        "environment_tags": ["production", "gpu"],
        "priority": 75,  # Higher priority than core (100)
        "conflict_resolution_strategy": "workspace_priority"
    }
}

# 2. Implement enhanced version
# File: developer_workspaces/developers/jane_smith/src/cursus_dev/steps/configs/enhanced_xgboost_training_config.py
from cursus.steps.configs.xgboost_training_config import XGBoostTrainingConfig

class EnhancedXGBoostTrainingConfig(XGBoostTrainingConfig):
    # Enhanced implementation with additional features
    custom_hyperparameter_optimization: bool = True
    advanced_early_stopping: bool = True

# 3. Test override behavior
from cursus.steps.registry import workspace_context, get_config_class_name

# Without workspace context - uses core implementation
config_class = get_config_class_name("XGBoostTraining")  # "XGBoostTrainingConfig"

# With workspace context - uses override
with workspace_context("jane_smith"):
    config_class = get_config_class_name("XGBoostTraining")  # "EnhancedXGBoostTrainingConfig"
```

### Example 3: Handling Step Name Conflicts

```python
# Scenario: Multiple developers define "ModelEvaluation" step

# Developer A's registry
LOCAL_STEPS = {
    "ModelEvaluation": {
        "config_class": "PyTorchModelEvaluationConfig",
        "builder_step_name": "PyTorchModelEvaluationStepBuilder",
        "spec_type": "PyTorchModelEvaluation",
        "sagemaker_step_type": "Processing",
        "description": "PyTorch model evaluation",
        "framework": "pytorch",
        "environment_tags": ["development", "gpu"],
        "priority": 85
    }
}

# Developer B's registry  
LOCAL_STEPS = {
    "ModelEvaluation": {
        "config_class": "TensorFlowModelEvaluationConfig",
        "builder_step_name": "TensorFlowModelEvaluationStepBuilder", 
        "spec_type": "TensorFlowModelEvaluation",
        "sagemaker_step_type": "Processing",
        "description": "TensorFlow model evaluation",
        "framework": "tensorflow",
        "environment_tags": ["development", "gpu"],
        "priority": 85
    }
}

# Resolution examples
from cursus.steps.registry.hybrid import get_global_registry_manager

registry_manager = get_global_registry_manager()

# Resolve by workspace
definition_a = registry_manager.get_step_definition("ModelEvaluation", workspace_id="developer_a")
# Returns PyTorch implementation

definition_b = registry_manager.get_step_definition("ModelEvaluation", workspace_id="developer_b") 
# Returns TensorFlow implementation

# Resolve by framework preference
definition_pytorch = registry_manager.get_step_definition_with_resolution(
    "ModelEvaluation",
    preferred_framework="pytorch"
)
# Returns PyTorch implementation regardless of workspace

definition_tf = registry_manager.get_step_definition_with_resolution(
    "ModelEvaluation", 
    preferred_framework="tensorflow"
)
# Returns TensorFlow implementation regardless of workspace
```

### Example 4: Multi-Developer Pipeline Collaboration

```python
# Scenario: Building pipeline using steps from multiple developers

from cursus.steps.registry import workspace_context
from cursus.pipeline.assembler import PipelineAssembler

# Create pipeline using steps from different workspaces
assembler = PipelineAssembler()

# Use core preprocessing step
assembler.add_step("TabularDataPreprocessing", config=core_preprocessing_config)

# Use developer A's custom feature engineering
with workspace_context("developer_a"):
    assembler.add_step("AdvancedFeatureEngineering", config=feature_config)

# Use developer B's custom training
with workspace_context("developer_b"):
    assembler.add_step("CustomXGBoostTraining", config=training_config)

# Use core model registration
assembler.add_step("RegisterModel", config=registration_config)

pipeline = assembler.build()
```

## Implementation Timeline

### Phase 1-2: Foundation (Weeks 1-4)
- **Week 1**: HybridStepDefinition, CoreStepRegistry, LocalStepRegistry
- **Week 2**: IntelligentConflictResolver, HybridRegistryManager
- **Week 3**: EnhancedBackwardCompatibilityLayer, ContextAwareRegistryProxy
- **Week 4**: Drop-in replacement functions, legacy API preservation

### Phase 3-4: Infrastructure (Weeks 5-8)
- **Week 5**: Local registry templates, workspace registry format
- **Week 6**: Registry management CLI, initialization scripts
- **Week 7**: StepBuilderBase integration, BasePipelineConfig enhancement
- **Week 8**: Builder registry integration, workspace-aware discovery

### Phase 5-6: Integration (Weeks 9-12)
- **Week 9**: step_names.py replacement, registry __init__.py update
- **Week 10**: Developer workspace registry initialization
- **Week 11**: Comprehensive backward compatibility testing
- **Week 12**: Integration testing, performance validation

### Phase 7-8: Production (Weeks 13-16)
- **Week 13**: Developer onboarding tools, setup scripts
- **Week 14**: Developer documentation, usage guides
- **Week 15**: Feature flag implementation, safe rollout strategy
- **Week 16**: Production deployment, monitoring setup

## Risk Mitigation

### Critical Risks and Mitigation Strategies

**Risk 1: Backward Compatibility Breakage**
- **Mitigation**: Comprehensive test suite covering all 232+ references
- **Fallback**: Feature flags with core-only fallback mode
- **Validation**: Automated compatibility testing in CI/CD

**Risk 2: Performance Degradation**
- **Mitigation**: Caching layer for registry access
- **Monitoring**: Performance benchmarks and alerts
- **Optimization**: Lazy loading and context-aware caching

**Risk 3: Complex Conflict Resolution**
- **Mitigation**: Clear resolution strategies and documentation
- **Tools**: CLI tools for conflict detection and resolution
- **Training**: Developer education on best practices

**Risk 4: Registry Corruption**
- **Mitigation**: Registry validation and health checks
- **Backup**: Automatic backup of registry changes
- **Recovery**: Registry repair and restoration tools

## Success Metrics

### Technical Metrics
- **Zero backward compatibility breaks**: All existing code continues to work
- **Performance maintained**: Registry access within 10ms baseline
- **Conflict resolution**: 95%+ automatic resolution success rate
- **Developer adoption**: 100% developer workspace setup within 4 weeks

### Developer Experience Metrics
- **Setup time**: New developer onboarding under 15 minutes
- **Development friction**: Reduced merge conflicts by 80%
- **Step development**: Custom step creation time reduced by 60%
- **Documentation clarity**: Developer guide comprehension score >90%

## Post-Migration Benefits

### For Individual Developers
- **Isolated Development**: Experiment with custom steps without affecting others
- **Rapid Prototyping**: Quick iteration on step implementations
- **Framework Flexibility**: Use preferred frameworks without conflicts
- **Reduced Friction**: No merge conflicts on registry changes

### For Team Collaboration
- **Parallel Development**: Multiple developers work simultaneously without conflicts
- **Selective Sharing**: Share successful experiments through controlled integration
- **Version Control**: Independent versioning of custom implementations
- **Quality Control**: Isolated testing before integration

### For System Architecture
- **Scalability**: Support unlimited number of developers
- **Maintainability**: Clear separation of concerns between core and custom
- **Flexibility**: Easy addition of new resolution strategies
- **Robustness**: Fallback mechanisms for registry failures

## Conclusion

This comprehensive migration plan transforms our centralized registry into a hybrid system that maintains all existing functionality while enabling isolated multi-developer workflows. The phased approach ensures minimal risk while delivering maximum benefit to developer productivity and system scalability.

The hybrid registry system preserves the simplicity of the current system for basic usage while providing advanced capabilities for complex multi-developer scenarios. Through intelligent conflict resolution and workspace isolation, developers can innovate freely while maintaining system stability and backward compatibility.

Key success factors:
1. **Gradual Migration**: Phased rollout with feature flags and fallback mechanisms
2. **Comprehensive Testing**: Extensive validation of backward compatibility
3. **Developer Education**: Clear documentation and onboarding tools
4. **Monitoring**: Production health monitoring and diagnostics
5. **Continuous Improvement**: Iterative enhancement based on developer feedback

The migration will be complete when all developers can work independently in their isolated workspaces while seamlessly accessing shared core functionality, with zero impact on existing code and workflows.

## References

### Core Design Documents
- **[Workspace-Aware Distributed Registry Design](../1_design/workspace_aware_distributed_registry_design.md)** - Foundational design for distributed registry architecture with namespaced step definitions and intelligent conflict resolution
- **[Design Principles](../1_design/design_principles.md)** - Architectural philosophy and quality standards that guide the migration design
- **[Documentation YAML Frontmatter Standard](../1_design/documentation_yaml_frontmatter_standard.md)** - YAML header format standard used in this migration plan

### Implementation Planning Context
- **[2025-08-28 Workspace-Aware Unified Implementation Plan](2025-08-28_workspace_aware_unified_implementation_plan.md)** - Overall implementation plan that includes Phase 7 registry migration as part of the broader workspace-aware system
- **[Step Names Integration Requirements Analysis](../4_analysis/step_names_integration_requirements_analysis.md)** - Critical analysis of 232+ existing step_names references and backward compatibility requirements

### Current System Analysis
- **Current Registry Location**: `src/cursus/steps/registry/` - Existing centralized registry system with step_names.py, builder_registry.py, and hyperparameter_registry.py
- **Current Step Definitions**: 17 core step definitions in STEP_NAMES dictionary with derived registries (CONFIG_STEP_REGISTRY, BUILDER_STEP_NAMES, SPEC_STEP_TYPES)
- **Integration Points**: Base classes (StepBuilderBase, BasePipelineConfig) and validation system (108+ references)

### Target Architecture
- **Developer Workspace Structure**: `developer_workspaces/developers/developer_k/` - Target structure for isolated local developer registries
- **Hybrid Registry Components**: CoreStepRegistry, LocalStepRegistry, HybridRegistryManager, IntelligentConflictResolver
- **Compatibility Layer**: EnhancedBackwardCompatibilityLayer, ContextAwareRegistryProxy, LegacyAPIPreservation

### Related Workspace Architecture
- **[Workspace-Aware Multi-Developer Management Design](../1_design/workspace_aware_multi_developer_management_design.md)** - Master design document for multi-developer workspace architecture
- **[Workspace-Aware Validation System Design](../1_design/workspace_aware_validation_system_design.md)** - Validation framework that integrates with the hybrid registry system
- **[Registry Single Source of Truth](../1_design/registry_single_source_of_truth.md)** - Original registry design principles and centralized registry concept

### Quality Assessment
- **[2025-09-02 Hybrid Registry Migration Plan Analysis](../4_analysis/2025-09-02_hybrid_registry_migration_plan_analysis.md)** - Comprehensive quality assessment of this migration plan against design principles, backward compatibility, and code redundancy criteria

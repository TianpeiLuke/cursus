---
tags:
  - analysis
  - integration
  - registry_system
  - backward_compatibility
  - workspace_management
keywords:
  - step names integration
  - distributed registry
  - backward compatibility
  - workspace context
  - registry migration
  - STEP_NAMES analysis
  - base class integration
  - import pattern analysis
topics:
  - registry system integration
  - workspace-aware architecture
  - backward compatibility strategy
  - system migration planning
language: python
date of note: 2025-08-29
---

# Step Names Integration Requirements Analysis

## Executive Summary

This analysis examines how the current system imports and uses `step_names` from `cursus/steps/registry` to understand the integration requirements for the distributed registry system. The goal is to ensure a smooth transition from one workspace under `src/cursus/steps` to multiple workspaces while keeping all existing pipelines and code unaffected.

## Current Usage Patterns

### 1. Core Registry Structure

The current `src/cursus/steps/registry/step_names.py` provides:

```python
# Main registry dictionary
STEP_NAMES = {
    "StepName": {
        "config_class": "ConfigClassName",
        "builder_step_name": "BuilderClassName", 
        "spec_type": "SpecTypeName",
        "sagemaker_step_type": "SageMakerStepType",
        "description": "Description"
    }
}

# Derived registries (CRITICAL for backward compatibility)
CONFIG_STEP_REGISTRY = {info["config_class"]: step_name for step_name, info in STEP_NAMES.items()}
BUILDER_STEP_NAMES = {step_name: info["builder_step_name"] for step_name, info in STEP_NAMES.items()}
SPEC_STEP_TYPES = {step_name: info["spec_type"] for step_name, info in STEP_NAMES.items()}
```

### 2. Import Patterns Analysis

Found **232 references** to step_names across the codebase with these patterns:

#### Direct Registry Imports
```python
from cursus.steps.registry.step_names import STEP_NAMES, CONFIG_STEP_REGISTRY, BUILDER_STEP_NAMES
from ...steps.registry.step_names import get_canonical_name_from_file_name, get_config_class_name
```

#### Function Imports
```python
from cursus.steps.registry.step_names import (
    get_sagemaker_step_type, 
    get_canonical_name_from_file_name,
    get_all_step_names,
    get_step_name_from_spec_type
)
```

### 3. Critical Base Class Dependencies

#### StepBuilderBase (src/cursus/core/base/builder_base.py)
```python
@property
def STEP_NAMES(self):
    """Lazy load step names to avoid circular imports while maintaining Single Source of Truth."""
    if not hasattr(self, '_step_names'):
        try:
            from ...steps.registry.step_names import BUILDER_STEP_NAMES
            self._step_names = BUILDER_STEP_NAMES
        except ImportError:
            self._step_names = {}
    return self._step_names
```

#### BasePipelineConfig (src/cursus/core/base/config_base.py)
```python
_STEP_NAMES: ClassVar[Dict[str, str]] = {}  # Will be populated via lazy loading

@classmethod
def _get_step_registry(cls) -> Dict[str, str]:
    """Lazy load step registry to avoid circular imports."""
    if not cls._STEP_NAMES:
        try:
            from ...steps.registry.step_names import CONFIG_STEP_REGISTRY
            cls._STEP_NAMES = CONFIG_STEP_REGISTRY
        except ImportError:
            logger.warning("Could not import step registry, using empty registry")
            cls._STEP_NAMES = {}
    return cls._STEP_NAMES
```

### 4. System Components Using Registry

#### Validation System (108+ references)
- **Alignment validation**: Uses STEP_NAMES for canonical name resolution
- **Builder testing**: Uses BUILDER_STEP_NAMES for builder discovery
- **Config analysis**: Uses CONFIG_STEP_REGISTRY for config class mapping
- **Dependency validation**: Uses get_all_step_names() for available steps

#### Core System Components
- **Pipeline assembler**: Uses CONFIG_STEP_REGISTRY for step type resolution
- **Compiler validation**: Uses CONFIG_STEP_REGISTRY for config type mapping
- **Workspace registry**: Uses STEP_NAMES for step type mapping

#### Step Specifications (40+ files)
- All step specs import `get_spec_step_type` or `get_spec_step_type_with_job_type`
- Used for consistent step_type assignment in specifications

## Integration Requirements for Distributed Registry

### 1. Backward Compatibility Requirements

**CRITICAL**: All existing code must continue to work without modification.

#### Must Maintain:
- **STEP_NAMES dictionary**: Core registry structure
- **CONFIG_STEP_REGISTRY**: Config class to step name mapping
- **BUILDER_STEP_NAMES**: Step name to builder class mapping  
- **SPEC_STEP_TYPES**: Step name to spec type mapping
- **All helper functions**: get_canonical_name_from_file_name, get_all_step_names, etc.

#### Import Compatibility:
```python
# These imports must continue to work unchanged
from cursus.steps.registry.step_names import STEP_NAMES
from cursus.steps.registry.step_names import CONFIG_STEP_REGISTRY, BUILDER_STEP_NAMES
from cursus.steps.registry.step_names import get_all_step_names, get_canonical_name_from_file_name
```

### 2. Workspace Context Integration

#### Base Class Integration Strategy:
```python
# StepBuilderBase enhancement
@property
def STEP_NAMES(self):
    """Enhanced with workspace context awareness."""
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
        return get_workspace_context()
    except:
        pass
    
    return None
```

#### BasePipelineConfig enhancement:
```python
@classmethod
def _get_step_registry(cls) -> Dict[str, str]:
    """Enhanced with workspace context."""
    if not cls._STEP_NAMES:
        # Get workspace context
        workspace_id = cls._get_workspace_context()
        
        compatibility_layer = get_enhanced_compatibility()
        if workspace_id:
            compatibility_layer.set_workspace_context(workspace_id)
        
        cls._STEP_NAMES = compatibility_layer.get_config_step_registry()
    return cls._STEP_NAMES
```

### 3. Transition Strategy

#### Phase 1: Enhanced Compatibility Layer
Create `src/cursus/registry/distributed/compatibility.py`:

```python
class EnhancedBackwardCompatibilityLayer:
    """Enhanced compatibility layer that maintains all derived registry structures."""
    
    def get_step_names(self, workspace_id: str = None) -> Dict[str, Dict[str, Any]]:
        """Returns STEP_NAMES format with workspace context."""
        
    def get_builder_step_names(self, workspace_id: str = None) -> Dict[str, str]:
        """Returns BUILDER_STEP_NAMES format with workspace context."""
        
    def get_config_step_registry(self, workspace_id: str = None) -> Dict[str, str]:
        """Returns CONFIG_STEP_REGISTRY format with workspace context."""
        
    def get_spec_step_types(self, workspace_id: str = None) -> Dict[str, str]:
        """Returns SPEC_STEP_TYPES format with workspace context."""

# Global registry replacement functions
def get_step_names() -> Dict[str, Dict[str, Any]]:
    return get_enhanced_compatibility().get_step_names()

def get_builder_step_names() -> Dict[str, str]:
    return get_enhanced_compatibility().get_builder_step_names()

# Dynamic module-level variables that update with workspace context
STEP_NAMES = get_step_names()
BUILDER_STEP_NAMES = get_builder_step_names()
CONFIG_STEP_REGISTRY = get_config_step_registry()
SPEC_STEP_TYPES = get_spec_step_types()
```

#### Phase 2: Drop-in Replacement
Replace `src/cursus/steps/registry/step_names.py` with compatibility layer that:
1. Maintains exact same API
2. Adds workspace context support
3. Preserves all existing functionality
4. Enables gradual workspace adoption

### 4. Workspace Context Management

#### Thread-Safe Context:
```python
import contextvars
from typing import Optional, ContextManager
from contextlib import contextmanager

# Thread-local workspace context
_workspace_context: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar('workspace_id', default=None)

def set_workspace_context(workspace_id: str) -> None:
    """Set the current workspace context."""
    _workspace_context.set(workspace_id)

def get_workspace_context() -> Optional[str]:
    """Get the current workspace context."""
    return _workspace_context.get()

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

## Implementation Recommendations

### 1. Minimal Disruption Approach

1. **Keep existing step_names.py structure** as the core registry
2. **Enhance with workspace awareness** through compatibility layer
3. **Maintain all existing imports** and function signatures
4. **Add workspace context detection** in base classes
5. **Enable gradual workspace adoption** without breaking existing code

### 2. Migration Path

#### Step 1: Create Distributed Registry Infrastructure
- Implement `src/cursus/registry/distributed/` module
- Create enhanced compatibility layer
- Add workspace context management
- Implement core and workspace registries

#### Step 2: Enhance Base Classes
- Update `StepBuilderBase.STEP_NAMES` property with workspace context detection
- Update `BasePipelineConfig._get_step_registry()` with workspace context detection
- Add workspace context detection methods

#### Step 3: Replace step_names.py
- Replace existing `step_names.py` with compatibility layer
- Maintain all existing APIs and imports
- Add workspace context support transparently

#### Step 4: Enable Workspace Features
- Allow workspace-specific step definitions
- Support workspace inheritance from core registry
- Enable workspace context switching

### 3. Risk Mitigation

#### Backward Compatibility Testing
- Comprehensive test suite for all existing imports
- Validation that all 232 references continue to work
- Performance testing to ensure no degradation

#### Gradual Rollout
- Feature flags for workspace functionality
- Fallback to existing behavior if workspace features fail
- Clear error messages and diagnostics

#### Documentation and Training
- Migration guide for developers
- Examples of workspace usage
- Troubleshooting guide for common issues

## Critical Success Factors

### 1. Zero Breaking Changes
- All existing code must continue to work without modification
- All existing imports must continue to function
- All existing APIs must maintain compatibility

### 2. Transparent Enhancement
- Workspace features should be opt-in
- Default behavior should remain unchanged
- Enhanced features should be discoverable but not required

### 3. Performance Preservation
- Registry access performance must not degrade
- Lazy loading patterns must be maintained
- Caching strategies must be preserved

### 4. Developer Experience
- Clear documentation for workspace features
- Easy migration path for adopting workspace functionality
- Comprehensive error messages and diagnostics

## Impact on Workspace-Aware Components

### Analysis of Existing Workspace Components

The distributed registry system will significantly impact the existing workspace-aware components in both `src/cursus/core/workspace` and `src/cursus/validation/workspace`. These components currently use the centralized STEP_NAMES registry and will need to be enhanced to work with the distributed registry system.

#### Core Workspace Components Impact

##### 1. WorkspaceComponentRegistry (src/cursus/core/workspace/registry.py)
**Current Implementation**:
- Directly imports and uses `STEP_NAMES` for step type mapping
- Uses core registry fallback for component discovery
- Implements component caching and discovery logic

**Required Changes**:
```python
# Current problematic code:
from ...steps.registry.step_names import STEP_NAMES

# Enhanced implementation:
class WorkspaceComponentRegistry:
    def __init__(self, workspace_root: str):
        self.workspace_root = workspace_root
        # Integration point: Use distributed registry for step definitions
        self.registry_manager = get_global_registry_manager()
        self.compatibility_layer = get_enhanced_compatibility()
    
    def find_builder_class(self, step_name: str, developer_id: str = None) -> Optional[Type]:
        """Enhanced with distributed registry integration."""
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
```

**Impact Level**: **HIGH** - Core component that bridges workspace discovery with registry system

##### 2. WorkspacePipelineAssembler (src/cursus/core/workspace/assembler.py)
**Current Implementation**:
- Extends PipelineAssembler with workspace component resolution
- Uses WorkspaceComponentRegistry for component discovery

**Required Changes**:
- No direct STEP_NAMES usage, but depends on WorkspaceComponentRegistry
- Will benefit from enhanced registry integration through WorkspaceComponentRegistry
- May need workspace context management for assembly operations

**Impact Level**: **MEDIUM** - Indirect impact through WorkspaceComponentRegistry dependency

##### 3. WorkspaceDAGCompiler (src/cursus/core/workspace/compiler.py)
**Current Implementation**:
- Extends PipelineDAGCompiler with workspace compilation logic
- Uses workspace component registry for validation

**Required Changes**:
- Similar to WorkspacePipelineAssembler, indirect impact through dependencies
- May need workspace context management during compilation

**Impact Level**: **MEDIUM** - Indirect impact through component registry dependencies

##### 4. Workspace Configuration Models (src/cursus/core/workspace/config.py)
**Current Implementation**:
- Pydantic V2 models for workspace step and pipeline definitions
- No direct registry dependencies currently

**Required Changes**:
- May need integration with distributed registry for step definition validation
- Could benefit from registry-based step type validation

**Impact Level**: **LOW** - Minimal direct impact, potential enhancement opportunities

#### Validation Workspace Components Impact

##### 1. WorkspaceUnifiedAlignmentTester (src/cursus/validation/workspace/workspace_alignment_tester.py)
**Current Implementation**:
- Extends UnifiedAlignmentTester for workspace support
- Uses workspace file resolution and module loading
- No direct STEP_NAMES usage identified

**Required Changes**:
- May benefit from distributed registry integration for component validation
- Could use registry for enhanced step type validation
- Workspace context management integration

**Impact Level**: **LOW** - Minimal direct impact, enhancement opportunities

##### 2. WorkspaceUniversalStepBuilderTest (src/cursus/validation/workspace/workspace_builder_test.py)
**Current Implementation**:
- Extends UniversalStepBuilderTest for workspace builder testing
- Uses workspace module loading for builder discovery

**Required Changes**:
- Could integrate with distributed registry for enhanced builder discovery
- May benefit from registry-based step type validation

**Impact Level**: **LOW** - Minimal direct impact, enhancement opportunities

##### 3. Workspace Infrastructure Components
**Components**: WorkspaceManager, WorkspaceFileResolver, WorkspaceModuleLoader
**Current Implementation**:
- Foundation infrastructure for workspace operations
- No direct registry dependencies

**Required Changes**:
- May need integration points with distributed registry
- Could benefit from registry-guided component discovery

**Impact Level**: **LOW** - Foundation components with enhancement opportunities

### Integration Strategy for Workspace Components

#### Phase 1: Core Registry Integration
1. **Update WorkspaceComponentRegistry** to use distributed registry system
2. **Implement workspace context management** in component discovery
3. **Add registry-guided component validation**

#### Phase 2: Enhanced Component Discovery
1. **Integrate distributed registry** with workspace file resolution
2. **Add step definition validation** before component loading
3. **Implement registry-based component caching**

#### Phase 3: Workspace Context Management
1. **Add workspace context detection** to all workspace components
2. **Implement context propagation** through component hierarchies
3. **Add context-aware error reporting**

### Benefits of Distributed Registry Integration

#### For Core Workspace Components
1. **Enhanced Component Discovery**: Registry-guided discovery improves accuracy
2. **Better Validation**: Step definition validation before component loading
3. **Improved Caching**: Registry-based caching strategies
4. **Context Awareness**: Workspace context propagation through all operations

#### For Validation Workspace Components
1. **Registry-Based Validation**: Enhanced validation using step definitions
2. **Better Error Reporting**: Registry context in error messages
3. **Improved Component Matching**: Registry-guided component matching
4. **Cross-Workspace Validation**: Registry-based dependency validation

### Implementation Recommendations

#### 1. Gradual Enhancement Approach
- **Phase 1**: Update WorkspaceComponentRegistry with distributed registry integration
- **Phase 2**: Enhance other components to use updated registry
- **Phase 3**: Add advanced features like cross-workspace validation

#### 2. Backward Compatibility Preservation
- **Maintain existing APIs** for all workspace components
- **Add new features** as optional enhancements
- **Preserve existing behavior** as default

#### 3. Performance Optimization
- **Registry-guided caching** for improved performance
- **Lazy loading** of registry data
- **Context-aware optimization** based on workspace usage patterns

## Conclusion

The analysis reveals that the distributed registry system must be implemented with extreme care to maintain backward compatibility. The 232+ references to step_names throughout the codebase, particularly in critical base classes, require a sophisticated compatibility layer that can transparently add workspace awareness while preserving all existing functionality.

### Impact Summary on Workspace Components

**High Impact Components**:
- **WorkspaceComponentRegistry**: Direct STEP_NAMES usage, requires significant enhancement

**Medium Impact Components**:
- **WorkspacePipelineAssembler**: Indirect impact through registry dependencies
- **WorkspaceDAGCompiler**: Indirect impact through registry dependencies

**Low Impact Components**:
- **Validation workspace components**: Minimal direct impact, enhancement opportunities
- **Workspace infrastructure**: Foundation components with integration opportunities

### Recommended Implementation Strategy

The recommended approach focuses on:
1. **Enhanced compatibility layer** that maintains all existing APIs
2. **Workspace context detection** in base classes
3. **Gradual migration path** that enables workspace features without breaking existing code
4. **Comprehensive testing** to ensure zero breaking changes
5. **Strategic enhancement** of workspace components to leverage distributed registry capabilities

This strategy ensures a smooth transition from single workspace to multiple workspaces while maintaining the stability and reliability of the existing

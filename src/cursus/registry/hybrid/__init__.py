"""
Optimized 3-Level Hybrid Registry System

This module provides the hybrid registry system with consolidated shared utilities
and optimized folder structure to maintain maximum 3-level depth while eliminating
code redundancy.

Architecture:
- utils.py: Consolidated shared utilities (RegistryLoader, StepDefinitionConverter, etc.)
- models.py: Data models (StepDefinition, ResolutionContext, etc.)
- manager.py: Registry management (CoreStepRegistry, LocalStepRegistry, HybridRegistryManager)
- resolver.py: Conflict resolution (ConflictDetector, ConflictResolver, StepResolver, etc.)
- compatibility.py: Backward compatibility (LegacyRegistryAdapter, EnhancedBackwardCompatibilityLayer, etc.)
- workspace.py: Workspace management (WorkspaceManager, WorkspaceIsolationManager, etc.)
"""

# Data Models
from .models import (
    StepDefinition,
    NamespacedStepDefinition,
    ResolutionContext,
    StepResolutionResult,
    RegistryValidationResult,
    ConflictAnalysis,
    StepComponentResolution,
    DistributedRegistryValidationResult
)

# Registry Management
from .manager import (
    RegistryConfig,
    CoreStepRegistry,
    LocalStepRegistry,
    HybridRegistryManager
)

# Conflict Resolution
from .resolver import (
    ConflictType,
    ResolutionStrategy,
    ConflictDetails,
    ResolutionPlan,
    DependencyAnalyzer,
    ConflictDetector,
    ConflictResolver,
    StepResolver,
    AdvancedStepResolver
)

# Backward Compatibility
from .compatibility import (
    LegacyRegistryAdapter,
    EnhancedBackwardCompatibilityLayer,
    APICompatibilityChecker,
    MigrationAssistant,
    BackwardCompatibilityValidator,
    LegacyRegistryInterface,
    deprecated_registry_method,
    create_legacy_registry_interface,
    get_compatibility_layer,
    set_compatibility_layer
)

# Workspace Management
from .workspace import (
    WorkspaceConfig,
    WorkspaceStatus,
    WorkspaceManager,
    WorkspaceIsolationManager,
    MultiDeveloperManager,
    WorkspaceConfigManager,
    WorkspaceAwareRegistryManager,
    create_workspace_aware_registry,
    get_default_workspace_config,
    create_developer_workspace_config
)

# Shared Utilities
from .utils import (
    RegistryLoader,
    StepDefinitionConverter,
    RegistryValidationUtils,
    RegistryErrorFormatter
)

__all__ = [
    # Data Models
    "StepDefinition",
    "NamespacedStepDefinition",
    "ResolutionContext",
    "StepResolutionResult",
    "RegistryValidationResult",
    "ConflictAnalysis",
    "StepComponentResolution",
    "DistributedRegistryValidationResult",
    
    # Registry Management
    "RegistryConfig",
    "CoreStepRegistry",
    "LocalStepRegistry",
    "HybridRegistryManager",
    
    # Conflict Resolution
    "ConflictType",
    "ResolutionStrategy",
    "ConflictDetails",
    "ResolutionPlan",
    "DependencyAnalyzer",
    "ConflictDetector",
    "ConflictResolver",
    "StepResolver",
    "AdvancedStepResolver",
    
    # Backward Compatibility
    "LegacyRegistryAdapter",
    "EnhancedBackwardCompatibilityLayer",
    "APICompatibilityChecker",
    "MigrationAssistant",
    "BackwardCompatibilityValidator",
    "LegacyRegistryInterface",
    "deprecated_registry_method",
    "create_legacy_registry_interface",
    "get_compatibility_layer",
    "set_compatibility_layer",
    
    # Workspace Management
    "WorkspaceConfig",
    "WorkspaceStatus",
    "WorkspaceManager",
    "WorkspaceIsolationManager",
    "MultiDeveloperManager",
    "WorkspaceConfigManager",
    "WorkspaceAwareRegistryManager",
    "create_workspace_aware_registry",
    "get_default_workspace_config",
    "create_developer_workspace_config",
    
    # Shared Utilities
    "RegistryLoader",
    "StepDefinitionConverter",
    "RegistryValidationUtils",
    "RegistryErrorFormatter"
]

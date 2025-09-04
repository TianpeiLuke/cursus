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

# Context-Aware Registry Proxy (NEW - Section 2.2)
from .proxy import (
    ContextAwareRegistryProxy,
    set_workspace_context,
    get_workspace_context,
    clear_workspace_context,
    workspace_context,
    get_global_registry_manager,
    get_enhanced_compatibility,
    get_context_aware_proxy,
    reset_global_instances,
    get_workspace_from_environment,
    auto_set_workspace_from_environment,
    validate_workspace_context,
    debug_workspace_context,
    with_workspace_context,
    auto_workspace_context,
    ensure_workspace_context,
    get_effective_workspace_context,
    update_compatibility_layer_context,
    sync_all_contexts,
    get_context_status
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
    
    # Context-Aware Registry Proxy (NEW - Section 2.2)
    "ContextAwareRegistryProxy",
    "set_workspace_context",
    "get_workspace_context",
    "clear_workspace_context",
    "workspace_context",
    "get_global_registry_manager",
    "get_enhanced_compatibility",
    "get_context_aware_proxy",
    "reset_global_instances",
    "get_workspace_from_environment",
    "auto_set_workspace_from_environment",
    "validate_workspace_context",
    "debug_workspace_context",
    "with_workspace_context",
    "auto_workspace_context",
    "ensure_workspace_context",
    "get_effective_workspace_context",
    "update_compatibility_layer_context",
    "sync_all_contexts",
    "get_context_status",
    
    # Shared Utilities
    "RegistryLoader",
    "StepDefinitionConverter",
    "RegistryValidationUtils",
    "RegistryErrorFormatter"
]

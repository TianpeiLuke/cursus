"""
Enhanced Pipeline Registry Module with Hybrid Registry Support.

This module contains registry components for tracking step types, specifications,
hyperparameters, and other metadata used in the pipeline system. It helps ensure 
consistency in step naming and configuration.

Enhanced Features:
- Workspace-aware step resolution
- Hybrid registry backend support
- Context management for multi-developer workflows
- Backward compatibility with existing code
"""

from .exceptions import RegistryError

# StepBuilderRegistry has been removed - use StepCatalog instead
# from .builder_registry import (
#     StepBuilderRegistry,
#     get_global_registry,
#     register_global_builder,
#     list_global_step_types,
# )

from .step_names import (
    # Core registry data structures
    STEP_NAMES,
    CONFIG_STEP_REGISTRY,
    BUILDER_STEP_NAMES,
    SPEC_STEP_TYPES,
    # Helper functions (now workspace-aware)
    get_config_class_name,
    get_builder_step_name,
    get_spec_step_type,
    get_spec_step_type_with_job_type,
    get_step_name_from_spec_type,
    get_all_step_names,
    validate_step_name,
    validate_spec_type,
    get_step_description,
    list_all_step_info,
    # SageMaker integration functions (now workspace-aware)
    get_sagemaker_step_type,
    get_steps_by_sagemaker_type,
    get_all_sagemaker_step_types,
    validate_sagemaker_step_type,
    get_sagemaker_step_type_mapping,
    # Advanced functions (now workspace-aware)
    get_canonical_name_from_file_name,
    validate_file_name,
    # NEW: Workspace context management
    set_workspace_context,
    get_workspace_context,
    clear_workspace_context,
    workspace_context,
    # NEW: Workspace-aware registry functions
    get_step_names,
    get_config_step_registry,
    get_builder_step_names,
    get_spec_step_types,
    # NEW: Workspace management functions
    list_available_workspaces,
    get_workspace_step_count,
    has_workspace_conflicts,
)

from .hyperparameter_registry import (
    HYPERPARAMETER_REGISTRY,
    get_all_hyperparameter_classes,
    get_hyperparameter_class_by_model_type,
    get_module_path,
    get_all_hyperparameter_info,
    validate_hyperparameter_class,
)

# NEW: Hybrid registry components (optional import)
try:
    from .hybrid.manager import UnifiedRegistryManager
    from .hybrid.models import (
        StepDefinition,
        ResolutionContext,
        StepResolutionResult,
        RegistryValidationResult,
        ConflictAnalysis,
        RegistryType,
        ResolutionMode,
        ResolutionStrategy,
        ConflictType,
    )

    # Add hybrid components to exports
    _HYBRID_EXPORTS = [
        "UnifiedRegistryManager",
        "StepDefinition",
        "ResolutionContext",
        "StepResolutionResult",
        "RegistryValidationResult",
        "ConflictAnalysis",
        "RegistryType",
        "ResolutionMode",
        "ResolutionStrategy",
        "ConflictType",
    ]

except ImportError:
    # Hybrid registry not available - continue with core functionality
    _HYBRID_EXPORTS = []

__all__ = [
    # Exceptions
    "RegistryError",
    # Builder registry - REMOVED: Use StepCatalog instead
    # "StepBuilderRegistry",
    # "get_global_registry", 
    # "register_global_builder",
    # "list_global_step_types",
    # Core step names and registry (backward compatible)
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
    # NEW: Workspace context management
    "set_workspace_context",
    "get_workspace_context",
    "clear_workspace_context",
    "workspace_context",
    # NEW: Workspace-aware registry functions
    "get_step_names",
    "get_config_step_registry",
    "get_builder_step_names",
    "get_spec_step_types",
    # NEW: Workspace management functions
    "list_available_workspaces",
    "get_workspace_step_count",
    "has_workspace_conflicts",
    # Hyperparameter registry
    "HYPERPARAMETER_REGISTRY",
    "get_all_hyperparameter_classes",
    "get_hyperparameter_class_by_model_type",
    "get_module_path",
    "get_all_hyperparameter_info",
    "validate_hyperparameter_class",
] + _HYBRID_EXPORTS


# Convenience functions for common workspace operations
def switch_to_workspace(workspace_id: str):
    """
    Switch to a specific workspace context.

    Args:
        workspace_id: Workspace identifier to switch to

    Example:
        switch_to_workspace("developer_1")
        step_names = STEP_NAMES  # Now uses developer_1 context
    """
    set_workspace_context(workspace_id)


def switch_to_core():
    """
    Switch back to core registry (no workspace context).

    Example:
        switch_to_core()
        step_names = STEP_NAMES  # Now uses core registry only
    """
    clear_workspace_context()


def get_registry_info(workspace_id: str = None) -> dict:
    """
    Get comprehensive registry information for a workspace or core.

    Args:
        workspace_id: Optional workspace identifier

    Returns:
        Dictionary with registry information
    """
    current_workspace = workspace_id or get_workspace_context()

    info = {
        "workspace_id": current_workspace,
        "step_count": len(get_step_names(current_workspace)),
        "available_steps": sorted(get_all_step_names(current_workspace)),
        "sagemaker_types": get_all_sagemaker_step_types(current_workspace),
        "has_conflicts": has_workspace_conflicts() if current_workspace else False,
    }

    if current_workspace:
        info["workspace_step_count"] = get_workspace_step_count(current_workspace)
        info["available_workspaces"] = list_available_workspaces()

    return info


# Add convenience functions to exports
__all__.extend(["switch_to_workspace", "switch_to_core", "get_registry_info"])

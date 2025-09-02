"""
Cursus Validation Workspace Package

Multi-developer workspace management system for Cursus validation framework.
Provides workspace-aware file resolution, module loading, and workspace management
utilities to support isolated developer environments.

Core Components:
- DeveloperWorkspaceFileResolver: Workspace-aware file discovery
- WorkspaceModuleLoader: Dynamic module loading with workspace isolation
- WorkspaceManager: Workspace discovery, validation, and management

Key Features:
- Multi-developer workspace isolation
- Shared workspace fallback support
- Dynamic module loading with Python path management
- Workspace structure validation and creation
- Configuration management for workspace settings

Usage:
    from cursus.validation.workspace import (
        WorkspaceManager,
        DeveloperWorkspaceFileResolver,
        WorkspaceModuleLoader
    )
    
    # Initialize workspace manager
    manager = WorkspaceManager("/path/to/workspaces")
    
    # Get workspace-aware components
    file_resolver = manager.get_file_resolver("developer_1")
    module_loader = manager.get_module_loader("developer_1")
    
    # Discover and validate workspaces
    workspace_info = manager.discover_workspaces()
    is_valid, issues = manager.validate_workspace_structure()
"""

from .workspace_file_resolver import DeveloperWorkspaceFileResolver
from .workspace_module_loader import WorkspaceModuleLoader
from .workspace_manager import (
    WorkspaceManager,
    WorkspaceConfig,
    DeveloperInfo,
    WorkspaceInfo
)

# NEW: Unified validation components
from .workspace_type_detector import WorkspaceTypeDetector
from .unified_validation_core import UnifiedValidationCore, ValidationConfig
from .unified_result_structures import (
    UnifiedValidationResult,
    WorkspaceValidationResult,
    ValidationSummary,
    ValidationResultBuilder,
    create_single_workspace_result,
    create_empty_result
)
from .unified_report_generator import (
    UnifiedReportGenerator,
    ReportConfig,
    generate_unified_report,
    export_unified_report
)
from .legacy_adapters import (
    LegacyWorkspaceValidationAdapter,
    create_legacy_adapter,
    validate_workspace_legacy,
    validate_all_workspaces_legacy
)

# Existing validation components
from .workspace_orchestrator import WorkspaceValidationOrchestrator

# PHASE 3: Consolidated validation system components
from .test_manager import (
    WorkspaceTestManager,
    TestEnvironment,
    IsolationReport,
    TestWorkspaceConfig,
    create_test_workspace_manager,
    validate_test_workspace_structure
)
# Import consolidated IsolationViolation from core module
from ...core.workspace.isolation import IsolationViolation

from .test_isolation import (
    TestWorkspaceIsolationManager,
    TestIsolationConfig,
    IsolationEnvironment,
    create_test_isolation_manager,
    validate_test_environment_isolation
)
from .cross_workspace_validator import (
    CrossWorkspaceValidator,
    ComponentConflict,
    DependencyResolution,
    ValidationResult,
    CrossWorkspaceConfig,
    create_cross_workspace_validator,
    validate_cross_workspace_compatibility
)

# Public API
__all__ = [
    # Core classes
    "DeveloperWorkspaceFileResolver",
    "WorkspaceModuleLoader", 
    "WorkspaceManager",
    
    # Configuration and data classes
    "WorkspaceConfig",
    "DeveloperInfo",
    "WorkspaceInfo",
    
    # NEW: Unified validation system
    "WorkspaceTypeDetector",
    "UnifiedValidationCore",
    "ValidationConfig",
    "UnifiedValidationResult",
    "WorkspaceValidationResult",
    "ValidationSummary",
    "ValidationResultBuilder",
    "UnifiedReportGenerator",
    "ReportConfig",
    
    # Legacy compatibility
    "LegacyWorkspaceValidationAdapter",
    "WorkspaceValidationOrchestrator",
    
    # PHASE 3: Consolidated validation system
    "WorkspaceTestManager",
    "TestEnvironment",
    "IsolationReport",
    "TestWorkspaceConfig",
    "TestWorkspaceIsolationManager",
    "TestIsolationConfig",
    "IsolationViolation",
    "IsolationEnvironment",
    "CrossWorkspaceValidator",
    "ComponentConflict",
    "DependencyResolution",
    "ValidationResult",
    "CrossWorkspaceConfig",
    
    # Utility functions
    "create_workspace_manager",
    "validate_workspace_structure",
    "discover_workspaces",
    
    # NEW: Unified validation functions
    "create_single_workspace_result",
    "create_empty_result",
    "generate_unified_report",
    "export_unified_report",
    "create_legacy_adapter",
    "validate_workspace_legacy",
    "validate_all_workspaces_legacy",
    
    # PHASE 3: Consolidated validation functions
    "create_test_workspace_manager",
    "validate_test_workspace_structure",
    "create_test_isolation_manager",
    "validate_test_environment_isolation",
    "create_cross_workspace_validator",
    "validate_cross_workspace_compatibility"
]


def create_workspace_manager(
    workspace_root: str,
    developer_id: str = None,
    config_file: str = None,
    auto_discover: bool = True
) -> WorkspaceManager:
    """
    Convenience function to create a configured WorkspaceManager.
    
    Args:
        workspace_root: Root directory containing developer workspaces
        developer_id: Default developer ID to use
        config_file: Path to workspace configuration file
        auto_discover: Whether to automatically discover workspaces
    
    Returns:
        Configured WorkspaceManager instance
    
    Example:
        manager = create_workspace_manager(
            workspace_root="/path/to/workspaces",
            developer_id="developer_1"
        )
    """
    manager = WorkspaceManager(
        workspace_root=workspace_root,
        config_file=config_file,
        auto_discover=auto_discover
    )
    
    # Set default developer if provided
    if developer_id and not manager.config:
        manager.config = WorkspaceConfig(
            workspace_root=workspace_root,
            developer_id=developer_id
        )
    
    return manager


def validate_workspace_structure(
    workspace_root: str,
    strict: bool = False
) -> tuple[bool, list[str]]:
    """
    Convenience function to validate workspace structure.
    
    Args:
        workspace_root: Root directory to validate
        strict: Whether to apply strict validation rules
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    
    Example:
        is_valid, issues = validate_workspace_structure("/path/to/workspaces")
        if not is_valid:
            print("Validation issues:", issues)
    """
    manager = WorkspaceManager(workspace_root=workspace_root, auto_discover=False)
    return manager.validate_workspace_structure(strict=strict)


def discover_workspaces(workspace_root: str) -> WorkspaceInfo:
    """
    Convenience function to discover workspace structure.
    
    Args:
        workspace_root: Root directory to discover
    
    Returns:
        WorkspaceInfo object with discovered information
    
    Example:
        workspace_info = discover_workspaces("/path/to/workspaces")
        print(f"Found {workspace_info.total_developers} developers")
    """
    manager = WorkspaceManager(workspace_root=workspace_root, auto_discover=True)
    return manager.workspace_info


# Module-level configuration
DEFAULT_WORKSPACE_STRUCTURE = {
    "developers": {
        "description": "Directory containing individual developer workspaces",
        "structure": {
            "{developer_id}": {
                "src": {
                    "cursus_dev": {
                        "steps": {
                            "builders": "Step builder implementations",
                            "contracts": "Script contract definitions", 
                            "specs": "Step specification files",
                            "scripts": "Execution scripts",
                            "configs": "Configuration files"
                        }
                    }
                }
            }
        }
    },
    "shared": {
        "description": "Shared workspace for common components",
        "structure": {
            "src": {
                "cursus_dev": {
                    "steps": {
                        "builders": "Shared step builders",
                        "contracts": "Shared contracts",
                        "specs": "Shared specifications", 
                        "scripts": "Shared scripts",
                        "configs": "Shared configurations"
                    }
                }
            }
        }
    }
}

# Workspace validation rules
VALIDATION_RULES = {
    "required_directories": ["developers", "shared"],
    "developer_structure": [
        "src",
        "src/cursus_dev", 
        "src/cursus_dev/steps"
    ],
    "module_directories": [
        "builders",
        "contracts", 
        "specs",
        "scripts",
        "configs"
    ],
    "required_files": [
        "__init__.py"
    ]
}

# Configuration file templates
CONFIG_TEMPLATES = {
    "basic": {
        "workspace_root": "/path/to/workspaces",
        "developer_id": "developer_1",
        "enable_shared_fallback": True,
        "cache_modules": True,
        "auto_create_structure": False,
        "validation_settings": {}
    },
    "multi_developer": {
        "workspace_root": "/path/to/workspaces", 
        "enable_shared_fallback": True,
        "cache_modules": True,
        "auto_create_structure": True,
        "validation_settings": {
            "strict_validation": False,
            "require_all_module_types": False,
            "validate_imports": True
        }
    }
}


def get_config_template(template_name: str = "basic") -> dict:
    """
    Get a configuration template for workspace setup.
    
    Args:
        template_name: Name of template ("basic" or "multi_developer")
    
    Returns:
        Dictionary containing configuration template
    
    Example:
        config = get_config_template("multi_developer")
        config["workspace_root"] = "/my/workspace/path"
    """
    if template_name not in CONFIG_TEMPLATES:
        raise ValueError(f"Unknown template: {template_name}")
    
    return CONFIG_TEMPLATES[template_name].copy()


def get_workspace_structure_info() -> dict:
    """
    Get information about expected workspace structure.
    
    Returns:
        Dictionary describing workspace structure requirements
    
    Example:
        structure_info = get_workspace_structure_info()
        print(structure_info["developers"]["description"])
    """
    return DEFAULT_WORKSPACE_STRUCTURE.copy()

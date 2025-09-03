"""
Cursus Workspace Package

High-level workspace management API and utilities for the Cursus system.
Provides a unified interface for all workspace operations, abstracting the
complexity of the underlying consolidated workspace management system.

This package serves as the primary entry point for workspace operations,
offering:
- Unified WorkspaceAPI for all workspace management tasks
- CLI commands for workspace operations
- Workspace templates and scaffolding utilities
- High-level utilities for common workspace tasks

The API is built on top of the consolidated Phase 1-3 architecture:
- Phase 1: Foundation Consolidation (WorkspaceManager with specialized managers)
- Phase 2: Pipeline Assembly Layer Optimization (enhanced pipeline components)
- Phase 3: Validation System Consolidation (advanced validation and testing)

Usage:
    from cursus.workspace import WorkspaceAPI
    
    # Initialize the unified API
    api = WorkspaceAPI()
    
    # Developer operations
    result = api.setup_developer_workspace("developer_1", "ml_template")
    pipeline = api.build_cross_workspace_pipeline(pipeline_spec)
    report = api.validate_workspace_components("developer_1")
    
    # Administrative operations
    workspaces = api.list_workspaces()
    health = api.get_workspace_health("developer_1")
    cleanup_result = api.cleanup_inactive_workspaces()
"""

from .api import (
    WorkspaceAPI,
    WorkspaceSetupResult,
    ValidationReport,
    PromotionResult,
    HealthReport,
    CleanupReport,
    WorkspaceInfo
)

from .templates import (
    WorkspaceTemplate,
    TemplateManager
)

# PHASE 1 CLEANUP: Removed commented-out utility imports
# These functions will be implemented when needed or removed if not required

# Layer imports for consolidated architecture (Phase 5)
from . import core
from . import validation

# Public API
__all__ = [
    # Core API
    "WorkspaceAPI",
    
    # Result and data classes
    "WorkspaceSetupResult",
    "ValidationReport", 
    "PromotionResult",
    "HealthReport",
    "CleanupReport",
    "WorkspaceInfo",
    
    # Template system
    "WorkspaceTemplate",
    "TemplateManager",
    
    # PHASE 1 CLEANUP: Removed commented-out utility function exports
    # These will be added when implemented or removed if not needed
    
    # Consolidated layers (Phase 5)
    "core",
    "validation"
]

# Default configuration
DEFAULT_CONFIG = {
    "workspace_root": None,  # Will be auto-detected or set by user
    "auto_discover": True,
    "enable_validation": True,
    "enable_isolation": True,
    "cache_components": True,
    "log_level": "INFO"
}

def get_default_config():
    """Get default configuration for workspace API."""
    return DEFAULT_CONFIG.copy()

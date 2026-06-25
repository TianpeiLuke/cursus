"""
Core utilities for cursus.

This module provides utility functions and classes that support the core
functionality of cursus, including path resolution, configuration management,
and other common operations.
"""

from .hybrid_path_resolution import (
    HybridPathResolver,
    resolve_hybrid_path,
    get_hybrid_resolution_metrics,
    HybridResolutionConfig,
)

from .generic_path_discovery import (
    find_project_folder_generic,
    get_generic_discovery_metrics,
)

from .project_discovery import (
    discover_pipeline_projects,
    summarize_project,
    ProjectInfo,
    ConfigSummary,
)

from .nvme_security import install_nvme_aware_security_patch

__all__ = [
    "HybridPathResolver",
    "resolve_hybrid_path",
    "get_hybrid_resolution_metrics",
    "HybridResolutionConfig",
    "find_project_folder_generic",
    "get_generic_discovery_metrics",
    "discover_pipeline_projects",
    "summarize_project",
    "ProjectInfo",
    "ConfigSummary",
    "install_nvme_aware_security_patch",
]

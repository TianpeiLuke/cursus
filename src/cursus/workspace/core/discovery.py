"""
Cross-workspace component discovery and dependency management.

This module provides the WorkspaceDiscoveryManager with sophisticated cross-workspace
capabilities while leveraging the unified step catalog system for component discovery.

DESIGN PRINCIPLES:
- Separation of Concerns: Workspace business logic separate from step catalog discovery
- Single Responsibility: Focused on cross-workspace coordination and dependency analysis
- Explicit Dependencies: Clear integration with step catalog for discovery operations
"""

from ...step_catalog.adapters.workspace_discovery import WorkspaceDiscoveryManagerAdapter as WorkspaceDiscoveryManager

# Import the workspace-specific data structures
from .inventory import ComponentInventory
from .dependency_graph import DependencyGraph

# Export the classes for backward compatibility
__all__ = [
    'WorkspaceDiscoveryManager',
    'ComponentInventory', 
    'DependencyGraph'
]

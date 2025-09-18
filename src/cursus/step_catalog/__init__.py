"""
Unified Step Catalog System

This module provides a unified interface for discovering and retrieving step-related
components (scripts, contracts, specifications, builders, configs) across multiple workspaces.

The system consolidates 16+ fragmented discovery mechanisms into a single, efficient
StepCatalog class that provides O(1) lookups and intelligent component discovery.
"""

import os
from pathlib import Path
from typing import Any, Optional

from .step_catalog import StepCatalog
from .models import StepInfo, FileMetadata, StepSearchResult
from .config_discovery import ConfigAutoDiscovery
from .adapters import (
    ContractDiscoveryEngineAdapter,
    ContractDiscoveryManagerAdapter,
    ContractDiscoveryResult,
    FlexibleFileResolverAdapter,
    DeveloperWorkspaceFileResolverAdapter,
    WorkspaceDiscoveryManagerAdapter,
    HybridFileResolverAdapter,
    LegacyDiscoveryWrapper,
)

__all__ = [
    "StepCatalog",
    "StepInfo", 
    "FileMetadata",
    "StepSearchResult",
    "ConfigAutoDiscovery",
    "create_step_catalog",
    # Legacy adapters
    "ContractDiscoveryEngineAdapter",
    "ContractDiscoveryManagerAdapter",
    "ContractDiscoveryResult",
    "FlexibleFileResolverAdapter",
    "DeveloperWorkspaceFileResolverAdapter",
    "WorkspaceDiscoveryManagerAdapter",
    "HybridFileResolverAdapter",
    "LegacyDiscoveryWrapper",
]


def create_step_catalog(workspace_root: Path, use_unified: Optional[bool] = None) -> Any:
    """
    Factory function for step catalog with feature flag support.
    
    Args:
        workspace_root: Root directory of the workspace
        use_unified: Whether to use unified catalog (None = check environment)
        
    Returns:
        StepCatalog instance or legacy wrapper based on feature flag
    """
    if use_unified is None:
        use_unified = os.getenv('USE_UNIFIED_CATALOG', 'false').lower() == 'true'
    
    if use_unified:
        return StepCatalog(workspace_root)
    else:
        # Return legacy wrapper with backward compatibility adapters
        return LegacyDiscoveryWrapper(workspace_root)


def get_rollout_percentage() -> int:
    """Get current rollout percentage from environment."""
    return int(os.getenv('UNIFIED_CATALOG_ROLLOUT', '0'))


def should_use_unified_catalog() -> bool:
    """Determine if request should use unified catalog based on rollout percentage."""
    import random
    rollout_percentage = get_rollout_percentage()
    return random.random() < (rollout_percentage / 100.0)


def create_step_catalog_with_rollout(workspace_root: Path) -> Any:
    """
    Factory function with gradual rollout support.
    
    Args:
        workspace_root: Root directory of the workspace
        
    Returns:
        StepCatalog or LegacyDiscoveryWrapper based on rollout percentage
    """
    # Check explicit feature flag first
    explicit_flag = os.getenv('USE_UNIFIED_CATALOG')
    if explicit_flag is not None:
        use_unified = explicit_flag.lower() == 'true'
        return create_step_catalog(workspace_root, use_unified)
    
    # Use rollout percentage for gradual deployment
    use_unified = should_use_unified_catalog()
    return create_step_catalog(workspace_root, use_unified)

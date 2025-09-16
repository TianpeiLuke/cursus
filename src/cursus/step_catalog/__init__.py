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

__all__ = [
    "StepCatalog",
    "StepInfo", 
    "FileMetadata",
    "StepSearchResult",
    "ConfigAutoDiscovery",
    "create_step_catalog",
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
        # For now, always return unified catalog during development
        # Legacy wrapper will be implemented in Phase 2
        return StepCatalog(workspace_root)

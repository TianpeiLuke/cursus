"""
Core workspace functionality layer.

This module provides the foundational workspace management components
that were consolidated from src.cursus.core.workspace during Phase 5
structural consolidation.
"""

from .manager import WorkspaceManager
from .lifecycle import WorkspaceLifecycleManager
from .discovery import WorkspaceDiscoveryManager
from .integration import WorkspaceIntegrationManager
from .isolation import WorkspaceIsolationManager
from .assembler import WorkspacePipelineAssembler
from .compiler import WorkspaceDAGCompiler
from .config import WorkspaceStepDefinition, WorkspacePipelineDefinition
from .registry import WorkspaceComponentRegistry

__all__ = [
    'WorkspaceManager',
    'WorkspaceLifecycleManager', 
    'WorkspaceDiscoveryManager',
    'WorkspaceIntegrationManager',
    'WorkspaceIsolationManager',
    'WorkspacePipelineAssembler',
    'WorkspaceDAGCompiler',
    'WorkspaceStepDefinition',
    'WorkspacePipelineDefinition',
    'WorkspaceComponentRegistry'
]

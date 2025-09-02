"""
Workspace-aware core system extensions.

This module provides workspace-aware extensions to the core pipeline assembly
and DAG compilation system, enabling support for developer workspace components
while maintaining full backward compatibility.

Phase 1 Implementation: Consolidated workspace management system with specialized
functional managers for lifecycle, isolation, discovery, and integration operations.
"""

from .config import WorkspaceStepDefinition, WorkspacePipelineDefinition
from .registry import WorkspaceComponentRegistry
from .assembler import WorkspacePipelineAssembler
from .compiler import WorkspaceDAGCompiler

# Phase 1: Consolidated workspace management system
from .manager import WorkspaceManager
from .lifecycle import WorkspaceLifecycleManager
from .isolation import WorkspaceIsolationManager
from .discovery import WorkspaceDiscoveryManager
from .integration import WorkspaceIntegrationManager

__all__ = [
    # Legacy workspace-aware components (backward compatibility)
    'WorkspaceStepDefinition',
    'WorkspacePipelineDefinition',
    'WorkspaceComponentRegistry',
    'WorkspacePipelineAssembler',
    'WorkspaceDAGCompiler',
    
    # Phase 1: Consolidated workspace management system
    'WorkspaceManager',
    'WorkspaceLifecycleManager',
    'WorkspaceIsolationManager',
    'WorkspaceDiscoveryManager',
    'WorkspaceIntegrationManager'
]

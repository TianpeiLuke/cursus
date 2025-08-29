"""
Workspace-aware core system extensions.

This module provides workspace-aware extensions to the core pipeline assembly
and DAG compilation system, enabling support for developer workspace components
while maintaining full backward compatibility.
"""

from .config import WorkspaceStepConfig, WorkspacePipelineConfig
from .registry import WorkspaceComponentRegistry
from .assembler import WorkspacePipelineAssembler
from .compiler import WorkspaceDAGCompiler

__all__ = [
    'WorkspaceStepConfig',
    'WorkspacePipelineConfig', 
    'WorkspaceComponentRegistry',
    'WorkspacePipelineAssembler',
    'WorkspaceDAGCompiler'
]

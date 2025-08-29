"""
Workspace-aware core system extensions.

This module provides workspace-aware extensions to the core pipeline assembly
and DAG compilation system, enabling support for developer workspace components
while maintaining full backward compatibility.
"""

from .config import WorkspaceStepDefinition, WorkspacePipelineDefinition
from .registry import WorkspaceComponentRegistry
from .assembler import WorkspacePipelineAssembler
from .compiler import WorkspaceDAGCompiler

__all__ = [
    'WorkspaceStepDefinition',
    'WorkspacePipelineDefinition',
    'WorkspaceComponentRegistry',
    'WorkspacePipelineAssembler',
    'WorkspaceDAGCompiler'
]

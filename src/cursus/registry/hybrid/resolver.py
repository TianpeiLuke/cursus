"""
Simplified Hybrid Registry Conflict Resolution

This module provides simple conflict resolution for the hybrid registry system,
focusing on workspace priority resolution without over-engineering.
"""

import logging
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from .models import (
    StepDefinition,
    NamespacedStepDefinition,
    ResolutionContext,
    StepResolutionResult,
    ConflictAnalysis
)

logger = logging.getLogger(__name__)


class SimpleConflictResolver:
    """
    Simplified conflict resolver focusing on workspace priority resolution.
    
    Eliminates over-engineering by focusing only on actual needs:
    - Simple workspace priority resolution
    - Basic conflict detection
    - No theoretical framework/environment resolution
    """
    
    def __init__(self):
        pass
    
    def resolve_step_conflict(self, step_name: str, context: ResolutionContext,
                            core_steps: Dict[str, StepDefinition],
                            local_steps: Dict[str, Dict[str, NamespacedStepDefinition]]) -> StepResolutionResult:
        """
        Simple step conflict resolution using workspace priority.
        
        Resolution logic:
        1. If workspace_id specified and step exists in that workspace, use it
        2. Otherwise, use core registry
        3. If step not found anywhere, return error
        """
        
        # Check workspace-specific registry first if workspace_id provided
        if context.workspace_id and context.workspace_id in local_steps:
            workspace_steps = local_steps[context.workspace_id]
            if step_name in workspace_steps:
                step_def = workspace_steps[step_name]
                return StepResolutionResult(
                    step_definition=step_def,
                    source_registry=context.workspace_id,
                    workspace_id=context.workspace_id,
                    resolution_strategy="workspace_priority",
                    conflict_detected=False,
                    conflict_analysis=None,
                    errors=[],
                    warnings=[]
                )
        
        # Fallback to core registry
        if step_name in core_steps:
            step_def = core_steps[step_name]
            return StepResolutionResult(
                step_definition=step_def,
                source_registry="core",
                workspace_id=context.workspace_id,
                resolution_strategy="workspace_priority",
                conflict_detected=False,
                conflict_analysis=None,
                errors=[],
                warnings=[]
            )
        
        # Step not found
        return StepResolutionResult(
            step_definition=None,
            source_registry="none",
            workspace_id=context.workspace_id,
            resolution_strategy="workspace_priority",
            conflict_detected=False,
            conflict_analysis=None,
            errors=[f"Step '{step_name}' not found in any registry"],
            warnings=[]
        )
    
    def detect_step_conflicts(self, core_steps: Dict[str, StepDefinition],
                            local_steps: Dict[str, Dict[str, NamespacedStepDefinition]]) -> Dict[str, List[str]]:
        """
        Simple conflict detection - identify steps defined in multiple registries.
        
        Returns:
            Dictionary mapping step names to list of registries that define them
        """
        conflicts = {}
        
        # Check each step in local registries
        for workspace_id, workspace_steps in local_steps.items():
            for step_name in workspace_steps.keys():
                sources = []
                
                # Check if also in core
                if step_name in core_steps:
                    sources.append("core")
                
                # Check if in other workspaces
                for other_workspace_id, other_workspace_steps in local_steps.items():
                    if other_workspace_id != workspace_id and step_name in other_workspace_steps:
                        sources.append(other_workspace_id)
                
                if sources:
                    sources.append(workspace_id)  # Add current workspace
                    conflicts[step_name] = sources
        
        return conflicts


# Simplified resolver instance for backward compatibility
def get_simple_resolver() -> SimpleConflictResolver:
    """Get a simple conflict resolver instance."""
    return SimpleConflictResolver()


# Legacy compatibility - simplified versions of complex classes
class ConflictResolver(SimpleConflictResolver):
    """Legacy compatibility wrapper for SimpleConflictResolver."""
    pass


class StepResolver:
    """Simplified step resolver without over-engineered dependency analysis."""
    
    def __init__(self):
        self._resolver = SimpleConflictResolver()
    
    def resolve_step(self, step_name: str, context: ResolutionContext,
                    core_steps: Dict[str, StepDefinition],
                    local_steps: Dict[str, Dict[str, NamespacedStepDefinition]]) -> StepResolutionResult:
        """Resolve a single step using simple workspace priority."""
        return self._resolver.resolve_step_conflict(step_name, context, core_steps, local_steps)

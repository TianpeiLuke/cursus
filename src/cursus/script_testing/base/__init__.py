"""
Script Testing Base Module

This module contains the base classes for DAG-guided script testing,
including specifications, execution plans, and test results.

These classes mirror the base classes in cursus/core but target script execution
with maximum component reuse from existing cursus infrastructure.
"""

from .script_execution_spec import ScriptExecutionSpec
from .script_execution_plan import ScriptExecutionPlan
from .script_test_result import ScriptTestResult

# Direct imports from existing cursus components for convenience
from ...api.dag.base_dag import PipelineDAG
from ...step_catalog import StepCatalog

__all__ = [
    # Core base classes
    "ScriptExecutionSpec",
    "ScriptExecutionPlan", 
    "ScriptTestResult",
    
    # Reused components (for convenience)
    "PipelineDAG",
    "StepCatalog",
]

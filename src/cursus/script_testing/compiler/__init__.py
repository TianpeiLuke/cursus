"""
Script Testing Compiler Module

Provides script compilation components that mirror cursus/core/compiler patterns.
Uses maximum component reuse from existing cursus infrastructure.

This module contains:
- ScriptDAGCompiler: Compiles PipelineDAG to ScriptExecutionPlan
- ScriptExecutionTemplate: Dynamic template for script execution plans
- Validation functions: DAG and script validation for compilation
- Exception classes: Script compilation error handling
"""

from .script_dag_compiler import ScriptDAGCompiler
from .script_execution_template import ScriptExecutionTemplate
from .validation import (
    validate_dag_for_script_execution,
    validate_script_structure,
    validate_execution_plan,
    validate_script_spec,
)
from .exceptions import (
    ScriptCompilationError,
    ScriptDiscoveryError,
    ScriptValidationError,
    DAGValidationError,
    ExecutionPlanValidationError,
    StepCatalogIntegrationError,
    InteractiveInputError,
    format_compilation_error,
)

# Main compiler API - mirrors cursus/core/compiler patterns
__all__ = [
    # Core compiler classes
    "ScriptDAGCompiler",
    "ScriptExecutionTemplate",
    
    # Validation functions
    "validate_dag_for_script_execution",
    "validate_script_structure", 
    "validate_execution_plan",
    "validate_script_spec",
    
    # Exception classes
    "ScriptCompilationError",
    "ScriptDiscoveryError",
    "ScriptValidationError",
    "DAGValidationError",
    "ExecutionPlanValidationError",
    "StepCatalogIntegrationError",
    "InteractiveInputError",
    "format_compilation_error",
]

# Convenience functions that mirror cursus/core/compiler API patterns
def compile_dag_to_script_execution_plan(
    dag,
    test_workspace_dir,
    step_catalog=None,
    user_inputs=None,
    collect_inputs=False,
    **kwargs
):
    """
    Convenience function to compile DAG to script execution plan.
    
    This function mirrors the convenience functions in cursus/core/compiler
    but targets script execution instead of pipeline compilation.
    
    Args:
        dag: PipelineDAG to compile
        test_workspace_dir: Test workspace directory
        step_catalog: Optional step catalog for enhanced discovery
        user_inputs: Optional pre-collected user inputs
        collect_inputs: Whether to collect inputs interactively
        **kwargs: Additional compiler configuration
        
    Returns:
        ScriptExecutionPlan ready for execution
        
    Raises:
        ScriptCompilationError: If compilation fails
    """
    compiler = ScriptDAGCompiler(
        dag=dag,
        test_workspace_dir=test_workspace_dir,
        step_catalog=step_catalog,
        **kwargs
    )
    
    return compiler.compile_dag_to_execution_plan(
        user_inputs=user_inputs,
        collect_inputs=collect_inputs
    )


def create_script_execution_template(
    dag,
    user_inputs,
    test_workspace_dir,
    step_catalog=None,
    **kwargs
):
    """
    Convenience function to create script execution template.
    
    Args:
        dag: PipelineDAG for template
        user_inputs: User inputs for script execution
        test_workspace_dir: Test workspace directory
        step_catalog: Optional step catalog for enhanced discovery
        **kwargs: Additional template configuration
        
    Returns:
        ScriptExecutionTemplate ready for plan generation
    """
    return ScriptExecutionTemplate(
        dag=dag,
        user_inputs=user_inputs,
        test_workspace_dir=test_workspace_dir,
        step_catalog=step_catalog,
        **kwargs
    )


def preview_script_compilation(
    dag,
    test_workspace_dir,
    step_catalog=None,
    **kwargs
):
    """
    Preview script compilation without creating full execution plan.
    
    Args:
        dag: PipelineDAG to preview
        test_workspace_dir: Test workspace directory
        step_catalog: Optional step catalog for enhanced discovery
        **kwargs: Additional compiler configuration
        
    Returns:
        Dictionary with compilation preview information
    """
    compiler = ScriptDAGCompiler(
        dag=dag,
        test_workspace_dir=test_workspace_dir,
        step_catalog=step_catalog,
        **kwargs
    )
    
    return compiler.preview_compilation()

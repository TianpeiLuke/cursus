"""
DAG-Guided Script Testing Engine

This module provides a comprehensive script testing framework that mirrors the 
proven patterns from cursus/core, enabling DAG-guided end-to-end testing with 
intelligent dependency resolution and step catalog integration.

Main API Functions:
    compile_dag_to_script_execution: Simple API for DAG-guided script testing
    
Core Components:
    ScriptDAGCompiler: Compiles DAGs to script execution plans
    ScriptAssembler: Executes scripts with dependency resolution
    InteractiveScriptTestingFactory: Interactive input collection
"""

from typing import Optional, Dict, Any
from pathlib import Path

# Import core components
from .compiler.script_dag_compiler import ScriptDAGCompiler
from .assembler.script_assembler import ScriptAssembler
from .factory.interactive_script_factory import InteractiveScriptTestingFactory
from .base.script_execution_spec import ScriptExecutionSpec
from .base.script_execution_plan import ScriptExecutionPlan
from .base.script_test_result import ScriptTestResult

# Direct imports from existing cursus components - MAXIMUM REUSE
from ..api.dag.base_dag import PipelineDAG
from ..step_catalog import StepCatalog
from ..core.deps.dependency_resolver import create_dependency_resolver
from ..core.deps.factory import create_pipeline_components
from ..registry.step_names import get_step_name_from_spec_type, get_spec_step_type


def compile_dag_to_script_execution(
    dag: PipelineDAG,
    test_workspace_dir: str = "test/integration/script_testing",
    step_catalog: Optional[StepCatalog] = None,
    collect_inputs: bool = True,
    **kwargs: Any,
) -> ScriptExecutionPlan:
    """
    Compile a PipelineDAG to a script execution plan with optional interactive input collection.
    
    This is the main entry point for users who want simple, one-call compilation 
    from DAG to script execution plan, mirroring compile_dag_to_pipeline.
    
    Args:
        dag: PipelineDAG instance defining the pipeline structure
        test_workspace_dir: Directory for test workspace and script discovery
        step_catalog: Optional step catalog for enhanced script discovery
        collect_inputs: Whether to collect user inputs interactively
        **kwargs: Additional arguments passed to compiler
        
    Returns:
        ScriptExecutionPlan ready for execution
        
    Raises:
        ValueError: If DAG nodes don't have corresponding scripts
        FileNotFoundError: If test workspace directory doesn't exist
        
    Example:
        >>> from cursus.script_testing import compile_dag_to_script_execution
        >>> from cursus.api.dag.base_dag import PipelineDAG
        >>> 
        >>> dag = PipelineDAG.from_json("configs/xgboost_training.json")
        >>> execution_plan = compile_dag_to_script_execution(
        ...     dag=dag,
        ...     test_workspace_dir="test/integration/script_testing",
        ...     collect_inputs=True
        ... )
        >>> results = execution_plan.execute()
        >>> print(f"Pipeline success: {results['pipeline_success']}")
    """
    try:
        # Validate inputs
        if not isinstance(dag, PipelineDAG):
            raise ValueError("dag must be a PipelineDAG instance")
            
        if not dag.nodes:
            raise ValueError("DAG must contain at least one node")
            
        # Ensure test workspace directory exists
        workspace_path = Path(test_workspace_dir)
        workspace_path.mkdir(parents=True, exist_ok=True)
        
        # Create step catalog if not provided - REUSE EXISTING COMPONENT
        if step_catalog is None:
            step_catalog = StepCatalog()
        
        # Create compiler with step catalog integration - REUSE EXISTING PATTERNS
        compiler = ScriptDAGCompiler(
            dag=dag,
            test_workspace_dir=test_workspace_dir,
            step_catalog=step_catalog,
            **kwargs
        )
        
        # Compile DAG to execution plan
        execution_plan = compiler.compile_dag_to_execution_plan(
            collect_inputs=collect_inputs
        )
        
        return execution_plan
        
    except Exception as e:
        raise RuntimeError(f"Failed to compile DAG to script execution plan: {e}") from e


def create_script_testing_components(context_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Create script testing components using existing cursus infrastructure.
    
    This function directly reuses create_pipeline_components from cursus/core
    to ensure maximum component reuse and consistency.
    
    Args:
        context_name: Optional context name for component creation
        
    Returns:
        Dictionary with script testing components
    """
    # DIRECT REUSE: Use existing pipeline components factory
    components = create_pipeline_components(context_name)
    
    # Add script-specific components
    components.update({
        "step_catalog": StepCatalog(),
        "dependency_resolver": create_dependency_resolver(),
    })
    
    return components


# Export main API components
__all__ = [
    # Main API function
    "compile_dag_to_script_execution",
    "create_script_testing_components",
    
    # Core components
    "ScriptDAGCompiler",
    "ScriptAssembler", 
    "InteractiveScriptTestingFactory",
    
    # Base classes
    "ScriptExecutionSpec",
    "ScriptExecutionPlan",
    "ScriptTestResult",
    
    # Reused components (for convenience)
    "PipelineDAG",
    "StepCatalog",
    "create_dependency_resolver",
    "get_step_name_from_spec_type",
    "get_spec_step_type",
]

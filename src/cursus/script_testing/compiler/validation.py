"""
Script Compilation Validation

Provides validation functions for script compilation and DAG validation.
Uses maximum component reuse from existing cursus infrastructure.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import ast
import logging

# Direct imports from existing cursus components - MAXIMUM REUSE
from ...api.dag.base_dag import PipelineDAG
from ...step_catalog import StepCatalog
from ..base.script_execution_spec import ScriptExecutionSpec
from ..base.script_execution_plan import ScriptExecutionPlan
from .exceptions import (
    DAGValidationError,
    ScriptValidationError,
    ScriptDiscoveryError,
    ExecutionPlanValidationError,
)

logger = logging.getLogger(__name__)


def validate_dag_for_script_execution(
    dag: PipelineDAG,
    test_workspace_dir: str,
    step_catalog: Optional[StepCatalog] = None,
) -> Dict[str, Any]:
    """
    Validate DAG for script execution compatibility.
    
    This function validates that a PipelineDAG is suitable for script execution,
    checking for cycles, node validity, and script availability.
    
    Args:
        dag: PipelineDAG to validate
        test_workspace_dir: Test workspace directory for script discovery
        step_catalog: Optional step catalog for enhanced validation
        
    Returns:
        Dictionary with validation results:
        {
            "valid": bool,
            "errors": List[str],
            "warnings": List[str],
            "node_analysis": Dict[str, Dict[str, Any]]
        }
    """
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "node_analysis": {},
    }
    
    try:
        # 1. Validate DAG structure (DIRECT REUSE of DAG validation)
        _validate_dag_structure(dag, validation_result)
        
        # 2. Validate nodes for script execution
        _validate_nodes_for_script_execution(
            dag, test_workspace_dir, step_catalog, validation_result
        )
        
        # 3. Validate dependencies
        _validate_dag_dependencies(dag, validation_result)
        
        # Set overall validity
        validation_result["valid"] = len(validation_result["errors"]) == 0
        
        logger.info(f"DAG validation completed: valid={validation_result['valid']}, "
                   f"errors={len(validation_result['errors'])}, "
                   f"warnings={len(validation_result['warnings'])}")
        
    except Exception as e:
        validation_result["valid"] = False
        validation_result["errors"].append(f"Validation error: {str(e)}")
        logger.error(f"DAG validation failed with exception: {e}")
    
    return validation_result


def _validate_dag_structure(dag: PipelineDAG, validation_result: Dict[str, Any]) -> None:
    """
    Validate basic DAG structure.
    
    Uses direct reuse of DAG validation methods.
    """
    # Check for empty DAG
    if not dag.nodes:
        validation_result["errors"].append("DAG has no nodes")
        return
    
    # Check for cycles using DAG's built-in validation (DIRECT REUSE)
    try:
        execution_order = dag.topological_sort()
        logger.debug(f"DAG topological sort successful: {execution_order}")
    except ValueError as e:
        validation_result["errors"].append(f"DAG contains cycles: {str(e)}")
        return
    
    # Validate node names
    for node_name in dag.nodes:
        if not node_name or not isinstance(node_name, str):
            validation_result["errors"].append(f"Invalid node name: {node_name}")
        elif not node_name.strip():
            validation_result["errors"].append("Empty node name found")
    
    # Validate edges
    for edge in dag.edges:
        if len(edge) != 2:
            validation_result["errors"].append(f"Invalid edge format: {edge}")
            continue
            
        source, target = edge
        if source not in dag.nodes:
            validation_result["errors"].append(f"Edge source '{source}' not in nodes")
        if target not in dag.nodes:
            validation_result["errors"].append(f"Edge target '{target}' not in nodes")


def _validate_nodes_for_script_execution(
    dag: PipelineDAG,
    test_workspace_dir: str,
    step_catalog: Optional[StepCatalog],
    validation_result: Dict[str, Any],
) -> None:
    """
    Validate that all DAG nodes can be resolved to executable scripts.
    """
    for node_name in dag.nodes:
        node_analysis = {
            "script_discovered": False,
            "script_path": None,
            "script_exists": False,
            "script_valid": False,
            "discovery_method": None,
            "validation_errors": [],
        }
        
        try:
            # Try script discovery using step catalog first (DIRECT REUSE)
            script_path = None
            discovery_method = None
            
            if step_catalog:
                try:
                    step_info = step_catalog.resolve_pipeline_node(node_name)
                    if step_info and step_info.file_components.get('script'):
                        script_metadata = step_info.file_components['script']
                        script_path = str(script_metadata.path)
                        discovery_method = "step_catalog"
                        node_analysis["script_discovered"] = True
                except Exception as e:
                    logger.debug(f"Step catalog discovery failed for {node_name}: {e}")
            
            # Fallback to traditional discovery
            if not script_path:
                script_name = ScriptExecutionSpec._convert_step_name_to_script_name(node_name)
                script_path = str(Path(test_workspace_dir) / "scripts" / f"{script_name}.py")
                discovery_method = "fallback"
                node_analysis["script_discovered"] = True
            
            node_analysis["script_path"] = script_path
            node_analysis["discovery_method"] = discovery_method
            
            # Check if script exists
            if script_path and Path(script_path).exists():
                node_analysis["script_exists"] = True
                
                # Validate script structure
                script_validation = validate_script_structure(script_path)
                node_analysis["script_valid"] = script_validation["valid"]
                node_analysis["validation_errors"] = script_validation.get("errors", [])
                
                if not script_validation["valid"]:
                    validation_result["warnings"].append(
                        f"Script validation issues for {node_name}: "
                        f"{'; '.join(script_validation['errors'])}"
                    )
            else:
                validation_result["warnings"].append(
                    f"Script not found for node '{node_name}': {script_path}"
                )
                
        except Exception as e:
            node_analysis["validation_errors"].append(str(e))
            validation_result["warnings"].append(
                f"Error analyzing node '{node_name}': {str(e)}"
            )
        
        validation_result["node_analysis"][node_name] = node_analysis


def _validate_dag_dependencies(dag: PipelineDAG, validation_result: Dict[str, Any]) -> None:
    """
    Validate DAG dependencies for script execution.
    
    Uses direct reuse of DAG dependency methods.
    """
    try:
        # Check for orphaned nodes (nodes with no dependencies and no dependents)
        orphaned_nodes = []
        for node_name in dag.nodes:
            dependencies = dag.get_dependencies(node_name)  # DIRECT REUSE
            dependents = [n for n in dag.nodes if node_name in dag.get_dependencies(n)]
            
            if not dependencies and not dependents and len(dag.nodes) > 1:
                orphaned_nodes.append(node_name)
        
        if orphaned_nodes:
            validation_result["warnings"].append(
                f"Orphaned nodes found (no dependencies or dependents): {orphaned_nodes}"
            )
        
        # Check for self-dependencies
        for node_name in dag.nodes:
            dependencies = dag.get_dependencies(node_name)
            if node_name in dependencies:
                validation_result["errors"].append(f"Node '{node_name}' depends on itself")
        
    except Exception as e:
        validation_result["warnings"].append(f"Dependency validation error: {str(e)}")


def validate_script_structure(script_path: str) -> Dict[str, Any]:
    """
    Validate script structure for execution compatibility.
    
    Checks for main() function, valid Python syntax, and other requirements.
    
    Args:
        script_path: Path to the script file to validate
        
    Returns:
        Dictionary with validation results:
        {
            "valid": bool,
            "errors": List[str],
            "warnings": List[str],
            "has_main_function": bool,
            "main_function_signature": Optional[str]
        }
    """
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "has_main_function": False,
        "main_function_signature": None,
    }
    
    try:
        script_path_obj = Path(script_path)
        
        # Check if file exists
        if not script_path_obj.exists():
            validation_result["valid"] = False
            validation_result["errors"].append(f"Script file does not exist: {script_path}")
            return validation_result
        
        # Check if file is readable
        if not script_path_obj.is_file():
            validation_result["valid"] = False
            validation_result["errors"].append(f"Path is not a file: {script_path}")
            return validation_result
        
        # Read and parse the script
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                script_content = f.read()
        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Cannot read script file: {str(e)}")
            return validation_result
        
        # Parse Python syntax
        try:
            tree = ast.parse(script_content, filename=script_path)
        except SyntaxError as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Python syntax error: {str(e)}")
            return validation_result
        
        # Check for main() function
        main_function = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'main':
                main_function = node
                validation_result["has_main_function"] = True
                break
        
        if not validation_result["has_main_function"]:
            validation_result["warnings"].append("No main() function found")
        else:
            # Analyze main function signature
            main_signature = _analyze_main_function_signature(main_function)
            validation_result["main_function_signature"] = main_signature
            
            # Validate main function parameters
            if not _validate_main_function_parameters(main_function):
                validation_result["warnings"].append(
                    "main() function parameters may not be compatible with script execution"
                )
        
        # Check for common issues
        _check_common_script_issues(tree, validation_result)
        
        # Set overall validity
        validation_result["valid"] = len(validation_result["errors"]) == 0
        
    except Exception as e:
        validation_result["valid"] = False
        validation_result["errors"].append(f"Script validation error: {str(e)}")
    
    return validation_result


def _analyze_main_function_signature(main_function: ast.FunctionDef) -> str:
    """
    Analyze the main() function signature.
    
    Args:
        main_function: AST node for the main function
        
    Returns:
        String representation of the function signature
    """
    args = []
    
    # Regular arguments
    for arg in main_function.args.args:
        args.append(arg.arg)
    
    # Default arguments
    defaults_count = len(main_function.args.defaults)
    if defaults_count > 0:
        # Mark arguments with defaults
        for i in range(len(args) - defaults_count, len(args)):
            args[i] = f"{args[i]}=..."
    
    # Keyword-only arguments
    for arg in main_function.args.kwonlyargs:
        args.append(f"{arg.arg}=...")
    
    # *args and **kwargs
    if main_function.args.vararg:
        args.append(f"*{main_function.args.vararg.arg}")
    if main_function.args.kwarg:
        args.append(f"**{main_function.args.kwarg.arg}")
    
    return f"main({', '.join(args)})"


def _validate_main_function_parameters(main_function: ast.FunctionDef) -> bool:
    """
    Validate that main() function parameters are compatible with script execution.
    
    Expected signature patterns:
    - main()
    - main(input_paths, output_paths, environ_vars, job_args)
    - main(**kwargs)
    - main(input_paths, output_paths, **kwargs)
    """
    args = main_function.args
    arg_names = [arg.arg for arg in args.args]
    
    # Allow no parameters
    if not arg_names and not args.kwarg:
        return True
    
    # Allow **kwargs pattern
    if args.kwarg and not arg_names:
        return True
    
    # Check for expected parameter names
    expected_params = {"input_paths", "output_paths", "environ_vars", "job_args"}
    
    # Allow if all parameters are in expected set
    if all(name in expected_params for name in arg_names):
        return True
    
    # Allow if has **kwargs for additional flexibility
    if args.kwarg:
        return True
    
    return False


def _check_common_script_issues(tree: ast.AST, validation_result: Dict[str, Any]) -> None:
    """
    Check for common script issues that might affect execution.
    """
    # Check for relative imports that might cause issues
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level > 0:
            validation_result["warnings"].append(
                f"Relative import found: may cause issues in script execution context"
            )
            break
    
    # Check for __name__ == "__main__" pattern
    has_main_guard = False
    for node in ast.walk(tree):
        if (isinstance(node, ast.If) and 
            isinstance(node.test, ast.Compare) and
            isinstance(node.test.left, ast.Name) and
            node.test.left.id == '__name__'):
            has_main_guard = True
            break
    
    if not has_main_guard:
        validation_result["warnings"].append(
            "No __name__ == '__main__' guard found: script may execute on import"
        )


def validate_execution_plan(execution_plan: ScriptExecutionPlan) -> Dict[str, Any]:
    """
    Validate a script execution plan.
    
    This function validates that a ScriptExecutionPlan is ready for execution,
    checking script availability, execution order, and dependencies.
    
    Args:
        execution_plan: ScriptExecutionPlan to validate
        
    Returns:
        Dictionary with validation results
    """
    # Use the built-in validation method (DIRECT REUSE)
    return execution_plan.validate_execution_plan()


def validate_script_spec(spec: ScriptExecutionSpec) -> Dict[str, Any]:
    """
    Validate a script execution specification.
    
    Args:
        spec: ScriptExecutionSpec to validate
        
    Returns:
        Dictionary with validation results
    """
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": [],
    }
    
    try:
        # Validate script path
        if not spec.script_path:
            validation_result["errors"].append("Script path is empty")
        elif not Path(spec.script_path).exists():
            validation_result["warnings"].append(f"Script file does not exist: {spec.script_path}")
        
        # Validate script name
        if not spec.script_name:
            validation_result["errors"].append("Script name is empty")
        
        # Validate step name
        if not spec.step_name:
            validation_result["errors"].append("Step name is empty")
        
        # Validate paths
        for path_type, paths in [("input", spec.input_paths), ("output", spec.output_paths)]:
            for logical_name, path in paths.items():
                if not logical_name:
                    validation_result["warnings"].append(f"Empty {path_type} path logical name")
                if not path:
                    validation_result["warnings"].append(f"Empty {path_type} path for '{logical_name}'")
        
        # Use built-in path validation (DIRECT REUSE)
        path_validation = spec.validate_paths_exist(check_inputs=False, check_outputs=False)
        if not path_validation["script_exists"]:
            validation_result["warnings"].append("Script file does not exist")
        
        validation_result["valid"] = len(validation_result["errors"]) == 0
        
    except Exception as e:
        validation_result["valid"] = False
        validation_result["errors"].append(f"Spec validation error: {str(e)}")
    
    return validation_result

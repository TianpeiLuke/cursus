"""
Simplified Script Testing API

This module provides a streamlined script testing framework that extends existing
cursus infrastructure instead of reimplementing it. The approach eliminates 
over-engineering by directly reusing DAGConfigFactory, StepCatalog, and 
UnifiedDependencyResolver components.

Key Functions:
    run_dag_scripts: Main entry point for DAG-guided script testing
    execute_single_script: Execute individual scripts with dependency management
    install_script_dependencies: Handle package dependencies (valid complexity)
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import subprocess
import sys
import ast
import importlib.util
import logging

# Direct reuse of existing cursus infrastructure
from ...api.dag.base_dag import PipelineDAG
from ...api.factory.dag_config_factory import DAGConfigFactory
from ...step_catalog import StepCatalog
from ...core.deps.dependency_resolver import create_dependency_resolver
from ...steps.configs.utils import load_configs, build_complete_config_classes

logger = logging.getLogger(__name__)


class ScriptTestResult:
    """Simple result model for script execution."""
    
    def __init__(self, success: bool, output_files: Optional[Dict[str, str]] = None, 
                 error_message: Optional[str] = None, execution_time: Optional[float] = None):
        self.success = success
        self.output_files = output_files or {}
        self.error_message = error_message
        self.execution_time = execution_time


def run_dag_scripts(
    dag: PipelineDAG,
    config_path: str,
    test_workspace_dir: str = "test/integration/script_testing",
    step_catalog: Optional[StepCatalog] = None,
    use_dependency_resolution: bool = True
) -> Dict[str, Any]:
    """
    ENHANCED: Run scripts with ScriptExecutionRegistry integration and message passing.

    This function now uses the ScriptExecutionRegistry for central state coordination,
    enabling intelligent dependency resolution and automatic message passing between
    script executions.

    Args:
        dag: PipelineDAG instance defining the pipeline structure
        config_path: Path to pipeline configuration JSON file for script validation
        test_workspace_dir: Directory for test workspace and script discovery
        step_catalog: Optional StepCatalog instance (will create if not provided)
        use_dependency_resolution: Whether to use two-phase dependency resolution

    Returns:
        Dictionary with execution results and metadata

    Example:
        >>> from cursus.validation.script_testing import run_dag_scripts
        >>> from cursus.api.dag.base_dag import PipelineDAG
        >>>
        >>> dag = PipelineDAG.from_json("configs/xgboost_training.json")
        >>> results = run_dag_scripts(
        ...     dag=dag,
        ...     config_path="pipeline_config/config_NA_xgboost_AtoZ.json"
        ... )
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

        # Initialize step catalog
        if not step_catalog:
            step_catalog = StepCatalog()

        logger.info(f"Starting DAG-guided script testing with {len(dag.nodes)} nodes")

        # ENHANCED: Use ScriptExecutionRegistry for state coordination
        from .script_execution_registry import create_script_execution_registry
        registry = create_script_execution_registry(dag, step_catalog)

        # REGISTRY-ONLY: Use registry for both dependency resolution modes
        if use_dependency_resolution:
            from .script_dependency_matcher import resolve_script_dependencies_with_registry
            user_inputs = resolve_script_dependencies_with_registry(dag, config_path, step_catalog, registry)
            logger.info(f"Used registry-coordinated dependency resolution for {len(user_inputs)} scripts")
        else:
            # REGISTRY-ONLY: Use existing registry functions with manual mode (no dependency resolution)
            from .script_dependency_matcher import (
                prepare_script_testing_inputs, 
                collect_user_inputs_with_registry_coordination
            )
            
            # Prepare with empty dependency matches (manual mode)
            prepared_data = prepare_script_testing_inputs(dag, config_path, step_catalog)
            prepared_data['dependency_matches'] = {}  # Clear dependency matches for manual mode
            
            # Initialize registry with manual mode data
            registry.initialize_from_dependency_matcher(prepared_data)
            
            # Use registry coordination but without automatic dependency resolution
            user_inputs = collect_user_inputs_with_registry_coordination(prepared_data, registry)
            logger.info(f"Used registry-coordinated manual input collection for {len(user_inputs)} scripts")

        # ENHANCED: Execute with registry coordination and message passing
        results = execute_scripts_with_registry_coordination(dag, registry)

        # Include registry summary in results
        results['execution_summary'] = registry.get_execution_summary()
        results['message_passing_history'] = registry.get_message_passing_history()

        logger.info(f"Script testing completed. Success: {results['pipeline_success']}")
        return results

    except Exception as e:
        logger.error(f"Script testing failed: {e}")
        raise RuntimeError(f"Failed to test DAG scripts: {e}") from e


def collect_script_inputs_using_dag_factory(dag: PipelineDAG, config_path: str) -> Dict[str, Any]:
    """
    Collect script inputs by extending DAGConfigFactory patterns.
    
    This function reuses the existing 600+ lines of proven interactive collection
    patterns instead of reimplementing them.
    
    Args:
        dag: PipelineDAG instance
        config_path: Path to configuration file for script validation
        
    Returns:
        Dictionary mapping script names to their input configurations
    """
    try:
        # REUSE: Existing DAGConfigFactory infrastructure (600+ lines of proven patterns)
        dag_factory = DAGConfigFactory(dag)
        
        # Load configs for script validation (eliminates phantom scripts)
        config_classes = build_complete_config_classes()
        all_configs = load_configs(config_path, config_classes)
        
        # Get validated scripts from config (eliminates phantom scripts)
        validated_scripts = get_validated_scripts_from_config(dag, all_configs)
        logger.info(f"Validated scripts (no phantoms): {validated_scripts}")
        
        user_inputs = {}
        for script_name in validated_scripts:
            # EXTEND: Use DAGConfigFactory patterns for input collection
            config = all_configs[script_name]
            script_inputs = collect_script_inputs(config)
            user_inputs[script_name] = script_inputs
        
        return user_inputs
        
    except Exception as e:
        logger.error(f"Failed to collect script inputs: {e}")
        raise ValueError(f"Input collection failed: {e}") from e


def get_validated_scripts_from_config(dag: PipelineDAG, configs: Dict[str, Any]) -> List[str]:
    """
    Get only scripts with actual entry points from config (eliminates phantom scripts).
    
    This addresses the phantom script issue by using config-based validation
    to ensure only scripts with actual entry points are discovered.
    
    Args:
        dag: PipelineDAG instance
        configs: Loaded configuration instances
        
    Returns:
        List of validated script names with actual entry points
    """
    validated_scripts = []
    
    for node_name in dag.nodes:
        if node_name in configs:
            config = configs[node_name]
            # Check if config has script entry point fields
            if hasattr(config, 'training_entry_point') or hasattr(config, 'inference_entry_point'):
                validated_scripts.append(node_name)
            elif hasattr(config, 'source_dir') and hasattr(config, 'entry_point'):
                validated_scripts.append(node_name)
    
    logger.info(f"Phantom script elimination: {len(dag.nodes)} nodes -> {len(validated_scripts)} validated scripts")
    return validated_scripts


def collect_script_inputs(config) -> Dict[str, Any]:
    """
    Extract script path, environment variables, and job arguments from config.
    
    This function uses proper config field access patterns instead of direct __dict__ access.
    It focuses on config-to-script transformation, not path management (that's InputCollector's job).
    
    Args:
        config: Populated config instance (BasePipelineConfig or derived)
        
    Returns:
        Dictionary with script_path, environment_variables, and job_arguments
    """
    # 1. Extract script path from config entry point fields
    script_path = extract_script_path_from_config(config)
    
    # 2. Extract environment variables using proper config access
    environ_vars = extract_environment_variables_from_config(config)
    
    # 3. Extract job arguments using proper config access  
    job_args = extract_job_arguments_from_config(config)
    
    return {
        'script_path': script_path,
        'environment_variables': environ_vars,
        'job_arguments': job_args
    }


def extract_script_path_from_config(config) -> Optional[str]:
    """
    Extract script path from config entry point fields using proper config access.
    
    Args:
        config: Config instance with entry point fields
        
    Returns:
        Resolved script path or None if not found
    """
    import os
    
    # Check for various entry point fields
    entry_point_fields = [
        'training_entry_point',
        'inference_entry_point', 
        'entry_point'
    ]
    
    for field in entry_point_fields:
        if hasattr(config, field):
            entry_point = getattr(config, field)
            if entry_point:
                # Combine with source_dir if available
                if hasattr(config, 'effective_source_dir') and config.effective_source_dir:
                    script_path = os.path.join(config.effective_source_dir, entry_point)
                else:
                    script_path = entry_point
                
                # Use hybrid path resolution if available
                if hasattr(config, 'resolve_hybrid_path'):
                    try:
                        resolved_path = config.resolve_hybrid_path(script_path)
                        if resolved_path and os.path.exists(resolved_path):
                            return resolved_path
                    except AttributeError as e:
                        # Skip hybrid resolution if project_root_folder is missing
                        logger.debug(f"Hybrid path resolution failed for {script_path}: {e}")
                        pass
                
                # Return the path as-is if it exists
                if os.path.exists(script_path):
                    return script_path
                    
                # Return the path even if it doesn't exist (for testing scenarios)
                return script_path
    
    return None


def extract_environment_variables_from_config(config) -> Dict[str, str]:
    """
    Extract environment variables from config using proper field access.
    
    Args:
        config: Config instance
        
    Returns:
        Dictionary of environment variables
    """
    environ_vars = {}
    
    # Try to use model_dump() but handle errors gracefully
    config_data = {}
    try:
        if hasattr(config, 'model_dump'):
            config_data = config.model_dump()
    except AttributeError as e:
        # If model_dump() fails due to missing fields, fall back to direct attribute access
        logger.debug(f"model_dump() failed for config, using direct attribute access: {e}")
        config_data = {}
    
    # Extract relevant fields that should become environment variables
    env_relevant_fields = [
        'framework_version', 'py_version', 'region', 'aws_region',
        'model_class', 'service_name', 'author', 'bucket', 'role'
    ]
    
    for field_name in env_relevant_fields:
        # Try from config_data first, then direct attribute access
        value = None
        if field_name in config_data and config_data[field_name] is not None:
            value = config_data[field_name]
        elif hasattr(config, field_name):
            try:
                value = getattr(config, field_name)
            except AttributeError:
                continue
        
        if value is not None:
            env_var_name = field_name.upper()
            environ_vars[env_var_name] = str(value)
    
    # Add derived fields that are commonly used as environment variables
    derived_env_fields = [
        'pipeline_name', 'pipeline_s3_loc', 'aws_region'
    ]
    
    for field_name in derived_env_fields:
        if hasattr(config, field_name):
            try:
                value = getattr(config, field_name)
                if value is not None:
                    env_var_name = field_name.upper()
                    environ_vars[env_var_name] = str(value)
            except Exception:
                # Skip fields that cause errors
                pass
    
    return environ_vars


def extract_job_arguments_from_config(config):
    """
    Extract job arguments from config using proper field access.
    
    Args:
        config: Config instance
        
    Returns:
        argparse.Namespace with job arguments
    """
    import argparse
    
    # Create argparse.Namespace with relevant job parameters
    job_args = argparse.Namespace()
    
    # Extract job-relevant fields using proper attribute access
    job_relevant_fields = [
        ('training_instance_type', 'instance_type'),
        ('training_instance_count', 'instance_count'), 
        ('training_volume_size', 'volume_size'),
        ('framework_version', 'framework_version'),
        ('py_version', 'py_version')
    ]
    
    for config_field, arg_name in job_relevant_fields:
        if hasattr(config, config_field):
            value = getattr(config, config_field)
            if value is not None:
                setattr(job_args, arg_name, value)
    
    # Add default job type if not specified
    if not hasattr(job_args, 'job_type'):
        job_args.job_type = getattr(config, 'job_type', 'training')
    
    return job_args


def execute_scripts_in_order(
    execution_order: List[str],
    user_inputs: Dict[str, Any]  # Complete inputs from two-phase system
) -> Dict[str, Any]:
    """
    DRAMATICALLY SIMPLIFIED: Execute scripts with complete pre-resolved inputs.
    
    All complexity (message passing, dependency matching, config extraction) 
    is handled in input collection phase.
    
    Args:
        execution_order: List of script names in topological order
        user_inputs: Complete inputs from two-phase dependency resolution
        
    Returns:
        Dictionary with execution results
    """
    results = {}
    
    for node_name in execution_order:
        try:
            logger.info(f"Executing script: {node_name}")
            
            # SIMPLIFIED: Get complete pre-resolved data
            node_inputs = user_inputs.get(node_name, {})
            
            # All information is complete from two-phase resolution:
            input_paths = node_inputs.get('input_paths', {})        # ✅ Auto-resolved or user-provided
            output_paths = node_inputs.get('output_paths', {})      # ✅ User-provided
            environ_vars = node_inputs.get('environment_variables', {})  # ✅ From config
            job_args = node_inputs.get('job_arguments', {})         # ✅ From config
            script_path = node_inputs.get('script_path')            # ✅ From config
            
            if not script_path:
                logger.warning(f"No script path found for {node_name}, skipping")
                continue
            
            # ULTRA-SIMPLIFIED: Just execute with complete information
            result = execute_single_script(script_path, input_paths, output_paths, environ_vars, job_args)
            results[node_name] = result
            
            if result.success:
                logger.info(f"✅ {node_name} executed successfully")
            else:
                logger.error(f"❌ {node_name} failed: {result.error_message}")
                
        except Exception as e:
            logger.error(f"❌ {node_name} execution failed: {e}")
            results[node_name] = ScriptTestResult(success=False, error_message=str(e))
    
    return {
        "pipeline_success": all(r.success for r in results.values()),
        "script_results": results,
        "execution_order": execution_order,
        "total_scripts": len(execution_order),
        "successful_scripts": sum(1 for r in results.values() if r.success)
    }


# Removed discover_script_with_config_validation - it was redundant
# Script paths are now properly extracted from config via extract_script_path_from_config()


def execute_single_script(script_path: str, input_paths: Dict[str, str], 
                         output_paths: Dict[str, str], environ_vars: Dict[str, str], 
                         job_args) -> ScriptTestResult:
    """
    Execute a single script with the fixed signature and dependency management.
    
    This function handles the one legitimate complexity in script testing:
    package dependency management (scripts import packages that need installation).
    
    Args:
        script_path: Path to the script file
        input_paths: Input paths from InputCollector (contract-based logical names)
        output_paths: Output paths from InputCollector (contract-based logical names)
        environ_vars: Environment variables from config
        job_args: Job arguments from config (argparse.Namespace)
        
    Returns:
        ScriptTestResult with execution outcome
    """
    try:
        # 1. Handle package dependencies (VALID COMPLEXITY)
        # Scripts import packages that need to be installed before execution
        # (In SageMaker pipeline, this was isolated as an environment)
        install_script_dependencies(script_path)
        
        # 2. Execute script with fixed signature
        result = import_and_execute_script(script_path, input_paths, output_paths, environ_vars, job_args)
        
        return ScriptTestResult(
            success=True,
            output_files=result.get('outputs', {}),
            execution_time=result.get('execution_time', 0)
        )
        
    except Exception as e:
        logger.error(f"Script execution failed for {script_path}: {e}")
        return ScriptTestResult(success=False, error_message=str(e))


def install_script_dependencies(script_path: str) -> None:
    """
    Install package dependencies for script execution.
    
    This is the ONE valid complexity in script testing - scripts import packages
    that need to be installed before execution. In SageMaker pipeline, this was
    isolated as an environment.
    
    Args:
        script_path: Path to the script file
    """
    try:
        # Parse script imports and install required packages
        required_packages = parse_script_imports(script_path)
        
        for package in required_packages:
            if not is_package_installed(package):
                logger.info(f"Installing package: {package}")
                install_package(package)
                
    except Exception as e:
        logger.warning(f"Dependency installation failed for {script_path}: {e}")


def parse_script_imports(script_path: str) -> List[str]:
    """
    Parse script file to extract required packages.
    
    Args:
        script_path: Path to the script file
        
    Returns:
        List of required package names
    """
    try:
        with open(script_path, 'r') as f:
            tree = ast.parse(f.read())
        
        packages = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    packages.append(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    packages.append(node.module.split('.')[0])
        
        # Filter out standard library modules
        standard_libs = {'os', 'sys', 'json', 'logging', 'pathlib', 'typing', 'datetime'}
        external_packages = [pkg for pkg in packages if pkg not in standard_libs]
        
        return list(set(external_packages))  # Remove duplicates
        
    except Exception as e:
        logger.warning(f"Failed to parse imports from {script_path}: {e}")
        return []


def is_package_installed(package_name: str) -> bool:
    """
    Check if a package is installed.
    
    Args:
        package_name: Name of the package to check
        
    Returns:
        True if package is installed, False otherwise
    """
    try:
        importlib.util.find_spec(package_name)
        return True
    except ImportError:
        return False


def install_package(package_name: str) -> None:
    """
    Install a package using pip.
    
    Args:
        package_name: Name of the package to install
    """
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
        logger.info(f"Successfully installed {package_name}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install {package_name}: {e}")
        raise


def execute_scripts_with_registry_coordination(dag: PipelineDAG, registry) -> Dict[str, Any]:
    """
    Execute scripts with registry coordination and message passing.
    
    This function uses the ScriptExecutionRegistry to coordinate script execution
    with automatic message passing between nodes.
    
    Args:
        dag: PipelineDAG instance
        registry: ScriptExecutionRegistry instance
        
    Returns:
        Dictionary with execution results and metadata
    """
    script_results = {}
    successful_scripts = 0
    execution_order = dag.topological_sort()
    
    for node_name in execution_order:
        try:
            # Get ready inputs from registry (Integration Point 5)
            script_inputs = registry.get_ready_node_inputs(node_name)
            
            if not script_inputs or 'script_path' not in script_inputs:
                logger.warning(f"⚠️  {node_name} has no script configuration, skipping")
                continue
            
            logger.info(f"🔄 Executing {node_name} with registry coordination")
            
            # Execute script with registry-coordinated inputs
            result = execute_single_script(
                script_path=script_inputs['script_path'],
                input_paths=script_inputs.get('input_paths', {}),
                output_paths=script_inputs.get('output_paths', {}),
                environ_vars=script_inputs.get('environment_variables', {}),
                job_args=script_inputs.get('job_arguments', {})
            )
            
            script_results[node_name] = result
            
            # Commit execution results to registry (Integration Point 6)
            registry.commit_execution_results(node_name, result)
            
            if result.success:
                successful_scripts += 1
                logger.info(f"✅ {node_name} completed successfully")
            else:
                logger.error(f"❌ {node_name} failed: {result.error_message}")
                
        except Exception as e:
            error_result = ScriptTestResult(success=False, error_message=f"Execution failed: {e}")
            script_results[node_name] = error_result
            registry.commit_execution_results(node_name, error_result)
            logger.error(f"❌ {node_name} failed with exception: {e}")
    
    return {
        'pipeline_success': successful_scripts == len([n for n in execution_order if registry.get_ready_node_inputs(n)]),
        'script_results': script_results,
        'execution_order': execution_order,
        'total_scripts': len(execution_order),
        'successful_scripts': successful_scripts
    }


def import_and_execute_script(script_path: str, input_paths: Dict[str, str], 
                            output_paths: Dict[str, str], environ_vars: Dict[str, str], 
                            job_args) -> Dict[str, Any]:
    """
    Import and execute a script with the fixed signature.
    
    Uses the testability pattern: main(input_paths, output_paths, environ_vars, job_args)
    
    Args:
        script_path: Path to the script file
        input_paths: Input paths from InputCollector (contract-based logical names)
        output_paths: Output paths from InputCollector (contract-based logical names)
        environ_vars: Environment variables from config
        job_args: Job arguments from config (argparse.Namespace)
        
    Returns:
        Dictionary with execution results
    """
    try:
        import time
        start_time = time.time()
        
        # Load script as module
        spec = importlib.util.spec_from_file_location("script_module", script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load script from {script_path}")
        
        script_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script_module)
        
        # Execute main function with fixed signature
        if hasattr(script_module, 'main'):
            result = script_module.main(input_paths, output_paths, environ_vars, job_args)
        else:
            raise ValueError(f"Script {script_path} does not have a main function with the required signature")
        
        execution_time = time.time() - start_time
        
        return {
            'outputs': result if isinstance(result, dict) else {'result': result},
            'execution_time': execution_time
        }
        
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        raise

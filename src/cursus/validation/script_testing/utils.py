"""
Script Testing Utilities

This module provides utility functions for script testing operations,
focusing on simple, reusable functions that support the main API.
"""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import logging
import json
import time
from datetime import datetime

logger = logging.getLogger(__name__)


def validate_dag_and_config(dag, config_path: str) -> Dict[str, Any]:
    """
    Validate DAG and config path inputs.
    
    Args:
        dag: PipelineDAG instance
        config_path: Path to configuration file
        
    Returns:
        Dictionary with validation results
        
    Raises:
        ValueError: If validation fails
    """
    validation_results = {
        'dag_valid': False,
        'config_valid': False,
        'dag_nodes': 0,
        'config_exists': False,
        'errors': []
    }
    
    try:
        # Validate DAG
        if dag is None:
            validation_results['errors'].append("DAG cannot be None")
        elif not hasattr(dag, 'nodes'):
            validation_results['errors'].append("DAG must have nodes attribute")
        elif not dag.nodes:
            validation_results['errors'].append("DAG must contain at least one node")
        else:
            validation_results['dag_valid'] = True
            validation_results['dag_nodes'] = len(dag.nodes)
        
        # Validate config path
        config_file = Path(config_path)
        if not config_file.exists():
            validation_results['errors'].append(f"Config file not found: {config_path}")
        elif not config_file.is_file():
            validation_results['errors'].append(f"Config path is not a file: {config_path}")
        elif not config_file.suffix.lower() == '.json':
            validation_results['errors'].append(f"Config file must be JSON: {config_path}")
        else:
            validation_results['config_valid'] = True
            validation_results['config_exists'] = True
        
        # Overall validation
        if validation_results['errors']:
            error_msg = "; ".join(validation_results['errors'])
            raise ValueError(f"Validation failed: {error_msg}")
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise


def create_test_workspace(workspace_dir: str) -> Path:
    """
    Create test workspace directory structure.
    
    Args:
        workspace_dir: Path to workspace directory
        
    Returns:
        Path to created workspace directory
    """
    try:
        workspace_path = Path(workspace_dir)
        workspace_path.mkdir(parents=True, exist_ok=True)
        
        # Create standard subdirectories
        subdirs = ['data', 'models', 'scripts', 'outputs', 'logs']
        for subdir in subdirs:
            (workspace_path / subdir).mkdir(exist_ok=True)
        
        logger.info(f"Created test workspace: {workspace_path}")
        return workspace_path
        
    except Exception as e:
        logger.error(f"Failed to create workspace {workspace_dir}: {e}")
        raise


def load_json_config(config_path: str) -> Dict[str, Any]:
    """
    Load JSON configuration file.
    
    Args:
        config_path: Path to JSON configuration file
        
    Returns:
        Dictionary with configuration data
    """
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        logger.info(f"Loaded config from {config_path}")
        return config_data
        
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise


def save_execution_results(results: Dict[str, Any], output_path: str) -> Path:
    """
    Save execution results to file.
    
    Args:
        results: Execution results dictionary
        output_path: Path to save results
        
    Returns:
        Path to saved file
    """
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp to results
        results_with_metadata = {
            **results,
            '_metadata': {
                'saved_at': datetime.now().isoformat(),
                'saved_to': str(output_file)
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_with_metadata, f, indent=2, default=str)
        
        logger.info(f"Saved execution results to {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Failed to save results to {output_path}: {e}")
        raise


def calculate_execution_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate execution summary statistics.
    
    Args:
        results: Execution results dictionary
        
    Returns:
        Dictionary with summary statistics
    """
    try:
        script_results = results.get('script_results', {})
        execution_order = results.get('execution_order', [])
        
        total_scripts = len(script_results)
        successful_scripts = sum(1 for result in script_results.values() if result.success)
        failed_scripts = total_scripts - successful_scripts
        
        # Calculate execution times
        execution_times = [
            result.execution_time for result in script_results.values() 
            if result.execution_time is not None
        ]
        
        total_execution_time = sum(execution_times) if execution_times else 0
        avg_execution_time = total_execution_time / len(execution_times) if execution_times else 0
        
        summary = {
            'total_scripts': total_scripts,
            'successful_scripts': successful_scripts,
            'failed_scripts': failed_scripts,
            'success_rate': successful_scripts / total_scripts if total_scripts > 0 else 0,
            'total_execution_time': total_execution_time,
            'average_execution_time': avg_execution_time,
            'pipeline_success': results.get('pipeline_success', False),
            'execution_order': execution_order
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"Failed to calculate execution summary: {e}")
        return {}


def format_execution_time(seconds: Optional[float]) -> str:
    """
    Format execution time for display.
    
    Args:
        seconds: Execution time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds is None:
        return "N/A"
    
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"


def get_script_info(script_path: str) -> Dict[str, Any]:
    """
    Get information about a script file.
    
    Args:
        script_path: Path to script file
        
    Returns:
        Dictionary with script information
    """
    try:
        script_file = Path(script_path)
        
        info = {
            'exists': script_file.exists(),
            'is_file': script_file.is_file() if script_file.exists() else False,
            'size_bytes': script_file.stat().st_size if script_file.exists() else 0,
            'modified_time': datetime.fromtimestamp(script_file.stat().st_mtime).isoformat() if script_file.exists() else None,
            'extension': script_file.suffix,
            'name': script_file.name,
            'parent': str(script_file.parent)
        }
        
        # Check if it's a Python file
        if script_file.suffix.lower() == '.py' and script_file.exists():
            info['is_python'] = True
            info['has_main_function'] = check_has_main_function(script_path)
        else:
            info['is_python'] = False
            info['has_main_function'] = False
        
        return info
        
    except Exception as e:
        logger.error(f"Failed to get script info for {script_path}: {e}")
        return {
            'exists': False,
            'error': str(e)
        }


def check_has_main_function(script_path: str) -> bool:
    """
    Check if a Python script has a main function.
    
    Args:
        script_path: Path to Python script
        
    Returns:
        True if script has main function, False otherwise
    """
    try:
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Simple check for main function definition
        return 'def main(' in content
        
    except Exception as e:
        logger.warning(f"Failed to check main function in {script_path}: {e}")
        return False


def create_default_paths(script_name: str, base_dir: str = "test") -> Dict[str, Dict[str, str]]:
    """
    Create default input and output paths for a script.
    
    Args:
        script_name: Name of the script
        base_dir: Base directory for paths
        
    Returns:
        Dictionary with input and output path mappings
    """
    base_path = Path(base_dir)
    
    return {
        'input_paths': {
            'data_input': str(base_path / 'data' / script_name / 'input'),
            'model_input': str(base_path / 'models' / script_name / 'input')
        },
        'output_paths': {
            'data_output': str(base_path / 'data' / script_name / 'output'),
            'model_output': str(base_path / 'models' / script_name / 'output'),
            'logs': str(base_path / 'logs' / f"{script_name}.log")
        }
    }


def merge_script_configs(base_config: Dict[str, Any], script_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge base configuration with script-specific configuration.
    
    Args:
        base_config: Base configuration dictionary
        script_config: Script-specific configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    try:
        merged_config = base_config.copy()
        
        # Merge nested dictionaries
        for key, value in script_config.items():
            if key in merged_config and isinstance(merged_config[key], dict) and isinstance(value, dict):
                merged_config[key].update(value)
            else:
                merged_config[key] = value
        
        return merged_config
        
    except Exception as e:
        logger.error(f"Failed to merge configs: {e}")
        return script_config


def validate_script_inputs(inputs: Dict[str, Any]) -> List[str]:
    """
    Validate script input configuration.
    
    Args:
        inputs: Script input configuration
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    try:
        # Check required keys
        required_keys = ['input_paths', 'output_paths']
        for key in required_keys:
            if key not in inputs:
                errors.append(f"Missing required key: {key}")
        
        # Validate input paths
        input_paths = inputs.get('input_paths', {})
        if not isinstance(input_paths, dict):
            errors.append("input_paths must be a dictionary")
        elif not input_paths:
            errors.append("input_paths cannot be empty")
        
        # Validate output paths
        output_paths = inputs.get('output_paths', {})
        if not isinstance(output_paths, dict):
            errors.append("output_paths must be a dictionary")
        elif not output_paths:
            errors.append("output_paths cannot be empty")
        
        # Validate environment variables (optional)
        env_vars = inputs.get('environment_variables', {})
        if env_vars and not isinstance(env_vars, dict):
            errors.append("environment_variables must be a dictionary")
        
        # Validate job arguments (optional)
        job_args = inputs.get('job_arguments', {})
        if job_args and not isinstance(job_args, dict):
            errors.append("job_arguments must be a dictionary")
        
        return errors
        
    except Exception as e:
        logger.error(f"Failed to validate script inputs: {e}")
        return [f"Validation error: {str(e)}"]


def get_testing_summary() -> Dict[str, Any]:
    """
    Get summary of script testing utilities.
    
    Returns:
        Dictionary with utility summary information
    """
    return {
        'utility_functions': [
            'validate_dag_and_config',
            'create_test_workspace',
            'load_json_config',
            'save_execution_results',
            'calculate_execution_summary',
            'format_execution_time',
            'get_script_info',
            'check_has_main_function',
            'create_default_paths',
            'merge_script_configs',
            'validate_script_inputs'
        ],
        'features': [
            'DAG and config validation',
            'Test workspace management',
            'Configuration loading and merging',
            'Execution result processing',
            'Script information extraction',
            'Path management utilities'
        ],
        'module_type': 'ScriptTestingUtils'
    }

"""
Script Input Resolution Pattern Adaptation

This module adapts step builder input resolution patterns for script testing,
providing contract-based path mapping and logical name transformation using
existing cursus infrastructure patterns.

Key Features:
- Direct adaptation of StepBuilder._get_inputs() patterns for script testing
- Contract-based path mapping using existing step catalog infrastructure
- Logical name to actual path transformation
- Same validation patterns as step builders
- Maximum component reuse from existing cursus infrastructure
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Direct reuse of existing cursus infrastructure
from ...step_catalog import StepCatalog
from ...core.base.specification_base import StepSpecification

logger = logging.getLogger(__name__)


def resolve_script_inputs_using_step_patterns(
    node_name: str,
    spec: StepSpecification,
    resolved_dependencies: Dict[str, str],
    step_catalog: StepCatalog
) -> Dict[str, str]:
    """
    Script input resolution adapted from StepBuilder._get_inputs() patterns.
    
    DIRECT ADAPTATION of step builder input resolution logic for script testing.
    This function mirrors the same patterns used in step builders for input resolution,
    providing consistent behavior between pipeline steps and script testing.
    
    Args:
        node_name: Name of the script/node
        spec: Step specification with dependencies and outputs
        resolved_dependencies: Dictionary of resolved dependency paths
        step_catalog: Step catalog for contract loading
        
    Returns:
        Dictionary mapping logical dependency names to actual script input paths
        
    Example:
        >>> spec = step_catalog.spec_discovery.load_spec_class('DataPreprocessing')
        >>> resolved_deps = {'training_data': '/data/input/train.csv'}
        >>> script_inputs = resolve_script_inputs_using_step_patterns(
        ...     'DataPreprocessing', spec, resolved_deps, step_catalog
        ... )
        >>> print(script_inputs)
        {'training_data': '/data/input/train.csv'}
    """
    logger.info(f"Resolving script inputs for {node_name} using step builder patterns")
    
    script_inputs = {}
    
    try:
        # Load contract (DIRECT REUSE of step catalog patterns)
        contract = step_catalog.load_contract_class(node_name)
        logger.debug(f"Loaded contract for {node_name}: {contract is not None}")
        
        # Process dependencies (SAME PATTERN as step builders)
        for dep_name, dep_spec in spec.dependencies.items():
            logger.debug(f"Processing dependency {dep_name} for {node_name}")
            
            # Skip optional unresolved dependencies (SAME LOGIC as step builders)
            if not dep_spec.required and dep_name not in resolved_dependencies:
                logger.debug(f"Skipping optional unresolved dependency {dep_name}")
                continue
            
            # Ensure required dependencies are resolved (SAME VALIDATION as step builders)
            if dep_spec.required and dep_name not in resolved_dependencies:
                raise ValueError(f"Required dependency '{dep_name}' not resolved for {node_name}")
            
            # Get actual path
            actual_path = resolved_dependencies[dep_name]
            logger.debug(f"Resolved {dep_name} to {actual_path}")
            
            # Map using contract (SAME PATTERN as step builders)
            if contract and hasattr(contract, 'expected_input_paths'):
                container_path = contract.expected_input_paths.get(dep_name)
                if container_path:
                    # Use contract-defined path mapping
                    script_inputs[dep_name] = actual_path
                    logger.debug(f"Used contract mapping for {dep_name}: {actual_path}")
                else:
                    # Fallback to direct mapping
                    script_inputs[dep_name] = actual_path
                    logger.debug(f"Used direct mapping for {dep_name}: {actual_path}")
            else:
                # No contract available, use direct mapping
                script_inputs[dep_name] = actual_path
                logger.debug(f"No contract available, used direct mapping for {dep_name}: {actual_path}")
        
        logger.info(f"Successfully resolved {len(script_inputs)} script inputs for {node_name}")
        return script_inputs
        
    except Exception as e:
        logger.error(f"Failed to resolve script inputs for {node_name}: {e}")
        raise RuntimeError(f"Script input resolution failed for {node_name}: {e}") from e


def adapt_step_input_patterns_for_scripts(
    node_name: str,
    inputs: Dict[str, Any],
    step_catalog: StepCatalog
) -> Dict[str, str]:
    """
    Adapt step builder input patterns for script testing.
    
    DIRECT ADAPTATION of step builder input resolution patterns.
    This function provides the same input validation and transformation
    patterns used in step builders, ensuring consistency across the system.
    
    Args:
        node_name: Name of the script/node
        inputs: Dictionary of input data to process
        step_catalog: Step catalog for specification and contract loading
        
    Returns:
        Dictionary mapping logical names to actual script input paths
        
    Raises:
        ValueError: If specification or contract not found, or required inputs missing
        
    Example:
        >>> inputs = {'training_data': '/data/train.csv', 'model_config': '/config/model.json'}
        >>> script_inputs = adapt_step_input_patterns_for_scripts(
        ...     'XGBoostTraining', inputs, step_catalog
        ... )
        >>> print(script_inputs)
        {'training_data': '/data/train.csv', 'model_config': '/config/model.json'}
    """
    logger.info(f"Adapting step input patterns for script {node_name}")
    
    try:
        # Load specification (DIRECT REUSE)
        spec = step_catalog.spec_discovery.load_spec_class(node_name)
        if not spec:
            raise ValueError(f"No specification found for {node_name}")
        
        logger.debug(f"Loaded specification for {node_name}: {len(spec.dependencies)} dependencies")
        
        # Load contract (DIRECT REUSE)
        contract = step_catalog.load_contract_class(node_name)
        if not contract:
            logger.warning(f"No contract found for {node_name}, using direct mapping")
        else:
            logger.debug(f"Loaded contract for {node_name}")
        
        script_inputs = {}
        
        # Process each dependency (SAME PATTERN as step builders)
        for dep_name, dep_spec in spec.dependencies.items():
            logger.debug(f"Processing dependency {dep_name} (required: {dep_spec.required})")
            
            # Skip optional dependencies not provided (SAME LOGIC as step builders)
            if not dep_spec.required and dep_name not in inputs:
                logger.debug(f"Skipping optional dependency {dep_name} (not provided)")
                continue
            
            # Ensure required dependencies are provided (SAME VALIDATION as step builders)
            if dep_spec.required and dep_name not in inputs:
                raise ValueError(f"Required input '{dep_name}' not provided for {node_name}")
            
            # Get container path from contract (SAME PATTERN as step builders)
            container_path = None
            if contract and hasattr(contract, 'expected_input_paths'):
                container_path = contract.expected_input_paths.get(dep_name)
                logger.debug(f"Contract container path for {dep_name}: {container_path}")
            
            if container_path:
                # Use logical name for script input mapping (SAME PATTERN as step builders)
                script_inputs[dep_name] = inputs[dep_name]
                logger.debug(f"Used contract-based mapping for {dep_name}")
            else:
                # Fallback to logical name (SAME FALLBACK as step builders)
                script_inputs[dep_name] = inputs[dep_name]
                logger.debug(f"Used direct mapping for {dep_name}")
        
        logger.info(f"Successfully adapted {len(script_inputs)} inputs for script {node_name}")
        return script_inputs
        
    except Exception as e:
        logger.error(f"Failed to adapt step input patterns for {node_name}: {e}")
        raise RuntimeError(f"Step input pattern adaptation failed for {node_name}: {e}") from e


def validate_script_input_resolution(
    node_name: str,
    script_inputs: Dict[str, str],
    step_catalog: StepCatalog
) -> bool:
    """
    Validate script input resolution using step builder validation patterns.
    
    This function applies the same validation logic used in step builders
    to ensure script inputs are properly resolved and valid.
    
    Args:
        node_name: Name of the script/node
        script_inputs: Dictionary of resolved script inputs
        step_catalog: Step catalog for specification loading
        
    Returns:
        True if validation passes, False otherwise
        
    Example:
        >>> script_inputs = {'training_data': '/data/train.csv'}
        >>> is_valid = validate_script_input_resolution(
        ...     'DataPreprocessing', script_inputs, step_catalog
        ... )
        >>> print(is_valid)
        True
    """
    try:
        logger.info(f"Validating script input resolution for {node_name}")
        
        # Load specification for validation
        spec = step_catalog.spec_discovery.load_spec_class(node_name)
        if not spec:
            logger.error(f"No specification found for {node_name}")
            return False
        
        # Validate all required dependencies are resolved
        for dep_name, dep_spec in spec.dependencies.items():
            if dep_spec.required and dep_name not in script_inputs:
                logger.error(f"Required dependency '{dep_name}' not resolved for {node_name}")
                return False
        
        # Validate paths exist (basic validation)
        for dep_name, path in script_inputs.items():
            if not path:
                logger.error(f"Empty path for dependency '{dep_name}' in {node_name}")
                return False
            
            # Check if path looks valid (basic format check)
            if not isinstance(path, str) or len(path.strip()) == 0:
                logger.error(f"Invalid path format for dependency '{dep_name}' in {node_name}: {path}")
                return False
        
        logger.info(f"Script input resolution validation passed for {node_name}")
        return True
        
    except Exception as e:
        logger.error(f"Script input resolution validation failed for {node_name}: {e}")
        return False


def get_script_input_resolution_summary(
    node_name: str,
    script_inputs: Dict[str, str],
    step_catalog: StepCatalog
) -> Dict[str, Any]:
    """
    Generate a summary of script input resolution for debugging and monitoring.
    
    Args:
        node_name: Name of the script/node
        script_inputs: Dictionary of resolved script inputs
        step_catalog: Step catalog for specification loading
        
    Returns:
        Dictionary with resolution summary information
        
    Example:
        >>> script_inputs = {'training_data': '/data/train.csv', 'config': '/config/model.json'}
        >>> summary = get_script_input_resolution_summary(
        ...     'XGBoostTraining', script_inputs, step_catalog
        ... )
        >>> print(summary['total_inputs'])
        2
    """
    try:
        # Load specification for analysis
        spec = step_catalog.spec_discovery.load_spec_class(node_name)
        contract = step_catalog.load_contract_class(node_name)
        
        # Count dependencies
        total_dependencies = len(spec.dependencies) if spec else 0
        required_dependencies = sum(1 for dep_spec in spec.dependencies.values() if dep_spec.required) if spec else 0
        optional_dependencies = total_dependencies - required_dependencies
        
        # Count resolved inputs
        resolved_inputs = len(script_inputs)
        
        # Check contract usage
        contract_available = contract is not None
        contract_paths_used = 0
        if contract and hasattr(contract, 'expected_input_paths'):
            contract_paths_used = len([
                dep_name for dep_name in script_inputs.keys()
                if dep_name in contract.expected_input_paths
            ])
        
        return {
            'node_name': node_name,
            'total_dependencies': total_dependencies,
            'required_dependencies': required_dependencies,
            'optional_dependencies': optional_dependencies,
            'resolved_inputs': resolved_inputs,
            'contract_available': contract_available,
            'contract_paths_used': contract_paths_used,
            'resolution_complete': resolved_inputs >= required_dependencies,
            'input_paths': script_inputs
        }
        
    except Exception as e:
        logger.error(f"Failed to generate script input resolution summary for {node_name}: {e}")
        return {
            'node_name': node_name,
            'error': str(e),
            'resolution_complete': False
        }


def transform_logical_names_to_actual_paths(
    logical_inputs: Dict[str, str],
    node_name: str,
    step_catalog: StepCatalog
) -> Dict[str, str]:
    """
    Transform logical dependency names to actual file paths using contract patterns.
    
    This function applies the same logical-to-actual path transformation
    used in step builders, ensuring consistent path resolution across the system.
    
    Args:
        logical_inputs: Dictionary mapping logical names to paths
        node_name: Name of the script/node
        step_catalog: Step catalog for contract loading
        
    Returns:
        Dictionary mapping logical names to actual file paths
        
    Example:
        >>> logical_inputs = {'training_data': '/container/input/data'}
        >>> actual_paths = transform_logical_names_to_actual_paths(
        ...     logical_inputs, 'DataPreprocessing', step_catalog
        ... )
        >>> print(actual_paths)
        {'training_data': '/opt/ml/input/data/training_data.csv'}
    """
    logger.info(f"Transforming logical names to actual paths for {node_name}")
    
    try:
        # Load contract for path transformation
        contract = step_catalog.load_contract_class(node_name)
        if not contract:
            logger.warning(f"No contract found for {node_name}, returning logical inputs as-is")
            return logical_inputs.copy()
        
        actual_paths = {}
        
        for logical_name, logical_path in logical_inputs.items():
            # Check if contract defines actual path mapping
            if hasattr(contract, 'expected_input_paths'):
                container_path = contract.expected_input_paths.get(logical_name)
                if container_path:
                    # Use contract-defined transformation
                    actual_paths[logical_name] = logical_path
                    logger.debug(f"Contract transformation for {logical_name}: {logical_path}")
                else:
                    # No contract mapping, use logical path as-is
                    actual_paths[logical_name] = logical_path
                    logger.debug(f"Direct mapping for {logical_name}: {logical_path}")
            else:
                # Contract has no path mappings, use logical path as-is
                actual_paths[logical_name] = logical_path
                logger.debug(f"No contract mappings, direct mapping for {logical_name}: {logical_path}")
        
        logger.info(f"Successfully transformed {len(actual_paths)} logical names to actual paths for {node_name}")
        return actual_paths
        
    except Exception as e:
        logger.error(f"Failed to transform logical names to actual paths for {node_name}: {e}")
        # Return original inputs as fallback
        return logical_inputs.copy()

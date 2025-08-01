"""
Contract Validation Utilities for Pipeline Scripts

Provides contract-aware helper functions for SageMaker pipeline scripts to ensure
alignment between step specifications, script contracts, and actual implementations.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


def validate_contract_environment(contract) -> None:
    """
    Validate SageMaker environment matches contract expectations.
    
    Args:
        contract: ScriptContract instance defining expected environment
        
    Raises:
        RuntimeError: If contract validation fails
    """
    errors = []
    
    # Check required environment variables
    for var in contract.required_env_vars:
        if var not in os.environ:
            errors.append(f"Missing required environment variable: {var}")
        else:
            logger.info(f"✓ Environment variable found: {var}")
    
    # Check optional environment variables and set defaults
    for var, default_value in contract.optional_env_vars.items():
        if var not in os.environ:
            os.environ[var] = default_value
            logger.info(f"Set default environment variable: {var}={default_value}")
        else:
            logger.info(f"✓ Optional environment variable found: {var}")
    
    # Check input paths exist (SageMaker mounts these)
    for logical_name, path in contract.expected_input_paths.items():
        if not os.path.exists(path):
            errors.append(f"Input path not found: {path} ({logical_name})")
        else:
            logger.info(f"✓ Input path exists: {path} ({logical_name})")
    
    # Ensure output directories exist
    for logical_name, path in contract.expected_output_paths.items():
        os.makedirs(path, exist_ok=True)
        logger.info(f"✓ Ensured output directory exists: {path} ({logical_name})")
    
    if errors:
        error_msg = f"Contract validation failed: {errors}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    logger.info("✅ Contract environment validation passed")


def get_contract_paths(contract) -> Dict[str, Dict[str, str]]:
    """
    Get input/output paths from contract for easy access.
    
    Args:
        contract: ScriptContract instance
        
    Returns:
        Dictionary with 'inputs' and 'outputs' keys containing path mappings
    """
    paths = {
        'inputs': contract.expected_input_paths.copy(),
        'outputs': contract.expected_output_paths.copy()
    }
    
    logger.info(f"Contract paths loaded - Inputs: {len(paths['inputs'])}, Outputs: {len(paths['outputs'])}")
    return paths


def get_input_path(contract, logical_name: str) -> str:
    """
    Get input path by logical name from contract.
    
    Args:
        contract: ScriptContract instance
        logical_name: Logical name of the input
        
    Returns:
        Absolute path to the input
        
    Raises:
        ValueError: If logical name not found in contract
    """
    if logical_name not in contract.expected_input_paths:
        available_inputs = list(contract.expected_input_paths.keys())
        raise ValueError(f"Unknown input '{logical_name}'. Available inputs: {available_inputs}")
    
    path = contract.expected_input_paths[logical_name]
    logger.info(f"Retrieved input path: {logical_name} -> {path}")
    return path


def get_output_path(contract, logical_name: str) -> str:
    """
    Get output path by logical name from contract.
    
    Args:
        contract: ScriptContract instance
        logical_name: Logical name of the output
        
    Returns:
        Absolute path to the output directory
        
    Raises:
        ValueError: If logical name not found in contract
    """
    if logical_name not in contract.expected_output_paths:
        available_outputs = list(contract.expected_output_paths.keys())
        raise ValueError(f"Unknown output '{logical_name}'. Available outputs: {available_outputs}")
    
    path = contract.expected_output_paths[logical_name]
    os.makedirs(path, exist_ok=True)
    logger.info(f"Retrieved output path: {logical_name} -> {path}")
    return path


def validate_required_files(contract, required_files: Optional[Dict[str, List[str]]] = None) -> None:
    """
    Validate that required files exist in input directories.
    
    Args:
        contract: ScriptContract instance
        required_files: Dict mapping logical input names to lists of required filenames
        
    Raises:
        RuntimeError: If required files are missing
    """
    if not required_files:
        logger.info("No required files specified for validation")
        return
    
    errors = []
    
    for logical_name, filenames in required_files.items():
        if logical_name not in contract.expected_input_paths:
            errors.append(f"Unknown input '{logical_name}' in required_files")
            continue
            
        input_dir = contract.expected_input_paths[logical_name]
        
        for filename in filenames:
            file_path = os.path.join(input_dir, filename)
            if not os.path.exists(file_path):
                errors.append(f"Required file not found: {file_path}")
            else:
                logger.info(f"✓ Required file found: {file_path}")
    
    if errors:
        error_msg = f"Required file validation failed: {errors}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    logger.info("✅ Required files validation passed")


def log_contract_summary(contract) -> None:
    """
    Log a summary of the contract for debugging purposes.
    
    Args:
        contract: ScriptContract instance
    """
    logger.info("=== Contract Summary ===")
    logger.info(f"Entry Point: {contract.entry_point}")
    logger.info(f"Description: {contract.description}")
    
    logger.info("Input Paths:")
    for logical_name, path in contract.expected_input_paths.items():
        logger.info(f"  {logical_name}: {path}")
    
    logger.info("Output Paths:")
    for logical_name, path in contract.expected_output_paths.items():
        logger.info(f"  {logical_name}: {path}")
    
    logger.info("Required Environment Variables:")
    for var in contract.required_env_vars:
        value = os.environ.get(var, "NOT SET")
        logger.info(f"  {var}: {value}")
    
    logger.info("Optional Environment Variables:")
    for var, default in contract.optional_env_vars.items():
        value = os.environ.get(var, default)
        logger.info(f"  {var}: {value} (default: {default})")
    
    logger.info("Framework Requirements:")
    for framework, version in contract.framework_requirements.items():
        logger.info(f"  {framework}: {version}")
    
    logger.info("========================")


def find_files_in_input(contract, logical_name: str, pattern: str = "*") -> List[str]:
    """
    Find files matching a pattern in an input directory.
    
    Args:
        contract: ScriptContract instance
        logical_name: Logical name of the input
        pattern: Glob pattern to match files (default: "*")
        
    Returns:
        List of file paths matching the pattern
        
    Raises:
        ValueError: If logical name not found in contract
    """
    input_path = get_input_path(contract, logical_name)
    input_dir = Path(input_path)
    
    if not input_dir.exists():
        logger.warning(f"Input directory does not exist: {input_path}")
        return []
    
    files = list(input_dir.glob(pattern))
    file_paths = [str(f) for f in files if f.is_file()]
    
    logger.info(f"Found {len(file_paths)} files matching '{pattern}' in {logical_name}: {file_paths}")
    return file_paths


def create_output_file_path(contract, logical_name: str, filename: str) -> str:
    """
    Create a full file path within an output directory.
    
    Args:
        contract: ScriptContract instance
        logical_name: Logical name of the output
        filename: Name of the file to create
        
    Returns:
        Full path to the output file
    """
    output_dir = get_output_path(contract, logical_name)
    file_path = os.path.join(output_dir, filename)
    
    logger.info(f"Created output file path: {logical_name}/{filename} -> {file_path}")
    return file_path


def validate_framework_requirements(contract) -> None:
    """
    Validate that required frameworks are available (basic import check).
    
    Args:
        contract: ScriptContract instance
        
    Raises:
        RuntimeError: If required frameworks cannot be imported
    """
    errors = []
    
    framework_import_map = {
        'pandas': 'pandas',
        'numpy': 'numpy', 
        'scikit-learn': 'sklearn',
        'xgboost': 'xgboost',
        'matplotlib': 'matplotlib',
        'torch': 'torch',
        'lightning': 'lightning'
    }
    
    for framework, version_req in contract.framework_requirements.items():
        import_name = framework_import_map.get(framework, framework)
        
        try:
            __import__(import_name)
            logger.info(f"✓ Framework available: {framework} (imported as {import_name})")
        except ImportError:
            errors.append(f"Required framework not available: {framework} ({version_req})")
    
    if errors:
        error_msg = f"Framework validation failed: {errors}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    logger.info("✅ Framework requirements validation passed")


class ContractEnforcer:
    """
    Context manager for contract enforcement in SageMaker scripts.
    
    Usage:
        with ContractEnforcer(contract) as enforcer:
            # Script logic here
            input_path = enforcer.get_input_path('data_input')
            output_path = enforcer.get_output_path('processed_output')
    """
    
    def __init__(self, contract, required_files: Optional[Dict[str, List[str]]] = None):
        self.contract = contract
        self.required_files = required_files
        
    def __enter__(self):
        """Enter the contract enforcement context"""
        logger.info("🔒 Entering contract enforcement context")
        
        # Log contract summary
        log_contract_summary(self.contract)
        
        # Validate environment
        validate_contract_environment(self.contract)
        
        # Validate framework requirements
        validate_framework_requirements(self.contract)
        
        # Validate required files if specified
        if self.required_files:
            validate_required_files(self.contract, self.required_files)
        
        logger.info("✅ Contract enforcement validation complete")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the contract enforcement context"""
        if exc_type is None:
            logger.info("✅ Script completed successfully within contract")
        else:
            logger.error(f"❌ Script failed with exception: {exc_type.__name__}: {exc_val}")
        
        logger.info("🔓 Exiting contract enforcement context")
        
    def get_input_path(self, logical_name: str) -> str:
        """Get input path by logical name"""
        return get_input_path(self.contract, logical_name)
        
    def get_output_path(self, logical_name: str) -> str:
        """Get output path by logical name"""
        return get_output_path(self.contract, logical_name)
        
    def create_output_file_path(self, logical_name: str, filename: str) -> str:
        """Create output file path"""
        return create_output_file_path(self.contract, logical_name, filename)
        
    def find_files_in_input(self, logical_name: str, pattern: str = "*") -> List[str]:
        """Find files in input directory"""
        return find_files_in_input(self.contract, logical_name, pattern)

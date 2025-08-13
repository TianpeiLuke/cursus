"""
Contract Loader Module

Handles loading and parsing of script contracts from Python files.
Provides robust import handling and contract object extraction.
"""

import os
import sys
import importlib.util
from typing import Dict, Any, Optional
from pathlib import Path


class ContractLoader:
    """
    Loads script contracts from Python files with robust import handling.
    
    Handles:
    - Dynamic module loading with proper sys.path management
    - Multiple contract naming patterns
    - Relative import resolution
    - Contract object to dictionary conversion
    """
    
    def __init__(self, contracts_dir: str):
        """
        Initialize the contract loader.
        
        Args:
            contracts_dir: Directory containing contract files
        """
        self.contracts_dir = Path(contracts_dir)
    
    def load_contract(self, contract_path: Path, contract_name: str) -> Dict[str, Any]:
        """
        Load contract from Python file using robust sys.path management.
        
        Args:
            contract_path: Path to the contract file
            contract_name: Name of the contract
            
        Returns:
            Contract dictionary
            
        Raises:
            Exception: If contract loading fails
        """
        try:
            # Add the project root to sys.path temporarily to handle relative imports
            # Go up to the project root (where src/ is located)
            project_root = str(contract_path.parent.parent.parent.parent)  # Go up to project root
            src_root = str(contract_path.parent.parent.parent)  # Go up to src/ level
            contract_dir = str(contract_path.parent)
            
            paths_to_add = [project_root, src_root, contract_dir]
            added_paths = []
            
            for path in paths_to_add:
                if path not in sys.path:
                    sys.path.insert(0, path)
                    added_paths.append(path)
            
            try:
                # Load the module
                spec = importlib.util.spec_from_file_location(f"{contract_name}_contract", contract_path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Could not load contract module from {contract_path}")
                
                module = importlib.util.module_from_spec(spec)
                
                # Set the module's package to handle relative imports
                module.__package__ = 'cursus.steps.contracts'
                
                spec.loader.exec_module(module)
            finally:
                # Remove added paths from sys.path
                for path in added_paths:
                    if path in sys.path:
                        sys.path.remove(path)
            
            # Look for the contract object - try multiple naming patterns
            contract_obj = self._find_contract_object(module, contract_name)
            
            if contract_obj is None:
                raise AttributeError(f"No contract object found in {contract_path}")
            
            # Convert ScriptContract object to dictionary format
            return self._contract_to_dict(contract_obj, contract_name)
            
        except Exception as e:
            raise Exception(f"Failed to load Python contract from {contract_path}: {str(e)}")
    
    def _find_contract_object(self, module, contract_name: str):
        """
        Find the contract object in the loaded module using multiple naming patterns.
        
        Args:
            module: Loaded Python module
            contract_name: Name of the contract
            
        Returns:
            Contract object or None if not found
        """
        # Try various naming patterns
        possible_names = [
            f"{contract_name.upper()}_CONTRACT",
            f"{contract_name}_CONTRACT", 
            f"{contract_name}_contract",
            "XGBOOST_MODEL_EVAL_CONTRACT",  # Specific for model_evaluation_xgb
            "MODEL_EVALUATION_CONTRACT",  # Legacy fallback
            "CONTRACT",
            "contract"
        ]
        
        # Also try to find any variable ending with _CONTRACT
        for attr_name in dir(module):
            if attr_name.endswith('_CONTRACT') and not attr_name.startswith('_'):
                possible_names.append(attr_name)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_names = []
        for name in possible_names:
            if name not in seen:
                seen.add(name)
                unique_names.append(name)
        
        for name in unique_names:
            if hasattr(module, name):
                contract_obj = getattr(module, name)
                # Verify it's actually a contract object
                if hasattr(contract_obj, 'entry_point'):
                    return contract_obj
        
        return None
    
    def _contract_to_dict(self, contract_obj, contract_name: str) -> Dict[str, Any]:
        """
        Convert ScriptContract object to dictionary format.
        
        Args:
            contract_obj: Contract object
            contract_name: Name of the contract
            
        Returns:
            Contract dictionary
        """
        contract_dict = {
            'entry_point': getattr(contract_obj, 'entry_point', f"{contract_name}.py"),
            'inputs': {},
            'outputs': {},
            'arguments': {},
            'environment_variables': {
                'required': getattr(contract_obj, 'required_env_vars', []),
                'optional': getattr(contract_obj, 'optional_env_vars', {})
            },
            'description': getattr(contract_obj, 'description', ''),
            'framework_requirements': getattr(contract_obj, 'framework_requirements', {})
        }
        
        # Convert expected_input_paths to inputs format
        if hasattr(contract_obj, 'expected_input_paths'):
            for logical_name, path in contract_obj.expected_input_paths.items():
                contract_dict['inputs'][logical_name] = {'path': path}
        
        # Convert expected_output_paths to outputs format
        if hasattr(contract_obj, 'expected_output_paths'):
            for logical_name, path in contract_obj.expected_output_paths.items():
                contract_dict['outputs'][logical_name] = {'path': path}
        
        # Convert expected_arguments to arguments format
        if hasattr(contract_obj, 'expected_arguments'):
            for arg_name, default_value in contract_obj.expected_arguments.items():
                contract_dict['arguments'][arg_name] = {
                    'default': default_value,
                    'required': default_value is None
                }
        
        return contract_dict

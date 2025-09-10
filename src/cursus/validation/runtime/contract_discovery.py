"""
Contract Discovery and Loading Module

Provides intelligent discovery and loading of script contracts for runtime testing.
Maps script names to contracts and adapts contract data for local testing environments.
"""

import importlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from pydantic import BaseModel, Field
from ...core.base.contract_base import ScriptContract


class ContractDiscoveryResult(BaseModel):
    """Result of contract discovery operation"""
    contract: Optional[ScriptContract] = Field(None, description="Discovered contract object")
    contract_name: str = Field(..., description="Name of the contract")
    discovery_method: str = Field(..., description="Method used to discover the contract")
    error_message: Optional[str] = Field(None, description="Error message if discovery failed")
    
    class Config:
        arbitrary_types_allowed = True  # Allow ScriptContract objects


class ContractDiscoveryManager:
    """
    Manager for discovering and loading script contracts.
    
    Handles the mapping between script names and contract objects,
    with intelligent fallback strategies and error recovery.
    """
    
    def __init__(self, test_data_dir: str = "test/integration/runtime"):
        self.test_data_dir = Path(test_data_dir)
        self._contract_cache: Dict[str, ContractDiscoveryResult] = {}
        
        # Common contract name patterns
        self.contract_patterns = [
            "{script_name}_contract",           # direct match
            "{script_name}",                    # simple name
            "{canonical_name}_contract",        # canonical form
            "{canonical_name}",                 # canonical simple
        ]
        
        # Contract module search paths
        self.contract_module_paths = [
            "cursus.steps.contracts",
            "src.cursus.steps.contracts",
            "steps.contracts",
            "contracts"
        ]
    
    def discover_contract(self, script_name: str, canonical_name: Optional[str] = None) -> ContractDiscoveryResult:
        """
        Discover and load contract for a given script name.
        
        Args:
            script_name: snake_case script name (e.g., "tabular_preprocess")
            canonical_name: PascalCase canonical name (e.g., "TabularPreprocessing")
            
        Returns:
            ContractDiscoveryResult with contract and metadata
        """
        # Check cache first
        cache_key = f"{script_name}:{canonical_name or 'None'}"
        if cache_key in self._contract_cache:
            return self._contract_cache[cache_key]
        
        # Try different discovery strategies
        strategies = [
            self._discover_by_direct_import,
            self._discover_by_pattern_matching,
            self._discover_by_fuzzy_search
        ]
        
        for strategy in strategies:
            result = strategy(script_name, canonical_name)
            if result.contract is not None:
                self._contract_cache[cache_key] = result
                return result
        
        # No contract found
        result = ContractDiscoveryResult(
            contract=None,
            contract_name="not_found",
            discovery_method="none",
            error_message=f"No contract found for script '{script_name}'"
        )
        self._contract_cache[cache_key] = result
        return result
    
    def _discover_by_direct_import(self, script_name: str, canonical_name: Optional[str] = None) -> ContractDiscoveryResult:
        """Try direct import using standardization rules from design documents"""
        
        # Following standardization rules: contract files are named {script_name}_contract.py
        # Contract objects are named {SCRIPT_NAME}_CONTRACT
        
        # Generate contract names to try based on standardization rules
        contract_names = []
        
        # Primary pattern: {SCRIPT_NAME}_CONTRACT (standardization rule)
        contract_names.append(f"{script_name.upper()}_CONTRACT")
        
        # Add canonical_name based patterns if available
        if canonical_name:
            canonical_upper = self._to_constant_case(canonical_name)
            contract_names.extend([
                f"{canonical_upper}_CONTRACT",
                f"{canonical_name.upper()}_CONTRACT"
            ])
        
        # Try importing from each module path
        for module_path in self.contract_module_paths:
            try:
                # Contract module naming follows standardization: {script_name}_contract
                contract_module_name = f"{script_name}_contract"
                full_module_path = f"{module_path}.{contract_module_name}"
                
                print(f"Attempting to import contract module: {full_module_path}")
                module = importlib.import_module(full_module_path)
                
                # Try each contract name pattern
                for contract_name in contract_names:
                    if hasattr(module, contract_name):
                        contract_obj = getattr(module, contract_name)
                        if isinstance(contract_obj, ScriptContract):
                            # Validate that contract entry_point matches script_name
                            if hasattr(contract_obj, 'entry_point'):
                                expected_entry_point = f"{script_name}.py"
                                if contract_obj.entry_point == expected_entry_point:
                                    print(f"Found valid contract: {contract_name} with matching entry_point: {expected_entry_point}")
                                    return ContractDiscoveryResult(
                                        contract=contract_obj,
                                        contract_name=contract_name,
                                        discovery_method="direct_import",
                                        error_message=None
                                    )
                                else:
                                    print(f"Contract {contract_name} entry_point mismatch: expected {expected_entry_point}, got {contract_obj.entry_point}")
                            else:
                                # Contract without entry_point - still valid
                                print(f"Found contract: {contract_name} (no entry_point validation)")
                                return ContractDiscoveryResult(
                                    contract=contract_obj,
                                    contract_name=contract_name,
                                    discovery_method="direct_import",
                                    error_message=None
                                )
                
                print(f"Contract module {full_module_path} found but no matching contract objects: {contract_names}")
                
            except ImportError as e:
                print(f"Failed to import contract module {full_module_path}: {e}")
                continue  # Try next module path
            except Exception as e:
                print(f"Error processing contract module {full_module_path}: {e}")
                continue
        
        return ContractDiscoveryResult(
            contract=None,
            contract_name="not_found",
            discovery_method="direct_import",
            error_message=f"No contract module found for script '{script_name}' in paths: {self.contract_module_paths}"
        )
    
    def _discover_by_pattern_matching(self, script_name: str, canonical_name: Optional[str] = None) -> ContractDiscoveryResult:
        """Try pattern-based discovery in contract modules"""
        
        for module_path in self.contract_module_paths:
            try:
                # Import the contracts package
                contracts_module = importlib.import_module(module_path)
                
                # Get all attributes from the module
                for attr_name in dir(contracts_module):
                    if attr_name.startswith('_'):
                        continue
                    
                    attr_obj = getattr(contracts_module, attr_name)
                    if isinstance(attr_obj, ScriptContract):
                        # Check if this contract matches our script using entry_point validation
                        if self._is_contract_match(attr_obj, script_name, canonical_name):
                            return ContractDiscoveryResult(
                                contract=attr_obj,
                                contract_name=attr_name,
                                discovery_method="pattern_matching",
                                error_message=None
                            )
            
            except ImportError:
                continue
        
        return ContractDiscoveryResult(
            contract=None,
            contract_name="not_found",
            discovery_method="pattern_matching",
            error_message="Pattern matching failed"
        )
    
    def _discover_by_fuzzy_search(self, script_name: str, canonical_name: Optional[str] = None) -> ContractDiscoveryResult:
        """Try fuzzy matching against available contracts"""
        
        # This is a placeholder for more sophisticated fuzzy matching
        # Could be implemented using string similarity algorithms
        
        return ContractDiscoveryResult(
            contract=None,
            contract_name="not_found",
            discovery_method="fuzzy_search",
            error_message="Fuzzy search not implemented"
        )
    
    def _is_contract_match(self, contract_obj: ScriptContract, script_name: str, canonical_name: Optional[str] = None) -> bool:
        """
        Check if a contract matches the given script using entry_point field.
        
        This is the most reliable way to validate contract-script matching
        as specified in the design documents.
        """
        # Primary validation: check entry_point field
        if hasattr(contract_obj, 'entry_point') and contract_obj.entry_point:
            expected_entry_point = f"{script_name}.py"
            if contract_obj.entry_point == expected_entry_point:
                print(f"Contract matches script via entry_point: {contract_obj.entry_point}")
                return True
            else:
                print(f"Contract entry_point mismatch: expected {expected_entry_point}, got {contract_obj.entry_point}")
                return False
        
        # Fallback: if no entry_point field, use name-based matching
        print(f"Contract has no entry_point field, falling back to name-based matching")
        return self._fallback_name_matching(contract_obj.__class__.__name__, script_name, canonical_name)
    
    def _fallback_name_matching(self, contract_name: str, script_name: str, canonical_name: Optional[str] = None) -> bool:
        """Fallback name-based matching when entry_point is not available"""
        
        contract_lower = contract_name.lower()
        script_lower = script_name.lower()
        
        # Direct substring match
        if script_lower in contract_lower or contract_lower in script_lower:
            return True
        
        # Check canonical name if available
        if canonical_name:
            canonical_lower = canonical_name.lower()
            if canonical_lower in contract_lower or contract_lower in canonical_lower:
                return True
        
        # Check for common variations
        script_parts = script_lower.split('_')
        contract_parts = contract_lower.split('_')
        
        # If most parts match, consider it a match
        common_parts = set(script_parts) & set(contract_parts)
        if len(common_parts) >= max(1, min(len(script_parts), len(contract_parts)) // 2):
            return True
        
        return False
    
    def _to_constant_case(self, pascal_case: str) -> str:
        """Convert PascalCase to CONSTANT_CASE"""
        # Handle special cases first
        special_cases = {
            'XGBoost': 'XGBOOST',
            'PyTorch': 'PYTORCH',
            'MLFlow': 'MLFLOW',
            'TensorFlow': 'TENSORFLOW',
            'SageMaker': 'SAGEMAKER',
            'AutoML': 'AUTOML'
        }
        
        processed = pascal_case
        for original, replacement in special_cases.items():
            processed = processed.replace(original, replacement)
        
        # Convert to constant case
        result = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', processed)
        result = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', result)
        
        return result.upper()
    
    def get_contract_input_paths(self, contract: ScriptContract, script_name: str) -> Dict[str, str]:
        """
        Extract and adapt input paths from contract for local testing.
        
        Args:
            contract: ScriptContract object
            script_name: snake_case script name for path adaptation
            
        Returns:
            Dictionary of logical_name -> local_path mappings
        """
        if not hasattr(contract, 'expected_input_paths') or not contract.expected_input_paths:
            return {}
        
        adapted_paths = {}
        base_data_dir = self.test_data_dir / 'data' / script_name
        
        for logical_name, original_path in contract.expected_input_paths.items():
            # Adapt SageMaker container paths to local testing paths
            local_path = self._adapt_path_for_local_testing(original_path, base_data_dir, 'input')
            adapted_paths[logical_name] = str(local_path)
        
        return adapted_paths
    
    def get_contract_output_paths(self, contract: ScriptContract, script_name: str) -> Dict[str, str]:
        """
        Extract and adapt output paths from contract for local testing.
        
        Args:
            contract: ScriptContract object
            script_name: snake_case script name for path adaptation
            
        Returns:
            Dictionary of logical_name -> local_path mappings
        """
        if not hasattr(contract, 'expected_output_paths') or not contract.expected_output_paths:
            return {}
        
        adapted_paths = {}
        base_data_dir = self.test_data_dir / 'data' / script_name
        
        for logical_name, original_path in contract.expected_output_paths.items():
            # Adapt SageMaker container paths to local testing paths
            local_path = self._adapt_path_for_local_testing(original_path, base_data_dir, 'output')
            adapted_paths[logical_name] = str(local_path)
        
        return adapted_paths
    
    def get_contract_environ_vars(self, contract: ScriptContract) -> Dict[str, str]:
        """
        Extract environment variables from contract.
        
        Args:
            contract: ScriptContract object
            
        Returns:
            Dictionary of environment variable mappings
        """
        environ_vars = {}
        
        # Add required environment variables
        if hasattr(contract, 'required_env_vars') and contract.required_env_vars:
            for env_var in contract.required_env_vars:
                if isinstance(env_var, dict):
                    environ_vars.update(env_var)
                elif isinstance(env_var, str):
                    # Provide a default value for required env vars
                    environ_vars[env_var] = f"test_value_for_{env_var.lower()}"
        
        # Add optional environment variables with defaults
        if hasattr(contract, 'optional_env_vars') and contract.optional_env_vars:
            for env_var in contract.optional_env_vars:
                if isinstance(env_var, dict):
                    environ_vars.update(env_var)
                elif isinstance(env_var, str):
                    # Provide a default value for optional env vars
                    environ_vars[env_var] = f"default_value_for_{env_var.lower()}"
        
        # Add standard testing environment variables
        environ_vars.update({
            "PYTHONPATH": str(Path("src").resolve()),
            "CURSUS_ENV": "testing"
        })
        
        return environ_vars
    
    def get_contract_job_args(self, contract: ScriptContract, script_name: str) -> Dict[str, Any]:
        """
        Extract job arguments from contract.
        
        Args:
            contract: ScriptContract object
            script_name: snake_case script name
            
        Returns:
            Dictionary of job argument mappings
        """
        job_args = {
            "script_name": script_name,
            "execution_mode": "testing",
            "log_level": "INFO"
        }
        
        # Extract contract-specific job arguments if available
        if hasattr(contract, 'job_args') and contract.job_args:
            job_args.update(contract.job_args)
        
        # Extract from contract metadata if available
        if hasattr(contract, 'metadata') and contract.metadata:
            if 'job_args' in contract.metadata:
                job_args.update(contract.metadata['job_args'])
        
        return job_args
    
    def _adapt_path_for_local_testing(self, original_path: str, base_data_dir: Path, path_type: str) -> Path:
        """
        Adapt SageMaker container paths to local testing paths.
        
        Args:
            original_path: Original path from contract (e.g., "/opt/ml/input/data/training")
            base_data_dir: Base directory for this script's test data
            path_type: 'input' or 'output' for path category
            
        Returns:
            Adapted local path for testing
        """
        # Common SageMaker path patterns to local path mappings
        sagemaker_patterns = {
            '/opt/ml/input/data': base_data_dir / 'input',
            '/opt/ml/output': base_data_dir / 'output',
            '/opt/ml/model': base_data_dir / 'model',
            '/opt/ml/processing/input': base_data_dir / 'input',
            '/opt/ml/processing/output': base_data_dir / 'output',
        }
        
        # Try to match and replace SageMaker paths
        for sagemaker_path, local_base in sagemaker_patterns.items():
            if original_path.startswith(sagemaker_path):
                # Replace the SageMaker prefix with local path
                relative_part = original_path[len(sagemaker_path):].lstrip('/')
                if relative_part:
                    return local_base / relative_part
                else:
                    return local_base
        
        # If no SageMaker pattern matches, create a reasonable local path
        # Extract the last meaningful part of the path
        path_parts = Path(original_path).parts
        if len(path_parts) > 1:
            # Use the last part as the subdirectory
            return base_data_dir / path_type / path_parts[-1]
        else:
            # Use the whole path as subdirectory name
            return base_data_dir / path_type / original_path.strip('/')

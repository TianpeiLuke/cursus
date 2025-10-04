"""
Modernized contract discovery adapters using unified StepCatalog system.

This module provides streamlined adapters that leverage the unified StepCatalog
for discovery operations while maintaining backward compatibility.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Any

from ..step_catalog import StepCatalog

logger = logging.getLogger(__name__)


class ContractDiscoveryResult:
    """
    Legacy result class for contract discovery operations.
    
    Maintains backward compatibility with existing tests and code.
    """
    
    def __init__(self, contract: Optional[Any] = None, contract_name: Optional[str] = None, 
                 discovery_method: str = "step_catalog", error_message: Optional[str] = None):
        """Initialize contract discovery result."""
        self.contract = contract
        self.contract_name = contract_name
        self.discovery_method = discovery_method
        self.error_message = error_message
        self.success = contract is not None and error_message is None
    
    def __repr__(self) -> str:
        if self.success:
            return f"ContractDiscoveryResult(contract={self.contract_name}, method={self.discovery_method})"
        else:
            return f"ContractDiscoveryResult(error={self.error_message})"


class ContractDiscoveryEngineAdapter:
    """
    Modernized adapter using unified StepCatalog for all discovery operations.
    
    Replaces: src/cursus/validation/alignment/discovery/contract_discovery.py
    All methods now use the step catalog's built-in discovery capabilities.
    """
    
    def __init__(self, workspace_root: Path):
        """Initialize with unified catalog."""
        # PORTABLE: Use workspace-aware discovery for contract discovery
        try:
            self.catalog = StepCatalog(workspace_dirs=[workspace_root])
        except Exception as e:
            logger.error(f"Failed to initialize StepCatalog: {e}")
            self.catalog = None
        self.logger = logging.getLogger(__name__)
    
    def discover_contracts_with_scripts(self) -> List[str]:
        """
        MODERNIZED: Use step catalog's built-in method.
        
        This is the primary method used by ContractSpecificationAlignmentTester.
        """
        try:
            if self.catalog is None:
                return []
            # Use step catalog's built-in method - no redundant code needed
            return self.catalog.discover_contracts_with_scripts()
            
        except Exception as e:
            self.logger.error(f"Error discovering contracts with scripts: {e}")
            return []
    
    def discover_all_contracts(self) -> List[str]:
        """
        MODERNIZED: Discover all steps that have contracts using step catalog.
        
        Used by alignment validation system.
        """
        try:
            if self.catalog is None:
                return []
            all_steps = self.catalog.list_available_steps()
            contracts = []
            
            for step in all_steps:
                step_info = self.catalog.get_step_info(step)
                if step_info and 'contract' in step_info.file_components:
                    contracts.append(step)
            
            return contracts
            
        except Exception as e:
            self.logger.error(f"Error discovering all contracts: {e}")
            return []
    
    def extract_contract_reference_from_spec(self, spec_file: str) -> Optional[str]:
        """
        MODERNIZED: Extract contract reference from spec file using step catalog.
        
        Used by alignment validation system to find which step corresponds to a spec file.
        
        Args:
            spec_file: Name of the spec file (e.g., "xgboost_training_spec.py")
            
        Returns:
            Step name if found, None otherwise
        """
        try:
            if self.catalog is None:
                return None
            # Use step catalog to find step by spec component
            step_name = self.catalog.find_step_by_component(spec_file)
            if step_name:
                # Verify the step has a contract
                step_info = self.catalog.get_step_info(step_name)
                if step_info and 'contract' in step_info.file_components:
                    return step_name
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting contract reference from {spec_file}: {e}")
            return None
    
    def build_entry_point_mapping(self) -> Dict[str, str]:
        """
        MODERNIZED: Build entry point mapping from step names to script paths.
        
        Used by tests and legacy systems that expect entry point mappings.
        
        Returns:
            Dictionary mapping step names to script file paths
        """
        try:
            if self.catalog is None:
                return {}
            mapping = {}
            all_steps = self.catalog.list_available_steps()
            
            for step_name in all_steps:
                step_info = self.catalog.get_step_info(step_name)
                if step_info and 'script' in step_info.file_components:
                    script_path = step_info.file_components['script'].path
                    mapping[step_name] = str(script_path)
            
            return mapping
            
        except Exception as e:
            self.logger.error(f"Error building entry point mapping: {e}")
            return {}


class ContractDiscoveryManagerAdapter:
    """
    Modernized adapter using unified StepCatalog with minimal business logic.
    
    Replaces: src/cursus/validation/runtime/contract_discovery.py
    Focuses on test-specific functionality while leveraging step catalog for discovery.
    """
    
    def __init__(self, test_data_dir: Optional[str] = None, workspace_root: Optional[Path] = None):
        """Initialize with unified catalog and test directory."""
        # Support both test_data_dir (for tests) and workspace_root (for production)
        if test_data_dir is not None:
            self.test_data_dir = Path(test_data_dir)
            workspace_root = Path(test_data_dir)
        elif workspace_root is not None:
            self.test_data_dir = workspace_root
        else:
            self.test_data_dir = Path('.')
            workspace_root = Path('.')
            
        # PORTABLE: Use workspace-aware discovery for contract discovery
        self.catalog = StepCatalog(workspace_dirs=[workspace_root])
        self.logger = logging.getLogger(__name__)
        
        # Minimal cache for test performance
        self._contract_cache = {}
    
    def discover_contract(self, step_name: str, canonical_name: Optional[str] = None) -> Optional[str]:
        """
        MODERNIZED: Use StepCatalog's load_contract_class for discovery.
        
        Args:
            step_name: Name of the step/script
            canonical_name: Optional canonical name for the step
            
        Returns:
            String path for backward compatibility, or None if not found
        """
        try:
            # Check cache first
            cache_key = f"{step_name}:{canonical_name}"
            if cache_key in self._contract_cache:
                cached_result = self._contract_cache[cache_key]
                # Return consistent type - always string path or None
                return cached_result
            
            # MODERNIZED: Use StepCatalog's load_contract_class method
            contract = self.catalog.load_contract_class(step_name)
            
            if contract:
                # For backward compatibility with tests, try to get contract path
                step_info = self.catalog.get_step_info(step_name)
                if step_info and step_info.file_components.get('contract'):
                    contract_path = str(step_info.file_components['contract'].path)
                    # Cache the string path for consistency
                    self._contract_cache[cache_key] = contract_path
                    return contract_path
                else:
                    # No file path available, cache None
                    self._contract_cache[cache_key] = None
                    return None
            else:
                # No contract found, cache None
                self._contract_cache[cache_key] = None
                return None
            
        except Exception as e:
            self.logger.error(f"Error discovering contract for {step_name}: {e}")
            return None
    
    
    # Contract analysis methods (business logic - cannot be replaced by step catalog)
    def get_contract_input_paths(self, contract: Any, step_name: str) -> Dict[str, str]:
        """Get contract input paths with local adaptation."""
        try:
            if not hasattr(contract, 'expected_input_paths') or contract.expected_input_paths is None:
                return {}
            
            adapted_paths = {}
            base_data_dir = self.test_data_dir / step_name
            
            for key, path in contract.expected_input_paths.items():
                adapted_path = self._adapt_path_for_local_testing(path, base_data_dir, "input")
                adapted_paths[key] = str(adapted_path)
            
            return adapted_paths
            
        except Exception as e:
            self.logger.error(f"Error getting contract input paths: {e}")
            return {}
    
    def get_contract_output_paths(self, contract: Any, step_name: str) -> Dict[str, str]:
        """Get contract output paths with local adaptation."""
        try:
            if not hasattr(contract, 'expected_output_paths') or contract.expected_output_paths is None:
                return {}
            
            adapted_paths = {}
            base_data_dir = self.test_data_dir / step_name
            
            for key, path in contract.expected_output_paths.items():
                adapted_path = self._adapt_path_for_local_testing(path, base_data_dir, "output")
                adapted_paths[key] = str(adapted_path)
            
            return adapted_paths
            
        except Exception as e:
            self.logger.error(f"Error getting contract output paths: {e}")
            return {}
    
    def get_contract_environ_vars(self, contract: Any) -> Dict[str, str]:
        """Get environment variables from contract."""
        try:
            environ_vars = {
                "PYTHONPATH": "/opt/ml/code",
                "CURSUS_ENV": "testing",
            }
            
            if hasattr(contract, 'required_env_vars') and contract.required_env_vars:
                for var in contract.required_env_vars:
                    if isinstance(var, dict):
                        environ_vars.update(var)
                    else:
                        environ_vars[var] = ""
            
            if hasattr(contract, 'optional_env_vars') and contract.optional_env_vars:
                for var in contract.optional_env_vars:
                    if isinstance(var, dict):
                        environ_vars.update(var)
                    else:
                        environ_vars[var] = ""
            
            return environ_vars
            
        except Exception as e:
            self.logger.error(f"Error getting contract environment variables: {e}")
            return {"CURSUS_ENV": "testing"}
    
    def get_contract_job_args(self, contract: Any, step_name: str) -> Dict[str, Any]:
        """Get job arguments from contract."""
        try:
            job_args = {
                "script_name": step_name,
                "execution_mode": "testing",
                "log_level": "INFO",
            }
            
            if hasattr(contract, 'job_args') and contract.job_args:
                job_args.update(contract.job_args)
            elif hasattr(contract, 'metadata') and contract.metadata and 'job_args' in contract.metadata:
                job_args.update(contract.metadata['job_args'])
            
            return job_args
            
        except Exception as e:
            self.logger.error(f"Error getting contract job args: {e}")
            return {"script_name": step_name, "execution_mode": "testing"}
    
    def _adapt_path_for_local_testing(self, path: str, base_data_dir: Path, path_type: str) -> Path:
        """Adapt SageMaker paths for local testing (test infrastructure - cannot be replaced)."""
        try:
            # Handle SageMaker paths
            if "/opt/ml/" in path:
                if "/input/" in path:
                    # Extract the part after /input/
                    suffix = path.split("/input/", 1)[1] if "/input/" in path else "data"
                    return base_data_dir / "input" / suffix
                elif "/output/" in path:
                    # Extract the part after /output/
                    suffix = path.split("/output/", 1)[1] if "/output/" in path else "data"
                    return base_data_dir / "output" / suffix
                elif "/processing/" in path:
                    if "/processing/input/" in path:
                        suffix = path.split("/processing/input/", 1)[1]
                        return base_data_dir / "input" / suffix
                    elif "/processing/output/" in path:
                        suffix = path.split("/processing/output/", 1)[1]
                        return base_data_dir / "output" / suffix
            
            # Handle custom paths
            path_parts = Path(path).parts
            if len(path_parts) > 1:
                return base_data_dir / path_type / path_parts[-1]
            else:
                return base_data_dir / path_type / "data"
                
        except Exception as e:
            self.logger.warning(f"Error adapting path {path}: {e}")
            return base_data_dir / path_type / "data"

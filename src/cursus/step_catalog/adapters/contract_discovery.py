"""
Contract discovery adapters for backward compatibility.

This module provides adapters that maintain existing contract discovery APIs
during the migration from legacy discovery systems to the unified StepCatalog system.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Any

from ..step_catalog import StepCatalog

logger = logging.getLogger(__name__)


class ContractDiscoveryResult:
    """
    Legacy result class for contract discovery operations.
    
    Maintains backward compatibility with existing tests and code that
    expect ContractDiscoveryResult objects from contract discovery operations.
    """
    
    def __init__(self, contract: Optional[Any] = None, contract_path: Optional[str] = None, 
                 error: Optional[str] = None, cached: bool = False):
        """
        Initialize contract discovery result.
        
        Args:
            contract: The discovered contract object (if any)
            contract_path: Path to the contract file
            error: Error message if discovery failed
            cached: Whether this result was retrieved from cache
        """
        self.contract = contract
        self.contract_path = contract_path
        self.error = error
        self.cached = cached
        self.success = contract is not None and error is None
    
    def __repr__(self) -> str:
        if self.success:
            return f"ContractDiscoveryResult(contract={self.contract}, path={self.contract_path}, cached={self.cached})"
        else:
            return f"ContractDiscoveryResult(error={self.error})"


class ContractDiscoveryEngineAdapter:
    """
    Adapter maintaining backward compatibility with ContractDiscoveryEngine.
    
    Replaces: src/cursus/validation/alignment/discovery/contract_discovery.py
    """
    
    def __init__(self, workspace_root: Path):
        """Initialize with unified catalog."""
        self.catalog = StepCatalog(workspace_root)
        self.logger = logging.getLogger(__name__)
    
    def discover_all_contracts(self) -> List[str]:
        """Legacy method: discover all contracts using unified catalog."""
        try:
            steps = self.catalog.list_available_steps()
            contracts = []
            
            for step_name in steps:
                step_info = self.catalog.get_step_info(step_name)
                if step_info and step_info.file_components.get('contract'):
                    contracts.append(step_name)
            
            self.logger.debug(f"Discovered {len(contracts)} contracts via unified catalog")
            return contracts
            
        except Exception as e:
            self.logger.error(f"Error discovering contracts: {e}")
            return []
    
    def discover_contracts_with_scripts(self) -> List[str]:
        """Legacy method: discover contracts that have associated scripts."""
        try:
            steps = self.catalog.list_available_steps()
            contracts_with_scripts = []
            
            for step_name in steps:
                step_info = self.catalog.get_step_info(step_name)
                if (step_info and 
                    step_info.file_components.get('contract') and 
                    step_info.file_components.get('script')):
                    contracts_with_scripts.append(step_name)
            
            self.logger.debug(f"Discovered {len(contracts_with_scripts)} contracts with scripts")
            return contracts_with_scripts
            
        except Exception as e:
            self.logger.error(f"Error discovering contracts with scripts: {e}")
            return []
    
    def extract_contract_reference_from_spec(self, spec_path: str) -> Optional[str]:
        """Legacy method: extract contract reference from specification."""
        try:
            # Use reverse lookup to find step, then get contract
            step_name = self.catalog.find_step_by_component(spec_path)
            if step_name:
                step_info = self.catalog.get_step_info(step_name)
                if step_info and step_info.file_components.get('contract'):
                    return step_name
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting contract reference from {spec_path}: {e}")
            return None
    
    def build_entry_point_mapping(self) -> Dict[str, str]:
        """Legacy method: build mapping of entry points."""
        try:
            steps = self.catalog.list_available_steps()
            entry_point_mapping = {}
            
            for step_name in steps:
                step_info = self.catalog.get_step_info(step_name)
                if step_info and step_info.file_components.get('script'):
                    script_path = step_info.file_components['script'].path
                    entry_point_mapping[step_name] = str(script_path)
            
            return entry_point_mapping
            
        except Exception as e:
            self.logger.error(f"Error building entry point mapping: {e}")
            return {}


class ContractDiscoveryManagerAdapter:
    """
    Adapter maintaining backward compatibility with ContractDiscoveryManager.
    
    Replaces: src/cursus/validation/runtime/contract_discovery.py
    """
    
    def __init__(self, workspace_root: Path):
        """Initialize with unified catalog."""
        self.catalog = StepCatalog(workspace_root)
        self.logger = logging.getLogger(__name__)
    
    def discover_contract(self, step_name: str) -> Optional[str]:
        """Legacy method: discover contract for a specific step."""
        try:
            step_info = self.catalog.get_step_info(step_name)
            if step_info and step_info.file_components.get('contract'):
                return str(step_info.file_components['contract'].path)
            return None
            
        except Exception as e:
            self.logger.error(f"Error discovering contract for {step_name}: {e}")
            return None
    
    def get_contract_input_paths(self, step_name: str) -> List[str]:
        """Legacy method: get contract input paths."""
        try:
            # This would require parsing the contract file, which is beyond
            # the scope of the unified catalog. Return empty list for now.
            self.logger.warning(f"get_contract_input_paths not fully implemented for {step_name}")
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting contract input paths for {step_name}: {e}")
            return []
    
    def get_contract_output_paths(self, step_name: str) -> List[str]:
        """Legacy method: get contract output paths."""
        try:
            # This would require parsing the contract file, which is beyond
            # the scope of the unified catalog. Return empty list for now.
            self.logger.warning(f"get_contract_output_paths not fully implemented for {step_name}")
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting contract output paths for {step_name}: {e}")
            return []
    
    def _adapt_path_for_local_testing(self, path: str) -> str:
        """Legacy method: adapt path for local testing."""
        # Simple passthrough for now
        return path

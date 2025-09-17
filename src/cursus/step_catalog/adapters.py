"""
Backward compatibility adapters for legacy discovery systems.

This module provides adapters that maintain existing APIs during the migration
from 16+ fragmented discovery systems to the unified StepCatalog system.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Union

from .step_catalog import StepCatalog

logger = logging.getLogger(__name__)


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


class FlexibleFileResolverAdapter:
    """
    Adapter maintaining backward compatibility with FlexibleFileResolver.
    
    Replaces: src/cursus/validation/alignment/file_resolver.py
    """
    
    def __init__(self, workspace_root: Path):
        """Initialize with unified catalog."""
        self.catalog = StepCatalog(workspace_root)
        self.logger = logging.getLogger(__name__)
    
    def find_contract_file(self, step_name: str) -> Optional[Path]:
        """Legacy method: find contract file for step."""
        try:
            step_info = self.catalog.get_step_info(step_name)
            if step_info and step_info.file_components.get('contract'):
                return step_info.file_components['contract'].path
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding contract file for {step_name}: {e}")
            return None
    
    def find_spec_file(self, step_name: str) -> Optional[Path]:
        """Legacy method: find spec file for step."""
        try:
            step_info = self.catalog.get_step_info(step_name)
            if step_info and step_info.file_components.get('spec'):
                return step_info.file_components['spec'].path
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding spec file for {step_name}: {e}")
            return None
    
    def find_builder_file(self, step_name: str) -> Optional[Path]:
        """Legacy method: find builder file for step."""
        try:
            step_info = self.catalog.get_step_info(step_name)
            if step_info and step_info.file_components.get('builder'):
                return step_info.file_components['builder'].path
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding builder file for {step_name}: {e}")
            return None
    
    def find_config_file(self, step_name: str) -> Optional[Path]:
        """Legacy method: find config file for step."""
        try:
            step_info = self.catalog.get_step_info(step_name)
            if step_info and step_info.file_components.get('config'):
                return step_info.file_components['config'].path
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding config file for {step_name}: {e}")
            return None
    
    def find_all_component_files(self, step_name: str) -> Dict[str, Optional[Path]]:
        """Legacy method: find all component files for step."""
        try:
            step_info = self.catalog.get_step_info(step_name)
            if step_info:
                return {
                    component_type: metadata.path if metadata else None
                    for component_type, metadata in step_info.file_components.items()
                }
            return {}
            
        except Exception as e:
            self.logger.error(f"Error finding component files for {step_name}: {e}")
            return {}


class DeveloperWorkspaceFileResolverAdapter(FlexibleFileResolverAdapter):
    """
    Adapter maintaining backward compatibility with DeveloperWorkspaceFileResolver.
    
    Replaces: src/cursus/workspace/validation/workspace_file_resolver.py
    """
    
    def __init__(self, workspace_root: Path, project_id: Optional[str] = None):
        """Initialize with workspace-aware unified catalog."""
        super().__init__(workspace_root)
        self.project_id = project_id
    
    def find_contract_file(self, step_name: str) -> Optional[Path]:
        """Workspace-aware contract file discovery."""
        try:
            # First try workspace-specific lookup
            if self.project_id:
                workspace_steps = self.catalog.list_available_steps(workspace_id=self.project_id)
                if step_name in workspace_steps:
                    step_info = self.catalog.get_step_info(step_name)
                    if step_info and step_info.workspace_id == self.project_id:
                        if step_info.file_components.get('contract'):
                            return step_info.file_components['contract'].path
            
            # Fallback to core lookup
            return super().find_contract_file(step_name)
            
        except Exception as e:
            self.logger.error(f"Error finding workspace contract file for {step_name}: {e}")
            return None


class WorkspaceDiscoveryManagerAdapter:
    """
    Adapter maintaining backward compatibility with WorkspaceDiscoveryManager.
    
    Replaces: src/cursus/workspace/core/discovery.py
    """
    
    def __init__(self, workspace_root: Path):
        """Initialize with unified catalog."""
        self.catalog = StepCatalog(workspace_root)
        self.logger = logging.getLogger(__name__)
    
    def discover_workspaces(self) -> List[str]:
        """Legacy method: discover available workspaces."""
        try:
            # Get all workspace IDs from the catalog
            metrics = self.catalog.get_metrics_report()
            # For now, return the workspaces we know about
            return ["core"]  # Could be extended to discover actual workspace directories
            
        except Exception as e:
            self.logger.error(f"Error discovering workspaces: {e}")
            return []
    
    def discover_components(self, workspace_id: Optional[str] = None) -> List[str]:
        """Legacy method: discover components in workspace."""
        try:
            return self.catalog.list_available_steps(workspace_id=workspace_id)
            
        except Exception as e:
            self.logger.error(f"Error discovering components for workspace {workspace_id}: {e}")
            return []
    
    def resolve_cross_workspace_dependencies(self, step_name: str) -> Dict[str, Any]:
        """Legacy method: resolve cross-workspace dependencies."""
        try:
            step_info = self.catalog.get_step_info(step_name)
            if step_info:
                return {
                    'step_name': step_info.step_name,
                    'workspace_id': step_info.workspace_id,
                    'components': list(step_info.file_components.keys()),
                    'registry_data': step_info.registry_data
                }
            return {}
            
        except Exception as e:
            self.logger.error(f"Error resolving dependencies for {step_name}: {e}")
            return {}


class HybridFileResolverAdapter:
    """
    Adapter maintaining backward compatibility with HybridFileResolver.
    
    Replaces: src/cursus/validation/alignment/patterns/file_resolver.py
    """
    
    def __init__(self, workspace_root: Path):
        """Initialize with unified catalog."""
        self.catalog = StepCatalog(workspace_root)
        self.logger = logging.getLogger(__name__)
    
    def resolve_file_pattern(self, pattern: str, component_type: str) -> List[Path]:
        """Legacy method: resolve files matching pattern."""
        try:
            results = []
            steps = self.catalog.list_available_steps()
            
            for step_name in steps:
                if pattern.lower() in step_name.lower():
                    step_info = self.catalog.get_step_info(step_name)
                    if step_info and step_info.file_components.get(component_type):
                        results.append(step_info.file_components[component_type].path)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error resolving file pattern {pattern}: {e}")
            return []


# Legacy wrapper for backward compatibility during migration
class LegacyDiscoveryWrapper:
    """
    Wrapper providing legacy discovery interfaces during migration period.
    
    This class provides a unified interface that can be used as a drop-in
    replacement for legacy discovery systems during the migration phase.
    """
    
    def __init__(self, workspace_root: Path):
        """Initialize with all legacy adapters."""
        self.workspace_root = workspace_root
        self.catalog = StepCatalog(workspace_root)
        
        # Initialize all adapters
        self.contract_discovery_engine = ContractDiscoveryEngineAdapter(workspace_root)
        self.contract_discovery_manager = ContractDiscoveryManagerAdapter(workspace_root)
        self.flexible_file_resolver = FlexibleFileResolverAdapter(workspace_root)
        self.workspace_discovery_manager = WorkspaceDiscoveryManagerAdapter(workspace_root)
        self.hybrid_file_resolver = HybridFileResolverAdapter(workspace_root)
        
        self.logger = logging.getLogger(__name__)
    
    def get_adapter(self, adapter_type: str) -> Any:
        """Get specific legacy adapter by type."""
        adapters = {
            'contract_discovery_engine': self.contract_discovery_engine,
            'contract_discovery_manager': self.contract_discovery_manager,
            'flexible_file_resolver': self.flexible_file_resolver,
            'workspace_discovery_manager': self.workspace_discovery_manager,
            'hybrid_file_resolver': self.hybrid_file_resolver,
        }
        
        return adapters.get(adapter_type)
    
    def get_unified_catalog(self) -> StepCatalog:
        """Get the underlying unified catalog."""
        return self.catalog

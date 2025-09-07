"""
Workspace component registry for discovering and managing workspace components.

This module provides registry functionality for workspace component discovery,
caching, and management, integrating with the unified registry system to eliminate redundancy.
"""

from typing import Dict, List, Any, Optional, Type, Set
from pathlib import Path
import logging
import time
from collections import defaultdict

from ...core.base import StepBuilderBase, BasePipelineConfig
from ...registry.builder_registry import StepBuilderRegistry
from ...registry.step_names import STEP_NAMES

logger = logging.getLogger(__name__)


class WorkspaceComponentRegistry:
    """
    Registry for workspace component discovery and management.
    
    Refactored to use UnifiedRegistryManager internally to eliminate redundancy
    and provide a consistent workspace-aware registry experience.
    """
    
    def __init__(self, workspace_root: str, discovery_manager: Optional['WorkspaceDiscoveryManager'] = None):
        """
        Initialize workspace component registry using UnifiedRegistryManager.
        
        Args:
            workspace_root: Root path of the workspace
            discovery_manager: Optional consolidated WorkspaceDiscoveryManager instance
        """
        self.workspace_root = workspace_root
        
        # Initialize unified registry manager for consistent workspace handling
        try:
            from ...registry.hybrid.manager import UnifiedRegistryManager
            self.unified_manager = UnifiedRegistryManager(workspaces_root=workspace_root)
            self._unified_available = True
            logger.info(f"Initialized workspace component registry with UnifiedRegistryManager for: {workspace_root}")
        except ImportError:
            logger.warning("UnifiedRegistryManager not available, using fallback implementation")
            self.unified_manager = None
            self._unified_available = False
        
        # Legacy discovery manager support for backward compatibility
        if discovery_manager:
            self.discovery_manager = discovery_manager
            self.workspace_manager = discovery_manager.workspace_manager
        else:
            # Create minimal workspace manager to avoid circular imports
            self.workspace_manager = None
            self.discovery_manager = None
        
        # Initialize cache attributes for backward compatibility (always present)
        self._component_cache = self.discovery_manager._component_cache if self.discovery_manager else {}
        self._builder_cache: Dict[str, Type[StepBuilderBase]] = {}
        self._config_cache: Dict[str, Type[BasePipelineConfig]] = {}
        self._cache_timestamp: Dict[str, float] = {}
        self.cache_expiry = 300
        
        # Core registry for fallback
        self.core_registry = StepBuilderRegistry()
    
    def discover_components(self, developer_id: str = None) -> Dict[str, Any]:
        """
        Discover components in workspace(s) using unified caching when available.
        
        Args:
            developer_id: Optional specific developer ID to discover components for
            
        Returns:
            Dictionary containing discovered components
        """
        cache_key = f"components_{developer_id or 'all'}"
        
        # Use unified caching when available
        if self._unified_available:
            cached_components = self.unified_manager.get_component_cache(cache_key)
            if cached_components:
                logger.debug(f"Returning cached components from UnifiedRegistryManager for {cache_key}")
                return cached_components
        elif self._is_cache_valid(cache_key):
            logger.debug(f"Returning cached components for {cache_key}")
            return self._component_cache[cache_key]
        
        logger.info(f"Discovering components for developer: {developer_id or 'all'}")
        start_time = time.time()
        
        components = {
            'builders': {},
            'configs': {},
            'contracts': {},
            'specs': {},
            'scripts': {},
            'summary': {
                'total_components': 0,
                'developers': [],
                'step_types': set()
            }
        }
        
        try:
            # Get workspace info
            if self.workspace_manager:
                workspace_info = self.workspace_manager.discover_workspaces()
                
                if developer_id:
                    # Discover components for specific developer
                    if developer_id in workspace_info.developers:
                        self._discover_developer_components(developer_id, components)
                    else:
                        logger.warning(f"Developer {developer_id} not found in workspace")
                else:
                    # Discover components for all developers
                    for dev_id in workspace_info.developers:
                        self._discover_developer_components(dev_id, components)
                
                # Update summary
                components['summary']['developers'] = list(workspace_info.developers)
            else:
                # Fallback when workspace_manager is not available
                logger.warning("Workspace manager not available, using fallback component discovery")
                components['summary']['developers'] = []
            
            components['summary']['step_types'] = list(components['summary']['step_types'])
            components['summary']['total_components'] = (
                len(components['builders']) + 
                len(components['configs']) + 
                len(components['contracts']) + 
                len(components['specs']) + 
                len(components['scripts'])
            )
            
            # Cache the results using unified caching when available
            if self._unified_available:
                self.unified_manager.set_component_cache(cache_key, components)
            else:
                self._component_cache[cache_key] = components
                self._cache_timestamp[cache_key] = time.time()
            
            elapsed_time = time.time() - start_time
            logger.info(f"Discovered {components['summary']['total_components']} components in {elapsed_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error discovering components: {e}")
            # Return empty structure on error
            components['summary']['error'] = str(e)
        
        return components
    
    def _discover_developer_components(self, developer_id: str, components: Dict[str, Any]) -> None:
        """Discover components for a specific developer."""
        try:
            # Get file resolver and module loader for this developer
            file_resolver = self.workspace_manager.get_file_resolver(developer_id)
            module_loader = self.workspace_manager.get_module_loader(developer_id)
            
            # Discover different component types
            self._discover_builders(developer_id, file_resolver, module_loader, components)
            self._discover_configs(developer_id, file_resolver, module_loader, components)
            self._discover_contracts(developer_id, file_resolver, components)
            self._discover_specs(developer_id, file_resolver, components)
            self._discover_scripts(developer_id, file_resolver, components)
            
        except Exception as e:
            logger.error(f"Error discovering components for developer {developer_id}: {e}")
    
    def _discover_builders(self, developer_id: str, file_resolver: Any, 
                          module_loader: Any, components: Dict[str, Any]) -> None:
        """Discover builder components."""
        try:
            builder_modules = module_loader.discover_workspace_modules('builders')
            
            for step_name, module_files in builder_modules.items():
                for module_file in module_files:
                    try:
                        # Load builder class
                        builder_class = module_loader.load_builder_class(step_name)
                        if builder_class:
                            components['builders'][f"{developer_id}:{step_name}"] = {
                                'developer_id': developer_id,
                                'step_name': step_name,
                                'class_name': builder_class.__name__,
                                'module_file': module_file,
                                'step_type': getattr(builder_class, 'step_type', 'Unknown')
                            }
                            components['summary']['step_types'].add(getattr(builder_class, 'step_type', 'Unknown'))
                    except Exception as e:
                        logger.warning(f"Could not load builder {step_name} for {developer_id}: {e}")
        except Exception as e:
            logger.warning(f"Error discovering builders for {developer_id}: {e}")
    
    def _discover_configs(self, developer_id: str, file_resolver: Any,
                         module_loader: Any, components: Dict[str, Any]) -> None:
        """Discover config components."""
        try:
            config_modules = module_loader.discover_workspace_modules('configs')
            
            for step_name, module_files in config_modules.items():
                for module_file in module_files:
                    try:
                        # Load config class
                        config_class = module_loader.load_contract_class(step_name)  # Reuse contract loading logic
                        if config_class:
                            components['configs'][f"{developer_id}:{step_name}"] = {
                                'developer_id': developer_id,
                                'step_name': step_name,
                                'class_name': config_class.__name__,
                                'module_file': module_file
                            }
                    except Exception as e:
                        logger.warning(f"Could not load config {step_name} for {developer_id}: {e}")
        except Exception as e:
            logger.warning(f"Error discovering configs for {developer_id}: {e}")
    
    def _discover_contracts(self, developer_id: str, file_resolver: Any,
                           components: Dict[str, Any]) -> None:
        """Discover contract components."""
        try:
            # Use file resolver to find contract files
            workspace_path = Path(file_resolver.workspace_root) / developer_id
            contracts_path = workspace_path / "contracts"
            
            if contracts_path.exists():
                for contract_file in contracts_path.glob("*.py"):
                    if contract_file.name != "__init__.py":
                        step_name = contract_file.stem.replace("_contract", "").replace("contract_", "")
                        components['contracts'][f"{developer_id}:{step_name}"] = {
                            'developer_id': developer_id,
                            'step_name': step_name,
                            'file_path': str(contract_file)
                        }
        except Exception as e:
            logger.warning(f"Error discovering contracts for {developer_id}: {e}")
    
    def _discover_specs(self, developer_id: str, file_resolver: Any,
                       components: Dict[str, Any]) -> None:
        """Discover spec components."""
        try:
            # Use file resolver to find spec files
            workspace_path = Path(file_resolver.workspace_root) / developer_id
            specs_path = workspace_path / "specs"
            
            if specs_path.exists():
                for spec_file in specs_path.glob("*.py"):
                    if spec_file.name != "__init__.py":
                        step_name = spec_file.stem.replace("_spec", "").replace("spec_", "")
                        components['specs'][f"{developer_id}:{step_name}"] = {
                            'developer_id': developer_id,
                            'step_name': step_name,
                            'file_path': str(spec_file)
                        }
        except Exception as e:
            logger.warning(f"Error discovering specs for {developer_id}: {e}")
    
    def _discover_scripts(self, developer_id: str, file_resolver: Any,
                         components: Dict[str, Any]) -> None:
        """Discover script components."""
        try:
            # Use file resolver to find script files
            workspace_path = Path(file_resolver.workspace_root) / developer_id
            scripts_path = workspace_path / "scripts"
            
            if scripts_path.exists():
                for script_file in scripts_path.glob("*.py"):
                    if script_file.name != "__init__.py":
                        step_name = script_file.stem
                        components['scripts'][f"{developer_id}:{step_name}"] = {
                            'developer_id': developer_id,
                            'step_name': step_name,
                            'file_path': str(script_file)
                        }
        except Exception as e:
            logger.warning(f"Error discovering scripts for {developer_id}: {e}")
    
    def find_builder_class(self, step_name: str, developer_id: str = None) -> Optional[Type[StepBuilderBase]]:
        """
        Find builder class for a step using UnifiedRegistryManager when available.
        
        Args:
            step_name: Name of the step
            developer_id: Optional developer ID to search in
            
        Returns:
            Builder class if found, None otherwise
        """
        # Use UnifiedRegistryManager for step resolution when available
        if self._unified_available:
            try:
                # Get step definition from unified manager
                step_def = self.unified_manager.get_step_definition(step_name, developer_id)
                if step_def:
                    # Use the builder_step_name to find the actual builder class
                    builder_name = step_def.builder_step_name
                    # Try to load builder class using workspace manager or core registry
                    if self.workspace_manager and developer_id:
                        module_loader = self.workspace_manager.get_module_loader(developer_id)
                        builder_class = module_loader.load_builder_class(step_name)
                        if builder_class:
                            return builder_class
                    
                    # Fallback to core registry
                    builder_class = self.core_registry.get_builder_for_step_type(step_name)
                    if builder_class:
                        return builder_class
                        
            except Exception as e:
                logger.debug(f"UnifiedRegistryManager lookup failed for {step_name}: {e}")
        
        # Fallback to legacy implementation
        cache_key = f"builder_{developer_id or 'any'}_{step_name}"
        
        # Check cache first
        if cache_key in self._builder_cache:
            return self._builder_cache[cache_key]
        
        try:
            if self.workspace_manager:
                if developer_id:
                    # Search in specific developer workspace
                    module_loader = self.workspace_manager.get_module_loader(developer_id)
                    builder_class = module_loader.load_builder_class(step_name)
                    if builder_class:
                        self._builder_cache[cache_key] = builder_class
                        return builder_class
                else:
                    # Search in all developer workspaces
                    workspace_info = self.workspace_manager.discover_workspaces()
                    for dev_id in workspace_info.developers:
                        module_loader = self.workspace_manager.get_module_loader(dev_id)
                        builder_class = module_loader.load_builder_class(step_name)
                        if builder_class:
                            if not self._unified_available:
                                self._builder_cache[cache_key] = builder_class
                            return builder_class
            
            # Final fallback to core registry
            builder_class = self.core_registry.get_builder_for_step_type(step_name)
            if builder_class:
                if not self._unified_available:
                    self._builder_cache[cache_key] = builder_class
                return builder_class
            
        except Exception as e:
            logger.error(f"Error finding builder class for {step_name}: {e}")
        
        return None
    
    def find_config_class(self, step_name: str, developer_id: str = None) -> Optional[Type[BasePipelineConfig]]:
        """
        Find config class for a step.
        
        Args:
            step_name: Name of the step
            developer_id: Optional developer ID to search in
            
        Returns:
            Config class if found, None otherwise
        """
        cache_key = f"config_{developer_id or 'any'}_{step_name}"
        
        # Check cache first
        if cache_key in self._config_cache:
            return self._config_cache[cache_key]
        
        try:
            if developer_id:
                # Search in specific developer workspace
                module_loader = self.workspace_manager.get_module_loader(developer_id)
                config_class = module_loader.load_contract_class(step_name)  # Reuse contract loading
                if config_class:
                    self._config_cache[cache_key] = config_class
                    return config_class
            else:
                # Search in all developer workspaces
                workspace_info = self.workspace_manager.discover_workspaces()
                for dev_id in workspace_info.developers:
                    module_loader = self.workspace_manager.get_module_loader(dev_id)
                    config_class = module_loader.load_contract_class(step_name)
                    if config_class:
                        self._config_cache[cache_key] = config_class
                        return config_class
            
            # Fallback to core registry (would need to be implemented)
            # For now, return None
            
        except Exception as e:
            logger.error(f"Error finding config class for {step_name}: {e}")
        
        return None
    
    def get_workspace_summary(self) -> Dict[str, Any]:
        """Get summary of workspace components."""
        try:
            components = self.discover_components()
            return {
                'workspace_root': self.workspace_root,
                'total_components': components['summary']['total_components'],
                'developers': components['summary']['developers'],
                'step_types': components['summary']['step_types'],
                'component_counts': {
                    'builders': len(components['builders']),
                    'configs': len(components['configs']),
                    'contracts': len(components['contracts']),
                    'specs': len(components['specs']),
                    'scripts': len(components['scripts'])
                }
            }
        except Exception as e:
            logger.error(f"Error getting workspace summary: {e}")
            return {
                'workspace_root': self.workspace_root,
                'error': str(e)
            }
    
    def validate_component_availability(self, workspace_config) -> Dict[str, Any]:
        """
        Validate component availability for pipeline assembly.
        
        Args:
            workspace_config: WorkspacePipelineConfig instance
            
        Returns:
            Validation result dictionary
        """
        validation_result = {
            'valid': True,
            'missing_components': [],
            'available_components': [],
            'warnings': []
        }
        
        try:
            for step in workspace_config.steps:
                step_key = f"{step.developer_id}:{step.step_name}"
                
                # Check if builder is available
                builder_class = self.find_builder_class(step.step_name, step.developer_id)
                if builder_class:
                    validation_result['available_components'].append({
                        'step_name': step.step_name,
                        'developer_id': step.developer_id,
                        'component_type': 'builder',
                        'class_name': builder_class.__name__
                    })
                else:
                    validation_result['valid'] = False
                    validation_result['missing_components'].append({
                        'step_name': step.step_name,
                        'developer_id': step.developer_id,
                        'component_type': 'builder'
                    })
                
                # Check if config is available
                config_class = self.find_config_class(step.step_name, step.developer_id)
                if config_class:
                    validation_result['available_components'].append({
                        'step_name': step.step_name,
                        'developer_id': step.developer_id,
                        'component_type': 'config',
                        'class_name': config_class.__name__
                    })
                else:
                    validation_result['warnings'].append(
                        f"Config class not found for {step.step_name} (developer: {step.developer_id})"
                    )
        
        except Exception as e:
            validation_result['valid'] = False
            validation_result['error'] = str(e)
            logger.error(f"Error validating component availability: {e}")
        
        return validation_result
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid."""
        if cache_key not in self._cache_timestamp:
            return False
        
        elapsed = time.time() - self._cache_timestamp[cache_key]
        return elapsed < self.cache_expiry
    
    def clear_cache(self) -> None:
        """Clear all caches."""
        self._component_cache.clear()
        self._builder_cache.clear()
        self._config_cache.clear()
        self._cache_timestamp.clear()
        logger.info("Cleared workspace component registry cache")

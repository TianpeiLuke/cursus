"""
Workspace Discovery Manager

Manages cross-workspace component discovery and resolution.
This module provides comprehensive component discovery across multiple workspaces,
dependency resolution, and component compatibility analysis.

Features:
- Cross-workspace component discovery and inventory
- Component dependency resolution and analysis
- Component compatibility validation
- Component metadata caching and management
- File resolver and module loader integration
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Set
import logging
from collections import defaultdict

from ..validation import DeveloperWorkspaceFileResolver, WorkspaceModuleLoader

logger = logging.getLogger(__name__)


class ComponentInventory:
    """Inventory of discovered workspace components."""
    
    def __init__(self):
        """Initialize component inventory."""
        self.builders: Dict[str, Dict[str, Any]] = {}
        self.configs: Dict[str, Dict[str, Any]] = {}
        self.contracts: Dict[str, Dict[str, Any]] = {}
        self.specs: Dict[str, Dict[str, Any]] = {}
        self.scripts: Dict[str, Dict[str, Any]] = {}
        self.summary: Dict[str, Any] = {
            'total_components': 0,
            'developers': [],
            'step_types': set()
        }
    
    def add_component(
        self,
        component_type: str,
        component_id: str,
        component_info: Dict[str, Any]
    ) -> None:
        """Add component to inventory."""
        if component_type == "builders":
            self.builders[component_id] = component_info
        elif component_type == "configs":
            self.configs[component_id] = component_info
        elif component_type == "contracts":
            self.contracts[component_id] = component_info
        elif component_type == "specs":
            self.specs[component_id] = component_info
        elif component_type == "scripts":
            self.scripts[component_id] = component_info
        
        # Update summary
        self.summary['total_components'] += 1
        if component_info.get('developer_id') not in self.summary['developers']:
            self.summary['developers'].append(component_info.get('developer_id'))
        if component_info.get('step_type'):
            self.summary['step_types'].add(component_info.get('step_type'))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert inventory to dictionary."""
        return {
            'builders': self.builders,
            'configs': self.configs,
            'contracts': self.contracts,
            'specs': self.specs,
            'scripts': self.scripts,
            'summary': {
                **self.summary,
                'step_types': list(self.summary['step_types'])
            }
        }


class DependencyGraph:
    """Represents component dependency relationships."""
    
    def __init__(self):
        """Initialize dependency graph."""
        self.nodes: Set[str] = set()
        self.edges: List[Tuple[str, str]] = []
        self.metadata: Dict[str, Dict[str, Any]] = {}
    
    def add_component(self, component_id: str, metadata: Dict[str, Any] = None) -> None:
        """Add component to dependency graph."""
        self.nodes.add(component_id)
        if metadata:
            self.metadata[component_id] = metadata
    
    def add_dependency(self, from_component: str, to_component: str) -> None:
        """Add dependency relationship."""
        self.edges.append((from_component, to_component))
    
    def get_dependencies(self, component_id: str) -> List[str]:
        """Get dependencies for a component."""
        return [to_comp for from_comp, to_comp in self.edges if from_comp == component_id]
    
    def get_dependents(self, component_id: str) -> List[str]:
        """Get components that depend on this component."""
        return [from_comp for from_comp, to_comp in self.edges if to_comp == component_id]
    
    def has_circular_dependencies(self) -> bool:
        """Check for circular dependencies."""
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            if node in rec_stack:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for dep in self.get_dependencies(node):
                if has_cycle(dep):
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in self.nodes:
            if node not in visited:
                if has_cycle(node):
                    return True
        
        return False


class WorkspaceDiscoveryManager:
    """
    Cross-workspace component discovery and resolution.
    
    Provides comprehensive component discovery across multiple workspaces,
    dependency resolution, and component compatibility analysis.
    """
    
    def __init__(self, workspace_manager):
        """
        Initialize workspace discovery manager.
        
        Args:
            workspace_manager: Parent WorkspaceManager instance
        """
        self.workspace_manager = workspace_manager
        
        # Component caches
        self._component_cache: Dict[str, ComponentInventory] = {}
        self._dependency_cache: Dict[str, DependencyGraph] = {}
        self._cache_timestamp: Dict[str, float] = {}
        
        # Cache expiration time (5 minutes)
        self.cache_expiry = 300
        
        logger.info("Initialized workspace discovery manager")
    
    def discover_workspaces(self, workspace_root: Path) -> Dict[str, Any]:
        """
        Discover and analyze workspace structure.
        
        Args:
            workspace_root: Root directory to discover
        
        Returns:
            Workspace discovery information
        """
        logger.info(f"Discovering workspaces in: {workspace_root}")
        
        discovery_result = {
            "workspace_root": str(workspace_root),
            "workspaces": [],
            "summary": {
                "total_workspaces": 0,
                "workspace_types": {},
                "total_developers": 0,
                "total_components": 0
            }
        }
        
        try:
            # Discover developer workspaces
            developers_dir = workspace_root / "developers"
            if developers_dir.exists():
                dev_workspaces = self._discover_development(developers_dir)
                discovery_result["workspaces"].extend(dev_workspaces)
                discovery_result["summary"]["total_developers"] = len(dev_workspaces)
            
            # Discover shared workspace
            shared_dir = workspace_root / "shared"
            if shared_dir.exists():
                shared_workspace = self._discover_shared_workspace(shared_dir)
                discovery_result["workspaces"].append(shared_workspace)
            
            # Update summary
            discovery_result["summary"]["total_workspaces"] = len(discovery_result["workspaces"])
            discovery_result["summary"]["workspace_types"] = {
                workspace_type: len([
                    ws for ws in discovery_result["workspaces"]
                    if ws.get("workspace_type") == workspace_type
                ])
                for workspace_type in ["developer", "shared", "test"]
            }
            
            # Calculate total components
            total_components = sum(
                ws.get("component_count", 0) for ws in discovery_result["workspaces"]
            )
            discovery_result["summary"]["total_components"] = total_components
            
            logger.info(f"Discovered {discovery_result['summary']['total_workspaces']} workspaces with {total_components} components")
            return discovery_result
            
        except Exception as e:
            logger.error(f"Failed to discover workspaces: {e}")
            discovery_result["error"] = str(e)
            return discovery_result
    
    def _discover_development(self, developers_dir: Path) -> List[Dict[str, Any]]:
        """Discover developer workspaces."""
        workspaces = []
        
        try:
            for item in developers_dir.iterdir():
                if not item.is_dir():
                    continue
                
                developer_id = item.name
                workspace_info = self._analyze_workspace(item, developer_id, "developer")
                workspaces.append(workspace_info)
        
        except Exception as e:
            logger.error(f"Error discovering developer workspaces: {e}")
        
        return workspaces
    
    def _discover_shared_workspace(self, shared_dir: Path) -> Dict[str, Any]:
        """Discover shared workspace."""
        return self._analyze_workspace(shared_dir, "shared", "shared")
    
    def _analyze_workspace(
        self,
        workspace_path: Path,
        workspace_id: str,
        workspace_type: str
    ) -> Dict[str, Any]:
        """Analyze individual workspace."""
        workspace_info = {
            "workspace_id": workspace_id,
            "workspace_path": str(workspace_path),
            "developer_id": workspace_id if workspace_type == "developer" else None,
            "workspace_type": workspace_type,
            "component_count": 0,
            "module_types": {},
            "last_modified": None,
            "metadata": {}
        }
        
        try:
            cursus_dev_dir = workspace_path / "src" / "cursus_dev" / "steps"
            
            if cursus_dev_dir.exists():
                # Count components by type
                module_dirs = ["builders", "contracts", "specs", "scripts", "configs"]
                total_components = 0
                
                for module_dir in module_dirs:
                    module_path = cursus_dev_dir / module_dir
                    if module_path.exists():
                        py_files = [
                            f for f in module_path.glob("*.py")
                            if f.name != "__init__.py"
                        ]
                        count = len(py_files)
                        workspace_info["module_types"][module_dir] = count
                        total_components += count
                
                workspace_info["component_count"] = total_components
            
            # Get last modified time
            try:
                workspace_info["last_modified"] = str(int(workspace_path.stat().st_mtime))
            except OSError:
                pass
        
        except Exception as e:
            logger.warning(f"Error analyzing workspace {workspace_id}: {e}")
            workspace_info["error"] = str(e)
        
        return workspace_info
    
    def discover_components(
        self,
        workspace_ids: Optional[List[str]] = None,
        developer_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Discover components across workspaces.
        
        Args:
            workspace_ids: Optional list of workspace IDs to search
            developer_id: Optional specific developer ID to search
        
        Returns:
            Dictionary containing discovered components
        """
        cache_key = f"components_{workspace_ids or 'all'}_{developer_id or 'all'}"
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            logger.debug(f"Returning cached components for {cache_key}")
            return self._component_cache[cache_key].to_dict()
        
        logger.info(f"Discovering components for workspaces: {workspace_ids or 'all'}")
        start_time = time.time()
        
        inventory = ComponentInventory()
        
        try:
            if not self.workspace_manager.workspace_root:
                raise ValueError("No workspace root configured")
            
            # Determine which workspaces to search
            target_workspaces = self._determine_target_workspaces(workspace_ids, developer_id)
            
            # Discover components in each workspace
            for workspace_id in target_workspaces:
                try:
                    self._discover_workspace_components(workspace_id, inventory)
                except Exception as e:
                    logger.error(f"Error discovering components in workspace {workspace_id}: {e}")
            
            # Cache the results
            self._component_cache[cache_key] = inventory
            self._cache_timestamp[cache_key] = time.time()
            
            elapsed_time = time.time() - start_time
            logger.info(f"Discovered {inventory.summary['total_components']} components in {elapsed_time:.2f}s")
            
            return inventory.to_dict()
            
        except Exception as e:
            logger.error(f"Failed to discover components: {e}")
            return {"error": str(e)}
    
    def _determine_target_workspaces(
        self,
        workspace_ids: Optional[List[str]],
        developer_id: Optional[str]
    ) -> List[str]:
        """Determine which workspaces to search."""
        if workspace_ids:
            return workspace_ids
        
        if developer_id:
            return [developer_id]
        
        # Search all available workspaces
        target_workspaces = []
        
        # Add developer workspaces
        developers_dir = self.workspace_manager.workspace_root / "developers"
        if developers_dir.exists():
            for item in developers_dir.iterdir():
                if item.is_dir():
                    target_workspaces.append(item.name)
        
        # Add shared workspace if it exists
        shared_dir = self.workspace_manager.workspace_root / "shared"
        if shared_dir.exists():
            target_workspaces.append("shared")
        
        return target_workspaces
    
    def _discover_workspace_components(
        self,
        workspace_id: str,
        inventory: ComponentInventory
    ) -> None:
        """Discover components in a specific workspace."""
        try:
            # Get file resolver and module loader for this workspace
            file_resolver = self.get_file_resolver(workspace_id)
            module_loader = self.get_module_loader(workspace_id)
            
            # Discover different component types
            self._discover_builders(workspace_id, module_loader, inventory)
            self._discover_configs(workspace_id, module_loader, inventory)
            self._discover_contracts(workspace_id, file_resolver, inventory)
            self._discover_specs(workspace_id, file_resolver, inventory)
            self._discover_scripts(workspace_id, file_resolver, inventory)
            
        except Exception as e:
            logger.error(f"Error discovering components for workspace {workspace_id}: {e}")
    
    def _discover_builders(
        self,
        workspace_id: str,
        module_loader: WorkspaceModuleLoader,
        inventory: ComponentInventory
    ) -> None:
        """Discover builder components."""
        try:
            builder_modules = module_loader.discover_workspace_modules('builders')
            
            for step_name, module_files in builder_modules.items():
                for module_file in module_files:
                    try:
                        # Load builder class
                        builder_class = module_loader.load_builder_class(step_name)
                        if builder_class:
                            component_id = f"{workspace_id}:{step_name}"
                            component_info = {
                                'developer_id': workspace_id,
                                'step_name': step_name,
                                'class_name': builder_class.__name__,
                                'module_file': module_file,
                                'step_type': getattr(builder_class, 'step_type', 'Unknown')
                            }
                            inventory.add_component("builders", component_id, component_info)
                    except Exception as e:
                        logger.warning(f"Could not load builder {step_name} for {workspace_id}: {e}")
        except Exception as e:
            logger.warning(f"Error discovering builders for {workspace_id}: {e}")
    
    def _discover_configs(
        self,
        workspace_id: str,
        module_loader: WorkspaceModuleLoader,
        inventory: ComponentInventory
    ) -> None:
        """Discover config components."""
        try:
            config_modules = module_loader.discover_workspace_modules('configs')
            
            for step_name, module_files in config_modules.items():
                for module_file in module_files:
                    try:
                        # Load config class
                        config_class = module_loader.load_contract_class(step_name)
                        if config_class:
                            component_id = f"{workspace_id}:{step_name}"
                            component_info = {
                                'developer_id': workspace_id,
                                'step_name': step_name,
                                'class_name': config_class.__name__,
                                'module_file': module_file
                            }
                            inventory.add_component("configs", component_id, component_info)
                    except Exception as e:
                        logger.warning(f"Could not load config {step_name} for {workspace_id}: {e}")
        except Exception as e:
            logger.warning(f"Error discovering configs for {workspace_id}: {e}")
    
    def _discover_contracts(
        self,
        workspace_id: str,
        file_resolver: DeveloperWorkspaceFileResolver,
        inventory: ComponentInventory
    ) -> None:
        """Discover contract components."""
        try:
            workspace_path = Path(file_resolver.workspace_root) / workspace_id
            contracts_path = workspace_path / "src" / "cursus_dev" / "steps" / "contracts"
            
            if contracts_path.exists():
                for contract_file in contracts_path.glob("*.py"):
                    if contract_file.name != "__init__.py":
                        step_name = contract_file.stem.replace("_contract", "").replace("contract_", "")
                        component_id = f"{workspace_id}:{step_name}"
                        component_info = {
                            'developer_id': workspace_id,
                            'step_name': step_name,
                            'file_path': str(contract_file)
                        }
                        inventory.add_component("contracts", component_id, component_info)
        except Exception as e:
            logger.warning(f"Error discovering contracts for {workspace_id}: {e}")
    
    def _discover_specs(
        self,
        workspace_id: str,
        file_resolver: DeveloperWorkspaceFileResolver,
        inventory: ComponentInventory
    ) -> None:
        """Discover spec components."""
        try:
            workspace_path = Path(file_resolver.workspace_root) / workspace_id
            specs_path = workspace_path / "src" / "cursus_dev" / "steps" / "specs"
            
            if specs_path.exists():
                for spec_file in specs_path.glob("*.py"):
                    if spec_file.name != "__init__.py":
                        step_name = spec_file.stem.replace("_spec", "").replace("spec_", "")
                        component_id = f"{workspace_id}:{step_name}"
                        component_info = {
                            'developer_id': workspace_id,
                            'step_name': step_name,
                            'file_path': str(spec_file)
                        }
                        inventory.add_component("specs", component_id, component_info)
        except Exception as e:
            logger.warning(f"Error discovering specs for {workspace_id}: {e}")
    
    def _discover_scripts(
        self,
        workspace_id: str,
        file_resolver: DeveloperWorkspaceFileResolver,
        inventory: ComponentInventory
    ) -> None:
        """Discover script components."""
        try:
            workspace_path = Path(file_resolver.workspace_root) / workspace_id
            scripts_path = workspace_path / "src" / "cursus_dev" / "steps" / "scripts"
            
            if scripts_path.exists():
                for script_file in scripts_path.glob("*.py"):
                    if script_file.name != "__init__.py":
                        step_name = script_file.stem
                        component_id = f"{workspace_id}:{step_name}"
                        component_info = {
                            'developer_id': workspace_id,
                            'step_name': step_name,
                            'file_path': str(script_file)
                        }
                        inventory.add_component("scripts", component_id, component_info)
        except Exception as e:
            logger.warning(f"Error discovering scripts for {workspace_id}: {e}")
    
    def resolve_cross_workspace_dependencies(
        self,
        pipeline_definition: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resolve dependencies across workspace boundaries.
        
        Args:
            pipeline_definition: Pipeline definition with cross-workspace dependencies
        
        Returns:
            Resolved dependency information
        """
        logger.info("Resolving cross-workspace dependencies")
        
        resolution_result = {
            "pipeline_definition": pipeline_definition,
            "resolved_dependencies": {},
            "dependency_graph": None,
            "issues": [],
            "warnings": []
        }
        
        try:
            # Create dependency graph
            dep_graph = DependencyGraph()
            
            # Extract steps from pipeline definition
            steps = pipeline_definition.get("steps", [])
            if isinstance(steps, dict):
                steps = list(steps.values())
            
            # Add components to dependency graph
            for step in steps:
                if isinstance(step, dict):
                    step_name = step.get("step_name", "")
                    workspace_id = step.get("developer_id", step.get("workspace_id", ""))
                    component_id = f"{workspace_id}:{step_name}"
                    
                    dep_graph.add_component(component_id, step)
                    
                    # Add dependencies
                    dependencies = step.get("dependencies", [])
                    for dep in dependencies:
                        if ":" not in dep:
                            # Assume same workspace if no workspace specified
                            dep = f"{workspace_id}:{dep}"
                        dep_graph.add_dependency(component_id, dep)
            
            # Check for circular dependencies
            if dep_graph.has_circular_dependencies():
                resolution_result["issues"].append("Circular dependencies detected")
            
            # Resolve component availability
            for component_id in dep_graph.nodes:
                workspace_id, step_name = component_id.split(":", 1)
                
                # Check if component exists
                component_exists = self._check_component_exists(workspace_id, step_name)
                if not component_exists:
                    resolution_result["issues"].append(f"Component not found: {component_id}")
                
                resolution_result["resolved_dependencies"][component_id] = {
                    "workspace_id": workspace_id,
                    "step_name": step_name,
                    "exists": component_exists,
                    "dependencies": dep_graph.get_dependencies(component_id),
                    "dependents": dep_graph.get_dependents(component_id)
                }
            
            resolution_result["dependency_graph"] = {
                "nodes": list(dep_graph.nodes),
                "edges": dep_graph.edges,
                "metadata": dep_graph.metadata
            }
            
            logger.info(f"Resolved {len(dep_graph.nodes)} components with {len(dep_graph.edges)} dependencies")
            return resolution_result
            
        except Exception as e:
            logger.error(f"Failed to resolve cross-workspace dependencies: {e}")
            resolution_result["issues"].append(f"Resolution error: {e}")
            return resolution_result
    
    def _check_component_exists(self, workspace_id: str, step_name: str) -> bool:
        """Check if a component exists in the specified workspace."""
        try:
            # Try to get file resolver for the workspace
            file_resolver = self.get_file_resolver(workspace_id)
            module_loader = self.get_module_loader(workspace_id)
            
            # Check if builder exists
            try:
                builder_class = module_loader.load_builder_class(step_name)
                if builder_class:
                    return True
            except Exception:
                pass
            
            # Check if script exists
            workspace_path = Path(file_resolver.workspace_root) / workspace_id
            scripts_path = workspace_path / "src" / "cursus_dev" / "steps" / "scripts"
            script_file = scripts_path / f"{step_name}.py"
            if script_file.exists():
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error checking component existence {workspace_id}:{step_name}: {e}")
            return False
    
    def get_file_resolver(
        self,
        developer_id: Optional[str] = None,
        **kwargs
    ) -> DeveloperWorkspaceFileResolver:
        """
        Get workspace-aware file resolver.
        
        Args:
            developer_id: Developer to target (uses config default if None)
            **kwargs: Additional arguments for file resolver
        
        Returns:
            Configured DeveloperWorkspaceFileResolver
        """
        if not self.workspace_manager.workspace_root:
            raise ValueError("No workspace root configured")
        
        # Use provided developer_id or fall back to config
        target_developer = developer_id
        if not target_developer and self.workspace_manager.config:
            target_developer = self.workspace_manager.config.developer_id
        
        # Get shared fallback setting
        enable_shared_fallback = kwargs.pop(
            'enable_shared_fallback',
            self.workspace_manager.config.enable_shared_fallback if self.workspace_manager.config else True
        )
        
        return DeveloperWorkspaceFileResolver(
            workspace_root=self.workspace_manager.workspace_root,
            developer_id=target_developer,
            enable_shared_fallback=enable_shared_fallback,
            **kwargs
        )
    
    def get_module_loader(
        self,
        developer_id: Optional[str] = None,
        **kwargs
    ) -> WorkspaceModuleLoader:
        """
        Get workspace-aware module loader.
        
        Args:
            developer_id: Developer to target (uses config default if None)
            **kwargs: Additional arguments for module loader
        
        Returns:
            Configured WorkspaceModuleLoader
        """
        if not self.workspace_manager.workspace_root:
            raise ValueError("No workspace root configured")
        
        # Use provided developer_id or fall back to config
        target_developer = developer_id
        if not target_developer and self.workspace_manager.config:
            target_developer = self.workspace_manager.config.developer_id
        
        # Get settings from config
        enable_shared_fallback = kwargs.pop(
            'enable_shared_fallback',
            self.workspace_manager.config.enable_shared_fallback if self.workspace_manager.config else True
        )
        
        cache_modules = kwargs.pop(
            'cache_modules',
            self.workspace_manager.config.cache_modules if self.workspace_manager.config else True
        )
        
        return WorkspaceModuleLoader(
            workspace_root=self.workspace_manager.workspace_root,
            developer_id=target_developer,
            enable_shared_fallback=enable_shared_fallback,
            cache_modules=cache_modules,
            **kwargs
        )
    
    def list_available_developers(self) -> List[str]:
        """
        Get list of available developer IDs.
        
        Returns:
            List of developer IDs found in the workspace
        """
        logger.info("Listing available developers")
        
        try:
            if not self.workspace_manager.workspace_root:
                return []
            
            developers = []
            developers_dir = self.workspace_manager.workspace_root / "developers"
            
            if developers_dir.exists():
                for item in developers_dir.iterdir():
                    if item.is_dir():
                        developers.append(item.name)
            
            # Add shared workspace if it exists
            shared_dir = self.workspace_manager.workspace_root / "shared"
            if shared_dir.exists():
                developers.append("shared")
            
            return sorted(developers)
            
        except Exception as e:
            logger.error(f"Failed to list available developers: {e}")
            return []
    
    def get_workspace_info(
        self,
        workspace_id: Optional[str] = None,
        developer_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get workspace information.
        
        Args:
            workspace_id: Optional workspace ID to get info for
            developer_id: Optional developer ID to get info for
            
        Returns:
            Workspace information dictionary
        """
        logger.info(f"Getting workspace info for: {workspace_id or developer_id or 'all'}")
        
        try:
            if not self.workspace_manager.workspace_root:
                return {"error": "No workspace root configured"}
            
            # If specific workspace requested, return its info
            target_id = workspace_id or developer_id
            if target_id:
                if target_id == "shared":
                    workspace_path = self.workspace_manager.workspace_root / "shared"
                else:
                    workspace_path = self.workspace_manager.workspace_root / "developers" / target_id
                
                if workspace_path.exists():
                    return self._analyze_workspace(workspace_path, target_id, 
                                                 "shared" if target_id == "shared" else "developer")
                else:
                    return {"error": f"Workspace not found: {target_id}"}
            
            # Return info for all workspaces
            return self.discover_workspaces(self.workspace_manager.workspace_root)
            
        except Exception as e:
            logger.error(f"Failed to get workspace info: {e}")
            return {"error": str(e)}
    
    def refresh_cache(self) -> None:
        """Refresh component discovery cache."""
        logger.info("Refreshing component discovery cache")
        
        try:
            self._component_cache.clear()
            self._dependency_cache.clear()
            self._cache_timestamp.clear()
            
            logger.info("Successfully refreshed component discovery cache")
            
        except Exception as e:
            logger.error(f"Failed to refresh cache: {e}")
            raise
    
    def get_discovery_summary(self) -> Dict[str, Any]:
        """Get summary of discovery activities."""
        try:
            return {
                "cached_discoveries": len(self._component_cache),
                "cache_entries": list(self._component_cache.keys()),
                "last_discovery": max(self._cache_timestamp.values()) if self._cache_timestamp else None,
                "available_developers": len(self.list_available_developers())
            }
        except Exception as e:
            logger.error(f"Failed to get discovery summary: {e}")
            return {"error": str(e)}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get discovery management statistics."""
        try:
            return {
                "discovery_operations": {
                    "cached_discoveries": len(self._component_cache),
                    "available_workspaces": len(self.list_available_developers()),
                    "cache_hit_ratio": self._calculate_cache_hit_ratio()
                },
                "component_summary": self._get_component_summary(),
                "discovery_summary": self.get_discovery_summary()
            }
        except Exception as e:
            logger.error(f"Failed to get discovery statistics: {e}")
            return {"error": str(e)}
    
    def _calculate_cache_hit_ratio(self) -> float:
        """Calculate cache hit ratio."""
        try:
            if not hasattr(self, '_cache_hits'):
                self._cache_hits = 0
            if not hasattr(self, '_cache_misses'):
                self._cache_misses = 0
            
            total_requests = self._cache_hits + self._cache_misses
            if total_requests == 0:
                return 0.0
            
            return self._cache_hits / total_requests
            
        except Exception:
            return 0.0
    
    def _get_component_summary(self) -> Dict[str, Any]:
        """Get summary of discovered components."""
        try:
            total_components = 0
            component_types = defaultdict(int)
            
            for inventory in self._component_cache.values():
                total_components += inventory.summary['total_components']
                for comp_type in ['builders', 'configs', 'contracts', 'specs', 'scripts']:
                    component_types[comp_type] += len(getattr(inventory, comp_type))
            
            return {
                "total_components": total_components,
                "component_types": dict(component_types)
            }
            
        except Exception as e:
            logger.error(f"Failed to get component summary: {e}")
            return {"error": str(e)}
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid."""
        if cache_key not in self._cache_timestamp:
            if hasattr(self, '_cache_misses'):
                self._cache_misses += 1
            else:
                self._cache_misses = 1
            return False
        
        elapsed = time.time() - self._cache_timestamp[cache_key]
        is_valid = elapsed < self.cache_expiry
        
        if is_valid:
            if hasattr(self, '_cache_hits'):
                self._cache_hits += 1
            else:
                self._cache_hits = 1
        else:
            if hasattr(self, '_cache_misses'):
                self._cache_misses += 1
            else:
                self._cache_misses = 1
        
        return is_valid

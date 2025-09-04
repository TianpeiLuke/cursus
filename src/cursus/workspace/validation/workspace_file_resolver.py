"""
Developer Workspace File Resolver

Extends FlexibleFileResolver to support multi-developer workspace structures.
Provides workspace-aware file discovery for contracts, specs, builders, and scripts.

Architecture:
- Extends existing FlexibleFileResolver capabilities
- Supports developer workspace directory structures
- Maintains backward compatibility with single workspace mode
- Provides workspace isolation and path management

Developer Workspace Structure:
development/
├── developers/
│   ├── developer_1/
│   │   └── src/cursus_dev/steps/
│   │       ├── builders/
│   │       ├── contracts/
│   │       ├── scripts/
│   │       ├── specs/
│   │       └── configs/
│   └── developer_2/
│       └── src/cursus_dev/steps/
│           ├── builders/
│           ├── contracts/
│           ├── scripts/
│           ├── specs/
│           └── configs/
└── shared/
    └── src/cursus_dev/steps/
        ├── builders/
        ├── contracts/
        ├── scripts/
        ├── specs/
        └── configs/
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import logging

from ...validation.alignment.file_resolver import FlexibleFileResolver


logger = logging.getLogger(__name__)


class DeveloperWorkspaceFileResolver(FlexibleFileResolver):
    """
    Workspace-aware file resolver that extends FlexibleFileResolver
    to support multi-developer workspace structures.
    
    Features:
    - Developer workspace discovery and validation
    - Workspace-specific file resolution with fallback to shared resources
    - Path isolation between developer workspaces
    - Backward compatibility with single workspace mode
    """
    
    def __init__(
        self,
        workspace_root: Optional[Union[str, Path]] = None,
        developer_id: Optional[str] = None,
        enable_shared_fallback: bool = True,
        **kwargs
    ):
        """
        Initialize workspace-aware file resolver.
        
        Args:
            workspace_root: Root directory containing developer workspaces
            developer_id: Specific developer workspace to target
            enable_shared_fallback: Whether to fallback to shared workspace
            **kwargs: Additional arguments passed to FlexibleFileResolver
        """
        # Initialize parent with default paths if not in workspace mode
        if workspace_root is None:
            # Provide default base directories for single workspace mode
            default_base_directories = kwargs.get('base_directories', {
                'contracts': 'src/cursus/steps/contracts',
                'specs': 'src/cursus/steps/specs', 
                'builders': 'src/cursus/steps/builders',
                'configs': 'src/cursus/steps/configs'
            })
            super().__init__(base_directories=default_base_directories)
            self.workspace_mode = False
            self.workspace_root = None
            self.developer_id = None
            self.enable_shared_fallback = False
            return
            
        self.workspace_mode = True
        self.workspace_root = Path(workspace_root)
        self.developer_id = developer_id
        self.enable_shared_fallback = enable_shared_fallback
        
        # Validate workspace structure
        self._validate_workspace_structure()
        
        # Build workspace-specific paths
        workspace_paths = self._build_workspace_paths()

        # Convert workspace paths to base_directories format expected by FlexibleFileResolver
        base_directories = {
            'contracts': workspace_paths['contracts_dir'],
            'specs': workspace_paths['specs_dir'],
            'builders': workspace_paths['builders_dir'],
            'configs': workspace_paths['configs_dir']
        }

        # Initialize parent with base_directories
        super().__init__(base_directories=base_directories, **kwargs)
        
        # Set workspace-specific attributes for direct access
        for key, value in workspace_paths.items():
            setattr(self, key, value)
        
        logger.info(f"Initialized workspace resolver for developer '{developer_id}' "
                   f"at '{workspace_root}'")
    
    def _validate_workspace_structure(self) -> None:
        """Validate that workspace root has expected structure."""
        if not self.workspace_root.exists():
            raise ValueError(f"Workspace root does not exist: {self.workspace_root}")
        
        developers_dir = self.workspace_root / "developers"
        shared_dir = self.workspace_root / "shared"
        
        if not developers_dir.exists() and not shared_dir.exists():
            raise ValueError(
                f"Workspace root must contain 'developers' or 'shared' directory: "
                f"{self.workspace_root}"
            )
        
        if self.developer_id:
            dev_workspace = developers_dir / self.developer_id
            if not dev_workspace.exists():
                raise ValueError(
                    f"Developer workspace does not exist: {dev_workspace}"
                )
    
    def _build_workspace_paths(self) -> Dict[str, Any]:
        """Build workspace-specific paths for FlexibleFileResolver."""
        paths = {}
        
        if self.developer_id:
            # Primary paths from developer workspace
            dev_base = (self.workspace_root / "developers" / self.developer_id / 
                       "src" / "cursus_dev" / "steps")
            
            if dev_base.exists():
                paths.update({
                    'contracts_dir': str(dev_base / "contracts"),
                    'specs_dir': str(dev_base / "specs"),
                    'builders_dir': str(dev_base / "builders"),
                    'scripts_dir': str(dev_base / "scripts"),
                    'configs_dir': str(dev_base / "configs"),
                })
        
        # Shared fallback paths
        if self.enable_shared_fallback:
            shared_base = (self.workspace_root / "shared" / "src" / 
                          "cursus_dev" / "steps")
            
            if shared_base.exists():
                # Add shared paths as fallback directories
                shared_paths = {
                    'shared_contracts_dir': str(shared_base / "contracts"),
                    'shared_specs_dir': str(shared_base / "specs"),
                    'shared_builders_dir': str(shared_base / "builders"),
                    'shared_scripts_dir': str(shared_base / "scripts"),
                    'shared_configs_dir': str(shared_base / "configs"),
                }
                paths.update(shared_paths)
        
        return paths
    
    def find_contract_file(self, step_name: str) -> Optional[str]:
        """
        Find contract file with workspace-aware search.
        
        Search order:
        1. Developer workspace contracts
        2. Shared workspace contracts (if enabled)
        3. Parent class fallback behavior
        """
        if not self.workspace_mode:
            return super().find_contract_file(step_name)
        
        # Try developer workspace first
        result = super().find_contract_file(step_name)
        if result:
            return result
        
        # Try shared workspace if enabled
        if self.enable_shared_fallback:
            result = self._find_in_shared_workspace(
                'contracts', step_name, None
            )
            if result:
                return result
        
        return None
    
    def find_spec_file(self, step_name: str) -> Optional[str]:
        """
        Find spec file with workspace-aware search.
        
        Search order:
        1. Developer workspace specs
        2. Shared workspace specs (if enabled)
        3. Parent class fallback behavior
        """
        if not self.workspace_mode:
            return super().find_spec_file(step_name)
        
        # Try developer workspace first
        result = super().find_spec_file(step_name)
        if result:
            return result
        
        # Try shared workspace if enabled
        if self.enable_shared_fallback:
            result = self._find_in_shared_workspace(
                'specs', step_name, None
            )
            if result:
                return result
        
        return None
    
    def find_builder_file(self, step_name: str) -> Optional[str]:
        """
        Find builder file with workspace-aware search.
        
        Search order:
        1. Developer workspace builders
        2. Shared workspace builders (if enabled)
        3. Parent class fallback behavior
        """
        if not self.workspace_mode:
            return super().find_builder_file(step_name)
        
        # Try developer workspace first
        result = super().find_builder_file(step_name)
        if result:
            return result
        
        # Try shared workspace if enabled
        if self.enable_shared_fallback:
            result = self._find_in_shared_workspace(
                'builders', step_name, None
            )
            if result:
                return result
        
        return None
    
    def find_script_file(
        self,
        step_name: str,
        script_name: Optional[str] = None
    ) -> Optional[str]:
        """
        Find script file with workspace-aware search.
        
        Search order:
        1. Developer workspace scripts
        2. Shared workspace scripts (if enabled)
        3. Parent class fallback behavior
        """
        if not self.workspace_mode:
            # Parent class doesn't have find_script_file, implement basic logic
            return self._find_file_in_directory(
                getattr(self, 'scripts_dir', 'src/cursus/steps/scripts'),
                step_name,
                script_name,
                ['.py']
            )
        
        # Try developer workspace first
        result = self._find_file_in_directory(
            getattr(self, 'scripts_dir', ''),
            step_name,
            script_name,
            ['.py']
        )
        if result:
            return result
        
        # Try shared workspace if enabled
        if self.enable_shared_fallback:
            result = self._find_in_shared_workspace(
                'scripts', step_name, script_name
            )
            if result:
                return result
        
        return None
    
    def find_config_file(
        self,
        step_name: str,
        config_name: Optional[str] = None
    ) -> Optional[str]:
        """
        Find config file with workspace-aware search.
        
        Search order:
        1. Developer workspace configs
        2. Shared workspace configs (if enabled)
        3. Parent class fallback behavior
        """
        if not self.workspace_mode:
            # Parent class doesn't have find_config_file, implement basic logic
            return self._find_file_in_directory(
                getattr(self, 'configs_dir', 'src/cursus/steps/configs'),
                step_name,
                config_name,
                ['.py']  # Config files are Python files in cursus/steps
            )
        
        # Try developer workspace first using parent class method
        result = super().find_config_file(step_name)
        if result:
            return result
        
        # Try shared workspace if enabled
        if self.enable_shared_fallback:
            result = self._find_in_shared_workspace(
                'configs', step_name, config_name
            )
            if result:
                return result
        
        return None
    
    def _find_in_shared_workspace(
        self,
        file_type: str,
        step_name: str,
        file_name: Optional[str] = None
    ) -> Optional[str]:
        """Find file in shared workspace directory."""
        shared_dir_attr = f'shared_{file_type}_dir'
        shared_dir = getattr(self, shared_dir_attr, None)
        
        if not shared_dir or not os.path.exists(shared_dir):
            return None
        
        # Determine file extensions based on type
        extensions = {
            'contracts': ['.py'],
            'specs': ['.json', '.yaml', '.yml'],
            'builders': ['.py'],
            'scripts': ['.py'],
            'configs': ['.json', '.yaml', '.yml']
        }.get(file_type, ['.py', '.json', '.yaml', '.yml'])
        
        return self._find_file_in_directory(
            shared_dir, step_name, file_name, extensions
        )
    
    def _find_file_in_directory(
        self,
        directory: str,
        step_name: str,
        file_name: Optional[str],
        extensions: List[str]
    ) -> Optional[str]:
        """Find file in specified directory with given extensions."""
        if not directory or not os.path.exists(directory):
            return None
        
        # Create a temporary FlexibleFileResolver for the specific directory
        temp_base_dirs = {}
        for ext in extensions:
            if ext == '.py':
                if 'contracts' not in temp_base_dirs:
                    temp_base_dirs['contracts'] = directory
                if 'builders' not in temp_base_dirs:
                    temp_base_dirs['builders'] = directory
            else:
                if 'specs' not in temp_base_dirs:
                    temp_base_dirs['specs'] = directory
                if 'configs' not in temp_base_dirs:
                    temp_base_dirs['configs'] = directory
        
        if temp_base_dirs:
            temp_resolver = FlexibleFileResolver(temp_base_dirs)
            
            # Try different component types based on extensions
            if '.py' in extensions:
                result = temp_resolver.find_contract_file(step_name)
                if result:
                    return result
                result = temp_resolver.find_builder_file(step_name)
                if result:
                    return result
            
            if any(ext in ['.json', '.yaml', '.yml'] for ext in extensions):
                result = temp_resolver.find_spec_file(step_name)
                if result:
                    return result
        
        # Fallback to basic file search
        search_names = []
        if file_name:
            search_names.append(file_name)
        search_names.append(step_name)
        
        for name in search_names:
            for ext in extensions:
                file_path = os.path.join(directory, f"{name}{ext}")
                if os.path.exists(file_path):
                    return file_path
        
        return None
    
    def get_workspace_info(self) -> Dict[str, Any]:
        """Get information about current workspace configuration."""
        return {
            'workspace_mode': self.workspace_mode,
            'workspace_root': str(self.workspace_root) if self.workspace_root else None,
            'developer_id': self.developer_id,
            'enable_shared_fallback': self.enable_shared_fallback,
            'developer_workspace_exists': (
                self.workspace_root and self.developer_id and
                (self.workspace_root / "developers" / self.developer_id).exists()
            ) if self.workspace_mode else False,
            'shared_workspace_exists': (
                self.workspace_root and
                (self.workspace_root / "shared").exists()
            ) if self.workspace_mode else False
        }
    
    def discover_workspace_components(self) -> Dict[str, Any]:
        """
        Enhanced method with consolidated discovery logic from WorkspaceDiscoveryManager.
        
        PHASE 4 CONSOLIDATION: Moved workspace discovery logic here instead of separate class.
        """
        if not self.workspace_mode or not self.workspace_root:
            return {"error": "Not in workspace mode or no workspace root"}
        
        discovery_result = {
            "workspace_root": str(self.workspace_root),
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
            developers_dir = self.workspace_root / "developers"
            if developers_dir.exists():
                dev_workspaces = self._discover_development(developers_dir)
                discovery_result["workspaces"].extend(dev_workspaces)
                discovery_result["summary"]["total_developers"] = len(dev_workspaces)
            
            # Discover shared workspace
            shared_dir = self.workspace_root / "shared"
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
    
    def discover_components_by_type(self, component_type: str) -> Dict[str, List[str]]:
        """
        Enhanced method to discover components by type across workspaces.
        
        PHASE 4 CONSOLIDATION: Consolidated component discovery logic.
        
        Args:
            component_type: Type of component ('builders', 'contracts', 'specs', 'scripts', 'configs')
        
        Returns:
            Dictionary mapping workspace_id to list of component names
        """
        if not self.workspace_mode or not self.workspace_root:
            return {}
        
        components = {}
        
        try:
            # Discover in developer workspaces
            developers_dir = self.workspace_root / "developers"
            if developers_dir.exists():
                for item in developers_dir.iterdir():
                    if item.is_dir():
                        developer_id = item.name
                        dev_components = self._discover_components_in_workspace(
                            item, developer_id, component_type
                        )
                        if dev_components:
                            components[developer_id] = dev_components
            
            # Discover in shared workspace
            shared_dir = self.workspace_root / "shared"
            if shared_dir.exists():
                shared_components = self._discover_components_in_workspace(
                    shared_dir, "shared", component_type
                )
                if shared_components:
                    components["shared"] = shared_components
            
            return components
            
        except Exception as e:
            logger.error(f"Failed to discover {component_type} components: {e}")
            return {}
    
    def _discover_components_in_workspace(
        self,
        workspace_path: Path,
        workspace_id: str,
        component_type: str
    ) -> List[str]:
        """Discover components of specific type in a workspace."""
        components = []
        
        try:
            component_dir = workspace_path / "src" / "cursus_dev" / "steps" / component_type
            
            if component_dir.exists():
                for component_file in component_dir.glob("*.py"):
                    if component_file.name != "__init__.py":
                        # Extract component name based on type
                        component_name = component_file.stem
                        
                        # Remove common prefixes/suffixes
                        if component_type == "contracts":
                            component_name = component_name.replace("_contract", "").replace("contract_", "")
                        elif component_type == "specs":
                            component_name = component_name.replace("_spec", "").replace("spec_", "")
                        elif component_type == "builders":
                            component_name = component_name.replace("_builder", "").replace("builder_", "")
                        
                        components.append(component_name)
            
            return sorted(components)
            
        except Exception as e:
            logger.warning(f"Error discovering {component_type} in workspace {workspace_id}: {e}")
            return []
    
    def resolve_component_path(
        self,
        component_type: str,
        component_name: str,
        workspace_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Enhanced method to resolve component path with workspace-aware search.
        
        PHASE 4 CONSOLIDATION: Consolidated path resolution logic.
        
        Args:
            component_type: Type of component ('builders', 'contracts', 'specs', 'scripts', 'configs')
            component_name: Name of the component
            workspace_id: Optional workspace ID to search in (uses current if None)
        
        Returns:
            Full path to component file if found, None otherwise
        """
        if not self.workspace_mode:
            # Use parent class methods for non-workspace mode
            if component_type == "contracts":
                return self.find_contract_file(component_name)
            elif component_type == "specs":
                return self.find_spec_file(component_name)
            elif component_type == "builders":
                return self.find_builder_file(component_name)
            elif component_type == "scripts":
                return self.find_script_file(component_name)
            elif component_type == "configs":
                return self.find_config_file(component_name)
            return None
        
        # Use workspace-aware resolution
        target_workspace = workspace_id or self.developer_id
        
        if target_workspace:
            # Search in specific workspace
            result = self._resolve_in_workspace(component_type, component_name, target_workspace)
            if result:
                return result
        
        # Search in shared workspace if enabled
        if self.enable_shared_fallback:
            result = self._resolve_in_workspace(component_type, component_name, "shared")
            if result:
                return result
        
        return None
    
    def _resolve_in_workspace(
        self,
        component_type: str,
        component_name: str,
        workspace_id: str
    ) -> Optional[str]:
        """Resolve component in specific workspace."""
        try:
            if workspace_id == "shared":
                workspace_path = self.workspace_root / "shared"
            else:
                workspace_path = self.workspace_root / "developers" / workspace_id
            
            component_dir = workspace_path / "src" / "cursus_dev" / "steps" / component_type
            
            if not component_dir.exists():
                return None
            
            # Try different naming patterns
            possible_names = [
                component_name,
                f"{component_name}_{component_type.rstrip('s')}",  # e.g., "step_contract"
                f"{component_type.rstrip('s')}_{component_name}",  # e.g., "contract_step"
            ]
            
            extensions = ['.py']
            if component_type in ['specs', 'configs']:
                extensions.extend(['.json', '.yaml', '.yml'])
            
            for name in possible_names:
                for ext in extensions:
                    file_path = component_dir / f"{name}{ext}"
                    if file_path.exists():
                        return str(file_path)
            
            return None
            
        except Exception as e:
            logger.warning(f"Error resolving {component_type}/{component_name} in {workspace_id}: {e}")
            return None
    
    def get_component_statistics(self) -> Dict[str, Any]:
        """
        Enhanced method to get component statistics across workspaces.
        
        PHASE 4 CONSOLIDATION: Consolidated statistics gathering.
        """
        if not self.workspace_mode or not self.workspace_root:
            return {"error": "Not in workspace mode or no workspace root"}
        
        stats = {
            "workspace_root": str(self.workspace_root),
            "total_workspaces": 0,
            "total_components": 0,
            "component_types": {
                "builders": 0,
                "contracts": 0,
                "specs": 0,
                "scripts": 0,
                "configs": 0
            },
            "workspaces": {}
        }
        
        try:
            # Analyze developer workspaces
            developers_dir = self.workspace_root / "developers"
            if developers_dir.exists():
                for item in developers_dir.iterdir():
                    if item.is_dir():
                        workspace_stats = self._get_workspace_statistics(item, item.name)
                        stats["workspaces"][item.name] = workspace_stats
                        stats["total_workspaces"] += 1
                        stats["total_components"] += workspace_stats["total_components"]
                        
                        for comp_type, count in workspace_stats["component_types"].items():
                            stats["component_types"][comp_type] += count
            
            # Analyze shared workspace
            shared_dir = self.workspace_root / "shared"
            if shared_dir.exists():
                workspace_stats = self._get_workspace_statistics(shared_dir, "shared")
                stats["workspaces"]["shared"] = workspace_stats
                stats["total_workspaces"] += 1
                stats["total_components"] += workspace_stats["total_components"]
                
                for comp_type, count in workspace_stats["component_types"].items():
                    stats["component_types"][comp_type] += count
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get component statistics: {e}")
            stats["error"] = str(e)
            return stats
    
    def _get_workspace_statistics(self, workspace_path: Path, workspace_id: str) -> Dict[str, Any]:
        """Get statistics for a specific workspace."""
        stats = {
            "workspace_id": workspace_id,
            "workspace_path": str(workspace_path),
            "total_components": 0,
            "component_types": {
                "builders": 0,
                "contracts": 0,
                "specs": 0,
                "scripts": 0,
                "configs": 0
            }
        }
        
        try:
            cursus_dev_dir = workspace_path / "src" / "cursus_dev" / "steps"
            
            if cursus_dev_dir.exists():
                for comp_type in stats["component_types"].keys():
                    comp_dir = cursus_dev_dir / comp_type
                    if comp_dir.exists():
                        count = len([
                            f for f in comp_dir.glob("*.py")
                            if f.name != "__init__.py"
                        ])
                        stats["component_types"][comp_type] = count
                        stats["total_components"] += count
            
            return stats
            
        except Exception as e:
            logger.warning(f"Error getting statistics for workspace {workspace_id}: {e}")
            stats["error"] = str(e)
            return stats
    
    def list_available_developers(self) -> List[str]:
        """List all available developer workspaces."""
        if not self.workspace_mode or not self.workspace_root:
            return []
        
        developers_dir = self.workspace_root / "developers"
        if not developers_dir.exists():
            return []
        
        developers = []
        for item in developers_dir.iterdir():
            if item.is_dir():
                # Check if it has the expected structure
                cursus_dev_dir = item / "src" / "cursus_dev" / "steps"
                if cursus_dev_dir.exists():
                    developers.append(item.name)
        
        return sorted(developers)
    
    def switch_developer(self, developer_id: str) -> None:
        """Switch to a different developer workspace."""
        if not self.workspace_mode:
            raise ValueError("Not in workspace mode")
        
        if developer_id not in self.list_available_developers():
            raise ValueError(f"Developer workspace not found: {developer_id}")
        
        self.developer_id = developer_id
        
        # Rebuild paths for new developer
        workspace_paths = self._build_workspace_paths()
        
        # Update base directories for FlexibleFileResolver
        new_base_directories = {
            'contracts': workspace_paths['contracts_dir'],
            'specs': workspace_paths['specs_dir'],
            'builders': workspace_paths['builders_dir'],
            'configs': workspace_paths['configs_dir']
        }
        
        # Update parent class base directories and refresh cache
        self.base_dirs = {k: Path(v) for k, v in new_base_directories.items()}
        self._discover_all_files()
        
        # Update instance attributes
        for key, value in workspace_paths.items():
            setattr(self, key, value)
        
        logger.info(f"Switched to developer workspace: {developer_id}")

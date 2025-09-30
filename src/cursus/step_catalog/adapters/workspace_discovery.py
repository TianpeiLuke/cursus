"""
Workspace discovery adapters for backward compatibility.

This module provides adapters that maintain existing workspace discovery APIs
during the migration from legacy discovery systems to the unified StepCatalog system.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from unittest.mock import Mock

from ..step_catalog import StepCatalog

logger = logging.getLogger(__name__)


class WorkspaceDiscoveryManagerAdapter:
    """
    Adapter maintaining backward compatibility with WorkspaceDiscoveryManager.
    
    Replaces: src/cursus/workspace/core/discovery.py
    """
    
    def __init__(self, workspace_root: Path):
        """Initialize with workspace root (updated for test compatibility)."""
        self.workspace_root = workspace_root
        # PORTABLE: Use workspace-aware discovery for workspace discovery
        self.catalog = StepCatalog(workspace_dirs=[workspace_root])
        self.logger = logging.getLogger(__name__)
        
        # Legacy compatibility attributes
        self._component_cache = {}
        self._dependency_cache = {}
        self._cache_timestamp = {}
        self.cache_expiry = 300
    
    def discover_workspaces(self, workspace_root: Path) -> Dict[str, Any]:
        """Legacy method: discover available workspaces."""
        try:
            discovery_result = {
                "workspace_root": str(workspace_root),
                "workspaces": [],
                "summary": {
                    "total_workspaces": 0,
                    "workspace_types": {},
                    "total_developers": 0,
                    "total_components": 0,
                },
            }
            
            # Discover developer workspaces
            developers_dir = workspace_root / "developers"
            if developers_dir.exists():
                for item in developers_dir.iterdir():
                    if item.is_dir():
                        # Count components in this workspace
                        component_count = self._count_workspace_components(item)
                        
                        workspace_info = {
                            "workspace_id": item.name,
                            "workspace_path": str(item),
                            "developer_id": item.name,
                            "workspace_type": "developer",
                            "component_count": component_count,
                        }
                        discovery_result["workspaces"].append(workspace_info)
                        discovery_result["summary"]["total_developers"] += 1
                        discovery_result["summary"]["total_components"] += component_count
            
            # Discover shared workspace
            shared_dir = workspace_root / "shared"
            if shared_dir.exists():
                component_count = self._count_workspace_components(shared_dir)
                
                shared_workspace = {
                    "workspace_id": "shared",
                    "workspace_path": str(shared_dir),
                    "developer_id": None,
                    "workspace_type": "shared",
                    "component_count": component_count,
                }
                discovery_result["workspaces"].append(shared_workspace)
                discovery_result["summary"]["total_components"] += component_count
            
            discovery_result["summary"]["total_workspaces"] = len(discovery_result["workspaces"])
            return discovery_result
            
        except Exception as e:
            self.logger.error(f"Error discovering workspaces: {e}")
            return {"error": str(e)}
    
    def _count_workspace_components(self, workspace_path: Path) -> int:
        """Count components in a workspace directory."""
        try:
            component_count = 0
            cursus_dev_path = workspace_path / "src" / "cursus_dev" / "steps"
            
            if cursus_dev_path.exists():
                # Count builders
                builders_path = cursus_dev_path / "builders"
                if builders_path.exists():
                    component_count += len([f for f in builders_path.glob("*.py") if f.is_file() and not f.name.startswith("__")])
                
                # Count configs
                configs_path = cursus_dev_path / "configs"
                if configs_path.exists():
                    component_count += len([f for f in configs_path.glob("*.py") if f.is_file() and not f.name.startswith("__")])
                
                # Count contracts
                contracts_path = cursus_dev_path / "contracts"
                if contracts_path.exists():
                    component_count += len([f for f in contracts_path.glob("*.py") if f.is_file() and not f.name.startswith("__")])
                
                # Count specs
                specs_path = cursus_dev_path / "specs"
                if specs_path.exists():
                    component_count += len([f for f in specs_path.glob("*.py") if f.is_file() and not f.name.startswith("__")])
                
                # Count scripts
                scripts_path = cursus_dev_path / "scripts"
                if scripts_path.exists():
                    component_count += len([f for f in scripts_path.glob("*.py") if f.is_file() and not f.name.startswith("__")])
            
            return component_count
            
        except Exception as e:
            self.logger.error(f"Error counting components in {workspace_path}: {e}")
            return 0
    
    def discover_components(self, workspace_ids: Optional[List[str]] = None, developer_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Enhanced method: discover ALL component types with workspace-specific functionality.
        
        When workspace_ids is None: focuses on shared package (cursus/steps - "core" workspace)
        When workspace_ids specified: discovers from those specific workspaces
        
        Returns comprehensive inventory of all 5 component types:
        - builders, configs, contracts, specs, scripts
        """
        try:
            # Check if workspace root is configured
            if not self.workspace_root:
                return {"error": "No workspace root configured"}
            
            # Enhanced inventory structure organized by component type and workspace
            inventory = {
                "builders": {},
                "configs": {},
                "contracts": {},
                "specs": {},
                "scripts": {},
                "metadata": {
                    "discovery_timestamp": time.time(),
                    "total_components": 0,
                    "workspaces_scanned": [],
                    "component_counts": {
                        "builders": 0,
                        "configs": 0,
                        "contracts": 0,
                        "specs": 0,
                        "scripts": 0
                    }
                }
            }
            
            # Determine target workspaces
            if workspace_ids is None and developer_id is None:
                # Focus on shared package (core workspace) - mirrors StepCatalog None behavior
                target_workspaces = ["core"]
                self.logger.debug("No workspace constraints specified, focusing on core workspace (shared package)")
            else:
                # Use specified workspace constraints
                target_workspaces = workspace_ids or ([developer_id] if developer_id else [])
                self.logger.debug(f"Discovering components in specified workspaces: {target_workspaces}")
            
            # Discover components using StepCatalog integration
            if self.catalog:
                steps = self.catalog.list_available_steps()
                
                for step_name in steps:
                    step_info = self.catalog.get_step_info(step_name)
                    if not step_info:
                        continue
                    
                    # Filter by target workspaces
                    if step_info.workspace_id not in target_workspaces:
                        continue
                    
                    # Track workspace
                    if step_info.workspace_id not in inventory["metadata"]["workspaces_scanned"]:
                        inventory["metadata"]["workspaces_scanned"].append(step_info.workspace_id)
                    
                    # Discover ALL component types for this step
                    self._discover_step_components(step_info, inventory)
            
            # Additional file system discovery for components not in StepCatalog
            self._discover_filesystem_components(target_workspaces, inventory)
            
            # Finalize metadata
            inventory["metadata"]["total_components"] = sum(inventory["metadata"]["component_counts"].values())
            
            self.logger.info(f"Discovered {inventory['metadata']['total_components']} components across {len(inventory['metadata']['workspaces_scanned'])} workspaces")
            return inventory
            
        except Exception as e:
            self.logger.error(f"Error discovering components: {e}")
            return {"error": str(e)}
    
    def _discover_step_components(self, step_info: Any, inventory: Dict[str, Any]) -> None:
        """
        Discover all component types for a specific step using StepCatalog data.
        
        Args:
            step_info: StepInfo object from StepCatalog
            inventory: Inventory dictionary to populate
        """
        try:
            workspace_id = step_info.workspace_id
            step_name = step_info.step_name
            
            # Process each component type that exists for this step
            for component_type, file_metadata in step_info.file_components.items():
                if file_metadata and component_type in inventory:
                    component_id = f"{workspace_id}:{step_name}"
                    
                    # Create comprehensive component info
                    component_info = {
                        "developer_id": workspace_id,
                        "workspace_id": workspace_id,
                        "step_name": step_name,
                        "component_type": component_type,
                        "file_path": str(file_metadata.path),
                        "modified_time": file_metadata.modified_time.isoformat() if hasattr(file_metadata, 'modified_time') and file_metadata.modified_time else None,
                        "registry_data": step_info.registry_data,
                    }
                    
                    # Add component-specific metadata
                    if hasattr(step_info, 'config_class') and step_info.config_class:
                        component_info["config_class"] = step_info.config_class
                    if hasattr(step_info, 'sagemaker_step_type') and step_info.sagemaker_step_type:
                        component_info["sagemaker_step_type"] = step_info.sagemaker_step_type
                    
                    # Add to inventory
                    if workspace_id not in inventory[component_type]:
                        inventory[component_type][workspace_id] = {}
                    
                    inventory[component_type][workspace_id][component_id] = component_info
                    inventory["metadata"]["component_counts"][component_type] += 1
                    
        except Exception as e:
            self.logger.warning(f"Error discovering components for step {step_info.step_name}: {e}")
    
    def _discover_filesystem_components(self, target_workspaces: List[str], inventory: Dict[str, Any]) -> None:
        """
        Discover additional components via direct filesystem scanning.
        
        This finds components that might not be registered in StepCatalog but exist
        in the workspace directories.
        
        Args:
            target_workspaces: List of workspace IDs to scan
            inventory: Inventory dictionary to populate
        """
        try:
            # Component type directory mapping (mirrors StepCatalog patterns)
            component_dirs = {
                "scripts": "scripts",
                "contracts": "contracts", 
                "specs": "specs",
                "builders": "builders",
                "configs": "configs"
            }
            
            # Scan workspace directories
            for workspace_id in target_workspaces:
                if workspace_id == "core":
                    # Core workspace is at package_root/steps/
                    workspace_path = self.catalog.package_root / "steps"
                else:
                    # Developer workspaces
                    workspace_path = self._find_workspace_path(workspace_id)
                
                if not workspace_path or not workspace_path.exists():
                    continue
                
                # Scan each component type directory
                for component_type, dir_name in component_dirs.items():
                    component_dir = workspace_path / dir_name
                    if not component_dir.exists():
                        continue
                    
                    self._scan_component_directory(component_dir, component_type, workspace_id, inventory)
                    
        except Exception as e:
            self.logger.error(f"Error in filesystem component discovery: {e}")
    
    def _find_workspace_path(self, workspace_id: str) -> Optional[Path]:
        """
        Find the filesystem path for a workspace ID.
        
        Args:
            workspace_id: ID of the workspace
            
        Returns:
            Path to workspace directory, or None if not found
        """
        try:
            # Check if workspace_id matches any of our configured workspace directories
            for workspace_dir in self.catalog.workspace_dirs:
                if workspace_dir.name == workspace_id:
                    return workspace_dir
            
            # Check traditional workspace structure
            developers_dir = self.workspace_root / "developers" / workspace_id
            if developers_dir.exists():
                # Look for src/cursus_dev/steps structure
                steps_dir = developers_dir / "src" / "cursus_dev" / "steps"
                if steps_dir.exists():
                    return steps_dir
            
            # Check shared workspace
            if workspace_id == "shared":
                shared_dir = self.workspace_root / "shared"
                if shared_dir.exists():
                    steps_dir = shared_dir / "src" / "cursus_dev" / "steps"
                    if steps_dir.exists():
                        return steps_dir
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error finding workspace path for {workspace_id}: {e}")
            return None
    
    def _scan_component_directory(self, component_dir: Path, component_type: str, workspace_id: str, inventory: Dict[str, Any]) -> None:
        """
        Scan a component directory for Python files.
        
        Args:
            component_dir: Directory to scan
            component_type: Type of component
            workspace_id: ID of the workspace
            inventory: Inventory dictionary to populate
        """
        try:
            for py_file in component_dir.glob("*.py"):
                if py_file.name.startswith("__"):
                    continue
                
                # Extract step name using StepCatalog patterns
                step_name = self._extract_step_name_from_file(py_file.name, component_type)
                if not step_name:
                    continue
                
                component_id = f"{workspace_id}:{step_name}"
                
                # Check if we already have this component (avoid duplicates)
                if (workspace_id in inventory[component_type] and 
                    component_id in inventory[component_type][workspace_id]):
                    continue
                
                # Create component info
                component_info = {
                    "developer_id": workspace_id,
                    "workspace_id": workspace_id,
                    "step_name": step_name,
                    "component_type": component_type,
                    "file_path": str(py_file),
                    "modified_time": datetime.fromtimestamp(py_file.stat().st_mtime).isoformat(),
                    "registry_data": {},
                    "source": "filesystem_discovery"
                }
                
                # Add to inventory
                if workspace_id not in inventory[component_type]:
                    inventory[component_type][workspace_id] = {}
                
                inventory[component_type][workspace_id][component_id] = component_info
                inventory["metadata"]["component_counts"][component_type] += 1
                
        except Exception as e:
            self.logger.warning(f"Error scanning component directory {component_dir}: {e}")
    
    def _extract_step_name_from_file(self, filename: str, component_type: str) -> Optional[str]:
        """
        Extract step name from filename using StepCatalog patterns.
        
        Args:
            filename: Name of the file
            component_type: Type of component
            
        Returns:
            Extracted step name, or None if not extractable
        """
        try:
            name = filename[:-3]  # Remove .py extension
            
            # Use StepCatalog naming patterns
            if component_type == "contracts" and name.endswith("_contract"):
                return name[:-9]  # Remove _contract
            elif component_type == "specs" and name.endswith("_spec"):
                return name[:-5]  # Remove _spec
            elif component_type == "builders" and name.startswith("builder_") and name.endswith("_step"):
                return name[8:-5]  # Remove builder_ and _step
            elif component_type == "configs" and name.startswith("config_") and name.endswith("_step"):
                return name[7:-5]  # Remove config_ and _step
            elif component_type == "scripts":
                return name  # Scripts use filename as-is
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error extracting step name from {filename}: {e}")
            return None
    
    
    def get_file_resolver(self, developer_id: Optional[str] = None, **kwargs):
        """Legacy method: get file resolver."""
        if not self.workspace_root:
            raise ValueError("No workspace root configured")
        
        from .file_resolver import DeveloperWorkspaceFileResolverAdapter
        return DeveloperWorkspaceFileResolverAdapter(
            self.workspace_root,
            project_id=developer_id
        )
    
    def get_module_loader(self, developer_id: Optional[str] = None, **kwargs):
        """Legacy method: get module loader."""
        if not self.workspace_root:
            raise ValueError("No workspace root configured")
        
        # Return a mock module loader for now
        mock_loader = Mock()
        mock_loader.workspace_root = self.workspace_root
        return mock_loader
    
    def list_available_developers(self) -> List[str]:
        """Legacy method: list available developers."""
        try:
            if not self.workspace_root:
                return []
            
            developers = []
            developers_dir = self.workspace_root / "developers"
            
            if developers_dir.exists():
                for item in developers_dir.iterdir():
                    if item.is_dir():
                        developers.append(item.name)
            
            # Add shared workspace if it exists
            shared_dir = self.workspace_root / "shared"
            if shared_dir.exists():
                developers.append("shared")
            
            return sorted(developers)
            
        except Exception as e:
            self.logger.error(f"Error listing developers: {e}")
            return []
    
    def get_workspace_info(self, workspace_id: Optional[str] = None, developer_id: Optional[str] = None) -> Dict[str, Any]:
        """Legacy method: get workspace info."""
        try:
            if not self.workspace_root:
                return {"error": "No workspace root configured"}
            
            target_id = workspace_id or developer_id
            if target_id:
                if target_id == "shared":
                    workspace_path = self.workspace_root / "shared"
                else:
                    workspace_path = self.workspace_root / "developers" / target_id
                
                if workspace_path.exists():
                    return {
                        "workspace_id": target_id,
                        "workspace_path": str(workspace_path),
                        "workspace_type": "shared" if target_id == "shared" else "developer",
                        "exists": True
                    }
                else:
                    return {"error": f"Workspace not found: {target_id}"}
            
            # Return info for all workspaces
            return self.discover_workspaces(self.workspace_root)
            
        except Exception as e:
            self.logger.error(f"Error getting workspace info: {e}")
            return {"error": str(e)}
    
    def refresh_cache(self) -> None:
        """Legacy method: refresh cache."""
        self._component_cache.clear()
        self._dependency_cache.clear()
        self._cache_timestamp.clear()
    
    def get_discovery_summary(self) -> Dict[str, Any]:
        """Legacy method: get discovery summary."""
        try:
            return {
                "cached_discoveries": len(self._component_cache),
                "cache_entries": list(self._component_cache.keys()),
                "last_discovery": (
                    max(self._cache_timestamp.values())
                    if self._cache_timestamp
                    else None
                ),
                "available_developers": len(self.list_available_developers()),
            }
        except Exception as e:
            self.logger.error(f"Error getting discovery summary: {e}")
            return {"error": str(e)}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Legacy method: get statistics."""
        try:
            return {
                "discovery_operations": {
                    "cached_discoveries": len(self._component_cache),
                    "available_workspaces": len(self.list_available_developers()),
                },
                "component_summary": {"total_components": 0},
                "discovery_summary": self.get_discovery_summary(),
            }
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Legacy method: check cache validity."""
        if cache_key not in self._cache_timestamp:
            return False
        
        import time
        elapsed = time.time() - self._cache_timestamp[cache_key]
        return elapsed < self.cache_expiry

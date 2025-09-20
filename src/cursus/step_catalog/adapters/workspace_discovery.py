"""
Workspace discovery adapters for backward compatibility.

This module provides adapters that maintain existing workspace discovery APIs
during the migration from legacy discovery systems to the unified StepCatalog system.
"""

import logging
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
        """Legacy method: discover components in workspace."""
        try:
            # Check if workspace root is configured
            if not self.workspace_root:
                return {"error": "No workspace root configured"}
            
            from ...workspace.core.inventory import ComponentInventory
            inventory = ComponentInventory()
            
            # Only discover components if we have specific workspace constraints
            # This respects workspace isolation - empty workspaces should return empty results
            if self.catalog and (workspace_ids or developer_id):
                # Filter by specific workspace IDs if provided
                target_workspaces = workspace_ids or ([developer_id] if developer_id else [])
                
                steps = self.catalog.list_available_steps()
                for step_name in steps:
                    step_info = self.catalog.get_step_info(step_name)
                    if step_info and step_info.workspace_id in target_workspaces:
                        component_id = f"{step_info.workspace_id}:{step_name}"
                        component_info = {
                            "developer_id": step_info.workspace_id,
                            "step_name": step_name,
                            "config_class": step_info.config_class,
                            "sagemaker_step_type": step_info.sagemaker_step_type,
                        }
                        inventory.add_component("builders", component_id, component_info)
            
            # If no workspace constraints, return empty inventory (respects workspace isolation)
            return inventory.to_dict()
            
        except Exception as e:
            self.logger.error(f"Error discovering components: {e}")
            return {"error": str(e)}
    
    def resolve_cross_workspace_dependencies(self, pipeline_definition: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy method: resolve cross-workspace dependencies."""
        try:
            from ...workspace.core.dependency_graph import DependencyGraph
            
            resolution_result = {
                "pipeline_definition": pipeline_definition,
                "resolved_dependencies": {},
                "dependency_graph": None,
                "issues": [],
                "warnings": [],
            }
            
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
            
            resolution_result["dependency_graph"] = dep_graph.to_dict()
            
            return resolution_result
            
        except Exception as e:
            self.logger.error(f"Error resolving dependencies: {e}")
            return {"error": str(e)}
    
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

"""
Workspace Type Detector

Unified workspace detection that normalizes single/multi-workspace scenarios.
This component eliminates dual-path complexity by treating single workspace
as multi-workspace with count=1.

Architecture:
- Unified workspace detection logic
- Normalization of single/multi-workspace scenarios
- Consistent workspace information structure
- Simplified validation path management

Features:
- Single workspace treated as multi-workspace with count=1
- Unified workspace dictionary structure
- Consistent workspace type detection
- Simplified validation logic paths
"""

import os
from pathlib import Path
from typing import Dict, Any, Union, Optional
import logging

from .workspace_manager import WorkspaceManager, WorkspaceInfo, DeveloperInfo

logger = logging.getLogger(__name__)


class WorkspaceTypeDetector:
    """
    Unified workspace detection that normalizes single/multi-workspace scenarios.
    
    This class eliminates the complexity of handling single vs multi-workspace
    scenarios by always returning a unified workspace dictionary structure.
    Single workspaces are treated as multi-workspace with count=1.
    
    Features:
    - Unified workspace detection regardless of workspace type
    - Consistent dictionary structure for all scenarios
    - Simplified validation logic paths
    - Automatic workspace type classification
    """
    
    def __init__(self, workspace_root: Union[str, Path]):
        """
        Initialize workspace type detector.
        
        Args:
            workspace_root: Root directory to analyze for workspace structure
        """
        self.workspace_root = Path(workspace_root)
        self.workspace_manager = WorkspaceManager(
            workspace_root=workspace_root,
            auto_discover=False
        )
        self._workspace_info: Optional[WorkspaceInfo] = None
        self._workspace_type: Optional[str] = None
        self._detected_workspaces: Optional[Dict[str, Any]] = None
    
    def detect_workspaces(self) -> Dict[str, Any]:
        """
        Returns unified workspace dictionary regardless of workspace type.
        
        This method normalizes workspace detection to always return a dictionary
        where keys are workspace identifiers and values are workspace information.
        
        For single workspace scenarios:
        - Returns {"default": WorkspaceInfo(...)}
        
        For multi-workspace scenarios:
        - Returns {"dev1": WorkspaceInfo(...), "dev2": WorkspaceInfo(...)}
        
        Returns:
            Dictionary mapping workspace IDs to workspace information
        """
        if self._detected_workspaces is not None:
            return self._detected_workspaces
        
        logger.info(f"Detecting workspace structure at: {self.workspace_root}")
        
        # Discover workspace structure
        try:
            self._workspace_info = self.workspace_manager.discover_workspaces()
        except Exception as e:
            logger.error(f"Failed to discover workspaces: {e}")
            self._detected_workspaces = {}
            return self._detected_workspaces
        
        # Determine workspace type and create unified structure
        if self._is_multi_workspace_structure():
            self._workspace_type = "multi"
            self._detected_workspaces = self._create_multi_workspace_dict()
        else:
            self._workspace_type = "single"
            self._detected_workspaces = self._create_single_workspace_dict()
        
        logger.info(f"Detected {self._workspace_type} workspace with "
                   f"{len(self._detected_workspaces)} workspace(s)")
        
        return self._detected_workspaces
    
    def is_single_workspace(self) -> bool:
        """
        Check if this is a single workspace scenario.
        
        Returns:
            True if single workspace, False if multi-workspace
        """
        if self._workspace_type is None:
            self.detect_workspaces()
        return self._workspace_type == "single"
    
    def is_multi_workspace(self) -> bool:
        """
        Check if this is a multi-workspace scenario.
        
        Returns:
            True if multi-workspace, False if single workspace
        """
        if self._workspace_type is None:
            self.detect_workspaces()
        return self._workspace_type == "multi"
    
    def get_workspace_type(self) -> str:
        """
        Get the workspace type classification.
        
        Returns:
            'single' for single workspace, 'multi' for multi-workspace
        """
        if self._workspace_type is None:
            self.detect_workspaces()
        return self._workspace_type or "unknown"
    
    def get_workspace_count(self) -> int:
        """
        Get the number of detected workspaces.
        
        Returns:
            Number of workspaces (1 for single, N for multi)
        """
        workspaces = self.detect_workspaces()
        return len(workspaces)
    
    def get_workspace_ids(self) -> list[str]:
        """
        Get list of workspace identifiers.
        
        Returns:
            List of workspace IDs
        """
        workspaces = self.detect_workspaces()
        return list(workspaces.keys())
    
    def get_workspace_info(self, workspace_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information for a specific workspace.
        
        Args:
            workspace_id: Workspace identifier
            
        Returns:
            Workspace information dictionary or None if not found
        """
        workspaces = self.detect_workspaces()
        return workspaces.get(workspace_id)
    
    def _is_multi_workspace_structure(self) -> bool:
        """
        Determine if the workspace structure is multi-workspace.
        
        Multi-workspace structure has:
        - developers/ directory with multiple developer subdirectories
        - OR shared/ directory with developers/ directory
        
        Single workspace structure has:
        - Direct src/cursus_dev/steps structure
        - OR only shared/ directory without developers/
        
        Returns:
            True if multi-workspace structure detected
        """
        if not self._workspace_info:
            return False
        
        # Check for developers directory with multiple developers
        if self._workspace_info.total_developers > 1:
            return True
        
        # Check for developers directory structure (even with 1 developer)
        developers_dir = self.workspace_root / "developers"
        if developers_dir.exists() and any(developers_dir.iterdir()):
            return True
        
        # Check for direct cursus_dev structure (single workspace)
        direct_cursus_dev = self.workspace_root / "src" / "cursus_dev" / "steps"
        if direct_cursus_dev.exists():
            return False
        
        # If only shared directory exists, treat as single workspace
        shared_dir = self.workspace_root / "shared"
        if shared_dir.exists() and not developers_dir.exists():
            return False
        
        # Default to single workspace for ambiguous cases
        return False
    
    def _create_multi_workspace_dict(self) -> Dict[str, Any]:
        """
        Create workspace dictionary for multi-workspace scenario.
        
        Returns:
            Dictionary mapping developer IDs to workspace information
        """
        workspaces = {}
        
        if not self._workspace_info or not self._workspace_info.developers:
            logger.warning("No developers found in multi-workspace structure")
            return workspaces
        
        for developer in self._workspace_info.developers:
            workspace_info = {
                "workspace_id": developer.developer_id,
                "workspace_type": "developer",
                "workspace_path": developer.workspace_path,
                "developer_info": {
                    "developer_id": developer.developer_id,
                    "has_builders": developer.has_builders,
                    "has_contracts": developer.has_contracts,
                    "has_specs": developer.has_specs,
                    "has_scripts": developer.has_scripts,
                    "has_configs": developer.has_configs,
                    "module_count": developer.module_count,
                    "last_modified": developer.last_modified
                },
                "workspace_root": str(self.workspace_root),
                "has_shared_fallback": self._workspace_info.has_shared
            }
            workspaces[developer.developer_id] = workspace_info
        
        # Add shared workspace if it exists
        if self._workspace_info.has_shared:
            shared_info = {
                "workspace_id": "shared",
                "workspace_type": "shared",
                "workspace_path": str(self.workspace_root / "shared"),
                "developer_info": {
                    "developer_id": "shared",
                    "has_builders": True,  # Assume shared has all types
                    "has_contracts": True,
                    "has_specs": True,
                    "has_scripts": True,
                    "has_configs": True,
                    "module_count": 0,  # Could be calculated if needed
                    "last_modified": None
                },
                "workspace_root": str(self.workspace_root),
                "has_shared_fallback": False  # Shared doesn't fall back to itself
            }
            workspaces["shared"] = shared_info
        
        return workspaces
    
    def _create_single_workspace_dict(self) -> Dict[str, Any]:
        """
        Create workspace dictionary for single workspace scenario.
        
        Single workspace is normalized to look like multi-workspace with count=1.
        Uses "default" as the workspace identifier.
        
        Returns:
            Dictionary with single "default" workspace entry
        """
        # Check for direct cursus_dev structure
        direct_cursus_dev = self.workspace_root / "src" / "cursus_dev" / "steps"
        shared_cursus_dev = self.workspace_root / "shared" / "src" / "cursus_dev" / "steps"
        
        workspace_path = str(self.workspace_root)
        workspace_type = "single"
        
        # Determine which structure exists
        if direct_cursus_dev.exists():
            cursus_dev_path = direct_cursus_dev
        elif shared_cursus_dev.exists():
            cursus_dev_path = shared_cursus_dev
            workspace_path = str(self.workspace_root / "shared")
            workspace_type = "shared"
        else:
            # No valid structure found
            logger.warning("No valid cursus_dev structure found in single workspace")
            cursus_dev_path = None
        
        # Analyze module availability
        has_builders = False
        has_contracts = False
        has_specs = False
        has_scripts = False
        has_configs = False
        module_count = 0
        
        if cursus_dev_path and cursus_dev_path.exists():
            builders_dir = cursus_dev_path / "builders"
            contracts_dir = cursus_dev_path / "contracts"
            specs_dir = cursus_dev_path / "specs"
            scripts_dir = cursus_dev_path / "scripts"
            configs_dir = cursus_dev_path / "configs"
            
            has_builders = builders_dir.exists()
            has_contracts = contracts_dir.exists()
            has_specs = specs_dir.exists()
            has_scripts = scripts_dir.exists()
            has_configs = configs_dir.exists()
            
            # Count Python modules
            for module_dir in [builders_dir, contracts_dir, scripts_dir]:
                if module_dir.exists():
                    module_count += len([
                        f for f in module_dir.iterdir()
                        if f.is_file() and f.suffix == '.py' and f.name != '__init__.py'
                    ])
        
        # Create normalized workspace info
        workspace_info = {
            "workspace_id": "default",
            "workspace_type": workspace_type,
            "workspace_path": workspace_path,
            "developer_info": {
                "developer_id": "default",
                "has_builders": has_builders,
                "has_contracts": has_contracts,
                "has_specs": has_specs,
                "has_scripts": has_scripts,
                "has_configs": has_configs,
                "module_count": module_count,
                "last_modified": None
            },
            "workspace_root": str(self.workspace_root),
            "has_shared_fallback": False  # Single workspace doesn't have fallback
        }
        
        return {"default": workspace_info}
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of workspace detection results.
        
        Returns:
            Summary dictionary with detection results
        """
        workspaces = self.detect_workspaces()
        
        return {
            "workspace_root": str(self.workspace_root),
            "workspace_type": self.get_workspace_type(),
            "workspace_count": len(workspaces),
            "workspace_ids": list(workspaces.keys()),
            "has_shared": self._workspace_info.has_shared if self._workspace_info else False,
            "total_modules": sum(
                ws.get("developer_info", {}).get("module_count", 0)
                for ws in workspaces.values()
            ),
            "detection_successful": len(workspaces) > 0
        }
    
    def validate_workspace_structure(self) -> tuple[bool, list[str]]:
        """
        Validate the detected workspace structure.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        workspaces = self.detect_workspaces()
        
        if not workspaces:
            return False, ["No valid workspace structure detected"]
        
        issues = []
        
        # Validate each workspace
        for workspace_id, workspace_info in workspaces.items():
            workspace_path = Path(workspace_info["workspace_path"])
            
            if not workspace_path.exists():
                issues.append(f"Workspace path does not exist: {workspace_path}")
                continue
            
            # Check for cursus_dev structure
            if workspace_info["workspace_type"] == "shared":
                cursus_dev_path = workspace_path / "src" / "cursus_dev" / "steps"
            elif workspace_info["workspace_type"] == "developer":
                cursus_dev_path = workspace_path / "src" / "cursus_dev" / "steps"
            else:  # single workspace
                cursus_dev_path = workspace_path / "src" / "cursus_dev" / "steps"
                if not cursus_dev_path.exists():
                    # Try shared structure
                    cursus_dev_path = workspace_path / "shared" / "src" / "cursus_dev" / "steps"
            
            if not cursus_dev_path.exists():
                issues.append(f"Missing cursus_dev structure in workspace: {workspace_id}")
            
            # Check for at least one module type
            developer_info = workspace_info.get("developer_info", {})
            has_any_modules = any([
                developer_info.get("has_builders", False),
                developer_info.get("has_contracts", False),
                developer_info.get("has_specs", False),
                developer_info.get("has_scripts", False),
                developer_info.get("has_configs", False)
            ])
            
            if not has_any_modules:
                issues.append(f"No module directories found in workspace: {workspace_id}")
        
        is_valid = len(issues) == 0
        return is_valid, issues

"""
Workspace Manager

Provides workspace discovery, validation, and management utilities for
multi-developer workspace environments.

Architecture:
- Workspace discovery and validation
- Developer workspace management
- Workspace structure creation and validation
- Integration with file resolver and module loader
- Configuration management for workspace settings

Features:
- Automatic workspace discovery
- Workspace structure validation
- Developer workspace creation and setup
- Workspace configuration management
- Integration utilities for validation frameworks
"""

import os
import json
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
import logging

from pydantic import BaseModel, Field, ConfigDict

from .workspace_file_resolver import DeveloperWorkspaceFileResolver
from .workspace_module_loader import WorkspaceModuleLoader

logger = logging.getLogger(__name__)


class WorkspaceConfig(BaseModel):
    """Configuration for a workspace."""
    model_config = ConfigDict(
        extra='forbid',
        validate_assignment=True,
        str_strip_whitespace=True
    )
    
    workspace_root: str
    developer_id: Optional[str] = None
    enable_shared_fallback: bool = True
    cache_modules: bool = True
    auto_create_structure: bool = False
    validation_settings: Dict[str, Any] = Field(default_factory=dict)


class DeveloperInfo(BaseModel):
    """Information about a developer workspace."""
    model_config = ConfigDict(
        extra='forbid',
        validate_assignment=True,
        str_strip_whitespace=True
    )
    
    developer_id: str
    workspace_path: str
    has_builders: bool = False
    has_contracts: bool = False
    has_specs: bool = False
    has_scripts: bool = False
    has_configs: bool = False
    module_count: int = 0
    last_modified: Optional[str] = None


class WorkspaceInfo(BaseModel):
    """Information about a workspace environment."""
    model_config = ConfigDict(
        extra='forbid',
        validate_assignment=True,
        str_strip_whitespace=True
    )
    
    workspace_root: str
    has_shared: bool = False
    developers: List[DeveloperInfo] = Field(default_factory=list)
    total_developers: int = 0
    total_modules: int = 0
    config_file: Optional[str] = None


class WorkspaceManager:
    """
    Manager for multi-developer workspace environments.
    
    Provides utilities for workspace discovery, validation, creation,
    and management of developer workspaces.
    
    Features:
    - Workspace discovery and validation
    - Developer workspace management
    - Configuration management
    - Integration with file resolver and module loader
    """
    
    def __init__(
        self,
        workspace_root: Optional[Union[str, Path]] = None,
        config_file: Optional[Union[str, Path]] = None,
        auto_discover: bool = True
    ):
        """
        Initialize workspace manager.
        
        Args:
            workspace_root: Root directory for workspaces
            config_file: Path to workspace configuration file
            auto_discover: Whether to automatically discover workspaces
        """
        self.workspace_root = Path(workspace_root) if workspace_root else None
        self.config_file = Path(config_file) if config_file else None
        self.config: Optional[WorkspaceConfig] = None
        
        # Discovered workspace information
        self.workspace_info: Optional[WorkspaceInfo] = None
        
        if auto_discover and self.workspace_root:
            self.discover_workspaces()
        
        if self.config_file and self.config_file.exists():
            self.load_config()
    
    def discover_workspaces(
        self,
        workspace_root: Optional[Union[str, Path]] = None
    ) -> WorkspaceInfo:
        """
        Discover and analyze workspace structure.
        
        Args:
            workspace_root: Root directory to discover (uses instance default if None)
        
        Returns:
            WorkspaceInfo object with discovered information
        """
        if workspace_root:
            self.workspace_root = Path(workspace_root)
        
        if not self.workspace_root or not self.workspace_root.exists():
            raise ValueError(f"Workspace root does not exist: {self.workspace_root}")
        
        workspace_info = WorkspaceInfo(workspace_root=str(self.workspace_root))
        
        # Check for shared workspace
        shared_dir = self.workspace_root / "shared"
        workspace_info.has_shared = shared_dir.exists()
        
        # Discover developer workspaces
        developers_dir = self.workspace_root / "developers"
        if developers_dir.exists():
            workspace_info.developers = self._discover_developers(developers_dir)
            workspace_info.total_developers = len(workspace_info.developers)
            workspace_info.total_modules = sum(
                dev.module_count for dev in workspace_info.developers
            )
        
        # Check for workspace config file
        config_candidates = [
            self.workspace_root / "workspace.json",
            self.workspace_root / "workspace.yaml",
            self.workspace_root / "workspace.yml",
            self.workspace_root / ".workspace.json",
            self.workspace_root / ".workspace.yaml"
        ]
        
        for config_path in config_candidates:
            if config_path.exists():
                workspace_info.config_file = str(config_path)
                break
        
        self.workspace_info = workspace_info
        logger.info(f"Discovered workspace with {workspace_info.total_developers} "
                   f"developers and {workspace_info.total_modules} modules")
        
        return workspace_info
    
    def _discover_developers(self, developers_dir: Path) -> List[DeveloperInfo]:
        """Discover developer workspaces in developers directory."""
        developers = []
        
        for item in developers_dir.iterdir():
            if not item.is_dir():
                continue
            
            developer_id = item.name
            cursus_dev_dir = item / "src" / "cursus_dev" / "steps"
            
            if not cursus_dev_dir.exists():
                continue
            
            dev_info = DeveloperInfo(
                developer_id=developer_id,
                workspace_path=str(item)
            )
            
            # Check for different module types
            builders_dir = cursus_dev_dir / "builders"
            contracts_dir = cursus_dev_dir / "contracts"
            specs_dir = cursus_dev_dir / "specs"
            scripts_dir = cursus_dev_dir / "scripts"
            configs_dir = cursus_dev_dir / "configs"
            
            dev_info.has_builders = builders_dir.exists()
            dev_info.has_contracts = contracts_dir.exists()
            dev_info.has_specs = specs_dir.exists()
            dev_info.has_scripts = scripts_dir.exists()
            dev_info.has_configs = configs_dir.exists()
            
            # Count modules
            module_count = 0
            for module_dir in [builders_dir, contracts_dir, scripts_dir]:
                if module_dir.exists():
                    module_count += len([
                        f for f in module_dir.iterdir()
                        if f.is_file() and f.suffix == '.py' and f.name != '__init__.py'
                    ])
            
            dev_info.module_count = module_count
            
            # Get last modified time
            try:
                dev_info.last_modified = str(int(item.stat().st_mtime))
            except OSError:
                pass
            
            developers.append(dev_info)
        
        return sorted(developers, key=lambda d: d.developer_id)
    
    def validate_workspace_structure(
        self,
        workspace_root: Optional[Union[str, Path]] = None,
        strict: bool = False
    ) -> Tuple[bool, List[str]]:
        """
        Validate workspace structure.
        
        Args:
            workspace_root: Root directory to validate
            strict: Whether to apply strict validation rules
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        if workspace_root:
            self.workspace_root = Path(workspace_root)
        
        if not self.workspace_root:
            return False, ["No workspace root specified"]
        
        issues = []
        
        # Check workspace root exists
        if not self.workspace_root.exists():
            issues.append(f"Workspace root does not exist: {self.workspace_root}")
            return False, issues
        
        # Check for required directories
        developers_dir = self.workspace_root / "developers"
        shared_dir = self.workspace_root / "shared"
        
        if not developers_dir.exists() and not shared_dir.exists():
            issues.append("Workspace must contain 'developers' or 'shared' directory")
        
        # Validate developer workspaces
        if developers_dir.exists():
            dev_issues = self._validate_development(developers_dir, strict)
            issues.extend(dev_issues)
        
        # Validate shared workspace
        if shared_dir.exists():
            shared_issues = self._validate_shared_workspace(shared_dir, strict)
            issues.extend(shared_issues)
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def _validate_development(
        self,
        developers_dir: Path,
        strict: bool
    ) -> List[str]:
        """Validate developer workspace structures."""
        issues = []
        
        if not any(developers_dir.iterdir()):
            if strict:
                issues.append("No developer workspaces found")
            return issues
        
        for item in developers_dir.iterdir():
            if not item.is_dir():
                continue
            
            developer_id = item.name
            cursus_dev_dir = item / "src" / "cursus_dev" / "steps"
            
            if not cursus_dev_dir.exists():
                issues.append(f"Developer '{developer_id}' missing cursus_dev structure")
                continue
            
            # Check for at least one module type directory
            module_dirs = ["builders", "contracts", "specs", "scripts", "configs"]
            has_any_module_dir = any(
                (cursus_dev_dir / module_dir).exists()
                for module_dir in module_dirs
            )
            
            if strict and not has_any_module_dir:
                issues.append(f"Developer '{developer_id}' has no module directories")
        
        return issues
    
    def _validate_shared_workspace(
        self,
        shared_dir: Path,
        strict: bool
    ) -> List[str]:
        """Validate shared workspace structure."""
        issues = []
        
        cursus_dev_dir = shared_dir / "src" / "cursus_dev" / "steps"
        
        if not cursus_dev_dir.exists():
            issues.append("Shared workspace missing cursus_dev structure")
            return issues
        
        # Check for at least one module type directory
        if strict:
            module_dirs = ["builders", "contracts", "specs", "scripts", "configs"]
            has_any_module_dir = any(
                (cursus_dev_dir / module_dir).exists()
                for module_dir in module_dirs
            )
            
            if not has_any_module_dir:
                issues.append("Shared workspace has no module directories")
        
        return issues
    
    def create_developer_workspace(
        self,
        developer_id: str,
        workspace_root: Optional[Union[str, Path]] = None,
        create_structure: bool = True
    ) -> Path:
        """
        Create a new developer workspace.
        
        Args:
            developer_id: ID for the new developer
            workspace_root: Root directory for workspaces
            create_structure: Whether to create directory structure
        
        Returns:
            Path to created developer workspace
        """
        if workspace_root:
            self.workspace_root = Path(workspace_root)
        
        if not self.workspace_root:
            raise ValueError("No workspace root specified")
        
        # Create workspace root if it doesn't exist
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        
        # Create developer workspace path
        developers_dir = self.workspace_root / "developers"
        developers_dir.mkdir(exist_ok=True)
        
        dev_workspace = developers_dir / developer_id
        
        if dev_workspace.exists():
            raise ValueError(f"Developer workspace already exists: {developer_id}")
        
        if create_structure:
            # Create directory structure
            cursus_dev_dir = dev_workspace / "src" / "cursus_dev" / "steps"
            cursus_dev_dir.mkdir(parents=True)
            
            # Create module directories
            module_dirs = ["builders", "contracts", "specs", "scripts", "configs"]
            for module_dir in module_dirs:
                (cursus_dev_dir / module_dir).mkdir()
                
                # Create __init__.py files
                (cursus_dev_dir / module_dir / "__init__.py").touch()
            
            # Create main __init__.py files
            (dev_workspace / "src" / "__init__.py").touch()
            (dev_workspace / "src" / "cursus_dev" / "__init__.py").touch()
            (cursus_dev_dir / "__init__.py").touch()
            
            logger.info(f"Created developer workspace structure for: {developer_id}")
        
        return dev_workspace
    
    def create_shared_workspace(
        self,
        workspace_root: Optional[Union[str, Path]] = None,
        create_structure: bool = True
    ) -> Path:
        """
        Create shared workspace.
        
        Args:
            workspace_root: Root directory for workspaces
            create_structure: Whether to create directory structure
        
        Returns:
            Path to created shared workspace
        """
        if workspace_root:
            self.workspace_root = Path(workspace_root)
        
        if not self.workspace_root:
            raise ValueError("No workspace root specified")
        
        # Create workspace root if it doesn't exist
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        
        shared_workspace = self.workspace_root / "shared"
        
        if create_structure:
            # Create directory structure
            cursus_dev_dir = shared_workspace / "src" / "cursus_dev" / "steps"
            cursus_dev_dir.mkdir(parents=True, exist_ok=True)
            
            # Create module directories
            module_dirs = ["builders", "contracts", "specs", "scripts", "configs"]
            for module_dir in module_dirs:
                (cursus_dev_dir / module_dir).mkdir(exist_ok=True)
                
                # Create __init__.py files
                (cursus_dev_dir / module_dir / "__init__.py").touch()
            
            # Create main __init__.py files
            (shared_workspace / "src" / "__init__.py").touch()
            (shared_workspace / "src" / "cursus_dev" / "__init__.py").touch()
            (cursus_dev_dir / "__init__.py").touch()
            
            logger.info("Created shared workspace structure")
        
        return shared_workspace
    
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
        if not self.workspace_root:
            raise ValueError("No workspace root configured")
        
        # Use provided developer_id or fall back to config
        target_developer = developer_id
        if not target_developer and self.config:
            target_developer = self.config.developer_id
        
        # Get shared fallback setting
        enable_shared_fallback = kwargs.pop(
            'enable_shared_fallback',
            self.config.enable_shared_fallback if self.config else True
        )
        
        return DeveloperWorkspaceFileResolver(
            workspace_root=self.workspace_root,
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
        if not self.workspace_root:
            raise ValueError("No workspace root configured")
        
        # Use provided developer_id or fall back to config
        target_developer = developer_id
        if not target_developer and self.config:
            target_developer = self.config.developer_id
        
        # Get settings from config
        enable_shared_fallback = kwargs.pop(
            'enable_shared_fallback',
            self.config.enable_shared_fallback if self.config else True
        )
        
        cache_modules = kwargs.pop(
            'cache_modules',
            self.config.cache_modules if self.config else True
        )
        
        return WorkspaceModuleLoader(
            workspace_root=self.workspace_root,
            developer_id=target_developer,
            enable_shared_fallback=enable_shared_fallback,
            cache_modules=cache_modules,
            **kwargs
        )
    
    def load_config(
        self,
        config_file: Optional[Union[str, Path]] = None
    ) -> WorkspaceConfig:
        """
        Load workspace configuration from file.
        
        Args:
            config_file: Path to configuration file
        
        Returns:
            Loaded WorkspaceConfig
        """
        if config_file:
            self.config_file = Path(config_file)
        
        if not self.config_file or not self.config_file.exists():
            raise ValueError(f"Config file does not exist: {self.config_file}")
        
        with open(self.config_file, 'r') as f:
            if self.config_file.suffix.lower() in ['.yaml', '.yml']:
                config_data = yaml.safe_load(f)
            else:
                config_data = json.load(f)
        
        self.config = WorkspaceConfig(**config_data)
        
        # Update workspace root if specified in config
        if self.config.workspace_root and not self.workspace_root:
            self.workspace_root = Path(self.config.workspace_root)
        
        logger.info(f"Loaded workspace config from: {self.config_file}")
        return self.config
    
    def save_config(
        self,
        config_file: Optional[Union[str, Path]] = None,
        config: Optional[WorkspaceConfig] = None
    ) -> None:
        """
        Save workspace configuration to file.
        
        Args:
            config_file: Path to save configuration
            config: Configuration to save (uses instance config if None)
        """
        if config_file:
            self.config_file = Path(config_file)
        
        if not self.config_file:
            raise ValueError("No config file specified")
        
        if not config:
            config = self.config
        
        if not config:
            raise ValueError("No configuration to save")
        
        # Ensure parent directory exists
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        config_data = config.model_dump()
        
        with open(self.config_file, 'w') as f:
            if self.config_file.suffix.lower() in ['.yaml', '.yml']:
                yaml.safe_dump(config_data, f, default_flow_style=False, indent=2)
            else:
                json.dump(config_data, f, indent=2)
        
        logger.info(f"Saved workspace config to: {self.config_file}")
    
    def get_workspace_summary(self) -> Dict[str, Any]:
        """Get summary of workspace information."""
        if not self.workspace_info:
            if self.workspace_root:
                self.discover_workspaces()
            else:
                return {"error": "No workspace configured"}
        
        summary = {
            "workspace_root": self.workspace_info.workspace_root,
            "has_shared": self.workspace_info.has_shared,
            "total_developers": self.workspace_info.total_developers,
            "total_modules": self.workspace_info.total_modules,
            "config_file": self.workspace_info.config_file,
            "developers": []
        }
        
        for dev in self.workspace_info.developers:
            dev_summary = {
                "developer_id": dev.developer_id,
                "module_count": dev.module_count,
                "has_builders": dev.has_builders,
                "has_contracts": dev.has_contracts,
                "has_specs": dev.has_specs,
                "has_scripts": dev.has_scripts,
                "has_configs": dev.has_configs
            }
            summary["developers"].append(dev_summary)
        
        return summary
    
    def list_available_developers(self) -> List[str]:
        """
        Get list of available developer IDs.
        
        Returns:
            List of developer IDs found in the workspace
        """
        if not self.workspace_info:
            if self.workspace_root:
                self.discover_workspaces()
            else:
                return []
        
        return [dev.developer_id for dev in self.workspace_info.developers]
    
    def get_workspace_info(self, developer_id: Optional[str] = None) -> WorkspaceInfo:
        """
        Get workspace information, optionally for a specific developer.
        
        Args:
            developer_id: Optional developer ID to get info for
            
        Returns:
            WorkspaceInfo object
        """
        if not self.workspace_info:
            if self.workspace_root:
                self.discover_workspaces()
            else:
                return WorkspaceInfo(workspace_root=str(self.workspace_root or ""))
        
        return self.workspace_info

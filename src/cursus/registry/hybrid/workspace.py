"""
Hybrid Registry Workspace Management

This module provides workspace management capabilities for the hybrid registry system,
including workspace isolation, multi-developer support, and workspace-specific configurations.
"""

import logging
import os
import json
from typing import Dict, List, Optional, Set, Any, Union
from pathlib import Path
from datetime import datetime
import threading
from contextlib import contextmanager
from pydantic import BaseModel, Field

from .models import (
    StepDefinition,
    NamespacedStepDefinition,
    ResolutionContext,
    RegistryValidationResult
)
from .manager import LocalStepRegistry, RegistryConfig
from .utils import RegistryValidationUtils, RegistryErrorFormatter

logger = logging.getLogger(__name__)


class WorkspaceConfig(BaseModel):
    """Configuration for a workspace."""
    workspace_id: str
    name: str
    description: str = ""
    registry_paths: List[str] = Field(default_factory=list)
    priority: int = 1
    isolation_level: str = Field(default="standard", description="Isolation level: strict, standard, permissive")
    conflict_resolution_strategy: str = "workspace_priority"
    enable_inheritance: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class WorkspaceStatus(BaseModel):
    """Status information for a workspace."""
    workspace_id: str
    is_active: bool
    step_count: int
    registry_paths: List[str]
    last_loaded: Optional[datetime] = None
    validation_status: Optional[RegistryValidationResult] = None
    conflicts: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class WorkspaceManager:
    """
    Manages multiple workspaces and their registries.
    
    This manager provides workspace isolation, configuration management,
    and multi-developer support for the hybrid registry system.
    """
    
    def __init__(self, base_config: RegistryConfig):
        self.base_config = base_config
        self._workspaces: Dict[str, WorkspaceConfig] = {}
        self._workspace_registries: Dict[str, LocalStepRegistry] = {}
        self._workspace_status: Dict[str, WorkspaceStatus] = {}
        self._active_workspace: Optional[str] = None
        
        self._validator = RegistryValidationUtils()
        self._error_formatter = RegistryErrorFormatter()
        self._lock = threading.RLock()
        
        # Default workspace
        self._create_default_workspace()
    
    def _create_default_workspace(self) -> None:
        """Create the default workspace."""
        default_config = WorkspaceConfig(
            workspace_id="default",
            name="Default Workspace",
            description="Default workspace for general use",
            registry_paths=[],
            priority=0,
            created_at=datetime.now()
        )
        
        self._workspaces["default"] = default_config
        self._active_workspace = "default"
    
    def create_workspace(self, config: WorkspaceConfig) -> bool:
        """Create a new workspace."""
        with self._lock:
            try:
                if config.workspace_id in self._workspaces:
                    logger.warning(f"Workspace {config.workspace_id} already exists")
                    return False
                
                # Validate workspace configuration
                validation_result = self._validate_workspace_config(config)
                if not validation_result.is_valid:
                    logger.error(f"Invalid workspace config: {validation_result.errors}")
                    return False
                
                # Set timestamps
                config.created_at = datetime.now()
                config.updated_at = datetime.now()
                
                # Create workspace registry
                workspace_registry = LocalStepRegistry(
                    config.workspace_id,
                    config.registry_paths,
                    self.base_config
                )
                
                # Store workspace
                self._workspaces[config.workspace_id] = config
                self._workspace_registries[config.workspace_id] = workspace_registry
                
                # Initialize status
                self._workspace_status[config.workspace_id] = WorkspaceStatus(
                    workspace_id=config.workspace_id,
                    is_active=False,
                    step_count=0,
                    registry_paths=config.registry_paths,
                    last_loaded=None,
                    validation_status=None
                )
                
                logger.info(f"Created workspace: {config.workspace_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to create workspace {config.workspace_id}: {str(e)}")
                return False
    
    def delete_workspace(self, workspace_id: str, force: bool = False) -> bool:
        """Delete a workspace."""
        with self._lock:
            try:
                if workspace_id == "default" and not force:
                    logger.error("Cannot delete default workspace without force=True")
                    return False
                
                if workspace_id not in self._workspaces:
                    logger.warning(f"Workspace {workspace_id} does not exist")
                    return False
                
                # Switch active workspace if deleting current one
                if self._active_workspace == workspace_id:
                    self._active_workspace = "default" if workspace_id != "default" else None
                
                # Clean up
                del self._workspaces[workspace_id]
                if workspace_id in self._workspace_registries:
                    del self._workspace_registries[workspace_id]
                if workspace_id in self._workspace_status:
                    del self._workspace_status[workspace_id]
                
                logger.info(f"Deleted workspace: {workspace_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to delete workspace {workspace_id}: {str(e)}")
                return False
    
    def activate_workspace(self, workspace_id: str) -> bool:
        """Activate a workspace."""
        with self._lock:
            if workspace_id not in self._workspaces:
                logger.error(f"Workspace {workspace_id} does not exist")
                return False
            
            self._active_workspace = workspace_id
            
            # Update status
            if workspace_id in self._workspace_status:
                self._workspace_status[workspace_id].is_active = True
            
            # Deactivate other workspaces
            for ws_id, status in self._workspace_status.items():
                if ws_id != workspace_id:
                    status.is_active = False
            
            logger.info(f"Activated workspace: {workspace_id}")
            return True
    
    def get_active_workspace(self) -> Optional[str]:
        """Get the currently active workspace."""
        return self._active_workspace
    
    def list_workspaces(self) -> List[str]:
        """List all workspace IDs."""
        with self._lock:
            return list(self._workspaces.keys())
    
    def get_workspace_config(self, workspace_id: str) -> Optional[WorkspaceConfig]:
        """Get workspace configuration."""
        return self._workspaces.get(workspace_id)
    
    def update_workspace_config(self, workspace_id: str, config: WorkspaceConfig) -> bool:
        """Update workspace configuration."""
        with self._lock:
            try:
                if workspace_id not in self._workspaces:
                    logger.error(f"Workspace {workspace_id} does not exist")
                    return False
                
                # Validate new configuration
                validation_result = self._validate_workspace_config(config)
                if not validation_result.is_valid:
                    logger.error(f"Invalid workspace config: {validation_result.errors}")
                    return False
                
                # Update timestamps
                config.updated_at = datetime.now()
                if config.created_at is None:
                    config.created_at = self._workspaces[workspace_id].created_at
                
                # Update workspace
                self._workspaces[workspace_id] = config
                
                # Update registry if paths changed
                old_paths = self._workspace_registries[workspace_id].registry_paths
                new_paths = [Path(p) for p in config.registry_paths]
                
                if old_paths != new_paths:
                    self._workspace_registries[workspace_id] = LocalStepRegistry(
                        workspace_id,
                        config.registry_paths,
                        self.base_config
                    )
                    
                    # Reset status
                    if workspace_id in self._workspace_status:
                        self._workspace_status[workspace_id].registry_paths = config.registry_paths
                        self._workspace_status[workspace_id].last_loaded = None
                        self._workspace_status[workspace_id].validation_status = None
                
                logger.info(f"Updated workspace: {workspace_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to update workspace {workspace_id}: {str(e)}")
                return False
    
    def load_workspace_registry(self, workspace_id: str) -> RegistryValidationResult:
        """Load registry for a specific workspace."""
        with self._lock:
            try:
                if workspace_id not in self._workspace_registries:
                    return RegistryValidationResult(
                        is_valid=False,
                        errors=[f"Workspace {workspace_id} does not exist"],
                        warnings=[],
                        step_count=0
                    )
                
                registry = self._workspace_registries[workspace_id]
                result = registry.load_registry()
                
                # Update status
                if workspace_id in self._workspace_status:
                    status = self._workspace_status[workspace_id]
                    status.last_loaded = datetime.now()
                    status.validation_status = result
                    status.step_count = result.step_count
                    status.warnings = result.warnings
                    
                    if not result.is_valid:
                        status.conflicts = result.errors
                
                return result
                
            except Exception as e:
                error_msg = f"Failed to load workspace {workspace_id}: {str(e)}"
                logger.error(error_msg)
                return RegistryValidationResult(
                    is_valid=False,
                    errors=[error_msg],
                    warnings=[],
                    step_count=0
                )
    
    def get_workspace_status(self, workspace_id: str) -> Optional[WorkspaceStatus]:
        """Get status for a specific workspace."""
        return self._workspace_status.get(workspace_id)
    
    def get_all_workspace_status(self) -> Dict[str, WorkspaceStatus]:
        """Get status for all workspaces."""
        return dict(self._workspace_status)
    
    def _validate_workspace_config(self, config: WorkspaceConfig) -> RegistryValidationResult:
        """Validate workspace configuration."""
        errors = []
        warnings = []
        
        # Validate workspace ID
        if not self._validator.validate_workspace_id(config.workspace_id):
            errors.append(f"Invalid workspace ID: {config.workspace_id}")
        
        # Validate registry paths
        for path in config.registry_paths:
            path_obj = Path(path)
            if not path_obj.exists():
                warnings.append(f"Registry path does not exist: {path}")
            elif not path_obj.is_dir():
                errors.append(f"Registry path is not a directory: {path}")
        
        # Validate priority
        if config.priority < 0:
            errors.append("Priority must be non-negative")
        
        # Validate isolation level
        valid_isolation_levels = ["strict", "standard", "permissive"]
        if config.isolation_level not in valid_isolation_levels:
            errors.append(f"Invalid isolation level: {config.isolation_level}")
        
        return RegistryValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            step_count=0
        )
    
    @contextmanager
    def workspace_context(self, workspace_id: str):
        """Context manager for workspace operations."""
        previous_workspace = self._active_workspace
        try:
            self.activate_workspace(workspace_id)
            yield workspace_id
        finally:
            if previous_workspace:
                self.activate_workspace(previous_workspace)


class WorkspaceIsolationManager:
    """
    Manages workspace isolation and access control.
    
    This manager ensures that workspaces are properly isolated according
    to their configuration and provides controlled access to shared resources.
    """
    
    def __init__(self, workspace_manager: WorkspaceManager):
        self.workspace_manager = workspace_manager
        self._access_rules: Dict[str, Dict[str, Any]] = {}
        self._isolation_policies: Dict[str, Dict[str, Any]] = {}
        self._validator = RegistryValidationUtils()
        self._lock = threading.RLock()
    
    def set_isolation_policy(self, workspace_id: str, policy: Dict[str, Any]) -> bool:
        """Set isolation policy for a workspace."""
        with self._lock:
            try:
                # Validate policy
                if not self._validate_isolation_policy(policy):
                    return False
                
                self._isolation_policies[workspace_id] = policy
                logger.info(f"Set isolation policy for workspace: {workspace_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to set isolation policy for {workspace_id}: {str(e)}")
                return False
    
    def check_access_permission(self, workspace_id: str, resource_type: str, 
                              resource_name: str) -> bool:
        """Check if workspace has access to a resource."""
        
        workspace_config = self.workspace_manager.get_workspace_config(workspace_id)
        if not workspace_config:
            return False
        
        isolation_level = workspace_config.isolation_level
        
        # Apply isolation rules
        if isolation_level == "strict":
            # Only allow access to workspace-specific resources
            return resource_name.startswith(f"{workspace_id}_") or resource_type == "core"
            
        elif isolation_level == "standard":
            # Allow access to core and workspace resources
            return resource_type in ["core", "workspace"] or resource_name.startswith(f"{workspace_id}_")
            
        elif isolation_level == "permissive":
            # Allow access to all resources
            return True
        
        return False
    
    def get_accessible_steps(self, workspace_id: str, all_steps: Dict[str, StepDefinition]) -> Dict[str, StepDefinition]:
        """Get steps accessible to a workspace based on isolation rules."""
        
        accessible_steps = {}
        
        for step_name, step_def in all_steps.items():
            if self.check_access_permission(workspace_id, "step", step_name):
                accessible_steps[step_name] = step_def
        
        return accessible_steps
    
    def _validate_isolation_policy(self, policy: Dict[str, Any]) -> bool:
        """Validate isolation policy configuration."""
        required_fields = ["allow_core_access", "allow_cross_workspace_access"]
        
        for field in required_fields:
            if field not in policy:
                logger.error(f"Missing required policy field: {field}")
                return False
        
        return True


class MultiDeveloperManager:
    """
    Manages multi-developer environments with workspace coordination.
    
    This manager provides coordination between multiple developers working
    in different workspaces while maintaining isolation and conflict resolution.
    """
    
    def __init__(self, workspace_manager: WorkspaceManager):
        self.workspace_manager = workspace_manager
        self._developer_workspaces: Dict[str, List[str]] = {}
        self._workspace_locks: Dict[str, threading.Lock] = {}
        self._shared_resources: Dict[str, Any] = {}
        self._lock = threading.RLock()
    
    def assign_developer_workspace(self, developer_id: str, workspace_id: str) -> bool:
        """Assign a workspace to a developer."""
        with self._lock:
            try:
                if workspace_id not in self.workspace_manager._workspaces:
                    logger.error(f"Workspace {workspace_id} does not exist")
                    return False
                
                if developer_id not in self._developer_workspaces:
                    self._developer_workspaces[developer_id] = []
                
                if workspace_id not in self._developer_workspaces[developer_id]:
                    self._developer_workspaces[developer_id].append(workspace_id)
                    
                    # Create workspace lock if needed
                    if workspace_id not in self._workspace_locks:
                        self._workspace_locks[workspace_id] = threading.Lock()
                    
                    logger.info(f"Assigned workspace {workspace_id} to developer {developer_id}")
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to assign workspace: {str(e)}")
                return False
    
    def get_developer_workspaces(self, developer_id: str) -> List[str]:
        """Get workspaces assigned to a developer."""
        return self._developer_workspaces.get(developer_id, [])
    
    def create_developer_workspace(self, developer_id: str, workspace_name: str,
                                 registry_paths: List[str]) -> Optional[str]:
        """Create a new workspace for a developer."""
        
        workspace_id = f"{developer_id}_{workspace_name}"
        
        config = WorkspaceConfig(
            workspace_id=workspace_id,
            name=f"{developer_id}'s {workspace_name} Workspace",
            description=f"Personal workspace for {developer_id}",
            registry_paths=registry_paths,
            priority=1,
            isolation_level="standard"
        )
        
        if self.workspace_manager.create_workspace(config):
            self.assign_developer_workspace(developer_id, workspace_id)
            return workspace_id
        
        return None
    
    @contextmanager
    def exclusive_workspace_access(self, workspace_id: str):
        """Context manager for exclusive workspace access."""
        if workspace_id not in self._workspace_locks:
            self._workspace_locks[workspace_id] = threading.Lock()
        
        with self._workspace_locks[workspace_id]:
            yield workspace_id
    
    def coordinate_workspace_changes(self, workspace_id: str, changes: Dict[str, Any]) -> bool:
        """Coordinate changes across workspaces to prevent conflicts."""
        
        with self.exclusive_workspace_access(workspace_id):
            try:
                # Validate changes don't conflict with other workspaces
                conflicts = self._detect_cross_workspace_conflicts(workspace_id, changes)
                
                if conflicts:
                    logger.warning(f"Cross-workspace conflicts detected: {conflicts}")
                    return False
                
                # Apply changes
                return self._apply_workspace_changes(workspace_id, changes)
                
            except Exception as e:
                logger.error(f"Failed to coordinate workspace changes: {str(e)}")
                return False
    
    def _detect_cross_workspace_conflicts(self, workspace_id: str, changes: Dict[str, Any]) -> List[str]:
        """Detect conflicts with other workspaces."""
        conflicts = []
        
        # Check for step name conflicts
        if "new_steps" in changes:
            for step_name in changes["new_steps"]:
                for other_ws_id, registry in self.workspace_manager._workspace_registries.items():
                    if other_ws_id != workspace_id and registry.has_step(step_name):
                        conflicts.append(f"Step {step_name} already exists in workspace {other_ws_id}")
        
        return conflicts
    
    def _apply_workspace_changes(self, workspace_id: str, changes: Dict[str, Any]) -> bool:
        """Apply changes to a workspace."""
        try:
            # This would implement the actual change application logic
            # For now, just log the changes
            logger.info(f"Applied changes to workspace {workspace_id}: {changes}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply changes to workspace {workspace_id}: {str(e)}")
            return False


class WorkspaceConfigManager:
    """
    Manages workspace configurations and persistence.
    
    This manager handles saving/loading workspace configurations and
    provides configuration validation and migration capabilities.
    """
    
    def __init__(self, config_dir: str = ".cursus/workspaces"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._validator = RegistryValidationUtils()
    
    def save_workspace_config(self, config: WorkspaceConfig) -> bool:
        """Save workspace configuration to disk."""
        try:
            config_file = self.config_dir / f"{config.workspace_id}.json"
            
            # Convert to serializable format
            config_data = {
                "workspace_id": config.workspace_id,
                "name": config.name,
                "description": config.description,
                "registry_paths": config.registry_paths,
                "priority": config.priority,
                "isolation_level": config.isolation_level,
                "conflict_resolution_strategy": config.conflict_resolution_strategy,
                "enable_inheritance": config.enable_inheritance,
                "metadata": config.metadata,
                "created_at": config.created_at.isoformat() if config.created_at else None,
                "updated_at": config.updated_at.isoformat() if config.updated_at else None
            }
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Saved workspace config: {config.workspace_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save workspace config {config.workspace_id}: {str(e)}")
            return False
    
    def load_workspace_config(self, workspace_id: str) -> Optional[WorkspaceConfig]:
        """Load workspace configuration from disk."""
        try:
            config_file = self.config_dir / f"{workspace_id}.json"
            
            if not config_file.exists():
                logger.warning(f"Workspace config file not found: {config_file}")
                return None
            
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Convert timestamps
            created_at = None
            if config_data.get("created_at"):
                created_at = datetime.fromisoformat(config_data["created_at"])
            
            updated_at = None
            if config_data.get("updated_at"):
                updated_at = datetime.fromisoformat(config_data["updated_at"])
            
            config = WorkspaceConfig(
                workspace_id=config_data["workspace_id"],
                name=config_data["name"],
                description=config_data.get("description", ""),
                registry_paths=config_data.get("registry_paths", []),
                priority=config_data.get("priority", 1),
                isolation_level=config_data.get("isolation_level", "standard"),
                conflict_resolution_strategy=config_data.get("conflict_resolution_strategy", "workspace_priority"),
                enable_inheritance=config_data.get("enable_inheritance", True),
                metadata=config_data.get("metadata", {}),
                created_at=created_at,
                updated_at=updated_at
            )
            
            logger.info(f"Loaded workspace config: {workspace_id}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load workspace config {workspace_id}: {str(e)}")
            return None
    
    def list_saved_configs(self) -> List[str]:
        """List all saved workspace configurations."""
        try:
            config_files = list(self.config_dir.glob("*.json"))
            return [f.stem for f in config_files]
            
        except Exception as e:
            logger.error(f"Failed to list workspace configs: {str(e)}")
            return []
    
    def delete_workspace_config(self, workspace_id: str) -> bool:
        """Delete workspace configuration from disk."""
        try:
            config_file = self.config_dir / f"{workspace_id}.json"
            
            if config_file.exists():
                config_file.unlink()
                logger.info(f"Deleted workspace config: {workspace_id}")
                return True
            else:
                logger.warning(f"Workspace config file not found: {config_file}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete workspace config {workspace_id}: {str(e)}")
            return False


class WorkspaceAwareRegistryManager:
    """
    Workspace-aware registry manager that integrates workspace management
    with the hybrid registry system.
    
    This manager provides a complete workspace-aware registry solution.
    """
    
    def __init__(self, base_config: RegistryConfig, config_dir: str = ".cursus/workspaces"):
        self.base_config = base_config
        self.workspace_manager = WorkspaceManager(base_config)
        self.isolation_manager = WorkspaceIsolationManager(self.workspace_manager)
        self.multi_developer_manager = MultiDeveloperManager(self.workspace_manager)
        self.config_manager = WorkspaceConfigManager(config_dir)
        
        # Load saved workspace configurations
        self._load_saved_workspaces()
    
    def _load_saved_workspaces(self) -> None:
        """Load all saved workspace configurations."""
        saved_configs = self.config_manager.list_saved_configs()
        
        for workspace_id in saved_configs:
            config = self.config_manager.load_workspace_config(workspace_id)
            if config:
                self.workspace_manager._workspaces[workspace_id] = config
                
                # Create registry
                registry = LocalStepRegistry(
                    workspace_id,
                    config.registry_paths,
                    self.base_config
                )
                self.workspace_manager._workspace_registries[workspace_id] = registry
                
                # Initialize status
                self.workspace_manager._workspace_status[workspace_id] = WorkspaceStatus(
                    workspace_id=workspace_id,
                    is_active=False,
                    step_count=0,
                    registry_paths=config.registry_paths,
                    last_loaded=None,
                    validation_status=None
                )
    
    def create_workspace(self, config: WorkspaceConfig, save_config: bool = True) -> bool:
        """Create a workspace and optionally save its configuration."""
        
        if self.workspace_manager.create_workspace(config):
            if save_config:
                return self.config_manager.save_workspace_config(config)
            return True
        
        return False
    
    def delete_workspace(self, workspace_id: str, delete_config: bool = True, force: bool = False) -> bool:
        """Delete a workspace and optionally its configuration."""
        
        if self.workspace_manager.delete_workspace(workspace_id, force):
            if delete_config:
                return self.config_manager.delete_workspace_config(workspace_id)
            return True
        
        return False
    
    def get_workspace_steps(self, workspace_id: str, include_inherited: bool = True) -> Dict[str, StepDefinition]:
        """Get all steps available to a workspace."""
        
        workspace_config = self.workspace_manager.get_workspace_config(workspace_id)
        if not workspace_config:
            return {}
        
        # Get workspace-specific steps
        workspace_registry = self.workspace_manager._workspace_registries.get(workspace_id)
        if not workspace_registry:
            return {}
        
        workspace_steps = {}
        for step_name in workspace_registry.list_steps():
            step_def = workspace_registry.get_step(step_name)
            if step_def:
                workspace_steps[step_name] = step_def
        
        # Add inherited steps if enabled
        if include_inherited and workspace_config.enable_inheritance:
            # This would include core steps and potentially other workspace steps
            # based on inheritance rules
            pass
        
        # Apply isolation filtering
        accessible_steps = self.isolation_manager.get_accessible_steps(workspace_id, workspace_steps)
        
        return accessible_steps
    
    def resolve_step_in_workspace(self, workspace_id: str, step_name: str) -> Optional[StepDefinition]:
        """Resolve a step within a specific workspace context."""
        
        context = ResolutionContext(
            workspace_id=workspace_id,
            resolution_strategy=self.base_config.conflict_resolution_strategy
        )
        
        # Get workspace steps
        workspace_steps = self.get_workspace_steps(workspace_id)
        
        # Check access permission
        if not self.isolation_manager.check_access_permission(workspace_id, "step", step_name):
            logger.warning(f"Access denied to step {step_name} for workspace {workspace_id}")
            return None
        
        return workspace_steps.get(step_name)
    
    def get_workspace_summary(self) -> Dict[str, Any]:
        """Get summary information about all workspaces."""
        
        summary = {
            "total_workspaces": len(self.workspace_manager._workspaces),
            "active_workspace": self.workspace_manager.get_active_workspace(),
            "workspaces": {},
            "total_steps": 0,
            "total_conflicts": 0
        }
        
        for workspace_id, config in self.workspace_manager._workspaces.items():
            status = self.workspace_manager.get_workspace_status(workspace_id)
            workspace_steps = self.get_workspace_steps(workspace_id)
            
            workspace_info = {
                "name": config.name,
                "description": config.description,
                "step_count": len(workspace_steps),
                "registry_paths": config.registry_paths,
                "isolation_level": config.isolation_level,
                "is_active": status.is_active if status else False,
                "conflicts": len(status.conflicts) if status else 0
            }
            
            summary["workspaces"][workspace_id] = workspace_info
            summary["total_steps"] += workspace_info["step_count"]
            summary["total_conflicts"] += workspace_info["conflicts"]
        
        return summary
    
    @contextmanager
    def developer_workspace_context(self, developer_id: str, workspace_name: str = None):
        """Context manager for developer workspace operations."""
        
        if workspace_name:
            workspace_id = f"{developer_id}_{workspace_name}"
        else:
            # Use first assigned workspace
            developer_workspaces = self.get_developer_workspaces(developer_id)
            if not developer_workspaces:
                raise ValueError(f"No workspaces assigned to developer {developer_id}")
            workspace_id = developer_workspaces[0]
        
        with self.workspace_manager.workspace_context(workspace_id):
            yield workspace_id


# Utility functions for workspace management
def create_workspace_aware_registry(base_registry_path: str, 
                                   workspace_configs: List[WorkspaceConfig]) -> WorkspaceAwareRegistryManager:
    """Factory function to create a workspace-aware registry manager."""
    
    base_config = RegistryConfig(
        core_registry_path=base_registry_path,
        conflict_resolution_strategy="workspace_priority"
    )
    
    manager = WorkspaceAwareRegistryManager(base_config)
    
    # Create workspaces
    for config in workspace_configs:
        manager.create_workspace(config)
    
    return manager


def get_default_workspace_config(workspace_id: str, registry_paths: List[str]) -> WorkspaceConfig:
    """Get a default workspace configuration."""
    return WorkspaceConfig(
        workspace_id=workspace_id,
        name=f"Workspace {workspace_id}",
        description=f"Default configuration for workspace {workspace_id}",
        registry_paths=registry_paths,
        priority=1,
        isolation_level="standard",
        conflict_resolution_strategy="workspace_priority",
        enable_inheritance=True,
        created_at=datetime.now()
    )


def create_developer_workspace_config(developer_id: str, workspace_name: str,
                                     registry_paths: List[str]) -> WorkspaceConfig:
    """Create a workspace configuration for a specific developer."""
    workspace_id = f"{developer_id}_{workspace_name}"
    
    return WorkspaceConfig(
        workspace_id=workspace_id,
        name=f"{developer_id}'s {workspace_name} Workspace",
        description=f"Personal workspace for {developer_id}",
        registry_paths=registry_paths,
        priority=1,
        isolation_level="standard",
        conflict_resolution_strategy="workspace_priority",
        enable_inheritance=True,
        metadata={"developer_id": developer_id, "workspace_type": "personal"},
        created_at=datetime.now()
    )

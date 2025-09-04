"""
Consolidated Workspace Manager

Centralized workspace management with functional separation through specialized managers.
This module provides the main WorkspaceManager class that coordinates all workspace
operations through specialized functional managers.

Architecture:
- WorkspaceManager: Main consolidated manager coordinating all workspace operations
- WorkspaceLifecycleManager: Workspace creation, setup, teardown operations
- WorkspaceIsolationManager: Workspace isolation and sandboxing utilities
- WorkspaceDiscoveryManager: Cross-workspace component discovery and resolution
- WorkspaceIntegrationManager: Integration staging coordination and management

This consolidates functionality previously distributed across:
- src/cursus/validation/workspace/workspace_manager.py
- src/cursus/validation/runtime/integration/workspace_manager.py
- development/workspace_manager/ (external - packaging violation)
"""

import os
import json
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
import logging
from datetime import datetime, timedelta

from pydantic import BaseModel, Field, ConfigDict

# Import specialized managers - avoid circular imports by importing at class level

logger = logging.getLogger(__name__)


class WorkspaceContext(BaseModel):
    """Context information for a workspace."""
    model_config = ConfigDict(
        extra='forbid',
        validate_assignment=True,
        str_strip_whitespace=True
    )
    
    workspace_id: str
    workspace_path: str
    developer_id: Optional[str] = None
    workspace_type: str = "developer"  # "developer", "shared", "test"
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed: datetime = Field(default_factory=datetime.now)
    status: str = "active"  # "active", "archived", "maintenance"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkspaceConfig(BaseModel):
    """Configuration for workspace management."""
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
    isolation_settings: Dict[str, Any] = Field(default_factory=dict)
    integration_settings: Dict[str, Any] = Field(default_factory=dict)


class WorkspaceManager:
    """
    Centralized workspace management with functional separation.
    
    This consolidated manager coordinates all workspace operations through
    specialized functional managers, providing a unified interface for:
    - Workspace lifecycle management (creation, setup, teardown)
    - Workspace isolation and sandboxing
    - Cross-workspace component discovery
    - Integration staging coordination
    
    Features:
    - Unified workspace management interface
    - Functional separation through specialized managers
    - Comprehensive error handling and diagnostics
    - Full backward compatibility with existing workspace operations
    - Integration with existing validation and core systems
    """
    
    def __init__(
        self,
        workspace_root: Optional[Union[str, Path]] = None,
        config_file: Optional[Union[str, Path]] = None,
        auto_discover: bool = True
    ):
        """
        Initialize consolidated workspace manager.
        
        Args:
            workspace_root: Root directory for workspaces
            config_file: Path to workspace configuration file
            auto_discover: Whether to automatically discover workspaces
        """
        self.workspace_root = Path(workspace_root) if workspace_root else None
        self.config_file = Path(config_file) if config_file else None
        self.config: Optional[WorkspaceConfig] = None
        
        # Initialize specialized managers - import at runtime to avoid circular imports
        from .lifecycle import WorkspaceLifecycleManager
        from .isolation import WorkspaceIsolationManager
        from .discovery import WorkspaceDiscoveryManager
        from .integration import WorkspaceIntegrationManager
        
        self.lifecycle_manager = WorkspaceLifecycleManager(self)
        self.isolation_manager = WorkspaceIsolationManager(self)
        self.discovery_manager = WorkspaceDiscoveryManager(self)
        self.integration_manager = WorkspaceIntegrationManager(self)
        
        # Workspace tracking
        self.active_workspaces: Dict[str, WorkspaceContext] = {}
        
        # Load configuration if provided
        if self.config_file and self.config_file.exists():
            self.load_config()
        
        # Auto-discover workspaces if requested
        if auto_discover and self.workspace_root:
            self.discover_workspaces()
        
        logger.info(f"Initialized consolidated workspace manager for: {self.workspace_root}")
    
    # Core Workspace Operations
    
    def create_workspace(
        self,
        developer_id: str,
        workspace_type: str = "developer",
        template: str = None,
        **kwargs
    ) -> WorkspaceContext:
        """
        Create a new workspace.
        
        Args:
            developer_id: Developer identifier for the workspace
            workspace_type: Type of workspace ("developer", "shared", "test")
            template: Optional template to use for workspace creation
            **kwargs: Additional arguments passed to lifecycle manager
        
        Returns:
            WorkspaceContext for the created workspace
        """
        logger.info(f"Creating workspace for developer: {developer_id}")
        
        try:
            # Use lifecycle manager to create workspace
            workspace_context = self.lifecycle_manager.create_workspace(
                developer_id=developer_id,
                workspace_type=workspace_type,
                template=template,
                **kwargs
            )
            
            # Register workspace
            self.active_workspaces[workspace_context.workspace_id] = workspace_context
            
            logger.info(f"Successfully created workspace: {workspace_context.workspace_id}")
            return workspace_context
            
        except Exception as e:
            logger.error(f"Failed to create workspace for {developer_id}: {e}")
            raise
    
    def configure_workspace(
        self,
        workspace_id: str,
        config: Dict[str, Any]
    ) -> WorkspaceContext:
        """
        Configure an existing workspace.
        
        Args:
            workspace_id: Workspace identifier
            config: Configuration dictionary
        
        Returns:
            Updated WorkspaceContext
        """
        logger.info(f"Configuring workspace: {workspace_id}")
        
        if workspace_id not in self.active_workspaces:
            raise ValueError(f"Workspace not found: {workspace_id}")
        
        try:
            # Use lifecycle manager to configure workspace
            workspace_context = self.lifecycle_manager.configure_workspace(
                workspace_id=workspace_id,
                config=config
            )
            
            # Update workspace tracking
            self.active_workspaces[workspace_id] = workspace_context
            
            logger.info(f"Successfully configured workspace: {workspace_id}")
            return workspace_context
            
        except Exception as e:
            logger.error(f"Failed to configure workspace {workspace_id}: {e}")
            raise
    
    def delete_workspace(self, workspace_id: str) -> bool:
        """
        Delete a workspace.
        
        Args:
            workspace_id: Workspace identifier
        
        Returns:
            True if deletion was successful
        """
        logger.info(f"Deleting workspace: {workspace_id}")
        
        try:
            # Use lifecycle manager to delete workspace
            success = self.lifecycle_manager.delete_workspace(workspace_id)
            
            # Remove from tracking
            if workspace_id in self.active_workspaces:
                del self.active_workspaces[workspace_id]
            
            logger.info(f"Successfully deleted workspace: {workspace_id}")
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete workspace {workspace_id}: {e}")
            raise
    
    # Component Discovery Operations
    
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
        logger.info(f"Discovering components for workspaces: {workspace_ids or 'all'}")
        
        try:
            return self.discovery_manager.discover_components(
                workspace_ids=workspace_ids,
                developer_id=developer_id
            )
        except Exception as e:
            logger.error(f"Failed to discover components: {e}")
            raise
    
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
        
        try:
            return self.discovery_manager.resolve_cross_workspace_dependencies(
                pipeline_definition
            )
        except Exception as e:
            logger.error(f"Failed to resolve cross-workspace dependencies: {e}")
            raise
    
    # Integration Operations
    
    def stage_for_integration(
        self,
        component_id: str,
        source_workspace: str,
        target_stage: str = "integration"
    ) -> Dict[str, Any]:
        """
        Stage component for integration.
        
        Args:
            component_id: Component identifier
            source_workspace: Source workspace identifier
            target_stage: Target staging area
        
        Returns:
            Staging result information
        """
        logger.info(f"Staging component {component_id} from {source_workspace}")
        
        try:
            return self.integration_manager.stage_for_integration(
                component_id=component_id,
                source_workspace=source_workspace,
                target_stage=target_stage
            )
        except Exception as e:
            logger.error(f"Failed to stage component {component_id}: {e}")
            raise
    
    def validate_integration_readiness(
        self,
        staged_components: List[str]
    ) -> Dict[str, Any]:
        """
        Validate integration readiness for staged components.
        
        Args:
            staged_components: List of staged component identifiers
        
        Returns:
            Integration readiness validation results
        """
        logger.info(f"Validating integration readiness for {len(staged_components)} components")
        
        try:
            return self.integration_manager.validate_integration_readiness(
                staged_components
            )
        except Exception as e:
            logger.error(f"Failed to validate integration readiness: {e}")
            raise
    
    # Workspace Discovery and Validation
    
    def discover_workspaces(
        self,
        workspace_root: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Discover and analyze workspace structure.
        
        Args:
            workspace_root: Root directory to discover (uses instance default if None)
        
        Returns:
            Workspace discovery information
        """
        if workspace_root:
            self.workspace_root = Path(workspace_root)
        
        if not self.workspace_root or not self.workspace_root.exists():
            raise ValueError(f"Workspace root does not exist: {self.workspace_root}")
        
        logger.info(f"Discovering workspaces in: {self.workspace_root}")
        
        try:
            # Use discovery manager for workspace discovery
            discovery_result = self.discovery_manager.discover_workspaces(self.workspace_root)
            
            # Update active workspaces tracking
            for workspace_info in discovery_result.get('workspaces', []):
                workspace_id = workspace_info.get('workspace_id')
                if workspace_id:
                    self.active_workspaces[workspace_id] = WorkspaceContext(
                        workspace_id=workspace_id,
                        workspace_path=workspace_info.get('workspace_path', ''),
                        developer_id=workspace_info.get('developer_id'),
                        workspace_type=workspace_info.get('workspace_type', 'developer'),
                        metadata=workspace_info.get('metadata', {})
                    )
            
            return discovery_result
            
        except Exception as e:
            logger.error(f"Failed to discover workspaces: {e}")
            raise
    
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
        
        logger.info(f"Validating workspace structure: {self.workspace_root}")
        
        try:
            # Use isolation manager for structure validation
            return self.isolation_manager.validate_workspace_structure(
                workspace_root=self.workspace_root,
                strict=strict
            )
        except Exception as e:
            logger.error(f"Failed to validate workspace structure: {e}")
            return False, [f"Validation error: {e}"]
    
    # Configuration Management
    
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
        
        logger.info(f"Loading workspace config from: {self.config_file}")
        
        try:
            with open(self.config_file, 'r') as f:
                if self.config_file.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
            
            self.config = WorkspaceConfig(**config_data)
            
            # Update workspace root if specified in config
            if self.config.workspace_root and not self.workspace_root:
                self.workspace_root = Path(self.config.workspace_root)
            
            logger.info(f"Successfully loaded workspace config")
            return self.config
            
        except Exception as e:
            logger.error(f"Failed to load workspace config: {e}")
            raise
    
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
        
        logger.info(f"Saving workspace config to: {self.config_file}")
        
        try:
            # Ensure parent directory exists
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            config_data = config.model_dump()
            
            with open(self.config_file, 'w') as f:
                if self.config_file.suffix.lower() in ['.yaml', '.yml']:
                    yaml.safe_dump(config_data, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_data, f, indent=2)
            
            logger.info(f"Successfully saved workspace config")
            
        except Exception as e:
            logger.error(f"Failed to save workspace config: {e}")
            raise
    
    # Workspace Information and Management
    
    def get_workspace_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of workspace information."""
        logger.info("Generating workspace summary")
        
        try:
            summary = {
                "workspace_root": str(self.workspace_root) if self.workspace_root else None,
                "config_file": str(self.config_file) if self.config_file else None,
                "active_workspaces": len(self.active_workspaces),
                "workspace_details": {},
                "discovery_summary": {},
                "validation_summary": {},
                "integration_summary": {}
            }
            
            # Add workspace details
            for workspace_id, context in self.active_workspaces.items():
                summary["workspace_details"][workspace_id] = {
                    "developer_id": context.developer_id,
                    "workspace_type": context.workspace_type,
                    "status": context.status,
                    "created_at": context.created_at.isoformat(),
                    "last_accessed": context.last_accessed.isoformat()
                }
            
            # Add discovery summary
            if self.workspace_root:
                summary["discovery_summary"] = self.discovery_manager.get_discovery_summary()
            
            # Add validation summary
            summary["validation_summary"] = self.isolation_manager.get_validation_summary()
            
            # Add integration summary
            summary["integration_summary"] = self.integration_manager.get_integration_summary()
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate workspace summary: {e}")
            return {"error": str(e)}
    
    def list_available_developers(self) -> List[str]:
        """
        Get list of available developer IDs.
        
        Returns:
            List of developer IDs found in the workspace
        """
        logger.info("Listing available developers")
        
        try:
            return self.discovery_manager.list_available_developers()
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
            if workspace_id and workspace_id in self.active_workspaces:
                context = self.active_workspaces[workspace_id]
                return {
                    "workspace_id": context.workspace_id,
                    "workspace_path": context.workspace_path,
                    "developer_id": context.developer_id,
                    "workspace_type": context.workspace_type,
                    "status": context.status,
                    "created_at": context.created_at.isoformat(),
                    "last_accessed": context.last_accessed.isoformat(),
                    "metadata": context.metadata
                }
            
            # Use discovery manager for broader workspace info
            return self.discovery_manager.get_workspace_info(
                workspace_id=workspace_id,
                developer_id=developer_id
            )
            
        except Exception as e:
            logger.error(f"Failed to get workspace info: {e}")
            return {"error": str(e)}
    
    # Backward Compatibility Methods
    # These methods maintain compatibility with existing workspace manager APIs
    
    def get_file_resolver(
        self,
        developer_id: Optional[str] = None,
        **kwargs
    ):
        """
        Get workspace-aware file resolver.
        
        Args:
            developer_id: Developer to target (uses config default if None)
            **kwargs: Additional arguments for file resolver
        
        Returns:
            Configured file resolver
        """
        logger.debug(f"Getting file resolver for developer: {developer_id}")
        
        try:
            return self.discovery_manager.get_file_resolver(
                developer_id=developer_id,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Failed to get file resolver: {e}")
            raise
    
    def get_module_loader(
        self,
        developer_id: Optional[str] = None,
        **kwargs
    ):
        """
        Get workspace-aware module loader.
        
        Args:
            developer_id: Developer to target (uses config default if None)
            **kwargs: Additional arguments for module loader
        
        Returns:
            Configured module loader
        """
        logger.debug(f"Getting module loader for developer: {developer_id}")
        
        try:
            return self.discovery_manager.get_module_loader(
                developer_id=developer_id,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Failed to get module loader: {e}")
            raise
    
    def create_developer_workspace(
        self,
        developer_id: str,
        workspace_root: Optional[Union[str, Path]] = None,
        create_structure: bool = True
    ) -> Path:
        """
        Create a new developer workspace (backward compatibility method).
        
        Args:
            developer_id: ID for the new developer
            workspace_root: Root directory for workspaces
            create_structure: Whether to create directory structure
        
        Returns:
            Path to created developer workspace
        """
        logger.info(f"Creating developer workspace (legacy API): {developer_id}")
        
        if workspace_root:
            self.workspace_root = Path(workspace_root)
        
        try:
            # Use new API but return Path for backward compatibility
            workspace_context = self.create_workspace(
                developer_id=developer_id,
                workspace_type="developer",
                create_structure=create_structure
            )
            
            return Path(workspace_context.workspace_path)
            
        except Exception as e:
            logger.error(f"Failed to create developer workspace: {e}")
            raise
    
    def create_shared_workspace(
        self,
        workspace_root: Optional[Union[str, Path]] = None,
        create_structure: bool = True
    ) -> Path:
        """
        Create shared workspace (backward compatibility method).
        
        Args:
            workspace_root: Root directory for workspaces
            create_structure: Whether to create directory structure
        
        Returns:
            Path to created shared workspace
        """
        logger.info("Creating shared workspace (legacy API)")
        
        if workspace_root:
            self.workspace_root = Path(workspace_root)
        
        try:
            # Use new API but return Path for backward compatibility
            workspace_context = self.create_workspace(
                developer_id="shared",
                workspace_type="shared",
                create_structure=create_structure
            )
            
            return Path(workspace_context.workspace_path)
            
        except Exception as e:
            logger.error(f"Failed to create shared workspace: {e}")
            raise
    
    # Health and Monitoring
    
    def get_workspace_health(self, workspace_id: str) -> Dict[str, Any]:
        """
        Get health information for a workspace.
        
        Args:
            workspace_id: Workspace identifier
        
        Returns:
            Health information dictionary
        """
        logger.info(f"Getting workspace health for: {workspace_id}")
        
        try:
            # Use isolation manager for health checks
            return self.isolation_manager.get_workspace_health(workspace_id)
        except Exception as e:
            logger.error(f"Failed to get workspace health: {e}")
            return {"error": str(e), "healthy": False}
    
    def cleanup_inactive_workspaces(
        self,
        inactive_threshold: timedelta = timedelta(days=30)
    ) -> Dict[str, Any]:
        """
        Clean up inactive workspaces.
        
        Args:
            inactive_threshold: Threshold for considering workspace inactive
        
        Returns:
            Cleanup result information
        """
        logger.info(f"Cleaning up workspaces inactive for more than {inactive_threshold}")
        
        try:
            return self.lifecycle_manager.cleanup_inactive_workspaces(inactive_threshold)
        except Exception as e:
            logger.error(f"Failed to cleanup inactive workspaces: {e}")
            return {"error": str(e), "cleaned_up": []}
    
    # Utility Methods
    
    def refresh_workspace_cache(self) -> None:
        """Refresh workspace component cache."""
        logger.info("Refreshing workspace cache")
        
        try:
            self.discovery_manager.refresh_cache()
            logger.info("Successfully refreshed workspace cache")
        except Exception as e:
            logger.error(f"Failed to refresh workspace cache: {e}")
            raise
    
    def get_workspace_statistics(self) -> Dict[str, Any]:
        """Get comprehensive workspace statistics."""
        logger.info("Generating workspace statistics")
        
        try:
            return {
                "total_workspaces": len(self.active_workspaces),
                "workspace_types": {
                    workspace_type: len([
                        ws for ws in self.active_workspaces.values()
                        if ws.workspace_type == workspace_type
                    ])
                    for workspace_type in ["developer", "shared", "test"]
                },
                "discovery_stats": self.discovery_manager.get_statistics(),
                "isolation_stats": self.isolation_manager.get_statistics(),
                "integration_stats": self.integration_manager.get_statistics(),
                "lifecycle_stats": self.lifecycle_manager.get_statistics()
            }
        except Exception as e:
            logger.error(f"Failed to generate workspace statistics: {e}")
            return {"error": str(e)}

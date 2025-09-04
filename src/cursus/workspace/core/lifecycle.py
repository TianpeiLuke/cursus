"""
Workspace Lifecycle Manager

Manages workspace creation, setup, teardown, and lifecycle operations.
This module provides comprehensive workspace lifecycle management including
workspace creation from templates, configuration, archiving, and cleanup.

Features:
- Workspace creation with template support
- Workspace structure initialization and validation
- Workspace configuration and environment setup
- Workspace archiving and restoration
- Inactive workspace cleanup and maintenance
"""

import os
import shutil
import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import logging
from datetime import datetime, timedelta

from .manager import WorkspaceContext

logger = logging.getLogger(__name__)


class WorkspaceTemplate:
    """Workspace template for creating new workspaces."""
    
    def __init__(self, template_name: str, template_path: Path):
        """
        Initialize workspace template.
        
        Args:
            template_name: Name of the template
            template_path: Path to template directory
        """
        self.template_name = template_name
        self.template_path = template_path
        self.metadata = self._load_template_metadata()
    
    def _load_template_metadata(self) -> Dict[str, Any]:
        """Load template metadata from template.json or template.yaml."""
        metadata_files = [
            self.template_path / "template.json",
            self.template_path / "template.yaml",
            self.template_path / "template.yml"
        ]
        
        for metadata_file in metadata_files:
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        if metadata_file.suffix.lower() in ['.yaml', '.yml']:
                            return yaml.safe_load(f)
                        else:
                            return json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load template metadata from {metadata_file}: {e}")
        
        # Default metadata if no file found
        return {
            "name": self.template_name,
            "description": f"Workspace template: {self.template_name}",
            "version": "1.0.0",
            "structure": {}
        }


class WorkspaceLifecycleManager:
    """
    Workspace lifecycle management.
    
    Handles workspace creation, configuration, archiving, restoration,
    and cleanup operations with comprehensive template support.
    """
    
    def __init__(self, workspace_manager):
        """
        Initialize workspace lifecycle manager.
        
        Args:
            workspace_manager: Parent WorkspaceManager instance
        """
        self.workspace_manager = workspace_manager
        self.templates: Dict[str, WorkspaceTemplate] = {}
        self._load_available_templates()
        
        logger.info("Initialized workspace lifecycle manager")
    
    def _load_available_templates(self) -> None:
        """Load available workspace templates."""
        try:
            # Look for templates in development/templates/
            if self.workspace_manager.workspace_root:
                templates_dir = self.workspace_manager.workspace_root.parent / "development" / "templates"
                if templates_dir.exists():
                    for template_dir in templates_dir.iterdir():
                        if template_dir.is_dir():
                            template = WorkspaceTemplate(template_dir.name, template_dir)
                            self.templates[template_dir.name] = template
                            logger.debug(f"Loaded template: {template_dir.name}")
            
            # Add default templates if none found
            if not self.templates:
                self._create_default_templates()
                
        except Exception as e:
            logger.warning(f"Failed to load workspace templates: {e}")
            self._create_default_templates()
    
    def _create_default_templates(self) -> None:
        """Create default workspace templates."""
        self.templates = {
            "basic": WorkspaceTemplate("basic", Path("basic")),
            "ml_workspace": WorkspaceTemplate("ml_workspace", Path("ml_workspace")),
            "advanced": WorkspaceTemplate("advanced", Path("advanced"))
        }
        logger.info("Created default workspace templates")
    
    def create_workspace(
        self,
        developer_id: str,
        workspace_type: str = "developer",
        template: str = None,
        create_structure: bool = True,
        **kwargs
    ) -> WorkspaceContext:
        """
        Create a new workspace.
        
        Args:
            developer_id: Developer identifier
            workspace_type: Type of workspace ("developer", "shared", "test")
            template: Optional template name to use
            create_structure: Whether to create directory structure
            **kwargs: Additional workspace configuration
        
        Returns:
            WorkspaceContext for the created workspace
        """
        logger.info(f"Creating {workspace_type} workspace for: {developer_id}")
        
        if not self.workspace_manager.workspace_root:
            raise ValueError("No workspace root configured")
        
        try:
            # Determine workspace path
            if workspace_type == "shared":
                workspace_path = self.workspace_manager.workspace_root / "shared"
                workspace_id = "shared"
            elif workspace_type == "test":
                workspace_path = self.workspace_manager.workspace_root / "test" / developer_id
                workspace_id = f"test_{developer_id}"
            else:  # developer workspace
                workspace_path = self.workspace_manager.workspace_root / "developers" / developer_id
                workspace_id = developer_id
            
            # Check if workspace already exists
            if workspace_path.exists():
                raise ValueError(f"Workspace already exists: {workspace_path}")
            
            # Create workspace structure
            if create_structure:
                self._create_workspace_structure(
                    workspace_path=workspace_path,
                    workspace_type=workspace_type,
                    template=template
                )
            
            # Create workspace context
            workspace_context = WorkspaceContext(
                workspace_id=workspace_id,
                workspace_path=str(workspace_path),
                developer_id=developer_id,
                workspace_type=workspace_type,
                metadata=kwargs
            )
            
            logger.info(f"Successfully created workspace: {workspace_id}")
            return workspace_context
            
        except Exception as e:
            logger.error(f"Failed to create workspace for {developer_id}: {e}")
            raise
    
    def _create_workspace_structure(
        self,
        workspace_path: Path,
        workspace_type: str,
        template: str = None
    ) -> None:
        """Create workspace directory structure."""
        logger.info(f"Creating workspace structure at: {workspace_path}")
        
        try:
            # Create base directory
            workspace_path.mkdir(parents=True, exist_ok=True)
            
            if template and template in self.templates:
                # Use template to create structure
                self._apply_template(workspace_path, self.templates[template])
            else:
                # Create default structure
                self._create_default_structure(workspace_path, workspace_type)
            
            logger.info(f"Successfully created workspace structure")
            
        except Exception as e:
            logger.error(f"Failed to create workspace structure: {e}")
            raise
    
    def _apply_template(self, workspace_path: Path, template: WorkspaceTemplate) -> None:
        """Apply template to workspace."""
        logger.info(f"Applying template: {template.template_name}")
        
        try:
            if template.template_path.exists():
                # Copy template structure
                shutil.copytree(template.template_path, workspace_path, dirs_exist_ok=True)
            else:
                # Create structure from metadata
                self._create_structure_from_metadata(workspace_path, template.metadata)
                
        except Exception as e:
            logger.error(f"Failed to apply template {template.template_name}: {e}")
            # Fallback to default structure
            self._create_default_structure(workspace_path, "developer")
    
    def _create_structure_from_metadata(self, workspace_path: Path, metadata: Dict[str, Any]) -> None:
        """Create workspace structure from template metadata."""
        structure = metadata.get("structure", {})
        
        def create_structure_recursive(base_path: Path, structure_dict: Dict[str, Any]):
            for name, content in structure_dict.items():
                item_path = base_path / name
                
                if isinstance(content, dict):
                    # Directory
                    item_path.mkdir(exist_ok=True)
                    create_structure_recursive(item_path, content)
                elif isinstance(content, str):
                    # File with content
                    item_path.parent.mkdir(parents=True, exist_ok=True)
                    item_path.write_text(content)
                else:
                    # Empty directory
                    item_path.mkdir(exist_ok=True)
        
        create_structure_recursive(workspace_path, structure)
    
    def _create_default_structure(self, workspace_path: Path, workspace_type: str) -> None:
        """Create default workspace structure."""
        logger.info(f"Creating default {workspace_type} workspace structure")
        
        try:
            if workspace_type == "shared":
                # Shared workspace structure
                cursus_dev_dir = workspace_path / "src" / "cursus_dev" / "steps"
            else:
                # Developer/test workspace structure
                cursus_dev_dir = workspace_path / "src" / "cursus_dev" / "steps"
            
            # Create directory structure
            cursus_dev_dir.mkdir(parents=True, exist_ok=True)
            
            # Create module directories
            module_dirs = ["builders", "contracts", "specs", "scripts", "configs"]
            for module_dir in module_dirs:
                (cursus_dev_dir / module_dir).mkdir(exist_ok=True)
                
                # Create __init__.py files
                (cursus_dev_dir / module_dir / "__init__.py").touch()
            
            # Create main __init__.py files
            (workspace_path / "src" / "__init__.py").touch()
            (workspace_path / "src" / "cursus_dev" / "__init__.py").touch()
            (cursus_dev_dir / "__init__.py").touch()
            
            # Create additional directories for developer workspaces
            if workspace_type == "developer":
                (workspace_path / "test").mkdir(exist_ok=True)
                (workspace_path / "validation_reports").mkdir(exist_ok=True)
                (workspace_path / "test" / "__init__.py").touch()
            
            logger.info(f"Successfully created default {workspace_type} workspace structure")
            
        except Exception as e:
            logger.error(f"Failed to create default workspace structure: {e}")
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
        
        if workspace_id not in self.workspace_manager.active_workspaces:
            raise ValueError(f"Workspace not found: {workspace_id}")
        
        try:
            workspace_context = self.workspace_manager.active_workspaces[workspace_id]
            
            # Update workspace metadata with configuration
            workspace_context.metadata.update(config)
            workspace_context.last_accessed = datetime.now()
            
            # Apply configuration to workspace
            self._apply_workspace_configuration(workspace_context, config)
            
            logger.info(f"Successfully configured workspace: {workspace_id}")
            return workspace_context
            
        except Exception as e:
            logger.error(f"Failed to configure workspace {workspace_id}: {e}")
            raise
    
    def _apply_workspace_configuration(
        self,
        workspace_context: WorkspaceContext,
        config: Dict[str, Any]
    ) -> None:
        """Apply configuration to workspace."""
        workspace_path = Path(workspace_context.workspace_path)
        
        try:
            # Create workspace config file
            config_file = workspace_path / "workspace_config.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Apply any structural changes if needed
            if config.get("enable_additional_modules"):
                additional_modules = config.get("additional_modules", [])
                cursus_dev_dir = workspace_path / "src" / "cursus_dev" / "steps"
                
                for module_name in additional_modules:
                    module_dir = cursus_dev_dir / module_name
                    module_dir.mkdir(exist_ok=True)
                    (module_dir / "__init__.py").touch()
            
        except Exception as e:
            logger.error(f"Failed to apply workspace configuration: {e}")
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
        
        if workspace_id not in self.workspace_manager.active_workspaces:
            raise ValueError(f"Workspace not found: {workspace_id}")
        
        try:
            workspace_context = self.workspace_manager.active_workspaces[workspace_id]
            workspace_path = Path(workspace_context.workspace_path)
            
            # Archive workspace before deletion if it contains data
            if self._workspace_has_data(workspace_path):
                archive_path = self._archive_workspace_data(workspace_context)
                logger.info(f"Archived workspace data to: {archive_path}")
            
            # Remove workspace directory
            if workspace_path.exists():
                shutil.rmtree(workspace_path)
            
            logger.info(f"Successfully deleted workspace: {workspace_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete workspace {workspace_id}: {e}")
            raise
    
    def _workspace_has_data(self, workspace_path: Path) -> bool:
        """Check if workspace contains user data."""
        try:
            # Check for Python files in step directories
            cursus_dev_dir = workspace_path / "src" / "cursus_dev" / "steps"
            if cursus_dev_dir.exists():
                for module_dir in cursus_dev_dir.iterdir():
                    if module_dir.is_dir():
                        python_files = list(module_dir.glob("*.py"))
                        # Filter out __init__.py files
                        user_files = [f for f in python_files if f.name != "__init__.py"]
                        if user_files:
                            return True
            
            # Check for test files
            test_dir = workspace_path / "test"
            if test_dir.exists():
                python_files = list(test_dir.glob("**/*.py"))
                if python_files:
                    return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error checking workspace data: {e}")
            return True  # Assume has data to be safe
    
    def _archive_workspace_data(self, workspace_context: WorkspaceContext) -> Path:
        """Archive workspace data before deletion."""
        try:
            # Create archive directory
            archive_root = self.workspace_manager.workspace_root / "archived_workspaces"
            archive_root.mkdir(exist_ok=True)
            
            # Create archive with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_name = f"{workspace_context.workspace_id}_{timestamp}"
            archive_path = archive_root / archive_name
            
            # Copy workspace to archive
            workspace_path = Path(workspace_context.workspace_path)
            shutil.copytree(workspace_path, archive_path)
            
            # Create archive metadata
            metadata = {
                "workspace_id": workspace_context.workspace_id,
                "developer_id": workspace_context.developer_id,
                "workspace_type": workspace_context.workspace_type,
                "archived_at": datetime.now().isoformat(),
                "original_path": workspace_context.workspace_path,
                "archive_reason": "workspace_deletion"
            }
            
            metadata_file = archive_path / "archive_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return archive_path
            
        except Exception as e:
            logger.error(f"Failed to archive workspace data: {e}")
            raise
    
    def archive_workspace(self, workspace_id: str) -> Dict[str, Any]:
        """
        Archive a workspace without deleting it.
        
        Args:
            workspace_id: Workspace identifier
        
        Returns:
            Archive result information
        """
        logger.info(f"Archiving workspace: {workspace_id}")
        
        if workspace_id not in self.workspace_manager.active_workspaces:
            raise ValueError(f"Workspace not found: {workspace_id}")
        
        try:
            workspace_context = self.workspace_manager.active_workspaces[workspace_id]
            
            # Archive workspace data
            archive_path = self._archive_workspace_data(workspace_context)
            
            # Update workspace status
            workspace_context.status = "archived"
            workspace_context.metadata["archived_at"] = datetime.now().isoformat()
            workspace_context.metadata["archive_path"] = str(archive_path)
            
            result = {
                "workspace_id": workspace_id,
                "archive_path": str(archive_path),
                "archived_at": workspace_context.metadata["archived_at"],
                "success": True
            }
            
            logger.info(f"Successfully archived workspace: {workspace_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to archive workspace {workspace_id}: {e}")
            return {
                "workspace_id": workspace_id,
                "success": False,
                "error": str(e)
            }
    
    def restore_workspace(
        self,
        workspace_id: str,
        archive_path: str
    ) -> Dict[str, Any]:
        """
        Restore a workspace from archive.
        
        Args:
            workspace_id: Workspace identifier
            archive_path: Path to archived workspace
        
        Returns:
            Restore result information
        """
        logger.info(f"Restoring workspace {workspace_id} from: {archive_path}")
        
        try:
            archive_path = Path(archive_path)
            if not archive_path.exists():
                raise ValueError(f"Archive path does not exist: {archive_path}")
            
            # Load archive metadata
            metadata_file = archive_path / "archive_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    archive_metadata = json.load(f)
            else:
                archive_metadata = {}
            
            # Determine restore path
            if workspace_id in self.workspace_manager.active_workspaces:
                raise ValueError(f"Workspace already exists: {workspace_id}")
            
            # Restore workspace
            if workspace_id == "shared":
                restore_path = self.workspace_manager.workspace_root / "shared"
            elif workspace_id.startswith("test_"):
                developer_id = workspace_id[5:]  # Remove "test_" prefix
                restore_path = self.workspace_manager.workspace_root / "test" / developer_id
            else:
                restore_path = self.workspace_manager.workspace_root / "developers" / workspace_id
            
            # Copy archive to restore location
            shutil.copytree(archive_path, restore_path, dirs_exist_ok=True)
            
            # Create workspace context
            workspace_context = WorkspaceContext(
                workspace_id=workspace_id,
                workspace_path=str(restore_path),
                developer_id=archive_metadata.get("developer_id", workspace_id),
                workspace_type=archive_metadata.get("workspace_type", "developer"),
                status="active",
                metadata=archive_metadata
            )
            
            # Register restored workspace
            self.workspace_manager.active_workspaces[workspace_id] = workspace_context
            
            result = {
                "workspace_id": workspace_id,
                "restore_path": str(restore_path),
                "restored_at": datetime.now().isoformat(),
                "success": True
            }
            
            logger.info(f"Successfully restored workspace: {workspace_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to restore workspace {workspace_id}: {e}")
            return {
                "workspace_id": workspace_id,
                "success": False,
                "error": str(e)
            }
    
    def _create_default_structure(self, workspace_path: Path, workspace_type: str) -> None:
        """Create default workspace structure."""
        try:
            # Create cursus_dev structure
            cursus_dev_dir = workspace_path / "src" / "cursus_dev" / "steps"
            cursus_dev_dir.mkdir(parents=True, exist_ok=True)
            
            # Create module directories
            module_dirs = ["builders", "contracts", "specs", "scripts", "configs"]
            for module_dir in module_dirs:
                (cursus_dev_dir / module_dir).mkdir(exist_ok=True)
                (cursus_dev_dir / module_dir / "__init__.py").touch()
            
            # Create __init__.py files
            (workspace_path / "src" / "__init__.py").touch()
            (workspace_path / "src" / "cursus_dev" / "__init__.py").touch()
            (cursus_dev_dir / "__init__.py").touch()
            
            # Create additional directories based on workspace type
            if workspace_type == "developer":
                (workspace_path / "test").mkdir(exist_ok=True)
                (workspace_path / "validation_reports").mkdir(exist_ok=True)
                (workspace_path / "test" / "__init__.py").touch()
            elif workspace_type == "test":
                (workspace_path / "test_data").mkdir(exist_ok=True)
                (workspace_path / "test_results").mkdir(exist_ok=True)
            
        except Exception as e:
            logger.error(f"Failed to create default structure: {e}")
            raise
    
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
        
        cleanup_result = {
            "cleaned_up": [],
            "archived": [],
            "errors": [],
            "total_processed": 0
        }
        
        try:
            current_time = datetime.now()
            
            for workspace_id, workspace_context in list(self.workspace_manager.active_workspaces.items()):
                cleanup_result["total_processed"] += 1
                
                try:
                    # Check if workspace is inactive
                    time_since_access = current_time - workspace_context.last_accessed
                    
                    if time_since_access > inactive_threshold:
                        # Archive workspace data if it has data
                        workspace_path = Path(workspace_context.workspace_path)
                        if self._workspace_has_data(workspace_path):
                            archive_result = self.archive_workspace(workspace_id)
                            if archive_result["success"]:
                                cleanup_result["archived"].append({
                                    "workspace_id": workspace_id,
                                    "archive_path": archive_result["archive_path"]
                                })
                        
                        # Delete workspace
                        if self.delete_workspace(workspace_id):
                            cleanup_result["cleaned_up"].append(workspace_id)
                        
                except Exception as e:
                    error_msg = f"Failed to cleanup workspace {workspace_id}: {e}"
                    logger.error(error_msg)
                    cleanup_result["errors"].append(error_msg)
            
            logger.info(f"Cleanup completed: {len(cleanup_result['cleaned_up'])} workspaces cleaned up")
            return cleanup_result
            
        except Exception as e:
            logger.error(f"Failed to cleanup inactive workspaces: {e}")
            cleanup_result["errors"].append(str(e))
            return cleanup_result
    
    def get_available_templates(self) -> List[Dict[str, Any]]:
        """
        Get list of available workspace templates.
        
        Returns:
            List of template information dictionaries
        """
        return [
            {
                "name": template.template_name,
                "path": str(template.template_path),
                "metadata": template.metadata
            }
            for template in self.templates.values()
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get lifecycle management statistics."""
        try:
            return {
                "available_templates": len(self.templates),
                "template_names": list(self.templates.keys()),
                "workspace_operations": {
                    "total_workspaces": len(self.workspace_manager.active_workspaces),
                    "active_workspaces": len([
                        ws for ws in self.workspace_manager.active_workspaces.values()
                        if ws.status == "active"
                    ]),
                    "archived_workspaces": len([
                        ws for ws in self.workspace_manager.active_workspaces.values()
                        if ws.status == "archived"
                    ])
                }
            }
        except Exception as e:
            logger.error(f"Failed to get lifecycle statistics: {e}")
            return {"error": str(e)}

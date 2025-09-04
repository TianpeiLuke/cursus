"""
Unified Workspace API - Phase 4 High-Level API Creation

This module provides a simplified, developer-friendly interface to the workspace-aware
system, abstracting the complexity of the underlying Phase 1-3 consolidated architecture.
"""

from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from enum import Enum
import logging

from pydantic import BaseModel, Field, ConfigDict
from pydantic.types import DirectoryPath

from .core import WorkspaceManager, WorkspaceDiscoveryManager, WorkspaceIsolationManager, WorkspaceLifecycleManager, WorkspaceIntegrationManager
from .validation import CrossWorkspaceValidator, WorkspaceTestManager


class WorkspaceStatus(Enum):
    """Workspace status enumeration."""
    HEALTHY = "healthy"
    WARNING = "warning"
    ERROR = "error"
    UNKNOWN = "unknown"


class WorkspaceSetupResult(BaseModel):
    """Result of workspace setup operation."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True
    )
    
    success: bool
    workspace_path: Path
    developer_id: str = Field(..., min_length=1, description="Unique identifier for the developer")
    message: str
    warnings: List[str] = Field(default_factory=list)


class ValidationReport(BaseModel):
    """Workspace validation report."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True
    )
    
    workspace_path: Path
    status: WorkspaceStatus
    issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    isolation_violations: List[Dict[str, Any]] = Field(default_factory=list)


class PromotionResult(BaseModel):
    """Result of workspace promotion operation."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True
    )
    
    success: bool
    source_workspace: Path
    target_environment: str = Field(..., min_length=1, description="Target environment name")
    message: str
    artifacts_promoted: List[str] = Field(default_factory=list)


class HealthReport(BaseModel):
    """Overall workspace system health report."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True
    )
    
    overall_status: WorkspaceStatus
    workspace_reports: List[ValidationReport] = Field(default_factory=list)
    system_issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class CleanupReport(BaseModel):
    """Result of workspace cleanup operation."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True
    )
    
    success: bool
    cleaned_workspaces: List[Path] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    space_freed: Optional[int] = Field(None, ge=0, description="Space freed in bytes")


class WorkspaceInfo(BaseModel):
    """Information about a workspace."""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True
    )
    
    path: Path
    developer_id: str = Field(..., min_length=1, description="Unique identifier for the developer")
    status: WorkspaceStatus
    created_at: Optional[str] = Field(None, description="ISO format timestamp")
    last_modified: Optional[str] = Field(None, description="ISO format timestamp")
    size_bytes: Optional[int] = Field(None, ge=0, description="Workspace size in bytes")
    active_pipelines: List[str] = Field(default_factory=list)


class WorkspaceAPI:
    """
    Unified high-level API for workspace-aware system operations.
    
    This class provides a simplified interface to the workspace system,
    abstracting the complexity of the underlying managers and providing
    developer-friendly methods for common operations.
    """
    
    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        """
        Initialize the WorkspaceAPI.
        
        Args:
            base_path: Base path for workspace operations. If None, uses default.
        """
        self.base_path = Path(base_path) if base_path else Path("development")
        self.logger = logging.getLogger(__name__)
        
        # Initialize underlying managers (lazy loading to avoid circular imports)
        self._workspace_manager = None
        self._discovery = None
        self._isolation_manager = None
        self._lifecycle_manager = None
        self._integration_manager = None
        self._validator = None
    
    @property
    def workspace_manager(self) -> WorkspaceManager:
        """Get workspace manager instance (lazy loaded)."""
        if self._workspace_manager is None:
            self._workspace_manager = WorkspaceManager(str(self.base_path))
        return self._workspace_manager
    
    @property
    def discovery(self) -> WorkspaceDiscoveryManager:
        """Get workspace discovery instance (lazy loaded)."""
        if self._discovery is None:
            self._discovery = WorkspaceDiscoveryManager(self.workspace_manager)
        return self._discovery
    
    @property
    def isolation_manager(self) -> WorkspaceIsolationManager:
        """Get isolation manager instance (lazy loaded)."""
        if self._isolation_manager is None:
            self._isolation_manager = WorkspaceIsolationManager(self.workspace_manager)
        return self._isolation_manager
    
    @property
    def lifecycle_manager(self) -> WorkspaceLifecycleManager:
        """Get lifecycle manager instance (lazy loaded)."""
        if self._lifecycle_manager is None:
            self._lifecycle_manager = WorkspaceLifecycleManager(self.workspace_manager)
        return self._lifecycle_manager
    
    @property
    def integration_manager(self) -> WorkspaceIntegrationManager:
        """Get integration manager instance (lazy loaded)."""
        if self._integration_manager is None:
            self._integration_manager = WorkspaceIntegrationManager(self.workspace_manager)
        return self._integration_manager
    
    @property
    def validator(self) -> CrossWorkspaceValidator:
        """Get cross-workspace validator instance (lazy loaded)."""
        if self._validator is None:
            self._validator = CrossWorkspaceValidator()
        return self._validator
    
    def setup_developer_workspace(
        self, 
        developer_id: str, 
        template: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> WorkspaceSetupResult:
        """
        Set up a new developer workspace.
        
        Args:
            developer_id: Unique identifier for the developer
            template: Optional template to use for workspace setup
            config_overrides: Optional configuration overrides
            
        Returns:
            WorkspaceSetupResult with setup details
        """
        try:
            self.logger.info(f"Setting up workspace for developer: {developer_id}")
            
            # Create workspace using lifecycle manager
            workspace_path = self.lifecycle_manager.create_workspace(
                developer_id, 
                template=template,
                config=config_overrides or {}
            )
            
            # Validate the new workspace
            validation_result = self.validate_workspace(workspace_path)
            
            warnings = []
            if validation_result.status == WorkspaceStatus.WARNING:
                warnings = validation_result.issues
            
            return WorkspaceSetupResult(
                success=True,
                workspace_path=workspace_path,
                developer_id=developer_id,
                message=f"Successfully created workspace at {workspace_path}",
                warnings=warnings
            )
            
        except Exception as e:
            self.logger.error(f"Failed to setup workspace for {developer_id}: {e}")
            return WorkspaceSetupResult(
                success=False,
                workspace_path=Path(""),
                developer_id=developer_id,
                message=f"Failed to create workspace: {str(e)}"
            )
    
    def validate_workspace(self, workspace_path: Union[str, Path]) -> ValidationReport:
        """
        Validate a workspace for compliance and isolation.
        
        Args:
            workspace_path: Path to the workspace to validate
            
        Returns:
            ValidationReport with validation results
        """
        workspace_path = Path(workspace_path)
        
        try:
            # Run cross-workspace validation
            violations = self.validator.validate_workspace_isolation(str(workspace_path))
            
            # Determine status based on violations
            if not violations:
                status = WorkspaceStatus.HEALTHY
                issues = []
                recommendations = []
            else:
                # Check severity of violations
                critical_violations = [v for v in violations if v.get('severity') == 'critical']
                if critical_violations:
                    status = WorkspaceStatus.ERROR
                else:
                    status = WorkspaceStatus.WARNING
                
                issues = [v.get('message', 'Unknown violation') for v in violations]
                recommendations = [v.get('recommendation', 'Review violation') for v in violations]
            
            return ValidationReport(
                workspace_path=workspace_path,
                status=status,
                issues=issues,
                recommendations=recommendations,
                isolation_violations=violations
            )
            
        except Exception as e:
            self.logger.error(f"Failed to validate workspace {workspace_path}: {e}")
            return ValidationReport(
                workspace_path=workspace_path,
                status=WorkspaceStatus.ERROR,
                issues=[f"Validation failed: {str(e)}"],
                recommendations=["Check workspace accessibility and permissions"],
                isolation_violations=[]
            )
    
    def list_workspaces(self) -> List[WorkspaceInfo]:
        """
        List all available workspaces.
        
        Returns:
            List of WorkspaceInfo objects
        """
        try:
            workspaces = self.discovery.discover_workspaces()
            workspace_infos = []
            
            for workspace_path in workspaces:
                # Get basic info
                path_obj = Path(workspace_path)
                developer_id = path_obj.name  # Assuming workspace name is developer ID
                
                # Validate to get status
                validation = self.validate_workspace(workspace_path)
                
                workspace_infos.append(WorkspaceInfo(
                    path=path_obj,
                    developer_id=developer_id,
                    status=validation.status
                ))
            
            return workspace_infos
            
        except Exception as e:
            self.logger.error(f"Failed to list workspaces: {e}")
            return []
    
    def promote_workspace_artifacts(
        self, 
        workspace_path: Union[str, Path], 
        target_environment: str = "staging"
    ) -> PromotionResult:
        """
        Promote artifacts from a workspace to target environment.
        
        Args:
            workspace_path: Path to the source workspace
            target_environment: Target environment (staging, production, etc.)
            
        Returns:
            PromotionResult with promotion details
        """
        workspace_path = Path(workspace_path)
        
        try:
            # Use integration manager for promotion
            promoted_artifacts = self.integration_manager.promote_artifacts(
                str(workspace_path), 
                target_environment
            )
            
            return PromotionResult(
                success=True,
                source_workspace=workspace_path,
                target_environment=target_environment,
                message=f"Successfully promoted {len(promoted_artifacts)} artifacts",
                artifacts_promoted=promoted_artifacts
            )
            
        except Exception as e:
            self.logger.error(f"Failed to promote artifacts from {workspace_path}: {e}")
            return PromotionResult(
                success=False,
                source_workspace=workspace_path,
                target_environment=target_environment,
                message=f"Promotion failed: {str(e)}"
            )
    
    def get_system_health(self) -> HealthReport:
        """
        Get overall system health report.
        
        Returns:
            HealthReport with system-wide health information
        """
        try:
            workspace_reports = []
            system_issues = []
            
            # Validate all workspaces
            workspaces = self.list_workspaces()
            for workspace_info in workspaces:
                validation = self.validate_workspace(workspace_info.path)
                workspace_reports.append(validation)
            
            # Determine overall status
            if not workspace_reports:
                overall_status = WorkspaceStatus.UNKNOWN
                system_issues.append("No workspaces found")
            else:
                error_count = sum(1 for r in workspace_reports if r.status == WorkspaceStatus.ERROR)
                warning_count = sum(1 for r in workspace_reports if r.status == WorkspaceStatus.WARNING)
                
                if error_count > 0:
                    overall_status = WorkspaceStatus.ERROR
                elif warning_count > 0:
                    overall_status = WorkspaceStatus.WARNING
                else:
                    overall_status = WorkspaceStatus.HEALTHY
            
            # Generate recommendations
            recommendations = []
            if error_count > 0:
                recommendations.append(f"Address {error_count} workspace(s) with errors")
            if warning_count > 0:
                recommendations.append(f"Review {warning_count} workspace(s) with warnings")
            
            return HealthReport(
                overall_status=overall_status,
                workspace_reports=workspace_reports,
                system_issues=system_issues,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get system health: {e}")
            return HealthReport(
                overall_status=WorkspaceStatus.ERROR,
                workspace_reports=[],
                system_issues=[f"Health check failed: {str(e)}"],
                recommendations=["Check system accessibility and permissions"]
            )
    
    def cleanup_workspaces(
        self, 
        inactive_days: int = 30, 
        dry_run: bool = True
    ) -> CleanupReport:
        """
        Clean up inactive workspaces.
        
        Args:
            inactive_days: Number of days of inactivity before cleanup
            dry_run: If True, only report what would be cleaned
            
        Returns:
            CleanupReport with cleanup results
        """
        try:
            # Use lifecycle manager for cleanup
            cleaned_workspaces = self.lifecycle_manager.cleanup_inactive_workspaces(
                inactive_days=inactive_days,
                dry_run=dry_run
            )
            
            return CleanupReport(
                success=True,
                cleaned_workspaces=[Path(w) for w in cleaned_workspaces],
                errors=[]
            )
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup workspaces: {e}")
            return CleanupReport(
                success=False,
                cleaned_workspaces=[],
                errors=[f"Cleanup failed: {str(e)}"]
            )

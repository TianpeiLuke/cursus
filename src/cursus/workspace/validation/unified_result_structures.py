"""
Unified Result Structures

Standardized data structures for workspace validation results that work
identically for single and multi-workspace scenarios.

Architecture:
- Unified result structures for all validation scenarios
- Consistent data models regardless of workspace count
- Standardized summary and reporting structures
- Backward compatibility with existing result formats

Features:
- Single validation result structure for count=1 or count=N
- Consistent summary statistics and reporting
- Standardized error handling and diagnostics
- Extensible result structure for future enhancements
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field, ConfigDict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseValidationResult(BaseModel):
    """
    Base class for all validation results with common fields.
    PHASE 1 CONSOLIDATION: Reduces duplication across result structures.
    """
    model_config = ConfigDict(
        extra='forbid',
        validate_assignment=True,
        str_strip_whitespace=True
    )
    
    success: bool = Field(
        description="Whether the validation was successful"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the validation was performed"
    )
    workspace_path: Path = Field(
        description="Path to the workspace that was validated"
    )
    messages: List[str] = Field(
        default_factory=list,
        description="Informational messages from validation"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages from validation"
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Error messages from validation"
    )
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0
    
    @property
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0
    
    @property
    def message_count(self) -> int:
        """Get total number of messages."""
        return len(self.messages) + len(self.warnings) + len(self.errors)
    
    def add_message(self, message: str) -> None:
        """Add an informational message."""
        if message not in self.messages:
            self.messages.append(message)
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        if warning not in self.warnings:
            self.warnings.append(warning)
    
    def add_error(self, error: str) -> None:
        """Add an error message."""
        if error not in self.errors:
            self.errors.append(error)


class WorkspaceValidationResult(BaseValidationResult):
    """
    Validation result for workspace validation.
    PHASE 1 CONSOLIDATION: Inherits common fields from BaseValidationResult.
    """
    violations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of validation violations found"
    )
    isolation_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Workspace isolation score (0.0 to 1.0)"
    )
    
    @property
    def has_violations(self) -> bool:
        """Check if there are any violations."""
        return len(self.violations) > 0
    
    @property
    def violation_count(self) -> int:
        """Get number of violations."""
        return len(self.violations)
    
    def add_violation(self, violation: Dict[str, Any]) -> None:
        """Add a validation violation."""
        self.violations.append(violation)


class AlignmentTestResult(BaseValidationResult):
    """
    Result for alignment testing validation.
    PHASE 1 CONSOLIDATION: Inherits common fields from BaseValidationResult.
    """
    alignment_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Alignment score (0.0 to 1.0)"
    )
    failed_checks: List[str] = Field(
        default_factory=list,
        description="List of failed alignment checks"
    )
    level_results: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Results for each validation level"
    )
    
    @property
    def has_failed_checks(self) -> bool:
        """Check if there are any failed checks."""
        return len(self.failed_checks) > 0
    
    @property
    def failed_check_count(self) -> int:
        """Get number of failed checks."""
        return len(self.failed_checks)
    
    def add_failed_check(self, check: str) -> None:
        """Add a failed check."""
        if check not in self.failed_checks:
            self.failed_checks.append(check)


class BuilderTestResult(BaseValidationResult):
    """
    Result for builder testing validation.
    PHASE 1 CONSOLIDATION: Inherits common fields from BaseValidationResult.
    """
    test_results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed test results for each builder"
    )
    total_builders: int = Field(
        default=0,
        ge=0,
        description="Total number of builders tested"
    )
    successful_tests: int = Field(
        default=0,
        ge=0,
        description="Number of successful builder tests"
    )
    failed_tests: int = Field(
        default=0,
        ge=0,
        description="Number of failed builder tests"
    )
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate for builder tests."""
        if self.total_builders == 0:
            return 0.0
        return self.successful_tests / self.total_builders
    
    @property
    def has_test_failures(self) -> bool:
        """Check if there are any test failures."""
        return self.failed_tests > 0
    
    def update_counts(self) -> None:
        """Update test counts based on test results."""
        if self.test_results:
            self.total_builders = len(self.test_results)
            self.successful_tests = sum(
                1 for result in self.test_results.values()
                if result.get('success', False)
            )
            self.failed_tests = self.total_builders - self.successful_tests


class IsolationTestResult(BaseValidationResult):
    """
    Result for isolation testing validation.
    PHASE 1 CONSOLIDATION: Inherits common fields from BaseValidationResult.
    """
    isolation_violations: List[str] = Field(
        default_factory=list,
        description="List of isolation violations"
    )
    boundary_checks: Dict[str, bool] = Field(
        default_factory=dict,
        description="Results of boundary checks"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for fixing issues"
    )
    
    @property
    def is_isolated(self) -> bool:
        """Check if workspace is properly isolated."""
        return len(self.isolation_violations) == 0
    
    @property
    def violation_count(self) -> int:
        """Get number of isolation violations."""
        return len(self.isolation_violations)
    
    def add_violation(self, violation: str) -> None:
        """Add an isolation violation."""
        if violation not in self.isolation_violations:
            self.isolation_violations.append(violation)
    
    def add_recommendation(self, recommendation: str) -> None:
        """Add a recommendation."""
        if recommendation not in self.recommendations:
            self.recommendations.append(recommendation)


class ValidationSummary(BaseModel):
    """
    Unified summary that works for count=1 or count=N workspaces.
    
    This summary structure provides consistent statistics regardless
    of whether validating a single workspace or multiple workspaces.
    """
    model_config = ConfigDict(
        extra='forbid',
        validate_assignment=True
    )
    
    total_workspaces: int = Field(
        description="Total number of workspaces validated"
    )
    successful_workspaces: int = Field(
        description="Number of workspaces that passed validation"
    )
    failed_workspaces: int = Field(
        description="Number of workspaces that failed validation"
    )
    success_rate: float = Field(
        description="Success rate as decimal (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    
    def __post_init__(self):
        """Validate summary consistency."""
        if self.total_workspaces != self.successful_workspaces + self.failed_workspaces:
            logger.warning("Inconsistent workspace counts in validation summary")
    
    @property
    def success_percentage(self) -> float:
        """Get success rate as percentage (0-100)."""
        return self.success_rate * 100.0
    
    @property
    def all_successful(self) -> bool:
        """Check if all workspaces passed validation."""
        return self.failed_workspaces == 0 and self.total_workspaces > 0
    
    @property
    def any_failed(self) -> bool:
        """Check if any workspaces failed validation."""
        return self.failed_workspaces > 0


class WorkspaceValidationEntry(BaseModel):
    """
    Validation result for a single workspace entry.
    
    This structure is used for both single workspace scenarios (count=1)
    and individual workspace results in multi-workspace scenarios.
    """
    model_config = ConfigDict(
        extra='forbid',
        validate_assignment=True
    )
    
    workspace_id: str = Field(
        description="Unique identifier for the workspace"
    )
    workspace_type: str = Field(
        description="Type of workspace (single, developer, shared)"
    )
    workspace_path: str = Field(
        description="File system path to the workspace"
    )
    
    # Validation results
    success: bool = Field(
        description="Overall validation success for this workspace"
    )
    validation_start_time: Optional[str] = Field(
        default=None,
        description="ISO timestamp when validation started"
    )
    validation_end_time: Optional[str] = Field(
        default=None,
        description="ISO timestamp when validation completed"
    )
    validation_duration_seconds: Optional[float] = Field(
        default=None,
        description="Duration of validation in seconds"
    )
    
    # Detailed results by validation type
    alignment_results: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Results from alignment validation"
    )
    builder_results: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Results from builder validation"
    )
    
    # Error information
    error: Optional[str] = Field(
        default=None,
        description="Error message if validation failed"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages from validation"
    )
    
    # Workspace metadata
    developer_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Information about the developer workspace"
    )
    
    @property
    def has_alignment_results(self) -> bool:
        """Check if alignment validation was performed."""
        return self.alignment_results is not None
    
    @property
    def has_builder_results(self) -> bool:
        """Check if builder validation was performed."""
        return self.builder_results is not None
    
    @property
    def validation_types_run(self) -> List[str]:
        """Get list of validation types that were executed."""
        types = []
        if self.has_alignment_results:
            types.append("alignment")
        if self.has_builder_results:
            types.append("builders")
        return types
    
    def get_validation_duration(self) -> Optional[float]:
        """Get validation duration, calculating if not set."""
        if self.validation_duration_seconds is not None:
            return self.validation_duration_seconds
        
        if self.validation_start_time and self.validation_end_time:
            try:
                start = datetime.fromisoformat(self.validation_start_time.replace('Z', '+00:00'))
                end = datetime.fromisoformat(self.validation_end_time.replace('Z', '+00:00'))
                return (end - start).total_seconds()
            except Exception:
                return None
        
        return None


class UnifiedValidationResult(BaseModel):
    """
    Standardized result structure for all validation scenarios.
    
    This structure works identically whether validating a single workspace
    (count=1) or multiple workspaces (count=N), eliminating the need for
    separate result handling logic.
    """
    model_config = ConfigDict(
        extra='forbid',
        validate_assignment=True
    )
    
    # Validation metadata
    workspace_root: str = Field(
        description="Root directory of the workspace(s)"
    )
    workspace_type: str = Field(
        description="Type of workspace structure (single or multi)"
    )
    validation_start_time: str = Field(
        description="ISO timestamp when validation started"
    )
    validation_end_time: Optional[str] = Field(
        default=None,
        description="ISO timestamp when validation completed"
    )
    validation_duration_seconds: Optional[float] = Field(
        default=None,
        description="Total validation duration in seconds"
    )
    
    # Workspace results
    workspaces: Dict[str, WorkspaceValidationEntry] = Field(
        description="Validation results for each workspace"
    )
    
    # Summary statistics
    summary: ValidationSummary = Field(
        description="Summary statistics for all workspaces"
    )
    
    # Recommendations and insights
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations based on validation results"
    )
    
    # Global error information
    global_error: Optional[str] = Field(
        default=None,
        description="Global error that prevented validation"
    )
    
    @property
    def is_single_workspace(self) -> bool:
        """Check if this represents a single workspace validation."""
        return self.workspace_type == "single" or len(self.workspaces) == 1
    
    @property
    def is_multi_workspace(self) -> bool:
        """Check if this represents a multi-workspace validation."""
        return self.workspace_type == "multi" and len(self.workspaces) > 1
    
    @property
    def workspace_count(self) -> int:
        """Get the number of workspaces validated."""
        return len(self.workspaces)
    
    @property
    def workspace_ids(self) -> List[str]:
        """Get list of workspace identifiers."""
        return list(self.workspaces.keys())
    
    @property
    def overall_success(self) -> bool:
        """Check if all workspaces passed validation."""
        return self.summary.all_successful and self.global_error is None
    
    @property
    def has_failures(self) -> bool:
        """Check if any workspaces failed validation."""
        return self.summary.any_failed or self.global_error is not None
    
    def get_workspace_result(self, workspace_id: str) -> Optional[WorkspaceValidationResult]:
        """Get validation result for a specific workspace."""
        return self.workspaces.get(workspace_id)
    
    def get_failed_workspaces(self) -> List[str]:
        """Get list of workspace IDs that failed validation."""
        return [
            workspace_id for workspace_id, result in self.workspaces.items()
            if not result.success
        ]
    
    def get_successful_workspaces(self) -> List[str]:
        """Get list of workspace IDs that passed validation."""
        return [
            workspace_id for workspace_id, result in self.workspaces.items()
            if result.success
        ]
    
    def get_validation_duration(self) -> Optional[float]:
        """Get total validation duration, calculating if not set."""
        if self.validation_duration_seconds is not None:
            return self.validation_duration_seconds
        
        if self.validation_start_time and self.validation_end_time:
            try:
                start = datetime.fromisoformat(self.validation_start_time.replace('Z', '+00:00'))
                end = datetime.fromisoformat(self.validation_end_time.replace('Z', '+00:00'))
                return (end - start).total_seconds()
            except Exception:
                return None
        
        return None
    
    def add_recommendation(self, recommendation: str) -> None:
        """Add a recommendation to the results."""
        if recommendation not in self.recommendations:
            self.recommendations.append(recommendation)
    
    def add_workspace_result(
        self,
        workspace_id: str,
        result: WorkspaceValidationResult
    ) -> None:
        """Add or update a workspace validation result."""
        self.workspaces[workspace_id] = result
        self._update_summary()
    
    def _update_summary(self) -> None:
        """Update summary statistics based on current workspace results."""
        total_workspaces = len(self.workspaces)
        successful_workspaces = len(self.get_successful_workspaces())
        failed_workspaces = total_workspaces - successful_workspaces
        success_rate = successful_workspaces / total_workspaces if total_workspaces > 0 else 0.0
        
        self.summary = ValidationSummary(
            total_workspaces=total_workspaces,
            successful_workspaces=successful_workspaces,
            failed_workspaces=failed_workspaces,
            success_rate=success_rate
        )
    
    def finalize_validation(self, end_time: Optional[datetime] = None) -> None:
        """Finalize validation results with end time and duration."""
        if end_time is None:
            end_time = datetime.now()
        
        self.validation_end_time = end_time.isoformat()
        
        # Calculate duration
        try:
            start = datetime.fromisoformat(self.validation_start_time.replace('Z', '+00:00'))
            self.validation_duration_seconds = (end_time - start).total_seconds()
        except Exception:
            logger.warning("Failed to calculate validation duration")
        
        # Update summary to ensure consistency
        self._update_summary()


class ValidationResultBuilder:
    """
    Builder class for creating unified validation results.
    
    This builder simplifies the creation of validation results and ensures
    consistency across different validation scenarios.
    """
    
    def __init__(
        self,
        workspace_root: str,
        workspace_type: str,
        start_time: Optional[datetime] = None
    ):
        """
        Initialize validation result builder.
        
        Args:
            workspace_root: Root directory of workspace(s)
            workspace_type: Type of workspace structure (single or multi)
            start_time: Validation start time (defaults to now)
        """
        self.workspace_root = workspace_root
        self.workspace_type = workspace_type
        self.start_time = start_time or datetime.now()
        self.workspaces: Dict[str, WorkspaceValidationEntry] = {}
        self.recommendations: List[str] = []
        self.global_error: Optional[str] = None
    
    def add_workspace_result(
        self,
        workspace_id: str,
        workspace_type: str,
        workspace_path: str,
        success: bool,
        alignment_results: Optional[Dict[str, Any]] = None,
        builder_results: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        warnings: Optional[List[str]] = None,
        developer_info: Optional[Dict[str, Any]] = None,
        validation_start_time: Optional[datetime] = None,
        validation_end_time: Optional[datetime] = None
    ) -> 'ValidationResultBuilder':
        """Add a workspace validation result."""
        
        # Calculate validation duration if times provided
        validation_duration = None
        if validation_start_time and validation_end_time:
            validation_duration = (validation_end_time - validation_start_time).total_seconds()
        
        workspace_result = WorkspaceValidationEntry(
            workspace_id=workspace_id,
            workspace_type=workspace_type,
            workspace_path=workspace_path,
            success=success,
            validation_start_time=validation_start_time.isoformat() if validation_start_time else None,
            validation_end_time=validation_end_time.isoformat() if validation_end_time else None,
            validation_duration_seconds=validation_duration,
            alignment_results=alignment_results,
            builder_results=builder_results,
            error=error,
            warnings=warnings or [],
            developer_info=developer_info
        )
        
        self.workspaces[workspace_id] = workspace_result
        return self
    
    def add_recommendation(self, recommendation: str) -> 'ValidationResultBuilder':
        """Add a recommendation."""
        if recommendation not in self.recommendations:
            self.recommendations.append(recommendation)
        return self
    
    def set_global_error(self, error: str) -> 'ValidationResultBuilder':
        """Set a global error that prevented validation."""
        self.global_error = error
        return self
    
    def build(self, end_time: Optional[datetime] = None) -> UnifiedValidationResult:
        """Build the final validation result."""
        if end_time is None:
            end_time = datetime.now()
        
        # Calculate summary statistics
        total_workspaces = len(self.workspaces)
        successful_workspaces = sum(1 for result in self.workspaces.values() if result.success)
        failed_workspaces = total_workspaces - successful_workspaces
        success_rate = successful_workspaces / total_workspaces if total_workspaces > 0 else 0.0
        
        summary = ValidationSummary(
            total_workspaces=total_workspaces,
            successful_workspaces=successful_workspaces,
            failed_workspaces=failed_workspaces,
            success_rate=success_rate
        )
        
        # Calculate total duration
        validation_duration = (end_time - self.start_time).total_seconds()
        
        return UnifiedValidationResult(
            workspace_root=self.workspace_root,
            workspace_type=self.workspace_type,
            validation_start_time=self.start_time.isoformat(),
            validation_end_time=end_time.isoformat(),
            validation_duration_seconds=validation_duration,
            workspaces=self.workspaces,
            summary=summary,
            recommendations=self.recommendations,
            global_error=self.global_error
        )


def create_single_workspace_result(
    workspace_root: str,
    workspace_id: str = "default",
    workspace_type: str = "single",
    workspace_path: Optional[str] = None,
    success: bool = True,
    alignment_results: Optional[Dict[str, Any]] = None,
    builder_results: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    recommendations: Optional[List[str]] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> UnifiedValidationResult:
    """
    Convenience function to create a single workspace validation result.
    
    This function creates a UnifiedValidationResult for a single workspace,
    treating it as a multi-workspace scenario with count=1.
    """
    builder = ValidationResultBuilder(
        workspace_root=workspace_root,
        workspace_type=workspace_type,
        start_time=start_time
    )
    
    builder.add_workspace_result(
        workspace_id=workspace_id,
        workspace_type=workspace_type,
        workspace_path=workspace_path or workspace_root,
        success=success,
        alignment_results=alignment_results,
        builder_results=builder_results,
        error=error,
        validation_start_time=start_time,
        validation_end_time=end_time
    )
    
    if recommendations:
        for rec in recommendations:
            builder.add_recommendation(rec)
    
    return builder.build(end_time)


def create_empty_result(
    workspace_root: str,
    workspace_type: str = "unknown",
    error: str = "No workspaces found",
    start_time: Optional[datetime] = None
) -> UnifiedValidationResult:
    """
    Create an empty validation result for cases where no workspaces are found.
    """
    builder = ValidationResultBuilder(
        workspace_root=workspace_root,
        workspace_type=workspace_type,
        start_time=start_time
    )
    
    builder.set_global_error(error)
    builder.add_recommendation("Ensure workspace structure is properly configured")
    
    return builder.build()

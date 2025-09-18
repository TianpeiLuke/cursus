"""
Simplified Data Models for Hybrid Registry System

This module contains essential Pydantic data models for the Phase 3 simplified
hybrid registry system with optimized validation using enums.

Models:
- StepDefinition: Core step definition with registry metadata
- ResolutionContext: Context for step resolution
- StepResolutionResult: Result of step resolution
- RegistryValidationResult: Results of registry validation
- ConflictAnalysis: Basic analysis of step name conflicts
"""

from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Dict, List, Any, Optional
from enum import Enum


# Shared validation enums (eliminates custom validators)
class RegistryType(str, Enum):
    """Registry type enumeration for automatic validation."""

    CORE = "core"
    WORKSPACE = "workspace"
    OVERRIDE = "override"


class ResolutionMode(str, Enum):
    """Resolution mode enumeration for automatic validation."""

    AUTOMATIC = "automatic"
    INTERACTIVE = "interactive"
    STRICT = "strict"


class ResolutionStrategy(str, Enum):
    """Resolution strategy enumeration for automatic validation."""

    WORKSPACE_PRIORITY = "workspace_priority"
    FRAMEWORK_MATCH = "framework_match"
    ENVIRONMENT_MATCH = "environment_match"
    MANUAL = "manual"
    HIGHEST_PRIORITY = "highest_priority"
    CORE_FALLBACK = "core_fallback"


class ConflictType(str, Enum):
    """Conflict type enumeration for automatic validation."""

    NAME_CONFLICT = "name_conflict"
    FRAMEWORK_CONFLICT = "framework_conflict"
    VERSION_CONFLICT = "version_conflict"
    WORKSPACE_CONFLICT = "workspace_conflict"


class StepDefinition(BaseModel):
    """Enhanced step definition with registry metadata using Pydantic V2 and enum validation."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        frozen=False,
        str_strip_whitespace=True,
    )

    name: str = Field(..., min_length=1, description="Step name identifier")
    registry_type: RegistryType = Field(
        ..., description="Registry type using enum validation"
    )
    config_class: Optional[str] = Field(None, description="Configuration class name")
    spec_type: Optional[str] = Field(None, description="Specification type")
    sagemaker_step_type: Optional[str] = Field(None, description="SageMaker step type")
    builder_step_name: Optional[str] = Field(None, description="Builder class name")
    description: Optional[str] = Field(None, description="Step description")
    framework: Optional[str] = Field(None, description="Framework used by step")
    job_types: List[str] = Field(
        default_factory=list, description="Supported job types"
    )
    workspace_id: Optional[str] = Field(
        None, description="Workspace identifier for workspace registrations"
    )
    override_source: Optional[str] = Field(
        None, description="Source of override for tracking"
    )

    # Conflict resolution metadata
    priority: int = Field(
        default=100, description="Resolution priority (lower = higher priority)"
    )
    compatibility_tags: List[str] = Field(
        default_factory=list, description="Compatibility tags for smart resolution"
    )
    framework_version: Optional[str] = Field(
        None, description="Framework version for compatibility checking"
    )
    environment_tags: List[str] = Field(
        default_factory=list, description="Environment compatibility tags"
    )
    conflict_resolution_strategy: ResolutionStrategy = Field(
        default=ResolutionStrategy.WORKSPACE_PRIORITY,
        description="Strategy for resolving conflicts using enum validation",
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    # Simplified validation for identifiers only (enum validation handles registry_type and strategy)
    @field_validator("name", "builder_step_name")
    @classmethod
    def validate_identifiers(cls, v: Optional[str]) -> Optional[str]:
        """Validate identifier fields using simple validation."""
        if v is not None:
            if not v or not v.strip():
                raise ValueError("Step name cannot be empty")
            if not v.replace("_", "").replace("-", "").isalnum():
                raise ValueError(f"Step name '{v}' contains invalid characters")
            return v.strip()
        return v

    @property
    def step_name(self) -> str:
        """Alias for name attribute to maintain compatibility with step catalog."""
        return self.name

    def to_legacy_format(self) -> Dict[str, Any]:
        """Convert to legacy STEP_NAMES format using shared converter."""
        from .utils import to_legacy_format

        return to_legacy_format(self)


class ResolutionContext(BaseModel):
    """Context for step resolution and conflict resolution using Pydantic V2 and enum validation."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid", frozen=False)

    workspace_id: Optional[str] = Field(None, description="Current workspace context")
    preferred_framework: Optional[str] = Field(
        None, description="Preferred framework for resolution"
    )
    environment_tags: List[str] = Field(
        default_factory=list, description="Current environment tags"
    )
    resolution_mode: ResolutionMode = Field(
        default=ResolutionMode.AUTOMATIC,
        description="Resolution mode using enum validation",
    )
    resolution_strategy: ResolutionStrategy = Field(
        default=ResolutionStrategy.WORKSPACE_PRIORITY,
        description="Strategy for conflict resolution using enum validation",
    )


class StepResolutionResult(BaseModel):
    """Result of step conflict resolution using Pydantic V2."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="allow",  # Allow extra fields for test compatibility
        frozen=False,
    )

    step_name: str = Field(..., description="Step name being resolved")
    resolved: bool = Field(..., description="Whether resolution was successful")
    selected_definition: Optional[StepDefinition] = Field(
        None, description="Selected step definition"
    )
    reason: Optional[str] = Field(None, description="Reason for resolution result")
    conflicting_definitions: List[StepDefinition] = Field(
        default_factory=list, description="Conflicting definitions found"
    )
    source_registry: str = Field(..., description="Source registry of resolved step")
    resolution_strategy: str = Field(..., description="Strategy used for resolution")
    workspace_id: Optional[str] = Field(None, description="Workspace context")
    conflict_detected: bool = Field(
        default=False, description="Whether conflicts were detected"
    )
    conflict_analysis: Optional["ConflictAnalysis"] = Field(
        None, description="Analysis of conflicts if any"
    )
    errors: List[str] = Field(default_factory=list, description="Resolution errors")
    warnings: List[str] = Field(default_factory=list, description="Resolution warnings")
    resolution_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional resolution metadata"
    )

    def get_resolution_summary(self) -> Dict[str, Any]:
        """Get a summary of the resolution result."""
        return {
            "step_name": self.step_name,
            "resolved": self.resolved,
            "strategy": self.resolution_strategy,
            "selected_workspace": (
                self.selected_definition.workspace_id
                if self.selected_definition
                else None
            ),
            "conflict_count": len(self.conflicting_definitions),
            "reason": self.reason,
        }


class RegistryValidationResult(BaseModel):
    """Results of registry validation using Pydantic V2."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="allow",  # Allow extra fields for test compatibility
        frozen=False,
    )

    is_valid: bool = Field(..., description="Whether validation passed")
    registry_type: str = Field(
        default="unknown", description="Type of registry validated"
    )
    issues: List[str] = Field(
        default_factory=list, description="List of validation issues"
    )
    errors: List[str] = Field(
        default_factory=list, description="List of validation errors"
    )
    warnings: List[str] = Field(
        default_factory=list, description="List of validation warnings"
    )
    step_count: int = Field(default=0, ge=0, description="Number of steps validated")

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of validation results."""
        return {
            "valid": self.is_valid,
            "issue_count": len(self.issues),
            "registry_type": self.registry_type,
            "step_count": self.step_count,
        }


class ConflictAnalysis(BaseModel):
    """Analysis of a step name conflict using Pydantic V2."""

    model_config = ConfigDict(
        validate_assignment=True,
        extra="allow",  # Allow extra fields for test compatibility
        frozen=False,
    )

    step_name: str = Field(..., min_length=1, description="Conflicting step name")
    conflicting_definitions: List[StepDefinition] = Field(
        default_factory=list, description="Conflicting step definitions"
    )
    resolution_strategies: List[str] = Field(
        default_factory=list, description="Available resolution strategies"
    )
    recommended_strategy: Optional[str] = Field(
        None, description="Recommended resolution strategy"
    )
    impact_assessment: Optional[str] = Field(
        None, description="Impact assessment of the conflict"
    )
    conflicting_sources: List[str] = Field(
        ..., description="Sources of conflicting definitions"
    )
    resolution_strategy: str = Field(..., description="Strategy used for resolution")
    workspace_context: Optional[str] = Field(
        None, description="Workspace context for resolution"
    )
    conflict_type: str = Field(
        default="name_conflict", description="Type of conflict identified"
    )

    def get_conflict_summary(self) -> Dict[str, Any]:
        """Get a summary of the conflict."""
        return {
            "step_name": self.step_name,
            "conflict_type": self.conflict_type,
            "definition_count": len(self.conflicting_definitions),
            "involved_workspaces": [
                d.workspace_id for d in self.conflicting_definitions if d.workspace_id
            ],
            "frameworks": list(
                {d.framework for d in self.conflicting_definitions if d.framework}
            ),
            "strategy_count": len(self.resolution_strategies),
        }

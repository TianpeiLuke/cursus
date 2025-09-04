"""
Data Models for Hybrid Registry System

This module contains all Pydantic data models used throughout the hybrid registry
system, including step definitions, resolution contexts, and validation results.

Models:
- StepDefinition: Enhanced step definition with registry metadata
- NamespacedStepDefinition: Step definition with namespace support for conflict resolution
- ResolutionContext: Context for step resolution and conflict resolution
- StepResolutionResult: Result of step conflict resolution
- RegistryValidationResult: Results of registry validation
- ConflictAnalysis: Analysis of step name conflicts
- StepComponentResolution: Result of step component resolution
- DistributedRegistryValidationResult: Results of distributed registry validation
"""

from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Dict, List, Any, Optional
from .utils import RegistryValidationUtils


class StepDefinition(BaseModel):
    """Enhanced step definition with registry metadata using Pydantic V2."""
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid',
        frozen=False,
        str_strip_whitespace=True
    )
    
    name: str = Field(..., min_length=1, description="Step name identifier")
    registry_type: str = Field(..., description="Registry type: 'core', 'workspace', 'override'")
    config_class: Optional[str] = Field(None, description="Configuration class name")
    spec_type: Optional[str] = Field(None, description="Specification type")
    sagemaker_step_type: Optional[str] = Field(None, description="SageMaker step type")
    builder_step_name: Optional[str] = Field(None, description="Builder class name")
    description: Optional[str] = Field(None, description="Step description")
    framework: Optional[str] = Field(None, description="Framework used by step")
    job_types: List[str] = Field(default_factory=list, description="Supported job types")
    workspace_id: Optional[str] = Field(None, description="Workspace identifier for workspace registrations")
    override_source: Optional[str] = Field(None, description="Source of override for tracking")
    
    # Conflict resolution metadata
    priority: int = Field(default=100, description="Resolution priority (lower = higher priority)")
    compatibility_tags: List[str] = Field(default_factory=list, description="Compatibility tags for smart resolution")
    framework_version: Optional[str] = Field(None, description="Framework version for compatibility checking")
    environment_tags: List[str] = Field(default_factory=list, description="Environment compatibility tags")
    conflict_resolution_strategy: str = Field(
        default="workspace_priority", 
        description="Strategy for resolving conflicts: 'workspace_priority', 'framework_match', 'environment_match', 'manual'"
    )
    
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @field_validator('registry_type')
    @classmethod
    def validate_registry_type(cls, v: str) -> str:
        """Validate registry type using shared validation utilities."""
        return RegistryValidationUtils.validate_registry_type(v)
    
    @field_validator('name', 'builder_step_name')
    @classmethod
    def validate_identifiers(cls, v: Optional[str]) -> Optional[str]:
        """Validate identifier fields using shared validation utilities."""
        if v is not None:
            return RegistryValidationUtils.validate_step_name(v)
        return v
    
    def to_legacy_format(self) -> Dict[str, Any]:
        """Convert to legacy STEP_NAMES format using shared converter."""
        from .utils import StepDefinitionConverter
        return StepDefinitionConverter.to_legacy_format(self)


class NamespacedStepDefinition(StepDefinition):
    """Enhanced step definition with namespace support using Pydantic V2."""
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid',
        frozen=False,
        str_strip_whitespace=True
    )
    
    # Core namespace fields
    namespace: str = Field(..., min_length=1, description="Step namespace (workspace_id or 'core')")
    qualified_name: str = Field(..., min_length=1, description="Fully qualified step name: namespace.name")
    
    # Conflict resolution metadata
    priority: int = Field(default=100, description="Resolution priority (lower = higher priority)")
    compatibility_tags: List[str] = Field(default_factory=list, description="Compatibility tags for smart resolution")
    framework_version: Optional[str] = Field(None, description="Framework version for compatibility checking")
    environment_tags: List[str] = Field(default_factory=list, description="Environment compatibility tags")
    
    # Conflict resolution hints
    conflict_resolution_strategy: str = Field(
        default="workspace_priority", 
        description="Strategy for resolving conflicts: 'workspace_priority', 'framework_match', 'environment_match', 'manual'"
    )
    
    def model_post_init(self, __context: Any) -> None:
        """Generate qualified name after initialization."""
        if not self.qualified_name:
            self.qualified_name = f"{self.namespace}.{self.name}"
    
    @field_validator('conflict_resolution_strategy')
    @classmethod
    def validate_resolution_strategy(cls, v: str) -> str:
        """Validate conflict resolution strategy."""
        allowed_strategies = {'workspace_priority', 'framework_match', 'environment_match', 'manual'}
        if v not in allowed_strategies:
            raise ValueError(f"conflict_resolution_strategy must be one of {allowed_strategies}")
        return v
    
    def is_compatible_with(self, other: 'NamespacedStepDefinition') -> bool:
        """Check if this step definition is compatible with another."""
        # Same framework compatibility
        if self.framework and other.framework:
            if self.framework != other.framework:
                return False
        
        # Environment compatibility
        if self.environment_tags and other.environment_tags:
            if not set(self.environment_tags).intersection(set(other.environment_tags)):
                return False
        
        # Compatibility tags
        if self.compatibility_tags and other.compatibility_tags:
            return bool(set(self.compatibility_tags).intersection(set(other.compatibility_tags)))
        
        return True
    
    def get_resolution_score(self, context: 'ResolutionContext') -> int:
        """Calculate resolution score for conflict resolution."""
        score = self.priority
        
        # Framework match bonus
        if context.preferred_framework and self.framework == context.preferred_framework:
            score -= 50
        
        # Environment match bonus
        if context.environment_tags:
            matching_env_tags = set(self.environment_tags).intersection(set(context.environment_tags))
            score -= len(matching_env_tags) * 10
        
        # Workspace preference bonus
        if context.workspace_id and self.workspace_id == context.workspace_id:
            score -= 30
        
        return score


class ResolutionContext(BaseModel):
    """Context for step resolution and conflict resolution using Pydantic V2."""
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid',
        frozen=False
    )
    
    workspace_id: Optional[str] = Field(None, description="Current workspace context")
    preferred_framework: Optional[str] = Field(None, description="Preferred framework for resolution")
    environment_tags: List[str] = Field(default_factory=list, description="Current environment tags")
    resolution_mode: str = Field(default="automatic", description="Resolution mode: 'automatic', 'interactive', 'strict'")
    resolution_strategy: str = Field(default="workspace_priority", description="Strategy for conflict resolution")
    
    @field_validator('resolution_mode')
    @classmethod
    def validate_resolution_mode(cls, v: str) -> str:
        """Validate resolution mode."""
        allowed_modes = {'automatic', 'interactive', 'strict'}
        if v not in allowed_modes:
            raise ValueError(f"resolution_mode must be one of {allowed_modes}")
        return v
    
    @field_validator('resolution_strategy')
    @classmethod
    def validate_resolution_strategy(cls, v: str) -> str:
        """Validate resolution strategy."""
        allowed_strategies = {'workspace_priority', 'framework_match', 'environment_match', 'manual', 'highest_priority', 'core_fallback'}
        if v not in allowed_strategies:
            raise ValueError(f"resolution_strategy must be one of {allowed_strategies}")
        return v


class StepResolutionResult(BaseModel):
    """Result of step conflict resolution using Pydantic V2."""
    model_config = ConfigDict(
        validate_assignment=True,
        extra='allow',  # Allow extra fields for test compatibility
        frozen=False
    )
    
    step_name: str = Field(..., description="Step name being resolved")
    resolved: bool = Field(..., description="Whether resolution was successful")
    selected_definition: Optional[StepDefinition] = Field(None, description="Selected step definition")
    reason: Optional[str] = Field(None, description="Reason for resolution result")
    conflicting_definitions: List[StepDefinition] = Field(default_factory=list, description="Conflicting definitions found")
    source_registry: str = Field(..., description="Source registry of resolved step")
    resolution_strategy: str = Field(..., description="Strategy used for resolution")
    workspace_id: Optional[str] = Field(None, description="Workspace context")
    conflict_detected: bool = Field(default=False, description="Whether conflicts were detected")
    conflict_analysis: Optional['ConflictAnalysis'] = Field(None, description="Analysis of conflicts if any")
    errors: List[str] = Field(default_factory=list, description="Resolution errors")
    warnings: List[str] = Field(default_factory=list, description="Resolution warnings")
    resolution_metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional resolution metadata"
    )
    
    def get_resolution_summary(self) -> Dict[str, Any]:
        """Get a summary of the resolution result."""
        return {
            'step_name': self.step_name,
            'resolved': self.resolved,
            'strategy': self.resolution_strategy,
            'selected_namespace': self.selected_definition.namespace if self.selected_definition else None,
            'conflict_count': len(self.conflicting_definitions),
            'reason': self.reason
        }


class RegistryValidationResult(BaseModel):
    """Results of registry validation using Pydantic V2."""
    model_config = ConfigDict(
        validate_assignment=True,
        extra='allow',  # Allow extra fields for test compatibility
        frozen=False
    )
    
    is_valid: bool = Field(..., description="Whether validation passed")
    registry_type: str = Field(default="unknown", description="Type of registry validated")
    issues: List[str] = Field(default_factory=list, description="List of validation issues")
    errors: List[str] = Field(default_factory=list, description="List of validation errors")
    warnings: List[str] = Field(default_factory=list, description="List of validation warnings")
    step_count: int = Field(default=0, ge=0, description="Number of steps validated")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of validation results."""
        return {
            'valid': self.is_valid,
            'issue_count': len(self.issues),
            'registry_type': self.registry_type,
            'step_count': self.step_count
        }


class ConflictAnalysis(BaseModel):
    """Analysis of a step name conflict using Pydantic V2."""
    model_config = ConfigDict(
        validate_assignment=True,
        extra='allow',  # Allow extra fields for test compatibility
        frozen=False
    )
    
    step_name: str = Field(..., min_length=1, description="Conflicting step name")
    conflicting_definitions: List[StepDefinition] = Field(default_factory=list, description="Conflicting step definitions")
    resolution_strategies: List[str] = Field(default_factory=list, description="Available resolution strategies")
    recommended_strategy: Optional[str] = Field(None, description="Recommended resolution strategy")
    impact_assessment: Optional[str] = Field(None, description="Impact assessment of the conflict")
    conflicting_sources: List[str] = Field(..., description="Sources of conflicting definitions")
    resolution_strategy: str = Field(..., description="Strategy used for resolution")
    workspace_context: Optional[str] = Field(None, description="Workspace context for resolution")
    conflict_type: str = Field(default="name_conflict", description="Type of conflict identified")
    
    def get_conflict_summary(self) -> Dict[str, Any]:
        """Get a summary of the conflict."""
        return {
            'step_name': self.step_name,
            'conflict_type': self.conflict_type,
            'definition_count': len(self.conflicting_definitions),
            'involved_namespaces': [d.namespace for d in self.conflicting_definitions],
            'frameworks': list({d.framework for d in self.conflicting_definitions if d.framework}),
            'recommendation_count': len(self.resolution_recommendations)
        }


class StepComponentResolution(BaseModel):
    """Result of step component resolution using Pydantic V2."""
    model_config = ConfigDict(
        validate_assignment=True,
        extra='allow',  # Allow extra fields for test compatibility
        frozen=False,
        str_strip_whitespace=True
    )
    
    step_name: str = Field(..., min_length=1, description="Step name being resolved")
    resolved: bool = Field(..., description="Whether resolution was successful")
    components: Dict[str, Optional[str]] = Field(default_factory=dict, description="Component paths")
    config_class: Optional[str] = Field(None, description="Config class name")
    builder_class: Optional[str] = Field(None, description="Builder class name")
    spec_class: Optional[str] = Field(None, description="Spec class name")
    script_path: Optional[str] = Field(None, description="Script path")
    contract_class: Optional[str] = Field(None, description="Contract class name")
    resolution_source: Optional[str] = Field(None, description="Resolution source")
    workspace_id: Optional[str] = Field(None, description="Workspace context for resolution")
    resolution_strategy: str = Field(default="workspace_priority", description="Strategy used for resolution")
    errors: List[str] = Field(default_factory=list, description="Resolution errors")
    warnings: List[str] = Field(default_factory=list, description="Resolution warnings")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    source_registry: Optional[str] = Field(None, description="Source registry")
    
    def get_available_components(self) -> List[str]:
        """Get list of available component types."""
        components = []
        if self.builder_path:
            components.append('builder')
        if self.config_path:
            components.append('config')
        if self.spec_path:
            components.append('spec')
        if self.contract_path:
            components.append('contract')
        if self.script_path:
            components.append('script')
        return components
    
    def is_complete(self) -> bool:
        """Check if all expected components are available."""
        # At minimum, we expect a builder
        return self.builder_path is not None


class DistributedRegistryValidationResult(BaseModel):
    """Results of distributed registry validation using Pydantic V2."""
    model_config = ConfigDict(
        validate_assignment=True,
        extra='forbid',
        frozen=False
    )
    
    is_valid: bool = Field(default=False, description="Overall validation status")
    core_validation: Optional[RegistryValidationResult] = Field(None, description="Core registry validation result")
    workspace_validations: Dict[str, RegistryValidationResult] = Field(
        default_factory=dict, 
        description="Workspace validation results by workspace ID"
    )
    conflicts: Dict[str, List[StepDefinition]] = Field(
        default_factory=dict, 
        description="Step conflicts between workspaces"
    )
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of validation results."""
        return {
            'overall_valid': self.is_valid,
            'core_valid': self.core_validation.is_valid if self.core_validation else False,
            'workspace_count': len(self.workspace_validations),
            'valid_workspaces': sum(1 for v in self.workspace_validations.values() if v.is_valid),
            'conflict_count': len(self.conflicts),
            'conflicted_steps': list(self.conflicts.keys())
        }

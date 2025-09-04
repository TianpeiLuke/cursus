"""
Streamlined Utility Functions for Hybrid Registry System

This module provides essential utility functions without over-engineering.
Replaces complex utility classes with simple, focused functions.
"""

import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, field_validator, ValidationError
from ..exceptions import RegistryError


class RegistryLoadError(RegistryError):
    """Error loading registry from file."""
    pass


# Simple loading functions (replaces RegistryLoader class)
def load_registry_module(file_path: str) -> Any:
    """
    Load registry module from file.
    
    Args:
        file_path: Path to the registry file
        
    Returns:
        Loaded module object
        
    Raises:
        RegistryLoadError: If module loading fails
    """
    try:
        if not Path(file_path).exists():
            raise RegistryLoadError(f"Registry file not found: {file_path}")
        
        spec = importlib.util.spec_from_file_location("registry", file_path)
        if spec is None or spec.loader is None:
            raise RegistryLoadError(f"Could not create module spec from {file_path}")
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
        
    except Exception as e:
        if isinstance(e, RegistryLoadError):
            raise
        raise RegistryLoadError(f"Failed to load registry from {file_path}: {e}")


def get_step_names_from_module(module: Any) -> Dict[str, Dict[str, Any]]:
    """
    Extract STEP_NAMES from loaded module.
    
    Args:
        module: Loaded registry module
        
    Returns:
        STEP_NAMES dictionary
    """
    return getattr(module, 'STEP_NAMES', {})


# Simple conversion functions (replaces StepDefinitionConverter class)
def from_legacy_format(step_name: str, 
                      step_info: Dict[str, Any], 
                      registry_type: str = 'core', 
                      workspace_id: str = None) -> 'StepDefinition':
    """
    Convert legacy STEP_NAMES format to StepDefinition.
    
    Args:
        step_name: Name of the step
        step_info: Legacy step information dictionary
        registry_type: Type of registry ('core', 'workspace', 'override')
        workspace_id: Workspace identifier for workspace steps
        
    Returns:
        StepDefinition object
    """
    from .models import StepDefinition
    
    return StepDefinition(
        name=step_name,
        registry_type=registry_type,
        workspace_id=workspace_id,
        config_class=step_info.get('config_class'),
        spec_type=step_info.get('spec_type'),
        sagemaker_step_type=step_info.get('sagemaker_step_type'),
        builder_step_name=step_info.get('builder_step_name'),
        description=step_info.get('description'),
        framework=step_info.get('framework'),
        job_types=step_info.get('job_types', []),
        metadata=step_info.get('metadata', {})
    )


def to_legacy_format(definition: 'StepDefinition') -> Dict[str, Any]:
    """
    Convert StepDefinition to legacy STEP_NAMES format.
    
    Args:
        definition: StepDefinition object
        
    Returns:
        Legacy format dictionary
    """
    legacy_dict = {}
    
    # Standard fields
    if definition.config_class:
        legacy_dict['config_class'] = definition.config_class
    if definition.builder_step_name:
        legacy_dict['builder_step_name'] = definition.builder_step_name
    if definition.spec_type:
        legacy_dict['spec_type'] = definition.spec_type
    if definition.sagemaker_step_type:
        legacy_dict['sagemaker_step_type'] = definition.sagemaker_step_type
    if definition.description:
        legacy_dict['description'] = definition.description
    if definition.framework:
        legacy_dict['framework'] = definition.framework
    if definition.job_types:
        legacy_dict['job_types'] = definition.job_types
    
    # Additional metadata
    if hasattr(definition, 'metadata') and definition.metadata:
        legacy_dict.update(definition.metadata)
    
    return legacy_dict


def convert_registry_dict(registry_dict: Dict[str, Dict[str, Any]], 
                         registry_type: str = 'core',
                         workspace_id: str = None) -> Dict[str, 'StepDefinition']:
    """
    Convert a complete registry dictionary to StepDefinition objects.
    
    Args:
        registry_dict: Dictionary of step_name -> step_info
        registry_type: Type of registry
        workspace_id: Workspace identifier
        
    Returns:
        Dictionary of step_name -> StepDefinition
    """
    return {
        step_name: from_legacy_format(step_name, step_info, registry_type, workspace_id)
        for step_name, step_info in registry_dict.items()
    }


# Simple validation using Pydantic (replaces RegistryValidationUtils class)
class RegistryValidationModel(BaseModel):
    """Pydantic model for registry validation."""
    registry_type: str
    step_name: str
    workspace_id: Optional[str] = None
    
    @field_validator('registry_type')
    @classmethod
    def validate_registry_type(cls, v: str) -> str:
        allowed_types = {'core', 'workspace', 'override'}
        if v not in allowed_types:
            raise ValueError(f"registry_type must be one of {allowed_types}, got: {v}")
        return v
    
    @field_validator('step_name')
    @classmethod
    def validate_step_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Step name cannot be empty")
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError(f"Step name '{v}' contains invalid characters")
        return v.strip()


def validate_registry_data(registry_type: str, step_name: str, workspace_id: str = None) -> bool:
    """
    Validate registry data using Pydantic model.
    
    Args:
        registry_type: Registry type to validate
        step_name: Step name to validate
        workspace_id: Optional workspace ID to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    RegistryValidationModel(
        registry_type=registry_type,
        step_name=step_name,
        workspace_id=workspace_id
    )
    return True


# Simple error formatting functions (replaces RegistryErrorFormatter class)
def format_step_not_found_error(step_name: str, 
                               workspace_context: str = None,
                               available_steps: List[str] = None) -> str:
    """Format step not found error messages."""
    context_info = f" (workspace: {workspace_context})" if workspace_context else " (core registry)"
    error_msg = f"Step '{step_name}' not found{context_info}"
    
    if available_steps:
        error_msg += f". Available steps: {', '.join(sorted(available_steps))}"
    
    return error_msg


def format_registry_load_error(registry_path: str, error_details: str) -> str:
    """Format registry loading error messages."""
    return f"Failed to load registry from '{registry_path}': {error_details}"


def format_validation_error(component_name: str, validation_issues: List[str]) -> str:
    """Format validation error messages."""
    error_msg = f"Validation failed for '{component_name}':"
    for i, issue in enumerate(validation_issues, 1):
        error_msg += f"\n  {i}. {issue}"
    return error_msg

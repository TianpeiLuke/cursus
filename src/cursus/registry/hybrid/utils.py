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
    return getattr(module, "STEP_NAMES", {})


# Simple conversion functions (replaces StepDefinitionConverter class)
def from_legacy_format(
    step_name: str,
    step_info: Dict[str, Any],
    registry_type: str = "core",
    workspace_id: str = None,
) -> "StepDefinition":
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
        config_class=step_info.get("config_class"),
        spec_type=step_info.get("spec_type"),
        sagemaker_step_type=step_info.get("sagemaker_step_type"),
        builder_step_name=step_info.get("builder_step_name"),
        description=step_info.get("description"),
        framework=step_info.get("framework"),
        job_types=step_info.get("job_types", []),
        metadata=step_info.get("metadata", {}),
    )


# Legacy format field names for conversion optimization
LEGACY_FIELDS = [
    "config_class",
    "builder_step_name",
    "spec_type",
    "sagemaker_step_type",
    "description",
    "framework",
    "job_types",
]


def to_legacy_format(definition: "StepDefinition") -> Dict[str, Any]:
    """
    Convert StepDefinition to legacy STEP_NAMES format using field list.

    Args:
        definition: StepDefinition object

    Returns:
        Legacy format dictionary
    """
    legacy_dict = {}

    # Convert standard fields using field list
    for field_name in LEGACY_FIELDS:
        value = getattr(definition, field_name, None)
        if value is not None:
            legacy_dict[field_name] = value

    # Add metadata if present
    if hasattr(definition, "metadata") and definition.metadata:
        legacy_dict.update(definition.metadata)

    return legacy_dict


def convert_registry_dict(
    registry_dict: Dict[str, Dict[str, Any]],
    registry_type: str = "core",
    workspace_id: str = None,
) -> Dict[str, "StepDefinition"]:
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


# Simple validation functions (replaces RegistryValidationModel class)
def validate_registry_type(registry_type: str) -> str:
    """
    Validate registry type using enum values.

    Args:
        registry_type: Registry type to validate

    Returns:
        Validated registry type

    Raises:
        ValueError: If registry type is invalid
    """
    from .models import RegistryType

    try:
        # Use enum validation
        validated = RegistryType(registry_type)
        return validated.value
    except ValueError:
        allowed_types = [rt.value for rt in RegistryType]
        raise ValueError(
            f"registry_type must be one of {allowed_types}, got: {registry_type}"
        )


def validate_step_name(step_name: str) -> str:
    """
    Validate step name format.

    Args:
        step_name: Step name to validate

    Returns:
        Validated and stripped step name

    Raises:
        ValueError: If step name is invalid
    """
    if not step_name or not step_name.strip():
        raise ValueError("Step name cannot be empty")
    if not step_name.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"Step name '{step_name}' contains invalid characters")
    return step_name.strip()


def validate_workspace_id(workspace_id: Optional[str]) -> Optional[str]:
    """
    Validate workspace ID format.

    Args:
        workspace_id: Workspace ID to validate

    Returns:
        Validated workspace ID or None

    Raises:
        ValueError: If workspace ID is invalid
    """
    if workspace_id is None:
        return None
    return validate_step_name(workspace_id)  # Same validation rules


def validate_registry_data(
    registry_type: str, step_name: str, workspace_id: str = None
) -> bool:
    """
    Validate registry data using direct validation functions.

    Args:
        registry_type: Registry type to validate
        step_name: Step name to validate
        workspace_id: Optional workspace ID to validate

    Returns:
        True if valid

    Raises:
        ValueError: If validation fails
    """
    validate_registry_type(registry_type)
    validate_step_name(step_name)
    validate_workspace_id(workspace_id)
    return True


# Generic error formatting with templates (replaces multiple specific error functions)
ERROR_TEMPLATES = {
    "step_not_found": "Step '{step_name}' not found{context}{suggestions}",
    "registry_load": "Failed to load registry from '{registry_path}': {error_details}",
    "validation": "Validation failed for '{component_name}':{issues}",
    "workspace_not_found": "Workspace '{workspace_id}' not found{suggestions}",
    "conflict_detected": "Step name conflict detected for '{step_name}'{context}",
    "invalid_registry_type": "Invalid registry type '{registry_type}'{suggestions}",
}


def format_registry_error(error_type: str, **kwargs) -> str:
    """
    Generic error formatter using templates.

    Args:
        error_type: Type of error to format
        **kwargs: Template variables

    Returns:
        Formatted error message
    """
    template = ERROR_TEMPLATES.get(error_type, "Registry error: {error}")

    # Special formatting for specific error types
    if error_type == "step_not_found":
        context = (
            f" (workspace: {kwargs.get('workspace_context')})"
            if kwargs.get("workspace_context")
            else " (core registry)"
        )
        suggestions = (
            f". Available steps: {', '.join(sorted(kwargs['available_steps']))}"
            if kwargs.get("available_steps")
            else ""
        )
        return template.format(context=context, suggestions=suggestions, **kwargs)

    elif error_type == "validation":
        issues = "".join(
            f"\n  {i}. {issue}"
            for i, issue in enumerate(kwargs.get("validation_issues", []), 1)
        )
        return template.format(issues=issues, **kwargs)

    elif error_type == "workspace_not_found":
        suggestions = (
            f". Available workspaces: {', '.join(sorted(kwargs['available_workspaces']))}"
            if kwargs.get("available_workspaces")
            else ""
        )
        return template.format(suggestions=suggestions, **kwargs)

    elif error_type == "invalid_registry_type":
        suggestions = (
            f". Valid types: {', '.join(kwargs['valid_types'])}"
            if kwargs.get("valid_types")
            else ""
        )
        return template.format(suggestions=suggestions, **kwargs)

    else:
        return template.format(**kwargs)


# Backward compatibility functions (now use generic formatter)
def format_step_not_found_error(
    step_name: str, workspace_context: str = None, available_steps: List[str] = None
) -> str:
    """Format step not found error messages using generic formatter."""
    return format_registry_error(
        "step_not_found",
        step_name=step_name,
        workspace_context=workspace_context,
        available_steps=available_steps,
    )


def format_registry_load_error(registry_path: str, error_details: str) -> str:
    """Format registry loading error messages using generic formatter."""
    return format_registry_error(
        "registry_load", registry_path=registry_path, error_details=error_details
    )


def format_validation_error(component_name: str, validation_issues: List[str]) -> str:
    """Format validation error messages using generic formatter."""
    return format_registry_error(
        "validation", component_name=component_name, validation_issues=validation_issues
    )

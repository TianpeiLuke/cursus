"""
Core base classes for the cursus framework.

This module provides the foundational base classes that are used throughout
the cursus framework for configuration, contracts, specifications, and builders.
"""

from typing import TYPE_CHECKING

# Always import enums first as they have no dependencies
from .enums import DependencyType, NodeType

# Import contract classes (no circular dependencies)
from .contract_base import ScriptContract, ValidationResult, ScriptAnalyzer

# Import hyperparameters (no circular dependencies)
from .hyperparameters_base import ModelHyperparameters

# Use lazy imports for classes that might have circular dependencies
if TYPE_CHECKING:
    from .config_base import BasePipelineConfig
    from .specification_base import DependencySpec, OutputSpec, StepSpecification
    from .builder_base import StepBuilderBase


def get_base_pipeline_config() -> type:
    """Lazy import for BasePipelineConfig to avoid circular imports."""
    from .config_base import BasePipelineConfig

    return BasePipelineConfig


def get_dependency_spec() -> type:
    """Lazy import for DependencySpec to avoid circular imports."""
    from .specification_base import DependencySpec

    return DependencySpec


def get_output_spec() -> type:
    """Lazy import for OutputSpec to avoid circular imports."""
    from .specification_base import OutputSpec

    return OutputSpec


def get_step_specification() -> type:
    """Lazy import for StepSpecification to avoid circular imports."""
    from .specification_base import StepSpecification

    return StepSpecification


def get_step_builder_base() -> type:
    """Lazy import for StepBuilderBase to avoid circular imports."""
    from .builder_base import StepBuilderBase

    return StepBuilderBase


# For backward compatibility, provide the classes via lazy loading
def __getattr__(name: str) -> type:
    """Provide lazy loading for backward compatibility."""
    if name == "BasePipelineConfig":
        return get_base_pipeline_config()
    elif name == "DependencySpec":
        return get_dependency_spec()
    elif name == "OutputSpec":
        return get_output_spec()
    elif name == "StepSpecification":
        return get_step_specification()
    elif name == "StepBuilderBase":
        return get_step_builder_base()
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    # Enums (always available)
    "DependencyType",
    "NodeType",
    # Contract classes (always available)
    "ScriptContract",
    "ValidationResult",
    "ScriptAnalyzer",
    # Hyperparameters (always available)
    "ModelHyperparameters",
    # Lazy-loaded classes (available via __getattr__)
    "BasePipelineConfig",
    "DependencySpec",
    "OutputSpec",
    "StepSpecification",
    "StepBuilderBase",
    # Lazy import functions
    "get_base_pipeline_config",
    "get_dependency_spec",
    "get_output_spec",
    "get_step_specification",
    "get_step_builder_base",
]

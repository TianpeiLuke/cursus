"""
Core base classes for the cursus framework.

This module provides the foundational base classes that are used throughout
the cursus framework for configuration, contracts, specifications, and builders.
"""

from typing import TYPE_CHECKING

# Always import enums first as they have no dependencies
from .enums import DependencyType, NodeType

# Import contract validation helpers (no circular dependencies).
# NOTE: the legacy ScriptContract data class was removed — the unified StepInterface
# (core/base/step_interface.py) is the single source of contract+spec data now.
from .contract_base import ValidationResult, ScriptAnalyzer

# Import the unified step interface model (replaces ScriptContract + StepSpecification)
from .step_interface import StepInterface

# Import hyperparameters (no circular dependencies)
from .hyperparameters_base import ModelHyperparameters

# Use lazy imports for classes that might have circular dependencies
if TYPE_CHECKING:
    from .config_base import BasePipelineConfig
    from .builder_base import StepBuilderBase


def get_base_pipeline_config() -> type:
    """Lazy import for BasePipelineConfig to avoid circular imports."""
    from .config_base import BasePipelineConfig

    return BasePipelineConfig


def get_step_builder_base() -> type:
    """Lazy import for StepBuilderBase to avoid circular imports."""
    from .builder_base import StepBuilderBase

    return StepBuilderBase


# For backward compatibility, provide the classes via lazy loading
def __getattr__(name: str) -> type:
    """Provide lazy loading for backward compatibility."""
    if name == "BasePipelineConfig":
        return get_base_pipeline_config()
    elif name == "StepBuilderBase":
        return get_step_builder_base()
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    # Enums (always available)
    "DependencyType",
    "NodeType",
    # Contract validation helpers (always available)
    "ValidationResult",
    "ScriptAnalyzer",
    # Unified step interface model
    "StepInterface",
    # Hyperparameters (always available)
    "ModelHyperparameters",
    # Lazy-loaded classes (available via __getattr__)
    "BasePipelineConfig",
    "StepBuilderBase",
    # Lazy import functions
    "get_base_pipeline_config",
    "get_step_builder_base",
]

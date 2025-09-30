"""
Pipeline compiler module.

This module provides high-level interfaces for compiling PipelineDAG structures
directly into executable SageMaker pipelines without requiring custom template classes.
"""

from .dag_compiler import compile_dag_to_pipeline, PipelineDAGCompiler
from ...step_catalog.adapters.config_resolver import StepConfigResolverAdapter as StepConfigResolver
from .validation import (
    ValidationResult,
    ResolutionPreview,
    ConversionReport,
    ValidationEngine,
)
from .name_generator import (
    generate_random_word,
    validate_pipeline_name,
    sanitize_pipeline_name,
    generate_pipeline_name,
)
from .exceptions import (
    PipelineAPIError,
    ConfigurationError,
    AmbiguityError,
    ValidationError,
    ResolutionError,
)


def _get_dynamic_pipeline_template() -> type:
    """Lazy import to avoid circular import issues."""
    from .dynamic_template import DynamicPipelineTemplate

    return DynamicPipelineTemplate


# Make DynamicPipelineTemplate available through lazy loading
def __getattr__(name: str) -> type:
    if name == "DynamicPipelineTemplate":
        return _get_dynamic_pipeline_template()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    # Main compilation functions
    "compile_dag_to_pipeline",
    # Compiler classes
    "PipelineDAGCompiler",
    "DynamicPipelineTemplate",  # Available through lazy loading
    "StepConfigResolver",
    # Validation and reporting
    "ValidationResult",
    "ResolutionPreview",
    "ConversionReport",
    "ValidationEngine",
    # Utilities
    "generate_random_word",
    "validate_pipeline_name",
    "sanitize_pipeline_name",
    "generate_pipeline_name",
    # Exceptions
    "PipelineAPIError",
    "ConfigurationError",
    "AmbiguityError",
    "ValidationError",
    "ResolutionError",
]

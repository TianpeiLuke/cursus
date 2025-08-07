"""
Pipeline compiler module.

This module provides high-level interfaces for compiling PipelineDAG structures
directly into executable SageMaker pipelines without requiring custom template classes.
"""

from .dag_compiler import compile_dag_to_pipeline, PipelineDAGCompiler
from .dynamic_template import DynamicPipelineTemplate
from .config_resolver import StepConfigResolver
from .validation import ValidationResult, ResolutionPreview, ConversionReport, ValidationEngine
from .name_generator import (
    generate_random_word,
    validate_pipeline_name,
    sanitize_pipeline_name,
    generate_pipeline_name
)
from .exceptions import (
    PipelineAPIError,
    ConfigurationError,
    AmbiguityError,
    ValidationError,
    ResolutionError
)

__all__ = [
    # Main compilation functions
    "compile_dag_to_pipeline",
    
    # Compiler classes
    "PipelineDAGCompiler",
    "DynamicPipelineTemplate",
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

__version__ = "1.0.0"

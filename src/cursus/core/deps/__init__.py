"""
Pipeline Dependencies module - Declarative dependency management for SageMaker pipelines.

This module provides declarative specifications and intelligent resolution
for pipeline step dependencies.
"""

from ..base import (
    DependencyType,
    NodeType,
    DependencySpec,
    OutputSpec,
    StepSpecification,
)
from .property_reference import PropertyReference
from .specification_registry import SpecificationRegistry
from .registry_manager import (
    RegistryManager,
    get_registry,
    get_pipeline_registry,
    get_default_registry,
    integrate_with_pipeline_builder,
    list_contexts,
    clear_context,
    get_context_stats,
)
from .dependency_resolver import (
    UnifiedDependencyResolver,
    DependencyResolutionError,
    create_dependency_resolver,
)
from .semantic_matcher import SemanticMatcher
from .factory import create_pipeline_components

__all__ = [
    # Core specification classes
    "DependencyType",
    "NodeType",
    "DependencySpec",
    "OutputSpec",
    "PropertyReference",
    "StepSpecification",
    # Registry management
    "SpecificationRegistry",
    "RegistryManager",
    "get_registry",
    "get_pipeline_registry",
    "get_default_registry",
    "integrate_with_pipeline_builder",
    "list_contexts",
    "clear_context",
    "get_context_stats",
    # Dependency resolution
    "UnifiedDependencyResolver",
    "DependencyResolutionError",
    "create_dependency_resolver",
    # Semantic matching
    "SemanticMatcher",
    # Factory functions
    "create_pipeline_components",
]

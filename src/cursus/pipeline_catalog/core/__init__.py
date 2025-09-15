"""
Pipeline Catalog Core Module

This module contains the core infrastructure for all pipelines in the catalog,
providing base classes and utilities that are shared across both MODS and
regular pipelines.

Key Components:
- BasePipeline: Abstract base class for all pipeline implementations
- Registry utilities: For pipeline metadata and catalog management
- Common utilities: Shared functionality across pipeline types
"""

from .base_pipeline import BasePipeline
from .catalog_registry import CatalogRegistry
from .connection_traverser import ConnectionTraverser
from .tag_discovery import TagBasedDiscovery
from .recommendation_engine import PipelineRecommendationEngine
from .registry_validator import RegistryValidator

__all__ = [
    "BasePipeline",
    "CatalogRegistry",
    "ConnectionTraverser",
    "TagBasedDiscovery",
    "PipelineRecommendationEngine",
    "RegistryValidator",
]

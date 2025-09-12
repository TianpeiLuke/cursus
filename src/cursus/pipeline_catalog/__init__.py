"""
Pipeline Catalog

Zettelkasten-inspired pipeline catalog with flat structure and connection-based discovery.
Provides atomic, independent pipeline implementations with enhanced metadata integration.
"""

from typing import Dict, List, Any

# Import key utilities
from .utils import (
    CatalogRegistry,
    ConnectionTraverser,
    TagBasedDiscovery,
    PipelineRecommendationEngine,
    RegistryValidator,
)

# Import pipeline discovery functions
from .pipelines import discover_pipelines, load_pipeline, get_registered_pipelines

from .mods_pipelines import (
    discover_mods_pipelines,
    load_mods_pipeline,
    get_registered_mods_pipelines,
)

# Import shared DAG utilities
from .shared_dags import DAGMetadata, validate_dag_metadata, get_all_shared_dags

__all__ = [
    # Utilities
    "CatalogRegistry",
    "ConnectionTraverser",
    "TagBasedDiscovery",
    "PipelineRecommendationEngine",
    "RegistryValidator",
    # Pipeline discovery
    "discover_pipelines",
    "load_pipeline",
    "get_registered_pipelines",
    "discover_mods_pipelines",
    "load_mods_pipeline",
    "get_registered_mods_pipelines",
    # DAG utilities
    "DAGMetadata",
    "validate_dag_metadata",
    "get_all_shared_dags",
]


def get_catalog_info() -> Dict[str, Any]:
    """
    Get information about the pipeline catalog.

    Returns:
        Dict containing catalog metadata and statistics
    """
    try:
        registry = CatalogRegistry()
        catalog_data = registry.load_catalog()

        return {
            "total_pipelines": catalog_data.get("metadata", {}).get(
                "total_pipelines", 0
            ),
            "frameworks": catalog_data.get("metadata", {}).get("frameworks", []),
            "complexity_levels": catalog_data.get("metadata", {}).get(
                "complexity_levels", []
            ),
            "last_updated": catalog_data.get("metadata", {}).get(
                "last_updated", "unknown"
            ),
            "standard_pipelines": len(get_registered_pipelines()),
            "mods_pipelines": len(get_registered_mods_pipelines()),
            "shared_dags": len(get_all_shared_dags()),
        }
    except Exception as e:
        return {
            "error": f"Failed to load catalog info: {e}",
            "standard_pipelines": len(get_registered_pipelines()),
            "mods_pipelines": len(get_registered_mods_pipelines()),
            "shared_dags": len(get_all_shared_dags()),
        }


def discover_all_pipelines() -> Dict[str, List[str]]:
    """
    Discover all available pipelines in the catalog.

    Returns:
        Dict with 'standard' and 'mods' pipeline lists
    """
    return {"standard": discover_pipelines(), "mods": discover_mods_pipelines()}

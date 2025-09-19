"""
Pipeline Catalog - Main Utilities Module

This module provides the main entry point for pipeline catalog utilities,
integrating all Zettelkasten-inspired functionality for pipeline discovery,
navigation, and management.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path

# Import all utility classes
from .core.catalog_registry import CatalogRegistry
from .core.connection_traverser import ConnectionTraverser
from .core.tag_discovery import TagBasedDiscovery
from .core.recommendation_engine import PipelineRecommendationEngine
from .core.registry_validator import RegistryValidator
from .shared_dags.registry_sync import DAGMetadataRegistrySync
from .shared_dags.enhanced_metadata import EnhancedDAGMetadata


class PipelineCatalogManager:
    """
    Main manager class for pipeline catalog operations.

    This class provides a unified interface for all pipeline catalog functionality,
    following Zettelkasten knowledge management principles.
    """

    def __init__(self, registry_path: Optional[str] = None):
        """
        Initialize the pipeline catalog manager.

        Args:
            registry_path: Path to the catalog registry JSON file.
                          If None, uses default location.
        """
        if registry_path is None:
            registry_path = str(Path(__file__).parent / "catalog_index.json")

        self.registry_path = registry_path

        # Initialize core components
        self.registry = CatalogRegistry(registry_path)
        self.traverser = ConnectionTraverser(self.registry)
        self.discovery = TagBasedDiscovery(self.registry)
        self.recommender = PipelineRecommendationEngine(
            self.registry, self.traverser, self.discovery
        )
        self.validator = RegistryValidator(self.registry)
        self.sync = DAGMetadataRegistrySync(registry_path)

    def discover_pipelines(self, **kwargs) -> List[str]:
        """
        Discover pipelines based on various criteria using step catalog with fallback.

        Args:
            **kwargs: Search criteria (framework, complexity, tags, etc.)

        Returns:
            List of pipeline IDs matching the criteria
        """
        # Try using step catalog first for enhanced discovery
        try:
            return self._discover_pipelines_with_catalog(**kwargs)
        except ImportError:
            # Step catalog not available, use legacy discovery
            pass
        except Exception as e:
            # Step catalog discovery failed, log and fall back
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Step catalog discovery failed: {e}, falling back to legacy")

        # FALLBACK METHOD: Legacy pipeline discovery
        return self._discover_pipelines_legacy(**kwargs)

    def _discover_pipelines_with_catalog(self, **kwargs) -> List[str]:
        """Discover pipelines using step catalog."""
        from ..step_catalog import StepCatalog
        
        # PORTABLE: Use package-only discovery for pipeline discovery
        try:
            catalog = StepCatalog(workspace_dirs=None)
        except Exception:
            # If step catalog initialization fails, fall back to legacy
            return self._discover_pipelines_legacy(**kwargs)
        
        # Use step catalog to discover relevant steps/pipelines
        if "framework" in kwargs:
            # Use catalog's framework detection
            all_steps = catalog.list_available_steps()
            framework_steps = []
            for step_name in all_steps:
                framework = catalog.detect_framework(step_name)
                if framework and framework.lower() == kwargs["framework"].lower():
                    framework_steps.append(step_name)
            return framework_steps
        elif "tags" in kwargs:
            # Use catalog's search functionality for tag-like queries
            search_results = catalog.search_steps(" ".join(kwargs["tags"]))
            return [result.step_name for result in search_results]
        elif "use_case" in kwargs:
            # Use catalog's search functionality for use case queries
            search_results = catalog.search_steps(kwargs["use_case"])
            return [result.step_name for result in search_results]
        else:
            # Return all available steps from catalog
            return catalog.list_available_steps()

    def _discover_pipelines_legacy(self, **kwargs) -> List[str]:
        """Legacy pipeline discovery method."""
        if "framework" in kwargs:
            return self.discovery.find_by_framework(kwargs["framework"])
        elif "complexity" in kwargs:
            return self.discovery.find_by_complexity(kwargs["complexity"])
        elif "tags" in kwargs:
            return self.discovery.find_by_tags(kwargs["tags"])
        elif "use_case" in kwargs:
            # Use text search for use case since find_by_use_case doesn't exist
            search_results = self.discovery.search_by_text(kwargs["use_case"])
            return [result[0] for result in search_results]  # Extract pipeline IDs from (id, score) tuples
        else:
            return self.registry.get_all_pipelines()

    def get_pipeline_connections(self, pipeline_id: str) -> Dict[str, List[str]]:
        """
        Get all connections for a pipeline.

        Args:
            pipeline_id: The pipeline to get connections for

        Returns:
            Dictionary of connection types and their targets
        """
        # get_all_connections returns Dict[str, List[PipelineConnection]]
        # We need to convert it to Dict[str, List[str]] for backwards compatibility
        connections = self.traverser.get_all_connections(pipeline_id)
        result = {}
        for conn_type, conn_list in connections.items():
            result[conn_type] = [conn.target_id for conn in conn_list]
        return result

    def find_path(self, source: str, target: str) -> Optional[List[str]]:
        """
        Find a path between two pipelines through connections.

        Args:
            source: Source pipeline ID
            target: Target pipeline ID

        Returns:
            List of pipeline IDs forming the path, or None if no path exists
        """
        return self.traverser.find_shortest_path(source, target)

    def get_recommendations(self, use_case: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Get pipeline recommendations for a use case.

        Args:
            use_case: The use case to get recommendations for
            **kwargs: Additional criteria for filtering

        Returns:
            List of recommendation results with scores
        """
        return self.recommender.recommend_for_use_case(use_case, **kwargs)

    def validate_registry(self) -> Dict[str, Any]:
        """
        Validate the registry integrity.

        Returns:
            Validation report with any issues found
        """
        report = self.validator.generate_validation_report()
        return {
            "is_valid": report.is_valid,
            "total_issues": report.total_issues,
            "issues_by_severity": report.issues_by_severity,
            "issues_by_category": report.issues_by_category,
            "all_issues": [issue.model_dump() for issue in report.all_issues],
        }

    def sync_pipeline(self, metadata: EnhancedDAGMetadata, filename: str) -> bool:
        """
        Sync a pipeline's metadata to the registry.

        Args:
            metadata: Enhanced DAG metadata
            filename: Pipeline filename

        Returns:
            True if sync was successful
        """
        try:
            self.sync.sync_metadata_to_registry(metadata, filename)
            return True
        except Exception:
            return False

    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Dictionary with registry statistics
        """
        return self.sync.get_registry_statistics()


# Convenience functions for direct access
def create_catalog_manager(
    registry_path: Optional[str] = None,
) -> PipelineCatalogManager:
    """Create a new pipeline catalog manager instance."""
    return PipelineCatalogManager(registry_path)


def discover_by_framework(
    framework: str, registry_path: Optional[str] = None
) -> List[str]:
    """Quick discovery by framework."""
    manager = create_catalog_manager(registry_path)
    return manager.discover_pipelines(framework=framework)


def discover_by_tags(tags: List[str], registry_path: Optional[str] = None) -> List[str]:
    """Quick discovery by tags."""
    manager = create_catalog_manager(registry_path)
    return manager.discover_pipelines(tags=tags)


def get_pipeline_alternatives(
    pipeline_id: str, registry_path: Optional[str] = None
) -> List[str]:
    """Get alternative pipelines for a given pipeline."""
    manager = create_catalog_manager(registry_path)
    connections = manager.get_pipeline_connections(pipeline_id)
    return connections.get("alternatives", [])


__all__ = [
    "PipelineCatalogManager",
    "create_catalog_manager",
    "discover_by_framework",
    "discover_by_tags",
    "get_pipeline_alternatives",
]

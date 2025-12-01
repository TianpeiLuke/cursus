"""
Pipeline Factory Module

Knowledge-driven factory for dynamic pipeline creation using DAGAutoDiscovery.
Eliminates redundant pipeline implementations by generating classes on-the-fly.

Features:
- Dynamic class generation from DAG definitions
- Multiple creation interfaces (direct, search, criteria)
- Caching for performance
- BasePipeline integration
- Registry-enriched metadata
"""

import logging
from typing import Dict, List, Optional, Type, Any
from pathlib import Path

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession

from .base_pipeline import BasePipeline
from .dag_discovery import DAGAutoDiscovery, DAGInfo
from ..shared_dags import DAGMetadata
from ..shared_dags.enhanced_metadata import EnhancedDAGMetadata
from ...api.dag.base_dag import PipelineDAG

logger = logging.getLogger(__name__)


class PipelineFactory:
    """
    Factory for dynamic pipeline creation using knowledge-driven approach.

    Instead of hardcoding pipeline classes, this factory:
    1. Discovers available DAGs via DAGAutoDiscovery
    2. Generates pipeline classes dynamically
    3. Instantiates pipelines with proper configuration
    4. Caches generated classes for reuse

    Usage:
        # Direct creation by ID
        factory = PipelineFactory()
        pipeline = factory.create("xgboost_complete_e2e", config_path="config.json")

        # Search-based creation
        pipeline = factory.create_by_search("xgboost end-to-end training")

        # Criteria-based creation
        pipeline = factory.create_by_criteria(
            framework="xgboost",
            features=["training", "calibration"]
        )
    """

    def __init__(
        self,
        package_root: Optional[Path] = None,
        workspace_dirs: Optional[List[Path]] = None,
        registry_path: Optional[str] = None,
        enable_caching: bool = True,
    ):
        """
        Initialize pipeline factory.

        Args:
            package_root: Root of package (for DAG discovery)
            workspace_dirs: Workspace directories to scan
            registry_path: Path to catalog registry
            enable_caching: Whether to cache generated classes
        """
        # Initialize discovery
        self.discovery = DAGAutoDiscovery(
            package_root=package_root,
            workspace_dirs=workspace_dirs,
            registry_path=registry_path,
        )

        # Class cache
        self.enable_caching = enable_caching
        self._class_cache: Dict[str, Type[BasePipeline]] = {}

        # Run discovery
        self.discovery.discover_all_dags()
        logger.info(
            f"PipelineFactory initialized with {len(self.discovery.list_available_dags())} DAGs"
        )

    def create(
        self,
        dag_id: str,
        config_path: Optional[str] = None,
        sagemaker_session: Optional[PipelineSession] = None,
        execution_role: Optional[str] = None,
        enable_mods: bool = True,
        validate: bool = True,
        pipeline_parameters: Optional[list] = None,
        **kwargs,
    ) -> BasePipeline:
        """
        Create pipeline directly by DAG ID.

        Args:
            dag_id: DAG identifier (e.g., "xgboost_complete_e2e")
            config_path: Path to configuration file
            sagemaker_session: SageMaker session
            execution_role: IAM execution role
            enable_mods: Enable MODS features
            validate: Validate DAG before compilation
            pipeline_parameters: Custom pipeline parameters
            **kwargs: Additional arguments for pipeline

        Returns:
            BasePipeline instance

        Raises:
            ValueError: If DAG not found
        """
        # Load DAG info
        dag_info = self.discovery.load_dag_info(dag_id)
        if dag_info is None:
            available = self.discovery.list_available_dags()
            raise ValueError(f"DAG '{dag_id}' not found. Available DAGs: {available}")

        # Get or generate pipeline class
        pipeline_class = self._get_or_create_pipeline_class(dag_info)

        # Instantiate pipeline
        pipeline = pipeline_class(
            config_path=config_path,
            sagemaker_session=sagemaker_session,
            execution_role=execution_role,
            enable_mods=enable_mods,
            validate=validate,
            pipeline_parameters=pipeline_parameters,
            **kwargs,
        )

        logger.info(f"Created pipeline instance for DAG '{dag_id}'")
        return pipeline

    def create_by_search(
        self, query: str, config_path: Optional[str] = None, **kwargs
    ) -> BasePipeline:
        """
        Create pipeline by natural language search.

        Searches DAG descriptions, features, and framework for matches.

        Args:
            query: Search query (e.g., "xgboost training calibration")
            config_path: Path to configuration file
            **kwargs: Additional arguments for create()

        Returns:
            BasePipeline instance

        Raises:
            ValueError: If no matching DAG found or multiple matches
        """
        # Tokenize query
        tokens = query.lower().split()

        # Score each DAG
        scores = {}
        for dag_id, dag_info in self.discovery._dag_cache.items():
            score = 0

            # Load metadata if needed
            if dag_info.metadata is None:
                dag_info.load_functions()

            # Score framework match
            if dag_info.framework.lower() in tokens:
                score += 10

            # Score complexity match
            if dag_info.complexity.lower() in tokens:
                score += 5

            # Score feature matches
            for feature in dag_info.features:
                if feature.lower() in tokens:
                    score += 3

            # Score DAG ID tokens
            dag_id_tokens = dag_id.replace("_", " ").split()
            for token in tokens:
                if token in dag_id_tokens:
                    score += 2

            if score > 0:
                scores[dag_id] = score

        if not scores:
            raise ValueError(f"No DAGs match query: '{query}'")

        # Get best match
        best_dag_id = max(scores.items(), key=lambda x: x[1])[0]
        best_score = scores[best_dag_id]

        # Check for ambiguous matches
        top_matches = [k for k, v in scores.items() if v == best_score]
        if len(top_matches) > 1:
            raise ValueError(
                f"Ambiguous query '{query}' matches multiple DAGs: {top_matches}. "
                "Please be more specific or use create() with exact dag_id."
            )

        logger.info(
            f"Search query '{query}' matched DAG '{best_dag_id}' (score: {best_score})"
        )

        # Create pipeline with matched DAG
        return self.create(best_dag_id, config_path=config_path, **kwargs)

    def create_by_criteria(
        self,
        framework: Optional[str] = None,
        complexity: Optional[str] = None,
        features: Optional[List[str]] = None,
        config_path: Optional[str] = None,
        **kwargs,
    ) -> BasePipeline:
        """
        Create pipeline by structured criteria.

        Args:
            framework: Framework filter (xgboost, pytorch, etc.)
            complexity: Complexity filter (simple, standard, comprehensive)
            features: Required features (all must be present)
            config_path: Path to configuration file
            **kwargs: Additional arguments for create()

        Returns:
            BasePipeline instance

        Raises:
            ValueError: If no matching DAG found or multiple matches
        """
        # Search with criteria
        matches = self.discovery.search_dags(
            framework=framework, complexity=complexity, features=features
        )

        if not matches:
            raise ValueError(
                f"No DAGs match criteria: framework={framework}, "
                f"complexity={complexity}, features={features}"
            )

        if len(matches) > 1:
            match_ids = list(matches.keys())
            raise ValueError(
                f"Multiple DAGs match criteria: {match_ids}. "
                "Please refine criteria or use create() with exact dag_id."
            )

        # Get single match
        dag_id = list(matches.keys())[0]
        logger.info(f"Criteria matched DAG '{dag_id}'")

        # Create pipeline
        return self.create(dag_id, config_path=config_path, **kwargs)

    def _get_or_create_pipeline_class(self, dag_info: DAGInfo) -> Type[BasePipeline]:
        """
        Get cached or generate new pipeline class.

        Args:
            dag_info: DAG information

        Returns:
            Generated pipeline class
        """
        # Check cache
        if self.enable_caching and dag_info.dag_id in self._class_cache:
            logger.debug(f"Using cached class for '{dag_info.dag_id}'")
            return self._class_cache[dag_info.dag_id]

        # Generate new class
        pipeline_class = self._generate_pipeline_class(dag_info)

        # Cache if enabled
        if self.enable_caching:
            self._class_cache[dag_info.dag_id] = pipeline_class
            logger.debug(f"Cached generated class for '{dag_info.dag_id}'")

        return pipeline_class

    def _generate_pipeline_class(self, dag_info: DAGInfo) -> Type[BasePipeline]:
        """
        Dynamically generate pipeline class from DAG info.

        Args:
            dag_info: DAG information

        Returns:
            Generated pipeline class inheriting from BasePipeline
        """
        # Load functions if not already loaded
        if dag_info.create_function is None:
            dag_info.load_functions()

        # Create class name
        class_name = self._dag_id_to_class_name(dag_info.dag_id)

        # Define create_dag method
        def create_dag(self) -> PipelineDAG:
            """Create the pipeline DAG."""
            return dag_info.create_function()

        # Define get_enhanced_dag_metadata method
        def get_enhanced_dag_metadata(self) -> EnhancedDAGMetadata:
            """Get enhanced DAG metadata."""
            # Get basic metadata
            if dag_info.metadata_function:
                basic_metadata = dag_info.metadata_function()
            else:
                # Fallback metadata if get_dag_metadata doesn't exist
                basic_metadata = DAGMetadata(
                    description=f"Pipeline for {dag_info.dag_id}",
                    complexity=dag_info.complexity,
                    features=dag_info.features,
                    framework=dag_info.framework,
                    node_count=dag_info.node_count,
                    edge_count=dag_info.edge_count,
                )

            # Convert to enhanced metadata
            from ..shared_dags.enhanced_metadata import create_enhanced_metadata

            enhanced = create_enhanced_metadata(
                atomic_id=dag_info.dag_id,
                title=f"{dag_info.framework.upper()} {dag_info.complexity.capitalize()} Pipeline",
                description=basic_metadata.description,
                framework=dag_info.framework,
                complexity=dag_info.complexity,
                features=dag_info.features,
                tags=[dag_info.framework, dag_info.complexity] + dag_info.features,
                node_count=dag_info.node_count,
                edge_count=dag_info.edge_count,
            )

            return enhanced

        # Create class dynamically
        pipeline_class = type(
            class_name,
            (BasePipeline,),
            {
                "create_dag": create_dag,
                "get_enhanced_dag_metadata": get_enhanced_dag_metadata,
                "__module__": __name__,
                "__doc__": f"Dynamically generated pipeline class for {dag_info.dag_id}",
                "_dag_info": dag_info,  # Store reference to DAG info
            },
        )

        logger.info(
            f"Generated pipeline class '{class_name}' for DAG '{dag_info.dag_id}'"
        )
        return pipeline_class

    @staticmethod
    def _dag_id_to_class_name(dag_id: str) -> str:
        """
        Convert DAG ID to class name.

        Args:
            dag_id: DAG identifier (e.g., "xgboost_complete_e2e")

        Returns:
            Class name (e.g., "XgboostCompleteE2EPipeline")
        """
        # Split by underscore and capitalize each part
        parts = dag_id.split("_")
        class_name = "".join(word.capitalize() for word in parts)
        return f"{class_name}Pipeline"

    def list_available_pipelines(self) -> List[Dict[str, Any]]:
        """
        List all available pipelines with metadata.

        Returns:
            List of pipeline information dicts
        """
        pipelines = []

        for dag_id in self.discovery.list_available_dags():
            dag_info = self.discovery.load_dag_info(dag_id)
            if dag_info:
                # Load metadata if not already loaded
                if dag_info.metadata is None:
                    dag_info.load_functions()

                pipeline_info = {
                    "dag_id": dag_id,
                    "class_name": self._dag_id_to_class_name(dag_id),
                    "framework": dag_info.framework,
                    "complexity": dag_info.complexity,
                    "features": dag_info.features,
                    "node_count": dag_info.node_count,
                    "edge_count": dag_info.edge_count,
                    "workspace": dag_info.workspace_id,
                }

                if dag_info.metadata:
                    pipeline_info["description"] = dag_info.metadata.description

                pipelines.append(pipeline_info)

        return pipelines

    def get_pipeline_info(self, dag_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific pipeline.

        Args:
            dag_id: DAG identifier

        Returns:
            Pipeline information dict

        Raises:
            ValueError: If DAG not found
        """
        dag_info = self.discovery.load_dag_info(dag_id)
        if dag_info is None:
            raise ValueError(f"DAG '{dag_id}' not found")

        # Load metadata if needed
        if dag_info.metadata is None:
            dag_info.load_functions()

        info = {
            "dag_id": dag_id,
            "dag_name": dag_info.dag_name,
            "class_name": self._dag_id_to_class_name(dag_id),
            "file_path": str(dag_info.dag_path),
            "workspace": dag_info.workspace_id,
            "framework": dag_info.framework,
            "complexity": dag_info.complexity,
            "features": dag_info.features,
            "node_count": dag_info.node_count,
            "edge_count": dag_info.edge_count,
        }

        if dag_info.metadata:
            info["description"] = dag_info.metadata.description
            info["full_metadata"] = dag_info.metadata.to_dict()

        return info

    def clear_cache(self) -> None:
        """Clear the pipeline class cache."""
        self._class_cache.clear()
        logger.info("Cleared pipeline class cache")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache statistics
        """
        return {
            "caching_enabled": self.enable_caching,
            "cached_classes": len(self._class_cache),
            "cache_keys": list(self._class_cache.keys()) if self.enable_caching else [],
        }

    def get_discovery_stats(self) -> Dict[str, Any]:
        """
        Get DAG discovery statistics.

        Returns:
            Dict with discovery statistics
        """
        return self.discovery.get_discovery_stats()

"""
Registry Synchronization Infrastructure

This module provides the infrastructure for synchronizing DAGMetadata with
the Zettelkasten registry system, enabling bidirectional data flow between
pipeline metadata and the connection registry.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from .enhanced_metadata import (
    EnhancedDAGMetadata,
    ZettelkastenMetadata,
    ComplexityLevel,
    PipelineFramework,
)

logger = logging.getLogger(__name__)


class RegistryValidationError(Exception):
    """Exception raised when registry validation fails."""

    pass


class DAGMetadataRegistrySync:
    """
    Synchronize DAGMetadata with Zettelkasten registry.

    Provides bidirectional synchronization between enhanced DAG metadata
    and the JSON-based connection registry, ensuring consistency and
    enabling the Zettelkasten knowledge management approach.
    """

    def __init__(self, registry_path: str = "catalog_index.json"):
        """
        Initialize registry synchronization.

        Args:
            registry_path: Path to the registry JSON file
        """
        self.registry_path = Path(registry_path)
        self._ensure_registry_exists()

    def _ensure_registry_exists(self) -> None:
        """Ensure the registry file exists with proper structure."""
        if not self.registry_path.exists():
            logger.info(f"Creating new registry at {self.registry_path}")
            self._create_empty_registry()

    def _create_empty_registry(self) -> None:
        """Create an empty registry with proper schema."""
        empty_registry = {
            "version": "1.0",
            "description": "Pipeline catalog connection registry - Zettelkasten-inspired knowledge network for independent pipelines",
            "metadata": {
                "total_pipelines": 0,
                "frameworks": [],
                "complexity_levels": [],
                "last_updated": datetime.now().isoformat(),
                "connection_types": ["alternatives", "related", "used_in"],
            },
            "nodes": {},
            "connection_graph_metadata": {
                "total_connections": 0,
                "connection_density": 0.0,
                "independent_pipelines": 0,
                "composition_opportunities": 0,
                "alternative_groups": 0,
                "isolated_nodes": [],
            },
            "tag_index": {
                "framework_tags": {},
                "task_tags": {},
                "complexity_tags": {},
                "independence_tags": {},
            },
        }

        self._save_registry(empty_registry)

    def load_registry(self) -> Dict[str, Any]:
        """
        Load the connection registry from JSON.

        Returns:
            Dict containing the complete registry data

        Raises:
            RegistryValidationError: If registry is corrupted or invalid
        """
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                registry = json.load(f)

            # Validate basic structure
            self._validate_registry_structure(registry)
            return registry

        except FileNotFoundError:
            logger.warning(
                f"Registry file not found at {self.registry_path}, creating new one"
            )
            self._create_empty_registry()
            return self.load_registry()
        except json.JSONDecodeError as e:
            raise RegistryValidationError(f"Invalid JSON in registry file: {e}")
        except Exception as e:
            raise RegistryValidationError(f"Failed to load registry: {e}")

    def _save_registry(self, registry: Dict[str, Any]) -> None:
        """
        Save the connection registry to JSON.

        Args:
            registry: Complete registry data to save
        """
        try:
            # Ensure parent directory exists
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)

            # Save with pretty formatting
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(registry, f, indent=2, ensure_ascii=False)

            logger.debug(f"Registry saved to {self.registry_path}")

        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
            raise

    def _validate_registry_structure(self, registry: Dict[str, Any]) -> None:
        """
        Validate registry structure and required fields.

        Args:
            registry: Registry data to validate

        Raises:
            RegistryValidationError: If structure is invalid
        """
        required_fields = ["version", "metadata", "nodes"]
        for field in required_fields:
            if field not in registry:
                raise RegistryValidationError(f"Missing required field: {field}")

        # Validate metadata structure
        metadata = registry["metadata"]
        required_metadata_fields = [
            "total_pipelines",
            "frameworks",
            "complexity_levels",
            "last_updated",
        ]
        for field in required_metadata_fields:
            if field not in metadata:
                raise RegistryValidationError(
                    f"Missing required metadata field: {field}"
                )

        # Validate nodes structure
        nodes = registry["nodes"]
        if not isinstance(nodes, dict):
            raise RegistryValidationError("Nodes must be a dictionary")

        # Validate individual nodes
        for node_id, node_data in nodes.items():
            self._validate_node_structure(node_id, node_data)

    def _validate_node_structure(self, node_id: str, node_data: Dict[str, Any]) -> None:
        """
        Validate individual node structure.

        Args:
            node_id: Node identifier
            node_data: Node data to validate

        Raises:
            RegistryValidationError: If node structure is invalid
        """
        required_node_fields = [
            "id",
            "title",
            "description",
            "atomic_properties",
            "zettelkasten_metadata",
        ]
        for field in required_node_fields:
            if field not in node_data:
                raise RegistryValidationError(
                    f"Node {node_id} missing required field: {field}"
                )

        # Validate atomic properties - match catalog_index.json structure
        atomic_props = node_data["atomic_properties"]
        required_atomic_fields = [
            "single_responsibility",
            "independence_level",
            "node_count",
            "edge_count",
        ]
        for field in required_atomic_fields:
            if field not in atomic_props:
                raise RegistryValidationError(
                    f"Node {node_id} missing atomic property: {field}"
                )

    def sync_metadata_to_registry(
        self, dag_metadata: EnhancedDAGMetadata, pipeline_file_path: str
    ) -> None:
        """
        Sync DAG metadata to registry entry.

        Args:
            dag_metadata: Enhanced DAG metadata to sync
            pipeline_file_path: Path to the pipeline file
        """
        try:
            # Load existing registry
            registry = self.load_registry()

            # Convert metadata to registry node
            node = dag_metadata.to_registry_node()
            node["file"] = pipeline_file_path
            node["last_updated"] = datetime.now().isoformat()

            # Update registry
            if "nodes" not in registry:
                registry["nodes"] = {}

            atomic_id = dag_metadata.zettelkasten_metadata.atomic_id
            registry["nodes"][atomic_id] = node

            # Update registry metadata
            self._update_registry_metadata(registry)

            # Update tag index
            self._update_tag_index(
                registry, atomic_id, dag_metadata.zettelkasten_metadata
            )

            # Save registry
            self._save_registry(registry)

            logger.info(f"Synced metadata for {atomic_id} to registry")

        except Exception as e:
            logger.error(f"Failed to sync metadata to registry: {e}")
            raise

    def sync_registry_to_metadata(
        self, pipeline_id: str
    ) -> Optional[EnhancedDAGMetadata]:
        """
        Sync registry entry back to DAG metadata.

        Args:
            pipeline_id: Pipeline identifier to sync

        Returns:
            EnhancedDAGMetadata instance or None if not found
        """
        try:
            registry = self.load_registry()

            if "nodes" not in registry or pipeline_id not in registry["nodes"]:
                logger.warning(f"Pipeline {pipeline_id} not found in registry")
                return None

            node = registry["nodes"][pipeline_id]

            # Extract core metadata
            framework = PipelineFramework(node["zettelkasten_metadata"]["framework"])
            complexity = ComplexityLevel(node["zettelkasten_metadata"]["complexity"])

            # Extract Zettelkasten metadata
            zettelkasten_metadata = self._extract_zettelkasten_metadata_from_node(node)

            # Create enhanced metadata
            enhanced_metadata = EnhancedDAGMetadata(
                description=node["description"],
                complexity=complexity,
                features=zettelkasten_metadata.task_tags,
                framework=framework,
                node_count=1,  # Default to 1 since we don't have actual DAG info
                edge_count=0,  # Default to 0 for edges
                zettelkasten_metadata=zettelkasten_metadata,
            )

            logger.debug(f"Synced registry data for {pipeline_id} to metadata")
            return enhanced_metadata

        except Exception as e:
            logger.error(f"Failed to sync registry to metadata for {pipeline_id}: {e}")
            return None

    def _extract_zettelkasten_metadata_from_node(
        self, node: Dict[str, Any]
    ) -> ZettelkastenMetadata:
        """
        Extract ZettelkastenMetadata from registry node.

        Args:
            node: Registry node data

        Returns:
            ZettelkastenMetadata instance
        """
        # Build manual connections from registry connections
        manual_connections = {}
        curated_connections = {}

        for conn_type, connections in node.get("connections", {}).items():
            if connections:  # Only add if there are connections
                manual_connections[conn_type] = [conn["id"] for conn in connections]
                for conn in connections:
                    curated_connections[conn["id"]] = conn["annotation"]

        # Extract atomic properties
        atomic_props = node.get("atomic_properties", {})
        zettel_meta = node.get("zettelkasten_metadata", {})
        multi_tags = node.get("multi_dimensional_tags", {})

        return ZettelkastenMetadata(
            atomic_id=node["id"],
            title=node.get("title", ""),
            single_responsibility=atomic_props.get("single_responsibility", ""),
            # Atomicity metadata
            input_interface=atomic_props.get("input_interface", []),
            output_interface=atomic_props.get("output_interface", []),
            side_effects=atomic_props.get("side_effects", "none"),
            independence_level=atomic_props.get(
                "independence_level", "fully_self_contained"
            ),
            node_count=atomic_props.get("node_count", 1),
            edge_count=atomic_props.get("edge_count", 0),
            # Core metadata (matches catalog structure)
            framework=zettel_meta.get("framework", ""),
            complexity=zettel_meta.get("complexity", ""),
            use_case=zettel_meta.get("use_case", ""),
            features=zettel_meta.get("features", []),
            mods_compatible=zettel_meta.get("mods_compatible", False),
            # File tracking
            source_file=node.get("source_file", ""),
            migration_source=node.get("migration_source", ""),
            created_date=node.get("created_date", ""),
            priority=node.get("priority", "standard"),
            # Connections
            manual_connections=manual_connections,
            curated_connections=curated_connections,
            # Tags
            framework_tags=multi_tags.get("framework_tags", []),
            task_tags=multi_tags.get("task_tags", []),
            complexity_tags=multi_tags.get("complexity_tags", []),
            domain_tags=multi_tags.get("domain_tags", []),
            pattern_tags=multi_tags.get("pattern_tags", []),
            integration_tags=multi_tags.get("integration_tags", []),
            quality_tags=multi_tags.get("quality_tags", []),
            data_tags=multi_tags.get("data_tags", []),
            # Discovery metadata
            creation_context=zettel_meta.get("creation_context", ""),
            usage_frequency=zettel_meta.get("usage_frequency", "unknown"),
            stability=zettel_meta.get("stability", "experimental"),
            maintenance_burden=zettel_meta.get("maintenance_burden", "unknown"),
            estimated_runtime=zettel_meta.get("estimated_runtime", "unknown"),
            resource_requirements=zettel_meta.get("resource_requirements", "unknown"),
            use_cases=zettel_meta.get("use_cases", []),
            skill_level=zettel_meta.get("skill_level", "unknown"),
        )

    def validate_consistency(
        self, dag_metadata: EnhancedDAGMetadata, pipeline_id: str
    ) -> List[str]:
        """
        Validate consistency between DAG metadata and registry.

        Args:
            dag_metadata: DAG metadata to validate
            pipeline_id: Pipeline identifier

        Returns:
            List of consistency errors (empty if consistent)
        """
        errors = []

        try:
            registry_metadata = self.sync_registry_to_metadata(pipeline_id)

            if registry_metadata is None:
                errors.append(f"Pipeline {pipeline_id} not found in registry")
                return errors

            # Check core metadata consistency
            if dag_metadata.description != registry_metadata.description:
                errors.append(
                    f"Description mismatch: DAG='{dag_metadata.description}' vs Registry='{registry_metadata.description}'"
                )

            if dag_metadata.complexity != registry_metadata.complexity:
                errors.append(
                    f"Complexity mismatch: DAG='{dag_metadata.complexity}' vs Registry='{registry_metadata.complexity}'"
                )

            if dag_metadata.framework != registry_metadata.framework:
                errors.append(
                    f"Framework mismatch: DAG='{dag_metadata.framework}' vs Registry='{registry_metadata.framework}'"
                )

            # Check Zettelkasten metadata consistency
            dag_zm = dag_metadata.zettelkasten_metadata
            reg_zm = registry_metadata.zettelkasten_metadata

            if dag_zm.atomic_id != reg_zm.atomic_id:
                errors.append(
                    f"Atomic ID mismatch: DAG='{dag_zm.atomic_id}' vs Registry='{reg_zm.atomic_id}'"
                )

            if dag_zm.single_responsibility != reg_zm.single_responsibility:
                errors.append(f"Single responsibility mismatch")

            # Check tag consistency
            for tag_category in ["framework_tags", "task_tags", "complexity_tags"]:
                dag_tags = set(getattr(dag_zm, tag_category))
                reg_tags = set(getattr(reg_zm, tag_category))
                if dag_tags != reg_tags:
                    errors.append(
                        f"{tag_category} mismatch: DAG={dag_tags} vs Registry={reg_tags}"
                    )

        except Exception as e:
            errors.append(f"Validation failed with error: {e}")

        return errors

    def _update_registry_metadata(self, registry: Dict[str, Any]) -> None:
        """
        Update registry-level metadata.

        Args:
            registry: Registry data to update
        """
        if "metadata" not in registry:
            registry["metadata"] = {}

        nodes = registry.get("nodes", {})
        registry["metadata"]["total_pipelines"] = len(nodes)
        registry["metadata"]["last_updated"] = datetime.now().isoformat()

        # Update framework and complexity statistics
        frameworks = set()
        complexities = set()
        total_connections = 0
        isolated_nodes = []

        for node_id, node in nodes.items():
            frameworks.add(node["zettelkasten_metadata"]["framework"])
            complexities.add(node["zettelkasten_metadata"]["complexity"])

            # Count connections
            connections = node.get("connections", {})
            node_connections = sum(len(conn_list) for conn_list in connections.values())
            total_connections += node_connections

            if node_connections == 0:
                isolated_nodes.append(node_id)

        registry["metadata"]["frameworks"] = list(frameworks)
        registry["metadata"]["complexity_levels"] = list(complexities)

        # Update connection graph metadata
        if "connection_graph_metadata" not in registry:
            registry["connection_graph_metadata"] = {}

        graph_meta = registry["connection_graph_metadata"]
        graph_meta["total_connections"] = total_connections
        graph_meta["independent_pipelines"] = len(nodes)
        graph_meta["isolated_nodes"] = isolated_nodes

        # Calculate connection density
        if len(nodes) > 1:
            max_possible_connections = len(nodes) * (len(nodes) - 1)
            graph_meta["connection_density"] = (
                total_connections / max_possible_connections
                if max_possible_connections > 0
                else 0.0
            )
        else:
            graph_meta["connection_density"] = 0.0

    def _update_tag_index(
        self,
        registry: Dict[str, Any],
        pipeline_id: str,
        zettelkasten_metadata: ZettelkastenMetadata,
    ) -> None:
        """
        Update the tag index with pipeline tags.

        Args:
            registry: Registry data to update
            pipeline_id: Pipeline identifier
            zettelkasten_metadata: Zettelkasten metadata containing tags
        """
        if "tag_index" not in registry:
            registry["tag_index"] = {}

        tag_index = registry["tag_index"]

        # Update each tag category
        tag_categories = {
            "framework_tags": zettelkasten_metadata.framework_tags,
            "task_tags": zettelkasten_metadata.task_tags,
            "complexity_tags": zettelkasten_metadata.complexity_tags,
            "domain_tags": zettelkasten_metadata.domain_tags,
            "pattern_tags": zettelkasten_metadata.pattern_tags,
            "integration_tags": zettelkasten_metadata.integration_tags,
            "quality_tags": zettelkasten_metadata.quality_tags,
            "data_tags": zettelkasten_metadata.data_tags,
        }

        for category, tags in tag_categories.items():
            if category not in tag_index:
                tag_index[category] = {}

            for tag in tags:
                if tag not in tag_index[category]:
                    tag_index[category][tag] = []

                if pipeline_id not in tag_index[category][tag]:
                    tag_index[category][tag].append(pipeline_id)

    def remove_pipeline_from_registry(self, pipeline_id: str) -> bool:
        """
        Remove a pipeline from the registry.

        Args:
            pipeline_id: Pipeline identifier to remove

        Returns:
            True if removed successfully, False if not found
        """
        try:
            registry = self.load_registry()

            if "nodes" not in registry or pipeline_id not in registry["nodes"]:
                logger.warning(f"Pipeline {pipeline_id} not found in registry")
                return False

            # Remove from nodes
            del registry["nodes"][pipeline_id]

            # Remove from tag index
            self._remove_from_tag_index(registry, pipeline_id)

            # Update metadata
            self._update_registry_metadata(registry)

            # Save registry
            self._save_registry(registry)

            logger.info(f"Removed pipeline {pipeline_id} from registry")
            return True

        except Exception as e:
            logger.error(f"Failed to remove pipeline {pipeline_id} from registry: {e}")
            return False

    def _remove_from_tag_index(
        self, registry: Dict[str, Any], pipeline_id: str
    ) -> None:
        """
        Remove pipeline from tag index.

        Args:
            registry: Registry data
            pipeline_id: Pipeline identifier to remove
        """
        tag_index = registry.get("tag_index", {})

        for category, tag_dict in tag_index.items():
            for tag, pipeline_list in tag_dict.items():
                if pipeline_id in pipeline_list:
                    pipeline_list.remove(pipeline_id)

            # Remove empty tag entries
            empty_tags = [
                tag for tag, pipeline_list in tag_dict.items() if not pipeline_list
            ]
            for tag in empty_tags:
                del tag_dict[tag]

    def get_registry_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive registry statistics.

        Returns:
            Dictionary containing registry statistics
        """
        try:
            registry = self.load_registry()

            stats = {
                "total_pipelines": len(registry.get("nodes", {})),
                "frameworks": registry.get("metadata", {}).get("frameworks", []),
                "complexity_levels": registry.get("metadata", {}).get(
                    "complexity_levels", []
                ),
                "total_connections": registry.get("connection_graph_metadata", {}).get(
                    "total_connections", 0
                ),
                "connection_density": registry.get("connection_graph_metadata", {}).get(
                    "connection_density", 0.0
                ),
                "isolated_nodes": registry.get("connection_graph_metadata", {}).get(
                    "isolated_nodes", []
                ),
                "last_updated": registry.get("metadata", {}).get(
                    "last_updated", "unknown"
                ),
            }

            # Add tag statistics
            tag_index = registry.get("tag_index", {})
            stats["tag_statistics"] = {}

            for category, tag_dict in tag_index.items():
                stats["tag_statistics"][category] = {
                    "total_tags": len(tag_dict),
                    "most_common_tags": sorted(
                        [(tag, len(pipelines)) for tag, pipelines in tag_dict.items()],
                        key=lambda x: x[1],
                        reverse=True,
                    )[
                        :5
                    ],  # Top 5 most common tags
                }

            return stats

        except Exception as e:
            logger.error(f"Failed to get registry statistics: {e}")
            return {"error": str(e)}


def create_empty_registry(registry_path: str = "catalog_index.json") -> None:
    """
    Create an empty registry file with proper structure.

    Args:
        registry_path: Path where to create the registry
    """
    sync = DAGMetadataRegistrySync(registry_path)
    sync._create_empty_registry()
    logger.info(f"Created empty registry at {registry_path}")


def validate_registry_file(registry_path: str = "catalog_index.json") -> List[str]:
    """
    Validate a registry file and return any errors.

    Args:
        registry_path: Path to registry file to validate

    Returns:
        List of validation errors (empty if valid)
    """
    try:
        sync = DAGMetadataRegistrySync(registry_path)
        registry = sync.load_registry()

        # Additional validation checks
        errors = []

        # Check for orphaned connections
        nodes = registry.get("nodes", {})
        for node_id, node in nodes.items():
            connections = node.get("connections", {})
            for conn_type, conn_list in connections.items():
                for conn in conn_list:
                    target_id = conn["id"]
                    if target_id not in nodes:
                        errors.append(f"Orphaned connection: {node_id} -> {target_id}")

        # Check tag index consistency
        tag_index = registry.get("tag_index", {})
        for category, tag_dict in tag_index.items():
            for tag, pipeline_list in tag_dict.items():
                for pipeline_id in pipeline_list:
                    if pipeline_id not in nodes:
                        errors.append(
                            f"Tag index references non-existent pipeline: {pipeline_id}"
                        )

        return errors

    except Exception as e:
        return [f"Registry validation failed: {e}"]

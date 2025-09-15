"""
Catalog Registry Management

Central registry manager that implements Zettelkasten principles for
pipeline discovery and navigation.
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..shared_dags.registry_sync import DAGMetadataRegistrySync, RegistryValidationError
from ..shared_dags.enhanced_metadata import EnhancedDAGMetadata, ZettelkastenMetadata

logger = logging.getLogger(__name__)


class CatalogRegistry:
    """
    Central registry for Zettelkasten-inspired pipeline catalog management.

    Implements the five core Zettelkasten principles:
    1. Atomicity - Each pipeline is an atomic unit
    2. Connectivity - Explicit connections between pipelines
    3. Anti-categories - Tag-based emergent organization
    4. Manual linking - Curated connections over search
    5. Dual-form structure - Metadata separate from implementation
    """

    def __init__(self, registry_path: str = "catalog_index.json"):
        """
        Initialize registry with connection index.

        Args:
            registry_path: Path to the registry JSON file
        """
        self.registry_path = registry_path
        self._sync = DAGMetadataRegistrySync(registry_path)
        self._cache = {}
        self._cache_valid = False

    def load_registry(self) -> Dict[str, Any]:
        """
        Load the connection registry from JSON.

        Returns:
            Dict containing the complete registry data
        """
        try:
            registry = self._sync.load_registry()
            self._cache = registry
            self._cache_valid = True
            return registry
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            raise

    def save_registry(self, registry: Dict[str, Any]) -> None:
        """
        Save the connection registry to JSON.

        Args:
            registry: Complete registry data to save
        """
        try:
            self._sync._save_registry(registry)
            self._cache = registry
            self._cache_valid = True
            logger.debug("Registry saved successfully")
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
            raise

    def get_pipeline_node(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """
        Get complete node information for a pipeline.

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            Pipeline node data or None if not found
        """
        try:
            if not self._cache_valid:
                self.load_registry()

            nodes = self._cache.get("nodes", {})
            return nodes.get(pipeline_id)

        except Exception as e:
            logger.error(f"Failed to get pipeline node {pipeline_id}: {e}")
            return None

    def get_all_pipelines(self) -> List[str]:
        """
        Get list of all pipeline IDs in the registry.

        Returns:
            List of pipeline identifiers
        """
        try:
            if not self._cache_valid:
                self.load_registry()

            nodes = self._cache.get("nodes", {})
            return list(nodes.keys())

        except Exception as e:
            logger.error(f"Failed to get all pipelines: {e}")
            return []

    def add_or_update_enhanced_node(
        self, enhanced_metadata: EnhancedDAGMetadata
    ) -> bool:
        """
        Add or update a pipeline node using EnhancedDAGMetadata.

        This is the preferred method for adding pipelines with the new enhanced metadata system.
        It automatically converts the EnhancedDAGMetadata to the registry format and handles
        all Zettelkasten metadata integration.

        Args:
            enhanced_metadata: EnhancedDAGMetadata object containing all pipeline metadata

        Returns:
            True if added/updated successfully, False otherwise
        """
        try:
            # Use the registry sync to convert and add the metadata
            # sync_metadata_to_registry returns None on success, raises exception on failure
            self._sync.sync_metadata_to_registry(
                dag_metadata=enhanced_metadata,
                pipeline_file_path=enhanced_metadata.zettelkasten_metadata.source_file,
            )

            # Clear cache to force reload with updated data
            self.clear_cache()
            logger.info(
                f"Successfully added/updated enhanced node {enhanced_metadata.zettelkasten_metadata.atomic_id}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to add/update enhanced node: {e}")
            return False

    def add_or_update_node(self, zettelkasten_metadata: ZettelkastenMetadata) -> bool:
        """
        Add or update a pipeline node using ZettelkastenMetadata (legacy method).

        This method is maintained for backward compatibility with existing pipeline files
        that haven't been migrated to the EnhancedDAGMetadata system yet.

        Args:
            zettelkasten_metadata: ZettelkastenMetadata object

        Returns:
            True if added/updated successfully, False otherwise
        """
        try:
            # Convert ZettelkastenMetadata to the registry node format
            node_data = self._convert_zettelkasten_to_node_data(zettelkasten_metadata)
            pipeline_id = zettelkasten_metadata.atomic_id

            # Use the existing add_pipeline_node method
            return self.add_pipeline_node(pipeline_id, node_data)

        except Exception as e:
            logger.error(f"Failed to add/update node from ZettelkastenMetadata: {e}")
            return False

    def add_pipeline_node(self, pipeline_id: str, node_data: Dict[str, Any]) -> bool:
        """
        Add a new pipeline node to the registry.

        Args:
            pipeline_id: Pipeline identifier
            node_data: Complete node data

        Returns:
            True if added successfully, False otherwise
        """
        try:
            if not self._cache_valid:
                self.load_registry()

            # Validate node structure
            required_fields = [
                "id",
                "title",
                "description",
                "atomic_properties",
                "zettelkasten_metadata",
            ]
            for field in required_fields:
                if field not in node_data:
                    logger.error(f"Missing required field {field} in node data")
                    return False

            # Add to registry
            if "nodes" not in self._cache:
                self._cache["nodes"] = {}

            self._cache["nodes"][pipeline_id] = node_data

            # Update tag index for the new pipeline
            self._update_tag_index_for_pipeline(pipeline_id, node_data)

            # Update metadata
            self._update_registry_metadata()

            # Save registry
            self.save_registry(self._cache)

            logger.info(f"Added pipeline node {pipeline_id} to registry")
            return True

        except Exception as e:
            logger.error(f"Failed to add pipeline node {pipeline_id}: {e}")
            return False

    def remove_pipeline_node(self, pipeline_id: str) -> bool:
        """
        Remove a pipeline node from the registry.

        Args:
            pipeline_id: Pipeline identifier to remove

        Returns:
            True if removed successfully, False if not found
        """
        try:
            if not self._cache_valid:
                self.load_registry()

            nodes = self._cache.get("nodes", {})
            if pipeline_id not in nodes:
                logger.warning(f"Pipeline {pipeline_id} not found in registry")
                return False

            # Remove from nodes
            del nodes[pipeline_id]

            # Remove from tag index
            self._remove_from_tag_index(pipeline_id)

            # Remove connections to this pipeline from other nodes
            self._remove_connections_to_pipeline(pipeline_id)

            # Update metadata
            self._update_registry_metadata()

            # Save registry
            self.save_registry(self._cache)

            logger.info(f"Removed pipeline node {pipeline_id} from registry")
            return True

        except Exception as e:
            logger.error(f"Failed to remove pipeline node {pipeline_id}: {e}")
            return False

    def update_pipeline_node(self, pipeline_id: str, node_data: Dict[str, Any]) -> bool:
        """
        Update an existing pipeline node in the registry.

        Args:
            pipeline_id: Pipeline identifier
            node_data: Updated node data

        Returns:
            True if updated successfully, False otherwise
        """
        try:
            if not self._cache_valid:
                self.load_registry()

            nodes = self._cache.get("nodes", {})
            if pipeline_id not in nodes:
                logger.warning(f"Pipeline {pipeline_id} not found in registry")
                return False

            # Update node
            nodes[pipeline_id] = node_data

            # Update tag index
            self._update_tag_index_for_pipeline(pipeline_id, node_data)

            # Update metadata
            self._update_registry_metadata()

            # Save registry
            self.save_registry(self._cache)

            logger.info(f"Updated pipeline node {pipeline_id} in registry")
            return True

        except Exception as e:
            logger.error(f"Failed to update pipeline node {pipeline_id}: {e}")
            return False

    def get_pipelines_by_framework(self, framework: str) -> List[str]:
        """
        Get all pipelines for a specific framework.

        Args:
            framework: Framework name (e.g., 'xgboost', 'pytorch')

        Returns:
            List of pipeline identifiers
        """
        try:
            if not self._cache_valid:
                self.load_registry()

            matching_pipelines = []
            nodes = self._cache.get("nodes", {})

            for pipeline_id, node in nodes.items():
                node_framework = node.get("zettelkasten_metadata", {}).get(
                    "framework", ""
                )
                if node_framework == framework:
                    matching_pipelines.append(pipeline_id)

            return matching_pipelines

        except Exception as e:
            logger.error(f"Failed to get pipelines by framework {framework}: {e}")
            return []

    def get_pipelines_by_complexity(self, complexity: str) -> List[str]:
        """
        Get all pipelines for a specific complexity level.

        Args:
            complexity: Complexity level (e.g., 'simple', 'standard', 'advanced')

        Returns:
            List of pipeline identifiers
        """
        try:
            if not self._cache_valid:
                self.load_registry()

            matching_pipelines = []
            nodes = self._cache.get("nodes", {})

            for pipeline_id, node in nodes.items():
                node_complexity = node.get("zettelkasten_metadata", {}).get(
                    "complexity", ""
                )
                if node_complexity == complexity:
                    matching_pipelines.append(pipeline_id)

            return matching_pipelines

        except Exception as e:
            logger.error(f"Failed to get pipelines by complexity {complexity}: {e}")
            return []

    def get_pipeline_connections(
        self, pipeline_id: str
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Get all connections for a specific pipeline.

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            Dictionary of connection types to connection lists
        """
        try:
            node = self.get_pipeline_node(pipeline_id)
            if node is None:
                return {}

            return node.get("connections", {})

        except Exception as e:
            logger.error(f"Failed to get connections for pipeline {pipeline_id}: {e}")
            return {}

    def add_connection(
        self, source_id: str, target_id: str, connection_type: str, annotation: str
    ) -> bool:
        """
        Add a connection between two pipelines.

        Args:
            source_id: Source pipeline identifier
            target_id: Target pipeline identifier
            connection_type: Type of connection ('alternatives', 'related', 'used_in')
            annotation: Human-readable annotation for the connection

        Returns:
            True if connection added successfully, False otherwise
        """
        try:
            if not self._cache_valid:
                self.load_registry()

            nodes = self._cache.get("nodes", {})

            # Validate both pipelines exist
            if source_id not in nodes:
                logger.error(f"Source pipeline {source_id} not found")
                return False

            if target_id not in nodes:
                logger.error(f"Target pipeline {target_id} not found")
                return False

            # Validate connection type
            valid_types = ["alternatives", "related", "used_in"]
            if connection_type not in valid_types:
                logger.error(f"Invalid connection type {connection_type}")
                return False

            # Add connection to source node
            source_node = nodes[source_id]
            if "connections" not in source_node:
                source_node["connections"] = {
                    "alternatives": [],
                    "related": [],
                    "used_in": [],
                }

            if connection_type not in source_node["connections"]:
                source_node["connections"][connection_type] = []

            # Check if connection already exists
            existing_connections = source_node["connections"][connection_type]
            for conn in existing_connections:
                if conn["id"] == target_id:
                    logger.info(f"Connection {source_id} -> {target_id} already exists")
                    return True

            # Add new connection
            source_node["connections"][connection_type].append(
                {"id": target_id, "annotation": annotation}
            )

            # Update metadata
            self._update_registry_metadata()

            # Save registry
            self.save_registry(self._cache)

            logger.info(
                f"Added connection {source_id} -> {target_id} ({connection_type})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to add connection {source_id} -> {target_id}: {e}")
            return False

    def remove_connection(
        self, source_id: str, target_id: str, connection_type: str
    ) -> bool:
        """
        Remove a connection between two pipelines.

        Args:
            source_id: Source pipeline identifier
            target_id: Target pipeline identifier
            connection_type: Type of connection to remove

        Returns:
            True if connection removed successfully, False otherwise
        """
        try:
            if not self._cache_valid:
                self.load_registry()

            nodes = self._cache.get("nodes", {})

            if source_id not in nodes:
                logger.error(f"Source pipeline {source_id} not found")
                return False

            source_node = nodes[source_id]
            connections = source_node.get("connections", {})

            if connection_type not in connections:
                logger.warning(
                    f"No {connection_type} connections found for {source_id}"
                )
                return False

            # Find and remove connection
            connection_list = connections[connection_type]
            for i, conn in enumerate(connection_list):
                if conn["id"] == target_id:
                    del connection_list[i]

                    # Update metadata
                    self._update_registry_metadata()

                    # Save registry
                    self.save_registry(self._cache)

                    logger.info(
                        f"Removed connection {source_id} -> {target_id} ({connection_type})"
                    )
                    return True

            logger.warning(f"Connection {source_id} -> {target_id} not found")
            return False

        except Exception as e:
            logger.error(f"Failed to remove connection {source_id} -> {target_id}: {e}")
            return False

    def validate_registry_integrity(self) -> Dict[str, Any]:
        """
        Validate registry structure and connection integrity.

        Returns:
            Validation result with errors and warnings
        """
        try:
            if not self._cache_valid:
                self.load_registry()

            errors = []
            warnings = []

            nodes = self._cache.get("nodes", {})

            # Check for orphaned connections
            for source_id, node in nodes.items():
                connections = node.get("connections", {})
                for conn_type, conn_list in connections.items():
                    for conn in conn_list:
                        target_id = conn["id"]
                        if target_id not in nodes:
                            errors.append(
                                f"Orphaned connection: {source_id} -> {target_id}"
                            )

            # Check tag index consistency
            tag_index = self._cache.get("tag_index", {})
            for category, tag_dict in tag_index.items():
                for tag, pipeline_list in tag_dict.items():
                    for pipeline_id in pipeline_list:
                        if pipeline_id not in nodes:
                            errors.append(
                                f"Tag index references non-existent pipeline: {pipeline_id}"
                            )

            # Check for isolated nodes
            isolated_nodes = []
            for pipeline_id, node in nodes.items():
                connections = node.get("connections", {})
                total_connections = sum(
                    len(conn_list) for conn_list in connections.values()
                )
                if total_connections == 0:
                    isolated_nodes.append(pipeline_id)

            if isolated_nodes:
                warnings.append(f"Isolated nodes (no connections): {isolated_nodes}")

            return {
                "is_valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "total_nodes": len(nodes),
                "isolated_nodes": len(isolated_nodes),
            }

        except Exception as e:
            logger.error(f"Failed to validate registry integrity: {e}")
            return {
                "is_valid": False,
                "errors": [f"Validation failed: {e}"],
                "warnings": [],
                "total_nodes": 0,
                "isolated_nodes": 0,
            }

    def get_registry_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive registry statistics.

        Returns:
            Dictionary containing registry statistics
        """
        return self._sync.get_registry_statistics()

    def _update_registry_metadata(self) -> None:
        """Update registry-level metadata."""
        self._sync._update_registry_metadata(self._cache)

    def _remove_from_tag_index(self, pipeline_id: str) -> None:
        """Remove pipeline from tag index."""
        self._sync._remove_from_tag_index(self._cache, pipeline_id)

    def _remove_connections_to_pipeline(self, target_id: str) -> None:
        """Remove all connections pointing to a specific pipeline."""
        nodes = self._cache.get("nodes", {})

        for source_id, node in nodes.items():
            connections = node.get("connections", {})
            for conn_type, conn_list in connections.items():
                # Remove connections to target_id
                connections[conn_type] = [
                    conn for conn in conn_list if conn["id"] != target_id
                ]

    def _update_tag_index_for_pipeline(
        self, pipeline_id: str, node_data: Dict[str, Any]
    ) -> None:
        """Update tag index for a specific pipeline."""
        # First remove old entries
        self._remove_from_tag_index(pipeline_id)

        # Then add new entries
        tags = node_data.get("multi_dimensional_tags", {})
        tag_index = self._cache.get("tag_index", {})

        for category, tag_list in tags.items():
            if category not in tag_index:
                tag_index[category] = {}

            for tag in tag_list:
                if tag not in tag_index[category]:
                    tag_index[category][tag] = []

                if pipeline_id not in tag_index[category][tag]:
                    tag_index[category][tag].append(pipeline_id)

    def _convert_zettelkasten_to_node_data(
        self, zettelkasten_metadata: ZettelkastenMetadata
    ) -> Dict[str, Any]:
        """
        Convert ZettelkastenMetadata to registry node data format.

        This method provides backward compatibility for pipeline files that haven't
        been migrated to the EnhancedDAGMetadata system yet. The format matches
        exactly what EnhancedDAGMetadata.to_registry_node() produces.

        Args:
            zettelkasten_metadata: ZettelkastenMetadata object to convert

        Returns:
            Dict containing node data in registry format matching catalog_index.json
        """
        try:
            # Build connections from manual connections
            connections = {"alternatives": [], "related": [], "used_in": []}

            # Convert manual connections to registry format
            for (
                conn_type,
                target_ids,
            ) in zettelkasten_metadata.manual_connections.items():
                if conn_type in connections:
                    for target_id in target_ids:
                        annotation = zettelkasten_metadata.curated_connections.get(
                            target_id, f"Connected via {conn_type}"
                        )
                        connections[conn_type].append(
                            {"id": target_id, "annotation": annotation}
                        )

            # Create the node data structure that matches EnhancedDAGMetadata.to_registry_node()
            node_data = {
                "id": zettelkasten_metadata.atomic_id,
                "title": zettelkasten_metadata.title
                or zettelkasten_metadata.atomic_id.replace("_", " ").title(),
                "description": zettelkasten_metadata.single_responsibility,
                "atomic_properties": {
                    "single_responsibility": zettelkasten_metadata.single_responsibility,
                    "independence_level": zettelkasten_metadata.independence_level,
                    "node_count": zettelkasten_metadata.node_count,
                    "edge_count": zettelkasten_metadata.edge_count,
                },
                "zettelkasten_metadata": {
                    "framework": zettelkasten_metadata.framework,
                    "complexity": zettelkasten_metadata.complexity,
                    "use_case": zettelkasten_metadata.use_case
                    or zettelkasten_metadata.single_responsibility,
                    "features": zettelkasten_metadata.features,
                    "mods_compatible": zettelkasten_metadata.mods_compatible,
                },
                "multi_dimensional_tags": {
                    "framework_tags": zettelkasten_metadata.framework_tags,
                    "task_tags": zettelkasten_metadata.task_tags,
                    "complexity_tags": zettelkasten_metadata.complexity_tags,
                },
                "source_file": zettelkasten_metadata.source_file,
                "migration_source": zettelkasten_metadata.migration_source,
                "connections": connections,
                "created_date": zettelkasten_metadata.created_date,
                "priority": zettelkasten_metadata.priority,
            }

            return node_data

        except Exception as e:
            logger.error(f"Failed to convert ZettelkastenMetadata to node data: {e}")
            raise

    def clear_cache(self) -> None:
        """Clear the internal cache, forcing reload on next access."""
        self._cache = {}
        self._cache_valid = False
        logger.debug("Registry cache cleared")

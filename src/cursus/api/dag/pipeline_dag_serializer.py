"""
Pipeline DAG Serialization Module.

This module provides utilities for serializing and deserializing PipelineDAG objects
to enable easy knowledge transfer and storage of pipeline topologies.
"""

import json
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime, timezone
import logging

from .base_dag import PipelineDAG

logger = logging.getLogger(__name__)


class PipelineDAGWriter:
    """
    Writer for serializing PipelineDAG to various formats.

    Supports:
    - JSON format with metadata
    - Pretty printing for readability
    - Validation before writing
    - Metadata tracking (creation time, etc.)
    """

    def __init__(self, dag: PipelineDAG, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize writer with a PipelineDAG instance.

        Args:
            dag: PipelineDAG instance to serialize
            metadata: Optional metadata to include in serialization
        """
        self.dag = dag
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert PipelineDAG to dictionary representation.

        Returns:
            Dictionary with complete DAG representation including metadata
        """
        return {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": self.metadata,
            "dag": {
                "nodes": self.dag.nodes,
                "edges": [[src, dst] for src, dst in self.dag.edges],
            },
            "statistics": self._compute_statistics(),
        }

    def to_json(self, pretty: bool = True, indent: int = 2) -> str:
        """
        Convert PipelineDAG to JSON string.

        Args:
            pretty: If True, format JSON with indentation
            indent: Number of spaces for indentation (if pretty=True)

        Returns:
            JSON string representation of the DAG
        """
        data = self.to_dict()
        if pretty:
            return json.dumps(data, indent=indent, sort_keys=False)
        return json.dumps(data)

    def write_to_file(
        self,
        filepath: Union[str, Path],
        pretty: bool = True,
        indent: int = 2,
        validate: bool = True,
    ) -> None:
        """
        Write PipelineDAG to a JSON file.

        Args:
            filepath: Path to output file
            pretty: If True, format JSON with indentation
            indent: Number of spaces for indentation
            validate: If True, validate DAG before writing

        Raises:
            ValueError: If DAG validation fails
            IOError: If file cannot be written
        """
        if validate:
            self._validate_dag()

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        json_str = self.to_json(pretty=pretty, indent=indent)

        try:
            with open(filepath, "w") as f:
                f.write(json_str)
            logger.info(f"Successfully wrote PipelineDAG to {filepath}")
        except IOError as e:
            logger.error(f"Failed to write file {filepath}: {e}")
            raise

    def _validate_dag(self) -> None:
        """
        Validate DAG structure before writing.

        Raises:
            ValueError: If DAG has cycles or invalid structure
        """
        # Check for cycles using topological sort
        try:
            self.dag.topological_sort()
        except ValueError as e:
            logger.error(f"DAG validation failed: {e}")
            raise

        # Check for empty nodes
        if not self.dag.nodes:
            raise ValueError("Cannot write empty DAG (no nodes)")

        # Check for dangling edges
        node_set = set(self.dag.nodes)
        for src, dst in self.dag.edges:
            if src not in node_set:
                raise ValueError(f"Edge references non-existent source node: {src}")
            if dst not in node_set:
                raise ValueError(
                    f"Edge references non-existent destination node: {dst}"
                )

        logger.debug("DAG validation passed")

    def _compute_statistics(self) -> Dict[str, Any]:
        """
        Compute statistics about the DAG.

        Returns:
            Dictionary with DAG statistics
        """
        try:
            topo_order = self.dag.topological_sort()
            has_cycles = False
        except ValueError:
            topo_order = []
            has_cycles = True

        # Count nodes by degree
        in_degrees = {
            node: len(self.dag.reverse_adj.get(node, [])) for node in self.dag.nodes
        }
        out_degrees = {
            node: len(self.dag.adj_list.get(node, [])) for node in self.dag.nodes
        }

        # Identify entry and exit nodes
        entry_nodes = [node for node, degree in in_degrees.items() if degree == 0]
        exit_nodes = [node for node, degree in out_degrees.items() if degree == 0]

        # Compute depth (longest path from any entry node)
        max_depth = self._compute_max_depth()

        return {
            "node_count": len(self.dag.nodes),
            "edge_count": len(self.dag.edges),
            "has_cycles": has_cycles,
            "entry_nodes": entry_nodes,
            "exit_nodes": exit_nodes,
            "max_depth": max_depth,
            "isolated_nodes": [
                node
                for node in self.dag.nodes
                if in_degrees[node] == 0 and out_degrees[node] == 0
            ],
        }

    def _compute_max_depth(self) -> int:
        """
        Compute maximum depth of the DAG (longest path from entry to exit).

        Returns:
            Maximum depth, or 0 if DAG has cycles
        """
        try:
            topo_order = self.dag.topological_sort()
        except ValueError:
            return 0

        # Dynamic programming to compute longest path
        depths = {node: 0 for node in self.dag.nodes}

        for node in topo_order:
            for predecessor in self.dag.reverse_adj.get(node, []):
                depths[node] = max(depths[node], depths[predecessor] + 1)

        return max(depths.values()) if depths else 0


class PipelineDAGReader:
    """
    Reader for deserializing PipelineDAG from various formats.

    Supports:
    - JSON format
    - Validation during reading
    - Metadata extraction
    """

    @classmethod
    def from_dict(cls, data: Dict[str, Any], validate: bool = True) -> PipelineDAG:
        """
        Create PipelineDAG from dictionary representation.

        Args:
            data: Dictionary containing DAG data
            validate: If True, validate data before creating DAG

        Returns:
            PipelineDAG instance

        Raises:
            ValueError: If data is invalid or version is unsupported
        """
        if validate:
            cls._validate_data(data)

        # Extract DAG data
        dag_data = data.get("dag", {})
        nodes = dag_data.get("nodes", [])
        edges = [tuple(edge) for edge in dag_data.get("edges", [])]

        # Create DAG instance
        dag = PipelineDAG(nodes=nodes, edges=edges)

        logger.info(
            f"Successfully loaded PipelineDAG with {len(nodes)} nodes and {len(edges)} edges"
        )

        return dag

    @classmethod
    def from_json(cls, json_str: str, validate: bool = True) -> PipelineDAG:
        """
        Create PipelineDAG from JSON string.

        Args:
            json_str: JSON string representation of DAG
            validate: If True, validate data before creating DAG

        Returns:
            PipelineDAG instance

        Raises:
            ValueError: If JSON is invalid or data is malformed
        """
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {e}")
            raise ValueError(f"Invalid JSON format: {e}")

        return cls.from_dict(data, validate=validate)

    @classmethod
    def read_from_file(
        cls, filepath: Union[str, Path], validate: bool = True
    ) -> PipelineDAG:
        """
        Read PipelineDAG from a JSON file.

        Args:
            filepath: Path to input file
            validate: If True, validate data before creating DAG

        Returns:
            PipelineDAG instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file content is invalid
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        try:
            with open(filepath, "r") as f:
                json_str = f.read()
            logger.info(f"Reading PipelineDAG from {filepath}")
            return cls.from_json(json_str, validate=validate)
        except IOError as e:
            logger.error(f"Failed to read file {filepath}: {e}")
            raise ValueError(f"Cannot read file: {e}")

    @classmethod
    def extract_metadata(cls, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract metadata from a DAG file without loading the full DAG.

        Args:
            filepath: Path to input file

        Returns:
            Dictionary containing metadata
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, "r") as f:
            data = json.load(f)

        return {
            "created_at": data.get("created_at"),
            "metadata": data.get("metadata", {}),
            "statistics": data.get("statistics", {}),
        }

    @classmethod
    def _validate_data(cls, data: Dict[str, Any]) -> None:
        """
        Validate data structure.

        Args:
            data: Dictionary to validate

        Raises:
            ValueError: If data is invalid
        """
        # Check required fields
        if "dag" not in data:
            raise ValueError("Missing required field: 'dag'")

        dag_data = data["dag"]

        if "nodes" not in dag_data:
            raise ValueError("Missing required field: 'dag.nodes'")

        if "edges" not in dag_data:
            raise ValueError("Missing required field: 'dag.edges'")

        # Validate nodes
        nodes = dag_data["nodes"]
        if not isinstance(nodes, list):
            raise ValueError("'dag.nodes' must be a list")

        # Validate edges
        edges = dag_data["edges"]
        if not isinstance(edges, list):
            raise ValueError("'dag.edges' must be a list")

        for edge in edges:
            if not isinstance(edge, list) or len(edge) != 2:
                raise ValueError(f"Invalid edge format: {edge}. Expected [src, dst]")

        logger.debug("Data validation passed")


def export_dag_to_json(
    dag: PipelineDAG,
    filepath: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
    pretty: bool = True,
) -> None:
    """
    Convenience function to export a PipelineDAG to JSON file.

    Args:
        dag: PipelineDAG instance to export
        filepath: Output file path
        metadata: Optional metadata to include
        pretty: If True, format JSON with indentation
    """
    writer = PipelineDAGWriter(dag, metadata=metadata)
    writer.write_to_file(filepath, pretty=pretty)


def import_dag_from_json(filepath: Union[str, Path]) -> PipelineDAG:
    """
    Convenience function to import a PipelineDAG from JSON file.

    Args:
        filepath: Input file path

    Returns:
        PipelineDAG instance
    """
    return PipelineDAGReader.read_from_file(filepath)

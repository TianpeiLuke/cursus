"""
Shared input resolvers for the cursus MCP tool handlers.

Several tool namespaces (``compile``, ``config``, ``execdoc``) each grew their own
near-identical "turn the args into a ``PipelineDAG``" helper, with subtly different
accepted shapes. This module is the single canonical resolver so every DAG/config-taking
tool advertises and accepts the *same* input contract:

DAG input — supply exactly one of:
  - ``dag_file``: path to a serialized DAG JSON file (loaded via ``import_dag_from_json``), or
  - ``dag``: an inline object, either the flat form ``{"nodes": [...], "edges": [[s, d], ...]}``
    or the serializer/wrapped form ``{"dag": {"nodes": [...], "edges": [...]}}``.

Config input:
  - ``config_file``: path to the pipeline configuration JSON.

Engine imports are lazy (inside the functions) so importing the tool modules stays cheap.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

from ..envelope import ToolError

# Reusable JSON-schema fragment for the inline ``dag`` object — both flat and wrapped
# forms are accepted (the wrapped ``dag`` key is optional). Tools spread DAG_INPUT_PROPS
# into their own schema ``properties`` so every tool advertises the identical contract.
_DAG_OBJECT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "description": (
        "Pipeline DAG topology. Either the flat form {'nodes': [...], 'edges': [[src, "
        "dst], ...]} or the serializer form {'dag': {'nodes': [...], 'edges': [...]}}. "
        "Provide this OR 'dag_file', not both."
    ),
    "properties": {
        "nodes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Step/node names, e.g. 'TabularPreprocessing_training'.",
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 2,
            },
            "description": "Directed dependency edges as [src, dst] pairs.",
        },
        "dag": {
            "type": "object",
            "description": "Serializer-wrapped body {'nodes': [...], 'edges': [...]}.",
        },
    },
}

# DAG input properties shared by every DAG-taking tool's schema.
DAG_INPUT_PROPS: Dict[str, Any] = {
    "dag": _DAG_OBJECT_SCHEMA,
    "dag_file": {
        "type": "string",
        "description": (
            "Path to a serialized DAG JSON file (loaded via import_dag_from_json). "
            "Provide this OR 'dag'."
        ),
    },
}

# Config input property shared by every config-taking tool's schema.
CONFIG_INPUT_PROPS: Dict[str, Any] = {
    "config_file": {
        "type": "string",
        "description": "Path to the pipeline configuration JSON file.",
    },
}


def _normalize_edges(raw_edges: Any) -> List[Tuple[str, str]]:
    """Validate + coerce a JSON edge list into a list of (src, dst) tuples."""
    if raw_edges is None:
        return []
    if not isinstance(raw_edges, list):
        raise ToolError(
            "'dag.edges' must be a list of [src, dst] pairs", code="invalid_input"
        )
    edges: List[Tuple[str, str]] = []
    for edge in raw_edges:
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            raise ToolError(
                f"invalid edge {edge!r}: each edge must be a [src, dst] pair",
                code="invalid_input",
            )
        edges.append((edge[0], edge[1]))
    return edges


def resolve_dag(args: Dict[str, Any]) -> Any:
    """
    Build a ``PipelineDAG`` from ``args``, accepting exactly one of ``dag_file`` or
    inline ``dag`` (flat or wrapped). Raises :class:`ToolError` on bad/ambiguous input.

    This is the canonical superset of the former per-tool resolvers: it errors when both
    sources are supplied (rather than silently preferring one) and accepts the wrapped
    ``{"dag": {...}}`` form everywhere.
    """
    dag_file = args.get("dag_file")
    inline = args.get("dag")

    if dag_file and inline is not None:
        raise ToolError(
            "provide either 'dag' (inline JSON) or 'dag_file' (path), not both",
            code="invalid_input",
        )
    if not dag_file and inline is None:
        raise ToolError(
            "must provide one of 'dag' (inline JSON) or 'dag_file' (path)",
            code="invalid_input",
        )

    if dag_file:
        if not os.path.exists(dag_file):
            raise ToolError(f"dag_file not found: {dag_file}", code="not_found")
        from ...api.dag import import_dag_from_json

        try:
            return import_dag_from_json(dag_file)
        except Exception as exc:  # noqa: BLE001 - surface as a handled tool error
            raise ToolError(
                f"failed to load DAG from '{dag_file}': {exc}",
                code="invalid_input",
            )

    # Inline DAG: accept the flat form or the serializer-wrapped {"dag": {...}} form.
    if not isinstance(inline, dict):
        raise ToolError(
            "'dag' must be a JSON object with 'nodes' and optional 'edges'",
            code="invalid_input",
        )
    body = inline.get("dag") if isinstance(inline.get("dag"), dict) else inline

    nodes = body.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ToolError(
            "'dag.nodes' must be a non-empty list of step names",
            code="invalid_input",
        )
    if not all(isinstance(n, str) for n in nodes):
        raise ToolError(
            "'dag.nodes' must be a list of step-name strings", code="invalid_input"
        )

    edges = _normalize_edges(body.get("edges", []))

    from ...api.dag.base_dag import PipelineDAG

    return PipelineDAG(nodes=list(nodes), edges=edges)


def require_config_exists(config_file: Any) -> None:
    """Raise :class:`ToolError` unless ``config_file`` is a path to an existing file."""
    if not config_file or not os.path.exists(config_file):
        raise ToolError(f"config_file not found: {config_file}", code="not_found")

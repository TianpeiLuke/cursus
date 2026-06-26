"""
Tests for the canonical DAG/config input resolver (cursus.mcp.tools.shared).

Locks the superset contract the three former per-tool resolvers were consolidated into:
inline (flat OR wrapped) dag, dag_file, the both-given / neither error policy, and edge
validation.
"""

import json

import pytest

from cursus.mcp.tools.shared import (
    resolve_dag,
    require_config_exists,
    DAG_INPUT_PROPS,
    CONFIG_INPUT_PROPS,
)
from cursus.mcp.envelope import ToolError


class TestResolveDagInline:
    def test_flat_form(self):
        dag = resolve_dag({"dag": {"nodes": ["a", "b"], "edges": [["a", "b"]]}})
        assert set(dag.nodes) == {"a", "b"}
        assert ("a", "b") in list(dag.edges)

    def test_wrapped_serializer_form(self):
        # config's historical input shape: {"dag": {"dag": {...}}}
        dag = resolve_dag({"dag": {"dag": {"nodes": ["x"], "edges": []}}})
        assert set(dag.nodes) == {"x"}

    def test_edges_optional(self):
        dag = resolve_dag({"dag": {"nodes": ["solo"]}})
        assert set(dag.nodes) == {"solo"}
        assert list(dag.edges) == []


class TestResolveDagErrors:
    def test_both_dag_and_dag_file_errors(self):
        with pytest.raises(ToolError) as ei:
            resolve_dag({"dag": {"nodes": ["a"]}, "dag_file": "/tmp/x.json"})
        assert ei.value.code == "invalid_input"

    def test_neither_errors(self):
        with pytest.raises(ToolError) as ei:
            resolve_dag({})
        assert ei.value.code == "invalid_input"

    def test_empty_nodes_errors(self):
        with pytest.raises(ToolError) as ei:
            resolve_dag({"dag": {"nodes": [], "edges": []}})
        assert ei.value.code == "invalid_input"

    def test_non_string_nodes_errors(self):
        with pytest.raises(ToolError) as ei:
            resolve_dag({"dag": {"nodes": [1, 2]}})
        assert ei.value.code == "invalid_input"

    def test_bad_edge_shape_errors(self):
        with pytest.raises(ToolError) as ei:
            resolve_dag({"dag": {"nodes": ["a", "b"], "edges": [["a"]]}})
        assert ei.value.code == "invalid_input"

    def test_dag_not_object_errors(self):
        with pytest.raises(ToolError) as ei:
            resolve_dag({"dag": "not-an-object"})
        assert ei.value.code == "invalid_input"

    def test_missing_dag_file_errors_not_found(self):
        with pytest.raises(ToolError) as ei:
            resolve_dag({"dag_file": "/nonexistent/path/to/dag.json"})
        assert ei.value.code == "not_found"


class TestResolveDagFromFile:
    def test_loads_a_serialized_dag(self, tmp_path):
        # Build a DAG, serialize it via the engine, then resolve it back through dag_file.
        from cursus.api.dag import export_dag_to_json
        from cursus.api.dag.base_dag import PipelineDAG

        src = PipelineDAG(nodes=["a", "b"], edges=[("a", "b")])
        path = tmp_path / "dag.json"
        export_dag_to_json(src, str(path))

        dag = resolve_dag({"dag_file": str(path)})
        assert set(dag.nodes) == {"a", "b"}
        assert ("a", "b") in list(dag.edges)


class TestRequireConfigExists:
    def test_existing_file_passes(self, tmp_path):
        p = tmp_path / "config.json"
        p.write_text("{}")
        require_config_exists(str(p))  # should not raise

    def test_missing_file_errors(self):
        with pytest.raises(ToolError) as ei:
            require_config_exists("/nope/config.json")
        assert ei.value.code == "not_found"

    def test_none_errors(self):
        with pytest.raises(ToolError) as ei:
            require_config_exists(None)
        assert ei.value.code == "not_found"


class TestSchemaFragments:
    def test_dag_input_props_advertise_both_sources(self):
        assert "dag" in DAG_INPUT_PROPS
        assert "dag_file" in DAG_INPUT_PROPS

    def test_fragments_are_json_serializable(self):
        json.dumps(DAG_INPUT_PROPS)
        json.dumps(CONFIG_INPUT_PROPS)
        assert "config_file" in CONFIG_INPUT_PROPS

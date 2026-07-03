"""
Tests for the cursus.mcp ``dag.*`` tool namespace — DAG construction, integrity validation,
execution planning, dependency lookup, and JSON serialize/deserialize round-trip.

Driven end-to-end through ``call_tool`` (the same entry point an agent/MCP client uses), so these
exercise the real ``PipelineDAG`` / ``PipelineDAGResolver`` / serializer wiring, not mocks.

The load-bearing case is ``undeclared_edge_nodes`` in ``dag.validate_integrity``: ``add_edge``
silently auto-creates any endpoint not already ``add_node``'d, so a single edge-name typo spawns a
phantom, unconfigured node and orphans the real one. The dag.* tool deliberately does NOT union edge
endpoints into the node list, so this detector can fire.
"""

import json
import os
import tempfile

from cursus.mcp import call_tool


class TestDagConstruct:
    def test_construct_returns_serialized_dag(self):
        r = call_tool(
            "dag.construct",
            {"nodes": ["preprocess", "train"], "edges": [["preprocess", "train"]], "metadata": {"author": "t"}},
        )
        assert r.ok
        assert set(["dag", "metadata", "statistics"]).issubset(r.data.keys())
        assert r.data["dag"]["nodes"] == ["preprocess", "train"]
        assert r.meta["node_count"] == 2
        assert r.meta["edge_count"] == 1

    def test_construct_flags_cycle_as_warning_not_error(self):
        # A cycle is a real topology problem but construct still succeeds (it just serializes);
        # the warning is how an agent learns the DAG is not a valid pipeline.
        r = call_tool("dag.construct", {"nodes": ["a", "b"], "edges": [["a", "b"], ["b", "a"]]})
        assert r.ok
        assert r.warnings
        assert any("cycle" in w.lower() for w in r.warnings)

    def test_construct_rejects_non_object_metadata(self):
        r = call_tool("dag.construct", {"nodes": ["a"], "metadata": "not-a-dict"})
        assert not r.ok
        assert r.code == "invalid_input"


class TestDagValidateIntegrity:
    # NOTE: made-up node names ("a"/"b") also trip the catalog's `missing_steps` category, so the
    # graph-structure assertions below check the SPECIFIC category rather than overall is_valid. The
    # fully-clean case uses real registered cursus step names so is_valid can be True.
    def test_clean_dag_is_valid(self):
        r = call_tool(
            "dag.validate_integrity",
            {
                "nodes": ["CradleDataLoading", "TabularPreprocessing"],
                "edges": [["CradleDataLoading", "TabularPreprocessing"]],
            },
        )
        assert r.ok
        assert r.data["is_valid"] is True
        assert r.data["issues"] == {}

    def test_undeclared_edge_node_flagged(self):
        # 'Bb' is an edge endpoint that was NOT listed in nodes — a likely typo. The dag.* tool
        # does not union it into nodes, so validate_node_declarations flags it as a phantom node.
        r = call_tool(
            "dag.validate_integrity",
            {"nodes": ["A", "B"], "edges": [["A", "Bb"]]},
        )
        assert r.ok
        assert "undeclared_edge_nodes" in r.data["issue_categories"]
        assert any("Bb" in msg for msg in r.data["issues"]["undeclared_edge_nodes"])

    def test_declared_edge_nodes_not_flagged(self):
        # When every edge endpoint is a declared node, there is no undeclared_edge_nodes category
        # (even though made-up names still trip missing_steps — that is a different category).
        r = call_tool(
            "dag.validate_integrity",
            {"nodes": ["A", "B"], "edges": [["A", "B"]]},
        )
        assert r.ok
        assert "undeclared_edge_nodes" not in r.data["issue_categories"]

    def test_cycle_flagged(self):
        r = call_tool(
            "dag.validate_integrity",
            {"nodes": ["a", "b"], "edges": [["a", "b"], ["b", "a"]]},
        )
        assert r.ok
        assert r.data["is_valid"] is False
        assert "cycles" in r.data["issue_categories"]


class TestDagResolvePlan:
    def test_topological_order(self):
        r = call_tool(
            "dag.resolve_plan",
            {"nodes": ["a", "b", "c"], "edges": [["a", "b"], ["b", "c"]]},
        )
        assert r.ok
        order = r.data["execution_order"]
        assert order.index("a") < order.index("b") < order.index("c")
        assert r.meta["step_count"] == 3

    def test_cycle_is_a_failure(self):
        r = call_tool(
            "dag.resolve_plan",
            {"nodes": ["a", "b"], "edges": [["a", "b"], ["b", "a"]]},
        )
        assert not r.ok
        assert r.code == "invalid_input"


class TestDagDependencies:
    def test_upstream_and_downstream(self):
        r = call_tool(
            "dag.dependencies",
            {"nodes": ["a", "b", "c"], "edges": [["a", "b"], ["b", "c"]], "step": "b"},
        )
        assert r.ok
        assert r.data["dependencies"] == ["a"]  # immediate parent
        assert r.data["dependents"] == ["c"]  # immediate child

    def test_unknown_step_is_not_found(self):
        r = call_tool(
            "dag.dependencies",
            {"nodes": ["a"], "edges": [], "step": "zzz"},
        )
        assert not r.ok
        assert r.code == "not_found"

    def test_missing_step_arg_is_invalid(self):
        r = call_tool("dag.dependencies", {"nodes": ["a"], "edges": []})
        assert not r.ok
        assert r.code == "invalid_input"


class TestDagSerializeRoundTrip:
    def test_serialize_to_json_string(self):
        r = call_tool("dag.serialize", {"nodes": ["a", "b"], "edges": [["a", "b"]]})
        assert r.ok
        parsed = json.loads(r.data["json"])
        assert parsed["dag"]["nodes"] == ["a", "b"]

    def test_serialize_to_path_then_deserialize(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "dag.json")
        s = call_tool(
            "dag.serialize",
            {"nodes": ["a", "b", "c"], "edges": [["a", "b"], ["b", "c"]], "path": path, "metadata": {"author": "t"}},
        )
        assert s.ok
        assert os.path.exists(path)

        ds = call_tool("dag.deserialize", {"path": path})
        assert ds.ok
        assert ds.data["nodes"] == ["a", "b", "c"]
        assert ds.data["edges"] == [["a", "b"], ["b", "c"]]
        assert ds.data["metadata"].get("author") == "t"

    def test_deserialize_missing_file_is_not_found(self):
        r = call_tool("dag.deserialize", {"path": "/nonexistent/does_not_exist.json"})
        assert not r.ok
        assert r.code == "not_found"


class TestDagToolsRegistered:
    def test_all_dag_tools_are_served(self):
        from cursus.mcp.registry import list_tools

        names = {t["name"] if isinstance(t, dict) else getattr(t, "name", t) for t in list_tools()}
        for expected in [
            "dag.construct",
            "dag.validate_integrity",
            "dag.resolve_plan",
            "dag.dependencies",
            "dag.serialize",
            "dag.deserialize",
        ]:
            assert expected in names

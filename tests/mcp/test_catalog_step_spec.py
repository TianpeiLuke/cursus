"""
Tests for the catalog.step_spec MCP tool (a step's dependencies + outputs).

Exercised end-to-end through call_tool against the real catalog — no engine mocking
needed (it reads the package's .step.yaml interfaces).
"""

from cursus.mcp import call_tool, get_registry


def test_registered():
    assert "catalog.step_spec" in get_registry()


def test_returns_dependencies_and_outputs_for_a_real_step():
    r = call_tool("catalog.step_spec", {"step_name": "XGBoostTraining"})
    assert r.ok
    d = r.data
    assert d["step_type"] == "XGBoostTraining"
    assert "node_type" in d
    # dependencies / outputs are lists of port dicts
    assert isinstance(d["dependencies"], list) and len(d["dependencies"]) >= 1
    assert isinstance(d["outputs"], list) and len(d["outputs"]) >= 1
    assert d["dependency_count"] == len(d["dependencies"])
    assert d["output_count"] == len(d["outputs"])
    # a dependency port carries the fields an agent needs to wire it
    dep = d["dependencies"][0]
    assert "logical_name" in dep
    assert "dependency_type" in dep


def test_json_serializable():
    import json

    r = call_tool("catalog.step_spec", {"step_name": "XGBoostTraining"})
    assert r.ok
    json.dumps(r.to_dict())  # must not raise


def test_missing_required_arg():
    r = call_tool("catalog.step_spec", {})
    assert not r.ok
    assert r.code == "invalid_input"  # schema requires step_name


def test_step_without_spec_returns_not_found_with_remedy():
    # 'Base' is an abstract registry step with no .step.yaml / spec.
    r = call_tool("catalog.step_spec", {"step_name": "Base"})
    assert not r.ok
    assert r.code == "not_found"
    assert r.remedy is not None
    assert "catalog.list_steps" in r.remedy.get("suggested_tools", [])

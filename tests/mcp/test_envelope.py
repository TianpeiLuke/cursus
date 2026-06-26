"""
Tests for the cursus.mcp result envelope (``ToolResult`` / ``ToolError``).

These cover the contract every tool relies on: the success/failure constructors, the
optional ``next_steps`` / ``remedy`` agent-guidance fields, and ``to_dict()`` selectively
emitting fields so payloads stay lean and backward-compatible.
"""

import json

import pytest

from cursus.mcp.envelope import ToolResult, ToolError


class TestToolResultSuccess:
    def test_minimal_success(self):
        r = ToolResult.success({"x": 1})
        assert r.ok is True
        assert r.data == {"x": 1}
        assert r.error is None  # regression guard: must NOT be the classmethod object
        assert r.code is None
        assert r.next_steps == []

    def test_success_to_dict_minimal(self):
        out = ToolResult.success({"x": 1}).to_dict()
        assert out == {"ok": True, "data": {"x": 1}}
        # empty optional fields are omitted entirely
        assert "next_steps" not in out
        assert "warnings" not in out
        assert "remedy" not in out

    def test_success_with_next_steps_emitted(self):
        steps = [{"tool": "compile.dag", "when": "now", "why": "valid"}]
        out = ToolResult.success({"ok": 1}, next_steps=steps).to_dict()
        assert out["next_steps"] == steps

    def test_success_with_warnings_and_meta(self):
        r = ToolResult.success({"a": 1}, warnings=["heads up"], tool="x.y")
        out = r.to_dict()
        assert out["warnings"] == ["heads up"]
        assert out["meta"] == {"tool": "x.y"}

    def test_success_never_emits_remedy(self):
        # remedy is a failure-only concept; success must never carry it.
        out = ToolResult.success({"x": 1}).to_dict()
        assert "remedy" not in out


class TestToolResultFailure:
    def test_minimal_failure(self):
        r = ToolResult.failure("boom")
        assert r.ok is False
        assert r.error == "boom"
        assert r.code == "tool_error"  # default code
        assert r.data is None

    def test_failure_to_dict(self):
        out = ToolResult.failure("nope", code="not_found").to_dict()
        assert out["ok"] is False
        assert out["error"] == "nope"
        assert out["code"] == "not_found"
        assert "data" not in out  # failures carry no data key

    def test_failure_with_remedy_emitted(self):
        remedy = {"suggested_tools": ["config.requirements"], "fix_action": "do X"}
        out = ToolResult.failure("x", code="unsupported", remedy=remedy).to_dict()
        assert out["remedy"] == remedy

    def test_failure_details_go_into_meta(self):
        out = ToolResult.failure(
            "bad", code="invalid_input", details={"field": "nodes"}
        ).to_dict()
        assert out["meta"]["details"] == {"field": "nodes"}

    def test_failure_never_emits_next_steps(self):
        out = ToolResult.failure("x").to_dict()
        assert "next_steps" not in out


class TestSerializable:
    @pytest.mark.parametrize(
        "result",
        [
            ToolResult.success({"k": "v"}),
            ToolResult.success(
                {"k": 1}, next_steps=[{"tool": "t", "when": "w", "why": "y"}]
            ),
            ToolResult.failure("e", code="not_found"),
            ToolResult.failure(
                "e", remedy={"suggested_tools": ["a"], "fix_action": "b"}
            ),
        ],
    )
    def test_to_dict_is_json_serializable(self, result):
        # Every envelope must round-trip through json with no custom encoder.
        s = json.dumps(result.to_dict())
        assert isinstance(s, str)


class TestToolError:
    def test_carries_code_and_details(self):
        e = ToolError("bad", code="invalid_input", details={"why": "x"})
        assert e.message == "bad"
        assert e.code == "invalid_input"
        assert e.details == {"why": "x"}

    def test_defaults(self):
        e = ToolError("oops")
        assert e.code == "tool_error"
        assert e.details == {}

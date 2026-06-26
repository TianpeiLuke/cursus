"""
Tests for the cursus.mcp meta/discovery namespace (tools.by_phase / tools.describe_tool).

These exercise the agent-ergonomics discovery layer end-to-end through call_tool against
the real assembled registry — they need no engine mocking (the meta tools only read the
registry).
"""

from cursus.mcp import call_tool


class TestByPhase:
    def test_validator_phase_returns_tools(self):
        r = call_tool("tools.by_phase", {"phase": "validator"})
        assert r.ok
        assert r.data["phase"] == "validator"
        assert r.data["count"] >= 1
        # every returned tool is actually tagged validator
        assert all("validator" in t["tags"] for t in r.data["tools"])

    def test_planner_phase_returns_tools(self):
        r = call_tool("tools.by_phase", {"phase": "planner"})
        assert r.ok
        assert r.data["count"] >= 1

    def test_programmer_phase_returns_tools(self):
        r = call_tool("tools.by_phase", {"phase": "programmer"})
        assert r.ok
        assert r.data["count"] >= 1

    def test_unknown_phase_rejected_by_enum(self):
        # The schema constrains 'phase' to the known phases, so the registry rejects an
        # out-of-enum value before the handler runs.
        r = call_tool("tools.by_phase", {"phase": "nonsense"})
        assert not r.ok
        assert r.code == "invalid_input"

    def test_missing_phase_is_invalid_input(self):
        r = call_tool("tools.by_phase", {})
        assert not r.ok
        assert r.code == "invalid_input"


class TestDescribeTool:
    def test_describe_known_tool_returns_schema(self):
        r = call_tool("tools.describe_tool", {"name": "compile.dag"})
        assert r.ok
        assert r.data["name"] == "compile.dag"
        assert "schema" in r.data
        assert r.data["schema"].get("type") == "object"
        assert "tags" in r.data
        assert "destructive" in r.data

    def test_describe_unknown_tool_gives_remedy(self):
        r = call_tool("tools.describe_tool", {"name": "nope.zzz"})
        assert not r.ok
        assert r.code == "not_found"
        # in-band remedy points the agent at the discovery tool
        assert r.remedy is not None
        assert "tools.by_phase" in r.remedy.get("suggested_tools", [])

    def test_describe_marks_destructive_tool(self):
        # compile.dag is the one tool flagged destructive (upsert path).
        r = call_tool("tools.describe_tool", {"name": "compile.dag"})
        assert r.ok
        assert r.data["destructive"] is True

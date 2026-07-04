"""
Tests for the cursus.mcp meta/discovery namespace (tools.help / tools.by_phase /
tools.describe_tool).

These exercise the agent-ergonomics discovery layer end-to-end through call_tool against
the real assembled registry — they need no engine mocking (the meta tools only read the
registry).
"""

from cursus.mcp import call_tool, get_registry


class TestHelp:
    def test_help_lists_every_tool_grouped_by_namespace(self):
        r = call_tool("tools.help", {})
        assert r.ok
        data = r.data
        # Front-door content: prose overview + phase taxonomy + full grouping.
        assert isinstance(data["overview"], str) and data["overview"]
        assert set(data["phases"]) == {"planner", "validator", "programmer"}
        # Unfiltered: every registered tool is shown exactly once across the groups.
        reg = get_registry()
        assert data["total_tools"] == len(reg)
        assert data["shown"] == len(reg)
        listed = [t["name"] for ns in data["namespaces"] for t in ns["tools"]]
        assert sorted(listed) == sorted(reg.keys())

    def test_help_includes_this_namespace_and_its_description(self):
        r = call_tool("tools.help", {})
        assert r.ok
        by_ns = {ns["namespace"]: ns for ns in r.data["namespaces"]}
        assert "tools" in by_ns
        assert by_ns["tools"]["description"]  # non-empty purpose line
        names = {t["name"] for t in by_ns["tools"]["tools"]}
        assert {"tools.help", "tools.by_phase", "tools.describe_tool"} <= names

    def test_help_namespace_filter(self):
        r = call_tool("tools.help", {"namespace": "compile"})
        assert r.ok
        assert r.data["filtered_namespace"] == "compile"
        assert [ns["namespace"] for ns in r.data["namespaces"]] == ["compile"]
        assert r.data["shown"] == r.data["namespaces"][0]["count"]

    def test_help_phase_filter_only_matching_tools(self):
        r = call_tool("tools.help", {"phase": "validator"})
        assert r.ok
        assert r.data["filtered_phase"] == "validator"
        for ns in r.data["namespaces"]:
            for t in ns["tools"]:
                assert "validator" in t["tags"]

    def test_help_include_schema_attaches_schema(self):
        r = call_tool("tools.help", {"namespace": "tools", "include_schema": True})
        assert r.ok
        for ns in r.data["namespaces"]:
            for t in ns["tools"]:
                assert t["schema"]["type"] == "object"

    def test_help_default_omits_schema(self):
        r = call_tool("tools.help", {"namespace": "tools"})
        assert r.ok
        for ns in r.data["namespaces"]:
            for t in ns["tools"]:
                assert "schema" not in t

    def test_help_unknown_namespace_gives_remedy(self):
        r = call_tool("tools.help", {"namespace": "nope"})
        assert not r.ok
        assert r.code == "not_found"
        assert "available_namespaces" in r.meta.get("details", {})
        assert r.remedy is not None

    def test_help_bad_phase_rejected_by_enum(self):
        r = call_tool("tools.help", {"phase": "nonsense"})
        assert not r.ok
        assert r.code == "invalid_input"

    def test_help_rejects_unknown_argument(self):
        r = call_tool("tools.help", {"bogus": 1})
        assert not r.ok
        assert r.code == "invalid_input"

    def test_help_next_steps_point_to_describe_and_by_phase(self):
        r = call_tool("tools.help", {})
        assert r.ok
        tools = {s["tool"] for s in r.next_steps}
        assert {"tools.describe_tool", "tools.by_phase"} <= tools


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
        # in-band remedy points the agent at the discovery tools
        assert r.remedy is not None
        suggested = r.remedy.get("suggested_tools", [])
        assert "tools.by_phase" in suggested
        assert "tools.help" in suggested

    def test_describe_marks_destructive_tool(self):
        # compile.dag is the one tool flagged destructive (upsert path).
        r = call_tool("tools.describe_tool", {"name": "compile.dag"})
        assert r.ok
        assert r.data["destructive"] is True

    def test_describe_surfaces_when_and_examples(self):
        # tools.help itself carries when + examples — describe must surface them.
        r = call_tool("tools.describe_tool", {"name": "tools.help"})
        assert r.ok
        assert r.data.get("when")
        assert r.data.get("examples")
        assert all(isinstance(ex, str) for ex in r.data["examples"])


class TestNamespaceHelp:
    """The auto-generated <ns>.help tools (one per non-meta namespace)."""

    def test_namespace_help_returns_only_that_namespace(self):
        r = call_tool("catalog.help", {})
        assert r.ok
        assert r.data["filtered_namespace"] == "catalog"
        assert [ns["namespace"] for ns in r.data["namespaces"]] == ["catalog"]

    def test_namespace_help_phase_filter(self):
        r = call_tool("validate.help", {"phase": "validator"})
        assert r.ok
        for ns in r.data["namespaces"]:
            for t in ns["tools"]:
                assert "validator" in t["tags"]

    def test_namespace_help_include_schema(self):
        r = call_tool("compile.help", {"include_schema": True})
        assert r.ok
        for ns in r.data["namespaces"]:
            for t in ns["tools"]:
                assert t["schema"]["type"] == "object"

    def test_namespace_help_rejects_namespace_arg(self):
        # <ns>.help pins its own namespace; passing one is an unknown argument.
        r = call_tool("compile.help", {"namespace": "dag"})
        assert not r.ok
        assert r.code == "invalid_input"

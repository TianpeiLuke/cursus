"""
Public MCP-server readiness guards.

These lock in the invariants that make ``cursus[mcp]`` safe and usable from real MCP
hosts (Claude Desktop, Cursor, ...):

- every externally-exposed tool name is accepted by the Anthropic/OpenAI tool-name
  pattern (``^[a-zA-Z0-9_-]{1,64}$``) — the dotted internal names are not,
- the on-the-wire ``__`` name round-trips back to the dotted name,
- the server is READ-ONLY by default: tools that mutate state or run code are gated
  behind opt-in environment variables,
- ``project.init`` cannot write outside its confinement root.
"""

from __future__ import annotations

import re

import pytest

from cursus.mcp.registry import (
    get_registry,
    get_tool,
    export_mcp_tools,
    export_openai_tools,
)

# Anthropic requires ^[a-zA-Z0-9_-]{1,128}$; OpenAI ^[a-zA-Z0-9_-]+$ with len<=64. The
# intersection every host accepts is the 64-char cap below.
HOST_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class TestWireNames:
    def test_every_exported_name_is_host_legal(self):
        for entry in export_mcp_tools():
            assert HOST_NAME.match(entry["name"]), entry["name"]
        for entry in export_openai_tools():
            assert HOST_NAME.match(entry["function"]["name"]), entry["function"]["name"]

    def test_wire_names_unique(self):
        wires = [td.wire_name for td in get_registry().values()]
        assert len(wires) == len(set(wires))

    def test_wire_name_round_trips_to_dotted(self):
        for td in get_registry().values():
            assert "." not in td.wire_name
            # both the dotted internal name and the wire name resolve to the same tool
            assert get_tool(td.name) is td
            assert get_tool(td.wire_name) is td


class TestSafeByDefaultGating:
    """The default server surface is read-only; mutating/code tools are opt-in."""

    def test_known_mutating_tools_are_gated(self, monkeypatch):
        from cursus.mcp import server

        monkeypatch.delenv("CURSUS_MCP_ENABLE_DESTRUCTIVE", raising=False)
        monkeypatch.delenv("CURSUS_MCP_ALLOW_SCRIPT_EXEC", raising=False)
        for name in (
            "compile.dag",
            "project.init",
            "dag.serialize",
            "validate.run_scripts",
        ):
            td = get_tool(name)
            assert server._gate_env(td) is not None, name
            assert server._tool_enabled(td) is False, name

    def test_gates_open_with_env(self, monkeypatch):
        from cursus.mcp import server

        monkeypatch.setenv("CURSUS_MCP_ENABLE_DESTRUCTIVE", "1")
        monkeypatch.setenv("CURSUS_MCP_ALLOW_SCRIPT_EXEC", "1")
        for name in (
            "compile.dag",
            "project.init",
            "dag.serialize",
            "validate.run_scripts",
        ):
            assert server._tool_enabled(get_tool(name)) is True, name

    def test_read_only_tools_never_gated(self, monkeypatch):
        from cursus.mcp import server

        monkeypatch.delenv("CURSUS_MCP_ENABLE_DESTRUCTIVE", raising=False)
        monkeypatch.delenv("CURSUS_MCP_ALLOW_SCRIPT_EXEC", raising=False)
        read_only = [
            td for td in get_registry().values() if server._gate_env(td) is None
        ]
        # the large majority of the surface is safe/read-only and always available
        assert len(read_only) >= 60
        assert all(server._tool_enabled(td) for td in read_only)


class TestProjectInitConfinement:
    def test_name_with_separators_rejected(self):
        from cursus.mcp.registry import call_tool

        r = call_tool("project.init", {"name": "../evil", "framework": "xgboost"})
        assert not r.ok and r.code == "invalid_input"

    def test_dotdot_target_dir_rejected(self):
        from cursus.mcp.registry import call_tool

        r = call_tool(
            "project.init",
            {"name": "ok", "framework": "xgboost", "target_dir": "../../.."},
        )
        assert not r.ok and r.code == "invalid_input"

    def test_confined_root_blocks_absolute_escape(self, tmp_path, monkeypatch):
        from cursus.mcp.registry import call_tool

        # With a confinement root set (as the server does), an absolute target_dir that
        # escapes it is refused.
        monkeypatch.setenv("CURSUS_MCP_PROJECT_ROOT", str(tmp_path))
        r = call_tool(
            "project.init",
            {
                "name": "ok",
                "framework": "xgboost",
                "target_dir": "/tmp/cursus_escape_test",
            },
        )
        assert not r.ok and r.code == "invalid_input"

    def test_confined_write_within_root_succeeds(self, tmp_path, monkeypatch):
        from cursus.mcp.registry import call_tool

        monkeypatch.setenv("CURSUS_MCP_PROJECT_ROOT", str(tmp_path))
        r = call_tool(
            "project.init",
            {"name": "demo", "framework": "xgboost", "target_dir": "sub"},
        )
        assert r.ok, r.error
        assert (tmp_path / "sub" / "demo_xgboost" / "README.md").exists()


class TestServerAnnotationsAndIsError:
    """Requires the mcp SDK; verifies host-facing annotations + isError signaling."""

    def test_tools_carry_annotations(self):
        pytest.importorskip("mcp")
        import anyio
        from cursus.mcp.server import build_server

        server = build_server()
        # invoke the registered list_tools handler
        import mcp.types as types

        handler = server.request_handlers[types.ListToolsRequest]

        async def _list():
            req = types.ListToolsRequest(method="tools/list")
            result = await handler(req)
            return result.root.tools

        tools = anyio.run(_list)
        assert tools, "server exposed no tools"
        for t in tools:
            assert t.annotations is not None
            assert t.annotations.readOnlyHint is not None

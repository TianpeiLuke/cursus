"""
Tests for ``cursus.mcp.server`` — the optional MCP stdio-server adapter.

The ``mcp`` SDK is an OPTIONAL dependency; the tool functions / schemas / registry work without it
(see the rest of tests/mcp). These tests cover the module's importable, SDK-independent surface: the
lazy-SDK guard and the actionable error it raises when the SDK is absent. When the SDK IS installed,
``build_server`` is exercised instead.
"""

import importlib.util

import pytest

from cursus.mcp import server as mcp_server

_HAS_MCP_SDK = importlib.util.find_spec("mcp") is not None


def test_module_imports_without_sdk():
    # The adapter module itself must import even when the optional SDK is missing.
    assert hasattr(mcp_server, "build_server")
    assert hasattr(mcp_server, "main")


def test_sdk_hint_is_actionable():
    # The hint names the missing package and reassures that the rest of cursus.mcp works.
    assert "mcp" in mcp_server._SDK_HINT
    assert "install" in mcp_server._SDK_HINT.lower()


@pytest.mark.skipif(_HAS_MCP_SDK, reason="mcp SDK is installed — the absent-SDK guard cannot fire")
def test_build_server_raises_actionable_error_without_sdk():
    with pytest.raises(RuntimeError) as exc:
        mcp_server.build_server()
    assert "mcp" in str(exc.value).lower()


@pytest.mark.skipif(not _HAS_MCP_SDK, reason="mcp SDK not installed")
def test_build_server_constructs_when_sdk_present():
    srv = mcp_server.build_server(name="cursus-test")
    assert srv is not None

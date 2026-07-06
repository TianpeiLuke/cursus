"""
Optional MCP server adapter for the cursus toolset.

This is a *thin* adapter: it mounts the framework-neutral tools from
:mod:`cursus.mcp.registry` onto an actual Model Context Protocol server. The official
``mcp`` Python SDK is an **optional** dependency, imported lazily here so that importing
``cursus.mcp`` (the tool functions, schemas, and registry) never requires the SDK.

Run as a stdio MCP server::

    python -m cursus.mcp.server

If the SDK is not installed, this raises a clear, actionable error pointing at the
extra to install. Everything else in ``cursus.mcp`` works without it.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from .registry import get_registry, call_tool, render_description

logger = logging.getLogger(__name__)

_SDK_HINT = (
    "The MCP server requires the optional 'mcp' SDK, which is not installed. "
    'Install it with `pip install "cursus[mcp]"` (or `pip install mcp anyio`) — the '
    "cursus.mcp tool functions, schemas, and registry work without it; only this server "
    "adapter needs it."
)


def _require_sdk():
    """Import the MCP SDK lazily, raising an actionable error if it is absent."""
    try:
        from mcp.server import Server  # type: ignore
        from mcp import types  # type: ignore

        return Server, types
    except Exception as exc:  # pragma: no cover - depends on optional dep
        raise RuntimeError(_SDK_HINT) from exc


def _cursus_version() -> str:
    """Best-effort cursus version for the MCP ``serverInfo`` (never the SDK's version).

    Without an explicit ``version``, ``Server`` reports the ``mcp`` SDK's version as the
    server's version — misleading to clients. Read the real cursus version, falling back
    to ``"0"`` if the package metadata can't be resolved (e.g. an odd source checkout).
    """
    try:
        from cursus import __version__  # type: ignore

        return str(__version__ or "0")
    except Exception:  # pragma: no cover - defensive
        return "0"


def build_server(name: str = "cursus") -> Any:
    """
    Build an MCP ``Server`` exposing every registered cursus tool.

    The server's ``list_tools`` is generated from the registry, and ``call_tool`` routes
    straight through :func:`cursus.mcp.registry.call_tool`, so the server and in-process
    callers share one code path and one result contract.
    """
    Server, types = _require_sdk()
    registry = get_registry()
    server = Server(name, version=_cursus_version())

    @server.list_tools()
    async def _list_tools():  # type: ignore[misc]
        return [
            types.Tool(
                name=td.name,
                description=render_description(td),
                inputSchema=td.schema,
            )
            for td in sorted(registry.values(), key=lambda t: t.name)
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: Dict[str, Any]):  # type: ignore[misc]
        result = call_tool(name, arguments or {})
        # MCP returns content blocks; serialize the envelope as JSON text.
        import json

        return [types.TextContent(type="text", text=json.dumps(result.to_dict()))]

    return server


def _protect_stdout_for_stdio() -> None:
    """Keep ``sys.stdout`` reserved for JSON-RPC framing.

    A stdio MCP server frames every protocol message on ``stdout``; any other byte written
    there corrupts the stream and the client fails with "Failed to parse JSONRPC message".

    The primary offender — ``sagemaker.config`` attaching a ``StreamHandler(stdout)`` and
    logging INFO the moment sagemaker is imported — is neutralized at the source in
    ``cursus/__init__.py`` (its logger is raised to WARNING before that import). This function
    is the defense-in-depth for anything ELSE that might log to stdout at runtime: it points
    our own logging at ``stderr`` and repoints any lingering stdout-bound ``StreamHandler``.
    """
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr)

    def _retarget(handler: logging.Handler) -> None:
        stream = getattr(handler, "stream", None)
        if isinstance(handler, logging.StreamHandler) and stream in (sys.stdout, sys.__stdout__):
            handler.setStream(sys.stderr)

    root = logging.getLogger()
    for h in list(root.handlers):
        _retarget(h)
    # A stdout handler may live on any named logger (not just root), so sweep the registry too.
    manager = getattr(logging.Logger, "manager", None)
    for lg in list(getattr(manager, "loggerDict", {}).values()):
        for h in list(getattr(lg, "handlers", []) or []):
            _retarget(h)


def main() -> int:
    """Entry point: run the cursus MCP server over stdio."""
    try:
        import anyio  # type: ignore
        from mcp.server.stdio import stdio_server  # type: ignore
    except Exception as exc:
        # Chain the original ImportError so the missing-module cause is preserved
        # (matches _require_sdk's `raise ... from exc` pattern).
        raise RuntimeError(_SDK_HINT) from exc

    # Scrub stdout (belt-and-suspenders; the sagemaker root-cause fix is in cursus/__init__.py)
    # then build + run the server.
    server = build_server()
    _protect_stdout_for_stdio()

    async def _run() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

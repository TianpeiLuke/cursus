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
import os
from typing import Any, Dict, Optional

from .envelope import ToolResult
from .registry import get_registry, get_tool, call_tool, render_description

logger = logging.getLogger(__name__)

# --- Safe-by-default gating ------------------------------------------------------------
# A PUBLIC MCP server is READ-ONLY by default. Tools that mutate persistent state (AWS
# upserts, filesystem writes) or execute code are neither listed nor callable unless the
# operator opts in via these environment variables.
_ENABLE_MUTATION = "CURSUS_MCP_ENABLE_DESTRUCTIVE"  # AWS upserts + filesystem writes
_ALLOW_SCRIPT_EXEC = "CURSUS_MCP_ALLOW_SCRIPT_EXEC"  # runs step scripts / pip installs


def _gate_env(td: Any) -> Optional[str]:
    """The env var that must be truthy to expose/allow this tool, or None if always safe."""
    if getattr(td, "exec_code", False):
        return _ALLOW_SCRIPT_EXEC
    if td.destructive or getattr(td, "writes", False):
        return _ENABLE_MUTATION
    return None


def _tool_enabled(td: Any) -> bool:
    gate = _gate_env(td)
    return gate is None or bool(os.getenv(gate))


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

    # Safe by default: only expose tools whose gate (if any) is enabled.
    exposed = sorted(
        (td for td in registry.values() if _tool_enabled(td)), key=lambda t: t.name
    )

    def _annotations(td: Any):
        read_only = not (
            td.destructive
            or getattr(td, "writes", False)
            or getattr(td, "exec_code", False)
        )
        return types.ToolAnnotations(
            title=td.name,  # the human-facing dotted name (the wire name uses '__')
            readOnlyHint=read_only,
            destructiveHint=bool(td.destructive),
            openWorldHint=bool(
                getattr(td, "network", False) or getattr(td, "exec_code", False)
            ),
        )

    @server.list_tools()
    async def _list_tools():  # type: ignore[misc]
        return [
            types.Tool(
                name=td.wire_name,  # host APIs reject the dotted internal name
                description=render_description(td),
                inputSchema=td.schema,
                annotations=_annotations(td),
            )
            for td in exposed
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: Dict[str, Any]):  # type: ignore[misc]
        # Enforce the gate at the boundary too (a client could call a hidden tool by name).
        td = get_tool(name)
        if td is not None and not _tool_enabled(td):
            gate = _gate_env(td)
            result: ToolResult = ToolResult.failure(
                f"tool '{td.name}' is disabled on this server; set {gate}=1 to enable it "
                f"(it mutates state or executes code)",
                code="tool_disabled",
                details={"env_var": gate},
            )
        else:
            result = call_tool(
                name, arguments or {}
            )  # get_tool maps wire -> dotted name

        # Serialize the envelope as JSON text and signal protocol-level success/failure so
        # hosts and models can tell a failed call from a successful one.
        import json

        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(result.to_dict()))],
            isError=not result.ok,
        )

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

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr
    )

    def _retarget(handler: logging.Handler) -> None:
        stream = getattr(handler, "stream", None)
        if isinstance(handler, logging.StreamHandler) and stream in (
            sys.stdout,
            sys.__stdout__,
        ):
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

    # Build the server with stdout redirected to stderr, so any import-time write (a
    # StreamHandler logging during the heavy tool-module imports) can't corrupt the
    # JSON-RPC stream before the handlers are repointed. Then install the runtime guard.
    import contextlib
    import sys

    # Confine filesystem-writing tools (e.g. project.init, when enabled) to the server's
    # working directory unless the operator picked an explicit root.
    os.environ.setdefault("CURSUS_MCP_PROJECT_ROOT", os.getcwd())

    with contextlib.redirect_stdout(sys.stderr):
        server = build_server()
    _protect_stdout_for_stdio()

    async def _run() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

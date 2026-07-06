"""
Regression test for the circular-import bug exposed while testing cursus.mcp.

`import cursus` used to emit a UserWarning ("Some Cursus features may not be available:
cannot import name 'PipelineAssembler' ... circular import") and silently fall back to
stub `compile_dag` / `PipelineDAGCompiler` that raise ImportError. The cycle was
api.dag.pipeline_dag_resolver importing core.compiler.exceptions at module scope, which
re-entered core.compiler before it finished initializing. Fixed by making that import
lazy.

This test runs the import in a clean subprocess (so module caching from other tests can't
mask the warning) and asserts the real top-level API is importable.
"""

import subprocess
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[2] / "src")


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2]),
    )


def test_import_cursus_has_no_circular_import_warning():
    # Turn the specific UserWarning into an error; a clean import exits 0.
    code = (
        f"import sys, warnings; sys.path.insert(0, {_SRC!r}); "
        "warnings.filterwarnings('error', message='Some Cursus features may not be available'); "
        "import cursus"
    )
    proc = _run(code)
    assert proc.returncode == 0, (
        "import cursus raised on the feature-availability UserWarning "
        f"(circular import regressed):\nSTDERR:\n{proc.stderr}"
    )


def test_top_level_api_is_real_not_stub():
    # If the cycle regresses, cursus.PipelineDAGCompiler resolves to the degraded stub
    # class defined inside __init__.py's except block (module 'cursus'). The real one
    # lives under cursus.core.compiler. Assert we got the real implementation.
    code = (
        f"import sys; sys.path.insert(0, {_SRC!r}); "
        "import cursus; "
        "mod = cursus.PipelineDAGCompiler.__module__; "
        "assert 'core.compiler' in mod, mod; "
        "print('ok')"
    )
    proc = _run(code)
    assert proc.returncode == 0, f"top-level API is the degraded stub:\n{proc.stderr}"
    assert "ok" in proc.stdout


def test_import_cursus_writes_nothing_to_stdout():
    # A stdio MCP server (`python -m cursus.mcp.server` / `cursus mcp serve`) frames JSON-RPC
    # on stdout, so ANY stray byte written there on import corrupts the protocol ("Failed to
    # parse JSONRPC message"). The known offender is sagemaker.config, which attaches a
    # StreamHandler(stdout) and logs INFO on import; cursus/__init__.py raises that logger to
    # WARNING before the import to prevent it. This asserts the invariant the server relies on:
    # importing cursus emits nothing on stdout. (Logging to stderr is fine and not checked here.)
    code = f"import sys; sys.path.insert(0, {_SRC!r}); import cursus"
    proc = _run(code)
    assert proc.returncode == 0, f"import cursus failed:\n{proc.stderr}"
    assert proc.stdout == "", (
        "import cursus wrote to stdout, which breaks the stdio MCP protocol.\n"
        f"STDOUT:\n{proc.stdout}"
    )

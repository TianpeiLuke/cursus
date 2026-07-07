# Installation

Cursus is published on [PyPI](https://pypi.org/project/cursus/) and requires
**Python 3.9 or newer** (tested on 3.9 through 3.12).

## Install

```bash
pip install cursus
```

This installs the core pipeline-authoring engine: the DAG model, the compiler, the
step catalog, the registry, the pipeline catalog, the CLI, and the MCP tool surface.
The 1.x/2.x line targets the **SageMaker Python SDK 2.x**.

## Optional extras

Cursus keeps heavy ML/data libraries out of the core install and groups them behind
extras, so you only pull in what your pipeline actually runs:

```bash
pip install "cursus[processing]"   # pandas / numpy data-processing utilities
pip install "cursus[nlp]"          # tokenizers / transformers for text steps
pip install "cursus[all]"          # everything
```

Install the docs toolchain (to build this site locally) with:

```bash
pip install "cursus[docs]"
```

## Verify

```bash
cursus --version
python -c "import cursus; print(cursus.__version__)"
```

Both should print the same version (Cursus derives `__version__` from the installed
package metadata, so they can never drift).

## Optional: the MCP server

Cursus ships a framework-neutral **agent tool surface** (70 tools). To expose it over
the Model Context Protocol for an LLM agent, install the optional MCP SDK and launch the
stdio server:

```bash
pip install "cursus[mcp]"          # pulls the MCP SDK (mcp, anyio)

cursus-mcp                         # stable launch command, or:
python -m cursus.mcp.server        # equivalent module entry point, or:
cursus mcp serve                   # equivalent via the CLI
cursus mcp help                    # inspect the tools without starting the server
```

Point your MCP host at `cursus-mcp`. **Use the absolute path** (`which cursus-mcp`) — GUI
hosts like Claude Desktop and Cursor don't inherit your shell `PATH`, so a bare
`"cursus-mcp"` usually fails with `ENOENT` — and pass any needed AWS env explicitly, since
hosts don't inherit your shell env either:

```json
{
  "mcpServers": {
    "cursus": {
      "command": "/absolute/path/to/cursus-mcp",
      "args": [],
      "env": { "AWS_PROFILE": "default", "AWS_REGION": "us-east-1" }
    }
  }
}
```

The server is **read-only by default**; opt into state-changing tools with
`CURSUS_MCP_ENABLE_DESTRUCTIVE=1` (filesystem writes + AWS upserts) or
`CURSUS_MCP_ALLOW_SCRIPT_EXEC=1` (running step scripts) in that `env` block. Per-host config
locations, the safety matrix, and troubleshooting are in the
[MCP server README](https://github.com/TianpeiLuke/cursus/blob/main/src/cursus/mcp/README.md);
the [MCP Tool Reference](../reference/generated/mcp_tools.md) lists the full toolset.

## AWS setup

To *upsert* or *run* a compiled pipeline (not just compile it locally), you need AWS
credentials and a SageMaker execution role, exactly as for any SageMaker SDK usage —
configure them via the standard `aws configure` / environment variables / instance
role. Compilation itself (`cursus compile` without `--upsert`) is fully offline.

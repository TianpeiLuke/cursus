# Cursus MCP Server

`cursus.mcp` exposes the cursus pipeline engine as a **framework-neutral tool surface** —
~70 JSON-in / JSON-out tools across 12 namespaces (catalog, dag, config, compile, validate,
execdoc, pipeline_catalog, project, strategies, steps, author, tools) — and mounts them on a
[Model Context Protocol](https://modelcontextprotocol.io) **stdio server** so any MCP host
(Claude Desktop, Cursor, Kiro, Continue, …) can drive cursus.

## Install

```bash
pip install "cursus[mcp]"      # adds the MCP SDK (mcp>=1.2, anyio) on top of cursus
```

The SDK is imported lazily, so a plain `pip install cursus` (compile pipelines only) never
pulls it in. Verify the launcher resolves:

```bash
cursus mcp help                # inspect the tools without starting a server
which cursus-mcp               # -> the absolute path you'll put in your host config
```

## Launch commands (all equivalent, all stdio)

```bash
cursus-mcp                     # console script (recommended)
python -m cursus.mcp.server    # module entry point
cursus mcp serve               # via the CLI
```

## Wiring it into a host

> **Use the absolute path to `cursus-mcp`.** GUI-launched hosts (Claude Desktop, Cursor)
> do **not** inherit your shell `PATH`, so a bare `"command": "cursus-mcp"` usually fails
> with `ENOENT`. Get the path with `which cursus-mcp` (venv installs live under the venv's
> `bin/`). Likewise hosts don't inherit your shell env — pass `AWS_PROFILE` / `AWS_REGION`
> explicitly if any tool needs AWS.

**Claude Desktop** — `~/Library/Application Support/Claude/claude_desktop_config.json`
(macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

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

**Cursor** — `~/.cursor/mcp.json` (global) or `.cursor/mcp.json` (per-project): same
`mcpServers` shape as above.

**Kiro / generic `mcp.json`:**

```json
{
  "mcpServers": {
    "cursus": {
      "command": "/absolute/path/to/cursus-mcp",
      "args": [],
      "env": { "AWS_PROFILE": "default", "AWS_REGION": "us-east-1" },
      "disabled": false
    }
  }
}
```

After editing, fully restart the host. It should list the cursus tools; start the model on
`tools.help` (namespaced `<ns>.help` tools exist too, e.g. `compile.help`).

> Tool names are exposed with `__` in place of the internal dot (e.g. `catalog__list_steps`),
> because host tool-calling APIs reject `.` in names. The human-facing help still shows the
> dotted names; both forms are accepted when calling.

## Safety: read-only by default

To be safe under an LLM, the public server is **read-only by default**. Tools that mutate
state or run code are neither listed nor callable unless you opt in with environment
variables (add them to the `env` block above):

| Env var | Enables | Tools |
|---|---|---|
| *(none)* | read-only surface (~66 tools) | catalog / dag / config / validate (analysis) / help / … |
| `CURSUS_MCP_ENABLE_DESTRUCTIVE=1` | filesystem writes + AWS upserts | `compile.dag` (upsert), `project.init`, `dag.serialize` |
| `CURSUS_MCP_ALLOW_SCRIPT_EXEC=1` | running step scripts locally (may `pip install`) | `validate.run_scripts` |
| `CURSUS_MCP_PROJECT_ROOT=/path` | confinement root for `project.init` writes (default: the server's working dir) | — |

Every tool also carries MCP **annotations** (`readOnlyHint` / `destructiveHint` /
`openWorldHint`) so a host can auto-approve read-only calls and prompt before mutating ones.
Failures are returned with the protocol-level `isError` flag set (not a silent success).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Host shows "server failed" / `spawn cursus-mcp ENOENT` | Bare command not on the host's `PATH`. Use the absolute path from `which cursus-mcp`. |
| Client error: `Failed to parse JSONRPC message` | Something wrote to stdout. cursus reserves stdout for JSON-RPC and routes logs to stderr; if you see this, a plugin/tool is printing — report it. |
| A write/deploy/run tool is "missing" | It's gated. Set the matching env var above and restart the host. |
| SageMaker tools fail with credential/region errors | The host didn't inherit your shell env. Add `AWS_PROFILE` / `AWS_REGION` (and any `AWS_*`) to the `env` block. |
| `RuntimeError: The MCP server requires the optional 'mcp' SDK` | Run `pip install "cursus[mcp]"` in the *same* environment `cursus-mcp` resolves to. |

See the [MCP tool reference](https://cursus.readthedocs.io/en/latest/reference/generated/mcp_tools.html)
for the full toolset.

# Reference

Generated reference material for Cursus's structured surfaces. **These pages are emitted
from live package data at every build** (by `docs/_ext/gen_reference.py`), so they never
drift from the installed version.

```{toctree}
:maxdepth: 1

generated/mcp_tools
generated/step_catalog
generated/pipeline_catalog
```

- **[MCP Tool Reference](generated/mcp_tools.md)** — the 70-tool agent surface, grouped by
  namespace, with each tool's description, when-to-use, examples, and parameters.
- **[Step Interface Catalog](generated/step_catalog.md)** — every declarative
  `<step>.step.yaml`: dependencies, outputs, env vars, and job arguments.
- **[Pipeline Catalog](generated/pipeline_catalog.md)** — the pre-built shared DAGs,
  filterable by framework, task type, and complexity.

For the code-level surfaces see the [Python API reference](../api/index.rst) and the
[CLI reference](../cli.rst).

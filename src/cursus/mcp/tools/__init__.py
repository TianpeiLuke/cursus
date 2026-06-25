"""
Tool namespaces for ``cursus.mcp``.

Each module in this package defines one namespace of agent-callable tools and exposes a
module-level ``TOOLS: List[ToolDef]`` that the registry collects. Namespaces:

- ``catalog``          — discover/search steps, configs, builders (step_catalog + registry).
- ``dag``              — construct/validate/serialize pipeline DAGs (api.dag).
- ``config``           — schema-driven config generation (api.factory + core.config_fields).
- ``compile``          — compile/validate/preview DAG → SageMaker pipeline (core.compiler).
- ``validate``         — alignment + dependency + script-execution checks (validation, core.deps).
- ``execdoc``          — MODS execution-document generation (mods.exe_doc).
- ``pipeline_catalog`` — recommend/select/load pre-built shared DAGs (pipeline_catalog).
"""

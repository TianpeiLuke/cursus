"""
Build-time reference generator for the Cursus docs.

Wired to Sphinx's ``builder-inited`` event (see ``docs/conf.py``), this emits three
Markdown (MyST) reference pages under ``docs/reference/generated/`` from their LIVE
self-describing sources, so the pages never drift from the code:

- ``mcp_tools.md``       <- ``cursus.mcp.registry`` (namespaces + ToolDef when/examples/schema)
- ``step_catalog.md``    <- ``steps/interfaces/*.step.yaml`` (declarative I/O contracts)
- ``pipeline_catalog.md``<- ``pipeline_catalog/shared_dags/catalog_index.json``

Each surface is generated independently and defensively: a failure in one emits a
placeholder page + a Sphinx warning rather than breaking the whole build.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger("sphinx.cursus")

_OUT_SUBDIR = Path("reference") / "generated"


# --------------------------------------------------------------------------- helpers
def _pkg_root() -> Path:
    import cursus

    return Path(cursus.__file__).resolve().parent


def _md_escape(text) -> str:
    """Make a value safe for a single Markdown table cell."""
    if text is None:
        return ""
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def _fence(lines) -> str:
    body = "\n".join(lines) if isinstance(lines, (list, tuple)) else str(lines)
    return "```text\n" + body + "\n```"


# --------------------------------------------------------------------------- MCP tools
def _gen_mcp_tools() -> str:
    from cursus.mcp.registry import get_namespaces, list_tools, render_description

    namespaces = get_namespaces()
    all_tools = list_tools()
    n_examples = sum(len(getattr(t, "examples", None) or []) for t in all_tools)

    out = []
    out.append("# MCP Tool Reference\n")
    out.append(
        f"Cursus exposes a framework-neutral, JSON-in/JSON-out **agent tool surface** of "
        f"**{len(all_tools)} tools** across **{len(namespaces)} namespaces** "
        f"(with **{n_examples}** worked examples). This page is generated from the live "
        f"`cursus.mcp.registry`, so it always matches the installed toolset.\n"
    )
    out.append(
        "Invoke tools programmatically with `cursus.mcp.registry.call_tool(name, args)`, "
        "over the MCP server (`python -m cursus.mcp.server`), or inspect them with "
        "`cursus mcp help` / the `tools.help` tool.\n"
    )

    by_ns = {}
    for t in all_tools:
        by_ns.setdefault(t.name.split(".", 1)[0], []).append(t)

    for ns in sorted(by_ns):
        out.append(f"\n## `{ns}`\n")
        desc = namespaces.get(ns)
        if desc:
            out.append(f"{desc}\n")
        for t in sorted(by_ns[ns], key=lambda x: x.name):
            out.append(f"\n### `{t.name}`\n")
            body = (render_description(t) or t.description or "").strip()
            if body:
                out.append(body + "\n")
            meta = []
            if getattr(t, "destructive", False):
                meta.append("**destructive**")
            if getattr(t, "tags", None):
                meta.append("tags: " + ", ".join(f"`{x}`" for x in t.tags))
            if meta:
                out.append("*(" + " · ".join(meta) + ")*\n")
            # Input schema (top-level params)
            schema = getattr(t, "schema", None) or {}
            props = (schema or {}).get("properties") or {}
            required = set((schema or {}).get("required") or [])
            if props:
                out.append("**Parameters**\n")
                out.append("| name | type | required | description |")
                out.append("|---|---|---|---|")
                for pname, pspec in props.items():
                    pspec = pspec or {}
                    ptype = pspec.get("type") or (
                        "enum" if pspec.get("enum") else ""
                    )
                    out.append(
                        f"| `{_md_escape(pname)}` | {_md_escape(ptype)} | "
                        f"{'yes' if pname in required else 'no'} | "
                        f"{_md_escape(pspec.get('description'))} |"
                    )
                out.append("")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- step catalog
def _gen_step_catalog() -> str:
    import yaml

    idir = _pkg_root() / "steps" / "interfaces"
    files = sorted(idir.glob("*.step.yaml"))

    out = []
    out.append("# Step Interface Catalog\n")
    out.append(
        f"Cursus ships **{len(files)} declarative step interfaces** "
        f"(`steps/interfaces/*.step.yaml`). Each step is one YAML file that unifies the "
        f"script **contract** (input/output paths, env vars, CLI job arguments) with the "
        f"**spec** (typed dependencies and outputs used for DAG dependency resolution). "
        f"Load one at runtime with `load_step_interface(\"<StepType>\")`.\n"
    )

    steps = []
    for f in files:
        try:
            steps.append((f, yaml.safe_load(f.read_text()) or {}))
        except Exception as exc:  # pragma: no cover
            logger.warning("step_catalog: could not parse %s: %s", f.name, exc)

    # Summary table
    out.append("## Summary\n")
    out.append("| Step | Node type | SageMaker step | Deps | Outputs |")
    out.append("|---|---|---|---|---|")
    for f, d in steps:
        st = d.get("step_type") or f.stem
        reg = d.get("registry") or {}
        spec = d.get("spec") or {}
        out.append(
            f"| [`{_md_escape(st)}`](#{str(st).lower()}) | "
            f"{_md_escape(d.get('node_type'))} | "
            f"{_md_escape(reg.get('sagemaker_step_type'))} | "
            f"{len(spec.get('dependencies') or {})} | "
            f"{len(spec.get('outputs') or {})} |"
        )
    out.append("")

    for f, d in steps:
        st = d.get("step_type") or f.stem
        reg = d.get("registry") or {}
        con = d.get("contract") or {}
        spec = d.get("spec") or {}
        out.append(f"\n## {st}\n")
        if con.get("description"):
            out.append(f"{_md_escape(con.get('description'))}\n")
        facts = []
        if d.get("node_type"):
            facts.append(f"node type `{d['node_type']}`")
        if reg.get("sagemaker_step_type"):
            facts.append(f"SageMaker `{reg['sagemaker_step_type']}`")
        if reg.get("config_class"):
            facts.append(f"config `{reg['config_class']}`")
        if con.get("entry_point"):
            facts.append(f"entry point `{con['entry_point']}`")
        if facts:
            out.append("- " + " · ".join(facts) + "\n")

        deps = spec.get("dependencies") or {}
        if deps:
            out.append("**Dependencies (inputs)**\n")
            out.append("| logical name | required | data type | compatible sources |")
            out.append("|---|---|---|---|")
            for name, info in deps.items():
                info = info or {}
                srcs = info.get("compatible_sources") or []
                out.append(
                    f"| `{_md_escape(name)}` | "
                    f"{'yes' if info.get('required') else 'no'} | "
                    f"{_md_escape(info.get('data_type'))} | "
                    f"{_md_escape(', '.join(srcs) if srcs else '')} |"
                )
            out.append("")

        outs = spec.get("outputs") or {}
        if outs:
            out.append("**Outputs**\n")
            out.append("| logical name | data type | property path |")
            out.append("|---|---|---|")
            for name, info in outs.items():
                info = info or {}
                out.append(
                    f"| `{_md_escape(name)}` | {_md_escape(info.get('data_type'))} | "
                    f"`{_md_escape(info.get('property_path'))}` |"
                )
            out.append("")

        env = con.get("env_vars") or {}
        req_env = env.get("required") or []
        opt_env = env.get("optional") or {}
        if req_env or opt_env:
            parts = []
            if req_env:
                parts.append("required: " + ", ".join(f"`{e}`" for e in req_env))
            if opt_env:
                keys = list(opt_env.keys()) if isinstance(opt_env, dict) else list(opt_env)
                parts.append("optional: " + ", ".join(f"`{e}`" for e in keys))
            out.append("**Environment variables** — " + "; ".join(parts) + "\n")

        jargs = con.get("job_arguments") or []
        if jargs:
            flags = [a.get("flag") for a in jargs if isinstance(a, dict) and a.get("flag")]
            if flags:
                out.append("**Job arguments** — " + ", ".join(f"`{fl}`" for fl in flags) + "\n")

    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- pipeline catalog
def _gen_pipeline_catalog() -> str:
    idx_path = _pkg_root() / "pipeline_catalog" / "shared_dags" / "catalog_index.json"
    idx = json.loads(idx_path.read_text())
    dags = idx.get("dags") or []
    frameworks = idx.get("frameworks") or sorted({d.get("framework") for d in dags})

    out = []
    out.append("# Pipeline Catalog\n")
    out.append(
        f"Cursus ships **{idx.get('total_dags', len(dags))} pre-built shared pipeline DAGs** "
        f"across **{len(frameworks)}** frameworks "
        f"({', '.join(f'`{fw}`' for fw in frameworks)}). "
        f"Discover and build them with the data-driven router:\n"
    )
    out.append(
        "```python\n"
        "from cursus.pipeline_catalog import recommend_dag, load_shared_dag\n"
        "from cursus import PipelineDAGCompiler\n\n"
        "# recommend_dag returns a ranked list of matches (dicts with 'id', 'score', ...)\n"
        "recommendations = recommend_dag(framework=\"xgboost\", task_type=\"end_to_end\")\n"
        "dag = load_shared_dag(recommendations[0][\"id\"])\n\n"
        "pipeline, report = PipelineDAGCompiler(config_path=\"config.json\").compile_with_report(dag)\n"
        "```\n"
    )

    out.append("## All DAGs\n")
    out.append("| DAG id | framework | task type | complexity | nodes | description |")
    out.append("|---|---|---|---|---|---|")
    for d in sorted(dags, key=lambda x: (x.get("framework") or "", x.get("id") or "")):
        out.append(
            f"| `{_md_escape(d.get('id'))}` | {_md_escape(d.get('framework'))} | "
            f"{_md_escape(d.get('task_type'))} | {_md_escape(d.get('complexity'))} | "
            f"{_md_escape(d.get('node_count'))} | {_md_escape(d.get('description'))} |"
        )
    out.append("")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- driver
_SURFACES = [
    ("mcp_tools.md", "MCP tool reference", _gen_mcp_tools),
    ("step_catalog.md", "step interface catalog", _gen_step_catalog),
    ("pipeline_catalog.md", "pipeline catalog", _gen_pipeline_catalog),
]


def _generate(app):
    out_dir = Path(app.srcdir) / _OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, label, fn in _SURFACES:
        target = out_dir / filename
        try:
            target.write_text(fn(), encoding="utf-8")
            logger.info("cursus: generated %s (%s)", filename, label)
        except Exception as exc:  # pragma: no cover - degrade, never fail the build
            logger.warning("cursus: FAILED to generate %s (%s): %s", filename, label, exc)
            target.write_text(
                f"# {label.title()}\n\n"
                f"> **Note:** this page could not be generated during the docs build "
                f"(`{type(exc).__name__}: {exc}`). It is emitted from live package data at "
                f"build time; see `docs/_ext/gen_reference.py`.\n",
                encoding="utf-8",
            )


def register(app):
    app.connect("builder-inited", _generate)

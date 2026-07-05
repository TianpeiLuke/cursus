# Execution Documents

An **execution document** is a JSON object that carries the runtime configuration a MODS
pipeline needs to actually *run* on top of a compiled SageMaker pipeline. Where a
[compiled pipeline](../concepts/dag_and_compilation.md) describes the *structure* of the
work (which steps exist and how they connect), the execution document supplies the extra
per-step payloads that certain steps require at execution time — for example the Cradle
data-load request for a data-loading step, or the model-registration payload for a
registration step.

Cursus generates these documents with a standalone component,
`ExecutionDocumentGenerator` (`src/cursus/mods/exe_doc/generator.py`). It takes a
`PipelineDAG` plus a serialized configuration file and fills in the runtime payloads for
the steps that need them. The same generator is exposed through the `cursus exec-doc` CLI
and the `execdoc.*` [MCP tools](../reference/generated/mcp_tools.md).

The generator is deliberately **independent from the pipeline generation system**: the
filling logic it runs was ported out of the dynamic pipeline template
(`DynamicPipelineTemplate._fill_*_configurations`, noted in the code as maintaining "exact
logic equivalence"). That means you can produce an execution document from just a DAG and a
config file — without compiling a full SageMaker pipeline first — and still get the same
per-step payloads the in-pipeline path would have produced.

This page explains:

- the shape of an execution document (`PIPELINE_STEP_CONFIGS`, `STEP_TYPE`,
  `STEP_CONFIG`);
- how `ExecutionDocumentGenerator` loads configs and dispatches to per-step *helpers*;
- the `cursus exec-doc generate` CLI;
- the `execdoc.*` MCP tools (`generate`, `template`, `validate`, `merge`);
- the `project_root` / `anchor_file` "caller hook" plumbing and why it matters for
  exec-doc-only flows.

---

## What an execution document looks like

An execution document is a plain dictionary with a single top-level key,
`PIPELINE_STEP_CONFIGS`, mapping each pipeline step name to a small object with two
fields:

```json
{
  "PIPELINE_STEP_CONFIGS": {
    "CradleDataLoading-Training": {
      "STEP_TYPE": ["PROCESSING_STEP", "CradleDataLoading"],
      "STEP_CONFIG": { }
    },
    "ModelRegistration-NA": {
      "STEP_TYPE": ["PROCESSING_STEP", "ModelRegistration"],
      "STEP_CONFIG": { }
    }
  }
}
```

- **`STEP_TYPE`** — a list of type tags MODS uses to classify the step (a base
  `PROCESSING_STEP` tag plus a more specific tag such as `CradleDataLoading` or
  `ModelRegistration`).
- **`STEP_CONFIG`** — the runtime payload for that step. It starts empty and is
  *filled in* by the generator for the steps that need it. Steps that do not require an
  execution-time payload keep an empty `STEP_CONFIG`.

The minimal validity rule is enforced by
`validate_execution_document_structure` (`src/cursus/mods/exe_doc/utils.py`): the document
must be a dict and must contain a `PIPELINE_STEP_CONFIGS` mapping. An empty
`PIPELINE_STEP_CONFIGS` is still structurally valid.

You can scaffold a blank document from a list of step names with
`create_execution_document_template(step_names)`, which produces one entry per step with a
default `STEP_TYPE` of `["PROCESSING_STEP"]` and an empty `STEP_CONFIG`.

---

## Which steps get filled

Most steps in a DAG need *nothing* in their execution document — their configuration is
baked into the compiled pipeline. Only a handful of step families carry a runtime payload,
and each is handled by a dedicated **helper** that subclasses
`ExecutionDocumentHelper` (`src/cursus/mods/exe_doc/base.py`). The generator wires up four
helpers:

| Helper | Module | Handles |
| --- | --- | --- |
| `CradleDataLoadingHelper` | `cradle_helper.py` | Cradle data-loading steps |
| `RegistrationHelper` | `registration_helper.py` | Model registration steps |
| `DataUploadingHelper` | `data_uploading_helper.py` | Data uploading steps |
| `RedshiftDataLoadingHelper` | `redshift_data_loading_helper.py` | Redshift data-loading steps |

Every helper implements two abstract methods from `ExecutionDocumentHelper`:

- `can_handle_step(step_name, config)` — whether this helper is responsible for a given
  step/config. For example `CradleDataLoadingHelper` returns `True` when the config is a
  `CradleDataLoadingConfig` (or, as a fallback, when the config class name contains
  `cradle` + `data` + `load`).
- `extract_step_config(step_name, config)` — build the `STEP_CONFIG` payload from the
  step's config object.

Helpers also expose `get_execution_step_name(step_name, config)`, which maps a DAG node
name to the step name used inside the execution document. Cradle's mapping, for instance,
turns `CradleDataLoading_training` into `CradleDataLoading-Training` (it strips the
`job_type` suffix and re-appends it capitalized after a hyphen), so the generated payload
lands under the key MODS expects.

---

## How generation works

`ExecutionDocumentGenerator.fill_execution_document(dag, execution_document)` is the core
entry point. Its flow:

1. **Guard the document shape.** If `PIPELINE_STEP_CONFIGS` is missing, it logs a warning
   and returns the document unchanged.
2. **Identify relevant steps.** For each DAG node it resolves the config (via the
   `StepConfigResolver`) and asks whether *any* helper `can_handle_step`. Steps no helper
   claims are skipped entirely.
3. **Dispatch per helper type.** It filters the relevant steps by helper and, only when
   matching steps exist, runs the corresponding fill routine:
   `_fill_cradle_configurations`, `_fill_registration_configurations`,
   `_fill_data_uploading_configurations`, and `_fill_redshift_configurations`.
4. **Write payloads.** Each fill routine looks up the execution step name, checks that key
   exists in `PIPELINE_STEP_CONFIGS`, calls the helper's `extract_step_config`, and stores
   the result under `STEP_CONFIG` (adding `STEP_TYPE` when missing).

Any failure is wrapped in `ExecutionDocumentGenerationError`. Individual helper extraction
failures are logged as warnings and skipped, so one bad step does not abort the whole
document.

Relevance is decided in `_identify_relevant_steps` via `_is_execution_doc_relevant`: it
asks each helper `can_handle_step`, and — as a safety net — also treats a config whose
class name contains `cradle` or `registration` as relevant. Only when at least one matching
step exists does the generator run the corresponding fill routine, so the common case (a
DAG of steps that need no payload) does almost no work.

### Configs feed generation

The generator is constructed with a **config file path**, not with live config objects:

```python
from cursus.mods.exe_doc.generator import ExecutionDocumentGenerator

generator = ExecutionDocumentGenerator(config_path="config.json")
print(generator.configs.keys())  # loaded config instances, keyed by name
```

At construction time `_load_configs` calls `load_configs`
(`src/cursus/steps/configs/utils.py`) to deserialize the saved config set — the same
JSON format produced when a pipeline's configs are saved (see
[The Configuration System](../concepts/config_system.md)). During filling, each DAG node
is matched to one of these configs. Matching is delegated to the step-catalog
`StepConfigResolver` (the `StepConfigResolverAdapter` from
`src/cursus/step_catalog/adapters/config_resolver.py`, via `resolve_config_for_step`); when
that fails the generator falls back to a direct name match and then a fuzzy `_names_match`
(word-overlap) heuristic.

Registration is a special case: rather than matching one config per node, the generator
scans all loaded configs for a registration config (plus optional *payload* and *package*
configs) and calls
`RegistrationHelper.create_execution_doc_config_with_related_configs(...)` to build the
combined payload.

### AWS access is optional

The constructor accepts an optional `sagemaker_session` (a `PipelineSession`) and `role`
(IAM role ARN). These are passed through for helpers that need AWS access. Without them,
generation still runs, but helpers that depend on AWS may produce limited payloads — the
MCP tool surfaces this as a warning (see below).

---

## The `cursus exec-doc` CLI

The CLI group is registered as `exec-doc` (`src/cursus/cli/exec_doc_cli.py`,
wired in `src/cursus/cli/__init__.py`). It has one command, `generate`:

```bash
# Basic usage: serialized DAG + config file, default output execution_doc.json
cursus exec-doc generate -d dag.json -c config.json

# Custom output path
cursus exec-doc generate -d dag.json -c config.json -o my_exec_doc.json

# Start from an existing base template instead of auto-generating one
cursus exec-doc generate -d dag.json -c config.json --template base_template.json

# YAML output
cursus exec-doc generate -d dag.json -c config.json --format yaml

# Supply an IAM role for helpers that need AWS access
cursus exec-doc generate -d dag.json -c config.json --role arn:aws:iam::123456789012:role/MyRole
```

### Options

| Option | Required | Purpose |
| --- | --- | --- |
| `--dag-file`, `-d` | yes | Path to a serialized DAG JSON file. |
| `--config-file`, `-c` | yes | Path to the configuration JSON file. |
| `--output`, `-o` | no | Output path (default `execution_doc.json`). |
| `--template` | no | Base execution-document template to fill instead of auto-generating one. |
| `--format` | no | `json` (default) or `yaml`. |
| `--role` | no | IAM role ARN for AWS operations. |
| `--project-root` | no | Project folder anchoring source-dir resolution (see below). |
| `--anchor-file` | no | A file inside the project folder; its parent is used as the project root. |
| `--verbose`, `-v` | no | Debug-level logging and per-step summaries. |

### What the command does

1. Loads the DAG with `import_dag_from_json`
   (`src/cursus/api/dag/pipeline_dag_serializer.py`).
2. Either loads the `--template` file or auto-generates a base template with one
   `{"STEP_CONFIG": {}, "STEP_TYPE": []}` entry per DAG node.
3. Constructs an `ExecutionDocumentGenerator(config_path=config_file, role=..., project_root=..., anchor_file=...)`.
4. Calls `fill_execution_document(dag, execution_document)`.
5. Writes the result as JSON or YAML and prints a summary (total steps, steps with a
   filled `STEP_CONFIG`).

See the full [CLI reference](../cli.rst) for how `exec-doc` sits alongside the other
command groups.

---

## The `execdoc.*` MCP tools

The same functionality is available to agents through the `execdoc.*` MCP namespace
(`src/cursus/mcp/tools/execdoc.py`). The module defines four tools (the registry also adds
a generic `execdoc.help`); only `execdoc.generate` touches configs/AWS, the rest are pure
JSON document operations. See the
[MCP tools reference](../reference/generated/mcp_tools.md) for the full catalog.

| Tool | Purpose |
| --- | --- |
| `execdoc.generate` | Fill an execution document from a DAG + config file. |
| `execdoc.template` | Build an empty template for a list of step names. |
| `execdoc.validate` | Check a document has a well-formed `PIPELINE_STEP_CONFIGS`. |
| `execdoc.merge` | Merge two documents, `additional_doc` winning on conflicts. |

### `execdoc.generate`

Requires `config_path`. The DAG is supplied either as `dag_file` (a serialized JSON path)
or inline via `dag` (`{"nodes": [...], "edges": [[src, dst], ...]}`). If no
`execution_document` is passed, a template is auto-generated from the DAG node names.

```json
{"config_path": "config.json", "dag_file": "pipeline_dag.json"}
```

```json
{
  "config_path": "config.json",
  "dag": {
    "nodes": ["TabularPreprocessing", "XGBoostTraining"],
    "edges": [["TabularPreprocessing", "XGBoostTraining"]]
  }
}
```

The tool returns the filled document plus metadata: `node_count`, `config_count`,
`config_names`, `step_count`, `steps_with_config`, and `auto_template`. If no `role` is
given and the document was auto-generated, it adds a warning that AWS-dependent helpers may
produce limited step configs.

### `execdoc.template`, `execdoc.validate`, `execdoc.merge`

These wrap the corresponding functions in `src/cursus/mods/exe_doc/utils.py`:

- `execdoc.template` → `create_execution_document_template(step_names)`.
- `execdoc.validate` → `validate_execution_document_structure(doc)`, returning
  `{"valid": bool, "issues": [...]}`.
- `execdoc.merge` → `merge_execution_documents(base_doc, additional_doc)`. On merge,
  matching steps' `STEP_CONFIG` dicts are combined (with `additional_doc` values winning),
  and steps present only in `additional_doc` are added.

A typical agent flow is `execdoc.template` → `execdoc.generate` → `execdoc.validate`, or
`execdoc.merge` to layer generated payloads onto a hand-authored base.

---

## `project_root` / `anchor_file`: the caller hook

Step configs carry source-directory fields (`source_dir`, `processing_source_dir`) that
must be resolved to real paths. When a pipeline is compiled through
`PipelineDAGCompiler`, the compiler pushes the project root before configs are loaded, so
those relative paths resolve correctly. But an **exec-doc-only** flow (the CLI or MCP tool)
runs the generator *without* a compiler first — so the generator has to establish that
anchor itself. This is the **caller hook** (referred to in the code as *Strategy 0*).

`ExecutionDocumentGenerator.__init__` resolves the anchor before loading configs:

- **`project_root`** — an absolute path to the user's project *folder*. Highest priority.
- **`anchor_file`** — a *file* inside the project folder; its parent directory becomes the
  project root. Pass `__file__` from a template module for a self-documenting form.
  Equivalent to `project_root=Path(__file__).parent`.
- **Fallback** — if neither is given, the root is *inferred from the config file's
  location*.

Precedence is: explicit `project_root` > `anchor_file` > config-anchored inference. If both
`project_root` and `anchor_file` are given and disagree, `project_root` wins. This logic is
not duplicated — `_resolve_project_root` delegates to
`PipelineDAGCompiler._resolve_project_root`, so the anchor precedence is identical to the
compiler's. Once resolved, the generator calls
`set_project_root` (`src/cursus/core/utils/hybrid_path_resolution`) so that configs loaded
immediately afterward resolve their source dirs against it.

```python
# Explicit project folder
ExecutionDocumentGenerator(config_path="config.json", project_root="/path/to/project")

# Self-documenting anchor from a template module
ExecutionDocumentGenerator(config_path="config.json", anchor_file=__file__)
```

Both the CLI (`--project-root` / `--anchor-file`) and `execdoc.generate`
(`project_root` / `anchor_file` arguments) forward straight into this constructor.

> **Tip.** If your generated `STEP_CONFIG` payloads reference the wrong script paths, an
> incorrect or missing project-root anchor is the usual cause. Pass `--project-root`
> (CLI) or `project_root` (MCP) explicitly to pin it.

---

## See also

- [The Configuration System](../concepts/config_system.md) — how the config file consumed
  by the generator is built and serialized.
- [DAG & Compilation](../concepts/dag_and_compilation.md) — how the DAG and compiler
  produce the pipeline the execution document accompanies.
- [Step interfaces](../concepts/step_interfaces.md) — the step configs and builders the
  helpers extract payloads from.
- [CLI reference](../cli.rst) and [MCP tools reference](../reference/generated/mcp_tools.md).

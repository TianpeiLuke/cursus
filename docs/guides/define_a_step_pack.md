# Define a Step Pack

A **step pack** is a folder of your own pipeline steps that lives *outside* the
installed `cursus` package. You author the step artifacts once, drop them in a
directory, point the compiler at that directory, and Cursus discovers your steps as
if they were built in — no fork, no vendored copy, no edit to the package source.

This is the task-oriented recipe. It walks the whole loop end to end:

1. Lay out the pack directory (`interfaces/` + `configs/` + `scripts/`).
2. Point the compiler at it — explicitly (`workspace_dirs`) or by anchor (`anchor_file`).
3. Verify discovery — via `get_registry_health()` and the `cursus` CLI.
4. Scaffold a pack-ready project with the `project.init` MCP tool.

For *why* packs work the way they do — the additive invariant, the registry-merge
mechanics, interface-resolution ordering — read the companion concept page,
[Step packs](../concepts/step_packs.md). For authoring a single step's three
artifacts in depth, see the [Step pack tutorial](../tutorials/step_pack.md).

> **Prerequisite:** you should already be able to compile a DAG with
> `PipelineDAGCompiler`. See [DAG and compilation](../concepts/dag_and_compilation.md).

---

## Step 1 — Lay out the pack directory

A pack is a directory holding three subdirectories, one per discovery source. The
minimum layout for a single custom step named `AcmeScoring`:

```
my_project/
└── step_pack/
    ├── interfaces/
    │   └── acme_scoring.step.yaml       # the step interface (registry + contract + spec)
    ├── configs/
    │   └── config_acme_scoring_step.py  # class AcmeScoringConfig(...)
    └── scripts/
        └── acme_scoring.py              # the step's entry-point script
```

Each subdirectory maps to exactly one discovery source:

| Subdirectory                | Cursus scans for                                                                                       | Discovered by                                              |
| --------------------------- | ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------- |
| `interfaces/`               | `*.step.yaml` files — the step's `registry:` block, contract, and spec                                 | `refresh_registry` / the interface loader                  |
| `configs/`                  | `*.py` config classes (`<Name>Config`, or a `BaseModel` / `BasePipelineConfig` / `ProcessingStepConfigBase` subclass) | `ConfigAutoDiscovery`                                      |
| `scripts/`                  | the step's entry-point script                                                                          | `StepCatalog` component discovery                          |
| `hyperparams/` *(optional)* | `*.py` hyperparameter classes (`<Name>Hyperparameters`, or a `ModelHyperparameters` subclass)          | `ConfigAutoDiscovery`                                      |

You do **not** ship a per-step builder module — under the current design builders are
synthesized from the interface. So `interfaces/` + `configs/` + `scripts/` is the
complete set of files you author.

### The interface file (`interfaces/*.step.yaml`)

The `.step.yaml` is the single source of truth for the step. Its `step_type` is the
canonical (PascalCase) step name, and its `registry:` block is what gets merged into
the step-name registry:

```yaml
# step_pack/interfaces/acme_scoring.step.yaml
step_type: AcmeScoring
node_type: internal
registry:
  sagemaker_step_type: Processing      # required — no fallback
  description: Acme domain scoring step
contract:
  entry_point: acme_scoring.py
  inputs:
    DATA:
      path: /opt/ml/processing/input/data
      required: true
  outputs:
    scored_data:
      path: /opt/ml/processing/output
```

`sagemaker_step_type` has no default: if you omit it the loader raises a
`ValueError` naming the file, so a misconfigured pack fails loudly instead of being
silently dropped. The `config_class` (`AcmeScoringConfig`) and `builder_step_name`
(`AcmeScoringStepBuilder`) are derived from the step name by convention.

### The config class (`configs/*.py`)

A class following the `<Name>Config` convention (or inheriting a known base). Because
the file is not under the package root, `ConfigAutoDiscovery` finds it with AST
parsing and imports it **by file location** under a unique, path-hashed synthetic
module name — so two packs that each ship a `config_..._step.py` never collide in
`sys.modules`:

```python
# step_pack/configs/config_acme_scoring_step.py
from cursus.core.base.config_base import BasePipelineConfig

class AcmeScoringConfig(BasePipelineConfig):
    threshold: float = 0.5
```

`ConfigAutoDiscovery.discover_config_classes()` matches a class when it either
inherits `BasePipelineConfig` / `ProcessingStepConfigBase` / `BaseModel`, **or** its
name ends in `Config` / `Configuration`. Hyperparameter classes (in an optional
`hyperparams/` dir) match by inheriting `ModelHyperparameters` / `BaseModel` or by a
`Hyperparameters` / `Hyperparams` name suffix.

### The script (`scripts/*.py`)

The entry-point named in the interface's `contract.entry_point`. It is indexed as the
step's `script` component and used at build time.

---

## Step 2 — Point the compiler at the pack

`PipelineDAGCompiler` gives you two ways to attach a pack. Precedence, top to bottom:
explicit `workspace_dirs` > derived from `project_root` > derived from `anchor_file` >
package-only.

### Option A — explicit `workspace_dirs`

Pass one directory (or a list). Each entry is a pack root — the directory that holds
`interfaces/` + `configs/` + `scripts/`:

```python
from cursus.core.compiler.dag_compiler import PipelineDAGCompiler

compiler = PipelineDAGCompiler(
    config_path="pipeline_config/config_NA.json",
    workspace_dirs="/abs/path/to/my_project/step_pack",   # str, Path, or list
)
```

When `workspace_dirs` is given it wins over any auto-derivation. On construction the
compiler:

1. Calls `refresh_registry(<pack>/interfaces)` for each pack, merging the pack's
   `.step.yaml` rows into the step registry (add-only — see Step 3).
2. Builds the `StepCatalog` with `workspace_dirs=[...]` so the pack's components are
   indexed as native.
3. Pushes the dirs as the process-level default via `set_default_workspace_dirs(...)`,
   so a bare `StepCatalog()` created elsewhere (validation, exec-doc generation) also
   sees the plugin steps.

### Option B — derived from `anchor_file` / `project_root`

Most pipelines already pass the **caller hook** — `anchor_file=__file__` (or
`project_root=Path(__file__).parent`) — so Cursus can resolve docker `source_dir`
paths against the project folder. That same anchor doubles as the pack anchor. Omit
`workspace_dirs` and the compiler derives a pack from the resolved project root:

```python
compiler = PipelineDAGCompiler(
    config_path="pipeline_config/config_NA.json",
    anchor_file=__file__,   # project folder = this file's directory
)
```

The derivation (`_derive_step_pack_dir`) checks, in order:

1. `<project_root>/step_pack` — if it has an `interfaces/` subdir, that is the pack.
2. `<project_root>` itself — if *it* has an `interfaces/` subdir.

The first candidate that actually contains an `interfaces/` directory wins; if
neither does, discovery is package-only. This is why, in a scaffolded project,
dropping a `step_pack/interfaces/` folder next to your `run_pipeline.py` is enough —
the `anchor_file=__file__` you already pass makes the pack discoverable with no extra
argument.

> **Note on paths:** pass **absolute** paths (or an `anchor_file=__file__` that
> resolves to one). `workspace_dirs` entries are expanded and resolved to absolute
> paths internally, but relative strings are resolved against the current working
> directory, which is easy to get wrong.

### Referencing the step from your DAG

Your DAG node name resolves to the pack step's config the same way any node does.
Name the node after the step (or after its config), then preview before compiling:

```python
from cursus.api.dag.base_dag import PipelineDAG

dag = PipelineDAG()
dag.add_node("acme_scoring")
# ... add the rest of your nodes and edges ...

preview = compiler.preview_resolution(dag)
print(preview.node_config_map)   # {"acme_scoring": "AcmeScoringConfig", ...}

pipeline = compiler.compile(dag)
```

`node_config_map` maps each DAG node to the **config class name**
(`type(config).__name__`, e.g. `AcmeScoringConfig`) that the resolver matched it
to — not the canonical step type. An unresolved node maps to the literal string
`"UNRESOLVED"`.

Because discovery is additive, this exact code compiles unchanged whether or not the
pack is present. With no pack the compiler is package-only and behaves as before.

---

## Step 3 — Verify discovery

You can confirm a pack was picked up without compiling a full pipeline. Use whichever
surface fits — a quick Python check, or the CLI.

### The registry merge, and collisions

When a pack is attached, `refresh_registry(pack_interfaces_dir)` merges its rows into
the live registry **add-only**: package steps are always present, and a pack step
whose name already exists in the package *shadows* it with a `WARNING` (plugin-wins)
rather than replacing anything else. It returns a dict of collisions:

```python
from cursus.registry.step_names import refresh_registry, get_step_names

collisions = refresh_registry("/abs/path/to/my_project/step_pack/interfaces")
# {} means every pack step is new; {"XGBoostTraining": "collision"} means a shadow

names = get_step_names()
assert "AcmeScoring" in names          # your pack step has a registry row
assert "XGBoostTraining" in names      # package steps are still there
```

You normally don't call `refresh_registry` yourself — the compiler does it — but any
collision it records is also readable later through the registry **health report**:

```python
from cursus.registry.step_names import get_registry_health

health = get_registry_health()
print(health["pack_collisions"])   # {} when clean; {"XGBoostTraining": "collision"} on a clash
print(health["hybrid_active"])     # False (+ health["init_error"]) if the registry fell back
```

Wire `pack_collisions` into monitoring so a pack silently shadowing a core step never
goes unnoticed. The fix for a reported collision is simply to rename the pack step so
it no longer clashes with a package name.

### The step catalog

The pack step is indexed as a native catalog entry:

```python
from cursus.step_catalog.step_catalog import StepCatalog

catalog = StepCatalog(workspace_dirs=["/abs/path/to/my_project/step_pack"])
assert catalog.get_step_info("AcmeScoring") is not None
assert catalog.get_step_info("XGBoostTraining") is not None       # package step still indexed
```

And the out-of-package config was imported by file location:

```python
assert "AcmeScoringConfig" in catalog.config_discovery.discover_config_classes()
```

### From the CLI

The `cursus` CLI reads the same catalog and registry. These commands are handy for a
quick eyeball check (see the full [CLI reference](../cli.rst)):

```bash
# List catalog steps (your pack step appears in the list)
cursus catalog list

# Inspect one step: workspace, components, framework
cursus catalog show AcmeScoring

# Show which component files were discovered for the step
cursus catalog components AcmeScoring

# Show the step's I/O connection view (inputs, outputs, property refs)
cursus steps io AcmeScoring

# Registry view, including any name conflicts
cursus registry list-steps --conflicts-only
```

> The bare `cursus catalog ...` commands build a package-scoped catalog. To have the
> CLI *itself* see a pack you must run inside a process where the pack dirs are the
> process default (e.g. after a compile that attached them). The most reliable
> discovery check for a fresh pack is the Python snippet above with an explicit
> `workspace_dirs=[...]`, or the `preview_resolution` call in Step 2.

---

## Step 4 — Scaffold a pack-ready project

The `project.init` MCP tool lays down a phase-0 project whose entry files already
pass `anchor_file=__file__`. That anchor is exactly what the compiler uses to derive a
`step_pack/`, so a scaffolded project is pack-ready out of the box.

```jsonc
// project.init arguments
{ "name": "secure_delivery", "framework": "xgboost" }
// -> projects/secure_delivery_xgboost/
```

`project.init` requires `name` (snake_case base name) and `framework` (one of
`xgboost`, `pytorch`, `lightgbmmt`, `bedrock`); optional `target_dir` (default
`projects`) and `overwrite`. It writes a fixed skeleton — a region-agnostic
`run_pipeline.py`, the `@MODSTemplate` deployment class that loads
`pipeline_config/dag.json`, a `generate_config.py` skeleton, an empty `dag.json`
stub, the folder tree with per-folder READMEs, and a root `README.md` action-item
ledger. Both generated entry files build the compiler with the caller hook:

```python
from cursus.core.compiler.dag_compiler import PipelineDAGCompiler

compiler = PipelineDAGCompiler(
    config_path=self.config_path,
    sagemaker_session=self.sagemaker_session,
    role=self.execution_role,
    anchor_file=__file__,   # doubles as the step-pack anchor
)
```

To make a scaffolded project host a pack, create `step_pack/interfaces/` (plus
`configs/`, `scripts/`) beside the generated `run_pipeline.py` and drop your three
artifacts in — the pack is then picked up with no further wiring.

For a brand-new step *type* that does not yet exist in the registry, the scaffold's
ledger points you at the `/cursus-author-step` workflow to author the `.step.yaml` +
config + script — the very artifacts you then place into your pack.

### End-to-end bring-up

`project.init` is scaffold-only. To bring a whole project up (scaffold → seed/author a
DAG → generate config), use `project.bring_up`, which returns the invocation for the
`cursus-new-project` orchestrator. That workflow composes:

1. **Scaffold** — `cursus-init-project` (the phase-0 skeleton + ledger).
2. **SeedDAG** — for `dag_source="catalog"`, recommend + load a shared catalog DAG
   into `pipeline_config/dag.json`; for `dag_source="manual"`, stop for a human to
   author the DAG.
3. **GateDAG** — refuse to proceed on an empty/invalid DAG.
4. **Configure** — `cursus-configure-pipeline` fills `generate_config.py` and writes
   `config_<region>.json`.

See the [MCP tools reference](../reference/generated/mcp_tools.md) for `project.init`
and `project.bring_up`, and the [Pipeline catalog](../reference/generated/pipeline_catalog.md)
for the DAGs the catalog branch can seed.

---

## Troubleshooting

| Symptom                                                            | Likely cause / fix                                                                                                                     |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Pack step never appears in the registry                           | The compiler didn't attach the pack. Confirm `workspace_dirs`/`anchor_file` resolves to the dir that contains `interfaces/` (not its parent). |
| `preview_resolution` shows the node as `UNRESOLVED`               | The DAG node name doesn't match the step or its config. Rename the node, or add an explicit config key / `metadata.config_types` entry. |
| Interface loads but the config class is missing                   | Config file naming/base doesn't match `ConfigAutoDiscovery` rules (`<Name>Config`, or a known base). Check the import doesn't raise (a failed import is logged and skipped, never fatal). |
| `ValueError` naming a `.step.yaml`                                | The `registry:` block is missing `sagemaker_step_type` (no fallback). Add it.                                                          |
| A package step disappeared after adding a pack                    | It can't — packs are add-only. If a *pack* step shadowed a package name, check `get_registry_health()["pack_collisions"]` and rename the pack step. |
| `get_registry_health()["hybrid_active"]` is `False`               | The workspace-aware registry manager fell back to the static registry; read `init_error` for the captured exception.                  |

---

## Reference

- Compiler — `cursus.core.compiler.dag_compiler.PipelineDAGCompiler`
  (`workspace_dirs`, `anchor_file`, `project_root`).
- Registry merge / health — `cursus.registry.step_names.refresh_registry`,
  `get_registry_health`, `get_step_names`.
- Config / hyperparameter discovery —
  `cursus.step_catalog.config_discovery.ConfigAutoDiscovery`.
- Catalog — `cursus.step_catalog.step_catalog.StepCatalog`, `set_default_workspace_dirs`.
  See the [step catalog reference](../reference/generated/step_catalog.md).
- Scaffolding — the `project.init` / `project.bring_up` MCP tools; see the
  [MCP tools reference](../reference/generated/mcp_tools.md).
- Concept & deep dive — [Step packs](../concepts/step_packs.md) ·
  [Step pack tutorial](../tutorials/step_pack.md) ·
  [Registry and discovery](../concepts/registry_and_discovery.md).

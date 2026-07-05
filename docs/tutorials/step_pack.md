# Ship Your Own Steps as a Step Pack

*New in Cursus 2.8.0.*

A **step pack** lets you define your own pipeline steps in a folder that lives
**outside** the installed `cursus` package — no fork, no vendored copy, no edit to
the package source — and have Cursus discover them as if they were native steps.
You point the compiler at the folder (or let it derive the folder from your project
anchor), and your steps show up in the registry and the step catalog alongside the
built-in ones.

The feature is built on one hard rule, the **additive invariant**: the steps that
ship inside the package are *always* available. A pack can only **add** steps. It
can never remove or replace a package step. On a deliberate name clash a pack step
shadows the package name — but only with a warning, and every other package step
stays untouched. With no pack active at all, the registry and catalog are
byte-identical to package-only behavior.

This tutorial shows the directory layout, how the compiler picks a pack up, how the
registry merge works, and how to verify the invariant holds.

> Prerequisites: you should already be comfortable compiling a DAG with
> `PipelineDAGCompiler`. See [Concepts](../concepts/index.md) and the
> [Author a step](author_a_step.md) tutorial for how a single step is defined.

---

## Why a step pack?

Cursus ships a fixed set of steps (XGBoost training, tabular preprocessing, model
evaluation, and so on). Sometimes your project needs a step the package does not
have — a domain-specific processing step, a bespoke training routine, a new
transform. Before 2.8.0 the only options were to fork the package or to vendor a
copy and edit it.

A step pack removes that friction. You author the same three artifacts a package
step has — an **interface** (`.step.yaml`), a **config class**, and a **script** —
but you keep them in your own project folder. Cursus discovers them at compile time.

---

## The directory layout

A pack is a directory that contains an `interfaces/` subdirectory (its `.step.yaml`
files) plus `configs/` and `scripts/`. The minimum layout for one custom step named
`AcmeScoring` looks like this:

```
my_project/
└── step_pack/
    ├── interfaces/
    │   └── acme_scoring.step.yaml     # the step interface (registry + contract + spec)
    ├── configs/
    │   └── config_acme_scoring_step.py  # class AcmeScoringConfig(...)
    └── scripts/
        └── acme_scoring.py            # the processing/training script
```

Each subdirectory maps to a discovery source:

| Subdirectory  | What Cursus scans for                                          | Discovered by |
| ------------- | ------------------------------------------------------------- | ------------- |
| `interfaces/` | `*.step.yaml` files — the step's `registry:` block, contract, spec | `refresh_registry` / the interface loader |
| `configs/`    | `*.py` config classes (`<Name>Config`, or a `BaseModel`/`BasePipelineConfig`/`ProcessingStepConfigBase` subclass) | `ConfigAutoDiscovery` |
| `scripts/`    | the step's entry-point script                                 | `StepCatalog` component discovery |
| `hyperparams/` *(optional)* | `*.py` hyperparameter classes (`<Name>Hyperparameters` or `ModelHyperparameters` subclass) | `ConfigAutoDiscovery` |

You do **not** ship a per-step builder module. Under the current design builders are
synthesized from the interface, so `interfaces/` + `configs/` + `scripts/` is the
complete set of files you author.

### The interface file

The `.step.yaml` is the single source of truth for the step. Its top-level
`step_type` is the canonical (PascalCase) step name, and its `registry:` block is
what gets merged into the step-name registry:

```yaml
# step_pack/interfaces/acme_scoring.step.yaml
step_type: AcmeScoring
node_type: internal
registry:
  sagemaker_step_type: Processing
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

From the `registry:` block Cursus derives the full registry row. Only
`sagemaker_step_type` and `description` are read verbatim; the rest follow
convention (and can be overridden in the `registry:` block):

| Registry field      | Value                                                        |
| ------------------- | ------------------------------------------------------------ |
| `spec_type`         | the step name (`AcmeScoring`) — always equal to `step_type`  |
| `config_class`      | `AcmeScoringConfig` (`<Name>Config` by convention)           |
| `builder_step_name` | `AcmeScoringStepBuilder` (`<Name>StepBuilder` by convention) |
| `sagemaker_step_type` | read from the `registry:` block (required)                 |
| `description`       | read from the `registry:` block                              |

Because `sagemaker_step_type` has no fallback, omitting it from the `registry:`
block raises a `ValueError` naming the file — a fast, explicit failure rather than a
silent drop.

### The config class

The config module lives in `configs/` and defines a class that follows the
`<Name>Config` naming convention (or inherits a known base). Cursus finds it with
AST parsing, then imports it. Crucially, because the file is *not* under the package
root, Cursus imports it **by file location** (`importlib.util.spec_from_file_location`)
under a unique, path-hashed synthetic module name — so two packs that each ship a
`config_..._step.py` never collide in `sys.modules`:

```python
# step_pack/configs/config_acme_scoring_step.py
from cursus.core.base.config_base import BasePipelineConfig

class AcmeScoringConfig(BasePipelineConfig):
    threshold: float = 0.5
```

---

## Letting the compiler discover the pack

There are two ways the `PipelineDAGCompiler` picks up a pack: you name it
explicitly, or the compiler derives it from your project anchor.

### Explicit: `workspace_dirs=...`

Pass one directory (or a list) as `workspace_dirs`. Each entry is a pack root — the
directory that holds `interfaces/` + `configs/` + `scripts/`:

```python
from cursus.core.compiler.dag_compiler import PipelineDAGCompiler

compiler = PipelineDAGCompiler(
    config_path="pipeline_config/config_NA.json",
    workspace_dirs="/abs/path/to/my_project/step_pack",
)
```

When `workspace_dirs` is given it wins over any auto-derivation. The compiler:

1. Calls `refresh_registry(<pack>/interfaces)` for each pack so the pack's
   `.step.yaml` rows are merged into the step registry.
2. Constructs the `StepCatalog` with `workspace_dirs=[...]` so the pack's
   components (configs, scripts, interfaces) are indexed as native.
3. Pushes the dirs as the process-level default via
   `set_default_workspace_dirs(...)`, so even a bare `StepCatalog()` created
   elsewhere (validation, exec-doc generation) sees the plugin steps too.

### Derived: `anchor_file` / `project_root`

Most pipelines already pass the **caller hook** — `anchor_file=__file__` (or
`project_root=Path(__file__).parent`) — so Cursus can resolve docker `source_dir`
paths against the project folder. That same anchor doubles as the pack anchor. When
you omit `workspace_dirs`, the compiler derives a pack from the resolved project
root:

```python
compiler = PipelineDAGCompiler(
    config_path="pipeline_config/config_NA.json",
    anchor_file=__file__,   # project folder = this file's directory
)
```

The derivation rule (`_derive_step_pack_dir`) checks, in order:

1. `<project_root>/step_pack` — if it has an `interfaces/` subdir, that is the pack.
2. `<project_root>` itself — if *it* has an `interfaces/` subdir (a project whose
   own interfaces live at its root).

The first candidate that actually contains an `interfaces/` directory wins. If
neither does, the result is `None` and discovery is package-only. This is why, in a
scaffolded project, dropping a `step_pack/interfaces/` folder next to your
`run_pipeline.py` is enough — the `anchor_file=__file__` you already pass makes the
pack discoverable with no extra argument.

Precedence, end to end: explicit `workspace_dirs` > derived-from-`project_root` >
derived-from-`anchor_file` > package-only.

---

## What `refresh_registry` does

`refresh_registry(pack_interfaces_dir)` in `cursus.registry.step_names` is the public
entry point that merges a pack into the live registry. You rarely call it yourself —
the compiler calls it for you — but it is worth understanding what it guarantees.

```python
from cursus.registry.step_names import refresh_registry

collisions = refresh_registry("/abs/path/to/my_project/step_pack/interfaces")
```

Its mechanism is **package-first, never replace**:

1. It derives *only the pack's* registry rows from `pack_interfaces_dir/*.step.yaml`
   (via `build_registry_from_interfaces`), dropping the interface-less `_EXTRAS`
   rows (`Base` / `Processing` / `HyperparameterPrep`) that are package concerns.
2. It layers those rows **on top** of the live package table with
   `merge_pack_registry` — an in-place `STEP_NAMES.update(...)`, so the package rows
   are preserved and import-time references to `STEP_NAMES` stay live.
3. It registers the pack's `interfaces/` with the interface loader (searched *after*
   the package dir), re-syncs the hybrid registry manager (`reload_core_registry`)
   so `get_step_names()` — and therefore the `StepCatalog` — sees the plugin steps,
   and refreshes the module-level snapshot globals.

It returns a dict of collisions: `{name: "collision"}` for any pack step whose name
already existed in the package registry. An empty dict means every pack step is
genuinely new. Passing `None`, or a directory that does not exist, is a safe no-op
that returns `{}`.

### Registry merge, visually

```
package STEP_NAMES         pack rows (.step.yaml)
──────────────────         ──────────────────────
XGBoostTraining            AcmeScoring
TabularPreprocessing
XGBoostModelEval
  ...
        │                          │
        └──────────  merge  ───────┘
                     (STEP_NAMES.update, in place)
                          │
                          ▼
        XGBoostTraining, TabularPreprocessing,
        XGBoostModelEval, ..., AcmeScoring   ← package rows kept, pack added on top
```

---

## The additive invariant, precisely

The invariant is the contract that makes step packs safe to enable anywhere. It has
three parts, each locked by a regression test in
`tests/step_catalog/test_plugin_pack_additive_invariant.py`:

1. **A pack only adds.** After merging a pack with one new step, exactly that one
   step appears and every package step is present and byte-for-byte unchanged. A
   pack that ships *only* its own step does not drop `XGBoostTraining`,
   `TabularPreprocessing`, or any other package step.

2. **A name clash shadows with a warning — nothing else is lost.** If a pack ships a
   step whose name already exists in the package (say `XGBoostTraining`), the pack
   value shadows it (plugin-wins) and a `WARNING` is logged. Every *other* package
   step is still present. The clash is also recorded so monitoring can see it (see
   below). The recommended fix is simply to rename the pack step so it does not
   shadow a core name.

3. **No pack means package-only.** With no pack active, the live registry equals the
   package-derived table exactly, and the golden-snapshot drift gate
   (`tests/registry/step_names_registry_snapshot.json`) re-derives from the package
   interfaces alone via `build_registry_from_interfaces()` — so an active pack can
   never trip drift detection.

### Interface resolution ordering

The interface loader (`cursus.steps.interfaces`) enforces the same rule at the file
level. Its search roots are the **package interfaces dir first**, then any
registered pack dirs. So on a name clash a *package* interface always wins during
loading; a pack interface is used only for names the package does not own (or when a
pack deliberately shadows). Registering a new pack dir invalidates the interface
cache so the pack's `.step.yaml` is picked up without a restart.

---

## Verifying discovery

You can confirm a pack was discovered without compiling a full pipeline.

**Registry** — the step now has a row:

```python
from cursus.registry.step_names import get_step_names, refresh_registry

refresh_registry("/abs/path/to/my_project/step_pack/interfaces")
names = get_step_names()
assert "AcmeScoring" in names
assert "XGBoostTraining" in names          # package steps still there
```

**Catalog** — the step is indexed as native:

```python
from pathlib import Path
from cursus.step_catalog.step_catalog import StepCatalog

# Continues the session above: refresh_registry has already merged the pack's registry
# row, which is what lets the catalog resolve the pack step by name. Pass Path objects —
# config discovery joins each dir with `/ "configs"`, so a bare string is not scanned.
catalog = StepCatalog(workspace_dirs=[Path("/abs/path/to/my_project/step_pack")])
assert catalog.get_step_info("AcmeScoring") is not None
assert catalog.get_step_info("XGBoostTraining") is not None
```

**Config class** — the out-of-package config was imported by file location:

```python
config_classes = catalog.config_discovery.discover_config_classes()
assert "AcmeScoringConfig" in config_classes
```

### Surfacing collisions

If a pack step shadowed a package name, the clash is recorded and readable via the
registry health report — useful to wire into monitoring so a silent shadow of a core
step does not go unnoticed:

```python
from cursus.registry.step_names import get_registry_health

health = get_registry_health()
print(health["pack_collisions"])   # {} when clean; {"XGBoostTraining": "collision"} on a clash
```

`get_registry_health()` also reports `hybrid_active` (and `init_error` if the
workspace-aware registry manager fell back to the static registry).

---

## Scaffolding a project that hosts a pack

The `project.init` MCP tool (and the `cursus-new-project` orchestrator it feeds)
scaffolds a phase-0 project whose entry files already pass `anchor_file=__file__`.
That anchor is exactly what the compiler uses to derive a `step_pack/`, so a
scaffolded project is pack-ready out of the box: create a `step_pack/interfaces/`
(plus `configs/`, `scripts/`) beside the generated `run_pipeline.py`, and the pack
is picked up with no further wiring.

The generated `@MODSTemplate` deployment class and `run_pipeline.py` both build the
compiler like this:

```python
from cursus.core.compiler.dag_compiler import PipelineDAGCompiler

compiler = PipelineDAGCompiler(
    config_path=self.config_path,
    sagemaker_session=self.sagemaker_session,
    role=self.execution_role,
    anchor_file=__file__,   # doubles as the step-pack anchor
)
```

For a brand-new step *type* that does not yet exist in the registry, the scaffold's
action-item ledger points you at the `/cursus-author-step` workflow to author the
`.step.yaml` + config + script — the very artifacts you then drop into your pack.

See the [MCP tools reference](../reference/generated/mcp_tools.md) for `project.init`
and `project.bring_up`, and the [CLI reference](../cli.rst) for the corresponding
commands.

---

## Putting it together

A minimal end-to-end flow:

```python
from pathlib import Path
from cursus.core.compiler.dag_compiler import PipelineDAGCompiler
from cursus.api.dag.base_dag import PipelineDAG

# 1. A DAG that references your custom step by its node name.
dag = PipelineDAG()
dag.add_node("acme_scoring")
# ... add the rest of your nodes and edges ...

# 2. Point the compiler at the project folder (which contains step_pack/).
compiler = PipelineDAGCompiler(
    config_path="pipeline_config/config_NA.json",
    anchor_file=__file__,          # derives <project>/step_pack automatically
)

# 3. Preview resolution — your pack step should resolve to AcmeScoringConfig.
preview = compiler.preview_resolution(dag)
print(preview.node_config_map)     # {"acme_scoring": "AcmeScoringConfig", ...}
                                   # values are config CLASS names (type(config).__name__)

# 4. Compile.
pipeline = compiler.compile(dag)
```

Because discovery is additive, this same code compiles unchanged whether or not the
pack is present — with no `step_pack/` folder, the compiler is package-only and
behaves exactly as before.

---

## Reference

- Registry merge — `cursus.registry.step_names.refresh_registry`,
  `cursus.registry.step_names_base.merge_pack_registry`,
  `cursus.registry.step_names.get_registry_health`.
- Config/hyperparameter discovery — `cursus.step_catalog.config_discovery.ConfigAutoDiscovery`.
- Interface loading — `cursus.steps.interfaces` (`register_pack_interface_dir`,
  `list_available_interfaces`, `clear_interface_cache`).
- Compiler — `cursus.core.compiler.dag_compiler.PipelineDAGCompiler`
  (`workspace_dirs`, `anchor_file`, `project_root`).
- Catalog — `cursus.step_catalog.step_catalog.StepCatalog`,
  `set_default_workspace_dirs`. See the
  [step catalog reference](../reference/generated/step_catalog.md).
- Regression tests — `tests/step_catalog/test_plugin_pack_additive_invariant.py`.
- [API reference](../api/index.rst) · [Tutorials index](index.md)

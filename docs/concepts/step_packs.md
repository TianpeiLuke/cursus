# Step Packs: External Steps

A **step pack** is a consumer-owned folder of pipeline steps that lives *outside* the
pip-installed `cursus` package and is discovered as if it were native. It lets a team add
its own steps — a new preprocessing step, a proprietary training step, a custom model
evaluation — without forking cursus, vendoring a copy of the package, or editing any file
under `cursus/`.

This capability landed in **2.8.0**. Its central guarantee is the **additive invariant**:

> The steps that ship inside the `cursus` package are ALWAYS available. An external pack is a
> strictly **additive** overlay — it can only ADD steps (and, on a deliberate name clash,
> shadow one with a warning). It can never remove or replace a package step. With no pack
> active, the registry and the step catalog are byte-identical to package-only.

If you are new to how cursus turns a DAG into a pipeline, read
[Compilation](../concepts/dag_and_compilation.md) and
[Registry and discovery](../concepts/registry_and_discovery.md) first — step packs plug into
exactly those two subsystems.

```{contents}
:local:
:depth: 2
```

## The shape of a pack

A pack mirrors the layout the package itself uses. It is a directory containing three
sibling subfolders:

```
my_project/
  step_pack/
    interfaces/      # one <step>.step.yaml per step (the source of truth)
    configs/         # config_<step>_step.py — the step's Pydantic config class
    scripts/         # <step>.py — the runtime script the step executes
```

- **`interfaces/*.step.yaml`** — the unified step interface. Each file carries a
  `step_type`, a `registry:` block (`sagemaker_step_type`, `description`, and the derived
  config/builder/spec names), and the contract/spec sections. This is the same
  `.step.yaml` format the package ships; see [Step interfaces](../concepts/step_interfaces.md).
- **`configs/config_<step>_step.py`** — a config class that either inherits a known base
  (`BasePipelineConfig`, `ProcessingStepConfigBase`, or Pydantic `BaseModel`) or follows the
  `<Name>Config` / `<Name>Configuration` naming convention.
- **`scripts/<step>.py`** — the executable body of the step.

A minimal one-step pack, taken from the regression fixture in
`tests/step_catalog/test_plugin_pack_additive_invariant.py`:

```yaml
# step_pack/interfaces/additive_probe_step.step.yaml
step_type: AdditiveProbeStep
registry:
  sagemaker_step_type: Processing
  description: plugin step AdditiveProbeStep
```

```python
# step_pack/configs/config_additive_probe_step_step.py
from pydantic import BaseModel

class AdditiveProbeStepConfig(BaseModel):
    field: str = "x"
```

Because the layout is exactly the package layout, the same discovery machinery — AST config
scanning, `.step.yaml` interface loading, registry derivation — works on a pack without any
special cases. A pack is just "another search root."

## The discovery anchor

Nothing about a pack is hard-wired to a path. cursus finds it through the **caller hook**:
the pipeline entry point tells cursus where the project lives, and the pack is derived from
there. There are three ways to supply the anchor, in priority order, all on
[`PipelineDAGCompiler`](../concepts/dag_and_compilation.md).

### 1. Explicit `workspace_dirs` (highest precedence)

Pass the pack root(s) directly. Each must be a directory holding `interfaces/` +
`configs/` + `scripts/`.

```python
from cursus.core.compiler.dag_compiler import PipelineDAGCompiler

compiler = PipelineDAGCompiler(
    config_path="configs/my_pipeline.json",
    workspace_dirs=["/abs/path/to/my_project/step_pack"],
)
```

`workspace_dirs` accepts a single `str`/`Path` or a list. When present, it wins over any
derived pack.

### 2. The project-root caller hook + auto-derivation

More commonly you give cursus the *project*, not the pack, and let it derive the pack. Pass
`project_root` (a project **directory**) or the self-documenting `anchor_file` (a **file**
inside the project — pass `__file__`):

```python
compiler = PipelineDAGCompiler(
    config_path="configs/my_pipeline.json",
    anchor_file=__file__,   # equivalent to project_root=Path(__file__).parent
)
```

`anchor_file=__file__` and `project_root=Path(__file__).parent` are equivalent; both are
normalized by `resolve_anchor` in `core/utils/hybrid_path_resolution.py`. If both are given
and disagree, `project_root` wins and a warning is logged
(`PipelineDAGCompiler._resolve_project_root`).

The same anchor is *also* the highest-priority "Strategy 0" input for resolving a step's
`source_dir` / `processing_source_dir` across deployment modes — it is pushed process-wide
via `set_project_root`, so configs resolve against it without needing the
`CURSUS_PROJECT_BASE` env var or a `project_root_folder` field. The pack discovery and the
path resolution share one anchor.

### 3. Config-anchored fallback

If you pass neither `project_root` nor `anchor_file`, the project root is inferred from
`config_path`: cursus walks up from the config file, treating a
`pipeline_config` / `pipeline_configs` parent directory as the config dir and using its
parent as the project root; otherwise it uses the config file's own directory.

### `_derive_step_pack_dir`

Once the project root is known, `PipelineDAGCompiler._derive_step_pack_dir(project_root)`
turns it into a pack directory by convention:

```python
# simplified from dag_compiler.py
for candidate in (root / "step_pack", root):
    if (candidate / "interfaces").is_dir():
        return str(candidate)
return None   # no interfaces/ anywhere -> package-only
```

It checks `<project_root>/step_pack` first, then `<project_root>` itself (for a project
whose own `interfaces/` sit at its root), and returns the first directory that actually
contains an `interfaces/` subfolder. If neither exists, it returns `None` and cursus stays
package-only. `_resolve_workspace_dirs` wraps this: explicit `workspace_dirs` win, otherwise
it falls back to the derived pack.

The upshot: **a caller usually needs only the anchor** (`anchor_file=__file__`), and a
conventionally-laid-out pack is found automatically.

## What the compiler does with a pack

When `PipelineDAGCompiler.__init__` resolves a non-empty `workspace_dirs`, it wires the pack
into every subsystem that needs to know about it:

1. **Register the pack in the registry.** For each pack dir it calls
   `refresh_registry(Path(pack_dir) / "interfaces")` (from `cursus.registry.step_names`).
   This merges the pack's `.step.yaml` rows into the live step registry so its steps get a
   registry row (config class, builder name, spec type, SageMaker type).
2. **Publish a process-wide default.** It calls `set_default_workspace_dirs(workspace_dirs)`
   so that bare `StepCatalog()` construction sites elsewhere (validation, authoring,
   execution-doc generation) also see the pack. `get_default_workspace_dirs()` in
   `step_catalog/step_catalog.py` returns this list when a catalog is built with no explicit
   `workspace_dirs`.
3. **Build a pack-aware catalog.** The compiler's own `StepCatalog` is constructed with
   `workspace_dirs=[Path(d) for d in self.workspace_dirs]`, so it indexes the pack's steps
   alongside the package's.

The registry refresh (1) and the process-wide default push (2) are each best-effort, wrapped
in `try/except` so a pack problem there never blocks a compile. The catalog (3) is a plain
`StepCatalog(...)` construction that builds its index lazily and records any discovery error
in `metrics["index_build_error"]` rather than raising — so a bad pack step degrades to an
empty/partial index instead of aborting the compile.

## Merging into the registry: `refresh_registry` / `merge_pack_registry`

The registry side is where the additive invariant is enforced. Two functions collaborate.

### `refresh_registry(pack_interfaces_dir)`

Defined in `cursus/registry/step_names.py`, this is the public entry point. Given the pack's
`interfaces/` directory it:

1. Derives the pack's registry rows from `pack_interfaces_dir/*.step.yaml` via
   `build_registry_from_interfaces(interfaces_dir=pack_dir)`.
2. Drops the interface-less `_EXTRAS` rows (`Base` / `Processing` / `HyperparameterPrep`)
   that `build_registry_from_interfaces` always seeds — those are package concerns, already
   present, so they are not "pack" rows.
3. Layers the remaining rows on top of the live package table with `merge_pack_registry`.
4. Registers the pack's `interfaces/` with the interface loader
   (`register_pack_interface_dir`, best-effort) so builder synthesis can load the plugin
   step's `.step.yaml`.
5. Re-syncs the hybrid manager (`manager.reload_core_registry()`) so `get_step_names()` — and
   therefore the `StepCatalog`, which reads names through the manager — sees the plugin steps.
6. Refreshes this module's own snapshot globals via `_refresh_module_variables()` so direct
   `STEP_NAMES` readers pick up the merge too.

It returns a `{name: "collision"}` dict for pack names that shadowed an existing package
step (empty when every pack step is new). `refresh_registry(None)` is a no-op returning `{}`;
a non-existent pack dir logs a warning and returns `{}`.

```python
from cursus.registry.step_names import refresh_registry

collisions = refresh_registry("/abs/path/to/my_project/step_pack/interfaces")
# {} means every pack step was genuinely new
```

### `merge_pack_registry(pack_rows)` — the in-place, never-replace primitive

Defined in `cursus/registry/step_names_base.py`, this is the low-level merge. The critical
detail is that it **mutates `STEP_NAMES` in place and never reassigns it**:

```python
# step_names_base.py
def merge_pack_registry(pack_rows):
    collisions = {name: "collision" for name in pack_rows if name in STEP_NAMES}
    STEP_NAMES.update(pack_rows)   # in place — package rows preserved, pack rows on top
    _rebuild_derived()
    return collisions
```

Why in-place? Many modules did `from ...step_names_base import STEP_NAMES` at import time and
hold a live reference to that dict. Reassigning `STEP_NAMES` to a new dict would leave those
references pointing at the stale table. Using `.update()` keeps every import-time reference
live, and `_rebuild_derived()` regenerates the `CONFIG_STEP_REGISTRY` / `BUILDER_STEP_NAMES`
/ `SPEC_STEP_TYPES` mappings from the mutated table.

Note the asymmetry with `build_registry_from_interfaces`, which is a **REPLACE** primitive
(it returns a fresh table). `refresh_registry` uses that to derive *only the pack's rows*,
then routes them through `merge_pack_registry` so they are layered **on top of** the package
table — the pack rows never stand alone as the registry. This is what makes package steps
the permanent floor.

## Collisions: shadow-with-warning + `pack_collisions` health

A pack step whose canonical name already belongs to a package step is a **collision**. The
policy is *plugin-wins*: the pack row shadows the package row in the merged table, but the
event is recorded and logged rather than silently accepted.

- `merge_pack_registry` returns the collision names.
- `refresh_registry` records them into the module-global `_pack_collisions` dict and logs a
  `WARNING` recommending you rename the pack step to avoid shadowing a core step.
- The collisions are surfaced for monitoring via `get_registry_health()`:

```python
from cursus.registry.step_names import get_registry_health

health = get_registry_health()
# {
#   "hybrid_active": True,          # False => static fallback registry in use
#   "init_error": None,            # stringified exception if the hybrid manager failed
#   "pack_collisions": {},         # {name: "collision"} — empty means clean
# }
```

`pack_collisions` sits alongside the other registry-health signals (`hybrid_active`,
`init_error`) so an operator can detect a silent shadow of a core step. An empty dict means
no pack shadowed a package step.

Even on a collision, every *other* package step is untouched — the invariant tests assert
`set(before).issubset(set(after))` after a deliberate `XGBoostTraining` clash.

## The ordered interface loader (package wins)

Registry rows tell the catalog a step *exists*; the interface loader
(`cursus/steps/interfaces/__init__.py`) tells it what the step *is* by loading its
`.step.yaml`. Packs plug in here through an **ordered search**.

- `register_pack_interface_dir(interfaces_dir)` appends a pack's `interfaces/` dir to the
  module-global `_pack_interface_dirs` list (idempotent; `None`/missing dirs ignored). It
  clears the interface cache because a new search root can change what resolves.
- `_search_dirs()` returns the search roots **package-first**: `[INTERFACES_DIR, *_pack_interface_dirs]`.
- `_resolve_interface_path(step_name)` walks those roots in order and returns the first
  match, so **a package interface always wins** on a name clash. Within each root it tries
  the convention-derived filename first, then a normalized (case/separator-insensitive) scan
  as a fallback.

```python
# steps/interfaces/__init__.py
def _search_dirs():
    """Ordered interface search roots: the package dir FIRST, then registered packs."""
    return [INTERFACES_DIR, *_pack_interface_dirs]
```

`list_available_interfaces()` merges names across all roots (deduplicated), and
`clear_interface_cache()` lets long-running processes pick up edited `.step.yaml` files.
This package-first ordering is the interface-loader half of the additive invariant, mirroring
the registry half in `merge_pack_registry`.

## External config import by file location

A pack's config classes live in files that are **not** under the cursus `package_root`, so
they cannot be imported by a relative dotted module path. This is a direct consequence of the
package-portability principle: cursus imports package components with *deployment-agnostic*
relative dotted paths (`file_path.relative_to(package_root)` + `importlib.import_module(...,
package=__package__)`) so the same code works whether cursus is pip-installed, vendored, or
running in a Lambda/container. That relative form only exists for files *under* the package
root. Before 2.8.0 a config file outside the package was AST-detected and then silently
dropped. `ConfigAutoDiscovery` in `step_catalog/config_discovery.py` now handles them.

When `_scan_config_directory` (and, for hyperparameters, `_scan_hyperparams_directory`) finds
a config class, it picks the import strategy by file location:

```python
# config_discovery.py — inside _scan_config_directory
relative_module_path = self._file_to_relative_module_path(py_file)
if relative_module_path:
    module = importlib.import_module(relative_module_path, package=__package__)
    class_type = getattr(module, node.name)
else:
    # File is NOT under package_root -> import by file location
    class_type = self._import_class_from_file(py_file, node.name)
```

`_file_to_relative_module_path` returns `None` for any file outside `package_root`, which is
exactly the case for a pack. `_import_class_from_file` then loads the class via
`importlib.util.spec_from_file_location` under a **unique, path-hashed synthetic module
name**:

```python
digest = hashlib.md5(abs_path.encode("utf-8")).hexdigest()[:12]
module_name = f"cursus._pack_configs.{file_path.stem}_{digest}"
```

The path-derived hash guarantees two packs each shipping, say, `config_xgboost_step.py` do
not collide in `sys.modules`, and it is stable so repeated discovery reuses the already-loaded
module. The module is registered in `sys.modules` *before* `exec_module` so intra-module
references resolve during execution, and on failure the partial registration is rolled back
so a later retry re-runs cleanly. A load failure never raises out of discovery — it is logged
and the class is skipped.

Discovery finds a pack's `configs/` via `_discover_workspace_configs` (and its `hyperparams/`
via `_discover_workspace_hyperparams`), each looking for that subfolder directly under a
`workspace_dir`. The regression test
`TestExternalConfigImport.test_external_config_class_is_discovered` writes a pack for
`ImportProbeStep` and asserts that its `ImportProbeStepConfig` is imported rather than dropped;
a sibling test `test_package_config_import_unchanged` asserts the package config import path is
unaffected.

## End to end

Putting the pieces together for a project laid out as `my_project/step_pack/`:

```python
from cursus.core.compiler.dag_compiler import PipelineDAGCompiler

# 1. anchor_file=__file__ -> project_root = my_project/
#    -> _derive_step_pack_dir finds my_project/step_pack (has interfaces/)
compiler = PipelineDAGCompiler(
    config_path="configs/my_pipeline.json",
    anchor_file=__file__,
)

# On construction, for the derived pack the compiler has already:
#   - refresh_registry("my_project/step_pack/interfaces")
#       -> merge_pack_registry (in-place, package steps preserved)
#       -> register_pack_interface_dir (interface loader, package-first)
#       -> reload_core_registry (catalog sees the new step)
#   - set_default_workspace_dirs([...])  (bare StepCatalog() also sees the pack)
#   - built self.step_catalog with workspace_dirs=[...]

pipeline = compiler.compile(dag)
```

The pack's steps now resolve exactly like package steps: they have a registry row, a
loadable interface, an importable config class, and a catalog entry. The regression suite
locks each guarantee:

| Invariant | Test |
| --- | --- |
| No pack ⇒ registry equals package-only | `test_no_pack_registry_equals_package_only` |
| Pack adds only its own step; all package steps kept | `test_pack_adds_only_new_step_and_keeps_all_package_steps` |
| A pack omitting a core step never drops it | `test_pack_omitting_a_core_step_does_not_remove_it` |
| Collision shadows with a warning, keeps other steps | `test_collision_shadows_with_warning_but_keeps_other_package_steps` |
| External config imported by file location | `test_external_config_class_is_discovered` |
| Catalog indexes pack + package | `test_catalog_indexes_pack_and_package` |
| Bare `StepCatalog()` uses the process default | `test_bare_catalog_uses_process_default` |
| Snapshot gate re-derives package-only | `test_pack_rows_excluded_from_package_derive` |

## Contrast: the removed `cursus.workspace` module

The word "workspace" appears throughout this feature (`workspace_dirs`,
`_discover_workspace_configs`, `get_workspace_context`), but it is **not** the old
`cursus.workspace` module. That module (the `api`/`manager`/`integrator`/`validator`
subsystem) was a dead island — never instantiated on any live path — and has been removed:
there is no `src/cursus/workspace/` package today, and `core/__init__.py` no longer references
it. Its one goal-relevant use, enumerating pipeline projects, is replaced by
`core/utils/project_discovery.py` and the `cursus projects list|show` CLI (see the
[CLI reference](../cli.rst)).

The step-pack model deliberately does **not** resurrect that machinery. The differences:

| Removed `cursus.workspace` module | Step packs (2.8.0) |
| --- | --- |
| A separate subsystem (API/manager/integrator/validator) | No new subsystem — reuses the registry, interface loader, and step catalog |
| Dead code, never instantiated | Wired into `PipelineDAGCompiler` and exercised by the compile path |
| Aimed at multi-workspace orchestration | Aims at one thing: adding external steps additively |
| — | Package steps are a permanent floor (additive invariant) |

The load-bearing pieces that *were* preserved are the `workspace_dirs` parameter and the
`step_catalog/adapters` — those are what step packs build on. The workspace-context helpers
still present in `registry/step_names.py` (`set_workspace_context`, `workspace_context`, …)
are a separate, older registry-scoping concern and are orthogonal to step packs.

## See also

- [Compilation](../concepts/dag_and_compilation.md) — where `PipelineDAGCompiler` and the
  caller hook live.
- [Registry and discovery](../concepts/registry_and_discovery.md) — the step registry that
  `refresh_registry` merges into.
- [Step interfaces](../concepts/step_interfaces.md) — the `.step.yaml` format a pack ships.
- [Config system](../concepts/config_system.md) — how config classes are discovered and used.
- [Step catalog](../reference/generated/step_catalog.md) — the catalog that indexes pack
  steps as native.

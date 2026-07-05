# Registry & Interface-First Discovery

The **step registry** is Cursus's canonical table of steps: for every step kind
(``XGBoostTraining``, ``TabularPreprocessing``, ``CradleDataLoading``, ...) it records the
config class, builder class, spec type, SageMaker step type, and a description. Almost every
other subsystem — the config resolver, the builder router, the step catalog, validation —
looks up "what is this step?" through this table.

The important thing to understand is that the registry is **derived**, not authored as a
standalone file. It is built *from the per-step ``.step.yaml`` interface files*. This page
explains that interface-first derivation, the layers that sit on top of it (workspace
awareness, the hybrid manager, plugin step-packs), the naming conventions that let
PascalCase canonical names and snake_case file stems round-trip, the health signals that
surface a degraded registry, and the separate **strategy registry** that maps a step to its
builder handler.

```{contents}
:local:
:depth: 2
```

## The shape of a registry row

Every step in the registry is one row keyed by its **canonical name** (PascalCase). The row
is a flat dict with exactly these fields:

| Field | Meaning | Example |
| --- | --- | --- |
| `config_class` | Configuration class name | `XGBoostTrainingConfig` |
| `builder_step_name` | Builder class name | `XGBoostTrainingStepBuilder` |
| `spec_type` | Step specification type — always **equal to the canonical name** | `XGBoostTraining` |
| `sagemaker_step_type` | SageMaker step category | `Training` |
| `description` | Human-readable prose | `XGBoost model training step` |

That table is exposed as the module-level dict `STEP_NAMES`, plus three derived mappings that
invert or project it:

- `CONFIG_STEP_REGISTRY` — `config_class` → canonical name (reverse lookup)
- `BUILDER_STEP_NAMES` — canonical name → `builder_step_name`
- `SPEC_STEP_TYPES` — canonical name → `spec_type`

## Interface-first derivation

The registry has a single source of truth: the per-step ``.step.yaml`` interface files under
`src/cursus/steps/interfaces/`. There is **no** standalone `step_names.yaml` table — it was
deleted; each `.step.yaml` now carries its own `registry:` block, and a golden snapshot
(`tests/registry/step_names_registry_snapshot.json`) gates drift.

A minimal interface file looks like this (from `xgboost_training.step.yaml`):

```yaml
step_type: XGBoostTraining        # the canonical name (== spec_type)
node_type: internal
registry:
  sagemaker_step_type: Training
  description: XGBoost model training step
# ...patterns / compute / contract / spec blocks follow...
```

`interface_registry_loader.build_registry_from_interfaces()` walks every `*.step.yaml`,
reads its `step_type` and `registry:` block, and applies these derivation rules
(from `interface_registry_loader.py`):

- `spec_type` = the canonical step name (it is `== step_type` for every row).
- `config_class` = `"<Name>Config"` by convention, unless the `registry:` block overrides it.
- `builder_step_name` = `"<Name>StepBuilder"` by convention, unless overridden.
- `sagemaker_step_type` = irreducible; read from the `registry:` block (missing it is a hard
  error — the loader raises `ValueError`).
- `description` = irreducible prose; read from the `registry:` block.

Because `config_class` and `builder_step_name` follow a naming convention, most steps only
need to declare the two irreducible fields (`sagemaker_step_type`, `description`) in their
`registry:` block. The loader also consults a Python `_CONFIG_CLASS_OVERRIDES` seam, but it is
now **empty**: the three convention-breakers (`BatchTransform` → `BatchTransformStepConfig`,
`PyTorchModel` → `PyTorchModelStepConfig`, `XGBoostModel` → `XGBoostModelStepConfig`, whose
real config classes end in `StepConfig` rather than the conventional `Config`) each declare
`config_class` explicitly in their own `.step.yaml` `registry:` block — so the truth stays in
the authored interface file, not a Python override.

### The three interface-less steps (`_EXTRAS`)

A few abstract/base steps have no `.step.yaml` interface at all, so they cannot be derived
from the walk. They are declared explicitly in the loader's `_EXTRAS` map:

| Canonical name | Why it has no interface |
| --- | --- |
| `Base` | Abstract base pipeline config (`BasePipelineConfig`) |
| `Processing` | Abstract base processing step (`ProcessingStepConfigBase`) |
| `HyperparameterPrep` | Builder-less Lambda step |

`build_registry_from_interfaces()` seeds the table with `_EXTRAS` first, then layers the
interface-derived rows on top.

```python
from cursus.registry.interface_registry_loader import build_registry_from_interfaces

table = build_registry_from_interfaces()   # {canonical_name: {config_class, ...}, ...}
table["XGBoostTraining"]
# {'config_class': 'XGBoostTrainingConfig',
#  'builder_step_name': 'XGBoostTrainingStepBuilder',
#  'spec_type': 'XGBoostTraining',
#  'sagemaker_step_type': 'Training',
#  'description': 'XGBoost model training step'}
```

## `step_names_base` — the dependency-free leaf

`registry/step_names_base.py` is a deliberately **dependency-free leaf module**. It calls
`build_registry_from_interfaces()` once at import to bind `STEP_NAMES`, then derives the three
mappings via `_rebuild_derived()`:

```python
from cursus.registry.step_names_base import (
    STEP_NAMES, CONFIG_STEP_REGISTRY, BUILDER_STEP_NAMES, SPEC_STEP_TYPES,
)
```

Keeping the raw data in this leaf (rather than in the access layer) is what breaks the
otherwise-circular import between the workspace-aware access layer (`step_names.py`) and the
hybrid manager (`hybrid/manager.py`) — both of them read `STEP_NAMES` from here. The leaf
itself imports only `typing` and lazily calls the loader; the loader
(`interface_registry_loader.py`) in turn imports only `pathlib` / `typing` / `yaml`. Neither
pulls in any cursus internals, so no import cycle is possible.

`step_names_base` also owns the low-level plugin merge, `merge_pack_registry()` (covered
below). It mutates `STEP_NAMES` **in place** (never reassigns it) so that any module that did
`from ...step_names_base import STEP_NAMES` keeps pointing at the live dict.

## `step_names` — the workspace-aware access layer

`registry/step_names.py` is the public accessor layer. Instead of reading the static
`STEP_NAMES` dict, code should call its functions, which accept an optional `workspace_id` and
resolve through the hybrid manager. The most common ones:

| Function | Returns |
| --- | --- |
| `get_step_names(workspace_id=None)` | The full `{name: row}` table for a workspace/core |
| `get_all_step_names(workspace_id=None)` | List of canonical names |
| `get_config_class_name(step_name, ...)` | The step's `config_class` |
| `get_builder_step_name(step_name, ...)` | The step's `builder_step_name` |
| `get_spec_step_type(step_name, ...)` | The step's `spec_type` |
| `get_sagemaker_step_type(step_name, ...)` | The step's `sagemaker_step_type` |
| `get_step_description(step_name, ...)` | The step's `description` |
| `get_canonical_name_from_file_name(file_name, ...)` | Resolve a file stem → canonical name |
| `get_step_name_from_spec_type(spec_type, ...)` | Reverse: spec type → canonical name |
| `get_valid_sagemaker_step_types(...)` | Authoritative set of valid SageMaker types |

```python
from cursus.registry import get_step_names, get_config_class_name, get_all_step_names

get_config_class_name("XGBoostTraining")     # 'XGBoostTrainingConfig'
"TabularPreprocessing" in get_step_names()   # True
sorted(get_all_step_names())[:3]
```

The module also keeps module-level snapshots (`STEP_NAMES`, `CONFIG_STEP_REGISTRY`,
`BUILDER_STEP_NAMES`, `SPEC_STEP_TYPES`) for backward compatibility. These are bound **once at
import**; workspace-aware reads must go through the `get_*` accessors. (A comment in the
source explicitly notes that the old module-level `@property STEP_NAMES` was dead code —
`@property` has no effect at module scope.)

### Workspace context

The access layer can carry a "current workspace" so lookups resolve workspace-local steps
first. This is set explicitly or scoped with a context manager:

```python
from cursus.registry import (
    set_workspace_context, clear_workspace_context, workspace_context, get_step_names,
)

with workspace_context("developer_1"):
    steps = get_step_names()      # developer_1's steps layered on top of core
# context restored on exit
```

Context is stored on the manager (and falls back to the `CURSUS_WORKSPACE_ID` env var).
Setting it invalidates the manager's caches so the next read reflects the new scope.

### Where the richer `StepInfo` lives

`get_step_names()` returns only the flat registry rows. The **richer** per-step record —
combining the registry row with discovered components (contract, spec, builder, script) — is a
`StepInfo` produced by the step catalog, not the registry. Fetch it with
`StepCatalog.get_step_info(step_name, job_type=None)`, which is backed by this registry. See
[Step catalog](../reference/generated/step_catalog.md) for that surface.

## The hybrid `UnifiedRegistryManager`

`registry/hybrid/manager.py` provides `UnifiedRegistryManager` — a single manager that
consolidates what were three classes (`CoreStepRegistry`, `LocalStepRegistry`,
`HybridRegistryManager`, kept as aliases). It:

- Loads the **core** table from `step_names_base.STEP_NAMES` (as `StepDefinition` objects).
- Optionally discovers and loads **workspace** registries (`workspace_registry.py` files
  declaring `LOCAL_STEPS`, `STEP_OVERRIDES`, `WORKSPACE_METADATA`) when a `workspaces_root`
  is provided.
- Resolves a step with **workspace priority**: workspace-local steps and overrides win over
  core; core is always the floor.
- Caches the legacy `{name: row}` dict and the `StepDefinition` map per workspace (with
  `threading.RLock` for thread safety), invalidating on context change or reload.

The access layer lazily constructs one global manager and delegates to it:

```python
# step_names.get_step_names() effectively does:
manager = _get_registry_manager()
manager.create_legacy_step_names_dict(effective_workspace)
```

### Static fallback

If `UnifiedRegistryManager()` fails to initialize, `_get_registry_manager()` does **not**
raise — it logs the failure (with traceback) and swaps in a minimal `FallbackManager` built
straight from `step_names_base.STEP_NAMES`. The registry still works, but **workspace-aware
resolution is unavailable** (core steps only). This degradation used to be invisible; it is
now recorded so callers can detect it (see health signals below).

## Health signals

Two functions in `step_names.py` surface whether the registry is running in its full,
workspace-aware mode or has silently fallen back. They are importable from the module:

```python
from cursus.registry.step_names import is_hybrid_active, get_registry_health

is_hybrid_active()      # True when the hybrid manager is live; False = static fallback
get_registry_health()
# {'hybrid_active': True, 'init_error': None, 'pack_collisions': {}}
```

- `is_hybrid_active()` → `True` when the hybrid `UnifiedRegistryManager` is in use; `False`
  means init failed and the static fallback is active.
- `get_registry_health()` returns:
  - `hybrid_active` — same signal as above.
  - `init_error` — the stringified exception that forced the fallback (`None` when healthy).
  - `pack_collisions` — plugin step-pack names that **shadowed** an existing package step
    (empty = clean). See the next section.

These are the signals monitoring should watch: a non-`None` `init_error` means degraded
resolution, and non-empty `pack_collisions` means a plugin quietly shadowed a core step.

## Plugin step-packs (add-only overlay)

External step-packs can contribute steps by shipping their own `.step.yaml` files. The public
entry point is `refresh_registry(pack_interfaces_dir)` in `step_names.py`. It enforces an
**additive invariant**: package steps are always present; a pack can only *add* steps (or,
on a deliberate name clash, shadow one with a warning). It never removes or replaces a
package step.

The mechanism is package-first:

1. Derive the pack's rows from `pack_interfaces_dir/*.step.yaml` via
   `build_registry_from_interfaces()`.
2. Layer them on top of the live package table with `step_names_base.merge_pack_registry()`
   (in-place `STEP_NAMES.update` — package rows preserved).
3. Re-sync the hybrid manager (`manager.reload_core_registry()`) so the step catalog, which
   reads `get_step_names()` through the manager, sees the plugin steps.

```python
from cursus.registry.step_names import refresh_registry

collisions = refresh_registry("/path/to/step_pack/interfaces")
# {} when every pack step is new; {name: "collision"} for any that shadowed a package step
```

Collisions are logged as warnings and recorded so `get_registry_health()['pack_collisions']`
surfaces them.

## Naming: canonical ↔ snake, compound acronyms

Steps have two names: the PascalCase **canonical name** in the registry
(`XGBoostTraining`) and the snake_case **file stem** on disk (`xgboost_training`). Discovery
has to round-trip between them, and the hard part is compound acronyms — a naive splitter
turns `PyTorchTraining` into `py_torch_training`. `step_catalog/naming.py` is the single
source of truth for this conversion (previously each discovery module carried its own drifting
table, the root cause of a class of "N of M discovered" bugs).

The canonical list of multi-word tokens whose internal capitals must be preserved is
`COMPOUND_ACRONYMS`: `LightGBMMT`, `LightGBM`, `XGBoost`, `PyTorch`, `TensorFlow`,
`SageMaker`, `MLFlow`, `AutoML` (longer tokens first so `LightGBMMT` matches before
`LightGBM`).

| Function | Purpose | Example |
| --- | --- | --- |
| `canonical_to_snake(name)` | PascalCase → snake stem | `XGBoostTraining` → `xgboost_training` |
| `parts_to_pascal(parts)` | snake parts → PascalCase (inverse) | `["xgboost","training"]` → `XGBoostTraining` |
| `canonical_key(name)` | case/separator-insensitive key | `XGBoost_Training` → `xgboosttraining` |
| `resolve_base_step_name(node, known)` | strip suffixes → known base name | `TabularPreprocessing_training` → `TabularPreprocessing` |
| `split_job_type_suffix(node, known)` | `(base, suffix)` split via registry | `CradleDataLoading_munged` → `('CradleDataLoading', 'munged')` |

```python
from cursus.step_catalog.naming import (
    canonical_to_snake, parts_to_pascal, canonical_key, resolve_base_step_name,
)

canonical_to_snake("PyTorchTraining")           # 'pytorch_training'
parts_to_pascal(["xgboost", "training"])        # 'XGBoostTraining'
canonical_key("XGBoost_Training")               # 'xgboosttraining'
```

`canonical_key` is the robustness lever: because it collapses case and separators, a
**normalized directory scan** resolves most new acronym steps with *no table edit at all*.
`resolve_base_step_name` and `split_job_type_suffix` use it to peel job-type / data-source
suffixes off a DAG node name (`_training`, `_munged`, `_sampling`, ...) by validating each
prefix against the actual step registry — not a hardcoded suffix list — so any open suffix
resolves while a distinct step like `XGBoostModel` is never mis-stripped.

The module also centralizes the job-type vocabulary (`JOB_TYPE_SUFFIXES`,
`JOB_TYPE_KEYWORDS`) and the `BASE_CONFIGS` set (`Base`, `Processing`) that discovery excludes.

## The strategy registry

Separate from the step-name registry, `registry/strategy_registry.py` is the single source of
truth for the **builder strategy library** — which construction handler builds a given step.
It is also a dependency-free leaf, so it never imports the heavy handler module at top level.

It maps a routing `(axis, name)` key to a `StrategyInfo` (the handler class plus its
declarative `KnobSpec` knobs). Two routing axes exist:

- `sagemaker_step_type` — `Training`, `CreateModel`, `Transform`, `CradleDataLoading`, ...
  (plus non-routable rows like `Base` / `Lambda`).
- `step_assembly` — `code` / `step_args` / `delegation`, the sub-discriminator used only for
  `Processing` steps.

Handlers self-register via the `@register_strategy(...)` decorator in
`core.base.builder_templates`; `ensure_strategies_loaded()` triggers those registrations
lazily on first read. Both the runtime router (`builder_templates.resolve_handler`) and the
introspection tooling read from this one registry, so the tool can never drift from what the
builder actually does.

```python
from cursus.registry import strategy_registry as sr

sr.axes()                                  # ['sagemaker_step_type', 'step_assembly']
sr.list_strategies(axis="sagemaker_step_type")

# Map a step's (sagemaker_step_type, step_assembly) to the registry key it will bind:
sr.axis_name_for_step_type("Training")               # ('sagemaker_step_type', 'Training')
sr.axis_name_for_step_type("Processing")             # ('step_assembly', 'code')
sr.axis_name_for_step_type("Processing", "step_args")  # ('step_assembly', 'step_args')

info = sr.resolve_strategy("sagemaker_step_type", "Training")  # StrategyInfo (or NoBuilderError)
info.handler, info.verb, info.knobs
```

`axis_name_for_step_type()` is the single source of the routing rule: routing is by
`sagemaker_step_type` only (never by step name), with `Processing` the one type
sub-discriminated by `step_assembly` (default `code`). Requesting a non-routable
`(axis, name)` (an abstract or builder-less type) raises `NoBuilderError`.

## Inspecting the registry from the CLI

Both registries are inspectable from the [CLI](../cli.rst):

```bash
# Step-name registry
cursus registry list-steps                        # all canonical step names
cursus registry list-steps --workspace developer_1
cursus registry list-steps --conflicts-only --include-source
cursus registry resolve-step XGBoostTraining --workspace developer_1
cursus registry validate-registry --check-conflicts
cursus registry validate-step-definition --name MyNewStep --auto-correct

# Strategy registry
cursus strategies axes                            # routing axes + counts
cursus strategies list --axis sagemaker_step_type
cursus strategies show Training                   # one strategy NAME: handler, verb, knobs, presets
cursus strategies for Training                    # what a SAGEMAKER_STEP_TYPE binds (authoring shortcut)
cursus strategies for Processing --step-assembly step_args
cursus strategies knobs --axis step_assembly --name code
```

`validate-step-definition` runs the **standardization enforcement** layer
(`registry/validation_utils.py`): it checks that a new step's canonical name is PascalCase,
that its builder/config names follow the conventions, and that its `sagemaker_step_type` is
one of the authoritative valid types — with `--auto-correct` applying `to_pascal_case`-style
fixes. This is what keeps the interface-first table self-consistent as new steps land.

The same strategy introspection is also exposed as MCP tools (`strategies.list_axes`,
`strategies.list`, `strategies.show`, `strategies.for_step_type`, `strategies.knobs`) — see
[MCP tools](../reference/generated/mcp_tools.md).

## How it fits together

```
.step.yaml registry: blocks  +  _EXTRAS (Base/Processing/HyperparameterPrep)
            │
            ▼  build_registry_from_interfaces()   (interface_registry_loader.py)
   step_names_base.STEP_NAMES        ← dependency-free leaf; merge_pack_registry() overlays packs
            │
            ▼  loaded as StepDefinitions, cached, workspace-layered
   UnifiedRegistryManager            (hybrid/manager.py; static FallbackManager on failure)
            │
            ▼  create_legacy_step_names_dict(workspace_id)
   step_names.get_step_names(...)    ← public accessors + workspace context + health signals
            │
            ▼
   StepCatalog / config resolver / builder router / validation
```

The whole chain is **interface-first**: to add or change a step you edit its `.step.yaml`
`registry:` block (or drop in a new interface file). Everything downstream — the registry
table, the derived mappings, the catalog, the strategy binding — is computed from that. There
is no separate table to keep in sync.

## Related pages

- [Step catalog](../reference/generated/step_catalog.md) — the discovery index and `get_step_info`.
- [MCP tools](../reference/generated/mcp_tools.md) — `strategies` and related introspection tools.
- [Pipeline catalog](../reference/generated/pipeline_catalog.md) — ready-made pipelines.
- [CLI](../cli.rst) — full `cursus registry` and `cursus strategies` command reference.
- [API reference](../api/index.rst) — module-level API docs.

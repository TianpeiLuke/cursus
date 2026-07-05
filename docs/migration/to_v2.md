# Migrating to 2.x

The 2.x line is the result of a multi-release **specification-unification** and
**classless-factory** rewrite that ran from 1.8.0 through the current 2.x line
(the folder-deletion tranche landed in 2.5.0; the head of the line is 2.8.x). It
removed a large amount of hand-maintained per-step code and replaced it with a
single declarative artifact per step (`<step>.step.yaml`) plus runtime synthesis.

The single most important fact for most readers:

> **The public pipeline-authoring API is unchanged.** `PipelineDAGCompiler`
> (and its `.compile()` / `.compile_with_report()` methods) and
> `compile_dag_to_pipeline` (aliased `compile_dag`) behave identically to 1.x.
> The **same DAG + the same config compiles to the same SageMaker pipeline.**
> If your code only imports those, you have nothing to change.

Everything else below matters only if you imported per-step *internals*
(builder / spec / contract modules), used the old class-based pipeline catalog,
or imported the removed `cursus.workspace` module.

See also: [Compilation](../concepts/dag_and_compilation.md),
[Step interfaces](../concepts/step_interfaces.md),
[Pipeline catalog](../concepts/pipeline_catalog.md),
[Registry & discovery](../concepts/registry_and_discovery.md).

---

## The invariance guarantee

The rewrite was explicitly designed to be **invisible at the public API**. The
DAG compiler, the assembler, the dependency resolver, and `pipeline.definition()`
produce the same result for any given DAG + config, because:

- The assembler instantiates a builder by calling `builder_cls(**kwargs)` with
  **no `isinstance` gate**. A runtime-synthesized builder class is
  indistinguishable from a hand-written one at that call site.
- The step-to-step wiring graph keys entirely on `.step.yaml` **spec data**
  (`step_type`, `compatible_sources`, `property_path`, `logical_name`) carried on
  `builder.spec`, plus the DAG node name — never on a Python class. So collapsing
  the 45 per-step builder classes into one shared facade leaves every edge intact.

A full compile of a real 11-node pipeline was confirmed to produce the same
pipeline definition with only the per-step builder *source* gone.

### What did NOT change

| Surface | Status |
|---|---|
| `from cursus import PipelineDAGCompiler` | Unchanged |
| `from cursus import compile_dag_to_pipeline`, `compile_dag` | Unchanged |
| `compiler.compile(...)`, `compiler.compile_with_report(...)` | Unchanged |
| `from cursus.api.dag import PipelineDAG` | Unchanged |
| Your `.step.yaml` interfaces and `<StepType>Config` config classes | Unchanged |
| The generated SageMaker pipeline definition for a given DAG + config | Unchanged |

```python
# This is the supported authoring API. It works identically in 1.x and 2.x.
from cursus import PipelineDAGCompiler
from cursus.api.dag import PipelineDAG

dag = PipelineDAG()
# ... add nodes / edges ...

compiler = PipelineDAGCompiler(config_path="config.json")
pipeline = compiler.compile(dag, pipeline_name="fraud-detection")
```

---

## What changed across the 1.x -> 2.x arc

The rewrite landed in four themes. The table maps each to the release that
introduced it and the sections below.

| Theme | First landed | Breaking? |
|---|---|---|
| Spec/contract merged into one `.step.yaml` per step | 1.8.0 | Only for internal spec/contract imports |
| Class-based pipeline catalog replaced by data-driven catalog | 1.8.0 | Yes, if you used the old catalog classes |
| `cursus.workspace` module removed | 1.8.0 | Only for external `cursus.workspace` imports |
| Per-step `builder_*.py` classes deleted; builders synthesized | 2.0.0 | Only for per-step builder imports |
| `ValidationLevel.CONTRACT_SPEC` (value 2) removed from the enum | 2.0.0 | Only if you referenced that member |
| `steps/{builders,contracts,specs}/` folders physically deleted | 2.5.0 | Yes — the lazy import shim is gone |

### 1. Spec-unification: one `.step.yaml` per step (1.8.0)

The legacy per-step Python pairs — a `*_contract.py` (`ScriptContract`, the I/O)
and a `*_spec.py` (`StepSpecification`, the demand/supply) — were merged into a
single declarative **`<step>.step.yaml`** file under `steps/interfaces/`, loaded
into one Pydantic `StepInterface`. Contract-to-spec alignment became a
**construction-time invariant** (a `@model_validator` raises if the contract's
paths are not a subset of the spec's dependencies/outputs) instead of a separate
runtime validation tier.

The unified `.step.yaml` superseded every per-step `*_spec.py` + `*_contract.py`
pair (90+ specs and 40+ contracts under `steps/specs/` and `steps/contracts/`);
`core/base/specification_base.py` was also removed (replaced by
`core/base/step_interface.py` and `core/base/step_contract.py`). The physical
`*_spec.py` / `*_contract.py` files lingered as import shims until 2.5.0, when
their folders were deleted outright (see §3 below).

If you loaded a step's contract or spec, use the catalog instead:

```python
from cursus.step_catalog import StepCatalog

catalog = StepCatalog()                            # package-only discovery
iface = catalog.get_step_interface("TabularPreprocessing")
# iface carries the unified contract + spec (StepInterface)
```

### 2. Classless-factory: builders are synthesized, not written (2.0.0)

All **45 hand-written `builder_*.py` step-builder classes** were deleted. A step
builder is now a single shared facade class, `TemplateStepBuilder`
(`core/base/builder_templates.py`), whose per-step behavior is routed by five
construction-strategy handlers — `ProcessingHandler`, `TrainingHandler`,
`ModelCreationHandler`, `TransformHandler`, and `SDKDelegationHandler` — that all
subclass a common `PatternHandler` ABC. The facade picks a handler at build time
via `resolve_handler(sagemaker_step_type, step_assembly)`: routing is keyed on the
interface's `registry.sagemaker_step_type` and, for Processing, the
`patterns.step_assembly` (`code` / `step_args` / `delegation`). Per-step
differences that used to live in Python are now declarative **knobs** read from
the `.step.yaml` `patterns:` section.

At runtime, for any registry step with a `.step.yaml` interface and no physical
builder file, the catalog fabricates the class on demand
(`step_catalog/builder_discovery.py::_synthesize_builder`), roughly:

```python
type(f"{Name}StepBuilder", (TemplateStepBuilder,), {"STEP_NAME": Name})
```

and caches it per process. The registry (`STEP_NAMES` and its derived maps) is no
longer a maintained table — it is **derived by construction** from the `.step.yaml`
`registry:` blocks (`registry/interface_registry_loader.py`); the standalone
`registry/step_names.yaml` table is gone.

**Authoring a new step is now: one `.step.yaml` (with a `registry:` block) + one
config class — no builder file.** The difference between, say, a Processing step
and a Training step is one string (`sagemaker_step_type`) in the interface.

### 3. `cursus.steps.builders` — lazy in 2.0.0, removed in 2.5.0

This is the sharpest edge to be aware of, because it changed twice:

- **2.0.0** turned `steps/builders/__init__.py` into a **PEP-562 lazy import
  surface**. `from cursus.steps.builders import XGBoostTrainingStepBuilder` still
  worked — the name resolved lazily to a synthesized `TemplateStepBuilder`
  subclass.
- **2.5.0** then **physically deleted the `steps/{builders,contracts,specs}/`
  folders**, and with them that lazy shim. In the current 2.x head,
  `import cursus.steps.builders` raises `ModuleNotFoundError`.

**Current correct way to obtain a builder class** (rarely needed for authoring):

```python
from cursus.step_catalog import StepCatalog

catalog = StepCatalog()
BuilderCls = catalog.load_builder_class("XGBoostTraining")   # synthesized on demand
```

Two related homes moved out of the deleted packages and are worth noting:

- `S3PathHandler` now lives in `cursus.steps.utils`.
- `StepBuilderBase` lives in `cursus.core.base.builder_base` (it gained a
  first-class `STEP_NAME` attribute, since a shared shell's class name no longer
  carries the canonical step name).

Note also (2.5.0): `StepCatalog.get_contract_entry_points()` keys changed from the
old file-stem (`tabular_preprocessing_contract`) to the PascalCase canonical step
name (`TabularPreprocessing`).

### 4. Data-driven pipeline catalog (1.8.0)

The previous **class-based catalog was removed** and replaced with data: 44
shared DAGs shipped as `*.dag.json` under `pipeline_catalog/shared_dags/`, indexed
by `catalog_index.json`, with a deterministic scoring router. Load and build a
catalogued DAG through the functional API:

```python
from cursus.pipeline_catalog.shared_dags import load_shared_dag
from cursus.pipeline_catalog.core.router import recommend_dag, auto_select_dag
from cursus.pipeline_catalog.core.builders import build_and_compile, build_mods_pipeline

dag = load_shared_dag("xgboost_mt_temporal_split_e2e")   # -> PipelineDAG
# recommend_dag(...) / auto_select_dag(...) score the index
# build_and_compile(...) -> compiled SageMaker Pipeline
# build_mods_pipeline(...) -> @MODSTemplate-decorated pipeline class (MODS)
```

See [Pipeline catalog](../concepts/pipeline_catalog.md) and the generated
[Pipeline catalog reference](../reference/generated/pipeline_catalog.md).

### 5. `cursus.workspace` module removed (1.8.0)

The entire `cursus.workspace` package (`api`, `integrator`, `manager`,
`validator`) was removed — it was a dead island with no in-tree caller. This is
**potentially breaking for any external code that imported `cursus.workspace`**.

Its one goal-relevant use — enumerating pipeline projects — is replaced by
`core/utils/project_discovery.py`:

```python
from cursus.core.utils import discover_pipeline_projects
from cursus.core.utils.project_discovery import summarize_project

projects = discover_pipeline_projects(root="/path/to/pipelines")   # -> List[ProjectInfo]
info = summarize_project("/path/to/pipelines/my_project")           # -> ProjectInfo | None
```

The load-bearing `workspace_dirs` parameter (on `StepCatalog` /
`PipelineDAGCompiler`) and the `step_catalog/adapters` were explicitly
**preserved** — only the dead `cursus.workspace` module was removed. For defining
your own step types outside the installed package, see the **step packs**
mechanism ([Step packs](../concepts/step_packs.md)), which lets a consumer add
native steps (`interfaces/*.step.yaml` + `configs/` + `scripts/`) with no fork.

### 6. `ValidationLevel` enum change (2.0.0)

Because contract-to-spec alignment became a construction-time Pydantic invariant,
the old **Level-2 `ValidationLevel.CONTRACT_SPEC` (value 2)** member was removed
from the enum (`validation/alignment/config/validation_ruleset.py`). Validation is
now re-grounded on the three boundaries construction cannot self-check:

| Member | Value | Boundary |
|---|---|---|
| `SCRIPT_CONTRACT` | 1 | B1 — script ↔ interface (contract) fidelity |
| `SPEC_DEPENDENCY` | 3 | B2 — cross-step DAG-resolvability (+ SageMaker property-path) |
| `BUILDER_CONFIG` | 4 | B3 — registry ↔ handler ↔ config binding |

The surviving members **keep their names and their non-contiguous integer values**
(1, 3, 4), so `ValidationLevel(1)`, `ValidationLevel(3)`, and `ValidationLevel(4)`
coercion still work. **Only the value-2 member is gone** — remove any reference to
`ValidationLevel.CONTRACT_SPEC` or `ValidationLevel(2)`.

The five per-step-type validators plus `validator_factory.py` were replaced by a
single `RegistryBindingValidator`
(`validation/alignment/validators/registry_binding_validator.py`) that proves a
step is *constructible* (the B3 boundary).

---

## New introspection surfaces

Because there is no longer a per-step builder class to open and read, two new
CLI/MCP surfaces make the now data-driven build self-describing:

- **`cursus steps io <Step>` / `cursus steps patterns <Step>`** — a step's
  resolved inputs/outputs/dependencies and which construction pattern + knobs it
  binds, read from the same `io_view` the build uses.
- **`cursus strategies axes | list | show | for | knobs`** — the strategy library
  itself: the handler verbs, which `sagemaker_step_type` maps to which handler,
  and the available knobs.

The MCP tool surface gained matching `steps.*` and `strategies.*` namespaces —
see [MCP tools](../reference/generated/mcp_tools.md) and the
[Step catalog reference](../reference/generated/step_catalog.md). Full CLI
reference: [CLI](../cli.rst).

---

## Before / after checklist

Work top to bottom. Most codebases only touch the first two rows.

| If your code did this (1.x)… | Do this instead (2.x) | Required? |
|---|---|---|
| `from cursus import PipelineDAGCompiler, compile_dag_to_pipeline` | Nothing — unchanged | — |
| Kept your `.step.yaml` + `<StepType>Config` classes | Nothing — unchanged | — |
| `from cursus.steps.builders.builder_xgboost_training_step import XGBoostTrainingStepBuilder` | `StepCatalog().load_builder_class("XGBoostTraining")` | Yes |
| `from cursus.steps.builders import XGBoostTrainingStepBuilder` (lazy shim) | Same as above — the shim was removed in 2.5.0 | Yes |
| Subclassed or monkeypatched a per-step builder in place | Move the behavior into the step's `.step.yaml` `patterns:` knobs, or a handler in `builder_templates.py` | Yes |
| Imported a per-step `*_spec.py` / `*_contract.py` module | `StepCatalog(...).get_step_interface("<StepName>")` (unified `StepInterface`) | Yes |
| `from cursus.steps.utils import S3PathHandler` | Unchanged — `S3PathHandler` still lives there | — |
| Imported `StepBuilderBase` | `from cursus.core.base.builder_base import StepBuilderBase` | If moved |
| Used the old class-based pipeline catalog | `load_shared_dag(...)` + `recommend_dag(...)` + `build_and_compile(...)` | Yes |
| `import cursus.workspace` (external only) | `cursus.core.utils.project_discovery` (project discovery) or step packs | Yes |
| Referenced `ValidationLevel.CONTRACT_SPEC` / `ValidationLevel(2)` | Remove it — use B1/B2/B3 (`SCRIPT_CONTRACT` / `SPEC_DEPENDENCY` / `BUILDER_CONFIG`) | Yes |
| Called `StepCatalog.get_contract_entry_points()` and keyed by file-stem | Key by PascalCase canonical step name (`TabularPreprocessing`) | Yes |
| Opened a builder class to understand how a step builds | `cursus steps patterns <Step>` / `cursus strategies show ...` | Optional |

### Quick verification

After upgrading, confirm the invariant that matters most — that your DAG still
compiles to the same pipeline:

```bash
# Compile from the CLI and inspect the result
cursus compile --help

# Or in Python
python -c "
from cursus import PipelineDAGCompiler
# build your dag + config, then:
# pipeline = PipelineDAGCompiler(config_path='config.json').compile(dag)
print('compiler import OK')
"
```

If you author steps, the interface validators are the fastest signal:

```bash
cursus validate step-interface --help
cursus steps io <StepName>          # resolved I/O for a step
cursus strategies list              # the strategy library the facade binds
cursus strategies knobs --axis sagemaker_step_type --name Training   # knobs for one strategy
```

---

## Summary

- **Do nothing** if you only use the public authoring API — it is byte-for-byte
  compatible.
- **Repoint imports** if you reached into per-step builder / spec / contract
  modules: those files, and (as of 2.5.0) the
  `steps/{builders,contracts,specs}/` folders themselves, no longer exist. Use
  `StepCatalog.load_builder_class(...)` / `.get_step_interface(...)`.
- **Switch to the functional pipeline catalog** (`load_shared_dag`,
  `recommend_dag`, `build_and_compile`) if you used the removed class-based
  catalog.
- **Replace `cursus.workspace`** with `core.utils.project_discovery` or step
  packs.
- **Drop `ValidationLevel.CONTRACT_SPEC`** — it is gone; the other three members
  keep their names and values.

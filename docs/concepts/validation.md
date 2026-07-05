# Validation & Alignment

Cursus assembles SageMaker pipelines out of loosely coupled parts — a script, its
contract, a step specification, a builder, and a config class — that must agree with one
another for a build to succeed. The **validation framework** is the set of tools that
checks those agreements *before* you deploy, plus the construction-time invariants that make
some of those checks unnecessary in the first place.

This page explains the layers:

- The **construction-time invariants** baked into `StepInterface` (the unified `.step.yaml`
  loader), which turned the old "contract ↔ spec" runtime tier into a Pydantic check that
  fails at load time.
- The **unified alignment tester** (`UnifiedAlignmentTester`) and its three surviving
  validation *boundaries* (B1 / B2 / B3).
- The **registry-binding validator** (`RegistryBindingValidator`) — the reframed Level‑4 /
  B3 gate that replaced per-step-type source scanning.
- The **universal builder test** (`UniversalStepBuilderTest`), which reuses the alignment
  system instead of re-implementing it.
- The CLI surfaces: `cursus alignment`, `cursus validate`.

Throughout, a recurring theme: validation reuses the *real* production machinery — the same
`StepCatalog`, the same dependency resolver, the same `resolve_handler` routing — so a
passing check means the same code path a real build takes actually works.

For background on the objects being validated, see [Step interfaces](step_interfaces.md),
[Dependency resolution](dependency_resolution.md), and
[Registry & discovery](registry_and_discovery.md).

## Why "alignment"?

A step is realized from several artifacts that each encode part of the same truth:

| Artifact | What it declares |
| --- | --- |
| Script (`*.py`) | The actual code, its input/output paths, CLI arguments, env vars |
| Contract | The interface the script promises (paths, arguments, env vars) |
| Specification | Dependencies (what the step consumes) and outputs (what it produces), with SageMaker property paths |
| Builder | How to turn a config into a SageMaker step |
| Config class | The typed fields the builder reads |

When these drift out of sync — a script reads an env var the contract never declares, a spec
output points at an invalid SageMaker property path, a config is missing a field the builder
reads — the pipeline fails at build or run time, often deep inside SageMaker. Alignment
validation surfaces those mismatches early and with a precise message.

## Construction-time invariants replaced a runtime tier

Historically the validator ran **four** levels, and Level‑2 was "Contract ↔ Specification":
does every contract input have a matching spec dependency, and every contract output a
matching spec output?

With the unified `.step.yaml` format, the contract and the spec are two *views onto one
aligned object* (`StepInterface`). Loading a `.step.yaml` runs `StepInterface._sync_and_align`
as a Pydantic model validator, which enforces exactly the old Level‑2 subset relationship at
**construction time**:

```python
# src/cursus/core/base/step_interface.py — _sync_and_align (excerpt)

# Contract inputs must each have a matching spec dependency.
missing_deps = set(self.contract.inputs.keys()) - set(self.spec.dependencies.keys())
if missing_deps:
    raise ValueError(f"Contract inputs missing from spec dependencies: {missing_deps}")

# Contract outputs must each have a matching spec output.
missing_outs = set(self.contract.outputs.keys()) - set(self.spec.outputs.keys())
if missing_outs:
    raise ValueError(f"Contract outputs missing from spec outputs: {missing_outs}")
```

Because a `StepInterface` *cannot be constructed* while violating this invariant, a separate
"contract ↔ spec" runtime validation level would only ever re-check a tautology. It was
therefore removed. `_sync_and_align` also reconciles the promoted top-level `compute:`
descriptor with the back-compat `contract.compute` mirror, raising if both are set and
disagree.

The `ValidationLevel` enum records this history:

```python
# src/cursus/validation/alignment/config/validation_ruleset.py

class ValidationLevel(Enum):
    SCRIPT_CONTRACT = 1   # B1 — Script <-> Interface (contract) fidelity
    SPEC_DEPENDENCY = 3   # B2 — Spec <-> Dependencies (+ SageMaker property-path), cross-step
    BUILDER_CONFIG  = 4   # B3 — Registry <-> handler <-> config binding
```

The member **values** are deliberately non-contiguous (1 / 3 / 4). Value `2` (the old
`CONTRACT_SPEC`) is gone, but `1/3/4` are preserved so that `ValidationLevel(1)`,
`ValidationLevel(3)`, `ValidationLevel(4)` coercion and every existing caller keep working.
The three survivors are the three **boundaries** construction *cannot* self-check:

- **B1 — Script ↔ Contract fidelity.** A script's real I/O must match what its contract
  promises. Construction can't see inside the `.py`.
- **B2 — Spec ↔ Dependencies.** Cross-step DAG resolvability, plus the SageMaker
  property-path correctness check (folded in from the old Level‑2). Construction validates a
  single step, not the graph.
- **B3 — Registry ↔ handler ↔ config binding.** Can the step actually be built from its
  registry row + `.step.yaml` + config? This is genuine residue no single object can
  self-check.

## The unified alignment tester

`UnifiedAlignmentTester`
(`src/cursus/validation/alignment/unified_alignment_tester.py`) is the orchestrator. It is
**configuration-driven**: which levels run for a given step is decided by a per-step-type
ruleset, not hard-coded.

### Step-type-aware rulesets

Each SageMaker step type maps to a `ValidationRuleset` that names its category, its
`enabled_levels`, and (for legacy signature compatibility) a `level_4_validator_class`
string:

```python
# src/cursus/validation/alignment/config/validation_ruleset.py

"Processing": ValidationRuleset(
    step_type="Processing",
    category=StepTypeCategory.SCRIPT_BASED,
    enabled_levels={
        ValidationLevel.SCRIPT_CONTRACT,
        ValidationLevel.SPEC_DEPENDENCY,
        ValidationLevel.BUILDER_CONFIG,
    },
    level_4_validator_class="RegistryBindingValidator",
),
```

Categories drive which levels are appropriate:

| Category | Meaning | Enabled boundaries (shipped rulesets) |
| --- | --- | --- |
| `SCRIPT_BASED` | Full validation — has a script (`Processing`, `Training`) | B1 + B2 + B3 |
| `CONTRACT_BASED` | Skip B1 (script check), keep the rest (`CradleDataLoading`, `MimsModelRegistrationProcessing`) | B2 + B3 |
| `NON_SCRIPT` | Skip the script boundary; run spec + binding (`CreateModel`, `Transform`, `RegisterModel`) | B2 + B3 |
| `CONFIG_ONLY` | Config-binding-focused (`Lambda`), but B2 stays on as a universal dependency check | B2 + B3 |
| `EXCLUDED` | No validation (`Base`, `Utility`) | — |

Note that `CONTRACT_BASED`, `NON_SCRIPT`, and `CONFIG_ONLY` all resolve to the same
`{SPEC_DEPENDENCY, BUILDER_CONFIG}` set in the shipped rulesets — the categories capture the
*reason* levels are skipped, not just the resulting set. In particular, `CONFIG_ONLY`'s enum
comment reads "Only Level 4 needed," but the sole member `Lambda` re-enables `SPEC_DEPENDENCY`
(commented "Universal Level 3 requirement") because every non-excluded step needs its
dependencies checked.

`Base` and `Utility` are `EXCLUDED` (base configs have no builder; utility steps don't create
SageMaker steps directly). `Lambda` keeps a ruleset (it still needs dependency validation) but
has no construction handler, so B3 skips it explicitly rather than emit a spurious error — see
the B3 section below.

### Orchestration flow

For a single step, `run_validation_for_step(step_name)`:

1. Reads the step's SageMaker type from the registry with `get_sagemaker_step_type`.
2. Fetches its `ValidationRuleset` via `get_validation_ruleset`.
3. If the type is excluded (`is_step_type_excluded`), returns an `EXCLUDED` result.
4. Otherwise runs **only** the `enabled_levels`, skipping the rest — the key performance
   optimization (a `CreateModel` step never runs a script-contract check it has no script
   for).

Discovery of "all steps" goes through the step catalog:

```python
def _discover_all_steps(self):
    return self.step_catalog.list_available_steps()
```

`run_validation_for_all_steps()` iterates that list; `get_validation_summary()` aggregates
pass/fail/excluded counts with a per-step-type breakdown; `export_report()` and
`print_summary()` render it. See [Step catalog](../reference/generated/step_catalog.md) for
the discovery layer.

### Level dispatch and the LevelValidators facade

`_run_validation_level` maps a `ValidationLevel` to a method on `LevelValidators`
(`core/level_validators.py`), a thin facade that instantiates the right tester per boundary:

| Level | Method | Backing tester |
| --- | --- | --- |
| B1 (`SCRIPT_CONTRACT`) | `run_level_1_validation` | `ScriptContractAlignmentTester` |
| B2 (`SPEC_DEPENDENCY`) | `run_level_3_validation` | `SpecificationDependencyAlignmentTester` |
| B3 (`BUILDER_CONFIG`) | `run_level_4_validation` | `RegistryBindingValidator` |

Note the level‑4 dispatch **ignores** the ruleset's `level_4_validator_class` string.
`_get_step_type_validator` always returns the single, step-type-agnostic
`RegistryBindingValidator`; the argument is kept only for signature compatibility with the
deleted per-step-type factory.

## B2 — Spec ↔ Dependencies, on the real resolver

`SpecificationDependencyAlignmentTester` (`core/spec_dependency_alignment.py`) is the most
"integration-like" boundary: it asks whether a step's declared dependencies can actually be
*resolved* against the other steps in the system.

The critical design choice is that it does **not** re-implement resolution — it constructs the
production dependency-resolver stack:

```python
from ....core.deps.factory import create_pipeline_components

self.pipeline_components = create_pipeline_components("level3_validation")
self.dependency_resolver = self.pipeline_components["resolver"]   # UnifiedDependencyResolver
self.spec_registry     = self.pipeline_components["registry"]
```

`create_pipeline_components` returns the same `UnifiedDependencyResolver` +
`SpecificationRegistry` + semantic matcher a real compilation uses (see
[Dependency resolution](dependency_resolution.md)). Validation registers every discovered
spec under its **canonical** registry name (via `get_canonical_name_from_file_name` /
`get_all_step_names`, the registry being the single source of truth) and then exercises the
resolver. A dependency that the real resolver can't wire is a real bug, not a validation
artifact.

`validate_specification_object` runs four checks and merges their issues:

1. **Dependency resolution** — compatibility scoring against candidate outputs
   (`DependencyValidator.validate_dependency_resolution`).
2. **Circular dependencies** — no cycles in the chain.
3. **Data-type consistency** — types match across a dependency edge.
4. **SageMaker property paths** — each spec output's `property_path` is a valid SageMaker
   path for the step type. This check (`SageMakerPropertyPathValidator`) is the one old
   Level‑2 concern with *no* construction-time equivalent, so it was folded onto the B2
   boundary rather than kept in a separate module.

A result is `passed` only when no issue has severity `CRITICAL` or `ERROR`. Specs are loaded
in bulk through `StepCatalog.load_all_specifications()` for efficiency, with an individual
fallback.

## B3 — the registry-binding validator (the reframed Level‑4)

`RegistryBindingValidator`
(`validators/registry_binding_validator.py`) is the most important change in the framework,
so it's worth understanding *why* it exists.

### What the old Level‑4 did, and why it broke

The old Level‑4 was a set of per-step-type validators that did **source-level** contract
testing: `inspect.getsource(builder_class)` substring scans and
`hasattr(builder_class, "_create_estimator")` method-presence checks. That assumed every step
had a hand-written, per-step builder whose Python source embodied its contract.

Once builders became thin declarative shells over a shared template
(`class XStepBuilder(TemplateStepBuilder): STEP_NAME = "X"`), those checks were meaningless:
the source is identical for every step, and `hasattr(shell, "_create_estimator")` is `False`
because the estimator factory lives in the bound handler (`make_compute`), not the shell. The
old validators reported **every** shell as `FAILED`.

### What B3 does instead

B3 replaces "does the builder *source* contain pattern X" with "can the step be *realized*
from its `.step.yaml` + config" — the genuine residue the construction invariant can't
self-check. It exposes the same method the deleted validators did,
`validate_builder_config_alignment(step_name)`, so the level dispatch and MCP/CLI consumers
are unchanged. It runs three sub-checks:

**B3‑1 — Handler binds.** The step's `(sagemaker_step_type, step_assembly)` must resolve to a
routable construction handler:

```python
from ....core.base.builder_templates import resolve_handler, NoBuilderError
return resolve_handler(sm_type, step_assembly)   # calling it IS the binding check
```

`Base` and `Lambda` are no-builder rows by design, so B3 raises an internal
`_SkipValidation` for them (returning `SKIPPED`) rather than a spurious `ERROR`. A genuine
`NoBuilderError` for any other type is an `ERROR`.

**B3‑2 — Builder loadable.** `step_catalog.load_builder_class(step_name)` must return a class
that is a `StepBuilderBase` subclass (the physical shell or a synthesized declarative shell),
i.e. no orphan registry row. A builder that's absent *in this environment* (e.g. an SDK
builder offline) is a `WARNING`, not an `ERROR` — B3‑1 already proved the handler binds.

**B3‑3 — Config-field coverage.** The resolved config class must supply every field the bound
handler and compute descriptor will read at build time. The required set is:

- the handler's declared `requires_config_fields`, UNION
- compute descriptor `*_field` names with non-`None` values
  (`framework_version_field`, `py_version_field`), UNION
- `contract.input_source_overrides` values.

A required field absent from the config class is an `ERROR`; a soft `job_arguments[].source`
provenance attr is a `WARNING`. Crucially, the field set is the **union** of pydantic
`model_fields` *and* class-level attributes (`dir(config_class)`), because a handler accepts a
method/property as a source too (e.g. `PackageConfig.inference_scripts_source` is a method,
not a field) — checking `model_fields` alone would false-`ERROR`.

The config class itself is resolved via `get_config_class_name` (honoring the naming
convention-breakers) and discovered through `step_catalog.discover_config_classes()`.

The result is issue-shaped: status `COMPLETED` when clean, `ISSUES_FOUND` when any
`CRITICAL`/`ERROR` issue is present, `SKIPPED` for no-builder rows, `ERROR` if B3 itself
crashes (a B3 bug must never mask the rest of the suite).

## The universal builder test

`UniversalStepBuilderTest`
(`src/cursus/validation/builders/universal_test.py`) validates step *builders*
specifically — but it is deliberately built **on top of** the alignment system rather than
duplicating it. Its constructor mirrors `UnifiedAlignmentTester` (same `workspace_dirs`, same
`StepCatalog`) and instantiates a `UnifiedAlignmentTester` internally.

`run_validation_for_step(step_name)` loads the builder class via
`step_catalog.load_builder_class` and runs a comprehensive set of components:

1. **Alignment validation** — delegates straight to
   `self.alignment_tester.run_validation_for_step(step_name)`, reusing the proven boundary
   checks instead of re-deriving them.
2. **Integration testing** — capability checks (dependency-resolution methods,
   cache configuration, structural step-instantiation requirements). These are *structural*:
   no step is actually instantiated.
3. **Step-creation capability** — is there a discoverable `<Step>Config` class, does the
   builder expose `create_step` / `validate_configuration`, can required fields be
   identified?
4. **Step-type-specific validation** — "can this step **produce** its compute?"

That last check is shell-aware, and mirrors the B3 fix. The old test did
`hasattr(builder_class, "_create_estimator")` per step type and failed every declarative
shell. The replacement, `_can_produce_compute`, is type-agnostic and passes if **any** of
these hold (mirroring `builder_templates.py`'s own disjunction):

```python
# _can_produce_compute (paraphrased)
# 1) the builder defines its own _create_<verb> override, OR
# 2) the .step.yaml declares a compute.kind, OR
# 3) the step routes to a handler at all (resolve_handler succeeds)
```

Scoring is optional (`enable_scoring`) via `StreamlinedStepBuilderScorer`, with a basic
fallback. Convenience class methods include `from_builder_class(...)` (single-builder mode)
and `test_all_builders_by_type(sagemaker_step_type, ...)`, which filters the registry by type
(excluding `BASE_CONFIGS`) and validates each concrete builder.

## Validation reuses the real machinery

The design invariant tying all of this together: **validation runs the same code a real
build/compile runs.** Concretely —

- **Discovery** is `StepCatalog.list_available_steps()` / `discover_config_classes()` /
  `load_builder_class()` — the same catalog compilation uses.
- **Dependency resolution** (B2) is the production `UnifiedDependencyResolver` from
  `create_pipeline_components`, registered under canonical registry names.
- **Handler binding** (B3‑1) is a literal call to `resolve_handler` — the routing a real
  build performs.
- **Config resolution** (B3‑3) uses `get_config_class_name` + `discover_config_classes` — the
  same lookup builders use.

The payoff: a green validation run is strong evidence the real path works, and a red one
points at the exact object (script, spec, handler, or config) that would have failed a build.

## CLI: `cursus alignment`

`cursus alignment` (`src/cursus/cli/alignment_cli.py`) is the front door to
`UnifiedAlignmentTester`. Commands:

```bash
# List every step the step catalog can validate
cursus alignment list-scripts

# Validate one step across its enabled boundaries
cursus alignment validate xgboost_training --verbose --show-scoring

# Save a JSON report
cursus alignment validate dummy_training --output-dir ./reports

# Validate a single boundary (1=Script↔Contract, 2=Contract↔Spec, 3=Spec↔Deps, 4=Builder↔Config)
cursus alignment validate-level currency_conversion 3

# Validate everything, continue past failures, write per-step reports
cursus alignment validate-all --output-dir ./reports --continue-on-error
```

`validate` exits `0` on `PASSED` or `EXCLUDED`, `1` on failure. `validate-all` discovers all
steps, reports how many have scripts (full validation) vs. rely on intelligent level
skipping, and writes a `validation_summary.json`. (The `validate-level` help text still labels
levels 1–4 for user familiarity even though the underlying boundaries are the three
described above.)

## CLI: `cursus validate`

`cursus validate` (`src/cursus/cli/validate_cli.py`) is a separate, lighter surface for two
author-time / pre-deployment checks:

### `cursus validate step-interface`

Validates a `.step.yaml` at **author time** — the fast feedback loop when you're editing an
interface. It loads through the production `StepInterface.from_yaml` path (via
`load_interface`), so it surfaces:

- Pydantic field errors, and
- the `_sync_and_align` contract↔spec alignment invariant (the construction-time check
  described above), as **blocking** errors;

plus non-blocking **warnings** for incompleteness — e.g. a `compatible_sources` entry that
case-insensitively matches a real step but differs in case, silently losing the resolver's
matching bonus.

```bash
# Validate one interface
cursus validate step-interface XGBoostTraining

# Validate a job_type variant
cursus validate step-interface RiskTableMapping --job-type validation

# Validate every .step.yaml (CI)
cursus validate step-interface --all
```

It exits non-zero if any interface has a blocking error, making it CI-friendly.

### `cursus validate run-scripts`

Executes a DAG's pipeline scripts locally, in dependency order, with data-flow simulation
between steps — a pre-deployment check that the scripts actually run and hand data to each
other. It wraps `validation.script_testing.api.run_dag_scripts`:

```bash
cursus validate run-scripts dag.json -c config.json
```

It exits non-zero if any script fails.

## Choosing the right tool

| You want to… | Use |
| --- | --- |
| Catch `.step.yaml` mistakes while editing / in CI | `cursus validate step-interface` |
| Check scripts actually run and pass data end-to-end | `cursus validate run-scripts` |
| Check a step's contract/spec/binding boundaries | `cursus alignment validate <step>` |
| Validate every step and get a summary | `cursus alignment validate-all` |
| Validate builders (with scoring/reports) | `UniversalStepBuilderTest` (programmatic) |

## Related pages

- [Step interfaces](step_interfaces.md) — the `.step.yaml` / `StepInterface` object being validated
- [Dependency resolution](dependency_resolution.md) — the resolver B2 reuses
- [Registry & discovery](registry_and_discovery.md) — canonical names, step types, handler routing
- [DAG & compilation](dag_and_compilation.md) — where a validated step ends up
- [CLI reference](../cli.rst) — full command listing
- [API reference](../api/index.rst)
- [Step catalog](../reference/generated/step_catalog.md) · [MCP tools](../reference/generated/mcp_tools.md)

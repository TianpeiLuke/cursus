# Design Principles

This page explains *why* Cursus is shaped the way it is. It is not a feature tour — it
is the set of load-bearing decisions that the rest of the codebase follows, each stated
as a principle, a justification, and a pointer to where it actually shows up in the
source. If you understand these six principles you can predict where almost any piece of
behavior lives.

The principles did not all arrive at once. Two of them — **specification-driven** and the
**classless factory** — are the two halves of a deliberate refactor arc: release `1.8.0`
collapsed the per-step *data*, and release `2.0.0` collapsed the per-step *code*. The
others (derived registry, additive extensibility, fail-loud, invariance) are the
constraints that made that arc safe to ship.

| # | Principle | One-line statement |
|---|-----------|--------------------|
| 1 | Specification-driven | One declarative `.step.yaml` per step is the single source of truth for a step's I/O and dependencies. |
| 2 | Classless factory | There are no hand-written step-builder classes; each is synthesized at runtime onto a shared facade. |
| 3 | Interface-first, derived registry | The registry is *computed* from the interface files, never separately maintained. |
| 4 | Additive extensibility | External step packs only add steps; built-in steps stay resolvable, and any name-clash is surfaced loudly, not silently applied. |
| 5 | Fail-loud | A misresolution or degraded subsystem is surfaced as an error or health signal, never silently swallowed. |
| 6 | Invariance guarantee | The same DAG plus the same config always compiles to the same pipeline. |

---

## 1. Specification-driven

**Principle.** A step is described by one declarative file — `steps/interfaces/<step>.step.yaml` —
that unifies what were historically two separate Python artifacts: the *script contract*
(the paths and environment variables a script expects) and the *step specification* (the
dependencies it demands and the outputs it supplies).

**Why.** When a step's I/O lived in a `*_contract.py` and its dependency graph lived in a
`*_spec.py`, the two could — and did — drift apart. Keeping them aligned was a separate
runtime validation *tier*, which meant the misalignment was only caught late, and only if
the tier ran. Collapsing them into one file makes the alignment a property of the *data
model itself*: it is impossible to load an interface whose contract inputs are not a
subset of its declared dependencies, because the loader refuses to construct the object.

**How it shows up in the code.** The unified model is `StepInterface`, a Pydantic
`BaseModel` in `src/cursus/core/base/step_interface.py`. It is intentionally a *superset*
of the two legacy classes so it can stand in for both:

- `StepInterface.contract` (a `ContractSection`) is a drop-in for the old `ScriptContract`
  — it exposes `entry_point`, `expected_input_paths`, `expected_output_paths`,
  `required_env_vars`, and so on.
- `StepInterface.spec` (a `SpecSection`) is a drop-in for the old `StepSpecification` —
  `step_type`, `node_type`, `dependencies`, `outputs`, plus accessors like
  `get_dependency()` / `get_output()`. Each dependency (`DependencyDecl`) and output
  (`OutputDecl`) is *declarative*: a dependency states `compatible_sources`,
  `semantic_keywords`, `data_type`, and `required` — *what* a port needs, not which step
  supplies it — and an output states its `property_path` and `aliases`. This is what makes the
  design "specification-driven" in the original sense: authors declare intent, and the
  dependency resolver does the semantic matching that wires ports together (see
  [Dependency resolution](dependency_resolution.md)).

The contract↔spec alignment that used to be a validation level is now a
**construction-time invariant** enforced by an `@model_validator(mode="after")` named
`_sync_and_align`:

```python
# src/cursus/core/base/step_interface.py  (inside StepInterface._sync_and_align)
missing_deps = set(self.contract.inputs.keys()) - set(self.spec.dependencies.keys())
if missing_deps:
    raise ValueError(f"Contract inputs missing from spec dependencies: {missing_deps}")

missing_outs = set(self.contract.outputs.keys()) - set(self.spec.outputs.keys())
if missing_outs:
    raise ValueError(f"Contract outputs missing from spec outputs: {missing_outs}")
```

Because the check runs *while building the object*, there is no "aligned" and "unaligned"
state a step can be in — a `StepInterface` that exists is aligned. This is the change the
`1.8.0` release describes as merging the per-step spec + contract into one `.step.yaml`
and removing the Contract↔Spec (Level-2) validation tier as redundant. The 113 old
per-step `*_spec.py` / `*_contract.py` files were deleted in the same release.

An interface is loaded by name via `load_interface("XGBoostTraining")` (used by the
runtime) and `StepInterface.from_yaml(data, job_type=...)`. Job-type *variants* are
handled by a **recursive deep merge** (`_deep_merge`) so a variant that restates only the
ports it changes preserves every base port it omits — a deliberate design choice, since a
shallow `{**base, **variant}` merge silently dropped base ports and broke the alignment
invariant.

See [Step interfaces](step_interfaces.md) for the full `.step.yaml` schema.

---

## 2. Classless factory

**Principle.** There are no hand-written per-step builder classes. A step needs one
`.step.yaml` and one config class — no builder file. At build time a builder class is
*synthesized* onto a single shared facade, `TemplateStepBuilder`, and its per-step
behavior is chosen by one of a small set of construction-strategy handlers.

**Why.** Before `2.0.0`, every step type had a `builder_*.py` containing a `create_step`
method that was ~90% identical to every other builder's `create_step`: extract inputs
from dependencies, merge overrides, assemble a SageMaker step. The differences were small,
declarative, and repeated by hand — exactly the kind of boilerplate that rots. Forty-five
near-duplicate classes is forty-five places for a subtle divergence to hide. Replacing the
*duplicated code* with *shared code + declarative knobs* means the difference between, say,
a Processing step and a Training step is one string (`sagemaker_step_type`) in the
interface, not a whole Python file.

**How it shows up in the code.** The facade and the strategy handlers live in
`src/cursus/core/base/builder_templates.py`:

- `TemplateStepBuilder` is a `StepBuilderBase` subclass. Its abstract methods
  (`create_step`, `_get_inputs`, `_get_outputs`) delegate to a **construction-verb
  handler** bound at `__init__`. A shell is nothing but a name:

  ```python
  class TabularPreprocessingStepBuilder(TemplateStepBuilder):
      STEP_NAME = "TabularPreprocessing"
  ```

- The five handlers are `ProcessingHandler`, `TrainingHandler`, `ModelCreationHandler`,
  `TransformHandler`, and `SDKDelegationHandler` (the last for MODS/SAIS-predefined steps
  such as Cradle / Redshift / Registration / DataUploading). `ProcessingHandler` covers
  both Processing assembly modes — the `code=` mode and the `processor.run()` → `step_args`
  mode — differing only by declarative knobs, not by subclass.

- Handler selection is `resolve_handler(sagemaker_step_type, step_assembly, knobs)`.
  Routing is by `sagemaker_step_type` *only* (never by step name — `DummyTraining` is a
  `Processing` step and must route as Processing). `Processing` is the one type that needs
  a sub-discriminator (`step_assembly`: `code` | `step_args` | `delegation`).

The deletion mechanism itself is in `src/cursus/step_catalog/builder_discovery.py`. For a
step that has a `.step.yaml`, routes to a handler, and has no physical builder file,
`_synthesize_builder` fabricates the class on demand and caches it per process:

```python
# src/cursus/step_catalog/builder_discovery.py  (inside _synthesize_builder)
synthesized = type(
    f"{step_name}StepBuilder",
    (TemplateStepBuilder,),
    {
        "STEP_NAME": step_name,
        "__doc__": f"Synthesized declarative shell for {step_name}.",
        **extra_attrs,  # e.g. an injected sdk_step_class knob for SAIS-delegation steps
    },
)
```

The directory `src/cursus/steps/builders/` no longer exists as a set of source files —
the builders are synthesized at runtime. Because a shell's class name no longer carries the
canonical step name, `STEP_NAME` is a first-class attribute on `StepBuilderBase`, and
`_get_step_name()` prefers it over stripping the class name.

Since there is no per-step class to open and read, two introspection surfaces replace that
workflow: `cursus steps io <Step>` / `cursus steps patterns <Step>` show a step's resolved
inputs/outputs and which construction pattern + knobs it binds, and `cursus strategies`
(`axes` | `list` | `show` | `for <SAGEMAKER_STEP_TYPE> [--step-assembly …]` | `knobs`) makes
the strategy library self-describing. Both read from the same data the build path reads, so the view
cannot drift from behavior.

---

## 3. Interface-first, derived registry

**Principle.** The step registry (`STEP_NAMES` and everything derived from it) is not a
maintained artifact. It is *computed* by walking the `.step.yaml` files. The interface
file is the single source of truth for a step's metadata as well as its I/O.

**Why.** A standalone registry table (formerly `registry/step_names.yaml`) is a second
place a step is described, and a second place to forget to update. If the registry is
derived from the interface files, then adding a step's `.step.yaml` *is* registering it —
there is no separate step, and no way for the two to disagree.

**How it shows up in the code.** `src/cursus/registry/interface_registry_loader.py`
provides `build_registry_from_interfaces()`, which globs `steps/interfaces/*.step.yaml`,
reads each file's `registry:` block, and produces the `{canonical_name: {config_class,
builder_step_name, spec_type, sagemaker_step_type, description}}` table:

```python
# src/cursus/registry/interface_registry_loader.py
for path in sorted(idir.glob("*.step.yaml")):
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    name = _interface_step_type(data)
    registry_block = data.get("registry") or {}
    sagemaker_step_type = registry_block.get("sagemaker_step_type")
    if not sagemaker_step_type:
        raise ValueError(f"{path.name}: cannot determine sagemaker_step_type — add a "
                         f"`registry:` block ...")
    table[name] = { ... }
```

`src/cursus/registry/step_names_base.py` calls this to build `STEP_NAMES` and then derives
the three companion maps `CONFIG_STEP_REGISTRY`, `BUILDER_STEP_NAMES`, and `SPEC_STEP_TYPES`
from it. The standalone `registry/step_names.yaml` table was removed in `2.0.0`.

The routing table is derived the same way. `src/cursus/registry/strategy_registry.py` is a
dependency-free leaf that maps a routing axis + name to a `StrategyInfo` (a handler class
plus its declarative `KnobSpec`s). Both the runtime router
(`builder_templates.resolve_handler`) and the `cursus strategies` introspection tool read
from this one `STRATEGY_REGISTRY`, so the tool can never describe a build that differs from
the one that actually runs. Handlers self-register via `@register_strategy(...)`.

"Interface-first" also reshaped **validation**. The old four-level alignment model assumed
three separately-maintained artifacts; with one self-aligning interface and source-less
builders, that premise is gone. Validation is re-grounded on the three boundaries
construction *cannot* self-check:

| Boundary | What it proves | Where |
|----------|----------------|-------|
| B1 | Script ↔ interface fidelity (AST) | the former Level 1 |
| B2 | Cross-step DAG resolvability (+ SageMaker property paths) | the former Level 3 |
| B3 | Registry ↔ handler ↔ config binding | `RegistryBindingValidator` |

`src/cursus/validation/alignment/validators/registry_binding_validator.py` implements B3:
it proves a step is *constructible* — the handler resolves
(`resolve_handler(sagemaker_step_type, patterns.step_assembly)`), the builder loads or
synthesizes, and the config class supplies every field the handler will read at build
time. If B3 passes, the step can be built. See [Validation](validation.md) and
[Registry & discovery](registry_and_discovery.md).

---

## 4. Additive extensibility (step packs never replace built-ins)

**Principle.** A consuming project can define its own steps (an
`interfaces/*.step.yaml` + `configs/` + `scripts/` layout) in a folder *outside* the
pip-installed package and have Cursus discover them as native steps — with no fork, no
vendored copy, and no edit to the package. The extension is additive and package-first: the
internal steps are *always* resolvable, a pack only *adds* steps, and on a deliberate
name-clash the collision is recorded and warned rather than silently applied.

**Why.** The usual ways to add project-specific steps — fork the package, or vendor a copy
— both create a divergent artifact that has to be re-synced forever. And an extension model
that lets a plugin *silently* override a built-in is a foot-gun: a name clash would change the
behavior of a step other pipelines depend on with no signal. Making extension package-first
means that with no pack installed the registry and catalog are byte-identical to the
package-only build, and any name-clash a pack does introduce is surfaced loudly (via
`get_registry_health()['pack_collisions']`) instead of quietly shadowing a core step.

**How it shows up in the code.** Registration of a pack is a *merge on top*, never a
replace. `refresh_registry(pack_interfaces_dir)` in `src/cursus/registry/step_names.py`
builds the pack's rows with the same `build_registry_from_interfaces` and hands them to
`merge_pack_registry` (`step_names_base.py`), which does an in-place `.update` — the package
table is the base and the pack rows are layered on top; any pack name that collides with a
core step is captured in the returned collision map and warned, never silently swallowed:

```python
# src/cursus/registry/step_names.py  (inside refresh_registry)
pack_rows = build_registry_from_interfaces(interfaces_dir=pack_dir)
collisions = step_names_base.merge_pack_registry(pack_rows)
...
_pack_collisions.update(collisions)   # surfaced via get_registry_health()['pack_collisions']
```

The two mechanisms are complementary. The *interface loader* (`load_interface` /
`load_step_interface` in `steps/interfaces/`) searches pack `interfaces/` directories *after*
the package (`_search_dirs()` returns `[INTERFACES_DIR, *_pack_interface_dirs]`), so the
package's `.step.yaml` always wins on a name clash and a core step stays resolvable. The
*registry table* is an add-only `.update` overlay: a colliding pack row does shadow the core
metadata row, but `merge_pack_registry` returns every such name so `refresh_registry` can log
it and record it in `get_registry_health()['pack_collisions']`. Either way the failure mode is
loud, not silent. Because the derived registry, the strategy table, and the classless factory
are all data-driven, a pack step gets discovered, resolved, and constructed through exactly
the same machinery as a built-in — it supplies only the `.step.yaml`, config, and script.
The additive invariant is locked by regression tests. See [Step packs](step_packs.md).

---

## 5. Fail-loud, no silent misresolution

**Principle.** When Cursus cannot resolve something, or when a subsystem degrades to a
fallback, that condition is surfaced — as a raised error, a non-zero exit code, or an
explicit health signal. It is never swallowed into a `None`, a `log.debug`, or a wrong-but-
plausible default.

**Why.** The most expensive failures in a pipeline compiler are the *quiet* ones: a DAG
node that resolves to the wrong config, a registry that silently fell back to a static
snapshot, a CLI that printed an error but exited `0`. Each of these looks like success until
something much later breaks in a way that no longer points back at the cause. A loud failure
at the point of misresolution is cheaper than a silent one debugged three steps downstream.

**How it shows up in the code.** This principle is visible as a steady stream of "surface
the swallowed failure" changes:

- **Registry degradation is observable.** When the hybrid (workspace-aware)
  `UnifiedRegistryManager` fails to initialize, the code falls back to a static manager —
  previously invisible. `is_hybrid_active()` and `get_registry_health()` (in
  `src/cursus/registry/step_names.py`) now expose `{hybrid_active, init_error,
  pack_collisions}`, and the fallback logs the full traceback rather than passing silently.
- **CLI exit codes are honest.** Error paths that did `return 1` reported *success* to the
  shell, because Click ignores a command's return value. Failures now
  `raise SystemExit(1)`, routed through a shared `safe_cli_command` decorator.
- **Misresolution surfaces as an error, not a wrong answer.** A config whose `job_type` was
  present-but-`None` used to raise an opaque `AttributeError` that masked a genuine
  DAG↔config mismatch; it now resolves cleanly so the *real* mismatch (if any) is the thing
  that surfaces. Conversely, `DAGConfigFactory.validate_dag_config_alignment()` turns
  step-*type* drift into a loud build-time error instead of a silently mis-built pipeline,
  and `is_dag_step()` / `configure_step_if_present()` replace the `if "X" in pending_steps`
  guard that hid typos and renamed steps.
- **Unknown job-type variants warn and fall back explicitly.** `StepInterface.from_yaml`
  logs a warning and uses the base spec when a requested `job_type` matches no declared
  variant — the fallback is documented, deliberate, and bounded (it can only under-tighten
  an optional dependency; a missing *required* dependency is still caught downstream by the
  dependency resolver).

The B3 `RegistryBindingValidator` (principle 3) is the fail-loud rule applied to
construction: it proves at validation time that a step can actually be built, rather than
letting an unconstructible step fail opaquely at pipeline-build time.

---

## 6. The invariance guarantee

**Principle.** For any given DAG plus config, Cursus compiles to the *same* SageMaker
pipeline — regardless of how the internals are refactored. The public pipeline-authoring
API (`PipelineDAGCompiler`, `compile_dag_to_pipeline`, `compile_with_report`) behaves
identically across the whole `1.8.0` → `2.0.0` arc.

**Why.** The specification-unification and classless-factory refactors deleted tens of
thousands of lines and 45 builder classes. A change that large is only safe if it is
*invisible at the seam*: pipeline authors, existing configs, and the assembler must not be
able to tell that the per-step builders no longer exist. The invariance guarantee is the
contract that made the rewrite shippable — it is the whole point of the exercise, not a
side effect.

**How it shows up in the code.** The guarantee holds because the two things that determine
the output pipeline never depended on the deleted code:

1. **The assembler calls `builder_cls(**five_kwargs)` with no `isinstance` gate.** A
   runtime-synthesized `TemplateStepBuilder` subclass is indistinguishable from a
   hand-written class at that call site — the assembler keeps the exact five-kwarg
   `__init__` contract it always called (see `TemplateStepBuilder.__init__` in
   `src/cursus/core/base/builder_templates.py`).
2. **The step↔step wiring graph keys entirely on `.step.yaml` spec data** — `spec.step_type`,
   `compatible_sources`, `property_path`, `logical_name` (carried on `builder.spec`) — plus
   the DAG node name. It never keys on a Python class. Collapsing 45 classes into one facade
   therefore leaves every edge of the dependency graph intact.

Because both invariants hold, the same DAG + config produces the same pipeline definition
with only the per-step builder *source* gone — validated end-to-end on a real 11-node
pipeline. For pipeline authors this means the classless-factory rewrite required **no
migration**: every `.step.yaml`, every config, and the DAG-compiler API are unchanged. The
only breaking surface is for code that imported a builder *by module path* (rather than by
name) or that monkeypatched a per-step builder in place.

See [DAG & compilation](dag_and_compilation.md) for how the compiler consumes these
invariants, and [Dependency resolution](dependency_resolution.md) for the spec-data wiring.

---

## How the principles reinforce each other

The six are not independent — they form a dependency chain:

- **Specification-driven (1)** makes each step a single declarative file, which is the
  precondition for **deriving the registry (3)** and for **synthesizing builders (2)** —
  both read that one file.
- **The classless factory (2)** is only safe because of the **invariance guarantee (6)**:
  deleting 45 classes is acceptable precisely because the output pipeline is defined by spec
  data, not by those classes.
- **The derived registry (3)** is what makes **additive extensibility (4)** trivial — a pack
  step is just more `.step.yaml` files fed to the same `build_registry_from_interfaces`.
- **Fail-loud (5)** is the discipline that keeps all of the above honest: a derived registry
  that silently fell back, or a misresolution that returned a wrong-but-plausible builder,
  would quietly break the invariance guarantee. Surfacing those conditions is what lets you
  *trust* that the same DAG + config really does give the same pipeline.

If you are extending Cursus, the practical takeaway is: **author data, not code.** Add a
`.step.yaml` and a config; let the registry, the builder, and the validation derive
themselves; and if something cannot be derived, expect a loud error at the point of failure
rather than a silent wrong answer downstream.

## Related reading

- [Architecture](architecture.md) — the subsystem map these principles produce.
- [Step interfaces](step_interfaces.md) — the `.step.yaml` schema in full.
- [Registry & discovery](registry_and_discovery.md) — how `STEP_NAMES` is derived and read.
- [Validation](validation.md) — the B1/B2/B3 boundary model.
- [Step packs](step_packs.md) — additive external step discovery.
- [Step catalog reference](../reference/generated/step_catalog.md) and
  [MCP tools reference](../reference/generated/mcp_tools.md).

# Step Interfaces & Builder Synthesis

```{admonition} What you'll learn
:class: tip

- What a `StepInterface` is and how it unifies the old contract + specification into one validated object loaded from `<step>.step.yaml`.
- How `load_step_interface()` / `load_interface()` read a step's YAML and resolve **job-type variants** by deep-merge.
- How a step builder is **synthesized at runtime** onto a single `TemplateStepBuilder` facade instead of hand-written `builder_*.py` classes.
- How the five `PatternHandler` construction strategies are selected by `resolve_handler()` and steered by declarative *knobs*.
- The `io_view`, the `output_path_token` escape hatch, and the **contract↔spec alignment invariant**.
```

This page describes the heart of the *classless-factory* design introduced in the `2.0.0` release: per-step *data* and per-step *code* both collapse into one declarative file. Where earlier releases already merged the per-step `*_contract.py` + `*_spec.py` pairs into a single `.step.yaml`, `2.0.0` deletes all 45 hand-written `builder_*.py` step-builder classes and *synthesizes* them at runtime from that same interface. The public authoring API (`PipelineDAGCompiler`, `compile_dag_to_pipeline`) is unchanged — the same DAG plus the same config compiles to the same pipeline.

If you are new to Cursus, read [Concepts](index.md) first for the big picture, then come back here.

---

## 1. The `StepInterface` — one object per step

A `StepInterface` is a single [Pydantic](https://docs.pydantic.dev/) model that represents everything about a step: the script's execution contract (container I/O paths, env vars, arguments), the dependency-resolution spec (what it demands and what it supplies), the compute descriptor, and the routing/registry metadata. It replaces what used to be a `(ScriptContract | StepContract, StepSpecification)` tuple with one validated, self-aligning message that is passed among the dependency resolver, the builder, and the assembler.

It lives in `src/cursus/core/base/step_interface.py`. The top-level model is:

```python
class StepInterface(BaseModel):
    step_type: str
    node_type: NodeType = NodeType.INTERNAL
    registry: RegistrySection      # sagemaker_step_type + build-time dep footprint
    compute: ComputeSpec           # the SDK processor/estimator/model/transformer descriptor
    patterns: PatternsSection      # per-axis strategy-selection knobs
    contract: ContractSection      # script I/O: entry_point, inputs, outputs, env_vars, arguments
    spec: SpecSection              # DAG wiring: dependencies + outputs
    variants: Dict[str, VariantDecl]  # job-type overrides
```

The design is deliberately a **superset** of the legacy data classes, so it can stand in for all of them:

| Legacy type | Drop-in on `StepInterface` |
| --- | --- |
| `ScriptContract` / `StepContract` | `StepInterface.contract` (`ContractSection`) — exposes `entry_point`, `expected_input_paths`, `expected_output_paths`, `expected_arguments`, `required_env_vars`, `optional_env_vars`, `framework_requirements`, `description` |
| `StepSpecification` | `StepInterface` / `StepInterface.spec` (`SpecSection`) — exposes `step_type`, `node_type`, `dependencies`, `outputs`, `get_dependency()`, `get_output()`, `get_output_by_name_or_alias()`, `list_required_dependencies()`, `list_optional_dependencies()`, `list_all_output_names()`, `validate_specification()`, `validate_contract_alignment()`, `script_contract` |
| `DependencySpec` / `OutputSpec` | `DependencyDecl` / `OutputDecl` — carry `logical_name` (auto-populated from the dict key), expose `dependency_type` / `output_type` (aliases of `type`), and support `matches_name_or_alias()` |

Because of these accessors, code written against the old `ScriptContract`/`StepSpecification` API keeps working: `iface.expected_input_paths`, `iface.get_output_by_name_or_alias("input_path")`, `iface.script_contract`, and so on all delegate to the appropriate section.

### The `.step.yaml` sections

A single YAML file drives everything. Here is a real (trimmed) example — `tabular_preprocessing.step.yaml`:

```yaml
step_type: TabularPreprocessing
node_type: internal
registry:
  sagemaker_step_type: Processing        # selects the construction handler
  description: Tabular data preprocessing step
patterns:
  direct_input_keys: [DATA, METADATA, SIGNATURE]
compute:
  kind: sklearn                          # build an SKLearnProcessor from config
  framework_version_field: processing_framework_version
contract:
  entry_point: tabular_preprocessing.py
  inputs:
    DATA:
      path: /opt/ml/processing/input/data
      required: true
    SIGNATURE:
      path: /opt/ml/processing/input/signature
      required: false
  outputs:
    processed_data:
      path: /opt/ml/processing/output
  job_arguments:
    - flag: --job_type
      source: job_type
  env_vars:
    required: []
    optional:
      LABEL_FIELD: ''
      OUTPUT_FORMAT: CSV
spec:
  dependencies:
    DATA:
      type: processing_output
      required: true
      compatible_sources: [CradleDataLoading, DummyDataLoading, RedshiftDataLoading]
      semantic_keywords: [data, input, raw, dataset]
  outputs:
    processed_data:
      type: processing_output
      property_path: properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri
      aliases: [input_path, training_data, model_input_data]
```

The sections map directly onto the sub-models:

- **`registry`** (`RegistrySection`) — `sagemaker_step_type` is the routing key that selects the construction handler (see §4). Its `requires` field declares the build-time third-party dependency (`none` for native SageMaker steps, `secure_ai_sandbox_workflow_python_sdk` for the SDK-delegation steps). The valid `sagemaker_step_type` values are pinned in a class-level tuple so a typo is caught at author time.
- **`compute`** (`ComputeSpec`) — a declarative descriptor of the SDK compute object the builder constructs (processor / estimator / model / transformer). `kind` picks the class family (`sklearn`, `xgboost`, `framework`, `script`, `estimator`, `model`, `transformer`). Some fields name config attributes the builder reads at build time (`framework_version_field`, `py_version_field`); others are literal switches or SDK identifiers (`sdk_class`, `framework_name`, `kms_network`, `instance_size_mode`, `lock_training_region`, `retrieve_image`, `requires`). When `kind` is unset the step keeps its own factory. The model validator enforces internal consistency (e.g. `framework`/`estimator`/`model` require an `sdk_class`; `kms_network` is `script`-only; `framework_name` is `model`-only).
- **`patterns`** (`PatternsSection`) — the per-axis strategy-selection knobs read into the bound handler at build time: `step_assembly` (`code` | `step_args` | `delegation`), `include_job_type_in_path`, and `direct_input_keys`. Editing these steers the build with no Python change.
- **`contract`** (`ContractSection`) — the script's execution requirements: `entry_point`, structured `inputs`/`outputs` ports, `env_vars`, `arguments`/`job_arguments`, plus a set of declarative deviation flags (`circular_ref_check`, `skip_inputs`, `input_source_overrides`, `sink`, `source_dir`, `output_path_token`, `include_job_type_in_path`, `computed_env_paths`) that let the handler cover a per-step quirk without a Python override.
- **`spec`** (`SpecSection`) — the DAG-wiring metadata: `dependencies` (what the step demands — each a `DependencyDecl` with `type`, `required`, `compatible_sources`, `semantic_keywords`) and `outputs` (what it supplies — each an `OutputDecl` with `type`, `property_path`, `aliases`, `semantic_keywords`). `compatible_sources` is dependency-only; `property_path`/`aliases` are output-only.

### Path validation

`InputPort.path` and `OutputPort.path` are validated against the SageMaker path conventions at load time:

```python
VALID_INPUT_PREFIXES  = ("/opt/ml/processing/", "/opt/ml/input/data",
                         "/opt/ml/input/config", "/opt/ml/code")
VALID_OUTPUT_PREFIXES = ("/opt/ml/processing/", "/opt/ml/model",
                         "/opt/ml/output/data", "/opt/ml/checkpoints")
```

`entry_point`, when present, must be a `.py` file. Both `entry_point` and the port `path` fields are `Optional`: script-less SageMaker steps (CreateModel / Transform — e.g. `xgboost_model`, `batch_transform`) legitimately declare them as `null`.

---

## 2. Loading an interface

The loader lives in `src/cursus/steps/interfaces/__init__.py`. There are two entry points:

```python
from cursus.steps.interfaces import load_interface, load_step_interface

# Preferred: get the StepInterface directly.
iface = load_interface("TabularPreprocessing")

# Backward-compatible: returns a (contract, spec) 2-tuple.
contract, spec = load_step_interface("CradleDataLoading", job_type="calibration")
```

- `load_interface(step_name, job_type=None)` returns the validated `StepInterface`.
- `load_step_interface(step_name, job_type=None)` returns a `(ContractSection, StepInterface)` tuple where both elements are *views onto the same object* — `[0]` is the `ContractSection` (a `ScriptContract` drop-in) and `[1]` is the whole `StepInterface` (a `StepSpecification` drop-in). New code should prefer `load_interface`.

The former per-step `steps/specs/` and `steps/contracts/` folders are gone; the `.step.yaml` files under `steps/interfaces/` are the sole source.

### Name resolution and caching

`_resolve_interface_path()` maps a PascalCase step name to a file. It first tries the naming-convention filename (`_step_name_to_filename` handles known acronyms like `PyTorch → pytorch`, `XGBoost → xgboost`), then falls back to a normalized scan that matches on a separator- and case-insensitive `_canonical_key` — so a new acronym step resolves even if it is not in the hardcoded abbreviation table.

Loaded interfaces are cached by `step_name:job_type`. `clear_interface_cache()` drops the cache (useful when hot-reloading edited YAML). External *step packs* can register additional interface directories via `register_pack_interface_dir()`; they are searched **after** the package directory, so a package interface always wins on a name clash (the additive invariant).

---

## 3. Job-type variants (deep merge)

Many steps run in several *job types* (`training`, `validation`, `calibration`, …) that tweak the spec — usually a distinct `step_type`, some required-flag changes, and different `compatible_sources` so the connection graph wires the right edges. These are declared in a `variants:` block:

```yaml
variants:
  training:
    step_type: RiskTableMapping_Training
    spec:
      dependencies:
        model_artifacts_input:
          required: false
  validation:
    step_type: RiskTableMapping_Validation
    spec:
      dependencies:
        model_artifacts_input:
          required: true
          compatible_sources: [RiskTableMapping_Training]
```

When `load_interface(..., job_type="validation")` is called, `StepInterface.from_yaml()` applies that variant's `spec` / `contract` / `patterns` overrides **before validation** using `_deep_merge()`. The merge is *recursive*:

```python
def _deep_merge(base, override):
    result = dict(base)
    for key, ov in override.items():
        bv = result.get(key)
        if isinstance(bv, dict) and isinstance(ov, dict):
            result[key] = _deep_merge(bv, ov)   # merge nested dicts key-by-key
        else:
            result[key] = ov                     # non-dict values replace outright
    return result
```

Deep merge matters because variants routinely restate **only the ports they tweak**. A shallow `{**base, **variant}` at the section level would drop every base port the variant happened to omit — that was a real latent bug (it dropped `hyperparameters_s3_uri` from a variant, which then violated the alignment invariant of §6 and raised at construction).

If a `job_type` is requested but the step declares variants and **none matches**, the loader falls back to the base spec and logs a warning rather than raising. Step configs deliberately do not restrict `job_type` to the declared variant set (most validate it only as "lowercase alphanumeric"), so a legitimate value like `munged` must resolve to the base spec. This can only ever *under-tighten* an optional dependency — a genuinely missing required dependency is still caught downstream by the dependency resolver and by the alignment check.

---

## 4. Builder synthesis onto `TemplateStepBuilder`

Historically each step had a hand-written `<Name>StepBuilder` class. In `2.0.0` all 45 of those files are deleted. A step builder is now a **thin shell** over one shared facade:

```python
class TabularPreprocessingStepBuilder(TemplateStepBuilder):
    STEP_NAME = "TabularPreprocessing"
```

`TemplateStepBuilder` (in `src/cursus/core/base/builder_templates.py`) is a `StepBuilderBase` subclass. It keeps the same `__init__` contract `StepBuilderBase` defines — `config` plus the four keyword components (`sagemaker_session`, `role`, `registry_manager`, `dependency_resolver`) that the `PipelineAssembler` passes, plus an optional trailing `spec` — and in `__init__` it:

1. Loads its own interface via `load_step_interface(self.STEP_NAME, job_type=getattr(config, "job_type", None))` when no `spec` is passed — the same loader everything else uses, threading `config.job_type` through so a variant-bearing step resolves its job-typed spec.
2. Calls `_auto_bind_handler()`, which reads the step's `sagemaker_step_type` from the registry (`get_sagemaker_step_type(STEP_NAME)`) plus the interface's `patterns:` knobs, and binds a `PatternHandler` via `resolve_handler(...)`.

The abstract builder methods delegate to that bound handler:

```python
def _get_inputs(self, inputs):  return self._handler.get_inputs(self, inputs)
def _get_outputs(self, outputs): return self._handler.get_outputs(self, outputs)
def create_step(self, **kwargs): return self._handler.build_step(self, **kwargs)
```

### Fabrication at runtime — no file at all

Even the two-line shell need not exist as a file. `StepCatalog`'s `builder_discovery._synthesize_builder(step_name)` fabricates it on demand:

```python
synthesized = type(
    f"{step_name}StepBuilder",
    (TemplateStepBuilder,),
    {"STEP_NAME": step_name, ...},
)
```

For any registry step that (a) has a `.step.yaml` interface and (b) routes via `resolve_handler`, the catalog builds this subclass and caches it per process (in `_synthesized_builders`, keyed on the canonical registry name). A subclass (not a `functools.partial`) is used so `__name__`, `issubclass`, `__mro__`, and `self.STEP_NAME` all behave. SDK-delegation steps are the one carve-out: they need a live SAIS `*Step` class injected as the `sdk_step_class` knob, so `_synthesize_builder` materializes that class through the lazy `sdk_bindings` helpers (`is_sdk_delegation_step` / `resolve_sdk_step_class`) and returns `None` — leaving them undiscoverable — when the SDK is absent (offline).

Because synthesis is registry-driven, there is **no importable per-step builder module** anymore: the whole `cursus.steps.builders/` folder (and its `builder_*.py` files) was deleted, and there is no `cursus.steps.builders` re-export to import a name from. Consumers obtain a builder class through the catalog instead — `StepCatalog.load_builder_class("XGBoostTraining")` (which delegates to `builder_discovery.load_builder_class` → `_synthesize_builder`), or `StepCatalog.get_builder_map()` / `builder_discovery.discover_builder_classes()` for the whole registry-wide map. Each returns a synthesized `TemplateStepBuilder` subclass with the correct `STEP_NAME`.

```{admonition} Invariance guarantee
:class: note

The assembler instantiates a builder by calling `builder_cls(**five_kwargs)` with **no `isinstance` gate** — a synthesized class is indistinguishable from a hand-written one at that call site. The step↔step wiring graph keys entirely on `.step.yaml` spec data (`spec.step_type`, `compatible_sources`, `property_path`, `logical_name`) carried on `builder.spec`, never on a Python class. So collapsing 45 classes into one facade leaves every edge intact.
```

---

## 5. The five construction patterns (`PatternHandler` strategies)

Routing is by `sagemaker_step_type` **only** (never by step name — `DummyTraining` is a `Processing` step and must route as Processing). `Processing` is the one type sub-discriminated by `step_assembly`. The dispatch table lives in `src/cursus/registry/strategy_registry.py`; handlers self-register via `@register_strategy(...)` decorations on their classes.

`axis_name_for_step_type()` encodes the routing rule:

```python
def axis_name_for_step_type(sagemaker_step_type, step_assembly=None):
    if sagemaker_step_type == "Processing":
        return "step_assembly", (step_assembly or "code")
    return "sagemaker_step_type", sagemaker_step_type
```

There are six construction *verbs*, but the two Processing assembly modes collapse to one handler, giving **five** `PatternHandler` classes:

| Handler | Routes on | SageMaker step built | Distinctive behavior |
| --- | --- | --- | --- |
| `ProcessingHandler` | `step_assembly` = `code` (2A) or `step_args` (2B) | `ProcessingStep` | Shared spec×contract input/output join; 2A passes `processor=` + `code=`, 2B calls `processor.run()` and passes `step_args=` only |
| `TrainingHandler` | `sagemaker_step_type` = `Training` | `TrainingStep` | `get_inputs` returns `Dict[str, TrainingInput]` keyed by **channel**; `input_path` fans out to `train`/`val`/`test`; output is a single `str`/`Join`; order is inputs → outputs → compute (estimator gets `output_path`) |
| `ModelCreationHandler` | `sagemaker_step_type` = `CreateModel` | `CreateModelStep` | `get_inputs` is a single-key `{"model_data": ...}` passthrough; `get_outputs` returns `None`; **caching is dropped** (warns on `enable_caching=True`) |
| `TransformHandler` | `sagemaker_step_type` = `Transform` | `TransformStep` | `get_inputs` returns a `(TransformInput, model_name)` tuple; compute runs **last**, consuming both `model_name` and `output_path` |
| `SDKDelegationHandler` | `sagemaker_step_type` = `CradleDataLoading` / `RedshiftDataLoading` / `MimsModelRegistrationProcessing`, or `step_assembly` = `delegation` | a SAIS `*Step` subclass | No `make_compute`; the SDK step class is injected via the `sdk_step_class` knob; three `input_mode`s (`none`, `resolve_s3`, `mims_ordered`) |

`resolve_handler(sagemaker_step_type, step_assembly, knobs)` does the lookup, merges the registry's `preset_knobs` **under** the caller's knobs, and instantiates the handler:

```python
def resolve_handler(sagemaker_step_type, step_assembly=None, knobs=None):
    axis, name = axis_name_for_step_type(sagemaker_step_type, step_assembly)
    info = resolve_strategy(axis, name)       # raises NoBuilderError for Base/Lambda/unknown
    merged = {**info.preset_knobs, **(knobs or {})}
    return info.handler(knobs=merged)
```

### Knobs: per-step behavior as data

A `PatternHandler` is **stateless config**. It holds a declarative `knobs` dict and receives the owning `TemplateStepBuilder` (`b`) on each call, reading `b.config` / `b.spec` / `b.contract` / `b.role` / `b.session` and calling base helpers (`b._get_step_name()`, `b._get_base_output_path()`, `b.extract_inputs_from_dependencies()`, …). Every knob a strategy accepts is described by a `KnobSpec` in the registry, so the same table drives both the build and the `cursus strategies` introspection tools — the docs can never drift from behavior.

Two smooth-migration behaviors are built in:

- **Per-step overrides still win.** `_overrides(builder, method_name)` detects whether a shell defines its own `_get_inputs` / `_get_outputs` / `_create_processor` / `_create_estimator` / …; if so the handler prefers it via the MRO. A migrating builder can keep just its genuinely deviating method and delete the boilerplate.
- **Compute resolution order.** In `build_step`, if no `make_compute` knob is set, a per-step `_create_processor`/`_create_estimator`/… override wins; else the declarative `compute` descriptor drives `b._create_compute()`; else a `NotImplementedError` is raised.

Finally, `TemplateStepBuilder.create_step()` guarantees the step carries its spec/contract by calling `PatternHandler._attach_spec(self, step)` (idempotent `setattr(step, "_spec", ...)` / `_contract`), since `step._spec` feeds the resolver-enrichment path.

---

## 6. The contract↔spec alignment invariant

The single most important property of a `StepInterface` is that its **contract and spec are aligned**. Because both are sections of one object validated together, alignment is a *construction-time invariant*, not a separate check that could rot.

`StepInterface._sync_and_align()` (a Pydantic `model_validator(mode="after")`) enforces it every time an interface is built:

```python
# Contract inputs must each have a matching spec dependency.
missing_deps = set(self.contract.inputs) - set(self.spec.dependencies)
if missing_deps:
    raise ValueError(f"Contract inputs missing from spec dependencies: {missing_deps}")

# Contract outputs must each have a matching spec output.
missing_outs = set(self.contract.outputs) - set(self.spec.outputs)
if missing_outs:
    raise ValueError(f"Contract outputs missing from spec outputs: {missing_outs}")
```

The same validator also (a) propagates `step_type`/`node_type` onto the spec so `SpecSection` is a self-contained `StepSpecification` stand-in, and (b) reconciles the top-level `compute` with the back-compat `contract.compute` mirror — exactly one side should be populated, and if both are (mid-migration) they must agree.

There is also a public `validate_contract_alignment()` that returns a `ValidationResult` (mirroring the legacy `StepSpecification.validate_contract_alignment`): every contract input must have a matching spec dependency, and every contract output must be satisfied by a matching spec output **logical name or alias**. `StepBuilderBase.__init__` runs this check whenever both a spec and a contract are present and raises if it fails.

In short: **you cannot construct a mis-aligned interface.** Every contract port must have a matching spec port; the reverse is allowed (a spec may declare extra dependencies/outputs and an output may carry aliases). This is why the old standalone "Contract↔Spec" validation tier could be deleted — re-checking it at runtime would be a tautology.

---

## 7. The `io_view` — introspecting wiring without building

Because a step is no longer a readable builder class, `src/cursus/steps/interfaces/io_view.py` renders a structured "what wires into / out of this step" view from the interface plus its bound handler. It is the path/wiring analogue of the `catalog.step_spec` view. Two functions back the `cursus steps` CLI and the `steps.*` MCP tools:

- **`describe_step_io(step_name, job_type=None)`** — for each dependency reports `container_path` (where the input lands in the container), `required`, `type`, `compatible_sources`, `semantic_keywords`, and — for Training steps — the SageMaker training `channels` it fans out into. For each output it reports `container_path` (source), `property_path` (the runtime `properties.*` reference a downstream step resolves against), `type`, `aliases`, and `data_type`. Pure introspection — no config, no SageMaker session.

  The channel fan-out delegates to `TrainingHandler.channels_for(...)`, the *single source* of the channel rule that `TrainingHandler.get_inputs` uses at build time, so the view can never drift from what the builder emits.

- **`describe_step_patterns(step_name, job_type=None)`** — the per-axis PATTERN view: which construction handler binds, the compute descriptor, declared env vars and job arguments, active input deviations, the output-destination shape, and a build-time-vs-runtime dependency rollup. Where a builder still hand-overrides a method it is marked `custom_override` so you can see exactly where the step departs from the declarative patterns.

```bash
# The CLI surfaces both views.
cursus steps io TabularPreprocessing
cursus steps patterns TabularPreprocessing
```

See the [Step catalog](../reference/generated/step_catalog.md) and [MCP tools](../reference/generated/mcp_tools.md) references for the full surface, and the [CLI](../cli.rst) reference for command details.

---

## 8. The `output_path_token` override

By default, the S3 output-destination prefix for a step's outputs is **derived from the step name** — `canonical_to_snake(step_type)` (the package's PascalCase→snake utility, acronyms handled) — the convention for essentially all steps. The `ProcessingHandler.get_outputs` (and the Training/Transform equivalents) build the destination as:

```python
token = getattr(b.contract, "output_path_token", None) or canonical_to_snake(b.spec.step_type)
values = [b._get_base_output_path(), token]
if include_job_type and getattr(b.config, "job_type", None):
    values.append(b.config.job_type)
values.append(logical_name)
destination = Join(on="/", values=values)
```

`contract.output_path_token` is an **opt-in, default-`None` escape hatch**. When set to a non-empty string it is used **verbatim** as that path segment instead of the derived token. This is needed only when an *external* consumer keys off a fixed S3 folder name that does not match the Cursus step name — for example, an external tool that scans `<pipeline>/Model_Metric_Generation_Step/` for `.metric` files. For every other step the derived convention holds and you never set this field.

The related `include_job_type_in_path` knob (default `True`, read knob → contract → default) controls whether `config.job_type` is a segment of the destination — genuinely variable across steps, so it stays a per-step knob rather than being derived.

---

## 9. Putting it together

The end-to-end flow for one node in a compiled pipeline:

1. The DAG compiler / assembler needs a builder for step `X`. `StepCatalog` returns either a physical shell or a **synthesized** `TemplateStepBuilder` subclass with `STEP_NAME = "X"`.
2. The assembler calls the builder with five keyword arguments — `builder_cls(config=config, sagemaker_session=..., role=..., registry_manager=..., dependency_resolver=...)`. It does **not** pass a `spec`; the shell loads its own.
3. `TemplateStepBuilder.__init__` loads `X.step.yaml` via `load_step_interface` (variant-resolved on `config.job_type`) when no `spec` was passed, which validates the **contract↔spec alignment invariant**, then `_auto_bind_handler()` picks a `PatternHandler` from `sagemaker_step_type` + `patterns` knobs.
4. The dependency resolver wires edges using `spec` data (`compatible_sources`, `property_path`, aliases) carried on `builder.spec`.
5. `create_step(**kwargs)` delegates to the handler's `build_step`, which runs the shared input/output join (or a per-step override), builds the compute object from the `compute` descriptor, and returns the concrete SageMaker step with `_spec`/`_contract` attached.

Authoring a new step is therefore: **one `.step.yaml` (with a `registry:` block) + one config class** — no builder file. The difference between a Processing step and a Training step is one string (`sagemaker_step_type`) in the interface.

---

## See also

- [Concepts](index.md) — the conceptual overview index.
- [Step catalog](../reference/generated/step_catalog.md) — the discovery layer that synthesizes and caches builders.
- [MCP tools](../reference/generated/mcp_tools.md) — the `steps.*` and `strategies.*` agent tools backed by `io_view` and the strategy registry.
- [Pipeline catalog](../reference/generated/pipeline_catalog.md) — ready-made pipelines built from these steps.
- [CLI](../cli.rst) — `cursus steps` and `cursus strategies` commands.
- [API reference](../api/index.rst) — module-level API docs.

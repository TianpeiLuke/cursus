# The Configuration System

Every pipeline that Cursus compiles is driven by **configuration objects** — one per
step — that carry the concrete values a step builder needs: the S3 bucket, the IAM role,
the instance type, the entry-point script, hyperparameters, and so on. Cursus models
these as **Pydantic** classes with a strict inheritance hierarchy, serializes a whole set
of them into a single JSON file, and records the type metadata (`__model_type__` markers
plus a `config_types` map) needed to map each node back to its config class when the file
is read back.

This page explains:

- the two base classes — [`BasePipelineConfig`](#the-config-class-hierarchy) and
  `ProcessingStepConfigBase` — and the three-tier field model they enforce;
- how a set of configs is merged and saved, and how `metadata.config_types` records the
  config **class** behind each DAG node;
- field inheritance and the `config_fields` managers
  (`UnifiedConfigManager`, `InheritanceAwareFieldGenerator`);
- the `DAGConfigFactory` workflow for building a full config set from a DAG, and the
  `ConfigurationGenerator` / `field_extractor` helpers underneath it;
- the `cursus config` CLI for introspecting what a DAG requires.

For how those configs are consumed once built, see
[DAG & Compilation](../concepts/dag_and_compilation.md) and
[Step interfaces](../concepts/step_interfaces.md).

---

## The config class hierarchy

All step configs descend from `BasePipelineConfig`
(`src/cursus/core/base/config_base.py`), an abstract Pydantic `BaseModel`. Processing
steps additionally descend from `ProcessingStepConfigBase`
(`src/cursus/steps/configs/config_processing_step_base.py`), which adds SageMaker
Processing fields:

```
BasePipelineConfig            (author, bucket, role, region, service_name, ...)
├── XGBoostTrainingConfig     (training-specific fields)
├── ...
└── ProcessingStepConfigBase  (processing_instance_count, processing_entry_point, ...)
    ├── TabularPreprocessingConfig
    ├── XGBoostModelEvalConfig
    └── ...
```

`BasePipelineConfig` declares the fields shared by every step — required user inputs like
`author`, `bucket`, `role`, `region`, `service_name`, `pipeline_version`, and
`project_root_folder`, plus optional system fields with defaults (`model_class`,
`framework_version`, `py_version`, `enable_caching`, `max_runtime_seconds`, …).

`ProcessingStepConfigBase` adds the ten processing-specific fields on top:
`processing_instance_count`, `processing_volume_size`,
`processing_instance_type_large`/`_small`, `use_large_processing_instance`,
`skip_volume_kms`, `processing_source_dir`, `processing_entry_point`,
`processing_script_arguments`, and `processing_framework_version`.

### The three-tier field model

Cursus organizes every config's fields into three **tiers**, and much of the tooling
keys off this distinction. The tiering embodies a **self-contained derivation** principle:
rather than a centralized field-derivation engine that has to understand every step type,
each config class owns its own derivations and its own encapsulation. Tier 3 fields are
private (`PrivateAttr`) with read-only properties precisely so a user *cannot* accidentally
override a computed value — the class stays the single source of truth. The categories are
derived automatically by `BasePipelineConfig.categorize_fields()` — nothing is
hand-maintained:

| Tier | Name | What it is | How it's detected |
|------|------|------------|-------------------|
| 1 | **Essential** | Required user inputs, no default | public model field where `field_info.is_required()` is `True` |
| 2 | **System** | Optional inputs with sensible defaults | public model field that is not required |
| 3 | **Derived** | Read-only computed values | a public `@property` that is not a model field |

Tier 3 is the interesting one: derived fields are **not** stored as writable Pydantic
fields. They are private attributes (`PrivateAttr`) exposed through read-only properties
and computed once in a `@model_validator(mode="after")`. For example, `pipeline_name`,
`aws_region`, `pipeline_s3_loc`, and `effective_source_dir` are all derived from Tier 1/2
inputs:

```python
@property
def pipeline_name(self) -> str:
    if self._pipeline_name is None:
        self._pipeline_name = (
            f"{self.author}-{self.service_name}-{self.model_class}-{self.region}"
        )
    return self._pipeline_name
```

`model_dump()` is overridden so these derived properties appear in the serialized output
even though they are not real fields. `ProcessingStepConfigBase` extends this pattern with
`effective_source_dir`, `effective_instance_type`, and `script_path`.

You can inspect the tiers of any instance:

```python
config.categorize_fields()
# {'essential': ['author', 'bucket', 'role', ...],
#  'system':    ['model_class', 'framework_version', ...],
#  'derived':   ['aws_region', 'pipeline_name', 'pipeline_s3_loc', ...]}
config.print_config()   # pretty tier-by-tier dump
```

### Inheritance via `from_base_config`

Because most fields are shared, you rarely fill a child config from scratch. Every config
inherits `BasePipelineConfig.from_base_config(base_config, **kwargs)`, which copies the
parent's **public** fields (via `get_public_init_fields()` — Tier 1 + non-`None` Tier 2)
and layers the child-specific `kwargs` on top:

```python
base = BasePipelineConfig(
    author="jdoe", bucket="my-bucket", role="arn:aws:iam::...:role/exec",
    region="NA", service_name="fraud", pipeline_version="1.2.0",
    project_root_folder="my_project",
)

train_cfg = XGBoostTrainingConfig.from_base_config(
    base,
    # training-specific fields only:
    training_entry_point="train_xgb.py",
    training_instance_type="ml.m5.4xlarge",
)
```

`ProcessingStepConfigBase.get_public_init_fields()` overrides the base to also propagate
the processing fields, so a processing child receives both the pipeline-level and the
processing-level inputs. This is exactly the mechanism the factory uses under the hood.

### Path resolution

Configs are deliberately portable: source-dir and entry-point paths are **not** validated
for existence at construction time. Instead `resolve_hybrid_path()` /
`effective_source_dir` resolve paths lazily against the project root at execution time,
so the same config JSON works across deployment layouts. See
[Path resolution](../concepts/path_resolution.md) for the resolution strategies.

---

## The config JSON: `metadata.config_types`

A pipeline's configs are persisted as a single JSON file via `merge_and_save_configs`
(exported from `cursus.core.config_fields`) and reconstructed via `load_configs`:

```python
from cursus.core.config_fields import merge_and_save_configs, load_configs

merge_and_save_configs([base_cfg, train_cfg, eval_cfg], "config.json")
loaded = load_configs("config.json")   # -> {"shared": {...}, "specific": {step: {...}}}
```

Both delegate to `UnifiedConfigManager` (`unified_config_manager.py`). The saved file has
two top-level sections:

```json
{
  "metadata": {
    "config_types": {
      "XGBoostTraining": "XGBoostTrainingConfig",
      "TabularPreprocessing_training": "TabularPreprocessingConfig"
    },
    "created_at": "2026-06-26T10:30:00",
    "field_sources": { "...": "..." }
  },
  "configuration": {
    "shared":   { "author": "jdoe", "bucket": "my-bucket", "...": "..." },
    "specific": {
      "XGBoostTraining":              { "training_entry_point": "train.py" },
      "TabularPreprocessing_training": { "job_type": "training" }
    }
  }
}
```

Two ideas make this work:

- **Field categorization.** `StepCatalogAwareConfigFieldCategorizer` splits fields into a
  `shared` block (identical across all configs — written once) and per-step `specific`
  blocks. `UnifiedConfigManager._verify_essential_structure()` asserts the `shared` /
  `specific` shape and warns on field conflicts.
- **`metadata.config_types`.** This maps each **step name** (the node key) to its config
  **class name**, recorded as metadata. The class-detection path
  (`detect_config_classes_from_json` / `ConfigClassDetector`) reads these class names to
  decide which config classes to import when a saved file is deserialized. The step name is
  produced by `TypeAwareConfigSerializer.generate_step_name()`, which looks the config class
  up in `CONFIG_STEP_REGISTRY` and appends distinguishing attributes (`job_type`,
  `data_type`, `mode`) — e.g. `TabularPreprocessing` + `job_type="training"` →
  `TabularPreprocessing_training`.

`load_configs` itself discovers the available config classes through the step catalog /
registry (see [Registry & discovery](../concepts/registry_and_discovery.md)) — so you never
pass a class map — and returns a `{"shared": …, "specific": {step: …}}` structure of
deserialized **field values** (each `specific` block is also tagged with its own
`__model_type__`). During this pass, nested objects that carry a `__model_type__` marker are
rebuilt into their real classes by `TypeAwareConfigSerializer`; to reconstruct a full typed
config instance from a whole block, hand that block to `deserialize_config`.

---

## The `config_fields` managers

`src/cursus/core/config_fields/` holds the machinery that saves, loads, categorizes, and
introspects configs.

### `UnifiedConfigManager`

The single entry point (`get_unified_config_manager(workspace_dirs=None)`, from
`cursus.core.config_fields.unified_config_manager`, returns a per-`workspace_dirs` cached
instance). It consolidates three formerly-separate systems and provides:

- `get_config_classes(project_id=None)` — discovers all available config classes through
  the [step catalog](../reference/generated/step_catalog.md)
  (`build_complete_config_classes`), with layered fallbacks.
- `get_field_tiers(instance)` — delegates to the config's own `categorize_fields()`.
- `save(...)` / `load(...)` — the implementations behind `merge_and_save_configs` /
  `load_configs`, including the metadata + `config_types` assembly shown above.
- `serialize_with_tier_awareness(obj)` — recursive serialization with a lightweight
  `SimpleTierAwareTracker` guarding against circular references.

### `inheritance_aware_field_generator`

`InheritanceAwareFieldGenerator` (obtained via
`get_inheritance_aware_field_generator(workspace_dirs, project_id)`) produces **form
field** descriptions used by interactive UIs. It extends the three-tier model with a
fourth, inheritance-aware tier:

| Tier | Meaning |
|------|---------|
| `essential` | required, new to this config |
| `system` | optional-with-default, new to this config |
| `inherited` | comes from a parent config — pre-populated with the parent's value, marked `can_override` |
| `derived` | computed; hidden from the UI |

Given an `inheritance_analysis` (parent values + immediate parent, typically from the step
catalog), `get_inheritance_aware_form_fields(config_class_name, ...)` marks any field
present in the parent's values as `inherited`, pre-fills its `default` from the parent,
and flips `required` to `False`. This is what lets a config UI show "already answered"
fields greyed-out but overridable. The generator builds on `UnifiedConfigManager` for the
underlying tier categorization.

---

## The `DAGConfigFactory` workflow

`DAGConfigFactory` (`src/cursus/api/factory/dag_config_factory.py`) is the high-level,
interactive way to build a **complete** config set for a DAG. You give it a DAG; it works
out which config class each node needs and walks you through supplying values.

### Construction: mapping nodes to config classes

```python
from cursus.api.dag import import_dag_from_json
from cursus.api.factory import DAGConfigFactory

dag = import_dag_from_json("dag.json")
factory = DAGConfigFactory(dag, anchor_file=__file__)   # anchor for path resolution
```

On construction the factory:

1. **Maps every DAG node to a config class.** A node name follows the pattern
   `{CanonicalStepName}_{job_type}` (e.g. `XGBoostModelEval_calibration`). The factory
   strips the job-type suffix, looks the canonical name up in the registry
   (`get_config_step_registry`), and resolves it to a class via `UnifiedConfigManager`.
2. **Pre-computes inheritance** for each class (does it derive from
   `ProcessingStepConfigBase`? from `BasePipelineConfig`?) and caches it.
3. **Pre-extracts per-step field requirements** so later calls are instant.

`project_root` / `anchor_file` push a project root for hybrid path resolution ("Strategy
0"), so generated configs anchor their `source_dir` / `processing_source_dir` correctly
even without a compiler running first.

### Supplying values

You set the shared base configs first, then per-step values:

```python
factory.set_base_config(
    author="jdoe", bucket="my-bucket", role="arn:...:role/exec",
    region="NA", service_name="fraud", pipeline_version="1.2.0",
    project_root_folder="my_project",
)
# Only needed if any node is a processing step:
factory.set_base_processing_config(processing_instance_count=2)

factory.set_step_config(
    "XGBoostTraining",
    training_entry_point="train_xgb.py",
    training_instance_type="ml.m5.4xlarge",
)
```

`set_step_config` validates the prerequisites (base config set? processing base set if the
step needs it?), then builds the instance immediately with the correct inheritance via
`from_base_config`, raising a detailed error if validation fails. It stores both the raw
inputs (for serialization) and the validated instance.

**Bare-name resolution.** DAG nodes carry a job-type suffix, but you can pass the bare
canonical name plus `job_type` and the factory resolves it to the real node key
(`_resolve_step_name_to_node`):

```python
# DAG node is "PercentileModelCalibration_calibration"; either form works:
factory.set_step_config("PercentileModelCalibration", job_type="calibration", ...)
```

Resolution order is: exact node-key match → `{step_name}_{job_type}` composed key →
the single node whose canonical base equals `step_name`. If several nodes share the base,
the factory warns and leaves the name unchanged so you disambiguate explicitly. Related
helpers: `is_dag_step()` and `configure_step_if_present()` (which **warns** instead of
silently skipping an unknown step — a guard against typo'd/renamed step names).

### Auto-configuration and generation

Steps whose only step-specific fields are Tier 2 (optional) need no user input — the
factory reports them via `get_pending_steps()` / `can_auto_configure_step()` and fills
them from inherited fields alone.

```python
factory.get_pending_steps()          # steps still needing required input
factory.get_configuration_status()   # per-config completion booleans

configs = factory.generate_all_configs()   # -> List[BaseModel]
```

`generate_all_configs()`:

1. auto-configures every eligible step;
2. raises if any step still has missing required input;
3. runs `validate_dag_config_alignment(raise_on_error=True)`;
4. returns the validated instances.

### `validate_dag_config_alignment`

This enforces the **DAG ↔ config invariant**: for each configured instance, the key it
will serialize under (`instance._derive_step_name()`, i.e.
`{registry_step_name}[_{job_type}]`) must equal the DAG node key it was stored under, and
every DAG node must have resolved to a config class. A mismatch means a node was
configured with the **wrong step type** — caught here at generate time with a clear
message, rather than surfacing later as an opaque "no config for node" at compile time.

The resulting list can be fed straight into `merge_and_save_configs` (or into the
compiler; see [DAG & Compilation](../concepts/dag_and_compilation.md)).

### `ConfigurationGenerator` and `field_extractor`

Two lower-level helpers sit under the factory:

- **`ConfigurationGenerator`** (`configuration_generator.py`) does the actual instance
  assembly. `generate_config_instance` / `generate_all_instances` pick the inheritance
  strategy (processing → base → standalone), merge base + processing + step inputs, and
  prefer `from_base_config` when present. It also offers `validate_generated_configs` and
  `get_config_summary`.
- **`field_extractor`** (`field_extractor.py`) reads field requirements straight off a
  Pydantic class:
  - `extract_field_requirements(cls)` → a list of `{name, type, description, required,
    default, ...}` dicts;
  - `extract_non_inherited_fields(derived, base)` → only the fields a subclass **adds**
    (this is how the factory shows *step-specific* requirements without repeating the base
    fields);
  - `extract_field_constraints(cls, field)` → the closed set of legal values for a field.

#### `allowed_values`

`extract_field_constraints` surfaces **`allowed_values`** for any field whose values are
constrained. It reads the constraint drift-proofly from the type first — a `Literal[...]`
or `Enum` annotation (`source: "literal"`) — and, failing that, scrapes the
`allowed = {...}` set out of the field's `@field_validator` source
(`source: "validator"`), also inferring `case_sensitive` from whether the validator
lower/upper-cases before comparing. When present it is attached to the field dict as
`allowed_values` + `case_sensitive`, so tooling can show a user the *legal* values rather
than just a bare type — closing the class of "wrong enum case / invalid enum value"
errors.

---

## The `cursus config` CLI

`cursus config` is the **introspection** side of the factory: it answers "what
configuration does this DAG require?" without running the interactive, stateful workflow.
Generating a populated config set is inherently multi-call (`set_base_config` →
`set_step_config` … → `generate_all_configs`), so that part lives in the Python API, not a
one-shot command.

```bash
# All requirements: base pipeline config + every step
cursus config requirements dag.json

# Just one step
cursus config requirements dag.json --step XGBoostTraining

# Machine-readable
cursus config requirements dag.json --format json
```

Under the hood the command loads the DAG, builds a `DAGConfigFactory`, and prints
`get_base_config_requirements()` plus `get_step_requirements(node)` for each node — the
same field dicts (`name`, `type`, `required`, `default`, `description`) that
`field_extractor` produces. Text output marks each field required/optional; JSON output
returns the raw structure. See the full [CLI reference](../cli.rst).

---

## How it fits together

```
DAG ──▶ DAGConfigFactory
          │  (registry + UnifiedConfigManager: node → config class)
          │  set_base_config / set_step_config  (from_base_config inheritance)
          ▼
       ConfigurationGenerator ──▶ List[BasePipelineConfig subclasses]
          │                          │
          │ field_extractor          │ merge_and_save_configs
          │ (requirements +          ▼
          │  allowed_values)     config.json
          │                       { metadata.config_types, configuration.{shared,specific} }
          ▼                          │ load_configs (TypeAwareConfigSerializer)
     cursus config requirements      ▼
                                  {shared, specific} field values ──▶ compiler / step builders
```

## Related pages

- [DAG & Compilation](../concepts/dag_and_compilation.md) — how configs feed the compiler.
- [Step interfaces](../concepts/step_interfaces.md) — the contracts configs satisfy.
- [Registry & discovery](../concepts/registry_and_discovery.md) — node ↔ class resolution.
- [Path resolution](../concepts/path_resolution.md) — hybrid source-dir resolution.
- [Step catalog](../reference/generated/step_catalog.md) — the class-discovery backbone.
- [CLI reference](../cli.rst) · [API reference](../api/index.rst)

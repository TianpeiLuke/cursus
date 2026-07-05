# Generate & Inspect Configs

Every Cursus pipeline is compiled from two inputs: a **DAG** (the topology of steps) and a
**config set** (the per-step settings that fill in each step's Pydantic config object). This
guide covers the second input — how to discover *what* a DAG requires, and how to *generate* a
populated config set for it using [`DAGConfigFactory`](../api/index.rst).

The workflow has two distinct modes:

| Mode | Question it answers | Where it lives |
| --- | --- | --- |
| **Inspect** | "What fields does this DAG need before I can compile it?" | `cursus config requirements` (CLI), `config.requirements` / `config.field_info` (MCP), `DAGConfigFactory.get_*_requirements()` |
| **Generate** | "Build me the actual config objects and save them to JSON." | `DAGConfigFactory.set_base_config` / `set_step_config` / `generate_all_configs`, then `merge_and_save_configs` |

Inspection is **stateless** and JSON-clean, so it is exposed over the CLI and MCP boundaries.
Generation is **stateful** — the factory accumulates a base config plus one config per step
across many calls and produces live Pydantic objects — so it is only available through the
Python API.

See also: [DAG & Compilation](../concepts/dag_and_compilation.md) for how the config set feeds
the compiler, and [Config system](../concepts/config_system.md) for the three-tier field model.

---

## 1. Inspect what a DAG requires

### From the CLI

`cursus config requirements` loads a serialized DAG JSON and reports the config fields each node
needs, without building anything:

```bash
# Base config + every step's fields
cursus config requirements dag.json

# One step only
cursus config requirements dag.json --step XGBoostTraining

# Machine-readable output
cursus config requirements dag.json --format json
```

The text output groups fields by scope. Each line shows the field name, its type string, and a
`[required]` / `[optional]` marker, followed by the field description:

```text
Config requirements for DAG: dag.json

Base pipeline config:
    - author (str) [required]
        ...
    - bucket (str) [required]
        ...

Per-step config:
  XGBoostTraining:
    - training_entry_point (str) [required]
        ...
```

Under the hood the command (`src/cursus/cli/config_cli.py`) calls
`import_dag_from_json(dag_file)`, constructs a `DAGConfigFactory(dag)`, and reads out
`get_base_config_requirements()` plus `get_step_requirements(node)` for each node. It is purely
introspective — it does **not** generate or save any config.

### From MCP tools

The same introspection is available as tools in the `config.*` namespace
(`src/cursus/mcp/tools/config.py`). See [MCP tools](../reference/generated/mcp_tools.md).

- **`config.requirements`** — pass a DAG; get back base-config requirements, base-processing
  requirements (if any step needs them), per-step non-inherited requirements, the
  `config_class_map` (node → config class name), and `pending_steps` (steps that still need
  user input).
- **`config.field_info`** — pass a single `config_class` name (e.g.
  `"XGBoostTrainingConfig"`); get its full field list. Add `categorized: true` to split into
  `required` and `optional` groups.
- **`config.load`** — summarize a previously saved merged-config JSON file (shared field names,
  per-step field names/counts).
- **`config.merge_save`** — intentionally *unsupported* over the JSON boundary; it returns an
  explanatory error because saving configs requires live Pydantic objects (see
  [Section 6](#6-why-generation-is-not-a-cli-command)).

```json
config.requirements {"dag": {"nodes": ["TabularPreprocessing", "XGBoostTraining"],
                             "edges": [["TabularPreprocessing", "XGBoostTraining"]]}}
```

### From Python

```python
from cursus.api.dag import import_dag_from_json
from cursus.api.factory import DAGConfigFactory

dag = import_dag_from_json("dag.json")
factory = DAGConfigFactory(dag)

# Node -> config class mapping
factory.get_config_class_map()          # {"XGBoostTraining": <class 'XGBoostTrainingConfig'>, ...}

# What each scope needs
factory.get_base_config_requirements()             # BasePipelineConfig fields
factory.get_base_processing_config_requirements()  # extra ProcessingStepConfigBase fields (may be [])
factory.get_step_requirements("XGBoostTraining")   # step-specific, non-inherited fields

# Which steps still need input (auto-configurable steps are excluded)
factory.get_pending_steps()
```

---

## 2. How field requirements are structured

Requirements come from `extract_field_requirements` in
`src/cursus/api/factory/field_extractor.py`, which reads a Pydantic V2 model's `model_fields`
directly. Each requirement is a plain dict:

| Key | Meaning |
| --- | --- |
| `name` | Field name |
| `type` | Human-readable type string (e.g. `str`, `list[str]`, `int (optional)`) |
| `description` | The Pydantic `Field(description=...)`, or a generated fallback |
| `required` | `True` if the field has no default (Pydantic `is_required()`) |
| `default` | Default value for optional fields (factory defaults are called, or shown as `<factory: name>`) |
| `allowed_values` | *(present only when the field is constrained)* the closed set of legal values |
| `case_sensitive` | *(present only with `allowed_values`)* whether matching is case-sensitive |

Because these are just dicts, you can print them with `print_field_requirements`, or split them
with `categorize_field_requirements(reqs)` → `{"required": [...], "optional": [...]}`.

### Inherited vs. step-specific fields

The factory deliberately separates the layers so you don't re-enter shared fields per step:

- **Base fields** come from `BasePipelineConfig` — set once via `set_base_config`.
- **Base-processing fields** are the *extra* fields `ProcessingStepConfigBase` adds on top of
  `BasePipelineConfig`. `get_base_processing_config_requirements()` returns only those
  non-inherited fields (via `extract_non_inherited_fields`). If no step in the DAG inherits from
  the processing base, this list is empty and you can skip `set_base_processing_config`.
- **Step fields** returned by `get_step_requirements(step)` exclude everything inherited from the
  relevant base — so you only see what is unique to that step.

This is why a step can be *auto-configurable*: if it has **no** required step-specific fields
(only inherited fields plus optional tier‑2+ fields), the factory can build it from the base
configs alone. Such steps do not appear in `get_pending_steps()`.

---

## 3. Field constraints (`allowed_values`)

Type strings alone don't tell you which *values* are legal. To prevent invalid-enum / wrong-case
mistakes, `extract_field_constraints` attaches a closed-value constraint to a requirement when it
can detect one. It uses two sources, in priority order:

1. **Literal / Enum annotation** (`source: "literal"`) — read straight off the type, e.g.
   `Literal["train", "test"]` or an `Enum` class. This is drift-proof. Case-sensitive.
2. **Validator source** (`source: "validator"`) — when a field is constrained by a
   `@field_validator` that checks against an `allowed = {...}` set. The extractor parses the
   validator's source; if it lowercases/uppercases the value before matching (`.lower()` /
   `.upper()`), `case_sensitive` is reported as `False`.

When present, the requirement dict carries `allowed_values` and `case_sensitive`:

```python
reqs = factory.get_step_requirements("TabularPreprocessing")
for req in reqs:
    if "allowed_values" in req:
        cs = "case-sensitive" if req["case_sensitive"] else "case-insensitive"
        print(f"{req['name']}: one of {req['allowed_values']} ({cs})")
```

If a field has no detectable constraint, no `allowed_values` key is added — treat it as
unconstrained. You can also query a single field directly:

```python
from cursus.api.factory.field_extractor import extract_field_constraints
from cursus.steps.configs.config_tabular_preprocessing_step import TabularPreprocessingConfig

extract_field_constraints(TabularPreprocessingConfig, "output_format")
# -> {"allowed_values": ["CSV", "TSV", "Parquet"], "case_sensitive": False, "source": "validator"}
# (a field with no detectable constraint, e.g. "job_type", returns None)
```

---

## 4. Generate a config set with `DAGConfigFactory`

Generation is a stateful, three-phase workflow: set the base config(s), configure each pending
step, then generate all instances.

```python
from cursus.api.dag import import_dag_from_json
from cursus.api.factory import DAGConfigFactory

dag = import_dag_from_json("dag.json")

# Pass project_root (a folder) or anchor_file=__file__ (a file whose parent is the
# folder) so generated configs resolve their source_dir / processing_source_dir against
# YOUR project. This is the highest-priority path anchor (the "caller hook", Strategy 0
# of hybrid path resolution), so config generation anchors correctly even when no
# compiler has run first. If both are given and disagree, project_root wins. A
# MODSTemplate author who only builds configs (never compiles) should pass
# anchor_file=__file__ (or project_root=Path(__file__).parent).
factory = DAGConfigFactory(dag, project_root="/path/to/my_project")

# --- Phase 1: base configs (set once, inherited by every step) ---
factory.set_base_config(
    author="my-team",
    bucket="my-bucket",
    role="arn:aws:iam::...:role/my-role",
    region="us-east-1",
    # ... whatever get_base_config_requirements() reported as required
)

# Only needed if get_base_processing_config_requirements() is non-empty
factory.set_base_processing_config(
    processing_instance_type_large="ml.m5.4xlarge",
    # ... processing-specific required fields
)

# --- Phase 2: per-step configs ---
for step in factory.get_pending_steps():
    reqs = factory.get_step_requirements(step)
    # inspect reqs, gather values, then:
    factory.set_step_config(step, **step_values)

# --- Phase 3: generate ---
configs = factory.generate_all_configs()   # list[BaseModel]
```

### What `set_step_config` does

`set_step_config(step_name, **kwargs)` (in `src/cursus/api/factory/dag_config_factory.py`):

1. Resolves `step_name` to an actual DAG node (see bare-name handling below).
2. Validates prerequisites — a processing step requires both `base_config` and
   `base_processing_config` to be set first; a base-derived step requires `base_config`. If a
   prerequisite is missing, it raises `ValueError`.
3. Builds the config instance **immediately** using `from_base_config(...)` so base and
   base-processing fields are inherited automatically — you only pass the step-specific values.
4. Validates the instance right away (early feedback) and stores both the raw kwargs and the
   validated instance.

It **returns** the validated instance, so you can assert on it inline.

### Bare step names and `job_type`

DAG nodes are named `{CanonicalStepName}[_{job_type}]` — e.g. a node might be
`CradleDataLoading_training`. `set_step_config` accepts either the full node name **or** the bare
canonical name together with `job_type`. `_resolve_step_name_to_node` resolves it in this order:

1. **Exact node match** — `set_step_config("CradleDataLoading_training", ...)` → used as-is.
2. **Bare name + `job_type`** — `set_step_config("CradleDataLoading", job_type="training", ...)`
   → composes and matches `CradleDataLoading_training`.
3. **Bare name, unique base** — if exactly one node has that canonical base name, it resolves
   unambiguously.

If a bare name matches **several** nodes (e.g. both `_training` and `_calibration` variants),
it's ambiguous — the factory logs a warning and you must pass the full node name.

```python
# All three configure the same node "CradleDataLoading_training":
factory.set_step_config("CradleDataLoading_training", s3_input_override="s3://...")
factory.set_step_config("CradleDataLoading", job_type="training", s3_input_override="s3://...")
```

### Safer step configuration helpers

Two wrappers make notebook code less error-prone than the common
`if "X" in pending_steps: set_step_config("X", ...)` guard, which silently skips typos:

- **`is_dag_step(step_name, job_type=None)`** — returns `True` if the (bare or suffixed) name
  resolves to a real DAG node. Use it for explicit, resolution-aware guards.
- **`configure_step_if_present(step_name, **kwargs)`** — configures the step if it exists,
  otherwise logs a **warning** (not silent) and returns `None`. This surfaces mistyped or
  renamed step names instead of hiding them.

Other useful state methods: `update_step_config(step, **kwargs)` (merge new values into an
already-configured step), `get_step_config_instance(step)`, `get_all_config_instances()`,
`get_configuration_status()`, and `get_factory_summary()`.

---

## 5. `generate_all_configs` and saving

`generate_all_configs()` finalizes the set:

1. **Auto-configures** every eligible step — those with only tier‑2+ (optional) step fields — so
   you don't have to call `set_step_config` on them explicitly.
2. Raises `ValueError("Missing configuration for steps: [...]")` if any step still lacks required
   input.
3. Runs `validate_dag_config_alignment(raise_on_error=True)` — the **DAG↔config invariant**: every
   configured instance must serialize under a key equal to its DAG node name. A mismatch means the
   configured config class is the *wrong step type* for that node, and it fails loudly here rather
   than as an opaque error at compile time.
4. Returns the list of validated config instances.

The returned list is a plain `list[BaseModel]`. To persist it as the merged JSON that the
compiler consumes, call `merge_and_save_configs`:

```python
from cursus.core.config_fields import merge_and_save_configs

configs = factory.generate_all_configs()
merge_and_save_configs(configs, "config/config.json")
```

You can also stash and resume an in-progress session with `save_partial_state(path)` /
`load_partial_state(path)` (these persist the raw kwargs, not full instances).

### Structure of the saved file

`merge_and_save_configs` writes a two-part JSON document:

```json
{
  "metadata": {
    "created_at": "2026-01-01T12:00:00",
    "config_types": { "XGBoostTraining": "XGBoostTrainingConfig", ... },
    "field_sources": { ... }
  },
  "configuration": {
    "shared":   { "author": "...", "bucket": "...", ... },
    "specific": { "XGBoostTraining": { ...step-only fields... }, ... }
  }
}
```

- **`metadata.config_types`** maps each **saved step name** to the **config class name** that
  produced it. It is built by `UnifiedConfigManager.save`, which derives the step name from each
  config object (via `TypeAwareConfigSerializer.generate_step_name`) and records
  `config.__class__.__name__`. On load, this map is what lets the deserializer reconstruct the
  correct Pydantic class for each step — it is the bridge between the flat JSON and typed config
  objects.
- **`configuration.shared`** holds fields common to all steps (entered once via the base configs);
  **`configuration.specific`** holds each step's unique fields. This split mirrors the
  inherited-vs-step-specific separation from [Section 2](#inherited-vs-step-specific-fields).

Inspect a saved file without loading full values via `config.load` (MCP) or:

```python
from cursus.core.config_fields import load_configs
loaded = load_configs("config/config.json")   # -> {"shared": {...}, "specific": {...}}
```

---

## 6. Why generation is not a CLI command

Generation produces **live, in-process Pydantic config objects**, and `merge_and_save_configs`
relies on each object's type metadata and categorization methods to serialize correctly. Those
objects can't cross a stateless JSON tool boundary, and the interactive
`set_base_config` → `set_step_config` (×N) → `generate_all_configs` flow doesn't map to a single
one-shot command. So:

- The **CLI** (`cursus config requirements`) and **MCP** tools (`config.requirements`,
  `config.field_info`, `config.load`) cover only the *inspect* side.
- `config.merge_save` deliberately returns an `unsupported` error pointing you back at the
  in-process API.
- Actual generation happens in **Python** (a notebook or script), as shown above.

---

## Quick reference

| Task | Call |
| --- | --- |
| List a DAG's field requirements (CLI) | `cursus config requirements dag.json` |
| One step's requirements (CLI) | `cursus config requirements dag.json --step NAME` |
| Requirements over MCP | `config.requirements {"dag": {...}}` |
| One config class's fields (MCP) | `config.field_info {"config_class": "...Config"}` |
| Node → config class map | `factory.get_config_class_map()` |
| Steps still needing input | `factory.get_pending_steps()` |
| Field's legal values | `extract_field_constraints(Cls, "field")` → `allowed_values` |
| Set base config | `factory.set_base_config(**fields)` |
| Set processing base | `factory.set_base_processing_config(**fields)` |
| Configure a step | `factory.set_step_config("Step", **fields)` |
| Configure by bare name | `factory.set_step_config("Step", job_type="training", ...)` |
| Configure only if present | `factory.configure_step_if_present("Step", ...)` |
| Finalize all configs | `factory.generate_all_configs()` |
| Save to JSON | `merge_and_save_configs(configs, "config/config.json")` |
| Inspect a saved file | `load_configs(path)` or `config.load {"path": ...}` |

## Related pages

- [DAG & Compilation](../concepts/dag_and_compilation.md)
- [Config system](../concepts/config_system.md)
- [API reference](../api/index.rst)
- [CLI reference](../cli.rst)
- [MCP tools](../reference/generated/mcp_tools.md)
- [Step catalog](../reference/generated/step_catalog.md)

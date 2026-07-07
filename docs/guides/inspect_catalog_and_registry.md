# Inspect Steps, Catalog & Registry

Before you wire up a DAG or author a config, you need to know **what exists**: which
steps are available, what each step consumes and produces, which builder strategy a
step type binds, what its config fields are, and which pre-built pipelines you could
start from. Cursus exposes all of this through read-only discovery commands.

Every discovery surface comes in two forms:

- a **CLI command** (`cursus <group> <command>`) for interactive use at a terminal, and
- an **MCP tool** (`<namespace>.<tool>`) that an agent calls to get the same answer as a
  structured JSON envelope.

Both read the same underlying engine — the unified
[step catalog](../concepts/registry_and_discovery.md) and the strategy registry — so the
CLI and the agent can never disagree about what a step actually does. This guide pairs
each command with its tool equivalent so you can move between the two freely.

## The five discovery surfaces at a glance

| CLI group | MCP namespace | What it answers |
| --- | --- | --- |
| `cursus catalog` | `catalog.*` | What steps / builders / configs / config fields exist? |
| `cursus steps` | `steps.*` | What are a step's I/O container paths and construction patterns? |
| `cursus strategies` | `strategies.*` | Which builder strategy + knobs does a step type bind? |
| `cursus registry` | *(via `catalog.resolve_step`)* | Registry contents, conflicts, name resolution, workspaces |
| `cursus pipeline-catalog` | `pipeline_catalog.*` | Which pre-built shared DAGs match my requirements? |

Most discovery commands accept `--format json` (the default is human-readable `text`;
for the `catalog` list-family, `table` is a deprecated alias for `text`), so any recipe
below can be piped into `jq` or consumed by scripts. The `registry` subcommands are the
exception — they print text only.

---

## `cursus catalog` — steps, builders, configs, fields

The catalog group wraps `cursus.step_catalog.StepCatalog`. Start here when you don't yet
know a step's exact name.

### List and search steps

```bash
# Every step known to cursus
cursus catalog list

# Filter by workspace, job type, or detected framework
cursus catalog list --framework xgboost --job-type training --limit 20

# Fuzzy search by name (scored matches, with component counts)
cursus catalog search xgboost
cursus catalog search preprocess --job-type training --format json
```

`list` calls `catalog.list_available_steps(...)` and can post-filter by
`detect_framework(...)`. `search` calls `catalog.search_steps(query, ...)` and returns
`match_score` / `match_reason` per hit.

**MCP equivalents:**

```text
catalog.list_steps  {"job_type": "training"}
catalog.search      {"query": "xgboost"}
```

`catalog.list_steps` accepts optional `workspace_id` and `job_type`; `catalog.search`
requires `query` and returns scored `matches`, each with `components_available`.

### Inspect one step

```bash
# Registry data + framework + available components + job-type variants
cursus catalog show XGBoostTraining
cursus catalog show XGBoostTraining --format json

# Just the file components (script / contract / spec / builder / config)
cursus catalog components XGBoostTraining
cursus catalog components XGBoostTraining --type builder

# Detailed info about one component, optionally loading the class
cursus catalog component-info XGBoostTraining builder --load
```

`show` renders `catalog.get_step_info(step_name)` — its `registry_data`,
`file_components`, detected framework, and any `_<job_type>` variants.
`component-info` takes a `component_type` from
`{config, builder, contract, spec, script}` and, with `--load`, actually imports the
class to report e.g. a config's field count or a spec's dependency names.

**MCP equivalent:**

```text
catalog.step_info {"step_name": "XGBoostTraining"}
```

Returns the step's `registry_data`, `config_class`, `builder_step_name`,
`sagemaker_step_type`, detected `framework`, and `components_available`.

### List builders, configs, contracts, specs, scripts

```bash
cursus catalog list-builders --step-type Training --show-path
cursus catalog list-configs --show-fields
cursus catalog list-contracts --with-scripts-only --show-entry-points
cursus catalog list-specs --job-type training --show-dependencies
cursus catalog list-scripts --show-path
```

- `list-builders` uses `get_builders_by_step_type(step_type)` (or `get_all_builders()`)
  and can print each builder's registered SageMaker step type and file path.
- `list-configs` uses `discover_config_classes(project_id)`; `--show-fields` counts each
  class's Pydantic `model_fields`.
- `list-contracts`, `list-specs`, and `list-scripts` enumerate the corresponding
  component type across the catalog.

**MCP equivalent (builders):**

```text
catalog.list_builders {"sagemaker_step_type": "Processing"}
```

Returns builder class names grouped by step name — with no instantiation.

### Explore config fields

Two commands answer "what config does this step take?":

```bash
# Every field on a step's config class, with types / defaults / required flags
cursus catalog fields XGBoostTraining --show-types --show-defaults --inherited

# Which config classes contain a given field name (reverse lookup)
cursus catalog search-field label_field --field-type str --show-default
```

`fields` looks up `<StepName>Config` in `discover_config_classes()` and walks its
`model_fields`. `--inherited` also prints the immediate parent config class via
`get_immediate_parent_config_class(...)`. `search-field` scans every discovered config
class for a field of that name.

**MCP equivalent:**

```text
catalog.config_fields {"step_name": "XGBoostTraining"}
```

Returns a `fields` list — each entry has `name`, `type`, `required`, `default`, and
`description` — resolved by mapping the step to its config class name via the registry.

### Other catalog utilities

```bash
cursus catalog frameworks           # frameworks detected across all steps + counts
cursus catalog workspaces           # workspaces and their step / component counts
cursus catalog list-by-type Processing   # steps whose sagemaker_step_type == Processing
cursus catalog metrics              # catalog query / index performance report
cursus catalog discover --workspace-dir ./my_workspace   # scan a directory for steps
```

### `catalog.resolve_step` and `catalog.step_spec` (MCP-only)

Two catalog-namespace tools have no direct `cursus catalog` subcommand:

```text
catalog.resolve_step {"step_name": "model_evaluation_xgb"}
catalog.step_spec    {"step_name": "XGBoostTraining"}
```

- `catalog.resolve_step` maps a canonical **or** file-style name (e.g.
  `model_evaluation_xgb`) to its `config_class`, `builder_class`, `spec_type`, and
  `sagemaker_step_type`, using `cursus.registry.step_names`. On the CLI, the closest
  equivalent is `cursus registry resolve-step` (below).
- `catalog.step_spec` serializes a step's specification to
  `{step_type, node_type, dependencies, outputs}` — the port-level I/O contract you need
  to wire a step into a DAG. It is complemented by `cursus steps io` / `steps.io`, which
  add container paths and channel fan-out.

---

## `cursus steps` — a step's I/O and construction view

Under the Strategy + Facade design, a step's builder is a near-empty shell: the
container source/destination paths, the runtime `property_path` references, and the
training-channel fan-out live in the step's `.step.yaml` interface plus its bound
handler, not in a readable builder class. The `steps` group renders that hidden wiring.

Both subcommands read `cursus.steps.interfaces.io_view`, so the view can never drift
from what the step actually wires.

### `steps io` — inputs, outputs, paths, channels

```bash
cursus steps io XGBoostTraining
cursus steps io RiskTableMapping --job-type validation
cursus steps io XGBoostTraining --format json
```

For each **input** (consumer) it prints the logical name, container path, required flag,
type, `compatible_sources`, and the SageMaker training channels the input fans into
(e.g. `input_path -> train / val / test`). For each **output** (producer) it prints the
container path and the runtime `property_path` a downstream step resolves against.

**MCP equivalent:**

```text
steps.io {"step_name": "XGBoostTraining"}
steps.io {"step_name": "TabularPreprocessing", "job_type": "training"}
```

Same view as JSON, plus convenience `dependency_count` / `output_count`. This is the
path/wiring complement to `catalog.step_spec` (ports + property paths, but no container
paths or channels).

### `steps patterns` — how the step is assembled

```bash
cursus steps patterns XGBoostTraining
cursus steps patterns TabularPreprocessing --job-type calibration --format json
```

Shows the construction "plugins" the `TemplateStepBuilder` composes for the step: the
bound `create_step` handler, plus the env-var, job-argument, input, output, and compute
patterns — all derived from the `.step.yaml` contract data and the registry binding. A
`⚠ custom override` marker flags any axis the builder still hand-overrides. A
`dependencies` rollup reports the step's third-party import footprint (build-time vs
runtime vs native-SageMaker-only).

**MCP equivalent:**

```text
steps.patterns {"step_name": "XGBoostTraining"}
```

Both `steps.*` tools return a `not_found` failure (with `catalog.list_steps` /
`catalog.search` suggested as a remedy) when the step name has no interface.

---

## `cursus strategies` — the builder strategy library

A step builder is no longer a class you open and read — it is a *selection* of
strategies plus knobs, bound at build time by the facade from the step's
`sagemaker_step_type` (and, for Processing, its `step_assembly`). The `strategies` group
makes that selection space discoverable. Every subcommand reads
`cursus.registry.strategy_registry`, the same single source the runtime router uses.

```bash
# The routing axes (sagemaker_step_type, step_assembly) + strategy counts
cursus strategies axes

# Every registered strategy; columns: axis | name | verb | #knobs
cursus strategies list
cursus strategies list --axis sagemaker_step_type

# Full detail for one strategy: verb, handler, knobs, preset knobs
cursus strategies show Training
cursus strategies show code --axis step_assembly

# The authoring shortcut: "what builder do I get if I declare this step type?"
cursus strategies for Training
cursus strategies for Processing --step-assembly code

# Just the declarative knobs a strategy accepts
cursus strategies knobs --axis sagemaker_step_type --name Training
```

`for` is the highest-value command: given a `sagemaker_step_type` (plus `--step-assembly`
for Processing: `code | step_args | delegation`), it resolves the exact strategy the
facade would bind and lists its preset and available knobs — the replacement for reading
a builder class.

**MCP equivalents:**

| CLI | MCP tool |
| --- | --- |
| `cursus strategies axes` | `strategies.list_axes {}` |
| `cursus strategies list [--axis A]` | `strategies.list {"axis": "A"}` |
| `cursus strategies show NAME [--axis A]` | `strategies.show {"name": "NAME"}` |
| `cursus strategies for TYPE [--step-assembly S]` | `strategies.for_step_type {"sagemaker_step_type": "TYPE"}` |
| `cursus strategies knobs --axis A --name N` | `strategies.knobs {"axis": "A", "name": "N"}` |

```text
strategies.for_step_type {"sagemaker_step_type": "Processing", "step_assembly": "code"}
```

When a name is ambiguous across axes, both the CLI and `strategies.show` ask you to pass
the disambiguating axis. When a step type binds no builder (e.g. builder-less types),
`strategies.for_step_type` returns a `not_found` failure pointing you at
`strategies.list`.

---

## `cursus registry` — registry contents, resolution, and workspaces

The `registry` group manages the hybrid registry (core registry plus per-developer
workspace registries). It is broader than pure discovery — it also scaffolds and
validates — but the read-only subcommands are the registry counterpart to
`catalog.resolve_step`.

### Inspect and resolve

```bash
# List steps in the (core or workspace) registry
cursus registry list-steps
cursus registry list-steps --workspace my_project --include-source
cursus registry list-steps --conflicts-only

# Validate a registry and optionally check for step-name conflicts
cursus registry validate-registry --workspace my_project --check-conflicts

# Resolve a step name to its config class / builder / framework
cursus registry resolve-step XGBoostTraining --workspace my_project
```

`list-steps` reads `get_all_step_names(workspace)`; with a hybrid registry available it
groups by source registry and can surface conflicts via `UnifiedRegistryManager`.
`resolve-step` runs the full hybrid resolution (source registry + resolution strategy),
falling back to `get_config_class_name` / `get_builder_step_name` when the hybrid
manager is unavailable.

**MCP equivalent** — there is no dedicated `registry.*` MCP namespace; registry
resolution is exposed through the catalog namespace:

```text
catalog.resolve_step {"step_name": "XGBoostTraining"}
```

### Validate step definitions

```bash
# Validate a proposed step definition against standardization rules
cursus registry validate-step-definition \
  --name MyCustomStep \
  --config-class MyCustomStepConfig \
  --builder-name MyCustomStepBuilder \
  --sagemaker-type Processing \
  --auto-correct

# Validation system status + performance metrics
cursus registry validation-status
cursus registry reset-validation-metrics
```

These wrap `cursus.registry.validation_utils` and are most useful when authoring new
steps — see [Define a step pack](define_a_step_pack.md).

### Scaffold a workspace

```bash
cursus registry init-workspace my_project --template standard
```

Creates a developer workspace (interface-first layout:
`src/cursus_dev/steps/{interfaces,configs,scripts}` plus a local
`src/cursus_dev/registry/workspace_registry.py`) so you can register custom steps that
the catalog then discovers. Templates: `minimal`, `standard`, `advanced`.

---

## `cursus pipeline-catalog` — pre-built shared DAGs

Rather than assemble a DAG from scratch, you can start from a pre-built shared DAG. The
`pipeline-catalog` group ranks and loads these, wrapping
`cursus.pipeline_catalog.core.agent_tool.pipeline_catalog_tool`.

```bash
# Rank pre-built DAGs against your requirements
cursus pipeline-catalog recommend --data-type tabular --framework xgboost
cursus pipeline-catalog recommend --data-type text --needs-llm --framework pytorch

# Frameworks available across the catalog (with DAG counts)
cursus pipeline-catalog list

# Nodes / edges / requirements for one DAG
cursus pipeline-catalog get-dag lightgbm_complete_e2e
```

`recommend` scores DAGs against semantic flags — `--data-type` (`text | tabular |
mixed`), `--has-labels/--no-labels`, `--needs-llm/--no-llm`, `--multi-task/--single-task`,
`--incremental/--first-time`, `--framework`, and `--gpu/--no-gpu`. Note that
`cursus pipeline-catalog list` reports **frameworks with DAG counts**; the per-DAG
listing lives in the MCP `pipeline_catalog.list` tool.

**MCP equivalents:**

| CLI | MCP tool |
| --- | --- |
| `cursus pipeline-catalog recommend ...` | `pipeline_catalog.recommend {...}` |
| `cursus pipeline-catalog get-dag <id>` | `pipeline_catalog.get_dag {"dag_id": "<id>"}` |
| *(framework counts)* `pipeline-catalog list` | `pipeline_catalog.list {}` (per-DAG metadata) |

The MCP namespace exposes three additional tools with no CLI subcommand:

```text
pipeline_catalog.config_guidance {"dag_id": "bedrock_pytorch_incremental_edx"}
pipeline_catalog.auto_select     {"framework": "xgboost", "min_score": 0.8}
pipeline_catalog.load_dag        {"dag_id": "lightgbm_complete_e2e"}
```

- `config_guidance` returns prerequisites, required-vs-default config values, and common
  pitfalls for a chosen DAG.
- `auto_select` returns the single best-matching `dag_id` (or `null` below the score
  threshold) instead of a ranked list.
- `load_dag` returns the concrete `nodes` + `edges` in JSON-safe form, ready to compile.

Once you have picked a DAG, hand it to compilation — see
[Compile and deploy](compile_and_deploy.md) and
[DAG & compilation](../concepts/dag_and_compilation.md).

---

## CLI ↔ MCP quick reference

| Task | CLI | MCP tool |
| --- | --- | --- |
| List steps | `cursus catalog list` | `catalog.list_steps` |
| Search steps | `cursus catalog search Q` | `catalog.search` |
| Step metadata | `cursus catalog show NAME` | `catalog.step_info` |
| Step I/O ports | *(none)* | `catalog.step_spec` |
| Config fields | `cursus catalog fields NAME` | `catalog.config_fields` |
| List builders | `cursus catalog list-builders` | `catalog.list_builders` |
| Resolve a name | `cursus registry resolve-step NAME` | `catalog.resolve_step` |
| Step I/O + paths + channels | `cursus steps io NAME` | `steps.io` |
| Step construction patterns | `cursus steps patterns NAME` | `steps.patterns` |
| Strategy axes | `cursus strategies axes` | `strategies.list_axes` |
| List strategies | `cursus strategies list` | `strategies.list` |
| Show a strategy | `cursus strategies show NAME` | `strategies.show` |
| Strategy for a step type | `cursus strategies for TYPE` | `strategies.for_step_type` |
| Strategy knobs | `cursus strategies knobs ...` | `strategies.knobs` |
| Registry step list | `cursus registry list-steps` | *(use `catalog.list_steps`)* |
| Recommend a pipeline | `cursus pipeline-catalog recommend` | `pipeline_catalog.recommend` |
| Inspect a shared DAG | `cursus pipeline-catalog get-dag ID` | `pipeline_catalog.get_dag` |
| List shared DAGs | *(framework counts only)* | `pipeline_catalog.list` |
| Config guidance for a DAG | *(none)* | `pipeline_catalog.config_guidance` |
| Auto-select a DAG | *(none)* | `pipeline_catalog.auto_select` |
| Load a DAG graph | *(none)* | `pipeline_catalog.load_dag` |

## See also

- [Registry & discovery](../concepts/registry_and_discovery.md) — how the catalog and
  registry are built.
- [Step interfaces](../concepts/step_interfaces.md) — the `.step.yaml` model behind
  `cursus steps`.
- [Define a step pack](define_a_step_pack.md) — author a new step the catalog discovers.
- [CLI reference](../cli.rst) — every `cursus` command.
- [Step catalog reference](../reference/generated/step_catalog.md),
  [MCP tools reference](../reference/generated/mcp_tools.md),
  [Pipeline catalog reference](../reference/generated/pipeline_catalog.md) — generated
  listings.
- [API reference](../api/index.rst).

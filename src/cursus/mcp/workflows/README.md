# cursus MCP workflows

Saved **Claude Code dynamic-workflow** scripts that orchestrate the cursus MCP tools into a
deterministic, resumable, batch-capable sequence. They are shipped with the package (FZ 31e1d3f5a)
so any consumer of `amzn-cursus` — not just this workspace — can drive the flow.

These are JavaScript orchestration scripts for the Claude Code `Workflow` runtime (the
`export const meta = {...}` header + the `agent()` / `parallel()` / `pipeline()` / `phase()`
primitives). They are reference/runnable artifacts, not importable Python.

## `cursus-author-step.js` — DAG-driven new-step creation

A new step is **never authored alone**. The input is a user `PipelineDAG` in which exactly ONE node
is a NEW, unregistered step sitting **between a producer and a consumer**. That placement imposes hard
constraints on the new step's interface, and the workflow enforces them as gates (grounded in the FZ 29
mods-migration SOP): the new step's *dependency-spec* must align with the producer *output-spec*, its
*output-spec* must align with the consumer *dependency-spec*, and the producer/consumer specs must be
**editable** (additively) so they accept the new step. It invokes the `author.*` + `strategies.*` +
`steps.io` + `validate.deps_resolve` + `validate.step_interface` MCP tools at each stage:

```
Resolve   → decompose each adjacent node → base step_type + _jobtype (the full node string IS the config
            key); climb the gap ladder reuse→extend→delete→new (author.checklist + strategies.for_step_type
            + catalog.step_info) → a PLAN that may conclude NO new step is needed
Challenge → verdict-override on the gap rung: one skeptic runs catalog.step_info/steps.io to try to REVERSE
            the rung on the SAME enum (a wrong cheaper reuse/extend/delete would silently short-circuit)
AlignEdges→ the spec-alignment gate (steps.io on producer + consumer): arity (1 PropertyReference per
            DependencySpec); GATE-1 enum (dependency_type/output_type exact-or-in-matrix, else the edge
            HARD-ZEROS); plan additive edits to the CONSUMER compatible_sources + OUTPUT aliases; GATE-2
            compute the 6-component score for BOTH edges (validate.deps_explain) — refuse to proceed <0.5
Guide     → author.rules + strategies.knobs + author.config_constraints + catalog.config_fields + the
            container-path roots per step type → per-axis field guidance honoring the aligned specs
Author    → the agent Writes the 3 artifacts (.step.yaml + <Name>Config + script) using the exemplar as a
            STARTING shape then applying the required divergences-from-exemplar (never a silent clone) AND
            edits the producer/consumer .step.yaml specs to accept the new step
Validate  → validate.step_interface (contract↔spec + container-path validators) + author.check_script
            (both directions incl. --job_type + the {job_type} subdir layout) + an EXECUTABLE parse oracle
            (py_compile the .py artifacts + yaml.safe_load the .step.yaml, as required booleans) + a
            DIVERGENCE-FIDELITY gate (every required delta-over-exemplar the Resolve phase named is actually
            present — catches the copy-the-exemplar-and-drop-the-distinguishing-feature bug) + re-resolve
            BOTH edges with the real resolver (validate.deps_resolve); fix loops report py_compile_ok so a
            fixer that injects a syntax error escalates instead of silently retrying
Preflight → author.preflight_step (offline constructibility = the CI merge gate) + compile.preview/validate
            on the whole DAG (the new node resolves to its config + builder)
Gaps      → completeness critic over the FULL dag_path: does the additive compatible_sources edit regress
            ANOTHER consumer's edge? a cross-branch (train vs eval) column-survival mismatch (habit 4)?
            → a step with open gaps is reported green_with_open_gaps, not silently shipped
Synthesize→ report what was authored, the rung (+ whether Challenge reversed it), the edge scores, the path
```

The edge-alignment mechanic is the resolver's own 6-component compatibility score
(`dependency_resolver._calculate_compatibility`): dependency_type↔output_type **40%** (exact) / 20%
(compatible-in-matrix) / 0%, data_type **20%** / 10%, semantic-name-with-aliases **25%**, exact
logical-name-or-alias bonus **5%**, source-compatibility **10%** (the CONSUMER dependency's
`compatible_sources` lists the job-type-normalized producer base step type) / 5% (empty) / 0% (wrong),
keyword **5%**; an edge resolves at **≥0.5**. A mismatched dependency_type/output_type hard-zeros the
40% and usually the edge, so enums are fixed before names; the lever for "editing room" is the consumer
`compatible_sources` + an output alias — you never mutate an existing step's output_type to force an edge.

The `author.*` value/constraint tools (`author.config_constraints`, `author.preflight_config`) and the
script gate (`author.check_script`) close the empirically-most-common bug classes from the Munged
Address migration (FZ 29d14m): wrong/invalid config enum values, missing required fields, a custom
script missing the `--job_type` argparse the builder passes, and a required env var read-and-ignored.
The workflow embeds a **4-step habit** (read the validator not the docstring; trace
config→builder→env→script→downstream-column; placeholder-is-a-bug; **a resolved edge is a wire proof,
not a runtime proof** — always run the separate {job_type}-subdir + column-survival checks).

The structural guarantee: the gates are non-skippable **pipeline stages**, so a step cannot reach
`Synthesize` green without passing the same checks CI runs as its merge gate — and a request whose gap
ladder resolves to a cheaper rung (config-only / optional-dependency / delete-node), or whose edges
cannot be made to resolve, **short-circuits** rather than authoring a step it should not.

### Run it

Copy the script into a Claude Code workflows location (`.claude/workflows/` in a project, or
`~/.claude/workflows/` for personal use) so it runs as `/cursus-author-step`, or invoke the
`Workflow` tool with this `scriptPath` directly. Each request is a **DAG-edge context** — the new node
plus its adjacent nodes; omit `producer_node` for a source node or `consumer_node` for a sink:

```
# one step (the undefined node between a producer and a consumer)
args: { "intent": "a post-training probability calibration step", "name": "BetaCalibration",
        "producer_node": "XGBoostModelEval_calibration", "consumer_node": "ModelMetricsComputation",
        "dag_path": "projects/atoz_xgboost/atoz_xgboost_na.py" }

# a batch (authored independently, pipelined)
args: [ { "intent": "an XGBoost evaluation step", "name": "XGBoostModelEval2",
          "producer_node": "XGBoostTraining", "consumer_node": "ModelMetricsComputation" },
        { "intent": "a feature-drift check step",  "name": "FeatureDriftCheck",
          "producer_node": "TabularPreprocessing_training", "consumer_node": "XGBoostTraining" } ]
```

With no `args` it authors a single illustrative `BetaCalibration` step between `XGBoostModelEval_calibration`
and `ModelMetricsComputation`.

### Tooling assumption + fallback

Stages call the cursus tools as **MCP tools** (`author.checklist`, `validate.step_interface`, …).
If the cursus MCP server is unreachable in the harness, the gate stages fall back to the `cursus`
CLI (`cursus validate step-interface …`); they **never** fall back to hand-editing the registry
(`STEP_NAMES` is derived by construction — writing the `.step.yaml` IS the registration).
The `author.*` tools are MCP-only today; the workflow embeds their guidance inline as a backstop.

See FZ 31e1d3f5 (the tool surface) and FZ 31e1d3f5a (this orchestrator) in the Cursus Simplification
Trail, and `slipbox/0_developer_guide/adding_new_pipeline_step.md` for the human SOP.

## `cursus-configure-pipeline.js` — agent-tool-driven config.json generation

`cursus-author-step` authors a NEW step TYPE. This workflow does the *other* (and, per the Munged
Address migration, far more common) job: produce a **pipeline `config.json` for a DAG of EXISTING
step types** — the activity where 19 of the 21 empirical config bugs (FZ 29d14m) lived (wrong enum
value/case, missing required fields, wrong type).

It drives an agent to author (or repair) a project's `generate_config.py` — the canonical
`DAGConfigFactory` pattern used across the BAMT projects (`munged_address_pytorch`, `neat_spam_pytorch`,
`mods_pipeline_adapter`, `rnr_pytorch_bedrock`, …): build/import the DAG → `DAGConfigFactory(dag)` →
`set_base_config(...)` → `set_base_processing_config(...)` → `set_step_config(node, **values)` per node →
`get_pending_steps()` empty → `generate_all_configs()` → `merge_and_save_configs(config_path)`.

```
Map       → DAG node -> config class + base/per-step field requirements
Constrain → per node: author.config_constraints (legal enum values + case + declared type + required-no-default)
Author    → the agent writes/repairs generate_config.py + self-checks it with `py_compile` (a required boolean)
Validate  → per node: author.preflight_config (config_class.model_validate the values) + bounded fix loop whose
            fix agent reports py_compile_ok (a fixer that injects a syntax error escalates, not silent-retries)
Generate  → run generate_config.py: get_pending_steps gate -> generate_all_configs -> merge_and_save_configs,
            THEN an executed `json.load(config_path)` parse check (config_json_parses) — not a self-report
DagCheck  → CROSS-NODE gate: node/edge integrity (every add_edge endpoint was add_node'd) + compile.validate
            (no missing config) + compile.preview (no low-confidence/ambiguous = no wrong-config-class collision)
            + validate.deps_resolve (edges wire)
Synthesize→ report the config.json + per-step validity + the cross-node verdict + the run path. `shippable` is
            true only if generated + parses + cross-node RAN and passed + zero value errors — a skipped/dead
            DagCheck (null) is NOT treated as a pass (distinct "did not run" branch)
```

`set_step_config` itself validates each config (Pydantic) at generation time, so the factory's
`get_pending_steps()` + `generate_all_configs()` is the final whole-config gate; `author.preflight_config`
is the per-step EARLY gate that catches a wrong value before the run. Those gates are per-NODE; the
**DagCheck** phase is the per-EDGE / whole-graph gate the migration learnings (FZ 29d14m) flagged as the
single largest miss — most SAIS-run failures are cross-node, and they pass every per-node check. It catches
the failure modes per-node validation cannot: a node renamed without regenerating its config (silently
fuzzy-resolves to the WRONG config class → duplicate-step `ValueError`), a missing config, an unresolved
edge, and — because `PipelineDAG.add_edge` silently auto-creates any endpoint not already `add_node`'d — a
typo'd edge name that spawns a phantom unconfigured node and orphans the real one (construction never
raises; the serializer's dangling-edge check is fooled because the phantom is already in `nodes`). Invoke
with `args`:

```
args: { "project": "munged_address_pytorch", "region": "NA",
        "project_root": "projects/munged_address_pytorch",   # or src/buyer_abuse_mods_template/<project> in BAMT
        "dag_nodes": ["CradleDataLoading_tagging", "TabularPreprocessing_sampling", "PyTorchTraining", ...] }
```

### Project folder layout

A pipeline project is a self-contained folder with the **same relative layout** wherever it lives —
`AmazonCursus/projects/<project>/` (dev) or `BuyerAbuseModsTemplate/src/buyer_abuse_mods_template/<project>/`
(deployed). Canonical worked examples: `munged_address_pytorch`, `neat_spam_pytorch`.

```
<project>/
├── <project>_<region>.py          # the DAG def (lowercase region, e.g. *_na.py): create_<project>_dag()
│                                   #   -> ONE worldwide (WW) DAG; topology is region-agnostic
├── generate_config.py             # authors pipeline_config/config_<REGION>.json via DAGConfigFactory (per-region)
├── generate_exe_doc.py            # authors pipeline_config/exe_doc_<REGION>.json (MODS execution document)
├── run_pipeline.py                # the SAIS run (see below)
├── pipeline_config/               # SINGULAR (plural pipeline_configs is back-compat only)
│   ├── config_<REGION>.json        # per-region step configs (regional SQL / marketplace IDs / ARNs)
│   ├── dag.json                    # optional serialized WW DAG (split only for genuine topology variants)
│   └── exe_doc_<REGION>.json
└── dockers/
    ├── scripts/                    # application-agnostic Processing scripts (processing_source_dir)
    ├── hyperparams/hyperparameters_<REGION>.json   # per-region training hyperparameters
    ├── <framework>_training.py     # model-dependent scripts in the dockers ROOT (source_dir): the per-project
    ├── <framework>_model_inference.py  #   model implementations the framework does NOT ship
    ├── <framework>_model_eval.py
    ├── <framework>_inference_handler.py
    └── requirements-secure.txt / requirements-gpu-secure.txt   # SAIS secure-PyPI pins (load-bearing)
```

The DAG topology is **one worldwide (WW)** DAG; only `config_<REGION>.json` + `hyperparameters_<REGION>.json`
+ the exe-doc are per-region.

### Running it in SAIS

After the config is generated, the pipeline is compiled + executed by `run_pipeline.py`:
1. `import sais_environment.sais_env_setup` (the load-bearing SAIS bootstrap — sets `PYTHONNOUSERSITE`).
2. `run_pipeline.py --preview` — offline DAG-resolution + `validate_dag_compatibility` (no session).
3. `run_pipeline.py --region <R>` — `setup_session()` (SaisSession / SecurityConfig /
   `create_secure_session_config` / `PipelineSession`) → `PipelineDAGCompiler.compile_with_report(dag)` →
   `ExecutionDocumentGenerator.fill_execution_document(dag, ...)` →
   `SagemakerPipelineHelper.start_pipeline_execution(...)`.

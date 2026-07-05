# Validate a Pipeline

Before you push a pipeline to SageMaker, Cursus gives you four independent checks that
catch different classes of mistake — from a config that doesn't match a DAG node, to a
`.step.yaml` that won't load, to a dependency edge that scores just under the resolution
threshold. This guide walks through each command, explains **what it catches**, and shows
**how to read the results**.

All four commands are read-only and fast — none of them deploys anything. Run them in this
order and you fix problems from the outside in:

| Check | Command | Catches |
| --- | --- | --- |
| DAG ↔ config alignment | `cursus compile --validate-only` | Nodes with no config; unresolvable builders |
| Interface validity | `cursus validate step-interface [--all]` | Broken `.step.yaml` (Pydantic errors, contract↔spec misalignment, `compatible_sources` typos) |
| 4-level alignment | `cursus alignment validate` / `validate-all` | Script↔contract↔spec↔dependencies↔builder drift |
| DAG structure + edge scoring | `cursus dag validate` / `cursus dag resolve` | Cycles, phantom nodes, unknown steps; per-edge resolver scores |

`compile`, `validate step-interface`, and `dag validate`/`dag resolve` all support
`--format text` (default, human-readable) and `--format json` (machine-readable, for CI
gates). The `alignment` group is the exception: it prints human-readable output and emits
machine-readable JSON reports via `--output-dir/-o` instead of a `--format` flag. Most of
these commands exit nonzero when they find a blocking problem, so you can wire them straight
into a pre-merge check. The exception is `cursus dag resolve`, which is a scoring/reporting
command: it exits `0` even when an edge fails to resolve (it only exits nonzero on a load or
parse error), so gate on its `all_edges_resolve` JSON field rather than on the exit code.

See also: [DAG and compilation](../concepts/dag_and_compilation.md),
[Dependency resolution](../concepts/dependency_resolution.md),
[Step interfaces](../concepts/step_interfaces.md), and the full [CLI reference](../cli.rst).

---

## 1. DAG ↔ config alignment — `cursus compile --validate-only`

This is the fastest way to confirm that a serialized DAG and a configuration file fit
together **before** you actually compile a SageMaker pipeline. It runs the real compiler's
compatibility check and stops short of building any steps.

```bash
cursus compile -d dag.json -c config.json --validate-only
```

Under the hood this calls `PipelineDAGCompiler.validate_dag_compatibility(dag)`
(`src/cursus/core/compiler/dag_compiler.py`), which returns a `ValidationResult`
(`src/cursus/core/compiler/validation.py`) with these fields:

- `is_valid` — overall pass/fail.
- `missing_configs` — DAG nodes that have **no matching configuration** in the config file.
- `unresolvable_builders` — configs whose **step builder could not be resolved** from the
  registry / step catalog.
- `warnings` — non-blocking notes.
- (`config_errors` and `dependency_issues` also exist on the model.)

### How to read it

A passing run:

```text
✓ DAG loaded: 5 nodes, 4 edges
✓ Config loaded: 5 step configurations

Validation Results:
✓ All DAG nodes have matching configurations
✓ All step builders resolved successfully
✓ No dependency issues found

Validation passed! Ready for compilation.
```

A failing run prints exactly which nodes and builders are the problem, and exits `1`:

```text
❌ Validation failed!

Missing configurations:
  - XGBoostTraining

Unresolvable builders:
  - CustomStep
```

- **Missing configuration** almost always means the node name in the DAG doesn't match the
  key/type of any config object — check for a typo or a config you forgot to add.
- **Unresolvable builder** means a config exists but Cursus can't find a step builder for it
  — usually a step that isn't registered in the [Step catalog](../reference/generated/step_catalog.md).

For CI, use JSON and gate on `is_valid`:

```bash
cursus compile -d dag.json -c config.json --validate-only --format json
```

```json
{
  "status": "validation_complete",
  "is_valid": false,
  "dag_nodes": 5,
  "dag_edges": 4,
  "missing_configs": ["XGBoostTraining"],
  "unresolvable_builders": [],
  "warnings": []
}
```

> `--validate-only` skips compilation entirely. Drop the flag (and add `--show-report`) once
> validation is green to see the resolution details. See
> [Compile and deploy](compile_and_deploy.md).

---

## 2. Interface validation — `cursus validate step-interface`

The compiler's check above assumes every step's `.step.yaml` interface actually loads. This
command validates that assumption at **author time**, loading each interface through the same
production path the build uses (`StepInterface.from_yaml`, via
`load_interface` in `src/cursus/steps/interfaces`).

Validate one step:

```bash
cursus validate step-interface XGBoostTraining
```

Resolve a job-type variant:

```bash
cursus validate step-interface RiskTableMapping --job-type validation
```

Validate **every** interface in the package (the CI form):

```bash
cursus validate step-interface --all
```

### What it catches

The command separates blocking **errors** from non-blocking **warnings**
(`src/cursus/cli/validate_cli.py`):

- **Errors (blocking, exit `1`):**
  - `FileNotFound` — no `.step.yaml` for that step / job-type.
  - Pydantic `ValidationError` — malformed fields in the YAML.
  - Cross-section misalignment — the contract↔spec consistency check raised inside
    `from_yaml`.
- **Warnings (non-blocking):**
  - `compatible_sources` **case typos** — an entry that case-insensitively matches a real
    step name but differs in case. It still "works" but silently loses the dependency
    resolver's source-compatibility bonus, weakening the edge. The warning names the
    correctly-cased step it probably meant.

### How to read it

```text
✅ XGBoostTraining
⚠️  RiskTableMapping
     warn:  dep 'risk_tables': compatible_sources 'cradledataloading' looks like a case typo of 'CradleDataLoading' (would silently lose the resolver bonus)
❌ BrokenStep
     ERROR: ValidationError: 1 validation error for StepInterface ...

validated 3 · 1 error(s) · 1 warning(s)
```

- `✅` — clean.
- `⚠️` — loads fine, but has a quality warning worth fixing.
- `❌` — blocking error; this interface will not load in a build.

If you pass neither a step name nor `--all`, the command exits `2` and reminds you to supply
one. JSON output rolls the counts up for gating:

```bash
cursus validate step-interface --all --format json
```

```json
{
  "validated": 42,
  "errors": 0,
  "warnings": 1,
  "results": [ ... ]
}
```

> Fix interface errors first — the alignment tests and the resolver checks below both load
> these same interfaces, so a broken `.step.yaml` will cascade into failures there too.
> For how interfaces are authored, see [Define a step pack](define_a_step_pack.md).

---

## 3. Four-level alignment tests — `cursus alignment`

Where `step-interface` validates one YAML file in isolation, `cursus alignment` validates the
**whole chain of artifacts** behind a step — script, contract, specification, dependencies,
and builder — and reports where they drift apart. It is driven by
`UnifiedAlignmentTester` (`src/cursus/validation/alignment/unified_alignment_tester.py`).

The four levels are:

| Level | Alignment | Typical failure |
| --- | --- | --- |
| 1 | Script ↔ Contract | Script reads/writes a path the contract doesn't declare |
| 2 | Contract ↔ Specification | Contract I/O doesn't match the spec's logical names |
| 3 | Specification ↔ Dependencies | A declared dependency has no compatible producer |
| 4 | Builder ↔ Configuration | Builder expects config fields the config doesn't define |

### Commands

Validate one step:

```bash
cursus alignment validate xgboost_training --verbose --show-scoring
```

Validate every step the catalog discovers (steps without a script file get intelligent
level-skipping):

```bash
cursus alignment validate-all --output-dir ./reports --verbose
```

Validate a single level (1–4) — handy when you're iterating on just the script/contract pair:

```bash
cursus alignment validate-level xgboost_training 1
```

List what can be validated:

```bash
cursus alignment list-scripts
```

Key options: `--workspace-dirs` (include workspace steps, repeatable), `--output-dir/-o`
(write per-step JSON reports plus a `validation_summary.json`), `--verbose/-v`,
`--show-scoring`, and `--continue-on-error` (for `validate-all`, keep going past a failing
step instead of stopping).

### How to read it

Each step gets an `overall_status` of `PASSED`, `FAILED`, or `EXCLUDED`, followed by a
per-level breakdown:

```text
✅ xgboost_training: PASSED
  ✅ Level 1 (Script ↔ Contract): PASS
  ✅ Level 2 (Contract ↔ Specification): PASS
  ✅ Level 3 (Specification ↔ Dependencies): PASS
  ✅ Level 4 (Builder ↔ Configuration): PASS

✅ xgboost_training passed all alignment validation checks!
```

- `PASSED` and `EXCLUDED` both exit `0` — **`EXCLUDED`** means the step was deliberately
  skipped (e.g. no script), not that it failed.
- `FAILED` exits `1`.

With `--show-scoring` you also get a 0–100 quality score and rating per step and (with
`-v`) per level:

```text
📊 Overall Score: 96.5/100 (Excellent)
📈 Level Scores:
  • Level 1 (Script ↔ Contract): 100.0/100
  • Level 2 (Contract ↔ Specification): 92.0/100
  ...
```

`validate-all` ends with a roll-up:

```text
🎯 FINAL VALIDATION SUMMARY
📊 Total Steps: 42
✅ Passed: 41 (97.6%)
❌ Failed: 1 (2.4%)
⚠️  Errors: 0 (0.0%)
```

When something fails, run `validate-level <step> <n> --verbose` to focus on the offending
level and read the `[ERROR]` message plus its `💡` recommendation. The concepts page
[Dependency resolution](../concepts/dependency_resolution.md) explains Level 3 in depth.

> Use `-o ./reports` in CI: each step's JSON report and the `validation_summary.json` are
> durable artifacts you can attach to a build.

---

## 4. DAG structure and edge scoring — `cursus dag`

The `dag` group works directly on a serialized DAG JSON file and on step interfaces, with no
config required. Use it to catch structural mistakes and to see the **actual resolver scores**
on dependency edges.

### 4a. Structural integrity — `cursus dag validate`

```bash
cursus dag validate dag.json
```

This runs `PipelineDAGResolver.validate_dag_integrity()`
(`src/cursus/api/dag/pipeline_dag_resolver.py`), which reports issues grouped by category:

- `cycles` — a dependency cycle (the DAG isn't acyclic).
- `dangling_dependencies` — an edge references a source/destination node that doesn't exist.
- `undeclared_edge_nodes` — an edge endpoint that was never declared via `add_node`; a
  classic symptom of an edge-name typo that spawned a phantom, unconfigured node.
- `isolated_nodes` — a node with no connections at all.
- `missing_steps` — a node that doesn't resolve to any known step in the
  [Step catalog](../reference/generated/step_catalog.md) (with a sample of available names).
- Plus component-availability and workspace-compatibility checks when the catalog is present.

Passing:

```text
DAG: dag.json
  nodes: 5 | edges: 4
✅ DAG is valid (no integrity issues found).
```

Failing (exits `1`):

```text
DAG: dag.json
  nodes: 5 | edges: 4
❌ DAG has integrity issues:
  cycles:
    - Cycle detected: A -> B -> A
  missing_steps:
    - Step 'XGBTrain' not found in catalog. Available steps: [...]...
```

`--format json` returns `is_valid`, `node_count`, `edge_count`, and the full `issues` map.

### 4b. Dependency edge scoring — `cursus dag resolve`

`dag validate` tells you the graph is well-formed; `dag resolve` tells you whether the edges
will **actually resolve** through the dependency resolver — and by how much. Give it two or
more step names (typically a producer, a step you're adding, and a consumer):

```bash
cursus dag resolve CradleDataLoading TSATabularPreprocessing
```

Each step's interface is loaded and registered, then the **real**
`UnifiedDependencyResolver` (`create_dependency_resolver` +
`resolve_with_scoring`, `src/cursus/core/deps/dependency_resolver.py`) scores every
dependency of every named step against the other named steps as candidate providers. No score
is computed by hand — this is the same resolver, weights, and threshold the compiler and CI
use:

- Weights: `type` 0.40, `data_type` 0.20, `semantic` 0.25, `exact-match` 0.05,
  `source-compat` 0.10, `keyword` 0.05.
- Threshold: an edge **resolves** when its best candidate scores **≥ 0.5**.

### How to read it

```text
Resolve: CradleDataLoading, TSATabularPreprocessing
  ✅ TSATabularPreprocessing.DATA <- CradleDataLoading (score 1.0, resolves=True)
```

- `✅ ... resolves=True` — the best provider cleared the 0.5 threshold. (Resolved edges always
  print `score 1.0` as a "cleared-threshold" placeholder rather than the raw compatibility
  score.)
- `❌ ... resolves=False` — the best candidate fell short; the printed `score` is that best
  score, so you can see how close you were. A score of `0.0` usually means no candidate
  produced a compatible output at all.
- `⚠ could not load <name>` — that step name is unknown or its YAML failed to load (fix it
  with `cursus validate step-interface <name>` from section 2).
- `(no dependency edges among these steps)` — none of the named steps declares a dependency
  that the others could satisfy; add the missing producer/consumer to the list.

JSON output is designed for an author-time gate:

```bash
cursus dag resolve CradleDataLoading TSATabularPreprocessing --format json
```

```json
{
  "steps": ["CradleDataLoading", "TSATabularPreprocessing"],
  "loaded": ["CradleDataLoading", "TSATabularPreprocessing"],
  "load_errors": {},
  "edges": [
    {
      "consumer": "TSATabularPreprocessing",
      "dependency": "DATA",
      "provider": "CradleDataLoading",
      "score": 1.0,
      "resolves": true
    }
  ],
  "all_edges_resolve": true,
  "threshold": 0.5
}
```

Gate on `all_edges_resolve` (and inspect any `edge` with `resolves: false` and its `score`).
A near-miss score (e.g. 0.45) is the signal that a `compatible_sources` case typo — the exact
warning from section 2 — cost you the 0.10 source-compat bonus.

> `dag resolve` scores among the steps you name; it is not the full-DAG resolution the
> compiler does. Use it to check a **new or changed** edge in isolation; use
> `compile --validate-only` (section 1) and `alignment validate-all` (section 3) for the
> whole pipeline.

---

## A practical validation loop

When adding or changing a step, run the checks from smallest scope to largest:

```bash
# 1. Does the interface load at all?
cursus validate step-interface MyNewStep

# 2. Does my new edge resolve against its neighbours?
cursus dag resolve UpstreamStep MyNewStep DownstreamStep

# 3. Are the script/contract/spec/builder in agreement?
cursus alignment validate my_new_step --verbose --show-scoring

# 4. Is the assembled DAG structurally sound?
cursus dag validate dag.json

# 5. Does the DAG line up with the config file?
cursus compile -d dag.json -c config.json --validate-only
```

Each step narrows down where a problem lives, so by the time you reach step 5 the only
failures left are genuine DAG↔config mismatches. For CI, gate on exit codes and machine
output:

- **Step 1** (interface): run the CI form `cursus validate step-interface --all --format json`.
- **Step 2** (edge scoring): run `cursus dag resolve ... --format json` and gate on the
  `all_edges_resolve` field — this command exits `0` even on unresolved edges.
- **Step 3** (alignment): run `cursus alignment validate-all -o ./reports`; alignment writes
  per-step JSON plus a `validation_summary.json` to the output directory rather than taking a
  `--format json` flag.
- **Steps 4 and 5** (`dag validate`, `compile --validate-only`): run with `--format json`.

Fail the build on any nonzero exit (and, for step 2, on `all_edges_resolve == false`).

## Related

- [Compile and deploy](compile_and_deploy.md) — the next step once validation is green.
- [Generate configs](generate_configs.md) — producing the config file section 1 checks.
- [Define a step pack](define_a_step_pack.md) — authoring the interfaces sections 2–4 load.
- [Dependency resolution](../concepts/dependency_resolution.md) — the scoring model behind
  `dag resolve` and Level 3.
- [CLI reference](../cli.rst) and [MCP tools](../reference/generated/mcp_tools.md).

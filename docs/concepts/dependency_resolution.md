# Dependency Resolution

Cursus does not ask you to hand-wire step inputs to step outputs. Instead, each step
declares — in its `.step.yaml` interface — the *dependencies* it consumes and the
*outputs* it produces, as typed, named, semantically-annotated ports. At compile time a
single component, the **`UnifiedDependencyResolver`**, reads those declarations and
*infers* the wiring: for every dependency of every step it scores each candidate output
across the pipeline and binds the best-scoring one, provided the score clears a fixed
threshold.

This page explains how that inference works: the resolver, the six-component
compatibility score, the `SemanticMatcher`, the role of `compatible_sources`, the `0.5`
threshold, and how one DAG edge (producer output → consumer dependency) is scored and
turned into a runtime property reference. It also covers `cursus dag resolve` — the CLI
that runs the *real* resolver so you can inspect a single edge's score — and the
silent-misresolution hardening that keeps a wrong-but-plausible match from binding
quietly.

See also: [Compilation](dag_and_compilation.md), the [CLI reference](../cli.rst), the
[Step catalog](../reference/generated/step_catalog.md), and the
[MCP tools](../reference/generated/mcp_tools.md).

## What gets resolved

A step interface declares two relevant sections. On the **consumer** side, each
dependency carries a `type`, a `required` flag, a `compatible_sources` list, and
`semantic_keywords`. On the **producer** side, each output carries a `type`, a
`property_path` (the SageMaker runtime property to read), `aliases`, and
`semantic_keywords`. Here is an (abbreviated) fragment from
`src/cursus/steps/interfaces/xgboost_training.step.yaml` — the shipped file lists several
more `compatible_sources` and `semantic_keywords` entries:

```yaml
  dependencies:
    input_path:
      type: training_data
      required: true
      compatible_sources:
      - TabularPreprocessing
      - StratifiedSampling
      - RiskTableMapping
      - ProcessingStep
      semantic_keywords:
      - data
      - input
      - training
      - dataset
  outputs:
    model_output:
      type: model_artifacts
      property_path: properties.ModelArtifacts.S3ModelArtifacts
      aliases:
      - ModelArtifacts
      - model_data
      data_type: S3Uri
```

These decls surface on the loaded interface as `DependencyDecl` and `OutputDecl`
objects (in `src/cursus/core/base/step_interface.py`). Each exposes the attributes the
resolver reads: a dependency has `logical_name`, `dependency_type` (an alias of `type`,
returning a `DependencyType` enum), `data_type`, `required`, `compatible_sources`, and
`semantic_keywords`; an output has `logical_name`, `output_type`, `data_type`,
`aliases`, `property_path`, and `semantic_keywords`. `logical_name` is auto-populated
from the YAML dict key.

`DependencyType` is a closed enum (`src/cursus/core/base/enums.py`):
`MODEL_ARTIFACTS`, `PROCESSING_OUTPUT`, `TRAINING_DATA`, `HYPERPARAMETERS`,
`PAYLOAD_SAMPLES`, `CUSTOM_PROPERTY`.

## The UnifiedDependencyResolver

The resolver lives in `src/cursus/core/deps/dependency_resolver.py`. It is constructed
from two collaborators:

```python
from cursus.core.deps.dependency_resolver import create_dependency_resolver
from cursus.core.deps.specification_registry import SpecificationRegistry

registry = SpecificationRegistry()
registry.register("XGBoostTraining", xgboost_spec)
registry.register("TabularPreprocessing", preproc_spec)

resolver = create_dependency_resolver(registry)   # attaches a SemanticMatcher
```

- **`SpecificationRegistry`** holds the registered specs keyed by step name and answers
  `get_specification(step_name)`.
- **`SemanticMatcher`** computes name-similarity scores (see below).

The resolver caches per-step results in `_resolution_cache`; registering a new spec via
`register_specification` clears the cache.

### Resolution entry points

The resolver exposes several methods, all of which reduce to the same per-edge scoring:

| Method | Returns | Used by |
| --- | --- | --- |
| `resolve_step_dependencies(consumer, available_steps)` | `{dep_name: PropertyReference}` for one step; raises `DependencyResolutionError` if a **required** dep is unresolved | Reports, ad-hoc resolution |
| `resolve_all_dependencies(available_steps)` | `{step: {dep_name: PropertyReference}}` for every registered step | Batch resolution |
| `resolve_with_scoring(consumer, available_steps)` | `{"resolved": …, "failed_with_scores": …, "resolution_details": …}` with per-candidate score breakdowns | Validation, `cursus dag resolve` |
| `get_resolution_report(available_steps)` | A debug report with per-step resolved/unresolved details and an overall `resolution_rate` | Diagnostics |

A `PropertyReference` (`src/cursus/core/deps/property_reference.py`) pairs the producer
`step_name` with the matched `output_spec`; its `to_runtime_property()` later becomes an
actual SageMaker `Properties` object at build time.

## The six-component compatibility score

For a single `(dependency, output)` pair the resolver computes one number in `[0, 1]`
via `_calculate_compatibility(dep_spec, output_spec, provider_spec)`. Six weighted
components contribute. A parallel `_get_score_breakdown` returns each component
separately for the failed-edge report; note that its source-compatibility component
reports the reward-only form (`0.1` if the provider is in `compatible_sources`, else
`0.0`): it still normalizes the provider's job-type suffix
(`_normalize_step_type_for_compatibility`), but applies neither the `−0.1` penalty nor
the `_provider_in_compatible_sources` alias/legacy-`Step` reconciliation that
`_calculate_compatibility` uses.

| # | Component | Max weight | How it is computed |
| --- | --- | --- | --- |
| 1 | **Dependency-type compatibility** | 0.40 | `0.4` if `dependency_type == output_type`; `0.2` if compatible via the type matrix; **returns `0.0` overall** if incompatible |
| 2 | **Data-type compatibility** | 0.20 | `0.2` if `data_type` equal; `0.1` if compatible (e.g. `S3Uri`↔`String`, `Integer`↔`Float`) |
| 3 | **Semantic name matching** | 0.25 | `SemanticMatcher.calculate_similarity_with_aliases(dep.logical_name, output_spec) * 0.25` |
| 4 | **Exact-name / alias bonus** | 0.05 | `+0.05` if the dependency's `logical_name` equals the output's `logical_name` or appears in its `aliases` |
| 5 | **Compatible-source check** | ±0.10 | `+0.1` if the (normalized) producer step type is in `compatible_sources`; **`−0.1` penalty** if `compatible_sources` is non-empty and the producer is *not* listed; `+0.05` if the dependency declares no sources at all |
| 6 | **Keyword matching** | 0.05 | fraction of `semantic_keywords` found as substrings of the output's `logical_name`, times `0.05` |

The final score is `min(sum, 1.0)`.

### Type compatibility is a hard gate

Component 1 is decisive: `_are_types_compatible` consults a fixed matrix and if the
dependency type and output type do not overlap, `_calculate_compatibility` returns `0.0`
immediately — no amount of name similarity can rescue an incompatible type. The matrix
allows a few cross-type pairs, e.g. `TRAINING_DATA` accepts `PROCESSING_OUTPUT` (so a
preprocessing step can feed a trainer), and `HYPERPARAMETERS` accepts `CUSTOM_PROPERTY`.

### Data-type compatibility

`_are_data_types_compatible` allows near-equivalents: `S3Uri` may be used as `String`
and vice versa, `Integer` as `Float` and vice versa, `Boolean` only as `Boolean`. An
unknown data type only matches itself.

## The SemanticMatcher

`SemanticMatcher` (`src/cursus/core/deps/semantic_matcher.py`) supplies component 3 — a
name-similarity score in `[0, 1]`. `calculate_similarity(name1, name2)` first normalizes
both names (lowercase, split on `_`/`-`/`.`, drop special characters, expand
abbreviations such as `config`→`configuration` and `pkg`→`package`, strip stop words),
returns `1.0` on an exact normalized match, and otherwise blends four sub-metrics:

| Sub-metric | Weight | Idea |
| --- | --- | --- |
| String similarity | 0.30 | `difflib.SequenceMatcher` ratio |
| Token overlap | 0.25 | Jaccard over word tokens |
| Semantic similarity | 0.25 | synonym-aware token matching (e.g. `model`≈`artifact`, `data`≈`dataset`) |
| Substring matching | 0.20 | one name contained in the other, or shared word substrings |

`calculate_similarity_with_aliases(name, output_spec)` runs this against the output's
`logical_name` *and* each alias, keeping the highest — which is why declaring an alias
like `model_data` on an output can lift a dependency's semantic score above the
threshold even when the logical names differ. `explain_similarity` returns the full
sub-metric breakdown for debugging.

## `compatible_sources` and the ±0.10 nudge

`compatible_sources` is the consumer's declared allowlist of producer *step types*. It
is deliberately **not** a hard gate. As the resolver's own comments record, roughly 40%
of the tokens declared across the shipped interfaces are generic categories or legacy
aliases (`ProcessingStep`, `TrainingStep`, `PayloadStep`, `S3Source`, `UserProvided`, …)
that are not real registry step types. Zeroing a score whenever the exact string is
absent would wrongly reject legitimate edges — for instance a provider whose canonical
type is `Payload` against a declared `PayloadStep`.

So component 5 *rewards* an in-list provider (`+0.1`) and only *mildly penalizes* an
out-of-list one (`−0.1`), rather than vetoing it. This disadvantages a
semantically-wrong producer without breaking correct-but-legacy-named matches, and it is
only decisive near the `0.5` threshold — the base `type (0.4) + data (0.2) + semantic
(0.25)` still dominates.

Two helpers make the membership test forgiving:

- **`_normalize_step_type_for_compatibility`** strips job-type suffixes
  (`TabularPreprocessing_Training` → `TabularPreprocessing`) using the registry's
  canonical-name functions, so a variant provider still matches its base entry in
  `compatible_sources`. It falls back to stripping `_Training` / `_Testing` /
  `_Validation` / `_Calibration` if the registry lookup fails.
- **`_provider_in_compatible_sources`** reconciles legacy spellings: exact match,
  `StepCatalog.LEGACY_ALIASES` (e.g. `MIMSPayload`→`Payload`, `MIMSPackaging`→`Package`),
  and a trailing-`Step` equivalence, so `Payload` ≡ `PayloadStep`.

## The 0.5 threshold

The resolver treats `0.5` as the pass mark. In `_resolve_single_dependency` a candidate
is only collected if `confidence > 0.5`; in `resolve_with_scoring` the best candidate is
bound only if `best_match["score"] >= 0.5`, otherwise it is recorded in
`failed_with_scores` with the top three candidates and their breakdowns. The threshold
is surfaced in `resolution_details["resolution_threshold"]` and echoed by the CLI.

When multiple candidates clear the bar, the resolver **sorts by score and takes the
highest**; alternatives are logged at debug level. A dependency that resolves to nothing
is fatal only if it is `required` — an unresolved optional dependency is logged and
skipped.

## How one DAG edge is scored and bound

At compile time the resolver runs inside the `PipelineAssembler`
(`src/cursus/core/assembler/pipeline_assembler.py`). Its `_propagate_messages` method
walks the DAG's declared edges and, for each `(src_step, dst_step)` edge, asks the
resolver to score every consumer dependency against every producer output:

```python
for dep_name, dep_spec in dst_builder.spec.dependencies.items():
    matches = []
    for out_name, out_spec in src_builder.spec.outputs.items():
        compatibility = resolver._calculate_compatibility(
            dep_spec, out_spec, src_builder.spec
        )
        if compatibility > 0.5:            # same threshold as the resolver
            matches.append((out_name, out_spec, compatibility))
    if matches:
        matches.sort(key=lambda x: x[2], reverse=True)
        best = matches[0]
        self.step_messages[dst_step][dep_name] = {
            "source_step": src_step,
            "source_output": best[0],
            "match_type": "specification_match",
            "compatibility": best[2],
        }
```

The winning `(source_step, source_output)` is stored in `step_messages`. Later, when the
assembler instantiates each step, it converts that message into a real runtime property:
it looks up the producer output spec via `get_output_by_name_or_alias`, builds a
`PropertyReference(step_name=src_step, output_spec=output_spec)`, and calls
`to_runtime_property(step_instances)` to produce the SageMaker `Properties` handle that
becomes the consumer's input channel.

Crucially, the assembler then **validates that every required dependency got a match**
and refuses to fabricate a placeholder if the runtime property cannot be built — see
"Silent-misresolution hardening" below.

## `cursus dag resolve`

`cursus dag resolve` (in `src/cursus/cli/dag_cli.py`) exposes exactly this scoring so you
can inspect an edge without compiling a pipeline. You name two or more steps; the command
loads each one's `.step.yaml` interface via `load_interface`, registers the specs, and
runs the **real** `UnifiedDependencyResolver.resolve_with_scoring` over them — the same
weights and threshold the compiler and CI use, with no re-implementation.

```bash
# Score every dependency edge among these steps
cursus dag resolve CradleDataLoading TabularPreprocessing XGBoostTraining

# Machine-readable, for gating an author-time check
cursus dag resolve TabularPreprocessing XGBoostTraining --format json
```

For each dependency of each named step, it reports the best-scoring provider among the
other named steps, the score, and whether the edge resolves (`>= 0.5`). Text output marks
each edge with a check or cross:

```
Resolve: CradleDataLoading, TabularPreprocessing, XGBoostTraining
  ✅ XGBoostTraining.input_path <- TabularPreprocessing (score 1.0, resolves=True)
  ❌ XGBoostTraining.hyperparameters_s3_uri <- None (score 0.0, resolves=False)
```

The JSON form emits `steps`, `loaded`, `load_errors`, an `edges` array, `all_edges_resolve`,
and `threshold: 0.5`. Each edge object carries `consumer`, `dependency`, `provider`, `score`,
and `resolves`; a *failed* edge additionally carries `required`. For a resolved edge the CLI
reports `score: 1.0` as a "cleared the threshold" marker; a failed edge reports its best
candidate's actual score (the per-component `score_breakdown` is computed internally by the
resolver but is not emitted by the CLI). A separate command, `cursus dag validate`, checks
structural integrity — cycles, dangling edges, isolated nodes, undeclared edge endpoints —
rather than scoring edges.

## The same resolver at compile and in validation

There is exactly one resolver implementation, and three surfaces run it:

1. **Compile** — `PipelineAssembler._propagate_messages` scores each DAG edge and binds
   the winning output, as shown above.
2. **Validation** — the alignment layer's `dependency_validator.py` calls
   `resolve_with_scoring(canonical_spec_name, available_steps)` to report resolved
   dependencies and to raise a `CRITICAL` issue for any *required* dependency with no
   candidate at or above threshold.
3. **CLI / authoring** — `cursus dag resolve` (and the MCP author-step workflow) call the
   same `resolve_with_scoring` so an author can confirm, before compiling, that a new
   step's edges will bind. Because it is the production resolver, a green result here is
   trustworthy.

This single-source-of-truth design means the score you see in `cursus dag resolve` is the
score the compiler will compute.

## Silent-misresolution hardening

A resolver that "guesses" is dangerous precisely when it guesses *wrong but plausibly* —
binding an edge to the nearest-looking producer and compiling a pipeline that reads the
wrong data at runtime. A series of hardening changes closed these silent-failure paths.
The relevant ones for resolution:

- **Job-type-aware config resolution.** `StepCatalog.get_step_info` and the config
  resolver strip job-type suffixes through the registry-driven `naming.resolve_base_step_name`
  / `split_job_type_suffix` (guarded by the actual step registry so a real base like
  `XGBoostModel` is never mis-stripped), fixing the catalog-tier miss where a suffixed
  node (`TabularPreprocessing_training`) with `job_type=None` failed to resolve.

- **`job_type=None` no longer crashes.** Four sites in
  `step_catalog/adapters/config_resolver.py` were changed to
  `(getattr(config, "job_type", "") or "").lower()`, so a config whose `job_type` is
  present-but-`None` no longer raises `AttributeError` that masked a real DAG↔config
  mismatch as an opaque "unresolvable node".

- **Bare-name resolution.** A suffix-less DAG node now resolves to the single config
  keyed with a suffix — node `PercentileModelCalibration` → config
  `PercentileModelCalibration_calibration` — but only when that base match is
  *unambiguous*: the node must itself be a base step name (checked with
  `split_job_type_suffix`) and exactly one config whose real step type equals that base
  may claim it. When several configs share the base and the node carries no `job_type`,
  the resolver does **not** guess — it defers to the scored matching strategies rather
  than binding an arbitrary one.

- **`compatible_sources` alias/suffix-awareness** — the `Payload` ≡ `PayloadStep`
  trailing-`Step` equivalence and the `LEGACY_ALIASES` reconciliation (e.g.
  `MIMSPackaging` ≡ `Package`) described above, plus the `−0.1` out-of-list penalty, so a
  correct-but-legacy-named producer is not silently demoted below threshold and a wrong
  producer is nudged down.

- **Interface variant loading uses a recursive deep-merge.** When a `job_type` names a
  declared variant, `StepInterface.from_yaml` deep-merges that variant's `spec`/`contract`
  overrides over the base sections *recursively*, so a variant that restates only a subset
  of ports (e.g. one that tightens a single optional dependency to `required`) overrides
  just those ports' fields and keeps the base ports it omits — a shallow
  `{**base, **variant}` merge previously dropped every base port a variant happened not to
  restate. When a `job_type` is requested and the step declares variants but none of them
  matches, `from_yaml` logs a **warning and falls back to the base spec** (it deliberately
  does **not** raise:
  step configs validate `job_type` only as open lowercase-alphanumeric, so legitimate
  values like `munged`/`tagging` are expected not to be enumerated variants). The hazard of
  an under-tightened variant is not masked — it is caught downstream, where the dependency
  resolver flags an unwired *required* dependency and `_sync_and_align` re-checks
  contract↔spec alignment, so a base-fallback can only under-tighten an *optional* port,
  never hide a missing required one.

- **Validation flags misresolutions as errors, not warnings.** In
  `core/compiler/validation.py`, `validate_dag_compatibility` runs a node-vs-config
  cross-check: when a node name encodes a step type (matching a known registry base) but
  the bound config resolves to a *different* step type, it records a `config_errors` entry
  (→ `is_valid=False`) telling the author to add an explicit config key or a
  `metadata.config_types` mapping. It also loads each node's interface with its resolved
  `job_type` — the same `load_interface(step_name, job_type=…)` call the builder makes at
  compile — so an interface-load failure surfaces in `cursus validate`, not only at build
  time. Routability is checked via `has_builder_provider`, so SDK-delegation steps
  (`CradleDataLoading`, `Registration`) whose builders don't import offline are not
  false-flagged as unresolvable.

- **No fabricated placeholders at build.** When a matched edge's runtime property cannot
  be built, or the producer spec has no output named/aliased as expected, the assembler
  now raises a `ValueError` instead of inventing an `s3://pipeline-reference/...`
  placeholder that would make an absent input *appear* present and wire a channel at a
  nonexistent bucket.

The through-line: an edge either resolves to a real, type-compatible, above-threshold
producer output, or the failure is reported loudly — at resolve, validate, or compile —
rather than compiled into a wrong pipeline.

## Related pages

- [DAG and compilation](dag_and_compilation.md) — how the DAG is built and compiled into a
  SageMaker pipeline.
- [Step catalog](../reference/generated/step_catalog.md) — how interfaces, specs, and
  builders are discovered.
- [CLI reference](../cli.rst) — full command surface, including `cursus dag`.
- [MCP tools](../reference/generated/mcp_tools.md) — the authoring/validation tools that
  wrap the resolver.

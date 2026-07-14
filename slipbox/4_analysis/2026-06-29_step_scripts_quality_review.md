---
tags:
  - analysis
  - step_scripts
  - critical_review
  - code_quality
  - script_contract_alignment
  - robustness
keywords:
  - main() testability contract
  - silent except swallowing
  - env-var under-declaration
  - false required env var
  - unguarded KeyError input_paths
  - inline pip-install bloat
  - duplicated processor classes
  - shared script library
  - secure-pypi hardcoded ARN
  - hybrid path resolution
topics:
  - step script code quality
  - script vs .step.yaml alignment
  - robustness / conciseness / generality
language: python
date of note: 2026-06-29
---
# Step Scripts Quality Review — 40 scripts on 4 axes (robust / concise / aligned / general)

> Review requested 2026-06-29: go over ALL scripts under `src/cursus/steps/scripts/`, and for each — judged against its stated objective and its `.step.yaml` interface — assess whether the code can be made (1) more robust, (2) more concise, (3) better aligned with the corresponding step interface, (4) more general/standard. This is a *code-quality* review of the scripts themselves; the narrower *env-var/job-arg declaration* alignment (axis 3) was cataloged separately in [2026-06-27_bamt_script_vs_stepyaml_env_arg_alignment.md](2026-06-27_bamt_script_vs_stepyaml_env_arg_alignment.md) and is referenced, not re-derived, here.
>
> Method: a 40-way dynamic-workflow fan-out (one reviewer agent per script, reading the real source + the matching `.step.yaml`), each scoring the four axes 1–5 and emitting concrete file:line findings; every HIGH-severity finding was then adversarially re-verified against source by a second agent. 37 scripts returned structured output on the first pass; 3 oversized scripts (`bedrock_prompt_template_generation`, `lightgbm_model_inference`, `lightgbmmt_model_eval`) were re-reviewed individually. Corpus: **313 findings (97 high / 163 med / 53 low)** across the 37 structured reviews, plus the 3 reruns.

## Headline: the scripts WORK but systematically over-trust their environment, and the big ones are bloated

The fleet runs in production today, so these are not outage-class bugs in aggregate — but the same handful of weaknesses recur across nearly every script, which makes them a *systemic* quality issue rather than 40 unrelated nits. Two axes are weak fleet-wide; two are healthier:

| Axis | Mean (of 5) | Scripts scoring ≤2 | Read |
|---|---|---|---|
| **interface-aligned** | 2.15 | 32 / 40 | Weakest. The interface systematically under-declares what the script reads, and several declare `required` vars the script silently defaults. |
| **concise** | 2.20 | 33 / 40 | The large eval/training/bedrock scripts carry heavy duplication + inline install boilerplate. |
| **robust** | 2.48 | 20 / 40 | Over-broad `except` and unguarded path/env access are pervasive; degrade-silently is the norm. |
| **general-standard** | 2.80 | 9 / 40 | Healthiest — 39/40 scripts DO follow the `main()` + try-wrapped `__main__` + `environ_vars`-dict convention. The deviations are concentrated. |

The single most striking fact: **the fleet is highly standardized in *shape* (axis 4) yet highly duplicated in *substance* (axis 2)** — every script independently re-implements the same preprocessing classes, the same pip-install preamble, and the same env-parsing idiom. The convention exists; the shared *library* behind it does not.

Lowest-scoring scripts (sum of 4 axes, max 20) — the priority queue: **`redshift_data_loading` (6)**, **`temporal_sequence_normalization` (6)**, `bspline_calibration` (8), `feature_selection` (8), `model_wiki_generator` (8), `pytorch_model_inference` (8), then a cluster at 9 (`bedrock_batch_processing`, `dummy_training`, `label_ruleset_generation`, `lightgbm_model_eval`, `lightgbm_model_inference`, `lightgbmmt_model_inference`).

## The recurring patterns (ranked by blast radius) — fix these once, fleet-wide

These are the cross-cutting findings: the same defect in many scripts. Fixing each as a *single coordinated change* (or a shared util) beats 40 local patches.

### 1. Silent `except` / swallowed errors — ~30 scripts (robustness)
Bare `except:` / `except Exception: pass|continue` and `os.environ.get(NAME, default)` with no "was it set?" signal are everywhere. Confirmed instances: `dummy_data_loading.detect_file_format()` has three bare `except:` (lines 256/263/270) that hide permission errors, corrupted files, and OOM behind "format not detected"; the bedrock/eval/training scripts catch-and-log per-file then continue, so a whole split can produce zero rows with only DEBUG noise. **This is the same "surface previously-swallowed exceptions" theme as the 1.8.x/1.9.0 discovery-layer sweep — it never reached the scripts.** Fix: a shared `read_required_env(name)` that raises with a clear message, narrow `except` to the specific exception, and re-raise under a `strict` mode.

### 2. Env-var under-declaration in `.step.yaml` — ~23 scripts (interface)
The interface declares **fewer** env vars than the script reads, verified per script: `active_sample_selection` reads 13, declares 5 (62% undeclared); `bedrock_processing` reads ~24, declares a handful; `dummy_data_loading` reads 6 undeclared (`MAX_WORKERS`/`BATCH_SIZE`/`OPTIMIZE_MEMORY`/`STREAMING_BATCH_SIZE`/`ENABLE_TRUE_STREAMING`/`METADATA_FORMAT`). It works only because of in-code defaults — so it is **contract drift, not breakage** (consistent with the 2026-06-27 note), but it hides the tunable surface from callers and makes the interface non-authoritative. Fix: back-fill `contract.env_vars.optional` with the actual names + in-code defaults; this is exactly what the env-vars single-source conformance gate ([../2_project_planning/2026-06-27_env_vars_config_single_source.md](../2_project_planning/2026-06-27_env_vars_config_single_source.md)) would enforce.

### 3. Misleading `required` — env vars/args declared `required` but silently defaulted — ~23 scripts (interface + robustness)
The inverse drift, and sharper because it is *dishonest*: the interface marks a var `required`, but the script reads it with a default, so a builder that enforced `required` would demand a value the script ignores. Verified: `currency_conversion` declares `CURRENCY_CONVERSION_VARS`/`CURRENCY_CONVERSION_DICT` required but both use `json.loads(environ_vars.get(...), default)`; `risk_table_mapping` / `pytorch_model_inference` declare `LABEL_FIELD` required while inference reads no label (already flagged 2026-06-27 — still present in the package scripts). Fix: re-classify to optional, OR add real fail-fast enforcement — pick one and make the flag honest.

### 4. Unguarded `KeyError` on `input_paths` / `output_paths` — ~19 scripts (robustness)
`main()` takes `input_paths`/`output_paths` as `Dict[str,str]` and indexes them directly (`output_paths["prompt_templates"]`, `input_paths["model_artifacts"]`) with no key guard, so a builder that omits a key crashes with a bare `KeyError` instead of a clear "missing channel X" message. Several scripts guard *inputs* but not *outputs* (e.g. `bedrock_prompt_template_generation` validates input keys at line 1287 but not the three output keys at 1421–1427). Fix: a shared `require_paths(d, [...], kind)` preflight at the top of `main()` that names the missing logical key.

### 5. Inline pip-install / secure-PyPI preamble — ~14 scripts (conciseness + generality)
Training/eval/bedrock scripts open with 80–105 LOC of runtime `pip install`, `USE_SECURE_PYPI` branching, and **hard-coded ARNs/accounts/region** (`arn:aws:iam::111111111111:role/SecurePyPIReadRole_`, domain-owner `222222222222`, `us-west-2`). This contradicts the container contract (deps should be pre-installed via `framework_requirements`), bloats the file, and is copy-pasted verbatim across siblings. Notably `xgboost_model_inference` omits it entirely — proof it is removable. Fix: delete the preamble; rely on `contract.framework_requirements` + the container image. If runtime install is genuinely needed, it belongs in ONE shared `ensure_packages()` helper with the ARN/region as params/env, not 14 copies.

### 6. Duplicated processor/helper classes across siblings — the eval/inference/training families (conciseness)
`RiskTableMappingProcessor` and `NumericalVariableImputationProcessor` (~200–250 LOC each) are embedded **verbatim** in `xgboost_model_inference`, `lightgbm_model_inference`, `lightgbmmt_model_inference`, `lightgbm_model_eval`, `lightgbmmt_model_eval`, and others; the comparison-metric/plot functions in the `*_model_eval` family are near-identical ports. Estimated ~1,200+ LOC of pure duplication across the model-eval/inference scripts. Fix: extract a shared `steps/scripts/lib/` (processors + comparison-metrics + plotting), import it. This is the highest-LOC-leverage change in the whole review.

### 7. `OUTPUT_FORMAT == "csv"` sentinel logic (robustness) + `plots_output` channel ghost (interface)
A subtler shared bug: `final_format = output_format if output_format != "csv" else input_format` (in the inference family + `active_sample_selection`) means a caller who *explicitly* sets `OUTPUT_FORMAT=csv` has it ignored — `"csv"` doubles as both the default and the "unset" sentinel. Fix: default to `None`, treat `None`→preserve-input. Separately, `lightgbmmt_model_eval` (and `model_wiki_generator`) declare a `plots_output` channel the script never writes to (it saves plots into the metrics dir) — a contract ghost; route the writes or drop the declaration.

## Notable per-script highs (adversarially verified)

- **`bspline_calibration` — orphan script, NO `.step.yaml`.** It has its own `main()` (a monotone B-spline calibrator, port of COSA `generic_rfuge.r`) but no interface file and no step that declares it as an `entry_point`. Either it is dead code, or a calibration step invokes it indirectly and the interface is missing. Highest-priority interface gap. (Also: `df[sf].values.astype(float)` at 616–617 with no NaN/non-numeric guard.)
- **`currency_conversion`** — divides by `exchange_rate_series` (line 214) with no non-zero guard; a `conversion_rate: 0` in the mapping yields `inf`/`NaN` silently.
- **`bedrock_processing`** — `config['primary_model_id'] = environ_vars.get('BEDROCK_PRIMARY_MODEL_ID')` (line 1988, no None check) → `None` → cryptic `NoneType` failure deep in `__init__` (line 471) instead of a clear "missing required env" at entry.
- **`lightgbm_model_inference`** — `output_df_json[col].astype(float)` (line 678) discards its result (missing assignment) → float columns stay uncast for JSON serialization. A real correctness bug.
- **`dummy_training`** — `compressed_size / total_size` (line 149) ZeroDivisionError when the source dir is empty; and the "input mode" log always prints `INTERNAL` (the ternary tests an always-truthy dict).
- **`dummy_data_loading`** — imports `numpy`/`boto3`/`botocore` that are never used (and absent from `framework_requirements`) — dead deps.

## Recommendations (ordered by leverage)

1. **Build a shared `steps/scripts/lib/` and migrate the duplicated processors + comparison/plot helpers into it** (theme 6). Biggest LOC win (~1,200+), zero behavior change, and it makes the eval/inference family consistent.
2. **Add a shared `script_io` preflight + env helper** — `require_paths(paths, keys, kind)` and `read_required_env(name)` — and adopt them in `main()`/`__main__` across the fleet (themes 1, 3, 4). Turns silent degradation into clear, early errors.
3. **Delete the inline pip-install preamble** from the ~14 scripts that carry it; move any genuine need into ONE `ensure_packages()` with parameterized ARN/region (theme 5). Removes the hard-coded account IDs from source.
4. **Back-fill `contract.env_vars.optional`** with the undeclared vars + defaults across the ~23 under-declaring scripts, and **re-classify the false-`required` vars** (theme 2, 3). Do this *with* the env-vars single-source conformance gate so it can't re-drift.
5. **Fix the `OUTPUT_FORMAT` csv-sentinel** in the inference family and resolve the `plots_output` channel ghosts (theme 7).
6. **Resolve `bspline_calibration`** — give it a `.step.yaml` (and a registry entry) if it is a real step, or delete it if it is dead.
7. **Sweep the confirmed correctness bugs** (`currency_conversion` zero-rate, `lightgbm_model_inference` discarded `astype`, `dummy_training` ZeroDivision, `bedrock_processing` None model-id) — small, high-certainty fixes.

## Scope & relation to other work

This review is the **script-code** complement to two existing notes: the 2026-06-27 alignment note (the *declaration* triangle `script reads = interface declares = config emits`) and the 2026-06-26 package brittleness review (the *framework* code). The script fleet shows the same brittleness signature the framework already had its sweep for — **silent error-swallowing and non-authoritative contracts** — which strongly argues for applying the same remedy (a shared library + a conformance gate) one layer down, in the scripts. None of the findings is an active production outage; all are latent fragility, hidden tunable surface, and maintenance drag that compound as steps are added.

## Provenance

40-way review workflow (run `wf_09139734-b66`, 77 agents, ~3.1M tokens) with per-finding adversarial verification of every HIGH; 3 oversized scripts re-reviewed individually. Per-script scores + findings corpus: 313 structured findings across the 37 first-pass reviews. Axis means and the recurring-theme tallies in this note are computed from that corpus; per-script file:line evidence is in the run output.

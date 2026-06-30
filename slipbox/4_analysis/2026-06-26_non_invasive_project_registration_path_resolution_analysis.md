---
tags:
  - analysis
  - cursus_core
  - path_resolution
  - deployment_portability
  - api_design
  - non_invasive
keywords:
  - CURSUS_PROJECT_BASE
  - project_root_folder
  - hybrid path resolution
  - config-anchored resolution
  - caller introspection
  - project registration
  - load_configs input_file
  - Strategy 0
  - non-invasive anchor
topics:
  - path resolution
  - deployment portability
  - consumer API ergonomics
language: python
date of note: 2026-06-26
status: active
---

# Non-Invasive Project Registration for Cursus Path Resolution

## Executive Summary

Cursus' hybrid path resolver needs **one anchor** to find a consumer project's
`dockers/`/`scripts/` directory across deployment scenarios (Lambda/MODS bundled,
pip-installed, dev monorepo). Today that anchor is supplied by **two consumer-side
obligations**, both of which are invasive:

1. **`CURSUS_PROJECT_BASE`** — an environment variable the consumer must emit
   (currently `buyer_abuse_mods_template/__init__.py` does `os.environ.setdefault(...)`).
   This forces the consumer to add cursus-specific code to *their* package init.
2. **`project_root_folder`** — a field every config must carry to name the consumer's
   project directory, so cursus can locate it as a sibling of the package.

Both make the consumer "mount their project to us." The question this note answers:
**how can cursus stop requiring consumers to emit `CURSUS_PROJECT_BASE` and to set
`project_root_folder`, while keeping the anchor inside cursus — i.e. make the project
*register itself implicitly*, with the consumer aware of (but not burdened by) the
registration?**

**The answer is a layered anchor cascade whose top layer is the config file's own
directory** — which cursus already receives in `load_configs(input_file, ...)` but
currently ignores for path resolution. The config file IS the registration artifact;
its location is the most reliable, zero-config project anchor available, and it works in
every deployment scenario including Lambda. `CURSUS_PROJECT_BASE` and `project_root_folder`
become optional overrides, not obligations.

This resolves the open dialectic in the project trail between the consumer self-declaration
solution (FZ 18g3h2a) and the industry-patterns counter (consumer should NOT self-declare;
use caller introspection — FZ 18g3h2b), both in the abuse_slipbox cursus project trail.

## Background: What the Anchor Is For

Hybrid resolution (`src/cursus/core/utils/hybrid_path_resolution.py`) resolves a config's
`source_dir`/`processing_source_dir` (the docker scripts directory) — **not** the cursus
package. See [Processing Step Path Resolution: Cache Poisoning Fix](2025-11-24_processing_step_path_resolution_cache_poisoning_fix.md)
and the design in `slipbox/1_design/hybrid_strategy_deployment_path_resolution_design.md`. The
package location is only a *discovery anchor*; the resolution **target** is the consumer's
project tree.

The resolver runs 5 strategies in order (`config_base.py:192-207` →
`resolve_hybrid_path` → the cascade):

| # | Strategy | Anchor it uses | Consumer burden |
|---|----------|----------------|-----------------|
| 0 | Explicit project base | `CURSUS_PROJECT_BASE` env var | **Consumer must emit it** |
| 1 | Package location | `Path(__file__)` of cursus, up 4 | none — but only works when cursus is a *sibling* of the project (bundled/vendored) |
| 2 | Working directory | `Path.cwd()` upward ≤10 | none — but fails in Lambda (cwd `/var/task` cut off from `/tmp/...`) |
| 3 | Generic search | cwd + cursus root, up5/down3 | needs `project_root_folder` to know what to search for |
| 4 | Default scripts | `cursus/steps/scripts/` | n/a — finds cursus' OWN scripts, never the project's |

The invasiveness comes from the fact that the only strategy that works **everywhere**
(including Lambda) is Strategy 0, and Strategy 0 today depends on a consumer-emitted env
var. `project_root_folder` is the second obligation, load-bearing for Strategies 0/1/3
because the project sits as a *named sibling* of cursus in the bundled layout.

## The Core Insight: The Config File Is the Registration

The whole problem is "how does cursus find the project root without the consumer telling
it?" But cursus is *already told* — by the call it cannot run without:

```python
# src/cursus/steps/configs/utils.py:212
def load_configs(input_file: str, config_classes=..., project_id=None): ...
```

Every compile begins by loading a config JSON from a path. That config file lives **inside
the project**, next to `dockers/`:

```
/tmp/buyer_abuse_mods_template/munged_address_pytorch/
├── pipeline_config/config_NA.json   ← input_file (cursus already has this path)
├── dockers/                         ← the thing we're trying to resolve
└── munged_address_pytorch_na.py
```

`Path(input_file).parent` (or its parent, walking up to the directory that contains a
`dockers/`/`scripts/`/`pipeline_config/` marker) **is the project root** — derived from an
argument cursus already receives, requiring **zero** consumer code, **zero** env var, and
**zero** `project_root_folder` field. It works in Lambda (the config path is absolute and
real on the bundled filesystem), in SAIS, and on a laptop.

This is exactly the *anchor* the resolver needs, and it is strictly more reliable than
either consumer-supplied mechanism because it cannot drift: the config and the scripts ship
together by construction.

Cursus already has the marker-detection primitive for this — `core/utils/project_discovery.py`
recognizes a project by `_CONFIG_DIR_NAMES = ("pipeline_config", "pipeline_configs")` and
`_PROJECT_MARKERS = ("dockers", "scripts")`. The same `_find_config_dir`/marker logic that
powers `discover_pipeline_projects` can anchor resolution.

## Proposed Design: A Layered Anchor Cascade with Implicit Registration

Reframe "registration" so the consumer is **aware** of how cursus finds their project, but
is not **obligated** to emit anything. The anchor is resolved from a cascade, most-implicit
first; each lower layer is an optional, increasingly-explicit override:

```
Anchor resolution (new Strategy 0 cascade), first hit wins:
  0a. Config-anchored      — Path(input_file) walked up to the nearest project marker
                             (pipeline_config|pipeline_configs + dockers|scripts).
                             ZERO consumer burden. Works in Lambda. ← the new default.
  0b. Caller-anchored      — the project module's __file__, captured when the consumer
                             calls compile()/generate (Flask pattern). Optional.
  0c. Explicit base        — CURSUS_PROJECT_BASE env var. Optional override / escape hatch.
  0d. Explicit field       — project_root_folder in the config. Optional override.
then existing Strategies 1-4 (package location, cwd, generic, default scripts) as fallbacks.
```

### Why config-anchored (0a) is the right default

- **Non-invasive**: derived from `load_configs(input_file)` — an argument the consumer
  already passes. No `__init__.py` edit, no env var, no per-config field.
- **Deployment-universal**: the config path is absolute and co-located with the scripts in
  *every* scenario, so it sidesteps the Lambda cwd-disconnect that kills Strategy 2 and the
  package-location break that kills Strategy 1 after a pip-install migration.
- **Self-consistent**: config + scripts are one shipping unit; the anchor cannot point at a
  stale or wrong project.
- **`project_root_folder` becomes derivable**: once the project root is found by walking up
  from the config file, the field is no longer needed to *name* the sibling — it degrades to
  an optional disambiguation hint for the rare multi-project-one-config-dir case.

### What "registration" means in this model

The consumer's project is registered by **convention, not declaration**: a directory is a
cursus project iff it contains a config dir + a scripts marker. This is the pytest/Kedro
"marker file" pattern (`pyproject.toml [tool.kedro]`, `pytest.ini`) — adapted so the marker
is the config the consumer *already* authors. If we want the registration to be *explicit
and visible* (the "user aware of their registering" the question asks for), add an optional
declarative marker the consumer can drop at their project root:

```toml
# cursus.toml (optional, at project root) — makes registration explicit & greppable
[tool.cursus]
project_root = "."           # or a subdir
scripts_dir  = "dockers"     # default marker
```

`cursus.toml` is opt-in: present → authoritative anchor (Kedro-style); absent → the
config-anchored convention (0a) just works. Either way the consumer is *aware* of the
contract (it's documented and discoverable) without being *forced* to wire env vars into
their package init.

## Resolution of the Trail Dialectic

| Position | Mechanism | Verdict under this design |
|----------|-----------|---------------------------|
| FZ 18g3h2a (self-declare) | `CURSUS_PROJECT_BASE` from consumer `__init__.py` | demoted to **0c**: optional escape hatch, no longer the primary path |
| FZ 18g3h2b (don't self-declare; caller `__file__`) | Flask-style introspection | kept as **0b**: useful when no config path is in play (pure-DAG entry points) |
| **This note (FZ 18g3h2c)** | **config-anchored (0a)** + optional `cursus.toml` | **new default**: anchor from the already-passed `input_file`; consumer-aware via convention/marker, not obligation |

18g3h2b's key truth still holds — *a pip-installed framework never magically finds user
files; it needs an anchor*. This design supplies that anchor from the **one argument the
consumer cannot avoid passing** (the config path), which is less invasive than env var
(0c), caller `__file__` (0b), or the `project_root_folder` field, while remaining
explicit-overridable for the cases that need it.

## Migration / Compatibility

- **Backward compatible**: 0c (`CURSUS_PROJECT_BASE`) and 0d (`project_root_folder`) remain
  as higher-priority *explicit* overrides, so existing BuyerAbuseModsTemplate configs and
  the `__init__.py` setdefault keep working unchanged. The change is purely additive: insert
  0a/0b *above* the legacy strategies and existing behavior is preserved where the new
  layers don't fire.
- **Threading the config path**: 0a requires `load_configs`/the compiler to make
  `input_file` available to `effective_source_dir` resolution (e.g. stash the config dir on
  the config objects at load time, or pass it through the resolver call). This is the only
  non-trivial plumbing.
- **Unblocks the vendored→installed migration** (see the migration analysis): the pytorch
  entry points that import top-level `from cursus...` and never run the template `__init__`
  (so `CURSUS_PROJECT_BASE` is unset) resolve correctly under 0a as long as they pass a
  config path — removing the single biggest risk of dropping the vendored copy.

## Definition-Time vs Runtime — and What This Means in MODS Lambda

The single most important constraint for *any* anchor design: **hybrid resolution runs at
DEFINITION (compile) time, on the host that compiles the pipeline — never inside the
SageMaker container at runtime.** Getting this wrong is the difference between a working and
a broken design, so it must be stated explicitly.

**Two execution modes** (confirmed in `MODSWorkflowHelper.SagemakerPipelineHelper`, which
branches on *"In an SAIS execution … In a MODS execution …"*; see abuse_slipbox
`cursus_mods_pipeline_lifecycle.md` and `cursus_mods_vs_sais_deployment_architecture.md`):

- **SAIS / notebook mode** — a human or notebook **compiles and starts in the same account**
  (`pipeline.upsert()` / `start()` run locally). Compile host = the notebook's project dir.
- **MODS mode** — `MODSPythonLambda` (MODS account) does **both** the compile
  (`generate_pipeline()`) **and** the kick-off (`upsert` / `start_pipeline_execution`), then
  assumes a **per-execution cross-account role** (`sagemakerPipelineExecutionRoleArn`) so the
  SageMaker pipeline resource is created and runs in the **user's onboarded AWS account** —
  not a single fixed "SAIS account." Compile host = the Lambda's `/tmp`.

In **both** modes the two phases are the same shape; only the compile host and the execution
account differ:

| | Definition time (compile) | Runtime (execution) |
|---|---|---|
| **Compile host** | MODS mode: Lambda `/tmp/buyer_abuse_mods_template/...` · SAIS mode: notebook project dir | n/a (already compiled) |
| **What resolves the path** | `effective_source_dir` → `HybridPathResolver` | nothing — SageMaker SDK already handled it |
| **Path namespace** | a real local dir on the compile host, e.g. `…/<project>/dockers` | `/opt/ml/code/` (container convention) |
| **How scripts get there** | SDK uploads `source_dir` to S3 at compile | S3 → container `/opt/ml/code` at job start |
| **Execution account** | n/a | MODS mode: user's onboarded account (via the role ARN) · SAIS mode: same account as compile |

Trace it in code:

1. `builder_xgboost_training_step.py:127` does `source_dir = self.config.effective_source_dir`
   and passes it as `XGBoost(source_dir=source_dir, ...)`. The SageMaker SDK **uploads that
   local directory to S3 at compile time** and bakes the S3 URI into the pipeline definition.
2. Therefore the resolver only ever needs to produce a **valid local path on the compile
   host**. The runtime container path is `/opt/ml/code/...` (the SDK's convention,
   validated by `training_script_contract.py:56` against `/opt/ml/...` prefixes) — a
   *completely different namespace* that hybrid resolution neither produces nor touches.
3. `_effective_source_dir` is a `PrivateAttr` (`config_base.py:132`), lazily computed
   (`:192-207`) and **not serialized** into the config JSON. So the resolved path is **never
   persisted** and is **re-resolved on the compile host every run** — the cached string in
   `config.json` is not authoritative (this is the cache-poisoning fix). This is exactly why
   the anchor must be valid *at definition time, on the compile host*, and need not survive
   into the container.

**Why config-anchored (0a) is the strongest possible anchor specifically for MODS Lambda:**

- In MODS, definition time runs inside the Lambda whose cwd is `/var/task/` — **disconnected**
  from the code at `/tmp/buyer_abuse_mods_template/`. That kills Strategy 2 (cwd traversal)
  and Strategy 3 (generic search from cwd): they cannot reach `/tmp/...`. The design note is
  explicit — *"From cwd(), we CANNOT trace back to dockers folder."* So in Lambda the anchor
  **must** be `/tmp`-absolute, supplied out-of-band.
- The MODS Lambda invokes `template.generate_pipeline()` with a config that lives at
  `/tmp/buyer_abuse_mods_template/<project>/pipeline_config/config.json`. That `input_file` is
  an **absolute `/tmp` path on the exact compile host** — i.e. config-anchored resolution (0a)
  gets a perfect anchor for free, on the one filesystem that matters, with no env var and no
  `project_root_folder`. It is strictly more robust here than Strategy 2/3 (which are dead)
  and at least as robust as Strategy 1 (which dies after the vendored→installed move) or
  Strategy 0c (`CURSUS_PROJECT_BASE`, which only fires if the template `__init__` ran).
- The execution account is irrelevant to resolution: paths are resolved + uploaded to S3 at
  compile time (in the MODS account for MODS mode, in the notebook's account for SAIS mode);
  whichever account ultimately runs SageMaker — the **user's onboarded account** in MODS mode
  via the cross-account role, or the same account in SAIS mode — only ever sees the S3 URI →
  `/opt/ml/code`. Nothing the resolver produces has to be valid in the execution account.

**Net rule for the design:** the only thing 0a must guarantee is *"on the compile host, the
config file's directory walks up to a dir that actually contains the project's `dockers/`."*
In MODS Lambda that host is the Lambda's `/tmp` and the config path is `/tmp`-absolute, so the
guarantee holds by construction. The runtime container path is the SDK's concern, not ours —
the definition-time/runtime gap is real but is **already bridged by the S3 upload**, not by
path resolution.

## Risks & Open Questions

1. **Pure-DAG entry points with no config file.** If a flow compiles a DAG without
   `load_configs` (no `input_file`), 0a cannot fire; it must fall to 0b (caller `__file__`)
   or 0c. Audit whether any production entry point compiles without a config path.
2. **Marker ambiguity.** Walking up from the config file to "nearest project marker" must
   match `project_discovery.py`'s markers exactly, and stop at the first dir containing
   `dockers/`|`scripts/` to avoid over-shooting to a monorepo root. Reuse `_find_config_dir`.
3. **Config dir nested away from scripts.** If a consumer puts `pipeline_config/` and
   `dockers/` in different ancestors, the walk-up heuristic needs a tie-break (prefer the
   dir that has BOTH, else the config dir's parent).
4. **`cursus.toml` scope creep.** Keep it minimal (`project_root`, `scripts_dir`); it is an
   optional override, not a new config system.

## Related Notes

### Design notes
- [Hybrid Strategy Deployment Path Resolution Design](../1_design/hybrid_strategy_deployment_path_resolution_design.md) — the 3-scenario design + 5 strategies (the system this extends).
- [Config Portability Path Resolution Design](../1_design/config_portability_path_resolution_design.md) — config-portability rationale; closest prior art to config-anchored resolution.
- [Deployment Context Agnostic Path Resolution Design](../1_design/deployment_context_agnostic_path_resolution_design.md) — the goal of one config working across deployment contexts.
- [Cursus Package Portability Architecture Design](../1_design/cursus_package_portability_architecture_design.md) — universal deployment compatibility (PyPI / source / bundled).

### Analysis notes
- [2025-11-24 Processing Step Path Resolution Cache Poisoning Fix](2025-11-24_processing_step_path_resolution_cache_poisoning_fix.md) — why `effective_source_dir` resolution is host-side + existence-gated (constrains where the anchor must point).
- [Deployment Portability Analysis — Step Catalog Import Failures](deployment_portability_analysis_step_catalog_import_failures.md) — adjacent deployment-mode failure analysis.

### Entry points & code
- [Cursus Package Overview](../00_entry_points/cursus_package_overview.md) · [Core and MODS Systems Index](../00_entry_points/core_and_mods_systems_index.md).
- `core/utils/project_discovery.py` — the existing marker-detection primitive (`_CONFIG_DIR_NAMES`, `_PROJECT_MARKERS`) this design reuses as the anchor.
- `core/utils/hybrid_path_resolution.py` + `core/base/config_base.py:192-207` — the resolver and its entry point.

### Cross-vault
- abuse_slipbox `thought_cursus_config_anchored_project_registration.md` (FZ 18g3h2c) — the project-trail synthesis resolving the 18g3h2a (self-declare) vs 18g3h2b (caller-introspection) dialectic.

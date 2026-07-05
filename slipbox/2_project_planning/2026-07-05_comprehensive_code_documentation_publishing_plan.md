---
tags:
  - project
  - planning
  - documentation
  - sphinx
  - read_the_docs
  - api_reference
  - mcp
keywords:
  - comprehensive code documentation
  - sphinx recursive autosummary
  - sphinx-click CLI reference
  - MCP tool reference generation
  - step interface catalog generation
  - pipeline catalog reference generation
  - read the docs hosting and versioning
  - interrogate docstring coverage gate
topics:
  - plan to build and publish comprehensive code documentation for the cursus package
language: python
date of note: 2026-07-05
---

# Cursus Documentation Build & Publish — Project Plan

**Date:** 2026-07-05 | **Status:** Proposed (ready for execution) | **Owner:** Docs lead + staff eng | **Target package:** `cursus` 2.8.0 (PyPI, MIT)

> Note on location: this lives in `slipbox/2_project_planning/` — there is no `3_project_planning` in the repo (the numbered dirs are `0`, `00`, `01`, `1`–`7`; `3_*` is `3_llm_developer`). Filed here to match the planning-doc convention.

---

## 1. Executive summary

**Goal:** Stand up a comprehensive, auto-regenerating, hosted code-documentation site for the `cursus` package (328 Python modules, a ~13-group Click CLI, a 70-tool MCP surface, 54 declarative step interfaces, and a 44-DAG pipeline catalog) that never drifts from the code despite a release cadence of multiple tags per day.

**Current state:** A Sphinx scaffold exists under `docs/` (autodoc + napoleon + myst_parser + `sphinx_rtd_theme`) but was **last touched 2025-09-11 ("black formatting")** — it predates the entire 1.8.0 → 2.8.0 arc. It is badly stale: it references the **deleted `cursus.workspace` module** (`docs/api/workspace.rst`), hand-maintains 8 per-subsystem `.rst` files, and is missing every post-1.8.0 surface (`cursus.mcp`, data-driven `pipeline_catalog`, declarative `.step.yaml` interfaces, project tools, kiro runtime, classless builder synthesis, step-pack discovery). There is **no hosted docs site**: `[project.urls] Documentation` points at the README on GitHub; there is no `.readthedocs.yaml`, no GH Pages workflow, no docs CI.

**Recommended stack (one paragraph):** Stay on **Sphinx** (reuse the working `conf.py`, napoleon, mock-imports, intersphinx, doctest) and eliminate all hand-maintained API `.rst` by switching to **recursive `autosummary`** driven from a single `cursus` root — napoleon already parses the Google-style docstrings, so zero docstring rewrites are needed (this rules out `sphinx-autodoc2`, which renders Google docstrings raw). Swap the semi-abandoned `sphinx_rtd_theme` for **Furo**. Auto-render the CLI with **sphinx-click** pointed at the already-composed root group `cursus.cli:cli`. Emit the three structured surfaces (MCP tools, step interfaces, pipeline catalog) from their **live self-describing sources** via one build-time generator wired to Sphinx's `builder-inited` event, so those pages regenerate on every build with no manual step. Host on **Read the Docs** via a repo-root `.readthedocs.yaml` with tag-triggered builds and native versioning (`stable` auto-follows the newest SemVer tag; per-minor `.0` snapshots via one Automation Rule). Add a **non-blocking, PR-only docstring coverage ratchet** (`interrogate`) plus Ruff `D` (google convention) on changed code. Curate — do **not** dump — the 954-file `slipbox/` corpus into a small narrative set (install/quickstart/tutorials/concepts/migration).

---

## 2. Current-state audit

### What exists
| Artifact | Path | Notes |
|---|---|---|
| Sphinx config | `docs/conf.py` | autodoc, napoleon, viewcode, intersphinx, autosummary, doctest, todo, coverage, ifconfig, githubpages, myst_parser; `sys.path.insert(0, ../src)`; `version = cursus.__version__` |
| Theme | `sphinx_rtd_theme` (in conf) | Semi-abandoned; replace with Furo |
| Root toctree | `docs/index.rst` | Points at the 8 hand-written api pages |
| Hand-written API pages | `docs/api/{api,cli,core,pipeline_catalog,registry,steps,validation,workspace}.rst` | 8 files, manually curated per subsystem |
| Committed autosummary stubs | `docs/api/generated/*.rst` (~18 files) | Stale partial snapshot (e.g. `cursus.compile_dag`, `cursus.PipelineDAG`) |
| Narrative | `docs/guides/quickstart.rst` | Single quickstart, likely stale |
| Build plumbing | `docs/Makefile`, `docs/_static/custom.css` | Present. `docs/_build/` is **already git-ignored** (`.gitignore:72`, zero tracked files) — only an untracked local HTML build exists on disk |
| Docs deps | `pyproject.toml [project.optional-dependencies].docs` | `sphinx>=6.0.0`, `sphinx-rtd-theme>=1.2.0`, `myst-parser>=2.0.0` |
| Internal KB | `slipbox/` (954 md) | `0_developer_guide`, `1_design`, `2_project_planning`, `3_llm_developer`, `4_analysis`, `5_tutorials`, `6_resources`, `7_archive`, plus per-module dirs (`api/`, `cli/`, `core/`, `steps/`, `pipeline_catalog/`, `registry/`, `validation/`, `workspace/`) — candidate narrative source, but unstructured/uneven |

### What is WRONG / stale (root causes of the rot)
| Problem | Evidence | Impact |
|---|---|---|
| References a **deleted module** | `docs/api/workspace.rst` documents `cursus.workspace` (removed) | autodoc import error; `sphinx-build -W` fails |
| **Missing** all post-1.8.0 surfaces | No pages for `cursus.mcp`, data-driven `pipeline_catalog`, `steps/interfaces` (`.step.yaml`), project tools, kiro runtime, step-pack discovery | Docs describe a package version that no longer exists |
| **Stale CLI** reference | `docs/api/cli.rst` hand-written; predates current 13 groups | Documents removed commands, omits `mcp`, `config`, `dag`, `exec-doc`, `projects`, `strategies` |
| **Hand-maintained per-module .rst** | 8 subsystem files + committed `api/generated/` stubs | Every structural change requires manual edits → guaranteed drift at multi-release/day cadence |
| **Obsolete `conf.py` cruft** | Hand-maintained `api_modules` list; `doctest_global_setup` importing `from cursus import PipelineDAG, compile_dag_to_pipeline` (stale top-level re-exports); `suppress_warnings = ["ref.doc"]` | Recursive-autosummary approach makes these redundant/misleading; prune in P0 |
| **No hosted site** | `[project.urls] Documentation = ".../blob/main/README.md"` | Users have no versioned API reference; `Documentation` URL is just the README |
| **No docs automation** | No `.readthedocs.yaml`, no docs GH Action | Nothing rebuilds docs on release tags |

**Verified live entry points for the rebuild** (scouted 2026-07-05):
- CLI root group: `cursus.cli:cli` at `src/cursus/cli/__init__.py:34` (`main = cli` is a wrapper at line 57 — point sphinx-click at `cli`, not `main`). 13 subgroups added via `add_command()`: `alignment`, `catalog`, `compile`, `config`, `dag`, `exec-doc`, `mcp`, `pipeline-catalog`, `projects`, `registry`, `steps`, `strategies`, `validate`.
- MCP registry: `src/cursus/mcp/registry.py` exposes `class ToolDef` (fields incl. `when`, `examples`), `get_registry(force_reload=False) -> Dict[str, ToolDef]`, `get_namespaces() -> Dict[str, str]`, `list_tools(namespace=None) -> List[ToolDef]`, `render_description(td) -> str`. Self-documenting: 70 tools / 12 namespaces, `tools.help` + auto `<ns>.help` tools, **163 examples** (sum of `ToolDef.examples` across `list_tools()`).
- Step interfaces: `load_step_interface()` at `src/cursus/steps/interfaces/__init__.py:177`; 54 `*.step.yaml` under `src/cursus/steps/interfaces/`.
- Pipeline catalog: 44 `*.dag.json` under `src/cursus/pipeline_catalog/shared_dags/` (nested), indexed by `src/cursus/pipeline_catalog/shared_dags/catalog_index.json`; router API `recommend_dag` / `load_shared_dag` / `build_and_compile`.
- Major auto-coverable subsystems (recursive autosummary): `cursus.core`, `cursus.api`, `cursus.cli`, `cursus.mcp`, `cursus.registry`, `cursus.step_catalog`, `cursus.steps`, `cursus.processing`, `cursus.pipeline_catalog`, `cursus.validation`, `cursus.mods`, `cursus.core.utils`.

---

## 3. Documentation architecture (information architecture)

Principle: **anything derivable from source is generated at build time**; only genuinely narrative pages are hand-authored. Site layout:

```
docs/
  index.md                     # landing: what is cursus, install, quickstart link, surface map
  narrative/
    install.md
    quickstart.md
    concepts/                  # architecture, DAG→pipeline compilation, config system, registry, step-pack model
    tutorials/                 # curated from slipbox/5_tutorials
    how-to/                    # task-oriented recipes
    migration/                 # 1.8.0 → 2.8.0 (workspace removal, data-driven catalog, declarative interfaces)
  api/
    index.rst                  # ONE recursive autosummary root over `cursus`
  cli.rst                      # sphinx-click on cursus.cli:cli
  reference/
    index.md                   # toctree over the generated pages below
    generated/                 # GIT-IGNORED, emitted by builder-inited hook
      mcp_tools.md
      step_catalog.md
      pipeline_catalog.md
  _templates/module.rst        # recursive autosummary template
  _ext/gen_reference.py        # build-time generator + Jinja templates
```

### (a) Narrative docs — curate `slipbox/`, do not dump it
The 954-file `slipbox/` is internal, uneven, and includes `2_project_planning`/`4_analysis`/`7_archive` that must **not** ship. Curation policy:
- **Promote** a small, edited set: `slipbox/5_tutorials` → `narrative/tutorials`; the readable parts of `slipbox/0_developer_guide` → `how-to`/`concepts`; `slipbox/1_design` → distilled `concepts/architecture` (rewrite, don't paste).
- **Author fresh**: `install.md`, `quickstart.md` (replace stale `guides/quickstart.rst`), and a `migration/` page covering the workspace removal + data-driven catalog + declarative interfaces.
- **Exclude** `slipbox/2_project_planning`, `4_analysis`, `7_archive`, `test/` and per-module scratch dirs from the published site (they stay as internal KB).
- Keep narrative in **Markdown (MyST)** to match the repo's markdown-first style and ease authoring.

### (b) Auto-generated API reference — all 328 modules
Single recursive `autosummary` root over `cursus`; a custom `_templates/module.rst` emits a stub page per submodule every build. New subsystems appear automatically; the deleted `cursus.workspace` simply stops being emitted. **This kills the stale/deleted-module drift class outright.**

> **Caveat (local vs. CI):** `cursus.workspace` is git-removed (0 tracked files, 0 `.py`), but an empty implicit-namespace remnant dir still exists locally — `import cursus.workspace` succeeds returning an empty module with `__file__=None`. On a **local** recursive-autosummary run this remnant can still be discovered and emitted as an empty page (and may raise `-W` warnings). Only a fresh RTD clone (dir absent) behaves exactly as claimed. P0 removes the local remnant dir so local and CI agree.

### (c) Auto-generated CLI reference
One `.. click:: cursus.cli:cli` directive with `:prog: cursus` `:nested: full` renders all 13 groups + every subcommand/option/argument from the live app.

### (d) Auto-generated MCP tool reference — exploit the self-documenting registry
Build-time generator walks `get_namespaces()` → `list_tools(ns)` → per `ToolDef`: `render_description(td)` (folds in `when` + `examples`) → one section per tool with its examples (163 across the surface). Never hand-written. Guard the import in `try/except` so a registry failure logs a warning instead of killing the build.

### (e) Auto-generated step-interface catalog — from 54 `.step.yaml`
Generator loads each `steps/interfaces/*.step.yaml` (via `load_step_interface()` or `yaml.safe_load`) and emits, per step: contract I/O paths, spec dependencies/outputs, env vars, `job_arguments`.

### (f) Auto-generated pipeline-catalog reference — from `catalog_index.json`
Generator reads `pipeline_catalog/shared_dags/catalog_index.json` into a filterable table (framework / task_type / complexity / features / node_count across the 44 DAGs) plus `recommend_dag` / `load_shared_dag` / `build_and_compile` usage.

| Surface | Source of truth | Generation mechanism | Drift risk after |
|---|---|---|---|
| Python API (328 modules) | docstrings | recursive `autosummary` + napoleon | none (root-discovered) |
| CLI (13 groups) | `cursus.cli:cli` | `sphinx-click` `:nested: full` | none |
| MCP tools (70/12 ns) | `cursus.mcp.registry` | `gen_reference.py` (builder-inited) | none |
| Step interfaces (54) | `steps/interfaces/*.step.yaml` | `gen_reference.py` | none |
| Pipeline catalog (44) | `shared_dags/catalog_index.json` | `gen_reference.py` | none |
| Narrative | curated `slipbox/` + fresh | hand-authored MyST | reviewed per PR |

---

## 4. Tooling decisions (with rationale)

| Decision | Choice | Why (grounded in cursus) | Rejected alternatives |
|---|---|---|---|
| Doc generator | **Stay on Sphinx** | Working `conf.py`, napoleon parses the Google-style docstrings today, mock-imports/intersphinx/doctest already configured. Only real problem is staleness, not tooling. | MkDocs+mkdocstrings/Material, pdoc — full migration, lose sphinx-click/intersphinx/doctest for no docstring-format win |
| API autogen | **Recursive `autosummary`** (one `cursus` root + `_templates/module.rst`) | Zero docstring changes; auto-discovers all 328 modules; deleted modules vanish. Set `autosummary_imported_members=False` (current `True` double-documents the many `__init__` re-exports). | `sphinx-autodoc2` — **napoleon-incompatible**, renders `Args:/Returns:` raw across 328 Google-style modules (dealbreaker); `sphinx-apidoc` — commits/regenerates coarse files, more churn |
| Theme | **Furo ≥ 2024.8.6** | Better nested nav, light/dark, mobile, near-zero config; Furo ignores rtd `html_sidebars`/`html_theme_options` so remove the old block. | `sphinx_rtd_theme` — semi-abandoned |
| CLI docs | **sphinx-click ≥ 6.2.0** on `cursus.cli:cli` | Root group already composes all 13 subgroups; one directive = full always-in-sync reference. Point at `cli`, not the `main` wrapper. | mkdocs-click (second toolchain), Typer rewrite (risky, docs-only win) |
| Structured-source docs | **One `gen_reference.py` on `builder-inited`** emitting MyST | Runs inside `sphinx-build` → identical locally, in CI, on RTD, with no second orchestration point to drift. Plain Jinja templates are debuggable. | Standalone pre-build script wired into Makefile+CI+RTD (three places to drift); custom docutils directive (verbose/version-sensitive); sphinx-jinja (lightly maintained) |
| Heavy deps at build | **`autodoc_mock_imports`** | autosummary/autodoc/sphinx-click import `cursus` at build; SageMaker SDK 2.x, torch, xgboost, etc. must be mocked or RTD import-fails. Keep the list current — the one recurring manual touchpoint. | Installing full heavy stack on RTD (slow/fragile) |
| Narrative authoring | **MyST (myst-parser ≥ 4.0)** | Markdown-native, matches slipbox style; RST only for the thin autosummary/CLI roots. | — |

**Concrete `conf.py` changes:** add `sphinx_click` to `extensions`; keep napoleon/autosummary/intersphinx/doctest; set `autosummary_generate=True`, `autosummary_imported_members=False`, `add_module_names=False`; replace theme with `html_theme='furo'` and delete the rtd `html_theme_options`/`html_sidebars` block; ensure `autodoc_mock_imports` covers `boto3, botocore, sagemaker, torch, pytorch_lightning, transformers, xgboost, sklearn, pandas, numpy, matplotlib, seaborn, plotly, networkx, lightgbm`; in `setup(app)` do `app.connect('builder-inited', ...)` to run the generator (wrapped in try/except). **Prune the now-obsolete blocks:** the hand-maintained `api_modules` list, the `doctest_global_setup` importing `from cursus import PipelineDAG, compile_dag_to_pipeline` (stale top-level re-exports), and `suppress_warnings = ["ref.doc"]`; and remove the `cursus.workspace` reference.

**`_templates/module.rst`** (recursive template — required, else `:recursive:` only lists names):
```
{{ fullname | escape | underline }}
.. automodule:: {{ fullname }}
   :members:
   :show-inheritance:
{% block modules %}{% if modules %}
.. rubric:: Modules
.. autosummary::
   :toctree:
   :recursive:
{% for item in modules %}   {{ item }}
{% endfor %}{% endif %}{% endblock %}
```

**`docs/api/index.rst`** (replaces all 8 hand-written pages):
```
API Reference
=============
.. autosummary::
   :toctree: generated
   :template: module.rst
   :recursive:

   cursus
```

**`docs/cli.rst`:** `.. click:: cursus.cli:cli` / `:prog: cursus` / `:nested: full`.

**`pyproject.toml` docs extra (bump):**
```
docs = [
  "sphinx>=8.1",
  "furo>=2024.8.6",
  "myst-parser>=4.0",
  "linkify-it-py>=2.0",        # REQUIRED: conf.py enables the MyST "linkify" extension (docs/conf.py:213)
  "sphinx-click>=6.2.0",
  "jinja2>=3.1",
  "pyyaml>=6",
]
```
> `myst_enable_extensions` in the current `conf.py` includes `"linkify"`, which needs `linkify-it-py` at build time. Either add the dep above (equivalently `myst-parser[linkify]`) **or** drop `"linkify"` from `myst_enable_extensions` — otherwise the P0 clean-build gate fails on a fresh environment.

---

## 5. Hosting, CI & versioning

**Host: Read the Docs (Community/OSS tier)** — lowest friction for a multi-release/day MIT package. Tag-triggered builds via webhook (no Actions docs workflow to keep green), `stable` auto-tracks the newest non-prerelease SemVer tag, PR previews + visual diff via the RTD GitHub App, hosted search/version switcher/redirects/HTTPS custom domain out of the box. GH Pages + Actions (`mike`/`sphinx-multiversion`) was rejected: you'd own the versioning tool, deploy action, alias logic, and a hand-rolled PR-preview pipeline — the opposite of the goal (and `sphinx-multiversion` rebuilds all refs per publish, untenable at this cadence).

**`.readthedocs.yaml` (repo ROOT):**
```
version: 2
build:
  os: ubuntu-24.04
  tools: {python: "3.12"}
sphinx:
  configuration: docs/conf.py
  fail_on_warning: false     # keep false until de-staled; then tighten
python:
  install:
    - method: pip
      path: .
      extra_requirements: [docs]
formats: [pdf]               # optional — drop to speed builds if not needed
```
No `build.jobs` needed — the `builder-inited` generator runs inside RTD's `sphinx-build`.

**Versioning policy** (RTD native, all tag-driven, none manual):
- `latest` → `main`; `stable` → greatest non-prerelease tag, auto-rebuilt on each newer tag.
- Incoming tags default **inactive** (prevents build sprawl from the patch flood).
- One Automation Rule — Custom regex `^v?\d+\.\d+\.0$`, Tag, **Activate** → a permanent snapshot per minor (`2.0.0`, `2.1.0`, `2.8.0`…), ignoring patches.
- Set project **Default version = `stable`**.
- Ensure release tags are SemVer-like and the tag commit's `VERSION` matches (conf reads `cursus.__version__`; RTD reads git tags — they must agree).

**PR previews:** connect via the RTD GitHub App; enable "Build pull requests"; previews get only public env vars.

**CI (separate from publishing):** a PR/push GitHub Action that runs `sphinx-build -b html docs docs/_build/html` (start without `-W`; add `-W --keep-going` once de-staled) as a pre-merge import/link check. Plus the docstring job (§6).

**`pyproject` URL update:** once the first `stable` build is green, set `[project.urls] Documentation = "https://cursus.readthedocs.io/en/stable/"`.

**Housekeeping:** add `docs/reference/generated/` to `.gitignore`. `docs/_build/` is **already** ignored (`.gitignore:72`) with zero tracked files — no `git rm --cached` is needed.

---

## 6. Docstring standards & coverage gate

Keep **Google style** (napoleon-compatible, zero conversion). Two-tier, **PR-only, never release-triggered** (a gate must block before merge, not break tag automation).

**Tier 1 — coverage ratchet (`interrogate`, blocking):** MEASURE FIRST, do not guess 80%.
```
interrogate -vv --ignore-init-method --ignore-magic --ignore-private \
  --ignore-nested-functions --ignore-property-decorators \
  --exclude 'tests,docs' src/cursus
```
Set `[tool.interrogate] fail-under = floor(measured %)`; ratchet up only when comfortably exceeded. Config: `ignore-private/semiprivate/magic/init/nested/property/overloaded = true`, `style = "google"`, `omit-covered-files = true`, exclude `tests`, `docs`, generated dirs.

**Tier 2 — style on changed code (Ruff `D`, google convention):** `[tool.ruff.lint] select = ["E","F","I","D"]`, `[tool.ruff.lint.pydocstyle] convention = "google"`, start with `ignore = ["D100","D104"]` and scope `D1xx` presence rules to changed files via pre-commit so the legacy backlog doesn't wall off merges.

**Drift (advisory, non-blocking):** standalone `pydoclint --style=google` (or Ruff `DOC` rules under `--preview`, pinned). **Do NOT adopt `darglint`** (maintenance mode).

**Doctest:** do **not** run `pytest --doctest-modules` over all 328 SageMaker-heavy modules. Curate a small `docs/examples/*.md` allowlist. More valuably, the MCP/CLI references are generated from live objects, so the 163 examples are validated structurally on every build.

**CI job** (`.github/workflows/docs-quality.yml`, on PR + push to main): `interrogate -c pyproject.toml src/cursus` (blocking) · `ruff check --select D src/cursus` (blocking on selected rules) · `pydoclint --style=google src/cursus || true` (advisory).

---

## 7. Phased rollout

| Phase | Scope | Key tasks | Effort | Depends on | Acceptance criteria |
|---|---|---|---|---|---|
| **P0 — De-stale & modernize scaffold** | Fix the rot; get a clean local build | `git rm docs/api/{workspace,api,cli,core,steps,registry,validation,pipeline_catalog}.rst`; `git rm -r docs/api/generated`; add `docs/reference/generated/` to `.gitignore` (`docs/_build/` already ignored); remove the empty local `src/cursus/workspace/` remnant dir; prune stale `conf.py` cruft (`api_modules`, the `PipelineDAG/compile_dag_to_pipeline` `doctest_global_setup`, `suppress_warnings=['ref.doc']`); swap theme → Furo; set `autosummary_imported_members=False`; extend `autodoc_mock_imports`; bump `docs` extra incl. `linkify-it-py` | 0.5–1 day | — | `pip install -e '.[docs]' && sphinx-build -b html docs docs/_build/html` succeeds with no import errors; no reference to `cursus.workspace` |
| **P1 — API autogen (328 modules)** | Recursive autosummary | Add `_templates/module.rst`; new `docs/api/index.rst` root over `cursus`; verify new subsystems (`cursus.mcp`, `pipeline_catalog`, `steps.interfaces`, `step_catalog`, `processing`) appear | 1 day | P0 | Every top-level `cursus.*` subpackage has a generated page incl. `mcp`; no page for deleted modules |
| **P2 — CLI + MCP + catalog generation** | The 3 structured surfaces + CLI | Add `sphinx_click`, `docs/cli.rst` on `cursus.cli:cli`; write `docs/_ext/gen_reference.py` + Jinja templates for MCP (`get_namespaces`/`list_tools`/`render_description`), step interfaces (54 `.step.yaml`), pipeline catalog (`catalog_index.json`); wire `builder-inited` (try/except) | 2–3 days | P1 | CLI page lists all 13 groups; MCP page lists 70 tools w/ when+examples; step page lists 54 interfaces; catalog page tabulates 44 DAGs; all regenerate on rebuild |
| **P3 — Narrative & tutorials** | Curate slipbox | Author `install.md`, `quickstart.md`, `concepts/*`, `migration/` (workspace removal, data-driven catalog, declarative interfaces); promote+edit `slipbox/5_tutorials`; build `reference/index.md` + top-level toctree | 2–4 days | P1 | Landing page + install + quickstart + ≥3 tutorials + migration page render and cross-link; no planning/analysis/archive dirs published |
| **P4 — Hosting, CI & versioning** | Publish | Add `.readthedocs.yaml`; connect RTD GitHub App; enable PR builds; add minor-tag Automation Rule; set Default=`stable`; add PR docs-build Action; update `[project.urls] Documentation` | 0.5–1 day | P0–P3 | `stable` + `latest` build green on RTD; a pushed `X.Y.0` tag auto-publishes; PR preview link appears; `Documentation` URL points to RTD |
| **P5 — Coverage gate & maintenance** | Quality ratchet | Measure with `interrogate`; set `fail-under=floor(%)`; add Ruff `D` google; pre-commit hooks; `docs-quality.yml`; add curated doctest allowlist | 1 day | P4 | CI blocks coverage regressions; Ruff `D` runs on changed files; drift check advisory; gate is PR-only |

Total rough effort: ~7–11 engineer-days, front-loaded on P2/P3.

---

## 8. Maintenance model

- **Generation-from-source is the core anti-drift mechanism.** The recursive autosummary root, `sphinx-click` directive, and `gen_reference.py` (`builder-inited`) all read the **live** package at build time, so every RTD build — triggered by each tagged release — snapshots the API/CLI/MCP/interface/catalog surfaces exactly as of that tag. No page is hand-maintained per release.
- **`stable` auto-follows the newest tag**; a single Automation Rule pins per-minor `.0` snapshots. No human touches the version list on the multi-release/day cadence.
- **The one recurring manual touchpoint:** when an upstream sync adds a new heavy dependency, add it to `autodoc_mock_imports` (else RTD import-fails). Add this to the release checklist.
- **Release checklist additions:** (1) `sphinx-build` passes locally / in PR CI; (2) new heavy deps mocked; (3) tag is SemVer and its `VERSION` matches the commit.
- **Guardrails:** `gen_reference.py` wrapped in try/except (a registry import failure logs, doesn't kill the build); `fail_on_warning: false` until warnings are triaged, then tighten to `-W` in PR CI first, RTD later.

---

## 9. Risks & open questions

**Risks**
- **Build-time import of `cursus`** (autosummary/autodoc/sphinx-click) — a newly added heavy dep not in `autodoc_mock_imports` breaks RTD builds. Mitigation: mock-list in release checklist; keep CLI/registry module imports side-effect-free.
- **328-module autosummary surfaces many warnings** — keep `fail_on_warning: false` initially; do not enable `-W` on RTD until the warning backlog is triaged.
- **MCP generator runs live registry code** at build — guard with try/except so a failure degrades to a warning, not a build failure (esp. under RTD where deps are mocked).
- **Autosummary sidebar limitation** — it doesn't add every symbol to the global sidebar; Furo's page-local TOC mitigates; don't promise a giant global symbol tree.
- **Tag/VERSION mismatch** — if a tag commit's `VERSION` differs from the tag string, the RTD switcher and in-page version disagree. Enforce in release process.
- **`cursus.workspace` local remnant** — the empty namespace dir on disk can still be emitted/warned on locally; P0's removal of the dir is required to make local builds match a clean RTD clone.
- **slipbox curation scope creep** — 954 files; strictly exclude `2_project_planning`/`4_analysis`/`7_archive`; timebox P3.

**Open questions**
- Which exact `slipbox/` subtrees (beyond `5_tutorials` and readable parts of `0_developer_guide`/`1_design`) should be promoted vs. remain internal-only?
- Custom domain (e.g. `docs.<domain>`) desired, or is `cursus.readthedocs.io` sufficient for the `Documentation` URL?
- Should pre-release/rc tags ever publish, or only final SemVer tags (affects the Automation Rule regex)?
- Confirm `load_step_interface()` is the intended API for the step-catalog generator vs. raw `yaml.safe_load` of `*.step.yaml` (both viable; pick one for consistency).
- Is `pdf`/`epub` output actually wanted, or drop `formats` to speed builds?
- Target measured docstring coverage floor is unknown until `interrogate -vv` is run — the `fail-under` number is set in P5, not now.

---

## Appendix — provenance

This plan was produced from a repo scout (2026-07-05) plus a 6-dimension web research pass (API autogen, CLI docs, hosting/CI, versioning, docstring coverage, generation-from-structured-sources; 88 sources) and an adversarial repo-fit review. Reviewer verdict: repo-fit accuracy **high** — the load-bearing counts and entry points (328 modules, 13 CLI groups, 70 MCP tools / 12 namespaces, 54 `.step.yaml`, 44 `.dag.json`, `load_step_interface` at `:177`, `main = cli` at `:57`, docs extra + README-only Documentation URL, no `.readthedocs.yaml`/CI) all verify against the tree. Corrections applied post-review: MCP example count 135 → **163**; CLI `def cli()` at `:34` (not `:32`); `docs/_build` is already git-ignored (no `git rm` no-op); added the required `linkify-it-py` dep; documented the `cursus.workspace` local-vs-CI remnant; enumerated `step_catalog`/`processing`; specified the exact stale `conf.py` blocks to prune.

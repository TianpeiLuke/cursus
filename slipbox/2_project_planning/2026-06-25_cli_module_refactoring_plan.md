---
tags:
  - project
  - planning
  - cli
  - refactoring_plan
  - click_unification
keywords:
  - cursus cli refactoring
  - click group unification
  - console_scripts entry point
  - argparse dispatcher removal
  - cli shared helpers
  - dead code removal
  - alignment html report
  - level3_mode no-op
  - new cli commands
  - pipeline-catalog recommend
topics:
  - cli module refactoring plan
  - click-based command composition
  - cli consistency and de-duplication
  - cli command surface expansion
language: python
date of note: 2026-06-25
---
# CLI Module Refactoring Plan

## Overview

This document outlines a concrete plan to refactor `src/cursus/cli/` (3,742 LOC across
8 files). The module is **functional but uneven**: all six subcommand modules are already
built on `click` and export clean command objects, but they are dispatched by an
`argparse` shim that mutates `sys.argv`, there is no `cursus` console entry point, and the
modules carry verified dead code, a deceptive no-op flag, cross-module inconsistencies,
and heavy duplication. The plan removes the rot, de-duplicates, unifies the dispatcher on
a single `click` group with a `cursus` entry point, and adds five high-value commands that
expose engine capabilities already reachable via the new `cursus.mcp` tool layer.

The findings below are grounded in a source-verified review of every file (each engine
symbol checked against real definitions).

## Goals

1. Remove genuinely stale / broken / dead code (highest value, lowest risk).
2. Eliminate a deceptive no-op API surface (`--level3-mode`) that implies behavior it has none of.
3. Standardize cross-module conventions: exit codes, error handling, `--format` semantics, logging.
4. De-duplicate repeated patterns (catalog instantiation, JSON/table output, try/except) into a shared module.
5. Unify the dispatcher on a single `click` group and add a `cursus` console_scripts entry point.
6. Add five high-value commands that are currently engine-only (`dag validate`, `pipeline-catalog recommend`, `validate run-scripts`, `config generate`, `mcp serve`).
7. Preserve backward compatibility for one release (keep `main = cli`, alias changed `--format` vocab).

## Alignment with Core Architectural Principles

### Single Source of Truth
A single root `click.Group` becomes the one place commands are registered; a single
`cli/_shared.py` becomes the one place output/error/instantiation patterns live. Today the
dispatcher hand-routes via an `if/elif` chain and each module re-implements formatting.

### Declarative Over Imperative
`click` groups compose declaratively (`cli.add_command(...)`) — this replaces the
imperative `sys.argv` mutation, `SystemExit` capture, and manual routing in the argparse
dispatcher.

### Explicit Over Implicit
Removing `--level3-mode` (accepted, stored in metadata, never consumed) and the broken
HTML report removes APIs that imply effects they do not have. Standardizing `return 1` on
error paths makes failure explicit to callers/CI.

### Convention Over Configuration
A `cursus` entry point + native `click` help (`cursus <sub> --help`) replaces the
`python -m cursus.cli` invocation and the hand-written argparse epilog.

## Current State (verified)

| File | LOC | Framework | State |
|------|-----|-----------|-------|
| `__init__.py` (dispatcher) | 136 | argparse | Shim over click; `sys.argv` mutation; no entry point |
| `catalog_cli.py` | 1115 | click group (17 cmds) | Disciplined; all 22 StepCatalog methods verified to exist; heavy duplication |
| `alignment_cli.py` | 946 | click group (4 cmds) | Contains broken HTML report + no-op `--level3-mode` |
| `registry_cli.py` | 756 | click group (7 cmds) | Bare-`return` error paths; unused imports |
| `compile_cli.py` | 377 | click command | Cleanest; effectively no stale code |
| `exec_doc_cli.py` | 253 | click group | Bare-`return` error paths; `--format` means file-format (divergent) |
| `project_cli.py` | 150 | click group (2 cmds) | New; case-sensitive `--format` bug; no try/except |

## Implementation Phases

### Phase 1: Dead-code & no-op removal (S, low risk) — do first

- **Remove broken HTML report** in `alignment_cli.py` (`generate_html_report`, ~lines 260–395).
  Verified dead/broken: it reads `results['level{N}']['passed'|'issues']` and checks
  `status == 'PASSING'`, but the engine (`unified_alignment_tester.py:271,283`) produces
  `results['validation_results']['level_N']` with `'status'` fields and overall `'PASSED'/'FAILED'`.
  Any `--format html|both` silently emits an empty report. Remove the `html`/`both`
  `--format` choices and the `save_report` html branch (~line 254). (File a follow-up if a
  correct HTML renderer is later wanted.)
- **Remove deceptive `--level3-mode`** from `validate` (424–427), `validate-all` (527–530),
  `validate-level` (754–757), `list-scripts` (885–887), plus its metadata stamping
  (482, 634, 794). `UnifiedAlignmentTester.__init__` swallows it via `**kwargs` and never uses it.
- **Drop unused imports / dead locals**: `catalog_cli` (`os`, `sys`, `List`, `Dict`),
  `registry_cli` (`os`, `sys`); move `shutil` import to module level; in `project_cli`
  remove/raise the unused logger and align its level to `INFO`.

### Phase 2: Consistency standardization (M, medium risk)

- **Exit codes**: convert bare `return` on error paths to `return 1` in `registry_cli`
  (70, 82, 157, 172, 206, …) and `exec_doc_cli` (124, 138, 171, 203, 232) to match
  `compile_cli`. (Behavior change — see Risks.)
- **`--format` case bug**: `project_cli` checks `format == "json"` case-sensitively despite
  `case_sensitive=False`; lowercase the value (lines 68, 119).
- **Error handling**: wrap `discover_pipeline_projects` calls in `project_cli` (66, 113)
  with `try/except` + `logger.error(..., exc_info=True)`; add `exc_info=True` to the
  generic `except Exception` blocks in `registry_cli` to stop losing stack traces.
- **`catalog_cli` defensive access**: guard `str(finfo.annotation)` / `finfo.default`
  (~819, 826), the `load_spec_class`→`serialize_spec` path in `list-specs` (745–749), and
  `builder_class.__name__` in `component-info` (1065–1067).

### Phase 3: De-duplication via `cli/_shared.py` (L, medium risk)

Create `cli/_shared.py` with:
- `get_catalog()` — `@lru_cache(maxsize=1)` lazy `StepCatalog` accessor, replacing the
  16 `StepCatalog()` instantiations in `catalog_cli`.
- `format_json_output(data)` / `format_table_output(title, rows)` — collapse the ~17
  duplicated json-vs-table branches (~250 LOC saved in `catalog_cli` alone).
- `safe_cli_command` decorator — wrap the repeated `try/except` + `logger.error` pattern.
- `echo_step(label, emoji)` + `SECTION_EMOJI` / `SEP_WIDTH` constants — shared by
  `catalog`/`compile`/`exec_doc`.
- Unify `--format` vocabulary: standardize `list-*` (`table|json`) and `show-*` (`text|json`)
  onto one set; alias old values for one release; document the chosen set.

Land this **after** Phase 1 and behind `click.testing.CliRunner` tests (largest blast radius).

### Phase 4: Dispatcher unification + `cursus` entry point (S, medium risk)

- Rewrite `cli/__init__.py` as a root `click.Group` named `cli`; register existing objects:
  `cli.add_command(alignment)`, `cli.add_command(catalog_cli, name="catalog")`,
  `cli.add_command(compile_pipeline, name="compile")`, `cli.add_command(exec_doc_cli)`,
  `cli.add_command(projects_cli, name="projects")`, `cli.add_command(registry_cli)`.
  Delete the argparse parser, `sys.argv` mutation, `SystemExit` wrapping, `if/elif` routing
  (137 → ~30 lines).
- Keep `main = cli` (or a `main()` that calls `cli()`) for one release (backward compat).
- Update `__main__.py` to call `cli()`.
- Add `[project.scripts]` to `pyproject.toml`: `cursus = "cursus.cli:cli"`.
- Update epilog examples from `python -m cursus.cli ...` to `cursus ...`.
- Verify with `click.testing.CliRunner`.

### Phase 5: New commands (L, medium risk) — on top of the unified group

| Command | Wraps | Priority |
|---------|-------|----------|
| `dag validate` | `api.dag:PipelineDAGResolver.validate_dag_integrity` | high |
| `pipeline-catalog recommend` | `pipeline_catalog.core.agent_tool:pipeline_catalog_tool(action='recommend')` | high |
| `validate run-scripts` | `validation.script_testing.api:run_dag_scripts` | high |
| `config generate` | `api.factory.dag_config_factory:DAGConfigFactory.generate_all_configs` | medium |
| `mcp serve` | `cursus.mcp.server:main` | medium |

Add as new modules (`dag_cli.py`, `pipeline_catalog_cli.py`, etc.) registered via
`cli.add_command`; reuse `_shared.py` helpers so they are consistent from day one. Use
lazy imports inside command bodies (as `compile_cli` already does) so `cursus --help` stays
fast and does not fail when an optional dependency (e.g. `mcp`, `script_testing`) is absent.

## Implementation Details

### Dispatcher target shape
```python
# cli/__init__.py (~30 lines)
import click
from .alignment_cli import alignment
from .catalog_cli import catalog_cli
from .compile_cli import compile_pipeline
from .exec_doc_cli import exec_doc_cli
from .project_cli import projects_cli
from .registry_cli import registry_cli

@click.group()
@click.version_option()
def cli():
    """Cursus — pipeline development and validation tools."""

cli.add_command(alignment)
cli.add_command(catalog_cli, name="catalog")
cli.add_command(compile_pipeline, name="compile")
cli.add_command(exec_doc_cli)
cli.add_command(projects_cli, name="projects")
cli.add_command(registry_cli)

main = cli  # backward-compat shim for one release
```

### Verified dead-code references
- `alignment_cli.generate_html_report` reads `results['level1']…` / `['passed']` /
  `status == 'PASSING'`; engine emits `results['validation_results']['level_1']` /
  `['status']` / `'PASSED'`.
- `--level3-mode` → `UnifiedAlignmentTester.__init__(**kwargs)` never reads it.

## Migration Strategy

1. Land Phases 1–2 as small, independent commits (pure subtraction + consistency); build-gate each.
2. Land Phase 3 behind `CliRunner` tests.
3. Land Phase 4 as one commit (group swap + entry point + `__main__` + epilog).
4. Land Phase 5 commands incrementally (high priority first).

Each phase ends with `brazil-build` green and `ruff check src/cursus/cli/` clean.

## Backward Compatibility

- Keep `main = cli` so `from cursus.cli import main` and `python -m cursus.cli <cmd>` keep working.
- Subcommand invocation strings (e.g. `catalog list`) are unchanged.
- Alias old `--format` values for one release; document the unified vocabulary.
- Removing `--level3-mode` and the broken `html` format are observable changes — note in
  CHANGELOG; treat per semver (minor, since neither had real effect).

## Testing Approach

- Add `tests/cli/` using `click.testing.CliRunner` to invoke `cli` directly (no `sys.argv` gymnastics).
- Smoke-test each subcommand's happy path + one error path (assert exit code).
- Regression-test the de-dup (`_shared.py`) against pre-refactor output for representative commands.

## Risks and Mitigations

- **Broken HTML report may be consumed by external CI** → confirm no consumer before deleting the `html` format; prefer rewrite if any.
- **Removing `level3_mode` / changing `main`** are observable API changes → keep `main = cli` shim, announce deprecation.
- **`return` → `return 1`** changes exit codes → may surface latent red in CI (intended); coordinate rollout.
- **`StepCatalog` lru_cache** assumes immutability within a process → scope/​key the cache by workspace if any command discovers against a different workspace-dir.
- **`_shared.py` blast radius** is large → land behind `CliRunner` tests, after cheap removals.
- **New commands pull heavier deps** (`mcp.server`, `script_testing`, `dag_config_factory`) → keep lazy imports inside command bodies so `cursus --help` stays fast and tolerant of missing optional deps.
- **Unifying `--format` vocab** is itself a behavior change → alias old values one release, document.

## Success Criteria

1. Zero references to deleted/no-op surfaces (`generate_html_report`, `--level3-mode`).
2. Consistent error exit codes (`return 1`) and `--format` handling across all commands.
3. `cli/_shared.py` exists; `catalog_cli` instantiates `StepCatalog` once; json/table duplication collapsed.
4. `cursus` console command works (`cursus --help`, `cursus catalog list`, …); `__init__.py` ≈ 30 lines.
5. Five new commands callable and consistent with `_shared.py`.
6. `brazil-build` green; `ruff check src/cursus/cli/` clean; `tests/cli/` passing.

## Conclusion

The CLI's building blocks (clean `click` modules) are sound; the work is removing rot,
standardizing conventions, collapsing duplication, and replacing an argparse shim with a
native `click` group plus a `cursus` entry point — then extending the surface with five
commands that expose already-built engine capabilities. Sequencing dead-code removal first
keeps risk low and lets the higher-blast-radius de-dup and unification land on clean modules.

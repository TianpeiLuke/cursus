# Path Resolution & the Caller Hook

Every Cursus step that runs code on SageMaker points at a **docker source
directory** — the folder that gets uploaded as the step's `source_dir`
(processing steps also support `processing_source_dir`). On a config object these
are stored as *relative* paths (e.g. `"scripts"` or `"src/training"`), because a
config that hard-codes an absolute path stops working the moment it moves between
a laptop, a notebook, a Lambda bundle, or a pip-installed environment.

The job of **path resolution** is to turn that relative `source_dir` into a real
absolute path on whatever machine is compiling the pipeline. Cursus does this
with a **hybrid, multi-strategy resolver** and, at the top of that stack, a
**caller hook (Strategy 0)** that lets the pipeline's entry point tell Cursus
exactly where its project lives. This page explains both.

The relevant code:

- `src/cursus/core/utils/hybrid_path_resolution.py` — the resolver, the strategy
  ladder, and the caller-hook functions (`set_project_root`, `get_project_root`,
  `resolve_anchor`).
- `src/cursus/core/base/config_base.py` — `resolve_hybrid_path()`, the config-level
  entry point that calls the resolver.
- `src/cursus/core/compiler/dag_compiler.py` — `PipelineDAGCompiler` and its
  `_resolve_project_root()` precedence logic.
- `src/cursus/core/compiler/single_node_compiler.py` — `SingleNodeCompiler`, which
  reuses the same precedence.

Related reading: [Configuration system](../concepts/config_system.md),
[Compilation](../concepts/dag_and_compilation.md), and the
[CLI reference](../cli.rst).

## The problem: one config, many deployment modes

The same config JSON and the same pipeline code may be compiled in very different
layouts:

| Deployment mode | Where `cursus` lives | Where the project lives |
| --- | --- | --- |
| Development monorepo | `<repo>/src/cursus/` | `<repo>/<project>/` |
| Pip-installed / SAIS notebook | `site-packages/cursus/` | user's working directory |
| Lambda / MODS bundled | bundled next to project code | sibling of `cursus` in the bundle |

A relative `source_dir` like `"scripts"` means nothing on its own — it must be
joined to *some* base directory, and that base differs per mode. Cursus resolves
it by trying a ladder of strategies until one produces a path that actually exists
on disk.

## The strategy ladder

`HybridPathResolver.resolve_path(project_root_folder, relative_path)` in
`hybrid_path_resolution.py` runs the strategies below **in order** and returns the
first candidate that exists. Each strategy only checks paths that physically
exist (`Path.exists()`), so a strategy that guesses wrong falls through to the
next one rather than returning a bad path.

| # | Strategy | Method | Base it joins `relative_path` to |
| --- | --- | --- | --- |
| 0 | **Caller hook** (pushed project root) | `_pushed_project_root_discovery` | the project folder pushed by the entry point → `project_root / relative_path` |
| 0b | Explicit project base | `_explicit_project_base_discovery` | `CURSUS_PROJECT_BASE` env var, then `/ project_root_folder / relative_path` |
| 1 | Package location discovery | `_package_location_discovery` | derived from `Path(__file__)` of the cursus package (bundled sibling, direct, or monorepo `src/` layouts) |
| 2 | Working directory discovery | `_working_directory_discovery` | walks upward from `Path.cwd()` (up to 10 levels) |
| 3 | Generic path discovery | `_generic_path_discovery` | recursive search for a uniquely-named `project_root_folder` |
| 4 | Default scripts discovery | `_default_scripts_discovery` | the packaged `cursus/steps/scripts/` directory |

Strategies 1–4 are the older "figure it out from the environment" heuristics.
They still work and remain the fallback path. Strategies 0 and 0b are the two
*explicit* anchors — you tell Cursus where the project is instead of making it
guess.

```{note}
The whole ladder can be toggled by environment variables read in
`HybridResolutionConfig`: `CURSUS_HYBRID_RESOLUTION_ENABLED` (default `true`) and
`CURSUS_HYBRID_RESOLUTION_MODE` (`full`, `fallback_only`, or `disabled`). Under
normal operation you never touch these; they exist for staged rollout.
```

### Why Strategy 0 is the most robust

Strategies 1–4 all *infer* the project location from something ambient — where
the `cursus` package file sits, what the current working directory is, or a
recursive folder search. Those signals can be wrong or ambiguous (two folders
with the same name, a notebook launched from an unexpected cwd, a bundler that
flattens the tree).

Strategy 0 removes the guessing. The module that defines your pipeline's
`generate_pipeline()` **physically lives inside your project folder**, so its
`__file__` is a rock-solid pointer to the project. If that module hands its own
location to Cursus, resolution becomes a single deterministic join:

```
resolved = project_root / relative_path
```

No package-layout assumptions, no cwd assumptions. This is why the code and the
CHANGELOG call it "the most robust anchor across deployment modes."

## The caller hook: `set_project_root` / `get_project_root` / `resolve_anchor`

Three functions in `hybrid_path_resolution.py` implement the caller hook. They
are re-exported from `cursus.core.utils`, so you can import them from either
location:

```python
from cursus.core.utils import (
    set_project_root,
    get_project_root,
    resolve_anchor,
)
```

### `resolve_anchor(anchor)` — the shared normalizer

`resolve_anchor` collapses *either* a file *or* a directory into a single
project-root string, so `anchor_file=__file__` and `project_root=<dir>` end up
identical everywhere:

- A **file** (e.g. `__file__`) resolves to its **parent** directory.
- A **directory** (e.g. `Path(__file__).parent`) resolves to itself.
- A path that does not exist yet but is *file-shaped* (has a suffix, like a `.py`
  path) is treated as a file and reduced to its parent — so `anchor_file=__file__`
  works even before the module is materialized on disk.
- A falsy anchor returns `None`.

```python
resolve_anchor("/proj/run_pipeline.py")   # -> "/proj"
resolve_anchor("/proj")                    # -> "/proj"
resolve_anchor(None)                        # -> None
```

Every entry point routes its `project_root` / `anchor_file` arguments through this
one helper, which is what guarantees the file form and the directory form behave
the same.

### `set_project_root(...)` and `get_project_root()`

`set_project_root` pushes the project folder into a process-level variable
(`_pushed_project_root`). It accepts either a directory or a file and normalizes
through `resolve_anchor`, so you can pass whichever you have on hand:

```python
from pathlib import Path
from cursus.core.utils import set_project_root, get_project_root

set_project_root(__file__)               # a file -> its parent is the project root
set_project_root(Path(__file__).parent)  # a directory -> used as-is
# both leave get_project_root() pointing at the same folder

get_project_root()   # -> "/abs/path/to/project"
set_project_root(None)  # clears it
```

Once set, Strategy 0 (`_pushed_project_root_discovery`) checks
`get_project_root() / relative_path` before any other strategy runs. Because the
value is process-wide, it is honored by the entire compile chain —
config → resolver → step builder — not just the object you set it on.

```{warning}
The pushed root is **process-level** (module global). It persists until
overwritten or cleared. In a long-lived process (a notebook kernel, a service)
that compiles multiple pipelines from different projects, the most recent
`set_project_root` / compiler construction wins. The compilers set it during
`__init__`, so constructing a new compiler for a new project re-points it
correctly.
```

## `project_root` vs `anchor_file` on the entry points

You rarely call `set_project_root` yourself. Instead the entry points expose two
constructor parameters and push the root for you. Four entry points carry the
same pair of parameters:

| Entry point | Module |
| --- | --- |
| `PipelineDAGCompiler` | `core/compiler/dag_compiler.py` |
| `SingleNodeCompiler` | `core/compiler/single_node_compiler.py` |
| `ExecutionDocumentGenerator` | `mods/exe_doc/generator.py` |
| `DAGConfigFactory` | `api/factory/dag_config_factory.py` |

The two parameters are two spellings of the same idea:

- **`project_root`** — the project **folder** as a directory, typically
  `Path(__file__).parent`.
- **`anchor_file`** — a **file** inside the project, typically `__file__`. Its
  parent directory becomes the project root.

```python
from pathlib import Path
from cursus.core.compiler.dag_compiler import PipelineDAGCompiler

# Self-documenting form — pass the module's own file.
compiler = PipelineDAGCompiler(
    config_path="pipeline_config/config.json",
    anchor_file=__file__,          # project root = folder containing this module
)

# Equivalent explicit form.
compiler = PipelineDAGCompiler(
    config_path="pipeline_config/config.json",
    project_root=Path(__file__).parent,
)
```

`anchor_file=__file__` is exactly equivalent to
`project_root=Path(__file__).parent` — both flow through `resolve_anchor` and land
on the same folder. Prefer `anchor_file=__file__`: it is self-documenting and
survives the module being moved, because it always resolves relative to wherever
the file currently lives.

### Precedence and the disagreement warning

`PipelineDAGCompiler._resolve_project_root(project_root, config_path, anchor_file)`
is the shared precedence function (`SingleNodeCompiler` and
`ExecutionDocumentGenerator` both delegate to it; `DAGConfigFactory` mirrors the
same rule). The order is:

1. **Explicit `project_root`** — highest priority.
2. **Explicit `anchor_file`** — used when `project_root` is not given.
3. **Config-anchored inference** — when neither is given, derive the root from
   `config_path` by walking up from the config file. If the config's directory is
   named `pipeline_config` or `pipeline_configs`, the project is its parent; if
   that recognized directory is one level up (a versioned config subdir), the
   project is two levels up; otherwise the config file's own directory is used.

If **both** `project_root` and `anchor_file` are supplied and they resolve to
different folders, `project_root` wins and a warning is logged:

```text
PipelineDAGCompiler received both project_root (...) and anchor_file (-> ...)
that disagree; using project_root.
```

Because step 3 always produces *something* usable from `config_path`, the caller
hook is effectively **always active** during compilation — even when you pass no
anchor at all, the compiler infers one and pushes it. Passing `anchor_file`
simply makes the anchor explicit and correct instead of inferred.

### CLI and factory surfaces

The exec-doc CLI exposes the same anchor through flags (see
`cli/exec_doc_cli.py`):

```bash
cursus exec-doc generate --project-root /abs/path/to/project ...
# or
cursus exec-doc generate --anchor-file /abs/path/to/project/run_pipeline.py ...
```

`--anchor-file` accepts a file and uses its parent directory as the project root —
the CLI mirror of `anchor_file=__file__`. See the [CLI reference](../cli.rst).

## How a config actually resolves its `source_dir`

Path resolution reaches the resolver through the config layer. On
`BasePipelineConfig` (`config_base.py`):

- `resolve_hybrid_path(relative_path)` is the bridge to
  `hybrid_path_resolution.resolve_hybrid_path(self.project_root_folder, relative_path)`.
- `resolved_source_dir` and `effective_source_dir` call `resolve_hybrid_path`
  with `self.source_dir`.
- Processing configs (`config_processing_step_base.py`) override
  `effective_source_dir` to resolve `processing_source_dir` first, then fall back
  to `source_dir`.

The key gate lives inside `resolve_hybrid_path`:

```python
# from config_base.py
if not self.project_root_folder and not get_project_root():
    logger.debug(
        "No project_root_folder and no pushed project root for hybrid resolution"
    )
    return None

return resolve_hybrid_path(self.project_root_folder, relative_path)
```

Resolution proceeds when **either** `project_root_folder` is set on the config
**or** a project root was pushed by the caller hook. That `or` is the whole point:

- **Before the caller hook**, a config *had to* declare `project_root_folder`
  (a required field on `BasePipelineConfig` in `config_base.py`, whose own
  description reads "required for hybrid resolution") so Strategies 0b/1/2/3 had a
  folder name to search for, and often you also needed `CURSUS_PROJECT_BASE` set
  for Strategy 0b.
- **With the caller hook**, the compiler pushes the project folder itself, so
  Strategy 0 resolves `project_root / source_dir` directly — no folder name and no
  env var required.

## What the caller hook makes optional

Two things that used to be mandatory become optional once the project root is
pushed:

### `CURSUS_PROJECT_BASE` (the env var)

`CURSUS_PROJECT_BASE` drives **Strategy 0b**. It is a *package base* directory
under which a named `project_root_folder` sibling is searched
(`base / project_root_folder / relative_path`). It typically had to be exported by
a consumer package's `__init__.py` at import time.

The pushed project root is different: it is the **project folder itself**, so the
join is `project_root / relative_path` with no intermediate folder name. When
Strategy 0 succeeds, Strategy 0b is never consulted, so `CURSUS_PROJECT_BASE` no
longer needs to be set.

### `project_root_folder` (the config field)

`project_root_folder` is the folder *name* that Strategies 0b/1/2/3 search for. It
still helps those fallbacks, but Strategy 0 does not use it — it anchors on the
pushed folder directly. As shown above, `resolve_hybrid_path` now proceeds when a
root has been pushed even if `project_root_folder` is empty. A config can omit it
and still resolve.

Net effect: a pipeline that passes `anchor_file=__file__` to its compiler needs
**neither** an environment variable **nor** a `project_root_folder` on every
config to get correct `source_dir` resolution.

## Anchoring step packs

The same pushed project root also anchors **external step packs** — a consumer's
own steps (`interfaces/*.step.yaml` + `configs/` + `scripts/`) living in a folder
*outside* the pip-installed `cursus` package. `PipelineDAGCompiler` accepts an
explicit `workspace_dirs`, but when you omit it the compiler **derives a pack from
the resolved project root**:

`_derive_step_pack_dir(project_root)` in `dag_compiler.py` looks for an
`interfaces/` directory under `<project_root>/step_pack` first, then under
`<project_root>` itself, and uses the first one that exists. So the same
`anchor_file=__file__` that fixes `source_dir` resolution also tells Cursus where
to find the project's own step definitions:

```python
compiler = PipelineDAGCompiler(
    config_path="pipeline_config/config.json",
    anchor_file=__file__,   # anchors BOTH source_dir resolution AND step-pack discovery
)
```

Step packs are strictly additive — the built-in package steps are always
available and are never removed; a pack step that clashes on name shadows with a
warning. See [Registry and discovery](../concepts/registry_and_discovery.md) and
the [Step catalog reference](../reference/generated/step_catalog.md) for how those
steps become visible.

## End-to-end example

A typical project entry module — the one that defines `generate_pipeline()` and
therefore lives inside the project folder:

```python
# /my_project/run_pipeline.py
from pathlib import Path
from cursus.core.compiler.dag_compiler import PipelineDAGCompiler
from cursus.api.dag import PipelineDAG


def generate_pipeline(session, role):
    dag = PipelineDAG()
    # ... add nodes/edges ...

    compiler = PipelineDAGCompiler(
        config_path=str(Path(__file__).parent / "pipeline_config" / "config.json"),
        sagemaker_session=session,
        role=role,
        anchor_file=__file__,   # push /my_project as the project root
    )
    return compiler.compile(dag)
```

What happens on construction:

1. The compiler calls `_resolve_project_root(None, config_path, anchor_file=__file__)`.
   `anchor_file` resolves (via `resolve_anchor`) to `/my_project`.
2. The compiler calls `set_project_root("/my_project")`, pushing it process-wide.
3. During compile, each config's `effective_source_dir` calls
   `resolve_hybrid_path`, which proceeds because `get_project_root()` is set —
   even if a config has no `project_root_folder`.
4. Strategy 0 joins `/my_project / <source_dir>` (e.g.
   `/my_project/scripts`), confirms it exists, and returns it.
5. Because no `workspace_dirs` was passed, the compiler also checks
   `/my_project/step_pack/interfaces` and `/my_project/interfaces` for an external
   step pack to fold in.

No `CURSUS_PROJECT_BASE`, no per-config `project_root_folder`, and correct
resolution regardless of whether this runs from a monorepo, a notebook, or a
bundled Lambda.

## Precedence, summarized

Resolution honors two independent precedence chains — one that decides *what the
project root is*, and one that decides *how a relative path is joined to a base*.

**Choosing the project root** (in `_resolve_project_root`):

```
explicit project_root  >  anchor_file  >  inferred from config_path
```

**Resolving a relative path** (in `HybridPathResolver.resolve_path`):

```
Strategy 0  (pushed project root, the caller hook)
Strategy 0b (CURSUS_PROJECT_BASE + project_root_folder)
Strategy 1  (cursus package location)
Strategy 2  (working directory walk-up)
Strategy 3  (generic recursive folder search)
Strategy 4  (packaged cursus/steps/scripts)
```

The first that yields an existing path wins. In the common, well-configured case,
that is Strategy 0 — a single, deterministic `project_root / source_dir` join.

## See also

- [Configuration system](../concepts/config_system.md) — where `source_dir`,
  `processing_source_dir`, and `project_root_folder` live on configs.
- [Compilation](../concepts/dag_and_compilation.md) — how `PipelineDAGCompiler`
  turns a DAG plus configs into a pipeline.
- [Registry and discovery](../concepts/registry_and_discovery.md) — step-pack
  discovery anchored on the same project root.
- [CLI reference](../cli.rst) — the `exec-doc generate --project-root` /
  `--anchor-file` flags.

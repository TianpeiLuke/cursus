# Author a Custom Step

This tutorial walks you end-to-end through creating a brand-new Cursus step from
scratch. By the end you will have written the three artifacts a step needs — its
`<step>.step.yaml` interface, its Pydantic config class, and its processing script —
and validated the result with `cursus validate step-interface`.

The key idea to internalize first: **you do not write a builder class.** Under
Cursus's declarative step model, the interface *is* the registration. Writing the
`.step.yaml` derives the registry entry, and the builder is synthesized at runtime
from a shared facade. Three files in, and your step is discoverable, buildable, and
wireable into any DAG.

```{contents} On this page
:local:
:depth: 2
```

## What a step is made of

A Cursus step is authored as exactly **three files**, all sharing one snake_case
stem derived from the PascalCase step name:

| Artifact | Location | Purpose |
| --- | --- | --- |
| `<snake>.step.yaml` | `src/cursus/steps/interfaces/` | The **interface** — contract I/O + spec dependencies/outputs + routing metadata. Writing it registers the step. |
| `config_<snake>_step.py` | `src/cursus/steps/configs/` | The **config class** (`<StepName>Config`) — the Pydantic model that supplies runtime values. |
| `<snake>.py` | `src/cursus/steps/scripts/` | The **script** — the container entrypoint with the fixed `main(...)` signature. |

There is deliberately **no** `builder_<snake>_step.py`. The builder is a runtime
synthesis (see [How the builder is synthesized](#how-the-builder-is-synthesized)),
and there is no registry file to edit — `build_registry_from_interfaces()` derives
the `STEP_NAMES` table directly from the `.step.yaml` files.

For this tutorial we will build a small Processing step called **`FeatureFlagger`**
that reads a preprocessed dataset, adds a set of derived boolean flag columns, and
writes the augmented dataset back out. It sits between an upstream preprocessing step
and any downstream consumer — an `internal` node with one required input and one
output.

## The `.step.yaml` interface

Open `src/cursus/steps/interfaces/feature_flagger.step.yaml`. The interface is a
single Pydantic-validated document (`StepInterface`) with a handful of top-level
sections. Here is a complete, minimal-but-real interface for our step:

```yaml
step_type: FeatureFlagger
node_type: internal
registry:
  sagemaker_step_type: Processing
  description: Adds derived boolean flag columns to a tabular dataset
patterns:
  direct_input_keys: [processed_data]
compute:
  kind: sklearn
  framework_version_field: processing_framework_version
contract:
  entry_point: feature_flagger.py
  inputs:
    processed_data:
      path: /opt/ml/processing/input/data
      required: true
  outputs:
    flagged_data:
      path: /opt/ml/processing/output
  job_arguments:
  - flag: --job_type
    source: job_type
  env_vars:
    required: []
    optional:
      FLAG_COLUMNS: ''
      DROP_ORIGINAL: 'false'
  framework_requirements:
    pandas: '>=1.3.0'
  description: Reads a tabular dataset and appends boolean flag columns.
spec:
  dependencies:
    processed_data:
      type: processing_output
      required: true
      compatible_sources:
      - TabularPreprocessing
      - RiskTableMapping
      semantic_keywords:
      - data
      - processed
      - tabular
  outputs:
    flagged_data:
      type: processing_output
      property_path: properties.ProcessingOutputConfig.Outputs['flagged_data'].S3Output.S3Uri
      aliases:
      - input_data
      - feature_data
      semantic_keywords:
      - flagged
      - features
```

Every field above maps to a real Pydantic model in
`src/cursus/core/base/step_interface.py`. Let's break the sections down.

### Top-level fields

- **`step_type`** — the PascalCase step name. This string is simultaneously the
  `.step.yaml`'s identity *and* the canonical registry name — the registry is derived
  by construction, so there is no second place to register the name.
- **`node_type`** — one of the `NodeType` enum values in
  `src/cursus/core/base/enums.py`: `source` (no dependencies, has outputs),
  `internal` (both), `sink` (dependencies, no outputs), or `singular` (neither).
  Our step has one input and one output, so it is `internal`.

### `registry`

The `RegistrySection` selects how the step is built and discovered:

- **`sagemaker_step_type`** — the routing verb that selects the build handler. It
  must be one of a closed, validated set: `Processing`, `Training`, `Transform`,
  `CreateModel`, the SAIS-delegation verbs (`CradleDataLoading`,
  `RedshiftDataLoading`, `MimsModelRegistrationProcessing`), plus the no-builder
  rows (`Base`, `Lambda`, `RegisterModel`, `Utility`). A typo here is caught at
  author time, not at build time.
- **`description`** — free text surfaced in the [step catalog](../reference/generated/step_catalog.md).
- **`config_class`** / **`builder_step_name`** *(optional)* — you almost never set
  these. By convention the config class is `<StepName>Config` and the builder name is
  `<StepName>StepBuilder`; the registry loader fills both in automatically. Only
  override `config_class` if your config class breaks the naming convention.

### `compute`

The `ComputeSpec` describes the SageMaker compute object (processor / estimator /
model / transformer) the synthesized builder constructs, so you don't hand-write a
`_create_processor` factory. For a scikit-learn based `ScriptProcessor`, declaring
the framework and the config field that holds its version is enough:

```yaml
compute:
  kind: sklearn
  framework_version_field: processing_framework_version
```

Valid `kind` values are `sklearn`, `xgboost`, `framework`, `script`, `estimator`,
`model`, and `transformer`. The `framework`, `estimator`, and `model` kinds
additionally require an `sdk_class` (one of `PyTorch`, `SKLearn`, `XGBoost`,
`PyTorchModel`, `XGBoostModel`). Leaving `compute` empty (`kind` unset) means the
step keeps its own factory — for a new step you almost always want to declare it.

### `patterns`

The `PatternsSection` holds per-axis strategy knobs read by the builder facade:

- **`step_assembly`** *(Processing only)* — `code` (a single-file
  `ProcessingStep(code=...)`, the default), `step_args` (a `FrameworkProcessor`
  `processor.run()`), or `delegation` (SDK delegation).
- **`direct_input_keys`** — logical input names passed straight through to the
  processor rather than resolved through the spec×contract join.
- **`include_job_type_in_path`** — whether `config.job_type` is a segment of the
  synthesized output S3 destination (default `True`).

### `contract` — the script's I/O

The `ContractSection` is the drop-in for the legacy `ScriptContract`. It declares
what the *script* sees inside the container:

- **`entry_point`** — the script filename (must end in `.py`).
- **`inputs`** — a map of logical name → `{path, required}`. The `path` must start
  with a valid SageMaker input prefix: `/opt/ml/processing/`, `/opt/ml/input/data`,
  `/opt/ml/input/config`, or `/opt/ml/code`.
- **`outputs`** — a map of logical name → `{path}`. The `path` must start with
  `/opt/ml/processing/`, `/opt/ml/model`, `/opt/ml/output/data`, or
  `/opt/ml/checkpoints`.
- **`job_arguments`** — a *declarative* record of the CLI flags the script accepts,
  each `{flag, source}`. This documents the argument surface and drives the reverse
  script check; the actual values come from `config.get_job_arguments()` at build
  time.
- **`env_vars`** — `required` (a list of names) and `optional` (a map of name →
  default). The interface declares *which* env vars the step uses; the config
  supplies the *values*.
- **`framework_requirements`** — pinned pip requirements for documentation and
  alignment.

### `spec` — dependency resolution metadata

The `SpecSection` is the drop-in for the legacy `StepSpecification`. It tells the
dependency resolver how to wire this step into a DAG:

- **`dependencies`** — for each input, a `type` (a `DependencyType` such as
  `processing_output`, `training_data`, `model_artifacts`, `hyperparameters`), a
  `required` flag, `compatible_sources` (the **exact, case-sensitive** upstream step
  names this input can wire from), and `semantic_keywords` used for fuzzy matching.
- **`outputs`** — for each output, a `type`, a SageMaker `property_path` (how the
  runtime reads the produced S3 URI off the step's properties), optional `aliases`
  (alternate names a downstream dependency may match on), and `semantic_keywords`.

```{admonition} The one alignment invariant to remember
:class: important
Every key in `contract.inputs` **must** have a matching key in
`spec.dependencies`, and every key in `contract.outputs` **must** have a matching
key in `spec.outputs`. This is enforced at load time by `StepInterface._sync_and_align`
— if the keys drift, the interface fails to parse. This replaced the old separate
"Level-2" alignment check with a hard Pydantic invariant.
```

```{admonition} compatible_sources is case-sensitive
:class: warning
`spec.dependencies.*.compatible_sources` must use the **exact** upstream step name.
A case typo (e.g. `Tabularpreprocessing`) silently loses the resolver's match bonus
— the edge may still resolve by keywords, but weakly. `cursus validate step-interface`
flags likely case typos as warnings.
```

### Job-type variants (optional)

If your step needs different dependencies or outputs per job type (e.g. `training`
vs `calibration`), declare a `variants:` block. Each entry is a partial override
that is *deep-merged* over the base sections when a builder requests that job type
via `load_interface(step_name, job_type=...)`. A variant that restates only a subset
of ports overrides just those ports and preserves the rest. Our simple step doesn't
need variants.

## The config class

Open `src/cursus/steps/configs/config_feature_flagger_step.py`. The config class
must be named `<StepName>Config` and follows the **three-tier field design**:

1. **Tier 1 — Essential user inputs.** Required fields, no default.
2. **Tier 2 — System fields.** Sensible defaults the user may override.
3. **Tier 3 — Derived fields.** Private attributes exposed via read-only properties.

For a Processing step, inherit from `ProcessingStepConfigBase` (which supplies
`processing_framework_version`, source-dir resolution, and the shared env/arg
plumbing):

```python
from pydantic import Field, field_validator
from typing import Any, Dict, List, Optional

from .config_processing_step_base import ProcessingStepConfigBase


class FeatureFlaggerConfig(ProcessingStepConfigBase):
    """Configuration for the FeatureFlagger step (three-tier field design)."""

    # ===== Tier 1: Essential User Inputs =====
    job_type: str = Field(
        description="One of ['training','validation','testing','calibration']",
    )
    flag_columns: List[str] = Field(
        description="Columns to derive boolean flags from.",
    )

    # ===== Tier 2: System Fields with Defaults =====
    processing_entry_point: str = Field(
        default="feature_flagger.py",
        description="Relative path (within the source dir) to the script.",
    )
    drop_original: bool = Field(
        default=False,
        description="Drop the source columns after deriving flags.",
    )

    # ===== Validators =====
    @field_validator("job_type")
    @classmethod
    def validate_job_type(cls, v: str) -> str:
        if not v.replace("_", "").isalnum() or v != v.lower():
            raise ValueError(
                f"job_type must be lowercase alphanumeric, got '{v}'"
            )
        return v

    # ===== Value collectors the builder facade reads =====
    def get_environment_variables(self, declared_names=None) -> Dict[str, str]:
        """Supply values for the env vars the interface declares."""
        return {
            "FLAG_COLUMNS": ",".join(self.flag_columns),
            "DROP_ORIGINAL": "true" if self.drop_original else "false",
        }

    def get_job_arguments(self) -> Optional[List[str]]:
        """CLI args — config is the single source of truth."""
        return ["--job_type", self.job_type]
```

A few things worth calling out, all grounded in how the synthesized builder consumes
the config:

- **`get_environment_variables`** is how the interface's declared `env_vars` names
  get their runtime values. `builder_base._get_environment_variables` prefers a
  config-owned `get_environment_variables` collector over the inherited generic
  resolver, so your bespoke logic is never bypassed. If you *don't* define one, the
  base resolver maps a declared name `FOO` to `self.foo` by convention.
- **`get_job_arguments`** returns the exact CLI argument list. This is the single
  source for the `--job_type` value the script parses — the interface's
  `job_arguments` block only *declares* the flag.
- **Field validators** enforce closed value sets. When you later validate config
  *values*, the `author.preflight_config` MCP tool and
  `TabularPreprocessingConfig`-style validators catch wrong enum casing and missing
  required fields.

## The processing script

Open `src/cursus/steps/scripts/feature_flagger.py`. Every Cursus script exposes the
**standard `main` signature** so the framework (and the local script-testing harness)
can drive it uniformly:

```python
def main(input_paths, output_paths, environ_vars, job_args):
    ...
```

The four parameters are, in order: a dict of logical input name → container path, a
dict of logical output name → container path, a dict of environment variables, and
the parsed `argparse.Namespace`. Those four names are the **required prefix**
`ScriptAnalyzer.validate_main_function_signature` checks for; a trailing *optional*
parameter such as `logger=None` is allowed and used by several shipped scripts —
the exemplar `tabular_preprocessing.py`, for instance, declares
`def main(input_paths, output_paths, environ_vars, job_args, logger=None)`. A
`__main__` block wires the real SageMaker container paths and environment into that
signature. Here is a complete skeleton that mirrors the shape of the shipped
`tabular_preprocessing.py`:

```python
import argparse
import os
import sys
from pathlib import Path

import pandas as pd


def main(input_paths, output_paths, environ_vars, job_args):
    """Append boolean flag columns to a tabular dataset."""
    input_dir = input_paths["processed_data"]
    output_dir = output_paths["flagged_data"]

    flag_columns = [c for c in environ_vars.get("FLAG_COLUMNS", "").split(",") if c]
    drop_original = environ_vars.get("DROP_ORIGINAL", "false").lower() == "true"
    job_type = job_args.job_type  # declared as --job_type in the interface

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(next(Path(input_dir).glob("*.csv")))
    for col in flag_columns:
        df[f"{col}_flag"] = df[col].notna()
    if drop_original:
        df = df.drop(columns=flag_columns)

    df.to_csv(Path(output_dir) / f"{job_type}_flagged.csv", index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_type", type=str, required=True)
    args = parser.parse_args()

    input_paths = {"processed_data": "/opt/ml/processing/input/data"}
    output_paths = {"flagged_data": "/opt/ml/processing/output"}
    environ_vars = {
        "FLAG_COLUMNS": os.environ.get("FLAG_COLUMNS", ""),
        "DROP_ORIGINAL": os.environ.get("DROP_ORIGINAL", "false"),
    }

    try:
        main(input_paths, output_paths, environ_vars, args)
        sys.exit(0)
    except Exception as e:  # surface a nonzero exit so the step fails loudly
        print(f"Error in feature_flagger: {e}")
        sys.exit(1)
```

```{admonition} Script ↔ contract must agree in both directions
:class: tip
- Every flag in `contract.job_arguments` must be parsed by an
  `parser.add_argument(...)` in the script (`--job_type` here).
- Every `env_vars.required` name must actually be *read* in `main()`.
- The container paths in the `__main__` block must match the `contract.inputs` /
  `contract.outputs` paths.

The `author.check_script` MCP tool checks all of this, in both directions, offline.
```

```{admonition} Keep the SAIS install preamble if your script has one
:class: note
Scripts that carry the secure-PyPI install preamble (the `USE_SECURE_PYPI` /
`CA_REPOSITORY_ARN` / CodeArtifact block) must keep it — it is load-bearing. Never
strip it when copying an exemplar script.
```

## How the builder is synthesized

You wrote no builder class — so how does the step get built? At build time the
`PipelineAssembler` resolves your step's builder class through the step catalog
(`get_builder_for_config`), and for a step with no physical `builder_*.py` that class
is a runtime-**fabricated** `TemplateStepBuilder` subclass — the discovery layer
synthesizes `class FeatureFlaggerStepBuilder(TemplateStepBuilder): STEP_NAME =
"FeatureFlagger"` on the fly. `TemplateStepBuilder`
(`src/cursus/core/base/builder_templates.py`) is the single facade that replaces the
per-step builder shells. Once instantiated with your `config`, it:

1. Loads your step's `StepInterface` from the `.step.yaml` (via
   `load_step_interface`), passing through `config.job_type` so variant-bearing
   steps resolve their job-typed spec.
2. Binds a construction handler in `_auto_bind_handler`, which calls
   `resolve_handler(sagemaker_step_type, step_assembly, knobs)` — the
   `sagemaker_step_type` comes from the registry (derived from your `registry:`
   block) and the `step_assembly` + knobs come from your `patterns:` section. For our
   step the bound handler is the `ProcessingHandler`. The five handlers
   (`ProcessingHandler`, `TrainingHandler`, `ModelCreationHandler`,
   `TransformHandler`, `SDKDelegationHandler`) cover all the SageMaker verbs.
3. Delegates the abstract `_get_inputs` / `_get_outputs` / `create_step` methods to
   that handler, which reads your `compute`, `contract`, and `patterns` sections plus
   the config to construct the processor, wire the inputs/outputs, and emit the
   `ProcessingStep`.

Because the registry (`STEP_NAMES`) is derived from the interface files by
`build_registry_from_interfaces()`, and the config class is discovered by the
`<StepName>Config` convention, the moment your three files exist and validate, the
step is a first-class, buildable citizen — no builder export, no registry edit.

## Validate with the CLI

The author-time gate is `cursus validate step-interface`. It loads your interface
through the exact production `StepInterface.from_yaml` path — surfacing Pydantic
field errors and the contract↔spec alignment check — then runs the non-blocking
incompleteness checks (like `compatible_sources` case typos).

```bash
# Validate one step
cursus validate step-interface FeatureFlagger

# Resolve and validate a job_type variant
cursus validate step-interface FeatureFlagger --job-type calibration

# Validate every .step.yaml (what CI runs)
cursus validate step-interface --all

# Machine-readable output
cursus validate step-interface FeatureFlagger --format json
```

A clean run prints a ✅ per step and exits `0`. Blocking errors (a missing spec
dependency for a contract input, an invalid `sagemaker_step_type`, a bad SageMaker
path prefix) print `ERROR:` lines and exit `1`. Warnings (`warn:`) don't fail the
command but are worth fixing. See the full [CLI reference](../cli.rst) for the other
`validate` subcommands, including `run-scripts` for local end-to-end script testing.

## The guided path: `author.*` tools and the workflow

Everything above can be driven manually, but Cursus ships a guided authoring path
built for agents (and useful as a checklist for humans). It centers on the
`author.*` MCP namespace and the `cursus-author-step` workflow.

### The `author.*` MCP tools

Defined in `src/cursus/mcp/tools/author.py`, these are read-only, offline-safe, and
compose the same engines the CLI and CI use — so their guidance can never drift from
what the build enforces:

| Tool | What it gives you |
| --- | --- |
| `author.checklist` | The ordered author→validate→integrate SOP for a given `sagemaker_step_type`, naming the exact tool to call at each step, plus the bound handler and a same-phase exemplar to copy shapes from. **Start here.** |
| `author.rules` | The restriction set for one topic (`naming`, `packaging`, `sdk_carveout`, `reuse_class`, `closure`), read live off the enforcement objects. |
| `author.config_constraints` | A config class's fields *with* their allowed values, case-sensitivity, and which required fields have no default. |
| `author.preflight_config` | Validates a concrete `{field: value}` set against the live config class — catches wrong enum case, invalid enum, wrong type, missing required fields. |
| `author.check_script` | Checks the script against its contract in both directions (main signature, declared args parsed, required env vars read). |
| `author.preflight_step` | Proves the step is **constructible** (not merely parseable) — the same four gates CI runs: interface validation, registry parity, `RegistryBindingValidator` B3 (handler binds + builder synthesizes + config-field coverage), and `resolve_strategy` routability. |

A typical loop: `author.checklist` → `author.rules` → write the three files →
`author.config_constraints` / `author.preflight_config` for the config values →
`cursus validate step-interface` → `author.check_script` → `author.preflight_step`.
See the generated [MCP tools reference](../reference/generated/mcp_tools.md) for the
full schema of each tool.

### The `cursus-author-step` workflow

The `cursus-author-step` workflow orchestrates that loop as non-skippable phases:
**Resolve** (locate the new node between its producer and consumer, bind the handler
and exemplar) → **Challenge** (skeptically confirm a new step is actually needed) →
**AlignEdges** (align the new dependency-spec to the producer's output-spec and the
new output-spec to the consumer's dependency-spec, refusing a resolution score below
0.5) → **Guide** → **Author** (write the three artifacts by exemplar-plus-required-
divergences, and edit the producer/consumer specs so they accept the new step) →
**Validate** → **Preflight** → **Gaps** → **Synthesize**. It is the recommended path
when the new step must slot into an *existing* DAG, because it verifies both DAG edges
resolve, not just that the interface parses in isolation.

## Recap and next steps

You created a step with three files and zero builder code:

1. **`feature_flagger.step.yaml`** — the interface; writing it *is* the registration.
2. **`FeatureFlaggerConfig`** — the Pydantic config supplying runtime values.
3. **`feature_flagger.py`** — the script with the fixed `main(input_paths,
   output_paths, environ_vars, job_args)` signature.

You then validated with `cursus validate step-interface FeatureFlagger`, and the
builder is synthesized at runtime by `TemplateStepBuilder` + the bound handler.

From here:

- Add the step to a DAG and compile it — see the [concepts overview](../concepts/index.md)
  for how the DAG resolver wires steps together.
- Browse the [step catalog](../reference/generated/step_catalog.md) for exemplars to
  copy section shapes from, and the [pipeline catalog](../reference/generated/pipeline_catalog.md)
  for ready-made pipelines to drop your step into.
- Explore the full [API reference](../api/index.rst) for `StepInterface`,
  `TemplateStepBuilder`, and the config base classes.

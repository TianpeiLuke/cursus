# Cursus: Automatic SageMaker Pipeline Generation

[![PyPI version](https://badge.fury.io/py/cursus.svg)](https://badge.fury.io/py/cursus)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/cursus/badge/?version=latest)](https://cursus.readthedocs.io/en/latest/)

**Turn a pipeline graph plus a JSON config into a complete, production-ready SageMaker pipeline — automatically.**

Cursus is a specification-driven pipeline generation system for Amazon SageMaker. You describe your ML workflow as a **DAG of step names**; Cursus resolves the dependencies between steps, wires their inputs and outputs, looks up each step's declarative interface, and assembles a runnable `sagemaker.workflow.pipeline.Pipeline`. You say *what* the pipeline is — Cursus figures out *how* to build it.

> **📖 Full documentation:** **[cursus.readthedocs.io](https://cursus.readthedocs.io/en/latest/)** (mirror: [tianpeiluke.github.io/cursus](https://tianpeiluke.github.io/cursus/)) — getting-started, tutorials, concepts & architecture, and the complete API / CLI / MCP / step-catalog reference.

---

## Installation

```bash
pip install cursus
```

Requires **Python 3.9+** and targets the **SageMaker Python SDK 2.x** (`sagemaker>=2.248.0,<3`).

Optional extras keep heavy ML/data libraries out of the core install — pull in only what your steps run:

```bash
pip install "cursus[processing]"   # pandas / numpy data-processing utilities
pip install "cursus[pytorch]"      # PyTorch / Lightning
pip install "cursus[gbm]"          # XGBoost / LightGBM
pip install "cursus[nlp]"          # tokenizers / transformers
pip install "cursus[mcp]"          # MCP server for LLM agents (mcp + anyio)
pip install "cursus[all]"          # everything
```

Verify:

```bash
cursus --version
python -c "import cursus; print(cursus.__version__)"
```

---

## Quick Start

There are three ways in, highest-level first. They all compile through the same engine, so the same DAG + config produces the same pipeline.

Every path needs two ingredients: a **DAG** (step names + edges) and a **config JSON** whose `metadata.config_types` maps each node to a configuration class. Compilation is offline; you only need AWS credentials to *deploy* (`--upsert`) or *run* (`--start`).

### 1. Start from the pre-built pipeline catalog

Cursus ships **44 validated DAGs** across 8 frameworks. Let the router recommend one and build it:

```python
from cursus.pipeline_catalog import recommend_dag, load_shared_dag
from cursus import PipelineDAGCompiler

# recommend_dag returns a ranked list of matches (dicts with 'id', 'score', ...)
recommendations = recommend_dag(framework="xgboost", task_type="end_to_end")
dag = load_shared_dag(recommendations[0]["id"])

pipeline, report = PipelineDAGCompiler(config_path="config.json").compile_with_report(dag)
print(pipeline.name, "-", len(pipeline.steps), "steps")
```

### 2. Compile from the command line

Reproducible, no-glue path — point it at a DAG JSON and a config JSON:

```bash
# compile only (writes the SageMaker pipeline definition to a file)
cursus compile -d my_dag.json -c my_config.json -o pipeline.json

# validate DAG <-> config alignment without compiling
cursus compile -d my_dag.json -c my_config.json --validate-only

# compile, deploy to SageMaker, and start an execution
cursus compile -d my_dag.json -c my_config.json \
    --upsert --start --role arn:aws:iam::123456789012:role/MySageMakerRole
```

### 3. Build a DAG in Python

```python
from cursus.api import PipelineDAG
from cursus.core import compile_dag_to_pipeline

# Nodes are step names; edges are data dependencies
dag = PipelineDAG()
for node in ["CradleDataLoading", "TabularPreprocessing", "XGBoostTraining"]:
    dag.add_node(node)
dag.add_edge("CradleDataLoading", "TabularPreprocessing")
dag.add_edge("TabularPreprocessing", "XGBoostTraining")

# config.json maps each node -> a config class (metadata.config_types)
pipeline = compile_dag_to_pipeline(dag=dag, config_path="config.json")

# Deploy / run when ready
pipeline.upsert(role_arn="arn:aws:iam::123456789012:role/MySageMakerRole")
pipeline.start()
```

See the [Quickstart guide](https://cursus.readthedocs.io/en/latest/getting_started/quickstart.html) for the full walkthrough.

---

## Key Features

- **🎯 Graph-to-pipeline automation** — a DAG of step names compiles straight to a SageMaker pipeline; the SageMaker step objects, wiring, and naming are generated for you.
- **🧠 Intelligent dependency resolution** — Cursus infers step connections and data flow by matching each step's declared outputs to the next step's declared inputs (semantic scoring), instead of hand-wiring `properties` paths.
- **📄 Declarative, data-driven steps** — every step is a single `<step>.step.yaml` interface unifying the script contract (I/O, env vars, job arguments) and the dependency spec; step builders are synthesized at runtime, with no hand-written builder classes to maintain.
- **📦 A pre-built pipeline catalog** — 44 ready-to-use DAGs across XGBoost, PyTorch, LightGBM, Bedrock and more, discoverable by framework, task type, and complexity.
- **🧩 Extensible via step packs** — define your own steps in a folder *outside* the installed package and Cursus discovers them as native, strictly additively (built-in steps are never removed).
- **🛡️ Built-in validation** — DAG↔config alignment, interface conformance, and dependency resolution are checked before you deploy (`cursus validate`, `cursus alignment`, `--validate-only`).
- **🤖 Agent-ready (MCP)** — a framework-neutral, self-documenting tool surface of **70 JSON-in/JSON-out tools** across 12 namespaces mirrors the CLI/API for LLM agents. `pip install "cursus[mcp]"`, then `cursus-mcp` (or `python -m cursus.mcp.server` / `cursus mcp serve`); `cursus mcp help` to inspect the tools.

---

## How It Works

A DAG + config flows through layered subsystems to a SageMaker pipeline:

| Subsystem | Package | Responsibility |
|---|---|---|
| **DAG model** | `cursus.api.dag` | `PipelineDAG` — nodes (step names) + edges (dependencies) |
| **Compiler** | `cursus.core.compiler` | `PipelineDAGCompiler` / `compile_dag_to_pipeline` — validate → resolve → build → assemble |
| **Assembler** | `cursus.core.assembler` | Turns resolved steps into a `sagemaker` `Pipeline` |
| **Dependency resolver** | `cursus.core.deps` | Matches producer outputs to consumer inputs (semantic scoring) |
| **Step interfaces** | `cursus.core.base` + `cursus.steps.interfaces` | Declarative `<step>.step.yaml`; builders synthesized at runtime |
| **Registry & discovery** | `cursus.registry` + `cursus.step_catalog` | Canonical step table, derived interface-first; step-pack discovery |
| **Config system** | `cursus.core.config_fields` + `cursus.api.factory` | Pydantic config classes; `metadata.config_types` node→class map |
| **Pipeline catalog** | `cursus.pipeline_catalog` | Pre-built shared DAGs + router (`recommend_dag` / `load_shared_dag`) |
| **Validation** | `cursus.validation` | Alignment / interface / dependency checks |
| **Agent surface** | `cursus.mcp` | The MCP tool surface |
| **CLI** | `cursus.cli` | 13 command groups |

Read the [Architecture & Design](https://cursus.readthedocs.io/en/latest/concepts/index.html) docs for the full picture.

---

## What's Included

| | Count | |
|---|---|---|
| **Step types** | 57 registered (54 declarative `.step.yaml` interfaces) | data loading, preprocessing, training, eval, calibration, registration, … |
| **Pre-built DAGs** | 44 across 8 frameworks | XGBoost, PyTorch, LightGBM, LightGBM-MT, XGBoost-MT, Bedrock, TSA, Dummy |
| **CLI command groups** | 13 | `compile`, `dag`, `config`, `catalog`, `steps`, `strategies`, `pipeline-catalog`, `validate`, `alignment`, `registry`, `projects`, `exec-doc`, `mcp` |
| **MCP agent tools** | 70 across 12 namespaces | discover, construct, validate, compile, author — for LLM agents |

---

## Command-Line Interface

```bash
cursus compile -d dag.json -c config.json -o pipeline.json   # DAG + config -> pipeline
cursus compile -d dag.json -c config.json --validate-only    # dry-run alignment report
cursus pipeline-catalog recommend --framework xgboost        # discover a pre-built DAG
cursus pipeline-catalog get-dag xgboost_complete_e2e         # inspect a catalog DAG
cursus catalog list                                          # browse available step types
cursus steps io XGBoostTraining                              # a step's declared I/O
cursus dag resolve TabularPreprocessing XGBoostTraining      # score dependency edges
cursus validate step-interface --all                         # validate every interface
cursus projects list                                         # discover pipeline projects
cursus mcp help                                              # explore the agent tool surface
```

Every group, subcommand, and flag is in the [CLI reference](https://cursus.readthedocs.io/en/latest/cli.html).

---

## Installation Options

| Extra | Installs | For |
|---|---|---|
| *(core)* | DAG model, compiler, catalog, CLI, MCP | everything except heavy ML/data libs |
| `processing` | pandas, numpy | data-processing utilities & scripts |
| `pytorch` | torch, lightning | PyTorch training/eval steps |
| `gbm` | xgboost, lightgbm | gradient-boosting steps |
| `nlp` | tokenizers, transformers | text steps |
| `jupyter` | notebook tooling | interactive development |
| `viz` | plotting libraries | reports/visualizations |
| `docs` | sphinx, furo, sphinx-click, … | building this documentation |
| `dev` | test/lint toolchain | contributing |
| `all` | pytorch + gbm + nlp + processing + jupyter + viz | full runtime install |

```bash
pip install "cursus[all]"          # full ML runtime
pip install "cursus[dev]"          # contributor toolchain
```

---

## 📖 Documentation

### 🌐 [Hosted Documentation → cursus.readthedocs.io](https://cursus.readthedocs.io/en/latest/)
**The full, auto-generated documentation site** — rebuilt from the source on every release (also mirrored at [tianpeiluke.github.io/cursus](https://tianpeiluke.github.io/cursus/)):
[Getting Started](https://cursus.readthedocs.io/en/latest/getting_started/index.html) ·
[Tutorials](https://cursus.readthedocs.io/en/latest/tutorials/index.html) ·
[Concepts & Architecture](https://cursus.readthedocs.io/en/latest/concepts/index.html) ·
[How-to Guides](https://cursus.readthedocs.io/en/latest/guides/index.html) ·
[API Reference](https://cursus.readthedocs.io/en/latest/api/index.html) ·
[CLI Reference](https://cursus.readthedocs.io/en/latest/cli.html) ·
[MCP Tools](https://cursus.readthedocs.io/en/latest/reference/generated/mcp_tools.html) ·
[Step Catalog](https://cursus.readthedocs.io/en/latest/reference/generated/step_catalog.html) ·
[Pipeline Catalog](https://cursus.readthedocs.io/en/latest/reference/generated/pipeline_catalog.html)

### Design & developer notes (in-repo)
- **[Developer Guide](https://github.com/TianpeiLuke/cursus/tree/main/slipbox/0_developer_guide/README.md)** — developing new pipeline steps and extending Cursus
- **[Design Documentation](https://github.com/TianpeiLuke/cursus/tree/main/slipbox/1_design/README.md)** — architectural design docs and principles
- **[Pipeline Catalog](https://github.com/TianpeiLuke/cursus/tree/main/src/cursus/pipeline_catalog/README.md)** — the prebuilt-DAG collection
- **[Changelog](https://github.com/TianpeiLuke/cursus/blob/main/CHANGELOG.md)** — release history

---

## Who Should Use Cursus?

- **Data scientists & ML practitioners** — go from a workflow sketch to a running SageMaker pipeline without writing SageMaker step glue; start from a catalog template and customize.
- **Platform & ML engineers** — standardize pipeline construction, enforce DAG↔config alignment in CI, and extend the step library with your own step packs.
- **Organizations** — a consistent, validated, reproducible path from graph to production pipeline, with less bespoke SageMaker code to maintain.

---

## 🤝 Contributing

Contributions are welcome! See the [Developer Guide](https://github.com/TianpeiLuke/cursus/tree/main/slipbox/0_developer_guide/README.md) for:

- **[Prerequisites](https://github.com/TianpeiLuke/cursus/tree/main/slipbox/0_developer_guide/prerequisites.md)** — what you need before starting
- **[Creation Process](https://github.com/TianpeiLuke/cursus/tree/main/slipbox/0_developer_guide/creation_process.md)** — adding a new pipeline step
- **[Validation Checklist](https://github.com/TianpeiLuke/cursus/tree/main/slipbox/0_developer_guide/validation_checklist.md)** — validating an implementation
- **[Common Pitfalls](https://github.com/TianpeiLuke/cursus/tree/main/slipbox/0_developer_guide/common_pitfalls.md)** — mistakes to avoid

Or author a step the guided way with the [`cursus mcp`](https://cursus.readthedocs.io/en/latest/tutorials/author_a_step.html) agent tools.

## 📄 License

Licensed under the MIT License — see [LICENSE](https://github.com/TianpeiLuke/cursus/blob/main/LICENSE).

## 🔗 Links

- **Documentation**: https://cursus.readthedocs.io/ (mirror: https://tianpeiluke.github.io/cursus/)
- **GitHub**: https://github.com/TianpeiLuke/cursus
- **PyPI**: https://pypi.org/project/cursus/
- **Issues**: https://github.com/TianpeiLuke/cursus/issues
- **Changelog**: https://github.com/TianpeiLuke/cursus/blob/main/CHANGELOG.md

# Getting Started

**Cursus** turns a pipeline *graph* (a DAG of step names) plus a JSON *configuration*
into a complete, production-ready **Amazon SageMaker pipeline** — resolving the
dependencies between steps, wiring their inputs/outputs, and generating the SageMaker
step objects for you. You describe *what* the pipeline is; Cursus figures out *how* to
build it.

This section takes you from `pip install` to a compiled pipeline in a few minutes.

```{toctree}
:maxdepth: 1

installation
quickstart
core_concepts
```

## The 30-second picture

There are three ways in, from highest-level to lowest:

| Path | You provide | Best for |
|---|---|---|
| **Pipeline catalog** | a framework/task + a config JSON | starting from a proven, pre-built pipeline (44 shipped DAGs) |
| **CLI** | a DAG JSON + a config JSON | reproducible builds, CI, no Python glue |
| **Python API** | a `PipelineDAG` + configs | full programmatic control / embedding in your own code |

All three converge on the same compiler, so they produce the same pipeline from the
same inputs.

## Fastest possible start

```bash
pip install cursus

# discover a pre-built pipeline and inspect what config it needs
cursus pipeline-catalog recommend --framework xgboost
cursus pipeline-catalog get-dag xgboost_complete_e2e

# compile a DAG + config into a SageMaker pipeline definition
cursus compile -d my_dag.json -c my_config.json -o pipeline.json
```

Then head to [Installation](installation.md) for the install options and
[Quickstart](quickstart.md) for the full walkthrough (catalog, CLI, and Python API).

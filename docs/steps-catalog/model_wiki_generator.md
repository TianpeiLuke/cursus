# ModelWikiGenerator

**Model wiki generator step for automated documentation creation**

| | |
|---|---|
| **SageMaker step type** | `Processing` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `model_wiki_generator.py` |
| **Interface file** | `steps/interfaces/model_wiki_generator.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `sklearn` |

## Functionality

Model wiki generator script. Loads metrics and visualizations, generates comprehensive multi-format model documentation (Wiki, HTML, Markdown).

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `metrics_output` | `processing_output` | yes | [ModelMetricsComputation](model_metrics_computation.md), [XGBoostModelEval](xgboost_model_eval.md), [XGBoostModelInference](xgboost_model_inference.md), [PyTorchModelInference](pytorch_model_inference.md) |
| `plots_output` | `processing_output` | no | [ModelMetricsComputation](model_metrics_computation.md), [XGBoostModelEval](xgboost_model_eval.md), [XGBoostModelInference](xgboost_model_inference.md), [PyTorchModelInference](pytorch_model_inference.md) |

## Outputs

| Output | Type |
|--------|------|
| `wiki_output` | `processing_output` |

## Consumers (downstream steps)

_No cataloged step lists this step as a compatible source (it may be a terminal/sink step, or consumed via a generic source name)._

## Framework requirements

| Package | Version |
|---------|---------|
| `jinja2` | `>=3.0.0` |
| `pandas` | `>=1.3.0` |

---

← [Back to the Step Catalog](index.md)

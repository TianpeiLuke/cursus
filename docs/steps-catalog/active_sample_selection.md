# ActiveSampleSelection

**Active sample selection step that intelligently selects high-value samples from model predictions for Semi-Supervised Learning (SSL) or Active Learning workflows using confidence-based, uncertainty-based, diversity-based, or hybrid strategies**

| | |
|---|---|
| **SageMaker step type** | `Processing` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `active_sample_selection.py` |
| **Interface file** | `steps/interfaces/active_sample_selection.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `sklearn` |

## Functionality

Active sample selection script. Intelligently selects high-value samples from model predictions for SSL or Active Learning workflows.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `evaluation_data` | `processing_output` | yes | [XGBoostModelInference](xgboost_model_inference.md), [LightGBMModelInference](lightgbm_model_inference.md), [PyTorchModelInference](pytorch_model_inference.md), [XGBoostModelEval](xgboost_model_eval.md), [LightGBMModelEval](lightgbm_model_eval.md), [PyTorchModelEval](pytorch_model_eval.md), [BedrockBatchProcessing](bedrock_batch_processing.md), [BedrockProcessing](bedrock_processing.md), [LabelRulesetExecution](label_ruleset_execution.md) |

## Outputs

| Output | Type |
|--------|------|
| `selected_samples` | `processing_output` |
| `selection_metadata` | `processing_output` |

## Consumers (downstream steps)

Steps that declare this step as a compatible input source:

- [PseudoLabelMerge](pseudo_label_merge.md)

## Framework requirements

| Package | Version |
|---------|---------|
| `pandas` | `>=1.3.0` |
| `numpy` | `>=1.21.0` |

---

← [Back to the Step Catalog](index.md)

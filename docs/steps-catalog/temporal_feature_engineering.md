# TemporalFeatureEngineering

**Temporal feature engineering step that extracts comprehensive temporal features from normalized sequences for machine learning models**

| | |
|---|---|
| **SageMaker step type** | `Processing` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `temporal_feature_engineering.py` |
| **Interface file** | `steps/interfaces/temporal_feature_engineering.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `sklearn` |

## Functionality

Temporal feature engineering script. Extracts comprehensive temporal features from normalized sequence data combining generic temporal features with time window aggregations.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `normalized_sequences` | `processing_output` | yes | [TemporalSequenceNormalization](temporal_sequence_normalization.md), ProcessingStep |

## Outputs

| Output | Type |
|--------|------|
| `temporal_feature_tensors` | `processing_output` |

## Consumers (downstream steps)

Steps that declare this step as a compatible input source:

- [BedrockBatchProcessing](bedrock_batch_processing.md)
- [BedrockProcessing](bedrock_processing.md)
- [LabelRulesetExecution](label_ruleset_execution.md)
- [PseudoLabelMerge](pseudo_label_merge.md)
- [TSATraining](tsa_training.md)

## Framework requirements

| Package | Version |
|---------|---------|
| `pandas` | `>=1.3.0` |
| `numpy` | `>=1.21.0` |

---

← [Back to the Step Catalog](index.md)

# TemporalSequenceNormalization

**Temporal sequence normalization step for machine learning models with configurable sequence operations**

| | |
|---|---|
| **SageMaker step type** | `Processing` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `temporal_sequence_normalization.py` |
| **Interface file** | `steps/interfaces/temporal_sequence_normalization.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `sklearn` |

## Functionality

Temporal sequence normalization script. Handles temporal sequence data loading, validation, normalization, and padding/truncation for ML models.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `DATA` | `processing_output` | yes | [CradleDataLoading](cradle_data_loading.md), [DummyDataLoading](dummy_data_loading.md), DataLoad, ProcessingStep, [TabularPreprocessing](tabular_preprocessing.md) |
| `SIGNATURE` | `processing_output` | no | [CradleDataLoading](cradle_data_loading.md), [DummyDataLoading](dummy_data_loading.md) |

## Outputs

| Output | Type |
|--------|------|
| `normalized_sequences` | `processing_output` |

## Consumers (downstream steps)

Steps that declare this step as a compatible input source:

- [BedrockBatchProcessing](bedrock_batch_processing.md)
- [BedrockProcessing](bedrock_processing.md)
- [LabelRulesetExecution](label_ruleset_execution.md)
- [PseudoLabelMerge](pseudo_label_merge.md)
- [TSATraining](tsa_training.md)
- [TemporalFeatureEngineering](temporal_feature_engineering.md)

## Framework requirements

| Package | Version |
|---------|---------|
| `pandas` | `>=1.3.0` |
| `numpy` | `>=1.21.0` |

---

← [Back to the Step Catalog](index.md)

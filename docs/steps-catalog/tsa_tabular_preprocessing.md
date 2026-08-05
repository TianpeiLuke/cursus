# TSATabularPreprocessing

**TSA (Temporal Self-Attention) tabular preprocessing with explicit output declarations for processed_data and preprocessor artifacts**

| | |
|---|---|
| **SageMaker step type** | `Processing` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `tsa_tabular_preprocessing.py` |
| **Interface file** | `steps/interfaces/tsa_tabular_preprocessing.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `framework` |
| **SDK class** | `PyTorch` (SageMaker DLC via `image_uris.retrieve`) |

## Functionality

TSA tabular preprocessing script that combines data shards, loads column signature, applies TSA-domain feature engineering (label encoding, ID field handling, date-based feature extraction), splits data into train/test/val, serialises the fitted sklearn preprocessor pipeline to preprocessor.pkl, and outputs both processed CSV and the preprocessor artifact. Supports streaming mode for large datasets.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `DATA` | `processing_output` | yes | [CradleDataLoading](cradle_data_loading.md), [DummyDataLoading](dummy_data_loading.md), [RedshiftDataLoading](redshift_data_loading.md), DataLoad, ProcessingStep, [BedrockProcessing](bedrock_processing.md), [StratifiedSampling](stratified_sampling.md) |
| `DATA_SECONDARY` | `processing_output` | no | [CradleDataLoading](cradle_data_loading.md), [DummyDataLoading](dummy_data_loading.md), [RedshiftDataLoading](redshift_data_loading.md), DataLoad, ProcessingStep, [BedrockProcessing](bedrock_processing.md), [StratifiedSampling](stratified_sampling.md) |
| `SIGNATURE` | `processing_output` | no | [CradleDataLoading](cradle_data_loading.md), [DummyDataLoading](dummy_data_loading.md) |

## Outputs

| Output | Type |
|--------|------|
| `tsa_processed_data` | `processing_output` |
| `preprocessor` | `processing_output` |

## Consumers (downstream steps)

Steps that declare this step as a compatible input source:

- [TSAModelCalibration](tsa_model_calibration.md)
- [TSAPreprocessing](tsa_preprocessing.md)

## Framework requirements

| Package | Version |
|---------|---------|
| `pandas` | `>=1.3.0` |
| `numpy` | `>=1.21.0` |
| `scikit-learn` | `>=1.0.0` |

---

← [Back to the Step Catalog](index.md)

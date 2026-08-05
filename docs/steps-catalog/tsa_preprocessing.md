# TSAPreprocessing

**TSA (Temporal Self-Attention) data preprocessing step that performs sequence processing with feature transformation and scaling for fraud detection models**

| | |
|---|---|
| **SageMaker step type** | `Processing` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `tsa_preprocessing.py` |
| **Interface file** | `steps/interfaces/tsa_preprocessing.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `framework` |
| **SDK class** | `PyTorch` (SageMaker DLC via `image_uris.retrieve`) |

## Functionality

TSA preprocessing script that performs CID sequence processing for fraud detection. It loads model artifacts (preprocessor, categorical mappings, default values, Python modules), loads and combines data from tabular preprocessing output, processes Customer ID (CID) sequences, applies feature transformations, scaling, and categorical encoding, handles time windowing and downsampling for different dataset types, and outputs numpy arrays for TSA model training. Inputs are artifacts (model artifacts read from /opt/ml/processing/input/code/artifacts), preprocessor (optional training-fitted scaling parameters, used via PREPROCESSOR_PATH when provided), and processed_data (tabular preprocessing output with train/test/val splits). Output is tsa_processed_data (5 numpy arrays per dataset - CID categorical sequences, CID numerical sequences, static features, labels, amounts). Supports streaming mode (ENABLE_TSA_STREAMING) for memory-efficient processing of large datasets.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `artifacts` | `processing_output` | no | [CradleDataLoading](cradle_data_loading.md), [DummyDataLoading](dummy_data_loading.md), DataLoad, ProcessingStep, [TabularPreprocessing](tabular_preprocessing.md) |
| `preprocessor` | `processing_output` | no | [TSATabularPreprocessing](tsa_tabular_preprocessing.md), ProcessingStep |
| `processed_data` | `processing_output` | yes | [TabularPreprocessing](tabular_preprocessing.md), [TSATabularPreprocessing](tsa_tabular_preprocessing.md), [CradleDataLoading](cradle_data_loading.md), [DummyDataLoading](dummy_data_loading.md), DataLoad, ProcessingStep |

## Outputs

| Output | Type |
|--------|------|
| `tsa_processed_data` | `processing_output` |

## Consumers (downstream steps)

Steps that declare this step as a compatible input source:

- [TSAModelCalibration](tsa_model_calibration.md)
- [TSAModelEval](tsa_model_eval.md)
- [TSATraining](tsa_training.md)

## Framework requirements

| Package | Version |
|---------|---------|
| `pandas` | `>=1.3.0` |
| `numpy` | `>=1.21.0` |

---

← [Back to the Step Catalog](index.md)

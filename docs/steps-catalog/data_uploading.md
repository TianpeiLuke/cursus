# DataUploading

**Upload processed data to BDT (EDX/Andes) via SAIS SDK delegation**

| | |
|---|---|
| **SageMaker step type** | `Processing` |
| **Node type** | sink (terminal — produces no pipeline outputs) |
| **Container entry point** | `scripts.py` |
| **Build-time requirement** | `secure_ai_sandbox_workflow_python_sdk` (SAIS SDK — fatal on load if absent) |
| **Interface file** | `steps/interfaces/data_uploading.step.yaml` |

## Functionality

SDK delegation step. Uploads S3 data to BDT (EDX/Andes). SINK node — no outputs. SDK DataUploadProcessor handles arguments internally.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `input_data` | `processing_output` | yes | [TabularPreprocessing](tabular_preprocessing.md), [StratifiedSampling](stratified_sampling.md), [XGBoostTraining](xgboost_training.md), [CradleDataLoading](cradle_data_loading.md), Processing |

## Outputs

_None — this is a sink step; it produces no downstream pipeline outputs._

## Consumers (downstream steps)

_No cataloged step lists this step as a compatible source (it may be a terminal/sink step, or consumed via a generic source name)._

## Framework requirements

| Package | Version |
|---------|---------|
| `python` | `>=3.7` |

---

← [Back to the Step Catalog](index.md)

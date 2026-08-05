# EdxUploading

**Upload S3 data to EDX via EdxDataLoader (SINK node, no Kale required)**

| | |
|---|---|
| **SageMaker step type** | `Processing` |
| **Node type** | sink (terminal — produces no pipeline outputs) |
| **Container entry point** | `edx_uploading.py` |
| **Interface file** | `steps/interfaces/edx_uploading.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `script` |
| **Network** | `kms_network` — the shared SAIS VpcConfig + volume KMS (script kind) |

## Functionality

EDX upload script. Uploads S3 data to EDX via EdxDataLoader. SINK node — data exits pipeline to EDX. No SageMaker outputs.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `input_data` | `processing_output` | yes | [CradleDataLoading](cradle_data_loading.md), [RedshiftDataLoading](redshift_data_loading.md), [TabularPreprocessing](tabular_preprocessing.md), [StratifiedSampling](stratified_sampling.md), [BedrockProcessing](bedrock_processing.md), ProcessingStep |

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

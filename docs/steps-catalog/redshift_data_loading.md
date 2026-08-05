# RedshiftDataLoading

**Redshift SQL data loading step (source node with optional EDX upload)**

| | |
|---|---|
| **SageMaker step type** | `RedshiftDataLoading` |
| **Node type** | source (no inputs — originates data) |
| **Container entry point** | `redshift_data_loading.py` |
| **Build-time requirement** | `secure_ai_sandbox_workflow_python_sdk` (SAIS SDK — fatal on load if absent) |
| **Interface file** | `steps/interfaces/redshift_data_loading.step.yaml` |

## Functionality

Redshift data loading script. Source node that executes SQL against Redshift and writes results as CSV to S3. Optionally uploads to EDX as side effect.

## Inputs (dependencies)

_None — this is a source step; it originates data with no pipeline inputs._

## Outputs

| Output | Type |
|--------|------|
| `output` | `processing_output` |

## Consumers (downstream steps)

Steps that declare this step as a compatible input source:

- [EdxUploading](edx_uploading.md)
- [TSATabularPreprocessing](tsa_tabular_preprocessing.md)
- [TabularPreprocessing](tabular_preprocessing.md)

## Framework requirements

| Package | Version |
|---------|---------|
| `boto3` | `>=1.26.0` |
| `pandas` | `>=1.2.0` |

---

← [Back to the Step Catalog](index.md)

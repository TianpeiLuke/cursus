# Registration

**Model registration step**

| | |
|---|---|
| **SageMaker step type** | `MimsModelRegistrationProcessing` |
| **Node type** | sink (terminal — produces no pipeline outputs) |
| **Container entry point** | `script.py` |
| **Build-time requirement** | `secure_ai_sandbox_workflow_python_sdk` (SAIS SDK — fatal on load if absent) |
| **Interface file** | `steps/interfaces/registration.step.yaml` |

## Functionality

MIMS model registration script that uploads model artifacts and payload samples, registers the model with MIMS service, tracks workflow execution ID, and cleans up temporary resources. No output files produced - registration is a side effect.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `PackagedModel` | `model_artifacts` | yes | PackagingStep, [Package](package.md), ProcessingStep |
| `GeneratedPayloadSamples` | `payload_samples` | yes | [Payload](payload.md), PayloadTestStep, PayloadStep, ProcessingStep |

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

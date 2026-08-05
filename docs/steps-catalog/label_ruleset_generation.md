# LabelRulesetGeneration

**Label ruleset generation step that validates and optimizes user-defined classification rules for transparent, maintainable rule-based label mapping in ML training pipelines**

| | |
|---|---|
| **SageMaker step type** | `Processing` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `label_ruleset_generation.py` |
| **Interface file** | `steps/interfaces/label_ruleset_generation.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `sklearn` |

## Functionality

Label ruleset generation script. Validates and optimizes user-defined classification rules for transparent, maintainable rule-based classification.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `ruleset_configs` | `processing_output` | no | RulesetConfiguration, ProcessingStep |

## Outputs

| Output | Type |
|--------|------|
| `validated_ruleset` | `processing_output` |
| `validation_report` | `processing_output` |

## Consumers (downstream steps)

Steps that declare this step as a compatible input source:

- [LabelRulesetExecution](label_ruleset_execution.md)

## Framework requirements

| Package | Version |
|---------|---------|
| `pydantic` | `>=2.0.0` |
| `pandas` | `>=1.3.0` |

---

← [Back to the Step Catalog](index.md)

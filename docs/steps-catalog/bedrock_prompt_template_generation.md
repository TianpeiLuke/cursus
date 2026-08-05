# BedrockPromptTemplateGeneration

**Bedrock prompt template generation step that assembles the prompt-config bundle (system prompt, category rules, output schema) into ONE standardized prompts.json prompt ruleset ({ruleset, rules} shape) with the output schema embedded in the prompt — the same contract the knowledge-routing producer emits, consumed by the Bedrock processing steps**

| | |
|---|---|
| **SageMaker step type** | `Processing` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `bedrock_prompt_template_generation.py` |
| **Interface file** | `steps/interfaces/bedrock_prompt_template_generation.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `sklearn` |

## Functionality

Bedrock prompt template generation script. Assembles the prompt-config bundle into ONE prompts.json prompt ruleset ({ruleset, rules}) with the output schema embedded in the prompt; a declarative meta-prompt assembler (validate blanks -> assemble), same downstream contract as the knowledge-routing producer.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `prompt_configs` | `processing_output` | no | PromptConfiguration, ProcessingStep |

## Outputs

| Output | Type |
|--------|------|
| `prompt_templates` | `processing_output` |
| `template_metadata` | `processing_output` |

## Consumers (downstream steps)

Steps that declare this step as a compatible input source:

- [BedrockBatchProcessing](bedrock_batch_processing.md)
- [BedrockProcessing](bedrock_processing.md)

## Framework requirements

| Package | Version |
|---------|---------|
| `pydantic` | `>=2.0.0` |
| `boto3` | `>=1.26.0` |

---

← [Back to the Step Catalog](index.md)

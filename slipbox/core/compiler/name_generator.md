---
tags:
  - code
  - pipeline_api
  - naming_convention
  - pipeline_generation
keywords:
  - name generator
  - pipeline
  - naming convention
  - versioning
topics:
  - pipeline API
  - pipeline naming
language: python
date of note: 2025-07-31
---

# Pipeline Name Generator

## Purpose

The `name_generator` module provides functionality to generate consistent, rule-based pipeline names that follow AWS SageMaker naming conventions and incorporate proper versioning information. This ensures pipelines are identifiable, versioned, and compliant with platform requirements.

## Core Problem Solved

SageMaker pipeline names must:
1. Be unique within an AWS account
2. Follow specific character and length limitations
3. Ideally include versioning information
4. Be human-readable and meaningful

Manual pipeline naming is error-prone and can lead to naming conflicts or inconsistencies. The name generator solves this by providing a deterministic and standardized approach to pipeline naming.

## Implementation Details

### Pipeline Name Generation Rules

The `generate_pipeline_name` function applies several rules to generate valid pipeline names:

1. **Base name sanitization**:
   - Convert to lowercase
   - Replace spaces with hyphens
   - Remove special characters
   - Truncate to maximum allowed length

2. **Version incorporation**:
   - Append version with 'v' prefix
   - Handle semantic versioning formats (e.g., '1.0.0')
   - Normalize version format

3. **Platform compliance**:
   - Ensure the name complies with SageMaker naming constraints
   - Apply character set restrictions
   - Enforce length limits

### Example

```python
from src.pipeline_api.name_generator import generate_pipeline_name

# Simple usage
name = generate_pipeline_name("my-pipeline", "1.0")
# Result: "my-pipeline-v1-0"

# With complex name and version
name = generate_pipeline_name("My Complex Pipeline Name!", "2.3.1-beta")
# Result: "my-complex-pipeline-name-v2-3-1-beta"
```

## Usage Patterns

### Direct Usage

```python
from src.pipeline_api.name_generator import generate_pipeline_name

pipeline_name = generate_pipeline_name(
    base_name="xgboost-training",
    version="1.0.0"
)
```

### Usage in Templates

```python
def _get_pipeline_name(self) -> str:
    pipeline_name = getattr(self.base_config, 'pipeline_name', 'mods')
    pipeline_version = getattr(self.base_config, 'pipeline_version', '1.0')
    
    # Use the rule-based generator
    return generate_pipeline_name(pipeline_name, pipeline_version)
```

## Key Features

### 1. Deterministic Generation
The same input parameters will always produce the same pipeline name, ensuring consistency across runs.

### 2. Version Normalization
Various version formats (1.0, 1.0.0, v1.0, etc.) are normalized to a consistent pattern.

### 3. Platform Compliance
Generated names comply with SageMaker's pipeline naming restrictions:
- Only alphanumeric characters and hyphens
- Maximum length enforced
- No consecutive hyphens

### 4. Human-Readable
Generated names preserve readability while ensuring uniqueness and compliance.

## Integration with Other Components

The name generator is used by:
- `PipelineTemplateBase`: For generating pipeline names from configuration
- `PipelineDAGCompiler`: For pipeline name generation during compilation
- `MODSPipelineDAGCompiler`: For consistent naming of MODS pipelines

This ensures that all pipelines generated through the Pipeline API have consistent, compliant, and meaningful names regardless of how they are created.

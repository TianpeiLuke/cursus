---
tags:
  - resource
  - documentation
  - standardization
  - script_documentation
keywords:
  - script documentation
  - processing scripts
  - documentation standard
  - technical writing
  - code documentation
  - algorithm documentation
topics:
  - documentation standards
  - script documentation
  - technical writing
  - knowledge management
language: python
date of note: 2025-11-18
---

# Script Documentation Guide

## Purpose

This guide defines the standardized format and content requirements for documenting processing scripts in the Cursus framework. Following this guide ensures consistent, comprehensive documentation that enables developers to understand, use, and maintain processing scripts effectively.

## Documentation Location

All script documentation files should be placed in:
```
slipbox/scripts/{script_name}_script.md
```

Example: `active_sample_selection.py` → `slipbox/scripts/active_sample_selection_script.md`

## Documentation Structure

Each script documentation file must follow this structure:

### 1. YAML Frontmatter (Required)

Every script documentation file must begin with standardized YAML frontmatter:

```yaml
---
tags:
  - code
  - processing_script
  - [script_category]
  - [specific_domain]
keywords:
  - [script_name]
  - [key_concept_1]
  - [key_concept_2]
  - [algorithm_name]
  - [integration_point]
  - ...  # 5-10 keywords total
topics:
  - [main_topic_1]
  - [main_topic_2]
  - [workflow_area]
  - [technical_domain]
language: python
date of note: YYYY-MM-DD
---
```

**Tag Guidelines**:
- First tag: Always `code` (indicates code documentation)
- Second tag: Always `processing_script` (identifies as script documentation)
- Third tag: Script category (e.g., `active_learning`, `bedrock`, `data_processing`)
- Fourth tag: Specific domain (e.g., `semi_supervised_learning`, `batch_inference`)

**Keyword Guidelines**:
- Include script name
- Include key algorithms (e.g., `BADGE algorithm`, `k-center`)
- Include integration points (e.g., `AWS Bedrock`, `S3 integration`)
- Include major functionalities (e.g., `cost optimization`, `automatic fallback`)
- 5-10 keywords total

**Topic Guidelines**:
- Main workflow area (e.g., `active learning workflows`, `AWS Bedrock integration`)
- Technical domain (e.g., `cost-efficient LLM inference`, `intelligent sampling`)
- 2-4 topics total

### 2. Overview Section (Required)

Provide a high-level overview of the script:

```markdown
## Overview

The `{script_name}.py` script [one-sentence primary purpose].

[2-3 paragraphs describing]:
- What the script does
- Key capabilities and features
- Integration with other components
- Use cases and workflows
```

**Example**:
```markdown
## Overview

The `active_sample_selection.py` script implements intelligent sample selection from model predictions for two primary machine learning workflows: Semi-Supervised Learning (SSL) and Active Learning (AL).

The script supports multiple selection strategies tailored to each use case and provides flexible integration with upstream prediction sources including XGBoost, LightGBM, PyTorch, Bedrock, and Label Rulesets.
```

### 3. Purpose and Major Tasks (Required)

Clearly state the purpose and enumerate major tasks:

```markdown
## Purpose and Major Tasks

### Primary Purpose
[One clear statement of the script's primary purpose]

### Major Tasks
1. **[Task 1 Name]**: [Brief description]
2. **[Task 2 Name]**: [Brief description]
3. **[Task 3 Name]**: [Brief description]
[Continue for 5-10 major tasks]
```

**Example**:
```markdown
### Primary Purpose
Intelligently select high-value samples from a pool of unlabeled data based on model predictions, enabling efficient use of computational or human labeling resources.

### Major Tasks
1. **Data Loading**: Load inference predictions from various upstream sources
2. **Score Normalization**: Convert diverse score formats into standardized probability distributions
3. **Strategy-Based Selection**: Apply appropriate sampling strategy
```

### 4. Script Contract (Required)

Document the script's contract comprehensively:

```markdown
## Script Contract

### Entry Point
```
{script_name}.py
```

### Input Paths
| Path | Location | Description |
|------|----------|-------------|
| `input_1` | `/opt/ml/processing/input/input_1` | [Description] |
| `input_2` | `/opt/ml/processing/input/input_2` | [Description] |

### Output Paths
| Path | Location | Description |
|------|----------|-------------|
| `output_1` | `/opt/ml/processing/output/output_1` | [Description] |
| `output_2` | `/opt/ml/processing/output/output_2` | [Description] |

### Required Environment Variables
| Variable | Description | Example |
|----------|-------------|---------|
| `VAR_NAME` | [Description] | `"example_value"` |

### Optional Environment Variables

#### [Category 1]
| Variable | Default | Description |
|----------|---------|-------------|
| `VAR_1` | `"default"` | [Description] |
| `VAR_2` | `"42"` | [Description] |

#### [Category 2]
| Variable | Default | Description | Range/Options |
|----------|---------|-------------|---------------|
| `VAR_3` | `"1.0"` | [Description] | 0.0 - 1.0 |

### Job Arguments
| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--arg_name` | `str` | Yes | [Description] |
```

**Key Points**:
- Document ALL environment variables with defaults
- Group optional variables by category (e.g., "Core Parameters", "Performance Tuning")
- Provide examples for complex values
- Include valid ranges/options where applicable

### 5. Input Data Structure (Required)

Describe expected input format in detail:

```markdown
## Input Data Structure

### Expected Input Format
```
input_directory/
├── subdirectory/ (if applicable)
│   ├── data.csv (or .tsv, .parquet)
│   └── ...
└── metadata.json (if applicable)
```

### Required Columns
- **Column Name**: Description and requirements
- **Another Column**: Description and requirements

### Optional Columns
- **Optional Column**: Description and use

### Supported Input Sources
1. **Source Type 1**: Description and format details
2. **Source Type 2**: Description and format details
```

**Example**:
```markdown
### Expected Input Format
The script accepts predictions from various upstream sources:

```
evaluation_data/
├── predictions.csv (or .tsv, .parquet)
└── _SUCCESS (optional marker)
```

### Required Columns
- **ID Column**: Configurable via `ID_FIELD` (default: `"id"`)
- **Probability Columns**: One of:
  - `prob_class_0`, `prob_class_1`, ... (standard format)
  - `confidence_score`, `prediction_score` (LLM format)
```

### 6. Output Data Structure (Required)

Describe output format and contents:

```markdown
## Output Data Structure

### Output Directory Structure
```
output_directory/
├── results.{format}
└── metadata.json
```

**Columns in Output**:
- All original input columns (preserved)
- `output_field_1`: Description
- `output_field_2`: Description
- `status`: Processing status (success/error)

### Metadata Output
```
metadata_directory/
└── metadata.json
```

**Metadata Contents**:
```json
{
  "field1": "value1",
  "field2": "value2",
  "statistics": {...}
}
```
```

### 7. Key Functions and Tasks (Required)

Document all major components with their algorithms:

```markdown
## Key Functions and Tasks

### [Component Name] Component

#### `function_name(parameters)`
**Purpose**: [Clear statement of function purpose]

**Algorithm**:
```python
1. [Step 1 description]
2. [Step 2 description]
   a. [Sub-step if needed]
   b. [Sub-step if needed]
3. [Step 3 description]
```

**Parameters**:
- `param1` (type): Description
- `param2` (type): Description

**Returns**: `ReturnType` - Description

**Example**:
```python
result = function_name(arg1, arg2)
```

#### `another_function(parameters)`
[Similar structure for each major function]
```

**Guidelines**:
- Group related functions into components
- Provide pseudocode algorithms for complex logic
- Include parameter descriptions
- Show example usage where helpful
- Document complexity for algorithms

### 8. Algorithms and Data Structures (Required for Complex Scripts)

Document key algorithms with detailed explanations:

```markdown
## Algorithms and Data Structures

### [Algorithm Name]
**Problem**: [Clear problem statement]

**Solution Strategy**:
1. [Approach 1]
2. [Approach 2]
3. [Key insight]

**Algorithm**:
```python
# Detailed pseudocode with comments
for item in items:
    if condition:
        process(item)
```

**Complexity**: 
- Time: O(n log n)
- Space: O(n)

**Key Features**:
- Feature 1 explanation
- Feature 2 explanation
```

**When to Include**:
- Unique or non-trivial algorithms
- Performance-critical operations
- Algorithms requiring special data structures
- Production-validated approaches

**Example Topics**:
- Sampling strategies (k-center, BADGE)
- JSON parsing/repair strategies
- Batch splitting algorithms
- Multipart upload logic

### 9. Performance Characteristics (Recommended)

Document performance metrics and complexity:

```markdown
## Performance Characteristics

### Complexity Analysis

| Operation | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| Operation 1 | O(n) | O(1) | [Notes] |
| Operation 2 | O(n log n) | O(n) | [Notes] |

### Processing Mode Comparison (if applicable)

| Metric | Mode 1 | Mode 2 | Mode 3 |
|--------|--------|--------|--------|
| Cost | Baseline | -50% | -25% |
| Latency | Low | High | Medium |
| Best For | < 1K | > 10K | 1K-10K |
```

### 10. Error Handling (Required)

Document error handling strategies:

```markdown
## Error Handling

### Error Types

#### Input Validation Errors
- **Error Type 1**: Cause and handling
- **Error Type 2**: Cause and handling

#### Processing Errors
- **Error Type 1**: Cause and handling
- **Error Type 2**: Cause and handling

### Error Response Structure
```python
{
    "status": "error",
    "error_message": "...",
    "error_type": "...",
    # Additional fields
}
```
```

### 11. Best Practices (Recommended)

Provide usage recommendations:

```markdown
## Best Practices

### For Production Deployments
1. **Practice 1**: Description and rationale
2. **Practice 2**: Description and rationale

### For Development
1. **Practice 1**: Description and rationale
2. **Practice 2**: Description and rationale

### For Performance Optimization
1. **Practice 1**: Description and rationale
2. **Practice 2**: Description and rationale
```

### 12. Example Configurations (Recommended)

Provide real-world configuration examples:

```markdown
## Example Configurations

### [Use Case 1]
```bash
export VAR_1="value1"
export VAR_2="value2"
export VAR_3="value3"
```

**Use Case**: Description of when to use this configuration

### [Use Case 2]
```bash
export VAR_1="different_value"
export VAR_2="different_value"
```

**Use Case**: Description of when to use this configuration
```

### 13. Integration Patterns (Recommended)

Document how the script integrates with other components:

```markdown
## Integration Patterns

### Upstream Integration
```
UpstreamStep
   ↓ (outputs: data, metadata)
CurrentScript
   ↓ (outputs: processed_data)
```

### Downstream Integration
```
CurrentScript
   ↓ (outputs: processed_data)
DownstreamStep
   ↓ (outputs: final_output)
```

### Workflow Example
1. Step 1: Description
2. Step 2: Description
3. Step 3: Description
```

### 14. Troubleshooting (Recommended)

Provide troubleshooting guidance:

```markdown
## Troubleshooting

### [Issue Category 1]

**Symptom**: Description of the problem

**Common Causes**:
1. **Cause 1**: Description
2. **Cause 2**: Description

**Solution**: Step-by-step resolution

### [Issue Category 2]
[Similar structure]
```

### 15. References (Required)

Link to related documentation using proper markdown link syntax:

```markdown
## References

### Related Scripts
- [`script1.py`](../path/to/script1_script.md): Brief description
- [`script2.py`](../path/to/script2_script.md): Brief description

### Related Documentation
- **Step Builder**: [`slipbox/steps/step_name.md`](../steps/step_name.md) (if exists)
- **Config Class**: [`slipbox/core/config_name.md`](../core/config_name.md) (if exists)
- **Contract**: [`src/cursus/steps/contracts/contract_name.py`](../../src/cursus/steps/contracts/contract_name.py)
- **Step Specification**: Brief description

### Related Design Documents
- **[Design Doc Title](../1_design/actual_design_doc.md)**: Brief description of design doc content
- **[Another Design Title](../1_design/another_real_doc.md)**: Brief description of design doc content

**IMPORTANT**: 
- Use markdown link format: `[Link Text](relative/path/to/file.md)`
- All paths must be relative from the script documentation location (`slipbox/scripts/`)
- Only include design documents that actually exist in the repository
- Verify file existence before adding links using `list_files` or `read_file`
- If no design docs exist, state: "No specific design documents currently exist for this script"

### External References (if applicable)
- [Paper Title](https://doi.org/example): Brief description
- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/): Specific feature
- [Framework Docs](https://example.com/docs): Specific topic
```

**Critical Guidelines for Related Design Documents**:
- **Use markdown link syntax**: `[Descriptive Title](../1_design/actual_file.md)`
- **Relative paths from `slipbox/scripts/`**: Use `../1_design/` to reference design docs
- **ONLY link to existing files**: Verify with `list_files` on `slipbox/1_design/` directory
- **Use descriptive link text**: "Template-Driven Processing Design" not "Design Doc"
- **Add brief descriptions**: Explain what each design doc covers after the link
- **Acceptable alternatives**:
  - If no design docs exist: "No specific design documents currently exist for this script"
  - If design docs exist but aren't specific: Link to general design patterns that apply
  - If uncertain: Search `slipbox/1_design/` directory first

**Example with Real Links**:
```markdown
### Related Design Documents
- **[Bedrock Processing Step Builder Patterns](../1_design/bedrock_processing_step_builder_patterns.md)**: Template-driven processing architecture and step builder integration patterns
- **[Concurrent Processing Architecture](../1_design/concurrent_bedrock_processing_design.md)**: Multi-threaded processing design with rate limiting
```

## Writing Guidelines

### Style and Tone

1. **Be Clear and Direct**
   - Use simple, precise language
   - Avoid jargon where possible
   - Define technical terms on first use

2. **Be Comprehensive**
   - Document all parameters and options
   - Explain the "why" not just the "what"
   - Include edge cases and limitations

3. **Be Practical**
   - Provide runnable examples
   - Include real-world use cases
   - Show both success and error scenarios

4. **Be Consistent**
   - Use consistent terminology
   - Follow the same structure across all scripts
   - Maintain parallel phrasing in lists

### Technical Writing Tips

1. **Use Tables for Structured Data**
   - Parameters, environment variables, options
   - Comparison matrices
   - Configuration options

2. **Use Code Blocks for Examples**
   - Configuration examples
   - Command-line usage
   - Algorithm pseudocode

3. **Use Lists for Sequential Information**
   - Workflow steps
   - Algorithm steps
   - Troubleshooting procedures

4. **Use Headings for Navigation**
   - Clear hierarchy (##, ###, ####)
   - Descriptive heading text
   - Logical grouping of content

## Quality Checklist

Before considering script documentation complete, verify:

- [ ] YAML frontmatter present and correct
- [ ] Overview clearly explains script purpose
- [ ] All contract elements documented (I/O, env vars, args)
- [ ] Input and output structures clearly described
- [ ] All major functions documented with algorithms
- [ ] Complex algorithms explained with pseudocode
- [ ] Error handling documented
- [ ] At least 2 example configurations provided
- [ ] Integration patterns described
- [ ] References section complete
- [ ] No spelling or grammatical errors
- [ ] Code examples are syntactically correct
- [ ] Tables are properly formatted
- [ ] All sections use consistent terminology

## Documentation Workflow

### Step-by-Step Process

1. **Read the Script Implementation**
   - Understand the full script logic
   - Identify major components and functions
   - Note key algorithms and data structures

2. **Read the Script Contract**
   - Extract I/O paths
   - Document environment variables
   - Note job arguments

3. **Check for Existing Documentation**
   - Look for partial documentation
   - Review design documents
   - Check related script documentation

4. **Create/Update Documentation File**
   - Start with YAML frontmatter
   - Follow the structure outlined above
   - Use examples from the script itself

5. **Validate Documentation**
   - Check against quality checklist
   - Verify all code examples
   - Test any provided configurations

6. **Review and Refine**
   - Ensure clarity and completeness
   - Check for consistency with other docs
   - Verify all cross-references

## Examples from Existing Documentation

### Example 1: Active Sample Selection

**Good YAML Frontmatter**:
```yaml
---
tags:
  - code
  - processing_script
  - active_learning
  - semi_supervised_learning
  - sample_selection
keywords:
  - active sample selection
  - semi-supervised learning
  - active learning
  - confidence-based sampling
  - uncertainty sampling
  - diversity sampling
  - BADGE algorithm
  - pseudo-labeling
  - sample selection strategies
topics:
  - active learning workflows
  - semi-supervised learning
  - intelligent sampling
  - model predictions
language: python
date of note: 2025-11-18
---
```

**Good Algorithm Documentation**:
```markdown
#### `UncertaintySampler.compute_scores(probabilities)`
**Purpose**: Compute uncertainty scores (higher = more uncertain)

**Uncertainty Modes**:

1. **Margin Sampling** (default):
   ```python
   uncertainty = 1 - (P(class_1st) - P(class_2nd))
   ```
   - Measures gap between top two predictions
   - High uncertainty when classes are close

2. **Entropy Sampling**:
   ```python
   uncertainty = -Σ P(class_i) * log(P(class_i))
   ```
   - Shannon entropy of probability distribution
```

### Example 2: Bedrock Batch Processing

**Good Contract Documentation**:
```markdown
### Required Environment Variables
| Variable | Description | Example |
|----------|-------------|---------|
| `BEDROCK_PRIMARY_MODEL_ID` | Primary Bedrock model ID | `"anthropic.claude-sonnet-4-20250514-v1:0"` |

### Optional Environment Variables

#### Batch-Specific Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `BEDROCK_BATCH_MODE` | `"auto"` | Batch processing mode: auto/batch/realtime |
| `BEDROCK_BATCH_THRESHOLD` | `"1000"` | Minimum records for automatic batch processing |
```

**Good Integration Pattern Documentation**:
```markdown
### Upstream Integration (Template Generation)
```
BedrockPromptTemplateGeneration
   ↓ (outputs: prompts.json, validation_schema.json)
BedrockBatchProcessing
   ↓ (outputs: processed_data, summary)
```
```

## Common Pitfalls to Avoid

1. **Incomplete Environment Variable Documentation**
   - Always document defaults
   - Show valid ranges/options
   - Explain impact of each variable

2. **Missing Algorithm Complexity**
   - Include Big-O notation for algorithms
   - Explain performance characteristics
   - Note any trade-offs

3. **Unclear Input/Output Formats**
   - Show actual directory structures
   - Document all required fields
   - Explain format preservation logic

4. **Lack of Examples**
   - Every configuration section needs examples
   - Show both simple and complex cases
   - Include real command-line usage

5. **Poor Cross-Referencing**
   - Link to related scripts
   - Reference design documents
   - Point to contracts and specs

## Maintenance and Updates

### When to Update Documentation

- Script functionality changes
- New environment variables added
- Algorithm improvements
- Bug fixes affecting behavior
- Integration changes
- Performance optimizations

### Update Process

1. Update the `date of note` field
2. Modify affected sections
3. Add notes about changes if significant
4. Verify examples still work
5. Update cross-references if needed

## Conclusion

Following this guide ensures that all processing script documentation is:

- **Consistent**: Same structure and format across all scripts
- **Comprehensive**: All aspects of the script are documented
- **Practical**: Users can understand and use the script effectively
- **Maintainable**: Documentation can be easily updated

This guide should be considered a living document. As we document more scripts and identify patterns or gaps, we should refine this guide to better serve our documentation needs.

## Related Resources

- **[YAML Frontmatter Standard](documentation_yaml_frontmatter_standard.md)**: Standard format for YAML frontmatter in documentation files
- **[Script Development Guide](../0_developer_guide/script_development_guide.md)**: Comprehensive guide for developing processing scripts in the Cursus framework
- **[Script Contract Guide](../0_developer_guide/script_contract.md)**: Understanding and implementing script contracts for pipeline steps
- **[Script Testability Implementation](../0_developer_guide/script_testability_implementation.md)**: Framework patterns and best practices for implementing testable processing scripts
- **[Active Sample Selection Script](../scripts/active_sample_selection_script.md)**: Complete example of script documentation following this guide's standardized format
- **[Bedrock Batch Processing Script](../scripts/bedrock_batch_processing_script.md)**: Complex script documentation example with comprehensive algorithm documentation
- **[Bedrock Processing Script](../scripts/bedrock_processing_script.md)**: Script documentation example demonstrating all sections and best practices

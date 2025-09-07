---
tags:
  - code
  - core
  - compiler
  - name_generation
  - pipeline_naming
keywords:
  - generate_pipeline_name
  - sanitize_pipeline_name
  - validate_pipeline_name
  - generate_random_word
  - SageMaker constraints
  - pipeline naming
topics:
  - pipeline naming
  - name generation
  - SageMaker compliance
language: python
date of note: 2025-09-07
---

# Name Generator

Name generator utilities for pipeline naming with consistent formats that comply with SageMaker naming constraints.

## Overview

The `name_generator` module provides utilities for generating pipeline names with consistent formats that comply with SageMaker naming constraints. It includes functions for generating random components, validating names against SageMaker requirements, sanitizing names to ensure compliance, and generating complete pipeline names.

SageMaker has specific constraints for pipeline names: they must match the pattern `[a-zA-Z0-9](-*[a-zA-Z0-9]){0,255}`, be no longer than 255 characters, and start and end with alphanumeric characters. This module ensures all generated names comply with these requirements.

## Classes and Methods

### Functions
- [`generate_pipeline_name`](#generate_pipeline_name) - Generate a valid pipeline name with consistent format
- [`sanitize_pipeline_name`](#sanitize_pipeline_name) - Sanitize a pipeline name to conform to SageMaker constraints
- [`validate_pipeline_name`](#validate_pipeline_name) - Validate that a pipeline name conforms to SageMaker constraints
- [`generate_random_word`](#generate_random_word) - Generate a random word of specified length

### Constants
- [`PIPELINE_NAME_PATTERN`](#pipeline_name_pattern) - SageMaker pipeline name constraint pattern

## API Reference

### generate_pipeline_name

generate_pipeline_name(_base_name_, _version="1.0"_)

Generate a valid pipeline name with the format: {base_name}-{version}-pipeline. This function ensures the generated name conforms to SageMaker constraints by sanitizing it before returning.

**Parameters:**
- **base_name** (_str_) – Base name for the pipeline
- **version** (_str_) – Version string to include in the name (default: "1.0")

**Returns:**
- **str** – A string with the generated pipeline name that passes SageMaker validation

```python
from cursus.core.compiler.name_generator import generate_pipeline_name

# Generate pipeline name with default version
pipeline_name = generate_pipeline_name("my-ml-model")
print(pipeline_name)  # Output: "my-ml-model-1.0-pipeline"

# Generate pipeline name with custom version
pipeline_name = generate_pipeline_name("fraud-detection", "2.1")
print(pipeline_name)  # Output: "fraud-detection-2.1-pipeline"

# Generate name with base name that needs sanitization
pipeline_name = generate_pipeline_name("my_model.v1", "1.0")
print(pipeline_name)  # Output: "my-model-v1-1.0-pipeline"
```

### sanitize_pipeline_name

sanitize_pipeline_name(_name_)

Sanitize a pipeline name to conform to SageMaker constraints. This function: 1. Replaces dots with hyphens, 2. Replaces underscores with hyphens, 3. Removes any other special characters, 4. Ensures the name starts with an alphanumeric character, 5. Ensures the name ends with an alphanumeric character.

**Parameters:**
- **name** (_str_) – The pipeline name to sanitize

**Returns:**
- **str** – A sanitized version of the name that conforms to SageMaker constraints

```python
from cursus.core.compiler.name_generator import sanitize_pipeline_name

# Sanitize name with dots and underscores
sanitized = sanitize_pipeline_name("my_model.v1.pipeline")
print(sanitized)  # Output: "my-model-v1-pipeline"

# Sanitize name with special characters
sanitized = sanitize_pipeline_name("model@#$%pipeline")
print(sanitized)  # Output: "modelpipeline"

# Sanitize name that starts with special character
sanitized = sanitize_pipeline_name("-my-pipeline")
print(sanitized)  # Output: "p-my-pipeline"

# Sanitize name that's too long
long_name = "a" * 300
sanitized = sanitize_pipeline_name(long_name)
print(len(sanitized))  # Output: 255 (truncated to max length)
```

### validate_pipeline_name

validate_pipeline_name(_name_)

Validate that a pipeline name conforms to SageMaker constraints.

**Parameters:**
- **name** (_str_) – The pipeline name to validate

**Returns:**
- **bool** – True if the name is valid, False otherwise

```python
from cursus.core.compiler.name_generator import validate_pipeline_name

# Valid pipeline names
print(validate_pipeline_name("my-pipeline"))  # Output: True
print(validate_pipeline_name("model-v1-0"))   # Output: True
print(validate_pipeline_name("Pipeline123"))  # Output: True

# Invalid pipeline names
print(validate_pipeline_name("my_pipeline"))  # Output: False (underscore not allowed)
print(validate_pipeline_name("-pipeline"))    # Output: False (starts with hyphen)
print(validate_pipeline_name("pipeline-"))    # Output: False (ends with hyphen)
print(validate_pipeline_name(""))             # Output: False (empty string)

# Check before using a name
name = "my-custom-pipeline"
if validate_pipeline_name(name):
    print(f"'{name}' is a valid pipeline name")
else:
    print(f"'{name}' needs to be sanitized")
    name = sanitize_pipeline_name(name)
```

### generate_random_word

generate_random_word(_length=4_)

Generate a random word of specified length.

**Parameters:**
- **length** (_int_) – Length of the random word (default: 4)

**Returns:**
- **str** – Random string of specified length

```python
from cursus.core.compiler.name_generator import generate_random_word

# Generate random 4-letter word (default)
random_word = generate_random_word()
print(random_word)  # Output: "XKCD" (example)

# Generate random word of custom length
random_word = generate_random_word(6)
print(random_word)  # Output: "ABCDEF" (example)

# Use in custom name generation
base_name = "model"
random_suffix = generate_random_word(3)
custom_name = f"{base_name}-{random_suffix}-pipeline"
print(custom_name)  # Output: "model-XYZ-pipeline" (example)
```

### PIPELINE_NAME_PATTERN

SageMaker pipeline name constraint pattern. Must match: `[a-zA-Z0-9](-*[a-zA-Z0-9]){0,255}`

**Type:** _str_

**Value:** `r'^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,255}$'`

```python
from cursus.core.compiler.name_generator import PIPELINE_NAME_PATTERN
import re

# Use pattern for custom validation
def custom_validate(name):
    return bool(re.match(PIPELINE_NAME_PATTERN, name))

print(custom_validate("valid-name"))    # Output: True
print(custom_validate("invalid_name"))  # Output: False
```

## SageMaker Naming Constraints

SageMaker pipeline names must comply with the following constraints:

### Pattern Requirements
- Must start with an alphanumeric character (`[a-zA-Z0-9]`)
- Can contain alphanumeric characters and hyphens
- Must end with an alphanumeric character
- Cannot have consecutive hyphens
- Cannot start or end with hyphens

### Length Requirements
- Must be between 1 and 255 characters long
- Empty strings are not allowed

### Common Issues and Solutions

| Issue | Example | Solution |
|-------|---------|----------|
| Underscores | `my_pipeline` | Replace with hyphens: `my-pipeline` |
| Dots | `model.v1` | Replace with hyphens: `model-v1` |
| Special characters | `model@v1` | Remove: `modelv1` |
| Leading hyphen | `-pipeline` | Add prefix: `p-pipeline` |
| Trailing hyphen | `pipeline-` | Remove: `pipeline` |
| Too long | 300+ characters | Truncate to 255 characters |

## Usage Patterns

### Basic Pipeline Naming

```python
# Simple pipeline name generation
name = generate_pipeline_name("fraud-detection", "1.2")
# Result: "fraud-detection-1.2-pipeline"
```

### Name Validation and Sanitization

```python
# Validate and sanitize user input
user_input = "my_model.v1@pipeline"

if not validate_pipeline_name(user_input):
    sanitized_name = sanitize_pipeline_name(user_input)
    print(f"Sanitized '{user_input}' to '{sanitized_name}'")
    # Output: Sanitized 'my_model.v1@pipeline' to 'my-model-v1pipeline'
```

### Custom Name Generation

```python
# Generate unique names with random components
def generate_unique_pipeline_name(base_name, version):
    random_suffix = generate_random_word(4)
    name = f"{base_name}-{random_suffix}-{version}-pipeline"
    return sanitize_pipeline_name(name)

unique_name = generate_unique_pipeline_name("model", "1.0")
# Result: "model-ABCD-1.0-pipeline" (ABCD is random)
```

## Related Documentation

- [DAG Compiler](dag_compiler.md) - Uses name generator for pipeline naming
- [Dynamic Template](dynamic_template.md) - May use name generation for pipeline creation
- [Compiler Overview](README.md) - System overview and integration

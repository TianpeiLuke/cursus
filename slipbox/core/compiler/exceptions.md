---
tags:
  - code
  - pipeline_api
  - error_handling
  - exceptions
keywords:
  - exceptions
  - error handling
  - pipeline API
  - validation
topics:
  - pipeline API
  - error handling
language: python
date of note: 2025-07-31
---

# Pipeline API Exceptions

## Purpose

The exceptions module defines a hierarchy of custom exception classes specific to the Pipeline API. These exceptions provide structured error information to help users diagnose and resolve issues during pipeline generation.

## Exception Hierarchy

```
PipelineAPIError (Base class)
├── ConfigurationError
│   └── AmbiguityError
├── ValidationError
└── ResolutionError
```

## Core Problem Solved

When errors occur during pipeline generation, generic Python exceptions often lack context about what went wrong and how to fix it. The custom exception classes:

1. Provide specific error types for different failure scenarios
2. Include detailed context information about the error
3. Allow for structured error handling
4. Make troubleshooting more efficient

## Exception Types

### PipelineAPIError

Base exception class for all Pipeline API exceptions.

```python
try:
    pipeline = compile_dag_to_pipeline(dag, config_path)
except PipelineAPIError as e:
    print(f"Pipeline generation failed: {e}")
```

### ConfigurationError

Raised when configuration issues prevent pipeline generation.

```python
try:
    pipeline = compile_dag_to_pipeline(dag, config_path)
except ConfigurationError as e:
    print(f"Configuration issue: {e}")
    print(f"Missing configs: {e.missing_configs}")
    print(f"Available configs: {e.available_configs}")
```

### AmbiguityError

A specialized configuration error raised when multiple configurations match a node with similar confidence.

```python
try:
    pipeline = compile_dag_to_pipeline(dag, config_path)
except AmbiguityError as e:
    print(f"Ambiguous configuration: {e}")
    print(f"Node: {e.node_name}")
    print(f"Candidates: {e.candidates}")
    print("Try using more specific node names or add job_type attributes")
```

### ValidationError

Raised when validation checks fail during pipeline generation.

```python
try:
    pipeline = compile_dag_to_pipeline(dag, config_path)
except ValidationError as e:
    print(f"Validation failed: {e}")
    for category, errors in e.validation_errors.items():
        print(f"  {category}: {errors}")
```

### ResolutionError

Raised when the resolver cannot map nodes to configurations.

```python
try:
    pipeline = compile_dag_to_pipeline(dag, config_path)
except ResolutionError as e:
    print(f"Resolution failed: {e}")
    print(f"Unresolved nodes: {e.unresolved_nodes}")
```

## Integration with External Exceptions

The Pipeline API also re-exports relevant exceptions from dependent modules:

- `RegistryError` from `pipeline_registry.exceptions`: Raised when the registry cannot map configurations to step builders.

```python
from src.pipeline_api.exceptions import RegistryError

try:
    pipeline = compile_dag_to_pipeline(dag, config_path)
except RegistryError as e:
    print(f"Registry issue: {e}")
    print(f"Unresolvable types: {e.unresolvable_types}")
    print(f"Available builders: {e.available_builders}")
```

## Best Practices

1. **Catch specific exceptions first**: More specific exceptions should be caught before more general ones.
2. **Extract context information**: Use the additional attributes on exception instances to provide helpful error messages.
3. **Provide resolution hints**: Use the exception information to suggest how the user might fix the issue.

```python
from src.pipeline_api.exceptions import ConfigurationError, ValidationError, RegistryError, PipelineAPIError

try:
    pipeline = compile_dag_to_pipeline(dag, config_path)
except ConfigurationError as e:
    print(f"Configuration issue: {e}")
    # Configuration-specific handling
except ValidationError as e:
    print(f"Validation failed: {e}")
    # Validation-specific handling
except RegistryError as e:
    print(f"Registry issue: {e}")
    # Registry-specific handling
except PipelineAPIError as e:
    print(f"Other pipeline error: {e}")
    # Generic handling
```

These exceptions enable better error handling, clearer error messages, and more efficient troubleshooting when working with the Pipeline API.

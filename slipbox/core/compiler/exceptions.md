---
tags:
  - code
  - core
  - compiler
  - exceptions
  - error_handling
keywords:
  - PipelineAPIError
  - ConfigurationError
  - AmbiguityError
  - ValidationError
  - ResolutionError
  - exception handling
topics:
  - error handling
  - exception classes
  - pipeline compilation
language: python
date of note: 2025-09-07
---

# Compiler Exceptions

Custom exception classes for the Pipeline API to provide clear, actionable error messages for users.

## Overview

The `exceptions` module defines custom exceptions used throughout the Pipeline API to provide clear, actionable error messages for users. These exceptions are designed to give detailed information about what went wrong during pipeline compilation, including specific nodes that failed, available alternatives, and suggestions for resolution.

The exception hierarchy provides specialized error types for different failure scenarios including configuration errors, resolution ambiguity, validation failures, and general pipeline API errors. Each exception includes relevant context information to help users diagnose and fix issues.

## Classes and Methods

### Exception Classes
- [`PipelineAPIError`](#pipelineapierror) - Base exception for all Pipeline API errors
- [`ConfigurationError`](#configurationerror) - Raised when configuration-related errors occur
- [`AmbiguityError`](#ambiguityerror) - Raised when multiple configurations could match a DAG node
- [`ValidationError`](#validationerror) - Raised when DAG-config validation fails
- [`ResolutionError`](#resolutionerror) - Raised when DAG node resolution fails

## API Reference

### PipelineAPIError

_class_ cursus.core.compiler.exceptions.PipelineAPIError

Base exception for all Pipeline API errors.

```python
from cursus.core.compiler.exceptions import PipelineAPIError

try:
    # Pipeline API operation
    pass
except PipelineAPIError as e:
    print(f"Pipeline API error: {e}")
```

### ConfigurationError

_class_ cursus.core.compiler.exceptions.ConfigurationError(_message_, _missing_configs=None_, _available_configs=None_)

Raised when configuration-related errors occur.

**Parameters:**
- **message** (_str_) – Error message describing the configuration issue
- **missing_configs** (_Optional[List[str]]_) – List of missing configuration names
- **available_configs** (_Optional[List[str]]_) – List of available configuration names

**Attributes:**
- **missing_configs** (_List[str]_) – List of missing configuration names
- **available_configs** (_List[str]_) – List of available configuration names

```python
from cursus.core.compiler.exceptions import ConfigurationError

try:
    # Configuration resolution
    pass
except ConfigurationError as e:
    print(f"Configuration error: {e}")
    print(f"Missing configs: {e.missing_configs}")
    print(f"Available configs: {e.available_configs}")
    
    # Suggest fixes
    if e.missing_configs and e.available_configs:
        print("Suggestions:")
        for missing in e.missing_configs:
            similar = [cfg for cfg in e.available_configs if missing.lower() in cfg.lower()]
            if similar:
                print(f"  - For '{missing}', consider: {similar}")
```

### AmbiguityError

_class_ cursus.core.compiler.exceptions.AmbiguityError(_message_, _node_name=None_, _candidates=None_)

Raised when multiple configurations could match a DAG node.

**Parameters:**
- **message** (_str_) – Error message describing the ambiguity
- **node_name** (_Optional[str]_) – Name of the ambiguous DAG node
- **candidates** (_Optional[List[Any]]_) – List of candidate configurations

**Attributes:**
- **node_name** (_Optional[str]_) – Name of the ambiguous DAG node
- **candidates** (_List[Any]_) – List of candidate configurations

```python
from cursus.core.compiler.exceptions import AmbiguityError

try:
    # Node resolution
    pass
except AmbiguityError as e:
    print(f"Ambiguity error: {e}")
    print(f"Ambiguous node: {e.node_name}")
    
    # Show candidates with details
    if e.candidates:
        print("Candidate configurations:")
        for candidate in e.candidates:
            if isinstance(candidate, dict):
                config_type = candidate.get('config_type', 'Unknown')
                confidence = candidate.get('confidence', 0.0)
                job_type = candidate.get('job_type', 'N/A')
                print(f"  - {config_type} (job_type='{job_type}', confidence={confidence:.2f})")
```

### ValidationError

_class_ cursus.core.compiler.exceptions.ValidationError(_message_, _validation_errors=None_)

Raised when DAG-config validation fails.

**Parameters:**
- **message** (_str_) – Error message describing the validation failure
- **validation_errors** (_Optional[Dict[str, List[str]]]_) – Dictionary of validation errors by category

**Attributes:**
- **validation_errors** (_Dict[str, List[str]]_) – Dictionary of validation errors by category

```python
from cursus.core.compiler.exceptions import ValidationError

try:
    # DAG validation
    pass
except ValidationError as e:
    print(f"Validation error: {e}")
    
    # Show detailed validation errors
    if e.validation_errors:
        for category, errors in e.validation_errors.items():
            print(f"{category} errors:")
            for error in errors:
                print(f"  - {error}")
                
    # Handle specific validation categories
    if 'missing_configs' in e.validation_errors:
        missing = e.validation_errors['missing_configs']
        print(f"Add configurations for: {missing}")
        
    if 'unresolvable_builders' in e.validation_errors:
        unresolvable = e.validation_errors['unresolvable_builders']
        print(f"Register step builders for: {unresolvable}")
```

### ResolutionError

_class_ cursus.core.compiler.exceptions.ResolutionError(_message_, _failed_nodes=None_, _suggestions=None_)

Raised when DAG node resolution fails.

**Parameters:**
- **message** (_str_) – Error message describing the resolution failure
- **failed_nodes** (_Optional[List[str]]_) – List of nodes that failed to resolve
- **suggestions** (_Optional[List[str]]_) – List of suggestions for fixing the issue

**Attributes:**
- **failed_nodes** (_List[str]_) – List of nodes that failed to resolve
- **suggestions** (_List[str]_) – List of suggestions for fixing the issue

```python
from cursus.core.compiler.exceptions import ResolutionError

try:
    # Node resolution
    pass
except ResolutionError as e:
    print(f"Resolution error: {e}")
    print(f"Failed nodes: {e.failed_nodes}")
    
    # Show suggestions
    if e.suggestions:
        print("Suggestions:")
        for suggestion in e.suggestions:
            print(f"  - {suggestion}")
            
    # Implement fixes based on suggestions
    for node in e.failed_nodes:
        print(f"Consider renaming node '{node}' or adding matching configuration")
```

## Exception Hierarchy

```
PipelineAPIError (base)
├── ConfigurationError
├── AmbiguityError  
├── ValidationError
└── ResolutionError
```

## Error Handling Patterns

### Comprehensive Error Handling

```python
from cursus.core.compiler.exceptions import (
    PipelineAPIError, ConfigurationError, AmbiguityError, 
    ValidationError, ResolutionError
)

try:
    # Pipeline compilation
    pipeline = compile_dag_to_pipeline(dag, config_path)
    
except ConfigurationError as e:
    print("Configuration issue - check your config file")
    print(f"Missing: {e.missing_configs}")
    print(f"Available: {e.available_configs}")
    
except AmbiguityError as e:
    print("Multiple configs match - be more specific")
    print(f"Ambiguous node: {e.node_name}")
    
except ValidationError as e:
    print("Validation failed - fix configuration issues")
    for category, errors in e.validation_errors.items():
        print(f"{category}: {errors}")
        
except ResolutionError as e:
    print("Resolution failed - check node names")
    print(f"Failed nodes: {e.failed_nodes}")
    
except PipelineAPIError as e:
    print(f"General pipeline error: {e}")
```

### Graceful Degradation

```python
def safe_compile_pipeline(dag, config_path):
    """Compile pipeline with graceful error handling."""
    try:
        return compile_dag_to_pipeline(dag, config_path)
    except AmbiguityError as e:
        # Try with first candidate
        print(f"Warning: Using first candidate for ambiguous node {e.node_name}")
        return compile_dag_to_pipeline(dag, config_path, use_first_match=True)
    except ConfigurationError as e:
        # Provide fallback configuration
        print(f"Warning: Configuration error, using defaults: {e}")
        return None
```

## Related Documentation

- [DAG Compiler](dag_compiler.md) - Uses these exceptions for error reporting
- [Configuration Resolver](config_resolver.md) - Raises ConfigurationError and AmbiguityError
- [Dynamic Template](dynamic_template.md) - Raises ValidationError during template creation
- [Validation](validation.md) - Raises ValidationError for validation failures
- [Compiler Overview](README.md) - System overview and integration

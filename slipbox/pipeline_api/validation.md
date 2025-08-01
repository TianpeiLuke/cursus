---
tags:
  - code
  - pipeline_api
  - validation
  - quality_assurance
keywords:
  - validation
  - compatibility
  - pipeline
  - DAG
topics:
  - pipeline API
  - validation engine
language: python
date of note: 2025-07-31
---

# Validation Engine

## Purpose

The `ValidationEngine` provides comprehensive validation capabilities for the Pipeline API, ensuring compatibility between DAGs, configurations, and step builders before pipeline generation. This helps catch issues early and provides detailed feedback to users.

## Core Problem Solved

Pipeline generation can fail for various reasons:
1. Missing configurations for DAG nodes
2. Missing step builders for configuration types
3. Configuration validation errors
4. Dependency resolution issues

The validation engine catches these issues early in the process and provides detailed reports to help users diagnose and fix problems.

## Validation Process

The validation engine performs multiple checks:

### 1. DAG-Configuration Compatibility

Ensures each DAG node has a corresponding configuration.

```python
validation_result = validation_engine.validate_dag_compatibility(
    dag_nodes=["data_load", "preprocess", "train"],
    available_configs=loaded_configs,
    config_map=config_map,
    builder_registry=builder_map
)

if not validation_result.is_valid:
    print(f"Missing configurations: {validation_result.missing_configs}")
```

### 2. Configuration-Builder Compatibility

Ensures each configuration has a corresponding step builder.

```python
if validation_result.unresolvable_builders:
    print(f"Missing builders: {validation_result.unresolvable_builders}")
    print(f"Available builders: {validation_result.builder_registry_stats}")
```

### 3. Configuration-Specific Validation

Runs specific validation logic for each configuration type.

```python
if validation_result.config_errors:
    for config_name, errors in validation_result.config_errors.items():
        print(f"Errors in {config_name}: {errors}")
```

### 4. Dependency Resolution Validation

Ensures all dependencies can be resolved correctly.

```python
if validation_result.dependency_issues:
    print(f"Dependency issues: {validation_result.dependency_issues}")
```

## Validation Result

The validation process returns a `ValidationResult` containing:

- `is_valid`: Boolean indicating if validation passed
- `missing_configs`: List of DAG nodes without configurations
- `unresolvable_builders`: List of configuration types without builders
- `config_errors`: Dictionary of configuration-specific errors
- `dependency_issues`: List of dependency resolution issues
- `warnings`: List of non-critical issues
- `detailed_report()`: Method to generate a detailed report
- `summary()`: Method to generate a summary report

## Resolution Preview

The validation engine also provides a `ResolutionPreview` that shows how DAG nodes will be resolved to configurations and builders:

```python
preview = validation_engine.preview_resolution(
    dag_nodes=["data_load", "preprocess", "train"],
    available_configs=loaded_configs
)

print(preview.display())
```

The preview includes:
- `node_config_map`: Mapping from nodes to config types
- `config_builder_map`: Mapping from config types to builder types
- `resolution_confidence`: Confidence scores for each resolution
- `ambiguous_resolutions`: List of potentially ambiguous resolutions
- `recommendations`: Suggested improvements

## Conversion Report

After pipeline generation, the validation engine can generate a `ConversionReport` with detailed information about the conversion process:

```python
report = ConversionReport(
    pipeline_name="my-pipeline",
    steps=dag_nodes,
    resolution_details=resolution_details,
    avg_confidence=0.85,
    warnings=warnings,
    metadata=metadata
)

print(report.detailed_report())
```

The report includes:
- Pipeline details (name, steps)
- Resolution details for each step
- Average confidence score
- Warnings and recommendations
- Metadata about the conversion process

## Integration with Other Components

The validation engine is used by:

- `PipelineDAGCompiler`: For pre-compilation validation
- `DynamicPipelineTemplate`: For configuration validation
- `MODSPipelineDAGCompiler`: For MODS-specific validation

It serves as a quality assurance layer, ensuring that only compatible and well-formed pipelines are generated.

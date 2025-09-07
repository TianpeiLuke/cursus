---
tags:
  - code
  - core
  - compiler
  - validation
  - dag_validation
keywords:
  - ValidationEngine
  - ValidationResult
  - ResolutionPreview
  - ConversionReport
  - DAG validation
  - configuration validation
topics:
  - validation
  - DAG compatibility
  - pipeline validation
language: python
date of note: 2025-09-07
---

# Validation

Validation and preview classes for the Pipeline API, providing classes for validating DAG-config compatibility and previewing resolution results before pipeline generation.

## Overview

The `validation` module provides classes for validating DAG-config compatibility and previewing resolution results before pipeline generation. It includes comprehensive validation of DAG nodes against available configurations, step builder resolution, configuration validation, and dependency checking.

The module provides detailed reporting capabilities with actionable recommendations for fixing validation issues. It supports both summary and detailed reporting formats, making it easy to diagnose and resolve pipeline compilation problems.

## Classes and Methods

### Classes
- [`ValidationEngine`](#validationengine) - Engine for validating DAG-config compatibility
- [`ValidationResult`](#validationresult) - Result of DAG-config compatibility validation
- [`ResolutionPreview`](#resolutionpreview) - Preview of how DAG nodes will be resolved
- [`ConversionReport`](#conversionreport) - Report generated after successful pipeline conversion

## API Reference

### ValidationEngine

_class_ cursus.core.compiler.validation.ValidationEngine

Engine for validating DAG-config compatibility.

```python
from cursus.core.compiler.validation import ValidationEngine

# Create validation engine
engine = ValidationEngine()
```

#### validate_dag_compatibility

validate_dag_compatibility(_dag_nodes_, _available_configs_, _config_map_, _builder_registry_)

Validate DAG-config compatibility.

**Parameters:**
- **dag_nodes** (_List[str]_) – List of DAG node names
- **available_configs** (_Dict[str, Any]_) – Available configuration instances
- **config_map** (_Dict[str, Any]_) – Resolved node-to-config mapping
- **builder_registry** (_Dict[str, Any]_) – Available step builders

**Returns:**
- **ValidationResult** – ValidationResult with detailed validation information

```python
# Validate DAG compatibility
validation_result = engine.validate_dag_compatibility(
    dag_nodes=["data_load", "training", "evaluation"],
    available_configs=available_configs,
    config_map=resolved_config_map,
    builder_registry=builder_registry
)

if validation_result.is_valid:
    print("✅ Validation passed")
else:
    print("❌ Validation failed")
    print(validation_result.detailed_report())
```

### ValidationResult

_class_ cursus.core.compiler.validation.ValidationResult(_is_valid_, _missing_configs_, _unresolvable_builders_, _config_errors_, _dependency_issues_, _warnings_)

Result of DAG-config compatibility validation.

**Parameters:**
- **is_valid** (_bool_) – Whether validation passed
- **missing_configs** (_List[str]_) – List of missing configuration names
- **unresolvable_builders** (_List[str]_) – List of unresolvable step builders
- **config_errors** (_Dict[str, List[str]]_) – Configuration errors by node
- **dependency_issues** (_List[str]_) – List of dependency issues
- **warnings** (_List[str]_) – List of validation warnings

```python
# Create validation result
result = ValidationResult(
    is_valid=False,
    missing_configs=["missing_node"],
    unresolvable_builders=["UnknownStep"],
    config_errors={"node1": ["Invalid parameter"]},
    dependency_issues=["Circular dependency"],
    warnings=["Low confidence resolution"]
)
```

#### summary

summary()

Human-readable validation summary.

**Returns:**
- **str** – Summary of validation results

```python
# Get validation summary
summary = validation_result.summary()
print(summary)
# Output: "❌ Validation failed: 1 missing configs, 1 unresolvable builders"
```

#### detailed_report

detailed_report()

Detailed validation report with recommendations.

**Returns:**
- **str** – Detailed validation report

```python
# Get detailed validation report
report = validation_result.detailed_report()
print(report)
# Output includes:
# - Missing configurations
# - Unresolvable step builders
# - Configuration errors
# - Dependency issues
# - Warnings
# - Recommendations for fixes
```

### ResolutionPreview

_class_ cursus.core.compiler.validation.ResolutionPreview(_node_config_map_, _config_builder_map_, _resolution_confidence_, _ambiguous_resolutions_, _recommendations_)

Preview of how DAG nodes will be resolved.

**Parameters:**
- **node_config_map** (_Dict[str, str]_) – Node to config type mapping
- **config_builder_map** (_Dict[str, str]_) – Config type to builder type mapping
- **resolution_confidence** (_Dict[str, float]_) – Confidence scores for resolutions
- **ambiguous_resolutions** (_List[str]_) – List of ambiguous resolutions
- **recommendations** (_List[str]_) – List of recommendations

```python
# Create resolution preview
preview = ResolutionPreview(
    node_config_map={"data_load": "CradleDataLoadConfig", "training": "XGBoostTrainingConfig"},
    config_builder_map={"CradleDataLoadConfig": "CradleDataLoadingBuilder"},
    resolution_confidence={"data_load": 0.95, "training": 0.87},
    ambiguous_resolutions=["training has 2 similar candidates"],
    recommendations=["Consider renaming 'training' for better matching"]
)
```

#### display

display()

Display-friendly resolution preview.

**Returns:**
- **str** – Formatted resolution preview

```python
# Display resolution preview
preview_text = preview.display()
print(preview_text)
# Output:
# Resolution Preview
# ==================================================
# 
# Node → Configuration Mappings:
#   🟢 data_load → CradleDataLoadConfig (confidence: 0.95)
#   🟡 training → XGBoostTrainingConfig (confidence: 0.87)
# 
# Configuration → Builder Mappings:
#   ✓ CradleDataLoadConfig → CradleDataLoadingBuilder
# 
# ⚠️  Ambiguous Resolutions:
#   - training has 2 similar candidates
# 
# 💡 Recommendations:
#   - Consider renaming 'training' for better matching
```

### ConversionReport

_class_ cursus.core.compiler.validation.ConversionReport(_pipeline_name_, _steps_, _resolution_details_, _avg_confidence_, _warnings_, _metadata_)

Report generated after successful pipeline conversion.

**Parameters:**
- **pipeline_name** (_str_) – Name of the generated pipeline
- **steps** (_List[str]_) – List of pipeline steps
- **resolution_details** (_Dict[str, Dict[str, Any]]_) – Detailed resolution information
- **avg_confidence** (_float_) – Average confidence score
- **warnings** (_List[str]_) – List of warnings
- **metadata** (_Dict[str, Any]_) – Additional metadata

```python
# Create conversion report
report = ConversionReport(
    pipeline_name="my-ml-pipeline",
    steps=["data_load", "training", "evaluation"],
    resolution_details={
        "data_load": {
            "config_type": "CradleDataLoadConfig",
            "builder_type": "CradleDataLoadingBuilder",
            "confidence": 0.95
        }
    },
    avg_confidence=0.89,
    warnings=["Low confidence for 'evaluation' step"],
    metadata={"dag_nodes": 3, "compilation_time": "2.3s"}
)
```

#### summary

summary()

Summary of conversion results.

**Returns:**
- **str** – Summary of conversion results

```python
# Get conversion summary
summary = report.summary()
print(summary)
# Output: "Pipeline 'my-ml-pipeline' created successfully with 3 steps (avg confidence: 0.89)"
```

#### detailed_report

detailed_report()

Detailed conversion report.

**Returns:**
- **str** – Detailed conversion report

```python
# Get detailed conversion report
detailed = report.detailed_report()
print(detailed)
# Output includes:
# - Pipeline name and step count
# - Average confidence score
# - Step resolution details
# - Warnings
# - Additional metadata
```

## Validation Process

The validation engine performs comprehensive checks:

### 1. Configuration Availability
Checks that all DAG nodes have corresponding configuration instances available.

### 2. Step Builder Resolution
Validates that all configuration types can be mapped to available step builders, including:
- Direct step type matching
- Job type variant handling
- Legacy alias resolution
- Special case handling for known step types

### 3. Configuration Validation
Runs individual configuration validation if available, checking:
- Required field presence
- Value constraints and formats
- Configuration-specific business rules

### 4. Dependency Resolution
Validates that DAG dependencies can be properly resolved (placeholder for future enhancement).

### 5. Warning Generation
Identifies potential issues that don't prevent compilation but may indicate problems:
- Low confidence resolutions
- Ambiguous node mappings
- Deprecated configuration patterns

## Error Categories

### Missing Configurations
DAG nodes that don't have corresponding configuration instances.

### Unresolvable Builders
Configuration types that can't be mapped to step builders.

### Configuration Errors
Individual configuration validation failures.

### Dependency Issues
Problems with DAG dependency resolution.

## Related Documentation

- [DAG Compiler](dag_compiler.md) - Uses ValidationEngine for DAG compatibility checking
- [Dynamic Template](dynamic_template.md) - Uses ValidationEngine during template creation
- [Compiler Exceptions](exceptions.md) - ValidationError raised for validation failures
- [Configuration Resolver](config_resolver.md) - Provides resolution data for validation
- [Compiler Overview](README.md) - System overview and integration

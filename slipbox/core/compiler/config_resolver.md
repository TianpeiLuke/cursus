---
tags:
  - code
  - core
  - compiler
  - configuration_resolution
  - dag_matching
keywords:
  - StepConfigResolver
  - configuration resolution
  - DAG node matching
  - job type matching
  - semantic matching
  - pattern matching
topics:
  - configuration management
  - DAG compilation
  - intelligent matching
language: python
date of note: 2025-09-07
---

# Configuration Resolver

Intelligent matching of DAG nodes to configuration instances using multiple resolution strategies with enhanced handling for job types and configuration variants.

## Overview

The `config_resolver` module provides intelligent matching of DAG nodes to configuration instances using multiple resolution strategies. The resolver implements a tiered approach that prioritizes exact matches and makes better use of node name patterns, job types, and semantic similarity to resolve DAG node names to appropriate configuration instances.

The module handles complex scenarios including job type variants, configuration ambiguity, and provides detailed diagnostic information for troubleshooting resolution issues. It supports multiple resolution strategies including direct name matching, metadata mapping, job type matching, semantic similarity, and pattern-based matching.

## Classes and Methods

### Classes
- [`StepConfigResolver`](#stepconfigresolver) - Main resolver for matching DAG nodes to configuration instances

## API Reference

### StepConfigResolver

_class_ cursus.core.compiler.config_resolver.StepConfigResolver(_confidence_threshold=0.7_)

Resolves DAG nodes to configuration instances using intelligent matching. This class implements multiple resolution strategies to match DAG node names to configuration instances from the loaded configuration file.

**Parameters:**
- **confidence_threshold** (_float_) – Minimum confidence score for automatic resolution (default: 0.7)

**Class Attributes:**
- **STEP_TYPE_PATTERNS** (_Dict[str, List[str]]_) – Pattern mappings for step type detection
- **JOB_TYPE_KEYWORDS** (_Dict[str, List[str]]_) – Job type keywords for matching

```python
from cursus.core.compiler.config_resolver import StepConfigResolver

# Create resolver with default confidence threshold
resolver = StepConfigResolver()

# Create resolver with custom confidence threshold
resolver = StepConfigResolver(confidence_threshold=0.8)
```

#### resolve_config_map

resolve_config_map(_dag_nodes_, _available_configs_, _metadata=None_)

Resolve DAG nodes to configurations with enhanced metadata handling. Resolution strategies (in order of preference): 1. Direct name matching with exact match, 2. Metadata mapping from config_types, 3. Job type + config type matching with pattern recognition, 4. Semantic similarity matching, 5. Pattern-based matching.

**Parameters:**
- **dag_nodes** (_List[str]_) – List of DAG node names
- **available_configs** (_Dict[str, BasePipelineConfig]_) – Available configuration instances
- **metadata** (_Optional[Dict[str, Any]]_) – Optional metadata from configuration file

**Returns:**
- **Dict[str, BasePipelineConfig]** – Dictionary mapping node names to configuration instances

**Raises:**
- **ConfigurationError** – If nodes cannot be resolved
- **AmbiguityError** – If multiple configs match with similar confidence

```python
# Resolve DAG nodes to configurations
dag_nodes = ["CradleDataLoading_training", "XGBoostTraining_training", "ModelEval_training"]
available_configs = {
    "training_data_load": training_data_config,
    "training_model": training_model_config,
    "evaluation": evaluation_config
}

# With metadata for enhanced resolution
metadata = {
    "config_types": {
        "CradleDataLoading_training": "CradleDataLoadConfig",
        "XGBoostTraining_training": "XGBoostTrainingConfig"
    }
}

resolved_map = resolver.resolve_config_map(dag_nodes, available_configs, metadata)

# Access resolved configurations
for node_name, config in resolved_map.items():
    print(f"Node '{node_name}' -> {type(config).__name__}")
```

#### preview_resolution

preview_resolution(_dag_nodes_, _available_configs_, _metadata=None_)

Preview resolution candidates for each DAG node with enhanced diagnostics.

**Parameters:**
- **dag_nodes** (_List[str]_) – List of DAG node names
- **available_configs** (_Dict[str, BasePipelineConfig]_) – Available configuration instances
- **metadata** (_Optional[Dict[str, Any]]_) – Optional metadata from configuration file

**Returns:**
- **Dict[str, Any]** – Dictionary with resolution preview information including node-to-config mapping and diagnostic recommendations

```python
# Preview resolution before actual resolution
preview = resolver.preview_resolution(dag_nodes, available_configs, metadata)

# Examine resolution details
for node, info in preview['node_resolution'].items():
    if 'error' in info:
        print(f"Node '{node}' failed to resolve: {info['error']}")
    else:
        print(f"Node '{node}' -> {info['config_type']} "
              f"(confidence: {info['confidence']:.2f}, method: {info['method']})")

# Check recommendations for ambiguous nodes
for recommendation in preview['recommendations']:
    print(f"Recommendation: {recommendation}")
```

## Resolution Strategies

The resolver implements multiple resolution strategies in order of preference:

### 1. Direct Name Matching
Exact matches between node names and configuration keys, including case-insensitive matching and metadata mapping.

### 2. Metadata Mapping
Uses `metadata.config_types` mapping to resolve nodes to specific configuration class names with job type validation.

### 3. Job Type Enhanced Matching
Matches configurations based on job type attributes with improved accuracy, supporting job type variants like training, calibration, evaluation.

### 4. Semantic Similarity Matching
Uses semantic mappings and sequence matching to find configurations with similar meaning or purpose.

### 5. Pattern-Based Matching
Uses regex patterns to match node names to configuration types based on common naming conventions.

## Pattern Mappings

The resolver includes comprehensive pattern mappings for common step types:

```python
STEP_TYPE_PATTERNS = {
    r'.*data_load.*': ['CradleDataLoading'],
    r'.*preprocess.*': ['TabularPreprocessing'],
    r'.*train.*': ['XGBoostTraining', 'PyTorchTraining', 'DummyTraining'],
    r'.*eval.*': ['XGBoostModelEval'],
    r'.*model.*': ['XGBoostModel', 'PyTorchModel'],
    r'.*calibrat.*': ['ModelCalibration'],
    r'.*packag.*': ['MIMSPackaging'],
    r'.*regist.*': ['ModelRegistration'],
    # ... additional patterns
}
```

## Job Type Keywords

The resolver recognizes common job type keywords for enhanced matching:

```python
JOB_TYPE_KEYWORDS = {
    'train': ['training', 'train'],
    'calib': ['calibration', 'calib'],
    'eval': ['evaluation', 'eval', 'test'],
    'inference': ['inference', 'infer', 'predict'],
    'validation': ['validation', 'valid'],
}
```

## Related Documentation

- [DAG Compiler](dag_compiler.md) - Primary consumer of the configuration resolver
- [Compiler Exceptions](exceptions.md) - Defines ConfigurationError and AmbiguityError exceptions
- [Dynamic Template](dynamic_template.md) - Uses resolved configurations for template generation
- [Compiler Overview](README.md) - System overview and integration

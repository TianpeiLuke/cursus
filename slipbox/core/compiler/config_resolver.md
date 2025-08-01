---
tags:
  - code
  - pipeline_api
  - config_resolution
  - matching_engine
keywords: 
  - configuration
  - resolver
  - matching
  - DAG
  - pipeline
topics: 
  - pipeline API
  - configuration resolution
language: python
date of note: 2025-07-31
---

# Config Resolver

## Purpose

The `StepConfigResolver` is an intelligent matching engine that maps DAG nodes to configuration instances using multiple resolution strategies. It bridges the gap between the abstract DAG structure and concrete step configurations.

## Core Problem Solved

When working with dynamically defined DAGs, there's a need to automatically connect each node in the DAG to an appropriate configuration instance. This is challenging because:

1. Node names may not exactly match configuration identifiers
2. Multiple configurations could potentially match a single node
3. The matching logic may need to consider various attributes beyond just names

## Resolution Strategies

The resolver employs multiple strategies in descending order of confidence:

### 1. Direct Name Matching (Confidence: 1.0)

Exact matching between a node name and a configuration identifier.

```python
# Example
dag.add_node("data_load_step")  # → matches config with ID "data_load_step"
```

### 2. Job Type Matching (Confidence: 0.7-1.0)

Matching based on the `job_type` attribute of configurations.

```python
# Example: Node "training_job" matches config with job_type="training"
dag.add_node("training_job")  # → XGBoostTrainingConfig(job_type="training")
```

### 3. Semantic Matching (Confidence: 0.5-0.8)

Using semantic similarity to match node names to configuration types.

```python
# Example: "preprocessing" semantically matches "TabularPreprocessingConfig"
dag.add_node("preprocessing")  # → TabularPreprocessingConfig
```

### 4. Pattern Matching (Confidence: 0.6-0.9)

Using regex patterns to identify step types from node names.

```python
# Example: "xgb_training" matches XGBoost training pattern
dag.add_node("xgb_training")  # → XGBoostTrainingConfig
```

## Implementation Details

The resolver:

1. Collects all available configurations from the configuration file
2. For each DAG node, applies the resolution strategies in order
3. Ranks potential matches by confidence score
4. Returns the highest confidence match for each node
5. Provides detailed resolution information for debugging

## Usage

```python
from src.pipeline_api.config_resolver import StepConfigResolver

# Create resolver
resolver = StepConfigResolver()

# Resolve node to config
config_map = resolver.resolve_config_map(
    dag_nodes=["data_load", "preprocess", "train"],
    available_configs=loaded_configs
)

# Preview resolution
preview = resolver.preview_resolution(
    dag_nodes=["data_load", "preprocess", "train"],
    available_configs=loaded_configs
)
```

## Customization

The resolver can be extended by subclassing and overriding specific resolution strategies:

```python
class CustomResolver(StepConfigResolver):
    def _semantic_matching(self, node_name, configs):
        # Custom semantic matching logic
        return super()._semantic_matching(node_name, configs)
```

## Key Features

1. **Multi-strategy approach**: Combines multiple resolution methods
2. **Confidence scoring**: Provides quantitative measure of match quality
3. **Flexible matching**: Handles various naming conventions and patterns
4. **Detailed feedback**: Provides resolution preview for debugging
5. **Extensible**: Can be customized with additional resolution strategies

This resolver is a key component of the dynamic template system, enabling automatic DAG-to-configuration mapping without requiring manual step-by-step definitions.

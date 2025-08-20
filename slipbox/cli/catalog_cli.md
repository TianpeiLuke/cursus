---
tags:
  - code
  - cli
  - pipeline_catalog
  - documentation
  - user_guide
keywords:
  - catalog CLI
  - pipeline catalog
  - MODS integration
  - dual-compiler
  - command-line interface
  - pipeline management
  - registry integration
topics:
  - CLI documentation
  - pipeline catalog management
  - MODS functionality
  - command reference
language: python
date of note: 2025-08-20
---

# Pipeline Catalog CLI Documentation

## Overview

The Pipeline Catalog CLI provides a comprehensive command-line interface for managing, browsing, and working with the pipeline catalog. It supports both standard pipelines and MODS-enhanced pipelines through a dual-compiler architecture, offering powerful search, filtering, and registry integration capabilities.

## Architecture

The catalog CLI is built on a dual-compiler architecture that supports:

- **Standard Pipelines**: Traditional DAG compilation using `PipelineDAGCompiler`
- **MODS Pipelines**: Enhanced compilation using `MODSPipelineDAGCompiler` with registry integration
- **Shared DAG Definitions**: Consistent pipeline definitions across both compiler types
- **Graceful Fallback**: Seamless operation when MODS is unavailable

## Installation and Setup

The catalog CLI is available through the main cursus CLI interface:

```bash
# Basic usage
cursus catalog <command> [options]

# Get help
cursus catalog --help
```

## Core Commands

### 1. List Pipelines

List all available pipelines in the catalog.

```bash
# Basic listing
cursus catalog list

# JSON output
cursus catalog list --format json

# Sort by different fields
cursus catalog list --sort name
cursus catalog list --sort framework
cursus catalog list --sort complexity
```

**Output Example:**
```
ID                   NAME                           FRAMEWORK    COMPLEXITY
------------------------------------------------------------------------
xgboost-simple       Simple XGBoost Pipeline        xgboost      simple
xgboost-standard     Standard XGBoost Pipeline      xgboost      standard
pytorch-simple       Simple PyTorch Pipeline        pytorch      simple

Total: 14 pipelines
```

### 2. Search Pipelines

Search pipelines using various criteria.

```bash
# Search by framework
cursus catalog search --framework xgboost

# Search by complexity
cursus catalog search --complexity simple

# Search by features (can use multiple)
cursus catalog search --feature training --feature evaluation

# Search by tags
cursus catalog search --tag ml --tag classification

# Combine multiple filters
cursus catalog search --framework pytorch --complexity standard --feature training
```

**Advanced Search with Dual-Compiler Support:**

```bash
# Enhanced search with compiler type filtering
cursus catalog search-enhanced --compiler-type mods

# Filter by MODS-specific features
cursus catalog search-enhanced --mods-feature template_registration

# Combine standard and MODS filters
cursus catalog search-enhanced --framework xgboost --compiler-type mods --mods-feature operational_integration
```

### 3. Show Pipeline Details

Display detailed information about a specific pipeline.

```bash
# Show pipeline details
cursus catalog show xgboost-simple

# JSON output
cursus catalog show xgboost-simple --format json
```

**Output Example:**
```
Simple XGBoost Pipeline
=======================
ID: xgboost-simple
Framework: xgboost
Complexity: simple
Features: training, evaluation
Tags: ml, classification, tabular
Path: pipelines/xgboost/simple.py

Description:
A simple XGBoost pipeline for binary classification tasks. Includes data
preprocessing, model training, and evaluation steps.

Usage Example:
from cursus.pipeline_catalog import get_pipeline
pipeline = get_pipeline('xgboost-simple')
```

### 4. Generate Pipeline

Generate a pipeline file from a template.

```bash
# Generate pipeline to specific location
cursus catalog generate xgboost-simple --output my_pipeline.py

# Generate with custom name
cursus catalog generate xgboost-simple --output my_pipeline.py --rename "My Custom Pipeline"
```

### 5. Index Management

Manage the pipeline catalog index.

```bash
# Update the index
cursus catalog update-index

# Force full regeneration
cursus catalog update-index --force

# Validate only (no changes)
cursus catalog update-index --validate-only

# Validate the index
cursus catalog validate
```

## MODS-Specific Commands

The CLI provides specialized commands for MODS (Model Operations Data Science) functionality.

### 1. List MODS Pipelines

```bash
# List all MODS pipelines
cursus catalog mods list

# JSON output
cursus catalog mods list --format json

# Sort by different fields
cursus catalog mods list --sort framework
```

**Output Example:**
```
MODS Pipelines:
ID                        NAME                                FRAMEWORK    COMPLEXITY
----------------------------------------------------------------------------------
xgboost-simple-mods       Simple XGBoost Pipeline (MODS)     xgboost      simple
pytorch-standard-mods     Standard PyTorch Pipeline (MODS)   pytorch      standard

Total: 6 MODS pipelines
MODS features available: metadata_extraction, operational_integration, template_registration
```

### 2. MODS Registry Status

Check the status of the MODS registry integration.

```bash
# Show registry status
cursus catalog mods registry-status

# JSON output
cursus catalog mods registry-status --format json
```

**Output Example:**
```
MODS Registry Status:
====================
Available: Yes
Connection Status: connected
Last Sync: 2025-08-20T09:30:00Z
Template Count: 42
Registry Version: 2.1.0
```

### 3. List Registry Templates

List templates available in the MODS registry.

```bash
# List all registry templates
cursus catalog mods list-registry

# Filter by framework
cursus catalog mods list-registry --framework pytorch

# JSON output
cursus catalog mods list-registry --format json
```

### 4. Check MODS Integration

Comprehensive check of MODS integration status.

```bash
# Check integration status
cursus catalog mods check-registry

# JSON output
cursus catalog mods check-registry --format json
```

**Output Example:**
```
MODS Integration Status:
========================
MODS Available: Yes
Integration Status: fully_operational
Registry Available: Yes
Registry Connection: connected
Registry Templates: 42
Cache Entries: 15
Valid Cache Entries: 15
```

### 5. Show Pipeline Pairs

Display standard/MODS pipeline pairs to understand relationships.

```bash
# Show all pipeline pairs
cursus catalog mods pairs

# JSON output
cursus catalog mods pairs --format json
```

**Output Example:**
```
Standard/MODS Pipeline Pairs:
STANDARD ID              MODS ID                      FRAMEWORK    SHARED DAG
----------------------------------------------------------------------------------------
xgboost-simple           xgboost-simple-mods          xgboost      xgboost_simple_dag
pytorch-standard         pytorch-standard-mods        pytorch      pytorch_standard_dag

Total: 8 pipeline pairs
With MODS variant: 6
Without MODS variant: 2
```

## Output Formats

Most commands support multiple output formats:

- **table** (default): Human-readable tabular format
- **json**: Machine-readable JSON format
- **text**: Plain text format (for show commands)

```bash
# Examples of different formats
cursus catalog list --format table
cursus catalog list --format json
cursus catalog show pipeline-id --format text
cursus catalog show pipeline-id --format json
```

## Filtering and Search Options

### Standard Filters

- `--framework`: Filter by ML framework (xgboost, pytorch, sklearn, etc.)
- `--complexity`: Filter by complexity level (simple, standard, advanced)
- `--feature`: Filter by pipeline features (can be used multiple times)
- `--tag`: Filter by tags (can be used multiple times)

### MODS-Specific Filters

- `--compiler-type`: Filter by compiler type (standard, mods)
- `--mods-feature`: Filter by MODS-specific features (can be used multiple times)

### Example Complex Searches

```bash
# Find all simple XGBoost pipelines with training capability
cursus catalog search --framework xgboost --complexity simple --feature training

# Find all MODS pipelines with template registration
cursus catalog search-enhanced --compiler-type mods --mods-feature template_registration

# Find PyTorch pipelines with both standard and MODS variants
cursus catalog search-enhanced --framework pytorch
```

## Error Handling and Troubleshooting

### Common Issues

1. **Index Not Found**
   ```bash
   # Solution: Generate the index
   cursus catalog update-index
   ```

2. **MODS Not Available**
   ```bash
   # Check MODS integration status
   cursus catalog mods check-registry
   
   # MODS commands will gracefully fall back to standard functionality
   ```

3. **Registry Connection Issues**
   ```bash
   # Check registry status
   cursus catalog mods registry-status
   
   # The system will use cached data when registry is unavailable
   ```

### Validation

```bash
# Validate the catalog index
cursus catalog validate

# Update and validate in one step
cursus catalog update-index --validate-only
```

## Integration with Python Code

The CLI commands correspond to Python API functions:

```python
# Equivalent Python code for CLI commands
from cursus.pipeline_catalog.utils import (
    load_index,
    get_pipeline_by_id,
    filter_pipelines,
    get_mods_pipelines,
    get_pipeline_pairs
)

# List all pipelines
index = load_index()
pipelines = index['pipelines']

# Get specific pipeline
pipeline = get_pipeline_by_id('xgboost-simple')

# Search pipelines
results = filter_pipelines(framework='xgboost', complexity='simple')

# MODS-specific operations
mods_pipelines = get_mods_pipelines()
pairs = get_pipeline_pairs()
```

## Performance Considerations

### Caching

- MODS registry data is cached for 5 minutes to improve performance
- Index validation results are cached during CLI session
- Large searches are optimized for memory usage

### Best Practices

1. **Use specific filters** to reduce search time
2. **Cache registry status** for batch operations
3. **Use JSON output** for programmatic processing
4. **Validate index periodically** to ensure consistency

## Advanced Usage

### Batch Operations

```bash
# Generate multiple pipelines
for pipeline in xgboost-simple pytorch-simple; do
    cursus catalog generate $pipeline --output "${pipeline}.py"
done

# Search and process results
cursus catalog search --framework xgboost --format json | jq '.[] | .id'
```

### Integration with CI/CD

```bash
# Validate catalog in CI pipeline
cursus catalog validate || exit 1

# Check MODS integration in deployment
cursus catalog mods check-registry --format json | jq '.integration_status'
```

### Custom Workflows

```bash
# Find pipelines without MODS variants
cursus catalog mods pairs --format json | jq '.[] | select(.mods == null) | .standard.id'

# List all available frameworks
cursus catalog list --format json | jq -r '.[].framework' | sort -u
```

## Configuration

The catalog CLI respects the following configuration:

- **Index Location**: `src/cursus/pipeline_catalog/index.json`
- **Pipeline Directory**: `src/cursus/pipeline_catalog/pipelines/`
- **MODS Registry**: Configured through MODS package settings
- **Cache Settings**: 5-minute timeout for registry data

## Development and Extension

### Adding New Commands

To add new CLI commands:

1. Add parser configuration in `setup_parser()`
2. Implement the command function
3. Add routing in `main()`
4. Update this documentation

### Custom Filters

New filter types can be added by:

1. Extending the utility functions in `utils.py`
2. Adding parser arguments
3. Updating the search functions

## Conclusion

The Pipeline Catalog CLI provides a comprehensive interface for managing both standard and MODS-enhanced pipelines. Its dual-compiler architecture ensures compatibility while providing advanced functionality for MODS-enabled environments.

The CLI is designed for both interactive use and automation, with consistent output formats and robust error handling. Whether you're exploring available pipelines, managing the catalog, or integrating with MODS registry, the CLI provides the tools needed for effective pipeline management.

For additional help with specific commands, use the `--help` flag:

```bash
cursus catalog --help
cursus catalog <command> --help
cursus catalog mods --help

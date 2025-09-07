---
tags:
  - code
  - cli
  - pipeline_catalog
  - dual-compiler
  - mods-integration
keywords:
  - catalog CLI
  - pipeline catalog
  - MODS integration
  - dual-compiler
  - command-line interface
  - pipeline management
  - registry integration
topics:
  - pipeline catalog management
  - MODS functionality
  - dual-compiler architecture
  - CLI tools
language: python
date of note: 2024-12-07
---

# Pipeline Catalog CLI

Command-line interface for managing, browsing, and working with the pipeline catalog, supporting both standard pipelines and MODS-enhanced pipelines through a dual-compiler architecture.

## Overview

The Pipeline Catalog CLI provides a comprehensive command-line interface for pipeline catalog management with dual-compiler architecture support. It enables powerful search, filtering, and registry integration capabilities for both standard pipelines using PipelineDAGCompiler and MODS-enhanced pipelines using MODSPipelineDAGCompiler with registry integration.

The CLI features shared DAG definitions for consistent pipeline definitions across both compiler types, graceful fallback for seamless operation when MODS is unavailable, and comprehensive MODS integration with template registration and operational integration capabilities.

## Commands and Functions

### list

cursus catalog list [_options_]

List all available pipelines in the catalog with sorting and formatting options.

**Options:**
- **--format** (_str_) – Output format (table, json)
- **--sort** (_Optional[str]_) – Sort by field (name, framework, complexity)

**Returns:**
- **int** – Exit code (0 for success, 1 for error)

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

### search

cursus catalog search [_options_]

Search pipelines using various criteria with multiple filter support.

**Options:**
- **--framework** (_Optional[str]_) – Filter by ML framework (xgboost, pytorch, sklearn)
- **--complexity** (_Optional[str]_) – Filter by complexity level (simple, standard, advanced)
- **--feature** (_List[str]_) – Filter by pipeline features (can be used multiple times)
- **--tag** (_List[str]_) – Filter by tags (can be used multiple times)
- **--format** (_str_) – Output format (table, json)

**Returns:**
- **int** – Exit code (0 for success, 1 for error)

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

### search-enhanced

cursus catalog search-enhanced [_options_]

Advanced search with dual-compiler support and MODS-specific filtering.

**Options:**
- **--compiler-type** (_Optional[str]_) – Filter by compiler type (standard, mods)
- **--mods-feature** (_List[str]_) – Filter by MODS-specific features (can be used multiple times)
- **--framework** (_Optional[str]_) – Filter by ML framework
- **--format** (_str_) – Output format (table, json)

**Returns:**
- **int** – Exit code (0 for success, 1 for error)

```bash
# Enhanced search with compiler type filtering
cursus catalog search-enhanced --compiler-type mods

# Filter by MODS-specific features
cursus catalog search-enhanced --mods-feature template_registration

# Combine standard and MODS filters
cursus catalog search-enhanced --framework xgboost --compiler-type mods --mods-feature operational_integration
```

### show

cursus catalog show _pipeline_id_ [_options_]

Display detailed information about a specific pipeline.

**Parameters:**
- **pipeline_id** (_str_) – ID of the pipeline to display

**Options:**
- **--format** (_str_) – Output format (text, json)

**Returns:**
- **int** – Exit code (0 for success, 1 for error)

```bash
# Show pipeline details
cursus catalog show xgboost-simple

# JSON output
cursus catalog show xgboost-simple --format json
```

### generate

cursus catalog generate _pipeline_id_ [_options_]

Generate a pipeline file from a template.

**Parameters:**
- **pipeline_id** (_str_) – ID of the pipeline template to generate

**Options:**
- **--output** (_str_) – Output file path for generated pipeline
- **--rename** (_Optional[str]_) – Custom name for the generated pipeline

**Returns:**
- **int** – Exit code (0 for success, 1 for error)

```bash
# Generate pipeline to specific location
cursus catalog generate xgboost-simple --output my_pipeline.py

# Generate with custom name
cursus catalog generate xgboost-simple --output my_pipeline.py --rename "My Custom Pipeline"
```

### update-index

cursus catalog update-index [_options_]

Manage the pipeline catalog index with update and validation options.

**Options:**
- **--force** (_bool_) – Force full regeneration of index
- **--validate-only** (_bool_) – Validate only without making changes

**Returns:**
- **int** – Exit code (0 for success, 1 for error)

```bash
# Update the index
cursus catalog update-index

# Force full regeneration
cursus catalog update-index --force

# Validate only (no changes)
cursus catalog update-index --validate-only
```

### validate

cursus catalog validate [_options_]

Validate the pipeline catalog index for consistency and integrity.

**Options:**
- **--format** (_str_) – Output format (text, json)

**Returns:**
- **int** – Exit code (0 for success, 1 for error)

```bash
# Validate the index
cursus catalog validate

# JSON validation output
cursus catalog validate --format json
```

## MODS Commands

### mods list

cursus catalog mods list [_options_]

List all MODS-enhanced pipelines with sorting and formatting options.

**Options:**
- **--format** (_str_) – Output format (table, json)
- **--sort** (_Optional[str]_) – Sort by field (framework, name, complexity)

**Returns:**
- **int** – Exit code (0 for success, 1 for error)

```bash
# List all MODS pipelines
cursus catalog mods list

# JSON output
cursus catalog mods list --format json

# Sort by different fields
cursus catalog mods list --sort framework
```

### mods registry-status

cursus catalog mods registry-status [_options_]

Check the status of the MODS registry integration.

**Options:**
- **--format** (_str_) – Output format (text, json)

**Returns:**
- **int** – Exit code (0 for success, 1 for error)

```bash
# Show registry status
cursus catalog mods registry-status

# JSON output
cursus catalog mods registry-status --format json
```

### mods list-registry

cursus catalog mods list-registry [_options_]

List templates available in the MODS registry.

**Options:**
- **--framework** (_Optional[str]_) – Filter by framework
- **--format** (_str_) – Output format (table, json)

**Returns:**
- **int** – Exit code (0 for success, 1 for error)

```bash
# List all registry templates
cursus catalog mods list-registry

# Filter by framework
cursus catalog mods list-registry --framework pytorch

# JSON output
cursus catalog mods list-registry --format json
```

### mods check-registry

cursus catalog mods check-registry [_options_]

Comprehensive check of MODS integration status.

**Options:**
- **--format** (_str_) – Output format (text, json)

**Returns:**
- **int** – Exit code (0 for success, 1 for error)

```bash
# Check integration status
cursus catalog mods check-registry

# JSON output
cursus catalog mods check-registry --format json
```

### mods pairs

cursus catalog mods pairs [_options_]

Display standard/MODS pipeline pairs to understand relationships.

**Options:**
- **--format** (_str_) – Output format (table, json)

**Returns:**
- **int** – Exit code (0 for success, 1 for error)

```bash
# Show all pipeline pairs
cursus catalog mods pairs

# JSON output
cursus catalog mods pairs --format json
```

## Architecture

### Dual-Compiler Support

The catalog CLI is built on a dual-compiler architecture:

- **Standard Pipelines** – Traditional DAG compilation using PipelineDAGCompiler
- **MODS Pipelines** – Enhanced compilation using MODSPipelineDAGCompiler with registry integration
- **Shared DAG Definitions** – Consistent pipeline definitions across both compiler types
- **Graceful Fallback** – Seamless operation when MODS is unavailable

### MODS Integration Features

- **Template Registration** – Integration with MODS template registry
- **Operational Integration** – Enhanced operational capabilities
- **Metadata Extraction** – Automatic metadata extraction from pipelines
- **Registry Caching** – 5-minute cache timeout for improved performance

## Output Formats

### Supported Formats

- **table** (_default_) – Human-readable tabular format
- **json** – Machine-readable JSON format
- **text** – Plain text format (for show commands)

### Format Examples

```bash
# Examples of different formats
cursus catalog list --format table
cursus catalog list --format json
cursus catalog show pipeline-id --format text
cursus catalog show pipeline-id --format json
```

## Filtering Options

### Standard Filters

- **--framework** – Filter by ML framework (xgboost, pytorch, sklearn, etc.)
- **--complexity** – Filter by complexity level (simple, standard, advanced)
- **--feature** – Filter by pipeline features (can be used multiple times)
- **--tag** – Filter by tags (can be used multiple times)

### MODS-Specific Filters

- **--compiler-type** – Filter by compiler type (standard, mods)
- **--mods-feature** – Filter by MODS-specific features (can be used multiple times)

## Error Handling

### Exit Codes

- **0** – Success
- **1** – General error or command failure
- **2** – Index not found or invalid
- **3** – MODS integration error
- **4** – Registry connection error

### Common Issues

- **Index Not Found** – Solution: Generate the index with `cursus catalog update-index`
- **MODS Not Available** – MODS commands gracefully fall back to standard functionality
- **Registry Connection Issues** – System uses cached data when registry is unavailable

## Integration Points

### Python API Integration

The CLI commands correspond to Python API functions from `cursus.pipeline_catalog.utils`:

- **load_index()** – Load pipeline catalog index
- **get_pipeline_by_id()** – Get specific pipeline by ID
- **filter_pipelines()** – Search and filter pipelines
- **get_mods_pipelines()** – Get MODS-enhanced pipelines
- **get_pipeline_pairs()** – Get standard/MODS pipeline pairs

### Configuration Integration

- **Index Location** – `src/cursus/pipeline_catalog/index.json`
- **Pipeline Directory** – `src/cursus/pipeline_catalog/pipelines/`
- **MODS Registry** – Configured through MODS package settings
- **Cache Settings** – 5-minute timeout for registry data

## Performance Considerations

### Caching Strategy

- MODS registry data is cached for 5 minutes to improve performance
- Index validation results are cached during CLI session
- Large searches are optimized for memory usage

### Best Practices

1. Use specific filters to reduce search time
2. Cache registry status for batch operations
3. Use JSON output for programmatic processing
4. Validate index periodically to ensure consistency

## Related Documentation

- [Pipeline Catalog Utils](../pipeline_catalog/utils.md) - Utility functions for catalog operations
- [MODS Integration](../mods/integration.md) - MODS integration documentation
- [Registry CLI](registry_cli.md) - Registry management tools
- [Dual Compiler Design](../../1_design/dual_compiler_architecture.md) - Dual-compiler architecture design

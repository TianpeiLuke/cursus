---
tags:
  - design
  - organization
  - pipeline_catalog
  - discoverability
  - structure
keywords:
  - pipeline catalog
  - directory structure
  - organization
  - discoverability
  - framework organization
  - indexing
  - search
  - CLI tools
  - pipeline selection
topics:
  - catalog design
  - pipeline organization
  - user interface
  - discovery system
language: python
date of note: 2025-08-19
---

# Pipeline Catalog Design

## Overview

The Pipeline Catalog is designed to be a comprehensive collection of prebuilt pipelines that users can select from based on their specific needs. This design document outlines the structure and organization of the catalog to make it easy for users to discover and choose the appropriate pipeline for their use case.

## Package Integration

The Pipeline Catalog should be placed under `src/cursus/pipeline_catalog/` to include it as part of the main package. This has several advantages:

1. Makes the pipeline catalog available to users when they install the Cursus package
2. Allows direct importing of pipelines as modules (e.g., `from cursus.pipeline_catalog.frameworks.xgboost import simple`)
3. Enables seamless CLI integration through the package's command system
4. Provides consistent versioning and distribution alongside the core package
5. Makes testing and maintenance more straightforward as part of the standard package workflows

## Folder Structure

```
src/cursus/pipeline_catalog/
├── README.md                          # Main index and guide
├── index.json                         # Machine-readable index for tooling
├── utils.py                           # Utilities for working with the catalog
├── frameworks/
│   ├── xgboost/                       # XGBoost-specific pipelines
│   │   ├── simple.py                  # Basic XGBoost training
│   │   ├── training/
│   │   │   ├── basic_training.py      # Simple training
│   │   │   ├── with_calibration.py    # Training with calibration
│   │   │   └── with_evaluation.py     # Training with evaluation
│   │   ├── end_to_end/
│   │   │   ├── standard_e2e.py        # Standard end-to-end workflow
│   │   │   └── complete_e2e.py        # Complete with calibration
│   │   └── components/
│   │       ├── data_loading.py        # Data loading only
│   │       └── preprocessing.py       # Preprocessing only
│   │
│   └── pytorch/                       # PyTorch-specific pipelines
│       ├── simple.py                  # Basic PyTorch training
│       ├── training/
│       │   └── basic_training.py      # Simple training
│       └── end_to_end/
│           └── standard_e2e.py        # End-to-end PyTorch
│
├── tasks/                             # Task-oriented views
│   ├── training/                      # Training-focused pipelines
│   │   ├── xgboost_training.py        # -> symlink to frameworks/xgboost/training/basic_training.py
│   │   └── pytorch_training.py        # -> symlink to frameworks/pytorch/training/basic_training.py
│   ├── evaluation/                    # Evaluation-focused pipelines
│   │   └── xgboost_evaluation.py      # -> symlink to relevant pipeline
│   ├── registration/                  # Model registration pipelines
│   │   ├── xgboost_register.py        # -> symlink to registration pipeline
│   │   └── pytorch_register.py        # -> symlink to registration pipeline
│   └── data_processing/               # Data processing pipelines
│       └── cradle_loading.py          # -> symlink to data loading pipeline
│
└── examples/                          # Special case pipelines
    └── dummy_training.py              # Simple dummy example
```

## Documentation and Indexing

### Main README.md

The main `README.md` serves as the entry point with:

1. **Overview table** with all pipelines organized by:
   - Framework (XGBoost, PyTorch)
   - Complexity (Simple, Standard, Complete)
   - Features (Training, Evaluation, Calibration, Registration)

2. **Decision tree** to help users select:
   ```
   Start → Which framework? → What components do you need? → How complex? → Recommended pipeline
   ```

3. **Usage examples** showing how to:
   - Import and use pipelines
   - Customize configurations
   - Execute pipelines

### Machine-readable Index (index.json)

A structured JSON file that tools can consume:

```json
{
  "pipelines": [
    {
      "id": "xgboost-simple",
      "name": "XGBoost Simple Pipeline",
      "path": "frameworks/xgboost/simple.py",
      "framework": "xgboost",
      "complexity": "simple",
      "features": ["training"],
      "description": "Basic XGBoost training pipeline",
      "tags": ["xgboost", "training", "beginner"]
    },
    {
      "id": "xgboost-complete",
      "name": "XGBoost Complete E2E Pipeline",
      "path": "frameworks/xgboost/end_to_end/complete_e2e.py",
      "framework": "xgboost",
      "complexity": "advanced",
      "features": ["training", "calibration", "evaluation", "registration"],
      "description": "Complete end-to-end XGBoost pipeline with calibration",
      "tags": ["xgboost", "end-to-end", "calibration", "registration"]
    }
    // Additional entries for all pipelines
  ]
}
```

### Per-Pipeline Documentation

Each pipeline file will include:
- Detailed docstrings explaining purpose
- Usage examples
- Configuration requirements
- Diagram of pipeline topology (as ASCII or link to image)
- Expected inputs/outputs

## Integration with CLI/Tools

The CLI functionality is placed in the main CLI module (`src/cursus/cli/catalog_cli.py`), following the project's organization standards for command-line interfaces:

```bash
# List all pipelines
python -m cursus.cli.catalog_cli list

# Search by criteria
python -m cursus.cli.catalog_cli search --framework xgboost --feature calibration

# Show details of a specific pipeline
python -m cursus.cli.catalog_cli show xgboost-complete

# Generate a pipeline template based on a catalog entry
python -m cursus.cli.catalog_cli generate xgboost-complete --output my_pipeline.py
```

## Categorization Tags

Standardized tags across all pipelines for consistent discovery:

- **Framework**: xgboost, pytorch
- **Complexity**: simple, standard, advanced
- **Features**: training, evaluation, calibration, registration, data_loading, preprocessing
- **Use Case**: tabular, image, text (for future expansion)

## Implementation Plan

To implement this design:

1. Create the new folder structure
2. Move and rename existing files according to the structure
3. Create the documentation and indexes
4. Add appropriate tagging in each file
5. Develop the CLI tools in `src/cursus/cli/catalog_cli.py` for discovery

## Benefits

This design provides several advantages:

1. **Discoverability**: Users can easily find pipelines that match their specific requirements.
2. **Organization**: Clear separation by framework and functionality.
3. **Consistency**: Standardized structure and documentation.
4. **Extensibility**: Easy to add new pipelines or frameworks.
5. **Multiple Entry Points**: Users can approach from framework perspective or task perspective.
6. **Tooling Support**: Machine-readable indexes enable integration with developer tools.
7. **Standardized CLI**: Integration with the main CLI module provides a consistent user experience.

## Migration from Current Structure

The current examples folder will be reorganized into this new pipeline catalog structure, with files moved to appropriate locations in the new hierarchy. This process should be done carefully to preserve all functionality while improving the organization and discoverability of the pipelines.

### Migration Steps

1. Create the new directory structure under `src/cursus/pipeline_catalog/`
2. Update imports and references to reflect the new package structure
3. Ensure all files include proper `__init__.py` files for package organization
4. Test imports and functionality from the new package structure
5. Update documentation references to point to the new locations
6. Develop the CLI tool integration with the main package's command system

## Knowledge Management Principles

This pipeline catalog design embodies several key knowledge management principles documented in our [Zettelkasten Knowledge Management Principles](zettelkasten_knowledge_management_principles.md):

### Atomicity in Pipeline Organization
- **Single-purpose pipelines**: Each pipeline file focuses on one specific workflow (e.g., `simple.py`, `basic_training.py`)
- **Modular components**: Separate files for distinct concerns (data loading, preprocessing, training)
- **Clear boundaries**: Each pipeline has a well-defined scope and responsibility

### Connectivity Through Multiple Access Paths
- **Framework-based organization**: Primary structure organized by ML framework
- **Task-based views**: Alternative access through task-oriented symlinks
- **Cross-references**: Machine-readable index enables discovery of related pipelines
- **Explicit relationships**: JSON index documents connections between pipelines

### Emergent Organization Over Rigid Categories
- **Flexible tagging**: Pipelines can belong to multiple categories through tags
- **Multiple classification schemes**: Framework-based and task-based views coexist
- **Organic growth**: New frameworks and tasks can be added without restructuring
- **Tag-based discovery**: Users can find pipelines through multiple dimensions

### Manual Curation with Search Support
- **Curated documentation**: Human-written descriptions and usage examples
- **Structured metadata**: Machine-readable index for tooling integration
- **CLI discovery tools**: Manual curation enhanced by search capabilities
- **Decision trees**: Guided selection process for pipeline discovery

This design demonstrates practical application of Zettelkasten principles to software organization, creating a system that balances structure with flexibility and supports both human navigation and automated tooling.

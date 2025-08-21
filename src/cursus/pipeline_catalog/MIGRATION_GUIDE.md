# Pipeline Catalog Migration Guide

This guide helps you migrate from the old hierarchical pipeline catalog structure to the new Zettelkasten-based flat structure.

## Overview of Changes

The pipeline catalog has been completely restructured from a 5-level deep hierarchy to a flat, connection-based system using Zettelkasten knowledge management principles.

### Key Changes

1. **Flat Structure**: Maximum 3 levels deep instead of 5
2. **Semantic Naming**: Pipelines use descriptive names like `xgb_training_simple`
3. **Connection-Based Discovery**: Find pipelines through relationships, not folders
4. **Enhanced Metadata**: Rich metadata with tags, connections, and use cases
5. **MODS Integration**: Enhanced pipelines with operational capabilities
6. **Registry System**: Centralized catalog with automatic synchronization

## Import Path Changes

### Standard Pipelines

| Old Import Path | New Import Path | Pipeline ID |
|----------------|-----------------|-------------|
| `from cursus.pipeline_catalog.frameworks.xgboost.simple import create_pipeline` | `from cursus.pipeline_catalog.pipelines.xgb_training_simple import create_pipeline` | `xgb_training_simple` |
| `from cursus.pipeline_catalog.frameworks.xgboost.training.with_calibration import create_pipeline` | `from cursus.pipeline_catalog.pipelines.xgb_training_calibrated import create_pipeline` | `xgb_training_calibrated` |
| `from cursus.pipeline_catalog.frameworks.pytorch.training.basic_training import create_pipeline` | `from cursus.pipeline_catalog.pipelines.pytorch_training_basic import create_pipeline` | `pytorch_training_basic` |
| `from cursus.pipeline_catalog.frameworks.xgboost.training.with_evaluation import create_pipeline` | `from cursus.pipeline_catalog.pipelines.xgb_training_evaluation import create_pipeline` | `xgb_training_evaluation` |
| `from cursus.pipeline_catalog.frameworks.xgboost.end_to_end.complete_e2e import create_pipeline` | `from cursus.pipeline_catalog.pipelines.xgb_e2e_comprehensive import create_pipeline` | `xgb_e2e_comprehensive` |
| `from cursus.pipeline_catalog.frameworks.pytorch.end_to_end.standard_e2e import create_pipeline` | `from cursus.pipeline_catalog.pipelines.pytorch_e2e_standard import create_pipeline` | `pytorch_e2e_standard` |

### MODS Pipelines (New)

MODS-enhanced pipelines are now available with enhanced operational capabilities:

| MODS Pipeline | Standard Equivalent | Description |
|---------------|-------------------|-------------|
| `cursus.pipeline_catalog.mods_pipelines.xgb_mods_training_simple` | `xgb_training_simple` | MODS-enhanced XGBoost simple training |
| `cursus.pipeline_catalog.mods_pipelines.xgb_mods_training_calibrated` | `xgb_training_calibrated` | MODS-enhanced XGBoost with calibration |
| `cursus.pipeline_catalog.mods_pipelines.pytorch_mods_training_basic` | `pytorch_training_basic` | MODS-enhanced PyTorch basic training |
| `cursus.pipeline_catalog.mods_pipelines.xgb_mods_training_evaluation` | `xgb_training_evaluation` | MODS-enhanced XGBoost with evaluation |
| `cursus.pipeline_catalog.mods_pipelines.xgb_mods_e2e_comprehensive` | `xgb_e2e_comprehensive` | MODS-enhanced XGBoost comprehensive |
| `cursus.pipeline_catalog.mods_pipelines.pytorch_mods_e2e_standard` | `pytorch_e2e_standard` | MODS-enhanced PyTorch standard |

## Function Signature Changes

### Standard Pipelines

The function signatures remain largely the same, but with enhanced return values:

```python
# OLD
def create_pipeline(config_path, session, role, pipeline_name=None, pipeline_description=None):
    return pipeline, report, dag_compiler

# NEW
def create_pipeline(config_path, session, role, pipeline_name=None, pipeline_description=None, validate=True):
    return pipeline, report, dag_compiler, pipeline_template
```

### MODS Pipelines

MODS pipelines have additional parameters and return values:

```python
def create_pipeline(config_path, session, role, pipeline_name=None, pipeline_description=None, 
                   enable_mods=True, validate=True):
    return pipeline, report, dag_compiler, template_instance
```

### New Functions

All pipelines now include these additional functions:

```python
def get_enhanced_dag_metadata() -> Dict[str, Any]:
    """Get enhanced DAG metadata with Zettelkasten integration."""
    
def sync_to_registry() -> bool:
    """Synchronize pipeline metadata to the catalog registry."""
```

## CLI Command Changes

### Old CLI Commands

```bash
# OLD - Basic commands
python -m cursus.cli.catalog_cli list
python -m cursus.cli.catalog_cli search --framework xgboost
python -m cursus.cli.catalog_cli show xgboost-simple
```

### New CLI Commands

The CLI has been completely redesigned with Zettelkasten principles:

#### Registry Management
```bash
# Validate registry integrity
cursus catalog registry validate

# Show registry statistics
cursus catalog registry stats

# Export pipeline metadata
cursus catalog registry export --pipelines xgb_training_simple,pytorch_training_basic
```

#### Discovery Commands
```bash
# Find pipelines by tags
cursus catalog find --tags training,xgboost

# Search by framework and complexity
cursus catalog find --framework pytorch --complexity standard

# Search by use case
cursus catalog find --use-case "Basic XGBoost training workflow"

# Find MODS-compatible pipelines
cursus catalog find --mods-compatible
```

#### Connection Navigation
```bash
# Show all connections for a pipeline
cursus catalog connections --pipeline xgb_training_simple

# Find alternative pipelines
cursus catalog alternatives --pipeline xgb_training_simple

# Find connection path between pipelines
cursus catalog path --from xgb_training_simple --to xgb_e2e_comprehensive
```

#### Recommendations
```bash
# Get recommendations for a use case
cursus catalog recommend --use-case "XGBoost training"

# Get next step recommendations
cursus catalog recommend --next-steps xgb_training_simple

# Generate a learning path
cursus catalog recommend --learning-path --framework xgboost
```

#### Legacy Commands (Still Available)
```bash
# List all pipelines (backward compatibility)
cursus catalog list

# Show pipeline details (backward compatibility)
cursus catalog show xgb_training_simple
```

## Migration Steps

### Step 1: Update Import Statements

Replace old import statements with new ones:

```python
# Before
from cursus.pipeline_catalog.frameworks.xgboost.simple import create_pipeline

# After
from cursus.pipeline_catalog.pipelines.xgb_training_simple import create_pipeline
```

### Step 2: Update Function Calls

Handle the additional return value:

```python
# Before
pipeline, report, dag_compiler = create_pipeline(
    config_path=config_path,
    session=pipeline_session,
    role=role
)

# After
pipeline, report, dag_compiler, pipeline_template = create_pipeline(
    config_path=config_path,
    session=pipeline_session,
    role=role
)
```

### Step 3: Consider MODS Pipelines

Evaluate whether MODS-enhanced pipelines would benefit your use case:

```python
# Standard pipeline
from cursus.pipeline_catalog.pipelines.xgb_training_simple import create_pipeline

# MODS-enhanced pipeline (with fallback)
from cursus.pipeline_catalog.mods_pipelines.xgb_mods_training_simple import create_pipeline

pipeline, report, dag_compiler, template = create_pipeline(
    config_path=config_path,
    session=pipeline_session,
    role=role,
    enable_mods=True  # Gracefully falls back if MODS unavailable
)
```

### Step 4: Update CLI Usage

Replace old CLI commands with new discovery-focused commands:

```bash
# Instead of browsing hierarchically
cursus catalog list | grep xgboost

# Use intelligent discovery
cursus catalog find --framework xgboost

# Explore connections
cursus catalog connections --pipeline xgb_training_simple

# Get recommendations
cursus catalog recommend --use-case "XGBoost training"
```

### Step 5: Leverage New Features

Take advantage of the new capabilities:

```python
# Registry synchronization (automatic)
from cursus.pipeline_catalog.pipelines.xgb_training_simple import sync_to_registry
success = sync_to_registry()

# Connection traversal
from cursus.pipeline_catalog.utils import ConnectionTraverser
traverser = ConnectionTraverser()
alternatives = traverser.find_alternatives("xgb_training_simple")

# Tag-based discovery
from cursus.pipeline_catalog.utils import TagBasedDiscovery
discovery = TagBasedDiscovery()
results = discovery.search_pipelines({'framework': 'xgboost', 'tags': ['training']})

# Recommendations
from cursus.pipeline_catalog.utils import PipelineRecommendationEngine
engine = PipelineRecommendationEngine()
recommendations = engine.recommend_by_use_case("XGBoost training")
```

## Common Migration Scenarios

### Scenario 1: Simple Pipeline Usage

**Before:**
```python
from cursus.pipeline_catalog.frameworks.xgboost.simple import create_pipeline
from sagemaker import Session
from sagemaker.workflow.pipeline_context import PipelineSession

sagemaker_session = Session()
role = sagemaker_session.get_caller_identity_arn()
pipeline_session = PipelineSession()

pipeline, report, dag_compiler = create_pipeline(
    config_path="config.json",
    session=pipeline_session,
    role=role
)
```

**After:**
```python
from cursus.pipeline_catalog.pipelines.xgb_training_simple import create_pipeline
from sagemaker import Session
from sagemaker.workflow.pipeline_context import PipelineSession

sagemaker_session = Session()
role = sagemaker_session.get_caller_identity_arn()
pipeline_session = PipelineSession()

pipeline, report, dag_compiler, pipeline_template = create_pipeline(
    config_path="config.json",
    session=pipeline_session,
    role=role
)
# Pipeline automatically syncs to registry
```

### Scenario 2: Pipeline Discovery

**Before:**
```python
# Manual browsing through directory structure
import os
frameworks_dir = "cursus/pipeline_catalog/frameworks"
for framework in os.listdir(frameworks_dir):
    print(f"Framework: {framework}")
```

**After:**
```python
# Intelligent discovery
from cursus.pipeline_catalog.utils import TagBasedDiscovery

discovery = TagBasedDiscovery()
results = discovery.search_pipelines({
    'framework': 'xgboost',
    'complexity': 'simple'
})

for result in results:
    print(f"Pipeline: {result.pipeline.atomic_id} (Score: {result.score})")
```

### Scenario 3: Finding Related Pipelines

**Before:**
```python
# Manual exploration of directory structure
# No systematic way to find related pipelines
```

**After:**
```python
# Connection-based discovery
from cursus.pipeline_catalog.utils import ConnectionTraverser

traverser = ConnectionTraverser()
alternatives = traverser.find_alternatives("xgb_training_simple")
print(f"Alternatives: {alternatives}")

path = traverser.find_path("xgb_training_simple", "xgb_e2e_comprehensive")
print(f"Learning path: {path}")
```

## Troubleshooting

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'cursus.pipeline_catalog.frameworks'`

**Solution:** Update import paths according to the migration table above.

### Function Signature Errors

**Error:** `ValueError: too many values to unpack (expected 3)`

**Solution:** Update function calls to handle the additional return value:
```python
# Add pipeline_template to the unpacking
pipeline, report, dag_compiler, pipeline_template = create_pipeline(...)
```

### CLI Command Not Found

**Error:** `cursus: 'catalog' is not a cursus command`

**Solution:** Ensure you're using the updated CLI commands. The catalog CLI is now integrated into the main cursus command.

### Registry Issues

**Error:** Registry validation failures or connection issues

**Solution:** 
```bash
# Validate and fix registry
cursus catalog registry validate

# If issues persist, contact support
```

### MODS Not Available

**Error:** `MODSNotAvailableError: MODS is not available in the environment`

**Solution:** MODS pipelines gracefully fall back to standard functionality. You can:
1. Install MODS for enhanced features
2. Use `enable_mods=False` to explicitly disable MODS
3. Use standard pipelines instead

## Best Practices for Migration

### 1. Gradual Migration

Migrate one pipeline at a time rather than all at once:

```python
# Phase 1: Update imports but keep same usage pattern
from cursus.pipeline_catalog.pipelines.xgb_training_simple import create_pipeline

# Phase 2: Leverage new features
from cursus.pipeline_catalog.utils import TagBasedDiscovery
```

### 2. Test Registry Integration

Verify that pipelines sync correctly to the registry:

```python
from cursus.pipeline_catalog.pipelines.xgb_training_simple import sync_to_registry
success = sync_to_registry()
assert success, "Registry sync failed"
```

### 3. Explore Connections

Use the new discovery features to find better pipeline alternatives:

```bash
# Before choosing a pipeline, explore options
cursus catalog find --framework xgboost
cursus catalog alternatives --pipeline xgb_training_simple
cursus catalog recommend --use-case "XGBoost training"
```

### 4. Consider MODS

Evaluate MODS pipelines for enhanced operational capabilities:

```python
# Try MODS first, fall back to standard if needed
from cursus.pipeline_catalog.mods_pipelines.xgb_mods_training_simple import create_pipeline

try:
    pipeline, report, dag_compiler, template = create_pipeline(
        config_path=config_path,
        session=pipeline_session,
        role=role,
        enable_mods=True
    )
    print("Using MODS-enhanced pipeline")
except Exception as e:
    print(f"MODS not available, using standard features: {e}")
```

## Getting Help

### Documentation

- **Main README**: `src/cursus/pipeline_catalog/README.md`
- **Design Documents**: `slipbox/1_design/pipeline_catalog_zettelkasten_refactoring.md`
- **CLI Help**: `cursus catalog --help`

### CLI Tools

```bash
# Discover available pipelines
cursus catalog find --help

# Validate your setup
cursus catalog registry validate

# Get recommendations
cursus catalog recommend --help
```

### Common Commands

```bash
# List all available pipelines
cursus catalog list

# Find pipelines by criteria
cursus catalog find --framework xgboost

# Show pipeline details
cursus catalog show xgb_training_simple

# Check registry health
cursus catalog registry stats
```

## Support

If you encounter issues during migration:

1. **Check the registry**: `cursus catalog registry validate`
2. **Explore alternatives**: `cursus catalog find --framework <your_framework>`
3. **Get recommendations**: `cursus catalog recommend --use-case "<your_use_case>"`
4. **Review connections**: `cursus catalog connections --pipeline <pipeline_id>`

The new Zettelkasten-based catalog is designed to be more discoverable and connected than the old hierarchical structure. Take advantage of the intelligent discovery features to find the best pipelines for your needs.

---

**Welcome to the new Zettelkasten Pipeline Catalog - where pipelines connect and knowledge flows naturally!**

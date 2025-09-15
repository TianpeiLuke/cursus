# Cursus Pipeline Catalog

Welcome to the Cursus Pipeline Catalog, a knowledge-driven collection of pipeline templates organized using Zettelkasten principles for maximum discoverability and connection-based navigation.

## Overview

The Pipeline Catalog provides a flat, connection-based knowledge system that offers:

- **Atomic Independence**: Each pipeline is a self-contained, atomic unit with clear responsibilities
- **Connection-Based Discovery**: Find related pipelines through semantic connections rather than hierarchical browsing
- **Tag-Based Search**: Multi-dimensional tagging system for precise pipeline discovery
- **Intelligent Recommendations**: AI-powered suggestions for next steps and learning paths
- **MODS API Integration**: Dynamic MODS enhancement using the new API approach

## Quick Start

### Finding Pipelines

Use the CLI commands for intelligent discovery:

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

### Connection-Based Navigation

Explore relationships between pipelines:

```bash
# Show all connections for a pipeline
cursus catalog connections --pipeline xgb_training_simple

# Find alternative pipelines
cursus catalog alternatives --pipeline xgb_training_simple

# Find connection path between pipelines
cursus catalog path --from xgb_training_simple --to xgb_e2e_comprehensive
```

### Get Recommendations

Let the system suggest pipelines for you:

```bash
# Get recommendations for a use case
cursus catalog recommend --use-case "XGBoost training"

# Get next step recommendations
cursus catalog recommend --next-steps xgb_training_simple

# Generate a learning path
cursus catalog recommend --learning-path --framework xgboost
```

## Current System Structure

The catalog uses a flat, 3-level maximum structure:

```
pipeline_catalog/
├── pipelines/                    # Standard pipelines (atomic units)
│   ├── xgb_training_simple.py    # Basic XGBoost training
│   ├── xgb_training_calibrated.py # XGBoost with calibration
│   ├── pytorch_training_basic.py  # Basic PyTorch training
│   ├── xgb_training_evaluation.py # XGBoost with evaluation
│   ├── xgb_e2e_comprehensive.py  # Complete XGBoost workflow
│   ├── pytorch_e2e_standard.py   # Standard PyTorch workflow
│   └── dummy_e2e_basic.py        # Testing/demo pipeline
├── mods_pipelines/               # MODS API and utilities
│   ├── __init__.py               # MODS API exports
│   ├── README.md                 # MODS documentation
│   └── xgb_mods_e2e_comprehensive_new.py # Example implementation
├── mods_api.py                   # MODS API for dynamic enhancement
├── core/                         # Core utilities
│   ├── base_pipeline.py          # Base pipeline class
│   ├── catalog_registry.py       # Registry management
│   ├── connection_traverser.py   # Connection navigation
│   ├── recommendation_engine.py  # AI recommendations
│   ├── registry_validator.py     # Registry validation
│   └── tag_discovery.py          # Tag-based search
├── shared_dags/                  # Shared DAG definitions
└── catalog_index.json           # Connection registry
```

## Pipeline Discovery Methods

### 1. Tag-Based Discovery

Pipelines are tagged across multiple dimensions:

- **Framework Tags**: `xgboost`, `pytorch`, `dummy`
- **Task Tags**: `training`, `evaluation`, `calibration`, `registration`, `end_to_end`
- **Complexity Tags**: `simple`, `standard`, `comprehensive`
- **Domain Tags**: `machine_learning`, `supervised_learning`, `deep_learning`
- **Quality Tags**: `production_ready`, `tested`, `mods_compatible`

### 2. Connection-Based Navigation

Pipelines are connected through semantic relationships:

- **Alternatives**: Different approaches to the same problem
- **Extensions**: Pipelines that build upon others
- **Components**: Pipelines that use shared components
- **Progressions**: Natural learning progressions

### 3. Use Case Matching

Search by natural language descriptions:

```bash
cursus catalog find --use-case "I need to train an XGBoost model with probability calibration"
```

## Pipeline Naming Convention

Pipelines follow semantic naming: `{framework}_{use_case}_{complexity}`

- `xgb_training_simple` - Basic XGBoost training
- `pytorch_training_basic` - Basic PyTorch training  
- `xgb_e2e_comprehensive` - Complete XGBoost end-to-end workflow
- `dummy_e2e_basic` - Testing/demo pipeline

## MODS API Integration

The new MODS API approach eliminates code duplication by creating MODS-enhanced pipelines dynamically from regular pipelines.

### MODS API vs Traditional Approach

| Feature | Traditional MODS | New MODS API |
|---------|:----------------:|:------------:|
| Code Duplication | ❌ (Separate files) | ✅ (Single source) |
| Maintenance | ❌ (Double work) | ✅ (Single pipeline) |
| Consistency | ❌ (Can drift) | ✅ (Always in sync) |
| Dynamic Creation | ❌ | ✅ |
| Config Extraction | ❌ | ✅ (Automatic) |
| Backward Compatibility | ✅ | ✅ |

### Using the MODS API

#### Method 1: Direct API Usage

```python
from cursus.pipeline_catalog.mods_api import create_mods_pipeline_from_config
from cursus.pipeline_catalog.pipelines.xgb_e2e_comprehensive import XGBoostE2EComprehensivePipeline

# Create MODS-enhanced pipeline from regular pipeline + config
MODSPipeline = create_mods_pipeline_from_config(
    XGBoostE2EComprehensivePipeline,
    config_path="config.json"  # Extracts author, description, version from 'Base' key
)

# Use like any other pipeline
pipeline_instance = MODSPipeline(
    config_path="config.json",
    sagemaker_session=session,
    execution_role=role
)
pipeline = pipeline_instance.generate_pipeline()
```

#### Method 2: Convenience Functions

```python
from cursus.pipeline_catalog.mods_api import create_mods_xgboost_e2e_comprehensive

# Convenience function for common pipelines
MODSPipeline = create_mods_xgboost_e2e_comprehensive(
    config_path="config.json"
)
```

#### Method 3: Dynamic Creation by Name

```python
from cursus.pipeline_catalog.mods_api import create_mods_pipeline_by_name

# Create any MODS pipeline by name
MODSPipeline = create_mods_pipeline_by_name(
    'xgb_e2e_comprehensive',
    config_path='config.json'
)
```

#### Method 4: Via mods_pipelines Module

```python
from cursus.pipeline_catalog.mods_pipelines import create_mods_pipeline_from_config
from cursus.pipeline_catalog.pipelines.pytorch_e2e_standard import PyTorchE2EStandardPipeline

# Through the mods_pipelines module
MODSPipeline = create_mods_pipeline_from_config(
    PyTorchE2EStandardPipeline,
    config_path="config.json"
)
```

### MODS Metadata Extraction

The MODS API automatically extracts metadata from your configuration file:

```json
{
  "Base": {
    "author": "lukexie",
    "service_name": "AtoZ", 
    "model_class": "xgboost",
    "region": "NA",
    "pipeline_version": "1.2.3"
  }
}
```

The API extracts:
- **Author**: `base_config.author`
- **Description**: `base_config.pipeline_description` (derived from service_name, model_class, region)
- **Version**: `base_config.pipeline_version`

## Registry Management

The catalog maintains a connection registry for all pipelines:

```bash
# Validate registry integrity
cursus catalog registry validate

# Show registry statistics
cursus catalog registry stats

# Export pipeline metadata
cursus catalog registry export --pipelines xgb_training_simple,pytorch_training_basic
```

## Example Usage

### Basic Pipeline Usage

```python
from cursus.pipeline_catalog.pipelines.xgb_training_simple import create_pipeline
from sagemaker import Session
from sagemaker.workflow.pipeline_context import PipelineSession

# Initialize session
sagemaker_session = Session()
role = sagemaker_session.get_caller_identity_arn()
pipeline_session = PipelineSession()

# Create the pipeline
pipeline, report, dag_compiler, pipeline_template = create_pipeline(
    config_path="path/to/config.json",
    session=pipeline_session,
    role=role
)

# Pipeline automatically syncs to registry
# Get execution document
execution_doc = fill_execution_document(
    pipeline=pipeline,
    document={"training_dataset": "my-dataset"},
    dag_compiler=dag_compiler
)

# Execute pipeline
pipeline.upsert()
execution = pipeline.start(execution_input=execution_doc)
```

### MODS Pipeline Usage

```python
from cursus.pipeline_catalog.mods_api import create_mods_pipeline_from_config
from cursus.pipeline_catalog.pipelines.xgb_e2e_comprehensive import XGBoostE2EComprehensivePipeline

# Create MODS-enhanced pipeline
MODSPipeline = create_mods_pipeline_from_config(
    XGBoostE2EComprehensivePipeline,
    config_path="config.json"
)

# Use exactly like regular pipeline
pipeline_instance = MODSPipeline(
    config_path="config.json",
    sagemaker_session=pipeline_session,
    execution_role=role
)

# Generate MODS-enhanced pipeline
pipeline = pipeline_instance.generate_pipeline()

# Enhanced features when MODS is available:
# - Template registration in MODS global registry
# - Enhanced metadata extraction
# - Operational integration
# - Advanced pipeline tracking
```

## Decision Trees

### Choosing the Right Pipeline

```
Need XGBoost? ──┐
                ├─ Yes ──┐
                │        ├─ Simple training? ──── xgb_training_simple
                │        ├─ Need calibration? ──── xgb_training_calibrated  
                │        ├─ Need evaluation? ───── xgb_training_evaluation
                │        └─ Full workflow? ────── xgb_e2e_comprehensive
                │
                └─ No ───┐
                         ├─ PyTorch? ──┐
                         │             ├─ Basic training? ── pytorch_training_basic
                         │             └─ Full workflow? ─── pytorch_e2e_standard
                         │
                         └─ Testing/Demo? ── dummy_e2e_basic
```

### MODS vs Standard

```
Need MODS Features? ──┐
                      ├─ Yes ──── Use MODS API (automatic fallback)
                      │
                      └─ No ───── Use standard pipeline
```

## Advanced Features

### Connection Traversal

```python
from cursus.pipeline_catalog.core.connection_traverser import ConnectionTraverser

traverser = ConnectionTraverser()

# Find alternatives
alternatives = traverser.find_alternatives("xgb_training_simple")

# Find connection path
path = traverser.find_path("xgb_training_simple", "xgb_e2e_comprehensive")
```

### Tag-Based Discovery

```python
from cursus.pipeline_catalog.core.tag_discovery import TagBasedDiscovery

discovery = TagBasedDiscovery()

# Multi-criteria search
results = discovery.search_pipelines({
    'framework': 'xgboost',
    'tags': ['training', 'calibration'],
    'complexity': 'standard'
})
```

### Recommendations

```python
from cursus.pipeline_catalog.core.recommendation_engine import PipelineRecommendationEngine

engine = PipelineRecommendationEngine()

# Get recommendations by use case
recommendations = engine.recommend_by_use_case(
    "I need to train an XGBoost model for production"
)

# Generate learning path
learning_path = engine.generate_learning_path("xgboost")
```

## Best Practices

### Pipeline Selection

1. **Start with Discovery**: Use `cursus catalog find` to explore options
2. **Check Connections**: Use `cursus catalog connections` to see related pipelines
3. **Consider MODS**: Use MODS API for enhanced operational capabilities
4. **Follow Learning Paths**: Use recommendations for structured learning

### Development Workflow

1. **Discover** → Use CLI tools to find relevant pipelines
2. **Explore** → Check connections and alternatives
3. **Select** → Choose based on requirements and complexity
4. **Enhance** → Use MODS API if operational features needed
5. **Deploy** → Use MODS features for operational excellence

## Troubleshooting

### Common Issues

**Pipeline not found**: Use `cursus catalog list` to see all available pipelines

**Import errors**: Check the current import paths in this README

**MODS not available**: MODS API gracefully falls back to standard functionality

**Connection issues**: Use `cursus catalog registry validate` to check registry integrity

### Getting Help

1. Use the CLI discovery tools: `cursus catalog find --help`
2. Check pipeline connections: `cursus catalog connections --pipeline <id>`
3. Get recommendations: `cursus catalog recommend --use-case "<description>"`
4. Validate registry: `cursus catalog registry validate`

## Contributing

To add a new pipeline to the catalog:

1. Create the pipeline file in the `pipelines/` directory
2. Implement the required functions: `create_dag()`, `get_enhanced_dag_metadata()`, `sync_to_registry()`
3. Add comprehensive metadata with connections and tags
4. Test the pipeline and registry synchronization
5. Update documentation and examples

The catalog automatically discovers new pipelines and updates the registry. MODS enhancement is available automatically through the API.

---

**The Cursus Pipeline Catalog: Where knowledge connects and pipelines discover

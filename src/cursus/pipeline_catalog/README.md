# Zettelkasten Pipeline Catalog

Welcome to the Cursus Pipeline Catalog, a knowledge-driven collection of pipeline templates organized using Zettelkasten principles for maximum discoverability and connection-based navigation.

## Overview

The Pipeline Catalog has been redesigned as a flat, connection-based knowledge system that provides:

- **Atomic Independence**: Each pipeline is a self-contained, atomic unit with clear responsibilities
- **Connection-Based Discovery**: Find related pipelines through semantic connections rather than hierarchical browsing
- **Tag-Based Search**: Multi-dimensional tagging system for precise pipeline discovery
- **Intelligent Recommendations**: AI-powered suggestions for next steps and learning paths
- **MODS Integration**: Enhanced pipelines with operational capabilities when MODS is available

## Quick Start

### Finding Pipelines

Use the new CLI commands for intelligent discovery:

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

## New Flat Structure

The catalog now uses a flat, 3-level maximum structure:

```
pipeline_catalog/
├── pipelines/                    # Standard pipelines (atomic units)
│   ├── xgb_training_simple.py    # Basic XGBoost training
│   ├── xgb_training_calibrated.py # XGBoost with calibration
│   ├── pytorch_training_basic.py  # Basic PyTorch training
│   ├── xgb_training_evaluation.py # XGBoost with evaluation
│   ├── xgb_e2e_comprehensive.py  # Complete XGBoost workflow
│   └── pytorch_e2e_standard.py   # Standard PyTorch workflow
├── mods_pipelines/               # MODS-enhanced pipelines
│   ├── xgb_mods_training_simple.py
│   ├── xgb_mods_training_calibrated.py
│   ├── pytorch_mods_training_basic.py
│   ├── xgb_mods_training_evaluation.py
│   ├── xgb_mods_e2e_comprehensive.py
│   └── pytorch_mods_e2e_standard.py
├── shared_dags/                  # Shared DAG definitions
├── utils/                        # Zettelkasten utilities
└── catalog_index.json           # Connection registry
```

## Pipeline Discovery Methods

### 1. Tag-Based Discovery

Pipelines are tagged across multiple dimensions:

- **Framework Tags**: `xgboost`, `pytorch`, `mods`
- **Task Tags**: `training`, `evaluation`, `calibration`, `registration`, `end_to_end`
- **Complexity Tags**: `simple`, `standard`, `comprehensive`
- **Domain Tags**: `machine_learning`, `supervised_learning`, `deep_learning`
- **Quality Tags**: `production_ready`, `tested`, `mods_enhanced`

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

## MODS Integration

MODS (Model Operations and Deployment System) pipelines provide enhanced capabilities:

### Standard vs MODS Pipelines

| Feature | Standard | MODS |
|---------|:--------:|:----:|
| Basic Training | ✅ | ✅ |
| Template Registration | ❌ | ✅ |
| Enhanced Metadata | ❌ | ✅ |
| Operational Integration | ❌ | ✅ |
| Advanced Tracking | ❌ | ✅ |
| Graceful Degradation | N/A | ✅ |

### Using MODS Pipelines

```python
from cursus.pipeline_catalog.mods_pipelines.xgb_mods_training_simple import create_pipeline

# MODS pipeline with enhanced features
pipeline, report, dag_compiler, mods_template = create_pipeline(
    config_path="config.json",
    session=pipeline_session,
    role=role,
    enable_mods=True  # Enable MODS features
)

# Automatic template registration in MODS global registry
# Enhanced metadata extraction and validation
# Integration with MODS operational tools
```

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
from cursus.pipeline_catalog.mods_pipelines.xgb_mods_training_simple import create_pipeline

# MODS pipeline with fallback
pipeline, report, dag_compiler, template = create_pipeline(
    config_path="config.json",
    session=pipeline_session,
    role=role,
    enable_mods=True  # Gracefully falls back to standard if MODS unavailable
)

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
                         └─ Other framework? ── Use discovery tools
```

### MODS vs Standard

```
MODS Available? ──┐
                  ├─ Yes ──┐
                  │        ├─ Need operational features? ── Use MODS pipeline
                  │        └─ Basic functionality? ────── Either (MODS has fallback)
                  │
                  └─ No ─── Use standard pipeline
```

## Migration Guide

### From Old Structure

If you were using the old hierarchical structure:

```python
# OLD (deprecated)
from cursus.pipeline_catalog.frameworks.xgboost.simple import create_pipeline

# NEW (current)
from cursus.pipeline_catalog.pipelines.xgb_training_simple import create_pipeline
```

### Import Path Changes

| Old Path | New Path |
|----------|----------|
| `frameworks.xgboost.simple` | `pipelines.xgb_training_simple` |
| `frameworks.xgboost.training.with_calibration` | `pipelines.xgb_training_calibrated` |
| `frameworks.pytorch.training.basic_training` | `pipelines.pytorch_training_basic` |
| `frameworks.xgboost.end_to_end.complete_e2e` | `pipelines.xgb_e2e_comprehensive` |

## Advanced Features

### Connection Traversal

```python
from cursus.pipeline_catalog.utils import ConnectionTraverser

traverser = ConnectionTraverser()

# Find alternatives
alternatives = traverser.find_alternatives("xgb_training_simple")

# Find connection path
path = traverser.find_path("xgb_training_simple", "xgb_e2e_comprehensive")
```

### Tag-Based Discovery

```python
from cursus.pipeline_catalog.utils import TagBasedDiscovery

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
from cursus.pipeline_catalog.utils import PipelineRecommendationEngine

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
3. **Consider MODS**: Use MODS pipelines for enhanced operational capabilities
4. **Follow Learning Paths**: Use recommendations for structured learning

### Development Workflow

1. **Discover** → Use CLI tools to find relevant pipelines
2. **Explore** → Check connections and alternatives
3. **Select** → Choose based on requirements and complexity
4. **Customize** → Adapt the pipeline to your specific needs
5. **Deploy** → Use MODS features for operational excellence

## Troubleshooting

### Common Issues

**Pipeline not found**: Use `cursus catalog list` to see all available pipelines

**Import errors**: Check the new import paths in the migration guide

**MODS not available**: MODS pipelines gracefully fall back to standard functionality

**Connection issues**: Use `cursus catalog registry validate` to check registry integrity

### Getting Help

1. Use the CLI discovery tools: `cursus catalog find --help`
2. Check pipeline connections: `cursus catalog connections --pipeline <id>`
3. Get recommendations: `cursus catalog recommend --use-case "<description>"`
4. Validate registry: `cursus catalog registry validate`

## Contributing

To add a new pipeline to the catalog:

1. Create the pipeline file in the appropriate directory
2. Implement the required functions: `create_dag()`, `get_enhanced_dag_metadata()`, `sync_to_registry()`
3. Add comprehensive ZettelkastenMetadata with connections and tags
4. Test the pipeline and registry synchronization
5. Update documentation and examples

The catalog automatically discovers new pipelines and updates the registry.

---

**The Zettelkasten Pipeline Catalog: Where knowledge connects and pipelines discover themselves.**

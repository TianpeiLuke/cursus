# Pipeline Catalog - Standard Pipelines

This directory contains all standard pipeline implementations organized in a flat structure following Zettelkasten knowledge management principles. These pipelines serve as the foundation for both direct use and MODS enhancement through the MODS API.

## Current Pipeline Collection

### Available Pipelines

| Pipeline | Framework | Use Case | Complexity | Description |
|----------|-----------|----------|------------|-------------|
| `xgb_training_simple.py` | XGBoost | Training | Simple | Basic XGBoost training workflow |
| `xgb_training_calibrated.py` | XGBoost | Training | Standard | XGBoost training with probability calibration |
| `xgb_training_evaluation.py` | XGBoost | Training | Standard | XGBoost training with comprehensive evaluation |
| `xgb_e2e_comprehensive.py` | XGBoost | End-to-End | Comprehensive | Complete XGBoost workflow from data to deployment |
| `pytorch_training_basic.py` | PyTorch | Training | Simple | Basic PyTorch training workflow |
| `pytorch_e2e_standard.py` | PyTorch | End-to-End | Standard | Standard PyTorch workflow with evaluation |
| `dummy_e2e_basic.py` | Dummy | Testing | Simple | Testing/demo pipeline for infrastructure validation |

## Pipeline Architecture

### Atomic Independence

Each pipeline is designed as an **atomic, independent unit** with:

- **Single Responsibility**: Clear, focused purpose
- **Complete Implementation**: All necessary components included
- **Enhanced Metadata**: Rich Zettelkasten-style metadata
- **Connection Awareness**: Links to related pipelines
- **MODS Compatibility**: Can be enhanced via MODS API

### Standard Interface

All pipelines implement the standard interface:

```python
def create_dag() -> PipelineDAG:
    """Create the pipeline DAG structure."""
    
def get_enhanced_dag_metadata() -> EnhancedDAGMetadata:
    """Get comprehensive metadata with Zettelkasten integration."""
    
def create_pipeline(config_path, session, role, pipeline_name=None, **kwargs):
    """Create the complete SageMaker pipeline."""
    
def sync_to_registry() -> bool:
    """Synchronize pipeline metadata to the catalog registry."""
```

## Discovery and Navigation

### Framework-Based Discovery

```python
from cursus.pipeline_catalog.core.tag_discovery import TagBasedDiscovery

discovery = TagBasedDiscovery()

# Find XGBoost pipelines
xgb_pipelines = discovery.search_pipelines({'framework': 'xgboost'})

# Find PyTorch pipelines
pytorch_pipelines = discovery.search_pipelines({'framework': 'pytorch'})
```

### Complexity-Based Discovery

```python
# Find simple pipelines for beginners
simple_pipelines = discovery.search_pipelines({'complexity': 'simple'})

# Find comprehensive pipelines for production
comprehensive_pipelines = discovery.search_pipelines({'complexity': 'comprehensive'})
```

### Use Case Discovery

```python
# Find training pipelines
training_pipelines = discovery.search_pipelines({'task_tags': ['training']})

# Find end-to-end pipelines
e2e_pipelines = discovery.search_pipelines({'task_tags': ['end_to_end']})
```

## Connection Navigation

### Find Alternatives

```python
from cursus.pipeline_catalog.core.connection_traverser import ConnectionTraverser

traverser = ConnectionTraverser()

# Find alternatives to XGBoost simple training
alternatives = traverser.find_alternatives("xgb_training_simple")
# Returns: ["pytorch_training_basic"] - alternative framework for basic training
```

### Explore Learning Paths

```python
# Find progression path from simple to comprehensive
path = traverser.find_path("xgb_training_simple", "xgb_e2e_comprehensive")
# Returns learning progression through related pipelines
```

### Get Recommendations

```python
from cursus.pipeline_catalog.core.recommendation_engine import PipelineRecommendationEngine

engine = PipelineRecommendationEngine()

# Get recommendations for a use case
recommendations = engine.recommend_by_use_case(
    "I need to train an XGBoost model for production"
)
```

## Usage Examples

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

# Execute pipeline
pipeline.upsert()
execution = pipeline.start()
```

### MODS Enhancement

```python
from cursus.pipeline_catalog.mods_api import create_mods_pipeline_from_config
from cursus.pipeline_catalog.pipelines.xgb_e2e_comprehensive import XGBoostE2EComprehensivePipeline

# Create MODS-enhanced version
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
pipeline = pipeline_instance.generate_pipeline()
```

### Pipeline Discovery

```python
from cursus.pipeline_catalog.core.tag_discovery import TagBasedDiscovery

discovery = TagBasedDiscovery()

# Multi-criteria search
results = discovery.search_pipelines({
    'framework': 'xgboost',
    'complexity': 'simple',
    'task_tags': ['training']
})

for result in results:
    print(f"Pipeline: {result.pipeline_id} (Score: {result.score})")
```

## Pipeline Categories

### By Framework

#### XGBoost Pipelines
- **xgb_training_simple**: Basic XGBoost training with minimal configuration
- **xgb_training_calibrated**: XGBoost training with probability calibration
- **xgb_training_evaluation**: XGBoost training with comprehensive evaluation
- **xgb_e2e_comprehensive**: Complete XGBoost workflow from data to deployment

#### PyTorch Pipelines
- **pytorch_training_basic**: Basic PyTorch training for deep learning models
- **pytorch_e2e_standard**: Standard PyTorch workflow with evaluation

#### Testing Pipelines
- **dummy_e2e_basic**: Infrastructure testing and demonstration pipeline

### By Complexity

#### Simple (Beginner-Friendly)
- `xgb_training_simple`: Minimal configuration, easy to understand
- `pytorch_training_basic`: Basic deep learning workflow
- `dummy_e2e_basic`: Testing and learning

#### Standard (Production-Ready)
- `xgb_training_calibrated`: Production features with calibration
- `xgb_training_evaluation`: Production features with evaluation
- `pytorch_e2e_standard`: Production PyTorch workflow

#### Comprehensive (Full-Featured)
- `xgb_e2e_comprehensive`: Complete workflow with all features

### By Use Case

#### Training Focus
- `xgb_training_simple`, `xgb_training_calibrated`, `xgb_training_evaluation`
- `pytorch_training_basic`

#### End-to-End Workflows
- `xgb_e2e_comprehensive`, `pytorch_e2e_standard`, `dummy_e2e_basic`

#### Specialized Features
- **Calibration**: `xgb_training_calibrated`
- **Evaluation**: `xgb_training_evaluation`
- **Deep Learning**: `pytorch_training_basic`, `pytorch_e2e_standard`

## Best Practices

### For Pipeline Users

1. **Start with Discovery**: Use tag-based search to find relevant pipelines
2. **Check Connections**: Explore alternatives and related pipelines
3. **Consider MODS**: Use MODS API for enhanced operational capabilities
4. **Follow Learning Paths**: Progress from simple to comprehensive pipelines

### For Pipeline Developers

1. **Atomic Design**: Each pipeline should be self-contained and focused
2. **Rich Metadata**: Provide comprehensive Zettelkasten metadata
3. **Clear Connections**: Document relationships to other pipelines
4. **MODS Compatibility**: Ensure pipelines work with MODS API
5. **Registry Integration**: Implement sync_to_registry() method

## Integration with MODS

All pipelines in this directory are **MODS-compatible**, meaning they can be enhanced with MODS features using the MODS API:

```python
# Any pipeline can be MODS-enhanced
from cursus.pipeline_catalog.mods_api import create_mods_pipeline_by_name

# Create MODS version of any pipeline
mods_pipeline = create_mods_pipeline_by_name(
    'xgb_training_simple',  # or any other pipeline
    config_path='config.json'
)
```

This eliminates the need for separate MODS pipeline files while providing all MODS functionality.

## Registry Integration

All pipelines automatically integrate with the catalog registry:

```python
from cursus.pipeline_catalog.pipelines.xgb_training_simple import sync_to_registry

# Sync pipeline metadata to registry
success = sync_to_registry()
```

The registry maintains connections, tags, and metadata for intelligent discovery and recommendations.

## Related Documentation

- [Main Pipeline Catalog README](../README.md) - Overall catalog documentation
- [MODS Pipelines](../mods_pipelines/README.md) - MODS API and enhancement
- [Catalog Index](../catalog_index.json) - Registry of all pipelines and connections
- [Zettelkasten Knowledge Management Principles](../../slipbox/1_design/zettelkasten_knowledge_management_principles.md) - Knowledge organization principles
- [Pipeline Catalog Zettelkasten Refactoring](../../slipbox/1_design/pipeline_catalog_zettelkasten_refactoring.md) - Catalog refactoring design
- [Implementation Plan](../../slipbox/2_project_planning/2025-08-20_pipeline_catalog_zettelkasten_refactoring_plan.md) - Project planning documentation

---

**Standard Pipelines: The foundation for intelligent pipeline discovery and MODS enhancement.**

# Pipeline Catalog - Standard Pipelines

This directory contains all standard pipeline implementations organized in a flat structure following Zettelkasten knowledge management principles.

## Structure

Each pipeline in this directory is an **atomic, independent unit** with:

- **Enhanced DAGMetadata integration** - Rich metadata with Zettelkasten extensions
- **Connection-based relationships** - Links to alternatives, related pipelines, and usage contexts
- **Multi-dimensional tagging** - Framework, complexity, use-case, and feature tags
- **Registry synchronization** - Automatic integration with the catalog registry

## Pipeline Organization

### Naming Convention

Pipelines follow semantic naming patterns:
- `{framework}_{use_case}_{complexity}.py`
- Examples: `xgb_training_simple.py`, `pytorch_training_comprehensive.py`

### Atomic Independence

Each pipeline:
- ✅ Can be understood and used independently
- ✅ Has clear, single responsibility
- ✅ Contains complete implementation
- ✅ Includes comprehensive metadata

## Discovery Methods

### By Framework
```python
from src.cursus.pipeline_catalog.utils import discover_by_framework
xgb_pipelines = discover_by_framework("xgboost")
```

### By Tags
```python
from src.cursus.pipeline_catalog.utils import discover_by_tags
training_pipelines = discover_by_tags(["training", "supervised"])
```

### By Use Case
```python
from src.cursus.pipeline_catalog.utils import create_catalog_manager
manager = create_catalog_manager()
classification_pipelines = manager.discover_pipelines(use_case="classification")
```

## Connection Navigation

### Find Alternatives
```python
from src.cursus.pipeline_catalog.utils import get_pipeline_alternatives
alternatives = get_pipeline_alternatives("xgb_training_simple")
```

### Explore Relationships
```python
manager = create_catalog_manager()
connections = manager.get_pipeline_connections("pytorch_training_basic")
# Returns: {"alternatives": [...], "related": [...], "used_in": [...]}
```

## Pipeline Categories

### Training Pipelines
- **Simple**: Basic training with minimal configuration
- **Standard**: Production-ready with validation and monitoring
- **Comprehensive**: Full feature set with advanced options

### Data Processing Pipelines
- **Standard**: Common preprocessing operations
- **Advanced**: Complex transformations and feature engineering

### Evaluation Pipelines
- **Basic**: Essential metrics and validation
- **Comprehensive**: Full evaluation suite with visualizations

### Deployment Pipelines
- **Batch Inference**: Batch prediction workflows
- **Model Registration**: Model packaging and registration

## Usage Examples

### Loading a Pipeline
```python
from src.cursus.pipeline_catalog.pipelines import load_pipeline

# Load specific pipeline
xgb_simple = load_pipeline("xgb_training_simple")

# Get enhanced metadata
metadata = xgb_simple.get_enhanced_dag_metadata()
print(f"Complexity: {metadata.complexity}")
print(f"Features: {metadata.features}")
```

### Getting Recommendations
```python
manager = create_catalog_manager()
recommendations = manager.get_recommendations(
    use_case="binary classification",
    framework="xgboost",
    complexity="simple"
)
```

## Best Practices

### For Pipeline Authors
1. **Single Responsibility**: Each pipeline should have one clear purpose
2. **Complete Metadata**: Include comprehensive Zettelkasten metadata
3. **Clear Connections**: Document relationships to other pipelines
4. **Atomic Design**: Ensure pipeline can be understood independently

### For Pipeline Users
1. **Use Discovery**: Leverage tag-based and connection-based discovery
2. **Explore Alternatives**: Check alternative implementations
3. **Follow Connections**: Use related pipelines for learning paths
4. **Validate Choices**: Use recommendation engine for guidance

## Migration Status

This directory is part of the Phase 2 implementation of the pipeline catalog refactoring. Pipelines will be migrated from the existing hierarchical structure to this flat, connection-based organization.

### Migration Priority
1. **High Priority**: Core training and preprocessing pipelines
2. **Medium Priority**: Evaluation and comprehensive workflows
3. **Lower Priority**: Specialized and utility pipelines

## Related Documentation

- [Zettelkasten Knowledge Management Principles](../../slipbox/1_design/zettelkasten_knowledge_management_principles.md)
- [Pipeline Catalog Zettelkasten Refactoring](../../slipbox/1_design/pipeline_catalog_zettelkasten_refactoring.md)
- [Implementation Plan](../../slipbox/2_project_planning/2025-08-20_pipeline_catalog_zettelkasten_refactoring_plan.md)

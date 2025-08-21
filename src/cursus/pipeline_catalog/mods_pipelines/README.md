# Pipeline Catalog - MODS Pipelines

This directory contains all MODS-compatible pipeline implementations organized in a flat structure following Zettelkasten knowledge management principles.

## Structure

Each MODS pipeline in this directory is an **atomic, independent unit** with:

- **Enhanced DAGMetadata integration** - Rich metadata with Zettelkasten extensions
- **MODS compiler compatibility** - Full integration with MODS compilation system
- **Connection-based relationships** - Links to alternatives, related pipelines, and usage contexts
- **Multi-dimensional tagging** - Framework, complexity, use-case, and feature tags
- **Registry synchronization** - Automatic integration with the catalog registry

## MODS Integration

### MODS Compiler Compatibility

All pipelines in this directory are designed to work seamlessly with the MODS (Model Operations Development System) compiler:

- **Specification-driven**: Based on MODS specification contracts
- **Type-safe**: Full type checking and validation
- **Modular**: Composable with other MODS components
- **Testable**: Built-in testing and validation support

### MODS-Specific Features

- **Enhanced Metadata Extraction**: Automatic metadata generation from MODS specifications
- **Dependency Resolution**: Intelligent handling of MODS dependencies
- **Configuration Management**: Advanced configuration merging and validation
- **Registry Integration**: Seamless sync with MODS registry systems

## Pipeline Organization

### Naming Convention

MODS pipelines follow semantic naming patterns:
- `{framework}_mods_{use_case}_{complexity}.py`
- Examples: `xgb_mods_training_simple.py`, `pytorch_mods_training_comprehensive.py`

### Atomic Independence

Each MODS pipeline:
- ✅ Can be understood and used independently
- ✅ Has clear, single responsibility
- ✅ Contains complete MODS-compatible implementation
- ✅ Includes comprehensive metadata
- ✅ Follows MODS specification contracts

## Discovery Methods

### By Framework
```python
from src.cursus.pipeline_catalog.utils import discover_by_framework
xgb_mods_pipelines = discover_by_framework("xgboost")
# Filter for MODS pipelines
mods_only = [p for p in xgb_mods_pipelines if "mods" in p]
```

### By MODS-Specific Tags
```python
from src.cursus.pipeline_catalog.utils import discover_by_tags
mods_training = discover_by_tags(["mods", "training", "supervised"])
```

### Using MODS Pipeline Registry
```python
from src.cursus.pipeline_catalog.mods_pipelines import get_registered_mods_pipelines
all_mods_pipelines = get_registered_mods_pipelines()
```

## Connection Navigation

### Find MODS Alternatives
```python
from src.cursus.pipeline_catalog.utils import get_pipeline_alternatives
alternatives = get_pipeline_alternatives("xgb_mods_training_simple")
```

### Explore MODS Relationships
```python
from src.cursus.pipeline_catalog.utils import create_catalog_manager
manager = create_catalog_manager()
connections = manager.get_pipeline_connections("pytorch_mods_training_basic")
# Returns: {"alternatives": [...], "related": [...], "used_in": [...]}
```

## Pipeline Categories

### MODS Training Pipelines
- **Simple**: Basic MODS training with minimal configuration
- **Standard**: Production-ready MODS training with validation
- **Comprehensive**: Full MODS feature set with advanced options

### MODS Data Processing Pipelines
- **Standard**: Common MODS preprocessing operations
- **Advanced**: Complex MODS transformations and feature engineering

### MODS Evaluation Pipelines
- **Basic**: Essential MODS metrics and validation
- **Comprehensive**: Full MODS evaluation suite with visualizations

### MODS Deployment Pipelines
- **Batch Inference**: MODS batch prediction workflows
- **Model Registration**: MODS model packaging and registration

## Usage Examples

### Loading a MODS Pipeline
```python
from src.cursus.pipeline_catalog.mods_pipelines import load_mods_pipeline

# Load specific MODS pipeline
xgb_mods_simple = load_mods_pipeline("xgb_mods_training_simple")

# Get enhanced metadata
metadata = xgb_mods_simple.get_enhanced_dag_metadata()
print(f"MODS Compatibility: {metadata.zettelkasten_metadata.mods_compatible}")
print(f"Complexity: {metadata.complexity}")
```

### MODS Compilation Integration
```python
from src.cursus.mods_dag_compiler import ModsDAGCompiler

# Compile MODS pipeline
compiler = ModsDAGCompiler()
compiled_dag = compiler.compile_pipeline("xgb_mods_training_simple")
```

### Getting MODS Recommendations
```python
manager = create_catalog_manager()
recommendations = manager.get_recommendations(
    use_case="binary classification",
    framework="xgboost",
    complexity="simple",
    tags=["mods"]
)
```

## MODS-Specific Best Practices

### For MODS Pipeline Authors
1. **MODS Specification Compliance**: Follow MODS specification contracts
2. **Type Safety**: Ensure full type checking and validation
3. **Modular Design**: Create composable, reusable components
4. **Complete Metadata**: Include MODS-specific metadata fields
5. **Testing Integration**: Implement MODS testing patterns

### For MODS Pipeline Users
1. **Use MODS Discovery**: Leverage MODS-specific discovery methods
2. **Validate Compatibility**: Check MODS compiler compatibility
3. **Follow MODS Patterns**: Use established MODS usage patterns
4. **Integration Testing**: Test with MODS compilation pipeline

## MODS Integration Features

### Automatic Metadata Extraction
```python
# MODS pipelines automatically extract metadata from specifications
metadata = pipeline.get_enhanced_dag_metadata()
assert metadata.zettelkasten_metadata.mods_compatible == True
```

### Specification-Driven Configuration
```python
# MODS pipelines use specification-driven configuration
config = pipeline.get_mods_configuration()
validated_config = pipeline.validate_mods_config(config)
```

### Registry Synchronization
```python
# MODS pipelines automatically sync with registry
from src.cursus.pipeline_catalog.shared_dags.registry_sync import DAGMetadataRegistrySync
sync = DAGMetadataRegistrySync()
sync.sync_mods_pipeline(pipeline_metadata, "xgb_mods_training_simple.py")
```

## Migration Status

This directory is part of the Phase 3 implementation of the pipeline catalog refactoring. MODS pipelines will be migrated after standard pipelines are complete.

### MODS Migration Priority
1. **High Priority**: Core MODS training and preprocessing pipelines
2. **Medium Priority**: MODS evaluation and comprehensive workflows
3. **Lower Priority**: Specialized MODS utility pipelines

## MODS Compiler Integration

### Compilation Process
1. **Pipeline Discovery**: MODS compiler discovers pipelines in this directory
2. **Metadata Extraction**: Enhanced metadata is automatically extracted
3. **Specification Validation**: MODS specifications are validated
4. **DAG Generation**: Optimized DAGs are generated
5. **Registry Update**: Pipeline registry is automatically updated

### Compatibility Requirements
- **Python 3.8+**: Minimum Python version for MODS compatibility
- **MODS Compiler**: Latest version of MODS DAG compiler
- **Specification Format**: MODS specification format compliance
- **Type Annotations**: Full type annotation coverage

## Related Documentation

- [MODS DAG Compiler Design](../../slipbox/1_design/mods_dag_compiler_design.md)
- [Zettelkasten Knowledge Management Principles](../../slipbox/1_design/zettelkasten_knowledge_management_principles.md)
- [Pipeline Catalog Zettelkasten Refactoring](../../slipbox/1_design/pipeline_catalog_zettelkasten_refactoring.md)
- [Implementation Plan](../../slipbox/2_project_planning/2025-08-20_pipeline_catalog_zettelkasten_refactoring_plan.md)

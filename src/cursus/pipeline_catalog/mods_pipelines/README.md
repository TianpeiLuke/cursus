# Pipeline Catalog - MODS Pipelines

This directory contains MODS-enhanced pipeline functionality using the new API approach that eliminates code duplication between regular pipelines and MODS pipelines.

## Overview

Instead of maintaining separate MODS pipeline implementations, this module now provides:

- **MODS API**: Converts regular pipelines to MODS-enhanced versions dynamically
- **Configuration-based metadata extraction**: Automatic extraction from config files
- **Dynamic pipeline creation and registration**: No code duplication needed
- **Backward compatibility**: Existing MODS pipeline interfaces continue to work

## New MODS API Approach

The MODS API eliminates the need for duplicate pipeline files by creating MODS-enhanced pipelines dynamically from regular pipelines.

### Key Benefits

| Feature | Traditional MODS | New MODS API |
|---------|:----------------:|:------------:|
| Code Duplication | ❌ (Separate files) | ✅ (Single source) |
| Maintenance | ❌ (Double work) | ✅ (Single pipeline) |
| Consistency | ❌ (Can drift) | ✅ (Always in sync) |
| Dynamic Creation | ❌ | ✅ |
| Config Extraction | ❌ | ✅ (Automatic) |
| Backward Compatibility | ✅ | ✅ |

### Current Directory Structure

```
mods_pipelines/
├── __init__.py                              # MODS API exports and registry
├── README.md                                # This documentation
├── mods_pipeline_adapter.py                 # Original adapter (reference)
└── xgb_mods_e2e_comprehensive_new.py       # Example of new API approach
```

## MODS API Usage

### Method 1: Direct API Usage

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

### Method 2: Via mods_pipelines Module

```python
from cursus.pipeline_catalog.mods_pipelines import create_mods_pipeline_from_config
from cursus.pipeline_catalog.pipelines.pytorch_e2e_standard import PyTorchE2EStandardPipeline

# Through the mods_pipelines module
MODSPipeline = create_mods_pipeline_from_config(
    PyTorchE2EStandardPipeline,
    config_path="config.json"
)
```

### Method 3: Convenience Functions

```python
from cursus.pipeline_catalog.mods_api import create_mods_xgboost_e2e_comprehensive

# Convenience function for common pipelines
MODSPipeline = create_mods_xgboost_e2e_comprehensive(
    config_path="config.json"
)
```

### Method 4: Dynamic Creation by Name

```python
from cursus.pipeline_catalog.mods_api import create_mods_pipeline_by_name

# Create any MODS pipeline by name
MODSPipeline = create_mods_pipeline_by_name(
    'xgb_e2e_comprehensive',
    config_path='config.json'
)
```

## MODS Metadata Extraction

The MODS API automatically extracts metadata from your configuration file using the 'Base' key pattern:

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

## Available MODS-Compatible Pipelines

All regular pipelines can be enhanced with MODS using the API:

| Pipeline | Description | MODS API Function |
|----------|-------------|-------------------|
| `xgb_training_simple` | Basic XGBoost training | `create_mods_pipeline_by_name('xgb_training_simple')` |
| `xgb_training_calibrated` | XGBoost with calibration | `create_mods_pipeline_by_name('xgb_training_calibrated')` |
| `xgb_training_evaluation` | XGBoost with evaluation | `create_mods_pipeline_by_name('xgb_training_evaluation')` |
| `xgb_e2e_comprehensive` | Complete XGBoost workflow | `create_mods_xgboost_e2e_comprehensive()` |
| `pytorch_training_basic` | Basic PyTorch training | `create_mods_pipeline_by_name('pytorch_training_basic')` |
| `pytorch_e2e_standard` | Standard PyTorch workflow | `create_mods_pytorch_e2e_standard()` |
| `dummy_e2e_basic` | Testing/demo pipeline | `create_mods_dummy_e2e_basic()` |

## Discovery and Registry

### Pipeline Discovery

```python
from cursus.pipeline_catalog.mods_pipelines import discover_mods_pipelines

# Discover all available MODS-compatible pipelines
available_pipelines = discover_mods_pipelines()
print(f"Available MODS pipelines: {available_pipelines}")
```

### Registry Management

```python
from cursus.pipeline_catalog.mods_pipelines import get_registered_mods_pipelines, create_all_mods_pipelines

# Get all registered MODS pipelines
registered = get_registered_mods_pipelines()

# Create MODS versions of all available pipelines
all_mods = create_all_mods_pipelines(config_path="config.json")
```

### Dynamic Loading

```python
from cursus.pipeline_catalog.mods_pipelines import load_mods_pipeline

# Dynamically load a MODS pipeline
MODSPipeline = load_mods_pipeline("xgb_e2e_comprehensive")
```

## Migration from Old Approach

### Before (Separate MODS Files)
```python
# Had to maintain separate MODS pipeline files
from cursus.pipeline_catalog.mods_pipelines.xgb_mods_e2e_comprehensive import XGBoostMODSPipeline

pipeline_instance = XGBoostMODSPipeline(
    config_path="config.json",
    sagemaker_session=session,
    execution_role=role
)
```

### After (MODS API)
```python
# Dynamic creation from regular pipeline
from cursus.pipeline_catalog.mods_api import create_mods_pipeline_from_config
from cursus.pipeline_catalog.pipelines.xgb_e2e_comprehensive import XGBoostE2EComprehensivePipeline

MODSPipeline = create_mods_pipeline_from_config(
    XGBoostE2EComprehensivePipeline,
    config_path="config.json"
)

pipeline_instance = MODSPipeline(
    config_path="config.json",
    sagemaker_session=session,
    execution_role=role
)
```

## MODS Integration Features

### Automatic MODS Enhancement

When MODS is available, pipelines get enhanced features:
- Template registration in MODS global registry
- Enhanced metadata extraction and validation
- Operational integration with MODS tools
- Advanced pipeline tracking and monitoring

### Graceful Fallback

When MODS is not available:
- Pipelines work as regular SageMaker pipelines
- No functionality is lost
- Warnings are logged but execution continues
- Full backward compatibility maintained

### Configuration Integration

```python
# MODS API integrates with existing configuration patterns
MODSPipeline = create_mods_pipeline_from_config(
    RegularPipeline,
    config_path="config.json",
    # Override metadata if needed
    author="custom-author",
    pipeline_description="Custom description",
    pipeline_version="2.0.0"
)
```

## Best Practices

### For Pipeline Users

1. **Use the MODS API**: Create MODS pipelines dynamically rather than maintaining separate files
2. **Leverage Config Extraction**: Let the API extract metadata from your config files
3. **Test Both Modes**: Ensure your pipelines work with and without MODS
4. **Use Convenience Functions**: Use provided convenience functions for common pipelines

### For Pipeline Developers

1. **Focus on Regular Pipelines**: Develop and maintain only regular pipelines
2. **Ensure MODS Compatibility**: Make sure regular pipelines work with MODS API
3. **Provide Good Metadata**: Ensure config files have proper 'Base' sections
4. **Test MODS Enhancement**: Test that MODS API works with your pipelines

## Example Implementation

See `xgb_mods_e2e_comprehensive_new.py` for a complete example of how to use the new MODS API approach:

```python
from cursus.pipeline_catalog.mods_api import create_mods_pipeline_from_config
from cursus.pipeline_catalog.pipelines.xgb_e2e_comprehensive import XGBoostE2EComprehensivePipeline

# Create MODS-enhanced pipeline class
XGBoostE2EComprehensiveMODSPipeline = create_mods_pipeline_from_config(
    pipeline_class=XGBoostE2EComprehensivePipeline
)

# Use like any other pipeline
pipeline_instance = XGBoostE2EComprehensiveMODSPipeline(
    config_path="config.json",
    sagemaker_session=session,
    execution_role=role
)
pipeline = pipeline_instance.generate_pipeline()
```

## Troubleshooting

### Common Issues

**MODS not available**: The API gracefully falls back to standard functionality
**Config extraction fails**: Provide metadata explicitly as parameters
**Import errors**: Ensure you're importing from the correct modules

### Getting Help

1. Check the main MODS API documentation: `../mods_api.py`
2. Review example implementations in this directory
3. Use the discovery functions to explore available pipelines
4. Test with both MODS available and unavailable

## Related Documentation

- [Main Pipeline Catalog README](../README.md) - Overall catalog documentation
- [MODS API](../mods_api.py) - Core MODS API implementation
- [Regular Pipelines](../pipelines/) - Source pipelines for MODS enhancement
- [Pipeline Catalog Index](../catalog_index.json) - Registry of all pipelines
- [MODS DAG Compiler Design](../../slipbox/1_design/mods_dag_compiler_design.md) - MODS compiler architecture
- [Zettelkasten Knowledge Management Principles](../../slipbox/1_design/zettelkasten_knowledge_management_principles.md) - Knowledge organization principles
- [Pipeline Catalog Zettelkasten Refactoring](../../slipbox/1_design/pipeline_catalog_zettelkasten_refactoring.md) - Catalog refactoring design
- [Implementation Plan](../../slipbox/2_project_planning/2025-08-20_pipeline_catalog_zettelkasten_refactoring_plan.md) - Project planning documentation

---

**The MODS API: Eliminating duplication while enhancing functionality.**

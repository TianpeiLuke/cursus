---
tags:
  - design
  - pipeline_catalog
  - mods_integration
  - architecture
  - compiler_abstraction
keywords:
  - pipeline catalog
  - MODS integration
  - dual compiler architecture
  - code reuse
  - DAG sharing
  - compiler abstraction
  - catalog expansion
topics:
  - pipeline catalog design
  - MODS integration
  - compiler architecture
  - code organization
language: python
date of note: 2025-08-20
---

# Expanded Pipeline Catalog with MODS Integration

## Overview

This design document outlines the architecture for expanding the existing pipeline catalog to support MODS (Model Operations Data Science) pipelines alongside regular SageMaker pipelines, while minimizing code duplication and maintaining clean separation of concerns.

## Problem Statement

The current pipeline catalog uses the standard `PipelineDAGCompiler` from `cursus.core.compiler.dag_compiler`. We need to add support for MODS pipelines that use the `MODSPipelineDAGCompiler` from `cursus.mods.compiler.mods_dag_compiler`, which applies the `MODSTemplate` decorator for enhanced pipeline metadata and operations.

**Key Challenges:**
1. Avoid duplicating pipeline logic between regular and MODS versions
2. Maintain clear separation between compiler types
3. Preserve backward compatibility with existing catalog structure
4. Enable easy discovery and selection of appropriate pipeline type
5. Support future compiler extensions

## Current Architecture Analysis

### Existing Pipeline Catalog Structure
```
src/cursus/pipeline_catalog/
├── index.json                    # Pipeline metadata and discovery
├── utils.py                      # Catalog utilities
├── frameworks/                   # Framework-specific pipelines
│   ├── xgboost/                 # XGBoost pipelines
│   └── pytorch/                 # PyTorch pipelines
└── tasks/                       # Task-oriented views
    ├── training/
    ├── evaluation/
    └── registration/
```

### Compiler Differences
- **Standard Compiler**: `PipelineDAGCompiler` creates `DynamicPipelineTemplate` instances
- **MODS Compiler**: `MODSPipelineDAGCompiler` extends standard compiler and applies `MODSTemplate` decorator
- **Key Difference**: MODS integration provides enhanced metadata, execution tracking, and operational capabilities

## Proposed Architecture: Dual-Compiler Design

### 1. Enhanced Directory Structure

```
src/cursus/pipeline_catalog/
├── index.json                    # Enhanced with compiler_type field
├── utils.py                      # Enhanced with compiler selection logic
├── shared_dags/                  # NEW: Shared DAG definitions
│   ├── __init__.py
│   ├── xgboost/
│   │   ├── simple_dag.py
│   │   ├── training_dag.py
│   │   └── end_to_end_dag.py
│   └── pytorch/
│       ├── training_dag.py
│       └── end_to_end_dag.py
├── frameworks/                   # Existing regular pipelines
│   ├── xgboost/
│   └── pytorch/
├── mods_frameworks/              # NEW: MODS-specific pipelines
│   ├── __init__.py
│   ├── xgboost/
│   │   ├── __init__.py
│   │   ├── simple_mods.py
│   │   ├── training/
│   │   │   ├── __init__.py
│   │   │   ├── with_calibration_mods.py
│   │   │   └── with_evaluation_mods.py
│   │   └── end_to_end/
│   │       ├── __init__.py
│   │       ├── complete_e2e_mods.py
│   │       └── standard_e2e_mods.py
│   └── pytorch/
│       ├── __init__.py
│       ├── simple_mods.py
│       └── training/
│           ├── __init__.py
│           └── basic_training_mods.py
└── tasks/                        # Enhanced to include MODS tasks
    ├── training/
    ├── evaluation/
    ├── registration/
    ├── mods_training/            # NEW: MODS-specific task views
    │   ├── __init__.py
    │   ├── xgboost_training_mods.py
    │   └── pytorch_training_mods.py
    ├── mods_evaluation/          # NEW: MODS evaluation tasks
    │   ├── __init__.py
    │   └── xgboost_evaluation_mods.py
    └── mods_registration/        # NEW: MODS registration tasks
        ├── __init__.py
        ├── xgboost_register_mods.py
        └── pytorch_register_mods.py
```

### 2. Shared DAG Definition Layer

**Core Innovation**: Extract DAG creation logic into reusable functions that both compiler types can use.

```python
# shared_dags/xgboost/simple_dag.py
"""
Shared DAG definition for XGBoost simple pipeline.

This DAG definition can be used by both regular and MODS compilers,
ensuring consistency while avoiding code duplication.
"""

from cursus.api.dag.base_dag import PipelineDAG

def create_xgboost_simple_dag() -> PipelineDAG:
    """
    Create a simple XGBoost training pipeline DAG.
    
    This DAG represents a basic XGBoost training workflow with separate paths
    for training and calibration data.
    
    Returns:
        PipelineDAG: The directed acyclic graph for the pipeline
    """
    dag = PipelineDAG()
    
    # Add nodes
    dag.add_node("CradleDataLoading_training")
    dag.add_node("TabularPreprocessing_training")
    dag.add_node("XGBoostTraining")
    dag.add_node("CradleDataLoading_calibration")
    dag.add_node("TabularPreprocessing_calibration")
    
    # Training flow
    dag.add_edge("CradleDataLoading_training", "TabularPreprocessing_training")
    dag.add_edge("TabularPreprocessing_training", "XGBoostTraining")
    
    # Calibration flow (independent of training)
    dag.add_edge("CradleDataLoading_calibration", "TabularPreprocessing_calibration")
    
    return dag

def get_dag_metadata() -> dict:
    """
    Get metadata for this DAG definition.
    
    Returns:
        dict: Metadata including description, complexity, features
    """
    return {
        "description": "Simple XGBoost training pipeline with data loading and preprocessing",
        "complexity": "simple",
        "features": ["training", "data_loading", "preprocessing"],
        "framework": "xgboost",
        "node_count": 5,
        "edge_count": 3
    }
```

### 3. Enhanced Index Schema

```json
{
  "pipelines": [
    {
      "id": "xgboost-simple",
      "name": "XGBoost Simple Pipeline",
      "path": "frameworks/xgboost/simple.py",
      "shared_dag": "shared_dags/xgboost/simple_dag.py",
      "compiler_type": "standard",
      "framework": "xgboost",
      "complexity": "simple",
      "features": ["training"],
      "description": "Basic XGBoost training pipeline",
      "tags": ["xgboost", "training", "beginner"]
    },
    {
      "id": "xgboost-simple-mods",
      "name": "XGBoost Simple Pipeline (MODS)",
      "path": "mods_frameworks/xgboost/simple_mods.py",
      "shared_dag": "shared_dags/xgboost/simple_dag.py",
      "compiler_type": "mods",
      "framework": "xgboost",
      "complexity": "simple",
      "features": ["training"],
      "description": "MODS-enabled XGBoost training pipeline with enhanced metadata",
      "tags": ["xgboost", "training", "mods", "beginner"],
      "mods_metadata": {
        "author": "default",
        "version": "1.0.0",
        "description": "MODS-enabled XGBoost simple pipeline"
      }
    }
  ]
}
```

### 4. Pipeline Implementation Pattern

**Regular Pipeline** (existing pattern, enhanced):
```python
# frameworks/xgboost/simple.py
from ...shared_dags.xgboost.simple_dag import create_xgboost_simple_dag
from ...core.compiler.dag_compiler import PipelineDAGCompiler

def create_pipeline(
    config_path: str,
    session: PipelineSession,
    role: str,
    pipeline_name: Optional[str] = None,
    **kwargs
) -> Tuple[Pipeline, Dict[str, Any], PipelineDAGCompiler]:
    """Create a SageMaker Pipeline using standard compiler."""
    dag = create_xgboost_simple_dag()
    
    dag_compiler = PipelineDAGCompiler(
        config_path=config_path,
        sagemaker_session=session,
        role=role
    )
    
    pipeline, report = dag_compiler.compile_with_report(dag=dag)
    return pipeline, report, dag_compiler
```

**MODS Pipeline** (new pattern):
```python
# mods_frameworks/xgboost/simple_mods.py
from ...shared_dags.xgboost.simple_dag import create_xgboost_simple_dag
from ...mods.compiler.mods_dag_compiler import MODSPipelineDAGCompiler

def create_pipeline(
    config_path: str,
    session: PipelineSession,
    role: str,
    pipeline_name: Optional[str] = None,
    author: Optional[str] = None,
    version: Optional[str] = None,
    description: Optional[str] = None,
    **kwargs
) -> Tuple[Pipeline, Dict[str, Any], MODSPipelineDAGCompiler]:
    """Create a SageMaker Pipeline using MODS compiler."""
    dag = create_xgboost_simple_dag()
    
    mods_compiler = MODSPipelineDAGCompiler(
        config_path=config_path,
        sagemaker_session=session,
        role=role
    )
    
    pipeline, report = mods_compiler.compile_with_report(
        dag=dag,
        author=author,
        version=version,
        description=description
    )
    return pipeline, report, mods_compiler
```

### 5. Enhanced Catalog Utilities

```python
# utils.py enhancements
from typing import Union, Optional, Dict, Any, Tuple
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession

def create_pipeline_from_catalog(
    pipeline_id: str,
    config_path: str,
    session: PipelineSession,
    role: str,
    **kwargs
) -> Tuple[Pipeline, Dict[str, Any], Union['PipelineDAGCompiler', 'MODSPipelineDAGCompiler']]:
    """
    Universal pipeline creation that auto-selects appropriate compiler.
    
    Args:
        pipeline_id: ID from catalog index
        config_path: Path to configuration file
        session: SageMaker pipeline session
        role: IAM role for pipeline execution
        **kwargs: Additional arguments passed to pipeline creation
        
    Returns:
        Tuple of (Pipeline, report, compiler)
    """
    catalog_entry = get_pipeline_info(pipeline_id)
    
    if catalog_entry["compiler_type"] == "mods":
        # Import MODS pipeline module
        module_path = catalog_entry["path"].replace("/", ".").replace(".py", "")
        module = importlib.import_module(f"cursus.pipeline_catalog.{module_path}")
        
        # Extract MODS-specific parameters
        mods_kwargs = {}
        if "mods_metadata" in catalog_entry:
            mods_kwargs.update(catalog_entry["mods_metadata"])
        mods_kwargs.update(kwargs)
        
        return module.create_pipeline(
            config_path=config_path,
            session=session,
            role=role,
            **mods_kwargs
        )
    else:
        # Import standard pipeline module
        module_path = catalog_entry["path"].replace("/", ".").replace(".py", "")
        module = importlib.import_module(f"cursus.pipeline_catalog.{module_path}")
        
        return module.create_pipeline(
            config_path=config_path,
            session=session,
            role=role,
            **kwargs
        )

def list_pipelines_by_compiler_type(compiler_type: str) -> List[Dict[str, Any]]:
    """List all pipelines of a specific compiler type."""
    catalog = load_catalog_index()
    return [p for p in catalog["pipelines"] if p["compiler_type"] == compiler_type]

def get_mods_pipelines() -> List[Dict[str, Any]]:
    """Get all MODS-enabled pipelines."""
    return list_pipelines_by_compiler_type("mods")

def get_standard_pipelines() -> List[Dict[str, Any]]:
    """Get all standard pipelines."""
    return list_pipelines_by_compiler_type("standard")
```

## Implementation Strategy

### Phase 1: Foundation Setup
1. Create `shared_dags/` directory structure
2. Extract DAG creation logic from existing pipelines
3. Update existing pipelines to use shared DAGs
4. Test backward compatibility

### Phase 2: MODS Pipeline Creation
1. Create `mods_frameworks/` directory structure
2. Implement MODS versions of existing pipelines using shared DAGs
3. Add MODS-specific metadata and configuration options
4. Test MODS pipeline compilation and execution

### Phase 3: Catalog Enhancement
1. Update `index.json` with compiler type information
2. Enhance `utils.py` with compiler selection logic
3. Add MODS-specific task views in `tasks/mods_*/`
4. Update documentation and examples

### Phase 4: Integration and Testing
1. Comprehensive testing of both pipeline types
2. Performance comparison between compilers
3. Documentation updates
4. CLI integration for MODS pipeline discovery

## Benefits of This Design

### 1. Zero Code Duplication
- DAG definitions are shared between regular and MODS versions
- Business logic remains consistent across compiler types
- Maintenance overhead minimized

### 2. Clear Separation of Concerns
- Regular pipelines remain unchanged
- MODS pipelines clearly identified and separated
- Compiler-specific logic isolated

### 3. Backward Compatibility
- Existing pipelines continue to work unchanged
- No breaking changes to current API
- Gradual migration path available

### 4. Unified Discovery Interface
- Single catalog system for both pipeline types
- Consistent metadata and indexing
- Easy comparison between regular and MODS versions

### 5. Extensibility
- Easy to add new compiler types in the future
- Shared DAG layer supports any compiler implementation
- Modular architecture enables independent evolution

### 6. Enhanced User Experience
- Clear choice between regular and MODS pipelines
- Consistent API across pipeline types
- Rich metadata for informed decision making

## Migration Path

### For Existing Users
1. No immediate changes required
2. Existing pipelines continue to work
3. Optional migration to MODS versions when needed

### For New Pipelines
1. Create shared DAG definition first
2. Implement both regular and MODS versions
3. Add appropriate catalog entries
4. Document differences and use cases

## Future Considerations

### Additional Compiler Types
The architecture supports adding new compiler types:
- Kubeflow Pipelines compiler
- Apache Airflow compiler
- Custom enterprise compilers

### Enhanced Metadata
MODS integration enables:
- Pipeline lineage tracking
- Performance monitoring
- Automated documentation generation
- Compliance reporting

### Advanced Features
- Pipeline versioning and rollback
- A/B testing capabilities
- Multi-environment deployment
- Cost optimization tracking

## Conclusion

This dual-compiler architecture provides a clean, extensible solution for integrating MODS pipelines into the existing catalog while maintaining code reuse and backward compatibility. The shared DAG layer ensures consistency while the enhanced indexing system provides clear discovery and selection capabilities.

The design positions the pipeline catalog for future growth while solving the immediate need for MODS integration without code duplication.

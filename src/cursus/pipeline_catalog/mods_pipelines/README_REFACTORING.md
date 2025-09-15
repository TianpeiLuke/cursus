# MODS Pipeline Refactoring Guide

This document explains the refactoring of MODS pipelines to use a unified class-based structure that incorporates PipelineDAGCompiler, maintains the same interface as the original adapter, and keeps pipeline metadata and registry integration.

## Overview

The refactoring introduces a new `BaseMODSPipeline` class that provides:

- **Unified interface** for all MODS pipelines
- **Integration with PipelineDAGCompiler** from `cursus.core.compiler`
- **Pipeline metadata and registry management**
- **Execution document handling**
- **Same init signature** as the original adapter
- **Same `generate_pipeline` method** interface

## Key Components

### 1. BaseMODSPipeline Class

Located in `base_mods_pipeline.py`, this abstract base class provides:

```python
class BaseMODSPipeline(ABC):
    def __init__(self, config_path, sagemaker_session, execution_role, enable_mods=True, validate=True, **kwargs)
    def generate_pipeline(self) -> Pipeline  # Main interface method
    def fill_execution_document(self, execution_doc) -> Dict[str, Any]
    def sync_to_registry(self) -> bool
    
    # Abstract methods that subclasses must implement:
    @abstractmethod
    def create_dag(self) -> PipelineDAG
    @abstractmethod
    def get_enhanced_dag_metadata(self) -> EnhancedDAGMetadata
```

### 2. Key Features

- **PipelineDAGCompiler Integration**: Uses the standard `PipelineDAGCompiler` from `cursus.core.compiler`
- **No MODS-specific compiler**: Simplified to use only the standard compiler
- **Registry Integration**: Built-in `sync_to_registry()` method
- **Execution Document Handling**: Built-in `fill_execution_document()` method
- **Validation**: Optional DAG validation before compilation
- **Template Management**: Automatic template storage and retrieval

## Migration Pattern

### Before (Functional Approach)
```python
# Old functional approach
def create_dag() -> PipelineDAG:
    # DAG creation logic
    pass

def create_pipeline(config_path, session, role, ...):
    # Pipeline creation logic
    pass

def fill_execution_document(pipeline, document, dag_compiler):
    # Document filling logic
    pass
```

### After (Class-based Approach)
```python
# New class-based approach
class MyMODSPipeline(BaseMODSPipeline):
    def create_dag(self) -> PipelineDAG:
        # DAG creation logic
        pass
    
    def get_enhanced_dag_metadata(self) -> EnhancedDAGMetadata:
        # Metadata definition
        pass

# Usage
pipeline_instance = MyMODSPipeline(config_path, session, role)
pipeline = pipeline_instance.generate_pipeline()
```

## Example Implementations

### 1. Dummy Pipeline Example

```python
from cursus.pipeline_catalog.mods_pipelines.dummy_mods_e2e_basic_refactored import DummyMODSE2EBasicPipeline

# Create pipeline instance
pipeline_instance = DummyMODSE2EBasicPipeline(
    config_path="path/to/config.json",
    sagemaker_session=pipeline_session,
    execution_role=role
)

# Generate pipeline
pipeline = pipeline_instance.generate_pipeline()

# Fill execution document
execution_doc = pipeline_instance.fill_execution_document({
    "dummy_model_config": "basic-config",
    "packaging_params": "standard-packaging",
})

# Sync to registry
pipeline_instance.sync_to_registry()
```

### 2. XGBoost Pipeline Example

```python
from cursus.pipeline_catalog.mods_pipelines.xgb_mods_e2e_comprehensive_refactored import XGBoostMODSE2EComprehensivePipeline

# Create pipeline instance
pipeline_instance = XGBoostMODSE2EComprehensivePipeline(
    config_path="path/to/config.json",
    sagemaker_session=pipeline_session,
    execution_role=role
)

# Generate pipeline
pipeline = pipeline_instance.generate_pipeline()

# Fill execution document
execution_doc = pipeline_instance.fill_execution_document({
    "training_dataset": "dataset-training",
    "calibration_dataset": "dataset-calibration",
})
```

## Benefits of the New Structure

### 1. Simplified Architecture
- **Single compiler**: Uses only `PipelineDAGCompiler` from `cursus.core.compiler`
- **No MODS-specific compiler complexity**
- **Cleaner dependencies**

### 2. Consistent Interface
- **Same init signature** as original adapter
- **Same `generate_pipeline()` method**
- **Maintains backward compatibility concepts**

### 3. Better Organization
- **Class-based structure** is more maintainable
- **Clear separation of concerns**
- **Reusable base functionality**

### 4. Enhanced Functionality
- **Built-in registry synchronization**
- **Integrated execution document handling**
- **Automatic template management**
- **Optional validation**

## Migration Steps

To migrate an existing functional pipeline to the new class-based structure:

### Step 1: Create Pipeline Class
```python
class MyPipeline(BaseMODSPipeline):
    def __init__(self, config_path=None, sagemaker_session=None, execution_role=None, **kwargs):
        super().__init__(config_path, sagemaker_session, execution_role, **kwargs)
```

### Step 2: Move DAG Creation
```python
def create_dag(self) -> PipelineDAG:
    # Move your existing create_dag() function logic here
    return dag
```

### Step 3: Move Metadata Definition
```python
def get_enhanced_dag_metadata(self) -> EnhancedDAGMetadata:
    # Move your existing get_enhanced_dag_metadata() function logic here
    return enhanced_metadata
```

### Step 4: Remove Redundant Functions
- Remove standalone `create_dag()` function
- Remove standalone `get_enhanced_dag_metadata()` function  
- Remove standalone `sync_to_registry()` function
- Remove standalone `create_pipeline()` function (use `generate_pipeline()` instead)
- Remove standalone `fill_execution_document()` function (use class method)
- Remove standalone `save_execution_document()` function (use class method)

### Step 5: Update Usage
```python
# Old way
pipeline, report, compiler, template = create_pipeline(config_path, session, role)

# New way
pipeline_instance = MyPipeline(config_path, session, role)
pipeline = pipeline_instance.generate_pipeline()
```

## File Structure

```
src/cursus/pipeline_catalog/mods_pipelines/
├── base_mods_pipeline.py                           # Base class
├── dummy_mods_e2e_basic_refactored.py             # Refactored dummy pipeline
├── xgb_mods_e2e_comprehensive_refactored.py       # Refactored XGBoost pipeline
├── README_REFACTORING.md                          # This guide
└── [other existing pipelines to be refactored]
```

## Best Practices

### 1. Keep DAG Creation Simple
```python
def create_dag(self) -> PipelineDAG:
    # Use existing shared DAG functions
    dag = create_my_shared_dag()
    logger.info(f"Created DAG with {len(dag.nodes)} nodes")
    return dag
```

### 2. Comprehensive Metadata
```python
def get_enhanced_dag_metadata(self) -> EnhancedDAGMetadata:
    # Provide detailed metadata for registry and documentation
    zettelkasten_metadata = ZettelkastenMetadata(
        atomic_id="my_pipeline_refactored",
        title="My Refactored Pipeline",
        # ... other metadata
    )
    return EnhancedDAGMetadata(...)
```

### 3. Use Class Methods
```python
# Use the class methods instead of standalone functions
pipeline_instance.generate_pipeline()      # Instead of create_pipeline()
pipeline_instance.fill_execution_document() # Instead of standalone function
pipeline_instance.sync_to_registry()       # Instead of standalone function
```

##

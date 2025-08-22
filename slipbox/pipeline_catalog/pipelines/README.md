---
tags:
  - code
  - pipeline_catalog
  - pipelines
  - atomic_workflows
  - standard_pipelines
keywords:
  - standard pipelines
  - atomic workflows
  - independent pipelines
  - ML pipelines
  - framework-agnostic
  - self-contained
  - semantic naming
topics:
  - standard pipeline collection
  - atomic pipeline design
  - pipeline independence
  - ML workflow implementation
language: python
date of note: 2025-08-22
---

# Standard Pipelines Collection

## Overview

The `pipelines/` directory contains the standard atomic pipeline collection, implementing Zettelkasten principles of atomicity and independence. Each pipeline represents one coherent workflow concept that can run standalone without dependencies on other pipelines.

## Design Principles

### 1. Atomicity
- **Single Responsibility**: Each pipeline focuses on one coherent ML workflow
- **Complete Functionality**: All necessary steps included within the pipeline
- **Clear Boundaries**: Well-defined inputs, outputs, and side effects
- **Self-Contained**: No dependencies on other pipelines for core functionality

### 2. Independence
- **Standalone Operation**: Can be executed without other pipelines
- **Flexible Input**: Accepts various input formats and preprocessing states
- **Minimal Dependencies**: Only essential external dependencies
- **Stateless Design**: No persistent state between executions

### 3. Semantic Naming
- **Pattern**: `{framework}_{purpose}_{complexity}.py`
- **Self-Documenting**: Names clearly indicate functionality and scope
- **Consistent Structure**: Predictable naming across all pipelines
- **Framework Identification**: Clear indication of ML framework used

## Pipeline Categories

### XGBoost Pipelines

#### xgb_training_simple.py
- **Purpose**: Basic XGBoost model training
- **Complexity**: Simple
- **Features**: Standard training without additional enhancements
- **Use Cases**: Baseline models, quick prototyping, simple classification/regression
- **Input**: Tabular data (raw or preprocessed)
- **Output**: Trained XGBoost model, basic training metrics

#### xgb_training_calibrated.py
- **Purpose**: XGBoost training with probability calibration
- **Complexity**: Standard
- **Features**: Training + probability calibration for better uncertainty estimation
- **Use Cases**: Risk modeling, probability estimation, calibrated classification
- **Input**: Tabular data (raw or preprocessed)
- **Output**: Calibrated XGBoost model, training metrics, calibration metrics

#### xgb_training_evaluation.py
- **Purpose**: XGBoost training with comprehensive evaluation
- **Complexity**: Standard
- **Features**: Training + detailed evaluation metrics and visualizations
- **Use Cases**: Model assessment, performance analysis, detailed reporting
- **Input**: Tabular data with train/validation split
- **Output**: Trained model, comprehensive evaluation report, performance visualizations

#### xgb_e2e_comprehensive.py
- **Purpose**: Complete end-to-end XGBoost workflow
- **Complexity**: Comprehensive
- **Features**: Data preprocessing + training + evaluation + model registration
- **Use Cases**: Production workflows, complete ML pipelines, automated model deployment
- **Input**: Raw tabular data
- **Output**: Deployed model, complete pipeline artifacts, evaluation reports

### PyTorch Pipelines

#### pytorch_training_basic.py
- **Purpose**: Basic PyTorch model training
- **Complexity**: Simple
- **Features**: Standard neural network training with basic configuration
- **Use Cases**: Deep learning baselines, neural network prototyping
- **Input**: Structured or unstructured data (depending on model architecture)
- **Output**: Trained PyTorch model, training history

#### pytorch_e2e_standard.py
- **Purpose**: Standard end-to-end PyTorch workflow
- **Complexity**: Standard
- **Features**: Data preparation + model training + evaluation
- **Use Cases**: Standard deep learning workflows, production neural networks
- **Input**: Raw data (format depends on model type)
- **Output**: Trained model, evaluation metrics, model artifacts

### Dummy Pipelines

#### dummy_e2e_basic.py
- **Purpose**: Basic demonstration and testing pipeline
- **Complexity**: Simple
- **Features**: Minimal workflow for testing and demonstration purposes
- **Use Cases**: Testing, examples, template for new pipelines
- **Input**: Any tabular data
- **Output**: Mock model artifacts, basic metrics

## Pipeline Structure

### Standard Pipeline Template

Each pipeline follows a consistent structure:

```python
"""
Pipeline Name - Brief Description

Detailed description of the pipeline's purpose, features, and use cases.
Includes information about inputs, outputs, and key characteristics.
"""

import logging
from typing import Dict, Any, Optional
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession

# Import DAG and compiler
from ....api.dag.base_dag import PipelineDAG
from ....core.compiler.dag_compiler import PipelineDAGCompiler
from ..shared_dags import EnhancedDAGMetadata, get_enhanced_dag_metadata

logger = logging.getLogger(__name__)

def get_pipeline_metadata() -> EnhancedDAGMetadata:
    """Get enhanced metadata for this pipeline."""
    return get_enhanced_dag_metadata(
        atomic_id="pipeline_id",
        description="Pipeline description",
        # ... other metadata
    )

def create_pipeline_dag() -> PipelineDAG:
    """Create the pipeline DAG."""
    dag = PipelineDAG()
    
    # Add nodes and edges
    # Implementation specific to pipeline purpose
    
    return dag

def compile_pipeline(
    config_path: str,
    sagemaker_session: Optional[PipelineSession] = None,
    role: Optional[str] = None,
    pipeline_name: Optional[str] = None,
    **kwargs
) -> Pipeline:
    """
    Compile the pipeline DAG into a SageMaker Pipeline.
    
    Args:
        config_path: Path to pipeline configuration
        sagemaker_session: SageMaker session (optional)
        role: IAM role for pipeline execution (optional)
        pipeline_name: Name for the pipeline (optional)
        **kwargs: Additional compilation parameters
    
    Returns:
        Compiled SageMaker Pipeline
    """
    # Create DAG
    dag = create_pipeline_dag()
    
    # Compile to pipeline
    compiler = PipelineDAGCompiler(
        config_path=config_path,
        sagemaker_session=sagemaker_session,
        role=role
    )
    
    pipeline = compiler.compile(dag, pipeline_name=pipeline_name, **kwargs)
    
    return pipeline

# Main entry point
def main():
    """Main entry point for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run pipeline")
    parser.add_argument("--config", required=True, help="Configuration file path")
    parser.add_argument("--name", help="Pipeline name")
    
    args = parser.parse_args()
    
    pipeline = compile_pipeline(
        config_path=args.config,
        pipeline_name=args.name
    )
    
    print(f"Pipeline compiled: {pipeline.name}")

if __name__ == "__main__":
    main()
```

### Key Components

#### 1. Metadata Integration
- **Enhanced DAGMetadata**: Type-safe metadata following Zettelkasten principles
- **Atomic Properties**: Clear definition of single responsibility and interfaces
- **Tag-Based Classification**: Multi-dimensional tagging for discovery
- **Connection Information**: Curated relationships with other pipelines

#### 2. DAG Creation
- **Modular Design**: Separate DAG creation from compilation
- **Clear Structure**: Well-organized node and edge definitions
- **Validation**: Built-in validation for DAG integrity
- **Flexibility**: Parameterizable for different configurations

#### 3. Compilation Interface
- **Standard API**: Consistent interface across all pipelines
- **Configuration Support**: External configuration file integration
- **Session Management**: Optional SageMaker session handling
- **Parameter Passing**: Flexible parameter support

#### 4. Standalone Execution
- **Command Line Interface**: Direct execution capability
- **Argument Parsing**: Standard argument handling
- **Error Handling**: Robust error management
- **Logging Integration**: Comprehensive logging support

## Usage Patterns

### Direct Pipeline Usage

```python
from cursus.pipeline_catalog.pipelines.xgb_training_simple import compile_pipeline

# Compile pipeline with configuration
pipeline = compile_pipeline(
    config_path="path/to/config.yaml",
    pipeline_name="my-xgb-training"
)

# Execute pipeline
execution = pipeline.start()
```

### Discovery-Based Usage

```python
from cursus.pipeline_catalog import discover_pipelines, load_pipeline

# Discover XGBoost pipelines
xgb_pipelines = discover_pipelines(framework="xgboost")

# Load specific pipeline
pipeline_func = load_pipeline("xgb_training_simple")

# Use loaded pipeline
pipeline = pipeline_func(
    config_path="path/to/config.yaml",
    pipeline_name="discovered-pipeline"
)
```

### Connection-Based Discovery

```python
from cursus.pipeline_catalog.utils import PipelineCatalogManager

manager = PipelineCatalogManager()

# Find alternatives to current pipeline
alternatives = manager.get_pipeline_connections("xgb_training_simple")["alternatives"]

# Load alternative pipeline
alt_pipeline_func = load_pipeline(alternatives[0])
```

## Integration with Catalog System

### 1. Registry Integration
- **Automatic Registration**: Pipelines automatically registered in catalog
- **Metadata Sync**: Pipeline metadata synced to connection registry
- **Connection Management**: Relationships maintained in registry
- **Discovery Support**: Pipelines discoverable through catalog API

### 2. Tag-Based Organization
- **Framework Tags**: Automatic framework identification
- **Complexity Tags**: Complexity level classification
- **Task Tags**: Purpose and functionality tagging
- **Domain Tags**: Application area identification

### 3. Connection Relationships
- **Alternatives**: Different approaches to same problem
- **Related**: Conceptually similar pipelines
- **Used In**: Composition opportunities in larger workflows

## Quality Assurance

### 1. Validation Requirements
- **DAG Integrity**: Valid DAG structure with proper connections
- **Metadata Completeness**: All required metadata fields populated
- **Interface Consistency**: Standard compilation and execution interface
- **Documentation Standards**: Comprehensive docstrings and comments

### 2. Testing Standards
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end pipeline testing
- **Configuration Tests**: Various configuration scenarios
- **Performance Tests**: Resource usage and execution time

### 3. Code Quality
- **Style Consistency**: Adherence to project coding standards
- **Error Handling**: Robust error management and recovery
- **Logging Integration**: Comprehensive logging throughout execution
- **Resource Management**: Proper cleanup and resource handling

## Performance Considerations

### 1. Resource Efficiency
- **Memory Management**: Efficient memory usage patterns
- **Compute Optimization**: Optimized algorithms and implementations
- **I/O Efficiency**: Minimized data transfer and storage operations
- **Parallel Processing**: Utilization of parallel processing where appropriate

### 2. Scalability
- **Data Size Handling**: Support for various data sizes
- **Compute Scaling**: Ability to scale compute resources
- **Storage Scaling**: Efficient storage utilization
- **Network Optimization**: Minimized network overhead

## Future Enhancements

### 1. Advanced Features
- **Auto-tuning**: Automatic hyperparameter optimization
- **Model Versioning**: Built-in model version management
- **A/B Testing**: Support for model comparison and testing
- **Monitoring Integration**: Built-in performance monitoring

### 2. Framework Expansion
- **Additional Frameworks**: Support for more ML frameworks
- **Custom Models**: Support for custom model architectures
- **Ensemble Methods**: Built-in ensemble pipeline support
- **AutoML Integration**: Integration with automated ML tools

## Related Documentation

- **MODS Pipelines**: See `../mods_pipelines/README.md` for MODS-compatible versions
- **Shared DAGs**: See `../shared_dags/README.md` for reusable components
- **Utilities**: See `../utils/README.md` for catalog utility functions
- **Design Principles**: See `slipbox/1_design/pipeline_catalog_zettelkasten_refactoring.md`

## Conclusion

The standard pipelines collection provides a comprehensive set of atomic, independent ML workflows that embody Zettelkasten principles of knowledge organization. Through consistent design, semantic naming, and robust integration with the catalog system, these pipelines offer both flexibility for individual use and discoverability for systematic exploration.

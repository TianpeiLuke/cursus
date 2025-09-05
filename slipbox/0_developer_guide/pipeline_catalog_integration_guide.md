---
tags:
  - developer_guide
  - pipeline_catalog
  - zettelkasten
  - integration
  - workflow
keywords:
  - pipeline catalog integration
  - zettelkasten pipeline system
  - pipeline discovery
  - catalog registration
  - pipeline connections
topics:
  - pipeline catalog integration
  - zettelkasten methodology
  - pipeline discovery workflow
  - catalog management
  - connection-based navigation
language: python
date of note: 2025-09-05
---

# Pipeline Catalog Integration Guide

## Overview

The Pipeline Catalog is a Zettelkasten-inspired knowledge management system for organizing and discovering pipeline templates. This guide covers how to integrate your newly created pipeline steps with the catalog system, enabling intelligent discovery, connection-based navigation, and automated recommendations.

## Table of Contents

1. [Understanding the Pipeline Catalog](#understanding-the-pipeline-catalog)
2. [Pipeline Catalog Architecture](#pipeline-catalog-architecture)
3. [Integration Workflow](#integration-workflow)
4. [Creating Pipeline Files](#creating-pipeline-files)
5. [Metadata and Connections](#metadata-and-connections)
6. [Registry Synchronization](#registry-synchronization)
7. [Discovery and Navigation](#discovery-and-navigation)
8. [CLI Integration](#cli-integration)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Understanding the Pipeline Catalog

### Zettelkasten Principles

The Pipeline Catalog follows five core Zettelkasten principles:

1. **Atomicity**: Each pipeline represents one atomic concept or workflow
2. **Connectivity**: Explicit connections between related pipelines
3. **Anti-categories**: Tag-based emergent organization instead of rigid hierarchies
4. **Manual linking**: Curated connections over algorithmic search
5. **Dual-form structure**: Separate metadata from implementation

### Catalog Structure

```
src/cursus/pipeline_catalog/
├── pipelines/                    # Standard atomic pipeline units
│   ├── xgb_training_simple.py    # Basic XGBoost training
│   ├── xgb_training_calibrated.py # XGBoost with calibration
│   ├── pytorch_training_basic.py  # Basic PyTorch training
│   └── xgb_e2e_comprehensive.py  # Complete XGBoost workflow
├── mods_pipelines/               # MODS-enhanced variants
├── shared_dags/                  # Shared DAG definitions
├── utils/                        # Zettelkasten utilities
├── catalog_index.json           # Connection registry
└── README.md                     # Catalog documentation
```

## Pipeline Catalog Architecture

### Core Components

1. **CatalogRegistry**: Central registry for pipeline metadata and connections
2. **ConnectionTraverser**: Navigate relationships between pipelines
3. **TagBasedDiscovery**: Find pipelines through emergent tag organization
4. **PipelineRecommendationEngine**: Intelligent pipeline suggestions
5. **EnhancedDAGMetadata**: Bridge between technical DAG data and knowledge metadata

### Metadata Flow

```
┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────┐
│   Shared DAGs   │    │   Pipeline Files     │    │ CatalogRegistry │
│                 │    │                      │    │                 │
│  DAGMetadata    │───▶│  EnhancedDAGMetadata │───▶│ sync_enhanced_  │
│  (Technical)    │    │  (Bridge Layer)      │    │ metadata()      │
│                 │    │                      │    │                 │
│ • description   │    │ • DAGMetadata        │    │                 │
│ • complexity    │    │ • ZettelkastenMeta   │    │                 │
│ • framework     │    │ • Type safety        │    │                 │
│ • node_count    │    │ • Validation         │    │                 │
│ • edge_count    │    │                      │    │                 │
└─────────────────┘    └──────────────────────┘    └─────────────────┘
                                │                           │
                                │                           ▼
                                │                  ┌─────────────────┐
                                │                  │catalog_index.json│
                                │                  │                 │
                                └─────────────────▶│ • Connections   │
                                  (Knowledge)      │ • Tags          │
                                                   │ • Discovery     │
                                                   │ • Navigation    │
                                                   └─────────────────┘
```

## Integration Workflow

### Step 1: Create Shared DAG (if needed)

If your pipeline uses a new DAG pattern, create a shared DAG definition:

```python
# src/cursus/pipeline_catalog/shared_dags/my_framework/my_dag.py
from cursus.api.dag.base_dag import PipelineDAG
from cursus.api.dag.dag_metadata import DAGMetadata

def create_my_custom_dag() -> PipelineDAG:
    """Create a custom DAG for my framework."""
    dag = PipelineDAG()
    
    # Add nodes and edges
    dag.add_node("data_loading", "DataLoadingStep")
    dag.add_node("preprocessing", "PreprocessingStep")
    dag.add_node("training", "MyFrameworkTrainingStep")
    
    dag.add_edge("data_loading", "preprocessing")
    dag.add_edge("preprocessing", "training")
    
    return dag

def get_dag_metadata() -> DAGMetadata:
    """Get technical metadata for the DAG."""
    return DAGMetadata(
        description="Custom framework training pipeline",
        complexity="standard",
        framework="my_framework",
        node_count=3,
        edge_count=2
    )
```

### Step 2: Create Pipeline File

Create your pipeline file in the appropriate directory:

```python
# src/cursus/pipeline_catalog/pipelines/my_framework_training_basic.py
"""
My Framework Basic Training Pipeline

This pipeline implements basic training for my custom framework.
"""

import logging
from typing import Dict, Any, Tuple, Optional
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession

from ...api.dag.base_dag import PipelineDAG
from ...core.compiler.dag_compiler import PipelineDAGCompiler
from ..shared_dags.my_framework.my_dag import create_my_custom_dag
from ..shared_dags.enhanced_metadata import EnhancedDAGMetadata, ZettelkastenMetadata
from ..utils.catalog_registry import CatalogRegistry

logger = logging.getLogger(__name__)

def get_enhanced_dag_metadata() -> EnhancedDAGMetadata:
    """Get enhanced DAG metadata with Zettelkasten integration."""
    zettelkasten_metadata = ZettelkastenMetadata(
        atomic_id="my_framework_training_basic",
        title="My Framework Basic Training Pipeline",
        single_responsibility="Basic training workflow for my custom framework",
        input_interface=["Training dataset path", "model hyperparameters"],
        output_interface=["Trained model artifact"],
        side_effects="Creates model artifacts in S3",
        independence_level="fully_self_contained",
        node_count=3,
        edge_count=2,
        framework="my_framework",
        complexity="basic",
        use_case="Basic training for my framework",
        features=["training", "my_framework", "supervised"],
        mods_compatible=False,
        source_file="pipelines/my_framework_training_basic.py",
        created_date="2025-09-05",
        priority="medium",
        framework_tags=["my_framework"],
        task_tags=["training", "supervised"],
        complexity_tags=["basic", "simple"],
        domain_tags=["machine_learning", "supervised_learning"],
        pattern_tags=["atomic_workflow", "independent"],
        integration_tags=["sagemaker", "s3"],
        quality_tags=["production_ready"],
        data_tags=["tabular", "structured"],
        creation_context="Basic training pipeline for custom framework",
        usage_frequency="medium",
        stability="stable",
        maintenance_burden="low",
        estimated_runtime="20-40 minutes",
        resource_requirements="ml.m5.large or equivalent",
        use_cases=[
            "Basic classification with custom framework",
            "Regression with custom models",
            "Prototyping custom algorithms"
        ],
        skill_level="intermediate",
        # Define connections to other pipelines
        connections={
            "alternatives": [
                {
                    "target_id": "xgb_training_simple",
                    "annotation": "Alternative using XGBoost for similar tasks",
                    "confidence": 0.7
                }
            ],
            "related": [
                {
                    "target_id": "pytorch_training_basic",
                    "annotation": "Similar basic training pattern with PyTorch",
                    "confidence": 0.8
                }
            ],
            "used_in": [
                {
                    "target_id": "my_framework_e2e_comprehensive",
                    "annotation": "Used as training component in comprehensive workflow",
                    "confidence": 0.9
                }
            ]
        }
    )
    
    enhanced_metadata = EnhancedDAGMetadata(
        dag_id="my_framework_training_basic",
        description="Basic training pipeline for my custom framework",
        complexity="basic",
        features=["training", "my_framework", "supervised"],
        framework="my_framework",
        node_count=3,
        edge_count=2,
        zettelkasten_metadata=zettelkasten_metadata
    )
    
    return enhanced_metadata

def create_dag() -> PipelineDAG:
    """Create the pipeline DAG."""
    dag = create_my_custom_dag()
    logger.info(f"Created DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges")
    return dag

def create_pipeline(
    config_path: str,
    session: PipelineSession,
    role: str,
    pipeline_name: Optional[str] = None,
    pipeline_description: Optional[str] = None,
    validate: bool = True
) -> Tuple[Pipeline, Dict[str, Any], PipelineDAGCompiler, Any]:
    """Create a SageMaker Pipeline from the DAG."""
    dag = create_dag()
    
    dag_compiler = PipelineDAGCompiler(
        config_path=config_path,
        sagemaker_session=session,
        role=role
    )
    
    if pipeline_name:
        dag_compiler.pipeline_name = pipeline_name
    if pipeline_description:
        dag_compiler.pipeline_description = pipeline_description
    
    if validate:
        validation = dag_compiler.validate_dag_compatibility(dag)
        if not validation.is_valid:
            logger.warning(f"DAG validation failed: {validation.summary()}")
    
    pipeline, report = dag_compiler.compile_with_report(dag=dag)
    pipeline_template = dag_compiler.get_last_template()
    
    logger.info(f"Pipeline '{pipeline.name}' created successfully")
    
    # Sync to registry after successful pipeline creation
    sync_to_registry()
    
    return pipeline, report, dag_compiler, pipeline_template

def sync_to_registry() -> bool:
    """Synchronize this pipeline's metadata to the catalog registry."""
    try:
        registry = CatalogRegistry()
        enhanced_metadata = get_enhanced_dag_metadata()
        
        success = registry.add_or_update_enhanced_node(enhanced_metadata)
        
        if success:
            logger.info(f"Successfully synchronized {enhanced_metadata.zettelkasten_metadata.atomic_id} to registry")
        else:
            logger.warning(f"Failed to synchronize {enhanced_metadata.zettelkasten_metadata.atomic_id} to registry")
            
        return success
        
    except Exception as e:
        logger.error(f"Error synchronizing to registry: {e}")
        return False
```

## Metadata and Connections

### ZettelkastenMetadata Structure

The `ZettelkastenMetadata` contains comprehensive information about your pipeline:

```python
zettelkasten_metadata = ZettelkastenMetadata(
    # Core Identity
    atomic_id="unique_pipeline_id",
    title="Human-readable pipeline title",
    single_responsibility="One clear responsibility statement",
    
    # Interface Definition
    input_interface=["List of expected inputs"],
    output_interface=["List of produced outputs"],
    side_effects="Description of side effects",
    
    # Technical Properties
    independence_level="fully_self_contained",  # or "requires_dependencies"
    node_count=5,
    edge_count=4,
    framework="framework_name",
    complexity="simple|standard|comprehensive",
    
    # Discovery Tags
    framework_tags=["framework", "library"],
    task_tags=["training", "evaluation", "preprocessing"],
    complexity_tags=["simple", "basic", "advanced"],
    domain_tags=["machine_learning", "deep_learning"],
    pattern_tags=["atomic_workflow", "composite"],
    integration_tags=["sagemaker", "s3", "mods"],
    quality_tags=["production_ready", "tested"],
    data_tags=["tabular", "image", "text"],
    
    # Operational Information
    estimated_runtime="15-30 minutes",
    resource_requirements="ml.m5.large or equivalent",
    skill_level="beginner|intermediate|advanced",
    
    # Connections to other pipelines
    connections={
        "alternatives": [
            {
                "target_id": "alternative_pipeline_id",
                "annotation": "Why this is an alternative",
                "confidence": 0.8
            }
        ],
        "related": [
            {
                "target_id": "related_pipeline_id", 
                "annotation": "How they're related",
                "confidence": 0.7
            }
        ],
        "used_in": [
            {
                "target_id": "composite_pipeline_id",
                "annotation": "How this pipeline is used in the composite",
                "confidence": 0.9
            }
        ]
    }
)
```

### Connection Types

1. **alternatives**: Different approaches to the same problem
2. **related**: Conceptually related pipelines
3. **used_in**: Pipelines that use this one as a component
4. **extends**: Pipelines that build upon this one
5. **requires**: Dependencies this pipeline needs

## Registry Synchronization

### Automatic Sync

Pipelines automatically sync to the registry when created:

```python
def sync_to_registry() -> bool:
    """Synchronize this pipeline's metadata to the catalog registry."""
    try:
        registry = CatalogRegistry()
        enhanced_metadata = get_enhanced_dag_metadata()
        
        success = registry.add_or_update_enhanced_node(enhanced_metadata)
        return success
        
    except Exception as e:
        logger.error(f"Error synchronizing to registry: {e}")
        return False
```

### Manual Sync

You can also sync manually using the CLI:

```bash
# Sync specific pipeline
python -m cursus.pipeline_catalog.pipelines.my_framework_training_basic --sync-registry

# Validate registry integrity
cursus catalog registry validate

# Show registry statistics
cursus catalog registry stats
```

## Discovery and Navigation

### CLI Discovery

Use the CLI for intelligent pipeline discovery:

```bash
# Find pipelines by framework
cursus catalog find --framework my_framework

# Find by tags
cursus catalog find --tags training,supervised

# Find by complexity
cursus catalog find --complexity basic

# Search by use case
cursus catalog find --use-case "basic training workflow"

# Find MODS-compatible pipelines
cursus catalog find --mods-compatible
```

### Connection Navigation

Explore relationships between pipelines:

```bash
# Show all connections for a pipeline
cursus catalog connections --pipeline my_framework_training_basic

# Find alternatives
cursus catalog alternatives --pipeline my_framework_training_basic

# Find connection path between pipelines
cursus catalog path --from my_framework_training_basic --to xgb_training_simple
```

### Recommendations

Get intelligent recommendations:

```bash
# Get recommendations for a use case
cursus catalog recommend --use-case "custom framework training"

# Get next step recommendations
cursus catalog recommend --next-steps my_framework_training_basic

# Generate learning path
cursus catalog recommend --learning-path --framework my_framework
```

### Programmatic Discovery

Use the Python API for discovery in code:

```python
from cursus.pipeline_catalog.utils import PipelineCatalogManager

# Create catalog manager
manager = PipelineCatalogManager()

# Discover by framework
pipelines = manager.discover_pipelines(framework="my_framework")

# Get connections
connections = manager.get_pipeline_connections("my_framework_training_basic")

# Get recommendations
recommendations = manager.get_recommendations(
    use_case="basic training with custom framework"
)

# Find path between pipelines
path = manager.find_path(
    source="my_framework_training_basic",
    target="xgb_training_simple"
)
```

## CLI Integration

### Available Commands

The catalog integrates with the cursus CLI:

```bash
# Registry management
cursus catalog registry validate          # Validate registry integrity
cursus catalog registry stats            # Show registry statistics
cursus catalog registry export           # Export registry data

# Pipeline discovery
cursus catalog find [criteria]           # Find pipelines
cursus catalog list                      # List all pipelines
cursus catalog show [pipeline_id]        # Show pipeline details

# Connection navigation
cursus catalog connections [pipeline_id] # Show connections
cursus catalog alternatives [pipeline_id] # Find alternatives
cursus catalog path [--from] [--to]      # Find connection paths

# Recommendations
cursus catalog recommend [criteria]      # Get recommendations
cursus catalog suggest [pipeline_id]     # Suggest related pipelines
```

### Integration with Workspace CLI

The catalog works with workspace-aware development:

```bash
# Find pipelines in current workspace
cursus catalog find --workspace current

# List workspace-specific pipelines
cursus catalog list --workspace my_project

# Sync workspace pipelines to global catalog
cursus catalog sync --workspace my_project
```

## Best Practices

### Pipeline Design

1. **Atomic Responsibility**: Each pipeline should have one clear responsibility
2. **Clear Interfaces**: Define explicit input and output interfaces
3. **Independence**: Minimize dependencies where possible
4. **Consistent Naming**: Follow the `{framework}_{use_case}_{complexity}` pattern

### Metadata Quality

1. **Comprehensive Tags**: Use multiple tag dimensions for better discovery
2. **Meaningful Connections**: Create connections that add real value
3. **Accurate Descriptions**: Write clear, helpful descriptions
4. **Regular Updates**: Keep metadata current as pipelines evolve

### Connection Strategy

1. **Quality over Quantity**: Focus on meaningful connections
2. **Bidirectional Thinking**: Consider both directions of relationships
3. **Context Annotations**: Explain why connections exist
4. **Confidence Scores**: Use realistic confidence levels

### Registry Management

1. **Regular Validation**: Run registry validation frequently
2. **Sync After Changes**: Always sync after pipeline modifications
3. **Monitor Statistics**: Track registry growth and health
4. **Clean Up Orphans**: Remove unused or broken pipelines

## Troubleshooting

### Common Issues

**Pipeline not found in discovery**
- Check if `sync_to_registry()` was called successfully
- Verify metadata is complete and valid
- Run `cursus catalog registry validate`

**Connection errors**
- Ensure target pipelines exist in registry
- Check connection syntax in metadata
- Validate bidirectional connections

**Import errors in pipeline files**
- Verify shared DAG imports are correct
- Check that all dependencies are available
- Ensure proper Python path setup

**Registry validation failures**
- Review validation report details
- Fix metadata inconsistencies
- Update broken connections

### Debugging Commands

```bash
# Validate specific pipeline
cursus catalog validate --pipeline my_framework_training_basic

# Show detailed registry information
cursus catalog registry stats --verbose

# Export pipeline metadata for inspection
cursus catalog registry export --pipelines my_framework_training_basic

# Test connection traversal
cursus catalog connections --pipeline my_framework_training_basic --verbose
```

### Getting Help

1. Check the pipeline catalog README: `src/cursus/pipeline_catalog/README.md`
2. Review existing pipeline examples in `src/cursus/pipeline_catalog/pipelines/`
3. Use CLI help: `cursus catalog --help`
4. Validate registry integrity: `cursus catalog registry validate`
5. Check logs for sync errors during pipeline creation

## Related Documentation

- [Adding New Pipeline Step](adding_new_pipeline_step.md) - How to create new pipeline steps
- [Creation Process](creation_process.md) - Complete pipeline creation workflow
- [Step Builder Registry Guide](step_builder_registry_guide.md) - Registry system for step builders
- [Step Builder Registry Usage](step_builder_registry_usage.md) - Practical registry usage examples

## Conclusion

The Pipeline Catalog integration enables powerful discovery and navigation of your pipeline ecosystem. By following the Zettelkasten principles and providing comprehensive metadata, your pipelines become part of an intelligent knowledge network that helps developers find, understand, and use the right tools for their tasks.

The key to successful integration is creating atomic, well-connected pipelines with rich metadata that accurately describes their purpose, capabilities, and relationships to other pipelines in the ecosystem.

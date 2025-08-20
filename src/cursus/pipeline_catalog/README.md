# Pipeline Catalog

Welcome to the Cursus Pipeline Catalog, a comprehensive collection of prebuilt pipeline templates to accelerate your model training and deployment workflows.

## Overview

The Pipeline Catalog provides a curated set of pipeline templates organized by machine learning framework and task. These templates are designed to be:

- **Ready to use**: Each pipeline is fully functional out of the box
- **Customizable**: Easily adapt templates to your specific needs
- **Well-documented**: Clear documentation and usage examples
- **Production-grade**: Built using best practices for reliability and scalability

## How to Use the Catalog

### Finding a Pipeline

You can find the right pipeline for your needs in several ways:

#### By Framework

If you know which framework you're using:

```python
# Import an XGBoost pipeline
from cursus.pipeline_catalog.frameworks.xgboost import simple

# Import a PyTorch pipeline
from cursus.pipeline_catalog.frameworks.pytorch.end_to_end import standard_e2e
```

#### By Task

If you're looking for a pipeline to perform a specific task:

```python
# Import a training pipeline
from cursus.pipeline_catalog.tasks.training import xgboost_training

# Import a registration pipeline
from cursus.pipeline_catalog.tasks.registration import pytorch_register
```

### Using the CLI

The pipeline catalog includes a command-line interface for discovery:

```bash
# List all available pipelines
python -m cursus.cli.catalog_cli list

# Search for pipelines matching criteria
python -m cursus.cli.catalog_cli search --framework xgboost --feature calibration

# Show details for a specific pipeline
python -m cursus.cli.catalog_cli show xgboost-complete

# Generate a new pipeline based on a template
python -m cursus.cli.catalog_cli generate xgboost-simple --output my_pipeline.py
```

## Catalog Structure

```
pipeline_catalog/
├── frameworks/               # Organized by ML framework
│   ├── xgboost/              # XGBoost pipelines
│   │   ├── simple.py         # Simple training
│   │   ├── training/         # Training variants
│   │   ├── end_to_end/       # End-to-end pipelines
│   │   └── components/       # Individual components
│   └── pytorch/              # PyTorch pipelines
│       ├── training/
│       └── end_to_end/
├── tasks/                    # Organized by task
│   ├── training/             # Training pipelines
│   ├── evaluation/           # Evaluation pipelines
│   ├── registration/         # Registration pipelines
│   └── data_processing/      # Data processing pipelines
└── examples/                 # Special case examples
```

## Pipeline Complexity Guide

Pipelines are categorized by complexity:

- **Simple**: Basic pipelines with minimal components, ideal for getting started
- **Standard**: Balanced pipelines with common features for typical use cases
- **Complete**: Full-featured pipelines with all components for production use

## Feature Availability

This table shows which features are available in different pipeline templates:

| Pipeline | Training | Evaluation | Calibration | Registration | Data Loading |
|----------|:--------:|:----------:|:-----------:|:------------:|:------------:|
| xgboost-simple | ✅ | ❌ | ❌ | ❌ | ✅ |
| xgboost-train-evaluate | ✅ | ✅ | ❌ | ❌ | ✅ |
| xgboost-train-calibrate | ✅ | ❌ | ✅ | ❌ | ✅ |
| xgboost-complete-e2e | ✅ | ✅ | ✅ | ✅ | ✅ |
| pytorch-training | ✅ | ❌ | ❌ | ❌ | ✅ |
| pytorch-e2e | ✅ | ✅ | ❌ | ✅ | ✅ |

## Example Usage

Here's a complete example of using a pipeline from the catalog:

```python
from cursus.pipeline_catalog.frameworks.xgboost import simple
from sagemaker import Session
from sagemaker.workflow.pipeline_context import PipelineSession

# Initialize session
sagemaker_session = Session()
role = sagemaker_session.get_caller_identity_arn()
pipeline_session = PipelineSession()

# Create the pipeline
pipeline, report, dag_compiler = simple.create_pipeline(
    config_path="path/to/config.json",
    session=pipeline_session,
    role=role
)

# Get execution document
execution_doc = simple.fill_execution_document(
    pipeline=pipeline,
    document={"param1": "value1"},
    dag_compiler=dag_compiler
)

# Upsert and execute
pipeline.upsert()
execution = pipeline.start(execution_input=execution_doc)
```

## Contributing

To add a new pipeline to the catalog, please follow the [contribution guide](CONTRIBUTING.md).

## Getting Help

If you're having trouble finding or using a pipeline, please:

1. Check the [documentation](https://cursus.readthedocs.io)
2. Use the CLI search tool to find alternatives
3. Reach out to the support team

Happy pipeline building!

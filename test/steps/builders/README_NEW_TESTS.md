# New Step Builder Tests

This document describes the new comprehensive test suites created for Training, Transform, and CreateModel step builders, following the pattern established by the Processing step builder tests.

## Overview

Based on the SageMaker step types defined in `cursus/steps/registry/step_names.py`, the following test suites have been created:

### Training Step Builders (sagemaker_step_type: "Training")
- **PyTorchTraining** - PyTorch model training step
- **XGBoostTraining** - XGBoost model training step

### Transform Step Builders (sagemaker_step_type: "Transform")  
- **BatchTransform** - Batch transform step

### CreateModel Step Builders (sagemaker_step_type: "CreateModel")
- **PyTorchModel** - PyTorch model creation step
- **XGBoostModel** - XGBoost model creation step

## Files Created

### Test Files
1. `test_training_step_builders.py` - Comprehensive test suite for Training step builders
2. `test_transform_step_builders.py` - Comprehensive test suite for Transform step builders  
3. `test_createmodel_step_builders.py` - Comprehensive test suite for CreateModel step builders

### Runner Scripts
1. `run_training_tests.py` - Runner script with reporting for Training tests
2. `run_transform_tests.py` - Runner script with reporting for Transform tests
3. `run_createmodel_tests.py` - Runner script with reporting for CreateModel tests

## Test Structure

Each test suite follows the same pattern as `test_processing_step_builders.py`:

### Universal Tests
- Inheritance validation
- Required methods validation
- Configuration validation
- Specification integration
- Contract alignment

### Step-Type-Specific Tests

#### Training Steps
- Estimator creation method validation
- Expected estimator type validation
- Training input/output methods
- Environment variables handling
- Hyperparameters handling
- Framework-specific validation
- Training step creation

#### Transform Steps
- Transformer creation method validation
- Expected transformer type validation
- Transform input/output methods
- Job type validation
- Model name handling
- Transform step creation

#### CreateModel Steps
- Model creation method validation
- Expected model type validation
- CreateModel input/output methods
- Environment variables handling
- Model data handling
- Framework-specific validation
- CreateModel step creation

## Report Generation

Each runner script generates:

### Directory Structure
```
test/steps/builders/
├── reports/
│   ├── training_step_builders_report_TIMESTAMP.json
│   ├── transform_step_builders_report_TIMESTAMP.json
│   ├── createmodel_step_builders_report_TIMESTAMP.json
│   ├── training_overall_summary.png
│   ├── transform_overall_summary.png
│   └── createmodel_overall_summary.png
├── pytorchtraining_training/
│   └── scoring_reports/
│       ├── PyTorchTrainingTrainingStepBuilder_score_chart.png
│       └── PyTorchTrainingTrainingStepBuilder_score_report.json
├── xgboosttraining_training/
│   └── scoring_reports/
│       ├── XGBoostTrainingTrainingStepBuilder_score_chart.png
│       └── XGBoostTrainingTrainingStepBuilder_score_report.json
├── batchtransform_transform/
│   └── scoring_reports/
│       ├── BatchTransformTransformStepBuilder_score_chart.png
│       └── BatchTransformTransformStepBuilder_score_report.json
├── pytorchmodel_createmodel/
│   └── scoring_reports/
│       ├── PyTorchModelCreateModelStepBuilder_score_chart.png
│       └── PyTorchModelCreateModelStepBuilder_score_report.json
└── xgboostmodel_createmodel/
    └── scoring_reports/
        ├── XGBoostModelCreateModelStepBuilder_score_chart.png
        └── XGBoostModelCreateModelStepBuilder_score_report.json
```

### Report Contents
- **JSON Reports**: Detailed test results, scoring data, and summary statistics
- **Score Charts**: Visual representation of test results with pie charts and category breakdowns
- **Overall Summary Charts**: Bar charts showing pass rates across all builders

## Usage

### Running Individual Test Suites

```bash
# Training step builders
python test/steps/builders/run_training_tests.py

# Transform step builders  
python test/steps/builders/run_transform_tests.py

# CreateModel step builders
python test/steps/builders/run_createmodel_tests.py
```

### Running Individual Test Files

```bash
# Using unittest
python -m unittest test.steps.builders.test_training_step_builders
python -m unittest test.steps.builders.test_transform_step_builders
python -m unittest test.steps.builders.test_createmodel_step_builders

# Using pytest
pytest test/steps/builders/test_training_step_builders.py -v
pytest test/steps/builders/test_transform_step_builders.py -v
pytest test/steps/builders/test_createmodel_step_builders.py -v
```

## Key Features

### Centralized Registry-Based Discovery
- **Centralized Discovery Utilities**: Tests use `cursus.validation.builders.registry_discovery` module for consistent step builder discovery across all test suites
- **Automatic Step Discovery**: Tests automatically discover step builders using `get_training_steps_from_registry()`, `get_transform_steps_from_registry()`, etc.
- **Dynamic Class Loading**: Builder classes are loaded using centralized `load_builder_class()` function from the validation framework
- **Adaptive to Changes**: When new step builders are added to the registry, tests automatically include them without code changes
- **Accurate Categorization**: All step categorizations are based on the `sagemaker_step_type` field in the registry
- **Reusable Components**: Discovery methods are now available in the main validation framework for use by other testing components

### Comprehensive Testing
- Universal compliance testing using the existing `UniversalStepBuilderTest` framework
- Step-type-specific testing for specialized requirements
- Structural validation without requiring full configuration setup
- Error handling and detailed reporting

### Visual Reporting
- Individual score charts for each step builder
- Overall summary charts for each step type
- Color-coded results (green=success, yellow=partial, red=error)
- Detailed JSON reports with scoring data

### Subfolder Organization
Following the pattern of `tabular_preprocessing/scoring_reports/`, each step builder gets its own subfolder with the naming convention:
- `{step_name_lower}_{step_type_lower}/scoring_reports/`
- Example: `pytorchtraining_training/scoring_reports/`

## Integration

These tests integrate seamlessly with the existing test infrastructure:
- Use the same `UniversalStepBuilderTest` framework
- Follow the same reporting patterns as Processing tests
- Generate compatible report formats
- Maintain the same directory structure conventions

## Dependencies

The tests require:
- `matplotlib` for chart generation
- `cursus.validation.builders.universal_test` for universal testing
- `cursus.steps.registry.step_names` for step categorization
- Standard Python libraries: `unittest`, `pytest`, `json`, `pathlib`, `datetime`

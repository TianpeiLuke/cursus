# Pipeline Step Specifications

This module contains concrete step specifications for all pipeline components. These specifications define the input/output requirements, parameters, and metadata for each step type, enabling automatic dependency resolution and pipeline construction.

## Step Categories

### Data Loading Specifications
- **data_loading_spec.py** - Base data loading specification
- **data_loading_training_spec.py** - Training data loading
- **data_loading_validation_spec.py** - Validation data loading  
- **data_loading_testing_spec.py** - Testing data loading
- **data_loading_calibration_spec.py** - Calibration data loading

### Preprocessing Specifications
- **preprocessing_spec.py** - Base preprocessing specification
- **preprocessing_training_spec.py** - Training data preprocessing
- **preprocessing_validation_spec.py** - Validation data preprocessing
- **preprocessing_testing_spec.py** - Testing data preprocessing
- **preprocessing_calibration_spec.py** - Calibration data preprocessing

### Model Specifications
- **pytorch_model_spec.py** - PyTorch model specification
- **xgboost_model_spec.py** - XGBoost model specification
- **pytorch_training_spec.py** - PyTorch training specification
- **xgboost_training_spec.py** - XGBoost training specification

### Pipeline Operations
- **model_eval_spec.py** - Model evaluation specification
- **packaging_spec.py** - Model packaging specification
- **payload_spec.py** - Payload generation specification
- **registration_spec.py** - Model registration specification

## Key Features

1. **Comprehensive Coverage** - Specifications for all pipeline step types
2. **Job Type Variants** - Separate specs for training/validation/testing/calibration
3. **Framework Support** - PyTorch and XGBoost model specifications
4. **Dependency Resolution** - Compatible input/output specifications
5. **Semantic Matching** - Rich semantic tags for intelligent matching
6. **Script Contract Integration** - Automated script validation against specifications
7. **Registry Integration** - Centralized specification management and discovery

## Usage Pattern

```python
from src.pipeline_step_specs import (
    DATA_LOADING_TRAINING_SPEC,
    PREPROCESSING_TRAINING_SPEC,
    XGBOOST_TRAINING_SPEC
)

# Access step specifications
print(DATA_LOADING_TRAINING_SPEC.inputs)
print(PREPROCESSING_TRAINING_SPEC.outputs)
print(XGBOOST_TRAINING_SPEC.parameters)
```

## Job Type Variants

Many specifications have variants for different job types:

### Training Pipeline
- `DATA_LOADING_TRAINING_SPEC`
- `PREPROCESSING_TRAINING_SPEC`
- `XGBOOST_TRAINING_SPEC`

### Validation Pipeline
- `DATA_LOADING_VALIDATION_SPEC`
- `PREPROCESSING_VALIDATION_SPEC`

### Testing Pipeline
- `DATA_LOADING_TESTING_SPEC`
- `PREPROCESSING_TESTING_SPEC`

### Calibration Pipeline
- `DATA_LOADING_CALIBRATION_SPEC`
- `PREPROCESSING_CALIBRATION_SPEC`

## Integration

This module integrates with:
- **Pipeline Dependencies** - Provides specifications for dependency resolution
- **Pipeline Builder** - Uses specifications for automatic pipeline construction
- **Step Builders** - Validates step configurations against specifications
- **Script Contracts** - Aligns with script execution requirements

## Related Design Documentation

For architectural context and design decisions, see:
- **[Specification Driven Design](../pipeline_design/specification_driven_design.md)** - Overall design philosophy
- **[Step Specification Design](../pipeline_design/step_specification.md)** - Step specification patterns and conventions
- **[Step Builder Design](../pipeline_design/step_builder.md)** - Step builder architecture
- **[Pipeline Template Builder](../pipeline_design/pipeline_template_builder_v2.md)** - Template-based pipeline construction
- **[Standardization Rules](../pipeline_design/standardization_rules.md)** - Naming and structure conventions
- **[Design Principles](../pipeline_design/design_principles.md)** - Core design principles

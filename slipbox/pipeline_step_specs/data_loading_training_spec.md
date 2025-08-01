# Data Loading Training Specification

## Overview
The Data Loading Training Specification defines the input/output requirements and parameters for loading training data from Cradle data sources. This specification enables automatic dependency resolution and pipeline construction for training data loading steps.

## Specification Details

### Step Information
- **Step Name**: `cradle_data_loading_training`
- **Step Type**: `data_loading`
- **Job Type**: `training`
- **Framework**: Cradle Data Loading

### Input Requirements

| Input Name | Data Type | Format | Semantic Tags | Description |
|------------|-----------|--------|---------------|-------------|
| source_config | config | json | ["data_source", "cradle_config"] | Cradle data source configuration |
| query_params | parameters | json | ["query_parameters", "training_data"] | Query parameters for training data |

### Output Specifications

| Output Name | Data Type | Format | Semantic Tags | Description |
|-------------|-----------|--------|---------------|-------------|
| raw_data | tabular | parquet | ["raw_data", "training_data", "cradle_source"] | Raw training data from Cradle |
| metadata | metadata | json | ["data_metadata", "training_info"] | Data loading metadata and statistics |
| signature | signature | json | ["data_signature", "schema_info"] | Data schema and signature information |

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| data_source | str | Required | Cradle data source identifier |
| query_filter | str | None | Optional query filter for data selection |
| date_range | dict | None | Date range for data loading |
| sample_size | int | None | Optional sample size limit |
| include_metadata | bool | true | Whether to include metadata output |

### Constraints

#### Input Constraints
- source_config must contain valid Cradle connection parameters
- query_params must include required query fields
- date_range must be valid date range if specified

#### Output Constraints
- raw_data must be non-empty dataset
- metadata must include row count and column information
- signature must contain valid schema definition

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| CRADLE_CONNECTION_STRING | Yes | Cradle database connection string |
| CRADLE_API_KEY | Yes | API key for Cradle access |
| DATA_LOADING_TIMEOUT | No | Timeout for data loading operations |

## Usage Example

```python
from src.pipeline_step_specs import DATA_LOADING_TRAINING_SPEC
from src.pipeline_deps import DependencyResolver

# Access specification details
print(f"Step Name: {DATA_LOADING_TRAINING_SPEC.step_name}")
print(f"Inputs: {list(DATA_LOADING_TRAINING_SPEC.inputs.keys())}")
print(f"Outputs: {list(DATA_LOADING_TRAINING_SPEC.outputs.keys())}")

# Use in dependency resolution
resolver = DependencyResolver()
dependencies = resolver.resolve_dependencies([DATA_LOADING_TRAINING_SPEC])
```

## Integration with Pipeline Builder

### Step Builder Mapping
```python
from src.pipeline_steps import CradleDataLoadingStepBuilder

step_builder_map = {
    "CradleDataLoadingStep": CradleDataLoadingStepBuilder
}
```

### Configuration Example
```python
from src.pipeline_steps import CradleDataLoadingConfig

config = CradleDataLoadingConfig(
    data_source="training_dataset",
    query_filter="status = 'active'",
    date_range={
        "start_date": "2024-01-01",
        "end_date": "2024-12-31"
    },
    sample_size=100000,
    job_type="training"
)
```

## Dependency Relationships

### Upstream Dependencies
- None (this is typically a source step)

### Downstream Dependencies
- **Preprocessing Steps** - Consumes raw_data output
- **Data Validation Steps** - Uses metadata and signature outputs
- **Quality Checks** - Validates data integrity

### Compatible Specifications
- `PREPROCESSING_TRAINING_SPEC` - Accepts raw_data as input
- `DATA_VALIDATION_SPEC` - Uses metadata for validation
- `QUALITY_CHECK_SPEC` - Processes signature information

## Semantic Matching

### Output Semantic Tags
- **raw_data**: `["raw_data", "training_data", "cradle_source"]`
  - Matches preprocessing steps expecting raw training data
  - Compatible with tabular data processors
  - Indicates Cradle data source origin

- **metadata**: `["data_metadata", "training_info"]`
  - Matches validation steps requiring metadata
  - Compatible with quality check processes
  - Provides training-specific information

- **signature**: `["data_signature", "schema_info"]`
  - Matches schema validation steps
  - Compatible with data profiling processes
  - Enables schema consistency checks

## Validation Rules

### Specification Validation
- Step name must follow naming convention: `{source}_data_loading_{job_type}`
- All required parameters must be specified
- Input/output specifications must be valid DataSpecification objects
- Semantic tags must be non-empty lists

### Runtime Validation
- Cradle connection must be accessible
- Query parameters must be valid for data source
- Output data must meet minimum quality thresholds
- Metadata must include required fields

## Related Specifications

### Job Type Variants
- `DATA_LOADING_VALIDATION_SPEC` - Validation data loading
- `DATA_LOADING_TESTING_SPEC` - Testing data loading
- `DATA_LOADING_CALIBRATION_SPEC` - Calibration data loading

### Related Steps
- `PREPROCESSING_TRAINING_SPEC` - Next step in training pipeline
- `DATA_QUALITY_SPEC` - Data quality validation
- `SCHEMA_VALIDATION_SPEC` - Schema consistency checking

## Best Practices

1. **Consistent Naming** - Use consistent naming patterns for job type variants
2. **Rich Metadata** - Include comprehensive metadata for downstream steps
3. **Schema Documentation** - Provide detailed schema information in signatures
4. **Error Handling** - Define clear error conditions and recovery strategies
5. **Performance Optimization** - Consider data loading performance implications

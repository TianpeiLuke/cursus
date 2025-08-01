# Essential Input Models Documentation

## Overview

The `essential_input_models` module implements Pydantic models for the essential user inputs (Tier 1) in the three-tier configuration architecture. These models represent the three key areas of configuration: Data, Model, and Registration, providing a structured and validated interface for user inputs.

## Purpose

The Essential Input Models serve several key purposes in the configuration system:

1. Provide a structured representation of essential user inputs
2. Enforce validation rules to ensure input correctness
3. Document the required and optional inputs clearly
4. Generate derived properties from basic inputs where appropriate
5. Simplify the user interface by focusing on core business decisions

## Implementation

The essential input models are implemented as Pydantic models, which provide automatic validation, documentation, and serialization capabilities:

```python
from datetime import datetime
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, validator

class DateRangePeriod(BaseModel):
    """Represents a date range period for training or calibration"""
    start_date: datetime = Field(..., description="Start date of the period")
    end_date: datetime = Field(..., description="End date of the period")

    @validator('end_date')
    def end_date_must_be_after_start_date(cls, end_date, values):
        """Validate that end_date is after start_date"""
        if 'start_date' in values and end_date <= values['start_date']:
            raise ValueError('end_date must be after start_date')
        return end_date

class DataConfig(BaseModel):
    """Essential data configuration"""
    region: str = Field(..., description="Region code (NA, EU, FE)")
    training_period: DateRangePeriod = Field(..., description="Training data date range")
    # Additional fields...
```

## Key Models

### 1. DateRangePeriod

A utility model that represents a time period with start and end dates:

- `start_date`: Beginning of the period
- `end_date`: End of the period
- Includes validation to ensure end_date is after start_date

### 2. DataConfig

Represents the data-related essential inputs:

- `region`: Region code (NA, EU, FE)
- `training_period`: Date range for training data
- `calibration_period`: Date range for calibration data
- `feature_groups`: Dictionary of feature groups to include (name -> boolean)
- `custom_fields`: Additional custom fields to include beyond feature groups

This model also provides derived properties:
- `tab_field_list`: List of tabular fields derived from feature groups and custom fields
- `cat_field_list`: List of categorical fields derived from feature groups
- `full_field_list`: Combined list of all fields

### 3. ModelConfig

Represents the model-related essential inputs:

- `is_binary`: Boolean indicating if this is a binary classification model
- `label_name`: Name of the target/label field
- `id_name`: Name of the ID field
- `marketplace_id_col`: Name of the marketplace ID column
- `num_round`: Number of boosting rounds
- `max_depth`: Maximum depth of a tree
- `min_child_weight`: Minimum sum of instance weight needed in a child

The model includes validators for:
- Ensuring `num_round` is positive
- Ensuring `max_depth` is positive

### 4. RegistrationConfig

Represents the model registration essential inputs:

- `model_owner`: Owner of the model (e.g., team name)
- `model_registration_domain`: Domain for model registration
- `expected_tps`: Expected transactions per second
- `max_latency_ms`: Maximum allowed latency in milliseconds
- `max_error_rate`: Maximum acceptable error rate

The model includes validators for:
- Ensuring `expected_tps` is positive
- Ensuring `max_latency_ms` is positive
- Ensuring `max_error_rate` is between 0 and 1

### 5. EssentialInputs

Combines all three configuration areas into a single model:

- `data`: Data configuration
- `model`: Model configuration
- `registration`: Registration configuration

This model provides an `expand()` method that flattens the nested structure into a dictionary of configuration values, suitable for creating configuration objects.

## Field Documentation and Defaults

Each field in the models is documented using Pydantic's `Field` with a description:

```python
region: str = Field(..., description="Region code (NA, EU, FE)")
```

Fields with default values are provided with those defaults in the model definition:

```python
is_binary: bool = Field(True, description="Whether this is a binary classification model")
expected_tps: int = Field(2, description="Expected transactions per second")
```

Required fields use the `...` ellipsis syntax to indicate they must be provided:

```python
region: str = Field(..., description="Region code (NA, EU, FE)")
model_owner: str = Field(..., description="Owner of the model (e.g., team name)")
```

## Validation Rules

The models implement validators using Pydantic's `@validator` decorator:

```python
@validator('end_date')
def end_date_must_be_after_start_date(cls, end_date, values):
    """Validate that end_date is after start_date"""
    if 'start_date' in values and end_date <= values['start_date']:
        raise ValueError('end_date must be after start_date')
    return end_date
```

These validators ensure data integrity and provide clear error messages when inputs don't meet the requirements.

## Derived Properties

Some models include derived properties that compute values based on the provided inputs:

```python
@property
def full_field_list(self) -> List[str]:
    """Generate the full field list (combination of tabular and categorical)"""
    return sorted(set(self.tab_field_list + self.cat_field_list))
```

These properties help bridge the gap between the simplified essential inputs and the more detailed configuration expected by the pipeline.

## Usage

The Essential Input Models are designed to be used at the beginning of the configuration process:

```python
# Create essential inputs
data_config = DataConfig(
    region="NA",
    training_period={"start_date": datetime(2023, 1, 1), "end_date": datetime(2023, 6, 30)},
    calibration_period={"start_date": datetime(2023, 7, 1), "end_date": datetime(2023, 7, 31)},
    feature_groups={"buyer_profile": True, "order_history": True},
    custom_fields=["custom_field_1", "custom_field_2"]
)

model_config = ModelConfig(
    is_binary=True,
    label_name="is_abuse",
    id_name="order_id"
)

registration_config = RegistrationConfig(
    model_owner="Test Team",
    model_registration_domain="test-domain"
)

# Combine into a single essential inputs object
inputs = EssentialInputs(
    data=data_config,
    model=model_config,
    registration=registration_config
)

# Expand to a flat dictionary of configuration values
expanded = inputs.expand()

# Use the expanded values to create configuration objects
model_hyperparams = ModelHyperparameters(**expanded)
pipeline_config = PipelineConfig(**expanded)
```

## Integration with Three-Tier Architecture

Within the three-tier architecture, the Essential Input Models implement Tier 1 (Essential User Inputs). The typical workflow is:

1. **Collect essential user inputs (Tier 1)** using the Essential Input Models
2. Expand the essential inputs into configuration objects
3. Apply system defaults (Tier 2) using `DefaultValuesProvider`
4. Derive dependent fields (Tier 3) using `FieldDerivationEngine`
5. Generate final configuration

## Benefits

1. **Input Validation**: Automatic validation ensures correct input values
2. **Clear Documentation**: Field descriptions and types are clearly documented
3. **Structured Interface**: Organized, logical grouping of related inputs
4. **Error Handling**: Descriptive error messages for invalid inputs
5. **Extensibility**: Easy to add new fields or validation rules

## Future Enhancements

1. **Advanced Validation**: More sophisticated cross-field validation rules
2. **Feature Group Integration**: Direct integration with the feature group registry
3. **UI Widgets**: Generation of UI widgets based on the model definitions
4. **Schema Generation**: Automatic generation of JSON Schema for documentation
5. **Versioning Support**: Handling versioned models with migration paths

## Implementation Status

The Essential Input Models have been fully implemented as part of Phase 1 of the Essential Inputs Implementation Plan. They provide a complete representation of all required Tier 1 inputs with appropriate validation rules and documentation.

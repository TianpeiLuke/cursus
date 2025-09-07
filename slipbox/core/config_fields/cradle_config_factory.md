---
tags:
  - code
  - core
  - config_fields
  - factory
  - cradle_integration
keywords:
  - cradle config factory
  - configuration generation
  - AWS region mapping
  - EDX manifest
  - SQL transformation
topics:
  - configuration management
  - factory pattern
  - cradle integration
language: python
date of note: 2025-09-07
---

# Cradle Config Factory

## Overview

The `cradle_config_factory` module provides factory functions for generating specialized configuration objects for Cradle data loading operations. It handles AWS region mapping, EDX manifest creation, and SQL transformation generation.

## Core Functions

### Region Mapping

#### `_map_region_to_aws_region(region: str) -> str`

Maps internal region identifiers to AWS region names.

**Parameters:**
- `region`: Internal region identifier

**Returns:**
- AWS region string

### Schema Generation

#### `_create_field_schema(fields: List[str]) -> List[Dict[str, str]]`

Creates field schema definitions from a list of field names.

**Parameters:**
- `fields`: List of field names

**Returns:**
- List of field schema dictionaries

### EDX Manifest Operations

#### `_format_edx_manifest_key(...)`

Formats EDX manifest keys according to Cradle specifications.

#### `_create_edx_manifest(...)`

Creates complete EDX manifest configurations.

#### `_create_edx_manifest_from_key(...)`

Creates EDX manifest from a specific key configuration.

### SQL Generation

#### `_generate_transform_sql(...)`

Generates SQL transformation queries for data processing.

#### `_get_all_fields(...)`

Retrieves all available fields for configuration.

### Main Factory Functions

#### `create_cradle_data_load_config(...)`

Creates comprehensive Cradle data loading configurations.

**Features:**
- AWS region configuration
- Data source mapping
- Field schema generation
- Transformation rules

#### `create_training_and_calibration_configs(...)`

Creates specialized configurations for training and calibration workflows.

**Features:**
- Training data configuration
- Calibration data setup
- Model parameter mapping
- Validation rules

## Usage Patterns

### Basic Configuration Creation

```python
from cursus.core.config_fields.cradle_config_factory import create_cradle_data_load_config

# Create data loading configuration
config = create_cradle_data_load_config(
    region="us-west-2",
    data_source="s3://my-bucket/data/",
    fields=["field1", "field2", "field3"]
)
```

### Training Configuration

```python
from cursus.core.config_fields.cradle_config_factory import create_training_and_calibration_configs

# Create training and calibration configs
train_config, calib_config = create_training_and_calibration_configs(
    model_type="xgboost",
    training_data="s3://bucket/train/",
    calibration_data="s3://bucket/calib/"
)
```

## Implementation Details

### Factory Pattern

1. **Encapsulation**: Hides complex configuration creation logic
2. **Standardization**: Ensures consistent configuration structure
3. **Flexibility**: Supports various configuration scenarios
4. **Validation**: Built-in validation for configuration parameters

### AWS Integration

- **Region Mapping**: Automatic AWS region resolution
- **S3 Path Handling**: Proper S3 URI formatting
- **Service Configuration**: AWS service-specific settings

### Data Processing

- **Schema Generation**: Automatic field schema creation
- **SQL Generation**: Dynamic SQL query construction
- **Transformation Rules**: Data transformation specifications

## Dependencies

- **AWS Services**: For region and service configuration
- **SQL Generation**: For query construction
- **Configuration Classes**: For structured configuration objects

## Related Components

- [`config_merger.md`](config_merger.md): Configuration merging functionality
- [`type_aware_config_serializer.md`](type_aware_config_serializer.md): Configuration serialization
- [`config_field_categorizer.md`](config_field_categorizer.md): Field categorization

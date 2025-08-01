# MIMS Payload Generation Module Documentation

## Overview

The `mims_payload.py` script is responsible for generating sample payload files for model inference in the Machine Intelligence Model System (MIMS). It extracts field information from a model's hyperparameters, determines appropriate data types for each field, and creates example request payloads in supported formats (JSON and CSV). These sample payloads serve as templates for making inference requests to deployed models.

## Key Features

- Extracts model field information from hyperparameters.json
- Generates sample payloads in multiple formats (JSON, CSV)
- Supports customization of default values through environment variables
- Provides special field handling with dynamic value generation
- Creates a compressed archive of payload samples for deployment
- Handles robust extraction of hyperparameters from various model artifact structures

## Core Components

### Environment Variable Configuration

The script uses several environment variables to customize its behavior:

- **CONTENT_TYPES**: Comma-separated list of content types to generate (default: "application/json")
- **DEFAULT_NUMERIC_VALUE**: Default value for numeric fields (default: "0.0")
- **DEFAULT_TEXT_VALUE**: Default value for text fields (default: "DEFAULT_TEXT")
- **SPECIAL_FIELD_***: Special field handling with custom templates (e.g., SPECIAL_FIELD_TIMESTAMP)

### Data Type Classification

Fields are classified into two main types:
- **NUMERIC**: Quantitative fields that contain numerical values
- **TEXT**: Categorical fields or text data

### Core Functions

#### Hyperparameters Extraction

- **extract_hyperparameters_from_tarball()**: Extracts hyperparameters.json from model artifacts with robust fallback mechanisms:
  1. Looks for hyperparameters.json within model.tar.gz archive
  2. Checks if model.tar.gz is actually a directory containing hyperparameters.json
  3. Searches the input model directory and subdirectories for hyperparameters.json

#### Field Processing

- **create_model_variable_list(full_field_list, tab_field_list, cat_field_list, label_name, id_name)**: Creates a list of [variable_name, variable_type] pairs, classifying each field as either NUMERIC or TEXT based on the provided lists.

#### Payload Generation

- **generate_csv_payload(input_vars, default_numeric_value, default_text_value, special_field_values)**: Creates a CSV format payload following the order in input_vars.

- **generate_json_payload(input_vars, default_numeric_value, default_text_value, special_field_values)**: Creates a JSON format payload using field names from input_vars.

- **generate_sample_payloads(input_vars, content_types, default_numeric_value, default_text_value, special_field_values)**: Generates sample payloads for each requested content type.

- **save_payloads(output_dir, input_vars, content_types, default_numeric_value, default_text_value, special_field_values)**: Saves generated payloads to files in the specified directory.

#### Archive Creation

- **create_payload_archive(payload_files)**: Creates a compressed tar.gz archive containing all payload samples.

### Special Field Handling

The script supports special field handling through environment variables prefixed with `SPECIAL_FIELD_`:

- Templates can include placeholders like `{timestamp}` which get replaced with dynamic values
- Example: Setting `SPECIAL_FIELD_TRANSACTION_DATE="{timestamp}"` would replace that field's value with the current date/time

## Workflow

1. **Extract hyperparameters** from the model artifact
2. **Extract field information** from hyperparameters:
   - full_field_list: All fields
   - tab_field_list: Numeric/tabular fields
   - cat_field_list: Categorical fields
   - label_name: Target/label field name
   - id_name: Identifier field name
3. **Create variable type list** by classifying each field as NUMERIC or TEXT
4. **Read configuration** from environment variables
5. **Generate sample payloads** for each configured content type
6. **Save payloads** to sample directory
7. **Create compressed archive** of all payload samples

## Input and Output Structure

### Input Sources

1. **Model Artifact**:
   - Path: `/opt/ml/processing/input/model/model.tar.gz` or `/opt/ml/processing/input/model/`
   - Required File: `hyperparameters.json` (within archive or directory)
   - Contains: Field lists, label information, model metadata

2. **Environment Variables**:
   - Configure content types, default values, and special field handling

### Output Files

1. **Individual Payload Samples**:
   - Path: `/tmp/mims_payload_work/payload_sample/`
   - Format: JSON and/or CSV files according to configuration
   - Purpose: Sample inference requests for each supported format

2. **Payload Archive**:
   - Path: `/opt/ml/processing/output/payload.tar.gz`
   - Contents: All generated payload samples
   - Purpose: Deployment artifact for model registration

## Payload Formats

### JSON Format

```json
{
  "field_name1": "value1",
  "field_name2": 0.0,
  "field_name3": "DEFAULT_TEXT"
}
```

### CSV Format

```
value1,0.0,DEFAULT_TEXT
```

## Example Use Cases

### 1. Standard ML Model Payload Generation

For a typical machine learning model with numeric and categorical features:

```
# Environment setup
export CONTENT_TYPES="application/json,text/csv"
export DEFAULT_NUMERIC_VALUE="0.0"
export DEFAULT_TEXT_VALUE="N/A"

# Run the script
python mims_payload.py
```

### 2. Custom Default Values with Timestamp

For a model requiring special handling of date/time fields:

```
# Environment setup
export CONTENT_TYPES="application/json"
export DEFAULT_NUMERIC_VALUE="0.0"
export DEFAULT_TEXT_VALUE="UNKNOWN"
export SPECIAL_FIELD_TRANSACTION_DATE="{timestamp}"

# Run the script
python mims_payload.py
```

## Best Practices

1. **Field Classification**: Ensure field lists in hyperparameters.json accurately categorize numeric and text fields

2. **Default Values**: Set appropriate DEFAULT_NUMERIC_VALUE and DEFAULT_TEXT_VALUE for your model

3. **Content Types**: Only include content types your inference endpoint will support

4. **Special Fields**: Use SPECIAL_FIELD_* environment variables for fields requiring dynamic or specific values

5. **Validation**: Review the generated payloads to ensure they match your model's expected format

## Error Handling

The script includes comprehensive error handling for:

1. Missing hyperparameters.json (with diagnostic information about search locations)
2. Invalid environment variable values (with fallbacks to defaults)
3. Template format errors in special field values
4. Archive creation issues

## Integration in the Pipeline

This script is typically used in conjunction with the MIMS packaging step:

1. **Dummy Training** or **Model Training**: Produces model artifacts with hyperparameters.json
2. **MIMS Packaging**: Packages model and inference code into model.tar.gz
3. **MIMS Payload Generation**: Creates payload examples for the packaged model
4. **Model Registration**: Uses both the model package and payload examples for deployment

# Risk Table Mapping Module Documentation

## Overview

The `risk_table_mapping.py` script provides a specialized approach to categorical feature encoding through risk tables. It transforms categorical variables into numerical risk scores based on their correlation with the target variable. This approach is particularly valuable for high-cardinality categorical features in risk modeling, as it captures the relationship between categorical values and the target outcome in a statistically meaningful way.

## Key Features

- Maps categorical values to numerical risk scores based on target correlation
- Supports both training (fit + transform) and inference (transform-only) modes
- Handles low-frequency categories through smoothing
- Applies consistent transformations across training, validation, and testing datasets
- Maintains persistence of risk tables for downstream pipeline steps
- Enforces consistent file naming conventions for artifact management

## Core Components

### OfflineBinning Class

The central component of the script is the `OfflineBinning` class, which:

- **Fits risk tables**: Calculates risk scores for each category based on target correlation
- **Applies smoothing**: Handles low-frequency categories with Bayesian smoothing
- **Transforms data**: Converts categorical values to their corresponding risk scores
- **Handles missing values**: Assigns default risk scores to unseen categories

### Data Processing Flow

#### Training Mode

1. **Validation**: Validates that categorical fields are suitable for risk mapping
2. **Fitting**: Fits risk tables on training data only
3. **Transformation**: Applies the fitted risk tables to all data splits
4. **Artifact Saving**: Saves risk tables for downstream use

#### Inference Mode (validation/testing/calibration)

1. **Loading**: Loads pre-trained risk tables
2. **Transformation**: Applies loaded risk tables to transform categorical features
3. **Artifact Passing**: Maintains consistent risk table artifacts for downstream steps

## Risk Table Calculation

Risk tables are calculated using the following approach:

1. Create a cross-tabulation of the categorical feature and target variable
2. Calculate the raw risk (proportion of positive targets) for each category
3. Apply Bayesian smoothing to handle low-frequency categories:
   ```
   smoothed_risk = (count * raw_risk + smooth_samples * default_risk) / (count + smooth_samples)
   ```
4. Use default risk for categories below the count threshold

## Input and Output

### Input Data

- **Training Mode**:
  - Train/test/val splits in `/opt/ml/processing/input/data/{split}/{split}_processed_data.csv`
  - Hyperparameters in `/opt/ml/processing/input/config/hyperparameters.json`

- **Inference Mode**:
  - Single data split in `/opt/ml/processing/input/data/{job_type}/{job_type}_processed_data.csv`
  - Pre-trained risk tables in `/opt/ml/processing/input/risk_tables/bin_mapping.pkl`
  - Hyperparameters in `/opt/ml/processing/input/config/hyperparameters.json`

### Output Data

- **Transformed Data**:
  - Processed data with risk-mapped features in `/opt/ml/processing/output/{split}/{split}_processed_data.csv`

- **Artifacts**:
  - Risk tables in `/opt/ml/processing/output/bin_mapping.pkl`
  - Copy of hyperparameters in `/opt/ml/processing/output/hyperparameters.json`

## Configuration

The script is configured through hyperparameters, which can include:

- **cat_field_list**: List of categorical fields to apply risk mapping
- **label_name**: Name of the target/label column (default: "target")
- **smooth_factor**: Smoothing factor for risk table calculation (default: 0.01)
- **count_threshold**: Minimum count threshold for category risk calculation (default: 5)

## Usage Examples

### Training Mode

```bash
python risk_table_mapping.py --job_type training
```

In training mode, the script:
1. Loads training, validation, and testing data
2. Fits risk tables on training data
3. Transforms all three data splits
4. Saves the risk tables for future use

### Inference Mode

```bash
python risk_table_mapping.py --job_type validation
```

In inference mode, the script:
1. Loads the data split specified by job_type
2. Loads pre-trained risk tables
3. Transforms the data using the loaded risk tables
4. Saves the transformed data and a copy of the risk tables

## Best Practices

1. **Field Selection**: Include only relevant categorical fields that have meaningful correlation with the target

2. **Smoothing Factor**: Adjust based on dataset size - smaller datasets may benefit from higher smoothing

3. **Count Threshold**: Set according to domain knowledge about minimum reliable sample size for a category

4. **Pipeline Integration**: 
   - Ensure this step runs after initial data loading and preprocessing
   - Place before model training and feature selection steps

## Error Handling

The script implements robust error handling for:

- Missing configuration files
- Invalid JSON format in hyperparameters
- Missing risk tables in non-training modes
- File permission issues
- Runtime errors with detailed logging

## Implementation Details

### Risk Table Structure

Each risk table contains:
- **varName**: Name of the categorical variable
- **type**: Always "categorical" for this implementation
- **mode**: "numeric" or "categorical" based on the original data type
- **default_bin**: Default risk score for unseen categories
- **bins**: Dictionary mapping category values to risk scores

### Workflow Flexibility

The implementation supports:
1. **Dependency Injection**: Custom data loading/saving functions can be provided for testing
2. **Artifact Consistency**: Enforced naming conventions ensure pipeline compatibility
3. **Graceful Fallbacks**: Handles missing configuration with sensible defaults
4. **Detailed Logging**: Comprehensive logging for debugging and monitoring

## Integration in the Pipeline

This script typically:
1. Follows initial data loading and basic preprocessing
2. Precedes feature selection and model training
3. Provides consistent risk mapping across training and inference
4. Supports both model development and production deployment workflows

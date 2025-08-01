# XGBoost Model Evaluation Module Documentation

## Overview

The `model_evaluation_xgb.py` script performs comprehensive evaluation of XGBoost models by loading trained model artifacts, preprocessing evaluation data with consistent transformations, computing performance metrics, and generating visual artifacts. It supports both binary and multi-class classification scenarios and enforces consistent evaluation practices through the contract enforcement system.

## Key Features

- Loads trained XGBoost models and associated preprocessing artifacts
- Applies consistent preprocessing to evaluation data
- Computes comprehensive metrics for binary and multi-class classification
- Generates and saves ROC and Precision-Recall curves
- Provides detailed metric logs and summaries
- Uses contract enforcement to validate paths and environment variables

## Core Components

### Model and Artifact Loading

- **load_model_artifacts(model_dir)**: Loads the trained XGBoost model and associated artifacts including:
  - The trained XGBoost model (.bst format)
  - Risk table mappings for categorical features
  - Numerical imputation dictionary
  - Feature column names and order
  - Model hyperparameters including classification type

### Data Preprocessing

- **load_eval_data(eval_data_dir)**: Loads evaluation data from CSV or Parquet files
- **preprocess_eval_data(df, feature_columns, risk_tables, impute_dict)**: Applies consistent preprocessing to evaluation data:
  - Risk table mapping for categorical features
  - Numerical imputation for missing values
  - Column reordering to match model expectations
  - Type conversion and handling of missing values

### Metric Computation

- **compute_metrics_binary(y_true, y_prob)**: Calculates binary classification metrics:
  - AUC-ROC
  - Average precision
  - F1 score (overall and at multiple thresholds)
  - Precision and recall at various operating points

- **compute_metrics_multiclass(y_true, y_prob, n_classes)**: Calculates multi-class classification metrics:
  - Per-class AUC-ROC, average precision, and F1 scores
  - Micro and macro averaged metrics
  - Class distribution statistics

### Visualization

- **plot_and_save_roc_curve(y_true, y_score, output_dir, prefix)**: Generates and saves ROC curve plots
- **plot_and_save_pr_curve(y_true, y_score, output_dir, prefix)**: Generates and saves Precision-Recall curve plots

### Output Generation

- **save_predictions(ids, y_true, y_prob, id_col, label_col, output_eval_dir)**: Saves predictions with original IDs, true labels, and class probabilities
- **save_metrics(metrics, output_metrics_dir)**: Saves metrics as JSON and human-readable summary text
- **log_metrics_summary(metrics, is_binary)**: Creates formatted logs of key metrics

## Workflow

1. **Load Model and Artifacts**: Load the trained XGBoost model and associated preprocessing artifacts
2. **Load Evaluation Data**: Load the evaluation dataset (CSV or Parquet)
3. **Preprocess Data**: Apply risk table mapping and numerical imputation
4. **Generate Predictions**: Run the model on preprocessed data
5. **Compute Metrics**: Calculate appropriate metrics based on classification type
6. **Generate Visualizations**: Create ROC and Precision-Recall curves
7. **Save Results**: Output predictions, metrics, and visualizations

## Input and Output Structure

### Input Sources

1. **Model Directory**:
   - Path: Contract-specified 'model_input' path
   - Contents: XGBoost model and preprocessing artifacts
   - Required Files:
     - xgboost_model.bst
     - risk_table_map.pkl
     - impute_dict.pkl
     - feature_columns.txt
     - hyperparameters.json

2. **Evaluation Data Directory**:
   - Path: Contract-specified 'eval_data_input' path
   - Contents: Dataset for evaluation (CSV or Parquet)

### Output Files

1. **Predictions**:
   - Path: Contract-specified 'eval_output' path
   - File: eval_predictions.csv
   - Contents: Original IDs, true labels, and class probabilities

2. **Metrics**:
   - Path: Contract-specified 'metrics_output' path
   - Files:
     - metrics.json: Detailed metrics in JSON format
     - metrics_summary.txt: Human-readable metrics summary
     - roc_curve.jpg: ROC curve visualization
     - pr_curve.jpg: Precision-Recall curve visualization
     - For multi-class: class_N_roc_curve.jpg and class_N_pr_curve.jpg for each class

## Environment Variables

The script relies on these environment variables (validated through contract):
- **ID_FIELD**: Column name for the ID field in evaluation data
- **LABEL_FIELD**: Column name for the target/label field in evaluation data

## Error Handling

The script implements error handling for:
- Missing model artifacts
- Missing or invalid evaluation data
- Inconsistent feature sets
- Classification type mismatches

## Integration in the Pipeline

This script is typically used:
1. After a model has been trained
2. For both validation and testing datasets
3. As part of model performance assessment
4. To generate artifacts for model registration

## Best Practices Demonstrated

1. **Contract Enforcement**: Uses ContractEnforcer to validate environment and paths
2. **Consistent Preprocessing**: Applies the same preprocessing as training
3. **Comprehensive Metrics**: Reports multiple metrics for thorough evaluation
4. **Visualization**: Generates visual artifacts for easier interpretation
5. **Detailed Logging**: Creates well-formatted logs with searchable metric keys
6. **Human-Readable Output**: Provides summaries in addition to machine-readable files
7. **Cross-Format Support**: Works with both CSV and Parquet data sources

## Example Usage

```python
# Example environment setup
export ID_FIELD="user_id"
export LABEL_FIELD="fraud_flag"

# Run evaluation script
python -m src.pipeline_scripts.model_evaluation_xgb --job_type validation
```

## Key Metrics

### Binary Classification
- AUC-ROC: Area under the ROC curve
- Average Precision: Area under the precision-recall curve
- F1 Score: Harmonic mean of precision and recall
- Precision and Recall at various thresholds

### Multi-class Classification
- Per-class metrics (AUC-ROC, AP, F1)
- Macro-averaged metrics (average across all classes)
- Micro-averaged metrics (aggregate across all classes)
- Class distribution statistics

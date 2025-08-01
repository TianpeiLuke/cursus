# Model Evaluation Script Contract

## Overview
The Model Evaluation Script Contract defines the execution requirements for the XGBoost model evaluation script (`model_eval_xgb.py`). This contract ensures the script complies with SageMaker processing job conventions for loading trained models, processing evaluation data, and generating performance metrics and visualizations.

## Contract Details

### Script Information
- **Entry Point**: `model_eval_xgb.py`
- **Container Type**: SageMaker Processing Job
- **Framework**: XGBoost with scikit-learn preprocessing
- **Purpose**: Model evaluation and performance analysis

### Input Path Requirements

| Logical Name | Expected Path | Description |
|--------------|---------------|-------------|
| model_input | `/opt/ml/processing/input/model` | Trained model artifacts and preprocessing components |
| processed_data | `/opt/ml/processing/input/eval_data` | Evaluation data (CSV or Parquet files) |
| code | `/opt/ml/processing/input/code` | Source code for processing modules |

### Output Path Requirements

| Logical Name | Expected Path | Description |
|--------------|---------------|-------------|
| eval_output | `/opt/ml/processing/output/eval` | Model predictions with probabilities |
| metrics_output | `/opt/ml/processing/output/metrics` | Performance metrics and visualizations |

### Environment Variables

#### Required Variables
- `ID_FIELD` - Name of the ID column in evaluation data
- `LABEL_FIELD` - Name of the label column in evaluation data

#### Optional Variables
- None specified in current implementation

### Framework Requirements

#### Core Dependencies
```python
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.6.0
matplotlib>=3.5.0
```

## Script Functionality

Based on the actual implementation in `dockers/xgboost_atoz/model_eval_xgb.py`:

### Environment Validation
1. **Directory Validation** - Validates all required input/output directories exist
2. **Processing Package Validation** - Ensures processing modules are available
3. **Path Setup** - Adds source code directory to Python path
4. **Logging Configuration** - Sets up comprehensive logging for debugging

### Model Artifact Loading
1. **Model Decompression** - Automatically extracts model.tar.gz if present
2. **XGBoost Model Loading** - Loads trained XGBoost model from .bst file
3. **Preprocessing Artifacts** - Loads risk tables, imputation dictionary, and feature columns
4. **Hyperparameter Loading** - Loads model configuration and hyperparameters
5. **Feature Order Preservation** - Maintains exact feature ordering from training

### Data Preprocessing Pipeline
1. **Risk Table Application**:
   - Applies pre-fitted risk tables to categorical features
   - Handles unseen categories gracefully
   - Transforms categorical values to risk scores

2. **Numerical Imputation**:
   - Applies pre-fitted imputation values to numerical features
   - Uses same imputation strategy as training
   - Handles missing values consistently

3. **Data Validation**:
   - Ensures all features are numeric
   - Reorders columns to match training feature order
   - Handles missing features gracefully

### Model Evaluation Features
1. **Binary Classification**:
   - Computes AUC-ROC, Average Precision, F1-Score
   - Generates ROC and Precision-Recall curves
   - Handles probability calibration

2. **Multiclass Classification**:
   - Computes per-class metrics (one-vs-rest)
   - Calculates micro and macro averaged metrics
   - Generates curves for each class separately

3. **Visualization Generation**:
   - ROC curves saved as JPG files
   - Precision-Recall curves saved as JPG files
   - Separate plots for each class in multiclass problems

### Key Implementation Details

#### Model Artifact Loading
```python
def load_model_artifacts(model_dir):
    """Load XGBoost model and preprocessing artifacts"""
    # Decompress model.tar.gz if present
    decompress_model_artifacts(model_dir)
    
    # Load model and artifacts
    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, "xgboost_model.bst"))
    
    # Load preprocessing components
    with open(os.path.join(model_dir, "risk_table_map.pkl"), "rb") as f:
        risk_tables = pkl.load(f)
    with open(os.path.join(model_dir, "feature_columns.txt"), "r") as f:
        feature_columns = [line.strip().split(",")[1] for line in f if not line.startswith("#")]
```

#### Data Preprocessing
```python
def preprocess_eval_data(df, feature_columns, risk_tables, impute_dict):
    """Apply same preprocessing as training"""
    # Apply risk table mapping
    for feature, risk_table in risk_tables.items():
        if feature in df.columns:
            proc = RiskTableMappingProcessor(
                column_name=feature,
                label_name="label",
                risk_tables=risk_table
            )
            df[feature] = proc.transform(df[feature])
    
    # Apply numerical imputation
    imputer = NumericalVariableImputationProcessor(imputation_dict=impute_dict)
    df = imputer.transform(df)
```

#### Metrics Computation
```python
def compute_metrics_binary(y_true, y_prob):
    """Compute binary classification metrics"""
    y_score = y_prob[:, 1]
    metrics = {
        "auc_roc": roc_auc_score(y_true, y_score),
        "average_precision": average_precision_score(y_true, y_score),
        "f1_score": f1_score(y_true, y_score > 0.5)
    }
    return metrics
```

### Output Artifacts
- **eval_predictions.csv** - Model predictions with ID, true labels, and class probabilities
- **metrics.json** - Comprehensive performance metrics
- **roc_curve.jpg** - ROC curve visualization (binary) or per-class curves (multiclass)
- **pr_curve.jpg** - Precision-Recall curve visualization
- **class_{i}_roc_curve.jpg** - Per-class ROC curves for multiclass problems
- **class_{i}_pr_curve.jpg** - Per-class PR curves for multiclass problems

### Supported Input Formats
- **CSV Files** - Standard comma-separated values
- **Parquet Files** - Columnar storage format
- **Compressed Archives** - Automatic extraction of model.tar.gz

## Configuration Examples

### Environment Variables Setup
```bash
export ID_FIELD="customer_id"
export LABEL_FIELD="target"
```

### Command Line Usage
```bash
python model_eval_xgb.py --job_type evaluation
```

## Usage Example

### Contract Access
```python
from src.pipeline_script_contracts import MODEL_EVALUATION_CONTRACT

# Access contract details
print(f"Entry Point: {MODEL_EVALUATION_CONTRACT.entry_point}")
print(f"Required Env Vars: {MODEL_EVALUATION_CONTRACT.required_env_vars}")
```

### Integration with Step Builder
```python
from src.pipeline_steps import ModelEvaluationStepBuilder

class ModelEvaluationStepBuilder(StepBuilderBase):
    def validate_configuration(self) -> None:
        validation = MODEL_EVALUATION_CONTRACT.validate_implementation(
            'dockers/xgboost_atoz/model_eval_xgb.py'
        )
        if not validation.is_valid:
            self.logger.warning(f"Script validation warnings: {validation.errors}")
```

## Integration Points

### Pipeline Position
- **Evaluation Node** - Evaluates trained models on held-out data
- **Input Dependencies** - Requires trained model and evaluation data
- **Output Consumers** - Provides metrics for model selection and reporting

### Data Flow
```
Trained Model + Evaluation Data → model_eval_xgb.py → Metrics + Predictions + Visualizations
```

## Best Practices

### Script Development
1. **Consistent Preprocessing** - Apply exact same preprocessing as training
2. **Feature Order** - Maintain consistent feature ordering
3. **Error Handling** - Handle missing features and data gracefully
4. **Comprehensive Metrics** - Compute multiple evaluation metrics

### Model Evaluation
1. **Data Quality** - Validate evaluation data quality and format
2. **Metric Selection** - Choose appropriate metrics for the problem type
3. **Visualization** - Generate clear and informative plots
4. **Documentation** - Document evaluation methodology and results

## Related Contracts

### Upstream Contracts
- `XGBOOST_TRAIN_CONTRACT` - Provides trained model artifacts
- `TABULAR_PREPROCESS_CONTRACT` - Provides evaluation data format

### Downstream Contracts
- `MODEL_DEPLOYMENT_CONTRACT` - May use evaluation results for deployment decisions
- `REPORTING_CONTRACT` - May use metrics for reporting and monitoring

## Troubleshooting

### Common Issues
1. **Missing Artifacts** - Ensure all model artifacts are present
2. **Feature Mismatch** - Verify evaluation data has same features as training
3. **Preprocessing Errors** - Check risk table and imputation compatibility
4. **Memory Issues** - Monitor memory usage with large evaluation datasets

### Validation Failures
1. **Path Validation** - Ensure all input/output paths exist and are accessible
2. **Environment Variables** - Check ID_FIELD and LABEL_FIELD are set correctly
3. **Data Format** - Validate evaluation data format matches expectations
4. **Model Compatibility** - Ensure model artifacts are compatible with script version

## Performance Considerations

### Optimization Strategies
- **Batch Processing** - Process evaluation data in batches for large datasets
- **Memory Management** - Efficient loading and processing of model artifacts
- **Parallel Evaluation** - Leverage XGBoost's built-in parallelization
- **Caching** - Cache preprocessing results for repeated evaluations

### Monitoring Metrics
- Evaluation processing time
- Memory usage during evaluation
- Model prediction accuracy
- Preprocessing pipeline performance

## Security Considerations

### Data Protection
- Secure handling of evaluation data
- No sensitive data in logs or outputs
- Proper cleanup of temporary files

### Model Security
- Validate model artifact integrity
- Secure loading of preprocessing components
- Protection against model tampering

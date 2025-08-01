# XGBoost Training Script Contract

## Overview
The XGBoost Training Script Contract defines the execution requirements for the XGBoost training script (`train_xgb.py`). This contract ensures the script complies with SageMaker training job conventions for tabular data classification with risk table mapping and numerical imputation.

## Contract Details

### Script Information
- **Entry Point**: `train_xgb.py`
- **Container Type**: SageMaker Training Job
- **Framework**: XGBoost with scikit-learn preprocessing
- **Purpose**: Tabular data classification training with preprocessing

### Input Path Requirements

| Logical Name | Expected Path | Description |
|--------------|---------------|-------------|
| input_path | `/opt/ml/input/data` | Root directory containing train/val/test subdirectories |
| hyperparameters_s3_uri | `/opt/ml/input/data/config/hyperparameters.json` | Model configuration and hyperparameters |

### Output Path Requirements

| Logical Name | Expected Path | Description |
|--------------|---------------|-------------|
| model_output | `/opt/ml/model` | Trained model artifacts and preprocessing components |
| evaluation_output | `/opt/ml/output/data` | Training outputs, metrics, and evaluation results |

### Environment Variables

#### Required Variables
- None (script uses hyperparameters.json for configuration)

#### Optional Variables
- None specified in current implementation

### Framework Requirements

#### Core Dependencies
```python
boto3>=1.26.0
xgboost==1.7.6
scikit-learn>=0.23.2,<1.0.0
pandas>=1.2.0,<2.0.0
pyarrow>=4.0.0,<6.0.0
beautifulsoup4>=4.9.3
flask>=2.0.0,<3.0.0
pydantic>=2.0.0,<3.0.0
typing-extensions>=4.2.0
matplotlib>=3.0.0
numpy>=1.19.0
```

## Script Functionality

Based on the actual implementation in `dockers/xgboost_atoz/train_xgb.py`:

### Data Loading and Validation
1. **Multi-Format Support** - Loads CSV, Parquet, and JSON files from train/val/test directories
2. **Configuration Validation** - Validates hyperparameters using Pydantic models
3. **Data Structure Validation** - Ensures required fields exist in datasets
4. **Class Weight Validation** - Validates class weights match number of classes

### Preprocessing Pipeline
1. **Numerical Imputation**:
   - Uses mean imputation strategy for missing numerical values
   - Fits imputation on training data only
   - Applies same imputation to validation and test sets
   - Saves imputation dictionary for inference

2. **Risk Table Processing**:
   - Fits risk tables on categorical features using target correlation
   - Applies smoothing and count thresholds for robust estimation
   - Transforms categorical values to risk scores
   - Handles unseen categories during inference

### Model Training Features
1. **Binary Classification**:
   - Uses `binary:logistic` objective
   - Supports `scale_pos_weight` for class imbalance
   - Generates ROC and PR curves
   - Computes AUC-ROC, Average Precision, F1-Score

2. **Multiclass Classification**:
   - Uses `multi:softprob` objective
   - Supports sample weights for class imbalance
   - Generates per-class and aggregate metrics
   - Computes micro/macro averaged metrics

3. **Training Configuration**:
   - Early stopping with configurable patience
   - Comprehensive hyperparameter support
   - Feature importance tracking
   - Model checkpointing

### Key Implementation Details

#### Data Loading Logic
```python
def load_datasets(input_path: str) -> tuple:
    """Loads training, validation, and test datasets"""
    train_file = find_first_data_file(os.path.join(input_path, "train"))
    val_file = find_first_data_file(os.path.join(input_path, "val"))
    test_file = find_first_data_file(os.path.join(input_path, "test"))
    # Supports .parquet, .csv, .json files
```

#### Risk Table Processing
```python
def fit_and_apply_risk_tables(config, train_df, val_df, test_df):
    """Fits risk tables on training data and applies to all splits"""
    for var in config['cat_field_list']:
        proc = RiskTableMappingProcessor(
            column_name=var,
            label_name=config['label_name'],
            smooth_factor=config.get('smooth_factor', 0.0),
            count_threshold=config.get('count_threshold', 0),
        )
```

#### Model Training with Class Weights
```python
def train_model(config, dtrain, dval):
    """Trains XGBoost model with proper class weight handling"""
    if config.get("is_binary", True):
        xgb_params["objective"] = "binary:logistic"
        if "class_weights" in config:
            xgb_params["scale_pos_weight"] = config["class_weights"][1] / config["class_weights"][0]
    else:
        xgb_params["objective"] = "multi:softprob"
        xgb_params["num_class"] = config["num_classes"]
```

### Output Artifacts
- **xgboost_model.bst** - Trained XGBoost model
- **risk_table_map.pkl** - Risk table mappings for categorical features
- **impute_dict.pkl** - Imputation values for numerical features
- **feature_importance.json** - Feature importance scores
- **feature_columns.txt** - Ordered feature column names with indices
- **hyperparameters.json** - Model hyperparameters and configuration
- **val.tar.gz** - Validation predictions, metrics, and plots
- **test.tar.gz** - Test predictions, metrics, and plots

### Evaluation and Metrics
1. **Performance Metrics**:
   - AUC-ROC, Average Precision, F1-Score
   - Per-class metrics for multiclass problems
   - Micro and macro averaged metrics

2. **Visualization**:
   - ROC curves for each class
   - Precision-Recall curves
   - Performance plots saved as JPG files

3. **Output Packaging**:
   - Predictions and metrics packaged in tar.gz files
   - Separate directories for validation and test results

## Hyperparameter Configuration

### Required Parameters
```json
{
  "tab_field_list": ["feature1", "feature2", "feature3"],
  "cat_field_list": ["category1", "category2"],
  "label_name": "target",
  "is_binary": true,
  "num_classes": 2
}
```

### XGBoost Parameters
```json
{
  "eta": 0.1,
  "gamma": 0,
  "max_depth": 6,
  "subsample": 1,
  "colsample_bytree": 1,
  "lambda_xgb": 1,
  "alpha_xgb": 0,
  "num_round": 100,
  "early_stopping_rounds": 10
}
```

### Risk Table Parameters
```json
{
  "smooth_factor": 0.0,
  "count_threshold": 0
}
```

### Class Weight Configuration
```json
{
  "class_weights": [1.0, 2.0]  // Must match num_classes
}
```

## Usage Example

### Contract Access
```python
from src.pipeline_script_contracts import XGBOOST_TRAIN_CONTRACT

# Access contract details
print(f"Entry Point: {XGBOOST_TRAIN_CONTRACT.entry_point}")
print(f"Framework Requirements: {XGBOOST_TRAIN_CONTRACT.framework_requirements}")
```

### Integration with Step Builder
```python
from src.pipeline_steps import XGBoostTrainingStepBuilder

class XGBoostTrainingStepBuilder(StepBuilderBase):
    def validate_configuration(self) -> None:
        validation = XGBOOST_TRAIN_CONTRACT.validate_implementation(
            'dockers/xgboost_atoz/train_xgb.py'
        )
        if not validation.is_valid:
            self.logger.warning(f"Script validation warnings: {validation.errors}")
```

## Integration Points

### Pipeline Position
- **Training Node** - Trains models on preprocessed data
- **Input Dependencies** - Requires preprocessed data from tabular preprocessing
- **Output Consumers** - Provides trained models for evaluation and inference

### Data Flow
```
Preprocessed Data → train_xgb.py → Trained Model + Evaluation Results
                                ↓
                         Model Artifacts + Metrics
```

## Best Practices

### Script Development
1. **Robust Preprocessing** - Handle missing values and categorical encoding properly
2. **Feature Ordering** - Maintain consistent feature ordering for inference
3. **Model Validation** - Validate model performance on held-out data
4. **Artifact Management** - Save all necessary artifacts for inference

### Configuration Management
1. **Parameter Validation** - Use Pydantic for configuration validation
2. **Class Balance** - Handle imbalanced datasets with appropriate weights
3. **Feature Engineering** - Apply consistent preprocessing across splits
4. **Hyperparameter Tuning** - Use systematic approach for parameter optimization

## Related Contracts

### Upstream Contracts
- `TABULAR_PREPROCESS_CONTRACT` - Provides preprocessed training data
- `HYPERPARAMETER_PREP_CONTRACT` - May provide hyperparameter configuration

### Downstream Contracts
- `MODEL_EVALUATION_CONTRACT` - Uses trained model for evaluation
- `XGBOOST_INFERENCE_CONTRACT` - Uses model artifacts for inference

## Troubleshooting

### Common Issues
1. **Data Format Errors** - Ensure consistent data formats across splits
2. **Feature Mismatch** - Verify feature columns match between train/val/test
3. **Class Weight Errors** - Ensure class weights match number of classes
4. **Memory Issues** - Monitor memory usage with large datasets

### Validation Failures
1. **Missing Features** - Check that all required features are present
2. **Label Encoding** - Ensure labels are properly encoded as integers
3. **Risk Table Fitting** - Verify categorical features have sufficient data
4. **Model Convergence** - Check for training convergence issues

## Performance Considerations

### Optimization Strategies
- **Early Stopping** - Prevent overfitting with early stopping
- **Feature Selection** - Use feature importance for feature selection
- **Memory Management** - Efficient data loading and processing
- **Parallel Training** - Leverage XGBoost's built-in parallelization

### Monitoring Metrics
- Training and validation loss curves
- Feature importance scores
- Model convergence indicators
- Resource utilization metrics

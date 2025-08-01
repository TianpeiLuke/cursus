# Risk Table Mapping Script Contract

## Overview
The Risk Table Mapping Script Contract defines the execution requirements for the risk table mapping script (`risk_table_mapping.py`). This contract ensures the script complies with SageMaker processing job conventions for creating risk tables for categorical features and performing missing value imputation for numerical features.

## Contract Details

### Script Information
- **Entry Point**: `risk_table_mapping.py`
- **Container Type**: SageMaker Processing Job
- **Framework**: Python with pandas, scikit-learn
- **Purpose**: Categorical feature encoding via risk tables and numerical imputation

### Input Path Requirements

| Logical Name | Expected Path | Description |
|--------------|---------------|-------------|
| DATA | `/opt/ml/processing/input/data` | Input data (single file for training, splits for inference) |
| CONFIG | `/opt/ml/processing/input/config` | Configuration files (config.json, metadata.csv) |

### Output Path Requirements

| Logical Name | Expected Path | Description |
|--------------|---------------|-------------|
| processed_data | `/opt/ml/processing/output` | Processed data with risk-encoded features |

### Environment Variables

#### Required Variables
- `TRAIN_RATIO` - Training data ratio (default: 0.7)
- `TEST_VAL_RATIO` - Test/validation split ratio (default: 0.5)

#### Optional Variables
- None specified in current implementation

### Framework Requirements

#### Core Dependencies
```python
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
```

## Script Functionality

Based on the actual implementation in `dockers/xgboost_atoz/pipeline_scripts/risk_table_mapping.py`:

### Risk Table Creation (OfflineBinning)
1. **Categorical Feature Processing**:
   - Creates risk tables mapping categorical values to target correlation scores
   - Uses cross-tabulation to calculate risk scores for each category
   - Applies smoothing and count thresholds for robust estimation
   - Handles missing values with default risk scores

2. **Risk Score Calculation**:
   - Computes risk as positive class rate for each categorical value
   - Applies Laplace smoothing with configurable smoothing factor
   - Uses count thresholds to filter low-frequency categories
   - Provides default risk score for unseen categories

### Missing Value Imputation (MissingValueImputation)
1. **Numerical Feature Imputation**:
   - Uses scikit-learn SimpleImputer for numerical features
   - Supports multiple imputation strategies (mean, median, most_frequent)
   - Fits imputation on training data only
   - Applies same imputation to all data splits

2. **Strategy Configuration**:
   - Reads imputation strategy from metadata configuration
   - Handles different strategies per numerical variable
   - Skips imputation for variables with "none" strategy

### Processing Modes
1. **Training Mode** (`job_type="training"`):
   - Loads single unsplit dataset from `data.csv`
   - Fits risk tables and imputers on entire dataset
   - Transforms data then splits into train/test/val
   - Uses stratified sampling to maintain label distribution

2. **Inference Modes** (`job_type="validation"` or `job_type="testing"`):
   - Loads pre-split data from train/validation/test directories
   - Fits risk tables and imputers on training data only
   - Transforms all splits using fitted transformers
   - Maintains existing split boundaries

### Key Implementation Details

#### Risk Table Fitting
```python
def fit(self, df: pd.DataFrame, smooth_factor: float = 0, count_threshold: int = 0):
    """Fits risk tables based on the provided dataframe"""
    fit_df = df.loc[(df[self.target] != -1) & (~df[self.target].isnull())].copy()
    default_risk = float(fit_df[self.target].mean())
    smooth_samples = int(len(fit_df) * smooth_factor)
    
    for var in self.variables:
        risk_table = self._create_risk_table(fit_df, var, default_risk, smooth_samples, count_threshold)
        self.risk_tables[var]["bins"] = risk_table
```

#### Risk Score Calculation
```python
def _create_risk_table(self, df, variable, default_risk, samples, count_threshold):
    """Helper to calculate the risk table for a single variable"""
    cross_tab = pd.crosstab(df[variable], df[self.target].astype(object), margins=True, margins_name="_count_", dropna=False)
    cross_tab["risk"] = cross_tab.apply(lambda x: x.get(1, 0.0) / (x.get(1, 0) + x.get(0, 0)), axis=1)
    cross_tab["smooth_risk"] = cross_tab.apply(
        lambda x: ((x["_count_"] * x["risk"] + samples * default_risk) / (x["_count_"] + samples))
        if x["_count_"] >= count_threshold else default_risk, axis=1
    )
```

#### Numerical Imputation
```python
def fit(self, df: pd.DataFrame):
    """Fits imputers for numeric variables based on the provided dataframe"""
    for var in self.numeric_variables:
        if var in df.columns:
            impute_strategy = self.metadata.loc[self.metadata["varname"] == var, "impute_strategy"].iat[0]
            if impute_strategy and impute_strategy != "none":
                imputer = SimpleImputer(missing_values=np.nan, strategy=impute_strategy)
                imputer.fit(df[var].values.reshape(-1, 1))
                self.imputers[var] = imputer
```

### Output Artifacts
- **bin_mapping.pkl** - Risk table mappings for categorical features
- **missing_value_imputation.pkl** - Imputation values for numerical features
- **config.pkl** - Configuration and metadata used for processing
- **{split}_processed_data.csv** - Processed data files for each split

### Configuration Structure

#### Config.json Format
```json
{
  "tag": "target_column_name",
  "metadata": "loaded_from_metadata.csv",
  "model_training_config": {
    "category_risk_params": {
      "smooth_factor": 0.1,
      "count_threshold": 10
    }
  }
}
```

#### Metadata.csv Format
```csv
varname,datatype,iscategory,impute_strategy
feature1,numeric,False,mean
feature2,categorical,True,none
feature3,numeric,False,median
```

### Command Line Usage
```bash
# Training mode
python risk_table_mapping.py --job_type training

# Inference modes
python risk_table_mapping.py --job_type validation
python risk_table_mapping.py --job_type testing
```

## Usage Example

### Contract Access
```python
from src.pipeline_script_contracts import RISK_TABLE_MAPPING_CONTRACT

# Access contract details
print(f"Entry Point: {RISK_TABLE_MAPPING_CONTRACT.entry_point}")
print(f"Required Env Vars: {RISK_TABLE_MAPPING_CONTRACT.required_env_vars}")
```

### Integration with Step Builder
```python
from src.pipeline_steps import RiskTableMappingStepBuilder

class RiskTableMappingStepBuilder(StepBuilderBase):
    def validate_configuration(self) -> None:
        validation = RISK_TABLE_MAPPING_CONTRACT.validate_implementation(
            'dockers/xgboost_atoz/pipeline_scripts/risk_table_mapping.py'
        )
        if not validation.is_valid:
            self.logger.warning(f"Script validation warnings: {validation.errors}")
```

## Integration Points

### Pipeline Position
- **Feature Engineering Node** - Transforms categorical and numerical features
- **Input Dependencies** - Requires raw or preprocessed data with metadata
- **Output Consumers** - Provides encoded features for training and evaluation

### Data Flow
```
Raw Data + Metadata → risk_table_mapping.py → Risk-Encoded Features → Training
```

## Best Practices

### Script Development
1. **Robust Risk Calculation** - Handle edge cases in risk score computation
2. **Smoothing Strategy** - Use appropriate smoothing to prevent overfitting
3. **Missing Value Handling** - Implement consistent missing value strategies
4. **Metadata Validation** - Validate metadata consistency with data

### Feature Engineering
1. **Risk Table Quality** - Ensure sufficient data for reliable risk estimates
2. **Smoothing Parameters** - Tune smoothing factor and count threshold appropriately
3. **Imputation Strategy** - Choose appropriate imputation methods per feature
4. **Category Handling** - Handle high-cardinality categorical features properly

## Related Contracts

### Upstream Contracts
- `CRADLE_DATA_LOADING_CONTRACT` - May provide raw data input
- `TABULAR_PREPROCESS_CONTRACT` - May provide preprocessed data

### Downstream Contracts
- `XGBOOST_TRAIN_CONTRACT` - Uses risk-encoded features for training
- `CURRENCY_CONVERSION_CONTRACT` - May process data after risk encoding
- `MODEL_EVALUATION_CONTRACT` - Uses risk-encoded features for evaluation

## Troubleshooting

### Common Issues
1. **Insufficient Data** - Ensure sufficient samples for reliable risk estimation
2. **Missing Metadata** - Verify metadata.csv contains all required columns
3. **Configuration Errors** - Check config.json format and parameter values
4. **Memory Issues** - Monitor memory usage with high-cardinality features

### Validation Failures
1. **Metadata Mismatch** - Ensure metadata matches actual data columns
2. **Target Variable** - Verify target column exists and has valid values
3. **Data Types** - Check data types match metadata specifications
4. **Split Consistency** - Ensure data splits maintain proper structure

## Performance Considerations

### Optimization Strategies
- **Memory Management** - Efficient processing of large categorical features
- **Risk Calculation** - Optimize cross-tabulation for large datasets
- **Caching** - Cache risk tables for repeated transformations
- **Parallel Processing** - Process multiple features in parallel where possible

### Monitoring Metrics
- Risk table creation time
- Memory usage during processing
- Feature transformation accuracy
- Missing value imputation quality

## Security Considerations

### Data Protection
- Secure handling of categorical mappings
- No sensitive category information in logs
- Proper cleanup of temporary risk tables

### Model Security
- Validate risk table integrity
- Protect against category manipulation
- Audit trail for feature transformations

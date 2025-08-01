# Currency Conversion Script Contract

## Overview
The Currency Conversion Script Contract defines the execution requirements for the currency conversion script (`currency_conversion.py`). This contract ensures the script complies with SageMaker processing job conventions for converting monetary values across different currencies in tabular datasets.

## Contract Details

### Script Information
- **Entry Point**: `currency_conversion.py`
- **Container Type**: SageMaker Processing Job
- **Framework**: Python with pandas and multiprocessing
- **Purpose**: Currency normalization and conversion for monetary features

### Input Path Requirements

| Logical Name | Expected Path | Description |
|--------------|---------------|-------------|
| DATA | `/opt/ml/processing/input/data` | Processed data with train/test/val splits |

### Output Path Requirements

| Logical Name | Expected Path | Description |
|--------------|---------------|-------------|
| processed_data | `/opt/ml/processing/output` | Currency-converted data with same split structure |

### Environment Variables

#### Required Variables
- `CURRENCY_CONVERSION_VARS` - JSON list of column names requiring currency conversion
- `CURRENCY_CONVERSION_DICT` - JSON dictionary mapping currency codes to exchange rates
- `MARKETPLACE_INFO` - JSON dictionary mapping marketplace IDs to currency information
- `LABEL_FIELD` - Name of the label column (for stratified splitting)

#### Optional Variables
- `TRAIN_RATIO` - Training data ratio (default: 0.7)
- `TEST_VAL_RATIO` - Test/validation split ratio (default: 0.5)

### Framework Requirements

#### Core Dependencies
```python
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
```

## Script Functionality

Based on the actual implementation in `dockers/xgboost_atoz/pipeline_scripts/currency_conversion.py`:

### Currency Code Resolution
1. **Marketplace-Based Currency Detection**:
   - Maps marketplace IDs to currency codes using marketplace information
   - Handles missing or invalid marketplace IDs with default currency
   - Combines existing currency columns with marketplace-derived currencies

2. **Currency Code Validation**:
   - Validates currency codes against available exchange rates
   - Handles invalid currencies by dropping rows or using default currency
   - Provides flexible handling of missing currency information

### Conversion Processing
1. **Parallel Currency Conversion**:
   - Uses multiprocessing for efficient conversion of multiple variables
   - Applies exchange rates to convert all monetary values to base currency
   - Configurable number of worker processes for optimization

2. **Exchange Rate Application**:
   - Looks up exchange rates from provided conversion dictionary
   - Defaults to 1.0 exchange rate for unknown currencies
   - Handles missing or invalid exchange rate data gracefully

### Processing Modes
1. **Per-Split Mode** (`per_split`):
   - Processes each data split (train/test/val) independently
   - Maintains existing split boundaries
   - Applies conversion to both processed and full data files

2. **Split-After-Conversion Mode** (`split_after_conversion`):
   - Combines all splits before conversion
   - Applies currency conversion to combined dataset
   - Re-splits data using stratified sampling after conversion

### Key Implementation Details

#### Currency Code Resolution
```python
def get_currency_code(marketplace_id, marketplace_info, default_currency):
    """Get currency code for a given marketplace ID"""
    try:
        if pd.isna(marketplace_id) or str(int(marketplace_id)) not in marketplace_info:
            return default_currency
        return marketplace_info[str(int(marketplace_id))]["currency_code"]
    except (ValueError, TypeError):
        return default_currency
```

#### Parallel Conversion Processing
```python
def parallel_currency_conversion(df, currency_col, currency_conversion_vars, 
                               currency_conversion_dict, n_workers=50):
    """Perform parallel currency conversion on multiple variables"""
    exchange_rate_series = df[currency_col].apply(
        lambda x: currency_conversion_dict.get(x, 1.0)
    )
    processes = min(cpu_count(), len(currency_conversion_vars), n_workers)
    
    with Pool(processes=processes) as pool:
        results = pool.map(
            currency_conversion_single_variable,
            [(df[[var]], var, exchange_rate_series) for var in currency_conversion_vars]
        )
```

#### Data Processing Pipeline
```python
def process_currency_conversion(df, marketplace_id_col, currency_conversion_vars,
                              currency_conversion_dict, marketplace_info):
    """Main currency conversion processing pipeline"""
    # Drop rows with missing marketplace IDs
    df = df.dropna(subset=[marketplace_id_col]).reset_index(drop=True)
    
    # Get and combine currency codes
    df, final_currency_col = combine_currency_codes(...)
    
    # Apply parallel conversion
    df = parallel_currency_conversion(...)
```

### Output Structure
```
/opt/ml/processing/output/
├── train/
│   ├── train_processed_data.csv  # Currency-converted processed data
│   └── train_full_data.csv       # Currency-converted full data
├── test/
│   ├── test_processed_data.csv
│   └── test_full_data.csv
└── val/
    ├── val_processed_data.csv
    └── val_full_data.csv
```

### Configuration Examples

#### Environment Variables Setup
```bash
export CURRENCY_CONVERSION_VARS='["price", "cost", "revenue"]'
export CURRENCY_CONVERSION_DICT='{"USD": 1.0, "EUR": 1.2, "GBP": 1.3}'
export MARKETPLACE_INFO='{"1": {"currency_code": "USD"}, "2": {"currency_code": "EUR"}}'
export LABEL_FIELD="target"
```

#### Command Line Usage
```bash
# Per-split mode
python currency_conversion.py --job-type training --mode per_split \
    --marketplace-id-col marketplace_id --enable-conversion true

# Split-after-conversion mode
python currency_conversion.py --job-type training --mode split_after_conversion \
    --marketplace-id-col marketplace_id --train-ratio 0.7 --test-val-ratio 0.5
```

## Usage Example

### Contract Access
```python
from src.pipeline_script_contracts import CURRENCY_CONVERSION_CONTRACT

# Access contract details
print(f"Entry Point: {CURRENCY_CONVERSION_CONTRACT.entry_point}")
print(f"Required Env Vars: {CURRENCY_CONVERSION_CONTRACT.required_env_vars}")
```

### Integration with Step Builder
```python
from src.pipeline_steps import CurrencyConversionStepBuilder

class CurrencyConversionStepBuilder(StepBuilderBase):
    def validate_configuration(self) -> None:
        validation = CURRENCY_CONVERSION_CONTRACT.validate_implementation(
            'dockers/xgboost_atoz/pipeline_scripts/currency_conversion.py'
        )
        if not validation.is_valid:
            self.logger.warning(f"Script validation warnings: {validation.errors}")
```

## Integration Points

### Pipeline Position
- **Preprocessing Node** - Processes data after initial preprocessing
- **Input Dependencies** - Requires preprocessed data with train/test/val splits
- **Output Consumers** - Provides currency-normalized data for training

### Data Flow
```
Preprocessed Data → currency_conversion.py → Currency-Normalized Data → Training
```

## Best Practices

### Script Development
1. **Exchange Rate Management** - Keep exchange rates up-to-date and validated
2. **Error Handling** - Handle missing currencies and marketplace IDs gracefully
3. **Performance Optimization** - Use parallel processing for large datasets
4. **Data Validation** - Validate conversion results for accuracy

### Currency Conversion
1. **Base Currency Selection** - Choose stable base currency (typically USD)
2. **Rate Accuracy** - Use accurate and current exchange rates
3. **Missing Data Handling** - Define clear strategy for missing currency information
4. **Conversion Validation** - Validate conversion results against expected ranges

## Related Contracts

### Upstream Contracts
- `TABULAR_PREPROCESS_CONTRACT` - Provides preprocessed data input
- `CRADLE_DATA_LOADING_CONTRACT` - May provide raw data with currency information

### Downstream Contracts
- `XGBOOST_TRAIN_CONTRACT` - Uses currency-normalized data for training
- `PYTORCH_TRAIN_CONTRACT` - Uses currency-normalized data for training
- `MODEL_EVALUATION_CONTRACT` - Uses currency-normalized data for evaluation

## Troubleshooting

### Common Issues
1. **Missing Exchange Rates** - Ensure all currencies have defined exchange rates
2. **Invalid Marketplace IDs** - Handle missing or invalid marketplace mappings
3. **Performance Issues** - Adjust number of worker processes for optimization
4. **Data Consistency** - Ensure consistent currency handling across splits

### Validation Failures
1. **Environment Variables** - Check all required JSON environment variables are valid
2. **Data Format** - Validate input data has expected columns and structure
3. **Currency Codes** - Ensure currency codes are consistent and valid
4. **Split Consistency** - Verify data splits maintain proper structure after conversion

## Performance Considerations

### Optimization Strategies
- **Parallel Processing** - Use multiprocessing for large datasets
- **Memory Management** - Process data in chunks for memory efficiency
- **Caching** - Cache exchange rate lookups for repeated conversions
- **Vectorization** - Use pandas vectorized operations where possible

### Monitoring Metrics
- Currency conversion processing time
- Memory usage during conversion
- Exchange rate lookup performance
- Data quality after conversion

## Security Considerations

### Data Protection
- Secure handling of financial data
- No sensitive exchange rate information in logs
- Proper cleanup of temporary conversion data

### Exchange Rate Security
- Validate exchange rate sources
- Protect against exchange rate manipulation
- Audit trail for conversion operations

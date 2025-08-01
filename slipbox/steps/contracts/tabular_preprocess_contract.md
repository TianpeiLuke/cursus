# Tabular Preprocessing Script Contract

## Overview
The Tabular Preprocessing Script Contract defines the execution requirements for the tabular preprocessing script (`tabular_preprocess.py`). This contract ensures the script complies with SageMaker processing job conventions for data loading, cleaning, and splitting operations.

## Contract Details

### Script Information
- **Entry Point**: `tabular_preprocess.py`
- **Container Type**: SageMaker Processing Job
- **Framework**: Python with pandas, scikit-learn
- **Purpose**: Data preprocessing and train/test/validation splitting

### Input Path Requirements

| Logical Name | Expected Path | Description |
|--------------|---------------|-------------|
| DATA | `/opt/ml/processing/input/data` | Raw data shards (CSV, JSON, Parquet files) |

### Output Path Requirements

| Logical Name | Expected Path | Description |
|--------------|---------------|-------------|
| processed_data | `/opt/ml/processing/output` | Processed data split into subdirectories |

### Environment Variables

#### Required Variables
- `LABEL_FIELD` - Name of the target label column
- `TRAIN_RATIO` - Proportion of data for training (e.g., 0.7)
- `TEST_VAL_RATIO` - Ratio for test/validation split from holdout data (e.g., 0.5)

#### Optional Variables
| Variable | Default | Description |
|----------|---------|-------------|
| CATEGORICAL_COLUMNS | "" | Comma-separated list of categorical column names |
| NUMERICAL_COLUMNS | "" | Comma-separated list of numerical column names |
| TEXT_COLUMNS | "" | Comma-separated list of text column names |
| DATE_COLUMNS | "" | Comma-separated list of date column names |

### Framework Requirements

#### Core Dependencies
```python
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
```

## Script Functionality

Based on the actual implementation in `dockers/xgboost_atoz/pipeline_scripts/tabular_preprocess.py`:

### Data Loading Pipeline
1. **Shard Detection** - Automatically detects and combines data shards from input directory
2. **Multi-Format Support** - Handles CSV, TSV, JSON, JSON Lines, and Parquet formats
3. **Compression Support** - Supports gzipped files (.gz) for all formats
4. **Separator Detection** - Uses CSV sniffer to automatically detect delimiters
5. **Data Concatenation** - Combines all shards into a single DataFrame

### Supported File Formats
- **CSV/TSV Files**: `part-*.csv`, `part-*.csv.gz`, `part-*.tsv`
- **JSON Files**: `part-*.json`, `part-*.json.gz` (both regular JSON and JSON Lines)
- **Parquet Files**: `part-*.parquet`, `part-*.snappy.parquet`, `part-*.parquet.gz`

### Data Processing Operations
1. **Column Name Normalization** - Replaces `__DOT__` with `.` in column names
2. **Label Field Validation** - Ensures label field exists in the dataset
3. **Label Encoding** - Converts categorical labels to numeric values automatically
4. **Data Type Conversion** - Converts labels to integer type with proper null handling
5. **Missing Value Handling** - Drops rows with missing labels

### Data Splitting Logic
- **Training Mode** (`job_type="training"`):
  - Splits data into train/holdout using `TRAIN_RATIO`
  - Further splits holdout into test/validation using `TEST_VAL_RATIO`
  - Uses stratified sampling to maintain label distribution
  - Outputs three directories: `train/`, `test/`, `val/`

- **Other Modes** (`job_type="validation"` or `job_type="testing"`):
  - Uses entire dataset as single split
  - Outputs single directory named after job_type

### Output Structure
```
/opt/ml/processing/output/
├── train/
│   └── train_processed_data.csv
├── test/
│   └── test_processed_data.csv
└── val/
    └── val_processed_data.csv
```

### Key Implementation Details

#### File Reading Logic
```python
def _read_file_to_df(file_path: Path) -> pd.DataFrame:
    """Handles multiple formats with automatic format detection"""
    # Supports CSV, TSV, JSON, JSON Lines, Parquet
    # Handles gzipped versions of all formats
    # Uses CSV sniffer for delimiter detection
```

#### Label Processing
```python
# Automatic categorical to numeric conversion
if not pd.api.types.is_numeric_dtype(df[label_field]):
    unique_labels = sorted(df[label_field].dropna().unique())
    label_map = {val: idx for idx, val in enumerate(unique_labels)}
    df[label_field] = df[label_field].map(label_map)
```

#### Stratified Splitting
```python
# Maintains label distribution across splits
train_df, holdout_df = train_test_split(
    df, train_size=train_ratio, random_state=42, 
    stratify=df[label_field]
)
```

## Configuration Examples

### Environment Variables Setup
```bash
export LABEL_FIELD="target"
export TRAIN_RATIO="0.7"
export TEST_VAL_RATIO="0.5"
export CATEGORICAL_COLUMNS="category1,category2"
export NUMERICAL_COLUMNS="feature1,feature2,feature3"
```

### Command Line Usage
```bash
python tabular_preprocess.py --job_type training
```

## Usage Example

### Contract Access
```python
from src.pipeline_script_contracts import TABULAR_PREPROCESS_CONTRACT

# Access contract details
print(f"Entry Point: {TABULAR_PREPROCESS_CONTRACT.entry_point}")
print(f"Required Env Vars: {TABULAR_PREPROCESS_CONTRACT.required_env_vars}")
```

### Integration with Step Builder
```python
from src.pipeline_steps import TabularPreprocessStepBuilder

class TabularPreprocessStepBuilder(StepBuilderBase):
    def validate_configuration(self) -> None:
        validation = TABULAR_PREPROCESS_CONTRACT.validate_implementation(
            'dockers/xgboost_atoz/pipeline_scripts/tabular_preprocess.py'
        )
        if not validation.is_valid:
            self.logger.warning(f"Script validation warnings: {validation.errors}")
```

## Integration Points

### Pipeline Position
- **Preprocessing Node** - Processes raw data from source nodes
- **Input Dependencies** - Requires data from Cradle loading or other data sources
- **Output Consumers** - Provides processed data for training and evaluation steps

### Data Flow
```
Raw Data Shards → tabular_preprocess.py → Train/Test/Val Splits → Training Steps
```

## Best Practices

### Script Development
1. **Format Support** - Handle multiple data formats gracefully
2. **Error Handling** - Implement robust error handling for file reading
3. **Memory Efficiency** - Process large datasets efficiently
4. **Logging** - Provide detailed logging for debugging

### Data Quality
1. **Label Validation** - Ensure label field exists and has valid values
2. **Missing Data** - Handle missing values appropriately
3. **Data Types** - Ensure proper data type conversion
4. **Stratification** - Maintain label distribution in splits

## Related Contracts

### Upstream Contracts
- `CRADLE_DATA_LOADING_CONTRACT` - Provides raw data input
- `CURRENCY_CONVERSION_CONTRACT` - May provide processed data

### Downstream Contracts
- `XGBOOST_TRAIN_CONTRACT` - Uses processed data for training
- `PYTORCH_TRAIN_CONTRACT` - Uses processed data for training
- `MODEL_EVALUATION_CONTRACT` - Uses processed data for evaluation

## Troubleshooting

### Common Issues
1. **File Format Errors** - Check supported formats and compression
2. **Label Field Missing** - Verify LABEL_FIELD environment variable
3. **Memory Issues** - Monitor memory usage with large datasets
4. **Split Ratio Errors** - Ensure ratios sum appropriately

### Validation Failures
1. **Path Validation** - Ensure input/output paths follow SageMaker conventions
2. **Environment Variables** - Check all required variables are set
3. **Data Format** - Validate input data format compatibility
4. **Output Structure** - Ensure output directories are created correctly

## Performance Considerations

### Optimization Strategies
- **Parallel Processing** - Uses multiprocessing for large datasets
- **Memory Management** - Efficient DataFrame operations
- **File I/O** - Optimized file reading with format detection
- **Chunking** - Processes data in chunks for large files

### Monitoring Metrics
- Data processing time and throughput
- Memory usage during processing
- Split distribution validation
- File format detection accuracy

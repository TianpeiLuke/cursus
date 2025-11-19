---
tags:
  - code
  - processing_script
  - currency_conversion
  - data_transformation
  - parallel_processing
keywords:
  - currency conversion
  - exchange rates
  - marketplace localization
  - parallel processing
  - monetary values
  - data normalization
topics:
  - currency conversion
  - data preprocessing
  - ML pipelines
language: python
date of note: 2025-11-18
---

# Currency Conversion Script Documentation

## Overview

The `currency_conversion.py` script performs automated currency conversion on monetary values within datasets, transforming amounts from various currencies to a standard target currency (typically USD) based on marketplace information and configurable exchange rates. It operates as a SageMaker Processing step with support for parallel processing and multiple job types.

The script provides two methods for identifying currencies: direct currency code lookup from a dedicated column, or indirect lookup via marketplace ID mapping. It applies exchange rate conversions to specified monetary variables while preserving all other data columns and maintaining the original data format (CSV/TSV/Parquet).

A key feature is its integration with upstream preprocessing steps through the stacked preprocessing pattern, enabling seamless currency normalization as part of broader feature engineering workflows.

## Purpose and Major Tasks

### Primary Purpose
Convert monetary values across different currencies to a standard currency for ML model training, ensuring consistent feature scales and enabling cross-market model development.

### Major Tasks
1. **Currency Code Detection**: Identify currency for each row using direct codes or marketplace ID mapping
2. **Exchange Rate Lookup**: Map detected currencies to conversion rates from configuration
3. **Parallel Conversion**: Apply exchange rates to multiple monetary variables efficiently using multiprocessing
4. **Format Preservation**: Maintain input file format (CSV/TSV/Parquet) in output
5. **Multi-Split Processing**: Handle train/val/test splits for training jobs or single splits for other job types
6. **Default Handling**: Apply default currency when currency information is missing or invalid
7. **Validation**: Ensure required variables exist in dataset before conversion
8. **Configuration Loading**: Parse JSON-formatted configuration from environment variables
9. **Worker Pool Management**: Optimize parallel processing based on CPU count and variable count
10. **Error Handling**: Handle missing columns, invalid rates, and malformed data gracefully

## Script Contract

### Entry Point
```
currency_conversion.py
```

### Input Paths
| Path | Location | Description |
|------|----------|-------------|
| `input_data` | `/opt/ml/processing/input/data` | Processed data with train/val/test or single split |

**Expected Input Files**:
- `input_data/{split}/{split}_processed_data.{csv|tsv|parquet}` (required): Processed data files from upstream steps

### Output Paths
| Path | Location | Description |
|------|----------|-------------|
| `processed_data` | `/opt/ml/processing/output` | Currency-converted data in same format as input |

**Output Files**:
- `processed_data/{split}/{split}_processed_data.{csv|tsv|parquet}`: Original data with converted monetary values

### Required Environment Variables

| Variable | Description | Format |
|----------|-------------|--------|
| `CURRENCY_CONVERSION_VARS` | List of column names containing monetary values to convert | JSON array of strings |
| `CURRENCY_CONVERSION_DICT` | Mapping of marketplace IDs to currencies and exchange rates | JSON object with mappings array |

**CURRENCY_CONVERSION_DICT Structure**:
```json
{
  "mappings": [
    {
      "marketplace_id": "1",
      "currency_code": "USD",
      "conversion_rate": 1.0
    },
    {
      "marketplace_id": "3",
      "currency_code": "EUR",
      "conversion_rate": 0.85
    }
  ]
}
```

### Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CURRENCY_CODE_FIELD` | `""` | Name of column containing direct currency codes |
| `MARKETPLACE_ID_FIELD` | `""` | Name of column containing marketplace IDs for currency lookup |
| `DEFAULT_CURRENCY` | `"USD"` | Default currency when lookup fails |
| `N_WORKERS` | `"50"` | Maximum number of parallel workers for conversion |

**Note**: At least one of `CURRENCY_CODE_FIELD` or `MARKETPLACE_ID_FIELD` must be provided for currency identification.

### Job Arguments
| Argument | Required | Choices | Description |
|----------|----------|---------|-------------|
| `--job_type` | Yes | training, validation, testing, calibration | Determines splits to process |

**Job Type Behavior**:
- `training`: Processes train/, val/, test/ subdirectories
- `validation`: Processes validation/ subdirectory only
- `testing`: Processes testing/ subdirectory only
- `calibration`: Processes calibration/ subdirectory only

## Input Data Structure

### Expected Input Format
```
input_data/
├── train/
│   └── train_processed_data.{csv|tsv|parquet}
├── val/
│   └── val_processed_data.{csv|tsv|parquet}
└── test/
    └── test_processed_data.{csv|tsv|parquet}
```

### Input Data Example (Before Conversion)

**With Direct Currency Code**:
```csv
txn_id,marketplace_id,currency_code,transaction_amount,shipping_cost,tax_amount
1,1,USD,100.00,5.00,8.00
2,3,EUR,85.00,4.25,7.00
3,2,GBP,120.00,6.00,12.00
4,1,USD,50.00,2.50,4.00
```

**With Marketplace ID Lookup**:
```csv
txn_id,marketplace_id,transaction_amount,shipping_cost,tax_amount
1,1,100.00,5.00,8.00
2,3,85.00,4.25,7.00
3,2,120.00,6.00,12.00
4,1,50.00,2.50,4.00
```

### Configuration Example

**Environment Variables**:
```bash
export CURRENCY_CODE_FIELD="currency_code"
export MARKETPLACE_ID_FIELD="marketplace_id"
export DEFAULT_CURRENCY="USD"
export N_WORKERS="50"

export CURRENCY_CONVERSION_VARS='["transaction_amount", "shipping_cost", "tax_amount"]'

export CURRENCY_CONVERSION_DICT='{
  "mappings": [
    {"marketplace_id": "1", "currency_code": "USD", "conversion_rate": 1.0},
    {"marketplace_id": "2", "currency_code": "GBP", "conversion_rate": 0.73},
    {"marketplace_id": "3", "currency_code": "EUR", "conversion_rate": 0.85},
    {"marketplace_id": "4", "currency_code": "JPY", "conversion_rate": 110.0}
  ]
}'
```

## Output Data Structure

### Output Directory Structure
```
processed_data/
├── train/
│   └── train_processed_data.{csv|tsv|parquet}
├── val/
│   └── val_processed_data.{csv|tsv|parquet}
└── test/
    └── test_processed_data.{csv|tsv|parquet}
```

### Output Data Example (After Conversion to USD)

**Converted Values** (all amounts now in USD):
```csv
txn_id,marketplace_id,currency_code,transaction_amount,shipping_cost,tax_amount
1,1,USD,100.00,5.00,8.00
2,3,EUR,100.00,5.00,8.24
3,2,GBP,164.38,8.22,16.44
4,1,USD,50.00,2.50,4.00
```

**Conversion Details**:
- Row 1: USD → USD (rate 1.0): No change
- Row 2: EUR → USD (rate 0.85): 85.00/0.85 = 100.00
- Row 3: GBP → USD (rate 0.73): 120.00/0.73 = 164.38
- Row 4: USD → USD (rate 1.0): No change

**Key Properties**:
- All original columns preserved
- Monetary columns converted to target currency
- Same row order maintained
- Same data types preserved
- Format matches input (CSV→CSV, Parquet→Parquet)

## Key Functions and Tasks

### Main Orchestration Component

#### `main(input_paths, output_paths, environ_vars, job_args)`
**Purpose**: Standardized entry point orchestrating the complete currency conversion workflow

**Algorithm**:
```python
1. Validate input parameters:
   - Check required input_paths keys (input_data)
   - Check required output_paths keys (processed_data)
   - Verify job_args contains job_type
2. Extract paths from parameters
3. Log input/output paths for traceability
4. Parse currency configuration from environment variables:
   - Load CURRENCY_CONVERSION_VARS JSON array
   - Load CURRENCY_CONVERSION_DICT JSON object
   - Extract field names and default settings
5. Build currency_config dictionary with all settings
6. Call internal_main() with extracted configuration
7. Handle exceptions:
   - Log detailed error messages
   - Re-raise exceptions for upstream handling
8. Return processed dataframes dictionary
```

**Parameters**:
- `input_paths` (dict): Input directory paths
- `output_paths` (dict): Output directory paths
- `environ_vars` (dict): Environment variable configuration
- `job_args` (Namespace): Command-line arguments with job_type

**Returns**: `Dict[str, pd.DataFrame]` - Converted DataFrames by split name

**Raises**: `ValueError` if required parameters missing

#### `internal_main(job_type, input_dir, output_dir, currency_config, load_data_func, save_data_func)`
**Purpose**: Core conversion logic with dependency injection support for testing

**Algorithm**:
```python
1. Create output directory if needed
2. Log currency configuration details
3. Load data according to job_type:
   - Call load_data_func (injectable for tests)
   - Returns dict with DataFrames and format metadata
4. Process currency conversion:
   - Call process_data() with loaded data
   - Returns dict with converted DataFrames
5. Save processed data:
   - Call save_data_func (injectable for tests)
   - Preserves detected format from load
6. Log completion message
7. Return converted DataFrames dictionary
```

**Parameters**:
- `job_type` (str): Job type determining splits
- `input_dir` (str): Input data directory
- `output_dir` (str): Output data directory
- `currency_config` (dict): Complete currency configuration
- `load_data_func` (callable): Data loading function (for DI)
- `save_data_func` (callable): Data saving function (for DI)

**Returns**: `Dict[str, pd.DataFrame]` - Converted DataFrames

### Data I/O Components

#### `_detect_file_format(split_dir, split_name)`
**Purpose**: Automatically detect data file format from available files

**Algorithm**:
```python
1. Define format preferences in order:
   - CSV: {split}_processed_data.csv
   - TSV: {split}_processed_data.tsv
   - Parquet: {split}_processed_data.parquet
2. For each format in preference order:
   - Build file path: split_dir / filename
   - Check if file exists
   - If found: return (file_path, format)
3. If no files found:
   - Raise RuntimeError with list of expected files
```

**Parameters**:
- `split_dir` (Path): Directory containing split data
- `split_name` (str): Name of split (train/val/test/etc.)

**Returns**: `Tuple[Path, str]` - File path and format string

**Raises**: `RuntimeError` if no data file found

#### `load_split_data(job_type, input_dir)`
**Purpose**: Load data splits based on job type with automatic format detection

**Algorithm**:
```python
1. Initialize result dictionary and input_path
2. Check job_type:
   
   a. If "training":
      - Define splits = ["train", "test", "val"]
      - Initialize detected_format = None
      - For each split_name:
        * Detect file format: _detect_file_format(split_dir, split_name)
        * Store format from first split
        * Read DataFrame based on detected format:
          - csv: pd.read_csv(file_path)
          - tsv: pd.read_csv(file_path, sep="\t")
          - parquet: pd.read_parquet(file_path)
        * Store in result[split_name]
      - Store format: result["_format"] = detected_format
      - Log shapes for all splits
   
   b. Else (other job types):
      - Use job_type as split_name
      - Detect file format: _detect_file_format(split_dir, job_type)
      - Read DataFrame based on detected format
      - Store in result[job_type]
      - Store format: result["_format"] = detected_format
      - Log shape

3. Return result dictionary with DataFrames and format
```

**Parameters**:
- `job_type` (str): Job type determining which splits to load
- `input_dir` (str): Input data directory path

**Returns**: `Dict[str, pd.DataFrame]` - DataFrames by split name plus "_format" key

**Raises**: `RuntimeError` if data files not found or format unsupported

**Complexity**: O(n × m) where n = splits, m = columns in each split

#### `save_output_data(job_type, output_dir, data_dict)`
**Purpose**: Save converted data preserving input format

**Algorithm**:
```python
1. Extract format from data_dict (stored during load)
   - Default to "csv" if "_format" key not found
2. For each split in data_dict:
   - Skip "_format" metadata key
   - Create split output directory
   - Build output filename based on format
   - Write DataFrame using appropriate method:
     * csv: df.to_csv(file, index=False)
     * tsv: df.to_csv(file, sep="\t", index=False)
     * parquet: df.to_parquet(file, index=False)
   - Log saved file path and shape
```

**Parameters**:
- `job_type` (str): Job type (not actively used, splits from dict keys)
- `output_dir` (str): Output directory path
- `data_dict` (dict): DataFrames by split name with "_format" metadata

**Raises**: `RuntimeError` if format unsupported

**Complexity**: O(n × m) where n = splits, m = columns per split

### Currency Identification Component

#### `get_currency_code(row, currency_code_field, marketplace_id_field, conversion_dict, default_currency)`
**Purpose**: Determine currency code for a data row using available information

**Algorithm**:
```python
1. Check if currency_code_field available:
   a. If currency_code_field AND field in row:
      - Get value: currency = row[currency_code_field]
      - If not null and not empty string:
        * Strip whitespace
        * Return currency code
   
2. Check if marketplace_id_field available:
   a. If marketplace_id_field AND field in row:
      - Get value: marketplace_id = row[marketplace_id_field]
      - If not null:
        * Extract mappings from conversion_dict
        * For each mapping in mappings:
          - If mapping.marketplace_id == marketplace_id:
            * Return mapping.currency_code (or default)
   
3. Return default_currency (fallback)
```

**Parameters**:
- `row` (pd.Series): Data row to process
- `currency_code_field` (str | None): Column with direct currency codes
- `marketplace_id_field` (str | None): Column with marketplace IDs
- `conversion_dict` (dict): Configuration with marketplace mappings
- `default_currency` (str): Fallback currency code

**Returns**: `str` - Currency code for the row

**Decision Logic**:
- Priority 1: Direct currency code from dedicated column
- Priority 2: Marketplace ID lookup from mappings
- Priority 3: Default currency fallback

### Conversion Processing Component

#### `process_data(data_dict, job_type, currency_config)`
**Purpose**: Core data processing orchestrator for currency conversion across all splits

**Algorithm**:
```python
1. Extract configuration values:
   - currency_code_field
   - marketplace_id_field  
   - currency_conversion_vars (list)
   - currency_conversion_dict (mappings)
   - default_currency
   - n_workers

2. Validate configuration:
   a. If neither currency_code_field nor marketplace_id_field:
      - Log warning
      - Return data_dict unchanged (skip conversion)
   b. If currency_conversion_vars empty:
      - Log warning
      - Return data_dict unchanged (skip conversion)

3. Log conversion configuration details

4. Initialize converted_data dictionary

5. For each (split_name, df) in data_dict.items():
   - Skip "_format" metadata key
   - Log processing start with row count
   - Call process_currency_conversion():
     * Pass df and all configuration parameters
     * Returns converted DataFrame
   - Store in converted_data[split_name]
   - Log completion with shape

6. Return converted_data dictionary
```

**Parameters**:
- `data_dict` (dict): DataFrames by split name
- `job_type` (str): Job type for logging context
- `currency_config` (dict): Complete currency configuration

**Returns**: `Dict[str, pd.DataFrame]` - Converted DataFrames by split

**Complexity**: O(s × n × r × v) where s = splits, n = rows per split, r = row processing overhead, v = conversion variables

#### `process_currency_conversion(df, currency_code_field, marketplace_id_field, currency_conversion_vars, currency_conversion_dict, default_currency, n_workers)`
**Purpose**: Perform currency conversion on a single DataFrame

**Algorithm**:
```python
1. Log start with DataFrame shape

2. Filter variables to those existing in DataFrame:
   - Keep only vars where var in df.columns
   - Update currency_conversion_vars list

3. If no variables remain after filtering:
   - Log warning
   - Return original DataFrame unchanged

4. Create temporary currency code column:
   - Apply get_currency_code() to each row
   - Store in df["__temp_currency_code__"]

5. Build exchange rate series:
   - Initialize empty exchange_rates list
   - Extract mappings from currency_conversion_dict
   - For each currency_code in temp column:
     * Default rate = 1.0 (no conversion)
     * Search mappings for matching currency_code
     * If found: rate = mapping.conversion_rate
     * Append rate to exchange_rates list
   - Create pd.Series with rates aligned to df.index

6. Log variables being converted

7. Call parallel_currency_conversion():
   - Pass df, exchange_rate_series, vars, n_workers
   - Returns DataFrame with converted values

8. Clean up temporary column:
   - Drop __temp_currency_code__

9. Log completion
10. Return converted DataFrame
```

**Parameters**:
- `df` (pd.DataFrame): DataFrame to convert
- `currency_code_field` (str | None): Currency code column name
- `marketplace_id_field` (str | None): Marketplace ID column name
- `currency_conversion_vars` (list): Variables to convert
- `currency_conversion_dict` (dict): Conversion mappings
- `default_currency` (str): Default currency fallback
- `n_workers` (int): Maximum parallel workers

**Returns**: `pd.DataFrame` - DataFrame with converted values

**Complexity**: O(n × v) where n = rows, v = variables to convert

#### `parallel_currency_conversion(df, exchange_rate_series, currency_conversion_vars, n_workers)`
**Purpose**: Apply exchange rate conversions to multiple variables in parallel

**Algorithm**:
```python
1. Calculate optimal process count:
   - processes = min(cpu_count(), len(vars), n_workers)
   - Limits: system CPUs, variable count, config maximum

2. Create multiprocessing Pool with calculated processes

3. Map currency_conversion_single_variable() over variables:
   - Build args tuples: (df[[var]], var, exchange_rate_series)
   - Pool.map() applies function to each variable
   - Collects results list

4. Combine results:
   - pd.concat(results, axis=1)
   - Assigns back to df[currency_conversion_vars]

5. Close pool

6. Return modified DataFrame
```

**Parameters**:
- `df` (pd.DataFrame): DataFrame with original values
- `exchange_rate_series` (pd.Series): Exchange rates per row
- `currency_conversion_vars` (list): Variables to convert
- `n_workers` (int): Maximum workers

**Returns**: `pd.DataFrame` - DataFrame with converted variables

**Complexity**: O(n × v / p) where n = rows, v = variables, p = processes

**Parallelization Benefit**: Near-linear speedup with number of cores for large variable counts

#### `currency_conversion_single_variable(args)`
**Purpose**: Convert single variable's values using exchange rates (worker function)

**Algorithm**:
```python
1. Unpack args tuple:
   - df: Single-column DataFrame
   - variable: Column name
   - exchange_rate_series: Exchange rates

2. Apply conversion:
   - df[variable] / exchange_rate_series.values
   - Element-wise division
   - Converts from source to target currency

3. Return converted Series
```

**Parameters**:
- `args` (tuple): (df, variable, exchange_rate_series)

**Returns**: `pd.Series` - Converted values for the variable

**Complexity**: O(n) where n = rows

**Note**: Designed for parallel execution via multiprocessing.Pool.map()

## Algorithms and Data Structures

### Exchange Rate Conversion Algorithm
**Problem**: Convert monetary values from multiple source currencies to a single target currency efficiently

**Solution Strategy**:
1. Identify source currency for each row
2. Look up exchange rate for each currency
3. Apply division to convert values (target = source / rate)
4. Process multiple variables in parallel

**Algorithm**:
```python
def convert_currencies(df, currency_col, vars, rates_dict):
    # Step 1: Get currency code per row
    currency_codes = df.apply(lambda r: get_currency(r), axis=1)
    
    # Step 2: Build exchange rate series
    rates = [rates_dict.get(code, 1.0) for code in currency_codes]
    rate_series = pd.Series(rates, index=df.index)
    
    # Step 3: Convert each variable
    for var in vars:
        df[var] = df[var] / rate_series
    
    return df
```

**Mathematical Foundation**:
```
target_amount = source_amount / exchange_rate

Example:
100 EUR → USD with rate 0.85:
target = 100 / 0.85 = 117.65 USD
```

**Complexity**: O(n × v) where n = rows, v = variables

**Key Features**:
- Vectorized operations for performance
- Handles multiple source currencies simultaneously
- Preserves null values (null / rate = null)
- Supports arbitrary exchange rates

### Parallel Processing Architecture
**Problem**: Convert multiple monetary variables efficiently without blocking

**Solution Strategy**:
1. Split variables across multiple worker processes
2. Each worker converts one variable independently
3. Combine results from all workers

**Data Structure**:
```python
# Worker pool configuration
processes = min(cpu_count(), len(variables), max_workers)

# Task distribution
tasks = [
    (df[[var]], var, exchange_rates)
    for var in currency_conversion_vars
]

# Parallel execution
with Pool(processes=processes) as pool:
    results = pool.map(worker_function, tasks)
    
# Result combination
df[vars] = pd.concat(results, axis=1)
```

**Complexity**:
- Sequential: O(n × v)
- Parallel: O(n × v / p) where p = processes
- Speedup: ~p× for large datasets

**Key Features**:
- Automatic process count optimization
- CPU-bound parallelization (GIL-free)
- Memory efficient (shares DataFrame via pickle)
- Graceful degradation (works with p=1)

### Currency Lookup Data Structure
**Problem**: Efficiently map marketplace IDs or currency codes to exchange rates

**Solution Strategy**:
1. Store mappings in dictionary structure
2. Linear search for marketplace lookup (small mapping count)
3. Priority-based fallback system

**Data Structure**:
```python
currency_conversion_dict = {
    "mappings": [
        {
            "marketplace_id": "1",
            "currency_code": "USD",
            "conversion_rate": 1.0
        },
        {
            "marketplace_id": "3",
            "currency_code": "EUR", 
            "conversion_rate": 0.85
        }
    ]
}

# Optimized lookup cache (built at runtime)
rate_cache = {
    "USD": 1.0,
    "EUR": 0.85,
    "GBP": 0.73
}
```

**Lookup Complexity**:
- Direct currency code: O(1) with cache
- Marketplace ID: O(m) where m = mappings (typically <100)
- Default fallback: O(1)

**Key Features**:
- JSON-serializable for environment variables
- Supports both lookup methods
- Extensible to additional marketplaces
- Cache-friendly for repeated lookups

## Performance Characteristics

### Complexity Analysis

| Operation | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| Load Data | O(n × m) | O(n × m) | n = rows, m = columns |
| Currency Lookup | O(n) | O(n) | Per-row currency identification |
| Rate Building | O(n) | O(n) | Create exchange rate series |
| Single Variable Conversion | O(n) | O(1) | Vectorized division |
| Parallel Conversion | O(n × v / p) | O(n × v) | v = vars, p = processes |
| Save Data | O(n × m) | O(n × m) | Write all data |

**Overall Complexity**: O(n × m + n × v / p) where n = rows, m = total columns, v = conversion variables, p = processes

**Typical Performance**:
- Small datasets (<10K rows, <10 vars): < 2 seconds
- Medium datasets (10K-100K rows, 10-50 vars): 2-15 seconds
- Large datasets (>100K rows, >50 vars): 15-120 seconds

### Memory Usage

**Peak Memory**: O(n × m) for DataFrame storage

- Input DataFrame: O(n × m)
- Exchange rate series: O(n)
- Temporary currency column: O(n)
- Worker processes: p × O(n) (shared via pickle)

**Optimization Opportunities**:
1. Chunked processing for very large datasets (>1M rows)
2. In-place conversion to reduce memory copies
3. Process splits independently (parallel job execution)
4. Use Parquet format for compressed storage

### Parallelization Analysis

**Speedup Factor**: ~p× for v >> p (many variables, optimal process count)

**Example Scaling**:
```
Variables = 50, Processes = 8
Sequential time: 100 seconds
Parallel time: 100/8 ≈ 12.5 seconds
Speedup: 8× (linear scaling)
```

**Diminishing Returns**:
- p > v: No benefit (fewer variables than workers)
- p > CPU count: Overhead from context switching
- Very small datasets: Multiprocessing overhead dominates

## Error Handling

### Error Types

#### Configuration Errors
- **Missing Currency Identification**: Neither CURRENCY_CODE_FIELD nor MARKETPLACE_ID_FIELD provided
  - **Handling**: Logs warning, skips conversion, returns original data
  
- **Empty Variable List**: CURRENCY_CONVERSION_VARS is empty or invalid JSON
  - **Handling**: Logs warning, skips conversion, returns original data

- **Malformed Conversion Dict**: CURRENCY_CONVERSION_DICT invalid JSON or missing mappings
  - **Handling**: Propagates JSON parse error with context

#### Data Errors
- **Missing Columns**: Specified conversion variables not in DataFrame
  - **Handling**: Filters to existing columns, logs which variables skipped

- **Missing Currency Fields**: Currency lookup columns not in DataFrame
  - **Handling**: Falls back to default currency for all rows

- **Null Currency Values**: Null or empty currency codes in data
  - **Handling**: Uses default currency for affected rows

#### I/O Errors
- **File Not Found**: Expected data files missing
  - **Handling**: Raises RuntimeError with list of expected files

- **Format Detection Failure**: No supported format files found
  - **Handling**: Raises RuntimeError with tried formats

- **Unsupported Format**: File format not in [csv, tsv, parquet]
  - **Handling**: Raises RuntimeError with unsupported format name

#### Processing Errors
- **Invalid Exchange Rates**: Non-numeric or zero conversion rates
  - **Handling**: Division by zero → infinity values (preserved in output)

- **Type Conversion Errors**: Non-numeric values in monetary columns
  - **Handling**: Propagates pandas type error

### Error Response Structure

When conversion fails, detailed logging provides:

```python
logger.error(f"Error in currency conversion: {str(e)}")
logger.error(traceback.format_exc())
sys.exit(3)  # Exit code indicates type of failure
```

**Exit Codes**:
- 0: Success
- 1: File not found
- 2: Value error (config/parameter issues)
- 3: General exception

## Best Practices

### For Production Deployments
1. **Validate Configuration**: Test currency mappings with sample data before production
2. **Monitor Conversion Rates**: Ensure rates are updated regularly and accurately
3. **Check Variable Existence**: Verify all specified variables exist in actual data
4. **Use Appropriate Worker Count**: Balance parallelization vs. memory overhead
5. **Format Consistency**: Use same format across all pipeline steps for efficiency

### For Development
1. **Start with Lenient Settings**: Use default currency liberally during exploration
2. **Test Both Lookup Methods**: Verify both direct codes and marketplace ID lookups work
3. **Validate Rates**: Check conversion results manually for sample rows
4. **Log Extensively**: Enable detailed logging to debug currency lookups
5. **Test Edge Cases**: Handle missing currencies, null values, zero amounts

### For Performance Optimization
1. **Optimize Process Count**: Use N_WORKERS based on dataset size and variable count
2. **Use Parquet Format**: Parquet is faster than CSV for large datasets
3. **Minimize Variables**: Only convert truly monetary columns
4. **Cache Exchange Rates**: Build rate lookup once, reuse for all rows
5. **Process Splits in Parallel**: Run separate jobs for train/val/test if needed

## Example Configurations

### Basic Conversion with Direct Currency Codes
```bash
export CURRENCY_CODE_FIELD="currency_code"
export DEFAULT_CURRENCY="USD"
export N_WORKERS="50"

export CURRENCY_CONVERSION_VARS='["amount", "tax", "shipping"]'

export CURRENCY_CONVERSION_DICT='{
  "mappings": [
    {"marketplace_id": "1", "currency_code": "USD", "conversion_rate": 1.0},
    {"marketplace_id": "3", "currency_code": "EUR", "conversion_rate": 0.85}
  ]
}'

python currency_conversion.py --job_type training
```

**Use Case**: Standard e-commerce with explicit currency codes per transaction

### Marketplace ID-Based Conversion
```bash
export MARKETPLACE_ID_FIELD="marketplace_id"
export DEFAULT_CURRENCY="USD"
export N_WORKERS="50"

export CURRENCY_CONVERSION_VARS='["price", "discount", "total"]'

export CURRENCY_CONVERSION_DICT='{
  "mappings": [
    {"marketplace_id": "1", "currency_code": "USD", "conversion_rate": 1.0},
    {"marketplace_id": "2", "currency_code": "GBP", "conversion_rate": 0.73},
    {"marketplace_id": "3", "currency_code": "EUR", "conversion_rate": 0.85},
    {"marketplace_id": "4", "currency_code": "JPY", "conversion_rate": 110.0},
    {"marketplace_id": "5", "currency_code": "CAD", "conversion_rate": 1.25}
  ]
}'

python currency_conversion.py --job_type training
```

**Use Case**: Multi-marketplace platform where currency is implicit from marketplace

### Dual Lookup with Fallback
```bash
export CURRENCY_CODE_FIELD="currency_code"
export MARKETPLACE_ID_FIELD="marketplace_id"
export DEFAULT_CURRENCY="USD"

export CURRENCY_CONVERSION_VARS='["amount"]'

export CURRENCY_CONVERSION_DICT='{
  "mappings": [
    {"marketplace_id": "1", "currency_code": "USD", "conversion_rate": 1.0},
    {"marketplace_id": "3", "currency_code": "EUR", "conversion_rate": 0.85}
  ]
}'

python currency_conversion.py --job_type validation
```

**Use Case**: Robust setup with primary and fallback currency identification

### Performance-Optimized Configuration
```bash
export MARKETPLACE_ID_FIELD="marketplace_id"
export DEFAULT_CURRENCY="USD"
export N_WORKERS="16"  # Optimized for system with 16 cores

export CURRENCY_CONVERSION_VARS='["amt1", "amt2", "amt3", "amt4", "amt5"]'

export CURRENCY_CONVERSION_DICT='{
  "mappings": [
    {"marketplace_id": "1", "currency_code": "USD", "conversion_rate": 1.0}
  ]
}'

python currency_conversion.py --job_type training
```

**Use Case**: Large dataset processing with worker count tuned to CPU availability

## Integration Patterns

### Upstream Integration
```
TabularPreprocessing → processed_data
   ↓
CurrencyConversion → processed_data (normalized currencies)
   ↓
FeatureEngineering → processed_data
```

**Input Source**: TabularPreprocessing or any preprocessing step producing monetary features

### Downstream Integration
```
CurrencyConversion → processed_data (normalized)
   ↓
MissingValueImputation → processed_data (filled)
   ↓
FeatureSelection → processed_data (reduced)
   ↓
TrainingStep (XGBoost/LightGBM/PyTorch)
```

**Output Consumers**:
- **Feature Engineering Steps**: Use normalized monetary values for ratio features
- **Missing Value Imputation**: May impute converted monetary columns
- **Training Steps**: Use consistent currency scale across all features

### Stacked Preprocessing Pattern

The script enables seamless preprocessing pipeline composition through shared `processed_data` directories:

```python
# Step 1: Tabular preprocessing
preprocessing_step = TabularPreprocessingStepBuilder(config).create_step()

# Step 2: Currency conversion (adds normalized monetary values)
currency_step = CurrencyConversionStepBuilder(config).create_step(
    inputs={'input_data': preprocessing_step.properties...}
)

# Step 3: Feature engineering (uses normalized values)
feature_eng_step = FeatureEngineeringStepBuilder(config).create_step(
    inputs={'input_data': currency_step.properties...}
)

# Step 4: Training (uses features with consistent currency scale)
training_step = XGBoostTrainingStepBuilder(config).create_step(
    inputs={'training_data': feature_eng_step.properties...}
)
```

### Workflow Example
1. **TabularPreprocessing**: Cleans and structures raw transaction data
2. **CurrencyConversion**: Normalizes all monetary amounts to USD
3. **FeatureEngineering**: Creates price ratios, percentages using normalized values
4. **Training**: XGBoost model trains on features with consistent currency scale

## Troubleshooting

### Missing Currency Information

**Symptom**: All rows using default currency, no actual conversion happening

**Common Causes**:
1. **Wrong field names**: CURRENCY_CODE_FIELD or MARKETPLACE_ID_FIELD don't match actual columns
2. **Null values**: Currency lookup columns contain only nulls
3. **Mismatched marketplace IDs**: IDs in data don't match mappings configuration

**Solution**:
1. Verify column names match actual DataFrame columns exactly (case-sensitive)
2. Check for null percentages in currency lookup columns
3. Review marketplace ID values in data vs. mappings configuration
4. Add more marketplace mappings to cover all IDs in data
5. Verify default currency is appropriate for your use case

### Conversion Not Applied

**Symptom**: Output values identical to input values

**Common Causes**:
1. **Empty variable list**: CURRENCY_CONVERSION_VARS is empty
2. **Variables not in data**: Specified variables don't exist in DataFrame
3. **All rates = 1.0**: No actual conversion needed (same currency)

**Solution**:
1. Check CURRENCY_CONVERSION_VARS contains variable names as JSON array
2. Verify variable names match DataFrame columns exactly
3. Review conversion_rate values in mappings (should vary by currency)
4. Check logs for "No variables require currency conversion" warning
5. Ensure at least some currencies have rate ≠ 1.0

### Performance Issues

**Symptom**: Conversion takes excessively long time

**Common Causes**:
1. **Too many workers**: N_WORKERS higher than available CPUs
2. **Too few workers**: N_WORKERS=1 prevents parallelization
3. **Large dataset**: Millions of rows with many variables
4. **CSV format**: Text format slower than binary

**Solution**:
1. Set N_WORKERS to match CPU count (use cpu_count())
2. For small variable counts (<10), reduce N_WORKERS to avoid overhead
3. For large datasets, consider chunked processing or Parquet format
4. Use Parquet format for 2-5× faster I/O
5. Monitor memory usage and adjust worker count accordingly

### Incorrect Conversion Results

**Symptom**: Converted values don't match expected calculations

**Common Causes**:
1. **Wrong conversion rates**: Rates inverted or incorrectly specified
2. **Currency code mismatch**: Lookup returns wrong currency
3. **Floating point precision**: Rounding differences in calculations

**Solution**:
1. Verify conversion rates are correct (target = source / rate)
2. Test conversion manually: 100 EUR / 0.85 = 117.65 USD
3. Check currency code lookup logic with sample rows
4. Review marketplace ID to currency code mappings
5. Accept minor floating point differences (normal behavior)

### Format Detection Errors

**Symptom**: Script fails to find input files

**Common Causes**:
1. **Wrong directory structure**: Files not in expected {split}/ subdirectories
2. **Wrong filenames**: Files not named {split}_processed_data.*
3. **Unsupported format**: Files in format other than CSV/TSV/Parquet

**Solution**:
1. Ensure files follow naming convention: train_processed_data.csv
2. Verify files are in correct subdirectories: input_data/train/
3. Use supported formats only: .csv, .tsv, .parquet, .csv.gz, .tsv.gz
4. Check logs for "No processed data file found" error with expected files
5. Ensure upstream step produced output in expected format

## References

### Related Scripts
- [`tabular_preprocessing.py`](tabular_preprocess_script.md): Upstream data preprocessing step
- [`missing_value_imputation.py`]: Missing value handling after currency conversion
- [`feature_selection.py`]: Feature selection using normalized monetary values

### Related Documentation
- **Step Builder**: Currency conversion step builder implementation
- **Config Class**: Configuration class for currency conversion step
- **Contract**: [`src/cursus/steps/contracts/currency_conversion_contract.py`](../../src/cursus/steps/contracts/currency_conversion_contract.py)
- **Step Specification**: Specification defining inputs, outputs, and step behavior

### Related Design Documents
- **[Data Format Preservation Patterns](../1_design/data_format_preservation_patterns.md)**: Format detection and preservation strategy used in conversion
- **[Processing Step Builder Patterns](../1_design/processing_step_builder_patterns.md)**: Step builder patterns for processing steps
- **[Job Type Variant Handling](../1_design/job_type_variant_handling.md)**: Multi-variant job type support

### External References
- [Exchange Rates API](https://exchangerate.host/): Free exchange rate data source
- [ISO 4217 Currency Codes](https://en.wikipedia.org/wiki/ISO_4217): Standard currency code reference
- [Pandas Performance Optimization](https://pandas.pydata.org/docs/user_guide/enhancingperf.html): DataFrame optimization techniques
- [Python Multiprocessing](https://docs.python.org/3/library/multiprocessing.html): Parallel processing documentation

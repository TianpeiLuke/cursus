# Currency Conversion Module Documentation

## Overview
The `currency_conversion.py` script handles automated currency conversion for numerical features in datasets. It's designed to work within the SageMaker processing framework, converting monetary values from various currencies to a standard currency (typically USD) based on provided exchange rates.

## Key Features
- Automatic currency detection from marketplace IDs
- Parallel processing for efficient currency conversion
- Flexible handling of missing or invalid currency data
- Support for different job types (training, validation, testing, calibration)
- Two conversion modes: per-split or split-after-conversion

## Core Functions

### Currency Detection and Handling

- **get_currency_code(marketplace_id, marketplace_info, default_currency)**:  
  Determines the currency code for a given marketplace ID using a marketplace information dictionary.

- **combine_currency_codes(df, marketplace_id_col, currency_col, marketplace_info, default_currency, skip_invalid_currencies)**:  
  Combines currency information from marketplace IDs and an optional explicit currency column, handling any conflicts or missing values.

### Currency Conversion Logic

- **currency_conversion_single_variable(args)**:  
  Converts a single variable's currency values based on exchange rates.

- **parallel_currency_conversion(df, currency_col, currency_conversion_vars, currency_conversion_dict, n_workers)**:  
  Performs parallel currency conversion on multiple variables using a multiprocessing pool.

- **process_currency_conversion(df, marketplace_id_col, currency_conversion_vars, currency_conversion_dict, marketplace_info, currency_col, default_currency, skip_invalid_currencies, n_workers)**:  
  Main processing function that combines currency detection and conversion in a complete workflow.

### Main Execution Flow

- **main(args, currency_vars, currency_dict, marketplace_info)**:  
  Orchestrates the entire currency conversion process based on provided arguments:
  - Reads input data from SageMaker processing directories
  - Applies currency conversion using the appropriate mode
  - Handles data splitting for training workflows
  - Writes converted outputs to the expected SageMaker output locations

## Usage

The script is designed to be run as a SageMaker Processing job with the following parameters:

```bash
python currency_conversion.py \
  --job-type training \
  --mode per_split \
  --marketplace-id-col marketplace_id \
  --currency-col currency_code \
  --default-currency USD \
  --enable-conversion true
```

### Required Environment Variables
- `CURRENCY_CONVERSION_VARS`: JSON array of column names to convert
- `CURRENCY_CONVERSION_DICT`: JSON dictionary mapping currency codes to exchange rates
- `MARKETPLACE_INFO`: JSON dictionary with marketplace ID information including currency codes
- `LABEL_FIELD`: Name of the target/label column (used for stratified splitting)

### Optional Environment Variables
- `TRAIN_RATIO`: Proportion of data for training (default: 0.7)
- `TEST_VAL_RATIO`: Proportion of test data from the non-training portion (default: 0.5)

## Conversion Modes

1. **per_split**:  
   Applies currency conversion separately to each data split (train/test/val) preserving the original split ratios.

2. **split_after_conversion**:  
   Combines all data, applies conversion, then re-splits the data according to specified ratios.

## Input/Output Structure

### Input Structure
```
/opt/ml/processing/input/data/
  ├── train/
  │   ├── train_processed_data.csv
  │   └── train_full_data.csv
  ├── test/
  │   ├── test_processed_data.csv
  │   └── test_full_data.csv
  └── val/
      ├── val_processed_data.csv
      └── val_full_data.csv
```

### Output Structure
```
/opt/ml/processing/output/
  ├── train/
  │   ├── train_processed_data.csv  # With converted currency values
  │   └── train_full_data.csv       # With converted currency values
  ├── test/
  │   ├── test_processed_data.csv   # With converted currency values
  │   └── test_full_data.csv        # With converted currency values
  └── val/
      ├── val_processed_data.csv    # With converted currency values
      └── val_full_data.csv         # With converted currency values
```

## Performance Considerations
- The script uses multiprocessing for efficient parallel conversion of multiple columns
- Default worker count is 50, but can be adjusted with `--n-workers` parameter
- The actual number of workers is limited by CPU count and the number of columns to convert

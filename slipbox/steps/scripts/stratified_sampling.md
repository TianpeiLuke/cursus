# Stratified Sampling Script

This script implements stratified sampling with three core allocation strategies for handling class imbalance, causal analysis, and variance optimization.

## Overview

The `stratified_sampling.py` script follows the same format as `tabular_preprocessing.py` and can be inserted between tabular preprocessing and downstream steps without changing their interfaces.

## Features

### Three Allocation Strategies

1. **Balanced Allocation** (`balanced`)
   - **Purpose**: Handle class imbalance
   - **Method**: Equal samples per stratum/class
   - **Use case**: Machine learning with imbalanced target variables

2. **Proportional with Minimum** (`proportional_min`)
   - **Purpose**: Causal analysis and controlling for confounders
   - **Method**: Proportional allocation with minimum sample constraints
   - **Use case**: Observational studies, ensuring adequate samples for statistical power

3. **Optimal Allocation (Neyman)** (`optimal`)
   - **Purpose**: Variance optimization
   - **Method**: Allocate based on stratum size and variability
   - **Use case**: Survey sampling, precision optimization

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `STRATA_COLUMN` | Yes | - | Column name to stratify by (e.g., target variable, confounding variable) |
| `SAMPLING_STRATEGY` | No | `balanced` | One of: `balanced`, `proportional_min`, `optimal` |
| `TARGET_SAMPLE_SIZE` | No | `1000` | Total desired sample size |
| `MIN_SAMPLES_PER_STRATUM` | No | `10` | Minimum samples per stratum for statistical power |
| `VARIANCE_COLUMN` | No | - | Column for variance calculation (needed for optimal strategy) |
| `RANDOM_STATE` | No | `42` | Random seed for reproducibility |

## Job Type Behavior

### Training Job Type (`--job_type training`)
- Processes `train` and `val` splits with stratified sampling
- Copies `test` split unchanged (preserves test set integrity)
- Output: `train/`, `val/`, and `test/` directories

### Non-Training Job Types (`validation`, `testing`, `calibration`)
- Processes only the specified split
- Applies stratified sampling to that single dataset
- Output: Single directory matching the job type

## Usage Examples

### Basic Usage (Class Imbalance)
```bash
python stratified_sampling.py --job_type training
```

Environment variables:
```bash
export STRATA_COLUMN="label"
export SAMPLING_STRATEGY="balanced"
export TARGET_SAMPLE_SIZE="1000"
export MIN_SAMPLES_PER_STRATUM="20"
```

### Causal Analysis
```bash
python stratified_sampling.py --job_type training
```

Environment variables:
```bash
export STRATA_COLUMN="treatment_group"
export SAMPLING_STRATEGY="proportional_min"
export TARGET_SAMPLE_SIZE="2000"
export MIN_SAMPLES_PER_STRATUM="50"
```

### Variance Optimization
```bash
python stratified_sampling.py --job_type validation
```

Environment variables:
```bash
export STRATA_COLUMN="age_group"
export SAMPLING_STRATEGY="optimal"
export TARGET_SAMPLE_SIZE="1500"
export MIN_SAMPLES_PER_STRATUM="30"
export VARIANCE_COLUMN="income"
```

## Input/Output Structure

### Input Structure (from tabular_preprocessing)
```
/opt/ml/processing/input/data/
├── train/
│   └── train_processed_data.csv
├── val/
│   └── val_processed_data.csv
└── test/
    └── test_processed_data.csv
```

### Output Structure (maintains compatibility)
```
/opt/ml/processing/output/
├── train/
│   └── train_processed_data.csv  (sampled)
├── val/
│   └── val_processed_data.csv    (sampled)
└── test/
    └── test_processed_data.csv    (unchanged for training job_type)
```

## Integration with SageMaker Processing

The script is designed to run in SageMaker Processing containers and follows the same contract as `tabular_preprocessing.py`:

- Uses standard SageMaker paths (`/opt/ml/processing/input/data`, `/opt/ml/processing/output`)
- Maintains the same folder structure for seamless pipeline integration
- Supports the same job types and argument structure

## Testing

Run the test suite:
```bash
python test/steps/scripts/test_stratified_sampling.py
```

The test suite validates:
- All three allocation strategies
- File I/O operations
- Job type handling
- Output structure compatibility

## Key Benefits

1. **Flexible**: Three strategies cover most stratified sampling needs
2. **Compatible**: Drop-in replacement maintaining existing pipeline structure
3. **Robust**: Handles edge cases like insufficient samples per stratum
4. **Testable**: Comprehensive test suite with realistic scenarios
5. **Configurable**: Environment variables allow easy parameter tuning

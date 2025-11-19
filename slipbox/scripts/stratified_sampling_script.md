---
tags:
  - code
  - processing_script
  - sampling
  - stratified_sampling
  - data_balancing
keywords:
  - stratified sampling
  - allocation strategies
  - class imbalance
  - causal analysis
  - Neyman allocation
  - variance optimization
  - format preservation
topics:
  - sampling techniques
  - data preparation
  - statistical methods
language: python
date of note: 2025-11-18
---

# Stratified Sampling Script Documentation

## Overview

The `stratified_sampling.py` script implements sophisticated stratified sampling with three allocation strategies designed for different machine learning scenarios: balanced allocation for class imbalance, proportional allocation with minimum constraints for causal analysis, and optimal (Neyman) allocation for variance minimization. This script enables controlled dataset size reduction while maintaining statistical properties critical for model training and evaluation.

The script operates within the preprocessing pipeline, reading formatted data from upstream steps and producing sampled datasets that preserve folder structure and format conventions. It handles job-type-specific workflows, sampling train/val splits for training jobs while preserving test sets, or sampling individual splits for validation/testing jobs.

Key capabilities:
- **Three allocation strategies**: Balanced, proportional with minimum, and optimal (Neyman)
- **Class imbalance handling**: Balanced allocation ensures equal representation across strata
- **Causal analysis support**: Proportional allocation maintains representativeness with minimum sample guarantees
- **Variance optimization**: Neyman allocation minimizes sampling variance using stratum variability
- **Format preservation**: Auto-detects and maintains input format (CSV/TSV/Parquet)
- **Job-type awareness**: Different sampling behavior for training vs inference workflows
- **Test set protection**: Preserves test set unchanged for training workflows

## Purpose and Major Tasks

### Primary Purpose
Reduce dataset size through stratified sampling while maintaining statistical representativeness and ensuring adequate samples per stratum for reliable inference, supporting three distinct use cases: class imbalance mitigation, causal analysis with minimum constraints, and variance-optimized sampling.

### Major Tasks

1. **Configuration Loading**: Extract sampling parameters from environment variables including strata column, strategy, target size, and minimum samples per stratum

2. **Format Detection and Data Loading**: Auto-detect data format (CSV/TSV/Parquet) and load data from split-based directory structure

3. **Stratum Information Extraction**: Analyze data to extract stratum sizes, populations, and variance information for allocation calculations

4. **Allocation Strategy Execution**:
   - **Balanced**: Calculate equal allocation per stratum
   - **Proportional with Minimum**: Compute proportional allocation respecting minimum constraints
   - **Optimal (Neyman)**: Calculate variance-minimizing allocation using stratum variability

5. **Stratified Sampling**: Perform random sampling within each stratum according to computed allocation

6. **Job-Type-Specific Processing**:
   - Training jobs: Sample train/val splits, copy test unchanged
   - Non-training jobs: Sample only the specified split

7. **Format-Preserving Output**: Save sampled data maintaining original format and folder structure

## Script Contract

### Entry Point
```
stratified_sampling.py
```

### Input Paths

| Path | Location | Description |
|------|----------|-------------|
| `input_data` | `/opt/ml/processing/input/data` | Preprocessed data from upstream steps |

**Input Structure**:
```
/opt/ml/processing/input/data/
├── train/
│   └── train_processed_data.{csv|tsv|parquet}
├── val/
│   └── val_processed_data.{csv|tsv|parquet}
└── test/
    └── test_processed_data.{csv|tsv|parquet}
```

### Output Paths

| Path | Location | Description |
|------|----------|-------------|
| `processed_data` | `/opt/ml/processing/output` | Sampled data by split |

**Output Structure**:
```
/opt/ml/processing/output/
├── train/
│   └── train_processed_data.{format}
├── val/
│   └── val_processed_data.{format}
└── test/
    └── test_processed_data.{format}  # Copied unchanged for training jobs
```

### Required Environment Variables

| Variable | Type | Description |
|----------|------|-------------|
| `STRATA_COLUMN` | `str` | Column name to stratify by (e.g., target class, geographic region) |

### Optional Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SAMPLING_STRATEGY` | `str` | `"balanced"` | Allocation strategy: `balanced`, `proportional_min`, or `optimal` |
| `TARGET_SAMPLE_SIZE` | `int` | `1000` | Total desired sample size per split |
| `MIN_SAMPLES_PER_STRATUM` | `int` | `10` | Minimum samples per stratum for statistical power |
| `VARIANCE_COLUMN` | `str` | `None` | Column for variance calculation (required for optimal strategy) |
| `RANDOM_STATE` | `int` | `42` | Random seed for reproducibility |

### Job Arguments

| Argument | Type | Required | Description | Choices |
|----------|------|----------|-------------|---------|
| `--job_type` | `str` | Yes | Type of processing job | `training`, `validation`, `testing`, `calibration` |

### Framework Dependencies

- **pandas** >= 1.3.0 (data manipulation)
- **numpy** >= 1.21.0 (numerical operations)
- **scikit-learn** >= 1.0.0 (train_test_split utilities)

## Input Data Structure

### Expected Input Format

**Training Job**:
```
/opt/ml/processing/input/data/
├── train/train_processed_data.{csv|tsv|parquet}
├── val/val_processed_data.{csv|tsv|parquet}
└── test/test_processed_data.{csv|tsv|parquet}
```

**Non-Training Jobs** (validation/testing/calibration):
```
/opt/ml/processing/input/data/
└── {job_type}/{job_type}_processed_data.{csv|tsv|parquet}
```

### Required Columns

- **Strata Column**: Specified by `STRATA_COLUMN` environment variable
  - Can be categorical or discrete numerical
  - Used to define sampling strata
  - Examples: target class, region, customer segment
  
- **Variance Column** (for optimal strategy): Specified by `VARIANCE_COLUMN`
  - Must be numerical
  - Used to calculate stratum variance for Neyman allocation
  - Examples: transaction amount, model confidence score

### Optional Columns

All other columns are preserved unchanged in sampled data:
- Feature columns
- ID columns
- Metadata fields
- Timestamp columns

### Input Data Requirements

1. **Strata Column**: Must exist and have well-defined discrete values
2. **Cardinality**: Reasonable number of unique strata (typically 2-50)
3. **Stratum Sizes**: Each stratum should have sufficient samples (ideally > MIN_SAMPLES_PER_STRATUM)
4. **Format Consistency**: All splits must use same format (CSV, TSV, or Parquet)

## Output Data Structure

### Output Directory Structure

```
/opt/ml/processing/output/
├── train/
│   └── train_processed_data.{format}
├── val/
│   └── val_processed_data.{format}
└── test/
    └── test_processed_data.{format}
```

### Output Data Characteristics

- **Schema**: Identical to input (all columns preserved)
- **Format**: Same as input (CSV/TSV/Parquet)
- **Size**: Reduced according to TARGET_SAMPLE_SIZE and allocation strategy
- **Stratum Distribution**: Varies by strategy:
  - Balanced: Equal samples per stratum
  - Proportional: Maintains population proportions with minimums
  - Optimal: Allocated based on size × variability

**Example Sampling**:

*Input (train split with 10,000 samples)*:
```
Stratum A: 7,000 samples (70%)
Stratum B: 2,000 samples (20%)
Stratum C: 1,000 samples (10%)
```

*Output with balanced strategy (TARGET_SAMPLE_SIZE=900)*:
```
Stratum A: 300 samples (33.3%)
Stratum B: 300 samples (33.3%)
Stratum C: 300 samples (33.3%)
```

*Output with proportional_min strategy (TARGET_SAMPLE_SIZE=900, MIN=10)*:
```
Stratum A: 630 samples (70%)
Stratum B: 180 samples (20%)
Stratum C: 90 samples (10%)
```

### Job-Type-Specific Output

**Training Job**:
- Samples: train, val splits
- Copies unchanged: test split
- Rationale: Preserve test set for unbiased evaluation

**Non-Training Jobs**:
- Samples: Only the specified job_type split
- Example: `job_type=validation` samples validation split only

## Key Functions and Tasks

### Sampler Core Component

#### `StratifiedSampler` Class

**Purpose**: Core stratified sampling engine implementing three allocation strategies

**Attributes**:
- `random_state`: Random seed for reproducibility
- `strategies`: Dictionary mapping strategy names to allocation functions

#### `StratifiedSampler.sample(df, strata_column, target_size, strategy, min_samples_per_stratum, variance_column)`

**Purpose**: Main entry point for stratified sampling

**Algorithm**:
```python
1. Validate strategy name
2. Extract stratum information:
   - Count samples per stratum
   - Calculate variance per stratum (if needed)
3. Calculate allocation using selected strategy:
   - Call balanced_allocation() OR
   - Call proportional_with_min() OR  
   - Call optimal_allocation()
4. Perform sampling:
   - FOR each stratum:
     - Sample allocated number of rows
     - Use random_state for reproducibility
5. Concatenate sampled strata
6. Return sampled DataFrame
```

**Time Complexity**: O(n + k×m) where n=total rows, k=strata, m=avg samples per stratum

### Stratum Analysis Component

#### `StratifiedSampler._get_strata_info(df, strata_column, variance_column)`

**Purpose**: Extract statistical information about each stratum

**Algorithm**:
```python
1. Initialize strata_info dict
2. FOR each unique value in strata_column:
   a. Filter DataFrame to stratum
   b. Count stratum size
   c. Store stratum data
   d. IF variance_column provided AND exists:
      - Calculate variance: stratum_df[variance_column].var()
      - Calculate std dev: stratum_df[variance_column].std()
   e. ELSE:
      - Use default: variance=1.0, std=1.0
   f. Store in strata_info[stratum_value]
3. Return strata_info dict
```

**Output Structure**:
```python
{
    "stratum_A": {
        "size": 7000,
        "data": DataFrame(...),
        "variance": 150.2,
        "std": 12.3
    },
    "stratum_B": {
        "size": 2000,
        "data": DataFrame(...),
        "variance": 200.5,
        "std": 14.2
    }
}
```

### Allocation Strategy Components

#### `StratifiedSampler._balanced_allocation(strata_info, target_size, min_samples)`

**Purpose**: Balanced allocation for class imbalance handling

**Algorithm**:
```python
1. Calculate base allocation:
   samples_per_stratum = max(min_samples, target_size // num_strata)

2. First pass - equal allocation:
   FOR each stratum:
       allocated = min(samples_per_stratum, stratum_size)
       allocation[stratum] = allocated
       total_allocated += allocated

3. Distribute remaining samples:
   remaining = target_size - total_allocated
   IF remaining > 0:
      a. Calculate available capacity per stratum:
         capacity[s] = stratum_size - allocation[s]
      b. Get strata with capacity > 0
      c. extra_per_stratum = remaining // num_with_capacity
      d. FOR each stratum with capacity:
         - extra = min(extra_per_stratum, capacity[stratum])
         - allocation[stratum] += extra

4. Return allocation dict
```

**Example**:
```
Input: 3 strata (sizes: 7000, 2000, 1000), target=900, min=10
Base: 900 // 3 = 300 per stratum

Allocation:
- Stratum A: 300 (capacity: 6700 remaining)
- Stratum B: 300 (capacity: 1700 remaining)
- Stratum C: 300 (capacity: 700 remaining)

Total: 900 (target reached)
```

**Complexity**: O(k) where k=number of strata

#### `StratifiedSampler._proportional_with_min(strata_info, target_size, min_samples)`

**Purpose**: Proportional allocation maintaining representativeness with minimum constraints

**Algorithm**:
```python
1. Calculate total population:
   total_pop = sum(stratum_sizes)

2. First pass - proportional allocation with minimums:
   FOR each stratum:
       proportion = stratum_size / total_pop
       proportional_size = int(target_size * proportion)
       allocation[stratum] = max(min_samples, proportional_size)

3. Handle over-allocation:
   total_allocated = sum(allocation.values())
   IF total_allocated > target_size:
      a. Calculate excess:
         excess = total_allocated - target_size
      b. Identify adjustable strata:
         adjustable = {s: allocation[s] - min_samples 
                      for s where allocation[s] > min_samples}
      c. IF sum(adjustable) >= excess:
         - total_adjustable = sum(adjustable.values())
         - FOR each adjustable stratum:
           - reduction = int(excess * adjustable[s] / total_adjustable)
           - allocation[s] -= reduction

4. Enforce stratum size constraints:
   FOR each stratum:
       allocation[stratum] = min(allocation[stratum], stratum_size)

5. Return allocation dict
```

**Example**:
```
Input: 3 strata (sizes: 7000, 2000, 1000), target=1000, min=50
Proportions: 70%, 20%, 10%

Initial proportional:
- Stratum A: 700 (70% of 1000)
- Stratum B: 200 (20% of 1000)  
- Stratum C: 100 (10% of 1000)

After minimum constraint:
- Stratum A: 700 (already > 50)
- Stratum B: 200 (already > 50)
- Stratum C: 100 (already > 50)

Total: 1000 (no adjustment needed)
```

**Complexity**: O(k) where k=number of strata

#### `StratifiedSampler._optimal_allocation(strata_info, target_size, min_samples)`

**Purpose**: Optimal (Neyman) allocation for variance minimization

**Algorithm**:
```python
1. Calculate Neyman numerators:
   FOR each stratum:
       numerator[s] = stratum_size × stratum_std
       total_numerator += numerator[s]

2. Calculate optimal allocation:
   FOR each stratum:
       IF total_numerator > 0:
           optimal_size = int(target_size × numerator[s] / total_numerator)
       ELSE:
           optimal_size = target_size // num_strata
       
       # Apply constraints
       allocation[s] = min(
           max(min_samples, optimal_size),
           stratum_size
       )

3. Return allocation dict
```

**Mathematical Foundation (Neyman Allocation)**:

For stratum $h$ with population size $N_h$ and standard deviation $S_h$:

$$n_h = n \cdot \frac{N_h S_h}{\sum_{i=1}^{L} N_i S_i}$$

Where:
- $n_h$ = sample size for stratum $h$
- $n$ = total sample size
- $L$ = number of strata
- $N_h$ = population size of stratum $h$
- $S_h$ = standard deviation in stratum $h$

**Properties**:
- Minimizes overall sampling variance
- Allocates more samples to:
  - Larger strata (more population)
  - More variable strata (higher std dev)
- Requires variance estimation

**Example**:
```
Input: 3 strata, target=1000, min=10
Stratum A: size=7000, std=10.0 → numerator=70,000
Stratum B: size=2000, std=20.0 → numerator=40,000
Stratum C: size=1000, std=30.0 → numerator=30,000
Total numerator: 140,000

Allocation:
- Stratum A: 1000 × (70,000/140,000) = 500 samples
- Stratum B: 1000 × (40,000/140,000) = 286 samples
- Stratum C: 1000 × (30,000/140,000) = 214 samples
```

**Complexity**: O(k) where k=number of strata

### Sampling Execution Component

#### `StratifiedSampler._perform_sampling(df, strata_column, allocation)`

**Purpose**: Execute sampling based on computed allocation

**Algorithm**:
```python
1. Initialize sampled_dfs = []

2. FOR each (stratum, sample_size) in allocation:
   IF sample_size > 0:
      a. Filter to stratum: stratum_df = df[df[strata_column] == stratum]
      b. IF len(stratum_df) >= sample_size:
         - Sample: sampled = stratum_df.sample(n=sample_size, 
                                               random_state=random_state)
      c. ELSE:
         - Take all: sampled = stratum_df
      d. Append to sampled_dfs

3. IF sampled_dfs not empty:
   - Concatenate: result = pd.concat(sampled_dfs, ignore_index=True)
4. ELSE:
   - Return empty DataFrame

5. Return result
```

**Properties**:
- Uses pandas.DataFrame.sample() with fixed random_state
- Handles edge case of insufficient stratum samples
- Resets index in final DataFrame

**Complexity**: O(n) where n=total samples to draw

### Data I/O Components

#### `_detect_file_format(split_dir, split_name)`

**Purpose**: Auto-detect data file format

**Algorithm**:
```python
1. Define format priority:
   formats = [
       ("*_processed_data.csv", "csv"),
       ("*_processed_data.tsv", "tsv"),
       ("*_processed_data.parquet", "parquet")
   ]

2. FOR each (filename_pattern, format) in formats:
   a. Construct file_path = split_dir / filename_pattern
   b. IF file_path.exists():
      - RETURN (file_path, format)

3. IF no file found:
   - Raise RuntimeError with formats tried
```

**Format Detection Priority**:
1. CSV (most common)
2. TSV (tab-separated variant)
3. Parquet (compressed format)

#### `_read_processed_data(input_dir, split_name)`

**Purpose**: Load data with format preservation metadata

**Algorithm**:
```python
1. Construct split directory path
2. Detect file format and path
3. Load based on format:
   - csv: pd.read_csv(file_path)
   - tsv: pd.read_csv(file_path, sep='\t')
   - parquet: pd.read_parquet(file_path)
4. Return (DataFrame, detected_format)
```

**Returns**: Tuple of (DataFrame, format_string)

#### `_save_sampled_data(df, output_dir, split_name, output_format, logger)`

**Purpose**: Save sampled data preserving format

**Algorithm**:
```python
1. Create output directory: output_dir/split_name/
2. Construct output filename based on format:
   - csv: {split_name}_processed_data.csv
   - tsv: {split_name}_processed_data.tsv
   - parquet: {split_name}_processed_data.parquet
3. Save based on format:
   - csv: df.to_csv(path, index=False)
   - tsv: df.to_csv(path, sep='\t', index=False)
   - parquet: df.to_parquet(path, index=False)
4. Log save location, format, and shape
```

**Format Preservation**: Output format matches input format

### Main Processing Logic

#### `main(input_paths, output_paths, environ_vars, job_args, logger)`

**Purpose**: Orchestrate end-to-end stratified sampling workflow

**Algorithm**:
```python
1. Extract parameters from environ_vars and job_args:
   - strata_column (required)
   - sampling_strategy (default: balanced)
   - target_sample_size (default: 1000)
   - min_samples_per_stratum (default: 10)
   - variance_column (optional)
   - random_state (default: 42)

2. Validate required parameters:
   - strata_column must be set
   - sampling_strategy must be valid

3. Initialize sampler:
   sampler = StratifiedSampler(random_state)

4. Determine splits to process based on job_type:
   IF job_type == "training":
       splits_to_process = ["train", "val"]
   ELSE:
       splits_to_process = [job_type]

5. FOR each split in splits_to_process:
   a. Load data with format detection
   b. Validate strata_column exists
   c. Validate variance_column (if using optimal strategy)
   d. Calculate split target size = min(target_size, len(df))
   e. Perform stratified sampling
   f. Log stratum distribution
   g. Save sampled data (preserving format)
   h. Store in sampled_splits dict

6. IF job_type == "training":
   - Load test split
   - Copy test split unchanged
   - Save test split (preserving format)

7. Return sampled_splits dict
```

**Job-Type-Specific Logic**:
- **Training**: Sample train & val, copy test
- **Validation/Testing/Calibration**: Sample only that split

**Error Handling**:
- Validates strata column existence
- Warns if variance column missing (optimal strategy)
- Logs detailed progress and distributions
- Raises clear errors for configuration issues

## Algorithms and Data Structures

### Algorithm 1: Balanced Allocation with Capacity Management

**Problem**: Allocate samples equally across strata while respecting stratum sizes and target total

**Solution Strategy**:
1. Start with equal base allocation
2. Respect stratum capacity constraints
3. Distribute remaining samples to strata with capacity

**Algorithm**:
```python
# Phase 1: Equal base allocation
base_per_stratum = max(min_samples, target_size // num_strata)
FOR each stratum:
    allocation[s] = min(base_per_stratum, stratum_size[s])

# Phase 2: Distribute remainder
remaining = target_size - sum(allocation.values())
WHILE remaining > 0 AND exists stratum with capacity:
    FOR each stratum with (allocation[s] < stratum_size[s]):
        IF remaining > 0:
            allocation[s] += 1
            remaining -= 1
```

**Complexity**: O(k) where k=number of strata

**Guarantees**:
- Each stratum gets at least min_samples (if available)
- Distribution is as equal as possible
- Never exceeds stratum capacity
- Total allocation ≤ target_size

### Algorithm 2: Proportional Allocation with Minimum Guarantee

**Problem**: Maintain population proportions while ensuring statistical power (minimum samples per stratum)

**Solution Strategy**:
1. Calculate proportional allocation
2. Apply minimum constraints
3. Adjust for over-allocation by reducing from adjustable strata

**Algorithm**:
```python
# Calculate proportions
total_pop = sum(stratum_sizes)
FOR each stratum:
    proportion = stratum_size[s] / total_pop
    allocation[s] = max(min_samples, int(target_size * proportion))

# Handle over-allocation
IF sum(allocation) > target_size:
    excess = sum(allocation) - target_size
    adjustable = {s: allocation[s] - min_samples 
                  for s if allocation[s] > min_samples}
    
    # Proportionally reduce from adjustable strata
    FOR each adjustable stratum:
        reduction = excess * (allocation[s] - min_samples) / sum(adjustable)
        allocation[s] -= int(reduction)
```

**Complexity**: O(k) where k=number of strata

**Properties**:
- Maintains approximate population proportions
- Guarantees minimum samples for all strata
- Reduces large strata first when adjusting
- Suitable for causal analysis

### Algorithm 3: Neyman Optimal Allocation

**Problem**: Minimize overall sampling variance by accounting for both stratum size and variability

**Solution Strategy**: Allocate proportional to $N_h \times S_h$ (population size × standard deviation)

**Algorithm**:
```python
# Calculate allocation weights
FOR each stratum:
    weight[s] = stratum_size[s] × stratum_std[s]
total_weight = sum(weight.values())

# Allocate proportionally to weights
FOR each stratum:
    optimal_allocation[s] = int(target_size × weight[s] / total_weight)
    
    # Apply constraints
    allocation[s] = min(
        max(min_samples, optimal_allocation[s]),
        stratum_size[s]
    )
```

**Mathematical Justification**:

The Neyman allocation minimizes the variance of the stratified sample mean:

$$\text{Var}(\bar{y}_{st}) = \sum_{h=1}^{L} \left(\frac{N_h}{N}\right)^2 \frac{S_h^2}{n_h}$$

Subject to constraint $\sum_{h=1}^{L} n_h = n$

Using Lagrange multipliers, the optimal allocation is:

$$n_h^* = n \cdot \frac{N_h S_h}{\sum_{i=1}^{L} N_i S_i}$$

**Complexity**: O(k) where k=number of strata

**When to Use**:
- Variance minimization is primary goal
- Stratum variances are known or estimable
- Target variable or proxy available
- Examples: survey sampling, A/B test sample selection

### Data Structure 1: Stratum Information Dictionary

```python
strata_info: Dict[Any, Dict[str, Any]] = {
    "stratum_value_1": {
        "size": 7000,              # Number of samples in stratum
        "data": DataFrame(...),     # Stratum data subset
        "variance": 150.2,          # Variance of variance_column
        "std": 12.26               # Std dev of variance_column
    },
    "stratum_value_2": {
        "size": 2000,
        "data": DataFrame(...),
        "variance": 200.5,
        "std": 14.16
    }
}
```

**Properties**:
- Keys: Unique values from strata_column
- Nested dictionaries with statistical metadata
- Contains full data subsets for sampling
- Variance/std only computed if variance_column provided

**Memory Considerations**:
- Stores full DataFrames per stratum
- Memory = O(total_rows + k×overhead) where k=strata
- For large datasets, consider streaming approaches

### Data Structure 2: Allocation Dictionary

```python
allocation: Dict[Any, int] = {
    "stratum_A": 300,  # Number of samples to draw
    "stratum_B": 200,
    "stratum_C": 100
}
```

**Properties**:
- Keys: Stratum values (same as strata_info)
- Values: Integer sample counts
- Sum of values ≤ target_sample_size
- Each value ≤ corresponding stratum size

**Generation**:
- Produced by allocation strategy functions
- Used by sampling execution function
- Represents sampling plan

## Performance Characteristics

### Processing Time

| Operation | Typical Time (10K samples, 5 strata, target=1K) |
|-----------|--------------------------------------------------|
| Parameter extraction | <0.01 seconds |
| Data loading (CSV) | 0.2-0.5 seconds |
| Data loading (Parquet) | 0.1-0.3 seconds |
| Stratum info extraction | 0.1-0.2 seconds |
| Balanced allocation | <0.01 seconds |
| Proportional allocation | <0.01 seconds |
| Optimal allocation | 0.01-0.02 seconds (includes variance calc) |
| Sampling execution | 0.1-0.2 seconds |
| Data saving (CSV) | 0.2-0.4 seconds |
| Data saving (Parquet) | 0.1-0.2 seconds |
| **Total (per split)** | **0.7-1.5 seconds** |
| **Training job (3 splits)** | **2-4 seconds** |

### Memory Usage

**Memory Profile**:
```
Input data: n × d × 8 bytes (float64 DataFrame)
Stratum info: k × (n/k) × d × 8 bytes (strata subsets)
  = n × d × 8 bytes (same as input, references)
Sampled data: m × d × 8 bytes (m=target_sample_size)
Peak: ~2× input data + sampled data

Example (10K samples, 50 features, sample to 1K):
- Input: 10K × 50 × 8 = 4 MB
- Stratum subsets: ~4 MB (references)
- Sampled: 1K × 50 × 8 = 0.4 MB  
- Peak: ~8-9 MB
```

**Scalability**:

| Input Size | Target Size | Memory (est.) | Time (est.) |
|------------|-------------|---------------|-------------|
| 1K samples | 500 | ~1 MB | 0.5 seconds |
| 10K samples | 1K | ~8 MB | 1.5 seconds |
| 100K samples | 5K | ~80 MB | 10 seconds |
| 1M samples | 10K | ~800 MB | 1-2 minutes |

**Recommendations**:
- For datasets > 1M samples: Use Parquet format
- For memory constraints: Process splits sequentially  
- For speed: Balanced allocation is fastest (no variance calculation)

### Strategy-Specific Performance

| Strategy | Computation Overhead | Memory Overhead | Best For |
|----------|---------------------|-----------------|----------|
| Balanced | O(k) - minimal | None | Fast execution, equal representation |
| Proportional | O(k) - minimal | None | Maintain proportions with minimums |
| Optimal | O(n×k) - variance calc | None | Variance minimization priority |

Where k=number of strata, n=total samples

## Error Handling

### Error Types

#### Configuration Errors

**Missing Required Environment Variable**:
- **Cause**: `STRATA_COLUMN` not set
- **Handling**: Raises `RuntimeError` immediately
- **Response**: Script exits with code 1, clear error message

**Invalid Sampling Strategy**:
- **Cause**: `SAMPLING_STRATEGY` not in ['balanced', 'proportional_min', 'optimal']
- **Handling**: Raises `RuntimeError` with valid options listed
- **Response**: Script exits with code 1

#### Data Validation Errors

**Strata Column Not Found**:
- **Cause**: Column specified in `STRATA_COLUMN` doesn't exist in DataFrame
- **Handling**: Raises `RuntimeError` listing available columns
- **Response**: Clear error showing column names in data

**Variance Column Not Found** (optimal strategy):
- **Cause**: `VARIANCE_COLUMN` specified but doesn't exist in data
- **Handling**: Logs warning, uses default variance (1.0)
- **Response**: Processing continues with equal variance assumption

**Format Detection Failure**:
- **Cause**: No processed data file found in expected location
- **Handling**: Raises `RuntimeError` listing attempted formats
- **Response**: Clear error showing directory and expected files

#### Processing Errors

**Insufficient Samples in Stratum**:
- **Cause**: Stratum has fewer samples than requested allocation
- **Handling**: Takes all available samples from that stratum
- **Response**: Warning logged, processing continues

**Empty Result After Sampling**:
- **Cause**: All strata exhausted or allocation zero
- **Handling**: Returns empty DataFrame
- **Response**: Warning logged, may cause downstream issues

### Error Response Structure

**Exit Codes**:
- **0**: Success
- **1**: Runtime/Configuration error
- **3**: Unexpected exception

**Error Message Format**:
```
ERROR: [Error description]
[Detailed context including available options/columns]
[Stack trace if unexpected]
```

**Example Error Messages**:
```
ERROR: STRATA_COLUMN environment variable must be set.

ERROR: Invalid SAMPLING_STRATEGY: unknown. Must be one of: balanced, proportional_min, optimal

ERROR: Strata column 'label' not found in train data. Available columns: ['id', 'feature1', 'feature2', 'target']

WARNING: Variance column 'amount' not found. Using default variance for optimal allocation.
```

## Best Practices

### For Production Deployments

1. **Choose Appropriate Strategy**
   - **Balanced**: Use for severe class imbalance (e.g., fraud detection with 1% positive class)
   - **Proportional with Minimum**: Use when maintaining data distribution is important but need statistical power per stratum
   - **Optimal**: Use when variance reduction is critical and you have a good variance proxy

2. **Set Reasonable Target Size**
   - Consider computational constraints of downstream steps
   - Ensure sufficient samples for statistical power
   - Typical range: 1,000-10,000 samples depending on use case

3. **Configure Minimum Samples Per Stratum**
   - Default of 10 is reasonable for most cases
   - Increase for complex models requiring more data per category
   - Consider statistical power requirements (e.g., 30+ for central limit theorem)

4. **Validate Stratum Definitions**
   - Ensure strata are well-defined and meaningful
   - Check for reasonable number of strata (2-50 typical)
   - Verify no stratum is too small relative to minimum

5. **Monitor Stratum Distributions**
   - Review logged stratum counts after sampling
   - Ensure critical minority classes are represented
   - Verify allocation matches strategy expectations

### For Development

1. **Start with Balanced Strategy**
   - Simplest and fastest for initial testing
   - Reveals class imbalance issues quickly
   - Good baseline for comparison

2. **Test with Small Samples First**
   - Use small TARGET_SAMPLE_SIZE to validate logic
   - Check stratum distributions in output
   - Verify format preservation works

3. **Experiment with Strategies**
   - Compare balanced vs proportional for your data
   - Test optimal if you have variance proxy
   - Measure impact on downstream model performance

4. **Verify Test Set Preservation**
   - For training jobs, confirm test split unchanged
   - Check test set size matches input
   - Validate no sampling applied to test

### For Performance Optimization

1. **Use Parquet Format**
   - 40-60% faster I/O vs CSV
   - Smaller file sizes
   - Better for large datasets

2. **Optimize Target Size**
   - Larger samples = slower processing
   - Find minimum viable sample size for your use case
   - Balance statistical power vs speed

3. **Consider Stratum Count**
   - Many strata (>50) may slow processing
   - Consider grouping rare strata
   - Monitor memory with high cardinality

## Example Configurations

### Example 1: Balanced Sampling for Class Imbalance
```bash
export STRATA_COLUMN="fraud_label"
export SAMPLING_STRATEGY="balanced"
export TARGET_SAMPLE_SIZE="1000"
export MIN_SAMPLES_PER_STRATUM="100"
export RANDOM_STATE="42"

python stratified_sampling.py --job_type training
```

**Use Case**: Fraud detection with 99% negative, 1% positive classes
**Result**: 500 negative, 500 positive samples (equal representation)

### Example 2: Proportional Sampling with Minimums
```bash
export STRATA_COLUMN="region"
export SAMPLING_STRATEGY="proportional_min"
export TARGET_SAMPLE_SIZE="5000"
export MIN_SAMPLES_PER_STRATUM="50"
export RANDOM_STATE="42"

python stratified_sampling.py --job_type training
```

**Use Case**: Regional analysis with varying region sizes but need minimum per region
**Result**: Maintains regional proportions while ensuring 50+ samples per region

### Example 3: Optimal Sampling for Variance Reduction
```bash
export STRATA_COLUMN="customer_segment"
export SAMPLING_STRATEGY="optimal"
export TARGET_SAMPLE_SIZE="2000"
export MIN_SAMPLES_PER_STRATUM="20"
export VARIANCE_COLUMN="transaction_amount"
export RANDOM_STATE="42"

python stratified_sampling.py --job_type training
```

**Use Case**: Transaction analysis where some segments have high variance
**Result**: More samples allocated to high-variance segments

### Example 4: Validation Split Sampling
```bash
export STRATA_COLUMN="product_category"
export SAMPLING_STRATEGY="balanced"
export TARGET_SAMPLE_SIZE="500"
export MIN_SAMPLES_PER_STRATUM="10"

python stratified_sampling.py --job_type validation
```

**Use Case**: Sample validation split only for quick evaluation
**Result**: 500 samples from validation split with equal category representation

## Integration Patterns

### Upstream Integration

```
TabularPreprocessing
   ↓ (train/val/test splits with features)
RiskTableMapping (optional)
   ↓ (risk-mapped categorical features)
StratifiedSampling
   ↓ (sampled train/val, unchanged test)
```

**Data Flow**:
1. TabularPreprocessing creates initial splits
2. Optional feature engineering steps
3. StratifiedSampling reduces dataset size
4. Preserves format and structure for downstream steps

### Downstream Integration

```
StratifiedSampling
   ↓ (sampled data maintaining structure)
   ├→ XGBoostTraining (uses sampled training data)
   ├→ LightGBMTraining (uses sampled training data)
   └→ PyTorchTraining (uses sampled training data)
```

**Benefits**:
- Faster training on reduced datasets
- Better class balance (balanced strategy)
- Maintains statistical properties
- Same folder structure as full pipeline

### Complete Pipeline Example

```
1. TabularPreprocessing (job_type=training)
   → train/val/test splits (10K/2K/2K samples)

2. MissingValueImputation (job_type=training)
   → Imputed splits (same sizes)

3. RiskTableMapping (job_type=training)
   → Risk-mapped splits (same sizes)

4. StratifiedSampling (job_type=training)
   → train/val/test (1K/200/2K samples)
   Note: test unchanged, train/val sampled

5. XGBoostTraining
   → Trains on 1K samples (faster iteration)
```

## Troubleshooting

### Issue 1: Unbalanced Output Despite Balanced Strategy

**Symptom**:
Output stratum distribution not equal (e.g., 400/300/300 instead of 333/333/334)

**Common Causes**:
1. Stratum too small relative to target allocation
2. Capacity constraints preventing equal distribution
3. Rounding errors with indivisible target sizes

**Solutions**:
1. Check stratum sizes: `df[STRATA_COLUMN].value_counts()`
2. Reduce TARGET_SAMPLE_SIZE or MIN_SAMPLES_PER_STRATUM
3. Increase data collection for small strata
4. Use proportional_min if true proportions acceptable

### Issue 2: Over-Sampling Small Strata with Proportional Strategy

**Symptom**:
Small strata getting MIN_SAMPLES_PER_STRATUM despite low proportion

**Common Causes**:
1. MIN_SAMPLES_PER_STRATUM too high relative to target size
2. Many small strata consuming allocation budget

**Solutions**:
1. Lower MIN_SAMPLES_PER_STRATUM (e.g., from 50 to 10)
2. Group rare strata into "other" category upstream
3. Use balanced strategy if equal representation desired
4. Increase TARGET_SAMPLE_SIZE to accommodate minimums

### Issue 3: All Strata Get Default Variance in Optimal Strategy

**Symptom**:
Optimal allocation behaves like proportional allocation

**Common Causes**:
1. VARIANCE_COLUMN not set or not found
2. Variance column has constant values
3. Variance calculation failing silently

**Solutions**:
1. Verify VARIANCE_COLUMN environment variable set correctly
2. Check column exists: `VARIANCE_COLUMN in df.columns`
3. Verify column has variation: `df[VARIANCE_COLUMN].var() > 0`
4. Use different proxy for variance (e.g., prediction confidence)

### Issue 4: Test Set Modified in Training Job

**Symptom**:
Test set size changed after stratified sampling

**Common Causes**:
1. Bug in script (unlikely, but check version)
2. Upstream step already sampled test
3. Misunderstanding of job_type behavior

**Solutions**:
1. Verify job_type=training (not testing)
2. Check test set size before and after:
   ```python
   # Before: ls /opt/ml/processing/input/data/test/
   # After: ls /opt/ml/processing/output/test/
   ```
3. Review script logs for test set handling messages
4. Confirm script copies test unchanged for training jobs

### Issue 5: Empty Output After Sampling

**Symptom**:
Sampled DataFrame is empty or has very few rows

**Common Causes**:
1. TARGET_SAMPLE_SIZE = 0 or very small
2. All strata smaller than MIN_SAMPLES_PER_STRATUM
3. Allocation calculation error

**Solutions**:
1. Check TARGET_SAMPLE_SIZE environment variable
2. Verify stratum sizes: `df.groupby(STRATA_COLUMN).size()`
3. Lower MIN_SAMPLES_PER_STRATUM to match data availability
4. Check logs for allocation dictionary

## References

### Related Scripts

- **[`tabular_preprocessing.py`](./tabular_preprocessing_script.md)**: Upstream preprocessing script
- **[`missing_value_imputation.py`](./missing_value_imputation_script.md)**: Imputation before sampling
- **[`risk_table_mapping.py`](./risk_table_mapping_script.md)**: Risk mapping before sampling
- **[`xgboost_training.py`](./xgboost_training_script.md)**: Downstream training on sampled data

### Related Documentation

- **Contract**: `src/cursus/steps/contracts/stratified_sampling_contract.py`
- **Step Specification**: StratifiedSampling step specification
- **Config Class**: `src/cursus/steps/configs/config_stratified_sampling_step.py`
- **Builder**: `src/cursus/steps/builders/builder_stratified_sampling_step.py`

### Related Design Documents

- **[Data Format Preservation Patterns](../1_design/data_format_preservation_patterns.md)**: Format detection and preservation strategy
- **[Processing Step Builder Patterns](../1_design/processing_step_builder_patterns.md)**: General processing step architecture
- **[Job Type Variant Handling](../1_design/job_type_variant_handling.md)**: Multi-variant processing patterns

### External References

- **Stratified Sampling**: https://en.wikipedia.org/wiki/Stratified_sampling
- **Neyman Allocation**: https://en.wikipedia.org/wiki/Stratified_sampling#Allocation_of_sample_sizes
- **Class Imbalance**: https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
- **Pandas Sampling**: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html

---

## Document Metadata

**Author**: Cursus Framework Team  
**Last Updated**: 2025-11-18  
**Script Version**: 2025-11-18  
**Documentation Version**: 1.0  
**Review Status**: Complete

**Change Log**:
- 2025-11-18: Initial comprehensive documentation created
- 2025-11-18: Stratified sampling script documented with three allocation strategies

**Related Scripts**: 
- Upstream: `tabular_preprocessing.py`, `missing_value_imputation.py`, `risk_table_mapping.py`
- Downstream: `xgboost_training.py`, `lightgbm_training.py`, `pytorch_training.py`
- Related: Data preparation and sampling scripts

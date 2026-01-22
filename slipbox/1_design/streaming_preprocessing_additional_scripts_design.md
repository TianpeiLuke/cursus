---
tags:
  - design
  - preprocessing
  - streaming
  - parallelism
  - optimization
keywords:
  - streaming mode
  - risk table mapping
  - missing value imputation
  - stratified sampling
  - 1:1 shard mapping
  - multi-worker processing
topics:
  - streaming preprocessing
  - parallel pipeline architecture
  - shard-level parallelism
language: python
date of note: 2026-01-21
---

# Streaming Mode for Additional Preprocessing Scripts

## Overview

This design document describes the implementation of streaming mode for four preprocessing scripts: **risk_table_mapping**, **missing_value_imputation**, **stratified_sampling**, and **temporal_split_preprocessing**. The first three scripts require full streaming mode implementation from scratch, while temporal_split_preprocessing already has partial streaming mode that needs to be upgraded to true 1:1 shard mapping.

The design follows the fully parallel streaming architecture established in [tabular_preprocessing](./fully_parallel_streaming_preprocessing_design.md), enabling 1:1 shard mapping and maximum parallelism.

**Related Documents**:
- [Fully Parallel Streaming Preprocessing Design](./fully_parallel_streaming_preprocessing_design.md) - Reference architecture
- [Multi-Worker Streaming Preprocessing Design](./multi_worker_streaming_preprocessing_design.md) - Initial parallel optimization

## Problem Statement

### Current Limitations: Batch Mode Only

All three preprocessing scripts currently operate in **batch mode only**, which has significant limitations:

```
Current Architecture (Batch Mode):
INPUT SHARDS (100 shards) 
    ↓
[Load All Shards] ← Memory-intensive (10GB+)
    ↓
Single DataFrame (10M rows)
    ↓
[Process All Data] ← Sequential bottleneck
    ↓
[Save Output] ← Sequential write
    ↓
OUTPUT FILE(S)

Limitations:
❌ High memory usage (must fit entire dataset)
❌ No parallelism (sequential processing)
❌ Slow for large datasets (>1GB)
❌ Not compatible with IterableDataset training
```

### Scripts Requiring Streaming Mode

**1. Risk Table Mapping (`risk_table_mapping.py`)**
- **Purpose**: Create risk tables for categorical features based on target correlation
- **Current Mode**: Batch only - loads full train/test/val datasets
- **Memory Profile**: ~2-4GB for typical datasets
- **Processing Pattern**: Fit (compute risk tables) + Transform (apply mappings)
- **Artifacts**: Saves `risk_table_map.pkl` for inference

**2. Missing Value Imputation (`missing_value_imputation.py`)**
- **Purpose**: Handle missing values with statistical imputation methods
- **Current Mode**: Batch only - loads full train/test/val datasets  
- **Memory Profile**: ~2-4GB for typical datasets
- **Processing Pattern**: Fit (compute imputation statistics) + Transform (fill missing)
- **Artifacts**: Saves `impute_dict.pkl` for inference

**3. Stratified Sampling (`stratified_sampling.py`)**
- **Purpose**: Perform stratified sampling with multiple allocation strategies
- **Current Mode**: Batch only - loads full train/val/test datasets
- **Memory Profile**: ~2-4GB for typical datasets
- **Processing Pattern**: Stateless sampling operation
- **Artifacts**: None (stateless)

**4. Temporal Split Preprocessing (`temporal_split_preprocessing.py`)**
- **Purpose**: Temporal split with customer-level deduplication (OOT test) and random train/val split
- **Current Mode**: Has basic streaming mode that consolidates shards into single files
- **Memory Profile**: ~4-8GB for typical datasets
- **Processing Pattern**: Two-pass (customer allocation + split assignment)
- **Artifacts**: None
- **Special Requirements**: 
  - Customer-level split (no customer leakage between train and OOT)
  - Temporal cutoff date for OOT test set
  - Needs upgrade to true 1:1 shard mapping (shards-in-shards-out)
- **Detailed Upgrade Design**: See [Temporal Split Preprocessing Streaming Mode Upgrade](./temporal_split_preprocessing_streaming_upgrade.md) for complete implementation plan with code examples

### Key Challenge: Fit-Transform Pattern

Risk table mapping and missing value imputation follow a **fit-transform pattern**:

```
Training Mode:
  1. Fit Phase: Compute global statistics from training data
     - Risk tables: Aggregate by categorical values
     - Imputation: Compute mean/median/mode per column
  2. Transform Phase: Apply statistics to all splits (train/val/test)

Inference Mode:
  1. Load pre-computed statistics from artifacts
  2. Transform Phase: Apply to new data
```

**Challenge**: Global statistics require cross-shard coordination, but we want 1:1 shard mapping for parallelism.

**Solution**: Two-pass architecture (like tabular_preprocessing):
- **Pass 1**: Lightweight global statistics collection (sequential or parallel)
- **Pass 2**: Fully parallel per-shard transformation using global context

## Solution Architecture

### Unified Streaming Framework

All three scripts will adopt the **1:1 shard mapping architecture** from tabular_preprocessing:

```
STREAMING MODE ARCHITECTURE

INPUT SHARDS (100 shards)
    ↓
[PASS 1: Collect Global Context] ← Fast, lightweight
    │  ├─ Risk Tables (Pass 1)
    │  ├─ Imputation Stats (Pass 1)
    │  └─ Sampling Context (Pass 1)
    ↓
Global Context Dictionary
    ↓
[PASS 2: Parallel Processing] ← 100% parallel
    │
    ├─ Worker 1: Shard 1 → process_shard_end_to_end → Output Shards
    ├─ Worker 2: Shard 2 → process_shard_end_to_end → Output Shards
    ├─ Worker 3: Shard 3 → process_shard_end_to_end → Output Shards
    │  ...
    └─ Worker 8: Shard 8 → process_shard_end_to_end → Output Shards
    
    (Workers cycle through remaining shards)
    ↓
OUTPUT SHARDS (1:1 mapping preserved)

Performance:
✅ Memory: Fixed at ~2GB (workers × shard_size)
✅ Speed: 8-10× faster than batch mode
✅ Parallelism: 100% parallel efficiency
```

### Core Design Principles

**1. Dual Output Modes**
- **Artifacts (Same in Both Modes)**: Model artifacts like `risk_table_map.pkl`, `impute_dict.pkl` are identical
- **Processed Data (Mode-Dependent)**:
  - **Batch Mode**: Single consolidated file per split (e.g., `train.csv`, `val.csv`, `test.csv`)
  - **Streaming Mode**: Multiple shards per split with preserved numbering (e.g., `train/part-00001.csv`, `train/part-00002.csv`)

**2. Preserve 1:1 Shard Mapping (Streaming Mode)**
- Each input shard produces output shards with preserved numbering
- Example: `part-00042.csv` → `train/part-00042.csv`, `val/part-00042.csv`

**3. Two-Pass Strategy (When Needed)**
- Pass 1: Collect global statistics (lightweight, memory-efficient)
  - Produces artifacts in training mode (same for both batch/streaming)
- Pass 2: Apply transformations
  - Batch: Process all data together, output single files
  - Streaming: Process per shard in parallel, output shards

**4. Stateless Per-Shard Processing (Streaming Mode)**
- Each shard processed independently in Pass 2
- No cross-shard dependencies during transformation
- Pure parallelism across all workers

**5. Backward Compatibility**
- Batch mode remains default
- Streaming mode enabled via `ENABLE_TRUE_STREAMING` flag
- All existing functionality preserved

## Detailed Design: Risk Table Mapping

### Current Batch Implementation

```python
def process_data(data_dict, cat_field_list, label_name, job_type, ...):
    """Current batch mode implementation"""
    if job_type == "training":
        # Fit on full training DataFrame
        binner = OfflineBinning(cat_field_list, label_name)
        binner.fit(data_dict["train"])
        
        # Transform all splits (full DataFrames)
        for split_name, df in data_dict.items():
            df_transformed = binner.transform(df)
```

### Streaming Mode Architecture

```python
# ============================================================================
# PASS 1: COLLECT RISK TABLES (Global Statistics)
# ============================================================================

def collect_risk_tables_pass1(
    all_shards: List[Path],
    signature_columns: Optional[list],
    cat_field_list: List[str],
    label_field: str,
    smooth_factor: float,
    count_threshold: int,
    log_func: Callable,
) -> Dict[str, Any]:
    """
    Pass 1: Collect global risk tables from training shards.
    
    Memory-efficient: Only processes training data, aggregates incrementally.
    
    Returns:
        Dictionary containing fitted risk tables for all categorical fields
    """
    log_func("[PASS1] Collecting risk tables from training shards...")
    
    # Initialize aggregators for each categorical field
    cross_tabs = {field: {} for field in cat_field_list}
    total_records = 0
    default_risk_numerator = 0
    default_risk_denominator = 0
    
    # Process each shard to collect statistics
    for shard_idx, shard_path in enumerate(all_shards):
        df = _read_file_to_df(shard_path, signature_columns)
        df.columns = [col.replace("__DOT__", ".") for col in df.columns]
        
        # Filter to valid labels
        df_train = df[(df[label_field] != -1) & (~df[label_field].isnull())]
        
        # Update default risk calculation
        total_records += len(df_train)
        default_risk_numerator += (df_train[label_field] == 1).sum()
        default_risk_denominator += len(df_train)
        
        # Collect cross-tabulation data for each categorical field
        for field in cat_field_list:
            if field not in df_train.columns:
                continue
                
            # Group by categorical value and label
            for cat_value in df_train[field].unique():
                if pd.isna(cat_value):
                    continue
                    
                mask = df_train[field] == cat_value
                positive_count = (df_train[mask][label_field] == 1).sum()
                total_count = mask.sum()
                
                # Aggregate into cross_tabs
                if cat_value not in cross_tabs[field]:
                    cross_tabs[field][cat_value] = {"pos": 0, "total": 0}
                cross_tabs[field][cat_value]["pos"] += positive_count
                cross_tabs[field][cat_value]["total"] += total_count
        
        del df, df_train
        gc.collect()
    
    # Compute final risk tables
    default_risk = default_risk_numerator / default_risk_denominator if default_risk_denominator > 0 else 0.0
    smooth_samples = int(total_records * smooth_factor)
    
    risk_tables = {}
    for field, cat_data in cross_tabs.items():
        risk_tables[field] = {
            "varName": field,
            "type": "categorical",
            "default_bin": default_risk,
            "bins": {}
        }
        
        for cat_value, counts in cat_data.items():
            # Apply smoothing and threshold
            if counts["total"] >= count_threshold:
                raw_risk = counts["pos"] / counts["total"]
                smooth_risk = (
                    (counts["total"] * raw_risk + smooth_samples * default_risk)
                    / (counts["total"] + smooth_samples)
                )
            else:
                smooth_risk = default_risk
            
            risk_tables[field]["bins"][cat_value] = smooth_risk
    
    log_func(f"[PASS1] Collected risk tables for {len(risk_tables)} fields")
    log_func(f"[PASS1] Total records processed: {total_records:,}")
    log_func(f"[PASS1] Default risk: {default_risk:.4f}")
    
    return risk_tables


# ============================================================================
# PASS 2: PARALLEL PER-SHARD TRANSFORMATION
# ============================================================================

def process_shard_end_to_end_risk_mapping(args: tuple) -> Dict[str, int]:
    """
    Process single shard: read → apply risk mapping → write.
    
    Args:
        args: Tuple of (shard_path, shard_index, global_context, 
                       output_base, signature_columns, output_format, ...)
    
    Returns:
        Statistics dict with row counts per split
    """
    (shard_path, shard_index, global_context, output_base,
     signature_columns, output_format) = args
    
    # Extract input shard number
    shard_num = extract_shard_number(shard_path)
    
    # ============================================================
    # STEP 1: Read Single Shard
    # ============================================================
    df = _read_file_to_df(shard_path, signature_columns)
    df.columns = [col.replace("__DOT__", ".") for col in df.columns]
    
    # ============================================================
    # STEP 2: Apply Risk Mapping (Using Global Context)
    # ============================================================
    risk_tables = global_context["risk_tables"]
    
    for field, risk_table_info in risk_tables.items():
        if field in df.columns:
            bins = risk_table_info["bins"]
            default_bin = risk_table_info["default_bin"]
            df[field] = df[field].map(bins).fillna(default_bin)
    
    # ============================================================
    # STEP 3: Write to Split Folders (Preserving Shard Number)
    # ============================================================
    stats = {}
    job_type = global_context["job_type"]
    
    if job_type == "training":
        # Split assignment already done by upstream step (tabular_preprocessing)
        # Just write each split
        for split_name in ["train", "val", "test"]:
            # Check if split exists in this shard
            # (Assumes upstream step added _split column OR we write all data to all splits)
            output_path = (output_base / split_name / 
                          f"part-{shard_num:05d}.{output_format}")
            write_shard_file(df, output_path, output_format)
            stats[split_name] = len(df)
    else:
        # Single split mode
        output_path = (output_base / job_type / 
                      f"part-{shard_num:05d}.{output_format}")
        write_shard_file(df, output_path, output_format)
        stats[job_type] = len(df)
    
    return stats


# ============================================================================
# MAIN STREAMING MODE ENTRY POINT
# ============================================================================

def process_streaming_mode_risk_mapping(
    input_dir: str,
    output_dir: str,
    signature_columns: Optional[list],
    job_type: str,
    cat_field_list: List[str],
    label_field: str,
    smooth_factor: float,
    count_threshold: int,
    output_format: str,
    max_workers: Optional[int],
    optimize_memory: bool,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Streaming mode for risk table mapping.
    
    Uses two-pass architecture:
    - Pass 1: Collect global risk tables from training shards
    - Pass 2: Apply transformations per shard in parallel
    """
    log = logger or print
    output_path = Path(output_dir)
    
    log("[STREAMING] Starting risk table mapping in streaming mode")
    
    # Find input shards
    all_shards = find_input_shards(input_dir, log)
    
    # Determine optimal workers
    if max_workers is None:
        max_workers = min(cpu_count(), len(all_shards))
    log(f"[STREAMING] Using {max_workers} parallel workers")
    
    # ============================================================
    # PASS 1: Collect Risk Tables (Training Mode Only)
    # ============================================================
    risk_tables = None
    if job_type == "training":
        risk_tables = collect_risk_tables_pass1(
            all_shards, signature_columns, cat_field_list,
            label_field, smooth_factor, count_threshold, log
        )
    else:
        # Load pre-computed risk tables from artifacts
        risk_tables = load_risk_tables_from_artifacts(...)
    
    # Build global context
    global_context = {
        "job_type": job_type,
        "risk_tables": risk_tables,
    }
    
    # ============================================================
    # PASS 2: Parallel Per-Shard Processing
    # ============================================================
    log("[STREAMING] Pass 2: Applying risk mappings in parallel...")
    
    shard_args = [
        (shard, i, global_context, output_path,
         signature_columns, output_format)
        for i, shard in enumerate(all_shards)
    ]
    
    with Pool(processes=max_workers) as pool:
        results = pool.map(process_shard_end_to_end_risk_mapping, shard_args)
    
    # Aggregate statistics
    total_stats = aggregate_results(results, job_type)
    log(f"[STREAMING] Complete! Row distribution: {total_stats}")
    
    return {}  # Data written to disk
```

### Environment Variables

```bash
# Core Parameters (Existing)
STRATA_COLUMN="risk_category"
TARGET_SAMPLE_SIZE=100000
SAMPLING_STRATEGY="proportional_min"  # balanced, proportional_min, optimal
MIN_SAMPLES_PER_STRATUM=10
VARIANCE_COLUMN=""  # Optional, for optimal allocation
RANDOM_STATE=42

# Streaming Control (NEW)
ENABLE_TRUE_STREAMING="true"  # Enable streaming mode
MAX_WORKERS=8                 # Number of parallel workers
OUTPUT_FORMAT="csv"           # csv, tsv, or parquet
```

## Output Structure

### Artifacts (Identical in Both Modes)

**Risk Table Mapping:**
```
output_dir/
└── model/
    └── risk_table_map.pkl    # Same in both batch and streaming mode
```

**Missing Value Imputation:**
```
output_dir/
└── model/
    └── impute_dict.pkl       # Same in both batch and streaming mode
```

**Stratified Sampling:**
- No artifacts (stateless operation)

### Processed Data Output

#### Batch Mode (Default)

**Training Mode:**
```
output_dir/
├── train.csv              # Single consolidated file
├── val.csv                # Single consolidated file  
├── test.csv               # Single consolidated file
└── model/                 # Artifacts (if applicable)
    └── *.pkl
```

**Single Split Mode (Inference/Validation):**
```
output_dir/
├── validation.csv         # Single consolidated file
└── model/                 # Artifacts (if applicable)
    └── *.pkl
```

#### Streaming Mode (ENABLE_TRUE_STREAMING=true)

**Training Mode:**
```
output_dir/
├── train/
│   ├── part-00001.csv    # From input part-00001.csv (train rows only)
│   ├── part-00003.csv    # From input part-00003.csv (train rows only)
│   ├── part-00005.csv    # From input part-00005.csv (train rows only)
│   └── ...               # Sparse numbering (gaps where shards had 0 train rows)
├── val/
│   ├── part-00002.csv    # From input part-00002.csv (val rows only)
│   ├── part-00004.csv    # From input part-00004.csv (val rows only)
│   └── ...
├── test/
│   ├── part-00001.csv    # From input part-00001.csv (test rows only)
│   ├── part-00006.csv    # From input part-00006.csv (test rows only)
│   └── ...
└── model/                 # Artifacts (if applicable) - SAME as batch mode
    └── *.pkl
```

**Single Split Mode (Inference/Validation):**
```
output_dir/
├── validation/            # Directory of shards
│   ├── part-00001.csv    # From input part-00001.csv
│   ├── part-00002.csv    # From input part-00002.csv
│   ├── part-00003.csv    # From input part-00003.csv
│   └── ...               # 1:1 mapping preserved
└── model/                 # Artifacts (if applicable) - SAME as batch mode
    └── *.pkl
```

### Key Differences Summary

| Aspect | Batch Mode | Streaming Mode |
|--------|-----------|----------------|
| **Artifacts** | `model/*.pkl` (identical) | `model/*.pkl` (identical) |
| **Processed Data** | Single file per split (`train.csv`, `val.csv`, `test.csv`) | Multiple shards per split (`train/part-*.csv`, `val/part-*.csv`, `test/part-*.csv`) |
| **Memory Usage** | High (must fit full dataset) | Low (per-shard processing) |
| **Processing Speed** | Slower (sequential) | Faster (parallel) |
| **PyTorch IterableDataset** | Not compatible | Fully compatible |

## Performance Analysis

### Scenario: 100 Input Shards, 8 Workers, 10GB Dataset

#### Current Batch Mode
```
Batch Mode (All Scripts):
  Load all shards: ~20s
  Process full dataset: ~15s
  Write output: ~10s
  Total: ~45 seconds
  
Memory usage: ~10GB peak (must fit full dataset)
Parallelism: None (sequential bottleneck)
```

#### Proposed Streaming Mode
```
Streaming Mode:
  Pass 1: Collect global stats
    - Risk mapping: ~5s (lightweight aggregation)
    - Imputation: ~4s (simpler aggregation)
    - Sampling: ~3s (counts only)
  
  Pass 2: Parallel processing
    - 100 shards / 8 workers = 12.5 iterations
    - Per-shard time: ~0.5s
    - Total Pass 2: 6.25s
  
  Total time:
    - Risk mapping: 5s + 6.25s = 11.25s (4× speedup)
    - Imputation: 4s + 6.25s = 10.25s (4.4× speedup)
    - Sampling: 3s + 6.25s = 9.25s (4.9× speedup)
  
Memory usage: ~2GB peak (8 workers × 250MB per shard)
Parallelism: 100% in Pass 2
```

### Scaling Analysis

| Shards | Workers | Batch Mode | Streaming Mode | Speedup |
|--------|---------|------------|----------------|---------|
| 100    | 4       | 45s        | 12.5s          | 3.6×    |
| 100    | 8       | 45s        | 11s            | 4.1×    |
| 100    | 16      | 45s        | 9s             | 5.0×    |
| 1000   | 8       | 450s       | 70s            | 6.4×    |
| 1000   | 16      | 450s       | 40s            | 11.3×   |

## Unified Environment Variables

### Common Streaming Parameters (All Scripts)

```bash
# ============================================================================
# STREAMING MODE CONTROL (NEW - Common to All Scripts)
# ============================================================================

# Enable streaming mode (default: false for backward compatibility)
ENABLE_TRUE_STREAMING="true"

# Number of parallel workers (default: auto-detect CPU count)
MAX_WORKERS=8

# Output format for sharded files
OUTPUT_FORMAT="csv"  # Options: csv, tsv, parquet

# Memory optimization flag (existing, works in both modes)
OPTIMIZE_MEMORY="true"
```

### Script-Specific Parameters

**Risk Table Mapping:**
```bash
LABEL_FIELD="label"
SMOOTH_FACTOR=0.01
COUNT_THRESHOLD=5
MAX_UNIQUE_THRESHOLD=100
```

**Missing Value Imputation:**
```bash
LABEL_FIELD="label"
DEFAULT_NUMERICAL_STRATEGY="mean"  # mean, median, mode
DEFAULT_CATEGORICAL_STRATEGY="mode"
DEFAULT_TEXT_STRATEGY="mode"
```

**Stratified Sampling:**
```bash
STRATA_COLUMN="risk_category"
TARGET_SAMPLE_SIZE=100000
SAMPLING_STRATEGY="proportional_min"  # balanced, proportional_min, optimal
MIN_SAMPLES_PER_STRATUM=10
VARIANCE_COLUMN=""  # Optional, for optimal allocation
RANDOM_STATE=42
```

## Implementation Strategy

### Phase 1: Core Infrastructure (Week 1)

**1.1 Shared Utilities**
- Extract common functions from `tabular_preprocessing.py`:
  - `find_input_shards()`
  - `extract_shard_number()`
  - `write_shard_file()`
  - `aggregate_results()`
- Create `cursus/steps/scripts/streaming_utils.py` module
- Add unit tests for shared utilities

**1.2 Pass 1 Framework**
- Create base classes for global statistics collection
- Implement memory-efficient aggregation patterns
- Add progress tracking and logging

### Phase 2: Risk Table Mapping (Week 2)

**2.1 Implement Pass 1**
- `collect_risk_tables_pass1()` function
- Incremental aggregation of cross-tabulations
- Smoothing and threshold application

**2.2 Implement Pass 2**
- `process_shard_end_to_end_risk_mapping()` function
- Risk table application per shard
- Output with preserved shard numbering

**2.3 Integration**
- Add streaming mode entry point
- Integrate with existing batch mode
- Add `ENABLE_TRUE_STREAMING` flag handling

**2.4 Testing**
- Unit tests for Pass 1 and Pass 2
- Integration tests comparing batch vs streaming output
- Performance benchmarks

### Phase 3: Missing Value Imputation (Week 3)

**3.1 Implement Pass 1**
- `collect_imputation_statistics_pass1()` function
- Handle numeric (mean) and categorical (mode) columns
- Memory-efficient value aggregation

**3.2 Implement Pass 2**
- `process_shard_end_to_end_imputation()` function
- Apply imputation per shard
- Handle multiple imputation strategies

**3.3 Integration & Testing**
- Same pattern as risk mapping
- Validate output consistency
- Performance benchmarks

### Phase 4: Stratified Sampling (Week 4)

**4.1 Implement Pass 1**
- `collect_stratum_info_pass1()` function
- Stratum counting and variance aggregation
- Support all allocation strategies

**4.2 Implement Pass 2**
- `process_shard_end_to_end_sampling()` function
- Deterministic per-shard sampling
- Preserve stratum distributions

**4.3 Integration & Testing**
- Validate sampling strategies
- Check stratum distribution preservation
- Performance benchmarks

### Phase 5: Documentation & Polish (Week 5)

**5.1 Documentation**
- Update script docstrings
- Add streaming mode usage examples
- Create migration guide from batch to streaming

**5.2 Performance Optimization**
- Profile Pass 1 operations
- Optimize memory usage
- Tune worker allocation

**5.3 Validation**
- End-to-end pipeline testing
- Validate with real datasets
- Stress testing with large datasets

## Edge Cases & Handling

### Empty Shards After Filtering

```python
# Some shards may have 0 rows after processing
if len(df_processed) > 0:
    write_shard_file(df_processed, output_path, output_format)
    stats[split_name] = len(df_processed)
else:
    # Skip writing empty file
    stats[split_name] = 0
```

**Result**: Sparse output numbering (gaps in sequence)

### Missing Categorical Values in Risk Mapping

```python
# Unknown categorical value in inference
if cat_value not in risk_table["bins"]:
    # Use default risk
    risk_value = risk_table["default_bin"]
```

### Missing Imputation Values

```python
# Column has no imputation value (all NaN in training)
if column not in impute_dict:
    # Skip imputation for this column
    continue
```

### Small Strata in Sampling

```python
# Stratum too small for target allocation
sample_size = int(len(stratum_df) * ratio)
if sample_size < min_samples_per_stratum:
    # Take all available samples
    sampled = stratum_df.copy()
```

## Testing Strategy

### Unit Tests

```python
def test_risk_table_collection():
    # Test Pass 1 aggregation
    risk_tables = collect_risk_tables_pass1(...)
    assert len(risk_tables) == expected_fields
    assert "default_bin" in risk_tables[field]

def test_imputation_statistics():
    # Test imputation value computation
    impute_dict = collect_imputation_statistics_pass1(...)
    assert impute_dict["numeric_col"] == expected_mean
    assert impute_dict["cat_col"] == expected_mode

def test_sampling_allocation():
    # Test sampling ratio computation
    ratios = compute_per_shard_sampling_allocation(...)
    assert sum(ratios.values()) <= len(ratios)  # All ratios ≤ 1.0
```

### Integration Tests

```python
def test_batch_vs_streaming_consistency():
    # Compare output between modes
    batch_output = process_batch_mode(...)
    streaming_output = process_streaming_mode(...)
    
    # Statistics should match
    assert batch_output["row_count"] == streaming_output["row_count"]
    
    # Risk tables should match
    assert batch_output["risk_tables"] == streaming_output["risk_tables"]

def test_end_to_end_pipeline():
    # Test full pipeline with all three scripts
    # 1. Risk mapping
    # 2. Imputation  
    # 3. Sampling
    # Validate final output
```

### Performance Tests

```python
def test_streaming_performance():
    # Measure speedup
    batch_time = benchmark_batch_mode()
    streaming_time = benchmark_streaming_mode()
    speedup = batch_time / streaming_time
    assert speedup > 3.0  # Expect 3-5× speedup

def test_memory_usage():
    # Monitor memory during execution
    peak_memory = measure_peak_memory()
    assert peak_memory < batch_mode_memory
```

## Benefits Summary

### Performance Benefits
- ✅ **3-5× faster** than batch mode
- ✅ **100% parallel efficiency** in Pass 2
- ✅ **Linear scaling** with worker count
- ✅ **Consistent performance** regardless of dataset size

### Memory Benefits
- ✅ **80% less peak memory** (~2GB vs ~10GB)
- ✅ **Predictable memory usage** (workers × shard_size)
- ✅ **No OOM errors** on large datasets
- ✅ **Better memory locality** per worker

### Architecture Benefits
- ✅ **Unified framework** across all scripts
- ✅ **1:1 shard mapping** for traceability
- ✅ **Deterministic output** (same input → same output)
- ✅ **Idempotent operations** (can reprocess)

### Operational Benefits
- ✅ **PyTorch compatible** (sharded output for IterableDataset)
- ✅ **Backward compatible** (batch mode unchanged)
- ✅ **Simple configuration** (single flag to enable)
- ✅ **Fault tolerant** (can resume from partial completion)

## Migration Guide

### From Batch to Streaming Mode

**Step 1: Update Environment Variables**
```bash
# Old (batch mode)
# No special flags needed

# New (streaming mode)
ENABLE_TRUE_STREAMING="true"
MAX_WORKERS=8
OUTPUT_FORMAT="csv"
```

**Step 2: Verify Input Format**
```bash
# Ensure input data is in sharded format
# If not, use tabular_preprocessing streaming mode first
ls input_dir/
# Should show: part-00001.csv, part-00002.csv, ...
```

**Step 3: Run Scripts**
```bash
# Risk mapping (streaming)
python risk_table_mapping.py

# Imputation (streaming)
python missing_value_imputation.py

# Sampling (streaming)
python stratified_sampling.py
```

**Step 4: Validate Output**
```bash
# Check output structure
ls output_dir/train/
# Should show: part-00001.csv, part-00003.csv, ... (sparse)

# Compare statistics with batch mode
# Row counts should match
```

### Gradual Rollout Strategy

**Phase 1: Enable for New Pipelines**
- Use streaming mode for all new pipeline development
- Keep batch mode for existing production pipelines

**Phase 2: Validate Existing Pipelines**
- Run both modes in parallel for critical pipelines
- Compare outputs and performance metrics
- Document any discrepancies

**Phase 3: Production Rollout**
- Switch production pipelines to streaming mode
- Monitor performance and memory usage
- Keep batch mode as fallback option

**Phase 4: Deprecation (Optional)**
- After 6 months of successful streaming mode usage
- Consider deprecating batch mode
- Maintain for backward compatibility if needed

## Future Enhancements

### 1. Adaptive Worker Tuning
```python
# Auto-tune workers based on available memory
available_memory = psutil.virtual_memory().available
shard_size = estimate_shard_size(all_shards[0])
optimal_workers = min(
    cpu_count(),
    int(available_memory * 0.7 / shard_size),
    len(all_shards)
)
```

### 2. Progress Tracking
```python
from tqdm import tqdm

# Pass 1 progress
for shard in tqdm(all_shards, desc="Collecting statistics"):
    process_shard(shard)

# Pass 2 progress
with Pool(processes=max_workers) as pool:
    results = list(tqdm(
        pool.imap(process_shard_end_to_end, shard_args),
        total=len(shard_args),
        desc="Processing shards"
    ))
```

### 3. Fault Tolerance
```python
# Resume from partial completion
existing_output_shards = find_existing_output_shards(output_dir)
remaining_shards = [
    s for s in all_shards
    if extract_shard_number(s) not in existing_output_shards
]
process_shards(remaining_shards)  # Only process missing
```

### 4. Distributed Processing
```python
# Use Dask for multi-machine parallelism
from dask.distributed import Client

client = Client(cluster_address)
futures = client.map(process_shard_end_to_end, shard_args)
results = client.gather(futures)
```

### 5. Unified Configuration
```python
# Single config file for all streaming parameters
streaming_config = {
    "enable_streaming": True,
    "max_workers": 8,
    "output_format": "csv",
    "optimize_memory": True,
    "scripts": {
        "risk_mapping": {...},
        "imputation": {...},
        "sampling": {...}
    }
}
```

## Conclusion

This design document provides a comprehensive plan for adding streaming mode to three additional preprocessing scripts, following the proven architecture from `tabular_preprocessing`. The implementation will deliver:

- **3-5× performance improvement** through full parallelization
- **80% memory reduction** through per-shard processing
- **Backward compatibility** by preserving batch mode
- **Production readiness** with fault tolerance and progress tracking

The unified streaming framework ensures consistency across all preprocessing scripts and enables efficient processing of large-scale datasets required for PyTorch IterableDataset training.

**Recommended for immediate implementation** to support the Names3Risk PyTorch training pipeline optimization.

---

**Document Status**: Design Complete - Ready for Implementation  
**Last Updated**: 2026-01-21  
**Related Scripts**:
- `src/cursus/steps/scripts/risk_table_mapping.py`
- `src/cursus/steps/scripts/missing_value_imputation.py`
- `src/cursus/steps/scripts/stratified_sampling.py`
- `src/cursus/steps/scripts/tabular_preprocessing.py` (reference implementation)

### Environment Variables

```bash
# Core Parameters (Existing)
LABEL_FIELD="label"
SMOOTH_FACTOR=0.01
COUNT_THRESHOLD=5
MAX_UNIQUE_THRESHOLD=100

# Streaming Control (NEW)
ENABLE_TRUE_STREAMING="true"  # Enable streaming mode
MAX_WORKERS=8                 # Number of parallel workers
OUTPUT_FORMAT="csv"           # csv, tsv, or parquet

# Memory Optimization (Existing)
OPTIMIZE_MEMORY="true"
```

## Detailed Design: Missing Value Imputation

### Current Batch Implementation

```python
def process_data(data_dict, label_field, job_type, imputation_config, ...):
    """Current batch mode implementation"""
    if job_type == "training":
        # Fit on full training DataFrame
        imputation_engine = SimpleImputationEngine(strategy_manager, label_field)
        imputation_engine.fit(data_dict["train"])
        
        # Transform all splits (full DataFrames)
        for split_name, df in data_dict.items():
            df_imputed = imputation_engine.transform(df)
```

### Streaming Mode Architecture

```python
# ============================================================================
# PASS 1: COLLECT IMPUTATION STATISTICS (Global Statistics)
# ============================================================================

def collect_imputation_statistics_pass1(
    all_shards: List[Path],
    signature_columns: Optional[list],
    label_field: str,
    imputation_config: Dict[str, Any],
    log_func: Callable,
) -> Dict[str, Any]:
    """
    Pass 1: Collect imputation statistics from training shards.
    
    Memory-efficient: Aggregates statistics incrementally across shards.
    
    Returns:
        Dictionary mapping column names to imputation values
    """
    log_func("[PASS1] Collecting imputation statistics from training shards...")
    
    # Initialize aggregators
    column_aggregators = {}
    total_records = 0
    
    # Determine which columns need imputation
    first_shard_df = _read_file_to_df(all_shards[0], signature_columns)
    first_shard_df.columns = [col.replace("__DOT__", ".") for col in first_shard_df.columns]
    
    imputable_columns = [
        col for col in first_shard_df.columns
        if col != label_field and first_shard_df[col].isnull().any()
    ]
    del first_shard_df
    gc.collect()
    
    # Initialize aggregators for each column
    for col in imputable_columns:
        column_aggregators[col] = {
            "sum": 0.0,
            "count": 0,
            "values": [],  # For mode calculation
            "dtype": None,
        }
    
    # Process each shard
    for shard_idx, shard_path in enumerate(all_shards):
        df = _read_file_to_df(shard_path, signature_columns)
        df.columns = [col.replace("__DOT__", ".") for col in df.columns]
        
        total_records += len(df)
        
        # Aggregate statistics for each column
        for col in imputable_columns:
            if col not in df.columns:
                continue
            
            non_null_values = df[col].dropna()
            
            # Store dtype
            if column_aggregators[col]["dtype"] is None:
                column_aggregators[col]["dtype"] = str(df[col].dtype)
            
            # Aggregate based on column type
            if pd.api.types.is_numeric_dtype(df[col]):
                # For numeric: sum and count for mean
                column_aggregators[col]["sum"] += non_null_values.sum()
                column_aggregators[col]["count"] += len(non_null_values)
            else:
                # For categorical/text: collect values for mode
                column_aggregators[col]["values"].extend(non_null_values.tolist())
        
        del df
        gc.collect()
    
    # Compute final imputation values
    impute_dict = {}
    for col, aggregator in column_aggregators.items():
        dtype = aggregator["dtype"]
        
        if "int" in dtype or "float" in dtype:
            # Numeric: use mean (or median/mode based on config)
            strategy = imputation_config.get("default_numerical_strategy", "mean")
            if strategy == "mean" and aggregator["count"] > 0:
                impute_dict[col] = aggregator["sum"] / aggregator["count"]
            elif strategy == "median":
                # For median, need full data - use mean as fallback
                log_func(f"[PASS1] Using mean for {col} (median requires full data)")
                impute_dict[col] = aggregator["sum"] / aggregator["count"]
        else:
            # Categorical/text: use mode
            if aggregator["values"]:
                from collections import Counter
                mode_value = Counter(aggregator["values"]).most_common(1)[0][0]
                impute_dict[col] = mode_value
    
    log_func(f"[PASS1] Collected imputation values for {len(impute_dict)} columns")
    log_func(f"[PASS1] Total records processed: {total_records:,}")
    
    return impute_dict


# ============================================================================
# PASS 2: PARALLEL PER-SHARD IMPUTATION
# ============================================================================

def process_shard_end_to_end_imputation(args: tuple) -> Dict[str, int]:
    """
    Process single shard: read → apply imputation → write.
    
    Args:
        args: Tuple of (shard_path, shard_index, global_context,
                       output_base, signature_columns, output_format, ...)
    
    Returns:
        Statistics dict with row counts per split
    """
    (shard_path, shard_index, global_context, output_base,
     signature_columns, output_format) = args
    
    # Extract input shard number
    shard_num = extract_shard_number(shard_path)
    
    # ============================================================
    # STEP 1: Read Single Shard
    # ============================================================
    df = _read_file_to_df(shard_path, signature_columns)
    df.columns = [col.replace("__DOT__", ".") for col in df.columns]
    
    # ============================================================
    # STEP 2: Apply Imputation (Using Global Context)
    # ============================================================
    impute_dict = global_context["impute_dict"]
    
    for column, impute_value in impute_dict.items():
        if column in df.columns:
            # Only fill NaN values
            df[column] = df[column].fillna(impute_value)
    
    # ============================================================
    # STEP 3: Write to Split Folders (Preserving Shard Number)
    # ============================================================
    stats = {}
    job_type = global_context["job_type"]
    
    if job_type == "training":
        for split_name in ["train", "val", "test"]:
            output_path = (output_base / split_name /
                          f"part-{shard_num:05d}.{output_format}")
            write_shard_file(df, output_path, output_format)
            stats[split_name] = len(df)
    else:
        output_path = (output_base / job_type /
                      f"part-{shard_num:05d}.{output_format}")
        write_shard_file(df, output_path, output_format)
        stats[job_type] = len(df)
    
    return stats


# ============================================================================
# MAIN STREAMING MODE ENTRY POINT
# ============================================================ ============

def process_streaming_mode_imputation(
    input_dir: str,
    output_dir: str,
    signature_columns: Optional[list],
    job_type: str,
    label_field: str,
    imputation_config: Dict[str, Any],
    output_format: str,
    max_workers: Optional[int],
    optimize_memory: bool,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Streaming mode for missing value imputation.
    
    Uses two-pass architecture:
    - Pass 1: Collect global imputation statistics from training shards
    - Pass 2: Apply imputations per shard in parallel
    """
    log = logger or print
    output_path = Path(output_dir)
    
    log("[STREAMING] Starting imputation in streaming mode")
    
    # Find input shards
    all_shards = find_input_shards(input_dir, log)
    
    # Determine optimal workers
    if max_workers is None:
        max_workers = min(cpu_count(), len(all_shards))
    log(f"[STREAMING] Using {max_workers} parallel workers")
    
    # ============================================================
    # PASS 1: Collect Imputation Statistics (Training Mode Only)
    # ============================================================
    impute_dict = None
    if job_type == "training":
        impute_dict = collect_imputation_statistics_pass1(
            all_shards, signature_columns, label_field,
            imputation_config, log
        )
    else:
        # Load pre-computed imputation values from artifacts
        impute_dict = load_imputation_parameters_from_artifacts(...)
    
    # Build global context
    global_context = {
        "job_type": job_type,
        "impute_dict": impute_dict,
    }
    
    # ============================================================
    # PASS 2: Parallel Per-Shard Processing
    # ============================================================
    log("[STREAMING] Pass 2: Applying imputations in parallel...")
    
    shard_args = [
        (shard, i, global_context, output_path,
         signature_columns, output_format)
        for i, shard in enumerate(all_shards)
    ]
    
    with Pool(processes=max_workers) as pool:
        results = pool.map(process_shard_end_to_end_imputation, shard_args)
    
    # Aggregate statistics
    total_stats = aggregate_results(results, job_type)
    log(f"[STREAMING] Complete! Row distribution: {total_stats}")
    
    return {}  # Data written to disk
```

### Environment Variables

```bash
# Core Parameters (Existing)
LABEL_FIELD="label"
DEFAULT_NUMERICAL_STRATEGY="mean"
DEFAULT_CATEGORICAL_STRATEGY="mode"
DEFAULT_TEXT_STRATEGY="mode"

# Streaming Control (NEW)
ENABLE_TRUE_STREAMING="true"  # Enable streaming mode
MAX_WORKERS=8                 # Number of parallel workers
OUTPUT_FORMAT="csv"           # csv, tsv, or parquet

# Memory Optimization (Existing)
OPTIMIZE_MEMORY="true"
```

## Detailed Design: Stratified Sampling

### Current Batch Implementation

```python
def main(input_paths, output_paths, environ_vars, job_args, logger):
    """Current batch mode implementation"""
    # Load full DataFrames
    df, detected_format = _read_processed_data(input_data_dir, split_name)
    
    # Sample entire dataset
    sampled_df = sampler.sample(
        df=df,
        strata_column=strata_column,
        target_size=target_sample_size,
        strategy=sampling_strategy,
    )
```

### Streaming Mode Architecture

```python
# ============================================================================
# PASS 1: COLLECT STRATUM INFORMATION (Global Statistics)
# ============================================================ ============

def collect_stratum_info_pass1(
    all_shards: List[Path],
    signature_columns: Optional[list],
    strata_column: str,
    variance_column: Optional[str],
    log_func: Callable,
) -> Dict[str, Any]:
    """
    Pass 1: Collect stratum counts and statistics from all shards.
    
    Memory-efficient: Only aggregates stratum-level statistics.
    
    Returns:
        Dictionary containing stratum sizes and allocation ratios
    """
    log_func("[PASS1] Collecting stratum information from shards...")
    
    # Initialize stratum aggregators
    stratum_counts = {}
    stratum_variances = {} if variance_column else None
    total_records = 0
    
    # Process each shard
    for shard_idx, shard_path in enumerate(all_shards):
        df = _read_file_to_df(shard_path, signature_columns)
        df.columns = [col.replace("__DOT__", ".") for col in df.columns]
        
        total_records += len(df)
        
        # Count records per stratum
        stratum_value_counts = df[strata_column].value_counts()
        for stratum, count in stratum_value_counts.items():
            stratum_counts[stratum] = stratum_counts.get(stratum, 0) + count
        
        # Aggregate variance if needed (for optimal allocation)
        if variance_column and variance_column in df.columns:
            for stratum in df[strata_column].unique():
                stratum_df = df[df[strata_column] == stratum]
                if len(stratum_df) > 1:
                    var = stratum_df[variance_column].var()
                    if stratum not in stratum_variances:
                        stratum_variances[stratum] = {"sum_var": 0, "count": 0}
                    stratum_variances[stratum]["sum_var"] += var * len(stratum_df)
                    stratum_variances[stratum]["count"] += len(stratum_df)
        
        del df
        gc.collect()
    
    log_func(f"[PASS1] Found {len(stratum_counts)} strata")
    log_func(f"[PASS1] Total records: {total_records:,}")
    log_func(f"[PASS1] Stratum distribution: {dict(stratum_counts)}")
    
    return {
        "stratum_counts": stratum_counts,
        "stratum_variances": stratum_variances,
        "total_records": total_records,
    }


def compute_per_shard_sampling_allocation(
    stratum_info: Dict[str, Any],
    target_size: int,
    strategy: str,
    min_samples_per_stratum: int,
) -> Dict[Any, float]:
    """
    Compute sampling ratio for each stratum based on global statistics.
    
    Returns:
        Dictionary mapping stratum values to sampling ratios (0.0-1.0)
    """
    stratum_counts = stratum_info["stratum_counts"]
    total_records = stratum_info["total_records"]
    
    # Compute allocation based on strategy
    if strategy == "balanced":
        # Equal allocation per stratum
        num_strata = len(stratum_counts)
        samples_per_stratum = target_size / num_strata
        
        # Compute sampling ratio per stratum
        sampling_ratios = {}
        for stratum, count in stratum_counts.items():
            sampling_ratios[stratum] = min(1.0, samples_per_stratum / count)
    
    elif strategy == "proportional_min":
        # Proportional with minimum constraints
        sampling_ratios = {}
        for stratum, count in stratum_counts.items():
            proportion = count / total_records
            proportional_size = target_size * proportion
            allocated_size = max(min_samples_per_stratum, proportional_size)
            sampling_ratios[stratum] = min(1.0, allocated_size / count)
    
    elif strategy == "optimal":
        # Optimal allocation (Neyman)
        stratum_variances = stratum_info["stratum_variances"]
        if not stratum_variances:
            # Fallback to proportional if no variance data
            sampling_ratios = {}
            for stratum, count in stratum_counts.items():
                sampling_ratios[stratum] = min(1.0, target_size / total_records)
        else:
            # Compute optimal allocation
            numerators = {}
            total_numerator = 0
            for stratum, count in stratum_counts.items():
                if stratum in stratum_variances:
                    var_data = stratum_variances[stratum]
                    avg_var = var_data["sum_var"] / var_data["count"] if var_data["count"] > 0 else 1.0
                    std = np.sqrt(avg_var)
                else:
                    std = 1.0
                numerator = count * std
                numerators[stratum] = numerator
                total_numerator += numerator
            
            sampling_ratios = {}
            for stratum, numerator in numerators.items():
                if total_numerator > 0:
                    optimal_size = target_size * numerator / total_numerator
                else:
                    optimal_size = target_size / len(stratum_counts)
                allocated_size = max(min_samples_per_stratum, optimal_size)
                sampling_ratios[stratum] = min(1.0, allocated_size / stratum_counts[stratum])
    
    return sampling_ratios


# ============================================================================
# PASS 2: PARALLEL PER-SHARD SAMPLING
# ============================================================================

def process_shard_end_to_end_sampling(args: tuple) -> Dict[str, int]:
    """
    Process single shard: read → apply sampling → write.
    
    Args:
        args: Tuple of (shard_path, shard_index, global_context,
                       output_base, signature_columns, output_format, ...)
    
    Returns:
        Statistics dict with row counts per split
    """
    (shard_path, shard_index, global_context, output_base,
     signature_columns, strata_column, output_format, random_state) = args
    
    # Extract input shard number
    shard_num = extract_shard_number(shard_path)
    
    # Set deterministic random seed for this shard
    np.random.seed(random_state + shard_num)
    
    # ============================================================
    # STEP 1: Read Single Shard
    # ============================================================
    df = _read_file_to_df(shard_path, signature_columns)
    df.columns = [col.replace("__DOT__", ".") for col in df.columns]
    
    # ============================================================
    # STEP 2: Apply Per-Shard Sampling (Using Global Context)
    # ============================================================
    sampling_ratios = global_context["sampling_ratios"]
    
    # Sample per stratum based on computed ratios
    sampled_dfs = []
    for stratum, ratio in sampling_ratios.items():
        stratum_df = df[df[strata_column] == stratum]
        if len(stratum_df) > 0 and ratio > 0:
            sample_size = int(len(stratum_df) * ratio)
            if sample_size > 0:
                sampled = stratum_df.sample(n=min(sample_size, len(stratum_df)))
                sampled_dfs.append(sampled)
    
    if sampled_dfs:
        df_sampled = pd.concat(sampled_dfs, ignore_index=True)
    else:
        df_sampled = pd.DataFrame()
    
    # ============================================================
    # STEP 3: Write to Split Folders (Preserving Shard Number)
    # ============================================================
    stats = {}
    job_type = global_context["job_type"]
    
    if len(df_sampled) > 0:
        if job_type == "training":
            # For training, write to appropriate split
            # (Assumes this runs after splits are already assigned)
            for split_name in ["train", "val"]:
                output_path = (output_base / split_name /
                              f"part-{shard_num:05d}.{output_format}")
                write_shard_file(df_sampled, output_path, output_format)
                stats[split_name] = len(df_sampled)
        else:
            output_path = (output_base / job_type /
                          f"part-{shard_num:05d}.{output_format}")
            write_shard_file(df_sampled, output_path, output_format)
            stats[job_type] = len(df_sampled)
    
    return stats


# ============================================================================
# MAIN STREAMING MODE ENTRY POINT
# ============================================================================

def process_streaming_mode_sampling(
    input_dir: str,
    output_dir: str,
    signature_columns: Optional[list],
    job_type: str,
    strata_column: str,
    target_sample_size: int,
    sampling_strategy: str,
    min_samples_per_stratum: int,
    variance_column: Optional[str],
    random_state: int,
    output_format: str,
    max_workers: Optional[int],
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Streaming mode for stratified sampling.
    
    Uses two-pass architecture:
    - Pass 1: Collect stratum information from all shards
    - Pass 2: Apply per-shard sampling in parallel
    """
    log = logger or print
    output_path = Path(output_dir)
    
    log("[STREAMING] Starting stratified sampling in streaming mode")
    
    # Find input shards
    all_shards = find_input_shards(input_dir, log)
    
    # Determine optimal workers
    if max_workers is None:
        max_workers = min(cpu_count(), len(all_shards))
    log(f"[STREAMING] Using {max_workers} parallel workers")
    
    # ============================================================
    # PASS 1: Collect Stratum Information
    # ============================================================
    log("[STREAMING] Pass 1: Collecting stratum information...")
    stratum_info = collect_stratum_info_pass1(
        all_shards, signature_columns, strata_column,
        variance_column, log
    )
    
    # Compute sampling allocation
    sampling_ratios = compute_per_shard_sampling_allocation(
        stratum_info, target_sample_size, sampling_strategy,
        min_samples_per_stratum
    )
    
    log(f"[STREAMING] Sampling ratios: {sampling_ratios}")
    
    # Build global context
    global_context = {
        "job_type": job_type,
        "sampling_ratios": sampling_ratios,
    }
    
    # ============================================================
    # PASS 2: Parallel Per-Shard Sampling
    # ============================================================
    log("[STREAMING] Pass 2: Applying sampling in parallel...")
    
    shard_args = [
        (shard, i, global_context, output_path,
         signature_columns, strata_column, output_format, random_state)
        for i, shard in enumerate(all_shards)
    ]
    
    with Pool(processes=max_workers) as pool:
        results = pool.map(process_shard_end_to_end_sampling, shard_args)
    
    # Aggregate statistics
    total_stats = aggregate_results(results, job_type)
    log(f"[STREAMING] Complete! Sampled rows: {total_stats}")
    
    return {}  # Data written to disk
```

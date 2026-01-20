---
tags:
  - design
  - implementation
  - preprocessing
  - streaming
  - parallelization
keywords:
  - multi-worker
  - streaming mode
  - parallel shard loading
  - memory efficiency
  - throughput optimization
  - tabular preprocessing
topics:
  - data preprocessing
  - parallel processing
  - streaming architecture
  - performance optimization
language: python
date of note: 2026-01-20
---

# Multi-Worker Streaming Preprocessing Design

## 1. Executive Summary

This document specifies the enhancement of the Names3Risk tabular preprocessing script to support **multi-worker parallel shard loading in streaming mode**. Currently, streaming mode loads shards sequentially, creating a significant I/O bottleneck. By adding multi-worker parallelization, we can achieve 5-10× throughput improvements while maintaining memory-efficient streaming behavior.

### 1.1 Problem Statement

**Current Limitation**: The `tabular_preprocessing.py` script supports two modes:
- ✅ **Batch Mode**: Uses `multiprocessing.Pool` for parallel shard reading (fast)
- ❌ **Streaming Mode**: Loads shards sequentially one-by-one (slow)

Despite having a `max_workers` parameter, **streaming mode does not utilize it**. All shards are loaded sequentially in `process_single_batch()`, causing:

- Poor I/O utilization (single-threaded disk/network access)
- Long preprocessing times for large datasets
- Underutilized CPU resources during I/O wait
- No performance scaling with available cores

### 1.2 Current vs. Proposed Performance

**Example Scenario**: 100 shards, 8 CPU cores, 2 seconds per shard I/O

| Mode | Current | Proposed | Speedup |
|------|---------|----------|---------|
| **Sequential Loading** | 200s (100 × 2s) | - | Baseline |
| **Parallel Loading (8 workers)** | - | 25s (~13 waves × 2s) | **8× faster** |

### 1.3 Key Design Principles

1. **Reuse Existing Infrastructure**: Leverage batch mode's `_read_shard_wrapper()` and `Pool.map()`
2. **Maintain Memory Safety**: `streaming_batch_size` controls maximum concurrent shards
3. **Backward Compatible**: No breaking changes to existing API
4. **Simple Implementation**: Minimal code changes, maximum impact

---

## 2. Background and Current Implementation

### 2.1 Script Architecture Overview

```
projects/names3risk_pytorch/dockers/scripts/tabular_preprocessing.py
├── Shared Utilities (Lines 1-437)
│   ├── optimize_dtypes()
│   ├── load_signature_columns()
│   ├── _read_file_to_df()
│   └── _read_shard_wrapper()        # ← Used by batch mode for parallel reading
├── Batch Mode (Lines 541-774)
│   ├── combine_shards()              # ← Uses Pool.map() with max_workers ✅
│   └── process_batch_mode_preprocessing()
└── Streaming Mode (Lines 1196-1465)
    ├── process_streaming_mode_preprocessing()
    ├── process_single_batch()        # ← Sequential loading ❌
    ├── process_training_splits_streaming()
    └── process_single_split_streaming()
```

### 2.2 Current Batch Mode Implementation (WORKING)

**Location**: Lines 541-609 in `combine_shards()`

```python
def combine_shards(
    input_dir: str,
    signature_columns: Optional[list] = None,
    max_workers: Optional[int] = None,  # ← Used!
    batch_size: int = 10,
    streaming_batch_size: Optional[int] = None,
) -> pd.DataFrame:
    """Batch mode: Parallel shard reading."""
    
    # Determine optimal number of workers
    if max_workers is None:
        max_workers = min(cpu_count(), total_shards)
    
    # Prepare arguments for parallel processing
    shard_args = [
        (shard, signature_columns, i, total_shards)
        for i, shard in enumerate(all_shards)
    ]
    
    # PARALLEL READING with Pool.map()
    if max_workers > 1 and total_shards > 1:
        with Pool(processes=max_workers) as pool:
            dataframes = pool.map(_read_shard_wrapper, shard_args)  # ✅ Parallel!
    else:
        dataframes = [_read_shard_wrapper(args) for args in shard_args]
    
    # Concatenate results
    result_df = _batch_concat_dataframes(dataframes, batch_size)
    return result_df
```

**Analysis**:
- ✅ Uses `multiprocessing.Pool` with `max_workers` processes
- ✅ Parallel I/O across CPU cores
- ✅ Efficient for moderately-sized datasets

### 2.3 Current Streaming Mode Implementation (BROKEN)

**Location**: Lines 1100-1151 in `process_single_batch()`

```python
def process_single_batch(
    shard_files: List[Path],
    signature_columns: Optional[list],
    batch_size: int,
    optimize_memory: bool,
    label_field: Optional[str],
    log_func: Callable,
    preserve_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Process a single batch of shards with Names3Risk preprocessing."""
    
    # ❌ SEQUENTIAL LOADING - No parallel workers used!
    batch_dfs = []
    for shard in shard_files:
        df = _read_file_to_df(shard, signature_columns)  # Blocking I/O
        batch_dfs.append(df)
    
    # Concatenate
    batch_df = _batch_concat_dataframes(batch_dfs, batch_size)
    
    # ... preprocessing ...
    
    return batch_df
```

**Problems**:
- ❌ Sequential `for` loop - one shard at a time
- ❌ No use of `multiprocessing.Pool`
- ❌ `max_workers` parameter exists but is never passed to this function
- ❌ I/O bottleneck dominates processing time

### 2.4 Parameter Relationship Analysis

Current streaming mode has these parameters:

| Parameter | Purpose | Current Usage | Line |
|-----------|---------|---------------|------|
| `max_workers` | Number of parallel CPU workers | ❌ Ignored | 1206 |
| `streaming_batch_size` | Shards per iteration (memory control) | ✅ Used | 1208, 1243 |
| `batch_size` | DataFrame concat batch size | ✅ Used | 1209, 437 |
| `shard_size` | Rows per output shard | ✅ Used | 1207 |

**Key Insight**: `streaming_batch_size` already serves as the "window size" for parallel loading. We just need to add actual parallelization within that window!

### 2.5 Data Flow - Current vs. Proposed

**Current (Sequential)**:
```
Streaming Batch (streaming_batch_size=20 shards)
├─ Load shard 0  [====] 2.0s  → CPU idle, I/O wait
├─ Load shard 1  [====] 2.0s  → CPU idle, I/O wait
├─ Load shard 2  [====] 2.0s  → CPU idle, I/O wait
├─ ... (17 more shards)
└─ Total: 40 seconds for 20 shards

Per-batch throughput: 0.5 shards/sec
Memory peak: 1 shard at a time (~500MB)
```

**Proposed (Parallel with max_workers=8)**:
```
Streaming Batch (streaming_batch_size=20 shards)
├─ Wave 1: Workers 0-7 load shards 0-7   [====] 2.0s (parallel)
├─ Wave 2: Workers 0-7 load shards 8-15  [====] 2.0s (parallel)
├─ Wave 3: Workers 0-3 load shards 16-19 [====] 2.0s (parallel)
└─ Total: 6 seconds for 20 shards

Per-batch throughput: 3.3 shards/sec (6.6× faster!)
Memory peak: 8 shards at a time (~4GB) - controlled by max_workers
```

---

## 3. Proposed Solution Architecture

### 3.1 High-Level Design

**Strategy**: Add parallel shard reading to `process_single_batch()` by reusing batch mode's proven implementation.

**Key Changes**:
1. Add `max_workers` parameter to `process_single_batch()`
2. Use `multiprocessing.Pool.map()` with `_read_shard_wrapper()` (already exists!)
3. Control memory via both `streaming_batch_size` and `max_workers`
4. Wire up parameter through call chain

**Design Diagram**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Streaming Mode Control Flow                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  process_streaming_mode_preprocessing()                              │
│  ├─ streaming_batch_size: 20   (outer loop control)                 │
│  ├─ max_workers: 8              (parallel I/O control)               │
│  └─ Calls process_training_splits_streaming()                        │
│                                                                       │
│      ├─ Loop over all shards in chunks of streaming_batch_size       │
│      │                                                               │
│      └─ For batch_start in range(0, len(all_shards), 20):           │
│          batch_shards = all_shards[batch_start:batch_start+20]       │
│          │                                                           │
│          └─ process_single_batch(batch_shards, max_workers=8, ...)   │
│              │                                                       │
│              ├─ [NEW] Parallel Reading Phase                        │
│              │   with Pool(processes=8) as pool:                     │
│              │       batch_dfs = pool.map(_read_shard_wrapper,      │
│              │                            batch_shards)              │
│              │   ↓                                                   │
│              │   20 shards → 8 workers → 3 waves → 6 seconds         │
│              │                                                       │
│              ├─ Concatenation Phase                                 │
│              │   batch_df = _batch_concat_dataframes(batch_dfs, 10) │
│              │                                                       │
│              └─ Preprocessing Phase                                 │
│                  detect_and_apply_names3risk_preprocessing(...)      │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Memory Management Strategy

**Two-Level Memory Control**:

1. **Outer Level**: `streaming_batch_size` (lines 1243-1260)
   - Controls how many shards are processed per iteration
   - Limits maximum memory footprint
   - Example: `streaming_batch_size=20` → max 20 shards in memory

2. **Inner Level**: `max_workers` (proposed addition)
   - Controls how many shards load **simultaneously** within a batch
   - Example: `max_workers=8` → 8 shards loading at once
   - When `max_workers < streaming_batch_size`, loading happens in waves

**Memory Calculation**:
```python
# Worst-case memory usage
max_concurrent_shards = min(max_workers, streaming_batch_size)
memory_per_shard = 500  # MB (typical)
peak_memory_mb = max_concurrent_shards * memory_per_shard

# Example configurations:
# Config 1: streaming_batch_size=10, max_workers=4
#   → 4 shards × 500MB = 2GB peak (safe)
# 
# Config 2: streaming_batch_size=50, max_workers=16
#   → 16 shards × 500MB = 8GB peak (requires large instance)
#
# Config 3: streaming_batch_size=5, max_workers=8
#   → 5 shards × 500MB = 2.5GB peak (max_workers limited by batch size)
```

### 3.3 Implementation Specification

#### 3.3.1 Enhanced `process_single_batch()` Function

**Location**: Lines 1100-1151 (modify existing function)

```python
def process_single_batch(
    shard_files: List[Path],
    signature_columns: Optional[list],
    batch_size: int,
    optimize_memory: bool,
    label_field: Optional[str],
    log_func: Callable,
    max_workers: Optional[int] = None,  # ← NEW PARAMETER
    preserve_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Process a single batch of shards with Names3Risk preprocessing.
    
    NEW: Supports parallel shard reading for improved throughput.
    
    Args:
        shard_files: List of shard file paths to process
        signature_columns: Optional column names for CSV/TSV files
        batch_size: Batch size for concatenation
        optimize_memory: Whether to optimize dtypes
        label_field: Name of label column (optional)
        log_func: Logging function
        max_workers: Number of parallel workers for shard reading (NEW)
                     If None or 1, falls back to sequential processing.
        preserve_columns: Optional columns to preserve during filtering
    
    Returns:
        Processed DataFrame from batch
    """
    # ============================================================
    # NEW: Parallel Shard Reading
    # ============================================================
    if max_workers and max_workers > 1 and len(shard_files) > 1:
        # Prepare arguments for parallel processing
        shard_args = [
            (shard, signature_columns, i, len(shard_files))
            for i, shard in enumerate(shard_files)
        ]
        
        # Use multiprocessing.Pool for parallel I/O
        with Pool(processes=max_workers) as pool:
            batch_dfs = pool.map(_read_shard_wrapper, shard_args)
        
        log_func(
            f"[STREAMING] Parallel loaded {len(shard_files)} shards "
            f"using {max_workers} workers"
        )
    else:
        # Fallback to sequential processing
        batch_dfs = []
        for shard in shard_files:
            df = _read_file_to_df(shard, signature_columns)
            batch_dfs.append(df)
        
        log_func(
            f"[STREAMING] Sequential loaded {len(shard_files)} shards "
            f"(single worker)"
        )
    
    # ============================================================
    # Existing: Concatenation and Preprocessing (unchanged)
    # ============================================================
    batch_df = _batch_concat_dataframes(batch_dfs, batch_size)
    del batch_dfs
    gc.collect()
    
    # Apply memory optimization if enabled
    if optimize_memory:
        batch_df = optimize_dtypes(batch_df, log_func)
    
    # Process columns
    batch_df.columns = [col.replace("__DOT__", ".") for col in batch_df.columns]
    
    # Apply Names3Risk-specific preprocessing
    batch_df = detect_and_apply_names3risk_preprocessing(batch_df, log_func)
    
    # Keep all columns (no conversion/filtering)
    log_func(
        "[STREAMING] Keeping all columns with original dtypes "
        "(no conversion or filtering)"
    )
    
    # Process labels if provided
    if label_field:
        if label_field not in batch_df.columns:
            raise RuntimeError(f"Label field '{label_field}' not found in columns")
        batch_df = process_label_column(batch_df, label_field, log_func)
    
    return batch_df
```

#### 3.3.2 Updated Function Call Chain

**Changes Required**:

1. **`process_training_splits_streaming()`** (line 1185):
```python
def process_training_splits_streaming(
    all_shards: List[Path],
    output_path: Path,
    signature_columns: Optional[list],
    label_field: Optional[str],
    train_ratio: float,
    test_val_ratio: float,
    output_format: str,
    streaming_batch_size: int,
    shard_size: int,
    batch_size: int,
    optimize_memory: bool,
    consolidate_shards: bool,
    max_workers: Optional[int],  # ← Already exists, needs wiring
    log_func: Callable,
) -> None:
    """Process training data with time-based splits in streaming mode."""
    # ... existing code ...
    
    # Process batch (NOW WITH PARALLEL WORKERS)
    batch_df = process_single_batch(
        batch_shards,
        signature_columns,
        batch_size,
        optimize_memory,
        label_field,
        log_func,
        max_workers=max_workers,  # ← ADD THIS LINE
        preserve_columns=preserve_cols,
    )
```

2. **`process_single_split_streaming()`** (line 1283):
```python
def process_single_split_streaming(
    all_shards: List[Path],
    output_path: Path,
    job_type: str,
    signature_columns: Optional[list],
    label_field: Optional[str],
    output_format: str,
    streaming_batch_size: int,
    shard_size: int,
    batch_size: int,
    optimize_memory: bool,
    consolidate_shards: bool,
    max_workers: Optional[int],  # ← Already exists, needs wiring
    log_func: Callable,
) -> None:
    """Process non-training data as single split in streaming mode."""
    # ... existing code ...
    
    # Process batch (NOW WITH PARALLEL WORKERS)
    batch_df = process_single_batch(
        batch_shards,
        signature_columns,
        batch_size,
        optimize_memory,
        label_field,
        log_func,
        max_workers=max_workers,  # ← ADD THIS LINE
    )
```

---

## 4. Performance Analysis

### 4.1 Theoretical Speedup Model

**Amdahl's Law Application**:
```
Sequential time (T_seq): T_load + T_process
Parallel time (T_par):   T_load/N + T_process

Where:
- T_load: Total I/O time (dominant)
- T_process: CPU processing time (small)
- N: Number of workers (max_workers)
```

**Realistic Scenario**:
```python
# Per-shard breakdown:
T_load = 2.0s      # Disk/network I/O
T_process = 0.3s   # DataFrame processing

# 20 shards sequential:
T_seq = 20 × (2.0 + 0.3) = 46s

# 20 shards with 8 workers:
T_par = (20 × 2.0) / 8 + 20 × 0.3 = 5.0s + 6.0s = 11s

# Speedup = 46s / 11s = 4.2× faster
```

### 4.2 Expected Performance Gains

| Configuration | Sequential Time | Parallel Time | Speedup |
|---------------|----------------|---------------|---------|
| **10 shards, 4 workers** | 23s | 8s | 2.9× |
| **20 shards, 8 workers** | 46s | 11s | 4.2× |
| **50 shards, 16 workers** | 115s | 18s | 6.4× |
| **100 shards, 8 workers** | 230s | 36s | 6.4× |

**Observations**:
1. Speedup increases with more shards (better worker utilization)
2. Diminishing returns beyond 8-16 workers (I/O contention)
3. Processing time becomes bottleneck at high worker counts

### 4.3 Memory Usage Comparison

**Current Sequential Mode**:
```
Peak memory = 1 shard + overhead
            ≈ 500MB + 200MB = 700MB
```

**Proposed Parallel Mode**:
```
Peak memory = min(max_workers, streaming_batch_size) × shard_size + overhead
            = min(8, 20) × 500MB + 200MB
            = 4,200MB (4.2GB)
```

**Memory Safety Guidelines**:
```python
# Conservative (for 8GB instance):
streaming_batch_size = 10
max_workers = 4
# Peak: 2.2GB

# Balanced (for 16GB instance):
streaming_batch_size = 20
max_workers = 8
# Peak: 4.2GB

# Aggressive (for 32GB+ instance):
streaming_batch_size = 50
max_workers = 16
# Peak: 8.2GB
```

### 4.4 Throughput Projections

**Current Performance** (from production logs):
```
Sequential: 0.5 shards/second
Time per epoch (10,000 shards): 5.5 hours
```

**Projected Performance** (with 8 workers):
```
Parallel: 3.2 shards/second (6.4× improvement)
Time per epoch (10,000 shards): 52 minutes
Time saved: 4 hours 38 minutes per epoch
```

---

## 5. Implementation Plan

### 5.1 Code Changes Summary

**Files to Modify**: 1 file
- `projects/names3risk_pytorch/dockers/scripts/tabular_preprocessing.py`

**Lines to Change**: ~20 lines total

| Function | Line Range | Change Type | LOC |
|----------|-----------|-------------|-----|
| `process_single_batch()` | 1100-1151 | Modify (add parallel loading) | +15 |
| `process_training_splits_streaming()` | 1185-1280 | Modify (wire max_workers) | +1 |
| `process_single_split_streaming()` | 1283-1380 | Modify (wire max_workers) | +1 |

**Total Impact**: ~17 lines of new/modified code

### 5.2 Step-by-Step Implementation

#### Step 1: Modify `process_single_batch()` (Estimated: 30 minutes)

**Location**: Lines 1100-1151

**Changes**:
1. Add `max_workers: Optional[int] = None` parameter
2. Add conditional block for parallel vs sequential loading
3. Add logging for worker count
4. Keep all other logic unchanged

**Risk**: Low (isolated change, well-tested pattern from batch mode)

#### Step 2: Wire `max_workers` in Training Splits (Estimated: 5 minutes)

**Location**: Line ~1267 in `process_training_splits_streaming()`

**Change**:
```python
# Add this parameter to process_single_batch() call:
max_workers=max_workers,
```

**Risk**: Minimal (parameter already exists, just needs passing)

#### Step 3: Wire `max_workers` in Single Split (Estimated: 5 minutes)

**Location**: Line ~1337 in `process_single_split_streaming()`

**Change**:
```python
# Add this parameter to process_single_batch() call:
max_workers=max_workers,
```

**Risk**: Minimal (parameter already exists, just needs passing)

#### Step 4: Update Documentation (Estimated: 15 minutes)

**Locations**:
- Docstring for `process_single_batch()`
- Docstring for `process_streaming_mode_preprocessing()`
- Main function docstring

**Changes**:
- Document `max_workers` parameter behavior
- Add performance guidance
- Note memory implications

### 5.3 Configuration Changes

**Environment Variables** (already supported):
```bash
# No new variables needed - MAX_WORKERS already exists!
export MAX_WORKERS=8               # Number of parallel I/O workers
export STREAMING_BATCH_SIZE=20     # Shards per iteration (memory control)
export BATCH_SIZE=10               # Concatenation batch size
```

**Recommended Configurations**:
```python
# Small instance (8GB RAM):
MAX_WORKERS=4
STREAMING_BATCH_SIZE=10

# Medium instance (16GB RAM):
MAX_WORKERS=8
STREAMING_BATCH_SIZE=20

# Large instance (32GB+ RAM):
MAX_WORKERS=16
STREAMING_BATCH_SIZE=50
```

---

## 6. Testing and Validation Strategy

### 6.1 Unit Tests

**Test File**: `tests/names3risk_pytorch/test_tabular_preprocessing_parallel.py` (new)

```python
import pytest
import tempfile
import pandas as pd
from pathlib import Path

def test_parallel_shard_loading():
    """Verify parallel loading produces same results as sequential."""
    # Create test shards
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write 10 test shards
        for i in range(10):
            df = pd.DataFrame({
                'id': [f'id_{i}_{j}' for j in range(100)],
                'label': [j % 2 for j in range(100)],
                'value': [j * 1.5 for j in range(100)]
            })
            df.to_parquet(Path(tmpdir) / f'part-{i:05d}.parquet')
        
        # Load sequentially
        result_seq = process_single_batch(
            shard_files=list(Path(tmpdir).glob('*.parquet')),
            max_workers=1,  # Sequential
            # ... other params ...
        )
        
        # Load in parallel
        result_par = process_single_batch(
            shard_files=list(Path(tmpdir).glob('*.parquet')),
            max_workers=4,  # Parallel
            # ... other params ...
        )
        
        # Results should be identical
        pd.testing.assert_frame_equal(result_seq, result_par)

def test_memory_bound_parallel_loading():
    """Verify memory usage stays within bounds."""
    # Monitor peak memory during parallel loading
    # Should not exceed max_workers × shard_size
    pass

def test_worker_count_edge_cases():
    """Test edge cases: 0, 1, None, > cpu_count()."""
    # max_workers=None → should use cpu_count()
    # max_workers=0 → should use sequential
    # max_workers=1 → should use sequential
    # max_workers=100 → should cap at reasonable limit
    pass
```

### 6.2 Integration Tests

**Test Scenarios**:

1. **Full Streaming Pipeline Test**:
   - 100 shards, various max_workers (1, 4, 8, 16)
   - Verify output consistency across configurations
   - Measure throughput improvements

2. **Memory Stress Test**:
   - Large streaming_batch_size (50-100)
   - Various max_workers
   - Monitor peak memory usage

3. **Error Handling Test**:
   - Corrupt shard with parallel workers
   - Verify error propagation
   - Check cleanup of Pool resources

### 6.3 Performance Benchmarks

**Benchmark Script**: `tests/performance/benchmark_streaming_parallel.py` (new)

```python
import time
import pandas as pd

def benchmark_configurations():
    """Benchmark different worker configurations."""
    
    configs = [
        {'max_workers': 1, 'name': 'Sequential'},
        {'max_workers': 4, 'name': '4 Workers'},
        {'max_workers': 8, 'name': '8 Workers'},
        {'max_workers': 16, 'name': '16 Workers'},
    ]
    
    results = []
    
    for config in configs:
        start = time.time()
        
        # Run preprocessing with this config
        process_streaming_mode_preprocessing(
            input_dir='/path/to/test/shards',
            max_workers=config['max_workers'],
            # ... other params ...
        )
        
        elapsed = time.time() - start
        results.append({
            'config': config['name'],
            'time': elapsed,
            'speedup': results[0]['time'] / elapsed if results else 1.0
        })
    
    # Print results table
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

# Expected output:
#       Config   Time  Speedup
#   Sequential  230.0      1.0
#    4 Workers   68.0      3.4
#    8 Workers   36.0      6.4
#   16 Workers   28.0      8.2
```

### 6.4 Validation Checklist

**Before Deployment**:
- [ ] Unit tests pass for sequential and parallel modes
- [ ] Output consistency verified (sequential == parallel)
- [ ] Memory usage within expected bounds
- [ ] Performance benchmark shows expected speedup (>3×)
- [ ] Error handling works correctly
- [ ] Documentation updated
- [ ] Code review completed

---

## 7. Migration and Deployment

### 7.1 Backward Compatibility

**Guaranteed Compatibility**:
- ✅ All existing code continues to work unchanged
- ✅ Default behavior: if `max_workers=None`, uses `cpu_count()` (similar to batch mode)
- ✅ Can explicitly set `max_workers=1` for sequential behavior
- ✅ No breaking API changes

**Configuration Migration**:
```bash
# Before (streaming mode ignores MAX_WORKERS):
export ENABLE_TRUE_STREAMING=true
export STREAMING_BATCH_SIZE=20
export MAX_WORKERS=8  # ← Was ignored!

# After (streaming mode uses MAX_WORKERS):
export ENABLE_TRUE_STREAMING=true
export STREAMING_BATCH_SIZE=20
export MAX_WORKERS=8  # ← Now used for parallel loading!
# No config changes needed - just works better!
```

### 7.2 Rollout Strategy

**Phase 1: Development Testing** (Week 1)
- Implement changes in feature branch
- Run unit tests and integration tests
- Performance benchmarking on test data

**Phase 2: Staging Validation** (Week 2)
- Deploy to staging environment
- Run preprocessing on production-sized dataset
- Compare output with current sequential mode
- Monitor memory usage and throughput

**Phase 3: Gradual Production Rollout** (Week 3)
- Deploy with conservative settings (max_workers=4)
- Monitor performance and memory metrics
- Gradually increase max_workers based on results
- Full rollout with optimal configuration

### 7.3 Monitoring and Metrics

**Key Metrics to Track**:
```python
# Performance metrics
- preprocessing_time_seconds
- shards_per_second_throughput
- worker_utilization_percent

# Resource metrics
- peak_memory_mb
- cpu_utilization_percent
- disk_io_wait_percent

# Quality metrics
- output_row_count (should match sequential)
- data_checksum (should match sequential)
- preprocessing_errors_count
```

**Alerting Thresholds**

---
tags:
  - design
  - preprocessing
  - streaming
  - temporal_split
  - upgrade
keywords:
  - temporal split
  - streaming mode upgrade
  - 1:1 shard mapping
  - customer-level deduplication
  - OOT test
topics:
  - temporal preprocessing
  - streaming upgrade
  - shard preservation
language: python
date of note: 2026-01-21
---

# Temporal Split Preprocessing Streaming Mode Upgrade

## Overview

This document provides a detailed design for upgrading the existing `temporal_split_preprocessing.py` script to support **two distinct output modes**:

1. **Batch Mode** (Current/Default): Outputs single consolidated files per split (train.csv, val.csv, test.csv, oot.csv)
2. **Streaming Mode** (Upgraded): Outputs sharded files with 1:1 mapping (train/part-*.csv, val/part-*.csv, test/part-*.csv, oot/part-*.csv)

The script currently has partial streaming mode that processes in batches but consolidates output into single files. The upgrade will:
- **Keep batch mode unchanged** for backward compatibility
- **Enhance streaming mode** to preserve 1:1 shard mapping for PyTorch IterableDataset compatibility

**Goal**: Remove the consolidation step in streaming mode to achieve true 1:1 shard mapping (shards-in-shards-out) while maintaining the existing two-pass architecture and preserving batch mode functionality.

## Dual-Mode Architecture

The script will support **two independent output modes** controlled by the `ENABLE_TRUE_STREAMING` environment variable:

### Batch Mode (Default: `ENABLE_TRUE_STREAMING=false` or not set)

**Output Structure**:
```
output_dir/
├── training_data/
│   ├── train/
│   │   └── train_processed_data.csv    # ✅ Single consolidated file
│   ├── val/
│   │   └── val_processed_data.csv      # ✅ Single consolidated file
│   └── test/
│       └── test_processed_data.csv     # ✅ Single consolidated file (OOT)
└── oot_data/
    └── oot_data.csv                     # ✅ Single consolidated file
```

**Characteristics**:
- Traditional single-file output per split
- Backward compatible with existing pipelines
- Easier to work with for small datasets
- No changes to existing behavior

### Streaming Mode (Upgraded: `ENABLE_TRUE_STREAMING=true`)

**Output Structure**:
```
output_dir/
├── training_data/
│   ├── train/
│   │   ├── part-00001.csv    # ✅ Sharded output
│   │   ├── part-00003.csv    # ✅ 1:1 mapping with input
│   │   └── ...
│   ├── val/
│   │   ├── part-00002.csv    # ✅ Sharded output
│   │   └── ...
│   └── test/
│       ├── part-00001.csv    # ✅ Sharded output (OOT)
│       └── ...
└── oot_data/
    ├── part-00001.csv        # ✅ Sharded output
    └── ...
```

**Characteristics**:
- Sharded output with preserved input shard numbers
- 1:1 mapping: `input/part-00042.csv` → `output/train/part-00042.csv`
- PyTorch IterableDataset compatible
- Distributed training ready
- Memory efficient for large datasets

**Key Difference**: Batch mode consolidates into single files; Streaming mode preserves shards for parallel processing.

## Current Implementation Analysis

### What Works Well

The existing streaming mode has a solid foundation:

```python
# ✅ PASS 1: Customer Allocation (Good!)
def collect_customer_allocation(
    all_shards: List[Path],
    signature_columns: Optional[list],
    date_column: str,
    group_id_column: str,
    split_date: str,
    train_ratio: float,
    random_seed: int,
    log_func: Callable,
) -> tuple:
    """
    Pass 1: Scan pre-split data to collect and allocate customers.
    
    Memory: O(unique_customers) - typically ~8MB for 1M customers
    """
    # Collect unique customers from pre-split period
    all_customers = set()
    for shard in all_shards:
        df = read_shard(shard, [date_column, group_id_column])
        pre_split_df = df[df[date_column] < split_date]
        all_customers.update(pre_split_df[group_id_column].unique())
    
    # Allocate customers to train/val
    customer_list = list(all_customers)
    random.shuffle(customer_list)
    train_customers = set(customer_list[:int(len(customer_list) * train_ratio)])
    val_customers = set(customer_list[int(len(customer_list) * train_ratio):])
    
    return train_customers, val_customers


# ✅ PASS 2: Streaming Processing (Good!)
def process_streaming_temporal_split(
    all_shards: List[Path],
    training_output_path: Path,
    train_customers: set,
    val_customers: set,
    ...
):
    """
    Pass 2: Process batches with customer allocation knowledge.
    """
    split_counters = {"train": 0, "val": 0, "oot": 0}
    
    for batch_start in range(0, len(all_shards), streaming_batch_size):
        batch_shards = all_shards[batch_start:batch_end]
        batch_df = process_single_batch(batch_shards, ...)
        
        # Assign splits using GLOBAL customer allocation
        def assign_split_global(row):
            customer = row[group_id_column]
            is_pre_split = row[date_column] < split_date
            
            if is_pre_split:
                return "train" if customer in train_customers else "val"
            else:
                return "oot" if customer not in train_customers else None
        
        batch_df["_split"] = batch_df.apply(assign_split_global, axis=1)
        batch_df = batch_df[batch_df["_split"].notna()]
        
        # ✅ Write to split directories WITH shard preservation
        write_splits_to_shards(batch_df, training_output_path, split_counters, ...)
```

### The Problem: Consolidation Step

After Pass 2 completes, the script **consolidates shards** into single files:

```python
# ❌ PROBLEM: This destroys 1:1 shard mapping
def consolidate_shards_to_single_files(
    training_output_path: Path, output_format: str, log_func: Callable
) -> Dict[str, pd.DataFrame]:
    """Consolidate temporary shards into single files per split."""
    log_func("[STREAMING] Consolidating shards into single files per split...")
    
    result = {}
    for split_name in ["train", "val", "oot"]:
        split_dir = training_output_path / split_name
        
        # Read ALL shards
        shard_files = sorted(split_dir.glob(f"part-*.{output_format}"))
        shard_dfs = [read_shard(f) for f in shard_files]
        
        # Concatenate into single DataFrame
        consolidated_df = pd.concat(shard_dfs, ignore_index=True)
        
        # ❌ Write as single file (breaks 1:1 mapping)
        save_name = "test" if split_name == "oot" else split_name
        output_file = split_dir / f"{save_name}_processed_data.{output_format}"
        consolidated_df.to_csv(output_file, index=False)
        
        # ❌ Delete shard files
        for shard_file in shard_files:
            shard_file.unlink()
    
    return result


# Called after Pass 2
splits = consolidate_shards_to_single_files(training_output_path, output_format, log)
```

**Impact on Streaming Mode**:
- Destroys shard-level parallelism (negating streaming benefits)
- Makes streaming mode output identical to batch mode
- Incompatible with PyTorch IterableDataset
- Unnecessary memory overhead from concatenation
- Loses shard number traceability

**Note**: This consolidation step is only a problem for streaming mode. Batch mode intentionally produces single files and will continue to do so for backward compatibility.

## Solution: Remove Consolidation

### Required Changes

**1. Modify `write_splits_to_shards()` to preserve input shard numbers**

```python
def write_splits_to_shards(
    df: pd.DataFrame,
    output_base: Path,
    split_counters: Dict[str, int],
    shard_size: int,
    output_format: str,
    input_shard_num: int,  # ✅ NEW: Preserve input shard number
    log_func: Callable,
) -> None:
    """
    Write DataFrame to separate split directories based on '_split' column.
    
    NEW: Preserves input shard number for 1:1 mapping.
    """
    for split_name in ["train", "val", "oot"]:
        split_data = df[df["_split"] == split_name].drop("_split", axis=1)
        
        if len(split_data) == 0:
            continue
        
        split_dir = output_base / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # ✅ Use input shard number (not counter)
        output_path = split_dir / f"part-{input_shard_num:05d}.{output_format}"
        
        if output_format == "csv":
            split_data.to_csv(output_path, index=False)
        elif output_format == "tsv":
            split_data.to_csv(output_path, sep="\t", index=False)
        elif output_format == "parquet":
            split_data.to_parquet(output_path, index=False)
        
        log_func(f"[STREAMING] Wrote {output_path} ({len(split_data)} rows)")
```

**2. Update `process_streaming_temporal_split()` to pass shard numbers**

```python
def process_streaming_temporal_split(
    all_shards: List[Path],
    training_output_path: Path,
    signature_columns: Optional[list],
    date_column: str,
    group_id_column: str,
    split_date: str,
    train_customers: set,
    val_customers: set,
    targets: Optional[list],
    main_task_index: Optional[int],
    label_field: Optional[str],
    output_format: str,
    streaming_batch_size: int,
    shard_size: int,
    batch_size: int,
    log_func: Callable,
) -> None:
    """
    Pass 2: Process batches with customer allocation knowledge.
    
    MODIFIED: Preserves input shard numbers for 1:1 mapping.
    """
    log_func("[STREAMING] ===== STARTING PASS 2: Processing with 1:1 shard mapping =====")
    
    for batch_start in range(0, len(all_shards), streaming_batch_size):
        batch_end = min(batch_start + streaming_batch_size, len(all_shards))
        batch_shards = all_shards[batch_start:batch_end]
        batch_num = (batch_start // streaming_batch_size) + 1
        
        log_func(f"[PASS 2] Processing batch {batch_num} ({len(batch_shards)} shards)")
        
        # ✅ Process each shard individually to preserve numbering
        for shard_path in batch_shards:
            # Extract shard number from filename
            shard_num = extract_shard_number(shard_path)
            
            # Read single shard
            df = _read_file_to_df(shard_path, signature_columns)
            df.columns = [col.replace("__DOT__", ".") for col in df.columns]
            
            # Apply multi-task label generation if needed
            if targets and main_task_index is not None:
                df = generate_main_task_label(df, targets, main_task_index, log_func)
            
            # Convert date column
            df[date_column] = pd.to_datetime(df[date_column])
            
            # Assign splits using GLOBAL customer allocation
            def assign_split_global(row):
                customer = row[group_id_column]
                is_pre_split = row[date_column] < split_date_dt
                
                if is_pre_split:
                    if customer in train_customers:
                        return "train"
                    elif customer in val_customers:
                        return "val"
                    else:
                        return None
                else:
                    if customer not in train_customers:
                        return "oot"
                    else:
                        return None
            
            df["_split"] = df.apply(assign_split_global, axis=1)
            
            # Filter out None assignments
            initial_rows = len(df)
            df = df[df["_split"].notna()]
            filtered_rows = initial_rows - len(df)
            
            if filtered_rows > 0:
                log_func(f"[PASS 2] Shard {shard_num}: Filtered {filtered_rows} rows")
            
            # ✅ Write with preserved shard number (1:1 mapping)
            if len(df) > 0:
                for split_name in ["train", "val", "oot"]:
                    split_data = df[df["_split"] == split_name].drop("_split", axis=1)
                    
                    if len(split_data) > 0:
                        split_dir = training_output_path / split_name
                        split_dir.mkdir(parents=True, exist_ok=True)
                        
                        output_path = split_dir / f"part-{shard_num:05d}.{output_format}"
                        
                        if output_format == "csv":
                            split_data.to_csv(output_path, index=False)
                        elif output_format == "tsv":
                            split_data.to_csv(output_path, sep="\t", index=False)
                        elif output_format == "parquet":
                            split_data.to_parquet(output_path, index=False)
            
            del df
            gc.collect()
    
    log_func("[STREAMING] ===== PASS 2 COMPLETE =====")


# Helper function to extract shard number
def extract_shard_number(shard_path: Path) -> int:
    """Extract shard number from filename like part-00042.csv"""
    stem = shard_path.stem
    if stem.endswith('.gz'):
        stem = Path(stem).stem
    
    # Extract number from part-XXXXX pattern
    import re
    match = re.search(r'part-(\d+)', stem)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Cannot extract shard number from {shard_path.name}")
```

**3. Remove consolidation step from main function**

```python
def process_streaming_mode_temporal_split(...):
    """
    Streaming mode for temporal split preprocessing.
    
    MODIFIED: No consolidation step - preserves 1:1 shard mapping.
    """
    # Pass 1: Collect customer allocation
    train_customers, val_customers = collect_customer_allocation(...)
    
    # Pass 2: Process shards with 1:1 mapping
    process_streaming_temporal_split(
        all_shards,
        training_output_path,
        signature_columns,
        date_column,
        group_id_column,
        split_date,
        train_customers,
        val_customers,
        targets,
        main_task_index,
        label_field,
        output_format,
        streaming_batch_size or 10,
        shard_size,
        batch_size,
        log_func,
    )
    
    # ❌ REMOVE THIS: consolidate_shards_to_single_files(...)
    
    # ✅ Save OOT data separately (already in sharded form)
    oot_output_dir = output_paths.get("oot_data", ...)
    oot_output_path = Path(oot_output_dir)
    oot_output_path.mkdir(parents=True, exist_ok=True)
    
    # Create symlink or copy shards from training_output_path/oot to oot_output_path
    oot_source_dir = training_output_path / "oot"
    if oot_source_dir.exists():
        for shard_file in oot_source_dir.glob(f"part-*.{output_format}"):
            dest_file = oot_output_path / shard_file.name
            shutil.copy2(shard_file, dest_file)
    
    log_func("[STREAMING] Temporal split preprocessing complete in streaming mode")
    
    return {}  # Data written to disk in sharded form
```

## Output Structure Comparison

### Before Upgrade (Current Streaming Mode)

```
output_dir/
├── training_data/
│   ├── train/
│   │   └── train_processed_data.csv          # ❌ Single consolidated file
│   ├── val/
│   │   └── val_processed_data.csv            # ❌ Single consolidated file
│   └── test/
│       └── test_processed_data.csv           # ❌ Single consolidated file (OOT)
└── oot_data/
    └── oot_data.csv                           # ❌ Single consolidated file
```

### After Upgrade (True Streaming Mode)

```
output_dir/
├── training_data/
│   ├── train/
│   │   ├── part-00001.csv    # ✅ From input part-00001.csv (train customers only)
│   │   ├── part-00003.csv    # ✅ From input part-00003.csv (train customers only)
│   │   ├── part-00005.csv    # ✅ From input part-00005.csv (train customers only)
│   │   └── ...               # Sparse numbering (gaps where shards had 0 train rows)
│   ├── val/
│   │   ├── part-00002.csv    # ✅ From input part-00002.csv (val customers only)
│   │   ├── part-00004.csv    # ✅ From input part-00004.csv (val customers only)
│   │   └── ...
│   └── test/
│       ├── part-00001.csv    # ✅ From input part-00001.csv (OOT customers only)
│       ├── part-00006.csv    # ✅ From input part-00006.csv (OOT customers only)
│       └── ...
└── oot_data/
    ├── part-00001.csv        # ✅ Copy/symlink from training_data/test/
    ├── part-00006.csv        # ✅ Copy/symlink from training_data/test/
    └── ...
```

## Environment Variables

No new environment variables needed - all existing variables work:

```bash
# Temporal Split Parameters
DATE_COLUMN="transaction_date"
GROUP_ID_COLUMN="customer_id"
SPLIT_DATE="2025-01-01"
TRAIN_RATIO=0.9
RANDOM_SEED=42

# Streaming Control (Existing)
ENABLE_TRUE_STREAMING="true"
STREAMING_BATCH_SIZE=10
SHARD_SIZE=100000
MAX_WORKERS=8
OUTPUT_FORMAT="csv"

# Optional Parameters
TARGETS="is_abuse,is_abusive_dnr,is_abusive_pda"
MAIN_TASK_INDEX=0
LABEL_FIELD="label"
```

## Benefits of Upgrade

### Performance
- ✅ **No change** - Pass 1 and Pass 2 remain the same speed
- ✅ **Removes consolidation overhead** (~10-20s saved for large datasets)
- ✅ **Total time reduced by ~20%** (eliminate concatenation + write step)

### Memory
- ✅ **Peak memory reduced** - No need to load all shards for consolidation
- ✅ **Predictable memory usage** - Only processes streaming batch size

### Architecture
- ✅ **True 1:1 shard mapping** - Input part-00042.csv → Output train/part-00042.csv
- ✅ **Sparse numbering** - Gaps in output indicate shards with zero rows for that split
- ✅ **Traceability** - Can trace any output row back to input shard

### PyTorch Compatibility
- ✅ **IterableDataset ready** - Sharded output works directly with IterableDataset
- ✅ **Distributed training ready** - Each worker can load different shards
- ✅ **Efficient streaming** - No need to load full dataset into memory

## Implementation Plan

### Phase 1: Code Changes (2-3 hours)

**Step 1**: Add `extract_shard_number()` helper function
```python
def extract_shard_number(shard_path: Path) -> int:
    """Extract shard number from filename"""
    # Implementation shown above
```

**Step 2**: Modify `write_splits_to_shards()` signature
```python
# Add input_shard_num parameter
def write_splits_to_shards(..., input_shard_num: int, ...):
    # Use input_shard_num instead of counter
```

**Step 3**: Update `process_streaming_temporal_split()`
```python
# Process shards individually
for shard_path in batch_shards:
    shard_num = extract_shard_number(shard_path)
    # ... process shard ...
    # Write with shard_num
```

**Step 4**: Remove consolidation call
```python
# Delete or comment out:
# consolidate_shards_to_single_files(...)
```

**Step 5**: Update OOT data handling
```python
# Copy/symlink shards instead of single file
for shard_file in oot_source_dir.glob(f"part-*.{output_format}"):
    shutil.copy2(shard_file, oot_output_path / shard_file.name)
```

### Phase 2: Testing (1-2 hours)

**Test 1**: Verify shard preservation
```bash
# Check that input and output shard numbers match
ls input_dir/ | sort > input_shards.txt
ls output_dir/train/ | sort > output_train_shards.txt
# Compare shard numbers
```

**Test 2**: Verify row counts
```python
# Compare total rows between old and new modes
old_mode_rows = count_rows_in_single_file("train_processed_data.csv")
new_mode_rows = sum(count_rows_in_shard(s) for s in glob("train/part-*.csv"))
assert old_mode_rows == new_mode_rows
```

**Test 3**: Verify split correctness
```python
# Verify no customer leakage
train_customers = get_customers_from_shards("train/part-*.csv")
oot_customers = get_customers_from_shards("test/part-*.csv")
assert len(train_customers & oot_customers) == 0  # No overlap
```

**Test 4**: PyTorch IterableDataset test
```python
from projects.names3risk_pytorch.dockers.processing.datasets.pipeline_iterable_datasets import (
    PipelineIterableDataset
)

# Test loading sharded data
dataset = PipelineIterableDataset(
    data_dir="output_dir/train",
    file_pattern="part-*.csv",
    ...
)

# Verify loading works
for batch in DataLoader(dataset, batch_size=32):
    assert batch is not None
    break
```

### Phase 3: Documentation (30 minutes)

**Update docstrings**:
- `process_streaming_temporal_split()` - Note 1:1 shard mapping
- `write_splits_to_shards()` - Document input_shard_num parameter

**Update README/Wiki**:
- Document new output structure (sharded vs single file)
- Add migration notes for existing pipelines

## Backward Compatibility

### Batch Mode Unchanged
```python
if enable_true_streaming:
    # New streaming mode with 1:1 mapping
    process_streaming_mode_temporal_split(...)
else:
    # Original batch mode (single files)
    process_batch_mode_temporal_split(...)
```

### Migration Path

**Option 1**: New pipelines use streaming mode by default
```bash
ENABLE_TRUE_STREAMING="true"
```

**Option 2**: Existing pipelines keep batch mode
```bash
# Don't set ENABLE_TRUE_STREAMING or set to "false"
```

**Option 3**: Gradual migration with validation
```bash
# Run both modes in parallel
# Compare outputs
# Switch to streaming once validated
```

## Edge Cases

### Empty Shards After Filtering
```python
# Some shards may have 0 rows after customer filtering
# Result: Sparse output numbering (gaps in sequence)

# Example:
# Input:  part-00001.csv, part-00002.csv, part-00003.csv
# Output: train/part-00001.csv (50 rows)
#         train/part-00003.csv (30 rows)  # part-00002 had 0 train customers
#         val/part-00002.csv (40 rows)
```

### Very Small Shards
```python
# If a shard has only 1-2 rows, still write it
# PyTorch DataLoader handles small batches gracefully
```

### All Customers in Train (No OOT)
```python
# If all post-split customers were in training:
# test/ directory will be empty or have very few shards
# This is expected behavior - log warning
log_func("[WARNING] OOT (test) data is very small or empty after filtering")
```

## Testing Checklist

- [ ] Verify shard number preservation (input → output)
- [ ] Verify total row counts match old mode
- [ ] Verify no customer leakage (train ∩ OOT = ∅)
- [ ] Verify temporal split correctness (dates < split_date → train/val)
- [ ] Test PyTorch IterableDataset loading
- [ ] Test with empty shards (0 rows after filtering)
- [ ] Test with very small shards (1-2 rows)
- [ ] Test with all three output formats (CSV, TSV, Parquet)
- [ ] Benchmark performance vs old streaming mode
- [ ] Validate memory usage unchanged

## Phase 3: Parallel Streaming Architecture (NEXT UPGRADE)

### Current Bottleneck: Sequential Pass 2

After implementing 1:1 shard mapping, Pass 2 is still sequential:

```python
# ❌ Current: Sequential processing
for shard_path in all_shards:  # One at a time
    shard_num = extract_shard_number(shard_path)
    df = _read_file_to_df(shard_path, ...)
    # ... process shard ...
```

**Performance**: 100 shards × 30 seconds/shard = **50 minutes**

### Parallel Architecture Design

Inspired by `tabular_preprocessing.py` fully parallel mode (lines 1350-1450):

```python
# ✅ NEW: Parallel processing with multiprocessing.Pool
with Pool(processes=max_workers) as pool:
    results = pool.map(process_shard_temporal_split, shard_args)
```

**Performance**: 100 shards / 8 workers × 30 seconds/shard = **6.25 minutes** (~8x speedup!)

### Key Design Decision: Customer Mapping Dictionary

**Problem**: Workers need customer allocation, but sets require two lookups

**Current Approach** (sets):
```python
train_customers = {"cust_1", "cust_2", ...}  # Set 1
val_customers = {"cust_3", "cust_4", ...}    # Set 2

# Requires checking both sets
if customer in train_customers:
    return "train"
elif customer in val_customers:
    return "val"
```

**NEW Approach** (dictionary):
```python
customer_split_map = {
    "cust_1": "train",
    "cust_2": "train", 
    "cust_3": "val",
    "cust_4": "val",
    ...
}

# Single lookup!
split = customer_split_map.get(customer)  # Returns "train" or "val" or None
```

**Benefits**:
- ✅ Single dictionary lookup vs two set checks
- ✅ More explicit (no implicit "not in train → val" logic)
- ✅ Easier to debug and validate
- ✅ Cleaner code in parallel workers

**Memory Impact**: ~3x more (50 MB vs 16 MB for 1M customers) - still very manageable

### Enhanced Pass 1: Build Customer Mapping Dictionary

```python
def collect_customer_allocation(
    all_shards: List[Path],
    signature_columns: Optional[list],
    date_column: str,
    group_id_column: str,
    split_date: str,
    train_ratio: float,
    random_seed: int,
    log_func: Callable,
) -> Dict[str, str]:  # ✅ Returns dictionary, not tuple of sets
    """
    Pass 1: Scan pre-split data and build complete customer allocation map.
    
    Returns:
        Dictionary mapping customer_id to split assignment
        Example: {"customer_1": "train", "customer_2": "val", ...}
        
    Memory: ~50 bytes per customer (~50 MB for 1M customers)
    """
    log_func("[PASS 1] Building customer→split mapping...")
    
    split_date_dt = pd.to_datetime(split_date)
    all_customers = set()
    
    # Collect all unique customers from pre-split period
    for shard in all_shards:
        # Read only 2 columns for efficiency
        if shard.suffix == ".parquet":
            df = pd.read_parquet(shard, columns=[date_column, group_id_column])
        else:
            df = _read_file_to_df(shard, signature_columns)
            df = df[[date_column, group_id_column]]
        
        df[date_column] = pd.to_datetime(df[date_column])
        pre_split_df = df[df[date_column] < split_date_dt]
        all_customers.update(pre_split_df[group_id_column].unique())
        del df, pre_split_df
        gc.collect()
    
    log_func(f"[PASS 1] Found {len(all_customers)} unique customers")
    
    # Allocate customers to train/val
    customer_list = list(all_customers)
    random.seed(random_seed)
    random.shuffle(customer_list)
    
    train_size = int(len(customer_list) * train_ratio)
    
    # ✅ Build dictionary mapping
    customer_split_map = {}
    for i, customer in enumerate(customer_list):
        customer_split_map[customer] = "train" if i < train_size else "val"
    
    # Memory usage estimate
    memory_mb = len(customer_split_map) * 50 / 1024 / 1024
    log_func(f"[PASS 1] Map size: ~{memory_mb:.2f} MB")
    log_func(f"[PASS 1] Allocated: {train_size} train ({train_ratio*100:.1f}%)")
    log_func(f"[PASS 1] Allocated: {len(customer_list)-train_size} val")
    
    return customer_split_map
```

### Parallel Pass 2: Worker Function with Vectorized Operations

**Critical**: Use vectorized operations, NOT `df.apply()` (20-30x faster!)

```python
def process_shard_temporal_split(args: tuple) -> Dict[str, int]:
    """
    Process single shard with global customer allocation map.
    Uses VECTORIZED operations for performance (not df.apply).
    
    This function runs in parallel across multiple workers.
    
    Args:
        args: Tuple of (shard_path, shard_num, customer_split_map, config)
    
    Returns:
        Statistics dict with row counts per split
    """
    shard_path, shard_num, customer_split_map, config = args
    
    # Extract config
    date_column = config["date_column"]
    group_id_column = config["group_id_column"]
    split_date = config["split_date"]
    output_base = config["output_base"]
    output_format = config["output_format"]
    signature_columns = config.get("signature_columns")
    targets = config.get("targets")
    main_task_index = config.get("main_task_index")
    label_field = config.get("label_field")
    
    # Read shard
    df = _read_file_to_df(shard_path, signature_columns)
    df.columns = [col.replace("__DOT__", ".") for col in df.columns]
    
    # Apply preprocessing (multi-task labels, label processing, etc.)
    if targets and main_task_index is not None:
        df = generate_main_task_label(df, targets, main_task_index, lambda x: None)
    
    if label_field and label_field in df.columns:
        df = process_label_column(df, label_field, lambda x: None)
    
    # Convert date column
    df[date_column] = pd.to_datetime(df[date_column])
    split_date_dt = pd.to_datetime(split_date)
    
    # ============================================
    # VECTORIZED TEMPORAL SPLIT ASSIGNMENT
    # ============================================
    
    # Step 1: Determine pre-split vs post-split (vectorized boolean)
    is_pre_split = df[date_column] < split_date_dt
    
    # Step 2: Map customer IDs to split assignments (vectorized)
    # customer_split_map.get() returns "train" or "val" or None
    customer_splits = df[group_id_column].map(customer_split_map)
    
    # Step 3: Assign final splits using vectorized conditions
    df["_split"] = None  # Default: will be filtered out
    
    # Pre-split data: use customer assignment ("train" or "val")
    df.loc[is_pre_split, "_split"] = customer_splits[is_pre_split]
    
    # Post-split data: assign "oot" only if customer is NOT in train set
    # (i.e., customer is in val set or unknown)
    is_post_split = ~is_pre_split
    is_not_train = customer_splits != "train"  # Val or None
    df.loc[is_post_split & is_not_train, "_split"] = "oot"
    
    # Step 4: Filter out None assignments (train customers in post-split period)
    initial_rows = len(df)
    df = df[df["_split"].notna()]
    filtered_rows = initial_rows - len(df)
    
    # Log filtering stats (will be captured by parent process)
    if filtered_rows > 0:
        print(f"[Shard {shard_num}] Filtered {filtered_rows} rows "
              f"({filtered_rows/initial_rows*100:.1f}%)")
    
    # Step 5: Write to split folders with preserved shard number
    stats = {}
    for split_name in ["train", "val", "oot"]:
        split_df = df[df["_split"] == split_name].drop("_split", axis=1)
        
        if len(split_df) > 0:
            split_dir = output_base / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = split_dir / f"part-{shard_num:05d}.{output_format}"
            
            if output_format == "csv":
                split_df.to_csv(output_path, index=False)
            elif output_format == "tsv":
                split_df.to_csv(output_path, sep="\t", index=False)
            elif output_format == "parquet":
                split_df.to_parquet(output_path, index=False)
            
            stats[split_name] = len(split_df)
            print(f"[Shard {shard_num}] Wrote {output_path.name} ({len(split_df)} rows)")
        else:
            stats[split_name] = 0
    
    return stats
```

### Main Parallel Processing Function

```python
def process_streaming_temporal_split_parallel(
    all_shards: List[Path],
    training_output_path: Path,
    customer_split_map: Dict[str, str],  # ✅ Pass the dictionary!
    config: Dict,
    max_workers: int,
    log_func: Callable,
) -> None:
    """
    Pass 2: Process all shards in parallel using customer map.
    
    Args:
        all_shards: List of all input shard paths
        training_output_path: Base output directory
        customer_split_map: Customer→split mapping from Pass 1
        config: Configuration dictionary
        max_workers: Number of parallel workers
        log_func: Logging function
    """
    log_func(f"[PASS 2] Processing {len(all_shards)} shards with {max_workers} workers")
    log_func("[PASS 2] Using PARALLEL processing with vectorized operations")
    
    # Prepare arguments for each shard
    shard_args = [
        (shard, extract_shard_number(shard), customer_split_map, config)
        for shard in all_shards
    ]
    
    # ✅ Process ALL shards in parallel
    with Pool(processes=max_workers) as pool:
        results = pool.map(process_shard_temporal_split, shard_args)
    
    # Aggregate statistics
    total_stats = {
        "train": sum(r.get("train", 0) for r in results),
        "val": sum(r.get("val", 0) for r in results),
        "oot": sum(r.get("oot", 0) for r in results),
    }
    
    # Count non-empty shards per split
    shard_counts = {
        "train": sum(1 for r in results if r.get("train", 0) > 0),
        "val": sum(1 for r in results if r.get("val", 0) > 0),
        "oot": sum(1 for r in results if r.get("oot", 0) > 0),
    }
    
    log_func(f"[PASS 2] Complete! Row distribution: {total_stats}")
    log_func(f"[PASS 2] Output shards - train={shard_counts['train']}, "
             f"val={shard_counts['val']}, oot={shard_counts['oot']}")
```

### Performance Comparison

| Mode | Pass 1 | Pass 2 | Total | Speedup |
|------|--------|--------|-------|---------|
| **Sequential** | 5 min | 50 min | 55 min | 1x (baseline) |
| **Parallel (8 workers)** | 5 min | 6.25 min | 11.25 min | **~5x faster** |

**Key Insights**:
- Pass 1 remains sequential (already efficient at ~5 min)
- Pass 2 gets ~8x speedup from parallelization
- Overall speedup ~5x (Pass 1 is small portion of total time)
- Vectorized operations provide additional 20-30x speedup within each worker

### Comparison with Tabular Preprocessing

| Aspect | Tabular Preprocessing | Temporal Split (Parallel) |
|--------|----------------------|---------------------------|
| **Pass 1** | Not needed | Customer allocation (5 min) |
| **Pass 2** | Fully parallel | Fully parallel (same pattern) |
| **Split Logic** | Random/stratified per-shard | Uses global customer map |
| **Coordination** | None needed | Customer map passed to workers |
| **Shard Ordering** | Not required | Not required |
| **Implementation** | `process_shard_end_to_end_generic()` | `process_shard_temporal_split()` |

**Why Parallel Works**:
1. ✅ Customer allocation done globally in Pass 1
2. ✅ Each shard processed independently in Pass 2
3. ✅ No inter-shard coordination needed
4. ✅ Same pattern as tabular_preprocessing.py

**Why Customer Map is Better Than Sets**:
1. ✅ Cleaner code (single lookup vs two set checks)
2. ✅ More explicit assignments
3. ✅ Easier to pass to workers (one object vs two)
4. ✅ Better debugging (can inspect the map)

### Implementation Checklist

**Phase 3A: Enhanced Pass 1** (1 hour)
- [ ] Modify `collect_customer_allocation()` to return dictionary
- [ ] Update return type: `-> Dict[str, str]`
- [ ] Build dictionary instead of two sets
- [ ] Add memory usage logging

**Phase 3B: Vectorized Worker Function** (2 hours)
- [ ] Create `process_shard_temporal_split()` function
- [ ] Implement vectorized split assignment (NOT df.apply)
- [ ] Add shard number extraction
- [ ] Write outputs with preserved shard numbers
- [ ] Return statistics dictionary

**Phase 3C: Parallel Orchestration** (1 hour)
- [ ] Create `process_streaming_temporal_split_parallel()` function
- [ ] Prepare shard arguments with customer map
- [ ] Use `multiprocessing.Pool.map()` for parallelization
- [ ] Aggregate and log statistics
- [ ] Update main function to call parallel version

**Phase 3D: Testing** (1-2 hours)
- [ ] Test with 8 workers vs sequential
- [ ] Verify row counts match
- [ ] Verify no customer leakage
- [ ] Verify shard number preservation
- [ ] Benchmark performance improvement
- [ ] Test with different worker counts

**Total Effort**: ~5-6 hours for parallel upgrade

## Conclusion

This design provides **two upgrade paths**:

### Immediate (Phase 1-2): 1:1 Shard Mapping
- ✅ **Low risk** - Only removes consolidation step
- ✅ **Quick win** - 2-3 hours implementation
- ✅ **20% speedup** - Eliminates consolidation overhead
- ✅ **PyTorch ready** - Enables IterableDataset compatibility

### Future (Phase 3): Parallel Processing
- ✅ **Medium effort** - 5-6 hours implementation
- ✅ **High impact** - ~5x overall speedup (8x in Pass 2)
- ✅ **Proven pattern** - Same approach as tabular_preprocessing.py
- ✅ **Scalable** - Speedup proportional to worker count

**Recommendation**:
1. **Implement Phase 1-2 immediately** for PyTorch compatibility
2. **Implement Phase 3 next sprint** for performance gains

Both upgrades are **low risk** with **high value**, providing immediate benefits for the Names3Risk PyTorch pipeline while preparing for distributed training at scale.

---

**Document Status**: Complete Design with Parallel Architecture  
**Estimated Effort**:
- Phase 1-2 (1:1 Mapping): 4-5 hours
- Phase 3 (Parallelization): 5-6 hours
- **Total**: 9-11 hours

**Last Updated**: 2026-01-21  
**Related Files**:
- Implementation: `src/cursus/steps/scripts/temporal_split_preprocessing.py`
- Reference: `src/cursus/steps/scripts/tabular_preprocessing.py` (lines 1350-1450)
- Dataset: `projects/names3risk_pytorch/dockers/processing/datasets/pipeline_iterable_datasets.py`

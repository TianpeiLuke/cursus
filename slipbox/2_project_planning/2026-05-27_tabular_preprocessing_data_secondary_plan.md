# Plan: TabularPreprocessing — DATA_SECONDARY Support in Script

**Date**: 2026-05-27
**Status**: 🟡 Ready to implement
**Prerequisite**: Spec + contract already updated (commit `1a436cf`). Builder already handles optional deps.
**Motivation**: FZ 29d12 — two CradleDataLoading nodes feed one TabularPreprocessing. Script currently reads only `input_paths["DATA"]`, ignoring `DATA_SECONDARY`.

## Component Status (5-Layer Check)

| Component | Status | Needs Change? | Evidence |
|-----------|--------|:-------------:|----------|
| **Spec** | ✅ Done | No | `DATA_SECONDARY` DependencySpec (required=False) already in `tabular_preprocessing_spec.py` |
| **Contract** | ✅ Done | No | `"DATA_SECONDARY": "/opt/ml/processing/input/data_secondary"` in `expected_input_paths` |
| **Builder** | ✅ Works | No | Line 176: `if not dependency_spec.required and logical_name not in inputs: continue` — skips when absent, wires when assembler provides it |
| **Config** | ✅ N/A | No | Input paths are resolved by assembler from DAG edges, not from config fields. No config field needed for `DATA_SECONDARY`. |
| **Script** | ❌ Missing | **YES** | Only reads `input_paths["DATA"]` (line 1912). Ignores `DATA_SECONDARY` entirely. |

## What Needs to Change (Script Only)

The script (`tabular_preprocessing.py`) must:
1. Read `DATA_SECONDARY` from `input_paths` when present
2. Load shards from BOTH directories and combine them
3. Pass the secondary path to `__main__` block

## Current State (Lines 1912-1913)

```python
input_data_dir = input_paths["DATA"]
input_signature_dir = input_paths["SIGNATURE"]
```

Only reads from `DATA`. If `DATA_SECONDARY` is present, it's ignored.

## Changes

### 1. Update `main()` — Read DATA_SECONDARY (~5 lines)

**Location**: After line 1913, before output_dir extraction

```python
input_data_dir = input_paths["DATA"]
input_data_secondary_dir = input_paths.get("DATA_SECONDARY")  # NEW — optional
input_signature_dir = input_paths["SIGNATURE"]
output_dir = output_paths["processed_data"]
```

### 2. Update Batch Mode — Combine Shards from Both Dirs (~10 lines)

**In `main()` where it calls `process_batch_mode_preprocessing`**, pass the secondary dir. In `process_batch_mode_preprocessing`, combine shards from both:

**Option A (simplest)**: Merge shards BEFORE calling processing function:
```python
# In main(), before routing to batch/streaming mode:
if input_data_secondary_dir and Path(input_data_secondary_dir).exists():
    log(f"[INFO] Secondary data input detected: {input_data_secondary_dir}")
    # For batch mode: will combine shards from both dirs
```

**Option B (cleaner)**: Update `combine_shards()` to accept a list of directories:
```python
def combine_shards(
    input_dirs,  # Changed from str to Union[str, List[str]]
    column_names=None,
    max_workers=None,
    batch_size=5,
    streaming_batch_size=None,
):
    if isinstance(input_dirs, str):
        input_dirs = [input_dirs]
    
    all_shards = []
    for input_dir in input_dirs:
        input_path = Path(input_dir)
        patterns = ["*.csv", "*.tsv", "*.json", "*.parquet", ...]
        all_shards.extend(sorted(set(p for pat in patterns for p in input_path.glob(pat))))
    ...
```

**Recommendation**: Option A is simpler and less risky (doesn't change `combine_shards` signature):

```python
# In main(), after extracting paths:
input_data_dir = input_paths["DATA"]
input_data_secondary_dir = input_paths.get("DATA_SECONDARY")
input_signature_dir = input_paths["SIGNATURE"]
output_dir = output_paths["processed_data"]

# If secondary input exists, merge its files into primary before processing
if input_data_secondary_dir and Path(input_data_secondary_dir).exists():
    secondary_path = Path(input_data_secondary_dir)
    primary_path = Path(input_data_dir)
    secondary_files = list(secondary_path.rglob("*"))
    data_files = [f for f in secondary_files if f.is_file() and f.suffix in (".csv", ".tsv", ".json", ".parquet", ".gz")]
    if data_files:
        log(f"[INFO] DATA_SECONDARY: found {len(data_files)} files in {input_data_secondary_dir}")
        # Symlink or copy secondary files into primary dir for unified processing
        for f in data_files:
            dest = primary_path / f"_secondary_{f.name}"
            if not dest.exists():
                os.symlink(f, dest)
        log(f"[INFO] DATA_SECONDARY: linked {len(data_files)} files into primary input dir")
```

**Problem with symlinks**: May not work in SageMaker containers (read-only input paths).

**Better approach**: Just pass both dirs to `combine_shards`:

```python
# In main(), build list of input dirs:
input_dirs = [input_data_dir]
if input_data_secondary_dir and Path(input_data_secondary_dir).exists():
    input_dirs.append(input_data_secondary_dir)
    log(f"[INFO] DATA_SECONDARY present: will combine from {len(input_dirs)} input directories")
```

Then update `combine_shards` to accept `Union[str, List[str]]`.

### 3. Update `combine_shards()` (~5 lines)

**Current signature** (line ~400):
```python
def combine_shards(input_data_dir, column_names=None, max_workers=None, batch_size=5, streaming_batch_size=None):
```

**New signature**:
```python
def combine_shards(input_data_dir, column_names=None, max_workers=None, batch_size=5, streaming_batch_size=None):
    # Support list of directories
    if isinstance(input_data_dir, list):
        input_dirs = input_data_dir
    else:
        input_dirs = [input_data_dir]
    
    all_shards = []
    for dir_path in input_dirs:
        input_path = Path(dir_path)
        patterns = ["*.csv", "*.csv.gz", "*.tsv", "*.tsv.gz", "*.json", "*.json.gz", "*.parquet", "*.snappy.parquet"]
        dir_shards = sorted(set(p for pat in patterns for p in input_path.glob(pat)))
        all_shards.extend(dir_shards)
    
    # ... rest unchanged (reads from all_shards list)
```

### 4. Update `__main__` Block (~3 lines)

```python
INPUT_DATA_DIR = "/opt/ml/processing/input/data"
INPUT_DATA_SECONDARY_DIR = "/opt/ml/processing/input/data_secondary"  # NEW
INPUT_SIGNATURE_DIR = "/opt/ml/processing/input/signature"

# ...

input_paths = {
    "DATA": INPUT_DATA_DIR,
    "DATA_SECONDARY": INPUT_DATA_SECONDARY_DIR,  # NEW
    "SIGNATURE": INPUT_SIGNATURE_DIR,
}
```

### 5. Update Streaming Mode (if needed)

The `process_fully_parallel_mode_preprocessing_generic` function also takes `input_dir`. Same pattern: accept list of dirs and collect shards from all.

**However**: For Munged Address, we use batch mode (dataset is ~100K rows, fits in memory). Streaming mode update can be deferred.

## Backward Compatibility

- When `DATA_SECONDARY` is not provided (not in `input_paths` or path doesn't exist), behavior is identical to current
- `combine_shards` with a single-element list `[input_data_dir]` produces same result as current `combine_shards(input_data_dir)`
- Existing pipelines have no `DATA_SECONDARY` in their config → assembler doesn't wire it → not in `input_paths` → skipped

## Total Effort

| File | Change | Lines |
|------|--------|-------|
| `tabular_preprocessing.py` — `main()` | Read `DATA_SECONDARY`, build dir list | ~5 |
| `tabular_preprocessing.py` — `combine_shards()` | Accept list of dirs | ~8 |
| `tabular_preprocessing.py` — `process_batch_mode_preprocessing()` | Pass dir list | ~2 |
| `tabular_preprocessing.py` — `__main__` | Add secondary path | ~3 |
| **Total** | | **~18 lines** |

## Verification

1. Syntax check + ruff format
2. Test: no `DATA_SECONDARY` → original behavior
3. Test: with `DATA_SECONDARY` → reads from both dirs, combines
4. brazil-build release

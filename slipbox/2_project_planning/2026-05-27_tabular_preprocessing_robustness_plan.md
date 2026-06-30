# Plan: TabularPreprocessing Script — Production Robustness Improvements

**Date**: 2026-05-27
**Status**: 🟡 Ready to implement
**Motivation**: FZ 29d13 identified 7 optimization areas in the ~2000-line script. None block Munged Address pipeline — all are production hardening for large-scale Cradle data processing.
**Total effort**: ~52 lines across the script (no config/contract/spec/builder changes needed)

## Priority Overview

| # | Optimization | Effort | Priority | Risk |
|---|---|---|---|---|
| 1 | Error recovery per shard | 15 lines | P1 | Low |
| 2 | Progress tracking + ETA | 8 lines | P2 | None |
| 3 | Schema validation across shards | 10 lines | P2 | None |
| 4 | Empty shard skip | 1 line | P3 | None |
| 5 | DATA_SECONDARY dedup | 3 lines | P3 | None |
| 6 | Output verification | 5 lines | P3 | None |
| 7 | ThreadPool for I/O reads | 10 lines | P2 | Low |

## Phase 1: P1 — Error Recovery Per Shard

**Problem**: A single corrupted shard kills the entire multiprocessing Pool → pipeline failure.

**File**: `src/cursus/steps/scripts/tabular_preprocessing.py`

### Change 1a: Update `_read_shard_wrapper` (line ~231)

```python
def _read_shard_wrapper(args: tuple) -> dict:
    """Wrapper for parallel shard reading with error isolation."""
    shard_path, signature_columns, idx, total = args
    try:
        df = _read_file_to_df(shard_path, signature_columns)
        print(f"[INFO] Processed shard {idx + 1}/{total}: {shard_path.name} ({df.shape[0]} rows)")
        return {"status": "success", "df": df, "path": str(shard_path)}
    except Exception as e:
        print(f"[WARNING] Failed to read shard {shard_path.name}: {e}")
        return {"status": "error", "error": str(e), "path": str(shard_path)}
```

### Change 1b: Update `combine_shards` — after Pool.map (line ~440)

```python
# After pool.map or sequential processing:
results = pool.map(_read_shard_wrapper, shard_args)

# Separate successes from failures
failures = [r for r in results if r["status"] == "error"]
dfs = [r["df"] for r in results if r["status"] == "success"]

if failures:
    print(f"[WARNING] {len(failures)}/{len(results)} shards failed:")
    for f in failures[:5]:  # Log first 5
        print(f"  - {f['path']}: {f['error']}")
    failure_rate = len(failures) / len(results)
    max_rate = 0.05  # 5% tolerance
    if failure_rate > max_rate:
        raise RuntimeError(
            f"Shard failure rate {failure_rate:.1%} exceeds {max_rate:.1%} threshold. "
            f"Failed: {[f['path'] for f in failures]}"
        )
```

**Note**: This changes the return type of `_read_shard_wrapper` from `pd.DataFrame` to `dict`. All callers in `combine_shards` and `_combine_shards_streaming` need updating.

## Phase 2: P2 — Progress, Schema, ThreadPool (Parallel)

### Change 2a: Progress Tracking (~8 lines)

In `_combine_shards_streaming`, add after each batch:
```python
import time

# At start:
start_time = time.time()

# After each batch:
elapsed = time.time() - start_time
rate = (batch_end) / elapsed if elapsed > 0 else 0
remaining = total_shards - batch_end
eta = remaining / rate if rate > 0 else 0
print(f"[PROGRESS] {batch_end}/{total_shards} shards ({batch_end/total_shards:.0%}), "
      f"rate={rate:.1f} shards/s, ETA={eta:.0f}s")
```

### Change 2b: Schema Validation (~10 lines)

In `combine_shards`, after collecting all DataFrames:
```python
# Validate schema consistency
if dfs:
    reference_cols = set(dfs[0].columns)
    for i, df in enumerate(dfs[1:], 1):
        if set(df.columns) != reference_cols:
            extra = set(df.columns) - reference_cols
            missing = reference_cols - set(df.columns)
            print(f"[WARNING] Shard {i} schema mismatch: extra={extra}, missing={missing}")
```

### Change 2c: ThreadPoolExecutor for I/O (~10 lines)

Replace `multiprocessing.Pool` with `concurrent.futures.ThreadPoolExecutor` in `combine_shards`:

```python
from concurrent.futures import ThreadPoolExecutor

# Replace:
# with Pool(processes=max_workers) as pool:
#     dfs = pool.map(_read_shard_wrapper, shard_args)

# With:
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    results = list(executor.map(_read_shard_wrapper, shard_args))
```

**Why**: Shard reading is I/O-bound (disk/S3). Threads avoid IPC serialization of large DataFrames between processes. GIL is released during I/O.

**Caveat**: Keep `multiprocessing.Pool` for streaming mode's per-shard PROCESSING (CPU-bound transforms). Only switch the READING phase to threads.

## Phase 3: P3 — Quick Fixes (Trivial)

### Change 3a: Empty Shard Skip (1 line)

In `combine_shards`, after collecting `all_shards`:
```python
all_shards = [s for s in all_shards if s.stat().st_size > 0]
```

### Change 3b: DATA_SECONDARY Dedup (3 lines)

In `combine_shards`, when building from multiple dirs:
```python
if len(input_dirs) > 1:
    seen_names = set()
    all_shards = [s for s in all_shards if s.name not in seen_names and not seen_names.add(s.name)]
```

### Change 3c: Output Verification (5 lines)

In `process_batch_mode_preprocessing`, after writing splits:
```python
for split_name in splits:
    proc_path = output_path / split_name / f"{split_name}_processed_data.{output_format}"
    if not proc_path.exists() or proc_path.stat().st_size == 0:
        raise RuntimeError(f"Output verification failed: {proc_path}")
    log(f"[VERIFY] {proc_path}: {proc_path.stat().st_size / 1024:.1f} KB ✓")
```

## Dependency Graph

```
Phase 1 (error recovery) ← Must be first (changes return type of wrapper)
    ↓
Phase 2a (progress)     ──┐
Phase 2b (schema)         ├── Independent, parallel
Phase 2c (threadpool)   ──┘
    ↓
Phase 3a-3c (quick fixes) ── Independent of above
```

## Backward Compatibility

All changes are internal to the script. No external interface changes:
- Same `main()` signature
- Same env vars
- Same input/output paths
- Same contract/spec/config/builder

The error recovery adds a 5% failure tolerance — stricter than current behavior (0% = any failure kills). This is MORE permissive, not less.

## Verification

1. `python3 -c "import ast; ast.parse(...)"`
2. `ruff format` + `ruff check`
3. Test: 100 shards, 1 corrupted → pipeline succeeds (within tolerance)
4. Test: 100 shards, 10 corrupted → pipeline fails (exceeds 5%)
5. Test: empty shards skipped, schema mismatch logged
6. `brazil-build release`

---
tags:
  - project_planning
  - implementation
  - streaming
  - preprocessing
  - risk_table_mapping
keywords:
  - risk_table_mapping
  - categorical_encoding
topics:
  - streaming upgrade
language: python
date: 2026-01-26
status: planning
---

# Risk Table Mapping Dual Mode Implementation Plan

## Overview

Implementation plan for adding streaming mode to `risk_table_mapping.py` following the same architecture used in [Missing Value Imputation Dual Mode](2026-01-21_missing_value_imputation_dual_mode_implementation_plan.md).

## Current State Analysis

### Existing Implementation
- **File**: `src/cursus/steps/scripts/risk_table_mapping.py`
- **Mode**: Batch only
- **Size**: ~700 lines
- **Pattern**: Fit-transform with OfflineBinning class
- **Artifacts**: `risk_table_map.pkl` (compatible with XGBoost training format)
- **Dependencies**: Requires `hyperparameters.json` for categorical field configuration

### Key Classes
1. **OfflineBinning**: Core risk table mapping engine
   - `fit()`: Create risk tables from training data (category → risk score)
   - `transform()`: Apply risk tables to data
   - `load_risk_tables()`: Load pre-trained risk tables

### Data Flow
```
Batch Mode:
load_split_data() → process_data() → save_output_data()
                     ↓
              OfflineBinning
                     ↓
           save_artifacts()
```

### Risk Table Structure
```python
risk_tables = {
    "category_field_1": {
        "varName": "category_field_1",
        "type": "categorical",
        "mode": "categorical",  # or "numeric"
        "default_bin": 0.045,   # default risk score
        "bins": {
            "category_A": 0.023,
            "category_B": 0.067,
            "category_C": 0.041,
            # ... more mappings
        }
    },
    # ... more categorical fields
}
```

## Key Differences from Missing Value Imputation

### Similarities
- ✅ Both use fit-transform pattern
- ✅ Both support training/non-training modes
- ✅ Both accumulate artifacts from previous steps
- ✅ Both preserve input format (CSV/TSV/Parquet)
- ✅ Both use train/val/test subdirectory structure

### Differences

| Aspect | Missing Value Imputation | Risk Table Mapping |
|--------|-------------------------|-------------------|
| **Fit Operation** | Compute aggregates (mean/mode) | Build crosstab risk tables |
| **Memory Pass 1** | ~50-100MB (simple aggregates) | **~200-500MB** (crosstabs per category) |
| **Artifact Format** | `{column: value}` | `{field: {bins: {}, default_bin: float}}` |
| **Dependencies** | Environment vars only | **Requires hyperparameters.json** |
| **Complexity** | Simpler (fillna) | **More complex (crosstab + smoothing)** |
| **Configuration** | Strategies from env vars | **cat_field_list from hyperparameters** |

### Critical Consideration: Hyperparameters
Risk table mapping requires `hyperparameters.json` which contains:
- `cat_field_list`: List of categorical fields to process
- `label_name`: Target variable name
- `smooth_factor`: Smoothing parameter
- `count_threshold`: Minimum count threshold

**Streaming Mode Solution**: Load hyperparameters at start, pass to Pass 1 function.

## Implementation Plan

### Streaming Mode Architecture Clarification

**Two-Pass Design with Different Parallelization Strategies**:

#### Training Job Type:
- **Pass 1 (Sequential)**: Loop through all training shards to construct risk tables
  - Must process shards in sequence (or accumulate results)
  - Builds crosstab statistics incrementally
  - Memory-efficient but inherently sequential
  - Output: `risk_table_map.pkl` containing risk tables for all categorical fields
- **Pass 2 (Parallel)**: Apply risk tables to train/val/test splits
  - Fully parallel - each shard processed independently
  - Uses risk tables from Pass 1 as global context
  - Multi-worker parallelization for maximum throughput

#### Other Job Types (validation/testing/calibration):
- **Pass 1**: Skipped (load pre-trained risk tables from artifacts)
- **Pass 2 (Fully Parallel)**: Transform only mode
  - Each shard can be processed completely independently
  - No dependencies between shards
  - Perfect for multi-worker parallelization
  - Simply applies loaded risk tables via `.map()` operation

**Key Insight**: Non-training modes are embarrassingly parallel since risk tables are pre-computed!

### Phase 1: Add Streaming Utilities (Reuse from missing_value_imputation)

**Location**: Top of file, after existing imports

**Functions to Add/Reuse**:
```python
# Reuse from missing_value_imputation (same implementation):
- find_input_shards()        # Find shards in flat directory
- find_split_shards()         # Find shards in split subdirectories
- extract_shard_number()      # Extract number from part-XXXXX.ext
- write_shard_file()          # Write shard in specified format
- detect_shard_format()       # Auto-detect format from filename
- _read_file_to_df()          # Read CSV/TSV/Parquet to DataFrame
- aggregate_shard_results()   # Aggregate parallel results
```

### Phase 2: Implement Pass 1 - Collect Risk Table Statistics

**Function**: `collect_risk_table_statistics_pass1()`

**Purpose**: Build risk tables from training shards incrementally

**Memory**: ~200-500MB (accumulates crosstabs per categorical field)

**Algorithm**:
```python
def collect_risk_table_statistics_pass1(
    all_shards: List[Path],
    signature_columns: Optional[List[str]],
    cat_field_list: List[str],
    label_name: str,
    smooth_factor: float,
    count_threshold: int,
    max_unique_threshold: int,
    log_func: Callable,
) -> Dict[str, Any]:
    """
    Pass 1: Collect risk table statistics from training shards.
    
    Memory-efficient incremental crosstab aggregation:
    - For each categorical field:
      - Accumulate counts per (category, label) combination
      - Compute risk scores after all shards processed
    
    Args:
        all_shards: List of training shard paths
        signature_columns: Optional column names for CSV/TSV
        cat_field_list: List of categorical field names
        label_name: Target variable name
        smooth_factor: Smoothing factor for risk calculation
        count_threshold: Minimum count threshold
        max_unique_threshold: Max unique values for validation
        log_func: Logging function
    
    Returns:
        Dictionary of risk tables (same format as OfflineBinning.risk_tables)
        Format: {field: {varName, type, mode, default_bin, bins: {category: risk}}}
    """
    
    log_func("[PASS1] Collecting risk table statistics from training shards...")
    
    # Step 1: Identify valid categorical fields from first shard
    first_shard = all_shards[0]
    df_first = _read_file_to_df(first_shard, signature_columns)
    df_first.columns = [col.replace("__DOT__", ".") for col in df_first.columns]
    
    # Validate categorical fields
    valid_cat_fields = []
    for field in cat_field_list:
        if field in df_first.columns:
            unique_count = df_first[field].nunique()
            if unique_count < max_unique_threshold:
                valid_cat_fields.append(field)
                log_func(f"[PASS1]   {field}: {unique_count} unique values")
            else:
                log_func(f"[PASS1]   {field}: SKIP ({unique_count} > {max_unique_threshold})")
    
    if not valid_cat_fields:
        log_func("[PASS1] No valid categorical fields found")
        return {}
    
    # Step 2: Initialize crosstab accumulators
    # For each field, accumulate: {(category, label): count}
    crosstab_accumulators = {}
    for field in valid_cat_fields:
        crosstab_accumulators[field] = {
            "counts": {},  # {(category, label): count}
            "data_mode": None,
        }
    
    # Compute default risk from first shard (rough approximation)
    df_fit = df_first.loc[
        (df_first[label_name] != -1) & (~df_first[label_name].isnull())
    ].copy()
    default_risk = float(df_fit[label_name].mean()) if len(df_fit) > 0 else 0.0
    
    del df_first, df_fit
    gc.collect()
    
    # Step 3: Process each shard and accumulate crosstabs
    log_func(f"[PASS1] Processing {len(all_shards)} training shards...")
    
    for i, shard_path in enumerate(all_shards):
        try:
            df = _read_file_to_df(shard_path, signature_columns)
            df.columns = [col.replace("__DOT__", ".") for col in df.columns]
            
            # Filter valid rows for fitting
            df_fit = df.loc[
                (df[label_name] != -1) & (~df[label_name].isnull())
            ].copy()
            
            if len(df_fit) == 0:
                continue
            
            # Accumulate crosstabs for each field
            for field in valid_cat_fields:
                if field not in df_fit.columns:
                    continue
                
                accumulator = crosstab_accumulators[field]
                
                # Detect data mode from first shard
                if accumulator["data_mode"] is None:
                    if pd.api.types.is_numeric_dtype(df_fit[field]):
                        accumulator["data_mode"] = "numeric"
                    else:
                        accumulator["data_mode"] = "categorical"
                
                # Accumulate counts per (category, label) pair
                for category, label in zip(df_fit[field], df_fit[label_name]):
                    if pd.isna(category):
                        continue
                    key = (category, label)
                    accumulator["counts"][key] = accumulator["counts"].get(key, 0) + 1
            
            del df, df_fit
            gc.collect()
            
            if (i + 1) % 100 == 0:
                log_func(f"[PASS1] Processed {i + 1}/{len(all_shards)} shards")
        
        except Exception as e:
            log_func(f"[PASS1 WARNING] Failed to read {shard_path.name}: {e}")
            continue
    
    # Step 4: Compute risk tables from accumulated crosstabs
    log_func("[PASS1] Computing final risk tables...")
    
    smooth_samples = int(len(all_shards) * 100 * smooth_factor)  # Rough estimate
    
    risk_tables = {}
    for field, accumulator in crosstab_accumulators.items():
        try:
            counts = accumulator["counts"]
            
            if not counts:
                log_func(f"[PASS1]   {field}: No valid data, using default")
                risk_tables[field] = {
                    "varName": field,
                    "type": "categorical",
                    "mode": accumulator["data_mode"],
                    "default_bin": default_risk,
                    "bins": {},
                }
                continue
            
            # Build risk table from accumulated counts
            # Group by category, compute risk = count(label=1) / count(total)
            category_stats = {}
            for (category, label), count in counts.items():
                if category not in category_stats:
                    category_stats[category] = {"total": 0, "positive": 0}
                category_stats[category]["total"] += count
                if label == 1:
                    category_stats[category]["positive"] += count
            
            # Compute smoothed risk scores
            bins = {}
            for category, stats in category_stats.items():
                total = stats["total"]
                positive = stats["positive"]
                
                # Base risk
                risk = positive / total if total > 0 else 0.0
                
                # Apply smoothing and threshold
                if total >= count_threshold:
                    smooth_risk = (
                        (total * risk + smooth_samples * default_risk)
                        / (total + smooth_samples)
                    )
                else:
                    smooth_risk = default_risk
                
                bins[category] = float(smooth_risk)
            
            risk_tables[field] = {
                "varName": field,
                "type": "categorical",
                "mode": accumulator["data_mode"],
                "default_bin": default_risk,
                "bins": bins,
            }
            
            log_func(f"[PASS1]   {field}: {len(bins)} categories mapped")
        
        except Exception as e:
            log_func(f"[PASS1 WARNING] Failed to compute risk table for {field}: {e}")
            risk_tables[field] = {
                "varName": field,
                "type": "categorical",
                "mode": accumulator["data_mode"],
                "default_bin": default_risk,
                "bins": {},
            }
    
    log_func(f"[PASS1] Complete! Created risk tables for {len(risk_tables)} fields")
    
    return risk_tables
```

**Key Points**:
- Accumulates crosstabs incrementally (memory-efficient)
- Computes risk scores after all shards processed
- Applies smoothing and count thresholds
- Returns dict compatible with OfflineBinning.risk_tables

### Phase 3: Implement Pass 2 - Parallel Per-Shard Risk Table Application

**Function**: `process_shard_end_to_end_risk_mapping()`

**Purpose**: Apply risk tables to single shard using global context

**Algorithm**:
```python
def process_shard_end_to_end_risk_mapping(args: tuple) -> Dict[str, int]:
    """
    Process single shard: read → apply risk tables → write.
    
    Stateless per-shard processing using global risk_tables from Pass 1.
    Preserves 1:1 shard mapping (input shard number → output shard number).
    
    Args:
        args: Tuple of (shard_path, shard_num, global_context,
                       output_base, signature_columns, output_format)
                       
        global_context must contain:
        - "risk_tables": Dictionary of risk tables
        - "split_name": Which split this shard belongs to ("train", "val", "test", etc.)
    
    Returns:
        Statistics dict with row count for this split
        Format: {"train": 1000} or {"val": 200} or {"validation": 500}
    
    Example:
        Input: train/part-00042.csv
        Output: train/part-00042.csv (risk-mapped)
    """
    shard_path, shard_num, global_context, output_base, signature_columns, output_format = args
    
    try:
        # ====================================================================
        # STEP 1: Read Single Shard
        # ====================================================================
        df = _read_file_to_df(shard_path, signature_columns)
        df.columns = [col.replace("__DOT__", ".") for col in df.columns]
        
        # ====================================================================
        # STEP 2: Apply Risk Table Mapping (Using Global Context)
        # ====================================================================
        risk_tables = global_context["risk_tables"]
        
        # Simple map operation for each categorical field
        for field, risk_table_info in risk_tables.items():
            if field in df.columns:
                bins = risk_table_info["bins"]
                default_bin = risk_table_info["default_bin"]
                df[field] = df[field].map(bins).fillna(default_bin)
        
        # ====================================================================
        # STEP 3: Write to Correct Split Folder (Preserving Shard Number)
        # ====================================================================
        split_name = global_context["split_name"]
        stats = {}
        
        if len(df) > 0:
            output_path = output_base / split_name / f"part-{shard_num:05d}.{output_format}"
            write_shard_file(df, output_path, output_format)
            stats[split_name] = len(df)
        else:
            stats[split_name] = 0
        
        return stats
    
    except Exception as e:
        # Log error but don't crash the entire pool
        print(f"[ERROR] Failed to process shard {shard_num} ({shard_path.name}): {e}")
        # Return zero stats for this shard
        split_name = global_context.get("split_name", "unknown")
        return {split_name: 0}
```

**Key Points**:
- Stateless per-shard processing
- Simple `.map()` operation (fast)
- Preserves 1:1 shard mapping
- Error handling without crashing pool

### Phase 4: Implement Main Streaming Entry Point

**Function**: `process_streaming_mode_risk_mapping()`

**Purpose**: Orchestrate two-pass streaming process

**Algorithm**:
```python
def process_streaming_mode_risk_mapping(
    input_dir: str,
    output_dir: str,
    signature_columns: Optional[List[str]],
    job_type: str,
    hyperparams: Dict[str, Any],
    environ_vars: Dict[str, str],
    max_workers: Optional[int],
    model_artifacts_input_dir: Optional[str] = None,
    model_artifacts_output_dir: Optional[str] = None,
    logger: Optional[Callable] = None,
) -> Dict[str, int]:
    """
    Streaming mode for risk table mapping with train/val/test subdirectories.
    
    Two-pass architecture:
    - Pass 1: Collect risk table statistics from training shards only
    - Pass 2: Apply risk tables per split in parallel
    
    Auto-detects output format from input shards (mirrors batch mode behavior).
    
    Input structure (training mode):
      input_dir/
        train/part-00000.csv, part-00001.csv, ...
        val/part-00000.csv, part-00001.csv, ...
        test/part-00000.csv, part-00001.csv, ...
    
    Output structure (training mode):
      output_dir/
        train/part-00000.csv, part-00001.csv, ... (risk-mapped, same format)
        val/part-00000.csv, part-00001.csv, ... (risk-mapped, same format)
        test/part-00000.csv, part-00001.csv, ... (risk-mapped, same format)
    
    Args:
        input_dir: Base input directory
        output_dir: Base output directory
        signature_columns: Optional column names for CSV/TSV
        job_type: 'training', 'validation', 'testing', 'calibration'
        hyperparams: Hyperparameters dict (contains cat_field_list, etc.)
        environ_vars: Environment variables dict
        max_workers: Number of parallel workers
        model_artifacts_input_dir: Input model artifacts directory
        model_artifacts_output_dir: Output model artifacts directory
        logger: Logging function
    
    Returns:
        Dictionary with total row counts per split
    """
    log = logger or print
    output_path = Path(output_dir)
    
    # Extract parameters from hyperparameters
    cat_field_list = hyperparams.get("cat_field_list", [])
    label_name = hyperparams.get("label_name", "target")
    smooth_factor = float(environ_vars.get("SMOOTH_FACTOR", hyperparams.get("smooth_factor", 0.01)))
    count_threshold = int(environ_vars.get("COUNT_THRESHOLD", hyperparams.get("count_threshold", 5)))
    max_unique_threshold = int(environ_vars.get("MAX_UNIQUE_THRESHOLD", hyperparams.get("max_unique_threshold", 100)))
    
    # Determine optimal workers (0 or None = auto-detect)
    if max_workers is None or max_workers == 0:
        max_workers = min(cpu_count(), 8)  # Default to 8 workers
    
    log(f"[STREAMING] Starting streaming mode risk table mapping")
    log(f"[STREAMING] Job type: {job_type}")
    log(f"[STREAMING] Max workers: {max_workers}")
    log(f"[STREAMING] Categorical fields: {cat_field_list}")
    
    # ========================================================================
    # PASS 1: Collect Risk Table Statistics (Training Only)
    # ========================================================================
    if job_type == "training":
        log("[STREAMING] PASS 1: Collecting risk table statistics from train split...")
        train_shards = find_split_shards(input_dir, "train", log)
        risk_tables = collect_risk_table_statistics_pass1(
            train_shards,
            signature_columns,
            cat_field_list,
            label_name,
            smooth_factor,
            count_threshold,
            max_unique_threshold,
            log,
        )
        
        # Save risk table artifacts
        if model_artifacts_output_dir:
            artifacts_path = Path(model_artifacts_output_dir)
            artifacts_path.mkdir(parents=True, exist_ok=True)
            
            # Save risk_table_map.pkl
            risk_table_path = artifacts_path / RISK_TABLE_FILENAME
            with open(risk_table_path, "wb") as f:
                pkl.dump(risk_tables, f)
            log(f"[STREAMING] Saved risk table to {risk_table_path}")
            
            # Save hyperparameters
            hyperparams_path = artifacts_path / HYPERPARAMS_FILENAME
            with open(hyperparams_path, "w") as f:
                json.dump(hyperparams, f, indent=2)
            log(f"[STREAMING] Saved hyperparameters to {hyperparams_path}")
    else:
        # Non-training: Load risk tables
        if not model_artifacts_input_dir:
            raise ValueError(f"model_artifacts_input_dir required for {job_type} mode")
        
        risk_table_path = Path(model_artifacts_input_dir) / RISK_TABLE_FILENAME
        if not risk_table_path.exists():
            raise FileNotFoundError(f"Risk table not found: {risk_table_path}")
        
        log(f"[STREAMING] Loading risk tables from {risk_table_path}")
        with open(risk_table_path, "rb") as f:
            risk_tables = pkl.load(f)
        log(f"[STREAMING] Loaded {len(risk_tables)} risk tables")
    
    # ========================================================================
    # PASS 2: Process Each Split Independently
    # ========================================================================
    log("[STREAMING] PASS 2: Processing splits in parallel...")
    
    # Determine which splits to process
    if job_type == "training":
        splits_to_process = ["train", "val", "test"]
    else:
        splits_to_process = [job_type]  # Single split (validation, testing, calibration)
    
    total_stats = {}
    
    for split_name in splits_to_process:
        log(f"[STREAMING] Processing {split_name} split...")
        
        # Find shards for this split
        split_shards = find_split_shards(input_dir, split_name, log)
        
        # Auto-detect format from first shard (mirrors batch mode behavior)
        output_format = detect_shard_format(split_shards[0])
        log(f"[STREAMING] Detected format: {output_format}")
        
        # Build global context for this split
        global_context = {
            "split_name": split_name,
            "risk_tables": risk_tables,
        }
        
        # Prepare arguments for parallel processing
        shard_args = [
            (
                shard,
                extract_shard_number(shard),
                global_context,
                output_path,
                signature_columns,
                output_format,
            )
            for shard in split_shards
        ]
        
        # Process shards in parallel
        log(f"[STREAMING] Processing {len(shard_args)} shards from {split_name} with {max_workers} workers")
        with Pool(processes=max_workers) as pool:
            results = pool.map(process_shard_end_to_end_risk_mapping, shard_args)
        
        # Aggregate results for this split
        split_total = sum(r.get(split_name, 0) for r in results)
        total_stats[split_name] = split_total
        log(f"[STREAMING] Completed {split_name} split: {split_total:,} rows")
    
    log(f"[STREAMING] Complete! Row distribution: {total_stats}")
    return total_stats
```

### Phase 5: Integrate with Existing main() Function

**Modifications to `internal_main()`**:

```python
def internal_main(
    job_type: str,
    input_dir: str,
    output_dir: str,
    hyperparams: Dict[str, Any],
    environ_vars: Dict[str, str],
    model_artifacts_input_dir: Optional[str] = None,
    model_artifacts_output_dir: Optional[str] = None,
    load_data_func: Callable = load_split_data,
    save_data_func: Callable = save_output_data,
) -> Tuple[Dict[str, pd.DataFrame], OfflineBinning]:
    """
    Main logic for risk table mapping with dual-mode support.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract configuration
    cat_field_list = hyperparams.get("cat_field_list", [])
    label_name = hyperparams.get("label_name", "target")
    # ... other params ...
    
    # NEW - Extract streaming mode configuration
    enable_true_streaming = environ_vars.get("ENABLE_TRUE_STREAMING", "false").lower() == "true"
    max_workers_str = environ_vars.get("MAX_WORKERS", "0")
    max_workers = int(max_workers_str) if max_workers_str else 0
    
    # Determine model artifacts output directory
    artifacts_output_dir = (
        Path(model_artifacts_output_dir)
        if model_artifacts_output_dir
        else output_path / "model_artifacts"
    )
    artifacts_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy existing artifacts from previous steps
    if model_artifacts_input_dir:
        copy_existing_artifacts(model_artifacts_input_dir, str(artifacts_output_dir))
    
    # ========================================================================
    # STREAMING MODE
    # ========================================================================
    if enable_true_streaming:
        logger.info("=" * 60)
        logger.info("STREAMING MODE ENABLED")
        logger.info("=" * 60)
        
        # Call streaming mode orchestration
        stats = process_streaming_mode_risk_mapping(
            input_dir=input_dir,
            output_dir=output_dir,
            signature_columns=None,  # Files have headers
            job_type=job_type,
            hyperparams=hyperparams,
            environ_vars=environ_vars,
            max_workers=max_workers,
            model_artifacts_input_dir=model_artifacts_input_dir,
            model_artifacts_output_dir=str(artifacts_output_dir),
            logger=logger.info,
        )
        
        logger.info(f"Streaming mode complete! Final statistics: {stats}")
        
        # Return empty data dict and None binner (data written to disk)
        return {}, None
    
    # ========================================================================
    # BATCH MODE (DEFAULT) - Existing code unchanged
    # ========================================================================
    logger.info("Running in BATCH MODE")
    
    # Load data according to job type
    data_dict = load_data_func(job_type, input_dir)
    
    # ... rest of existing batch mode logic ...
```

### Phase 6: Update Contract and Config

**Contract Updates** (`src/cursus/steps/contracts/risk_table_mapping_contract.py`):

```python
optional_env_vars={
    "SMOOTH_FACTOR": "0.01",
    "COUNT_THRESHOLD": "5",
    "MAX_UNIQUE_THRESHOLD": "100",
    
    # NEW - Streaming mode configuration
    "ENABLE_TRUE_STREAMING": "false",
    "MAX_WORKERS": "0",
}
```

Update description to include streaming mode details.

**Config Updates** (`src/cursus/steps/configs/config_risk_table_mapping_step.py`):

```python
class RiskTableMappingConfig(ProcessingStepConfigBase):
    # ... existing fields ...
    
    # NEW - Streaming mode configuration
    enable_true_streaming: bool = Field(
        default=False,
        description="Enable memory-efficient streaming mode for large datasets",
    )
    
    max_workers: int = Field(
        default=0,
        description="Number of parallel workers for streaming mode (0 = auto-detect, >0 = specific count)",
    )
    
    # Update environment_variables property
    @property
    def environment_variables(self) -> Dict[str, str]:
        """Get environment variables for the risk table mapping script."""
        if self._environment_variables is None:
            self._environment_variables = {
                "SMOOTH_FACTOR": str(self.smooth_factor),
                "COUNT_THRESHOLD": str(self.count_threshold),
                "MAX_UNIQUE_THRESHOLD": str(self.max_unique_threshold),
                # NEW - Streaming mode configuration
                "ENABLE_TRUE_STREAMING": str(self.enable_true_streaming).lower(),
                "MAX_WORKERS": str(self.max_workers),
            }
        return self._environment_variables
    
    # Add validator for max_workers
    @field_validator("max_workers")
    @classmethod
    def validate_max_workers(cls, v: int) -> int:
        """Ensure max_workers is 0 (auto-detect) or a positive integer."""
        if not isinstance(v, int) or v < 0:
            raise ValueError("max_workers must be 0 (auto-detect) or a positive integer")
        return v
```

## Implementation Checklist

### Code Changes
- [x] **PHASE 1 COMPLETE** - Add streaming utility functions
  - [x] Add `find_input_shards()` utility (from missing_value_imputation)
  - [x] Add `find_split_shards()` utility (from missing_value_imputation)
  - [x] Add `extract_shard_number()` utility (from missing_value_imputation)
  - [x] Add `write_shard_file()` utility (from missing_value_imputation)
  - [x] Add `detect_shard_format()` utility (from missing_value_imputation)
  - [x] Add `_read_file_to_df()` helper (from missing_value_imputation)
  - [x] Add `aggregate_shard_results()` utility (from missing_value_imputation)
- [x] **PHASE 2 COMPLETE** - Implement `collect_risk_table_statistics_pass1()`
  - [x] Add necessary imports (gc, Counter, Pool, cpu_count, pkl, json) - done in Phase 1
  - [x] Implement crosstab accumulation logic
  - [x] Implement risk score computation with smoothing
  - [x] Handle data mode detection (numeric vs categorical)
  - [x] Add progress logging every 100 shards
  - [x] Memory optimization for large category sets
- [x] **PHASE 3 COMPLETE** - Implement `process_shard_end_to_end_risk_mapping()`
  - [x] Read single shard
  - [x] Apply risk table mapping using global context
  - [x] Write to SINGLE split folder (not all three)
  - [x] Preserve 1:1 shard mapping
  - [x] Error handling without crashing pool
- [x] **PHASE 4 COMPLETE** - Implement `process_streaming_mode_risk_mapping()`
  - [x] Two-pass orchestration
  - [x] Pass 1: Collect statistics from train split only
  - [x] Pass 2: Process each split independently
  - [x] Training mode: process train/val/test splits
  - [x] Non-training mode: process single split
  - [x] Parallel processing with multiprocessing Pool
  - [x] Artifact management (save/load risk_table_map.pkl)
  - [x] Hyperparameters handling (load from file, pass to Pass 1)
- [x] **PHASE 5 COMPLETE** - Update `internal_main()` and `__main__` block with streaming integration
  - [x] Add streaming parameters to signature (already present: hyperparams, environ_vars)
  - [x] Add mode routing logic (if enable_true_streaming → streaming, else → batch)
  - [x] Extract streaming flags from environ_vars
  - [x] Pass hyperparams to streaming function
  - [x] Log mode selection for debugging
  - [x] Update `__main__` block environ_vars to include ENABLE_TRUE_STREAMING
  - [x] Update `__main__` block environ_vars to include MAX_WORKERS
- [x] **PHASE 6 COMPLETE** - Update contract and config
  - [x] Add `ENABLE_TRUE_STREAMING` to optional_env_vars in contract
  - [x] Add `MAX_WORKERS` to optional_env_vars in contract
  - [x] Update contract description with streaming mode details
  - [x] Add `enable_true_streaming` field to config
  - [x] Add `max_workers` field to config
  - [x] Add validator for `max_workers` to config
  - [x] Update `environment_variables` property to include streaming flags in config

### Testing
- [ ] Unit test for `collect_risk_table_statistics_pass1()`
  - Test crosstab accumulation
  - Test risk score computation
  - Test smoothing application
  - Test empty/invalid data handling
- [ ] Unit test for `process_shard_end_to_end_risk_mapping()`
  - Test single shard processing
  - Test preserved shard numbering
  - Test risk mapping application
- [ ] Integration test comparing batch vs streaming
  - Same input data
  - Compare risk_table_map.pkl artifacts
  - Compare row counts
  - Compare sample risk scores

### Documentation
- [ ] Update script docstring with streaming mode description
- [ ] Add streaming mode usage examples
- [ ] Document memory usage characteristics
- [ ] Document performance expectations
- [ ] Document hyperparameters.json requirement for streaming mode

## Testing Strategy

### Unit Tests

```python
def test_collect_risk_table_statistics():
    """Test Pass 1 crosstab aggregation and risk computation"""
    # Create test shards with known categories and labels
    # Verify risk_tables matches expected structure
    # Verify smoothing is applied correctly
    
def test_process_shard_risk_mapping():
    """Test Pass 2 transformation"""
    # Create test shard with categorical values
    # Apply known risk_tables
    # Verify output correctness
```

### Integration Tests

```python
def test_batch_vs_streaming_consistency():
    """Compare batch and streaming outputs"""
    # Run both modes on same input
    # Compare:
    #   - risk_table_map.pkl artifacts (should be similar within smoothing tolerance)
    #   - Total row counts (should match)
    #   - Sample risk scores (should match)
```

### Performance Tests

```python
def test_streaming_performance():
    """Measure speedup and memory usage"""
    # Benchmark batch mode
    # Benchmark streaming mode
    # Verify 3-5× speedup
    # Verify memory reduction
```

## Expected Outcomes

### Performance
- **Speed**: 3-5× faster than batch mode (similar to missing_value_imputation)
- **Memory**: 70-80% reduction (Pass 1 uses ~200-500MB vs full dataset in memory)
- **Scalability**: Linear scaling with worker count in Pass 2

### Compatibility
- **Artifacts**: Compatible `risk_table_map.pkl` format (same structure as OfflineBinning)
- **Batch Mode**: Unchanged, remains default
- **Output**: Sharded format compatible with PyTorch IterableDataset

### Code Quality
- **Reusability**: Shared utilities with missing_value_imputation
- **Testability**: Separate Pass 1 and Pass 2 functions
- **Maintainability**: Clear separation of concerns

## Risks and Mitigation

### Risk 1: Risk Tables Differ Between Modes
**Risk**: Batch and streaming compute slightly different risk scores due to different aggregation order

**Mitigation**: 
- Use same smoothing logic in both modes
- Add integration tests comparing outputs
- Accept minor differences within tolerance (e.g., 0.001)
- Document any known differences

### Risk 2: High Memory in Pass 1 for High-Cardinality Categories
**Risk**: Fields with many unique categories (e.g., 10K+ categories) use excessive memory

**Mitigation**:
- Validate fields upfront (max_unique_threshold)
- Skip fields exceeding threshold
- Add memory monitoring and warnings
- Consider sampling for very high cardinality fields

### Risk 3: Hyperparameters Loading Complexity
**Risk**: Streaming mode requires hyperparameters.json which may not always be available

**Mitigation**:
- Fail fast with clear error if hyperparameters missing
- Document hyperparameters requirement prominently
- Provide fallback to environment variables where possible
- Test both embedded and external hyperparameters paths

### Risk 4: Empty Shards After Risk Mapping
**Risk**: Some shards may have 0 rows after processing

**Mitigation**:
- Skip writing empty shards (sparse numbering OK)
- Log warning when skipping
- Document expected behavior

## Timeline

- **Phase 1**: 1-2 hours (utilities - mostly copy from missing_value_imputation)
- **Phase 2**: 3-4 hours (Pass 1 - more complex than imputation due to crosstab logic)
- **Phase 3**: 1-2 hours (Pass 2 - similar to imputation)
- **Phase 4**: 2-3 hours (orchestration + hyperparameters handling)
- **Phase 5**: 1 hour (main() integration)
- **Phase 6**: 1-2 hours (contract + config updates)
- **Testing**: 2-3 hours
- **Documentation**: 1 hour

**Total**: ~12-18 hours (slightly longer than missing_value_imputation due to crosstab complexity)

## Dependencies

- Completed missing_value_imputation streaming implementation (for reference and code reuse)
- Access to test data with categorical fields and labels
- Understanding of existing batch mode OfflineBinning behavior
- Hyperparameters.json file structure

## Success Criteria

1. ✅ Streaming mode produces compatible artifacts with batch mode
2. ✅ 3-5× performance improvement demonstrated
3. ✅ Memory usage reduced by 70-80%
4. ✅ All tests passing
5. ✅ Documentation updated
6. ✅ Backward compatible (batch mode unchanged)
7. ✅ Hyperparameters correctly loaded and passed to Pass 1

## Key Implementation Notes

### Differences from Missing Value Imputation

1. **Crosstab Accumulation**: More complex than simple mean/mode
   - Need to track (category, label) pairs
   - Compute risk scores from accumulated counts
   - Apply smoothing after accumulation complete

2. **Hyperparameters Dependency**: 
   - Missing value imputation uses only environment variables
   - Risk mapping REQUIRES hyperparameters.json
   - Must load hyperparameters at start and pass to Pass 1

3. **Memory Characteristics**:
   - Pass 1 uses more memory (~200-500MB vs ~50-100MB)
   - High-cardinality categories can increase memory significantly
   - Need validation upfront to skip problematic fields

4. **Artifact Format**:
   - More complex nested structure than simple dict
   - Must maintain compatibility with OfflineBinning format
   - Includes metadata (varName, type, mode, default_bin)

### Code Reuse Opportunities

- All Phase 1 utilities can be copied directly from missing_value_imputation
- Phase 2 and Phase 4 orchestration logic very similar
- Phase 5 integration logic nearly identical
- Phase 6 contract/config updates follow same pattern

---

**Status**: Planning  
**Last Updated**: 2026-01-26  
**Related Documents**:
- [Missing Value Imputation Dual Mode Implementation Plan](2026-01-21_missing_value_imputation_dual_mode_implementation_plan.md)
- [Streaming Preprocessing Additional Scripts Design](../1_design/streaming_preprocessing_additional_scripts_design.md)
- [Temporal Split Preprocessing Streaming Upgrade](../1_design/temporal_split_preprocessing_streaming_upgrade.md)

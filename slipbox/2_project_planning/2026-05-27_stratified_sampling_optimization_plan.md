# Plan: StratifiedSampling Step — Robustness + External Proportional Enhancement

**Date**: 2026-05-27
**Status**: ✅ ALL PHASES COMPLETE
**Commits**: `fc9dc38` (external_proportional + robustness), `23c37b2` (bug fixes)
**Build**: https://build.amazon.com/7856838511 (published)
**Motivation**: FZ 29d9b identified 6 gaps vs industry standards. Gaps 1-2 block Munged Address pipeline. Gaps 3-4 prevent silent production failures.
**Related Notes**: FZ 29d9a1 (design), 29d9a2 (Approach E sidecar), 29d9a3 (final action items), 29d9b (optimization analysis)

## Scope

Add `external_proportional` strategy with `allow_replacement`, fix NaN/empty-frame bugs, add diagnostics output. Fully backward-compatible — existing strategies unchanged.

## Files Modified

| File | Location | Change |
|------|----------|--------|
| `stratified_sampling.py` | `steps/scripts/` | +strategy, +replace, +NaN guard, +empty guard, +diagnostics |
| `config_stratified_sampling_step.py` | `steps/configs/` | +3 fields, update validator |
| `stratified_sampling_contract.py` | `steps/contracts/` | +3 optional env vars |
| `builder_stratified_sampling_step.py` | `steps/builders/` | +3 env var emissions |
| `stratified_sampling_spec.py` | `steps/specs/` | NO CHANGE (single input_data dep, Approach E) |

## Phase 1: Config + Contract + Builder (Parallel, Independent)

### 1A. Config (`config_stratified_sampling_step.py`)

Add 3 new Tier 2 fields after `variance_column`:

```python
sampling_multiplier: float = Field(
    default=1.0,
    ge=0.1,
    description="Multiplier for external reference counts (e.g., 5.0 for 5× oversampling).",
)

allow_replacement: bool = Field(
    default=False,
    description="Allow sampling with replacement when target exceeds available per stratum.",
)

reference_counts_json: Optional[str] = Field(
    default=None,
    description="JSON string of reference distribution {stratum: count}. Fallback when reference_counts.json sidecar file is absent.",
)
```

Update `validate_sampling_strategy` validator:
```python
allowed = {"balanced", "proportional_min", "optimal", "external_proportional"}
```

Update `get_public_init_fields`:
```python
sampling_fields["sampling_multiplier"] = self.sampling_multiplier
sampling_fields["allow_replacement"] = self.allow_replacement
if self.reference_counts_json is not None:
    sampling_fields["reference_counts_json"] = self.reference_counts_json
```

### 1B. Contract (`stratified_sampling_contract.py`)

Add 3 entries to `optional_env_vars`:
```python
"SAMPLING_MULTIPLIER": "1.0",
"ALLOW_REPLACEMENT": "false",
"REFERENCE_COUNTS_JSON": "",
```

### 1C. Builder (`builder_stratified_sampling_step.py`)

In the env var emission section, add:
```python
"SAMPLING_MULTIPLIER": str(self.config.sampling_multiplier),
"ALLOW_REPLACEMENT": str(self.config.allow_replacement).lower(),
"REFERENCE_COUNTS_JSON": self.config.reference_counts_json or "",
```

## Phase 2: Script Changes (After Phase 1)

### 2A. Add `external_proportional` Strategy

In `StratifiedSampler.__init__`, register:
```python
self.strategies = {
    "balanced": self._balanced_allocation,
    "proportional_min": self._proportional_with_min,
    "optimal": self._optimal_allocation,
    "external_proportional": self._external_proportional,
}
```

Add method:
```python
def _external_proportional(
    self, strata_info, target_size, min_samples,
    reference_counts=None, multiplier=1.0,
):
    """Allocate proportionally to an external reference distribution."""
    if not reference_counts:
        raise ValueError(
            "external_proportional strategy requires reference_counts "
            "(from sidecar file or REFERENCE_COUNTS_JSON env var)"
        )
    allocation = {}
    for stratum, info in strata_info.items():
        ref_count = reference_counts.get(str(stratum), 0)
        allocation[stratum] = max(min_samples, int(ref_count * multiplier))
    return allocation
```

### 2B. Add `allow_replacement` to `_perform_sampling`

Update signature:
```python
def _perform_sampling(self, df, strata_column, allocation, allow_replacement=False):
```

Update sampling logic:
```python
for stratum, sample_size in allocation.items():
    if sample_size > 0:
        stratum_df = df[df[strata_column] == stratum]
        if len(stratum_df) >= sample_size:
            sampled = stratum_df.sample(n=sample_size, random_state=self.random_state)
        elif allow_replacement and len(stratum_df) > 0:
            sampled = stratum_df.sample(
                n=sample_size, replace=True, random_state=self.random_state
            )
        else:
            sampled = stratum_df
        sampled_dfs.append(sampled)
```

Update `sample()` method signature:
```python
def sample(self, df, strata_column, target_size, strategy="balanced",
           min_samples_per_stratum=10, variance_column=None,
           reference_counts=None, multiplier=1.0, allow_replacement=False):
```

Thread `allow_replacement` to `_perform_sampling` and `reference_counts`/`multiplier` to strategy function.

### 2C. Add NaN Guard

At top of `_get_strata_info`:
```python
nan_count = df[strata_column].isna().sum()
if nan_count > 0:
    logger.warning(
        f"Found {nan_count} NaN values in strata column '{strata_column}'. "
        f"Excluding from sampling."
    )
    df = df.dropna(subset=[strata_column])
```

Return the filtered df alongside strata_info (or filter before calling).

### 2D. Add Empty DataFrame Guard

At top of `sample()`:
```python
if df.empty:
    logger.warning("Empty DataFrame received, returning empty result")
    return pd.DataFrame(columns=df.columns)
```

### 2E. Add Diagnostics Output

After sampling in `main()`, write diagnostics:
```python
import json as json_module

diagnostics = {
    "strategy": sampling_strategy,
    "strata_column": strata_column,
    "target_sample_size": target_sample_size,
    "input_size": len(df),
    "output_size": len(sampled_df),
    "allow_replacement": allow_replacement,
    "multiplier": sampling_multiplier,
    "per_stratum": {
        str(stratum): {
            "requested": allocation.get(stratum, 0),
            "achieved": int((sampled_df[strata_column] == stratum).sum()),
            "available": int((df[strata_column] == stratum).sum()),
            "replacement_used": allocation.get(stratum, 0) > int((df[strata_column] == stratum).sum()),
        }
        for stratum in sampled_df[strata_column].unique()
    },
}

diagnostics_file = Path(output_dir) / split_name / "sampling_diagnostics.json"
diagnostics_file.write_text(json_module.dumps(diagnostics, indent=2, default=str))
log(f"[INFO] Saved sampling diagnostics to {diagnostics_file}")
```

### 2F. Update `main()` — Read Reference Counts

In `main()`, after reading env vars, add reference counts loading:
```python
sampling_multiplier = float(environ_vars.get("SAMPLING_MULTIPLIER", "1.0"))
allow_replacement = environ_vars.get("ALLOW_REPLACEMENT", "false").lower() == "true"

# Load reference counts (sidecar file > env var > None)
reference_counts = None
if sampling_strategy == "external_proportional":
    # Try sidecar file first (Approach E from FZ 29d9a2)
    reference_path = Path(input_data_dir) / "reference_counts.json"
    if reference_path.exists():
        reference_counts = json.loads(reference_path.read_text())
        log(f"[INFO] Loaded reference counts from sidecar: {reference_path}")
    else:
        # Fallback to env var
        ref_json = environ_vars.get("REFERENCE_COUNTS_JSON", "")
        if ref_json:
            reference_counts = json.loads(ref_json)
            log("[INFO] Loaded reference counts from REFERENCE_COUNTS_JSON env var")
        else:
            raise RuntimeError(
                "external_proportional strategy requires reference_counts.json sidecar "
                "or REFERENCE_COUNTS_JSON environment variable"
            )
```

Update the strategy validation:
```python
valid_strategies = ["balanced", "proportional_min", "optimal", "external_proportional"]
if sampling_strategy not in valid_strategies:
    raise RuntimeError(f"Invalid SAMPLING_STRATEGY: {sampling_strategy}. Must be one of: {valid_strategies}")
```

Pass new params to `sampler.sample()`:
```python
sampled_df = sampler.sample(
    df=df,
    strata_column=strata_column,
    target_size=split_target_size,
    strategy=sampling_strategy,
    min_samples_per_stratum=min_samples_per_stratum,
    variance_column=variance_column,
    reference_counts=reference_counts,
    multiplier=sampling_multiplier,
    allow_replacement=allow_replacement,
)
```

### 2G. Update `__main__` Block

Add env var reads:
```python
SAMPLING_MULTIPLIER = float(os.environ.get("SAMPLING_MULTIPLIER", "1.0"))
ALLOW_REPLACEMENT = os.environ.get("ALLOW_REPLACEMENT", "false").lower() == "true"
REFERENCE_COUNTS_JSON = os.environ.get("REFERENCE_COUNTS_JSON", "")
```

Add to environ_vars dict:
```python
"SAMPLING_MULTIPLIER": str(SAMPLING_MULTIPLIER),
"ALLOW_REPLACEMENT": str(ALLOW_REPLACEMENT).lower(),
"REFERENCE_COUNTS_JSON": REFERENCE_COUNTS_JSON,
```

Update `choices` in argparser for job_type (remove hard constraint — allow any lowercase):
```python
parser.add_argument("--job_type", type=str, required=True)
```

## Phase 3: Verification

1. `ruff check` + `ruff format` on all 4 modified files
2. `python3 -c "import ast; ast.parse(...)"` syntax check
3. Unit tests:
   - `external_proportional` with mock reference_counts
   - `allow_replacement=True` with target > available
   - NaN in strata column → warning + exclusion
   - Empty DataFrame → empty result
   - Diagnostics JSON written correctly
4. `brazil-build release`
5. `brazil pb build`

## Dependency Graph

```
Phase 1A (config)   ──┐
Phase 1B (contract)   ├── Independent, parallel
Phase 1C (builder)   ──┘
                       │
                       ▼
Phase 2A-2G (script)  ── Sequential (depends on env var names from 1B)
                       │
                       ▼
Phase 3 (verify)      ── After all changes
```

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| `reference_counts.json` not found | Fallback chain: sidecar file → env var → clear error message |
| Replacement creates duplicates | Documented behavior; downstream dedup possible |
| NaN filtering changes row count | Warning logged with count; original df unchanged (copy) |
| Diagnostics file name collision | Per-split naming: `{split}/sampling_diagnostics.json` |
| Backward compat | All new params have defaults matching current behavior (multiplier=1.0, replace=False, ref=None) |

## Estimated Lines

| File | Lines Added |
|------|-------------|
| `stratified_sampling.py` (script) | ~55 |
| `config_stratified_sampling_step.py` | ~15 |
| `stratified_sampling_contract.py` | ~3 |
| `builder_stratified_sampling_step.py` | ~10 |
| **Total** | **~83** |

---

## Phase 2: Pass-Through Filter (Sampling Subset, Pass Rest)

**Status**: 🟡 Ready to implement
**Motivation**: FZ 29d11d/29d11e — Munged Address pipeline passes BOTH good + bad cohorts through StratifiedSampling. Need to sample only good (matching `__cohort__=='good'`) and pass bad through unchanged.
**Effort**: ~29 lines across 4 files

### Why Needed

`StratifiedSampling_sampling` receives combined data with `__cohort__` column:
- `__cohort__=='good'` → sample 5× (external_proportional)
- `__cohort__=='bad'` → pass through unchanged (do NOT sample)

Without filter, `external_proportional` samples ALL rows (including bad), which is wrong.

### Changes

#### Script (`stratified_sampling.py`)

In `main()`, after env var reads, add:
```python
filter_column = environ_vars.get("SAMPLING_FILTER_COLUMN", "")
filter_value = environ_vars.get("SAMPLING_FILTER_VALUE", "")
```

Replace the `sampled_df = sampler.sample(...)` block (around line 550) with:
```python
if filter_column and filter_value and filter_column in df.columns:
    to_sample = df[df[filter_column] == filter_value].copy()
    to_passthrough = df[df[filter_column] != filter_value].copy()
    log(
        f"[INFO] Filter: sampling {len(to_sample)} rows "
        f"({filter_column}=={filter_value}), "
        f"passing through {len(to_passthrough)} rows"
    )

    if not to_sample.empty:
        sampled_df = sampler.sample(
            df=to_sample,
            strata_column=strata_column,
            target_size=split_target_size,
            strategy=sampling_strategy,
            min_samples_per_stratum=min_samples_per_stratum,
            variance_column=variance_column,
            reference_counts=reference_counts,
            multiplier=sampling_multiplier,
            allow_replacement=allow_replacement,
        )
    else:
        sampled_df = to_sample

    # Combine: sampled subset + passed-through rest
    sampled_df = pd.concat([sampled_df, to_passthrough], ignore_index=True)
else:
    # No filter — sample entire DataFrame (original behavior)
    sampled_df = sampler.sample(
        df=df,
        strata_column=strata_column,
        target_size=split_target_size,
        strategy=sampling_strategy,
        min_samples_per_stratum=min_samples_per_stratum,
        variance_column=variance_column,
        reference_counts=reference_counts,
        multiplier=sampling_multiplier,
        allow_replacement=allow_replacement,
    )
```

Also add to `__main__` block env var reads:
```python
SAMPLING_FILTER_COLUMN = os.environ.get("SAMPLING_FILTER_COLUMN", "")
SAMPLING_FILTER_VALUE = os.environ.get("SAMPLING_FILTER_VALUE", "")
```

And to environ_vars dict:
```python
"SAMPLING_FILTER_COLUMN": SAMPLING_FILTER_COLUMN,
"SAMPLING_FILTER_VALUE": SAMPLING_FILTER_VALUE,
```

#### Contract (`stratified_sampling_contract.py`)

Add to `optional_env_vars`:
```python
"SAMPLING_FILTER_COLUMN": "",
"SAMPLING_FILTER_VALUE": "",
```

#### Config (`config_stratified_sampling_step.py`)

Add 2 Tier 2 fields:
```python
sampling_filter_column: Optional[str] = Field(
    default=None,
    description="Column to filter on before sampling. Only matching rows are sampled; rest pass through unchanged.",
)

sampling_filter_value: Optional[str] = Field(
    default=None,
    description="Value to match in filter_column for sampling subset selection.",
)
```

Update `get_public_init_fields`:
```python
if self.sampling_filter_column is not None:
    sampling_fields["sampling_filter_column"] = self.sampling_filter_column
if self.sampling_filter_value is not None:
    sampling_fields["sampling_filter_value"] = self.sampling_filter_value
```

#### Builder (`builder_stratified_sampling_step.py`)

Add to `_get_environment_variables`:
```python
if self.config.sampling_filter_column:
    env_vars["SAMPLING_FILTER_COLUMN"] = self.config.sampling_filter_column
if self.config.sampling_filter_value:
    env_vars["SAMPLING_FILTER_VALUE"] = self.config.sampling_filter_value
```

### Backward Compatibility

When `SAMPLING_FILTER_COLUMN` is empty (default), the script follows the original path — samples the entire DataFrame. Zero change for existing pipelines.

### Verification

1. ruff check + format
2. Syntax validation
3. Test: with filter → only matching rows sampled, rest passed through
4. Test: without filter → original behavior
5. brazil-build release + brazil pb build

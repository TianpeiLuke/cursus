#!/usr/bin/env python
"""
Stratified Sampling Script

Applies stratified sampling to input data with four allocation strategies:
1. Balanced — equal samples per stratum (class imbalance correction)
2. Proportional with minimum — proportional allocation with floor constraints (causal analysis)
3. Optimal (Neyman) — variance-weighted allocation (minimizes sampling error)
4. External proportional — sample to match an external reference distribution with multiplier

Features:
- Sampling with replacement (allow_replacement) for oversampling when target > available
- NaN guard: warns and excludes NaN strata values
- Empty DataFrame guard: returns empty result gracefully
- Per-split diagnostics JSON output (requested vs achieved per stratum)
- Format preservation: reads and writes CSV/TSV/Parquet maintaining input format
- Split-aware: processes train/val splits for training job type, copies test unchanged
- Reference counts loaded from sidecar file (reference_counts.json) or env var fallback

Input: /opt/ml/processing/input/data/{split}/{split}_processed_data.{csv|tsv|parquet}
Output: /opt/ml/processing/output/{split}/{split}_processed_data.{csv|tsv|parquet}
Diagnostics: /opt/ml/processing/output/{split}/sampling_diagnostics.json
"""

import os
import argparse
import json
import logging
import shutil
import sys
import traceback
from pathlib import Path
from typing import Dict, Optional, Callable, Any

import pandas as pd

# Big-parquet streaming path: a parquet whose string column (e.g. `dialogue` — full
# conversations) sums to > 2 GiB across the file overflows Arrow's int32-offset string-array
# limit (2^31 bytes) when loaded in one shot via pandas.read_parquet -> to_pandas():
#   pyarrow.lib.ArrowCapacityError: array cannot contain more than 2147483646 bytes
# This is a GENERAL cursus issue (any large-text feature crosses it), not project-specific. Fix:
# never materialize the full file. Pass 1 reads only the tiny (id, tag) key columns and runs the
# existing allocation to pick which rows; Pass 2 streams row-groups in bounded batches, keeps the
# selected rows (by id, else by position), and writes them incrementally. Each batch's big column
# stays well under 2 GiB, and the sampled output is far under it. Test is copied byte-for-byte.
# CSV/TSV and the filter / external_proportional cases keep the original in-memory path.
STREAM_BATCH_ROWS = (
    200_000  # rows per pyarrow batch in the streaming passes (bounds per-batch bytes)
)
_ROW_POS_COL = "__cursus_row_pos__"  # transient positional key mapping Pass-1 selection -> Pass-2 rows


# --- Stratified Sampling Core Logic ---


class StratifiedSampler:
    """
    Stratified sampling implementation with four allocation strategies:
    1. Balanced allocation - for class imbalance
    2. Proportional with minimum constraints - for causal analysis
    3. Optimal allocation (Neyman) - for variance optimization
    4. External proportional - sample to match an external reference distribution
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.strategies = {
            "balanced": self._balanced_allocation,
            "proportional_min": self._proportional_with_min,
            "optimal": self._optimal_allocation,
            "external_proportional": self._external_proportional,
        }

    def sample(
        self,
        df: pd.DataFrame,
        strata_column: str,
        target_size: int,
        strategy: str = "balanced",
        min_samples_per_stratum: int = 10,
        variance_column: Optional[str] = None,
        reference_counts: Optional[Dict[str, int]] = None,
        multiplier: float = 1.0,
        allow_replacement: bool = False,
    ) -> pd.DataFrame:
        """
        Perform stratified sampling on a DataFrame.

        Args:
            df: Input DataFrame
            strata_column: Column name to stratify by
            target_size: Total desired sample size
            strategy: Sampling strategy ('balanced', 'proportional_min', 'optimal', 'external_proportional')
            min_samples_per_stratum: Minimum samples per stratum
            variance_column: Column for variance calculation (needed for optimal strategy)
            reference_counts: External reference distribution {stratum: count} (for external_proportional)
            multiplier: Multiplier for reference counts (e.g., 5.0 for 5× oversampling)
            allow_replacement: Allow sampling with replacement when target > available

        Returns:
            Sampled DataFrame
        """
        if strategy not in self.strategies:
            raise ValueError(
                f"Unknown strategy: {strategy}. Available: {list(self.strategies.keys())}"
            )

        # Guard: empty DataFrame
        if df.empty:
            logging.warning("Empty DataFrame received, returning empty result")
            return pd.DataFrame(columns=df.columns)

        # Guard: NaN in strata column
        nan_count = df[strata_column].isna().sum()
        if nan_count > 0:
            logging.warning(
                f"Found {nan_count} NaN values in strata column '{strata_column}'. "
                f"Excluding from sampling."
            )
            df = df.dropna(subset=[strata_column]).copy()

        # Get stratum information
        strata_info = self._get_strata_info(df, strata_column, variance_column)

        # Calculate allocation (external_proportional needs extra params)
        if strategy == "external_proportional":
            allocation = self.strategies[strategy](
                strata_info,
                target_size,
                min_samples_per_stratum,
                reference_counts=reference_counts,
                multiplier=multiplier,
            )
        else:
            allocation = self.strategies[strategy](
                strata_info, target_size, min_samples_per_stratum
            )

        # Perform sampling
        # When allow_replacement is True, uncap allocations that were limited by stratum size
        # (balanced/proportional_min/optimal all cap at info["size"], making replacement a no-op)
        if allow_replacement:
            num_strata = len(strata_info)
            desired_per_stratum = max(
                min_samples_per_stratum, target_size // num_strata
            )
            for stratum in allocation:
                if allocation[stratum] < desired_per_stratum:
                    allocation[stratum] = desired_per_stratum

        return self._perform_sampling(
            df, strata_column, allocation, allow_replacement=allow_replacement
        )

    def _get_strata_info(
        self,
        df: pd.DataFrame,
        strata_column: str,
        variance_column: Optional[str] = None,
    ) -> Dict:
        """Extract stratum size and variance information from DataFrame."""
        strata_info = {}

        for stratum in df[strata_column].unique():
            stratum_df = df[df[strata_column] == stratum]
            info = {"size": len(stratum_df)}

            if variance_column and variance_column in df.columns:
                info["variance"] = stratum_df[variance_column].var()
                info["std"] = stratum_df[variance_column].std()
            else:
                info["variance"] = 1.0
                info["std"] = 1.0

            strata_info[stratum] = info

        return strata_info

    def _balanced_allocation(
        self, strata_info: Dict, target_size: int, min_samples: int
    ) -> Dict[Any, int]:
        """
        Balanced allocation strategy - equal samples per stratum.
        Handles class imbalance by giving equal representation to all classes.
        """
        num_strata = len(strata_info)
        samples_per_stratum = max(min_samples, target_size // num_strata)

        allocation = {}
        total_allocated = 0

        for stratum, info in strata_info.items():
            # Don't exceed available samples in stratum
            allocated = min(samples_per_stratum, info["size"])
            allocation[stratum] = allocated
            total_allocated += allocated

        # Distribute remaining samples proportionally if we're under target
        remaining = target_size - total_allocated
        if remaining > 0:
            # Sort strata by available capacity (size - current allocation)
            available_capacity = {
                stratum: info["size"] - allocation[stratum]
                for stratum, info in strata_info.items()
            }

            # Distribute remaining samples to strata with capacity
            strata_with_capacity = [
                s for s, cap in available_capacity.items() if cap > 0
            ]
            if strata_with_capacity:
                extra_per_stratum = remaining // len(strata_with_capacity)
                for stratum in strata_with_capacity:
                    extra = min(extra_per_stratum, available_capacity[stratum])
                    allocation[stratum] += extra

        return allocation

    def _proportional_with_min(
        self, strata_info: Dict, target_size: int, min_samples: int
    ) -> Dict[Any, int]:
        """
        Proportional allocation with minimum constraints.
        Maintains representativeness while ensuring adequate samples for causal inference.
        """
        total_population = sum(info["size"] for info in strata_info.values())
        allocation = {}

        # First pass: allocate proportionally
        for stratum, info in strata_info.items():
            proportion = info["size"] / total_population
            proportional_size = int(target_size * proportion)
            allocation[stratum] = max(min_samples, proportional_size)

        # Second pass: adjust if we exceeded target due to minimum constraints
        total_allocated = sum(allocation.values())
        if total_allocated > target_size:
            # Scale down while respecting minimums
            excess = total_allocated - target_size
            adjustable_strata = {
                stratum: allocation[stratum] - min_samples
                for stratum in allocation
                if allocation[stratum] > min_samples
            }

            if sum(adjustable_strata.values()) >= excess:
                # Proportionally reduce from adjustable strata
                total_adjustable = sum(adjustable_strata.values())
                for stratum, adjustable in adjustable_strata.items():
                    reduction = int(excess * adjustable / total_adjustable)
                    allocation[stratum] -= reduction

        # Ensure we don't exceed available samples in each stratum
        for stratum, info in strata_info.items():
            allocation[stratum] = min(allocation[stratum], info["size"])

        return allocation

    def _optimal_allocation(
        self, strata_info: Dict, target_size: int, min_samples: int
    ) -> Dict[Any, int]:
        """
        Optimal allocation (Neyman) strategy.
        Minimizes sampling variance by allocating based on stratum size and variability.
        """
        # Calculate Neyman allocation: n_h = n * (N_h * S_h) / sum(N_i * S_i)
        numerators = {}
        total_numerator = 0

        for stratum, info in strata_info.items():
            numerator = info["size"] * info["std"]
            numerators[stratum] = numerator
            total_numerator += numerator

        allocation = {}
        for stratum, numerator in numerators.items():
            if total_numerator > 0:
                optimal_size = int(target_size * numerator / total_numerator)
            else:
                optimal_size = target_size // len(strata_info)

            # Apply minimum constraint and don't exceed stratum size
            allocation[stratum] = min(
                max(min_samples, optimal_size), strata_info[stratum]["size"]
            )

        return allocation

    def _external_proportional(
        self,
        strata_info: Dict,
        target_size: int,
        min_samples: int,
        reference_counts: Optional[Dict[str, int]] = None,
        multiplier: float = 1.0,
    ) -> Dict[Any, int]:
        """
        External proportional allocation — sample to match an external reference distribution.
        Each stratum gets reference_count × multiplier samples.
        """
        if not reference_counts:
            raise ValueError(
                "external_proportional strategy requires reference_counts "
                "(from sidecar file or REFERENCE_COUNTS_JSON env var)"
            )
        allocation = {}
        for stratum in strata_info:
            ref_count = reference_counts.get(str(stratum), 0)
            allocation[stratum] = max(min_samples, int(ref_count * multiplier))
        return allocation

    def _perform_sampling(
        self,
        df: pd.DataFrame,
        strata_column: str,
        allocation: Dict[Any, int],
        allow_replacement: bool = False,
    ) -> pd.DataFrame:
        """Perform the actual sampling based on allocation."""
        sampled_dfs = []

        for stratum, sample_size in allocation.items():
            if sample_size > 0:
                stratum_df = df[df[strata_column] == stratum]
                if len(stratum_df) >= sample_size:
                    sampled = stratum_df.sample(
                        n=sample_size, random_state=self.random_state
                    )
                elif allow_replacement and len(stratum_df) > 0:
                    sampled = stratum_df.sample(
                        n=sample_size, replace=True, random_state=self.random_state
                    )
                else:
                    sampled = stratum_df
                sampled_dfs.append(sampled)

        if sampled_dfs:
            return pd.concat(sampled_dfs, ignore_index=True)
        else:
            return pd.DataFrame()


# --- File I/O Helper Functions with Format Preservation ---


def _detect_file_format(split_dir: Path, split_name: str) -> tuple[Path, str]:
    """
    Detect the format of processed data file.

    Returns:
        Tuple of (file_path, format) where format is 'csv', 'tsv', or 'parquet'
    """
    # Try different formats in order of preference
    formats = [
        (f"{split_name}_processed_data.csv", "csv"),
        (f"{split_name}_processed_data.tsv", "tsv"),
        (f"{split_name}_processed_data.parquet", "parquet"),
    ]

    for filename, fmt in formats:
        file_path = split_dir / filename
        if file_path.exists():
            return file_path, fmt

    raise RuntimeError(
        f"No processed data file found in {split_dir}. "
        f"Looked for: {[f[0] for f in formats]}"
    )


def _read_processed_data(input_dir: str, split_name: str) -> tuple[pd.DataFrame, str]:
    """
    Read processed data from tabular_preprocessing output structure.
    Automatically detects and preserves the input format.

    Returns:
        Tuple of (DataFrame, format) where format is 'csv', 'tsv', or 'parquet'
    """
    input_path = Path(input_dir)
    split_dir = input_path / split_name

    # Detect format and read file
    file_path, detected_format = _detect_file_format(split_dir, split_name)

    if detected_format == "csv":
        df = pd.read_csv(file_path)
    elif detected_format == "tsv":
        df = pd.read_csv(file_path, sep="\t")
    elif detected_format == "parquet":
        df = pd.read_parquet(file_path)
    else:
        raise RuntimeError(f"Unsupported format: {detected_format}")

    return df, detected_format


def _save_sampled_data(
    df: pd.DataFrame,
    output_dir: str,
    split_name: str,
    output_format: str,
    logger: Callable[[str], None],
):
    """
    Save sampled data maintaining the same folder structure and format as input.

    Args:
        df: DataFrame to save
        output_dir: Output directory path
        split_name: Name of the split (train/val/test)
        output_format: Format to save in ('csv', 'tsv', or 'parquet')
        logger: Logger function
    """
    output_path = Path(output_dir)
    split_dir = output_path / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    # Determine file extension and save based on format
    if output_format == "csv":
        output_file = split_dir / f"{split_name}_processed_data.csv"
        df.to_csv(output_file, index=False)
    elif output_format == "tsv":
        output_file = split_dir / f"{split_name}_processed_data.tsv"
        df.to_csv(output_file, sep="\t", index=False)
    elif output_format == "parquet":
        output_file = split_dir / f"{split_name}_processed_data.parquet"
        df.to_parquet(output_file, index=False)
    else:
        raise RuntimeError(f"Unsupported output format: {output_format}")

    logger(f"[INFO] Saved {output_file} (format={output_format}, shape={df.shape})")


# --- Streaming parquet path (avoids the >2 GiB Arrow int32-offset overflow) ---


def _read_strata_index_parquet(
    file_path: Path,
    strata_column: str,
    id_column: Optional[str],
    log: Callable[[str], None],
) -> pd.DataFrame:
    """Pass 1: read ONLY the tiny key columns — the strata (tag) column, plus the id column when
    present — never the multi-GiB text column.

    Projecting to just these small columns keeps every Arrow array tiny, so this never hits the
    2^31 offset limit even on a very large file. Always synthesizes ``_ROW_POS_COL`` (ordinal
    position across the file) as a fallback selection key. Returns [strata_column, _ROW_POS_COL]
    and, if id_column exists in the schema, [id_column].
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(file_path))
    names = pf.schema_arrow.names
    if strata_column not in names:
        raise RuntimeError(
            f"Strata column '{strata_column}' not found in {file_path.name}. "
            f"Available columns: {names}"
        )
    have_id = bool(id_column) and id_column in names
    cols = [strata_column] + ([id_column] if have_id else [])
    frames = []
    pos = 0
    for batch in pf.iter_batches(batch_size=STREAM_BATCH_ROWS, columns=cols):
        n = batch.num_rows
        data = {strata_column: batch.column(0).to_pandas().values}
        if have_id:
            data[id_column] = batch.column(1).to_pandas().values
        data[_ROW_POS_COL] = range(pos, pos + n)
        frames.append(pd.DataFrame(data))
        pos += n
    proj = f"[{strata_column}" + (f", {id_column}]" if have_id else "]")
    log(
        f"[INFO] Pass 1: scanned {pos:,} rows of {file_path.name} (key-only projection {proj})"
    )
    empty_cols = cols + [_ROW_POS_COL]
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=empty_cols)
    )


def _write_selected_rows_parquet(
    file_path: Path,
    output_file: Path,
    log: Callable[[str], None],
    selected_ids=None,
    id_column: Optional[str] = None,
    selected_positions: Optional[set] = None,
) -> int:
    """Pass 2: stream the full file row-group-by-row-group, keep the selected rows, write them out
    incrementally with a single ParquetWriter.

    Selection is BY ID when (id_column, selected_ids) are given (order-independent, the robust
    primary path); falls back to BY POSITION otherwise. Each streamed batch is <= STREAM_BATCH_ROWS
    rows, so its big text slice is far under 2 GiB; the selected output is likewise well under the
    limit. Never holds the whole file.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    by_id = selected_ids is not None and id_column is not None
    pf = pq.ParquetFile(str(file_path))
    id_col_idx = pf.schema_arrow.names.index(id_column) if by_id else None

    writer = None
    written = 0
    pos = 0
    try:
        for batch in pf.iter_batches(batch_size=STREAM_BATCH_ROWS):
            n = batch.num_rows
            if by_id:
                ids = batch.column(id_col_idx).to_pandas()
                take = [i for i, v in enumerate(ids) if v in selected_ids]
            else:
                take = [p - pos for p in range(pos, pos + n) if p in selected_positions]
            pos += n
            if not take:
                continue
            tbl = pa.Table.from_batches([batch]).take(take)
            if writer is None:
                output_file.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(str(output_file), tbl.schema)
            writer.write_table(tbl)
            written += tbl.num_rows
    finally:
        if writer is not None:
            writer.close()
    log(
        f"[INFO] Pass 2: wrote {written:,} selected rows -> {output_file} "
        f"(by {'id' if by_id else 'position'})"
    )
    return written


def _stratified_sample_parquet_streaming(
    input_file: Path,
    output_file: Path,
    strata_column: str,
    id_column: Optional[str],
    sampler: "StratifiedSampler",
    split_target_size_fn: Callable[[int], int],
    sampling_strategy: str,
    min_samples_per_stratum: int,
    variance_column: Optional[str],
    reference_counts: Optional[Dict[str, int]],
    multiplier: float,
    allow_replacement: bool,
    log: Callable[[str], None],
) -> Dict[str, Any]:
    """Two-pass streaming stratified sample for a big parquet split.

    Pass 1 reads the tiny (id, tag) key columns and runs the SAME allocation logic on that index to
    pick which rows to keep. Pass 2 streams the full file and writes only those rows — selecting BY
    ID (order-independent) when a unique id column is present, else BY POSITION. ``variance_column``
    (optimal strategy) is unsupported here (the value column isn't in the key projection) — falls
    back to uniform variance, which is fine for the balanced strategy. Returns a diagnostics dict.
    """
    idx = _read_strata_index_parquet(input_file, strata_column, id_column, log)
    input_size = len(idx)
    if input_size == 0:
        log(f"[WARNING] {input_file.name} has 0 rows; writing empty output")
        shutil.copyfile(input_file, output_file)
        return {"input_size": 0, "output_size": 0, "per_stratum": {}}

    # Prefer id-based selection; require the id column present AND unique (else id-set membership in
    # Pass 2 would pull duplicate-id rows and break the exact sampled count — fall back to position).
    use_id = bool(id_column) and id_column in idx.columns
    if use_id and idx[id_column].nunique() != input_size:
        log(
            f"[WARNING] id column '{id_column}' has duplicate values; "
            f"falling back to positional selection to preserve exact sample counts."
        )
        use_id = False

    split_target = split_target_size_fn(input_size)
    sampled_idx = sampler.sample(
        df=idx,
        strata_column=strata_column,
        target_size=split_target,
        strategy=sampling_strategy,
        min_samples_per_stratum=min_samples_per_stratum,
        variance_column=None,  # value column not available in the key projection
        reference_counts=reference_counts,
        multiplier=multiplier,
        allow_replacement=allow_replacement,
    )

    if use_id:
        selected_ids = set(sampled_idx[id_column].tolist())
        log(
            f"[INFO] Selected {len(selected_ids):,} of {input_size:,} rows by id '{id_column}' "
            f"(target {split_target:,}); streaming full columns for the write"
        )
        written = _write_selected_rows_parquet(
            input_file, output_file, log, selected_ids=selected_ids, id_column=id_column
        )
    else:
        selected_positions = set(int(p) for p in sampled_idx[_ROW_POS_COL].tolist())
        log(
            f"[INFO] Selected {len(selected_positions):,} of {input_size:,} rows by position "
            f"(target {split_target:,}); streaming full columns for the write"
        )
        written = _write_selected_rows_parquet(
            input_file, output_file, log, selected_positions=selected_positions
        )

    avail = idx[strata_column].value_counts().to_dict()
    samp = sampled_idx[strata_column].value_counts().to_dict()
    per_stratum = {
        str(s): {
            "available": int(avail.get(s, 0)),
            "sampled": int(samp.get(s, 0)),
            "replacement_used": int(samp.get(s, 0)) > int(avail.get(s, 0)),
        }
        for s in samp
    }
    return {
        "input_size": input_size,
        "output_size": written,
        "per_stratum": per_stratum,
    }


# --- Main Processing Logic ---


def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Main logic for stratified sampling, following tabular_preprocessing format.

    Args:
        input_paths: Dictionary of input paths with logical names
        output_paths: Dictionary of output paths with logical names
        environ_vars: Dictionary of environment variables
        job_args: Command line arguments
        logger: Optional logger object (defaults to print if None)

    Returns:
        Dictionary of sampled DataFrames by split name
    """
    # Extract parameters from arguments and environment variables
    job_type = job_args.job_type
    strata_column = environ_vars.get("STRATA_COLUMN")
    sampling_strategy = environ_vars.get("SAMPLING_STRATEGY", "balanced")
    target_sample_size = int(environ_vars.get("TARGET_SAMPLE_SIZE", 1000))
    # Optional per-split override for the VAL split only. TARGET_SAMPLE_SIZE applies to every split
    # (train/val/test) identically; with balanced + allow_replacement a large target oversamples the
    # minority class up to the target for EACH split, so val can balloon to the same size as train.
    # VAL_TARGET_SAMPLE_SIZE (0/unset = disabled → fall back to TARGET_SAMPLE_SIZE) lets callers keep
    # a large train sample while capping val, so downstream per-epoch + post-training eval stays cheap.
    val_target_sample_size = int(environ_vars.get("VAL_TARGET_SAMPLE_SIZE", 0)) or None
    min_samples_per_stratum = int(environ_vars.get("MIN_SAMPLES_PER_STRATUM", 10))
    variance_column = environ_vars.get("VARIANCE_COLUMN")
    random_state = int(environ_vars.get("RANDOM_STATE", 42))
    sampling_multiplier = float(environ_vars.get("SAMPLING_MULTIPLIER", "1.0"))
    allow_replacement = environ_vars.get("ALLOW_REPLACEMENT", "false").lower() == "true"
    filter_column = environ_vars.get("SAMPLING_FILTER_COLUMN", "")
    filter_value = environ_vars.get("SAMPLING_FILTER_VALUE", "")
    # ID column for the big-parquet streaming path: sampling only needs (id, tag), then rows are
    # selected by id. Optional env var; defaults to the common primary key `order_id`. If the column
    # is absent or non-unique, the streaming path falls back to positional selection.
    id_column = environ_vars.get("ID_COLUMN", "order_id")

    # Extract paths - no defaults, require explicit paths
    input_data_dir = input_paths.get("input_data")
    output_dir = output_paths.get("processed_data")

    # Validate required paths
    if not input_data_dir:
        raise ValueError("input_paths must contain 'input_data' key")
    if not output_dir:
        raise ValueError("output_paths must contain 'processed_data' key")

    # Use print function if no logger is provided
    log = logger or print

    # Validate required parameters
    if not strata_column:
        raise RuntimeError("STRATA_COLUMN environment variable must be set.")

    valid_strategies = [
        "balanced",
        "proportional_min",
        "optimal",
        "external_proportional",
    ]
    if sampling_strategy not in valid_strategies:
        raise RuntimeError(
            f"Invalid SAMPLING_STRATEGY: {sampling_strategy}. "
            f"Must be one of: {valid_strategies}"
        )

    # Load reference counts for external_proportional strategy
    reference_counts = None
    if sampling_strategy == "external_proportional":
        reference_path = Path(input_data_dir) / "reference_counts.json"
        if reference_path.exists():
            try:
                reference_counts = json.loads(reference_path.read_text())
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in reference_counts.json ({reference_path}): {e}"
                )
            log(f"[INFO] Loaded reference counts from sidecar: {reference_path}")
        else:
            ref_json = environ_vars.get("REFERENCE_COUNTS_JSON", "")
            if ref_json:
                try:
                    reference_counts = json.loads(ref_json)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON in REFERENCE_COUNTS_JSON env var: {e}"
                    )
                log("[INFO] Loaded reference counts from REFERENCE_COUNTS_JSON env var")
            else:
                raise RuntimeError(
                    "external_proportional strategy requires reference_counts.json "
                    "sidecar file in input directory or REFERENCE_COUNTS_JSON env var"
                )

    # Initialize sampler
    sampler = StratifiedSampler(random_state=random_state)

    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    log(f"[INFO] Starting stratified sampling with strategy: {sampling_strategy}")
    log(f"[INFO] Strata column: {strata_column}")
    log(f"[INFO] Target sample size: {target_sample_size}")
    log(f"[INFO] Min samples per stratum: {min_samples_per_stratum}")
    log(
        f"[INFO] Multiplier: {sampling_multiplier}, allow_replacement: {allow_replacement}"
    )

    # Determine which splits to process based on job_type
    if job_type == "training":
        # For training job_type, process train and val splits (not test)
        splits_to_process = ["train", "val"]
        log("[INFO] Training job type detected - processing train and val splits only")
    else:
        # For other job types, process only that specific split
        splits_to_process = [job_type]
        log(f"[INFO] Non-training job type detected - processing {job_type} split only")

    sampled_splits = {}

    # Process each split
    for split_name in splits_to_process:
        try:
            log(f"[INFO] Processing {split_name} split...")

            # Streaming path: for a PARQUET split with NO row-level filter, use the two-pass
            # id/position sampler so we never materialize the full file (a >2 GiB text column
            # overflows Arrow's 2^31 offset limit in a single-array to_pandas()). The filter case
            # and CSV/TSV keep the in-memory path below; external_proportional keeps it too.
            split_dir = Path(input_data_dir) / split_name
            parquet_in = split_dir / f"{split_name}_processed_data.parquet"
            use_streaming = (
                parquet_in.exists()
                and not (filter_column and filter_value)
                and sampling_strategy != "external_proportional"
            )
            if use_streaming:
                out_file = (
                    Path(output_dir)
                    / split_name
                    / f"{split_name}_processed_data.parquet"
                )

                # Use the val-specific cap for the val split when configured; else TARGET_SAMPLE_SIZE.
                _effective_target = (
                    val_target_sample_size
                    if (split_name == "val" and val_target_sample_size is not None)
                    else target_sample_size
                )

                def _target_fn(n, _t=_effective_target):
                    return min(_t, n)

                diag = _stratified_sample_parquet_streaming(
                    input_file=parquet_in,
                    output_file=out_file,
                    strata_column=strata_column,
                    id_column=id_column,
                    sampler=sampler,
                    split_target_size_fn=_target_fn,
                    sampling_strategy=sampling_strategy,
                    min_samples_per_stratum=min_samples_per_stratum,
                    variance_column=None,  # value col not in the key projection (balanced uses none)
                    reference_counts=reference_counts,
                    multiplier=sampling_multiplier,
                    allow_replacement=allow_replacement,
                    log=log,
                )
                log(
                    f"[INFO] Sampled {split_name} (streaming): "
                    f"{diag['output_size']:,} rows from {diag['input_size']:,}"
                )
                diagnostics = {
                    "strategy": sampling_strategy,
                    "strata_column": strata_column,
                    "input_size": diag["input_size"],
                    "output_size": diag["output_size"],
                    "allow_replacement": allow_replacement,
                    "multiplier": sampling_multiplier,
                    "streaming": True,
                    "per_stratum": diag["per_stratum"],
                }
                diag_path = Path(output_dir) / split_name / "sampling_diagnostics.json"
                diag_path.parent.mkdir(parents=True, exist_ok=True)
                diag_path.write_text(json.dumps(diagnostics, indent=2, default=str))
                log(f"[INFO] Saved diagnostics to {diag_path}")
                sampled_splits[split_name] = (
                    None  # streamed to disk; not held in memory
                )
                continue

            # Read the processed data from tabular_preprocessing output
            df, detected_format = _read_processed_data(input_data_dir, split_name)
            log(
                f"[INFO] Loaded {split_name} data with shape: {df.shape}, format: {detected_format}"
            )

            # Validate strata column exists
            if strata_column not in df.columns:
                raise RuntimeError(
                    f"Strata column '{strata_column}' not found in {split_name} data. Available columns: {df.columns.tolist()}"
                )

            # Check if variance column exists (for optimal strategy)
            effective_variance_column = variance_column
            if (
                sampling_strategy == "optimal"
                and variance_column
                and variance_column not in df.columns
            ):
                log(
                    f"[WARNING] Variance column '{variance_column}' not found. Using default variance for optimal allocation."
                )
                effective_variance_column = None

            # Calculate target size for this split
            # For external_proportional, target_size is ignored (allocation from reference_counts)
            if sampling_strategy == "external_proportional":
                split_target_size = target_sample_size
            else:
                # Use the val-specific cap for the val split when configured; else TARGET_SAMPLE_SIZE.
                effective_target = (
                    val_target_sample_size
                    if (split_name == "val" and val_target_sample_size is not None)
                    else target_sample_size
                )
                split_target_size = min(effective_target, len(df))

            # Apply filter: sample only matching rows, pass rest through
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
                        variance_column=effective_variance_column,
                        reference_counts=reference_counts,
                        multiplier=sampling_multiplier,
                        allow_replacement=allow_replacement,
                    )
                else:
                    sampled_df = to_sample

                sampled_df = pd.concat([sampled_df, to_passthrough], ignore_index=True)
            else:
                # No filter — sample entire DataFrame (original behavior)
                sampled_df = sampler.sample(
                    df=df,
                    strata_column=strata_column,
                    target_size=split_target_size,
                    strategy=sampling_strategy,
                    min_samples_per_stratum=min_samples_per_stratum,
                    variance_column=effective_variance_column,
                    reference_counts=reference_counts,
                    multiplier=sampling_multiplier,
                    allow_replacement=allow_replacement,
                )

            log(
                f"[INFO] Sampled {split_name} data: {len(sampled_df)} rows from {len(df)} original rows"
            )

            # Log stratum distribution
            strata_counts = sampled_df[strata_column].value_counts().sort_index()
            log(f"[INFO] {split_name} stratum distribution: {dict(strata_counts)}")

            # Save sampled data (preserve format)
            _save_sampled_data(sampled_df, output_dir, split_name, detected_format, log)
            sampled_splits[split_name] = sampled_df

            # Save sampling diagnostics
            diagnostics = {
                "strategy": sampling_strategy,
                "strata_column": strata_column,
                "input_size": len(df),
                "output_size": len(sampled_df),
                "allow_replacement": allow_replacement,
                "multiplier": sampling_multiplier,
                "per_stratum": {
                    str(s): {
                        "available": int((df[strata_column] == s).sum()),
                        "sampled": int((sampled_df[strata_column] == s).sum()),
                        "replacement_used": int((sampled_df[strata_column] == s).sum())
                        > int((df[strata_column] == s).sum()),
                    }
                    for s in sampled_df[strata_column].unique()
                },
            }
            diag_path = Path(output_dir) / split_name / "sampling_diagnostics.json"
            diag_path.parent.mkdir(parents=True, exist_ok=True)
            diag_path.write_text(json.dumps(diagnostics, indent=2, default=str))
            log(f"[INFO] Saved diagnostics to {diag_path}")

        except Exception as e:
            log(f"[ERROR] Failed to process {split_name} split: {str(e)}")
            raise

    # For training job_type, also copy test split unchanged (if it exists).
    # If test is parquet, copy the file BYTE-FOR-BYTE (shutil.copyfile) instead of
    # read->to_pandas->to_parquet: a >2 GiB text column would hit the same Arrow overflow on read,
    # and "unchanged" means an exact copy anyway.
    if job_type == "training":
        try:
            test_parquet = Path(input_data_dir) / "test" / "test_processed_data.parquet"
            if test_parquet.exists():
                out_file = Path(output_dir) / "test" / "test_processed_data.parquet"
                out_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(test_parquet, out_file)
                log(f"[INFO] Copied test split unchanged (byte-for-byte) -> {out_file}")
                sampled_splits["test"] = None  # copied on disk; not held in memory
            else:
                test_df, test_format = _read_processed_data(input_data_dir, "test")
                log(
                    f"[INFO] Copying test split unchanged (shape: {test_df.shape}, format: {test_format})"
                )
                _save_sampled_data(test_df, output_dir, "test", test_format, log)
                sampled_splits["test"] = test_df
        except Exception as e:
            log(f"[WARNING] Could not copy test split: {str(e)}")

    log("[INFO] Stratified sampling complete.")
    return sampled_splits


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--job_type",
            type=str,
            required=True,
            help="Job type (e.g., 'training', 'validation', 'testing', 'calibration', 'sampling')",
        )
        args = parser.parse_args()

        # Read configuration from environment variables
        STRATA_COLUMN = os.environ.get("STRATA_COLUMN")
        if not STRATA_COLUMN:
            raise RuntimeError("STRATA_COLUMN environment variable must be set.")

        SAMPLING_STRATEGY = os.environ.get("SAMPLING_STRATEGY", "balanced")
        TARGET_SAMPLE_SIZE = int(os.environ.get("TARGET_SAMPLE_SIZE", 1000))
        MIN_SAMPLES_PER_STRATUM = int(os.environ.get("MIN_SAMPLES_PER_STRATUM", 10))
        VARIANCE_COLUMN = os.environ.get("VARIANCE_COLUMN")  # Optional
        RANDOM_STATE = int(os.environ.get("RANDOM_STATE", 42))
        SAMPLING_MULTIPLIER = float(os.environ.get("SAMPLING_MULTIPLIER", "1.0"))
        ALLOW_REPLACEMENT = os.environ.get("ALLOW_REPLACEMENT", "false")
        REFERENCE_COUNTS_JSON = os.environ.get("REFERENCE_COUNTS_JSON", "")
        SAMPLING_FILTER_COLUMN = os.environ.get("SAMPLING_FILTER_COLUMN", "")
        SAMPLING_FILTER_VALUE = os.environ.get("SAMPLING_FILTER_VALUE", "")
        ID_COLUMN = os.environ.get(
            "ID_COLUMN", "order_id"
        )  # big-parquet streaming id-selection key

        # Define standard SageMaker paths - use contract-declared paths directly
        INPUT_DATA_DIR = "/opt/ml/processing/input/data"
        OUTPUT_DIR = "/opt/ml/processing/output"

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logger = logging.getLogger(__name__)

        # Log key parameters
        logger.info("Starting stratified sampling with parameters:")
        logger.info(f"  Job Type: {args.job_type}")
        logger.info(f"  Strata Column: {STRATA_COLUMN}")
        logger.info(f"  Sampling Strategy: {SAMPLING_STRATEGY}")
        logger.info(f"  Target Sample Size: {TARGET_SAMPLE_SIZE}")
        logger.info(f"  Min Samples Per Stratum: {MIN_SAMPLES_PER_STRATUM}")
        logger.info(f"  Variance Column: {VARIANCE_COLUMN}")
        logger.info(f"  Random State: {RANDOM_STATE}")
        logger.info(f"  Input Directory: {INPUT_DATA_DIR}")
        logger.info(f"  Output Directory: {OUTPUT_DIR}")

        # Set up path dictionaries
        input_paths = {"input_data": INPUT_DATA_DIR}
        output_paths = {"processed_data": OUTPUT_DIR}

        # Environment variables dictionary
        environ_vars = {
            "STRATA_COLUMN": STRATA_COLUMN,
            "SAMPLING_STRATEGY": SAMPLING_STRATEGY,
            "TARGET_SAMPLE_SIZE": str(TARGET_SAMPLE_SIZE),
            "MIN_SAMPLES_PER_STRATUM": str(MIN_SAMPLES_PER_STRATUM),
            "VARIANCE_COLUMN": VARIANCE_COLUMN,
            "RANDOM_STATE": str(RANDOM_STATE),
            "SAMPLING_MULTIPLIER": str(SAMPLING_MULTIPLIER),
            "ALLOW_REPLACEMENT": ALLOW_REPLACEMENT,
            "REFERENCE_COUNTS_JSON": REFERENCE_COUNTS_JSON,
            "SAMPLING_FILTER_COLUMN": SAMPLING_FILTER_COLUMN,
            "SAMPLING_FILTER_VALUE": SAMPLING_FILTER_VALUE,
            "ID_COLUMN": ID_COLUMN,
        }

        # Execute the main processing logic
        result = main(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,
            job_args=args,
            logger=logger.info,
        )

        # Log completion summary (streamed splits return None — written to disk, not held in memory)
        splits_summary = ", ".join(
            [
                f"{name}: {df.shape if df is not None else 'streamed-to-disk'}"
                for name, df in result.items()
            ]
        )
        logger.info(
            f"Stratified sampling completed successfully. Splits: {splits_summary}"
        )
        sys.exit(0)

    except Exception as e:
        logging.error(f"Error in stratified sampling script: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)

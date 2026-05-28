"""
Custom TabularPreprocessing script for Munged Address — Sampling Phase.

Reads bad addresses (DATA) + good addresses (DATA_SECONDARY),
deduplicates on (saddr, marketplaceId), adds __cohort__ column,
and emits reference_counts.json for downstream StratifiedSampling.

Source: FZ 29d16b (adapted from 04_process_address.ipynb)
"""

import argparse
import json
import os
import sys
import logging
from pathlib import Path
from typing import Dict

import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _read_cradle_shards(directory, log):
    """Read Cradle output shards (sig.csv + res.csv or part-* files)."""
    dir_path = Path(directory)
    dfs = []

    sig_file = dir_path / "sig.csv"
    res_file = dir_path / "res.csv"
    if sig_file.exists() and res_file.exists():
        header = pd.read_csv(sig_file)
        df = pd.read_csv(
            res_file,
            names=header.columns.tolist(),
            escapechar="\\",
            on_bad_lines="skip",
        )
        log(f"[INFO] Read sig/res format: {len(df)} rows from {dir_path}")
        return df

    parquet_files = list(dir_path.rglob("*.parquet"))
    csv_files = list(dir_path.rglob("*.csv"))

    if parquet_files:
        for f in sorted(parquet_files):
            if f.stat().st_size > 0:
                dfs.append(pd.read_parquet(f))
    elif csv_files:
        for f in sorted(csv_files):
            if f.stat().st_size > 0 and f.name != "sig.csv":
                dfs.append(pd.read_csv(f, escapechar="\\", on_bad_lines="skip"))

    if dfs:
        result = pd.concat(dfs, ignore_index=True)
        log(f"[INFO] Read {len(dfs)} shards: {len(result)} rows from {dir_path}")
        return result
    return pd.DataFrame()


def main(input_paths, output_paths, environ_vars, job_args, logger=None):
    log = logger or globals()["logger"].info

    dedup_columns = environ_vars.get("DEDUP_COLUMNS", "saddr,marketplaceId").split(",")
    sort_columns = environ_vars.get("SORT_COLUMNS", "marketplaceId,orderDate").split(
        ","
    )
    cohort_column = environ_vars.get("COHORT_COLUMN", "__cohort__")
    sampling_multiplier = int(environ_vars.get("SAMPLING_MULTIPLIER", "5"))
    strata_column = environ_vars.get("STRATA_COLUMN", "marketplaceId")
    output_format = environ_vars.get("OUTPUT_FORMAT", "parquet").lower()

    data_dir = input_paths["DATA"]
    data_secondary_dir = input_paths.get("DATA_SECONDARY")
    output_dir = Path(output_paths["processed_data"])
    output_dir.mkdir(parents=True, exist_ok=True)

    df_bad = _read_cradle_shards(data_dir, log)
    log(f"[INFO] Bad addresses loaded: {len(df_bad)} rows")

    df_good = pd.DataFrame()
    if data_secondary_dir and Path(data_secondary_dir).exists():
        df_good = _read_cradle_shards(data_secondary_dir, log)
        log(f"[INFO] Good addresses loaded: {len(df_good)} rows")

    if sort_columns[0] in df_bad.columns:
        df_bad = df_bad.sort_values(by=sort_columns)
    df_bad = df_bad.drop_duplicates(subset=dedup_columns, keep="last")
    log(f"[INFO] Bad after dedup: {len(df_bad)} rows")

    if not df_good.empty:
        if sort_columns[0] in df_good.columns:
            df_good = df_good.sort_values(by=sort_columns)
        df_good = df_good.drop_duplicates(subset=dedup_columns, keep="last")
        log(f"[INFO] Good after dedup: {len(df_good)} rows")

    df_bad[cohort_column] = "bad"
    if not df_good.empty:
        df_good[cohort_column] = "good"

    reference_counts = df_bad[strata_column].value_counts().to_dict()
    reference_counts = {
        str(k): int(v) * sampling_multiplier for k, v in reference_counts.items()
    }

    ref_counts_path = output_dir / "reference_counts.json"
    with open(ref_counts_path, "w") as f:
        json.dump(reference_counts, f, indent=2)
    log(f"[INFO] Saved reference_counts.json: {reference_counts}")

    df_combined = pd.concat([df_bad, df_good], ignore_index=True)
    log(
        f"[INFO] Combined dataset: {len(df_combined)} rows "
        f"(bad={len(df_bad)}, good={len(df_good)})"
    )

    keep_cols = ["saddr", "marketplaceId", "orderDate", cohort_column]
    available_cols = [c for c in keep_cols if c in df_combined.columns]
    df_combined = df_combined[available_cols]

    output_file = output_dir / f"processed_data.{output_format}"
    if output_format == "parquet":
        df_combined.to_parquet(output_file, index=False)
    else:
        df_combined.to_csv(output_file, index=False)
    log(f"[INFO] Saved {output_file} ({len(df_combined)} rows)")


if __name__ == "__main__":
    input_paths = {
        "DATA": "/opt/ml/processing/input/data",
        "DATA_SECONDARY": "/opt/ml/processing/input/data_secondary",
    }
    output_paths = {"processed_data": "/opt/ml/processing/output"}

    environ_vars = {
        "DEDUP_COLUMNS": os.environ.get("DEDUP_COLUMNS", "saddr,marketplaceId"),
        "SORT_COLUMNS": os.environ.get("SORT_COLUMNS", "marketplaceId,orderDate"),
        "COHORT_COLUMN": os.environ.get("COHORT_COLUMN", "__cohort__"),
        "SAMPLING_MULTIPLIER": os.environ.get("SAMPLING_MULTIPLIER", "5"),
        "STRATA_COLUMN": os.environ.get("STRATA_COLUMN", "marketplaceId"),
        "OUTPUT_FORMAT": os.environ.get("OUTPUT_FORMAT", "parquet"),
    }

    args = argparse.Namespace()

    try:
        main(input_paths, output_paths, environ_vars, args)
        sys.exit(0)
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)

#!/usr/bin/env python
import os
import gzip
import tempfile
import shutil
import csv
import json
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# --- Helper Functions ---

def _is_gzipped(path: str) -> bool:
    return path.lower().endswith(".gz")

def _detect_separator_from_sample(sample_lines: str) -> str:
    """Use csv.Sniffer to detect a delimiter, defaulting to comma."""
    try:
        dialect = csv.Sniffer().sniff(sample_lines)
        return dialect.delimiter
    except Exception:
        return ","

def peek_json_format(file_path: Path, open_func=open) -> str:
    """Check if the JSON file is in JSON Lines or regular format."""
    try:
        with open_func(str(file_path), "rt") as f:
            first_char = f.read(1)
            if not first_char:
                raise ValueError("Empty file")
            f.seek(0)
            first_line = f.readline().strip()
            try:
                json.loads(first_line)
                return "lines" if first_char != "[" else "regular"
            except json.JSONDecodeError:
                f.seek(0)
                json.loads(f.read())
                return "regular"
    except (json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(f"Error checking JSON format for {file_path}: {e}")

def _read_json_file(file_path: Path) -> pd.DataFrame:
    """Read a JSON or JSON Lines file into a DataFrame."""
    open_func = gzip.open if _is_gzipped(str(file_path)) else open
    fmt = peek_json_format(file_path, open_func)
    if fmt == "lines":
        return pd.read_json(str(file_path), lines=True, compression="infer")
    else:
        with open_func(str(file_path), "rt") as f:
            data = json.load(f)
        return pd.json_normalize(data if isinstance(data, list) else [data])

def _read_file_to_df(file_path: Path) -> pd.DataFrame:
    """Read a single file (CSV, TSV, JSON, Parquet) into a DataFrame."""
    suffix = file_path.suffix.lower()
    if suffix == ".gz":
        inner_ext = Path(file_path.stem).suffix.lower()
        if inner_ext in [".csv", ".tsv"]:
            with gzip.open(str(file_path), "rt") as f:
                sep = _detect_separator_from_sample(f.readline() + f.readline())
            return pd.read_csv(str(file_path), sep=sep, compression="gzip")
        elif inner_ext == ".json":
            return _read_json_file(file_path)
        elif inner_ext.endswith(".parquet"):
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                with gzip.open(str(file_path), "rb") as f_in, open(tmp.name, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                df = pd.read_parquet(tmp.name)
            os.unlink(tmp.name)
            return df
        else:
            raise ValueError(f"Unsupported gzipped file type: {file_path}")
    elif suffix in [".csv", ".tsv"]:
        with open(str(file_path), "rt") as f:
            sep = _detect_separator_from_sample(f.readline() + f.readline())
        return pd.read_csv(str(file_path), sep=sep)
    elif suffix == ".json":
        return _read_json_file(file_path)
    elif suffix.endswith(".parquet"):
        return pd.read_parquet(str(file_path))
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

def combine_shards(input_dir: str) -> pd.DataFrame:
    """Detect and combine all supported data shards in a directory."""
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise RuntimeError(f"Input directory does not exist: {input_dir}")
    patterns = [
        "part-*.csv", "part-*.csv.gz", "part-*.json", "part-*.json.gz",
        "part-*.parquet", "part-*.snappy.parquet", "part-*.parquet.gz"
    ]
    all_shards = sorted([p for pat in patterns for p in input_path.glob(pat)])
    if not all_shards:
        raise RuntimeError(f"No CSV/JSON/Parquet shards found under {input_dir}")
    try:
        dfs = [_read_file_to_df(shard) for shard in all_shards]
        return pd.concat(dfs, axis=0, ignore_index=True)
    except Exception as e:
        raise RuntimeError(f"Failed to read or concatenate shards: {e}")

# --- Main Processing Logic ---

def main(job_type: str, label_field: str, train_ratio: float, test_val_ratio: float, input_base_dir: str, output_dir: str):
    """
    Main logic for preprocessing data, now refactored for testability.
    """
    # 1. Setup paths
    input_data_dir = os.path.join(input_base_dir, "data")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 2. Combine data shards
    print(f"[INFO] Combining data shards from {input_data_dir}…")
    df = combine_shards(input_data_dir)
    print(f"[INFO] Combined data shape: {df.shape}")

    # 3. Process columns and labels
    df.columns = [col.replace("__DOT__", ".") for col in df.columns]
    if label_field not in df.columns:
        raise RuntimeError(f"Label field '{label_field}' not found in columns: {df.columns.tolist()}")

    if not pd.api.types.is_numeric_dtype(df[label_field]):
        unique_labels = sorted(df[label_field].dropna().unique())
        label_map = {val: idx for idx, val in enumerate(unique_labels)}
        df[label_field] = df[label_field].map(label_map)
    
    df[label_field] = pd.to_numeric(df[label_field], errors="coerce").astype("Int64")
    df.dropna(subset=[label_field], inplace=True)
    df[label_field] = df[label_field].astype(int)
    print(f"[INFO] Data shape after cleaning labels: {df.shape}")
    
    # 4. Split data if training, otherwise use the job_type as the single split
    if job_type == "training":
        train_df, holdout_df = train_test_split(df, train_size=train_ratio, random_state=42, stratify=df[label_field])
        test_df, val_df = train_test_split(holdout_df, test_size=test_val_ratio, random_state=42, stratify=holdout_df[label_field])
        splits = {"train": train_df, "test": test_df, "val": val_df}
    else:
        splits = {job_type: df}

    # 5. Save output files
    for split_name, split_df in splits.items():
        subfolder = output_path / split_name
        subfolder.mkdir(exist_ok=True)
        
        # Only output processed_data.csv
        proc_path = subfolder / f"{split_name}_processed_data.csv"
        split_df.to_csv(proc_path, index=False)
        print(f"[INFO] Saved {proc_path} (shape={split_df.shape})")

    print("[INFO] Preprocessing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_type", type=str, required=True, help="One of ['training','validation','testing']")
    args = parser.parse_args()

    # Read configuration from environment variables
    LABEL_FIELD = os.environ.get("LABEL_FIELD")
    if not LABEL_FIELD:
        raise RuntimeError("LABEL_FIELD environment variable must be set.")
    TRAIN_RATIO = float(os.environ.get("TRAIN_RATIO", 0.7))
    TEST_VAL_RATIO = float(os.environ.get("TEST_VAL_RATIO", 0.5))
    
    # Define standard SageMaker paths
    INPUT_BASE_DIR = "/opt/ml/processing/input"
    OUTPUT_DIR = "/opt/ml/processing/output"

    # Execute the main processing logic by calling the refactored main function
    main(
        job_type=args.job_type,
        label_field=LABEL_FIELD,
        train_ratio=TRAIN_RATIO,
        test_val_ratio=TEST_VAL_RATIO,
        input_base_dir=INPUT_BASE_DIR,
        output_dir=OUTPUT_DIR,
    )
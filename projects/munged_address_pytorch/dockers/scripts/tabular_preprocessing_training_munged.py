"""
Custom TabularPreprocessing script for Munged Address — Training Phase.

Reads LLM-scored data, applies label flip (bad→1, good+score>3→1, else→0),
extracts shippingAddress from saddr (first ||| segment), selects columns,
and performs stratified train/val/test split.

Source: FZ 29d16b (adapted from 21_data_prepare.ipynb)
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main(input_paths, output_paths, environ_vars, job_args, logger=None):
    log = logger or globals()["logger"].info

    threshold = int(environ_vars.get("LABEL_FLIP_THRESHOLD", "3"))
    cohort_col = environ_vars.get("COHORT_COLUMN", "__cohort__")
    score_col = environ_vars.get("STRANGENESS_COLUMN", "llm_strangeness_rating")
    address_col = environ_vars.get("ADDRESS_COLUMN", "saddr")
    address_delim = environ_vars.get("ADDRESS_DELIMITER", "|||")
    train_ratio = float(environ_vars.get("TRAIN_RATIO", "0.8"))
    test_val_ratio = float(environ_vars.get("TEST_VAL_RATIO", "0.5"))
    output_format = environ_vars.get("OUTPUT_FORMAT", "csv").lower()

    input_dir = input_paths["DATA"]
    output_path = Path(output_paths["processed_data"])
    output_path.mkdir(parents=True, exist_ok=True)

    all_files = list(Path(input_dir).rglob("*.parquet")) + list(
        Path(input_dir).rglob("*.csv")
    )
    dfs = []
    for f in all_files:
        if f.stat().st_size > 0:
            if f.suffix == ".parquet":
                dfs.append(pd.read_parquet(f))
            else:
                dfs.append(pd.read_csv(f))
    df = pd.concat(dfs, ignore_index=True)
    log(f"[INFO] Loaded {len(df)} rows from {len(dfs)} files")

    df["__tag__"] = 0
    df.loc[df[cohort_col] == "bad", "__tag__"] = 1
    df.loc[(df[cohort_col] == "good") & (df[score_col] > threshold), "__tag__"] = 1

    pos_count = (df["__tag__"] == 1).sum()
    neg_count = (df["__tag__"] == 0).sum()
    log(
        f"[INFO] Labels: positive={pos_count}, negative={neg_count}, "
        f"ratio=1:{neg_count / max(pos_count, 1):.1f}"
    )

    df["shippingAddress"] = (
        df[address_col].astype(str).str.split(address_delim).str[0].str.strip()
    )

    keep_cols = ["shippingAddress", "__tag__", "orderDate", "marketplaceId"]
    df = df[[c for c in keep_cols if c in df.columns]]

    train_df, holdout_df = train_test_split(
        df, train_size=train_ratio, stratify=df["__tag__"], random_state=42
    )
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=test_val_ratio,
        stratify=holdout_df["__tag__"],
        random_state=42,
    )

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_dir = output_path / name
        split_dir.mkdir(parents=True, exist_ok=True)
        out_file = split_dir / f"{name}_processed_data.{output_format}"
        if output_format == "parquet":
            split_df.to_parquet(out_file, index=False)
        else:
            split_df.to_csv(out_file, index=False)
        log(f"[INFO] {out_file} ({len(split_df)} rows)")


if __name__ == "__main__":
    input_paths = {"DATA": "/opt/ml/processing/input/data"}
    output_paths = {"processed_data": "/opt/ml/processing/output"}

    environ_vars = {
        "LABEL_FLIP_THRESHOLD": os.environ.get("LABEL_FLIP_THRESHOLD", "3"),
        "COHORT_COLUMN": os.environ.get("COHORT_COLUMN", "__cohort__"),
        "STRANGENESS_COLUMN": os.environ.get(
            "STRANGENESS_COLUMN", "llm_strangeness_rating"
        ),
        "ADDRESS_COLUMN": os.environ.get("ADDRESS_COLUMN", "saddr"),
        "ADDRESS_DELIMITER": os.environ.get("ADDRESS_DELIMITER", "|||"),
        "TRAIN_RATIO": os.environ.get("TRAIN_RATIO", "0.8"),
        "TEST_VAL_RATIO": os.environ.get("TEST_VAL_RATIO", "0.5"),
        "OUTPUT_FORMAT": os.environ.get("OUTPUT_FORMAT", "csv"),
    }

    args = argparse.Namespace()

    try:
        main(input_paths, output_paths, environ_vars, args)
        sys.exit(0)
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)

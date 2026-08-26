#!/usr/bin/env python
"""
TabularLookupModelBuilding Processing Script.

Reads a tabular dataset (parquet) from an upstream data-loading step and turns
the dataset itself into a non-parametric model, then packages it into
``model.tar.gz`` for downstream MIMS ``Package`` + ``Payload``:

- ``MODEL_KIND=lookup``          -> group VALUE_COLUMNS by KEY_COLUMNS into a
                                    ``{key: [values]}`` map, sharded into JSON.
- ``MODEL_KIND=set_membership``  -> de-duplicated set of KEY_COLUMNS, sharded
                                    into JSON.

There is no training and there are no learned parameters ("the dataset IS the
model"). This script does NOT write an inference handler — the per-project
handler is bundled by the ``Package`` step (``inference_scripts_input``).

Contract (matches tabular_lookup_model_building.step.yaml):
  Input:  /opt/ml/processing/input/data      (parquet shards)
  Output: /opt/ml/processing/output/model/model.tar.gz

model.tar.gz structure:
    code/
        config.json                # the model manifest (kind, keys, values, shards)
        lookup/<shard>.json        # {"records": [{"key": [...], "values": [...]}]}  (lookup)
        keyset/<shard>.json        # {"keys": [[...], ...]}                            (set_membership)
    hyperparameters.json           # required by downstream MIMS packaging

Dependencies (pandas / numpy / pyarrow) are declared in the interface's
``framework_requirements`` and provided by the SKLearn framework container — this
script does no runtime pip bootstrap.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import tarfile
import tempfile
import traceback
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# HELPERS
# ============================================================================


def _shard_for(key: str, shard_count: int) -> str:
    """Deterministic shard name for a stringified key."""
    if shard_count <= 1:
        return "shard_000"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % shard_count
    return f"shard_{bucket:03d}"


def load_data(input_dir: str) -> pd.DataFrame:
    """Load parquet shards from the upstream (Cradle) output directory."""
    input_path = Path(input_dir)
    parquet_files = list(input_path.rglob("*.parquet"))
    if not parquet_files:
        logger.info("No .parquet files found via rglob; trying pd.read_parquet on the directory")
        df = pd.read_parquet(input_dir)
    else:
        logger.info(f"Found {len(parquet_files)} parquet files in {input_dir}")
        df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    logger.info(f"Loaded {len(df)} rows with columns: {list(df.columns)}")
    return df


def _require_columns(df: pd.DataFrame, columns: List[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s) {missing}. Got: {list(df.columns)}")


def _key_str(row_key) -> str:
    """Stringify a (possibly multi-column) key deterministically for sharding."""
    if isinstance(row_key, tuple):
        return "\x1f".join("" if v is None else str(v) for v in row_key)
    return "" if row_key is None else str(row_key)


def build_lookup(df: pd.DataFrame, key_columns: List[str], value_columns: List[str], dedup: bool) -> List[dict]:
    """Group value_columns by key_columns into a list of {key, values} records."""
    _require_columns(df, key_columns + value_columns)
    df = df.dropna(subset=key_columns)
    records: List[dict] = []
    for key, group in df.groupby(key_columns if len(key_columns) > 1 else key_columns[0]):
        key_list = list(key) if isinstance(key, tuple) else [key]
        values = []
        for _, r in group[value_columns].iterrows():
            values.append([None if pd.isna(v) else v for v in r.tolist()])
        if dedup:
            seen = set()
            deduped = []
            for v in values:
                marker = json.dumps(v, default=str)
                if marker not in seen:
                    seen.add(marker)
                    deduped.append(v)
            values = deduped
        records.append({"key": [None if pd.isna(k) else k for k in key_list], "values": values})
    logger.info(f"Built lookup: {len(records)} keys, {sum(len(r['values']) for r in records)} total values")
    return records


def build_keyset(df: pd.DataFrame, key_columns: List[str], dedup: bool) -> List[list]:
    """De-duplicated set of key_columns tuples."""
    _require_columns(df, key_columns)
    df = df.dropna(subset=key_columns)
    keys: List[list] = []
    seen = set()
    for _, r in df[key_columns].iterrows():
        key_list = [None if pd.isna(v) else v for v in r.tolist()]
        marker = json.dumps(key_list, default=str)
        if dedup and marker in seen:
            continue
        seen.add(marker)
        keys.append(key_list)
    logger.info(f"Built key set: {len(keys)} unique keys")
    return keys


def build_model_directory(
    model_dir: Path,
    model_kind: str,
    key_columns: List[str],
    value_columns: List[str],
    records,
    shard_count: int,
) -> None:
    """Write the model manifest + sharded JSON into the model directory."""
    code_dir = model_dir / "code"
    code_dir.mkdir(parents=True, exist_ok=True)

    # Config manifest the inference handler reads at load time.
    config = {
        "model_kind": model_kind,
        "key_columns": key_columns,
        "value_columns": value_columns,
        "shard_count": shard_count,
    }
    with open(code_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    if model_kind == "lookup":
        shard_dir = code_dir / "lookup"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shards: Dict[str, list] = {}
        for rec in records:
            shards.setdefault(_shard_for(_key_str(tuple(rec["key"])), shard_count), []).append(rec)
        for shard_name, recs in shards.items():
            with open(shard_dir / f"{shard_name}.json", "w") as f:
                json.dump({"records": recs}, f)
        total = sum(len(r["values"]) for r in records)
        n_keys = len(records)
    else:  # set_membership
        shard_dir = code_dir / "keyset"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shards = {}
        for key_list in records:
            shards.setdefault(_shard_for(_key_str(tuple(key_list)), shard_count), []).append(key_list)
        for shard_name, keys in shards.items():
            with open(shard_dir / f"{shard_name}.json", "w") as f:
                json.dump({"keys": keys}, f)
        total = len(records)
        n_keys = len(records)

    logger.info(f"Wrote {len(shards)} shard file(s) under {shard_dir.name}/")

    hyperparams = {
        "model_class": "non_parametric",
        "model_type": f"tabular_{model_kind}",
        "key_columns": key_columns,
        "value_columns": value_columns,
        "shard_count": shard_count,
        "total_keys": n_keys,
        "total_values": total,
        "full_field_list": key_columns + value_columns,
    }
    with open(model_dir / "hyperparameters.json", "w") as f:
        json.dump(hyperparams, f, indent=2)
    logger.info("Saved hyperparameters.json")


def create_model_tarball(model_dir: Path, output_path: Path) -> None:
    """Package the model directory into model.tar.gz (script tars its own artifact)."""
    logger.info(f"Creating model.tar.gz at: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_path, "w:gz") as tar:
        for item in model_dir.rglob("*"):
            if item.is_file():
                tar.add(item, arcname=item.relative_to(model_dir))
    logger.info(f"model.tar.gz created: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


# ============================================================================
# MAIN
# ============================================================================


def _parse_list(env_value: str) -> List[str]:
    """Parse a JSON list env var; tolerate a bare comma-separated string."""
    env_value = (env_value or "").strip()
    if not env_value:
        return []
    try:
        parsed = json.loads(env_value)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
        return [str(parsed)]
    except json.JSONDecodeError:
        return [tok.strip() for tok in env_value.split(",") if tok.strip()]


def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
) -> None:
    """
    Entry point.

    Args:
        input_paths:  {"input_data": "/opt/ml/processing/input/data"}
        output_paths: {"model_output": "/opt/ml/processing/output/model"}
        environ_vars: MODEL_KIND / KEY_COLUMNS / VALUE_COLUMNS / DEDUP / SHARD_COUNT
        job_args:     parsed CLI args (unused; kept for the standard signature)
    """
    logger.info("=" * 70)
    logger.info("TABULAR LOOKUP MODEL BUILDING")
    logger.info("=" * 70)

    input_dir = input_paths["input_data"]
    output_dir = Path(output_paths["model_output"])

    model_kind = (environ_vars.get("MODEL_KIND") or "lookup").lower()
    key_columns = _parse_list(environ_vars.get("KEY_COLUMNS", "[]"))
    value_columns = _parse_list(environ_vars.get("VALUE_COLUMNS", "[]"))
    dedup = (environ_vars.get("DEDUP", "true") or "true").lower() == "true"
    shard_count = int(environ_vars.get("SHARD_COUNT", "1") or "1")

    if model_kind not in ("lookup", "set_membership"):
        raise ValueError(f"MODEL_KIND must be 'lookup' or 'set_membership', got '{model_kind}'")
    if not key_columns:
        raise ValueError("KEY_COLUMNS must be a non-empty list")

    logger.info(
        f"model_kind={model_kind} key_columns={key_columns} "
        f"value_columns={value_columns} dedup={dedup} shard_count={shard_count}"
    )

    df = load_data(input_dir)

    if model_kind == "lookup":
        if not value_columns:
            raise ValueError("model_kind='lookup' requires a non-empty VALUE_COLUMNS")
        records = build_lookup(df, key_columns, value_columns, dedup)
    else:
        records = build_keyset(df, key_columns, dedup)

    with tempfile.TemporaryDirectory() as temp_dir:
        model_dir = Path(temp_dir) / "model"
        model_dir.mkdir()
        build_model_directory(model_dir, model_kind, key_columns, value_columns, records, shard_count)
        output_dir.mkdir(parents=True, exist_ok=True)
        create_model_tarball(model_dir, output_dir / "model.tar.gz")

    logger.info("=" * 70)
    logger.info("MODEL BUILDING COMPLETE")
    logger.info(f"  Output: {output_dir / 'model.tar.gz'}")
    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        CONTAINER_PATHS = {
            "INPUT_DATA": "/opt/ml/processing/input/data",
            "MODEL_OUTPUT": "/opt/ml/processing/output/model",
        }

        parser = argparse.ArgumentParser(description="Tabular lookup/membership model building")
        job_args = parser.parse_args()

        input_paths = {"input_data": CONTAINER_PATHS["INPUT_DATA"]}
        output_paths = {"model_output": CONTAINER_PATHS["MODEL_OUTPUT"]}
        environ_vars = {
            "MODEL_KIND": os.environ.get("MODEL_KIND", "lookup"),
            "KEY_COLUMNS": os.environ.get("KEY_COLUMNS", "[]"),
            "VALUE_COLUMNS": os.environ.get("VALUE_COLUMNS", "[]"),
            "DEDUP": os.environ.get("DEDUP", "true"),
            "SHARD_COUNT": os.environ.get("SHARD_COUNT", "1"),
        }

        if not os.path.exists(CONTAINER_PATHS["INPUT_DATA"]):
            raise FileNotFoundError(f"Input directory not found: {CONTAINER_PATHS['INPUT_DATA']}")

        main(input_paths, output_paths, environ_vars, job_args)
        logger.info("Script completed successfully")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Script failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

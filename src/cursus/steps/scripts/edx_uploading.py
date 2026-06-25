"""
Upload S3 data files to EDX via EdxDataLoader.

This script is executed as a SageMaker ProcessingStep inside the SAIS
Docker container. It reads files from the input directory (mounted from
upstream S3) and uploads each to an EDX manifest.

Input: /opt/ml/processing/input/data (files from upstream step)
Output: None (SINK — data uploaded to EDX manifest)
"""

import argparse
import json
import os
import sys
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from secure_ai_sandbox_python_lib.session import Session as SandboxSession

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
)
logger = logging.getLogger(__name__)

MAX_WORKERS = 4


# Characters that MUST NOT survive into a headerless TSV that is read positionally:
#   \t (delimiter) shifts columns; \n \r \v \f inject phantom rows; " triggers pandas quote-wrapping
#   that a positional tab-splitter misreads; \x00 and other C0 control chars abort some parsers.
_TSV_BREAK_RE = r"[\t\r\n\v\f]+"
_TSV_STRIP_RE = r"[\"\x00-\x08\x0b\x0c\x0e-\x1f]"


def _parse_output_columns(raw) -> Optional[List[str]]:
    """Parse the optional EDX_OUTPUT_COLUMNS contract (JSON array string, list, or empty)."""
    if not raw:
        return None
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, list) or not raw:
        return None
    return [str(c) for c in raw]


def _assert_tsv_safe(df, log):
    """Fail-loud guardrail: after sanitization, no cell may still contain a tab/newline/CR/quote.
    A single bad cell would shift columns or inject rows in the headerless positional read, so we
    refuse to upload rather than silently corrupt the EDX manifest."""
    import re as _re

    bad = _re.compile(r'[\t\r\n"]')
    for col in df.columns:
        mask = df[col].astype(str).str.contains(bad, regex=True, na=False)
        if mask.any():
            n = int(mask.sum())
            sample = df.loc[mask, col].astype(str).iloc[0][:80]
            raise ValueError(
                f"TSV guardrail: column '{col}' still has {n} cell(s) with a delimiter/break/quote "
                f"after sanitization (e.g. {sample!r}). Refusing to upload a corruptible file."
            )


def _project_file_to_tsv(file_path: Path, output_columns: List[str], log) -> Path:
    """Read a parquet/CSV shard, project to an ORDERED canonical column set, and write a headerless
    TSV next to it.

    EDX is read positionally with a headerless schema, so the uploaded width must be deterministic:
      - missing canonical columns are added as empty (so every row has exactly len(output_columns))
      - non-canonical columns are dropped (parse-failure rows otherwise leak extra columns and
        widen the union)
      - NaN is filled to "" (so absent values do not serialize as the literal "nan")
      - delimiter/break/quote chars are stripped, then a guardrail asserts none remain (so a cell
        cannot shift fields or inject rows), and the write uses QUOTE_NONE so pandas never re-adds
        quotes.
    Returns the path to the projected headerless TSV.
    """
    import csv as _csv

    import pandas as pd

    if file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)

    missing = [c for c in output_columns if c not in df.columns]
    if missing:
        log(f"[WARN] {file_path.name}: missing columns {missing}; filling empty")
        for c in missing:
            df[c] = ""
    dropped = [c for c in df.columns if c not in output_columns]
    if dropped:
        log(f"[INFO] {file_path.name}: dropping non-canonical columns {dropped}")

    df = df[output_columns].copy().fillna("")
    for c in df.columns:
        df[c] = (
            df[c]
            .astype(str)
            .str.replace(_TSV_BREAK_RE, " ", regex=True)
            .str.replace(_TSV_STRIP_RE, "", regex=True)
        )
    _assert_tsv_safe(df, log)

    tsv_path = file_path.with_suffix(".projected.tsv")
    df.to_csv(
        tsv_path,
        sep="\t",
        header=False,
        index=False,
        quoting=_csv.QUOTE_NONE,
        escapechar="\\",
    )
    log(
        f"[INFO] {file_path.name}: projected to {len(output_columns)} cols -> {tsv_path.name}"
    )
    return tsv_path


def upload_file(edx_loader, manifest_arn, file_path, file_index, total_files):
    """Upload a single file to EDX."""
    logger.info(f"Uploading {file_path.name} ({file_index}/{total_files})...")
    edx_loader.upload_data_to_edx(manifest_arn, str(file_path))
    logger.info(f"Uploaded {file_path.name} successfully.")
    return file_path.name


def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
    logger_fn=None,
):
    """
    Main entry point for EDX upload — standard Cursus script signature.

    Args:
        input_paths: {"input_data": "/opt/ml/processing/input/data"}
        output_paths: {} (SINK — no S3 output)
        environ_vars: {"EDX_DATASET_ARN": ..., "EDX_MANIFEST_KEY": ..., etc.}
        job_args: argparse.Namespace (unused)
        logger_fn: optional logger
    """
    log = logger_fn or logger.info

    dataset_arn = environ_vars.get("EDX_DATASET_ARN", "")
    manifest_key = environ_vars.get("EDX_MANIFEST_KEY", "")
    if not dataset_arn or not manifest_key:
        raise ValueError("EDX_DATASET_ARN and EDX_MANIFEST_KEY env vars required")

    # Resolve template manifest key if placeholders present
    if "{" in manifest_key:
        parts_str = environ_vars.get("EDX_MANIFEST_KEY_PARTS", "{}")
        key_parts = json.loads(parts_str) if isinstance(parts_str, str) else parts_str
        key_parts.setdefault(
            "execution_id",
            environ_vars.get("MODS_WORKFLOW_EXECUTION_ID", str(uuid.uuid4())[:8]),
        )
        manifest_key = manifest_key.format(**key_parts)

    log(f"[INFO] Initializing SAIS session...")
    sandbox_session = SandboxSession(session_folder="/tmp/")
    edx_loader = sandbox_session.resource("EdxDataLoader")

    input_dir = Path(input_paths["input_data"])
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files = sorted(
        f for f in input_dir.iterdir() if f.is_file() and f.stat().st_size > 0
    )

    if not files:
        log(f"[WARN] No non-empty files found in {input_dir}")
        return

    log(f"[INFO] Found {len(files)} file(s) to upload")

    # Optional ordered column projection. When EDX_OUTPUT_COLUMNS is provided, each parquet/CSV
    # shard is reprojected to that exact column set and rewritten as a headerless TSV before upload,
    # guaranteeing a deterministic positional layout for the downstream Cradle read. When unset, the
    # original raw-passthrough behavior is preserved (files uploaded byte-for-byte as-is).
    output_columns = _parse_output_columns(environ_vars.get("EDX_OUTPUT_COLUMNS"))
    if output_columns:
        log(f"[INFO] Projecting all shards to canonical columns: {output_columns}")
        projected = []
        for f in files:
            try:
                projected.append(_project_file_to_tsv(f, output_columns, log))
            except Exception as e:
                log(f"[ERROR] Failed to project {f.name}: {e}")
                raise
        files = projected

    # Construct manifest ARN
    key_components = [k.strip() for k in manifest_key.split(",")]
    if len(key_components) == 1:
        manifest_arn = f'{dataset_arn}/["{key_components[0]}"]'
    else:
        quoted = ",".join(f'"{k}"' for k in key_components)
        manifest_arn = f"{dataset_arn}/[{quoted}]"
    log(f"[INFO] Target manifest: {manifest_arn}")

    uploaded = []
    failed = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                upload_file, edx_loader, manifest_arn, f, i + 1, len(files)
            ): f
            for i, f in enumerate(files)
        }
        for future in as_completed(futures):
            file_path = futures[future]
            try:
                name = future.result()
                uploaded.append(name)
            except Exception as e:
                log(f"[ERROR] Failed to upload {file_path.name}: {e}")
                failed.append(file_path.name)

    log(
        f"[INFO] Upload complete: {len(uploaded)} succeeded, {len(failed)} failed "
        f"out of {len(files)} total files."
    )

    if failed:
        raise RuntimeError(f"Failed to upload {len(failed)} files: {failed}")


if __name__ == "__main__":
    input_paths = {
        "input_data": "/opt/ml/processing/input/data",
    }
    output_paths = {}

    environ_vars = {
        "EDX_DATASET_ARN": os.environ.get("EDX_DATASET_ARN", ""),
        "EDX_MANIFEST_KEY": os.environ.get("EDX_MANIFEST_KEY", ""),
        "EDX_MANIFEST_KEY_PARTS": os.environ.get("EDX_MANIFEST_KEY_PARTS", "{}"),
        "EDX_OUTPUT_COLUMNS": os.environ.get("EDX_OUTPUT_COLUMNS", ""),
        "MODS_WORKFLOW_EXECUTION_ID": os.environ.get("MODS_WORKFLOW_EXECUTION_ID", ""),
    }

    args = argparse.Namespace()

    try:
        main(input_paths, output_paths, environ_vars, args)
        logger.info("EDX upload script completed successfully")
        sys.exit(0)
    except Exception as e:
        logger.error(f"EDX upload failed: {e}")
        sys.exit(1)

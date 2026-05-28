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
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from secure_ai_sandbox_python_lib.session import Session as SandboxSession

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
)
logger = logging.getLogger(__name__)

MAX_WORKERS = 4


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

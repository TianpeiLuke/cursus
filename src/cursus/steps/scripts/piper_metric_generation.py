#!/usr/bin/env python
import os
import json
import sys

from subprocess import check_call
import boto3
import logging

# ============================================================================
# PACKAGE INSTALLATION CONFIGURATION
# ============================================================================

# Control which PyPI source to use via environment variable
# Set USE_SECURE_PYPI=true to use secure CodeArtifact PyPI
# Set USE_SECURE_PYPI=false or leave unset to use public PyPI
USE_SECURE_PYPI = os.environ.get("USE_SECURE_PYPI", "false").lower() == "true"

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_secure_pypi_access_token() -> str:
    """
    Get CodeArtifact access token for secure PyPI.

    Returns:
        str: Authorization token for CodeArtifact

    Raises:
        Exception: If token retrieval fails
    """
    try:
        os.environ["AWS_STS_REGIONAL_ENDPOINTS"] = "regional"
        sts = boto3.client("sts", region_name="us-east-1")
        caller_identity = sts.get_caller_identity()
        assumed_role_object = sts.assume_role(
            RoleArn="arn:aws:iam::675292366480:role/SecurePyPIReadRole_"
            + caller_identity["Account"],
            RoleSessionName="SecurePypiReadRole",
        )
        credentials = assumed_role_object["Credentials"]
        code_artifact_client = boto3.client(
            "codeartifact",
            aws_access_key_id=credentials["AccessKeyId"],
            aws_secret_access_key=credentials["SecretAccessKey"],
            aws_session_token=credentials["SessionToken"],
            region_name="us-west-2",
        )
        token = code_artifact_client.get_authorization_token(
            domain="amazon", domainOwner="149122183214"
        )["authorizationToken"]

        logger.info("Successfully retrieved secure PyPI access token")
        return token

    except Exception as e:
        logger.error(f"Failed to retrieve secure PyPI access token: {e}")
        raise


def install_packages_from_public_pypi(packages: list) -> None:
    """
    Install packages from standard public PyPI.

    Args:
        packages: List of package specifications (e.g., ["pandas==1.5.0", "numpy"])
    """
    logger.info(f"Installing {len(packages)} packages from public PyPI")
    logger.info(f"Packages: {packages}")

    try:
        check_call([sys.executable, "-m", "pip", "install", *packages])
        logger.info("✓ Successfully installed packages from public PyPI")
    except Exception as e:
        logger.error(f"✗ Failed to install packages from public PyPI: {e}")
        raise


def install_packages_from_secure_pypi(packages: list) -> None:
    """
    Install packages from secure CodeArtifact PyPI.

    Args:
        packages: List of package specifications (e.g., ["pandas==1.5.0", "numpy"])
    """
    logger.info(f"Installing {len(packages)} packages from secure PyPI")
    logger.info(f"Packages: {packages}")

    try:
        token = _get_secure_pypi_access_token()
        index_url = f"https://aws:{token}@amazon-149122183214.d.codeartifact.us-west-2.amazonaws.com/pypi/secure-pypi/simple/"

        check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--index-url",
                index_url,
                *packages,
            ]
        )

        logger.info("✓ Successfully installed packages from secure PyPI")
    except Exception as e:
        logger.error(f"✗ Failed to install packages from secure PyPI: {e}")
        raise


def install_packages(packages: list, use_secure: bool = USE_SECURE_PYPI) -> None:
    """
    Install packages from PyPI source based on configuration.

    This is the main installation function that delegates to either public or
    secure PyPI based on the USE_SECURE_PYPI environment variable.

    Args:
        packages: List of package specifications (e.g., ["pandas==1.5.0", "numpy"])
        use_secure: If True, use secure CodeArtifact PyPI; if False, use public PyPI.
                   Defaults to USE_SECURE_PYPI environment variable.

    Environment Variables:
        USE_SECURE_PYPI: Set to "true" to use secure PyPI, "false" for public PyPI

    Example:
        # Install from public PyPI (default)
        install_packages(["pandas==1.5.0", "numpy"])

        # Install from secure PyPI
        os.environ["USE_SECURE_PYPI"] = "true"
        install_packages(["pandas==1.5.0", "numpy"])
    """
    if not packages:
        logger.info("No additional packages to install; skipping installation")
        return

    logger.info("=" * 70)
    logger.info("PACKAGE INSTALLATION")
    logger.info("=" * 70)
    logger.info(f"PyPI Source: {'SECURE (CodeArtifact)' if use_secure else 'PUBLIC'}")
    logger.info(
        f"Environment Variable USE_SECURE_PYPI: {os.environ.get('USE_SECURE_PYPI', 'not set')}"
    )
    logger.info(f"Number of packages: {len(packages)}")
    logger.info("=" * 70)

    try:
        if use_secure:
            install_packages_from_secure_pypi(packages)
        else:
            install_packages_from_public_pypi(packages)

        logger.info("=" * 70)
        logger.info("✓ PACKAGE INSTALLATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)

    except Exception as e:
        logger.error("=" * 70)
        logger.error("✗ PACKAGE INSTALLATION FAILED")
        logger.error("=" * 70)
        raise


# ============================================================================
# INSTALL REQUIRED PACKAGES
# ============================================================================

# PIPER metric generation relies only on pandas / numpy / scikit-learn, which are
# provided by the SKLearnProcessor container image (framework_version 1.2-1). No
# extra packages are required (unlike ModelMetricsComputation which installs
# matplotlib for .jpg plots — PIPER renders from CSVs so no plotting is needed).
required_packages: list = []

# Install packages using unified installation function
install_packages(required_packages)

print("***********************Package Installation Complete*********************")


import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
)
import csv
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONTAINER PATHS
# ============================================================================
# NOTE: the OUTPUT_DIR is a single FLAT root (NOT /opt/ml/processing/output/metrics
# or /plots). PIPER scans the output root for .metric + .csv files, so every
# artifact this script emits is written directly under OUTPUT_DIR. This is the one
# deliberate divergence from ModelMetricsComputation.
CONTAINER_PATHS = {
    "EVAL_DATA_DIR": "/opt/ml/processing/input/eval_data",
    "OUTPUT_DIR": "/opt/ml/processing/output",
}


# ============================================================================
# PREDICTION DATA LOADING (reused verbatim from model_metrics_computation.py)
# ============================================================================
def _detect_file_format(file_path: str) -> str:
    """
    Detect file format based on extension.

    Args:
        file_path: Path to file

    Returns:
        Format string: 'csv', 'tsv', 'parquet', or 'json'
    """
    file_path_lower = file_path.lower()

    if file_path_lower.endswith((".parquet", ".pq")):
        return "parquet"
    elif file_path_lower.endswith(".tsv"):
        return "tsv"
    elif file_path_lower.endswith(".json"):
        return "json"
    elif file_path_lower.endswith(".csv"):
        return "csv"
    else:
        # Default to CSV for unknown extensions
        return "csv"


# Prediction output base names produced by the upstream inference/eval steps, in
# resolution priority. Each cursus producer writes a SINGLE (non-sharded) file named
# <base>.<ext> to /opt/ml/processing/output/eval (mounted here as EVAL_DATA_DIR):
#   - eval_predictions_with_comparison : XGBoostModelEval comparison mode (most specific)
#   - eval_predictions                 : XGBoostModelEval standard mode (carries the label)
#   - inference_predictions            : PyTorchModelInference (NO label column)
#   - predictions                      : XGBoostModelInference / LightGBMMTModelInference
# Comparison output is checked before the plain eval file because both are never emitted
# together and the comparison base is the more specific artifact.
_PREDICTION_BASENAMES = (
    "eval_predictions_with_comparison",
    "eval_predictions",
    "inference_predictions",
    "predictions",
)


def _read_by_format(file_path: str) -> pd.DataFrame:
    """Read a single prediction file by its extension-detected format."""
    detected_format = _detect_file_format(file_path)
    logger.info(f"Loading predictions from {file_path} (format: {detected_format})")
    if detected_format == "parquet":
        return pd.read_parquet(file_path)
    elif detected_format == "tsv":
        return pd.read_csv(file_path, sep="\t")
    elif detected_format == "json":
        return pd.read_json(file_path)
    else:  # csv or default
        return pd.read_csv(file_path)


def detect_and_load_predictions(
    input_dir: str, preferred_format: str = None
) -> pd.DataFrame:
    """
    Load the upstream prediction file, robust to the naming/format divergence across
    cursus inference & eval steps.

    Filenames differ by producer (predictions.* / inference_predictions.* /
    eval_predictions[_with_comparison].*), all single-file and non-sharded, in one of
    csv/tsv/parquet/json. This globs the known base names (priority order) across the
    format list (preferred_format first) and loads the first match.
    """
    # Format resolution order: preferred first, then the rest.
    exts = []
    if preferred_format and preferred_format != "auto":
        exts.append(preferred_format)
    for fmt in ["parquet", "csv", "tsv", "json"]:
        if fmt not in exts:
            exts.append(fmt)

    for base in _PREDICTION_BASENAMES:
        for fmt in exts:
            file_path = os.path.join(input_dir, f"{base}.{fmt}")
            if os.path.exists(file_path):
                return _read_by_format(file_path)

    raise FileNotFoundError(
        "No predictions file found in "
        f"{input_dir}. Looked for {list(_PREDICTION_BASENAMES)} with extensions "
        f"{exts}. Upstream must be a *ModelInference or *ModelEval step writing its "
        "eval output (predictions/inference_predictions/eval_predictions) here."
    )


# ============================================================================
# SERIES / SCORE RESOLUTION
# ============================================================================
def resolve_score_column(
    df: pd.DataFrame,
    score_field: str,
    id_field: Optional[str] = None,
    label_field: Optional[str] = None,
) -> str:
    """
    Resolve which column holds the positive-class model score, robust to the
    naming divergence across cursus inference/eval producers.

    Resolution order (first match wins):
      1. explicit ``score_field`` (config SCORE_FIELD) if present;
      2. ``prob_class_1`` — the positive-class prob from XGBoostModelInference /
         PyTorchModelInference / XGBoostModelEval (standard mode);
      3. ``new_model_prob_class_1`` — XGBoostModelEval COMPARISON mode (where
         ``prob_class_1`` is renamed away);
      4. the sole ``*_prob`` column — LightGBMMTModelInference single-task output
         (``<task_name>_prob``), excluding id/label columns;
      5. the sole non-id / non-label numeric column in [0, 1].

    For MULTI-task LightGBMMT (several ``*_prob`` columns) the intended positive
    class is ambiguous — ``score_field`` MUST be set explicitly; this raises.
    """
    if score_field and score_field in df.columns:
        return score_field

    for candidate in ("prob_class_1", "new_model_prob_class_1"):
        if candidate in df.columns:
            logger.warning(
                f"Score field '{score_field}' not found; falling back to '{candidate}'"
            )
            return candidate

    reserved = {c for c in (id_field, label_field) if c}

    # LightGBMMT single-task: exactly one '<task>_prob' column (excluding reserved)
    prob_cols = [
        c for c in df.columns if str(c).endswith("_prob") and c not in reserved
    ]
    if len(prob_cols) == 1:
        logger.warning(
            f"Score field '{score_field}' not found; using sole '*_prob' column "
            f"'{prob_cols[0]}' (LightGBMMT single-task convention)"
        )
        return prob_cols[0]
    if len(prob_cols) > 1:
        raise ValueError(
            f"Multiple '*_prob' score columns {prob_cols} present (multi-task "
            f"output) — set SCORE_FIELD explicitly to pick the positive-class column."
        )

    # Last resort: the sole non-id/non-label numeric column bounded in [0, 1]
    numeric_cols = [
        c
        for c in df.columns
        if c not in reserved and pd.api.types.is_numeric_dtype(df[c])
    ]
    in_unit_range = [
        c
        for c in numeric_cols
        if df[c].dropna().between(0.0, 1.0).all() and not df[c].dropna().empty
    ]
    if len(in_unit_range) == 1:
        logger.warning(
            f"Score field '{score_field}' not found; using sole probability-like "
            f"numeric column '{in_unit_range[0]}'"
        )
        return in_unit_range[0]

    raise ValueError(
        f"Score field '{score_field}' not found and no score column could be resolved "
        f"(tried prob_class_1, new_model_prob_class_1, a sole '*_prob', and a sole "
        f"[0,1] numeric). Available columns: {df.columns.tolist()}. "
        f"Set SCORE_FIELD explicitly."
    )


# ============================================================================
# METADATA (record-count / fraud-count / fraud-rate / date-range / dataset-type)
# ============================================================================
def compute_metadata(
    df: pd.DataFrame,
    y_true: np.ndarray,
    dataset_type: str,
    amount_field: Optional[str],
) -> Dict[str, Any]:
    """
    Build the PIPER metadata block shared by every Graph-Line / Tabular file.

    Keys use the hyphenated PIPER contract names (record-count, fraud-count,
    fraud-rate, date-range-start, date-range-end, dataset-type). These are JSON
    string keys, so they are emitted as literal hyphenated strings.
    """
    record_count = int(len(df))
    fraud_count = int(np.nansum(y_true)) if record_count else 0
    fraud_rate = (fraud_count / record_count) if record_count else 0.0

    # date-range is best-effort: look for common timestamp columns. Per the PIPER
    # contract example, dates are formatted YYYY-MM-DD (calendar date, not full ISO).
    date_range_start = None
    date_range_end = None
    for candidate in ("date", "event_date", "timestamp", "order_date", "event_time"):
        if candidate in df.columns:
            try:
                dates = pd.to_datetime(df[candidate], errors="coerce").dropna()
                if not dates.empty:
                    date_range_start = dates.min().strftime("%Y-%m-%d")
                    date_range_end = dates.max().strftime("%Y-%m-%d")
                    break
            except Exception:
                continue

    # PIPER contract: fraud-rate is a STRING (e.g. "0.078"), record-count/fraud-count
    # are ints, dataset-type/date-range are strings. Match the example verbatim.
    return {
        "record-count": record_count,
        "fraud-count": fraud_count,
        "fraud-rate": f"{fraud_rate:.3f}",
        "date-range-start": date_range_start,
        "date-range-end": date_range_end,
        "dataset-type": dataset_type,
    }


# ============================================================================
# CSV WRITERS (2-column paired data files for PIPER Graph-Line)
# ============================================================================
def write_curve_csv(
    output_dir: str,
    filename: str,
    header: Tuple[str, str],
    x: np.ndarray,
    y: np.ndarray,
) -> str:
    """
    Write a 2-column data CSV (with header) FLAT to output_dir.

    Args:
        header: (x_header, y_header) e.g. ('FPR', 'TPR') or ('Recall', 'Precision')
        x, y: paired arrays of equal length
    """
    out_path = os.path.join(output_dir, filename)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([header[0], header[1]])
        for xi, yi in zip(x, y):
            writer.writerow([float(xi), float(yi)])
    logger.info(f"Wrote curve CSV: {out_path} ({len(x)} points)")
    return out_path


def write_metric_json(output_dir: str, filename: str, payload: Dict[str, Any]) -> str:
    """Write a .metric JSON file FLAT to output_dir."""
    out_path = os.path.join(output_dir, filename)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Wrote metric file: {out_path}")
    return out_path


# ============================================================================
# GRAPH-LINE (.metric) BUILDERS — ROC and PR
# ============================================================================
def emit_roc(
    output_dir: str,
    y_true: np.ndarray,
    variant_score: np.ndarray,
    control_score: Optional[np.ndarray],
    variant_model_id: str,
    control_model_id: Optional[str],
    metadata: Dict[str, Any],
) -> str:
    """
    Compute ROC curve(s), write per-series CSVs, and emit roc_curve.metric
    (Graph-Line). Control series is only added when control_score is present.
    """
    series = []

    # Variant series
    fpr, tpr, _ = roc_curve(y_true, variant_score)
    write_curve_csv(output_dir, "variant_roc.csv", ("FPR", "TPR"), fpr, tpr)
    series.append(
        {
            "label": "Variant",
            "modelId": variant_model_id,
            "data-file": "variant_roc.csv",
            "summary": {
                "auc-roc-value": round(float(roc_auc_score(y_true, variant_score)), 6)
            },
        }
    )

    # Control series (only when a control model / previous score is configured)
    if control_score is not None:
        fpr_c, tpr_c, _ = roc_curve(y_true, control_score)
        write_curve_csv(output_dir, "control_roc.csv", ("FPR", "TPR"), fpr_c, tpr_c)
        series.append(
            {
                "label": "Control",
                "modelId": control_model_id,
                "data-file": "control_roc.csv",
                "summary": {
                    "auc-roc-value": round(
                        float(roc_auc_score(y_true, control_score)), 6
                    )
                },
            }
        )

    payload = {
        "display-name": "AUC ROC - Count",
        "visualization-type": "Graph-Line",
        "series": series,
        "metadata": metadata,
    }
    return write_metric_json(output_dir, "roc_curve.metric", payload)


def emit_pr(
    output_dir: str,
    y_true: np.ndarray,
    variant_score: np.ndarray,
    control_score: Optional[np.ndarray],
    variant_model_id: str,
    control_model_id: Optional[str],
    metadata: Dict[str, Any],
) -> str:
    """
    Compute PR curve(s), write per-series CSVs, and emit pr_curve.metric
    (Graph-Line). Control series is only added when control_score is present.

    precision_recall_curve returns (precision, recall, thresholds). We plot
    Recall (x) vs Precision (y) so the CSV header is 'Recall,Precision'.
    """
    series = []

    # Variant series
    precision, recall, _ = precision_recall_curve(y_true, variant_score)
    write_curve_csv(
        output_dir, "variant_pr.csv", ("Recall", "Precision"), recall, precision
    )
    series.append(
        {
            "label": "Variant",
            "modelId": variant_model_id,
            "data-file": "variant_pr.csv",
            "summary": {
                "auc-pr-value": round(
                    float(average_precision_score(y_true, variant_score)), 6
                )
            },
        }
    )

    # Control series
    if control_score is not None:
        precision_c, recall_c, _ = precision_recall_curve(y_true, control_score)
        write_curve_csv(
            output_dir, "control_pr.csv", ("Recall", "Precision"), recall_c, precision_c
        )
        series.append(
            {
                "label": "Control",
                "modelId": control_model_id,
                "data-file": "control_pr.csv",
                "summary": {
                    "auc-pr-value": round(
                        float(average_precision_score(y_true, control_score)), 6
                    )
                },
            }
        )

    payload = {
        "display-name": "AUC PR - Count",
        "visualization-type": "Graph-Line",
        "series": series,
        "metadata": metadata,
    }
    return write_metric_json(output_dir, "pr_curve.metric", payload)


# ============================================================================
# TABULAR (.metric) BUILDER — data statistics
# ============================================================================
def emit_data_statistics(output_dir: str, metadata: Dict[str, Any]) -> str:
    """
    Emit data_preprocessing_statistic.metric (Tabular). Reuses the same metadata
    block computed for the Graph-Line files.
    """
    headers = [
        "Record Count",
        "Fraud Count",
        "Fraud Rate",
        "Date Range Start",
        "Date Range End",
        "Dataset Type",
    ]
    values_row = [
        metadata["record-count"],
        metadata["fraud-count"],
        metadata["fraud-rate"],
        metadata["date-range-start"],
        metadata["date-range-end"],
        metadata["dataset-type"],
    ]
    payload = {
        "display-name": "Data Statistics",
        "visualization-type": "Tabular",
        "data": {"headers": headers, "values": [values_row]},
        "metadata": metadata,
    }
    return write_metric_json(output_dir, "data_preprocessing_statistic.metric", payload)


# ============================================================================
# MAIN
# ============================================================================
def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
) -> None:
    """
    Main entry point for PIPER Metric Generation.

    Reads eval predictions, recomputes ROC/PR curves, and emits the PIPER
    contract (.metric JSON + paired 2-column CSVs) FLAT to the output root so
    PIPER's output-root scan finds every artifact.

    Args:
        input_paths (Dict[str, str]): Dictionary of input paths (logical -> path)
        output_paths (Dict[str, str]): Dictionary of output paths (logical -> path)
        environ_vars (Dict[str, str]): Dictionary of environment variables
        job_args (argparse.Namespace): Command line arguments
    """
    # Extract paths using CONTRACT logical names ('eval_output' / 'metric_output')
    eval_data_dir = input_paths.get("eval_output")
    output_dir = output_paths.get("metric_output")

    if not eval_data_dir:
        raise ValueError("input path 'eval_output' is required but was not provided")
    if not output_dir:
        raise ValueError("output path 'metric_output' is required but was not provided")

    os.makedirs(output_dir, exist_ok=True)

    # ---- Environment variables ------------------------------------------------
    id_field = environ_vars.get("ID_FIELD", "id")
    label_field = environ_vars.get("LABEL_FIELD", "label")
    score_field = environ_vars.get("SCORE_FIELD", "").strip()
    amount_field = environ_vars.get("AMOUNT_FIELD", "").strip() or None
    input_format = environ_vars.get("INPUT_FORMAT", "auto")
    previous_score_field = environ_vars.get("PREVIOUS_SCORE_FIELD", "").strip()
    comparison_mode = environ_vars.get("COMPARISON_MODE", "false").lower() == "true"
    generate_plots = environ_vars.get("GENERATE_PLOTS", "true").lower() == "true"

    # PIPER additions
    variant_model_id = environ_vars.get("VARIANT_MODEL_ID", "").strip()
    control_model_id = environ_vars.get("CONTROL_MODEL_ID", "").strip() or None
    pipeline_name = environ_vars.get("PIPELINE_NAME", "").strip() or None
    dataset_type = (
        environ_vars.get("DATASET_TYPE", "Validation").strip() or "Validation"
    )
    metrics_to_render = [
        m.strip()
        for m in environ_vars.get(
            "METRICS_TO_RENDER", "auc_roc,auc_pr,data_statistics"
        ).split(",")
        if m.strip()
    ]

    logger.info("=" * 70)
    logger.info("PIPER METRIC GENERATION")
    logger.info("=" * 70)
    logger.info(f"job_type          : {getattr(job_args, 'job_type', None)}")
    logger.info(f"eval_data_dir     : {eval_data_dir}")
    logger.info(f"output_dir (FLAT) : {output_dir}")
    logger.info(f"id_field          : {id_field}")
    logger.info(f"label_field       : {label_field}")
    logger.info(f"score_field       : {score_field}")
    logger.info(f"previous_score    : {previous_score_field}")
    logger.info(f"variant_model_id  : {variant_model_id}")
    logger.info(f"control_model_id  : {control_model_id}")
    logger.info(f"pipeline_name     : {pipeline_name}")
    logger.info(f"dataset_type      : {dataset_type}")
    logger.info(f"metrics_to_render : {metrics_to_render}")
    logger.info("=" * 70)

    # ---- Load predictions -----------------------------------------------------
    preferred_format = None if input_format in (None, "", "auto") else input_format
    df = detect_and_load_predictions(eval_data_dir, preferred_format)
    logger.info(f"Loaded {len(df)} prediction records; columns: {df.columns.tolist()}")

    # ---- Resolve series data --------------------------------------------------
    # y_true is REQUIRED for ROC/PR. The label is reliably present only in
    # *ModelEval output (e.g. XGBoostModelEval writes {id, label, prob_class_*}).
    # PyTorchModelInference DROPS the label; XGBoost/LightGBMMT inference pass it
    # through only if it was in the inference input. So this step must be wired
    # downstream of an EVAL step (or an inference step whose input carried the label).
    if label_field not in df.columns:
        raise ValueError(
            f"Label field '{label_field}' not found in the prediction output "
            f"(columns: {df.columns.tolist()}). PIPER metrics need ground-truth "
            f"labels: wire this step downstream of a *ModelEval step (which writes "
            f"the label), or set LABEL_FIELD, or ensure the inference input carried "
            f"the label column. Note PyTorchModelInference does not emit labels."
        )
    y_true = df[label_field].to_numpy()

    variant_col = resolve_score_column(df, score_field, id_field, label_field)
    variant_score = df[variant_col].to_numpy()
    logger.info(f"Variant series score column resolved to: '{variant_col}'")

    # Control series only when comparison is configured AND a previous-score column
    # exists. Resolve it as: explicit PREVIOUS_SCORE_FIELD, else the conventions
    # XGBoostModelEval comparison mode emits ('previous_model_score', or the renamed
    # 'new_model_prob_class_1' is the VARIANT so it is NOT the control).
    control_score = None
    have_control = bool(comparison_mode or previous_score_field or control_model_id)
    if have_control:
        control_col = None
        if previous_score_field and previous_score_field in df.columns:
            control_col = previous_score_field
        elif "previous_model_score" in df.columns:
            control_col = "previous_model_score"
        if control_col is not None:
            control_score = df[control_col].to_numpy()
            logger.info(f"Control series score column resolved to: '{control_col}'")
        else:
            logger.warning(
                f"Control configuration present (comparison_mode={comparison_mode}, "
                f"previous_score_field='{previous_score_field}', "
                f"control_model_id='{control_model_id}') but no usable previous score "
                f"column (tried PREVIOUS_SCORE_FIELD and 'previous_model_score') — "
                f"emitting single (variant-only) series."
            )

    # ---- Shared metadata ------------------------------------------------------
    metadata = compute_metadata(df, y_true, dataset_type, amount_field)
    if pipeline_name:
        metadata["pipeline-name"] = pipeline_name
    logger.info(f"Computed metadata: {metadata}")

    written = []

    # ---- ROC (Graph-Line) -----------------------------------------------------
    if "auc_roc" in metrics_to_render:
        written.append(
            emit_roc(
                output_dir,
                y_true,
                variant_score,
                control_score,
                variant_model_id,
                control_model_id,
                metadata,
            )
        )
    else:
        logger.info("Skipping ROC — 'auc_roc' not in METRICS_TO_RENDER")

    # ---- PR (Graph-Line) ------------------------------------------------------
    if "auc_pr" in metrics_to_render:
        written.append(
            emit_pr(
                output_dir,
                y_true,
                variant_score,
                control_score,
                variant_model_id,
                control_model_id,
                metadata,
            )
        )
    else:
        logger.info("Skipping PR — 'auc_pr' not in METRICS_TO_RENDER")

    # ---- Data statistics (Tabular) -------------------------------------------
    if "data_statistics" in metrics_to_render:
        written.append(emit_data_statistics(output_dir, metadata))
    else:
        logger.info(
            "Skipping data statistics — 'data_statistics' not in METRICS_TO_RENDER"
        )

    logger.info("=" * 70)
    logger.info(
        f"PIPER metric generation complete. Wrote {len(written)} .metric files:"
    )
    for p in written:
        logger.info(f"  - {p}")
    logger.info("=" * 70)


def create_health_check_file(output_path: str) -> str:
    """Create a health check file to signal script completion."""
    health_path = output_path
    with open(health_path, "w") as f:
        f.write(f"healthy: {datetime.now().isoformat()}")
    return health_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_type", type=str, required=True)
    args = parser.parse_args()

    # Set up paths using contract-defined logical names only.
    # NOTE: input key is 'eval_output' (contract logical name) — this matches
    # what main() reads, avoiding the latent key mismatch in the template.
    input_paths = {
        "eval_output": CONTAINER_PATHS["EVAL_DATA_DIR"],
    }

    output_paths = {
        "metric_output": CONTAINER_PATHS["OUTPUT_DIR"],
    }

    # Collect environment variables
    environ_vars = {
        # Basic field configuration
        "ID_FIELD": os.environ.get("ID_FIELD", "id"),
        "LABEL_FIELD": os.environ.get("LABEL_FIELD", "label"),
        "SCORE_FIELD": os.environ.get("SCORE_FIELD", ""),
        "SCORE_FIELDS": os.environ.get("SCORE_FIELDS", ""),
        "TASK_LABEL_NAMES": os.environ.get("TASK_LABEL_NAMES", ""),
        "AMOUNT_FIELD": os.environ.get("AMOUNT_FIELD", ""),
        "INPUT_FORMAT": os.environ.get("INPUT_FORMAT", "auto"),
        # Domain metrics configuration
        "COMPUTE_DOLLAR_RECALL": os.environ.get("COMPUTE_DOLLAR_RECALL", "true"),
        "COMPUTE_COUNT_RECALL": os.environ.get("COMPUTE_COUNT_RECALL", "true"),
        "DOLLAR_RECALL_FPR": os.environ.get("DOLLAR_RECALL_FPR", "0.1"),
        "COUNT_RECALL_CUTOFF": os.environ.get("COUNT_RECALL_CUTOFF", "0.1"),
        # Visualization configuration
        "GENERATE_PLOTS": os.environ.get("GENERATE_PLOTS", "true"),
        # Comparison mode configuration
        "COMPARISON_MODE": os.environ.get("COMPARISON_MODE", "false"),
        "PREVIOUS_SCORE_FIELD": os.environ.get("PREVIOUS_SCORE_FIELD", ""),
        "PREVIOUS_SCORE_FIELDS": os.environ.get("PREVIOUS_SCORE_FIELDS", ""),
        "COMPARISON_METRICS": os.environ.get("COMPARISON_METRICS", "all"),
        "STATISTICAL_TESTS": os.environ.get("STATISTICAL_TESTS", "true"),
        "COMPARISON_PLOTS": os.environ.get("COMPARISON_PLOTS", "true"),
        # PIPER additions
        "VARIANT_MODEL_ID": os.environ.get("VARIANT_MODEL_ID", ""),
        "CONTROL_MODEL_ID": os.environ.get("CONTROL_MODEL_ID", ""),
        "PIPELINE_NAME": os.environ.get("PIPELINE_NAME", ""),
        "DATASET_TYPE": os.environ.get("DATASET_TYPE", "Validation"),
        "METRICS_TO_RENDER": os.environ.get(
            "METRICS_TO_RENDER", "auc_roc,auc_pr,data_statistics"
        ),
    }

    try:
        # Call main function with testability parameters
        main(input_paths, output_paths, environ_vars, args)

        # Signal success
        success_path = os.path.join(output_paths["metric_output"], "_SUCCESS")
        Path(success_path).touch()
        logger.info(f"Created success marker: {success_path}")

        # Create health check file
        health_path = os.path.join(output_paths["metric_output"], "_HEALTH")
        create_health_check_file(health_path)
        logger.info(f"Created health check file: {health_path}")

        sys.exit(0)
    except Exception as e:
        # Log error and create failure marker
        logger.exception(f"Script failed with error: {e}")
        failure_path = os.path.join(
            output_paths.get("metric_output", "/tmp"), "_FAILURE"
        )
        os.makedirs(os.path.dirname(failure_path), exist_ok=True)
        with open(failure_path, "w") as f:
            f.write(f"Error: {str(e)}")
        sys.exit(1)

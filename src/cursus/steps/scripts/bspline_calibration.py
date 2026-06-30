#!/usr/bin/env python3
"""
Monotone B-Spline Calibration Script for SageMaker Processing.

==============================================================================
This is an ALTERNATIVE entry_point for the ModelCalibration step — NOT a
separate step. It is fully interchangeable with model_calibration.py: same
main(input_paths, output_paths, environ_vars, job_args) contract, same I/O
channels (input `evaluation_data` -> outputs `calibration_output` /
`metrics_output` / `calibrated_data`), and the same env-var surface (it sets
CALIBRATION_METHOD=bspline where model_calibration.py defaults to gam, plus the
B-spline knobs MONOTONIC_CONSTRAINT / DEGREE / N_KNOTS / ERROR_THRESHOLD).

It therefore has NO `.step.yaml` of its own (and no registry entry) by design —
it satisfies the existing `model_calibration.step.yaml` interface. To use the
B-spline method instead of GAM for the ModelCalibration step, point the step's
`contract.entry_point` (or the project's source_dir runner) at this script.
See model_calibration.py for the GAM/isotonic/Platt sibling.
==============================================================================

Python port of COSA's generic_rfuge.r — fits a monotone logistic B-spline
to calibrate raw model scores to probabilities. Monotonicity is enforced
via quadratic programming (hard constraint: delta[j+1] >= delta[j]).

Algorithm:
    1. Validate inputs (score/tag ranges, minimum records/fraud)
    2. Adaptive knot placement (quantile-based, dense near 1.0)
    3. Aggregate by score (group → freq, nbad)
    4. Fit monotone B-spline via IRLS + constrained QP
    5. Output bspline_parameters.json

Dependencies:
    - numpy, scipy (B-spline basis, initial GLM)
    - quadprog (constrained QP solver — pip install quadprog)
    - pandas (data aggregation)

Replaces: generic_rfuge.r (R script requiring R container + 17 R packages)
"""

import os
import sys
import json
import logging
import argparse
import traceback
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.optimize import minimize

try:
    import quadprog

    HAS_QUADPROG = True
except ImportError:
    quadprog = None
    HAS_QUADPROG = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Validation thresholds (match R script)
MIN_RECORD = 1000
MIN_FRAUD = 10
MAX_COEFFICIENTS = 1e12
MIN_UNIQUE_VALUES = 10


# ============================================================================
# B-SPLINE BASIS CONSTRUCTION
# ============================================================================


def bspline_basis(x: np.ndarray, knots: np.ndarray, degree: int = 3) -> np.ndarray:
    """
    Construct B-spline basis matrix with intercept.

    Equivalent to R's bs(x, knots=interior, degree=degree,
    Boundary.knots=c(0,1), intercept=TRUE).

    Args:
        x: Evaluation points (N,)
        knots: Full knot sequence including boundary (K,)
        degree: Spline degree (default: 3 = cubic)

    Returns:
        B: Basis matrix (N, n_basis) where n_basis = len(knots) + degree - 1
    """
    interior_knots = knots[1:-1]
    # Augmented knot vector: degree+1 copies of boundary on each side
    t = np.concatenate(
        [
            np.repeat(knots[0], degree + 1),
            interior_knots,
            np.repeat(knots[-1], degree + 1),
        ]
    )
    n_basis = len(t) - degree - 1

    B = np.zeros((len(x), n_basis))
    for i in range(n_basis):
        coeffs = np.zeros(n_basis)
        coeffs[i] = 1.0
        spline = BSpline(t, coeffs, degree, extrapolate=False)
        B[:, i] = spline(x)

    # Handle boundary: NaN at exact boundaries → fill with 0
    B = np.nan_to_num(B, nan=0.0)

    return B


# ============================================================================
# ADAPTIVE KNOT PLACEMENT
# ============================================================================


def compute_adaptive_knots(
    x: np.ndarray,
    n_knots: int = 50,
    min_count_per_interval: int = 10,
) -> np.ndarray:
    """
    Compute adaptive knots based on data quantiles with dense placement near 1.0.

    Exact port of R script's knot placement logic:
    1. Build quantile grid with extra density near 1.0
    2. Compute quantiles of x at grid points
    3. Add uniform grid seq(0,1,11)
    4. Filter: drop intervals with <= min_count_per_interval data points

    Args:
        x: Score values (N,)
        n_knots: Base number of knots
        min_count_per_interval: Minimum data points per interval

    Returns:
        knots: Filtered knot positions including 0 and 1
    """
    # Build quantile grid (matches R exactly)
    quantile_grid = np.sort(
        np.unique(
            np.concatenate(
                [
                    np.linspace(0, 1, n_knots + 1),
                    np.linspace(0, 1, 10),
                    np.linspace(0.5, 1, 10),
                    np.linspace(0.9, 1, 10),
                    np.linspace(0.99, 1, 10),
                    np.linspace(0.999, 1, 10),
                    np.linspace(0.9999, 1, 10),
                ]
            )
        )
    )

    # Compute quantiles of x
    x_quantiles = np.round(np.quantile(x, quantile_grid), 8)

    # Keep unique interior points
    x_quantiles_interior = np.unique(x_quantiles[1:-1])

    # Add uniform grid
    x_grid = np.linspace(0, 1, 11)

    # Combine all knot candidates
    all_knots = np.sort(
        np.unique(
            np.concatenate(
                [
                    [0.0],
                    x_quantiles_interior,
                    x_grid,
                    [1.0],
                ]
            )
        )
    )

    # Count data points in each interval (excluding points on knot boundaries)
    x_not_on_knots = x[~np.isin(x, all_knots)]
    if len(x_not_on_knots) > 0:
        counts, _ = np.histogram(x_not_on_knots, bins=all_knots)
    else:
        counts = np.zeros(len(all_knots) - 1, dtype=int)

    # Filter: keep right endpoints of intervals with > min_count data points
    # knots = c(0, x_quantiles[-1][ncounts > 10])
    surviving_right_endpoints = all_knots[1:][counts > min_count_per_interval]
    knots = np.sort(np.unique(np.concatenate([[0.0], surviving_right_endpoints])))

    # Ensure 1.0 is the last knot
    if knots[-1] < 1.0:
        knots[-1] = 1.0

    logger.info(
        f"Adaptive knots: {len(all_knots)} candidates → {len(knots)} surviving "
        f"(min_count={min_count_per_interval})"
    )

    return knots


# ============================================================================
# MONOTONE SPLINE FITTING
# ============================================================================


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def _initial_glm_fit(
    B: np.ndarray,
    tpr: np.ndarray,
    size: np.ndarray,
    max_samples: int = 1_000_000,
) -> np.ndarray:
    """
    Initial GLM fit (unconstrained logistic regression on B-spline basis).

    Equivalent to R's glm2(tpr ~ -1 + B, family=binomial, weights=size).
    Uses scipy minimize with L-BFGS-B as a simple substitute.

    Args:
        B: Basis matrix (N, n_basis)
        tpr: True positive rate per group (N,)
        size: Group sizes (N,)
        max_samples: Subsample if larger

    Returns:
        delta: Initial coefficient vector (n_basis,)
    """
    n = B.shape[0]
    if n > max_samples:
        idx = np.random.choice(n, max_samples, replace=False)
        B_sub, tpr_sub, size_sub = B[idx], tpr[idx], size[idx]
    else:
        B_sub, tpr_sub, size_sub = B, tpr, size

    def neg_log_lik(delta):
        eta = B_sub @ delta
        mu = np.clip(_sigmoid(eta), 1e-9, 1 - 1e-9)
        ll = size_sub * (tpr_sub * np.log(mu) + (1 - tpr_sub) * np.log(1 - mu))
        return -np.sum(ll)

    result = minimize(neg_log_lik, x0=np.zeros(B.shape[1]), method="L-BFGS-B")
    return result.x


def monotone_spline(
    y: np.ndarray,
    x: np.ndarray,
    size: np.ndarray,
    degree: int = 3,
    boundary: Tuple[float, float] = (0.0, 1.0),
    knots: np.ndarray = None,
    tolerance: float = 1e-6,
    max_iteration: int = 1000,
    lam: float = 1e-10,
) -> Dict[str, Any]:
    """
    Fit monotone B-spline via IRLS + constrained QP.

    Exact port of R's monotone_spline() function. At each IRLS iteration,
    solves a QP with monotonicity constraints delta[j+1] >= delta[j].

    Args:
        y: Number of positives per group (N,)
        x: Score values per group (N,)
        size: Group sizes (N,)
        degree: B-spline degree
        boundary: Score range
        knots: Knot positions (including boundaries)
        tolerance: Convergence tolerance
        max_iteration: Maximum IRLS iterations
        lam: P-spline penalty strength

    Returns:
        Dict with keys: delta, fitted_values, knots, degree, converged
    """
    if knots is None:
        knots = np.linspace(0, 1, 11)

    nknots = len(knots)
    n_basis = nknots + degree - 1

    # Build B-spline basis
    B = bspline_basis(x, knots, degree)
    if B.shape[1] != n_basis:
        logger.warning(
            f"Basis dimension mismatch: expected {n_basis}, got {B.shape[1]}. "
            f"Using actual dimension."
        )
        n_basis = B.shape[1]

    tpr = y / np.maximum(size, 1)

    # Initial unconstrained GLM fit
    delta_old = _initial_glm_fit(B, tpr, size)

    # Monotonicity constraint matrix: delta[j+1] - delta[j] >= 0
    # R: Dmat[i,i] = -1, Dmat[i,i+1] = 1 → Dmat @ delta >= 0
    C = np.zeros((n_basis - 1, n_basis))
    for i in range(n_basis - 1):
        C[i, i] = -1.0
        C[i, i + 1] = 1.0

    # P-spline second-difference penalty matrix
    D2 = np.zeros((n_basis - 2, n_basis))
    for i in range(n_basis - 2):
        D2[i, i] = 1.0
        D2[i, i + 1] = -2.0
        D2[i, i + 2] = 1.0
    DD = D2.T @ D2

    converged = False
    for j in range(max_iteration):
        # IRLS working quantities
        eta = B @ delta_old
        mu = _sigmoid(eta)
        mu = np.clip(mu, 1e-9, 1 - 1e-9)
        wt = size * mu * (1 - mu)
        z = y - size * mu

        # QP matrices
        # Pmat = B^T W B + lambda * DD
        Pmat = (B.T * wt) @ B + lam * DD

        # Ensure Pmat is symmetric positive definite
        Pmat = 0.5 * (Pmat + Pmat.T)
        Pmat += np.eye(n_basis) * 1e-12  # Numerical stability

        # dvec = B^T (z + W B delta_old)
        dvec = B.T @ (z + (wt * (B @ delta_old)))

        # Solve constrained QP
        if HAS_QUADPROG:
            # quadprog.solve_qp(G, a, C^T, b)
            # minimizes 0.5 x^T G x - a^T x  s.t.  C^T x >= b
            delta_new = quadprog.solve_qp(Pmat, dvec, C.T, np.zeros(n_basis - 1))[0]
        else:
            # Fallback: scipy SLSQP
            def qp_obj(d):
                return 0.5 * d @ Pmat @ d - dvec @ d

            constraints = [
                {"type": "ineq", "fun": lambda d, i=i: d[i + 1] - d[i]}
                for i in range(n_basis - 1)
            ]
            result = minimize(
                qp_obj,
                x0=delta_old,
                method="SLSQP",
                constraints=constraints,
                options={"maxiter": 500},
            )
            delta_new = result.x

        # Check convergence
        diff = np.sqrt(np.mean((delta_new - delta_old) ** 2))
        if diff <= tolerance:
            converged = True
            break
        delta_old = delta_new

    fitted_values = _sigmoid(B @ delta_new)

    logger.info(
        f"Monotone spline: {'converged' if converged else 'max iter reached'} "
        f"in {j + 1} iterations (diff={diff:.2e})"
    )

    return {
        "delta": delta_new,
        "fitted_values": fitted_values,
        "knots": knots,
        "degree": degree,
        "converged": converged,
        "n_iterations": j + 1,
    }


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def validate_inputs(tags: np.ndarray, scores: np.ndarray) -> Tuple[bool, str]:
    """Validate inputs (matches R script checks exactly)."""
    if len(tags) != len(scores):
        return False, "Length of tags and scores are not equal."
    if len(tags) < MIN_RECORD:
        return False, f"Too few records: {len(tags)} < {MIN_RECORD}."
    if np.sum(tags) < MIN_FRAUD:
        return False, f"Too few positives: {np.sum(tags)} < {MIN_FRAUD}."
    if not np.all((tags >= 0) & (tags <= 1)):
        return False, "Tags not in range [0, 1]."
    if not np.all((scores >= 0) & (scores <= 1)):
        return False, "Scores not in range [0, 1]."
    return True, "OK"


def validate_result(
    coefficients: np.ndarray, fitted_values: np.ndarray, tpr: np.ndarray
) -> Tuple[bool, str]:
    """Validate calibration result (matches R script checks)."""
    if np.any(np.isnan(coefficients)):
        return False, "Coefficients contain NaN."
    if len(np.unique(np.round(fitted_values, 6))) <= MIN_UNIQUE_VALUES:
        return False, f"Too few unique fitted values (<= {MIN_UNIQUE_VALUES})."
    model_mse = np.mean(np.sqrt((fitted_values - tpr) ** 2))
    max_mse = np.mean(np.sqrt((np.mean(tpr) - tpr) ** 2))
    if model_mse >= max_mse:
        return False, f"Model MSE ({model_mse:.6f}) >= max MSE ({max_mse:.6f})."
    if np.max(np.abs(coefficients)) >= MAX_COEFFICIENTS:
        return (
            False,
            f"Coefficients too large (max={np.max(np.abs(coefficients)):.2e}).",
        )
    return True, "OK"


def run_bspline_calibration(
    tags: np.ndarray,
    scores: np.ndarray,
    degree: int = 3,
    n_knots: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run full B-spline calibration pipeline.

    Args:
        tags: Binary labels (N,)
        scores: Raw model scores (N,)
        degree: B-spline degree (default: 3)
        n_knots: Number of knots (auto if None)

    Returns:
        Dict with coefficients, augmented knots, degree, status
    """
    # Validate
    valid, msg = validate_inputs(tags, scores)
    if not valid:
        logger.error(f"Validation failed: {msg}")
        return {"coefficients": None, "knots": None, "degree": degree, "status": "Fail"}

    # Auto knots based on data size (matches R)
    n = len(scores)
    if n_knots is None:
        if n > 1_000_000:
            n_knots = 100
        elif n < 10_000:
            n_knots = 20
        else:
            n_knots = 50

    # Round scores (matches R: round(score_data[[1]], 6))
    x = np.round(scores, 6)
    y = tags

    # Adaptive knot placement
    knots = compute_adaptive_knots(x, n_knots=n_knots)

    if len(knots) <= 2:
        logger.error("Too few knots (need at least 1 interior point).")
        return {"coefficients": None, "knots": None, "degree": degree, "status": "Fail"}

    # Aggregate by score (matches R: group_by(x) %>% summarise(freq, nbad))
    df = pd.DataFrame({"x": x, "y": y})
    agg = df.groupby("x").agg(freq=("y", "count"), nbad=("y", "sum")).reset_index()

    logger.info(f"Aggregated: {len(df)} rows → {len(agg)} unique scores")

    # Fit monotone spline
    result = monotone_spline(
        y=agg["nbad"].values,
        x=agg["x"].values,
        size=agg["freq"].values,
        knots=knots,
        degree=degree,
    )

    # Validate result
    tpr = agg["nbad"].values / np.maximum(agg["freq"].values, 1)
    valid, msg = validate_result(result["delta"], result["fitted_values"], tpr)

    if valid:
        status = "Success"
        logger.info(f"Calibration succeeded: {len(result['delta'])} coefficients")
    else:
        status = "Fail"
        logger.error(f"Result validation failed: {msg}")

    # Build augmented knot vector (matches R: c(rep(0, degree), knots, rep(1, degree)))
    augmented_knots = np.concatenate(
        [
            np.repeat(0.0, degree),
            knots,
            np.repeat(1.0, degree),
        ]
    )

    return {
        "coefficients": result["delta"].tolist()
        if result["delta"] is not None
        else None,
        "knots": augmented_knots.tolist(),
        "degree": degree,
        "status": status,
    }


# ============================================================================
# MAIN ENTRY POINT (Cursus script contract — drop-in for model_calibration.py)
# ============================================================================


def main(
    input_paths: dict,
    output_paths: dict,
    environ_vars: dict,
    job_args: argparse.Namespace = None,
) -> dict:
    """Main entry point for the B-spline calibration script.

    Drop-in replacement for model_calibration.py when CALIBRATION_METHOD=bspline.
    Same input/output contract: reads evaluation data with label + score columns,
    outputs calibration_summary.json, calibration_metrics.json, calibration_model.pkl,
    and calibrated data.

    Args:
        input_paths: Dictionary of input paths with logical names
            - "evaluation_data": Path to eval data (CSV/TSV/Parquet with label + score columns)
        output_paths: Dictionary of output paths with logical names
            - "calibration_output": Path for calibration artifacts
            - "metrics_output": Path for metrics JSON
            - "calibrated_data": Path for calibrated predictions
        environ_vars: Dictionary of environment variables
            - "CALIBRATION_METHOD": Must be "bspline" (default)
            - "LABEL_FIELD": Label column name (default: "label")
            - "SCORE_FIELD": Score column name (default: "prob_class_1")
            - "SCORE_FIELDS": Multi-task comma-separated score fields (optional)
            - "TASK_LABEL_NAMES": Multi-task comma-separated label fields (optional)
            - "IS_BINARY": "True"/"False" (default: "True")
            - "MONOTONIC_CONSTRAINT": "True"/"False" (default: "True")
            - "CALIBRATION_SAMPLE_POINTS": Lookup table size (default: "1000")
            - "DEGREE": B-spline degree (default: "3")
            - "N_KNOTS": Number of knots (default: auto)
            - "ERROR_THRESHOLD": Min improvement threshold (default: "0.05")
        job_args: Command line arguments (optional)

    Returns:
        Dictionary with status and results (same contract as model_calibration.py)
    """
    try:
        logger.info("=" * 60)
        logger.info("B-Spline Calibration (Python port of generic_rfuge.r)")
        logger.info("=" * 60)

        # Parse environment variables (same as model_calibration.py)
        label_field = environ_vars.get("LABEL_FIELD", "label")
        score_field = environ_vars.get("SCORE_FIELD", "prob_class_1")
        score_fields_str = environ_vars.get("SCORE_FIELDS", "")
        task_label_names_str = environ_vars.get("TASK_LABEL_NAMES", "")
        is_binary = environ_vars.get("IS_BINARY", "True").lower() == "true"
        degree = int(environ_vars.get("DEGREE", "3"))
        n_knots_str = environ_vars.get("N_KNOTS", "")
        n_knots = int(n_knots_str) if n_knots_str else None
        sample_points = int(environ_vars.get("CALIBRATION_SAMPLE_POINTS", "1000"))

        # Resolve paths
        input_data_path = input_paths.get(
            "evaluation_data", "/opt/ml/processing/input/eval_data"
        )
        calibration_output = output_paths.get(
            "calibration_output", "/opt/ml/processing/output/calibration"
        )
        metrics_output = output_paths.get(
            "metrics_output", "/opt/ml/processing/output/metrics"
        )
        calibrated_data_output = output_paths.get(
            "calibrated_data", "/opt/ml/processing/output/calibrated_data"
        )

        os.makedirs(calibration_output, exist_ok=True)
        os.makedirs(metrics_output, exist_ok=True)
        os.makedirs(calibrated_data_output, exist_ok=True)

        # Load evaluation data (same format detection as model_calibration.py)
        data_file = _find_first_data_file(input_data_path)
        df, input_format = _load_dataframe_with_format(data_file)
        logger.info(f"Loaded {len(df)} rows from {data_file} (format={input_format})")

        # Determine score fields (multi-task support)
        if score_fields_str:
            score_field_list = [
                s.strip() for s in score_fields_str.split(",") if s.strip()
            ]
        else:
            score_field_list = [score_field]

        if task_label_names_str:
            label_field_list = [
                s.strip() for s in task_label_names_str.split(",") if s.strip()
            ]
        else:
            label_field_list = [label_field] * len(score_field_list)

        # Run calibration per score field
        all_results = {}
        for sf, lf in zip(score_field_list, label_field_list):
            logger.info(f"Calibrating score_field={sf}, label_field={lf}")

            if sf not in df.columns:
                logger.error(f"Score field '{sf}' not found in data")
                continue
            if lf not in df.columns:
                logger.error(f"Label field '{lf}' not found in data")
                continue

            # Coerce to numeric (non-numeric values become NaN) so we can report
            # bad rows with context instead of an uninformative ValueError.
            score_series = pd.to_numeric(df[sf], errors="coerce")
            label_series = pd.to_numeric(df[lf], errors="coerce")

            # Drop rows where either field is missing/non-numeric. Rows with
            # fully valid numeric data are unaffected, preserving prior behavior.
            valid_mask = score_series.notna() & label_series.notna()
            n_dropped = int((~valid_mask).sum())
            if n_dropped:
                logger.warning(
                    f"Dropping {n_dropped} row(s) with missing/non-numeric values "
                    f"in '{sf}' or '{lf}' before calibration"
                )
            if not valid_mask.any():
                logger.error(
                    f"No valid numeric rows for score_field='{sf}', "
                    f"label_field='{lf}'; skipping calibration"
                )
                all_results[sf] = {
                    "status": "Fail",
                    "error": "No valid numeric rows",
                }
                continue

            valid_idx = df.index[valid_mask]
            scores = score_series[valid_mask].values.astype(float)
            tags = label_series[valid_mask].values.astype(float)

            # Run B-spline calibration
            result = run_bspline_calibration(
                tags, scores, degree=degree, n_knots=n_knots
            )

            if result["status"] == "Success" and result["coefficients"] is not None:
                # Build lookup table (same format as model_calibration's _model_to_lookup_table)
                augmented_knots = np.array(result["knots"])
                coefficients = np.array(result["coefficients"])
                score_range = np.linspace(0, 1, sample_points)
                t = augmented_knots
                spline = BSpline(t, coefficients, degree, extrapolate=True)
                calibrated_values = _sigmoid(spline(score_range))
                lookup_table = list(
                    zip(score_range.tolist(), calibrated_values.tolist())
                )

                # Apply calibration to data (only to the valid numeric rows)
                raw_scores = np.array([s for s, _ in lookup_table])
                cal_scores = np.array([c for _, c in lookup_table])
                calibrated_scores = np.interp(scores, raw_scores, cal_scores)
                df.loc[valid_idx, f"{sf}_calibrated"] = calibrated_scores

                # Compute metrics
                from sklearn.metrics import brier_score_loss

                uncal_brier = brier_score_loss(tags, scores)
                cal_brier = brier_score_loss(tags, calibrated_scores)

                all_results[sf] = {
                    "status": "Success",
                    "bspline_params": result,
                    "lookup_table": lookup_table,
                    "uncalibrated_brier": float(uncal_brier),
                    "calibrated_brier": float(cal_brier),
                    "improvement_percentage": float(
                        (uncal_brier - cal_brier) / max(uncal_brier, 1e-10) * 100
                    ),
                }
            else:
                all_results[sf] = {
                    "status": "Fail",
                    "error": result.get("status", "Unknown"),
                }

        # Save outputs (same structure as model_calibration.py)
        # 1. calibration_summary.json
        summary = {
            "status": "success"
            if any(r["status"] == "Success" for r in all_results.values())
            else "failure",
            "mode": "binary" if is_binary else "multiclass",
            "calibration_method": "bspline",
            "tasks": {
                sf: {
                    "uncalibrated_brier": r.get("uncalibrated_brier"),
                    "calibrated_brier": r.get("calibrated_brier"),
                    "improvement_percentage": r.get("improvement_percentage"),
                }
                for sf, r in all_results.items()
            },
        }
        with open(
            os.path.join(calibration_output, "calibration_summary.json"), "w"
        ) as f:
            json.dump(summary, f, indent=2)

        # 2. calibration_model.pkl (bspline params + lookup tables)
        import pickle

        calibrators = {}
        for sf, r in all_results.items():
            if r["status"] == "Success":
                calibrators[sf] = r["lookup_table"]
        with open(os.path.join(calibration_output, "calibration_model.pkl"), "wb") as f:
            pickle.dump(calibrators, f)

        # 3. calibration_metrics.json
        metrics = {
            sf: {
                "uncalibrated_brier_score": r.get("uncalibrated_brier"),
                "calibrated_brier_score": r.get("calibrated_brier"),
            }
            for sf, r in all_results.items()
        }
        with open(os.path.join(metrics_output, "calibration_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # 4. Calibrated data
        output_base = os.path.join(calibrated_data_output, "calibrated_predictions")
        _save_dataframe_with_format(df, output_base, input_format)

        # 5. bspline_parameters.json (B-spline specific — for inference reconstruction)
        for sf, r in all_results.items():
            if r["status"] == "Success":
                bspline_file = os.path.join(
                    calibration_output, f"bspline_parameters_{sf}.json"
                )
                with open(bspline_file, "w") as f:
                    json.dump(r["bspline_params"], f, indent=2)

        logger.info(f"Calibration complete. Summary: {summary['status']}")
        return summary

    except Exception as e:
        logger.error(f"B-spline calibration failed: {str(e)}")
        logger.error(traceback.format_exc())
        return {"status": "failure", "error_message": str(e)}


# ============================================================================
# FILE I/O HELPERS (same as model_calibration.py)
# ============================================================================


def _detect_file_format(file_path: str) -> str:
    from pathlib import Path

    suffix = Path(file_path).suffix.lower()
    if suffix == ".csv":
        return "csv"
    elif suffix == ".tsv":
        return "tsv"
    elif suffix == ".parquet":
        return "parquet"
    else:
        raise RuntimeError(f"Unsupported file format: {suffix}")


def _load_dataframe_with_format(file_path: str) -> Tuple[pd.DataFrame, str]:
    fmt = _detect_file_format(file_path)
    if fmt == "csv":
        df = pd.read_csv(file_path)
    elif fmt == "tsv":
        df = pd.read_csv(file_path, sep="\t")
    elif fmt == "parquet":
        df = pd.read_parquet(file_path)
    return df, fmt


def _save_dataframe_with_format(df: pd.DataFrame, output_base: str, fmt: str) -> str:
    from pathlib import Path

    output_base = Path(output_base)
    if fmt == "csv":
        path = output_base.with_suffix(".csv")
        df.to_csv(path, index=False)
    elif fmt == "tsv":
        path = output_base.with_suffix(".tsv")
        df.to_csv(path, sep="\t", index=False)
    elif fmt == "parquet":
        path = output_base.with_suffix(".parquet")
        df.to_parquet(path, index=False)
    return str(path)


def _find_first_data_file(data_dir: str) -> str:
    for fname in sorted(os.listdir(data_dir)):
        if fname.lower().endswith((".csv", ".tsv", ".parquet")):
            return os.path.join(data_dir, fname)
    raise FileNotFoundError(f"No data file found in {data_dir}")


# ============================================================================
# COMMAND LINE ENTRY POINT (same contract as model_calibration.py)
# ============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Monotone B-Spline Calibration Script for SageMaker Processing"
    )
    parser.add_argument(
        "--job_type",
        type=str,
        default="calibration",
        help="Job type - one of: training, calibration, validation, testing",
    )
    args = parser.parse_args()

    logger.info(f"Starting B-spline calibration with job_type: {args.job_type}")

    # Define standard SageMaker paths (same as model_calibration.py)
    INPUT_DATA_PATH = "/opt/ml/processing/input/eval_data"
    OUTPUT_CALIBRATION_PATH = "/opt/ml/processing/output/calibration"
    OUTPUT_METRICS_PATH = "/opt/ml/processing/output/metrics"
    OUTPUT_CALIBRATED_DATA_PATH = "/opt/ml/processing/output/calibrated_data"

    # Parse environment variables (same as model_calibration.py + bspline-specific)
    environ_vars = {
        "CALIBRATION_METHOD": os.environ.get("CALIBRATION_METHOD", "bspline"),
        "LABEL_FIELD": os.environ.get("LABEL_FIELD", "label"),
        "SCORE_FIELD": os.environ.get("SCORE_FIELD", "prob_class_1"),
        "SCORE_FIELDS": os.environ.get("SCORE_FIELDS", ""),
        "TASK_LABEL_NAMES": os.environ.get("TASK_LABEL_NAMES", ""),
        "IS_BINARY": os.environ.get("IS_BINARY", "True"),
        "MONOTONIC_CONSTRAINT": os.environ.get("MONOTONIC_CONSTRAINT", "True"),
        "CALIBRATION_SAMPLE_POINTS": os.environ.get(
            "CALIBRATION_SAMPLE_POINTS", "1000"
        ),
        "DEGREE": os.environ.get("DEGREE", "3"),
        "N_KNOTS": os.environ.get("N_KNOTS", ""),
        "ERROR_THRESHOLD": os.environ.get("ERROR_THRESHOLD", "0.05"),
    }

    # Set up input and output paths (same as model_calibration.py)
    input_paths = {"evaluation_data": INPUT_DATA_PATH}
    output_paths = {
        "calibration_output": OUTPUT_CALIBRATION_PATH,
        "metrics_output": OUTPUT_METRICS_PATH,
        "calibrated_data": OUTPUT_CALIBRATED_DATA_PATH,
    }

    # Call the main function
    try:
        results = main(input_paths, output_paths, environ_vars, args)
        if results.get("status") == "success":
            logger.info("B-spline calibration completed successfully")
            sys.exit(0)
        else:
            logger.error(
                f"B-spline calibration failed: {results.get('error_message', 'Unknown error')}"
            )
            sys.exit(1)
    except Exception as e:
        logger.error(f"B-spline calibration failed: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

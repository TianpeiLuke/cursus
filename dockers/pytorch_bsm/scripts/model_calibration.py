#!/usr/bin/env python
"""Model Calibration Script for SageMaker Processing.

This script calibrates model prediction scores to accurate probabilities,
which is essential for risk-based decision-making and threshold setting.
It supports multiple calibration methods including GAM, Isotonic Regression,
and Platt Scaling, with options for monotonicity constraints.
It supports both binary and multi-class classification scenarios.
"""

import os
import sys
import json
import logging
import traceback
import argparse
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

# Import pygam for GAM implementation if available
try:
    from pygam import LogisticGAM, s

    HAS_PYGAM = True
except ImportError:
    HAS_PYGAM = False

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define standard SageMaker paths
INPUT_DATA_PATH = "/opt/ml/processing/input/eval_data"
OUTPUT_CALIBRATION_PATH = "/opt/ml/processing/output/calibration"
OUTPUT_METRICS_PATH = "/opt/ml/processing/output/metrics"
OUTPUT_CALIBRATED_DATA_PATH = "/opt/ml/processing/output/calibrated_data"


class CalibrationConfig:
    """Configuration class for model calibration."""

    def __init__(
        self,
        input_data_path: str = "/opt/ml/processing/input/eval_data",
        output_calibration_path: str = "/opt/ml/processing/output/calibration",
        output_metrics_path: str = "/opt/ml/processing/output/metrics",
        output_calibrated_data_path: str = "/opt/ml/processing/output/calibrated_data",
        calibration_method: str = "gam",
        label_field: str = "label",
        score_field: str = "prob_class_1",
        is_binary: bool = True,
        monotonic_constraint: bool = True,
        gam_splines: int = 10,
        error_threshold: float = 0.05,
        num_classes: int = 2,
        score_field_prefix: str = "prob_class_",
        multiclass_categories: Optional[List[str]] = None,
    ):
        """Initialize configuration with paths and parameters."""
        # I/O Paths
        self.input_data_path = input_data_path
        self.output_calibration_path = output_calibration_path
        self.output_metrics_path = output_metrics_path
        self.output_calibrated_data_path = output_calibrated_data_path

        # Calibration parameters
        self.calibration_method = calibration_method.lower()
        self.label_field = label_field
        self.score_field = score_field
        self.is_binary = is_binary
        self.monotonic_constraint = monotonic_constraint
        self.gam_splines = gam_splines
        self.error_threshold = error_threshold

        # Multi-class parameters
        self.num_classes = num_classes
        self.score_field_prefix = score_field_prefix

        # Initialize multiclass_categories
        if multiclass_categories:
            self.multiclass_categories = multiclass_categories
        else:
            self.multiclass_categories = [str(i) for i in range(num_classes)]

    @classmethod
    def from_env(cls) -> 'CalibrationConfig':
        """Create configuration from environment variables."""
        # Parse multiclass categories from environment
        multiclass_categories = None
        if os.environ.get("IS_BINARY", "True").lower() != "true":
            multiclass_cats = os.environ.get("MULTICLASS_CATEGORIES", None)
            if multiclass_cats:
                try:
                    multiclass_categories = json.loads(multiclass_cats)
                except json.JSONDecodeError:
                    # Fallback to simple parsing if not valid JSON
                    multiclass_categories = multiclass_cats.split(",")

        # Use global path variables for input/output paths (fixed paths from contract)
        return cls(
            input_data_path=INPUT_DATA_PATH,
            output_calibration_path=OUTPUT_CALIBRATION_PATH,
            output_metrics_path=OUTPUT_METRICS_PATH,
            output_calibrated_data_path=OUTPUT_CALIBRATED_DATA_PATH,
            calibration_method=os.environ.get("CALIBRATION_METHOD", "gam"),
            label_field=os.environ.get("LABEL_FIELD", "label"),
            score_field=os.environ.get("SCORE_FIELD", "prob_class_1"),
            is_binary=os.environ.get("IS_BINARY", "True").lower() == "true",
            monotonic_constraint=os.environ.get("MONOTONIC_CONSTRAINT", "True").lower()
            == "true",
            gam_splines=int(os.environ.get("GAM_SPLINES", "10")),
            error_threshold=float(os.environ.get("ERROR_THRESHOLD", "0.05")),
            num_classes=int(os.environ.get("NUM_CLASSES", "2")),
            score_field_prefix=os.environ.get("SCORE_FIELD_PREFIX", "prob_class_"),
            multiclass_categories=multiclass_categories,
        )


def create_directories(config: Optional['CalibrationConfig'] = None) -> None:
    """Create output directories if they don't exist."""
    config = config or CalibrationConfig.from_env()
    os.makedirs(config.output_calibration_path, exist_ok=True)
    os.makedirs(config.output_metrics_path, exist_ok=True)
    os.makedirs(config.output_calibrated_data_path, exist_ok=True)


def find_first_data_file(data_dir: Optional[str] = None, config: Optional['CalibrationConfig'] = None) -> str:
    """Find the first supported data file in directory.

    Args:
        data_dir: Directory to search for data files (defaults to config input_data_path)
        config: Configuration object (optional, created from environment if not provided)

    Returns:
        str: Path to the first supported data file found

    Raises:
        FileNotFoundError: If no supported data file is found
    """
    config = config or CalibrationConfig.from_env()
    data_dir = data_dir or config.input_data_path

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Directory does not exist: {data_dir}")

    for fname in sorted(os.listdir(data_dir)):
        if fname.lower().endswith((".csv", ".parquet", ".json")):
            return os.path.join(data_dir, fname)

    raise FileNotFoundError(
        f"No supported data file (.csv, .parquet, .json) found in {data_dir}"
    )


def load_data(config: Optional['CalibrationConfig'] = None) -> pd.DataFrame:
    """Load evaluation data with predictions.

    Args:
        config: Configuration object (optional, created from environment if not provided)

    Returns:
        pd.DataFrame: Loaded evaluation data

    Raises:
        FileNotFoundError: If no data file is found
        ValueError: If required columns are missing
    """
    config = config or CalibrationConfig.from_env()
    data_file = find_first_data_file(config.input_data_path, config)

    logger.info(f"Loading data from {data_file}")
    if data_file.endswith(".parquet"):
        df = pd.read_parquet(data_file)
    elif data_file.endswith(".csv"):
        df = pd.read_csv(data_file)
    else:
        raise ValueError(f"Unsupported file format: {data_file}")

    # Validate required columns
    if config.label_field not in df.columns:
        raise ValueError(f"Label field '{config.label_field}' not found in data")

    if config.is_binary:
        # Binary classification case
        if config.score_field not in df.columns:
            raise ValueError(f"Score field '{config.score_field}' not found in data")
    else:
        # Multi-class classification case
        found_classes = 0
        for i in range(config.num_classes):
            class_name = config.multiclass_categories[i]
            col_name = f"{config.score_field_prefix}{class_name}"
            if col_name in df.columns:
                found_classes += 1
            else:
                logger.warning(f"Probability column '{col_name}' not found in data")

        if found_classes == 0:
            raise ValueError(
                f"No probability columns found with prefix '{config.score_field_prefix}'"
            )
        elif found_classes < config.num_classes:
            logger.warning(
                f"Only {found_classes}/{config.num_classes} probability columns found"
            )

    logger.info(f"Loaded data with shape {df.shape}")
    return df


def log_section(title: str) -> None:
    """Log a section title with delimiters for better visibility."""
    delimiter = "=" * 80
    logger.info(delimiter)
    logger.info(f"  {title}")
    logger.info(delimiter)


def extract_and_load_nested_tarball_data(config: Optional['CalibrationConfig'] = None) -> pd.DataFrame:
    """Extract and load data from nested tar.gz files in SageMaker output structure.

    Handles SageMaker's specific output structure:
    - output.tar.gz (outer archive)
      - val.tar.gz (inner archive)
        - val/predictions.csv (actual data)
        - val_metrics/... (metrics and plots)
      - test.tar.gz (inner archive)
        - test/predictions.csv (actual data)
        - test_metrics/... (metrics and plots)

    Also handles cases where the input path contains:
    - Direct output.tar.gz file
    - Path to a job directory that contains output/output.tar.gz
    - Path to a parent directory with job subdirectories

    Args:
        config: Configuration object (optional, created from environment if not provided)

    Returns:
        pd.DataFrame: Combined dataset with predictions from extracted tar.gz files

    Raises:
        FileNotFoundError: If necessary tar.gz files or prediction data not found
    """
    import tarfile
    import tempfile
    import shutil

    config = config or CalibrationConfig.from_env()
    input_dir = config.input_data_path
    log_section("NESTED TARBALL EXTRACTION")
    logger.info(f"Looking for SageMaker output archive in {input_dir}")

    # Check if we have a direct data file first (non-tarball case)
    try:
        direct_file = find_first_data_file(input_dir)
        if direct_file:
            logger.info(
                f"Found direct data file: {direct_file}, using standard loading"
            )
            return load_data(config)
    except FileNotFoundError:
        # No direct data file, continue with tarball extraction
        pass

    # First check: Direct tarball in the input directory
    output_archive = None
    for fname in os.listdir(input_dir):
        if fname.lower() == "output.tar.gz":
            output_archive = os.path.join(input_dir, fname)
            logger.info(f"Found output.tar.gz directly in input directory")
            break

    # Second check: Look for job-specific directories containing output/output.tar.gz
    if not output_archive:
        logger.info(
            "No output.tar.gz found directly in input directory, checking for job directories"
        )
        for item in os.listdir(input_dir):
            item_path = os.path.join(input_dir, item)
            if os.path.isdir(item_path):
                # Check if this directory has an output/output.tar.gz file
                output_dir = os.path.join(item_path, "output")
                if os.path.isdir(output_dir):
                    nested_archive = os.path.join(output_dir, "output.tar.gz")
                    if os.path.isfile(nested_archive):
                        output_archive = nested_archive
                        logger.info(f"Found nested output.tar.gz at {output_archive}")
                        break

    # Third check: Recursive search for output.tar.gz (most robust but potentially slower)
    if not output_archive:
        logger.info(
            "No output.tar.gz found in expected locations, performing recursive search"
        )
        for root, _, files in os.walk(input_dir):
            for fname in files:
                if fname.lower() == "output.tar.gz":
                    output_archive = os.path.join(root, fname)
                    logger.info(
                        f"Found output.tar.gz from recursive search at {output_archive}"
                    )
                    break
            if output_archive:
                break

    # If we still don't have it, fall back to standard data loading
    if not output_archive:
        logger.warning(
            "No output.tar.gz found anywhere, falling back to standard data loading"
        )
        return load_data(config)

    logger.info(f"Found SageMaker output archive: {output_archive}")
    logger.info(f"File size: {os.path.getsize(output_archive) / (1024*1024):.2f} MB")

    # Create temporary directories for extraction
    outer_temp_dir = tempfile.mkdtemp(prefix="outer_")
    inner_temp_dir = tempfile.mkdtemp(prefix="inner_")
    combined_df = None

    try:
        # Step 1: Extract the outer archive (output.tar.gz)
        logger.info(f"Extracting outer archive: {output_archive}")
        with tarfile.open(output_archive, "r:gz") as tar:
            # Log the contents of the tar file
            members = tar.getmembers()
            logger.info(f"Outer archive contains {len(members)} files:")
            for member in members:
                logger.info(f"  - {member.name} ({member.size / 1024:.2f} KB)")
            tar.extractall(path=outer_temp_dir)
        logger.info(f"Extracted to: {outer_temp_dir}")

        # Step 2: Find and extract the inner archives (val.tar.gz, test.tar.gz)
        inner_archives = []
        for fname in os.listdir(outer_temp_dir):
            if fname.lower().endswith(".tar.gz"):
                inner_archives.append(os.path.join(outer_temp_dir, fname))

        if not inner_archives:
            raise FileNotFoundError(
                "No val.tar.gz or test.tar.gz found in output.tar.gz"
            )

        logger.info(
            f"Found {len(inner_archives)} inner archives: {[os.path.basename(a) for a in inner_archives]}"
        )

        # Process each inner archive (val.tar.gz, test.tar.gz)
        for inner_archive in inner_archives:
            archive_name = os.path.basename(inner_archive).split(".")[
                0
            ]  # 'val' or 'test'
            logger.info(f"Processing {archive_name} archive: {inner_archive}")

            # Extract the inner archive
            inner_extract_dir = os.path.join(inner_temp_dir, archive_name)
            os.makedirs(inner_extract_dir, exist_ok=True)

            with tarfile.open(inner_archive, "r:gz") as tar:
                # Log the contents of the tar file
                members = tar.getmembers()
                logger.info(f"Inner archive contains {len(members)} files:")
                for member in members:
                    logger.info(f"  - {member.name} ({member.size / 1024:.2f} KB)")
                tar.extractall(path=inner_extract_dir)
            logger.info(f"Extracted inner archive to: {inner_extract_dir}")

            # Look for predictions.csv in the correct structure
            predictions_path = os.path.join(
                inner_extract_dir, archive_name, "predictions.csv"
            )
            if not os.path.exists(predictions_path):
                logger.warning(
                    f"Could not find predictions.csv in {inner_archive}, skipping"
                )
                continue

            # Load the predictions
            df = pd.read_csv(predictions_path)
            logger.info(f"Loaded {len(df)} rows from {predictions_path}")
            # Log data preview and column info for debugging
            logger.info(f"Columns in {predictions_path}: {df.columns.tolist()}")
            logger.info(f"Data types: {df.dtypes.to_dict()}")
            if len(df) > 0:
                logger.info(f"First row sample: {df.iloc[0].to_dict()}")

            # Add dataset origin column
            df["dataset_origin"] = archive_name

            # Combine with previous data
            if combined_df is None:
                combined_df = df
            else:
                # Check if columns match
                if set(df.columns) != set(combined_df.columns):
                    logger.warning(
                        f"Column mismatch between datasets. Common columns will be used."
                    )
                    common_cols = list(
                        set(df.columns).intersection(set(combined_df.columns))
                    )
                    combined_df = pd.concat([combined_df[common_cols], df[common_cols]])
                else:
                    combined_df = pd.concat([combined_df, df])

        if combined_df is None or len(combined_df) == 0:
            raise FileNotFoundError(
                "No valid prediction data found in extracted archives"
            )

        # Log information about the final combined dataset
        logger.info(
            f"Combined dataset contains {len(combined_df)} rows with {len(combined_df.columns)} columns"
        )
        logger.info(f"Final columns: {combined_df.columns.tolist()}")
        # Check for NaN values
        nan_counts = combined_df.isna().sum()
        if nan_counts.sum() > 0:
            logger.warning(
                f"Dataset contains NaN values: {nan_counts[nan_counts > 0].to_dict()}"
            )

        return combined_df

    finally:
        # Clean up temporary directories
        shutil.rmtree(outer_temp_dir, ignore_errors=True)
        shutil.rmtree(inner_temp_dir, ignore_errors=True)


def load_and_prepare_data(config: Optional['CalibrationConfig'] = None, job_type: str = "calibration") -> Tuple[pd.DataFrame, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Load evaluation data and prepare it for calibration based on classification type.

    Args:
        config: Configuration object (optional, created from environment if not provided)
        job_type: The job type to determine how to load data

    Returns:
        tuple: Different return values based on classification type:
            - Binary: (df, y_true, y_prob, None)
            - Multi-class: (df, y_true, None, y_prob_matrix)

    Raises:
        FileNotFoundError: If no data file is found
        ValueError: If required columns are missing
    """
    config = config or CalibrationConfig.from_env()

    log_section("DATA PREPARATION")

    # Load data differently based on job_type
    if job_type == "training":
        # Training job outputs are nested tarballs from XGBoostTraining output
        logger.info(
            f"Loading data for job_type=training using nested tarball extraction"
        )
        try:
            df = extract_and_load_nested_tarball_data(config)
        except Exception as e:
            logger.warning(f"Failed to extract data from nested tarballs: {e}")
            logger.warning(f"Exception details: {traceback.format_exc()}")
            logger.info("Falling back to standard data loading")
            df = load_data(config)
    else:
        # Calibration, validation, and testing job outputs are direct files from XGBoostModelEval
        logger.info(f"Loading data for job_type={job_type} using standard loading")
        df = load_data(config)

    if config.is_binary:
        # Binary case - single score field
        y_true = df[config.label_field].values
        y_prob = df[config.score_field].values
        return df, y_true, y_prob, None
    else:
        # Multi-class case - multiple probability columns
        y_true = df[config.label_field].values

        # Get all probability columns
        prob_columns = []
        for i in range(config.num_classes):
            class_name = config.multiclass_categories[i]
            col_name = f"{config.score_field_prefix}{class_name}"
            if col_name not in df.columns:
                # Try numeric index as fallback
                col_name = f"{config.score_field_prefix}{i}"
                if col_name not in df.columns:
                    raise ValueError(
                        f"Could not find probability column for class {class_name}"
                    )
            prob_columns.append(col_name)

        logger.info(f"Found probability columns for multi-class: {prob_columns}")

        # Extract probability matrix (samples × classes)
        y_prob_matrix = df[prob_columns].values

        return df, y_true, None, y_prob_matrix


def train_gam_calibration(scores: np.ndarray, labels: np.ndarray, config: Optional['CalibrationConfig'] = None):
    """Train a GAM calibration model with optional monotonicity constraints.

    Args:
        scores: Raw prediction scores to calibrate
        labels: Ground truth binary labels (0/1)
        config: Configuration object (optional, created from environment if not provided)

    Returns:
        LogisticGAM: Trained GAM calibration model

    Raises:
        ImportError: If pygam is not installed
    """
    config = config or CalibrationConfig.from_env()

    if not HAS_PYGAM:
        raise ImportError(
            "pygam package is required for GAM calibration but not installed"
        )

    scores = scores.reshape(-1, 1)  # Reshape for GAM

    # Configure GAM with monotonic constraint if specified
    if config.monotonic_constraint:
        gam = LogisticGAM(
            s(0, n_splines=config.gam_splines, constraints="monotonic_inc")
        )
        logger.info(
            f"Training GAM with monotonic constraint, {config.gam_splines} splines"
        )
    else:
        gam = LogisticGAM(s(0, n_splines=config.gam_splines))
        logger.info(
            f"Training GAM without monotonic constraint, {config.gam_splines} splines"
        )

    gam.fit(scores, labels)
    logger.info(f"GAM training complete, deviance: {gam.statistics_['deviance']}")
    return gam


def train_isotonic_calibration(scores: np.ndarray, labels: np.ndarray, config: Optional['CalibrationConfig'] = None) -> IsotonicRegression:
    """Train an isotonic regression calibration model.

    Args:
        scores: Raw prediction scores to calibrate
        labels: Ground truth binary labels (0/1)
        config: Configuration object (optional)

    Returns:
        IsotonicRegression: Trained isotonic regression model
    """
    logger.info("Training isotonic regression calibration model")
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(scores, labels)
    logger.info("Isotonic regression training complete")
    return ir


def train_platt_scaling(scores: np.ndarray, labels: np.ndarray, config: Optional['CalibrationConfig'] = None) -> LogisticRegression:
    """Train a Platt scaling (logistic regression) calibration model.

    Args:
        scores: Raw prediction scores to calibrate
        labels: Ground truth binary labels (0/1)
        config: Configuration object (optional)

    Returns:
        LogisticRegression: Trained logistic regression model
    """
    logger.info("Training Platt scaling (logistic regression) calibration model")
    scores = scores.reshape(-1, 1)  # Reshape for LogisticRegression
    lr = LogisticRegression(C=1e5)  # High C for minimal regularization
    lr.fit(scores, labels)
    logger.info("Platt scaling training complete")
    return lr


def train_multiclass_calibration(y_prob_matrix: np.ndarray, y_true: np.ndarray, method: str = "isotonic", config: Optional['CalibrationConfig'] = None) -> List[Any]:
    """Train calibration models for each class in one-vs-rest fashion.

    Args:
        y_prob_matrix: Matrix of prediction probabilities (samples × classes)
        y_true: Ground truth class labels
        method: Calibration method to use ("gam", "isotonic", "platt")
        config: Configuration object (optional, created from environment if not provided)

    Returns:
        list: List of calibration models, one for each class
    """
    config = config or CalibrationConfig.from_env()
    calibrators = []
    n_classes = y_prob_matrix.shape[1]

    # One-hot encode true labels for one-vs-rest approach
    y_true_onehot = np.zeros((len(y_true), n_classes))
    for i in range(len(y_true)):
        class_idx = int(y_true[i])
        if 0 <= class_idx < n_classes:
            y_true_onehot[i, class_idx] = 1

    # Train a calibrator for each class
    for i in range(n_classes):
        class_name = config.multiclass_categories[i]
        logger.info(f"Training calibration model for class {class_name}")

        if method == "gam":
            if HAS_PYGAM:
                calibrator = train_gam_calibration(
                    y_prob_matrix[:, i], y_true_onehot[:, i], config
                )
            else:
                logger.warning("pygam not installed, falling back to Platt scaling")
                calibrator = train_platt_scaling(
                    y_prob_matrix[:, i], y_true_onehot[:, i], config
                )
        elif method == "isotonic":
            calibrator = train_isotonic_calibration(
                y_prob_matrix[:, i], y_true_onehot[:, i], config
            )
        elif method == "platt":
            calibrator = train_platt_scaling(
                y_prob_matrix[:, i], y_true_onehot[:, i], config
            )
        else:
            raise ValueError(f"Unknown calibration method: {method}")

        calibrators.append(calibrator)

    return calibrators


def apply_multiclass_calibration(y_prob_matrix: np.ndarray, calibrators: List[Any], config: Optional['CalibrationConfig'] = None) -> np.ndarray:
    """Apply calibration to each class probability and normalize.

    Args:
        y_prob_matrix: Matrix of uncalibrated probabilities (samples × classes)
        calibrators: List of calibration models, one for each class
        config: Configuration object (optional, created from environment if not provided)

    Returns:
        np.ndarray: Matrix of calibrated probabilities (samples × classes)
    """
    config = config or CalibrationConfig.from_env()
    n_samples = y_prob_matrix.shape[0]
    n_classes = y_prob_matrix.shape[1]
    calibrated_probs = np.zeros((n_samples, n_classes))

    # Apply each calibrator to corresponding class probabilities
    for i in range(n_classes):
        class_name = config.multiclass_categories[i]
        logger.info(f"Applying calibration for class {class_name}")

        if isinstance(calibrators[i], IsotonicRegression):
            calibrated_probs[:, i] = calibrators[i].transform(y_prob_matrix[:, i])
        elif isinstance(calibrators[i], LogisticRegression):
            calibrated_probs[:, i] = calibrators[i].predict_proba(
                y_prob_matrix[:, i].reshape(-1, 1)
            )[:, 1]
        else:  # GAM
            calibrated_probs[:, i] = calibrators[i].predict_proba(
                y_prob_matrix[:, i].reshape(-1, 1)
            )

    # Normalize to ensure sum of probabilities = 1
    row_sums = calibrated_probs.sum(axis=1)
    calibrated_probs = calibrated_probs / row_sums[:, np.newaxis]

    return calibrated_probs


def compute_calibration_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> Dict[str, Any]:
    """Compute comprehensive calibration metrics including ECE, MCE, and reliability diagram.

    This function calculates:
    - Expected Calibration Error (ECE): weighted average of absolute calibration errors
    - Maximum Calibration Error (MCE): maximum calibration error across all bins
    - Reliability diagram data: points for plotting calibration curve
    - Bin statistics: detailed information about each probability bin
    - Brier score: quadratic scoring rule for probabilistic predictions
    - Preservation of discrimination: comparison of AUC before/after calibration

    Args:
        y_true: Ground truth binary labels (0/1)
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration curve

    Returns:
        Dict: Dictionary containing calibration metrics
    """
    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    # Get bin assignments and counts
    bin_indices = np.minimum(n_bins - 1, (y_prob * n_bins).astype(int))
    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    bin_counts = bin_counts.astype(np.float64)

    # Compute mean predicted probability in each bin
    bin_probs = np.bincount(bin_indices, weights=y_prob, minlength=n_bins) / np.maximum(
        bin_counts, 1
    )

    # Compute mean true label in each bin
    bin_true = np.bincount(bin_indices, weights=y_true, minlength=n_bins) / np.maximum(
        bin_counts, 1
    )

    # Compute calibration errors per bin
    abs_errors = np.abs(bin_probs - bin_true)

    # Expected Calibration Error (weighted average of absolute errors)
    ece = np.sum(bin_counts / len(y_true) * abs_errors)

    # Maximum Calibration Error
    mce = np.max(abs_errors)

    # Brier score - quadratic scoring rule for probabilistic predictions
    brier = brier_score_loss(y_true, y_prob)

    # Discrimination preservation (AUC)
    auc = roc_auc_score(y_true, y_prob)

    # Create detailed bin information
    bins = []
    for i in range(n_bins):
        if bin_counts[i] > 0:
            bins.append(
                {
                    "bin_index": i,
                    "bin_start": i / n_bins,
                    "bin_end": (i + 1) / n_bins,
                    "sample_count": int(bin_counts[i]),
                    "mean_predicted": float(bin_probs[i]),
                    "mean_true": float(bin_true[i]),
                    "calibration_error": float(abs_errors[i]),
                }
            )

    # Compile all metrics
    metrics = {
        "expected_calibration_error": float(ece),
        "maximum_calibration_error": float(mce),
        "brier_score": float(brier),
        "auc_roc": float(auc),
        "reliability_diagram": {
            "true_probs": prob_true.tolist(),
            "pred_probs": prob_pred.tolist(),
        },
        "bin_statistics": {
            "bin_counts": bin_counts.tolist(),
            "bin_predicted_probs": bin_probs.tolist(),
            "bin_true_probs": bin_true.tolist(),
            "calibration_errors": abs_errors.tolist(),
            "detailed_bins": bins,
        },
        "num_samples": len(y_true),
        "num_bins": n_bins,
    }

    return metrics


def compute_multiclass_calibration_metrics(
    y_true: np.ndarray, y_prob_matrix: np.ndarray, n_bins: int = 10, config: Optional['CalibrationConfig'] = None
) -> Dict[str, Any]:
    """Compute calibration metrics for multi-class scenario.

    Args:
        y_true: Ground truth class labels
        y_prob_matrix: Matrix of prediction probabilities (samples × classes)
        n_bins: Number of bins for calibration curve
        config: Configuration object (optional, created from environment if not provided)

    Returns:
        dict: Dictionary containing calibration metrics
    """
    config = config or CalibrationConfig.from_env()
    n_classes = y_prob_matrix.shape[1]

    # Convert y_true to one-hot encoding
    y_true_onehot = np.zeros((len(y_true), n_classes))
    for i in range(len(y_true)):
        class_idx = int(y_true[i])
        if 0 <= class_idx < n_classes:
            y_true_onehot[i, class_idx] = 1

    # Per-class metrics
    class_metrics = []
    for i in range(n_classes):
        class_name = config.multiclass_categories[i]
        logger.info(f"Computing calibration metrics for class {class_name}")
        metrics = compute_calibration_metrics(
            y_true_onehot[:, i], y_prob_matrix[:, i], n_bins
        )
        class_metrics.append(metrics)

    # Multi-class brier score
    multiclass_brier = 0
    for i in range(len(y_true)):
        true_class = int(y_true[i])
        for j in range(n_classes):
            if j == true_class:
                multiclass_brier += (1 - y_prob_matrix[i, j]) ** 2
            else:
                multiclass_brier += y_prob_matrix[i, j] ** 2
    multiclass_brier /= len(y_true)

    # Aggregate metrics
    macro_ece = np.mean([m["expected_calibration_error"] for m in class_metrics])
    macro_mce = np.mean([m["maximum_calibration_error"] for m in class_metrics])
    max_mce = np.max([m["maximum_calibration_error"] for m in class_metrics])

    metrics = {
        "multiclass_brier_score": float(multiclass_brier),
        "macro_expected_calibration_error": float(macro_ece),
        "macro_maximum_calibration_error": float(macro_mce),
        "maximum_calibration_error": float(max_mce),
        "per_class_metrics": [
            {
                "class_index": i,
                "class_name": config.multiclass_categories[i],
                "metrics": class_metrics[i],
            }
            for i in range(n_classes)
        ],
        "num_samples": len(y_true),
        "num_bins": n_bins,
        "num_classes": n_classes,
    }

    return metrics


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob_uncalibrated: np.ndarray,
    y_prob_calibrated: np.ndarray,
    n_bins: int = 10,
    config=None,
) -> str:
    """Create reliability diagram comparing uncalibrated and calibrated probabilities.

    Args:
        y_true: Ground truth binary labels (0/1)
        y_prob_uncalibrated: Uncalibrated prediction probabilities
        y_prob_calibrated: Calibrated prediction probabilities
        n_bins: Number of bins for calibration curve
        config: Configuration object (optional, created from environment if not provided)

    Returns:
        str: Path to the saved figure
    """
    config = config or CalibrationConfig.from_env()
    fig = plt.figure(figsize=(10, 8))

    # Plot calibration curves
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    # Plot uncalibrated curve
    prob_true_uncal, prob_pred_uncal = calibration_curve(
        y_true, y_prob_uncalibrated, n_bins=n_bins
    )
    ax1.plot(prob_pred_uncal, prob_true_uncal, "s-", label="Uncalibrated")

    # Plot calibrated curve
    prob_true_cal, prob_pred_cal = calibration_curve(
        y_true, y_prob_calibrated, n_bins=n_bins
    )
    ax1.plot(prob_pred_cal, prob_true_cal, "s-", label="Calibrated")

    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Fraction of positives")
    ax1.set_title("Calibration Curve (Reliability Diagram)")
    ax1.legend(loc="lower right")

    # Plot histogram of predictions
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax2.hist(
        y_prob_uncalibrated,
        range=(0, 1),
        bins=n_bins,
        label="Uncalibrated",
        alpha=0.5,
        edgecolor="k",
    )
    ax2.hist(
        y_prob_calibrated,
        range=(0, 1),
        bins=n_bins,
        label="Calibrated",
        alpha=0.5,
        edgecolor="r",
    )
    ax2.set_xlabel("Mean predicted probability")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center")

    plt.tight_layout()

    # Save figure
    figure_path = os.path.join(config.output_metrics_path, "reliability_diagram.png")
    plt.savefig(figure_path)
    plt.close(fig)

    return figure_path


def plot_multiclass_reliability_diagram(
    y_true, y_prob_uncalibrated, y_prob_calibrated, n_bins=10, config=None
):
    """Create reliability diagrams for multi-class case, one plot per class.

    Args:
        y_true: Ground truth class labels
        y_prob_uncalibrated: Matrix of uncalibrated probabilities (samples × classes)
        y_prob_calibrated: Matrix of calibrated probabilities (samples × classes)
        n_bins: Number of bins for calibration curve
        config: Configuration object (optional, created from environment if not provided)

    Returns:
        str: Path to the saved figure
    """
    config = config or CalibrationConfig.from_env()
    n_classes = y_prob_uncalibrated.shape[1]

    # Create a plot grid based on number of classes
    n_cols = min(3, n_classes)
    n_rows = (n_classes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))

    # Convert to one-hot encoding
    y_true_onehot = np.zeros((len(y_true), n_classes))
    for i in range(len(y_true)):
        class_idx = int(y_true[i])
        if 0 <= class_idx < n_classes:
            y_true_onehot[i, class_idx] = 1

    # For each class
    for i in range(n_classes):
        class_name = config.multiclass_categories[i]
        logger.info(f"Creating reliability diagram for class {class_name}")

        # Get appropriate axis
        if n_rows == 1 and n_cols == 1:
            ax = axes
        elif n_rows == 1:
            ax = axes[i % n_cols]
        elif n_cols == 1:
            ax = axes[i % n_rows]
        else:
            ax = axes[i // n_cols, i % n_cols]

        # Plot calibration curve for this class
        ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

        prob_true_uncal, prob_pred_uncal = calibration_curve(
            y_true_onehot[:, i], y_prob_uncalibrated[:, i], n_bins=n_bins
        )
        ax.plot(prob_pred_uncal, prob_true_uncal, "s-", label="Uncalibrated")

        prob_true_cal, prob_pred_cal = calibration_curve(
            y_true_onehot[:, i], y_prob_calibrated[:, i], n_bins=n_bins
        )
        ax.plot(prob_pred_cal, prob_true_cal, "s-", label="Calibrated")

        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(f"Calibration Curve for {class_name}")
        ax.legend(loc="lower right")

    # Hide empty subplots
    for i in range(n_classes, n_rows * n_cols):
        if n_rows == 1 and n_cols == 1:
            pass  # Single plot, nothing to hide
        elif n_rows == 1:
            axes[i].axis("off")
        elif n_cols == 1:
            axes[i].axis("off")
        else:
            axes[i // n_cols, i % n_cols].axis("off")

    plt.tight_layout()
    figure_path = os.path.join(
        config.output_metrics_path, "multiclass_reliability_diagram.png"
    )
    plt.savefig(figure_path)
    plt.close(fig)

    return figure_path


def main(
    input_paths: dict,
    output_paths: dict,
    environ_vars: dict,
    job_args: argparse.Namespace = None,
) -> dict:
    """Main entry point for the calibration script.

    Args:
        input_paths: Dictionary of input paths with logical names
        output_paths: Dictionary of output paths with logical names
        environ_vars: Dictionary of environment variables
        job_args: Command line arguments (optional)

    Returns:
        Dictionary with metrics and results
    """
    try:
        # Parse multiclass categories from environment variable
        multiclass_categories = None
        multiclass_cats_str = environ_vars.get("MULTICLASS_CATEGORIES")
        if multiclass_cats_str:
            try:
                import ast
                multiclass_categories = ast.literal_eval(multiclass_cats_str)
            except (ValueError, SyntaxError):
                # Fallback to simple comma-separated parsing
                multiclass_categories = [cat.strip() for cat in multiclass_cats_str.split(",")]

        # Create config from environment variables and input/output paths
        config = CalibrationConfig(
            input_data_path=input_paths.get("evaluation_data"),
            output_calibration_path=output_paths.get("calibration_output"),
            output_metrics_path=output_paths.get("metrics_output"),
            output_calibrated_data_path=output_paths.get("calibrated_data"),
            calibration_method=environ_vars.get("CALIBRATION_METHOD", "gam"),
            label_field=environ_vars.get("LABEL_FIELD", "label"),
            score_field=environ_vars.get("SCORE_FIELD", "prob_class_1"),
            is_binary=environ_vars.get("IS_BINARY", "True").lower() == "true",
            monotonic_constraint=environ_vars.get(
                "MONOTONIC_CONSTRAINT", "True"
            ).lower()
            == "true",
            gam_splines=int(environ_vars.get("GAM_SPLINES", "10")),
            error_threshold=float(environ_vars.get("ERROR_THRESHOLD", "0.05")),
            num_classes=int(environ_vars.get("NUM_CLASSES", "2")),
            score_field_prefix=environ_vars.get(
                "SCORE_FIELD_PREFIX", "prob_class_"
            ),
            multiclass_categories=multiclass_categories,
        )

        logger.info("Starting model calibration")
        logger.info(
            f"Running in {'binary' if config.is_binary else 'multi-class'} mode"
        )

        # Create output directories
        create_directories(config)

        results = {}

        # Get job_type from command line arguments if available
        job_type = "calibration"  # default
        if job_args and hasattr(job_args, "job_type"):
            job_type = job_args.job_type
            logger.info(f"Using job_type from command line: {job_type}")

        if config.is_binary:
            # Binary classification workflow
            # Load data and extract features and target based on job_type
            df, y_true, y_prob_uncalibrated, _ = load_and_prepare_data(config, job_type)

            # Select and train calibration model
            if config.calibration_method == "gam":
                if not HAS_PYGAM:
                    logger.warning("pygam not installed, falling back to Platt scaling")
                    calibrator = train_platt_scaling(
                        y_prob_uncalibrated, y_true, config
                    )
                else:
                    calibrator = train_gam_calibration(
                        y_prob_uncalibrated, y_true, config
                    )
            elif config.calibration_method == "isotonic":
                calibrator = train_isotonic_calibration(
                    y_prob_uncalibrated, y_true, config
                )
            elif config.calibration_method == "platt":
                calibrator = train_platt_scaling(y_prob_uncalibrated, y_true, config)
            else:
                raise ValueError(
                    f"Unknown calibration method: {config.calibration_method}"
                )

            # Apply calibration to get calibrated probabilities
            if isinstance(calibrator, IsotonicRegression):
                y_prob_calibrated = calibrator.transform(y_prob_uncalibrated)
            elif isinstance(calibrator, LogisticRegression):
                y_prob_calibrated = calibrator.predict_proba(
                    y_prob_uncalibrated.reshape(-1, 1)
                )[:, 1]
            else:  # GAM
                y_prob_calibrated = calibrator.predict_proba(
                    y_prob_uncalibrated.reshape(-1, 1)
                )

            # Compute calibration metrics for before and after
            uncalibrated_metrics = compute_calibration_metrics(
                y_true, y_prob_uncalibrated
            )
            calibrated_metrics = compute_calibration_metrics(y_true, y_prob_calibrated)

            # Create visualization
            plot_path = plot_reliability_diagram(
                y_true, y_prob_uncalibrated, y_prob_calibrated, config=config
            )

            # Create comprehensive metrics report
            metrics_report = {
                "mode": "binary",
                "calibration_method": config.calibration_method,
                "uncalibrated": uncalibrated_metrics,
                "calibrated": calibrated_metrics,
                "improvement": {
                    "ece_reduction": uncalibrated_metrics["expected_calibration_error"]
                    - calibrated_metrics["expected_calibration_error"],
                    "mce_reduction": uncalibrated_metrics["maximum_calibration_error"]
                    - calibrated_metrics["maximum_calibration_error"],
                    "brier_reduction": uncalibrated_metrics["brier_score"]
                    - calibrated_metrics["brier_score"],
                    "auc_change": calibrated_metrics["auc_roc"]
                    - uncalibrated_metrics["auc_roc"],
                },
                "visualization_paths": {"reliability_diagram": plot_path},
                "config": {
                    "label_field": config.label_field,
                    "score_field": config.score_field,
                    "monotonic_constraint": config.monotonic_constraint,
                    "gam_splines": config.gam_splines,
                    "error_threshold": config.error_threshold,
                    "is_binary": config.is_binary,
                },
            }

            # Save metrics report
            metrics_path = os.path.join(
                config.output_metrics_path, "calibration_metrics.json"
            )
            with open(metrics_path, "w") as f:
                json.dump(metrics_report, f, indent=2)

            # Save calibrator model
            calibrator_path = os.path.join(
                config.output_calibration_path, "calibration_model.pkl"
            )
            with open(calibrator_path, "wb") as f:
                pkl.dump(calibrator, f)

            # Add calibrated scores to dataframe and save
            df["calibrated_" + config.score_field] = y_prob_calibrated
            output_path = os.path.join(
                config.output_calibrated_data_path, "calibrated_data.csv"
            )
            df.to_csv(output_path, index=False)

            # Write summary
            summary = {
                "status": "success",
                "mode": "binary",
                "calibration_method": config.calibration_method,
                "uncalibrated_ece": uncalibrated_metrics["expected_calibration_error"],
                "calibrated_ece": calibrated_metrics["expected_calibration_error"],
                "improvement_percentage": (
                    1
                    - calibrated_metrics["expected_calibration_error"]
                    / max(uncalibrated_metrics["expected_calibration_error"], 1e-10)
                )
                * 100,
                "output_files": {
                    "metrics": metrics_path,
                    "calibrator": calibrator_path,
                    "calibrated_data": output_path,
                },
            }

            summary_path = os.path.join(
                config.output_calibration_path, "calibration_summary.json"
            )
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

            # Check if calibration improved by error threshold
            if summary["improvement_percentage"] < 0:
                logger.warning(
                    "Calibration did not improve expected calibration error!"
                )
            elif summary["improvement_percentage"] < 5:
                logger.warning(
                    "Calibration only marginally improved expected calibration error"
                )

            logger.info(
                f"Binary calibration complete. ECE reduced from {uncalibrated_metrics['expected_calibration_error']:.4f} to {calibrated_metrics['expected_calibration_error']:.4f}"
            )

        else:
            # Multi-class classification workflow
            # Load data with all probability columns based on job_type
            df, y_true, _, y_prob_matrix = load_and_prepare_data(config, job_type)

            # Train calibration models for each class
            logger.info(
                f"Training {config.calibration_method} calibration for {config.num_classes} classes"
            )
            calibrators = train_multiclass_calibration(
                y_prob_matrix, y_true, config.calibration_method, config
            )

            # Apply calibration to get calibrated probabilities
            y_prob_calibrated = apply_multiclass_calibration(
                y_prob_matrix, calibrators, config
            )

            # Compute metrics
            uncalibrated_metrics = compute_multiclass_calibration_metrics(
                y_true, y_prob_matrix, config=config
            )
            calibrated_metrics = compute_multiclass_calibration_metrics(
                y_true, y_prob_calibrated, config=config
            )

            # Create visualizations
            plot_path = plot_multiclass_reliability_diagram(
                y_true, y_prob_matrix, y_prob_calibrated, config=config
            )

            # Create metrics report
            metrics_report = {
                "mode": "multi-class",
                "calibration_method": config.calibration_method,
                "num_classes": config.num_classes,
                "class_names": config.multiclass_categories,
                "uncalibrated": uncalibrated_metrics,
                "calibrated": calibrated_metrics,
                "improvement": {
                    "macro_ece_reduction": uncalibrated_metrics[
                        "macro_expected_calibration_error"
                    ]
                    - calibrated_metrics["macro_expected_calibration_error"],
                    "multiclass_brier_reduction": uncalibrated_metrics[
                        "multiclass_brier_score"
                    ]
                    - calibrated_metrics["multiclass_brier_score"],
                },
                "visualization_paths": {"reliability_diagram": plot_path},
                "config": {
                    "label_field": config.label_field,
                    "score_field_prefix": config.score_field_prefix,
                    "num_classes": config.num_classes,
                    "class_names": config.multiclass_categories,
                    "monotonic_constraint": config.monotonic_constraint,
                    "gam_splines": config.gam_splines,
                    "error_threshold": config.error_threshold,
                    "is_binary": config.is_binary,
                },
            }

            # Save metrics report
            metrics_path = os.path.join(
                config.output_metrics_path, "calibration_metrics.json"
            )
            with open(metrics_path, "w") as f:
                json.dump(metrics_report, f, indent=2)

            # Save calibrator models
            calibrator_dir = os.path.join(
                config.output_calibration_path, "calibration_models"
            )
            os.makedirs(calibrator_dir, exist_ok=True)

            calibrator_paths = {}
            for i, calibrator in enumerate(calibrators):
                class_name = config.multiclass_categories[i]
                calibrator_path = os.path.join(
                    calibrator_dir, f"calibration_model_class_{class_name}.pkl"
                )
                with open(calibrator_path, "wb") as f:
                    pkl.dump(calibrator, f)
                calibrator_paths[f"class_{class_name}"] = calibrator_path

            # Add calibrated scores to dataframe and save
            for i in range(config.num_classes):
                class_name = config.multiclass_categories[i]
                col_name = f"{config.score_field_prefix}{class_name}"
                df[f"calibrated_{col_name}"] = y_prob_calibrated[:, i]

            output_path = os.path.join(
                config.output_calibrated_data_path, "calibrated_data.csv"
            )
            df.to_csv(output_path, index=False)

            # Write summary
            summary = {
                "status": "success",
                "mode": "multi-class",
                "num_classes": config.num_classes,
                "class_names": config.multiclass_categories,
                "calibration_method": config.calibration_method,
                "uncalibrated_macro_ece": uncalibrated_metrics[
                    "macro_expected_calibration_error"
                ],
                "calibrated_macro_ece": calibrated_metrics[
                    "macro_expected_calibration_error"
                ],
                "improvement_percentage": (
                    1
                    - calibrated_metrics["macro_expected_calibration_error"]
                    / max(
                        uncalibrated_metrics["macro_expected_calibration_error"], 1e-10
                    )
                )
                * 100,
                "output_files": {
                    "metrics": metrics_path,
                    "calibrators": calibrator_paths,
                    "calibrated_data": output_path,
                },
            }

            summary_path = os.path.join(
                config.output_calibration_path, "calibration_summary.json"
            )
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

            # Check if calibration improved by error threshold
            if summary["improvement_percentage"] < 0:
                logger.warning(
                    "Calibration did not improve expected calibration error!"
                )
            elif summary["improvement_percentage"] < 5:
                logger.warning(
                    "Calibration only marginally improved expected calibration error"
                )

            logger.info(
                f"Multi-class calibration complete. Macro ECE reduced from "
                + f"{uncalibrated_metrics['macro_expected_calibration_error']:.4f} to "
                + f"{calibrated_metrics['macro_expected_calibration_error']:.4f}"
            )

        logger.info(
            f"All outputs saved to: {config.output_calibration_path}, {config.output_metrics_path}, and {config.output_calibrated_data_path}"
        )

        # Return results dictionary as promised by function signature
        return {
            "status": "success",
            "mode": "binary" if config.is_binary else "multi-class",
            "calibration_method": config.calibration_method,
            "metrics_report": metrics_report,
            "summary": summary
        }

    except Exception as e:
        logger.error(f"Error in model calibration: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Model Calibration Script for SageMaker Processing"
    )
    parser.add_argument(
        "--job_type",
        type=str,
        default="calibration",
        help="Job type - one of: training, calibration, validation, testing",
    )
    args = parser.parse_args()

    logger.info(f"Starting model calibration with job_type: {args.job_type}")

    # Define standard SageMaker paths
    INPUT_DATA_PATH = "/opt/ml/processing/input/eval_data"
    OUTPUT_CALIBRATION_PATH = "/opt/ml/processing/output/calibration"
    OUTPUT_METRICS_PATH = "/opt/ml/processing/output/metrics"
    OUTPUT_CALIBRATED_DATA_PATH = "/opt/ml/processing/output/calibrated_data"

    # Parse environment variables
    environ_vars = {
        "CALIBRATION_METHOD": os.environ.get("CALIBRATION_METHOD", "gam"),
        "LABEL_FIELD": os.environ.get("LABEL_FIELD", "label"),
        "SCORE_FIELD": os.environ.get("SCORE_FIELD", "prob_class_1"),
        "IS_BINARY": os.environ.get("IS_BINARY", "True"),
        "MONOTONIC_CONSTRAINT": os.environ.get("MONOTONIC_CONSTRAINT", "True"),
        "GAM_SPLINES": os.environ.get("GAM_SPLINES", "10"),
        "ERROR_THRESHOLD": os.environ.get("ERROR_THRESHOLD", "0.05"),
        "NUM_CLASSES": os.environ.get("NUM_CLASSES", "2"),
        "SCORE_FIELD_PREFIX": os.environ.get("SCORE_FIELD_PREFIX", "prob_class_"),
        "MULTICLASS_CATEGORIES": os.environ.get("MULTICLASS_CATEGORIES"),
    }

    # Set up input and output paths
    input_paths = {"evaluation_data": INPUT_DATA_PATH}

    output_paths = {
        "calibration_output": OUTPUT_CALIBRATION_PATH,
        "metrics_output": OUTPUT_METRICS_PATH,
        "calibrated_data": OUTPUT_CALIBRATED_DATA_PATH,
    }

    # Call the main function with parsed arguments
    try:
        main(input_paths, output_paths, environ_vars, args)
        logger.info("Calibration completed successfully")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Calibration failed: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

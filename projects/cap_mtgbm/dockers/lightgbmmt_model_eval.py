#!/usr/bin/env python
"""
LightGBMMT Multi-Task Model Evaluation Script

Evaluates trained LightGBMMT models on evaluation datasets.
Generates per-task and aggregate metrics, predictions, and visualizations.
"""

import os
import sys
from subprocess import check_call
import logging

# ============================================================================
# PACKAGE INSTALLATION CONFIGURATION
# ============================================================================

USE_SECURE_PYPI = os.environ.get("USE_SECURE_PYPI", "true").lower() == "true"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_secure_pypi_access_token() -> str:
    """Get CodeArtifact access token for secure PyPI."""
    import boto3

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
    """Install packages from standard public PyPI."""
    logger.info(f"Installing {len(packages)} packages from public PyPI")
    try:
        check_call([sys.executable, "-m", "pip", "install", *packages])
        logger.info("✓ Successfully installed packages from public PyPI")
    except Exception as e:
        logger.error(f"✗ Failed to install packages from public PyPI: {e}")
        raise


def install_packages_from_secure_pypi(packages: list) -> None:
    """Install packages from secure CodeArtifact PyPI."""
    logger.info(f"Installing {len(packages)} packages from secure PyPI")
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
    """Install packages from PyPI source based on configuration."""
    logger.info("=" * 70)
    logger.info("PACKAGE INSTALLATION")
    logger.info("=" * 70)
    logger.info(f"PyPI Source: {'SECURE (CodeArtifact)' if use_secure else 'PUBLIC'}")
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

required_packages = [
    "pyarrow>=10.0.0",
    "lightgbm>=3.3.0",
]

install_packages(required_packages)

print("***********************Package Installation Complete*********************")

# Now import packages after installation
import json
import argparse
import pandas as pd
import numpy as np
import pickle as pkl
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
)
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import lightgbm as lgbm
import matplotlib.pyplot as plt
import time
import tarfile
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union

# Embedded processor classes to remove external dependencies


class RiskTableMappingProcessor:
    """
    A processor that performs risk-table-based mapping on a specified categorical variable.
    The 'process' method (called via __call__) handles single values.
    The 'transform' method handles pandas Series or DataFrames.
    """

    def __init__(
        self,
        column_name: str,
        label_name: str,
        smooth_factor: float = 0.0,
        count_threshold: int = 0,
        risk_tables: Optional[Dict] = None,
    ):
        """
        Initialize RiskTableMappingProcessor.

        Args:
            column_name: Name of the categorical column to be binned.
            label_name: Name of label/target variable (expected to be binary 0 or 1).
            smooth_factor: Smoothing factor for risk calculation (0 to 1).
            count_threshold: Minimum count for considering a category's calculated risk.
            risk_tables: Optional pre-computed risk tables.
        """
        self.processor_name = "risk_table_mapping_processor"
        self.function_name_list = ["process", "transform", "fit"]

        if not isinstance(column_name, str) or not column_name:
            raise ValueError("column_name must be a non-empty string.")
        self.column_name = column_name
        self.label_name = label_name
        self.smooth_factor = smooth_factor
        self.count_threshold = count_threshold

        self.is_fitted = False
        if risk_tables:
            self._validate_risk_tables(risk_tables)
            self.risk_tables = risk_tables
            self.is_fitted = True
        else:
            self.risk_tables = {}

    def get_name(self) -> str:
        return self.processor_name

    def _validate_risk_tables(self, risk_tables: Dict) -> None:
        if not isinstance(risk_tables, dict):
            raise ValueError("Risk tables must be a dictionary.")
        if "bins" not in risk_tables or "default_bin" not in risk_tables:
            raise ValueError("Risk tables must contain 'bins' and 'default_bin' keys.")
        if not isinstance(risk_tables["bins"], dict):
            raise ValueError("Risk tables 'bins' must be a dictionary.")
        if not isinstance(
            risk_tables["default_bin"], (int, float, np.floating, np.integer)
        ):
            raise ValueError(
                f"Risk tables 'default_bin' must be a number, got {type(risk_tables['default_bin'])}."
            )

    def set_risk_tables(self, risk_tables: Dict) -> None:
        self._validate_risk_tables(risk_tables)
        self.risk_tables = risk_tables
        self.is_fitted = True

    def process(self, input_value: Any) -> float:
        """
        Process a single input value (for the configured 'column_name'),
        mapping it to its binned risk value.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "RiskTableMappingProcessor must be fitted or initialized with risk tables before processing."
            )
        str_value = str(input_value)
        return self.risk_tables["bins"].get(str_value, self.risk_tables["default_bin"])

    def transform(
        self, data: Union[pd.DataFrame, pd.Series, Any]
    ) -> Union[pd.DataFrame, pd.Series, float]:
        """
        Transform data using the computed risk tables.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "RiskTableMappingProcessor must be fitted or initialized with risk tables before transforming."
            )

        if isinstance(data, pd.DataFrame):
            if self.column_name not in data.columns:
                raise ValueError(
                    f"Column '{self.column_name}' not found in input DataFrame for transform operation."
                )
            output_data = data.copy()
            output_data[self.column_name] = (
                data[self.column_name]
                .astype(str)
                .map(self.risk_tables["bins"])
                .fillna(self.risk_tables["default_bin"])
            )
            return output_data
        elif isinstance(data, pd.Series):
            return (
                data.astype(str)
                .map(self.risk_tables["bins"])
                .fillna(self.risk_tables["default_bin"])
            )
        else:
            return self.process(data)

    def get_risk_tables(self) -> Dict:
        if not self.is_fitted:
            raise RuntimeError(
                "RiskTableMappingProcessor has not been fitted or initialized with risk tables."
            )
        return self.risk_tables


class NumericalVariableImputationProcessor:
    """
    A processor that performs imputation on numerical variables using predefined or computed values.
    Supports mean, median, and mode imputation strategies.
    """

    def __init__(
        self,
        variables: Optional[List[str]] = None,
        imputation_dict: Optional[Dict[str, Union[int, float]]] = None,
        strategy: str = "mean",
    ):
        self.processor_name = "numerical_variable_imputation_processor"
        self.function_name_list = ["fit", "process", "transform"]

        self.variables = variables
        self.strategy = strategy
        self.is_fitted = False

        if imputation_dict:
            self._validate_imputation_dict(imputation_dict)
            self.imputation_dict = imputation_dict
            self.is_fitted = True
        else:
            self.imputation_dict = None

    def get_name(self) -> str:
        return self.processor_name

    def __call__(self, input_data):
        return self.process(input_data)

    def _validate_imputation_dict(self, imputation_dict: Dict[str, Any]) -> None:
        if not isinstance(imputation_dict, dict):
            raise ValueError("imputation_dict must be a dictionary")
        if not imputation_dict:
            raise ValueError("imputation_dict cannot be empty")
        for k, v in imputation_dict.items():
            if not isinstance(k, str):
                raise ValueError(f"All keys must be strings, got {type(k)} for key {k}")
            if not isinstance(v, (int, float, np.number)):
                raise ValueError(
                    f"All values must be numeric, got {type(v)} for key {k}"
                )

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_fitted:
            raise RuntimeError(
                "Processor is not fitted. Call 'fit' with appropriate arguments before using this method."
            )

        output_data = input_data.copy()
        for var, value in input_data.items():
            if var in self.imputation_dict and pd.isna(value):
                output_data[var] = self.imputation_dict[var]
        return output_data

    def transform(
        self, X: Union[pd.DataFrame, pd.Series]
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Transform input data by imputing missing values.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Processor is not fitted. Call 'fit' with appropriate arguments before using this method."
            )

        # Handle Series input
        if isinstance(X, pd.Series):
            if X.name not in self.imputation_dict:
                raise ValueError(f"No imputation value found for series name: {X.name}")
            return X.fillna(self.imputation_dict[X.name])

        # Handle DataFrame input
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be pandas Series or DataFrame")

        # Make a copy to avoid modifying the input
        df = X.copy()

        # Apply imputation only to variables in imputation_dict and only to NaN values
        for var, impute_value in self.imputation_dict.items():
            if var in df.columns:
                # Create mask for NaN values
                nan_mask = df[var].isna()
                # Only replace NaN values
                df.loc[nan_mask, var] = impute_value

        return df

    def get_params(self) -> Dict[str, Any]:
        return {
            "variables": self.variables,
            "imputation_dict": self.imputation_dict,
            "strategy": self.strategy,
        }


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Container path constants - aligned with script contract
CONTAINER_PATHS = {
    "MODEL_DIR": "/opt/ml/processing/input/model",
    "EVAL_DATA_DIR": "/opt/ml/processing/input/eval_data",
    "OUTPUT_EVAL_DIR": "/opt/ml/processing/output/eval",
    "OUTPUT_METRICS_DIR": "/opt/ml/processing/output/metrics",
}


# ============================================================================
# FILE I/O HELPER FUNCTIONS WITH FORMAT PRESERVATION
# ============================================================================


def _detect_file_format(file_path: Path) -> str:
    """Detect the format of a data file based on its extension."""
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    elif suffix == ".tsv":
        return "tsv"
    elif suffix == ".parquet":
        return "parquet"
    else:
        raise RuntimeError(f"Unsupported file format: {suffix}")


def load_dataframe_with_format(file_path: Path) -> Tuple[pd.DataFrame, str]:
    """Load DataFrame and detect its format."""
    detected_format = _detect_file_format(file_path)

    if detected_format == "csv":
        df = pd.read_csv(file_path)
    elif detected_format == "tsv":
        df = pd.read_csv(file_path, sep="\t")
    elif detected_format == "parquet":
        df = pd.read_parquet(file_path)
    else:
        raise RuntimeError(f"Unsupported format: {detected_format}")

    return df, detected_format


def save_dataframe_with_format(
    df: pd.DataFrame, output_path: Path, format_str: str
) -> Path:
    """Save DataFrame in specified format."""
    if format_str == "csv":
        file_path = output_path.with_suffix(".csv")
        df.to_csv(file_path, index=False)
    elif format_str == "tsv":
        file_path = output_path.with_suffix(".tsv")
        df.to_csv(file_path, sep="\t", index=False)
    elif format_str == "parquet":
        file_path = output_path.with_suffix(".parquet")
        df.to_parquet(file_path, index=False)
    else:
        raise RuntimeError(f"Unsupported output format: {format_str}")

    return file_path


def safe_extract_tar(tar: tarfile.TarFile, path: str) -> None:
    """
    Safely extract tar file, preventing path traversal attacks (zip slip).

    Validates that each member's extracted path stays within the target directory.

    Args:
        tar: Open TarFile object
        path: Target extraction directory

    Raises:
        ValueError: If a member would extract outside the target directory
    """

    def is_within_directory(directory: str, target: str) -> bool:
        """Check if target path is within directory."""
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
        return os.path.commonprefix([abs_directory, abs_target]) == abs_directory

    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not is_within_directory(path, member_path):
            raise ValueError(f"Attempted path traversal in tar file: {member.name}")

    # If all paths are safe, extract
    tar.extractall(path=path)


def decompress_model_artifacts(model_dir: str):
    """Extract model.tar.gz if it exists with path traversal protection."""
    model_tar_path = Path(model_dir) / "model.tar.gz"
    if model_tar_path.exists():
        logger.info(f"Found model.tar.gz at {model_tar_path}. Extracting...")
        with tarfile.open(model_tar_path, "r:gz") as tar:
            safe_extract_tar(tar, model_dir)
        logger.info("Extraction complete.")
    else:
        logger.info("No model.tar.gz found. Assuming artifacts are directly available.")


# ============================================================================
# MULTI-TASK LABEL PARSING
# ============================================================================


def parse_task_label_names(env_value: str) -> List[str]:
    """
    Parse TASK_LABEL_NAMES from environment variable.

    Supports:
    - Comma-separated: "isFraud,isCCfrd,isDDfrd"
    - JSON array: '["isFraud","isCCfrd","isDDfrd"]'

    Args:
        env_value: Environment variable value

    Returns:
        List of task label names
    """
    if not env_value or env_value.strip() == "":
        raise ValueError("TASK_LABEL_NAMES environment variable is empty")

    # Try JSON format first
    if env_value.strip().startswith("["):
        try:
            task_names = json.loads(env_value)
            if not isinstance(task_names, list):
                raise ValueError("JSON value must be an array")
            return [str(t).strip() for t in task_names]
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format for TASK_LABEL_NAMES: {e}")

    # Comma-separated format
    task_names = [t.strip() for t in env_value.split(",") if t.strip()]
    if not task_names:
        raise ValueError("TASK_LABEL_NAMES contains no valid task names")

    return task_names


# ============================================================================
# MODEL ARTIFACT LOADING
# ============================================================================


def load_model_artifacts(
    model_dir: str,
) -> Tuple[lgbm.Booster, Dict[str, Any], Dict[str, Any], List[str], Dict[str, Any]]:
    """
    Load trained LightGBMMT model and preprocessing artifacts.

    Returns: model, risk_tables, impute_dict, feature_columns, hyperparameters
    """
    logger.info(f"Loading model artifacts from {model_dir}")

    # Decompress if needed
    decompress_model_artifacts(model_dir)

    # Load LightGBM model
    model_file = os.path.join(model_dir, "lightgbmmt_model.txt")
    model = lgbm.Booster(model_file=model_file)
    logger.info(f"Loaded lightgbmmt_model.txt")

    # Load preprocessing artifacts
    with open(os.path.join(model_dir, "risk_table_map.pkl"), "rb") as f:
        risk_tables = pkl.load(f)
    logger.info("Loaded risk_table_map.pkl")

    with open(os.path.join(model_dir, "impute_dict.pkl"), "rb") as f:
        impute_dict = pkl.load(f)
    logger.info("Loaded impute_dict.pkl")

    with open(os.path.join(model_dir, "feature_columns.txt"), "r") as f:
        feature_columns = [
            line.strip().split(",")[1] for line in f if not line.startswith("#")
        ]
    logger.info(f"Loaded feature_columns.txt: {feature_columns}")

    with open(os.path.join(model_dir, "hyperparameters.json"), "r") as f:
        hyperparams = json.load(f)
    logger.info("Loaded hyperparameters.json")

    return model, risk_tables, impute_dict, feature_columns, hyperparams


# ============================================================================
# DATA PREPROCESSING
# ============================================================================


def preprocess_eval_data(
    df: pd.DataFrame,
    feature_columns: List[str],
    risk_tables: Dict[str, Any],
    impute_dict: Dict[str, Any],
) -> pd.DataFrame:
    """Apply risk table mapping and numerical imputation to evaluation data."""
    result_df = df.copy()

    available_features = [col for col in feature_columns if col in df.columns]
    logger.info(
        f"Found {len(available_features)} out of {len(feature_columns)} expected features"
    )

    # Risk table mapping
    logger.info("Applying risk table mapping")
    for feature, risk_table in risk_tables.items():
        if feature in available_features:
            proc = RiskTableMappingProcessor(
                column_name=feature, label_name="label", risk_tables=risk_table
            )
            result_df[feature] = proc.transform(df[feature])

    # Numerical imputation
    logger.info("Applying numerical imputation")
    feature_df = result_df[available_features].copy()
    imputer = NumericalVariableImputationProcessor(imputation_dict=impute_dict)
    imputed_df = imputer.transform(feature_df)
    for col in available_features:
        if col in imputed_df:
            result_df[col] = imputed_df[col]

    # Ensure numeric
    result_df[available_features] = (
        result_df[available_features].apply(pd.to_numeric, errors="coerce").fillna(0)
    )

    logger.info(f"Preprocessed data shape: {result_df.shape}")
    return result_df


def load_eval_data(eval_data_dir: str) -> Tuple[pd.DataFrame, str]:
    """Load evaluation data from directory."""
    logger.info(f"Loading eval data from {eval_data_dir}")
    eval_files = sorted(
        [
            f
            for f in Path(eval_data_dir).glob("**/*")
            if f.suffix in [".csv", ".tsv", ".parquet"]
        ]
    )
    if not eval_files:
        raise RuntimeError("No eval data file found in eval_data input.")

    eval_file = eval_files[0]
    logger.info(f"Using eval data file: {eval_file}")

    df, input_format = load_dataframe_with_format(eval_file)
    logger.info(f"Loaded eval data shape: {df.shape}, format: {input_format}")
    return df, input_format


def get_id_column(df: pd.DataFrame, id_field: str) -> str:
    """Determine ID column."""
    id_col = id_field if id_field in df.columns else df.columns[0]
    logger.info(f"Using id_col: {id_col}")
    return id_col


# ============================================================================
# MULTI-TASK INFERENCE
# ============================================================================


def predict_multitask(
    model: lgbm.Booster, df: pd.DataFrame, feature_columns: List[str]
) -> np.ndarray:
    """
    Generate multi-task predictions.

    Returns: np.ndarray of shape (n_samples, n_tasks) with probabilities
    """
    X = df[feature_columns]
    predictions = model.predict(X)

    # LightGBM multi-task output is already (n_samples, n_tasks)
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)

    logger.info(f"Generated predictions shape: {predictions.shape}")
    return predictions


# ============================================================================
# MULTI-TASK METRICS
# ============================================================================


def compute_multitask_metrics(
    y_true_tasks: Dict[int, np.ndarray],
    y_pred_tasks: np.ndarray,
    task_names: List[str],
) -> Dict[str, Any]:
    """Compute per-task and aggregate metrics."""
    logger.info("Computing multi-task metrics")
    metrics = {}
    auc_rocs = []
    aps = []
    f1s = []

    for i, task_name in enumerate(task_names):
        y_true = y_true_tasks[i]
        y_pred = y_pred_tasks[:, i]

        try:
            auc_roc = roc_auc_score(y_true, y_pred)
            ap = average_precision_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred > 0.5)

            metrics[f"task_{i}_{task_name}"] = {
                "auc_roc": float(auc_roc),
                "average_precision": float(ap),
                "f1_score": float(f1),
            }

            auc_rocs.append(auc_roc)
            aps.append(ap)
            f1s.append(f1)

            logger.info(
                f"Task {i} ({task_name}): AUC={auc_roc:.4f}, AP={ap:.4f}, F1={f1:.4f}"
            )

        except ValueError as e:
            logger.warning(f"Task {i} ({task_name}): {e}")
            metrics[f"task_{i}_{task_name}"] = {
                "auc_roc": 0.5,
                "average_precision": 0.5,
                "f1_score": 0.0,
            }

    # Aggregate metrics
    if auc_rocs:
        metrics["aggregate"] = {
            "mean_auc_roc": float(np.mean(auc_rocs)),
            "median_auc_roc": float(np.median(auc_rocs)),
            "mean_average_precision": float(np.mean(aps)),
            "median_average_precision": float(np.median(aps)),
            "mean_f1_score": float(np.mean(f1s)),
            "median_f1_score": float(np.median(f1s)),
        }

        logger.info("Aggregate Metrics:")
        logger.info(f"  Mean AUC-ROC: {metrics['aggregate']['mean_auc_roc']:.4f}")
        logger.info(f"  Mean AP: {metrics['aggregate']['mean_average_precision']:.4f}")
        logger.info(f"  Mean F1: {metrics['aggregate']['mean_f1_score']:.4f}")

    return metrics


# ============================================================================
# VISUALIZATION
# ============================================================================


def plot_multitask_curves(
    y_true_tasks: Dict[int, np.ndarray],
    y_pred_tasks: np.ndarray,
    task_names: List[str],
    output_dir: str,
) -> None:
    """Generate ROC and PR curves for each task."""
    logger.info("Generating multi-task curves")
    os.makedirs(output_dir, exist_ok=True)

    for i, task_name in enumerate(task_names):
        y_true = y_true_tasks[i]
        y_pred = y_pred_tasks[:, i]

        if len(np.unique(y_true)) < 2:
            logger.warning(
                f"Task {i} ({task_name}): Only one class present, skipping plots"
            )
            continue

        try:
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred)

            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
            plt.plot([0, 1], [0, 1], "--", color="gray")
            plt.title(f"Task {i} ({task_name}) ROC Curve")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            plt.savefig(
                os.path.join(output_dir, f"task_{i}_{task_name}_roc.jpg"), dpi=150
            )
            plt.close()

            # PR Curve
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            ap = average_precision_score(y_true, y_pred)

            plt.figure()
            plt.plot(recall, precision, label=f"AP={ap:.3f}")
            plt.title(f"Task {i} ({task_name}) PR Curve")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.legend()
            plt.savefig(
                os.path.join(output_dir, f"task_{i}_{task_name}_pr.jpg"), dpi=150
            )
            plt.close()

            logger.info(f"Generated plots for task {i} ({task_name})")

        except Exception as e:
            logger.warning(f"Error plotting task {i} ({task_name}): {e}")


# ============================================================================
# OUTPUT SAVING
# ============================================================================


def save_predictions(
    ids: np.ndarray,
    y_true_tasks: Dict[int, np.ndarray],
    y_pred_tasks: np.ndarray,
    task_names: List[str],
    id_col: str,
    output_dir: str,
    input_format: str = "csv",
) -> None:
    """Save multi-task predictions preserving input format."""
    logger.info(f"Saving predictions to {output_dir} in {input_format} format")

    # Build predictions DataFrame
    pred_df = pd.DataFrame({id_col: ids})

    for i, task_name in enumerate(task_names):
        pred_df[f"{task_name}_true"] = y_true_tasks[i]
        pred_df[f"{task_name}_prob"] = y_pred_tasks[:, i]

    output_base = Path(output_dir) / "eval_predictions"
    output_path = save_dataframe_with_format(pred_df, output_base, input_format)
    logger.info(f"Saved predictions (format={input_format}): {output_path}")


def save_metrics(
    metrics: Dict[str, Union[int, float, str]], output_metrics_dir: str
) -> None:
    """Save computed metrics as JSON."""
    out_path = os.path.join(output_metrics_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {out_path}")

    # Create summary
    summary_path = os.path.join(output_metrics_dir, "metrics_summary.txt")
    with open(summary_path, "w") as f:
        f.write("MULTI-TASK METRICS SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        # Per-task metrics
        f.write("PER-TASK METRICS\n")
        f.write("-" * 50 + "\n")
        for key, value in sorted(metrics.items()):
            if key.startswith("task_") and isinstance(value, dict):
                f.write(f"\n{key}:\n")
                for metric_name, metric_value in value.items():
                    f.write(f"  {metric_name}: {metric_value:.4f}\n")

        # Aggregate metrics
        if "aggregate" in metrics:
            f.write("\nAGGREGATE METRICS\n")
            f.write("-" * 50 + "\n")
            for metric_name, metric_value in metrics["aggregate"].items():
                f.write(f"{metric_name}: {metric_value:.4f}\n")

    logger.info(f"Saved metrics summary to {summary_path}")


def create_health_check_file(output_path: str) -> str:
    """Create health check file to signal script completion."""
    with open(output_path, "w") as f:
        f.write(f"healthy: {datetime.now().isoformat()}")
    return output_path


# ============================================================================
# MAIN EVALUATION
# ============================================================================


def evaluate_model(
    model: lgbm.Booster,
    df: pd.DataFrame,
    feature_columns: List[str],
    task_names: List[str],
    id_col: str,
    output_eval_dir: str,
    output_metrics_dir: str,
    input_format: str = "csv",
) -> None:
    """Run multi-task model evaluation."""
    logger.info("Starting multi-task evaluation")

    # Extract task labels
    y_true_tasks = {}
    for i, task_name in enumerate(task_names):
        if task_name not in df.columns:
            raise ValueError(f"Task label '{task_name}' not found in data")
        y_true_tasks[i] = df[task_name].astype(int).values

    # Get IDs
    ids = df[id_col].values

    # Generate predictions
    y_pred_tasks = predict_multitask(model, df, feature_columns)

    # Compute metrics
    metrics = compute_multitask_metrics(y_true_tasks, y_pred_tasks, task_names)

    # Generate plots
    plot_multitask_curves(y_true_tasks, y_pred_tasks, task_names, output_metrics_dir)

    # Save outputs
    save_predictions(
        ids,
        y_true_tasks,
        y_pred_tasks,
        task_names,
        id_col,
        output_eval_dir,
        input_format,
    )
    save_metrics(metrics, output_metrics_dir)

    logger.info("Multi-task evaluation complete")


def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
) -> None:
    """Main entry point for LightGBMMT model evaluation."""
    # Extract paths
    model_dir = input_paths.get("model_input", input_paths.get("model_dir"))
    eval_data_dir = input_paths.get("processed_data", input_paths.get("eval_data_dir"))
    output_eval_dir = output_paths.get(
        "eval_output", output_paths.get("output_eval_dir")
    )
    output_metrics_dir = output_paths.get(
        "metrics_output", output_paths.get("output_metrics_dir")
    )

    # Extract environment variables
    id_field = environ_vars.get("ID_FIELD", "id")
    task_label_names_str = environ_vars.get("TASK_LABEL_NAMES", "")

    # Parse task label names
    task_names = parse_task_label_names(task_label_names_str)
    logger.info(f"Parsed task names: {task_names}")

    # Ensure output directories exist
    os.makedirs(output_eval_dir, exist_ok=True)
    os.makedirs(output_metrics_dir, exist_ok=True)

    logger.info("Starting LightGBMMT model evaluation")
    logger.info(f"Job type: {job_args.job_type}")

    # Load model artifacts
    model, risk_tables, impute_dict, feature_columns, hyperparams = (
        load_model_artifacts(model_dir)
    )

    # Verify task names match hyperparameters
    hp_task_names = hyperparams.get("task_label_names", [])
    if hp_task_names and hp_task_names != task_names:
        logger.warning(
            f"Environment TASK_LABEL_NAMES {task_names} differs from "
            f"hyperparameters task_label_names {hp_task_names}. "
            f"Using environment variable."
        )

    # Load and preprocess data
    df, input_format = load_eval_data(eval_data_dir)
    id_col = get_id_column(df, id_field)

    # Preprocess
    df = preprocess_eval_data(df, feature_columns, risk_tables, impute_dict)

    # Get available features
    available_features = [col for col in feature_columns if col in df.columns]
    logger.info(f"Using {len(available_features)} features for inference")

    # Evaluate
    evaluate_model(
        model,
        df,
        available_features,
        task_names,
        id_col,
        output_eval_dir,
        output_metrics_dir,
        input_format,
    )

    logger.info("LightGBMMT model evaluation complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_type", type=str, required=True)
    args = parser.parse_args()

    # Set up paths
    input_paths = {
        "model_input": CONTAINER_PATHS["MODEL_DIR"],
        "processed_data": CONTAINER_PATHS["EVAL_DATA_DIR"],
    }

    output_paths = {
        "eval_output": CONTAINER_PATHS["OUTPUT_EVAL_DIR"],
        "metrics_output": CONTAINER_PATHS["OUTPUT_METRICS_DIR"],
    }

    # Collect environment variables
    environ_vars = {
        "ID_FIELD": os.environ.get("ID_FIELD", "id"),
        "TASK_LABEL_NAMES": os.environ.get("TASK_LABEL_NAMES", ""),
    }

    try:
        main(input_paths, output_paths, environ_vars, args)

        # Signal success
        success_path = os.path.join(output_paths["metrics_output"], "_SUCCESS")
        Path(success_path).touch()
        logger.info(f"Created success marker: {success_path}")

        # Create health check
        health_path = os.path.join(output_paths["metrics_output"], "_HEALTH")
        create_health_check_file(health_path)
        logger.info(f"Created health check file: {health_path}")

        sys.exit(0)
    except Exception as e:
        logger.exception(f"Script failed with error: {e}")
        failure_path = os.path.join(
            output_paths.get("metrics_output", "/tmp"), "_FAILURE"
        )
        os.makedirs(os.path.dirname(failure_path), exist_ok=True)
        with open(failure_path, "w") as f:
            f.write(f"Error: {str(e)}")
        sys.exit(1)

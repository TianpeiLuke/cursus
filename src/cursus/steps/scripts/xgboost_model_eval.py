#!/usr/bin/env python
import os
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
import xgboost as xgb
import matplotlib.pyplot as plt
import time
import sys
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

    def fit(self, data: pd.DataFrame) -> "RiskTableMappingProcessor":
        if not isinstance(data, pd.DataFrame):
            raise TypeError("fit() requires a pandas DataFrame.")
        if self.label_name not in data.columns:
            raise ValueError(
                f"Label variable '{self.label_name}' not found in input data."
            )
        if self.column_name not in data.columns:
            raise ValueError(
                f"Column to bin '{self.column_name}' not found in input data."
            )

        filtered_data = data[data[self.label_name] != -1].dropna(
            subset=[self.label_name, self.column_name]
        )

        if filtered_data.empty:
            # Handle case with no valid data for fitting
            print(
                f"Warning: Filtered data for column '{self.column_name}' is empty during fit. "
                "Risk tables will be empty, default_bin will be 0.5 or NaN if no labels at all."
            )
            # Attempt to get a global mean if any data existed before filtering for column_name
            overall_label_mean = data[self.label_name][
                data[self.label_name] != -1
            ].mean()
            self.risk_tables = {
                "bins": {},
                "default_bin": (
                    0.5 if pd.isna(overall_label_mean) else float(overall_label_mean)
                ),
            }
            self.is_fitted = True
            return self

        default_risk = float(filtered_data[self.label_name].mean())
        smooth_samples = int(len(filtered_data) * self.smooth_factor)

        cross_tab_result = pd.crosstab(
            index=filtered_data[self.column_name].astype(str),
            columns=filtered_data[self.label_name].astype(int),
            margins=True,
            margins_name="_count_",
            dropna=False,
        )

        positive_label_col = 1
        negative_label_col = 0

        if positive_label_col not in cross_tab_result.columns:
            cross_tab_result[positive_label_col] = 0
        if negative_label_col not in cross_tab_result.columns:
            cross_tab_result[negative_label_col] = 0

        calc_df = cross_tab_result[cross_tab_result.index != "_count_"].copy()

        if calc_df.empty:
            self.risk_tables = {"bins": {}, "default_bin": default_risk}
            self.is_fitted = True
            return self

        calc_df["risk"] = calc_df.apply(
            lambda x: (
                x[positive_label_col] / (x[positive_label_col] + x[negative_label_col])
                if (x[positive_label_col] + x[negative_label_col]) > 0
                else 0.0
            ),
            axis=1,
        )

        calc_df["_category_count_"] = cross_tab_result.loc[calc_df.index, "_count_"]

        calc_df["smooth_risk"] = calc_df.apply(
            lambda x: (
                (x["_category_count_"] * x["risk"] + smooth_samples * default_risk)
                / (x["_category_count_"] + smooth_samples)
                if (
                    x["_category_count_"] >= self.count_threshold
                    and (x["_category_count_"] + smooth_samples) > 0
                )
                else default_risk
            ),
            axis=1,
        )

        self.risk_tables = {
            "bins": dict(zip(calc_df.index.astype(str), calc_df["smooth_risk"])),
            "default_bin": default_risk,
        }

        self.is_fitted = True
        return self

    def process(self, input_value: Any) -> float:
        """
        Process a single input value (for the configured 'column_name'),
        mapping it to its binned risk value.
        This method is called when the processor instance is called as a function.
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
        - If data is a DataFrame, transforms the 'column_name' Series within it.
        - If data is a Series, transforms the Series (assumed to be the target column).
        - If data is a single value, uses the 'process' method.
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
            return self.process(data)  # Consistent with __call__

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

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "NumericalVariableImputationProcessor":
        if self.imputation_dict is None:
            self.imputation_dict = {}

            if self.variables is None:
                self.variables = X.select_dtypes(include=np.number).columns.tolist()

            for var in self.variables:
                if var not in X.columns:
                    raise ValueError(f"Variable {var} not found in the input data")

                if X[var].isna().all():
                    self.imputation_dict[var] = np.nan
                    continue

                if self.strategy == "mean":
                    self.imputation_dict[var] = float(X[var].mean())
                elif self.strategy == "median":
                    self.imputation_dict[var] = float(X[var].median())
                elif self.strategy == "mode":
                    self.imputation_dict[var] = float(X[var].mode()[0])
                else:
                    raise ValueError(f"Unknown strategy: {self.strategy}")

        self._validate_imputation_dict(self.imputation_dict)
        self.is_fitted = True
        return self

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

        Args:
            X: Input DataFrame or Series

        Returns:
            Transformed DataFrame or Series with imputed values
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

import logging

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


def load_model_artifacts(model_dir: str) -> Tuple[xgb.Booster, Dict[str, Any], Dict[str, Any], List[str], Dict[str, Any]]:
    """
    Load the trained XGBoost model and all preprocessing artifacts from the specified directory.
    Returns model, risk_tables, impute_dict, feature_columns, and hyperparameters.
    """
    logger.info(f"Loading model artifacts from {model_dir}")
    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, "xgboost_model.bst"))
    logger.info("Loaded xgboost_model.bst")
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


def preprocess_eval_data(df: pd.DataFrame, feature_columns: List[str], risk_tables: Dict[str, Any], impute_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply risk table mapping and numerical imputation to the evaluation DataFrame.
    Ensures all features are numeric and columns are ordered as required by the model.
    Preserves any non-feature columns like id and label.
    """
    # Make a copy of the input dataframe to avoid modifying the original
    result_df = df.copy()
    
    # Get available feature columns (features that exist in the input data)
    available_features = [col for col in feature_columns if col in df.columns]
    logger.info(f"Found {len(available_features)} out of {len(feature_columns)} expected feature columns")
    
    # Process only feature columns
    logger.info("Starting risk table mapping for categorical features")
    for feature, risk_table in risk_tables.items():
        if feature in available_features:
            logger.info(f"Applying risk table mapping for feature: {feature}")
            proc = RiskTableMappingProcessor(
                column_name=feature, label_name="label", risk_tables=risk_table
            )
            result_df[feature] = proc.transform(df[feature])
    logger.info("Risk table mapping complete")
    
    # For numerical imputation, only process features, not all columns
    logger.info("Starting numerical imputation")
    feature_df = result_df[available_features].copy()
    imputer = NumericalVariableImputationProcessor(imputation_dict=impute_dict)
    imputed_df = imputer.transform(feature_df)
    # Update only the feature columns in the result dataframe
    for col in available_features:
        if col in imputed_df:
            result_df[col] = imputed_df[col]
    logger.info("Numerical imputation complete")
    
    # Convert feature columns to numeric, leaving other columns unchanged
    logger.info("Ensuring feature columns are numeric")
    result_df[available_features] = (
        result_df[available_features].apply(pd.to_numeric, errors="coerce").fillna(0)
    )
    
    logger.info(f"Preprocessed data shape: {result_df.shape} (preserving all original columns)")
    
    # Return the result with all original columns preserved
    return result_df


def log_metrics_summary(metrics: Dict[str, Union[int, float, str]], is_binary: bool = True) -> None:
    """
    Log a nicely formatted summary of metrics for easy visibility in logs.

    Args:
        metrics: Dictionary of metrics to log
        is_binary: Whether these are binary classification metrics
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    logger.info("=" * 80)
    logger.info(f"METRICS SUMMARY - {timestamp}")
    logger.info("=" * 80)

    # Log each metric with a consistent format
    for name, value in metrics.items():
        # Format numeric values to 4 decimal places
        if isinstance(value, (int, float)):
            formatted_value = f"{value:.4f}"
        else:
            formatted_value = str(value)

        # Add a special prefix for easy searching in logs
        logger.info(f"METRIC: {name.ljust(25)} = {formatted_value}")

    # Highlight key metrics based on task type
    logger.info("=" * 80)
    logger.info("KEY PERFORMANCE METRICS")
    logger.info("=" * 80)

    if is_binary:
        logger.info(
            f"METRIC_KEY: AUC-ROC               = {metrics.get('auc_roc', 'N/A'):.4f}"
        )
        logger.info(
            f"METRIC_KEY: Average Precision     = {metrics.get('average_precision', 'N/A'):.4f}"
        )
        logger.info(
            f"METRIC_KEY: F1 Score              = {metrics.get('f1_score', 'N/A'):.4f}"
        )
    else:
        logger.info(
            f"METRIC_KEY: Macro AUC-ROC         = {metrics.get('auc_roc_macro', 'N/A'):.4f}"
        )
        logger.info(
            f"METRIC_KEY: Micro AUC-ROC         = {metrics.get('auc_roc_micro', 'N/A'):.4f}"
        )
        ap_macro = metrics.get("average_precision_macro", "N/A")
        if isinstance(ap_macro, (int, float)):
            logger.info(f"METRIC_KEY: Macro Average Precision = {ap_macro:.4f}")
        else:
            logger.info(f"METRIC_KEY: Macro Average Precision = {ap_macro}")
        logger.info(
            f"METRIC_KEY: Macro F1              = {metrics.get('f1_score_macro', 'N/A'):.4f}"
        )
        logger.info(
            f"METRIC_KEY: Micro F1              = {metrics.get('f1_score_micro', 'N/A'):.4f}"
        )

    # Add a summary section with pass/fail criteria if defined
    logger.info("=" * 80)


def compute_metrics_binary(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Compute binary classification metrics: AUC-ROC, average precision, and F1 score.
    """
    logger.info("Computing binary classification metrics")
    y_score = y_prob[:, 1]
    metrics = {
        "auc_roc": roc_auc_score(y_true, y_score),
        "average_precision": average_precision_score(y_true, y_score),
        "f1_score": f1_score(y_true, y_score > 0.5),
    }

    # Add more detailed metrics
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    metrics["precision_at_threshold_0.5"] = precision[0]
    metrics["recall_at_threshold_0.5"] = recall[0]

    # Thresholds at different operating points
    for threshold in [0.3, 0.5, 0.7]:
        y_pred = (y_score >= threshold).astype(int)
        metrics[f"f1_score_at_{threshold}"] = f1_score(y_true, y_pred)

    # Log basic summary and detailed formatted metrics
    logger.info(
        f"Binary metrics computed: AUC={metrics['auc_roc']:.4f}, AP={metrics['average_precision']:.4f}, F1={metrics['f1_score']:.4f}"
    )
    log_metrics_summary(metrics, is_binary=True)

    return metrics


def compute_metrics_multiclass(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int) -> Dict[str, Union[int, float]]:
    """
    Compute multiclass metrics: one-vs-rest AUC-ROC, average precision, F1 for each class,
    and micro/macro averages for all metrics.
    """
    logger.info("Computing multiclass metrics")
    metrics = {}

    # Per-class metrics
    for i in range(n_classes):
        y_true_bin = (y_true == i).astype(int)
        y_score = y_prob[:, i]
        metrics[f"auc_roc_class_{i}"] = roc_auc_score(y_true_bin, y_score)
        metrics[f"average_precision_class_{i}"] = average_precision_score(
            y_true_bin, y_score
        )
        metrics[f"f1_score_class_{i}"] = f1_score(y_true_bin, y_score > 0.5)

    # Micro and macro averages
    metrics["auc_roc_micro"] = roc_auc_score(
        y_true, y_prob, multi_class="ovr", average="micro"
    )
    metrics["auc_roc_macro"] = roc_auc_score(
        y_true, y_prob, multi_class="ovr", average="macro"
    )
    metrics["average_precision_micro"] = average_precision_score(
        y_true, y_prob, average="micro"
    )
    metrics["average_precision_macro"] = average_precision_score(
        y_true, y_prob, average="macro"
    )

    y_pred = np.argmax(y_prob, axis=1)
    metrics["f1_score_micro"] = f1_score(y_true, y_pred, average="micro")
    metrics["f1_score_macro"] = f1_score(y_true, y_pred, average="macro")

    # Class distribution metrics
    unique, counts = np.unique(y_true, return_counts=True)
    for cls, count in zip(unique, counts):
        metrics[f"class_{cls}_count"] = int(count)
        metrics[f"class_{cls}_ratio"] = float(count) / len(y_true)

    # Log basic summary and detailed formatted metrics
    logger.info(
        f"Multiclass metrics computed: Macro AUC={metrics['auc_roc_macro']:.4f}, Micro AUC={metrics['auc_roc_micro']:.4f}"
    )
    log_metrics_summary(metrics, is_binary=False)

    return metrics


def load_eval_data(eval_data_dir: str) -> pd.DataFrame:
    """
    Load the first .csv or .parquet file found in the evaluation data directory.
    Returns a pandas DataFrame.
    """
    logger.info(f"Loading eval data from {eval_data_dir}")
    eval_files = sorted(
        [
            f
            for f in Path(eval_data_dir).glob("**/*")
            if f.suffix in [".csv", ".parquet"]
        ]
    )
    if not eval_files:
        logger.error("No eval data file found in eval_data input.")
        raise RuntimeError("No eval data file found in eval_data input.")
    eval_file = eval_files[0]
    logger.info(f"Using eval data file: {eval_file}")
    if eval_file.suffix == ".parquet":
        df = pd.read_parquet(eval_file)
    else:
        df = pd.read_csv(eval_file)
    logger.info(f"Loaded eval data shape: {df.shape}")
    return df


def get_id_label_columns(df: pd.DataFrame, id_field: str, label_field: str) -> Tuple[str, str]:
    """
    Determine the ID and label columns in the DataFrame.
    Falls back to the first and second columns if not found.
    """
    id_col = id_field if id_field in df.columns else df.columns[0]
    label_col = label_field if label_field in df.columns else df.columns[1]
    logger.info(f"Using id_col: {id_col}, label_col: {label_col}")
    return id_col, label_col


def save_predictions(ids: np.ndarray, y_true: np.ndarray, y_prob: np.ndarray, id_col: str, label_col: str, output_eval_dir: str) -> None:
    """
    Save predictions to a CSV file, including id, true label, and class probabilities.
    """
    logger.info(f"Saving predictions to {output_eval_dir}")
    prob_cols = [f"prob_class_{i}" for i in range(y_prob.shape[1])]
    out_df = pd.DataFrame({id_col: ids, label_col: y_true})
    for i, col in enumerate(prob_cols):
        out_df[col] = y_prob[:, i]
    out_path = os.path.join(output_eval_dir, "eval_predictions.csv")
    out_df.to_csv(out_path, index=False)
    logger.info(f"Saved predictions to {out_path}")


def save_metrics(metrics: Dict[str, Union[int, float, str]], output_metrics_dir: str) -> None:
    """
    Save computed metrics as a JSON file.
    """
    out_path = os.path.join(output_metrics_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {out_path}")

    # Also create a plain text summary for easy viewing
    summary_path = os.path.join(output_metrics_dir, "metrics_summary.txt")
    with open(summary_path, "w") as f:
        f.write("METRICS SUMMARY\n")
        f.write("=" * 50 + "\n")

        # Write key metrics at the top
        if "auc_roc" in metrics:  # Binary classification
            f.write(f"AUC-ROC:           {metrics['auc_roc']:.4f}\n")
            if "average_precision" in metrics:
                f.write(f"Average Precision: {metrics['average_precision']:.4f}\n")
            if "f1_score" in metrics:
                f.write(f"F1 Score:          {metrics['f1_score']:.4f}\n")
        else:  # Multiclass classification
            f.write(f"AUC-ROC (Macro):   {metrics.get('auc_roc_macro', 'N/A'):.4f}\n")
            f.write(f"AUC-ROC (Micro):   {metrics.get('auc_roc_micro', 'N/A'):.4f}\n")
            f.write(f"F1 Score (Macro):  {metrics.get('f1_score_macro', 'N/A'):.4f}\n")

        f.write("=" * 50 + "\n\n")

        # Write all metrics
        f.write("ALL METRICS\n")
        f.write("=" * 50 + "\n")
        for name, value in sorted(metrics.items()):
            if isinstance(value, (int, float)):
                f.write(f"{name}: {value:.6f}\n")
            else:
                f.write(f"{name}: {value}\n")

    logger.info(f"Saved metrics summary to {summary_path}")


def plot_and_save_roc_curve(y_true: np.ndarray, y_score: np.ndarray, output_dir: str, prefix: str = "") -> None:
    """
    Plot ROC curve and save as JPG.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    out_path = os.path.join(output_dir, f"{prefix}roc_curve.jpg")
    plt.savefig(out_path, format="jpg")
    plt.close()
    logger.info(f"Saved ROC curve to {out_path}")


def plot_and_save_pr_curve(y_true: np.ndarray, y_score: np.ndarray, output_dir: str, prefix: str = "") -> None:
    """
    Plot Precision-Recall curve and save as JPG.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure()
    plt.plot(recall, precision, label=f"PR curve (AP = {ap:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    out_path = os.path.join(output_dir, f"{prefix}pr_curve.jpg")
    plt.savefig(out_path, format="jpg")
    plt.close()
    logger.info(f"Saved PR curve to {out_path}")


def evaluate_model(
    model: xgb.Booster,
    df: pd.DataFrame,
    feature_columns: List[str],
    id_col: str,
    label_col: str,
    hyperparams: Dict[str, Any],
    output_eval_dir: str,
    output_metrics_dir: str,
) -> None:
    """
    Run model prediction and evaluation, then save predictions and metrics.
    Also generate and save ROC and PR curves as JPG.
    """
    logger.info("Evaluating model")
    y_true = df[label_col].values
    ids = df[id_col].values
    X = df[feature_columns].values

    dmatrix = xgb.DMatrix(X, feature_names=feature_columns)
    y_prob = model.predict(dmatrix)
    logger.info(f"Model prediction shape: {y_prob.shape}")
    if len(y_prob.shape) == 1:
        y_prob = np.column_stack([1 - y_prob, y_prob])
        logger.info("Converted binary prediction to two-column probabilities")

    # Determine the classification type from the model's saved hyperparameters,
    # which is the definitive source of truth.
    is_binary_model = hyperparams.get("is_binary", True)

    if is_binary_model:
        logger.info(
            "Detected binary classification task based on model hyperparameters."
        )
        # Ensure y_true is also binary (0 or 1) for consistent metric calculation
        y_true = (y_true > 0).astype(int)
        metrics = compute_metrics_binary(y_true, y_prob)
        plot_and_save_roc_curve(y_true, y_prob[:, 1], output_metrics_dir)
        plot_and_save_pr_curve(y_true, y_prob[:, 1], output_metrics_dir)
    else:
        n_classes = y_prob.shape[1]
        logger.info(
            f"Detected multiclass classification task with {n_classes} classes."
        )
        metrics = compute_metrics_multiclass(y_true, y_prob, n_classes)
        for i in range(n_classes):
            y_true_bin = (y_true == i).astype(int)
            if len(np.unique(y_true_bin)) > 1:
                plot_and_save_roc_curve(
                    y_true_bin, y_prob[:, i], output_metrics_dir, prefix=f"class_{i}_"
                )
                plot_and_save_pr_curve(
                    y_true_bin, y_prob[:, i], output_metrics_dir, prefix=f"class_{i}_"
                )

    save_predictions(ids, y_true, y_prob, id_col, label_col, output_eval_dir)
    save_metrics(metrics, output_metrics_dir)
    logger.info("Evaluation complete")


def create_health_check_file(output_path: str) -> str:
    """Create a health check file to signal script completion."""
    health_path = output_path
    with open(health_path, "w") as f:
        f.write(f"healthy: {datetime.now().isoformat()}")
    return health_path


def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
) -> None:
    """
    Main entry point for XGBoost model evaluation script.
    Loads model and data, runs evaluation, and saves results.

    Args:
        input_paths (Dict[str, str]): Dictionary of input paths
        output_paths (Dict[str, str]): Dictionary of output paths
        environ_vars (Dict[str, str]): Dictionary of environment variables
        job_args (argparse.Namespace): Command line arguments
    """
    # Extract paths from parameters - using contract-defined logical names
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
    label_field = environ_vars.get("LABEL_FIELD", "label")

    # Log job info
    job_type = job_args.job_type
    logger.info(f"Running model evaluation with job_type: {job_type}")

    # Ensure output directories exist
    os.makedirs(output_eval_dir, exist_ok=True)
    os.makedirs(output_metrics_dir, exist_ok=True)

    logger.info("Starting model evaluation script")

    # Load model artifacts
    model, risk_tables, impute_dict, feature_columns, hyperparams = (
        load_model_artifacts(model_dir)
    )

    # Load and preprocess data
    df = load_eval_data(eval_data_dir)
    
    # Get ID and label columns before preprocessing
    id_col, label_col = get_id_label_columns(df, id_field, label_field)
    
    # Process the data - our updated preprocess_eval_data preserves all columns including id and label
    df = preprocess_eval_data(df, feature_columns, risk_tables, impute_dict)
    
    # No need to filter or re-add id and label columns since they're already preserved
    logger.info(f"Final evaluation DataFrame shape: {df.shape}")

    # Get the available features (those that exist in the DataFrame)
    available_features = [col for col in feature_columns if col in df.columns]
    
    # Evaluate model using the final DataFrame with both features and ID/label columns
    evaluate_model(
        model,
        df,
        available_features,  # Only use available feature columns for prediction
        id_col,
        label_col,
        hyperparams,
        output_eval_dir,
        output_metrics_dir,
    )

    logger.info("Model evaluation script complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_type", type=str, required=True)
    args = parser.parse_args()

    # Set up paths using contract-defined paths only
    input_paths = {
        "model_input": CONTAINER_PATHS["MODEL_DIR"],
        "processed_data": CONTAINER_PATHS["EVAL_DATA_DIR"],
    }

    output_paths = {
        "eval_output": CONTAINER_PATHS["OUTPUT_EVAL_DIR"],
        "metrics_output": CONTAINER_PATHS["OUTPUT_METRICS_DIR"],
    }

    # Collect environment variables - ID_FIELD and LABEL_FIELD are required per contract
    environ_vars = {
        "ID_FIELD": os.environ.get("ID_FIELD", "id"),  # Fallback for testing
        "LABEL_FIELD": os.environ.get("LABEL_FIELD", "label"),  # Fallback for testing
    }

    try:
        # Call main function with testability parameters
        main(input_paths, output_paths, environ_vars, args)

        # Signal success
        success_path = os.path.join(output_paths["metrics_output"], "_SUCCESS")
        Path(success_path).touch()
        logger.info(f"Created success marker: {success_path}")

        # Create health check file
        health_path = os.path.join(output_paths["metrics_output"], "_HEALTH")
        create_health_check_file(health_path)
        logger.info(f"Created health check file: {health_path}")

        sys.exit(0)
    except Exception as e:
        # Log error and create failure marker
        logger.exception(f"Script failed with error: {e}")
        failure_path = os.path.join(
            output_paths.get("metrics_output", "/tmp"), "_FAILURE"
        )
        os.makedirs(os.path.dirname(failure_path), exist_ok=True)
        with open(failure_path, "w") as f:
            f.write(f"Error: {str(e)}")
        sys.exit(1)

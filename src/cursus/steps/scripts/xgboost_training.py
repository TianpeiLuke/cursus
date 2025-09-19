#!/usr/bin/env python3
import os
import sys
import argparse
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

import pandas as pd
import numpy as np
import pickle as pkl
import xgboost as xgb

import tarfile
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    roc_curve,
    precision_recall_curve,
)


# -------------------------------------------------------------------------
# Assuming the processor is in a directory that can be imported
# -------------------------------------------------------------------------
from ...processing.risk_table_processor import RiskTableMappingProcessor
from ...processing.numerical_imputation_processor import (
    NumericalVariableImputationProcessor,
)


# -------------------------------------------------------------------------
# Logging setup - Updated for CloudWatch compatibility
# -------------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    """Configure logging for CloudWatch compatibility"""
    # Remove any existing handlers
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
        handlers=[
            # StreamHandler with stdout for CloudWatch
            logging.StreamHandler(sys.stdout)
        ],
    )

    # Configure our module's logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = True  # Allow propagation to root logger

    # Force flush stdout
    sys.stdout.flush()

    return logger


# Initialize logger
logger = setup_logging()

# -------------------------------------------------------------------------
# Pydantic V2 model for all hyperparameters
# -------------------------------------------------------------------------
from pydantic import BaseModel, Field, model_validator
from ..hyperparams.hyperparameters_xgboost import XGBoostModelHyperparameters


class XGBoostConfig(XGBoostModelHyperparameters):
    """
    Load everything from your pipeline’s XGBoostModelHyperparameters,
    plus the two risk-table params this script needs.
    """

    smooth_factor: float = Field(
        default=0.0, description="Smoothing factor for risk table"
    )
    count_threshold: int = Field(
        default=0, description="Minimum count threshold for risk table"
    )


# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------
def load_and_validate_config(hparam_path: str) -> dict:
    """Loads and validates the hyperparameters JSON file."""
    try:
        with open(hparam_path, "r") as f:
            config = json.load(f)

        required_keys = [
            "tab_field_list",
            "cat_field_list",
            "label_name",
            "multiclass_categories",
        ]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key in config: {key}")

        if "num_classes" not in config:
            config['num_classes'] = len(config['multiclass_categories'])

        if 'is_binary' not in config:
            config['is_binary'] = (config['num_classes'] == 2)

        # Validate class_weights if present
        if "class_weights" in config:
            if len(config["class_weights"]) != config["num_classes"]:
                raise ValueError(
                    f"Number of class weights ({len(config['class_weights'])}) "
                    f"does not match number of classes ({config['num_classes']})"
                )

        return config
    except Exception as err:
        logger.error(f"Failed to load/validate hyperparameters: {err}")
        raise


def find_first_data_file(data_dir: str) -> str:
    """Finds the first supported data file in a directory."""
    if not os.path.isdir(data_dir):
        return None
    for fname in sorted(os.listdir(data_dir)):
        if fname.lower().endswith((".csv", ".parquet", ".json")):
            return os.path.join(data_dir, fname)
    return None


def load_datasets(input_path: str) -> tuple:
    """Loads the training, validation, and test datasets."""
    train_file = find_first_data_file(os.path.join(input_path, "train"))
    val_file = find_first_data_file(os.path.join(input_path, "val"))
    test_file = find_first_data_file(os.path.join(input_path, "test"))

    if not train_file or not val_file or not test_file:
        raise FileNotFoundError(
            "Training, validation, or test data file not found in the expected subfolders."
        )

    train_df = (
        pd.read_parquet(train_file)
        if train_file.endswith(".parquet")
        else pd.read_csv(train_file)
    )
    val_df = (
        pd.read_parquet(val_file)
        if val_file.endswith(".parquet")
        else pd.read_csv(val_file)
    )
    test_df = (
        pd.read_parquet(test_file)
        if test_file.endswith(".parquet")
        else pd.read_csv(test_file)
    )

    logger.info(
        f"Loaded data -> train: {train_df.shape}, val: {val_df.shape}, test: {test_df.shape}"
    )
    return train_df, val_df, test_df


def apply_numerical_imputation(
    config: dict, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple:
    """Applies numerical imputation to the datasets."""
    imputer = NumericalVariableImputationProcessor(
        variables=config["tab_field_list"], strategy="mean"
    )
    imputer.fit(train_df)

    train_df_imputed = imputer.transform(train_df)
    val_df_imputed = imputer.transform(val_df)
    test_df_imputed = imputer.transform(test_df)

    return (
        train_df_imputed,
        val_df_imputed,
        test_df_imputed,
        imputer.get_params()["imputation_dict"],
    )


def fit_and_apply_risk_tables(
    config: dict, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple:
    """Fits risk tables on training data and applies them to all splits."""
    risk_processors = {}
    train_df_transformed = train_df.copy()
    val_df_transformed = val_df.copy()
    test_df_transformed = test_df.copy()

    for var in config["cat_field_list"]:
        proc = RiskTableMappingProcessor(
            column_name=var,
            label_name=config["label_name"],
            smooth_factor=config.get("smooth_factor", 0.0),
            count_threshold=config.get("count_threshold", 0),
        )
        proc.fit(train_df)
        risk_processors[var] = proc

        train_df_transformed[var] = proc.transform(train_df_transformed[var])
        val_df_transformed[var] = proc.transform(val_df_transformed[var])
        test_df_transformed[var] = proc.transform(test_df_transformed[var])

    consolidated_risk_tables = {
        var: proc.get_risk_tables() for var, proc in risk_processors.items()
    }
    return (
        train_df_transformed,
        val_df_transformed,
        test_df_transformed,
        consolidated_risk_tables,
    )


def prepare_dmatrices(
    config: dict, train_df: pd.DataFrame, val_df: pd.DataFrame
) -> Tuple[xgb.DMatrix, xgb.DMatrix, List[str]]:
    """
    Prepares XGBoost DMatrix objects from dataframes.

    Returns:
        Tuple containing:
        - Training DMatrix
        - Validation DMatrix
        - List of feature columns in the exact order used for the model
    """
    # Maintain exact ordering of features as they'll be used in the model
    feature_columns = config["tab_field_list"] + config["cat_field_list"]

    # Check for any remaining NaN/inf values
    X_train = train_df[feature_columns].astype(float)
    X_val = val_df[feature_columns].astype(float)

    if X_train.isna().any().any() or np.isinf(X_train).any().any():
        raise ValueError("Training data contains NaN or inf values after preprocessing")
    if X_val.isna().any().any() or np.isinf(X_val).any().any():
        raise ValueError(
            "Validation data contains NaN or inf values after preprocessing"
        )

    dtrain = xgb.DMatrix(
        X_train.values, label=train_df[config["label_name"]].astype(int).values
    )
    dval = xgb.DMatrix(
        X_val.values, label=val_df[config["label_name"]].astype(int).values
    )

    # Set feature names in DMatrix to ensure they're preserved
    dtrain.feature_names = feature_columns
    dval.feature_names = feature_columns

    return dtrain, dval, feature_columns


def train_model(config: dict, dtrain: xgb.DMatrix, dval: xgb.DMatrix) -> xgb.Booster:
    """
    Trains the XGBoost model.

    Args:
        config: Configuration dictionary containing model parameters
        dtrain: Training data as XGBoost DMatrix
        dval: Validation data as XGBoost DMatrix

    Returns:
        Trained XGBoost model
    """
    # Base parameters
    xgb_params = {
        "eta": config.get("eta", 0.1),
        "gamma": config.get("gamma", 0),
        "max_depth": config.get("max_depth", 6),
        "subsample": config.get("subsample", 1),
        "colsample_bytree": config.get("colsample_bytree", 1),
        "lambda": config.get("lambda_xgb", 1),
        "alpha": config.get("alpha_xgb", 0),
    }

    # Set objective and num_class based on hyperparameters
    # Handle class weights
    if config.get("is_binary", True):
        xgb_params["objective"] = "binary:logistic"
        if "class_weights" in config and len(config["class_weights"]) == 2:
            # For binary classification, use scale_pos_weight
            xgb_params["scale_pos_weight"] = (
                config["class_weights"][1] / config["class_weights"][0]
            )
    else:
        xgb_params["objective"] = "multi:softprob"
        xgb_params["num_class"] = config["num_classes"]

    logger.info(f"Starting XGBoost training with params: {xgb_params}")
    logger.info(f"Number of classes from config: {config.get('num_classes', 2)}")

    # Print label distribution for debugging
    y_train = dtrain.get_label()
    y_val = dval.get_label()
    logger.info(
        f"Label distribution in training data: {pd.Series(y_train).value_counts().sort_index()}"
    )
    logger.info(
        f"Label distribution in validation data: {pd.Series(y_val).value_counts().sort_index()}"
    )

    # Handle class weights for multiclass
    if not config.get("is_binary", True) and "class_weights" in config:
        sample_weights = np.ones(len(y_train))
        for i, weight in enumerate(config["class_weights"]):
            sample_weights[y_train == i] = weight
        dtrain.set_weight(sample_weights)

    return xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=config.get("num_round", 100),
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=config.get("early_stopping_rounds", 10),
        verbose_eval=True,
    )


def save_artifacts(
    model: xgb.Booster,
    risk_tables: dict,
    impute_dict: dict,
    model_path: str,
    feature_columns: List[str],
    config: dict,
):
    """
    Saves the trained model and preprocessing artifacts.

    Args:
        model: Trained XGBoost model
        risk_tables: Dictionary of risk tables
        impute_dict: Dictionary of imputation values
        model_path: Path to save model artifacts
        feature_columns: List of feature column names
        config: Configuration dictionary containing hyperparameters
    """
    os.makedirs(model_path, exist_ok=True)

    # Save XGBoost model
    model_file = os.path.join(model_path, "xgboost_model.bst")
    model.save_model(model_file)
    logger.info(f"Saved XGBoost model to {model_file}")

    # Save risk tables
    risk_map_file = os.path.join(model_path, "risk_table_map.pkl")
    with open(risk_map_file, "wb") as f:
        pkl.dump(risk_tables, f)
    logger.info(f"Saved consolidated risk table map to {risk_map_file}")

    # Save imputation dictionary
    impute_file = os.path.join(model_path, "impute_dict.pkl")
    with open(impute_file, "wb") as f:
        pkl.dump(impute_dict, f)
    logger.info(f"Saved imputation dictionary to {impute_file}")

    # Save feature importance
    fmap_json = os.path.join(model_path, "feature_importance.json")
    with open(fmap_json, "w") as f:
        json.dump(model.get_fscore(), f, indent=2)
    logger.info(f"Saved feature importance to {fmap_json}")

    # Save feature columns with ordering information
    feature_columns_file = os.path.join(model_path, "feature_columns.txt")
    with open(feature_columns_file, "w") as f:
        # Add a header comment to document the importance of ordering
        f.write(
            "# Feature columns in exact order required for XGBoost model inference\n"
        )
        f.write("# DO NOT MODIFY THE ORDER OF THESE COLUMNS\n")
        f.write("# Each line contains: <column_index>,<column_name>\n")
        for idx, column in enumerate(feature_columns):
            f.write(f"{idx},{column}\n")
    logger.info(f"Saved ordered feature columns to {feature_columns_file}")

    # Save hyperparameters configuration
    hyperparameters_file = os.path.join(model_path, "hyperparameters.json")
    with open(hyperparameters_file, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    logger.info(f"Saved hyperparameters configuration to {hyperparameters_file}")


# -------------------------------------------------------------------------
# New: inference + evaluation helpers
# -------------------------------------------------------------------------
def save_preds_and_metrics(ids, y_true, y_prob, id_col, label_col, out_dir, is_binary) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # metrics
    metrics = {}
    if is_binary:
        score = y_prob[:, 1]
        metrics = {
            "auc_roc": roc_auc_score(y_true, score),
            "average_precision": average_precision_score(y_true, score),
            "f1_score": f1_score(y_true, score > 0.5),
        }
        logger.info(f"AUC-ROC: {metrics['auc_roc']}")
        logger.info(f"Average Precision: {metrics['average_precision']}")
        logger.info(f"F1-Score: {metrics['f1_score']}")
    else:
        n = y_prob.shape[1]
        for i in range(n):
            y_bin = (y_true == i).astype(int)
            metrics[f"auc_roc_class_{i}"] = roc_auc_score(y_bin, y_prob[:, i])
            metrics[f"average_precision_class_{i}"] = average_precision_score(
                y_bin, y_prob[:, i]
            )
            metrics[f"f1_score_class_{i}"] = f1_score(y_bin, y_prob[:, i] > 0.5)
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
        logger.info(f"AUC-ROC (micro): {metrics['auc_roc_micro']}")
        logger.info(f"AUC-ROC (macro): {metrics['auc_roc_macro']}")
        logger.info(f"Average Precision (micro): {metrics['average_precision_micro']}")
        logger.info(f"Average Precision (macro): {metrics['average_precision_macro']}")
        logger.info(f"F1-Score (micro): {metrics['f1_score_micro']}")
        logger.info(f"F1-Score (macro): {metrics['f1_score_macro']}")
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    # preds
    df = pd.DataFrame({id_col: ids, label_col: y_true})
    for i in range(y_prob.shape[1]):
        df[f"prob_class_{i}"] = y_prob[:, i]
    df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)


def plot_curves(y_true, y_prob, out_dir, prefix, is_binary) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if is_binary:
        score = y_prob[:, 1]
        fpr, tpr, _ = roc_curve(y_true, score)
        auc = roc_auc_score(y_true, score)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
        plt.plot([0, 1], [0, 1], "--")
        plt.title(f"{prefix} ROC")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        plt.savefig(os.path.join(out_dir, f"{prefix}roc.jpg"))
        plt.close()
        precision, recall, _ = precision_recall_curve(y_true, score)
        ap = average_precision_score(y_true, score)
        plt.figure()
        plt.plot(recall, precision, label=f"AP={ap:.3f}")
        plt.title(f"{prefix} PR")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.savefig(os.path.join(out_dir, f"{prefix}pr.jpg"))
        plt.close()
    else:
        n = y_prob.shape[1]
        for i in range(n):
            y_bin = (y_true == i).astype(int)
            if len(np.unique(y_bin)) > 1:
                fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
                auc = roc_auc_score(y_bin, y_prob[:, i])
                plt.figure()
                plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
                plt.plot([0, 1], [0, 1], "--")
                plt.title(f"{prefix} class {i} ROC")
                plt.xlabel("FPR")
                plt.ylabel("TPR")
                plt.legend()
                plt.savefig(os.path.join(out_dir, f"{prefix}class_{i}_roc.jpg"))
                plt.close()
                precision, recall, _ = precision_recall_curve(y_bin, y_prob[:, i])
                ap = average_precision_score(y_bin, y_prob[:, i])
                plt.figure()
                plt.plot(recall, precision, label=f"AP={ap:.3f}")
                plt.title(f"{prefix} class {i} PR")
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.legend()
                plt.savefig(os.path.join(out_dir, f"{prefix}class_{i}_pr.jpg"))
                plt.close()


def evaluate_split(name, df, feats, model, cfg, prefix="/opt/ml/output/data") -> None:
    is_bin = cfg.get("is_binary", True)
    label = cfg["label_name"]
    idi = cfg.get("id_name", "id")

    ids = df.get(idi, np.arange(len(df)))
    y_true = df[label].astype(int).values

    # Build DMatrix *with* feature names
    X = df[feats]
    dmat = xgb.DMatrix(data=X, feature_names=feats)

    y_prob = model.predict(dmat)
    if y_prob.ndim == 1:
        y_prob = np.vstack([1 - y_prob, y_prob]).T

    # directories
    out_base = os.path.join(prefix, name)
    out_metrics = os.path.join(prefix, f"{name}_metrics")

    # save preds & metrics, then plots, then tar
    save_preds_and_metrics(ids, y_true, y_prob, idi, label, out_base, is_bin)
    plot_curves(y_true, y_prob, out_metrics, f"{name}_", is_bin)

    tar = os.path.join(prefix, f"{name}.tar.gz")
    with tarfile.open(tar, "w:gz") as t:
        t.add(out_base, arcname=name)
        t.add(out_metrics, arcname=f"{name}_metrics")

    logger.info(f"{name} outputs packaged → {tar}")


# -------------------------------------------------------------------------
# Main Orchestrator
# -------------------------------------------------------------------------
def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
) -> None:
    """
    Main function to execute the XGBoost training logic.

    Args:
        input_paths: Dictionary of input paths with logical names
            - "input_path": Directory containing train/val/test data
            - "hyperparameters_s3_uri": Path to hyperparameters directory (now points to /opt/ml/code/hyperparams)
        output_paths: Dictionary of output paths with logical names
            - "model_output": Directory to save model artifacts
            - "evaluation_output": Directory to save evaluation outputs
        environ_vars: Dictionary of environment variables
        job_args: Command line arguments
    """
    try:
        logger.info("====== STARTING MAIN EXECUTION ======")

        # Extract paths from parameters using contract logical names
        data_dir = input_paths["input_path"]
        model_dir = output_paths["model_output"]
        output_dir = output_paths["evaluation_output"]

        # Build hyperparameters path - handle both file path and directory cases
        if "hyperparameters_s3_uri" in input_paths:
            hparam_path = input_paths["hyperparameters_s3_uri"]
            # If it's a directory path, append the filename
            if not hparam_path.endswith("hyperparameters.json"):
                hparam_path = os.path.join(hparam_path, "hyperparameters.json")
        else:
            # Fallback to source directory if not provided
            hparam_path = "/opt/ml/code/hyperparams/hyperparameters.json"

        logger.info("Starting XGBoost training process...")
        logger.info(f"Loading configuration from {hparam_path}")
        config = load_and_validate_config(hparam_path)
        logger.info("Configuration loaded successfully")

        logger.info("Loading datasets...")
        train_df, val_df, test_df = load_datasets(data_dir)
        logger.info("Datasets loaded successfully")

        # Apply numerical imputation
        logger.info("Starting numerical imputation...")
        train_df, val_df, test_df, impute_dict = apply_numerical_imputation(
            config, train_df, val_df, test_df
        )
        logger.info("Numerical imputation completed")

        # Apply risk table mapping
        logger.info("Starting risk table mapping...")
        train_df, val_df, test_df, risk_tables = fit_and_apply_risk_tables(
            config, train_df, val_df, test_df
        )
        logger.info("Risk table mapping completed")

        logger.info("Preparing DMatrices for XGBoost...")
        dtrain, dval, feature_columns = prepare_dmatrices(config, train_df, val_df)
        logger.info("DMatrices prepared successfully")
        logger.info(
            f"Using {len(feature_columns)} features in order: {feature_columns}"
        )

        logger.info("Starting model training...")
        model = train_model(config, dtrain, dval)
        logger.info("Model training completed")

        logger.info("Saving model artifacts...")
        logger.info(f"Model path: {model_dir}, Output path: {output_dir}")
        logger.info(f"Output path exists: {os.path.exists(output_dir)}")

        save_artifacts(
            model=model,
            risk_tables=risk_tables,
            impute_dict=impute_dict,
            model_path=model_dir,
            feature_columns=feature_columns,
            config=config,
        )
        logger.info("✓ Model artifacts saved successfully")

        # --- inference + evaluation on val and test ---
        logger.info("====== STARTING EVALUATION PHASE ======")

        # Add explicit directory checks
        logger.info(f"Checking output directory: {output_dir}")
        if not os.path.exists(output_dir):
            logger.warning(f"Output directory {output_dir} does not exist, creating...")
            os.makedirs(output_dir, exist_ok=True)

        # Validation evaluation with exception handling
        logger.info("Starting inference & evaluation on validation set")
        try:
            evaluate_split("val", val_df, feature_columns, model, config, output_dir)
            logger.info("✓ Validation evaluation completed successfully")
        except Exception as e:
            logger.error(f"ERROR in validation evaluation: {str(e)}")
            logger.error(traceback.format_exc())

        # Test evaluation with exception handling
        logger.info("Starting inference & evaluation on test set")
        try:
            evaluate_split("test", test_df, feature_columns, model, config, output_dir)
            logger.info("✓ Test evaluation completed successfully")
        except Exception as e:
            logger.error(f"ERROR in test evaluation: {str(e)}")
            logger.error(traceback.format_exc())

        logger.info("All evaluation steps complete.")
        logger.info("====== MAIN EXECUTION COMPLETED SUCCESSFULLY ======")
        logger.info("Training script finished successfully.")
    except Exception as e:
        logger.error(f"FATAL ERROR in main execution: {str(e)}")
        logger.error(traceback.format_exc())
        raise


# -------------------------------------------------------------------------
# Script Entry Point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Script starting...")

    # Container path constants
    CONTAINER_PATHS = {
        "INPUT_DATA": "/opt/ml/input/data",
        "MODEL_DIR": "/opt/ml/model",
        "OUTPUT_DATA": "/opt/ml/output/data",
        "CONFIG_DIR": "/opt/ml/code/hyperparams",  # Source directory path
    }

    # Define input and output paths using contract logical names
    # Use container defaults (no CLI arguments per contract)
    input_paths = {
        "input_path": CONTAINER_PATHS["INPUT_DATA"],
        "hyperparameters_s3_uri": CONTAINER_PATHS["CONFIG_DIR"],
    }

    output_paths = {
        "model_output": CONTAINER_PATHS["MODEL_DIR"],
        "evaluation_output": CONTAINER_PATHS["OUTPUT_DATA"],
    }

    # Collect environment variables (none currently used, but following the pattern)
    environ_vars = {
        # Add any environment variables the script needs here
        # Example: "LOG_LEVEL": os.environ.get("LOG_LEVEL", "INFO")
    }

    # Create empty args namespace to maintain function signature
    args = argparse.Namespace()

    try:
        logger.info(f"Starting main process with paths:")
        logger.info(f"  Data directory: {input_paths['input_path']}")
        logger.info(f"  Config directory: {input_paths['hyperparameters_s3_uri']}")
        logger.info(f"  Model directory: {output_paths['model_output']}")
        logger.info(f"  Output directory: {output_paths['evaluation_output']}")

        # Call the refactored main function
        main(input_paths, output_paths, environ_vars, args)

        logger.info("XGBoost training script completed successfully")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Exception during training: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

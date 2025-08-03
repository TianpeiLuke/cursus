import os
import json
import argparse
import pandas as pd
import numpy as np
import pickle as pkl
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, f1_score
import xgboost as xgb
import matplotlib.pyplot as plt
import time

from ...processing.risk_table_processor import RiskTableMappingProcessor
from ...processing.numerical_imputation_processor import NumericalVariableImputationProcessor
from .contract_utils import ContractEnforcer

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model_artifacts(model_dir):
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
        feature_columns = [line.strip().split(",")[1] for line in f if not line.startswith("#")]
    logger.info(f"Loaded feature_columns.txt: {feature_columns}")
    with open(os.path.join(model_dir, "hyperparameters.json"), "r") as f:
        hyperparams = json.load(f)
    logger.info("Loaded hyperparameters.json")
    return model, risk_tables, impute_dict, feature_columns, hyperparams

def preprocess_eval_data(df, feature_columns, risk_tables, impute_dict):
    """
    Apply risk table mapping and numerical imputation to the evaluation DataFrame.
    Ensures all features are numeric and columns are ordered as required by the model.
    """
    logger.info("Starting risk table mapping for categorical features")
    for feature, risk_table in risk_tables.items():
        if feature in df.columns:
            logger.info(f"Applying risk table mapping for feature: {feature}")
            proc = RiskTableMappingProcessor(
                column_name=feature,
                label_name="label",
                risk_tables=risk_table
            )
            df[feature] = proc.transform(df[feature])
    logger.info("Risk table mapping complete")
    logger.info("Starting numerical imputation")
    imputer = NumericalVariableImputationProcessor(imputation_dict=impute_dict)
    df = imputer.transform(df)
    logger.info("Numerical imputation complete")
    logger.info("Ensuring all features are numeric and reordering columns")
    df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    df = df.copy()
    df = df[[col for col in feature_columns if col in df.columns]]
    logger.info(f"Preprocessed eval data shape: {df.shape}")
    return df

def log_metrics_summary(metrics, is_binary=True):
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
        logger.info(f"METRIC_KEY: AUC-ROC               = {metrics.get('auc_roc', 'N/A'):.4f}")
        logger.info(f"METRIC_KEY: Average Precision     = {metrics.get('average_precision', 'N/A'):.4f}")
        logger.info(f"METRIC_KEY: F1 Score              = {metrics.get('f1_score', 'N/A'):.4f}")
    else:
        logger.info(f"METRIC_KEY: Macro AUC-ROC         = {metrics.get('auc_roc_macro', 'N/A'):.4f}")
        logger.info(f"METRIC_KEY: Micro AUC-ROC         = {metrics.get('auc_roc_micro', 'N/A'):.4f}")
        logger.info(f"METRIC_KEY: Macro Average Precision = {metrics.get('average_precision_macro', 'N/A'):.4f}")
        logger.info(f"METRIC_KEY: Macro F1              = {metrics.get('f1_score_macro', 'N/A'):.4f}")
        logger.info(f"METRIC_KEY: Micro F1              = {metrics.get('f1_score_micro', 'N/A'):.4f}")
    
    # Add a summary section with pass/fail criteria if defined
    logger.info("=" * 80)

def compute_metrics_binary(y_true, y_prob):
    """
    Compute binary classification metrics: AUC-ROC, average precision, and F1 score.
    """
    logger.info("Computing binary classification metrics")
    y_score = y_prob[:, 1]
    metrics = {
        "auc_roc": roc_auc_score(y_true, y_score),
        "average_precision": average_precision_score(y_true, y_score),
        "f1_score": f1_score(y_true, y_score > 0.5)
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
    logger.info(f"Binary metrics computed: AUC={metrics['auc_roc']:.4f}, AP={metrics['average_precision']:.4f}, F1={metrics['f1_score']:.4f}")
    log_metrics_summary(metrics, is_binary=True)
    
    return metrics

def compute_metrics_multiclass(y_true, y_prob, n_classes):
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
        metrics[f"average_precision_class_{i}"] = average_precision_score(y_true_bin, y_score)
        metrics[f"f1_score_class_{i}"] = f1_score(y_true_bin, y_score > 0.5)
    
    # Micro and macro averages
    metrics["auc_roc_micro"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="micro")
    metrics["auc_roc_macro"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    metrics["average_precision_micro"] = average_precision_score(y_true, y_prob, average="micro")
    metrics["average_precision_macro"] = average_precision_score(y_true, y_prob, average="macro")
    
    y_pred = np.argmax(y_prob, axis=1)
    metrics["f1_score_micro"] = f1_score(y_true, y_pred, average="micro")
    metrics["f1_score_macro"] = f1_score(y_true, y_pred, average="macro")
    
    # Class distribution metrics
    unique, counts = np.unique(y_true, return_counts=True)
    for cls, count in zip(unique, counts):
        metrics[f"class_{cls}_count"] = int(count)
        metrics[f"class_{cls}_ratio"] = float(count) / len(y_true)
    
    # Log basic summary and detailed formatted metrics
    logger.info(f"Multiclass metrics computed: Macro AUC={metrics['auc_roc_macro']:.4f}, Micro AUC={metrics['auc_roc_micro']:.4f}")
    log_metrics_summary(metrics, is_binary=False)
    
    return metrics

def load_eval_data(eval_data_dir):
    """
    Load the first .csv or .parquet file found in the evaluation data directory.
    Returns a pandas DataFrame.
    """
    logger.info(f"Loading eval data from {eval_data_dir}")
    eval_files = sorted([f for f in Path(eval_data_dir).glob("**/*") if f.suffix in [".csv", ".parquet"]])
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

def get_id_label_columns(df, id_field, label_field):
    """
    Determine the ID and label columns in the DataFrame.
    Falls back to the first and second columns if not found.
    """
    id_col = id_field if id_field in df.columns else df.columns[0]
    label_col = label_field if label_field in df.columns else df.columns[1]
    logger.info(f"Using id_col: {id_col}, label_col: {label_col}")
    return id_col, label_col

def save_predictions(ids, y_true, y_prob, id_col, label_col, output_eval_dir):
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

def save_metrics(metrics, output_metrics_dir):
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
            if 'average_precision' in metrics:
                f.write(f"Average Precision: {metrics['average_precision']:.4f}\n")
            if 'f1_score' in metrics:
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

def plot_and_save_roc_curve(y_true, y_score, output_dir, prefix=""):
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

def plot_and_save_pr_curve(y_true, y_score, output_dir, prefix=""):
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

def evaluate_model(model, df, feature_columns, id_col, label_col, hyperparams, output_eval_dir, output_metrics_dir):
    """
    Run model prediction and evaluation, then save predictions and metrics.
    Also generate and save ROC and PR curves as JPG.
    """
    logger.info("Evaluating model")
    y_true = df[label_col].values
    ids = df[id_col].values
    X = df[feature_columns].values

    dmatrix = xgb.DMatrix(X)
    y_prob = model.predict(dmatrix)
    logger.info(f"Model prediction shape: {y_prob.shape}")
    if len(y_prob.shape) == 1:
        y_prob = np.column_stack([1 - y_prob, y_prob])
        logger.info("Converted binary prediction to two-column probabilities")

    # FIX: Determine the classification type from the model's saved hyperparameters,
    # which is the definitive source of truth.
    is_binary_model = hyperparams.get("is_binary", True)
    
    if is_binary_model:
        logger.info("Detected binary classification task based on model hyperparameters.")
        # Ensure y_true is also binary (0 or 1) for consistent metric calculation
        y_true = (y_true > 0).astype(int)
        metrics = compute_metrics_binary(y_true, y_prob)
        plot_and_save_roc_curve(y_true, y_prob[:, 1], output_metrics_dir)
        plot_and_save_pr_curve(y_true, y_prob[:, 1], output_metrics_dir)
    else:
        n_classes = y_prob.shape[1]
        logger.info(f"Detected multiclass classification task with {n_classes} classes.")
        metrics = compute_metrics_multiclass(y_true, y_prob, n_classes)
        for i in range(n_classes):
            y_true_bin = (y_true == i).astype(int)
            if len(np.unique(y_true_bin)) > 1:
                plot_and_save_roc_curve(y_true_bin, y_prob[:, i], output_metrics_dir, prefix=f"class_{i}_")
                plot_and_save_pr_curve(y_true_bin, y_prob[:, i], output_metrics_dir, prefix=f"class_{i}_")

    save_predictions(ids, y_true, y_prob, id_col, label_col, output_eval_dir)
    save_metrics(metrics, output_metrics_dir)
    logger.info("Evaluation complete")

def get_script_contract():
    """Get the contract for this script"""
    # Import at runtime to avoid circular imports
    from ..contracts.model_evaluation_contract import MODEL_EVALUATION_CONTRACT
    return MODEL_EVALUATION_CONTRACT

def main():
    """
    Main entry point for XGBoost model evaluation script.
    Loads model and data, runs evaluation, and saves results.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_type", type=str, required=True)
    args = parser.parse_args()

    # Get and validate contract
    contract = get_script_contract()
    
    # Use contract enforcement context manager
    with ContractEnforcer(contract) as enforcer:
        # Access validated environment variables (contract ensures these exist)
        ID_FIELD = os.environ["ID_FIELD"]
        LABEL_FIELD = os.environ["LABEL_FIELD"]

        # Use contract paths instead of hardcoded paths
        model_dir = enforcer.get_input_path('model_input')
        eval_data_dir = enforcer.get_input_path('eval_data_input')
        output_eval_dir = enforcer.get_output_path('eval_output')
        output_metrics_dir = enforcer.get_output_path('metrics_output')

        logger.info("Starting model evaluation script")
        model, risk_tables, impute_dict, feature_columns, hyperparams = load_model_artifacts(model_dir)
        df = load_eval_data(eval_data_dir)
        df = preprocess_eval_data(df, feature_columns, risk_tables, impute_dict)
        df = df[[col for col in feature_columns if col in df.columns]]
        id_col, label_col = get_id_label_columns(df, ID_FIELD, LABEL_FIELD)
        evaluate_model(
            model, df, feature_columns, id_col, label_col, hyperparams, output_eval_dir, output_metrics_dir
        )
        logger.info("Model evaluation script complete")

if __name__ == "__main__":
    main()

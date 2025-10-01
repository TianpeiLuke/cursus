#!/usr/bin/env python
import os
import json
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
import matplotlib.pyplot as plt
import time
import sys
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Container path constants - aligned with script contract
CONTAINER_PATHS = {
    "EVAL_DATA_DIR": "/opt/ml/processing/input/eval_data",
    "OUTPUT_METRICS_DIR": "/opt/ml/processing/output/metrics",
    "OUTPUT_PLOTS_DIR": "/opt/ml/processing/output/plots",
}


def detect_and_load_predictions(
    input_dir: str, 
    preferred_format: str = None
) -> pd.DataFrame:
    """
    Auto-detect and load predictions file in CSV, Parquet, or JSON format.
    Supports intelligent format detection and graceful fallback.
    """
    formats_to_try = []
    if preferred_format:
        formats_to_try.append(preferred_format)
    
    # Add other formats as fallback
    for fmt in ["parquet", "csv", "json"]:
        if fmt not in formats_to_try:
            formats_to_try.append(fmt)
    
    for fmt in formats_to_try:
        file_path = os.path.join(input_dir, f"predictions.{fmt}")
        if os.path.exists(file_path):
            logger.info(f"Loading predictions from {file_path}")
            if fmt == "parquet":
                return pd.read_parquet(file_path)
            elif fmt == "json":
                return pd.read_json(file_path)
            else:
                return pd.read_csv(file_path)
    
    # Also try eval_predictions.csv from xgboost_model_eval output
    eval_pred_path = os.path.join(input_dir, "eval_predictions.csv")
    if os.path.exists(eval_pred_path):
        logger.info(f"Loading predictions from {eval_pred_path}")
        return pd.read_csv(eval_pred_path)
    
    raise FileNotFoundError("No predictions file found in supported formats (csv, parquet, json)")


def validate_prediction_data(
    df: pd.DataFrame,
    id_field: str,
    label_field: str,
    amount_field: str = None
) -> Dict[str, Any]:
    """
    Validate prediction data schema and return validation report.
    """
    validation_report = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "data_summary": {}
    }
    
    # Check required columns
    required_cols = [id_field, label_field]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        validation_report["errors"].append(f"Missing required columns: {missing_cols}")
        validation_report["is_valid"] = False
    
    # Check prediction probability columns
    prob_cols = [col for col in df.columns if col.startswith("prob_class_")]
    if not prob_cols:
        validation_report["errors"].append("No prediction probability columns found")
        validation_report["is_valid"] = False
    
    # Check amount column if specified
    if amount_field and amount_field not in df.columns:
        validation_report["warnings"].append(f"Amount field '{amount_field}' not found - dollar recall will be skipped")
    
    # Generate data summary
    validation_report["data_summary"] = {
        "total_records": len(df),
        "prediction_columns": prob_cols,
        "has_amount_data": amount_field in df.columns if amount_field else False,
        "label_distribution": df[label_field].value_counts().to_dict() if label_field in df.columns else {}
    }
    
    return validation_report


def compute_standard_metrics(
    y_true: np.ndarray, 
    y_prob: np.ndarray,
    is_binary: bool = True
) -> Dict[str, float]:
    """
    Compute comprehensive standard ML metrics matching xgboost_model_eval.py.
    Supports both binary and multiclass classification with full metric coverage.
    """
    metrics = {}
    
    if is_binary:
        # Binary classification metrics - matching compute_metrics_binary()
        y_score = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob
        
        # Core metrics (exact match with original)
        metrics["auc_roc"] = roc_auc_score(y_true, y_score)
        metrics["average_precision"] = average_precision_score(y_true, y_score)
        metrics["f1_score"] = f1_score(y_true, y_score > 0.5)
        
        # Precision-Recall curve analysis
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        metrics["precision_at_threshold_0.5"] = precision_score(y_true, (y_score > 0.5).astype(int))
        metrics["recall_at_threshold_0.5"] = recall_score(y_true, (y_score > 0.5).astype(int))
        
        # Threshold-based metrics (matching original)
        for threshold in [0.3, 0.5, 0.7]:
            y_pred = (y_score >= threshold).astype(int)
            metrics[f"f1_score_at_{threshold}"] = f1_score(y_true, y_pred)
            metrics[f"precision_at_{threshold}"] = precision_score(y_true, y_pred)
            metrics[f"recall_at_{threshold}"] = recall_score(y_true, y_pred)
        
        # Additional analysis metrics
        metrics["max_f1_score"] = np.max(2 * precision * recall / (precision + recall + 1e-8))
        
        # ROC curve analysis
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
        metrics["optimal_threshold"] = roc_thresholds[np.argmax(tpr - fpr)]
        
    else:
        # Multiclass classification metrics - matching compute_metrics_multiclass()
        n_classes = y_prob.shape[1]
        
        # Per-class metrics (exact match with original)
        for i in range(n_classes):
            y_true_bin = (y_true == i).astype(int)
            y_score = y_prob[:, i]
            metrics[f"auc_roc_class_{i}"] = roc_auc_score(y_true_bin, y_score)
            metrics[f"average_precision_class_{i}"] = average_precision_score(y_true_bin, y_score)
            metrics[f"f1_score_class_{i}"] = f1_score(y_true_bin, y_score > 0.5)
        
        # Micro and macro averages (exact match with original)
        metrics["auc_roc_micro"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="micro")
        metrics["auc_roc_macro"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        metrics["average_precision_micro"] = average_precision_score(y_true, y_prob, average="micro")
        metrics["average_precision_macro"] = average_precision_score(y_true, y_prob, average="macro")
        
        y_pred = np.argmax(y_prob, axis=1)
        metrics["f1_score_micro"] = f1_score(y_true, y_pred, average="micro")
        metrics["f1_score_macro"] = f1_score(y_true, y_pred, average="macro")
        
        # Class distribution metrics (matching original)
        unique, counts = np.unique(y_true, return_counts=True)
        for cls, count in zip(unique, counts):
            metrics[f"class_{cls}_count"] = int(count)
            metrics[f"class_{cls}_ratio"] = float(count) / len(y_true)
    
    return metrics


def calculate_count_recall(scores, labels, amounts, cutoff=0.1):
    """
    Calculate count recall - imported from evaluation.py
    """
    assert len(scores) == len(labels), "Input lengths don't match!"

    threshold = np.quantile(scores, 1 - cutoff)
    abuse_order_total = len(labels[labels == 1])
    abuse_order_above_threshold = len(labels[(labels == 1) & (scores >= threshold)])

    order_count_recall = abuse_order_above_threshold / abuse_order_total
    return order_count_recall


def calculate_dollar_recall(scores, labels, amounts, fpr=0.1):
    """
    Calculate dollar recall - imported from evaluation.py
    """
    assert len(scores) == len(labels) == len(amounts), "Input lengths don't match!"

    threshold = np.quantile(scores[labels == 0], 1 - fpr)
    abuse_amount_total = amounts[labels == 1].sum()
    abuse_amount_above_threshold = amounts[(labels == 1) & (scores > threshold)].sum()

    dollar_recall = abuse_amount_above_threshold / abuse_amount_total
    return dollar_recall


def compute_domain_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    amounts: np.ndarray = None,
    compute_dollar_recall: bool = True,
    compute_count_recall: bool = True,
    dollar_recall_fpr: float = 0.1,
    count_recall_cutoff: float = 0.1
) -> Dict[str, float]:
    """
    Compute domain-specific metrics including dollar and count recall.
    Integrates functions from evaluation.py for business impact analysis.
    """
    domain_metrics = {}
    
    if compute_count_recall:
        # Count recall - percentage of abuse orders caught
        count_recall = calculate_count_recall(
            scores=scores,
            labels=labels,
            amounts=amounts,  # Not used but required by function signature
            cutoff=count_recall_cutoff
        )
        domain_metrics["count_recall"] = count_recall
        domain_metrics["count_recall_cutoff"] = count_recall_cutoff
    
    if compute_dollar_recall and amounts is not None:
        # Dollar recall - percentage of abuse dollar amount caught
        dollar_recall = calculate_dollar_recall(
            scores=scores,
            labels=labels,
            amounts=amounts,
            fpr=dollar_recall_fpr
        )
        domain_metrics["dollar_recall"] = dollar_recall
        domain_metrics["dollar_recall_fpr"] = dollar_recall_fpr
        
        # Additional amount-based analysis
        domain_metrics["total_abuse_amount"] = amounts[labels == 1].sum()
        domain_metrics["average_abuse_amount"] = amounts[labels == 1].mean()
        domain_metrics["total_legitimate_amount"] = amounts[labels == 0].sum()
        domain_metrics["amount_ratio_abuse_to_total"] = (
            amounts[labels == 1].sum() / amounts.sum()
        )
    
    return domain_metrics


def plot_and_save_roc_curve(
    y_true: np.ndarray, y_score: np.ndarray, output_dir: str, prefix: str = ""
) -> str:
    """
    Plot ROC curve and save as JPG (exact match with xgboost_model_eval.py).
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
    plt.savefig(out_path, format="jpg", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved ROC curve to {out_path}")
    return out_path


def plot_and_save_pr_curve(
    y_true: np.ndarray, y_score: np.ndarray, output_dir: str, prefix: str = ""
) -> str:
    """
    Plot Precision-Recall curve and save as JPG (exact match with xgboost_model_eval.py).
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
    plt.savefig(out_path, format="jpg", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved PR curve to {out_path}")
    return out_path


def generate_performance_visualizations(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metrics: Dict[str, float],
    output_dir: str,
    is_binary: bool = True
) -> Dict[str, str]:
    """
    Generate comprehensive performance visualizations matching xgboost_model_eval.py.
    Returns dictionary of plot file paths.
    """
    plot_paths = {}
    
    if is_binary:
        y_score = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob
        
        # ROC Curve (matching plot_and_save_roc_curve)
        plot_paths["roc_curve"] = plot_and_save_roc_curve(y_true, y_score, output_dir)
        
        # Precision-Recall Curve (matching plot_and_save_pr_curve)
        plot_paths["precision_recall_curve"] = plot_and_save_pr_curve(y_true, y_score, output_dir)
        
        # Score Distribution (enhanced version)
        plt.figure(figsize=(10, 6))
        plt.hist(y_score[y_true == 0], bins=50, alpha=0.7, label='Legitimate', density=True)
        plt.hist(y_score[y_true == 1], bins=50, alpha=0.7, label='Abuse', density=True)
        plt.xlabel('Prediction Score')
        plt.ylabel('Density')
        plt.title('Score Distribution by Class')
        plt.legend()
        plt.grid(True, alpha=0.3)
        dist_path = os.path.join(output_dir, "score_distribution.jpg")
        plt.savefig(dist_path, format="jpg", dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths["score_distribution"] = dist_path
        
        # Threshold Analysis (enhanced version)
        thresholds = np.linspace(0, 1, 101)
        f1_scores = []
        precisions = []
        recalls = []
        
        for threshold in thresholds:
            y_pred = (y_score >= threshold).astype(int)
            if len(np.unique(y_pred)) > 1:  # Avoid division by zero
                f1_scores.append(f1_score(y_true, y_pred))
                precisions.append(precision_score(y_true, y_pred))
                recalls.append(recall_score(y_true, y_pred))
            else:
                f1_scores.append(0)
                precisions.append(0)
                recalls.append(0)
        
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, f1_scores, label='F1 Score')
        plt.plot(thresholds, precisions, label='Precision')
        plt.plot(thresholds, recalls, label='Recall')
        plt.axvline(x=metrics.get("optimal_threshold", 0.5), color='red', linestyle='--', label='Optimal Threshold')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Threshold Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        threshold_path = os.path.join(output_dir, "threshold_analysis.jpg")
        plt.savefig(threshold_path, format="jpg", dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths["threshold_analysis"] = threshold_path
    
    else:
        # Multiclass visualizations (matching original per-class approach)
        n_classes = y_prob.shape[1]
        
        # Per-class ROC curves (matching original with prefix)
        for i in range(n_classes):
            y_true_bin = (y_true == i).astype(int)
            if len(np.unique(y_true_bin)) > 1:  # Only plot if class exists
                plot_paths[f"roc_curve_class_{i}"] = plot_and_save_roc_curve(
                    y_true_bin, y_prob[:, i], output_dir, prefix=f"class_{i}_"
                )
                plot_paths[f"pr_curve_class_{i}"] = plot_and_save_pr_curve(
                    y_true_bin, y_prob[:, i], output_dir, prefix=f"class_{i}_"
                )
        
        # Combined multiclass ROC curves
        plt.figure(figsize=(10, 8))
        for i in range(n_classes):
            y_true_bin = (y_true == i).astype(int)
            if len(np.unique(y_true_bin)) > 1:
                fpr, tpr, _ = roc_curve(y_true_bin, y_prob[:, i])
                auc = metrics.get(f"auc_roc_class_{i}", 0)
                plt.plot(fpr, tpr, label=f'Class {i} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class ROC Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        multiclass_roc_path = os.path.join(output_dir, "multiclass_roc_curves.jpg")
        plt.savefig(multiclass_roc_path, format="jpg", dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths["multiclass_roc_curves"] = multiclass_roc_path
    
    return plot_paths


def generate_performance_insights(metrics: Dict[str, float]) -> List[str]:
    """
    Generate actionable performance insights based on metrics.
    """
    insights = []
    
    # AUC analysis
    auc = metrics.get("auc_roc", 0)
    if auc >= 0.9:
        insights.append("Excellent discrimination capability (AUC ≥ 0.9)")
    elif auc >= 0.8:
        insights.append("Good discrimination capability (AUC ≥ 0.8)")
    elif auc >= 0.7:
        insights.append("Fair discrimination capability (AUC ≥ 0.7)")
    else:
        insights.append("Poor discrimination capability (AUC < 0.7) - model may need improvement")
    
    # Dollar vs Count recall comparison
    dollar_recall = metrics.get("dollar_recall")
    count_recall = metrics.get("count_recall")
    if dollar_recall and count_recall:
        if dollar_recall > count_recall * 1.2:
            insights.append("Model is particularly effective at catching high-value abuse cases")
        elif count_recall > dollar_recall * 1.2:
            insights.append("Model catches many abuse cases but may miss high-value ones")
        else:
            insights.append("Balanced performance across abuse case values")
    
    # Threshold analysis
    optimal_threshold = metrics.get("optimal_threshold")
    if optimal_threshold:
        if optimal_threshold < 0.3:
            insights.append("Optimal threshold is low - consider if this aligns with business tolerance")
        elif optimal_threshold > 0.7:
            insights.append("Optimal threshold is high - model is conservative")
    
    return insights


def generate_recommendations(metrics: Dict[str, float]) -> List[str]:
    """
    Generate actionable recommendations based on performance analysis.
    """
    recommendations = []
    
    auc = metrics.get("auc_roc", 0)
    if auc < 0.75:
        recommendations.append("Consider feature engineering or model architecture improvements")
        recommendations.append("Investigate data quality and label accuracy")
    
    dollar_recall = metrics.get("dollar_recall")
    count_recall = metrics.get("count_recall")
    if dollar_recall and count_recall and dollar_recall < 0.6:
        recommendations.append("Focus on improving detection of high-value abuse cases")
        recommendations.append("Consider amount-weighted loss functions during training")
    
    if count_recall and count_recall < 0.7:
        recommendations.append("Consider lowering decision threshold to catch more abuse cases")
        recommendations.append("Evaluate if additional features could improve recall")
    
    # F1 score analysis
    max_f1 = metrics.get("max_f1_score", 0)
    if max_f1 < 0.6:
        recommendations.append("Model shows poor precision-recall balance - consider class balancing techniques")
    
    return recommendations


def generate_comprehensive_report(
    standard_metrics: Dict[str, float],
    domain_metrics: Dict[str, float],
    plot_paths: Dict[str, str],
    validation_report: Dict[str, Any],
    output_dir: str
) -> Dict[str, str]:
    """
    Generate comprehensive metrics report with insights and recommendations.
    """
    # Combine all metrics
    all_metrics = {**standard_metrics, **domain_metrics}
    
    # Generate JSON report
    json_report = {
        "timestamp": datetime.utcnow().isoformat(),
        "data_summary": validation_report["data_summary"],
        "standard_metrics": standard_metrics,
        "domain_metrics": domain_metrics,
        "visualizations": plot_paths,
        "performance_insights": generate_performance_insights(all_metrics),
        "recommendations": generate_recommendations(all_metrics)
    }
    
    json_path = os.path.join(output_dir, "metrics_report.json")
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2)
    
    # Generate text summary
    text_summary = generate_text_summary(json_report)
    text_path = os.path.join(output_dir, "metrics_summary.txt")
    with open(text_path, "w") as f:
        f.write(text_summary)
    
    return {
        "json_report": json_path,
        "text_summary": text_path
    }


def generate_text_summary(json_report: Dict[str, Any]) -> str:
    """
    Generate a human-readable text summary from the JSON report.
    """
    summary = []
    summary.append("MODEL METRICS COMPUTATION REPORT")
    summary.append("=" * 50)
    summary.append(f"Generated: {json_report['timestamp']}")
    summary.append("")
    
    # Data summary
    data_summary = json_report["data_summary"]
    summary.append("DATA SUMMARY")
    summary.append("-" * 20)
    summary.append(f"Total Records: {data_summary['total_records']}")
    summary.append(f"Prediction Columns: {', '.join(data_summary['prediction_columns'])}")
    summary.append(f"Has Amount Data: {data_summary['has_amount_data']}")
    summary.append("")
    
    # Standard metrics
    standard_metrics = json_report["standard_metrics"]
    summary.append("STANDARD ML METRICS")
    summary.append("-" * 20)
    for name, value in standard_metrics.items():
        if isinstance(value, (int, float)):
            summary.append(f"{name}: {value:.4f}")
        else:
            summary.append(f"{name}: {value}")
    summary.append("")
    
    # Domain metrics
    domain_metrics = json_report["domain_metrics"]
    if domain_metrics:
        summary.append("DOMAIN-SPECIFIC METRICS")
        summary.append("-" * 25)
        for name, value in domain_metrics.items():
            if isinstance(value, (int, float)):
                summary.append(f"{name}: {value:.4f}")
            else:
                summary.append(f"{name}: {value}")
        summary.append("")
    
    # Performance insights
    insights = json_report["performance_insights"]
    if insights:
        summary.append("PERFORMANCE INSIGHTS")
        summary.append("-" * 20)
        for insight in insights:
            summary.append(f"• {insight}")
        summary.append("")
    
    # Recommendations
    recommendations = json_report["recommendations"]
    if recommendations:
        summary.append("RECOMMENDATIONS")
        summary.append("-" * 15)
        for rec in recommendations:
            summary.append(f"• {rec}")
        summary.append("")
    
    # Visualizations
    visualizations = json_report["visualizations"]
    if visualizations:
        summary.append("GENERATED VISUALIZATIONS")
        summary.append("-" * 25)
        for name, path in visualizations.items():
            summary.append(f"• {name}: {os.path.basename(path)}")
    
    return "\n".join(summary)


def log_metrics_summary(metrics: Dict[str, Union[int, float, str]], is_binary: bool = True) -> None:
    """
    Log a nicely formatted summary of metrics for easy visibility in logs.
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

    logger.info("=" * 80)


def save_metrics(metrics: Dict[str, Union[int, float, str]], output_metrics_dir: str) -> None:
    """
    Save computed metrics as a JSON file (matching xgboost_model_eval.py).
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
    Main entry point for Model Metrics Computation script.
    Loads prediction data, computes metrics, generates visualizations, and saves results.

    Args:
        input_paths (Dict[str, str]): Dictionary of input paths
        output_paths (Dict[str, str]): Dictionary of output paths
        environ_vars (Dict[str, str]): Dictionary of environment variables
        job_args (argparse.Namespace): Command line arguments
    """
    # Extract paths from parameters - using contract-defined logical names
    eval_data_dir = input_paths.get("processed_data", input_paths.get("eval_data_dir"))
    output_metrics_dir = output_paths.get(
        "metrics_output", output_paths.get("output_metrics_dir")
    )
    output_plots_dir = output_paths.get(
        "plots_output", output_paths.get("output_plots_dir", output_metrics_dir)
    )

    # Extract environment variables
    id_field = environ_vars.get("ID_FIELD", "id")
    label_field = environ_vars.get("LABEL_FIELD", "label")
    amount_field = environ_vars.get("AMOUNT_FIELD", None)
    input_format = environ_vars.get("INPUT_FORMAT", "auto")
    compute_dollar_recall = environ_vars.get("COMPUTE_DOLLAR_RECALL", "true").lower() == "true"
    compute_count_recall = environ_vars.get("COMPUTE_COUNT_RECALL", "true").lower() == "true"
    dollar_recall_fpr = float(environ_vars.get("DOLLAR_RECALL_FPR", "0.1"))
    count_recall_cutoff = float(environ_vars.get("COUNT_RECALL_CUTOFF", "0.1"))
    generate_plots = environ_vars.get("GENERATE_PLOTS", "true").lower() == "true"

    # Log job info
    logger.info("Running model metrics computation")

    # Ensure output directories exist
    os.makedirs(output_metrics_dir, exist_ok=True)
    os.makedirs(output_plots_dir, exist_ok=True)

    logger.info("Starting model metrics computation script")

    # Load and validate prediction data
    df = detect_and_load_predictions(eval_data_dir, preferred_format=input_format if input_format != "auto" else None)
    validation_report = validate_prediction_data(df, id_field, label_field, amount_field)
    
    if not validation_report["is_valid"]:
        logger.error("Data validation failed:")
        for error in validation_report["errors"]:
            logger.error(f"  - {error}")
        raise ValueError("Input data validation failed")
    
    # Log warnings
    for warning in validation_report["warnings"]:
        logger.warning(warning)

    # Extract data for metrics computation
    y_true = df[label_field].values
    prob_cols = [col for col in df.columns if col.startswith("prob_class_")]
    y_prob = df[prob_cols].values
    
    # Get prediction scores for domain metrics (use class 1 probability for binary, max probability for multiclass)
    if y_prob.shape[1] == 2:
        scores = y_prob[:, 1]  # Binary classification - use positive class probability
        is_binary = True
    else:
        scores = np.max(y_prob, axis=1)  # Multiclass - use max probability
        is_binary = False
    
    # Get amounts if available
    amounts = df[amount_field].values if amount_field and amount_field in df.columns else None

    logger.info(f"Computing metrics for {'binary' if is_binary else 'multiclass'} classification")
    logger.info(f"Data shape: {df.shape}, Predictions shape: {y_prob.shape}")

    # Compute standard ML metrics
    standard_metrics = compute_standard_metrics(y_true, y_prob, is_binary=is_binary)
    log_metrics_summary(standard_metrics, is_binary=is_binary)

    # Compute domain-specific metrics
    domain_metrics = compute_domain_metrics(
        scores=scores,
        labels=y_true,
        amounts=amounts,
        compute_dollar_recall=compute_dollar_recall and amounts is not None,
        compute_count_recall=compute_count_recall,
        dollar_recall_fpr=dollar_recall_fpr,
        count_recall_cutoff=count_recall_cutoff
    )
    
    if domain_metrics:
        logger.info("Domain-specific metrics computed:")
        for name, value in domain_metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {name}: {value:.4f}")
            else:
                logger.info(f"  {name}: {value}")

    # Generate visualizations if enabled
    plot_paths = {}
    if generate_plots:
        logger.info("Generating performance visualizations")
        plot_paths = generate_performance_visualizations(
            y_true, y_prob, standard_metrics, output_plots_dir, is_binary=is_binary
        )
        logger.info(f"Generated {len(plot_paths)} visualization plots")

    # Combine all metrics for saving
    all_metrics = {**standard_metrics, **domain_metrics}
    
    # Save metrics in original format (matching xgboost_model_eval.py)
    save_metrics(all_metrics, output_metrics_dir)

    # Generate comprehensive report
    report_paths = generate_comprehensive_report(
        standard_metrics, domain_metrics, plot_paths, validation_report, output_metrics_dir
    )
    
    logger.info(f"Generated comprehensive report: {report_paths['json_report']}")
    logger.info(f"Generated text summary: {report_paths['text_summary']}")

    logger.info("Model metrics computation script complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # Set up paths using contract-defined paths only
    input_paths = {
        "processed_data": CONTAINER_PATHS["EVAL_DATA_DIR"],
    }

    output_paths = {
        "metrics_output": CONTAINER_PATHS["OUTPUT_METRICS_DIR"],
        "plots_output": CONTAINER_PATHS["OUTPUT_PLOTS_DIR"],
    }

    # Collect environment variables
    environ_vars = {
        "ID_FIELD": os.environ.get("ID_FIELD", "id"),
        "LABEL_FIELD": os.environ.get("LABEL_FIELD", "label"),
        "AMOUNT_FIELD": os.environ.get("AMOUNT_FIELD", None),
        "INPUT_FORMAT": os.environ.get("INPUT_FORMAT", "auto"),
        "COMPUTE_DOLLAR_RECALL": os.environ.get("COMPUTE_DOLLAR_RECALL", "true"),
        "COMPUTE_COUNT_RECALL": os.environ.get("COMPUTE_COUNT_RECALL", "true"),
        "DOLLAR_RECALL_FPR": os.environ.get("DOLLAR_RECALL_FPR", "0.1"),
        "COUNT_RECALL_CUTOFF": os.environ.get("COUNT_RECALL_CUTOFF", "0.1"),
        "GENERATE_PLOTS": os.environ.get("GENERATE_PLOTS", "true"),
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

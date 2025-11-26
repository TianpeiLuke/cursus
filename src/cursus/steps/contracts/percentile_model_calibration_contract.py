#!/usr/bin/env python
"""Script Contract for Percentile Model Calibration Step.

This file defines the contract for the percentile model calibration processing script,
specifying input/output paths, environment variables, and required dependencies.
"""

from ...core.base.contract_base import ScriptContract

PERCENTILE_MODEL_CALIBRATION_CONTRACT = ScriptContract(
    entry_point="percentile_model_calibration.py",
    expected_input_paths={
        "evaluation_data": "/opt/ml/processing/input/eval_data",
        "calibration_config": "/opt/ml/code/calibration",
    },
    expected_output_paths={
        "calibration_output": "/opt/ml/processing/output/calibration",
        "metrics_output": "/opt/ml/processing/output/metrics",
        "calibrated_data": "/opt/ml/processing/output/calibrated_data",
    },
    required_env_vars=[],  # At least one of SCORE_FIELD or SCORE_FIELDS must be provided
    optional_env_vars={
        "SCORE_FIELD": "prob_class_1",  # Single-task mode (backward compatible)
        "SCORE_FIELDS": "",  # Multi-task mode: comma-separated list of score fields
        "N_BINS": "1000",
        "ACCURACY": "1e-5",
    },
    framework_requirements={
        "scikit-learn": ">=0.23.2,<1.0.0",
        "pandas": ">=1.2.0,<2.0.0",
        "numpy": ">=1.20.0",
    },
    description="""Contract for percentile model calibration processing step.
    
    The percentile model calibration step performs percentile score mapping calibration
    to convert raw model scores to calibrated percentile values using ROC curve analysis.
    Supports both single-task and multi-task calibration with full backward compatibility.
    
    Input Structure:
    - /opt/ml/processing/input/eval_data: Evaluation dataset with model prediction scores
      Supported formats: CSV, TSV, Parquet
      Priority order: eval_predictions.csv, eval_predictions_with_comparison.csv,
                     predictions.csv, predictions.parquet, predictions.json, processed_data.csv
    - /opt/ml/code/calibration: Optional calibration configuration directory containing
      standard_calibration_dictionary.json (falls back to built-in default if not provided)
    
    Output Structure:
    - /opt/ml/processing/output/calibration: Percentile score mapping artifacts
      * Single-task: percentile_score.pkl
      * Multi-task: percentile_score_{task_name}.pkl (per task) + percentile_score.pkl (first task, for compatibility)
    - /opt/ml/processing/output/metrics: Calibration quality metrics and statistics
      * calibration_metrics.json with per-task metrics and aggregate statistics
    - /opt/ml/processing/output/calibrated_data: Dataset with calibrated percentile scores
      * Format preserved from input (CSV/TSV/Parquet)
      * Adds {score_field}_percentile column(s) for calibrated values
    
    Environment Variables (at least one score field variable required):
    - SCORE_FIELD: Name of the prediction score column to calibrate (single-task mode)
      Example: "prob_class_1"
      Default: "prob_class_1"
    - SCORE_FIELDS: Comma-separated list of score columns to calibrate (multi-task mode)
      Example: "task_0_prob,task_1_prob,task_2_prob"
      Priority: If both SCORE_FIELD and SCORE_FIELDS are set, SCORE_FIELDS takes precedence
    - N_BINS: Number of bins for calibration analysis (optional, default=1000)
    - ACCURACY: Accuracy threshold for calibration mapping (optional, default=1e-5)
    
    Modes:
    1. Single-Task Mode (backward compatible):
       - Set SCORE_FIELD="prob_class_1"
       - Calibrates one score column
       - Outputs: percentile_score.pkl, calibration_metrics.json, calibrated_data with {field}_percentile column
    
    2. Multi-Task Mode:
       - Set SCORE_FIELDS="task_0_prob,task_1_prob,task_2_prob"
       - Calibrates multiple score columns independently using the same calibration dictionary
       - Outputs: percentile_score_{task}.pkl per task, percentile_score.pkl (first task),
                 calibration_metrics.json with per-task and aggregate metrics,
                 calibrated_data with {field}_percentile columns for each task
    
    Calibration Process:
    The script uses ROC curve analysis to map raw scores to percentile values based on a
    calibration dictionary that defines score thresholds and target volume ratios. Each task
    is calibrated independently, allowing different score distributions while using the same
    target calibration mapping.
    
    Error Handling:
    - Missing score fields are reported and skipped (calibration continues with valid fields)
    - NaN values are preserved in output but excluded from calibration
    - Invalid score ranges (outside [0,1]) are clipped with warnings
    - Constant scores (std < 1e-10) are skipped with error messages
    """,
)

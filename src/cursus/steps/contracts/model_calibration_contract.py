#!/usr/bin/env python
"""Script Contract for Model Calibration Step.

This file defines the contract for the model calibration processing script,
specifying input/output paths, environment variables, and required dependencies.
"""

from ...core.base.contract_base import ScriptContract

MODEL_CALIBRATION_CONTRACT = ScriptContract(
    entry_point="model_calibration.py",
    expected_input_paths={"evaluation_data": "/opt/ml/processing/input/eval_data"},
    expected_output_paths={
        "calibration_output": "/opt/ml/processing/output/calibration",
        "metrics_output": "/opt/ml/processing/output/metrics",
        "calibrated_data": "/opt/ml/processing/output/calibrated_data",
    },
    required_env_vars=["CALIBRATION_METHOD", "LABEL_FIELD", "IS_BINARY"],
    optional_env_vars={
        "SCORE_FIELD": "prob_class_1",
        "SCORE_FIELDS": "",
        "MONOTONIC_CONSTRAINT": "True",
        "GAM_SPLINES": "10",
        "ERROR_THRESHOLD": "0.05",
        "NUM_CLASSES": "2",
        "SCORE_FIELD_PREFIX": "prob_class_",
        "MULTICLASS_CATEGORIES": "[0, 1]",
        "CALIBRATION_SAMPLE_POINTS": "1000",
        "USE_SECURE_PYPI": "false",
    },
    framework_requirements={
        "scikit-learn": ">=0.23.2,<1.0.0",
        "pandas": ">=1.2.0,<2.0.0",
        "numpy": ">=1.20.0",
        "pygam": ">=0.8.0",
        "matplotlib": ">=3.3.0",
    },
    description="""Contract for model calibration processing step.
    
    The model calibration step takes a trained model's raw prediction scores and
    calibrates them to better reflect true probabilities, which is essential for
    risk-based decision-making, threshold setting, and confidence in model outputs.
    Supports binary classification, multi-class classification, and multi-task scenarios.
    
    Input Structure:
    - /opt/ml/processing/input/eval_data: Evaluation dataset with ground truth labels and model predictions
    
    Output Structure:
    - /opt/ml/processing/output/calibration: Calibration mapping and artifacts (per-task calibrators for multi-task)
    - /opt/ml/processing/output/metrics: Calibration quality metrics (aggregate metrics for multi-task)
    - /opt/ml/processing/output/calibrated_data: Dataset with calibrated probabilities
    
    Environment Variables (Required):
    - CALIBRATION_METHOD: Method to use for calibration (gam, isotonic, platt)
    - LABEL_FIELD: Name of the label column
    - IS_BINARY: Whether this is a binary classification task (true/false)
    
    Environment Variables (Optional):
    - SCORE_FIELD: Name of the prediction score column for single-task binary classification (default: prob_class_1)
    - SCORE_FIELDS: Comma-separated list of score fields for multi-task binary classification (e.g., "task1_prob,task2_prob,task3_prob")
      * If provided, enables multi-task mode and applies calibration independently to each task
      * Takes precedence over SCORE_FIELD when both are set
      * Requires IS_BINARY=true (multi-class multi-task not supported)
    - MONOTONIC_CONSTRAINT: Whether to enforce monotonicity in GAM (default: True)
    - GAM_SPLINES: Number of splines for GAM (default: 10)
    - ERROR_THRESHOLD: Acceptable calibration error threshold (default: 0.05)
    - CALIBRATION_SAMPLE_POINTS: Number of sample points for lookup table generation (default: 1000)
    - NUM_CLASSES: Number of classes for multi-class classification (default: 2)
    - SCORE_FIELD_PREFIX: Prefix for probability columns in multi-class scenario (default: prob_class_)
    - MULTICLASS_CATEGORIES: JSON string of class names/values for multi-class (default: [0, 1])
    - USE_SECURE_PYPI: Whether to use secure CodeArtifact PyPI for package installation (default: false)
    
    Multi-Task Support:
    - Use SCORE_FIELDS for calibrating multiple independent binary tasks with shared calibration method
    - Each task gets its own calibrator model saved as calibration_model_{task_name}.pkl
    - Aggregate metrics computed across all tasks for overall performance assessment
    - Output includes per-task metrics and aggregated statistics
    """,
)

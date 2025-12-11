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
    required_env_vars=["CALIBRATION_METHOD", "IS_BINARY", "LABEL_FIELD"],
    optional_env_vars={
        "SCORE_FIELD": "prob_class_1",
        "SCORE_FIELDS": "",
        "TASK_LABEL_NAMES": "",  # Required when SCORE_FIELDS is provided
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
    - IS_BINARY: Whether this is a binary classification task (true/false)
    - LABEL_FIELD: Name of the main label column
      * For single-task mode: this is the only label field used
      * For multi-task mode: this represents the main task label field
    
    Environment Variables (Optional):
    
    Single-Task Binary:
    - SCORE_FIELD: Name of the prediction score column (default: prob_class_1)
    
    Multi-Task Binary:
    - SCORE_FIELDS: Comma-separated list of score fields (e.g., "task1_prob,task2_prob,task3_prob")
      * Enables multi-task mode and applies calibration independently to each task
      * Takes precedence over SCORE_FIELD when both are set
      * Requires IS_BINARY=true (multi-class multi-task not supported)
      * LABEL_FIELD represents the main task in multi-task mode
    - TASK_LABEL_NAMES: Comma-separated list of label fields for each task (e.g., "task1_true,task2_true,task3_true")
      * REQUIRED when SCORE_FIELDS is provided (multi-task mode)
      * Must match length of SCORE_FIELDS
      * NOT required for single-task mode (when only SCORE_FIELD is used)
      * Validation rules:
        - Single value without comma: Must equal LABEL_FIELD
        - Multiple values with comma: LABEL_FIELD must be included in the list
      * Example: LABEL_FIELD="task1_true", SCORE_FIELDS="task1_prob,task2_prob", 
        TASK_LABEL_NAMES="task1_true,task2_true"
    
    Multi-Class Single-Task:
    - NUM_CLASSES: Number of classes (default: 2)
    - SCORE_FIELD_PREFIX: Prefix for probability columns (default: prob_class_)
    - MULTICLASS_CATEGORIES: JSON string of class names/values (default: [0, 1])
    
    Calibration Configuration:
    - MONOTONIC_CONSTRAINT: Whether to enforce monotonicity in GAM (default: True)
    - GAM_SPLINES: Number of splines for GAM (default: 10)
    - ERROR_THRESHOLD: Acceptable calibration error threshold (default: 0.05)
    - CALIBRATION_SAMPLE_POINTS: Number of sample points for lookup table generation (default: 1000)
    
    Infrastructure:
    - USE_SECURE_PYPI: Whether to use secure CodeArtifact PyPI for package installation (default: false)
    
    Multi-Task Output Structure:
    - Each task produces: calibration_model_{task_name}.pkl in /opt/ml/processing/output/calibration
    - Metrics include per-task calibration quality and aggregate statistics across all tasks
    - Calibrated data contains all original columns plus calibrated_{task_name} for each task
    
    Supported Scenarios:
    1. Single-task binary: One score field with one label field
    2. Multi-class single-task: Multiple score fields (one per class) with categorical labels
    3. Multi-task binary: Multiple independent binary tasks, each with its own score and label field
    """,
)

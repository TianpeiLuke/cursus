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
    required_env_vars=["CALIBRATION_METHOD", "LABEL_FIELD", "SCORE_FIELD", "IS_BINARY"],
    optional_env_vars={
        "MONOTONIC_CONSTRAINT": "True",
        "GAM_SPLINES": "10",
        "ERROR_THRESHOLD": "0.05",
        "NUM_CLASSES": "2",
        "SCORE_FIELD_PREFIX": "prob_class_",
        "MULTICLASS_CATEGORIES": "[0, 1]",
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
    Supports both binary and multi-class classification scenarios.
    
    Input Structure:
    - /opt/ml/processing/input/eval_data: Evaluation dataset with ground truth labels and model predictions
    
    Output Structure:
    - /opt/ml/processing/output/calibration: Calibration mapping and artifacts
    - /opt/ml/processing/output/metrics: Calibration quality metrics
    - /opt/ml/processing/output/calibrated_data: Dataset with calibrated probabilities
    
    Environment Variables:
    - CALIBRATION_METHOD: Method to use for calibration (gam, isotonic, platt)
    - LABEL_FIELD: Name of the label column
    - SCORE_FIELD: Name of the prediction score column (for binary classification)
    - IS_BINARY: Whether this is a binary classification task (true/false)
    - MONOTONIC_CONSTRAINT: Whether to enforce monotonicity in GAM (optional)
    - GAM_SPLINES: Number of splines for GAM (optional)
    - ERROR_THRESHOLD: Acceptable calibration error threshold (optional)
    - NUM_CLASSES: Number of classes for multi-class classification (optional, default=2)
    - SCORE_FIELD_PREFIX: Prefix for probability columns in multi-class scenario (optional)
    - MULTICLASS_CATEGORIES: JSON string of class names/values for multi-class (optional)
    """,
)

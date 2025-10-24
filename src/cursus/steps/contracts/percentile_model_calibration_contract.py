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
        "calibration_config": "/opt/ml/code/calibration"
    },
    expected_output_paths={
        "calibration_output": "/opt/ml/processing/output/calibration",
        "metrics_output": "/opt/ml/processing/output/metrics",
        "calibrated_data": "/opt/ml/processing/output/calibrated_data",
    },
    required_env_vars=["SCORE_FIELD"],
    optional_env_vars={
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
    This replicates the functionality of percentile_score_mapping.py but follows the
    cursus framework patterns with environment variable configuration and standardized
    I/O channels.
    
    Input Structure:
    - /opt/ml/processing/input/eval_data: Evaluation dataset with model prediction scores
    - /opt/ml/code/calibration: Optional calibration configuration directory containing
      standard_calibration_dictionary.json (falls back to built-in default if not provided)
    
    Output Structure:
    - /opt/ml/processing/output/calibration: Percentile score mapping artifacts (percentile_score.pkl)
    - /opt/ml/processing/output/metrics: Calibration quality metrics and statistics
    - /opt/ml/processing/output/calibrated_data: Dataset with calibrated percentile scores
    
    Environment Variables:
    - SCORE_FIELD: Name of the prediction score column to calibrate (required)
    - N_BINS: Number of bins for calibration analysis (optional, default=1000)
    - ACCURACY: Accuracy threshold for calibration mapping (optional, default=1e-5)
    
    The script expects the evaluation data to contain a file named 'processed_data.csv.out'
    or falls back to the first supported data file (.csv, .parquet, .json) in the directory.
    The calibration uses ROC curve analysis to map raw scores to percentile values based
    on a calibration dictionary that defines score thresholds and target volume ratios.
    """,
)

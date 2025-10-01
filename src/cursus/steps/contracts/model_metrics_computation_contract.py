"""
Model Metrics Computation Script Contract

Defines the contract for the model metrics computation script that loads prediction data,
computes comprehensive performance metrics, generates visualizations, and creates detailed reports.
"""

from ...core.base.contract_base import ScriptContract

MODEL_METRICS_COMPUTATION_CONTRACT = ScriptContract(
    entry_point="model_metrics_computation.py",
    expected_input_paths={
        "eval_output": "/opt/ml/processing/input/eval_data",
    },
    expected_output_paths={
        "metrics_output": "/opt/ml/processing/output/metrics",
        "plots_output": "/opt/ml/processing/output/plots",
    },
    expected_arguments={
        # No expected arguments - job_type comes from config
    },
    required_env_vars=["ID_FIELD", "LABEL_FIELD"],
    optional_env_vars={
        "AMOUNT_FIELD": "order_amount",
        "INPUT_FORMAT": "auto",
        "COMPUTE_DOLLAR_RECALL": "true",
        "COMPUTE_COUNT_RECALL": "true",
        "DOLLAR_RECALL_FPR": "0.1",
        "COUNT_RECALL_CUTOFF": "0.1",
        "GENERATE_PLOTS": "true",
    },
    framework_requirements={
        "pandas": ">=1.3.0",
        "numpy": ">=1.21.0",
        "scikit-learn": ">=1.0.0",
        "matplotlib": ">=3.5.0",
    },
    description="""
    Model metrics computation script that:
    1. Loads prediction data from various formats (CSV, Parquet, JSON)
    2. Validates prediction data schema and structure
    3. Computes comprehensive standard ML metrics (AUC-ROC, precision, recall, F1)
    4. Calculates domain-specific metrics (dollar recall, count recall)
    5. Generates performance visualizations (ROC curves, PR curves, distributions)
    6. Creates detailed reports with insights and recommendations
    7. Supports both binary and multiclass classification
    
    Input Structure:
    - /opt/ml/processing/input/eval_data: Prediction data directory containing:
      - predictions.csv, predictions.parquet, or predictions.json: Prediction data with:
        - ID column (configurable via ID_FIELD)
        - Label column (configurable via LABEL_FIELD)
        - Prediction probability columns (prob_class_0, prob_class_1, etc.)
        - Optional amount column (configurable via AMOUNT_FIELD)
      - eval_predictions.csv: Alternative prediction file from xgboost_model_eval
    
    Output Structure:
    - /opt/ml/processing/output/metrics/metrics.json: Standard metrics in JSON format
    - /opt/ml/processing/output/metrics/metrics_summary.txt: Human-readable metrics summary
    - /opt/ml/processing/output/metrics/metrics_report.json: Comprehensive report with insights
    - /opt/ml/processing/output/metrics/metrics_summary.txt: Detailed text summary
    - /opt/ml/processing/output/plots/roc_curve.jpg: ROC curve visualization
    - /opt/ml/processing/output/plots/pr_curve.jpg: Precision-Recall curve visualization
    - /opt/ml/processing/output/plots/score_distribution.jpg: Score distribution plot
    - /opt/ml/processing/output/plots/threshold_analysis.jpg: Threshold analysis plot
    - /opt/ml/processing/output/plots/class_*_roc_curve.jpg: Per-class ROC curves (multiclass)
    - /opt/ml/processing/output/plots/class_*_pr_curve.jpg: Per-class PR curves (multiclass)
    - /opt/ml/processing/output/plots/multiclass_roc_curves.jpg: Combined multiclass ROC curves
    
    Environment Variables:
    - ID_FIELD (required): Name of the ID column in prediction data
    - LABEL_FIELD (required): Name of the label column in prediction data
    - AMOUNT_FIELD (optional): Name of the amount column for dollar recall computation
    - INPUT_FORMAT (optional): Preferred input format ("csv", "parquet", "json", "auto")
    - COMPUTE_DOLLAR_RECALL (optional): Enable dollar recall computation ("true"/"false")
    - COMPUTE_COUNT_RECALL (optional): Enable count recall computation ("true"/"false")
    - DOLLAR_RECALL_FPR (optional): False positive rate for dollar recall (default: "0.1")
    - COUNT_RECALL_CUTOFF (optional): Cutoff percentile for count recall (default: "0.1")
    - GENERATE_PLOTS (optional): Enable plot generation ("true"/"false")
    
    Arguments:
    - job_type: Type of metrics computation job to perform (e.g., "evaluation", "validation")
    
    Features:
    - Multi-format Support: Automatically detects and loads CSV, Parquet, and JSON files
    - Binary/Multiclass: Automatically detects classification type and computes appropriate metrics
    - Domain Metrics: Computes business-specific metrics like dollar and count recall
    - Comprehensive Reporting: Generates detailed reports with actionable insights
    - Visualization: Creates publication-quality plots and charts
    - Flexible Configuration: Extensive environment variable configuration options
    - Error Handling: Robust error handling with detailed validation and logging
    
    Compatibility:
    - Input: Compatible with output from xgboost_model_eval.py and xgboost_model_inference.py
    - Output: Provides metrics in same format as xgboost_model_eval.py plus enhanced reporting
    - Framework: Works with any ML framework predictions (not limited to XGBoost)
    """,
)

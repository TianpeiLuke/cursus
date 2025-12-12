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
        # Basic field configuration
        "AMOUNT_FIELD": "order_amount",
        "INPUT_FORMAT": "auto",
        # Multi-task configuration
        "SCORE_FIELDS": "",  # Comma-separated score fields for multi-task (e.g., "isFraud_prob,isAbuse_prob")
        "SCORE_FIELD": "",  # Single score field for backward compatibility
        "TASK_LABEL_NAMES": "",  # Optional explicit task labels (comma-separated)
        # Domain metrics configuration
        "COMPUTE_DOLLAR_RECALL": "true",
        "COMPUTE_COUNT_RECALL": "true",
        "DOLLAR_RECALL_FPR": "0.1",
        "COUNT_RECALL_CUTOFF": "0.1",
        # Visualization configuration
        "GENERATE_PLOTS": "true",
        # Comparison mode configuration (single-task)
        "COMPARISON_MODE": "false",
        "PREVIOUS_SCORE_FIELD": "",
        # Multi-task comparison mode configuration
        "PREVIOUS_SCORE_FIELDS": "",  # Comma-separated previous score fields (e.g., "isFraud_v1,isAbuse_v1")
        # Comparison mode options
        "COMPARISON_METRICS": "all",
        "STATISTICAL_TESTS": "true",
        "COMPARISON_PLOTS": "true",
        # Installation configuration
        "USE_SECURE_PYPI": "false",
    },
    framework_requirements={
        "pandas": ">=1.3.0",
        "numpy": ">=1.21.0",
        "scikit-learn": ">=1.0.0",
        "matplotlib": ">=3.5.0",
        "scipy": ">=1.7.0",
    },
    description="""
    Model metrics computation script that:
    1. Loads prediction data from various formats (CSV, Parquet, JSON)
    2. Validates prediction data schema and structure
    3. Computes comprehensive standard ML metrics (AUC-ROC, precision, recall, F1)
    4. Calculates domain-specific metrics (dollar recall, count recall)
    5. Optionally compares performance with previous model scores
    6. Generates performance visualizations (ROC curves, PR curves, distributions)
    7. Creates detailed reports with insights and recommendations
    8. Supports single-task, multi-task, binary, and multiclass classification
    
    ===== CLASSIFICATION MODE DETECTION =====
    
    Single-Task Mode (Auto-detected):
    - Triggered when: SCORE_FIELD is set OR SCORE_FIELDS contains one field
    - Environment Variables:
      * SCORE_FIELD: Single score field name (e.g., "prob_class_1")
      * LABEL_FIELD: Single label field name (e.g., "label")
    - Behavior: Computes metrics for one classification task
    - Output: Single set of metrics and visualizations
    
    Multi-Task Mode (Auto-detected):
    - Triggered when: SCORE_FIELDS contains 2+ comma-separated fields
    - Environment Variables:
      * SCORE_FIELDS: Comma-separated score fields (e.g., "isFraud_prob,isAbuse_prob,isScam_prob")
      * TASK_LABEL_NAMES: Optional explicit labels (e.g., "isFraud,isAbuse,isScam")
      * If TASK_LABEL_NAMES not provided: labels inferred by removing "_prob" suffix
    - Behavior: Computes per-task metrics + aggregate metrics (mean/median across tasks)
    - Output: Per-task metrics, aggregate metrics, task-prefixed visualizations
    
    ===== INPUT STRUCTURE =====
    
    Single-Task Input:
    - /opt/ml/processing/input/eval_data: Prediction data directory containing:
      - predictions.{csv,parquet,json}: Prediction data with:
        - ID column (configurable via ID_FIELD)
        - Label column (configurable via LABEL_FIELD)
        - Score column (configurable via SCORE_FIELD)
        - Optional: prob_class_0, prob_class_1 columns
        - Optional: amount column (configurable via AMOUNT_FIELD)
        - Optional: previous model score column (for comparison mode)
    
    Multi-Task Input:
    - /opt/ml/processing/input/eval_data: Prediction data directory containing:
      - predictions.{csv,parquet,json}: Prediction data with:
        - ID column (configurable via ID_FIELD)
        - Multiple score columns (configurable via SCORE_FIELDS)
        - Multiple label columns (configurable via TASK_LABEL_NAMES or inferred)
        - Optional: amount column (configurable via AMOUNT_FIELD)
        - Optional: previous score columns (for multi-task comparison mode)
    
    ===== OUTPUT STRUCTURE =====
    
    Single-Task Output:
    - /opt/ml/processing/output/metrics/metrics.json: Standard metrics
      * auc_roc, average_precision, f1_score, precision, recall
      * threshold-based metrics at 0.3, 0.5, 0.7
      * optional: dollar_recall, count_recall
      * optional: comparison metrics (auc_delta, correlation, etc.)
    - /opt/ml/processing/output/plots/:
      * roc_curve.jpg, pr_curve.jpg
      * score_distribution.jpg, threshold_analysis.jpg
      * comparison_roc_curves.jpg, comparison_pr_curves.jpg (if comparison mode)
      * score_scatter_plot.jpg, score_distributions.jpg (if comparison mode)
    
    Multi-Task Output:
    - /opt/ml/processing/output/metrics/metrics.json: Per-task + aggregate metrics
      * aggregate: {mean_auc_roc, median_auc_roc, mean_average_precision, etc.}
      * task_{label}: {auc_roc, average_precision, f1_score, etc.} for each task
      * task_{label}_{metric}: comparison metrics for each task (if comparison mode)
      * aggregate_comparison: {mean_auc_delta, mean_correlation, etc.} (if comparison mode)
    - /opt/ml/processing/output/plots/:
      * task_{label}_roc_curve.jpg for each task
      * task_{label}_pr_curve.jpg for each task
      * task_{label}_comparison_roc_curves.jpg (if comparison mode)
      * task_{label}_comparison_pr_curves.jpg (if comparison mode)
      * task_{label}_score_scatter_plot.jpg (if comparison mode)
      * task_{label}_score_distributions.jpg (if comparison mode)
    
    ===== ENVIRONMENT VARIABLES =====
    
    Required:
    - ID_FIELD: Name of the ID column in prediction data
    
    Optional - Basic Configuration:
    - LABEL_FIELD: Label column name (default: "label", single-task only)
    - AMOUNT_FIELD: Amount column for dollar recall (default: None)
    - INPUT_FORMAT: Input format - "csv", "parquet", "json", "auto" (default: "auto")
    
    Optional - Multi-Task Configuration:
    - SCORE_FIELDS: Comma-separated score fields for multi-task mode
      * Example: "isFraud_prob,isAbuse_prob,isScam_prob"
      * When set with 2+ fields: enables multi-task mode
    - SCORE_FIELD: Single score field for single-task mode (backward compatible)
      * Example: "prob_class_1"
      * When set: enables single-task mode
    - TASK_LABEL_NAMES: Explicit task label field names (comma-separated)
      * Example: "isFraud,isAbuse,isScam"
      * If not provided: inferred from score field names by removing "_prob" suffix
    
    Optional - Domain Metrics:
    - COMPUTE_DOLLAR_RECALL: Enable dollar recall (default: "true")
    - COMPUTE_COUNT_RECALL: Enable count recall (default: "true")
    - DOLLAR_RECALL_FPR: False positive rate for dollar recall (default: "0.1")
    - COUNT_RECALL_CUTOFF: Cutoff percentile for count recall (default: "0.1")
    
    Optional - Visualization:
    - GENERATE_PLOTS: Enable plot generation (default: "true")
    
    Optional - Single-Task Comparison Mode:
    - COMPARISON_MODE: Enable comparison (default: "false")
    - PREVIOUS_SCORE_FIELD: Column with previous model scores
      * Example: "prev_model_score"
      * Enables comparison metrics and plots
    
    Optional - Multi-Task Comparison Mode:
    - PREVIOUS_SCORE_FIELDS: Comma-separated previous score fields
      * Example: "isFraud_v1,isAbuse_v1,isScam_v1"
      * Must match length of SCORE_FIELDS
      * Enables per-task comparison + aggregate comparison
    
    Optional - Comparison Mode Options:
    - COMPARISON_METRICS: "all" or "basic" (default: "all")
    - STATISTICAL_TESTS: Enable statistical tests (default: "true")
    - COMPARISON_PLOTS: Enable comparison plots (default: "true")
    
    Optional - Installation:
    - USE_SECURE_PYPI: Use secure CodeArtifact PyPI (default: "false")
    
    ===== ARGUMENTS =====
    
    - job_type: Type of metrics computation job (e.g., "evaluation", "validation")
    
    ===== COMPARISON MODE FEATURES =====
    
    Single-Task Comparison:
    - Performance delta metrics (AUC-ROC, AP, F1 deltas)
    - Statistical tests (McNemar, paired t-test, Wilcoxon)
    - Correlation analysis (Pearson, Spearman)
    - Comparison visualizations (ROC, PR, scatter, distributions)
    
    Multi-Task Comparison:
    - Per-task performance deltas (AUC-ROC, AP, F1 deltas per task)
    - Per-task correlation analysis
    - Aggregate comparison metrics (mean/median deltas across tasks)
    - Per-task comparison visualizations (4 plots per task)
    
    ===== KEY FEATURES =====
    
    - Multi-Task Support: Automatically detects and handles multi-task classification
    - Single-Task Support: Backward compatible with existing single-task pipelines
    - Multi-format: Auto-detects CSV, Parquet, and JSON input formats
    - Binary/Multiclass: Automatically detects classification type
    - Domain Metrics: Business-specific metrics (dollar/count recall)
    - Model Comparison: Compares new vs previous model (single-task or multi-task)
    - Comprehensive Reporting: Detailed reports with actionable insights
    - Visualization: Publication-quality plots with task prefixes
    - Flexible Configuration: Extensive environment variable options
    - Error Handling: Robust validation and logging
    - Smart Inference: Label fields inferred from score field names when not explicit
    
    ===== COMPATIBILITY =====
    
    - Input: Compatible with xgboost_model_eval.py, lightgbmmt_model_eval.py outputs
    - Output: Same format as xgboost_model_eval.py plus enhanced multi-task reporting
    - Framework: Works with any ML framework (XGBoost, LightGBM, PyTorch, etc.)
    - Comparison: Full comparison support for both single-task and multi-task
    
    ===== USAGE EXAMPLES =====
    
    Single-Task Example:
    ```
    ID_FIELD="transaction_id"
    LABEL_FIELD="is_fraud"
    SCORE_FIELD="fraud_score"
    AMOUNT_FIELD="transaction_amount"
    ```
    
    Multi-Task Example:
    ```
    ID_FIELD="transaction_id"
    SCORE_FIELDS="isFraud_prob,isAbuse_prob,isScam_prob"
    TASK_LABEL_NAMES="isFraud,isAbuse,isScam"  # Optional, can infer
    AMOUNT_FIELD="transaction_amount"
    ```
    
    Single-Task Comparison Example:
    ```
    ID_FIELD="transaction_id"
    SCORE_FIELD="fraud_score"
    PREVIOUS_SCORE_FIELD="fraud_score_v1"
    ```
    
    Multi-Task Comparison Example:
    ```
    ID_FIELD="transaction_id"
    SCORE_FIELDS="isFraud_prob,isAbuse_prob,isScam_prob"
    PREVIOUS_SCORE_FIELDS="isFraud_v1,isAbuse_v1,isScam_v1"
    ```
    """,
)

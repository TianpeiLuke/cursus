"""
Multi-Task Model Evaluation Script Contract

Defines the contract for the LightGBMMT multi-task model evaluation script that loads trained models,
processes evaluation data, and generates per-task and aggregate performance metrics and visualizations.
"""

from ...core.base.contract_base import ScriptContract

LIGHTGBMMT_MODEL_EVAL_CONTRACT = ScriptContract(
    entry_point="lightgbmmt_model_eval.py",
    expected_input_paths={
        "model_input": "/opt/ml/processing/input/model",
        "processed_data": "/opt/ml/processing/input/eval_data",
    },
    expected_output_paths={
        "eval_output": "/opt/ml/processing/output/eval",
        "metrics_output": "/opt/ml/processing/output/metrics",
    },
    expected_arguments={
        # No expected arguments - job_type comes from config
    },
    required_env_vars=["ID_FIELD", "TASK_LABEL_NAMES"],
    optional_env_vars={
        # Note: Comparison mode not yet implemented for multi-task
        # Future enhancement opportunity
    },
    framework_requirements={
        "pandas": ">=1.2.0,<2.0.0",
        "numpy": ">=1.21.0",
        "scikit-learn": ">=0.23.2,<1.0.0",
        "lightgbm": ">=3.3.0",
        "matplotlib": ">=3.0.0",
        "scipy": ">=1.7.0",
        "pyarrow": ">=4.0.0,<6.0.0",
    },
    description="""
    LightGBMMT multi-task model evaluation script that:
    1. Loads trained LightGBMMT model and preprocessing artifacts
    2. Loads and preprocesses evaluation data using risk tables and imputation
    3. Generates multi-task predictions (one probability per task)
    4. Computes per-task and aggregate performance metrics
    5. Creates ROC and Precision-Recall curve visualizations for each task
    6. Saves predictions, metrics, and plots preserving input format
    
    Input Structure:
    - /opt/ml/processing/input/model: Model artifacts directory containing:
      - lightgbmmt_model.txt: Trained LightGBMMT model (multi-task)
      - risk_table_map.pkl: Risk table mappings for categorical features
      - impute_dict.pkl: Imputation dictionary for numerical features
      - feature_columns.txt: Feature column names and order
      - hyperparameters.json: Model hyperparameters and metadata
      - training_state.json: Training state (multi-task specific)
      - weight_evolution.json: Task weight evolution (optional, multi-task specific)
      - feature_importance.json: Feature importance scores (optional)
    - /opt/ml/processing/input/eval_data: Evaluation data (CSV, TSV, or Parquet files)
      - Must contain all task label columns specified in TASK_LABEL_NAMES
      - Must contain ID column specified in ID_FIELD
    
    Output Structure:
    - /opt/ml/processing/output/eval/eval_predictions.[csv|tsv|parquet]: Multi-task predictions
      Format: id, task1_true, task1_prob, task2_true, task2_prob, ...
      Example: id, isFraud_true, isFraud_prob, isCCfrd_true, isCCfrd_prob, isDDfrd_true, isDDfrd_prob
    
    - /opt/ml/processing/output/metrics/metrics.json: Comprehensive metrics including:
      * Per-task metrics: {"task_0_isFraud": {"auc_roc": 0.85, "average_precision": 0.78, "f1_score": 0.72}}
      * Aggregate metrics: {"aggregate": {"mean_auc_roc": 0.87, "median_auc_roc": 0.86, ...}}
    
    - /opt/ml/processing/output/metrics/metrics_summary.txt: Human-readable metrics summary
    
    - /opt/ml/processing/output/metrics/task_<i>_<taskname>_roc.jpg: Per-task ROC curves
      Example: task_0_isFraud_roc.jpg, task_1_isCCfrd_roc.jpg, task_2_isDDfrd_roc.jpg
    
    - /opt/ml/processing/output/metrics/task_<i>_<taskname>_pr.jpg: Per-task PR curves
      Example: task_0_isFraud_pr.jpg, task_1_isCCfrd_pr.jpg, task_2_isDDfrd_pr.jpg
    
    - /opt/ml/processing/output/metrics/_SUCCESS: Success marker file
    - /opt/ml/processing/output/metrics/_HEALTH: Health check file with timestamp
    
    Required Environment Variables:
    - ID_FIELD: Name of the ID column in evaluation data (e.g., "id", "transaction_id")
    - TASK_LABEL_NAMES: Comma-separated list or JSON array of task label column names
      * Comma format: "isFraud,isCCfrd,isDDfrd"
      * JSON format: '["isFraud","isCCfrd","isDDfrd"]'
      * Must match task_label_names from training hyperparameters
      * All specified columns must exist in evaluation data
    
    Arguments:
    - job_type: Type of evaluation job to perform (e.g., "evaluation", "validation")
    
    Multi-Task Features:
    - Supports any number of binary classification tasks (n_tasks >= 2)
    - Generates independent metrics for each task
    - Computes aggregate performance across all tasks (mean, median)
    - Creates separate visualization for each task
    - Preserves input data format (CSV, TSV, or Parquet) in output
    
    Per-Task Metrics:
    - AUC-ROC: Area under ROC curve for each task
    - Average Precision: Area under PR curve for each task
    - F1 Score: F1 score at 0.5 threshold for each task
    
    Aggregate Metrics:
    - Mean/Median AUC-ROC across all tasks
    - Mean/Median Average Precision across all tasks
    - Mean/Median F1 Score across all tasks
    
    Data Format Preservation:
    - Automatically detects input format (CSV, TSV, Parquet)
    - Preserves format in output predictions
    - Supports mixed formats across different inputs
    
    Error Handling:
    - Validates task labels exist in evaluation data
    - Handles single-class tasks gracefully (skips metrics/plots)
    - Creates failure markers on error for debugging
    - Comprehensive logging for troubleshooting
    
    Alignment with Training:
    - Uses identical preprocessing pipeline (risk tables, imputation)
    - Validates task_label_names consistency with hyperparameters
    - Supports same feature column ordering as training
    - Compatible with all LightGBMMT loss function types (fixed, adaptive, adaptive_kd)
    
    Note: Multi-task comparison mode (comparing with previous model scores per task) is not yet
    implemented but can be added as a future enhancement following the single-task pattern.
    """,
)

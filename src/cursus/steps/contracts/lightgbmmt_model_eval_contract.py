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
        "GENERATE_PLOTS": "true",
        "COMPARISON_MODE": "false",
        "PREVIOUS_SCORE_FIELDS": "",
        "COMPARISON_METRICS": "all",
        "STATISTICAL_TESTS": "true",
        "COMPARISON_PLOTS": "true",
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
    
    Optional Environment Variables:
    - GENERATE_PLOTS: Enable visualization generation (default: "true")
      * Set to "false" to skip plot generation when evaluating many tasks
      * Recommended to keep enabled for detailed analysis
    
    Optional Environment Variables (Comparison Mode):
    - COMPARISON_MODE: Enable multi-task model comparison functionality (default: "false")
    - PREVIOUS_SCORE_FIELDS: Comma-separated list of columns with previous model scores
      * Format: "prev_isFraud,prev_isCCfrd,prev_isDDfrd"
      * Must provide one field per task in same order as TASK_LABEL_NAMES
      * All specified columns must exist in evaluation data
    - COMPARISON_METRICS: Metrics to compute - "all" or "basic" (default: "all")
    - STATISTICAL_TESTS: Enable statistical significance tests per task (default: "true")
    - COMPARISON_PLOTS: Enable comparison visualizations per task (default: "true")
    
    Arguments:
    - job_type: Type of evaluation job to perform (e.g., "evaluation", "validation")
    
    Standard Mode Output (COMPARISON_MODE=false):
    - eval_predictions.[csv|tsv|parquet]: Multi-task predictions with true labels and probabilities
    - metrics.json: Comprehensive per-task and aggregate metrics
    - metrics_summary.txt: Human-readable metrics summary
    - task_{i}_{taskname}_roc.jpg: ROC curve per task (300 DPI)
    - task_{i}_{taskname}_pr.jpg: Precision-Recall curve per task (300 DPI)
    - task_{i}_{taskname}_score_distribution.jpg: Score distribution by class per task
    - task_{i}_{taskname}_threshold_analysis.jpg: F1/Precision/Recall vs threshold per task
    - multitask_combined_roc_curves.jpg: All tasks on single ROC plot
    - _SUCCESS: Success marker file
    - _HEALTH: Health check file with timestamp
    
    Comparison Mode Output (COMPARISON_MODE=true):
    - eval_predictions_with_comparison.[csv|tsv|parquet]: Enhanced predictions with:
      * {taskname}_true: True label
      * {taskname}_new_prob: New model probability
      * {taskname}_prev_prob: Previous model probability
      * {taskname}_score_diff: Score difference (new - previous)
    - metrics.json: Enhanced with comparison metrics per task and aggregate comparison
    - metrics_summary.txt: Enhanced summary with comparison statistics
    - multitask_comparison_report.txt: Executive summary with per-task recommendations
    - Per-task comparison plots (300 DPI):
      * {taskname}_comparison_roc.jpg: Side-by-side ROC curves
      * {taskname}_comparison_pr.jpg: Side-by-side PR curves
      * {taskname}_score_scatter.jpg: New vs previous score correlation
      * {taskname}_score_distributions.jpg: 4-panel distribution comparison
    - Standard per-task plots (if GENERATE_PLOTS=true)
    
    Multi-Task Features:
    - Supports any number of binary classification tasks (n_tasks >= 2)
    - Generates independent metrics for each task
    - Computes aggregate performance across all tasks (mean, median)
    - Creates separate visualizations for each task
    - Preserves input data format (CSV, TSV, or Parquet) in output
    
    Enhanced Per-Task Metrics:
    - Core metrics: AUC-ROC, Average Precision, F1 Score @ 0.5 threshold
    - Threshold analysis: Precision/Recall @ 0.3, 0.5, 0.7 thresholds
    - Optimal threshold: Threshold maximizing (TPR - FPR)
    - Max F1 score: Maximum F1 across all thresholds
    - F1 scores @ multiple thresholds: 0.3, 0.5, 0.7
    
    Aggregate Metrics:
    - Mean/Median AUC-ROC, Average Precision, F1 Score across all tasks
    - Mean/Median Max F1 Score and Optimal Threshold across tasks
    
    Comparison Features (per task):
    - Performance delta metrics: AUC-ROC, Average Precision, F1-score improvements
    - Statistical significance: McNemar's test, paired t-test, Wilcoxon test
    - Correlation analysis: Pearson and Spearman correlations
    - Score distribution comparison: Mean, std, agreement metrics
    - Comprehensive visualizations comparing task performance
    - Aggregate comparison: Mean/median deltas, tasks improved/degraded count
    
    Data Format Preservation:
    - Automatically detects input format (CSV, TSV, Parquet)
    - Preserves format in output predictions
    - Supports mixed formats across different inputs
    
    Error Handling:
    - Validates task labels exist in evaluation data
    - Validates previous score fields exist when comparison mode enabled
    - Guard rail: Disables comparison if PREVIOUS_SCORE_FIELDS empty
    - Handles single-class tasks gracefully (skips metrics/plots)
    - Creates failure markers on error for debugging
    - Comprehensive logging for troubleshooting
    
    Alignment with Training:
    - Uses identical preprocessing pipeline (risk tables, imputation)
    - Validates task_label_names consistency with hyperparameters
    - Supports same feature column ordering as training
    - Compatible with all LightGBMMT loss function types (fixed, adaptive, adaptive_kd)
    
    Performance Note:
    - Plot generation can be disabled via GENERATE_PLOTS=false for faster evaluation
    - Comparison mode adds per-task statistical tests and visualizations
    - All plots generated at 300 DPI for publication quality
    """,
)

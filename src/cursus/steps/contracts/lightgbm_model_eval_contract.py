"""
Model Evaluation Script Contract

Defines the contract for the LightGBM model evaluation script that loads trained models,
processes evaluation data, and generates performance metrics and visualizations.
"""

from ...core.base.contract_base import ScriptContract

LIGHTGBM_MODEL_EVAL_CONTRACT = ScriptContract(
    entry_point="lightgbm_model_eval.py",
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
    required_env_vars=["ID_FIELD", "LABEL_FIELD"],
    optional_env_vars={
        "COMPARISON_MODE": "false",
        "PREVIOUS_SCORE_FIELD": "",
        "COMPARISON_METRICS": "all",
        "STATISTICAL_TESTS": "true",
        "COMPARISON_PLOTS": "true",
    },
    framework_requirements={
        "pandas": ">=1.3.0",
        "numpy": ">=1.21.0",
        "scikit-learn": ">=1.0.0",
        "lightgbm": ">=3.3.0",
        "matplotlib": ">=3.5.0",
        "scipy": ">=1.7.0",
    },
    description="""
    LightGBM model evaluation script that:
    1. Loads trained LightGBM model and preprocessing artifacts
    2. Loads and preprocesses evaluation data using risk tables and imputation
    3. Generates predictions and computes performance metrics
    4. Creates ROC and Precision-Recall curve visualizations
    5. Optionally compares performance with previous model scores
    6. Saves predictions, metrics, plots, and comparison reports
    
    Input Structure:
    - /opt/ml/processing/input/model: Model artifacts directory containing:
      - lightgbm_model.txt: Trained LightGBM model
      - risk_table_map.pkl: Risk table mappings for categorical features
      - impute_dict.pkl: Imputation dictionary for numerical features
      - feature_columns.txt: Feature column names and order
      - hyperparameters.json: Model hyperparameters and metadata
    - /opt/ml/processing/input/eval_data: Evaluation data (CSV or Parquet files)
      - For comparison mode: must include column with previous model scores
    
    Standard Output Structure:
    - /opt/ml/processing/output/eval/eval_predictions.csv: Model predictions with probabilities
    - /opt/ml/processing/output/metrics/metrics.json: Performance metrics
    - /opt/ml/processing/output/metrics/roc_curve.jpg: ROC curve visualization
    - /opt/ml/processing/output/metrics/pr_curve.jpg: Precision-Recall curve visualization
    
    Additional Output Structure (Comparison Mode):
    - /opt/ml/processing/output/eval/eval_predictions_with_comparison.csv: Enhanced predictions with both models
    - /opt/ml/processing/output/metrics/comparison_report.txt: Executive summary with recommendations
    - /opt/ml/processing/output/metrics/comparison_roc_curves.jpg: Side-by-side ROC comparison
    - /opt/ml/processing/output/metrics/comparison_pr_curves.jpg: Side-by-side PR comparison
    - /opt/ml/processing/output/metrics/score_scatter_plot.jpg: Model score correlation analysis
    - /opt/ml/processing/output/metrics/score_distributions.jpg: 4-panel distribution comparison
    - /opt/ml/processing/output/metrics/new_model_*.jpg: Individual plots for new model
    - /opt/ml/processing/output/metrics/previous_model_*.jpg: Individual plots for previous model
    
    Required Environment Variables:
    - ID_FIELD: Name of the ID column in evaluation data
    - LABEL_FIELD: Name of the label column in evaluation data
    
    Optional Environment Variables (Comparison Mode):
    - COMPARISON_MODE: Enable model comparison functionality (default: "false")
    - PREVIOUS_SCORE_FIELD: Column name containing previous model scores (default: "previous_score")
    - COMPARISON_METRICS: Metrics to compute - "all" or "basic" (default: "all")
    - STATISTICAL_TESTS: Enable statistical significance tests (default: "true")
    - COMPARISON_PLOTS: Enable comparison visualizations (default: "true")
    
    Arguments:
    - job_type: Type of evaluation job to perform (e.g., "evaluation", "validation")
    
    Comparison Features:
    - Performance delta metrics (AUC-ROC, Average Precision, F1-score improvements)
    - Statistical significance testing (McNemar's test, paired t-test, Wilcoxon test)
    - Correlation analysis between model scores
    - Comprehensive visualizations comparing model performance
    - Automated recommendations for model deployment decisions
    
    Supports both binary and multiclass classification with appropriate metrics for each.
    Binary classification has full comparison functionality; multiclass has limited comparison support.
    """,
)

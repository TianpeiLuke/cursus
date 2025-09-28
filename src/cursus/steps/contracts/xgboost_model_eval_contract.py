"""
Model Evaluation Script Contract

Defines the contract for the XGBoost model evaluation script that loads trained models,
processes evaluation data, and generates performance metrics and visualizations.
"""

from ...core.base.contract_base import ScriptContract

XGBOOST_MODEL_EVAL_CONTRACT = ScriptContract(
    entry_point="xgboost_model_eval.py",
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
    optional_env_vars={},
    framework_requirements={
        "pandas": ">=1.3.0",
        "numpy": ">=1.21.0",
        "scikit-learn": ">=1.0.0",
        "xgboost": ">=1.6.0",
        "matplotlib": ">=3.5.0",
    },
    description="""
    XGBoost model evaluation script that:
    1. Loads trained XGBoost model and preprocessing artifacts
    2. Loads and preprocesses evaluation data using risk tables and imputation
    3. Generates predictions and computes performance metrics
    4. Creates ROC and Precision-Recall curve visualizations
    5. Saves predictions, metrics, and plots
    
    Input Structure:
    - /opt/ml/processing/input/model: Model artifacts directory containing:
      - xgboost_model.bst: Trained XGBoost model
      - risk_table_map.pkl: Risk table mappings for categorical features
      - impute_dict.pkl: Imputation dictionary for numerical features
      - feature_columns.txt: Feature column names and order
      - hyperparameters.json: Model hyperparameters and metadata
    - /opt/ml/processing/input/eval_data: Evaluation data (CSV or Parquet files)
    
    Output Structure:
    - /opt/ml/processing/output/eval/eval_predictions.csv: Model predictions with probabilities
    - /opt/ml/processing/output/metrics/metrics.json: Performance metrics
    - /opt/ml/processing/output/metrics/roc_curve.jpg: ROC curve visualization
    - /opt/ml/processing/output/metrics/pr_curve.jpg: Precision-Recall curve visualization
    
    Environment Variables:
    - ID_FIELD: Name of the ID column in evaluation data
    - LABEL_FIELD: Name of the label column in evaluation data
    
    Arguments:
    - job_type: Type of evaluation job to perform (e.g., "evaluation", "validation")
    
    Supports both binary and multiclass classification with appropriate metrics for each.
    """,
)

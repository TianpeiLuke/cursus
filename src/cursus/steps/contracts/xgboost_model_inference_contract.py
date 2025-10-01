"""
Model Inference Script Contract

Defines the contract for the XGBoost model inference script that loads trained models,
processes evaluation data, and generates predictions without metrics computation.
"""

from ...core.base.contract_base import ScriptContract

XGBOOST_MODEL_INFERENCE_CONTRACT = ScriptContract(
    entry_point="xgboost_model_inference.py",
    expected_input_paths={
        "model_input": "/opt/ml/processing/input/model",
        "processed_data": "/opt/ml/processing/input/eval_data",
    },
    expected_output_paths={
        "eval_output": "/opt/ml/processing/output/eval",
    },
    expected_arguments={
        # No expected arguments - job_type comes from config
    },
    required_env_vars=["ID_FIELD", "LABEL_FIELD"],
    optional_env_vars={
        "OUTPUT_FORMAT": "csv",
        "JSON_ORIENT": "records",
    },
    framework_requirements={
        "pandas": ">=1.3.0",
        "numpy": ">=1.21.0",
        "xgboost": ">=1.6.0",
    },
    description="""
    XGBoost model inference script that:
    1. Loads trained XGBoost model and preprocessing artifacts
    2. Loads and preprocesses evaluation data using risk tables and imputation
    3. Generates predictions using the trained model
    4. Saves predictions with original data in multiple formats (CSV, Parquet, JSON)
    
    This script focuses solely on inference (prediction generation) without metrics
    computation or visualization, enabling modular pipeline architectures where
    inference results can be cached, reused, and processed by different downstream
    components.
    
    Input Structure:
    - /opt/ml/processing/input/model: Model artifacts directory containing:
      - xgboost_model.bst: Trained XGBoost model
      - risk_table_map.pkl: Risk table mappings for categorical features
      - impute_dict.pkl: Imputation dictionary for numerical features
      - feature_columns.txt: Feature column names and order
      - hyperparameters.json: Model hyperparameters and metadata
    - /opt/ml/processing/input/eval_data: Evaluation data (CSV or Parquet files)
    
    Output Structure:
    - /opt/ml/processing/output/eval/predictions.csv: Model predictions with probabilities (default)
    - /opt/ml/processing/output/eval/predictions.parquet: Parquet format output (if OUTPUT_FORMAT=parquet)
    - /opt/ml/processing/output/eval/predictions.json: JSON format output (if OUTPUT_FORMAT=json)
    
    Output Format:
    The output preserves all original data columns and adds prediction columns:
    - Original columns: All columns from input data (ID, label, features, metadata)
    - Prediction columns: prob_class_0, prob_class_1, ... (probability scores for each class)
    
    Environment Variables:
    - ID_FIELD (required): Name of the ID column in evaluation data
    - LABEL_FIELD (required): Name of the label column in evaluation data
    - OUTPUT_FORMAT (optional): Output format - "csv", "parquet", or "json" (default: "csv")
    - JSON_ORIENT (optional): JSON orientation - "records", "index", "values", "split", "table" (default: "records")
    
    Arguments:
    - job_type: Type of inference job to perform (e.g., "inference", "validation")
    
    Multi-Format Support:
    - CSV: Human-readable, compatible with existing workflows
    - Parquet: Efficient binary format for large datasets with compression
    - JSON: Universal compatibility with configurable orientations:
      - records: [{column -> value}, ..., {column -> value}]
      - index: {index -> {column -> value}}
      - values: [[row values], [row values], ...]
      - split: {'index': [index], 'columns': [columns], 'data': [values]}
      - table: {'schema': {schema}, 'data': [{row}, {row}, ...]}
    
    Downstream Integration:
    The inference output is designed to be consumed by:
    - Model metrics computation scripts (expects ID + label + prob_class_* columns)
    - Model calibration scripts (expects LABEL_FIELD + prob_class_* columns)
    - Model deployment validation (structured predictions for testing)
    
    Supports both binary and multiclass classification with consistent output format.
    """,
)

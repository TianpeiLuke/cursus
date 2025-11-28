"""
Multi-Task Model Inference Script Contract

Defines the contract for the LightGBMMT multi-task model inference script that loads trained models,
processes input data, and generates multi-task predictions WITHOUT evaluation, metrics, or plots.
"""

from ...core.base.contract_base import ScriptContract

LIGHTGBMMT_MODEL_INFERENCE_CONTRACT = ScriptContract(
    entry_point="lightgbmmt_model_inference.py",
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
    required_env_vars=["ID_FIELD", "TASK_LABEL_NAMES"],
    optional_env_vars={
        "OUTPUT_FORMAT": "csv",  # Output format: csv, tsv, parquet, or json
        "JSON_ORIENT": "records",  # JSON orientation when OUTPUT_FORMAT=json
    },
    framework_requirements={
        "pandas": ">=1.2.0,<2.0.0",
        "numpy": ">=1.21.0",
        "scikit-learn": ">=0.23.2,<1.0.0",
        "lightgbm": ">=3.3.0",
        "scipy": ">=1.7.0",
        "pyarrow": ">=4.0.0,<6.0.0",
    },
    description="""
    LightGBMMT multi-task model inference script that:
    1. Loads trained LightGBMMT model and preprocessing artifacts
    2. Loads and preprocesses inference data using risk tables and imputation
    3. Generates multi-task predictions (one probability per task)
    4. Saves predictions preserving input format (NO metrics, NO plots)
    
    Input Structure:
    - /opt/ml/processing/input/model: Model artifacts directory containing:
      - lightgbmmt_model.txt: Trained LightGBMMT model (multi-task)
      - risk_table_map.pkl: Risk table mappings for categorical features
      - impute_dict.pkl: Imputation dictionary for numerical features
      - feature_columns.txt: Feature column names and order
      - hyperparameters.json: Model hyperparameters and metadata
    
    - /opt/ml/processing/input/eval_data: Input data for inference (CSV, TSV, or Parquet files)
      - Must contain ID column specified in ID_FIELD
      - Task label columns (specified in TASK_LABEL_NAMES) are OPTIONAL for inference
        * If present, they will be included in output for reference
        * If absent, only ID and prediction columns will be in output
      - Must contain feature columns expected by the model
    
    Output Structure:
    - /opt/ml/processing/output/eval/predictions.[csv|tsv|parquet|json]: Multi-task predictions
      Format: id, [optional_task_labels], task1_prob, task2_prob, ...
      
      Example WITH labels:
      id, isFraud, isCCfrd, isDDfrd, isFraud_prob, isCCfrd_prob, isDDfrd_prob
      1,  1,       0,       0,       0.85,         0.12,         0.05
      2,  0,       1,       0,       0.03,         0.92,         0.08
      
      Example WITHOUT labels (inference only):
      id, isFraud_prob, isCCfrd_prob, isDDfrd_prob
      1,  0.85,         0.12,         0.05
      2,  0.03,         0.92,         0.08
    
    - /opt/ml/processing/output/eval/_SUCCESS: Success marker file
    - /opt/ml/processing/output/eval/_HEALTH: Health check file with timestamp
    
    Required Environment Variables:
    - ID_FIELD: Name of the ID column in input data (e.g., "id", "transaction_id")
      * Must exist in input data
      * Will be preserved in output
    
    - TASK_LABEL_NAMES: Comma-separated list or JSON array of task names
      * Comma format: "isFraud,isCCfrd,isDDfrd"
      * JSON format: '["isFraud","isCCfrd","isDDfrd"]'
      * Must match task_label_names from training hyperparameters
      * Used to name prediction columns: {task_name}_prob
      * Corresponding label columns in input data are OPTIONAL
    
    Optional Environment Variables:
    - OUTPUT_FORMAT: Output file format (default: auto-detect from input)
      * "csv": Save as CSV file
      * "tsv": Save as TSV (tab-separated) file
      * "parquet": Save as Parquet file
      * "json": Save as JSON file
      * If not set, uses input file format
    
    - JSON_ORIENT: JSON orientation when OUTPUT_FORMAT=json (default: "records")
      * "records": [{col1: val1, col2: val2}, ...]
      * "index": {index -> {col -> val}}
      * "split": {index -> [values], columns -> [labels], data -> [values]}
      * "values": [[val1, val2], ...]
    
    Arguments:
    - job_type: Type of inference job to perform (e.g., "inference", "batch_prediction")
    
    Multi-Task Features:
    - Supports any number of binary classification tasks (n_tasks >= 2)
    - Generates independent predictions for each task
    - Output includes one probability column per task: {task_name}_prob
    - Preserves all original input columns (including optional labels)
    - NO metrics computation - pure inference only
    - NO plots or visualizations
    
    Prediction Output:
    - Each task gets one probability column: {task_name}_prob
    - Values are probabilities in range [0.0, 1.0]
    - Higher values indicate higher confidence in positive class
    - All original columns from input data are preserved
    - Column order: [original_columns, task1_prob, task2_prob, ...]
    
    Data Format Preservation:
    - Automatically detects input format (CSV, TSV, Parquet)
    - Preserves format in output by default
    - Can override output format via OUTPUT_FORMAT env var
    - Handles JSON output with configurable orientation
    
    Error Handling:
    - Validates ID_FIELD exists in input data
    - Validates TASK_LABEL_NAMES is properly formatted
    - Handles missing feature columns gracefully
    - Creates failure markers on error for debugging
    - Comprehensive logging for troubleshooting
    
    Alignment with Training and Evaluation:
    - Uses identical preprocessing pipeline (risk tables, imputation)
    - Uses same task names from training hyperparameters
    - Supports same feature column ordering as training
    - Compatible with all LightGBMMT loss function types (fixed, adaptive, adaptive_kd)
    - Can process data with or without ground truth labels
    
    Comparison with Evaluation Script:
    - Inference: Predictions only (fast, no labels required)
    - Evaluation: Predictions + Metrics + Plots (slower, labels required)
    
    Use Cases:
    - Batch prediction on unlabeled data
    - Real-time scoring (after loading artifacts once)
    - A/B testing with multiple model versions
    - Pre-computing scores for downstream systems
    - Generating predictions for model monitoring
    
    Performance Considerations:
    - Faster than evaluation (no metrics computation)
    - Lighter memory footprint (no plot generation)
    - Suitable for large-scale batch inference
    - Can process data without ground truth labels
    - Preserves input format to avoid conversion overhead
    
    Example Usage Workflow:
    1. Train multi-task model with LightGBMMT training step
    2. Save model artifacts (lightgbmmt_model.txt, preprocessors, etc.)
    3. Run inference on new data (with or without labels)
    4. Get predictions for all tasks in single pass
    5. Use predictions for downstream decision-making
    
    Integration Points:
    - Input: Output from data preprocessing steps
    - Output: Input to downstream scoring/decision systems
    - Model: Output from LightGBMMT training step
    - Format: Compatible with SageMaker Processing/Batch Transform
    """,
)

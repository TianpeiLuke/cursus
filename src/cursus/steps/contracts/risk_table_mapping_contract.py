"""
Risk Table Mapping Script Contract

Defines the contract for the risk table mapping script that creates risk tables
for categorical features and handles missing value imputation for numeric features.
"""

from ...core.base.contract_base import ScriptContract

RISK_TABLE_MAPPING_CONTRACT = ScriptContract(
    entry_point="risk_table_mapping.py",
    expected_input_paths={
        "input_data": "/opt/ml/processing/input/data",
        "hyperparameters_s3_uri": "/opt/ml/code/hyperparams",
        "model_artifacts_input": "/opt/ml/processing/input/model_artifacts",  # Optional for non-training modes
    },
    expected_output_paths={
        "processed_data": "/opt/ml/processing/output/data",
        "model_artifacts_output": "/opt/ml/processing/output/model_artifacts",
    },
    expected_arguments={
        # No expected arguments - job_type comes from config
    },
    required_env_vars=[
        # No strictly required environment variables - script has defaults
    ],
    optional_env_vars={
        "SMOOTH_FACTOR": "0.01",
        "COUNT_THRESHOLD": "5",
        "MAX_UNIQUE_THRESHOLD": "100",
    },
    framework_requirements={
        "pandas": ">=1.3.0",
        "numpy": ">=1.21.0",
        "scikit-learn": ">=1.0.0",
    },
    description="""
    Risk table mapping script that:
    1. Creates risk tables for categorical features based on target variable correlation
    2. Handles missing value imputation for numeric features
    3. Supports both training mode (fit and transform) and inference mode (transform only)
    4. Applies smoothing and count thresholds for robust risk estimation
    5. Saves fitted artifacts for reuse in inference
    
    Input Structure:
    - /opt/ml/processing/input/data: Data files from missing_value_imputation or tabular preprocessing
      - Training mode: train/, test/, val/ subdirectories with processed data
      - Other modes: job_type/ subdirectory with processed data
    - /opt/ml/code/hyperparams: Configuration files
      - hyperparameters.json: Model configuration including category risk parameters (cat_field_list, smooth_factor, count_threshold)
    - /opt/ml/processing/input/model_artifacts: Pre-trained model artifacts (for non-training modes)
      - risk_table_map.pkl: Risk table mappings for categorical features
      - impute_dict.pkl: Imputation values (from previous missing_value_imputation step)
      - Other artifacts from previous steps (parameter accumulator pattern)
    
    Output Structure:
    - /opt/ml/processing/output/data/{split}/{split}_processed_data.csv: Transformed data by split
    - /opt/ml/processing/output/model_artifacts/risk_table_map.pkl: Risk table mappings
    - /opt/ml/processing/output/model_artifacts/hyperparameters.json: Copy of hyperparameters
    - /opt/ml/processing/output/model_artifacts/impute_dict.pkl: Copied from previous step (accumulator pattern)
    
    Job Types (from config):
    - training: Fits risk tables on training data, transforms all splits
    - validation/testing/calibration: Uses pre-trained risk tables, transforms single split
    
    Training Mode:
    - Fits risk tables on training data
    - Transforms train/test/val splits
    - Saves risk tables and imputation models
    
    Non-Training Modes:
    - Loads pre-trained risk tables from model_artifacts_input
    - Copies all existing artifacts from previous steps (parameter accumulator pattern)
    - Transforms data using loaded artifacts
    - Maintains the same output structure as training mode
    
    Parameter Accumulator Pattern:
    - Copies all existing artifacts from model_artifacts_input to model_artifacts_output
    - Adds its own risk table mappings to the accumulated artifacts
    - Enables downstream steps to access all preprocessing parameters in one location
    """,
)

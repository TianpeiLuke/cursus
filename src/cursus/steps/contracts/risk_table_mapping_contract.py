"""
Risk Table Mapping Script Contract

Defines the contract for the risk table mapping script that creates risk tables
for categorical features and handles missing value imputation for numeric features.
"""

from ...core.base.contract_base import ScriptContract

RISK_TABLE_MAPPING_CONTRACT = ScriptContract(
    entry_point="risk_table_mapping.py",
    expected_input_paths={
        "data_input": "/opt/ml/processing/input/data",
        "hyperparameters_s3_uri": "/opt/ml/processing/input/config",
        "risk_tables": "/opt/ml/processing/input/risk_tables",  # Optional for non-training modes
    },
    expected_output_paths={
        "processed_data": "/opt/ml/processing/output",
        "risk_tables": "/opt/ml/processing/output",  # Shared path with processed_data
    },
    expected_arguments={
        # No expected arguments - job_type comes from config
    },
    required_env_vars=[
        # No strictly required environment variables - script has defaults
    ],
    optional_env_vars={
        # No optional environment variables needed
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
    - /opt/ml/processing/input/data: Data files from tabular preprocessing
      - Training mode: train/, test/, val/ subdirectories with processed data
      - Other modes: job_type/ subdirectory with processed data
    - /opt/ml/processing/input/config: Configuration files
      - config.json: Model configuration including category risk parameters
      - metadata.csv: Variable metadata with types and imputation strategies
      - job_type: Configuration parameter specifying job type (training, validation, testing, calibration)
    - /opt/ml/processing/input/risk_tables: Pre-trained risk tables (for non-training modes)
      - bin_mapping.pkl: Risk table mappings for categorical features
      - missing_value_imputation.pkl: Imputation values for numeric features
    
    Output Structure:
    - /opt/ml/processing/output/{split}/{split}_processed_data.csv: Transformed data by split
    - /opt/ml/processing/output/bin_mapping.pkl: Risk table mappings for categorical features
    - /opt/ml/processing/output/missing_value_imputation.pkl: Imputation values for numeric features
    - /opt/ml/processing/output/config.pkl: Serialized configuration with metadata
    
    Job Types (from config):
    - training: Fits risk tables on training data, transforms all splits
    - validation/testing/calibration: Uses pre-trained risk tables, transforms single split
    
    Training Mode:
    - Fits risk tables on training data
    - Transforms train/test/val splits
    - Saves risk tables and imputation models
    
    Non-Training Modes:
    - Loads pre-trained risk tables and imputation models
    - Transforms data using loaded artifacts
    - Maintains the same output structure as training mode
    """,
)

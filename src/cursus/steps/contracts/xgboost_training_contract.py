"""
XGBoost Training Script Contract

Defines the contract for the XGBoost training script that handles tabular data
training with risk table mapping and numerical imputation.
"""

from .training_script_contract import TrainingScriptContract

XGBOOST_TRAIN_CONTRACT = TrainingScriptContract(
    entry_point="xgboost_training.py",
    expected_input_paths={
        "input_path": "/opt/ml/input/data",
        "hyperparameters_s3_uri": "/opt/ml/code/hyperparams/hyperparameters.json",
        "model_artifacts_input": "/opt/ml/input/data/model_artifacts_input",  # Optional: Pre-computed preprocessing artifacts
    },
    expected_output_paths={
        "model_output": "/opt/ml/model",
        "evaluation_output": "/opt/ml/output/data",
    },
    expected_arguments={
        # No expected arguments - using standard paths from contract
    },
    required_env_vars=[
        # No strictly required environment variables - script uses hyperparameters.json
    ],
    optional_env_vars={
        "USE_SECURE_PYPI": "true",  # Controls PyPI source for package installation (default: secure CodeArtifact)
        "USE_PRECOMPUTED_IMPUTATION": "false",  # If true, uses pre-computed imputation artifacts and skips inline computation
        "USE_PRECOMPUTED_RISK_TABLES": "false",  # If true, uses pre-computed risk table artifacts and skips inline computation
        "USE_PRECOMPUTED_FEATURES": "false",  # If true, uses pre-computed feature selection and skips inline computation
        "REGION": "NA",  # Region identifier (NA/EU/FE) for loading region-specific hyperparameters
    },
    framework_requirements={
        "boto3": ">=1.26.0",
        "xgboost": "==1.7.6",
        "scikit-learn": ">=0.23.2,<1.0.0",
        "pandas": ">=1.2.0,<2.0.0",
        "pyarrow": ">=4.0.0,<6.0.0",
        "beautifulsoup4": ">=4.9.3",
        "flask": ">=2.0.0,<3.0.0",
        "pydantic": ">=2.0.0,<3.0.0",
        "typing-extensions": ">=4.2.0",
        "matplotlib": ">=3.0.0",
        "numpy": ">=1.19.0",
    },
    description="""
    XGBoost training script for tabular data classification that:
    1. Loads training, validation, and test datasets from split directories
    2. Optionally uses pre-computed preprocessing artifacts from previous steps OR computes inline
    3. Applies numerical imputation using mean strategy for missing values (inline or pre-computed)
    4. Fits risk tables on categorical features using training data (inline or pre-computed)
    5. Transforms all datasets using preprocessing artifacts (skipped if data already processed)
    6. Trains XGBoost model with configurable hyperparameters
    7. Supports both binary and multiclass classification
    8. Handles class weights for imbalanced datasets
    9. Evaluates model performance with comprehensive metrics
    10. Saves model artifacts and preprocessing components
    11. Generates prediction files and performance visualizations
    
    Pre-Computed Artifact Support:
    - USE_PRECOMPUTED_IMPUTATION=true: Input data already imputed, loads impute_dict.pkl, skips transformation
    - USE_PRECOMPUTED_RISK_TABLES=true: Input data already risk-mapped, loads risk_table_map.pkl, skips transformation
    - USE_PRECOMPUTED_FEATURES=true: Input data already feature-selected, loads selected_features.json, skips selection
    - Default (all false): Computes all preprocessing inline and transforms data
    - Validates data state matches environment variable flags
    - All artifacts (pre-computed or inline) packaged into model.tar.gz
    
    Input Structure:
    - /opt/ml/input/data: Root directory containing train/val/test subdirectories
      - /opt/ml/input/data/train: Training data files (.csv, .parquet, .json)
      - /opt/ml/input/data/val: Validation data files
      - /opt/ml/input/data/test: Test data files
    - /opt/ml/input/data/model_artifacts_input: Optional directory with pre-computed artifacts
      - /opt/ml/input/data/model_artifacts_input/impute_dict.pkl: Pre-computed imputation parameters
      - /opt/ml/input/data/model_artifacts_input/risk_table_map.pkl: Pre-computed risk tables
      - /opt/ml/input/data/model_artifacts_input/selected_features.json: Pre-computed feature selection
    - /opt/ml/input/data/config/hyperparameters.json: Model configuration (optional)
    
    Output Structure:
    - /opt/ml/model: Model artifacts directory
      - /opt/ml/model/xgboost_model.bst: Trained XGBoost model
      - /opt/ml/model/risk_table_map.pkl: Risk table mappings for categorical features
      - /opt/ml/model/impute_dict.pkl: Imputation values for numerical features
      - /opt/ml/model/feature_importance.json: Feature importance scores
      - /opt/ml/model/feature_columns.txt: Ordered feature column names
      - /opt/ml/model/hyperparameters.json: Model hyperparameters
    - /opt/ml/output/data: Evaluation results directory
      - /opt/ml/output/data/val.tar.gz: Validation predictions and metrics
      - /opt/ml/output/data/test.tar.gz: Test predictions and metrics
    
    Contract aligned with step specification:
    - Inputs: input_path (required), hyperparameters_s3_uri (optional)
    - Outputs: model_output (primary), evaluation_output (secondary)
    
    Hyperparameters (via JSON config):
    - Data fields: tab_field_list, cat_field_list, label_name, id_name
    - Model: is_binary, num_classes, class_weights
    - XGBoost: eta, gamma, max_depth, subsample, colsample_bytree, lambda_xgb, alpha_xgb
    - Training: num_round, early_stopping_rounds
    - Risk tables: smooth_factor, count_threshold
    
    Binary Classification:
    - Uses binary:logistic objective
    - Supports scale_pos_weight for class imbalance
    - Generates ROC and PR curves
    - Computes AUC-ROC, Average Precision, F1-Score
    
    Multiclass Classification:
    - Uses multi:softprob objective
    - Supports sample weights for class imbalance
    - Generates per-class and aggregate metrics
    - Computes micro/macro averaged metrics
    
    Risk Table Processing:
    - Fits risk tables on categorical features using target correlation
    - Applies smoothing and count thresholds for robust estimation
    - Transforms categorical values to risk scores
    
    Numerical Imputation:
    - Uses mean imputation strategy for missing numerical values
    - Fits imputation on training data only
    - Applies same imputation to validation and test sets
    """,
)

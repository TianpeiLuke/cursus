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
    optional_env_vars={},
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
    2. Applies numerical imputation using mean strategy for missing values
    3. Fits risk tables on categorical features using training data
    4. Transforms all datasets using fitted preprocessing artifacts
    5. Trains XGBoost model with configurable hyperparameters
    6. Supports both binary and multiclass classification
    7. Handles class weights for imbalanced datasets
    8. Evaluates model performance with comprehensive metrics
    9. Saves model artifacts and preprocessing components
    10. Generates prediction files and performance visualizations
    
    Input Structure:
    - /opt/ml/input/data: Root directory containing train/val/test subdirectories
      - /opt/ml/input/data/train: Training data files (.csv, .parquet, .json)
      - /opt/ml/input/data/val: Validation data files
      - /opt/ml/input/data/test: Test data files
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

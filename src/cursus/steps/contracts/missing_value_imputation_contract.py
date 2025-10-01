"""
Missing Value Imputation Script Contract

Defines the contract for the missing value imputation script that handles missing values
in tabular data using simple statistical methods (mean, median, mode, constant).
Supports both training mode (fit and transform) and inference mode (transform only).
"""

from ...core.base.contract_base import ScriptContract

MISSING_VALUE_IMPUTATION_CONTRACT = ScriptContract(
    entry_point="missing_value_imputation.py",
    expected_input_paths={
        "data_input": "/opt/ml/processing/input/data",
        "imputation_params_input": "/opt/ml/processing/input/imputation_params",  # Optional for non-training modes
    },
    expected_output_paths={
        "data_output": "/opt/ml/processing/output",
    },
    expected_arguments={
        # No expected arguments - job_type comes from config
    },
    required_env_vars=[
        "LABEL_FIELD",
    ],
    optional_env_vars={
        "DEFAULT_NUMERICAL_STRATEGY": "mean",
        "DEFAULT_CATEGORICAL_STRATEGY": "mode", 
        "DEFAULT_TEXT_STRATEGY": "mode",
        "NUMERICAL_CONSTANT_VALUE": "0",
        "CATEGORICAL_CONSTANT_VALUE": "Unknown",
        "TEXT_CONSTANT_VALUE": "Unknown",
        "CATEGORICAL_PRESERVE_DTYPE": "true",
        "AUTO_DETECT_CATEGORICAL": "true",
        "CATEGORICAL_UNIQUE_RATIO_THRESHOLD": "0.1",
        "VALIDATE_FILL_VALUES": "true",
        "EXCLUDE_COLUMNS": "",
    },
    framework_requirements={
        "pandas": ">=1.3.0",
        "numpy": ">=1.21.0",
        "scikit-learn": ">=1.0.0",
    },
    description="""
    Missing value imputation script that:
    1. Handles missing values using simple statistical methods (mean, median, mode, constant)
    2. Supports numerical, categorical, and text/string data types with appropriate strategies
    3. Provides pandas-safe imputation values to avoid NA interpretation issues
    4. Supports both training mode (fit and transform) and inference mode (transform only)
    5. Saves fitted imputation parameters for reuse in inference jobs
    6. Generates comprehensive reports and quality metrics
    
    Input Structure:
    - /opt/ml/processing/input/data: Preprocessed data from tabular_preprocessing
      - Training mode: train/, test/, val/ subdirectories with *_processed_data.csv files
      - Other modes: {job_type}/ subdirectory with {job_type}_processed_data.csv file
    - /opt/ml/processing/input/imputation_params: Pre-trained imputation parameters (for non-training modes)
      - imputation_parameters.pkl: Fitted SimpleImputer objects and statistics
      - imputation_summary.json: Human-readable summary of imputation process
    
    Output Structure:
    - /opt/ml/processing/output/{split}/{split}_processed_data.csv: Imputed data by split
    - /opt/ml/processing/output/imputation_parameters.pkl: Fitted imputation parameters (training mode only)
    - /opt/ml/processing/output/imputation_summary.json: Human-readable imputation summary
    - /opt/ml/processing/output/imputation_report.json: Comprehensive analysis report
    - /opt/ml/processing/output/imputation_summary.txt: Text summary with recommendations
    
    Job Types:
    - training: Fits imputation parameters on training data, transforms all splits, saves parameters
    - validation/testing/calibration: Uses pre-trained parameters, transforms single split
    
    Data Type Support:
    - Numerical: mean, median, constant imputation strategies
    - Categorical: mode, constant imputation with pandas categorical dtype preservation
    - Text/String: mode, constant, empty string strategies with pandas-safe validation
    
    Training Mode:
    - Fits imputation parameters on training data only
    - Transforms train/test/val splits using fitted parameters
    - Saves imputation parameters and comprehensive reports
    
    Non-Training Modes:
    - Loads pre-trained imputation parameters from imputation_params_input
    - Transforms data using loaded parameters for consistent imputation
    - Maintains same output structure as training mode
    
    Dependency Path Compatibility:
    - Input: Consumes output from tabular_preprocessing (processed_data)
    - Output: Provides imputed data for downstream steps like risk_table_mapping
    - Path Structure: Maintains same directory structure as tabular_preprocessing
      - tabular_preprocessing outputs: /opt/ml/processing/output/{split}/{split}_processed_data.csv
      - missing_value_imputation inputs: /opt/ml/processing/input/data/{split}/{split}_processed_data.csv
      - missing_value_imputation outputs: /opt/ml/processing/output/{split}/{split}_processed_data.csv
    
    Environment Variables:
    - LABEL_FIELD: Target column name to exclude from imputation
    - DEFAULT_*_STRATEGY: Default imputation strategies by data type
    - *_CONSTANT_VALUE: Constant values for constant imputation strategies
    - CATEGORICAL_*: Configuration for categorical data handling
    - AUTO_DETECT_CATEGORICAL: Enable automatic categorical vs text detection
    - VALIDATE_FILL_VALUES: Enable pandas NA value validation
    - EXCLUDE_COLUMNS: Comma-separated list of columns to exclude from imputation
    - COLUMN_STRATEGY_<column_name>: Column-specific imputation strategies
    
    Pandas-Safe Features:
    - Validates fill values against pandas NA interpretation list
    - Automatically replaces problematic values with safe alternatives
    - Supports custom validation and safe value suggestions
    """,
)

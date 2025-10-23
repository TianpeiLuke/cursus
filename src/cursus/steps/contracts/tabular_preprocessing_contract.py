"""
Tabular Preprocessing Script Contract

Defines the contract for the tabular preprocessing script that handles data loading,
cleaning, and splitting for training/validation/testing.
"""

from ...core.base.contract_base import ScriptContract

TABULAR_PREPROCESSING_CONTRACT = ScriptContract(
    entry_point="tabular_preprocessing.py",
    expected_input_paths={
        "DATA": "/opt/ml/processing/input/data",
        "SIGNATURE": "/opt/ml/processing/input/signature",
    },
    expected_output_paths={"processed_data": "/opt/ml/processing/output"},
    expected_arguments={
        # No expected arguments - job_type comes from config
    },
    required_env_vars=["TRAIN_RATIO", "TEST_VAL_RATIO"],
    optional_env_vars={
        "LABEL_FIELD": "",
        "CATEGORICAL_COLUMNS": "",
        "NUMERICAL_COLUMNS": "",
        "TEXT_COLUMNS": "",
        "DATE_COLUMNS": "",
    },
    framework_requirements={
        "pandas": ">=1.3.0",
        "numpy": ">=1.21.0",
        "scikit-learn": ">=1.0.0",
    },
    description="""
    Tabular preprocessing script that:
    1. Combines data shards from input directory
    2. Loads column signature for CSV/TSV files if provided
    3. Cleans and processes label field
    4. Splits data into train/test/val for training jobs
    5. Outputs processed CSV files by split
    
    Contract aligned with actual script implementation:
    - Inputs: 
      * DATA (required) - reads from /opt/ml/processing/input/data
      * SIGNATURE (optional) - reads from /opt/ml/processing/input/signature
    - Outputs: processed_data (primary) - writes to /opt/ml/processing/output
    - Arguments: job_type (required) - defines processing mode (training/validation/testing)
    
    Script Implementation Details:
    - Reads data shards (CSV, JSON, Parquet) from input/data directory
    - Loads signature file containing column names for CSV/TSV files
    - Supports gzipped files and various formats
    - Uses signature column names for CSV/TSV files when available
    - Processes labels (converts categorical to numeric if needed)
    - Splits data based on job_type (training creates train/test/val splits)
    - Outputs processed files to split subdirectories under /opt/ml/processing/output
    
    Signature File Format:
    - CSV format with comma-separated column names
    - Applied only to CSV/TSV files, ignored for JSON/Parquet formats
    - Backward compatible - works without signature file
    """,
)

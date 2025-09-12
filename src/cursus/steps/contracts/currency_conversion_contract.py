"""
Currency Conversion Script Contract

Defines the contract for the currency conversion script that converts monetary values
across different currencies based on marketplace information and exchange rates.
"""

from ...core.base.contract_base import ScriptContract

CURRENCY_CONVERSION_CONTRACT = ScriptContract(
    entry_point="currency_conversion.py",
    expected_input_paths={"data_input": "/opt/ml/processing/input/data"},
    expected_output_paths={"converted_data": "/opt/ml/processing/output"},
    expected_arguments={
        "job-type": "training",  # Type of job (training, validation, testing, calibration)
        "mode": "per_split",  # Conversion mode (per_split or split_after_conversion)
        "train-ratio": "0.7",  # Training data ratio (when mode=split_after_conversion)
        "test-val-ratio": "0.5",  # Test/validation split ratio (when mode=split_after_conversion)
        "marketplace-id-col": "marketplace_id",  # Column containing marketplace IDs
        "currency-col": "",  # Optional existing currency column (empty string for optional)
        "default-currency": "USD",  # Default currency code
        "skip-invalid-currencies": "false",  # Skip rows with invalid currencies
        "enable-conversion": "true",  # Enable/disable conversion
        "n-workers": "50",  # Number of parallel workers
    },
    required_env_vars=[
        "CURRENCY_CONVERSION_VARS",
        "CURRENCY_CONVERSION_DICT",
        "MARKETPLACE_INFO",
        "LABEL_FIELD",
    ],
    optional_env_vars={"TRAIN_RATIO": "0.7", "TEST_VAL_RATIO": "0.5"},
    framework_requirements={
        "pandas": ">=1.3.0",
        "numpy": ">=1.21.0",
        "scikit-learn": ">=1.0.0",
    },
    description="""
    Currency conversion script that:
    1. Loads processed data from input splits (train/test/val or single split)
    2. Applies currency conversion to specified monetary variables
    3. Uses marketplace information to determine currency codes
    4. Supports parallel processing for performance
    5. Handles two modes: per-split conversion or conversion before re-splitting
    
    Input Structure:
    - /opt/ml/processing/input/data/{split}/{split}_processed_data.csv: Input data files
    - /opt/ml/processing/input/data/{split}/{split}_full_data.csv: Optional full data files
    
    Output Structure:
    - /opt/ml/processing/output/{split}/{split}_processed_data.csv: Converted processed data
    - /opt/ml/processing/output/{split}/{split}_full_data.csv: Converted full data (if exists)
    
    Environment Variables:
    - CURRENCY_CONVERSION_VARS: JSON list of variables requiring currency conversion
    - CURRENCY_CONVERSION_DICT: JSON dict mapping currency codes to exchange rates
    - MARKETPLACE_INFO: JSON dict mapping marketplace IDs to currency information
    - LABEL_FIELD: Name of the label column for stratified splitting
    - TRAIN_RATIO: Training data ratio (default: 0.7)
    - TEST_VAL_RATIO: Test/validation split ratio (default: 0.5)
    
    Command Line Arguments:
    - --job-type: Type of job (training, validation, testing, calibration)
    - --mode: Conversion mode (per_split or split_after_conversion)
    - --marketplace-id-col: Column containing marketplace IDs
    - --currency-col: Optional existing currency column
    - --default-currency: Default currency code (default: USD)
    - --skip-invalid-currencies: Skip rows with invalid currencies
    - --enable-conversion: Enable/disable conversion (default: true)
    - --n-workers: Number of parallel workers (default: 50)
    """,
)

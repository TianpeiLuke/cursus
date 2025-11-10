"""
Currency Conversion Script Contract

Defines the contract for the currency conversion script that converts monetary values
across different currencies based on marketplace information and exchange rates.
"""

from ...core.base.contract_base import ScriptContract

CURRENCY_CONVERSION_CONTRACT = ScriptContract(
    entry_point="currency_conversion.py",
    expected_input_paths={"input_data": "/opt/ml/processing/input/data"},
    expected_output_paths={"processed_data": "/opt/ml/processing/output"},
    expected_arguments={
        # No expected arguments - job_type comes from config
    },
    required_env_vars=[
        "CURRENCY_CONVERSION_VARS",
        "CURRENCY_CONVERSION_DICT",
    ],
    optional_env_vars={
        "CURRENCY_CODE_FIELD": "",
        "MARKETPLACE_ID_FIELD": "",
        "DEFAULT_CURRENCY": "USD",
        "N_WORKERS": "50",
    },
    framework_requirements={
        "pandas": ">=1.3.0",
        "numpy": ">=1.21.0",
        "scikit-learn": ">=1.0.0",
    },
    description="""
    Currency conversion script that:
    1. Loads processed data from input splits (train/test/val or single split)
    2. Applies currency conversion to specified monetary variables
    3. Uses marketplace information or direct currency codes to determine currency codes
    4. Supports parallel processing for performance
    
    Input Structure:
    - /opt/ml/processing/input/data/{split}/{split}_processed_data.csv: Input data files
    
    Output Structure:
    - /opt/ml/processing/output/{split}/{split}_processed_data.csv: Converted processed data
    
    Environment Variables:
    - CURRENCY_CONVERSION_VARS: JSON list of variables requiring currency conversion
    - CURRENCY_CONVERSION_DICT: JSON dict with mappings containing marketplace_id, currency_code, and conversion_rate
    - CURRENCY_CODE_FIELD: Name of column containing currency codes directly (optional but at least one of CURRENCY_CODE_FIELD or MARKETPLACE_ID_FIELD required)
    - MARKETPLACE_ID_FIELD: Name of column containing marketplace IDs (optional but at least one of CURRENCY_CODE_FIELD or MARKETPLACE_ID_FIELD required)
    - DEFAULT_CURRENCY: Default currency code (default: USD)
    - N_WORKERS: Number of parallel workers (default: 50)
    
    Command Line Arguments:
    - --job_type: Type of job (training, validation, testing, calibration)    
    
    Note: At least one of CURRENCY_CODE_FIELD or MARKETPLACE_ID_FIELD must be provided for currency lookup. If neither is provided, conversion will be skipped.
    """,
)

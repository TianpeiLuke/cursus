"""
MIMS Payload Script Contract

Defines the contract for the MIMS payload generation script that creates
sample payloads and metadata for model inference testing.
"""

from ...core.base.contract_base import ScriptContract

PAYLOAD_CONTRACT = ScriptContract(
    entry_point="payload.py",
    expected_input_paths={"model_input": "/opt/ml/processing/input/model"},
    expected_output_paths={"payload_sample": "/opt/ml/processing/output"},
    expected_arguments={
        # No expected arguments - using standard paths from contract
    },
    required_env_vars=[
        # No strictly required environment variables - script has defaults
    ],
    optional_env_vars={
        # Only these environment variables are actually used by the script:
        "CONTENT_TYPES": "application/json",
        "DEFAULT_NUMERIC_VALUE": "0.0",
        "DEFAULT_TEXT_VALUE": "DEFAULT_TEXT",
        # Special field environment variables follow pattern SPECIAL_FIELD_<fieldname>
    },
    framework_requirements={
        "python": ">=3.7"
        # Uses only standard library modules: json, logging, os, tarfile, tempfile, pathlib, enum, typing, datetime
    },
    description="""
    MIMS payload generation script that:
    1. Extracts hyperparameters from model artifacts (model.tar.gz or directory)
    2. Creates model variable list from field information
    3. Generates sample payloads in multiple formats (JSON, CSV)
    4. Archives payload files for deployment
    
    Note: This script extracts pipeline name, version, and model objective from hyperparameters,
    not from environment variables. It does not use PIPELINE_NAME, REGION, PAYLOAD_S3_KEY, or 
    BUCKET_NAME environment variables.
    
    Input Structure:
    - /opt/ml/processing/input/model: Model artifacts containing hyperparameters.json
    
    Output Structure:
    - /tmp/mims_payload_work/payload_sample/: Sample payload files (temporary)
    - /opt/ml/processing/output/: Output directory containing payload.tar.gz file
    
    Environment Variables:
    - CONTENT_TYPES: Comma-separated list of content types (default: "application/json")
    - DEFAULT_NUMERIC_VALUE: Default value for numeric fields (default: "0.0")
    - DEFAULT_TEXT_VALUE: Default value for text fields (default: "DEFAULT_TEXT")
    - SPECIAL_FIELD_<fieldname>: Custom values for specific fields
    
    Arguments:
    - mode: Operating mode for the script (default: "standard")
    """,
)

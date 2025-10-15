"""
Dummy Data Loading Script Contract

Defines the contract for the Dummy data loading script that processes
user-provided data instead of calling internal Cradle services.
This step serves as a drop-in replacement for CradleDataLoadingStep.
"""

from ...core.base.contract_base import ScriptContract

DUMMY_DATA_LOADING_CONTRACT = ScriptContract(
    entry_point="dummy_data_loading.py",
    expected_input_paths={
        "INPUT_DATA": "/opt/ml/processing/input/data",  # Input data channel
    },
    expected_output_paths={
        "SIGNATURE": "/opt/ml/processing/output/signature",
        "METADATA": "/opt/ml/processing/output/metadata",
        "DATA": "/opt/ml/processing/output/place_holder",  # Placeholder since actual data goes to S3
    },
    expected_arguments={
        # No expected arguments - using standard paths from contract
    },
    required_env_vars=[
        # No strictly required environment variables
    ],
    optional_env_vars={},
    framework_requirements={"python": ">=3.7", "boto3": ">=1.26.0"},
    description="""
    Dummy data loading script that:
    1. Reads user-provided data from input data channel
    2. Processes and validates the input data
    3. Writes output signature for data schema
    4. Writes metadata file with field type information
    5. Copies/processes data to output location
    
    Input Structure:
    - /opt/ml/processing/input/data: User-provided data files (CSV, Parquet, JSON)
    - Configuration is provided via the job configuration and not through input files
    - /opt/ml/processing/config/config: Data loading configuration is provided by the step creation process
    
    Output Structure:
    - /opt/ml/processing/output/signature/signature: Schema information for the loaded data
    - /opt/ml/processing/output/metadata/metadata: Metadata about fields (type information)
    - Data is processed and made available at the specified output location
    
    Environment Variables:
    - No environment variables required - data source is provided via INPUT_DATA input channel
    
    The script performs the following operations:
    - Reads user-provided data from the input data channel
    - Auto-detects data format (CSV, Parquet, JSON)
    - Generates schema signature based on the actual data
    - Writes metadata files with field type information
    - Processes and copies data to the output location
    - Processes data from the INPUT_DATA input channel
    
    This script is designed to replace CradleDataLoadingStep by processing
    user-provided data instead of calling internal Cradle services.
    As an Internal Node, this step requires input data dependencies.
    """,
)

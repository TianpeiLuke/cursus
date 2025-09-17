"""
Cradle Data Loading Script Contract

Defines the contract for the Cradle data loading script that downloads
data from Cradle data sources and processes it for use in ML pipelines.
"""

from ...core.base.contract_base import ScriptContract

CRADLE_DATA_LOADING_CONTRACT = ScriptContract(
    entry_point="scripts.py",
    expected_input_paths={
        # No inputs as this is a source node
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
    optional_env_vars={"OUTPUT_PATH": ""},  # Optional override for data output path
    framework_requirements={"python": ">=3.7"},
    description="""
    Cradle data loading script that:
    1. Reads data loading configuration from config file
    2. Writes output signature for data schema
    3. Writes metadata file with field type information
    4. Creates and executes a Cradle data load job
    5. Waits for job completion
    
    Input Structure:
    - Configuration is provided via the job configuration and not through input files
    - /opt/ml/processing/config/config: Data loading configuration is provided by the step creation process
    
    Output Structure:
    - /opt/ml/processing/output/signature/signature: Schema information for the loaded data
    - /opt/ml/processing/output/metadata/metadata: Metadata about fields (type information)
    - Data is loaded directly to S3 by the Cradle service
    
    Environment Variables:
    - OUTPUT_PATH: Optional override for the data output path
    
    The script performs the following operations:
    - Reads the data loading configuration from the config file
    - Writes the output signature based on the configuration
    - Writes metadata files with field type information
    - Creates a SandboxSession to interact with secure resources
    - Starts a Cradle data download job
    - Waits for the download job to complete
    
    This script is designed to interface with Amazon's internal Cradle data service
    to securely download and prepare data for machine learning pipelines.
    As a source node, this step doesn't have any dependencies on other steps.
    """,
)

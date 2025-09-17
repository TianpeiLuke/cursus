"""
MIMS Registration Script Contract

Defines the contract for the MIMS model registration script that registers
trained models with the MIMS service.
"""

from ...core.base.contract_base import ScriptContract

MIMS_REGISTRATION_CONTRACT = ScriptContract(
    entry_point="script.py",
    expected_input_paths={
        "PackagedModel": "/opt/ml/processing/input/model",
        "GeneratedPayloadSamples": "/opt/ml/processing/mims_payload",
    },
    expected_output_paths={
        # No output paths as this is a registration step with side effects only
    },
    expected_arguments={
        # No expected arguments - using standard paths from contract
    },
    required_env_vars=[
        # Environment variables required for registration
        "MODS_WORKFLOW_EXECUTION_ID"
    ],
    optional_env_vars={"PERFORMANCE_METADATA_PATH": ""},
    framework_requirements={"python": ">=3.7"},
    description="""
    MIMS model registration script that:
    1. Uploads model artifacts to temporary S3 location
    2. Optionally uploads payload samples if provided
    3. Registers the model with MIMS service using configuration
    4. Handles performance metadata if provided
    5. Tracks workflow execution ID for lineage
    
    Input Structure:
    - /opt/ml/processing/input/model: Packaged model artifacts (.tar.gz)
    - /opt/ml/processing/mims_payload: Optional payload samples for inference testing
    - /opt/ml/processing/config/config: Configuration file for registration
    - /opt/ml/processing/input/metadata: Optional performance metadata
    
    Environment Variables:
    - MODS_WORKFLOW_EXECUTION_ID: Workflow execution ID for tracking
    - PERFORMANCE_METADATA_PATH: Optional S3 path to performance metadata
    
    The script performs the following operations:
    - Creates a SandboxSession to interact with secure resources
    - Uploads model artifacts to a temporary S3 location
    - Optionally uploads payload samples if provided
    - Reads registration configuration from the config file
    - Sets appropriate environment variables
    - Registers the model using the MIMS resource API
    - Waits for registration completion
    - Cleans up temporary S3 resources
    
    This script does not generate output files but registers the model as a side effect.
    """,
)

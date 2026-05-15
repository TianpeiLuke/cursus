"""
Data Uploading Script Contract

Defines the contract for the SAIS SDK data uploading script that uploads
data from S3 to BDT (EDX/Andes). This is a SINK node — no SageMaker outputs.
"""

from ...core.base.contract_base import ScriptContract

DATA_UPLOADING_CONTRACT = ScriptContract(
    entry_point="scripts.py",
    expected_input_paths={
        # SDK delegation: no ProcessingInput mount.
        # Input arrives via --input-path argument (handled by SDK processor internally).
        # Logical name 'input_data' declared for contract↔spec alignment only.
        "input_data": "/opt/ml/processing/input/data",
    },
    expected_output_paths={
        # Sink node — data goes to BDT (Andes), not S3
    },
    expected_arguments={
        # Documented for reference only — the builder does NOT create this argument.
        # SDK DataUploadProcessor.get_processing_job_arguments() creates it internally
        # from the input_s3_location passed to DataUploadingStep constructor.
        "input-path": "S3 URI resolved from upstream step output (pipeline variable)",
    },
    required_env_vars=[],
    optional_env_vars={},
    framework_requirements={"python": ">=3.7"},
    description="""
    SDK delegation step — no Cursus script. The SAIS SDK provides the script.

    SDK script (data_uploading/scripts/scripts.py):
    1. Reads upload configuration from /opt/ml/processing/config/config
    2. Receives --input-path S3 URI via argparse (merged into config as inputS3Url)
    3. Creates a SAIS Session and DataUploader resource
    4. Calls create_data_upload_job() with CreateDataUploadJobRequest
    5. Polls wait_for_done(sleep_interval=120) until completion

    Builder responsibility:
    - Extracts upstream S3 (Properties object) from assembler inputs["input_data"]
    - Passes to DataUploadingStep(input_s3_location=<Properties>)
    - SDK internally converts to --input-path job argument
    - Builder does NOT handle arguments directly

    This is a SINK node — data exits the pipeline to BDT (Andes).
    No SageMaker outputs are produced.
    """,
)

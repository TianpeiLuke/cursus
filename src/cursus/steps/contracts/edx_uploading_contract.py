"""
EDX Uploading Script Contract

Defines the contract for the EDX upload script that uploads S3 data to EDX
via EdxDataLoader. This is a SINK node — no SageMaker outputs.
"""

from ...core.base.contract_base import ScriptContract

EDX_UPLOADING_CONTRACT = ScriptContract(
    entry_point="edx_upload.py",
    expected_input_paths={
        "input_data": "/opt/ml/processing/input/data",
    },
    expected_output_paths={
        # SINK node — data goes to EDX, not S3
    },
    expected_arguments={},
    required_env_vars=["EDX_DATASET_ARN", "EDX_MANIFEST_KEY"],
    optional_env_vars={
        "EDX_MANIFEST_KEY_PARTS": "{}",
        "AWS_STS_REGIONAL_ENDPOINTS": "regional",
        "AWS_DEFAULT_REGION": "us-east-1",
    },
    framework_requirements={
        "python": ">=3.7",
    },
    description="""
    EDX upload script that uploads S3 data to EDX via EdxDataLoader.

    SINK node — data exits the pipeline to EDX. No SageMaker outputs.
    No Kale attestation required (unlike DataUploading which uses Andes).

    Input: /opt/ml/processing/input/data (files from upstream step)
    Output: None (data uploaded to EDX manifest)

    Environment Variables:
    - EDX_DATASET_ARN (required): Base ARN for EDX dataset
    - EDX_MANIFEST_KEY (required): Manifest key (static or template with {placeholders})
    - EDX_MANIFEST_KEY_PARTS: JSON dict of placeholder values for template keys
    - AWS_STS_REGIONAL_ENDPOINTS: Must be 'regional' for SAIS
    - AWS_DEFAULT_REGION: AWS region (default: us-east-1)

    Manifest Key Modes:
    - Static: "munged_na" → uploads to /["munged_na"]
    - Template: "{marketplace},{date}" with EDX_MANIFEST_KEY_PARTS for resolution
    """,
)

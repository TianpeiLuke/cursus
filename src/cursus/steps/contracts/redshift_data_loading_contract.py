"""
Redshift Data Loading Script Contract.

Defines the contract for the Redshift data loading script that connects to
Redshift, executes SQL, and writes results as CSV to S3 (optionally uploading to EDX).
"""

from ...core.base.contract_base import ScriptContract

REDSHIFT_DATA_LOADING_CONTRACT = ScriptContract(
    entry_point="redshift_data_loading.py",
    expected_input_paths={},  # SOURCE NODE — no S3 inputs, data comes from Redshift
    expected_output_paths={
        "output": "/opt/ml/processing/output/data",
    },
    expected_arguments={},
    required_env_vars=[],
    optional_env_vars={
        "AWS_DEFAULT_REGION": "us-east-1",
        "AWS_STS_REGIONAL_ENDPOINTS": "regional",
    },
    framework_requirements={
        "boto3": ">=1.26.0",
        "pandas": ">=1.2.0",
    },
    description="""
    Redshift data loading script. Source node that:
    1. Reads step config from /opt/ml/processing/config/config (3-spec JSON)
    2. Dynamically installs DB connector (pg8000 or redshift_connector) from CodeArtifact
    3. Connects to Redshift via IAM role assumption (supports Andes3 session tags)
    4. Executes SQL query → DataFrame → CSV
    5. Writes output to /opt/ml/processing/output/data/output.csv
    6. Optionally uploads to EDX if outputSpecification.dataSourceType == "EDX"

    Config structure (read from /opt/ml/processing/config/config):
    - clusterSpecification: clusterId, dbName, roleArn, clusterEndpoint, port
    - querySpecification: query, connectorType, isUsingAndes3
    - outputSpecification: dataSourceType, edxArn, dropHeader (optional)

    Source node — no ProcessingInput. Data comes from Redshift, not S3.
    S3 output always produced. EDX upload is optional side effect.
    """,
)

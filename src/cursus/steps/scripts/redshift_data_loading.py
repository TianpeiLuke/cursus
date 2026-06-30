import json
import os
import sys
from subprocess import CalledProcessError, check_call

import boto3
import pandas as pd
from secure_ai_sandbox_python_lib.session import Session

AWS_DEFAULT_REGION = "us-east-1"
CLUSTER_SPECIFICATION_PARAMETER = "clusterSpecification"
OUTPUT_SPECIFICATION_PARAMETER = "outputSpecification"
QUERY_SPECIFICATION_PARAMETER = "querySpecification"
OUTPUT_DATA_PATH = "/opt/ml/processing/output/data/"
STEP_CONFIG_PATH = "/opt/ml/processing/config/config"

CLUSTER_ENDPOINT_FIELD = "clusterEndpoint"
CLUSTER_ID_FIELD = "clusterId"
DB_NAME_FIELD = "dbName"
IS_USING_ANDES_3_FIELD = "isUsingAndes3"
PORT_FIELD = "port"
ROLE_ARN_FIELD = "roleArn"
QUERY_FIELD = "query"
DATA_SOURCE_TYPE_FIELD = "dataSourceType"
EDX_ARN_FIELD = "edxArn"
DROP_HEADER_FIELD = "dropHeader"
CONNECTOR_TYPE_FIELD = "connectorType"
PG8000_CONNECTOR_TYPE = "pg8000"
REDSHIFT_CONNECTOR_TYPE = "redshift_connector"
PORT_FIELD_DEFAULT_VALUE = 5439


def _get_codeartifact_index_url():
    """Get the CodeArtifact index URL for installing packages."""
    os.environ["AWS_STS_REGIONAL_ENDPOINTS"] = "regional"
    sts = boto3.client("sts", region_name="us-east-1")
    caller_identity = sts.get_caller_identity()
    assumed_role_object = sts.assume_role(
        RoleArn=f"arn:aws:iam::675292366480:role/SecurePyPIReadRole_{caller_identity['Account']}",
        RoleSessionName="SecurePypiReadRole",
    )
    credentials = assumed_role_object["Credentials"]
    code_artifact_client = boto3.client(
        "codeartifact",
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
        region_name="us-west-2",
    )
    token = code_artifact_client.get_authorization_token(
        domain="amazon", domainOwner="149122183214"
    )["authorizationToken"]
    return f"https://aws:{token}@amazon-149122183214.d.codeartifact.us-west-2.amazonaws.com/pypi/secure-pypi/simple/"


def _install_package(package_name):
    """Install a package from CodeArtifact."""
    index_url = _get_codeartifact_index_url()
    try:
        check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--index-url",
                index_url,
                package_name,
            ]
        )
    except CalledProcessError as e:
        raise RuntimeError(
            f"Failed to install {package_name} (exit code {e.returncode})"
        ) from e


def import_redshift_connector():
    """Import redshift_connector, installing it first if necessary."""
    try:
        import redshift_connector

        return redshift_connector
    except ImportError:
        _install_package("redshift-connector")
        import redshift_connector

        return redshift_connector


def import_pg8000():
    """Import pg8000, installing it first if necessary."""
    try:
        import pg8000

        return pg8000
    except ImportError:
        _install_package("pg8000")
        import pg8000

        return pg8000


def read_step_config():
    with open(STEP_CONFIG_PATH) as config_file:
        redshift_data_loading_config = json.load(config_file)
    print(
        f"Starting Redshift Data Loading with config : {redshift_data_loading_config}"
    )
    return redshift_data_loading_config


def _assume_role(sts_client, role_arn, is_using_andes_3=False):
    """Assume IAM role, with Andes3 session tags if required."""
    if is_using_andes_3:
        return sts_client.assume_role(
            RoleArn=role_arn,
            RoleSessionName="sais_session",
            Tags=[{"Key": "currentOwnerAlias", "Value": "mods_db_user"}],
            TransitiveTagKeys=["currentOwnerAlias"],
        )["Credentials"]
    else:
        return sts_client.assume_role(RoleArn=role_arn, RoleSessionName="sais_session")[
            "Credentials"
        ]


def _get_redshift_client(sts_client, role_arn, is_using_andes_3=False):
    """Create a Redshift client using assumed role credentials."""
    creds = _assume_role(sts_client, role_arn, is_using_andes_3)
    return boto3.client(
        "redshift",
        aws_access_key_id=creds["AccessKeyId"],
        aws_secret_access_key=creds["SecretAccessKey"],
        aws_session_token=creds["SessionToken"],
    )


def _connect_with_pg8000(sts_client, cluster_specification, query_specification):
    """Connect to Redshift using pg8000 via host/port with DB credentials."""
    dbapi = import_pg8000()
    role_arn = cluster_specification.get(ROLE_ARN_FIELD)
    db_name = cluster_specification.get(DB_NAME_FIELD)
    cluster_id = cluster_specification.get(CLUSTER_ID_FIELD)
    is_using_andes_3 = query_specification.get(IS_USING_ANDES_3_FIELD)

    redshift_client = _get_redshift_client(sts_client, role_arn, is_using_andes_3)

    if is_using_andes_3:
        redshift_login = redshift_client.get_cluster_credentials_with_iam(
            ClusterIdentifier=cluster_id,
            DurationSeconds=3600,
            DbName=db_name,
        )
    else:
        redshift_login = redshift_client.get_cluster_credentials(
            ClusterIdentifier=cluster_id,
            DbUser="mods_db_user",
            DbName=db_name,
            DbGroups=["sais_users"],
            AutoCreate=True,
            DurationSeconds=3600,
        )

    conn = dbapi.connect(
        database=db_name,
        host=cluster_specification.get(CLUSTER_ENDPOINT_FIELD),
        port=cluster_specification.get(PORT_FIELD, PORT_FIELD_DEFAULT_VALUE),
        user=redshift_login["DbUser"],
        password=redshift_login["DbPassword"],
        ssl_context=True,
    )
    conn.autocommit = True
    return conn


def _connect_with_redshift_connector(
    sts_client, cluster_specification, query_specification
):
    """Connect to Redshift using redshift_connector."""
    redshift_connector = import_redshift_connector()
    role_arn = cluster_specification.get(ROLE_ARN_FIELD)
    db_name = cluster_specification.get(DB_NAME_FIELD)
    is_using_andes_3 = query_specification.get(IS_USING_ANDES_3_FIELD)

    if is_using_andes_3:
        redshift_client = _get_redshift_client(
            sts_client, role_arn, is_using_andes_3=True
        )
        redshift_login = redshift_client.get_cluster_credentials_with_iam(
            ClusterIdentifier=cluster_specification.get(CLUSTER_ID_FIELD),
            DurationSeconds=3600,
            DbName=db_name,
        )
        return redshift_connector.connect(
            database=db_name,
            host=cluster_specification.get(CLUSTER_ENDPOINT_FIELD),
            port=cluster_specification.get(PORT_FIELD, PORT_FIELD_DEFAULT_VALUE),
            user=redshift_login["DbUser"],
            password=redshift_login["DbPassword"],
        )
    else:
        assume_role_creds = _assume_role(sts_client, role_arn)
        return redshift_connector.connect(
            access_key_id=assume_role_creds["AccessKeyId"],
            secret_access_key=assume_role_creds["SecretAccessKey"],
            session_token=assume_role_creds["SessionToken"],
            cluster_identifier=cluster_specification.get(CLUSTER_ID_FIELD),
            database=db_name,
            region="us-east-1",
            db_user="mods_db_user",
            db_groups=["sais_users"],
            password="",
            auto_create=True,
            user="",
            iam=True,
        )


def start_redshift_data_loading(redshift_data_load_config):
    sandbox_session = Session(session_folder="/tmp/")
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)

    cluster_specification = redshift_data_load_config.get(
        CLUSTER_SPECIFICATION_PARAMETER
    )
    query_specification = redshift_data_load_config.get(QUERY_SPECIFICATION_PARAMETER)
    output_specification = redshift_data_load_config.get(OUTPUT_SPECIFICATION_PARAMETER)

    if not cluster_specification:
        raise ValueError(
            f"Missing required config field: {CLUSTER_SPECIFICATION_PARAMETER}"
        )
    if not query_specification:
        raise ValueError(
            f"Missing required config field: {QUERY_SPECIFICATION_PARAMETER}"
        )

    connector_type = query_specification.get(
        CONNECTOR_TYPE_FIELD, REDSHIFT_CONNECTOR_TYPE
    )

    print(
        f"RoleArn - {cluster_specification.get(ROLE_ARN_FIELD)}, "
        f"DbName - {cluster_specification.get(DB_NAME_FIELD)}, "
        f"ClusterId - {cluster_specification.get(CLUSTER_ID_FIELD)}, "
        f"ConnectorType - {connector_type}"
    )

    sts_client = boto3.client("sts", region_name=AWS_DEFAULT_REGION)

    if connector_type == PG8000_CONNECTOR_TYPE:
        conn = _connect_with_pg8000(
            sts_client, cluster_specification, query_specification
        )
        try:
            result: pd.DataFrame = pd.read_sql(
                query_specification.get(QUERY_FIELD), con=conn
            )
        finally:
            conn.close()
    else:
        conn = _connect_with_redshift_connector(
            sts_client, cluster_specification, query_specification
        )
        try:
            cursor = conn.cursor()
            cursor.execute(query_specification.get(QUERY_FIELD))
            result: pd.DataFrame = cursor.fetch_dataframe()
        finally:
            conn.close()

    print(result.head(10))
    print(f"Total number of rows in DataFrame: {len(result)}")

    output_file = os.path.join(OUTPUT_DATA_PATH, "output.csv")
    drop_header = (
        output_specification.get(DROP_HEADER_FIELD, False)
        if output_specification
        else False
    )
    result.to_csv(output_file, index=False, header=not drop_header)

    if output_specification:
        if output_specification.get(DATA_SOURCE_TYPE_FIELD) == "EDX":
            edx_data_loader = sandbox_session.resource("EdxDataLoader")
            edx_data_loader.upload_data_to_edx(
                output_specification.get(EDX_ARN_FIELD), output_file
            )
    else:
        print(
            f"Output Specification not provided. Output file is written to {OUTPUT_DATA_PATH}"
        )


job_config = read_step_config()
start_redshift_data_loading(job_config)

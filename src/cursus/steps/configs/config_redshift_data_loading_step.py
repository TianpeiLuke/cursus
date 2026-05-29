"""
Redshift Data Loading Step Configuration.

Flattens the SAIS SDK's 3-specification JSON structure (clusterSpecification,
querySpecification, outputSpecification) into Pydantic fields. The computed
property `step_config_json` reassembles the 3-spec dict for the container script.
"""

from typing import Optional, Dict, Any
from pydantic import Field, computed_field, model_validator

from ...core.base.config_base import BasePipelineConfig


class RedshiftDataLoadingConfig(BasePipelineConfig):
    """
    Configuration for RedshiftDataLoading step.

    Source node — executes SQL against Redshift, writes CSV to S3.
    Optionally uploads to EDX as side effect.
    """

    # ===== Tier 1: Essential (user must provide) =====

    cluster_id: str = Field(description="Redshift cluster identifier")
    db_name: str = Field(description="Database name")
    role_arn: str = Field(description="IAM role ARN for Redshift access")
    cluster_endpoint: str = Field(
        description="Cluster endpoint hostname (required for pg8000 and Andes3)"
    )
    query: str = Field(description="SQL query to execute against Redshift")

    # ===== Tier 2: System with Defaults =====

    connector_type: str = Field(
        default="redshift_connector",
        description="DB connector: 'pg8000' or 'redshift_connector'",
    )
    port: int = Field(default=5439, description="Redshift cluster port")
    is_using_andes3: bool = Field(
        default=True,
        description="Use Andes3 IAM authentication with session tags",
    )
    output_data_source_type: str = Field(
        default="S3",
        description="Output destination: 'S3' (default, S3 only) or 'EDX' (S3 + EDX upload)",
    )
    edx_arn: Optional[str] = Field(
        default=None,
        description="EDX manifest ARN. Required if output_data_source_type='EDX'.",
    )
    drop_header: bool = Field(
        default=False,
        description="Whether to drop CSV header in output",
    )
    job_type: Optional[str] = Field(
        default=None,
        description="Job type suffix for step naming (e.g., 'mo_orders')",
    )
    max_runtime_in_seconds: int = Field(
        default=43200,
        description="Max processing time in seconds (default: 12 hours)",
    )

    # ===== Validation =====

    @model_validator(mode="after")
    def validate_edx_config(self) -> "RedshiftDataLoadingConfig":
        """Validate EDX ARN is provided when output type is EDX."""
        if self.output_data_source_type == "EDX" and not self.edx_arn:
            raise ValueError("edx_arn is required when output_data_source_type='EDX'")
        if self.connector_type not in ("pg8000", "redshift_connector"):
            raise ValueError(
                f"connector_type must be 'pg8000' or 'redshift_connector', got '{self.connector_type}'"
            )
        return self

    # ===== Tier 3: Derived =====

    @computed_field
    @property
    def step_config_json(self) -> Dict[str, Any]:
        """
        Generate the 3-spec JSON config that the container script reads
        from /opt/ml/processing/config/config.
        """
        config: Dict[str, Any] = {
            "clusterSpecification": {
                "clusterId": self.cluster_id,
                "dbName": self.db_name,
                "roleArn": self.role_arn,
                "clusterEndpoint": self.cluster_endpoint,
                "port": self.port,
            },
            "querySpecification": {
                "query": self.query,
                "connectorType": self.connector_type,
                "isUsingAndes3": self.is_using_andes3,
            },
            "outputSpecification": {},
        }

        if self.output_data_source_type == "EDX":
            config["outputSpecification"] = {
                "dataSourceType": "EDX",
                "edxArn": self.edx_arn,
                "dropHeader": self.drop_header,
            }
        elif self.drop_header:
            config["outputSpecification"] = {"dropHeader": True}

        return config

    def get_script_contract(self):
        """Get the script contract for this configuration."""
        from ..contracts.redshift_data_loading_contract import (
            REDSHIFT_DATA_LOADING_CONTRACT,
        )

        return REDSHIFT_DATA_LOADING_CONTRACT

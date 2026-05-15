"""
Data Uploading Step Configuration.

Pydantic v2 config for the DataUploading step that uploads data from S3 to
Andes via the SAIS SDK. Maps to CreateDataUploadJobRequest fields.
"""

from typing import Optional, Dict, Any
from pydantic import Field, model_validator

from ...core.base.config_base import BasePipelineConfig


class DataUploadingConfig(BasePipelineConfig):
    """Configuration for the DataUploading step (S3 → Andes via SAIS SDK)."""

    # ===== Tier 1 (Essential — user must provide) =====

    provider_name: str = Field(
        description="Andes provider name (maps to dataSink.andesDataSinkProperties.providerName)"
    )

    table_name: str = Field(
        description="Andes table name (maps to dataSink.andesDataSinkProperties.tableName)"
    )

    table_version: int = Field(
        description="Andes table version (maps to dataSink.andesDataSinkProperties.tableVersion)"
    )

    cradle_account: str = Field(
        description="Cradle account identifier (maps to cradleAccount, required by API)"
    )

    schema_sdl: str = Field(
        description="SDL schema string describing the data format (maps to schema)"
    )

    # ===== Tier 2 (System — with defaults) =====

    job_type: Optional[str] = Field(
        default=None,
        description="Variant suffix for node uniqueness when multiple DataUploading steps "
        "exist in one DAG (e.g., training, calibration, inference, scoring, export).",
    )

    input_s3_location: Optional[str] = Field(
        default=None,
        description="Optional S3 URI override for manual/debug execution. "
        "Normally resolved from upstream DAG output by the assembler.",
    )

    input_format: str = Field(
        default="CSV",
        description="Input data format (maps to inputFormat)",
    )

    has_header: bool = Field(
        default=True,
        description="Whether input data has a header row (maps to hasHeader)",
    )

    cluster_type: str = Field(
        default="MEDIUM",
        description="Cradle cluster type for the upload job (maps to clusterType)",
    )

    instance_type: str = Field(
        default="ml.m5.xlarge",
        description="SageMaker processing instance type",
    )

    instance_count: int = Field(
        default=1,
        ge=1,
        description="SageMaker processing instance count",
    )

    volume_size_in_gb: int = Field(
        default=30,
        ge=1,
        description="SageMaker processing volume size in GB",
    )

    # ===== Tier 3 (Derived — computed at runtime) =====

    @property
    def job_config(self) -> Dict[str, Any]:
        """
        Assemble the full job config dict for CreateDataUploadJobRequest.

        This is what gets written to /opt/ml/processing/config/config and
        deserialized via CreateDataUploadJobRequest.__from_coral__(**config).
        Note: inputS3Url is NOT included here — it's injected at runtime
        via --input-path argument by the SDK processor.
        """
        return {
            "schema": self.schema_sdl,
            "inputFormat": self.input_format,
            "cradleAccount": self.cradle_account,
            "hasHeader": self.has_header,
            "clusterType": self.cluster_type,
            "dataSink": {
                "dataSinkType": "ANDES",
                "andesDataSinkProperties": {
                    "providerName": self.provider_name,
                    "tableName": self.table_name,
                    "tableVersion": self.table_version,
                },
            },
        }

    @model_validator(mode="after")
    def validate_cluster_type(self) -> "DataUploadingConfig":
        """Validate cluster_type is a known value."""
        valid_types = {"SMALL", "MEDIUM", "LARGE"}
        if self.cluster_type.upper() not in valid_types:
            raise ValueError(
                f"cluster_type must be one of {valid_types}, got: {self.cluster_type}"
            )
        return self

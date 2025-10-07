"""
Request schemas for Cradle Data Load Config UI API

This module defines Pydantic models for API request validation.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List


class StepValidationRequest(BaseModel):
    """Request model for step validation."""
    step: int = Field(..., ge=1, le=4, description="Step number (1-4)")
    data: Dict[str, Any] = Field(..., description="Step data to validate")


class DataSourceValidationRequest(BaseModel):
    """Request model for data source validation."""
    data: Dict[str, Any] = Field(..., description="Data source configuration to validate")


class ConfigBuildRequest(BaseModel):
    """Request model for building final configuration."""
    data_sources_spec: Dict[str, Any] = Field(..., description="Data sources specification")
    transform_spec: Dict[str, Any] = Field(..., description="Transform specification")
    output_spec: Dict[str, Any] = Field(..., description="Output specification")
    cradle_job_spec: Dict[str, Any] = Field(..., description="Cradle job specification")
    job_type: str = Field(..., description="Job type (training/validation/testing/calibration)")
    save_location: Optional[str] = Field(None, description="Optional file path to automatically save the configuration")


class ConfigExportRequest(BaseModel):
    """Request model for configuration export."""
    config: Dict[str, Any] = Field(..., description="Configuration to export")
    format: str = Field("json", description="Export format (json/python)")
    include_comments: bool = Field(True, description="Include comments in exported code")


class DataSourceBlockData(BaseModel):
    """Data model for a single data source block."""
    id: str = Field(..., description="Unique identifier for the data source block")
    data_source_name: str = Field(..., description="Name of the data source")
    data_source_type: str = Field(..., description="Type of data source (MDS/EDX/ANDES)")
    mds_properties: Optional[Dict[str, Any]] = Field(None, description="MDS-specific properties")
    edx_properties: Optional[Dict[str, Any]] = Field(None, description="EDX-specific properties")
    andes_properties: Optional[Dict[str, Any]] = Field(None, description="ANDES-specific properties")


class JobSplitOptionsData(BaseModel):
    """Data model for job split options."""
    split_job: bool = Field(False, description="Whether to split the job")
    days_per_split: int = Field(7, description="Number of days per split")
    merge_sql: Optional[str] = Field(None, description="SQL for merging split results")


class TransformSpecData(BaseModel):
    """Data model for transform specification."""
    transform_sql: str = Field(..., description="SQL transformation query")
    job_split_options: JobSplitOptionsData = Field(default_factory=JobSplitOptionsData)


class OutputSpecData(BaseModel):
    """Data model for output specification."""
    output_schema: List[str] = Field(..., description="List of output field names")
    output_format: str = Field("PARQUET", description="Output format")
    output_save_mode: str = Field("ERRORIFEXISTS", description="Output save mode")
    output_file_count: int = Field(0, description="Number of output files (0 = auto)")
    keep_dot_in_output_schema: bool = Field(False, description="Keep dots in output schema")
    include_header_in_s3_output: bool = Field(True, description="Include header in S3 output")


class CradleJobSpecData(BaseModel):
    """Data model for cradle job specification."""
    cradle_account: str = Field(..., description="Cradle account name")
    cluster_type: str = Field("STANDARD", description="Cluster type")
    extra_spark_job_arguments: str = Field("", description="Extra Spark job arguments")
    job_retry_count: int = Field(1, description="Number of job retries")

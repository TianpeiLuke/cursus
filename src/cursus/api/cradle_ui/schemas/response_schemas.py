"""
Response schemas for Cradle Data Load Config UI API

This module defines Pydantic models for API response validation.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List


class ConfigDefaultsResponse(BaseModel):
    """Response model for configuration defaults."""
    defaults: Dict[str, Any] = Field(..., description="Default values for all configuration sections")


class ValidationResponse(BaseModel):
    """Base response model for validation operations."""
    is_valid: bool = Field(..., description="Whether the validation passed")
    errors: Dict[str, List[str]] = Field(default_factory=dict, description="Validation errors by field")


class StepValidationResponse(ValidationResponse):
    """Response model for step validation."""
    warnings: Dict[str, List[str]] = Field(default_factory=dict, description="Validation warnings by field")


class ConfigBuildResponse(BaseModel):
    """Response model for configuration building."""
    success: bool = Field(..., description="Whether the configuration was built successfully")
    config: Optional[Dict[str, Any]] = Field(None, description="Built configuration object")
    python_code: Optional[str] = Field(None, description="Python code to create the configuration")
    errors: List[str] = Field(default_factory=list, description="Build errors")
    message: Optional[str] = Field(None, description="Success or informational message")


class ConfigExportResponse(BaseModel):
    """Response model for configuration export."""
    success: bool = Field(..., description="Whether the export was successful")
    content: str = Field(..., description="Exported configuration content")
    format: str = Field(..., description="Export format used")


class FieldInfo(BaseModel):
    """Information about a configuration field."""
    type: str = Field(..., description="Field type")
    required: bool = Field(..., description="Whether the field is required")
    default: Any = Field(None, description="Default value for the field")
    description: Optional[str] = Field(None, description="Field description")
    validation: Dict[str, Any] = Field(default_factory=dict, description="Validation rules")


class FieldSchema(BaseModel):
    """Schema information for a configuration class."""
    fields: Dict[str, FieldInfo] = Field(..., description="Field information by field name")
    categories: Dict[str, List[str]] = Field(..., description="Field categories (essential/system/derived)")


class FieldSchemaResponse(BaseModel):
    """Response model for field schema requests."""
    config_type: str = Field(..., description="Configuration type")
    field_schema: FieldSchema = Field(..., description="Schema information")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    code: Optional[str] = Field(None, description="Error code")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    timestamp: Optional[str] = Field(None, description="Response timestamp")


class DataSourceSchemaResponse(BaseModel):
    """Response model for data source schema information."""
    schemas: Dict[str, FieldSchema] = Field(..., description="Schemas for all data source types")


class ConfigSummary(BaseModel):
    """Summary information about a configuration."""
    data_sources_count: int = Field(..., description="Number of configured data sources")
    time_range: str = Field(..., description="Data time range")
    transform_type: str = Field(..., description="Type of transformation")
    output_format: str = Field(..., description="Output format")
    cluster_type: str = Field(..., description="Cluster type")
    job_type: Optional[str] = Field(None, description="Job type")


class ConfigSummaryResponse(BaseModel):
    """Response model for configuration summary."""
    summary: ConfigSummary = Field(..., description="Configuration summary")
    is_complete: bool = Field(..., description="Whether the configuration is complete")
    missing_fields: List[str] = Field(default_factory=list, description="List of missing required fields")

"""
FastAPI routes for Cradle Data Load Config UI

This module provides REST API endpoints for the web-based configuration wizard.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
import logging

from ..schemas.request_schemas import (
    StepValidationRequest,
    DataSourceValidationRequest,
    ConfigBuildRequest,
    ConfigExportRequest
)
from ..schemas.response_schemas import (
    ConfigDefaultsResponse,
    StepValidationResponse,
    ValidationResponse,
    ConfigBuildResponse,
    ConfigExportResponse,
    FieldSchemaResponse
)
from ..services.config_builder import ConfigBuilderService
from ..services.validation_service import ValidationService
from ..utils.field_extractors import extract_field_schema, get_data_source_variant_schemas

logger = logging.getLogger(__name__)

# Create router with prefix and tags
router = APIRouter(prefix="/api/cradle-ui", tags=["cradle-ui"])

# Initialize services
config_builder = ConfigBuilderService()
validation_service = ValidationService()


@router.get("/config-defaults", response_model=ConfigDefaultsResponse)
async def get_config_defaults() -> ConfigDefaultsResponse:
    """
    Get default values for all configuration fields.
    
    Returns:
        ConfigDefaultsResponse: Default values organized by configuration section
    """
    try:
        defaults = {
            "dataSourcesSpec": {
                "startDate": "",
                "endDate": "",
                "dataSources": []
            },
            "transformSpec": {
                "transformSql": "",
                "jobSplitOptions": {
                    "splitJob": False,
                    "daysPerSplit": 7,
                    "mergeSql": ""
                }
            },
            "outputSpec": {
                "outputSchema": [],
                "outputFormat": "PARQUET",
                "outputSaveMode": "ERRORIFEXISTS",
                "outputFileCount": 0,
                "keepDotInOutputSchema": False,
                "includeHeaderInS3Output": True
            },
            "cradleJobSpec": {
                "cradleAccount": "",
                "clusterType": "STANDARD",
                "extraSparkJobArguments": "",
                "jobRetryCount": 1
            },
            "jobType": ""
        }
        
        return ConfigDefaultsResponse(defaults=defaults)
        
    except Exception as e:
        logger.error(f"Error getting config defaults: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get config defaults: {str(e)}")


@router.post("/validate-step", response_model=StepValidationResponse)
async def validate_step(request: StepValidationRequest) -> StepValidationResponse:
    """
    Validate a specific step's configuration.
    
    Args:
        request: Step validation request containing step number and data
        
    Returns:
        StepValidationResponse: Validation results with errors and warnings
    """
    try:
        errors = validation_service.validate_step_data(request.step, request.data)
        is_valid = len(errors) == 0
        
        return StepValidationResponse(
            is_valid=is_valid,
            errors=errors,
            warnings={}  # TODO: Implement warnings logic
        )
        
    except Exception as e:
        logger.error(f"Error validating step {request.step}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@router.post("/validate-data-source", response_model=ValidationResponse)
async def validate_data_source(request: DataSourceValidationRequest) -> ValidationResponse:
    """
    Validate a single data source configuration.
    
    Args:
        request: Data source validation request
        
    Returns:
        ValidationResponse: Validation results
    """
    try:
        errors = validation_service._validate_data_source(request.data)
        is_valid = len(errors) == 0
        
        return ValidationResponse(
            is_valid=is_valid,
            errors={"dataSource": errors} if errors else {}
        )
        
    except Exception as e:
        logger.error(f"Error validating data source: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Data source validation failed: {str(e)}")


@router.post("/build-config", response_model=ConfigBuildResponse)
async def build_config(request: ConfigBuildRequest) -> ConfigBuildResponse:
    """
    Build the final CradleDataLoadConfig from UI data.
    
    Args:
        request: Configuration build request with all UI data
        
    Returns:
        ConfigBuildResponse: Built configuration or error details
    """
    try:
        # Convert request to dictionary format expected by validation service
        ui_data = {
            "dataSourcesSpec": request.data_sources_spec,
            "transformSpec": request.transform_spec,
            "outputSpec": request.output_spec,
            "cradleJobSpec": request.cradle_job_spec,
            "jobType": request.job_type
        }
        
        # Build the configuration
        config = validation_service.build_final_config(ui_data)
        
        # Convert to dictionary for response
        config_dict = config.model_dump()
        
        return ConfigBuildResponse(
            success=True,
            config=config_dict,
            errors=[]
        )
        
    except Exception as e:
        logger.error(f"Error building config: {str(e)}")
        return ConfigBuildResponse(
            success=False,
            config=None,
            errors=[str(e)]
        )


@router.post("/export-config", response_model=ConfigExportResponse)
async def export_config(request: ConfigExportRequest) -> ConfigExportResponse:
    """
    Export configuration as JSON or Python code.
    
    Args:
        request: Configuration export request
        
    Returns:
        ConfigExportResponse: Exported configuration in requested format
    """
    try:
        exported_content = config_builder.export_config(
            request.config,
            request.format,
            request.include_comments
        )
        
        return ConfigExportResponse(
            success=True,
            content=exported_content,
            format=request.format
        )
        
    except Exception as e:
        logger.error(f"Error exporting config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@router.get("/field-schema/{config_type}", response_model=FieldSchemaResponse)
async def get_field_schema(config_type: str) -> FieldSchemaResponse:
    """
    Get field schema for dynamic form generation.
    
    Args:
        config_type: Type of configuration (e.g., 'MDS', 'EDX', 'ANDES')
        
    Returns:
        FieldSchemaResponse: Field schema information
    """
    try:
        if config_type in ['MDS', 'EDX', 'ANDES']:
            schemas = get_data_source_variant_schemas()
            if config_type in schemas:
                return FieldSchemaResponse(
                    config_type=config_type,
                    schema=schemas[config_type]
                )
            else:
                raise HTTPException(status_code=404, detail=f"Schema not found for type: {config_type}")
        else:
            raise HTTPException(status_code=400, detail=f"Invalid config type: {config_type}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting field schema for {config_type}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get field schema: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "cradle-ui-api"}

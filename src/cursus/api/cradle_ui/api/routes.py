"""
FastAPI routes for Cradle Data Load Config UI

This module provides REST API endpoints for the web-based configuration wizard.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Optional
import logging
import os
import re

from ..schemas.request_schemas import (
    StepValidationRequest,
    DataSourceValidationRequest,
    ConfigBuildRequest,
    ConfigExportRequest,
)
from ..schemas.response_schemas import (
    ConfigDefaultsResponse,
    StepValidationResponse,
    ValidationResponse,
    ConfigBuildResponse,
    ConfigExportResponse,
    FieldSchemaResponse,
)
from ..services.config_builder import ConfigBuilderService
from ..services.validation_service import ValidationService
from ...factory import (
    extract_field_requirements,
)

def get_data_source_variant_schemas():
    """
    Get schemas for all data source variants using factory field extraction.
    
    Returns:
        Dict mapping data source type to schema information
    """
    try:
        # Import the data source config classes
        from ....steps.configs.config_cradle_data_loading_step import (
            MdsDataSourceConfig,
            EdxDataSourceConfig,
            AndesDataSourceConfig
        )
        
        return {
            "MDS": {"fields": extract_field_requirements(MdsDataSourceConfig)},
            "EDX": {"fields": extract_field_requirements(EdxDataSourceConfig)},
            "ANDES": {"fields": extract_field_requirements(AndesDataSourceConfig)}
        }
    except Exception as e:
        logger.error(f"Error getting data source variant schemas: {e}")
        return {}

logger = logging.getLogger(__name__)


# Security functions to prevent path traversal attacks
def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal attacks.

    Args:
        filename: The filename to sanitize

    Returns:
        str: Sanitized filename safe for file operations

    Raises:
        HTTPException: If filename contains invalid characters
    """
    if not filename:
        raise HTTPException(status_code=400, detail="Filename cannot be empty")

    # Remove any path separators and dangerous characters
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", filename)
    sanitized = sanitized.replace("..", "")  # Remove path traversal attempts
    sanitized = sanitized.strip(". ")  # Remove leading/trailing dots and spaces

    # Ensure filename is not empty after sanitization
    if not sanitized:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Limit filename length
    if len(sanitized) > 255:
        sanitized = sanitized[:255]

    return sanitized


def validate_file_path(file_path: str, allowed_dir: str) -> str:
    """
    Validate that a file path is within the allowed directory.

    Args:
        file_path: The file path to validate
        allowed_dir: The allowed base directory

    Returns:
        str: Validated absolute file path

    Raises:
        HTTPException: If path is outside allowed directory
    """
    try:
        # Convert to absolute paths
        abs_file_path = os.path.abspath(file_path)
        abs_allowed_dir = os.path.abspath(allowed_dir)

        # Check if file path is within allowed directory
        if not abs_file_path.startswith(abs_allowed_dir + os.sep):
            raise HTTPException(
                status_code=403, detail="Access denied: Path outside allowed directory"
            )

        return abs_file_path
    except Exception as e:
        logger.error(f"Path validation failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid file path")


# Create router with prefix and tags
router = APIRouter(prefix="/api/cradle-ui", tags=["cradle-ui"])

# Initialize services
config_builder = ConfigBuilderService()
validation_service = ValidationService()

# Global variable to store the latest configuration
latest_config = None


@router.get("/config-defaults", response_model=ConfigDefaultsResponse)
async def get_config_defaults() -> ConfigDefaultsResponse:
    """
    Get default values for all configuration fields.

    Returns:
        ConfigDefaultsResponse: Default values organized by configuration section
    """
    try:
        defaults = {
            "dataSourcesSpec": {"startDate": "", "endDate": "", "dataSources": []},
            "transformSpec": {
                "transformSql": "",
                "jobSplitOptions": {
                    "splitJob": False,
                    "daysPerSplit": 7,
                    "mergeSql": "",
                },
            },
            "outputSpec": {
                "outputSchema": [],
                "outputFormat": "PARQUET",
                "outputSaveMode": "ERRORIFEXISTS",
                "outputFileCount": 0,
                "keepDotInOutputSchema": False,
                "includeHeaderInS3Output": True,
            },
            "cradleJobSpec": {
                "cradleAccount": "",
                "clusterType": "STANDARD",
                "extraSparkJobArguments": "",
                "jobRetryCount": 1,
            },
            "jobType": "",
        }

        return ConfigDefaultsResponse(defaults=defaults)

    except Exception as e:
        logger.error(f"Error getting config defaults: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get config defaults: {str(e)}"
        )


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
            warnings={},  # TODO: Implement warnings logic
        )

    except Exception as e:
        logger.error(f"Error validating step {request.step}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@router.post("/validate-data-source", response_model=ValidationResponse)
async def validate_data_source(
    request: DataSourceValidationRequest,
) -> ValidationResponse:
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
            is_valid=is_valid, errors={"dataSource": errors} if errors else {}
        )

    except Exception as e:
        logger.error(f"Error validating data source: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Data source validation failed: {str(e)}"
        )


@router.post("/build-config", response_model=ConfigBuildResponse)
async def build_config(request: ConfigBuildRequest) -> ConfigBuildResponse:
    """
    Build the final CradleDataLoadingConfig from UI data and optionally save to file.

    Args:
        request: Configuration build request with all UI data

    Returns:
        ConfigBuildResponse: Built configuration or error details
    """
    global latest_config

    try:
        # Convert request to dictionary format expected by validation service
        ui_data = {
            "data_sources_spec": request.data_sources_spec,
            "transform_spec": request.transform_spec,
            "output_spec": request.output_spec,
            "cradle_job_spec": request.cradle_job_spec,
            "job_type": request.job_type,
        }

        # Build the configuration
        config = validation_service.build_final_config(ui_data)

        # Generate Python code
        python_code = validation_service.generate_python_code(config)

        # Convert to dictionary for response
        config_dict = config.model_dump()

        # Store the latest configuration for retrieval by Jupyter widget
        latest_config = config_dict

        # Auto-save to file if save_location is provided
        save_message = ""
        if hasattr(request, "save_location") and request.save_location:
            try:
                import json
                from pathlib import Path
                import tempfile

                # Sanitize the save location to prevent path traversal
                save_location = str(request.save_location)

                # Create a safe temporary directory for saving
                temp_dir = tempfile.mkdtemp()

                # Extract filename and sanitize it
                save_path = Path(save_location)
                safe_filename = sanitize_filename(save_path.name)

                # Create the safe file path within temp directory
                safe_save_path = Path(temp_dir) / safe_filename

                # Validate the path is within the temp directory
                validated_path = validate_file_path(str(safe_save_path), temp_dir)

                # Save configuration to JSON file
                with open(validated_path, "w") as f:
                    json.dump(config_dict, f, indent=2, default=str)

                save_message = f"Configuration automatically saved to: {safe_filename} (in secure location)"
                logger.info(f"Configuration saved to: {validated_path}")

            except Exception as save_error:
                logger.error(f"Error saving configuration: {str(save_error)}")
                save_message = (
                    f"Warning: Failed to save configuration: {str(save_error)}"
                )

        return ConfigBuildResponse(
            success=True,
            config=config_dict,
            python_code=python_code,
            errors=[],
            message=save_message
            if save_message
            else "Configuration built successfully",
        )

    except Exception as e:
        logger.error(f"Error building config: {str(e)}")
        return ConfigBuildResponse(
            success=False, config=None, python_code=None, errors=[str(e)]
        )


@router.get("/get-latest-config")
async def get_latest_config():
    """
    Get the latest generated configuration for Jupyter widget retrieval.

    Returns:
        Dict: Latest configuration data or 404 if none available
    """
    global latest_config

    if latest_config is None:
        raise HTTPException(
            status_code=404,
            detail="No configuration available. Please complete the configuration wizard first.",
        )

    return latest_config


@router.post("/clear-config")
async def clear_config():
    """
    Clear the stored configuration.

    This endpoint is called when the user navigates away from the finish page
    to disable the "Get Configuration" button in the Jupyter widget.

    Returns:
        Dict: Success message
    """
    global latest_config

    latest_config = None

    return {"success": True, "message": "Configuration cleared"}


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
            request.config, request.format, request.include_comments
        )

        return ConfigExportResponse(
            success=True, content=exported_content, format=request.format
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
        if config_type in ["MDS", "EDX", "ANDES"]:
            schemas = get_data_source_variant_schemas()
            if config_type in schemas:
                return FieldSchemaResponse(
                    config_type=config_type, schema=schemas[config_type]
                )
            else:
                raise HTTPException(
                    status_code=404, detail=f"Schema not found for type: {config_type}"
                )
        else:
            raise HTTPException(
                status_code=400, detail=f"Invalid config type: {config_type}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting field schema for {config_type}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get field schema: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "cradle-ui-api"}

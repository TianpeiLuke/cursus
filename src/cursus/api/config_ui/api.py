"""
Basic API Endpoints for Config UI

Provides basic FastAPI endpoints for configuration management and testing.
This is a minimal implementation for Phase 1 testing.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from cursus.core.base.config_base import BasePipelineConfig
from cursus.steps.configs.config_processing_step_base import ProcessingStepConfigBase
from .utils import (
    discover_available_configs,
    get_config_info,
    create_example_base_config,
    create_example_processing_config,
    validate_config_instance,
    export_config_to_dict,
    import_config_from_dict
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Cursus Config UI API",
    description="Universal Configuration Interface API",
    version="1.0.0"
)


# Request/Response Models
class ConfigDiscoveryRequest(BaseModel):
    workspace_dirs: Optional[List[str]] = None


class ConfigInfoRequest(BaseModel):
    config_class_name: str
    workspace_dirs: Optional[List[str]] = None


class ConfigValidationRequest(BaseModel):
    config_data: Dict[str, Any]
    config_class_name: str
    workspace_dirs: Optional[List[str]] = None


class ConfigImportRequest(BaseModel):
    config_dict: Dict[str, Any]
    config_class_name: str
    workspace_dirs: Optional[List[str]] = None


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Cursus Config UI API",
        "version": "1.0.0",
        "phase": "Phase 1 - Core Infrastructure",
        "endpoints": {
            "discovery": "/api/discover-configs",
            "info": "/api/config-info",
            "validate": "/api/validate-config",
            "examples": "/api/examples",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "phase": "Phase 1", "timestamp": "2025-10-07"}


@app.post("/api/discover-configs")
async def discover_configs(request: ConfigDiscoveryRequest):
    """
    Discover available configuration classes.
    
    Args:
        request: Discovery request with optional workspace directories
        
    Returns:
        Dictionary of available configuration classes
    """
    try:
        logger.info(f"Discovering configs with workspace_dirs: {request.workspace_dirs}")
        
        config_classes = discover_available_configs(workspace_dirs=request.workspace_dirs)
        
        # Convert to serializable format
        result = {}
        for name, cls in config_classes.items():
            result[name] = {
                "name": name,
                "module": cls.__module__,
                "docstring": cls.__doc__ or "No documentation available",
                "has_from_base_config": hasattr(cls, 'from_base_config')
            }
        
        return {
            "success": True,
            "count": len(result),
            "config_classes": result
        }
        
    except Exception as e:
        logger.error(f"Error discovering configs: {e}")
        raise HTTPException(status_code=500, detail=f"Discovery failed: {str(e)}")


@app.post("/api/config-info")
async def get_config_information(request: ConfigInfoRequest):
    """
    Get detailed information about a configuration class.
    
    Args:
        request: Config info request
        
    Returns:
        Detailed configuration class information
    """
    try:
        logger.info(f"Getting config info for: {request.config_class_name}")
        
        info = get_config_info(request.config_class_name, workspace_dirs=request.workspace_dirs)
        
        if not info["found"]:
            raise HTTPException(
                status_code=404, 
                detail=f"Configuration class '{request.config_class_name}' not found. "
                       f"Available: {info['available_classes']}"
            )
        
        # Make serializable
        serializable_info = {
            "found": info["found"],
            "config_class_name": info["config_class_name"],
            "fields": info["fields"],
            "inheritance_chain": info["inheritance_chain"],
            "field_count": info["field_count"],
            "required_fields": info["required_fields"],
            "optional_fields": info["optional_fields"],
            "has_from_base_config": info["has_from_base_config"],
            "docstring": info["docstring"]
        }
        
        return {
            "success": True,
            "config_info": serializable_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting config info: {e}")
        raise HTTPException(status_code=500, detail=f"Info retrieval failed: {str(e)}")


@app.post("/api/validate-config")
async def validate_configuration(request: ConfigValidationRequest):
    """
    Validate a configuration instance.
    
    Args:
        request: Validation request
        
    Returns:
        Validation results
    """
    try:
        logger.info(f"Validating config: {request.config_class_name}")
        
        # Import config from dict
        config_instance = import_config_from_dict(
            request.config_dict, 
            request.config_class_name,
            workspace_dirs=request.workspace_dirs
        )
        
        # Validate
        validation_result = validate_config_instance(config_instance)
        
        # Make serializable
        serializable_result = {
            "valid": validation_result["valid"],
            "config_type": validation_result["config_type"],
            "errors": validation_result["errors"]
        }
        
        if validation_result["valid"] and validation_result["validated_instance"]:
            serializable_result["validated_data"] = export_config_to_dict(
                validation_result["validated_instance"]
            )
        
        return {
            "success": True,
            "validation": serializable_result
        }
        
    except Exception as e:
        logger.error(f"Error validating config: {e}")
        return {
            "success": False,
            "validation": {
                "valid": False,
                "config_type": request.config_class_name,
                "errors": [str(e)]
            }
        }


@app.get("/api/examples")
async def get_examples():
    """
    Get example configurations for testing.
    
    Returns:
        Example configuration instances
    """
    try:
        logger.info("Creating example configurations")
        
        # Create examples
        base_config = create_example_base_config()
        processing_config = create_example_processing_config(base_config)
        
        return {
            "success": True,
            "examples": {
                "base_config": {
                    "type": "BasePipelineConfig",
                    "data": export_config_to_dict(base_config)
                },
                "processing_config": {
                    "type": "ProcessingStepConfigBase", 
                    "data": export_config_to_dict(processing_config)
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error creating examples: {e}")
        raise HTTPException(status_code=500, detail=f"Example creation failed: {str(e)}")


@app.post("/api/import-config")
async def import_configuration(request: ConfigImportRequest):
    """
    Import configuration from dictionary.
    
    Args:
        request: Import request
        
    Returns:
        Imported configuration data
    """
    try:
        logger.info(f"Importing config: {request.config_class_name}")
        
        config_instance = import_config_from_dict(
            request.config_dict,
            request.config_class_name,
            workspace_dirs=request.workspace_dirs
        )
        
        return {
            "success": True,
            "imported_config": {
                "type": request.config_class_name,
                "data": export_config_to_dict(config_instance)
            }
        }
        
    except Exception as e:
        logger.error(f"Error importing config: {e}")
        raise HTTPException(status_code=400, detail=f"Import failed: {str(e)}")


# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

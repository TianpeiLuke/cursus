"""
FastAPI endpoints for universal configuration management

This module provides REST API endpoints for the universal configuration system,
enabling web-based configuration management for all Cursus pipeline components.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from fastapi import APIRouter, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pathlib import Path

from .core import UniversalConfigCore
from .utils import discover_available_configs
from .specialized_widgets import SpecializedComponentRegistry

logger = logging.getLogger(__name__)

# Global state management (similar to Cradle UI pattern)
latest_config = None
active_sessions = {}

# Create router
router = APIRouter(prefix="/api/config-ui", tags=["config-ui"])


# Request/Response Models
class ConfigDiscoveryRequest(BaseModel):
    """Request model for configuration discovery."""
    workspace_dirs: Optional[List[str]] = Field(None, description="Optional workspace directories")


class ConfigDiscoveryResponse(BaseModel):
    """Response model for configuration discovery."""
    configs: Dict[str, Dict[str, Any]] = Field(description="Discovered configuration classes")
    count: int = Field(description="Number of discovered configurations")


class ConfigWidgetRequest(BaseModel):
    """Request model for creating configuration widgets."""
    config_class_name: str = Field(description="Name of the configuration class")
    base_config: Optional[Dict[str, Any]] = Field(None, description="Optional base configuration")
    workspace_dirs: Optional[List[str]] = Field(None, description="Optional workspace directories")


class ConfigWidgetResponse(BaseModel):
    """Response model for configuration widgets."""
    config_class_name: str = Field(description="Configuration class name")
    fields: List[Dict[str, Any]] = Field(description="Form fields")
    values: Dict[str, Any] = Field(description="Pre-populated values")
    specialized_component: bool = Field(False, description="Whether this uses a specialized component")


class ConfigSaveRequest(BaseModel):
    """Request model for saving configurations."""
    config_class_name: str = Field(description="Configuration class name")
    form_data: Dict[str, Any] = Field(description="Form data to save")


class ConfigSaveResponse(BaseModel):
    """Response model for saved configurations."""
    success: bool = Field(description="Whether save was successful")
    config: Dict[str, Any] = Field(description="Saved configuration")
    config_type: str = Field(description="Configuration type")
    python_code: Optional[str] = Field(None, description="Generated Python code")


class PipelineWizardRequest(BaseModel):
    """Request model for pipeline wizard creation."""
    dag: Dict[str, Any] = Field(description="DAG definition")
    base_config: Optional[Dict[str, Any]] = Field(None, description="Base configuration")
    processing_config: Optional[Dict[str, Any]] = Field(None, description="Processing configuration")


class PipelineWizardResponse(BaseModel):
    """Response model for pipeline wizard."""
    steps: List[Dict[str, Any]] = Field(description="Wizard steps")
    wizard_id: str = Field(description="Wizard identifier")


# Endpoints
@router.post("/discover", response_model=ConfigDiscoveryResponse)
async def discover_configurations(request: ConfigDiscoveryRequest):
    """
    Discover available configuration classes.
    
    Args:
        request: Discovery request with optional workspace directories
        
    Returns:
        ConfigDiscoveryResponse with discovered configurations
    """
    try:
        logger.info(f"Discovering configurations with workspace_dirs: {request.workspace_dirs}")
        
        # Use utility function to discover configurations
        configs = discover_available_configs(workspace_dirs=request.workspace_dirs)
        
        # Format response
        formatted_configs = {}
        for name, config_class in configs.items():
            formatted_configs[name] = {
                "module": getattr(config_class, "__module__", "unknown"),
                "description": getattr(config_class, "__doc__", "").split('\n')[0] if getattr(config_class, "__doc__", None) else None,
                "field_count": len(getattr(config_class, "model_fields", {}))
            }
        
        logger.info(f"Successfully discovered {len(configs)} configurations")
        
        return ConfigDiscoveryResponse(
            configs=formatted_configs,
            count=len(configs)
        )
        
    except Exception as e:
        logger.error(f"Configuration discovery failed: {e}")
        raise HTTPException(status_code=500, detail=f"Discovery failed: {str(e)}")


@router.post("/create-widget", response_model=ConfigWidgetResponse)
async def create_configuration_widget(request: ConfigWidgetRequest):
    """
    Create a configuration widget for the specified class.
    
    Args:
        request: Widget creation request
        
    Returns:
        ConfigWidgetResponse with widget data
    """
    try:
        logger.info(f"Creating widget for {request.config_class_name}")
        
        # Check for specialized components
        registry = SpecializedComponentRegistry()
        
        if registry.has_specialized_component(request.config_class_name):
            logger.info(f"Using specialized component for {request.config_class_name}")
            return ConfigWidgetResponse(
                config_class_name=request.config_class_name,
                fields=[],
                values={},
                specialized_component=True
            )
        
        # Create widget data using core directly (not the Jupyter widget)
        from .core import UniversalConfigCore
        
        core = UniversalConfigCore(workspace_dirs=request.workspace_dirs)
        config_classes = core.discover_config_classes()
        config_class = config_classes.get(request.config_class_name)
        
        if not config_class:
            raise HTTPException(
                status_code=404,
                detail=f"Configuration class '{request.config_class_name}' not found"
            )
        
        # Get form fields
        fields = core._get_form_fields(config_class)
        
        # Get pre-populated values if base_config provided
        values = {}
        if request.base_config and hasattr(config_class, 'from_base_config'):
            try:
                pre_populated = config_class.from_base_config(request.base_config)
                values = pre_populated.model_dump() if hasattr(pre_populated, 'model_dump') else {}
            except Exception as e:
                logger.warning(f"Failed to pre-populate config: {e}")
                values = {}
        
        logger.info(f"Successfully created widget data for {request.config_class_name}")
        
        return ConfigWidgetResponse(
            config_class_name=request.config_class_name,
            fields=fields,
            values=values,
            specialized_component=False
        )
        
    except Exception as e:
        logger.error(f"Widget creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Widget creation failed: {str(e)}")


@router.post("/save-config", response_model=ConfigSaveResponse)
async def save_configuration(request: ConfigSaveRequest):
    """
    Save a configuration from form data.
    
    Args:
        request: Configuration save request
        
    Returns:
        ConfigSaveResponse with saved configuration
    """
    global latest_config
    
    try:
        logger.info(f"Saving configuration for {request.config_class_name}")
        
        # Discover configuration class
        configs = discover_available_configs()
        config_class = configs.get(request.config_class_name)
        
        if not config_class:
            raise HTTPException(
                status_code=404, 
                detail=f"Configuration class '{request.config_class_name}' not found"
            )
        
        # Create configuration instance with enhanced Pydantic error handling
        try:
            config_instance = config_class(**request.form_data)
        except Exception as validation_error:
            # Handle Pydantic validation errors specifically
            if hasattr(validation_error, 'errors'):
                # Pydantic ValidationError - format for frontend
                validation_details = []
                for error in validation_error.errors():
                    field_path = '.'.join(str(loc) for loc in error['loc'])
                    validation_details.append({
                        'field': field_path,
                        'message': error['msg'],
                        'type': error['type'],
                        'input': error.get('input', 'N/A')
                    })
                
                raise HTTPException(
                    status_code=422,
                    detail={
                        'error_type': 'validation_error',
                        'message': 'Configuration validation failed',
                        'validation_errors': validation_details
                    }
                )
            else:
                # Other configuration errors
                raise HTTPException(
                    status_code=400,
                    detail={
                        'error_type': 'configuration_error',
                        'message': f'Configuration creation failed: {str(validation_error)}'
                    }
                )
        
        # Convert to dictionary
        if hasattr(config_instance, 'model_dump'):
            config_dict = config_instance.model_dump()
        else:
            config_dict = config_instance.__dict__
        
        # Store latest configuration globally (Cradle UI pattern)
        latest_config = {
            "config": config_dict,
            "config_type": request.config_class_name,
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }
        
        # Generate Python code (optional)
        python_code = None
        try:
            python_code = f"""# {request.config_class_name} Configuration
from cursus.steps.configs import {request.config_class_name}

config = {request.config_class_name}(
{_format_python_args(request.form_data)}
)
"""
        except Exception as e:
            logger.warning(f"Failed to generate Python code: {e}")
        
        logger.info(f"Successfully saved configuration for {request.config_class_name}")
        
        return ConfigSaveResponse(
            success=True,
            config=config_dict,
            config_type=request.config_class_name,
            python_code=python_code
        )
        
    except Exception as e:
        logger.error(f"Configuration save failed: {e}")
        raise HTTPException(status_code=500, detail=f"Save failed: {str(e)}")


@router.post("/create-pipeline-wizard", response_model=PipelineWizardResponse)
async def create_pipeline_wizard(request: PipelineWizardRequest):
    """
    Create a pipeline configuration wizard from DAG definition.
    
    Args:
        request: Pipeline wizard creation request
        
    Returns:
        PipelineWizardResponse with wizard data
    """
    try:
        logger.info("Creating pipeline configuration wizard")
        
        # For now, return a placeholder response
        # Full implementation would use the DAG to create wizard steps
        steps = [
            {
                "title": "Base Configuration",
                "config_class_name": "BasePipelineConfig",
                "required": True
            },
            {
                "title": "Processing Configuration", 
                "config_class_name": "ProcessingStepConfigBase",
                "required": True
            }
        ]
        
        # Add steps based on DAG nodes (simplified)
        if "nodes" in request.dag:
            for node in request.dag["nodes"]:
                if "config_type" in node:
                    steps.append({
                        "title": node.get("name", node["config_type"]),
                        "config_class_name": node["config_type"],
                        "required": True
                    })
        
        wizard_id = f"wizard_{len(steps)}_{hash(str(request.dag)) % 10000}"
        
        logger.info(f"Created pipeline wizard with {len(steps)} steps")
        
        return PipelineWizardResponse(
            steps=steps,
            wizard_id=wizard_id
        )
        
    except Exception as e:
        logger.error(f"Pipeline wizard creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Wizard creation failed: {str(e)}")


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
            detail="No configuration available. Please complete the configuration wizard first."
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


# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "config-ui", "phase": "Enhanced with Cradle UI patterns"}


# Static file serving
def setup_static_files(app):
    """Setup static file serving for the web interface."""
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/config-ui/static", StaticFiles(directory=str(static_dir)), name="config-ui-static")
        
        @app.get("/config-ui")
        async def serve_config_ui():
            """Serve the main config UI page."""
            return FileResponse(str(static_dir / "index.html"))


# Helper functions
def _format_python_args(form_data: Dict[str, Any], indent: int = 4) -> str:
    """Format form data as Python constructor arguments."""
    lines = []
    indent_str = " " * indent
    
    for key, value in form_data.items():
        if isinstance(value, str):
            lines.append(f'{indent_str}{key}="{value}",')
        elif isinstance(value, (list, dict)):
            lines.append(f'{indent_str}{key}={repr(value)},')
        else:
            lines.append(f'{indent_str}{key}={value},')
    
    return "\n".join(lines)


# Factory function to create FastAPI app with config UI
def create_config_ui_app():
    """Create FastAPI app with config UI endpoints."""
    from fastapi import FastAPI
    
    app = FastAPI(
        title="Cursus Config UI",
        description="Universal Configuration Management Interface",
        version="2.0.0"
    )
    
    # Include router
    app.include_router(router)
    
    # Setup static files
    setup_static_files(app)
    
    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "message": "Cursus Config UI API",
            "version": "2.0.0",
            "phase": "Phase 2 - Specialized Components",
            "web_interface": "/config-ui",
            "api_docs": "/docs"
        }
    
    return app


# For direct execution
if __name__ == "__main__":
    import uvicorn
    app = create_config_ui_app()
    uvicorn.run(app, host="0.0.0.0", port=8003)

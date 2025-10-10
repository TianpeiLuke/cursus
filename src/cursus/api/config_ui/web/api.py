"""
FastAPI endpoints for universal configuration management

This module provides REST API endpoints for the universal configuration system,
enabling web-based configuration management for all Cursus pipeline components.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Union
from fastapi import APIRouter, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pathlib import Path

# Handle both relative and absolute imports using centralized path setup
try:
    # Try relative imports first (when run as module)
    from ..core.core import UniversalConfigCore
    from ..core.utils import discover_available_configs
    from ..widgets.specialized_widgets import SpecializedComponentRegistry
except ImportError:
    # Fallback: Set up cursus path and use absolute imports
    import sys
    from pathlib import Path

    # Add the core directory to path for import_utils
    current_dir = Path(__file__).parent
    core_dir = current_dir.parent / "core"
    if str(core_dir) not in sys.path:
        sys.path.insert(0, str(core_dir))

    from import_utils import ensure_cursus_path

    ensure_cursus_path()

    from cursus.api.config_ui.core.core import UniversalConfigCore
    from cursus.api.config_ui.core.utils import discover_available_configs
    from cursus.api.config_ui.widgets.specialized_widgets import (
        SpecializedComponentRegistry,
    )

logger = logging.getLogger(__name__)

# Global state management (similar to Cradle UI pattern)
latest_config = None
active_sessions = {}

# Create router
router = APIRouter(prefix="/api/config-ui", tags=["config-ui"])


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
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', filename)
    sanitized = sanitized.replace('..', '')  # Remove path traversal attempts
    sanitized = sanitized.strip('. ')  # Remove leading/trailing dots and spaces
    
    # Ensure filename is not empty after sanitization
    if not sanitized:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    # Limit filename length
    if len(sanitized) > 255:
        sanitized = sanitized[:255]
    
    return sanitized


def sanitize_download_id(download_id: str) -> str:
    """
    Sanitize download ID to prevent path traversal attacks.
    
    Args:
        download_id: The download ID to sanitize
        
    Returns:
        str: Sanitized download ID
        
    Raises:
        HTTPException: If download ID contains invalid characters
    """
    if not download_id:
        raise HTTPException(status_code=400, detail="Download ID cannot be empty")
    
    # Only allow alphanumeric characters and underscores
    if not re.match(r'^[a-zA-Z0-9_]+$', download_id):
        raise HTTPException(status_code=400, detail="Invalid download ID format")
    
    # Limit length
    if len(download_id) > 100:
        raise HTTPException(status_code=400, detail="Download ID too long")
    
    return download_id


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
                status_code=403, 
                detail="Access denied: Path outside allowed directory"
            )
        
        return abs_file_path
    except Exception as e:
        logger.error(f"Path validation failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid file path")


# Request/Response Models
class ConfigDiscoveryRequest(BaseModel):
    """Request model for configuration discovery."""

    workspace_dirs: Optional[List[str]] = Field(
        None, description="Optional workspace directories"
    )


class ConfigDiscoveryResponse(BaseModel):
    """Response model for configuration discovery."""

    configs: Dict[str, Dict[str, Any]] = Field(
        description="Discovered configuration classes"
    )
    count: int = Field(description="Number of discovered configurations")


class ConfigWidgetRequest(BaseModel):
    """Request model for creating configuration widgets."""

    config_class_name: str = Field(description="Name of the configuration class")
    base_config: Optional[Dict[str, Any]] = Field(
        None, description="Optional base configuration"
    )
    workspace_dirs: Optional[List[str]] = Field(
        None, description="Optional workspace directories"
    )


class ConfigWidgetResponse(BaseModel):
    """Response model for configuration widgets."""

    config_class_name: str = Field(description="Configuration class name")
    fields: List[Dict[str, Any]] = Field(description="Form fields")
    values: Dict[str, Any] = Field(description="Pre-populated values")
    specialized_component: bool = Field(
        False, description="Whether this uses a specialized component"
    )


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
    base_config: Optional[Dict[str, Any]] = Field(
        None, description="Base configuration"
    )
    processing_config: Optional[Dict[str, Any]] = Field(
        None, description="Processing configuration"
    )


class PipelineWizardResponse(BaseModel):
    """Response model for pipeline wizard."""

    steps: List[Dict[str, Any]] = Field(description="Wizard steps")
    wizard_id: str = Field(description="Wizard identifier")


class MergeConfigsRequest(BaseModel):
    """Request model for merging and saving all configurations."""

    session_configs: Dict[str, Dict[str, Any]] = Field(
        description="All configurations from current session"
    )
    filename: Optional[str] = Field(
        None, description="Optional filename for the merged config"
    )
    workspace_dirs: Optional[List[str]] = Field(
        None, description="Optional workspace directories"
    )


class MergeConfigsResponse(BaseModel):
    """Response model for merged configurations."""

    success: bool = Field(description="Whether merge was successful")
    merged_config: Dict[str, Any] = Field(
        description="The merged configuration structure"
    )
    filename: str = Field(description="Generated filename")
    download_url: str = Field(description="URL to download the merged config file")


class DAGAnalysisRequest(BaseModel):
    """Request model for DAG analysis."""

    pipeline_dag: Dict[str, Any] = Field(description="Pipeline DAG definition")
    workspace_dirs: Optional[List[str]] = Field(
        None, description="Optional workspace directories"
    )


class DAGAnalysisResponse(BaseModel):
    """Response model for DAG analysis."""

    discovered_steps: List[Dict[str, Any]] = Field(
        description="Discovered pipeline steps"
    )
    required_configs: List[Dict[str, Any]] = Field(
        description="Required configuration classes"
    )
    workflow_steps: List[Dict[str, Any]] = Field(
        description="Generated workflow structure"
    )
    total_steps: int = Field(description="Total number of workflow steps")
    hidden_configs_count: int = Field(
        description="Number of hidden/unused config types"
    )


class DAGCatalogResponse(BaseModel):
    """Response model for DAG catalog discovery."""

    dags: List[Dict[str, Any]] = Field(
        description="Available DAG definitions from catalog"
    )
    count: int = Field(description="Number of available DAGs")


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
        logger.info(
            f"Discovering configurations with workspace_dirs: {request.workspace_dirs}"
        )

        # Use utility function to discover configurations
        configs = discover_available_configs(workspace_dirs=request.workspace_dirs)

        # Format response
        formatted_configs = {}
        for name, config_class in configs.items():
            formatted_configs[name] = {
                "module": getattr(config_class, "__module__", "unknown"),
                "description": getattr(config_class, "__doc__", "").split("\n")[0]
                if getattr(config_class, "__doc__", None)
                else None,
                "field_count": len(getattr(config_class, "model_fields", {})),
            }

        logger.info(f"Successfully discovered {len(configs)} configurations")

        return ConfigDiscoveryResponse(configs=formatted_configs, count=len(configs))

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
                specialized_component=True,
            )

        # Create widget data using core directly (not the Jupyter widget)
        from ..core.core import UniversalConfigCore

        core = UniversalConfigCore(workspace_dirs=request.workspace_dirs)
        config_classes = core.discover_config_classes()
        config_class = config_classes.get(request.config_class_name)

        if not config_class:
            raise HTTPException(
                status_code=404,
                detail=f"Configuration class '{request.config_class_name}' not found",
            )

        # Get form fields with proper serialization handling
        fields = core._get_form_fields(config_class)

        # Clean fields to ensure they're serializable
        cleaned_fields = []
        for field in fields:
            cleaned_field = {}
            for key, value in field.items():
                if key == "default":
                    # Handle PydanticUndefinedType and other non-serializable defaults
                    if value is not None:
                        try:
                            # Check if value is serializable
                            import json

                            json.dumps(value)  # Test serialization
                            cleaned_field[key] = value
                        except (TypeError, ValueError):
                            # If not serializable, convert to string or set to None
                            if hasattr(value, "__name__"):
                                cleaned_field[key] = str(value.__name__)
                            elif (
                                str(type(value))
                                == "<class 'pydantic_core._pydantic_core.PydanticUndefinedType'>"
                            ):
                                cleaned_field[key] = None
                            else:
                                try:
                                    cleaned_field[key] = str(value)
                                except:
                                    cleaned_field[key] = None
                    else:
                        cleaned_field[key] = None
                else:
                    cleaned_field[key] = value
            cleaned_fields.append(cleaned_field)

        # Get pre-populated values if base_config provided
        values = {}
        if request.base_config and hasattr(config_class, "from_base_config"):
            try:
                pre_populated = config_class.from_base_config(request.base_config)
                values = (
                    pre_populated.model_dump()
                    if hasattr(pre_populated, "model_dump")
                    else {}
                )
            except Exception as e:
                logger.warning(f"Failed to pre-populate config: {e}")
                values = {}

        logger.info(f"Successfully created widget data for {request.config_class_name}")

        return ConfigWidgetResponse(
            config_class_name=request.config_class_name,
            fields=cleaned_fields,
            values=values,
            specialized_component=False,
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
                detail=f"Configuration class '{request.config_class_name}' not found",
            )

        # Create configuration instance with enhanced Pydantic error handling
        try:
            config_instance = config_class(**request.form_data)
        except Exception as validation_error:
            # Handle Pydantic validation errors specifically
            if hasattr(validation_error, "errors"):
                # Pydantic ValidationError - format for frontend
                validation_details = []
                for error in validation_error.errors():
                    field_path = ".".join(str(loc) for loc in error["loc"])
                    validation_details.append(
                        {
                            "field": field_path,
                            "message": error["msg"],
                            "type": error["type"],
                            "input": error.get("input", "N/A"),
                        }
                    )

                raise HTTPException(
                    status_code=422,
                    detail={
                        "error_type": "validation_error",
                        "message": "Configuration validation failed",
                        "validation_errors": validation_details,
                    },
                )
            else:
                # Other configuration errors
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error_type": "configuration_error",
                        "message": f"Configuration creation failed: {str(validation_error)}",
                    },
                )

        # Convert to dictionary
        if hasattr(config_instance, "model_dump"):
            config_dict = config_instance.model_dump()
        else:
            config_dict = config_instance.__dict__

        # Store latest configuration globally (Cradle UI pattern)
        latest_config = {
            "config": config_dict,
            "config_type": request.config_class_name,
            "timestamp": __import__("datetime").datetime.now().isoformat(),
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
            python_code=python_code,
        )

    except Exception as e:
        logger.error(f"Configuration save failed: {e}")
        raise HTTPException(status_code=500, detail=f"Save failed: {str(e)}")


@router.post("/analyze-dag", response_model=DAGAnalysisResponse)
async def analyze_pipeline_dag(request: DAGAnalysisRequest):
    """
    Analyze PipelineDAG to discover required configuration classes.

    This endpoint implements the DAG-driven discovery approach from the design,
    providing intelligent configuration filtering based on actual pipeline needs.

    Args:
        request: DAG analysis request with pipeline definition

    Returns:
        DAGAnalysisResponse with discovered steps and workflow structure
    """
    try:
        logger.info("Analyzing pipeline DAG for configuration discovery")

        # Create core engine for DAG analysis
        core = UniversalConfigCore(workspace_dirs=request.workspace_dirs)

        # Create a mock DAG object from the request data
        class MockDAG:
            def __init__(self, dag_data):
                self.nodes = dag_data.get("nodes", [])
                if isinstance(self.nodes, list) and len(self.nodes) > 0:
                    # Handle list of node names or node objects
                    if isinstance(self.nodes[0], str):
                        self.nodes = self.nodes  # Already list of strings
                    else:
                        # Extract node names from objects
                        self.nodes = [
                            node.get("name", str(i))
                            for i, node in enumerate(self.nodes)
                        ]

        mock_dag = MockDAG(request.pipeline_dag)

        # Extract DAG nodes for analysis
        dag_nodes = list(mock_dag.nodes) if hasattr(mock_dag, "nodes") else []

        # Discover required config classes using the new discovery methods
        resolver = None
        try:
            from cursus.step_catalog.adapters.config_resolver import (
                StepConfigResolverAdapter,
            )

            resolver = StepConfigResolverAdapter()
        except ImportError:
            logger.warning(
                "StepConfigResolverAdapter not available, using fallback discovery"
            )

        required_config_classes = core._discover_required_config_classes(
            dag_nodes, resolver
        )

        # Create workflow structure
        workflow_steps = core._create_workflow_structure(required_config_classes)

        # Format discovered steps for response
        discovered_steps = []
        for node_name in dag_nodes:
            discovered_steps.append(
                {
                    "step_name": node_name,
                    "step_type": "pipeline_step",  # Could be enhanced with actual step type detection
                    "dependencies": [],  # Could be enhanced with dependency analysis
                }
            )

        # Format required configs for response
        formatted_required_configs = []
        for config in required_config_classes:
            formatted_required_configs.append(
                {
                    "config_class_name": config["config_class_name"],
                    "node_name": config["node_name"],
                    "inheritance_pattern": config["inheritance_pattern"],
                    "is_specialized": config["is_specialized"],
                    "inferred": config.get("inferred", False),
                }
            )

        # Calculate hidden configs count
        all_configs = core.discover_config_classes()
        hidden_configs_count = (
            len(all_configs) - len(required_config_classes) - 2
        )  # -2 for base configs

        logger.info(
            f"DAG analysis complete: {len(discovered_steps)} steps, "
            f"{len(required_config_classes)} required configs, "
            f"{len(workflow_steps)} workflow steps"
        )

        return DAGAnalysisResponse(
            discovered_steps=discovered_steps,
            required_configs=formatted_required_configs,
            workflow_steps=workflow_steps,
            total_steps=len(workflow_steps),
            hidden_configs_count=max(0, hidden_configs_count),
        )

    except Exception as e:
        logger.error(f"DAG analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"DAG analysis failed: {str(e)}")


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
                "required": True,
            },
            {
                "title": "Processing Configuration",
                "config_class_name": "ProcessingStepConfigBase",
                "required": True,
            },
        ]

        # Add steps based on DAG nodes (simplified)
        if "nodes" in request.dag:
            for node in request.dag["nodes"]:
                if "config_type" in node:
                    steps.append(
                        {
                            "title": node.get("name", node["config_type"]),
                            "config_class_name": node["config_type"],
                            "required": True,
                        }
                    )

        wizard_id = f"wizard_{len(steps)}_{hash(str(request.dag)) % 10000}"

        logger.info(f"Created pipeline wizard with {len(steps)} steps")

        return PipelineWizardResponse(steps=steps, wizard_id=wizard_id)

    except Exception as e:
        logger.error(f"Pipeline wizard creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Wizard creation failed: {str(e)}")


@router.post("/merge-and-save-configs", response_model=MergeConfigsResponse)
async def merge_and_save_configurations(request: MergeConfigsRequest):
    """
    Merge and save all configurations using the same logic as demo_config.ipynb.

    This endpoint replicates the merge_and_save_configs() experience from the notebook,
    creating a unified hierarchical JSON structure with shared vs specific fields.

    Args:
        request: Merge request with all session configurations

    Returns:
        MergeConfigsResponse with merged config and download info
    """
    try:
        logger.info(f"Merging {len(request.session_configs)} configurations")

        # Import the merge_and_save_configs function
        from cursus.core.config_fields import merge_and_save_configs

        # Discover configuration classes
        configs = discover_available_configs(workspace_dirs=request.workspace_dirs)

        # Create config instances from session data
        config_list = []
        for config_class_name, form_data in request.session_configs.items():
            config_class = configs.get(config_class_name)
            if not config_class:
                logger.warning(
                    f"Configuration class '{config_class_name}' not found, skipping"
                )
                continue

            try:
                # Create configuration instance
                config_instance = config_class(**form_data)
                config_list.append(config_instance)
                logger.info(f"Created {config_class_name} instance")
            except Exception as e:
                logger.error(f"Failed to create {config_class_name} instance: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to create {config_class_name}: {str(e)}",
                )

        if not config_list:
            raise HTTPException(
                status_code=400, detail="No valid configurations to merge"
            )

        # Generate filename if not provided
        if not request.filename:
            # Extract region and model info from configs for naming
            region = "NA"  # Default
            model_class = "xgboost"  # Default
            service_name = "pipeline"  # Default

            # Try to extract from base config if available
            for config in config_list:
                if hasattr(config, "region") and config.region:
                    region = config.region
                if hasattr(config, "service_name") and config.service_name:
                    service_name = config.service_name
                # Could add model_class detection logic here

            filename = f"config_{region}_{model_class}_{service_name}_v2.json"
        else:
            filename = request.filename
            if not filename.endswith(".json"):
                filename += ".json"

        # Sanitize filename to prevent path traversal attacks
        filename = sanitize_filename(filename)

        # Create temporary file path for merge_and_save_configs
        import tempfile
        import os

        temp_dir = tempfile.mkdtemp()
        # Validate that the file path is within the temp directory
        temp_file_path = validate_file_path(os.path.join(temp_dir, filename), temp_dir)

        try:
            # Call the actual merge_and_save_configs function
            merged_config = merge_and_save_configs(
                config_list=config_list,
                output_file=temp_file_path,
                workspace_dirs=request.workspace_dirs,
            )

            # Read the generated file
            with open(temp_file_path, "r") as f:
                merged_json_content = f.read()

            # Store the merged config for download
            # In a real implementation, you'd store this in a proper file storage system
            download_id = f"merged_{hash(merged_json_content) % 100000}"

            # For now, we'll store it in a simple in-memory cache
            # In production, use proper file storage (S3, local filesystem, etc.)
            if not hasattr(merge_and_save_configurations, "_download_cache"):
                merge_and_save_configurations._download_cache = {}

            merge_and_save_configurations._download_cache[download_id] = {
                "content": merged_json_content,
                "filename": filename,
                "content_type": "application/json",
            }

            logger.info(
                f"Successfully merged {len(config_list)} configurations into {filename}"
            )

            return MergeConfigsResponse(
                success=True,
                merged_config=merged_config,
                filename=filename,
                download_url=f"/api/config-ui/download/{download_id}",
            )

        finally:
            # Clean up temporary file
            try:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                os.rmdir(temp_dir)
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temp file: {cleanup_error}")

    except Exception as e:
        logger.error(f"Configuration merge failed: {e}")
        raise HTTPException(status_code=500, detail=f"Merge failed: {str(e)}")


@router.get("/download/{download_id}")
async def download_merged_config(download_id: str):
    """
    Download a merged configuration file.

    Args:
        download_id: The download identifier from merge response

    Returns:
        FileResponse with the merged configuration JSON
    """
    try:
        # Sanitize download ID to prevent path traversal attacks
        download_id = sanitize_download_id(download_id)
        
        # Get cached download data
        if not hasattr(merge_and_save_configurations, "_download_cache"):
            raise HTTPException(status_code=404, detail="Download not found")

        download_data = merge_and_save_configurations._download_cache.get(download_id)
        if not download_data:
            raise HTTPException(status_code=404, detail="Download not found or expired")

        # Sanitize the filename from cached data
        safe_filename = sanitize_filename(download_data["filename"])

        # Create temporary file for download
        import tempfile
        import os

        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)

        try:
            temp_file.write(download_data["content"])
            temp_file.close()

            return FileResponse(
                path=temp_file.name,
                filename=safe_filename,
                media_type=download_data["content_type"],
            )
        except Exception as e:
            # Clean up on error
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            raise e

    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


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


# Health check endpoint
@router.get("/catalog/dags", response_model=DAGCatalogResponse)
async def get_dag_catalog():
    """
    Discover and return available DAG definitions from the pipeline catalog.

    Uses the existing get_all_shared_dags() function to properly handle imports
    and avoid relative import issues.

    Returns:
        DAGCatalogResponse with available DAG definitions
    """
    try:
        logger.info("Discovering DAG catalog using get_all_shared_dags()")

        # Use the existing function that properly handles imports
        try:
            from cursus.pipeline_catalog.shared_dags import get_all_shared_dags

            shared_dags_metadata = get_all_shared_dags()
            logger.info(
                f"Found {len(shared_dags_metadata)} DAGs from get_all_shared_dags()"
            )
        except ImportError as e:
            logger.error(f"Failed to import get_all_shared_dags: {e}")
            return DAGCatalogResponse(dags=[], count=0)

        discovered_dags = []

        # Process each DAG metadata
        for dag_id, metadata in shared_dags_metadata.items():
            try:
                # Parse the DAG ID to get framework and name
                if "." in dag_id:
                    framework, dag_name = dag_id.split(".", 1)
                else:
                    framework = "unknown"
                    dag_name = dag_id

                # Handle complexity mapping for compatibility
                complexity = metadata.complexity
                if complexity == "medium":
                    complexity = "standard"  # Map medium to standard

                # Create DAG info for frontend
                dag_info = {
                    "id": dag_id.replace(".", "_"),
                    "name": metadata.extra_metadata.get("name", dag_name),
                    "display_name": f"{framework.title()} - {metadata.description}",
                    "framework": framework,
                    "description": metadata.description,
                    "complexity": complexity,
                    "features": metadata.features,
                    "node_count": metadata.node_count,
                    "edge_count": metadata.edge_count,
                    "dag_structure": {
                        "nodes": [],  # Will be populated when DAG is actually loaded
                        "edges": [],
                    },
                }

                # Try to get the actual DAG structure if possible
                try:
                    if framework == "xgboost" and "complete_e2e" in dag_name:
                        from cursus.pipeline_catalog.shared_dags.xgboost.complete_e2e_dag import (
                            create_xgboost_complete_e2e_dag,
                        )

                        dag = create_xgboost_complete_e2e_dag()
                        dag_info["dag_structure"] = {
                            "nodes": [
                                {"name": node, "type": _get_sagemaker_step_type(node)}
                                for node in dag.nodes
                            ],
                            "edges": [
                                {"from": edge[0], "to": edge[1]} for edge in dag.edges
                            ],
                        }
                    elif framework == "xgboost" and "simple" in dag_name:
                        from cursus.pipeline_catalog.shared_dags.xgboost.simple_dag import (
                            create_xgboost_simple_dag,
                        )

                        dag = create_xgboost_simple_dag()
                        dag_info["dag_structure"] = {
                            "nodes": [
                                {"name": node, "type": "pipeline_step"}
                                for node in dag.nodes
                            ],
                            "edges": [
                                {"from": edge[0], "to": edge[1]} for edge in dag.edges
                            ],
                        }
                    elif framework == "pytorch" and "standard_e2e" in dag_name:
                        from cursus.pipeline_catalog.shared_dags.pytorch.standard_e2e_dag import (
                            create_pytorch_standard_e2e_dag,
                        )

                        dag = create_pytorch_standard_e2e_dag()
                        dag_info["dag_structure"] = {
                            "nodes": [
                                {"name": node, "type": "pipeline_step"}
                                for node in dag.nodes
                            ],
                            "edges": [
                                {"from": edge[0], "to": edge[1]} for edge in dag.edges
                            ],
                        }
                    elif framework == "dummy" and "e2e_basic" in dag_name:
                        from cursus.pipeline_catalog.shared_dags.dummy.e2e_basic_dag import (
                            create_dummy_e2e_basic_dag,
                        )

                        dag = create_dummy_e2e_basic_dag()
                        dag_info["dag_structure"] = {
                            "nodes": [
                                {"name": node, "type": "pipeline_step"}
                                for node in dag.nodes
                            ],
                            "edges": [
                                {"from": edge[0], "to": edge[1]} for edge in dag.edges
                            ],
                        }
                    # Add more DAG types as needed

                except Exception as dag_error:
                    logger.warning(
                        f"Could not load DAG structure for {dag_id}: {dag_error}"
                    )
                    # Keep the DAG info but without structure

                discovered_dags.append(dag_info)
                logger.info(f"Successfully processed DAG: {dag_info['name']}")

            except Exception as e:
                logger.error(f"Failed to process DAG {dag_id}: {e}")
                continue

        # Sort DAGs by framework and complexity
        discovered_dags.sort(key=lambda x: (x["framework"], x["complexity"], x["name"]))

        logger.info(f"Successfully discovered {len(discovered_dags)} DAGs from catalog")

        return DAGCatalogResponse(dags=discovered_dags, count=len(discovered_dags))

    except Exception as e:
        logger.error(f"DAG catalog discovery failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Catalog discovery failed: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "config-ui",
        "phase": "Enhanced with Cradle UI patterns",
    }


# Static file serving
def setup_static_files(app):
    """Setup static file serving for the web interface."""
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount(
            "/config-ui/static",
            StaticFiles(directory=str(static_dir)),
            name="config-ui-static",
        )

        @app.get("/config-ui")
        async def serve_config_ui():
            """Serve the main config UI page."""
            # Validate that index.html is within the static directory
            index_file_path = validate_file_path(
                str(static_dir / "index.html"), 
                str(static_dir)
            )
            return FileResponse(index_file_path)


# Helper functions
def _get_sagemaker_step_type(node_name: str) -> str:
    """
    Get the proper SageMaker step type for a node name using the registry.

    Args:
        node_name: The pipeline step name (e.g., "CradleDataLoading_training")

    Returns:
        str: The SageMaker step type (e.g., "Processing", "Training", etc.)
    """
    try:
        # Import the step registry
        from cursus.registry.step_names_original import STEP_NAMES

        # Extract the base step name (remove suffixes like _training, _calibration)
        base_name = node_name

        # Handle common suffixes
        suffixes = [
            "_training",
            "_calibration",
            "_validation",
            "_evaluation",
            "_inference",
        ]
        for suffix in suffixes:
            if base_name.endswith(suffix):
                base_name = base_name[: -len(suffix)]
                break

        # Look up in registry
        if base_name in STEP_NAMES:
            return STEP_NAMES[base_name]["sagemaker_step_type"]

        # Handle special cases
        if "CradleDataLoading" in node_name:
            return "CradleDataLoading"
        elif "TabularPreprocessing" in node_name:
            return "Processing"
        elif "XGBoostTraining" in node_name:
            return "Training"
        elif "PyTorchTraining" in node_name:
            return "Training"
        elif "DummyTraining" in node_name:
            return "Processing"
        elif "XGBoostModelEval" in node_name:
            return "Processing"
        elif "PyTorchModelEval" in node_name:
            return "Processing"
        elif "ModelCalibration" in node_name:
            return "Processing"
        elif "Package" in node_name:
            return "Processing"
        elif "Registration" in node_name:
            return "MimsModelRegistrationProcessing"
        elif "Payload" in node_name:
            return "Processing"
        else:
            # Default fallback
            return "Processing"

    except Exception as e:
        logger.warning(f"Failed to get SageMaker step type for {node_name}: {e}")
        return "Processing"  # Safe fallback


def _format_python_args(form_data: Dict[str, Any], indent: int = 4) -> str:
    """Format form data as Python constructor arguments."""
    lines = []
    indent_str = " " * indent

    for key, value in form_data.items():
        if isinstance(value, str):
            lines.append(f'{indent_str}{key}="{value}",')
        elif isinstance(value, (list, dict)):
            lines.append(f"{indent_str}{key}={repr(value)},")
        else:
            lines.append(f"{indent_str}{key}={value},")

    return "\n".join(lines)


# Factory function to create FastAPI app with config UI
def create_config_ui_app():
    """Create FastAPI app with config UI endpoints."""
    from fastapi import FastAPI

    app = FastAPI(
        title="Cursus Config UI",
        description="Universal Configuration Management Interface",
        version="2.0.0",
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
            "api_docs": "/docs",
        }

    return app


# For direct execution
if __name__ == "__main__":
    import uvicorn

    app = create_config_ui_app()
    uvicorn.run(app, host="0.0.0.0", port=8003)

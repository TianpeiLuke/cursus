"""
Utility functions for the pipeline catalog.

This module provides functions for loading and working with the pipeline catalog index,
including support for dual-compiler architecture with MODS integration.
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

# Type alias for pipeline entries
PipelineEntry = Dict[str, Any]

# Setup logging
logger = logging.getLogger(__name__)


def get_catalog_root() -> Path:
    """
    Get the path to the root of the pipeline catalog.
    
    Returns:
        Path: The path to the root of the pipeline catalog
    """
    return Path(os.path.dirname(os.path.abspath(__file__)))


def load_index() -> Dict[str, List[PipelineEntry]]:
    """
    Load the pipeline catalog index.
    
    Returns:
        Dict[str, List[PipelineEntry]]: The pipeline catalog index
    """
    index_path = get_catalog_root() / "index.json"
    try:
        with open(index_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to load pipeline catalog index: {e}")


def get_pipeline_by_id(pipeline_id: str) -> Optional[PipelineEntry]:
    """
    Get a pipeline entry by its ID.
    
    Args:
        pipeline_id: The ID of the pipeline to retrieve
        
    Returns:
        Optional[PipelineEntry]: The pipeline entry, or None if not found
    """
    index = load_index()
    for pipeline in index.get("pipelines", []):
        if pipeline.get("id") == pipeline_id:
            return pipeline
    return None


def filter_pipelines(
    framework: Optional[str] = None, 
    complexity: Optional[str] = None,
    features: Optional[List[str]] = None,
    tags: Optional[List[str]] = None
) -> List[PipelineEntry]:
    """
    Filter pipelines by criteria.
    
    Args:
        framework: Filter by framework (e.g., "xgboost", "pytorch")
        complexity: Filter by complexity (e.g., "simple", "standard", "advanced")
        features: Filter by features (e.g., ["training", "evaluation"])
        tags: Filter by tags
        
    Returns:
        List[PipelineEntry]: List of pipeline entries matching the criteria
    """
    index = load_index()
    pipelines = index.get("pipelines", [])
    
    results = pipelines
    
    # Apply filters
    if framework:
        results = [p for p in results if p.get("framework") == framework]
    
    if complexity:
        results = [p for p in results if p.get("complexity") == complexity]
    
    if features:
        # Pipeline must have ALL specified features
        results = [
            p for p in results 
            if all(feature in p.get("features", []) for feature in features)
        ]
    
    if tags:
        # Pipeline must have ANY of the specified tags
        results = [
            p for p in results 
            if any(tag in p.get("tags", []) for tag in tags)
        ]
    
    return results


def get_pipeline_path(pipeline_id: str) -> Optional[Path]:
    """
    Get the full path to a pipeline module.
    
    Args:
        pipeline_id: The ID of the pipeline
        
    Returns:
        Optional[Path]: The path to the pipeline module, or None if not found
    """
    pipeline = get_pipeline_by_id(pipeline_id)
    if not pipeline:
        return None
    
    return get_catalog_root() / pipeline.get("path", "")


def get_all_frameworks() -> List[str]:
    """
    Get a list of all available frameworks in the catalog.
    
    Returns:
        List[str]: List of framework names
    """
    index = load_index()
    frameworks = set()
    
    for pipeline in index.get("pipelines", []):
        if "framework" in pipeline:
            frameworks.add(pipeline["framework"])
    
    return sorted(list(frameworks))


def get_all_features() -> List[str]:
    """
    Get a list of all available features across pipelines in the catalog.
    
    Returns:
        List[str]: List of feature names
    """
    index = load_index()
    features = set()
    
    for pipeline in index.get("pipelines", []):
        if "features" in pipeline:
            features.update(pipeline["features"])
    
    return sorted(list(features))


# ============================================================================
# Dual-Compiler Architecture Support
# ============================================================================

def list_pipelines_by_compiler_type(compiler_type: str = "standard") -> List[PipelineEntry]:
    """
    List pipelines by compiler type.
    
    Args:
        compiler_type: The compiler type to filter by ("standard" or "mods")
        
    Returns:
        List[PipelineEntry]: List of pipeline entries for the specified compiler type
    """
    index = load_index()
    pipelines = index.get("pipelines", [])
    
    return [
        p for p in pipelines 
        if p.get("compiler_type", "standard") == compiler_type
    ]


def get_standard_pipelines() -> List[PipelineEntry]:
    """
    Get all standard (non-MODS) pipelines.
    
    Returns:
        List[PipelineEntry]: List of standard pipeline entries
    """
    return list_pipelines_by_compiler_type("standard")


def get_mods_pipelines() -> List[PipelineEntry]:
    """
    Get all MODS pipelines.
    
    Returns:
        List[PipelineEntry]: List of MODS pipeline entries
    """
    return list_pipelines_by_compiler_type("mods")


def is_mods_pipeline(pipeline_id: str) -> bool:
    """
    Check if a pipeline is a MODS pipeline.
    
    Args:
        pipeline_id: The ID of the pipeline to check
        
    Returns:
        bool: True if the pipeline is a MODS pipeline, False otherwise
    """
    pipeline = get_pipeline_by_id(pipeline_id)
    if not pipeline:
        return False
    
    return pipeline.get("compiler_type", "standard") == "mods"


def get_pipeline_compiler_type(pipeline_id: str) -> Optional[str]:
    """
    Get the compiler type for a pipeline.
    
    Args:
        pipeline_id: The ID of the pipeline
        
    Returns:
        Optional[str]: The compiler type ("standard" or "mods"), or None if pipeline not found
    """
    pipeline = get_pipeline_by_id(pipeline_id)
    if not pipeline:
        return None
    
    return pipeline.get("compiler_type", "standard")


def get_shared_dag_path(pipeline_id: str) -> Optional[str]:
    """
    Get the shared DAG path for a pipeline.
    
    Args:
        pipeline_id: The ID of the pipeline
        
    Returns:
        Optional[str]: The shared DAG path, or None if not found
    """
    pipeline = get_pipeline_by_id(pipeline_id)
    if not pipeline:
        return None
    
    return pipeline.get("shared_dag")


def get_mods_metadata(pipeline_id: str) -> Optional[Dict[str, Any]]:
    """
    Get MODS metadata for a pipeline.
    
    Args:
        pipeline_id: The ID of the pipeline
        
    Returns:
        Optional[Dict[str, Any]]: The MODS metadata, or None if not a MODS pipeline
    """
    pipeline = get_pipeline_by_id(pipeline_id)
    if not pipeline or not is_mods_pipeline(pipeline_id):
        return None
    
    return pipeline.get("mods_metadata", {})


def create_pipeline_from_catalog(
    pipeline_id: str,
    config_path: str,
    session,
    role: str,
    **kwargs
) -> Tuple[Any, Dict[str, Any], Any, Any]:
    """
    Create a pipeline from the catalog with automatic compiler selection.
    
    Args:
        pipeline_id: The ID of the pipeline to create
        config_path: Path to the configuration file
        session: SageMaker session
        role: IAM role for pipeline execution
        **kwargs: Additional arguments to pass to the pipeline creation function
        
    Returns:
        Tuple: (Pipeline, Report, Compiler, Template) - 4-tuple with pipeline components
        
    Raises:
        RuntimeError: If pipeline not found or creation fails
        ImportError: If MODS dependencies are missing for MODS pipelines
    """
    pipeline_entry = get_pipeline_by_id(pipeline_id)
    if not pipeline_entry:
        raise RuntimeError(f"Pipeline '{pipeline_id}' not found in catalog")
    
    compiler_type = pipeline_entry.get("compiler_type", "standard")
    pipeline_path = pipeline_entry.get("path")
    
    if not pipeline_path:
        raise RuntimeError(f"No path specified for pipeline '{pipeline_id}'")
    
    # Import the pipeline module
    try:
        # Convert path to module import format
        module_path = pipeline_path.replace("/", ".").replace(".py", "")
        if module_path.startswith("src.cursus.pipeline_catalog."):
            module_path = module_path[len("src.cursus.pipeline_catalog."):]
        
        # Import the module
        import importlib
        module = importlib.import_module(f"cursus.pipeline_catalog.{module_path}")
        
        # Get the create_pipeline function
        if not hasattr(module, "create_pipeline"):
            raise RuntimeError(f"Pipeline module '{module_path}' does not have a 'create_pipeline' function")
        
        create_pipeline_func = getattr(module, "create_pipeline")
        
        # Call the function with provided arguments
        result = create_pipeline_func(
            config_path=config_path,
            session=session,
            role=role,
            **kwargs
        )
        
        # Ensure we return a 4-tuple
        if len(result) == 3:
            # Legacy 3-tuple, add None for template
            pipeline, report, compiler = result
            return pipeline, report, compiler, None
        elif len(result) == 4:
            # Modern 4-tuple
            return result
        else:
            raise RuntimeError(f"Unexpected return format from pipeline '{pipeline_id}'")
            
    except ImportError as e:
        if compiler_type == "mods":
            logger.error(f"Failed to import MODS pipeline '{pipeline_id}': {e}")
            raise ImportError(
                f"MODS dependencies not available for pipeline '{pipeline_id}'. "
                f"Please install MODS or use a standard pipeline variant."
            ) from e
        else:
            logger.error(f"Failed to import standard pipeline '{pipeline_id}': {e}")
            raise RuntimeError(f"Failed to import pipeline '{pipeline_id}': {e}") from e
    except Exception as e:
        logger.error(f"Failed to create pipeline '{pipeline_id}': {e}")
        raise RuntimeError(f"Failed to create pipeline '{pipeline_id}': {e}") from e


def filter_pipelines_enhanced(
    framework: Optional[str] = None,
    complexity: Optional[str] = None,
    features: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    compiler_type: Optional[str] = None,
    mods_features: Optional[List[str]] = None
) -> List[PipelineEntry]:
    """
    Enhanced pipeline filtering with dual-compiler support.
    
    Args:
        framework: Filter by framework (e.g., "xgboost", "pytorch")
        complexity: Filter by complexity (e.g., "simple", "standard", "advanced")
        features: Filter by features (e.g., ["training", "evaluation"])
        tags: Filter by tags
        compiler_type: Filter by compiler type ("standard" or "mods")
        mods_features: Filter by MODS-specific features
        
    Returns:
        List[PipelineEntry]: List of pipeline entries matching the criteria
    """
    # Start with basic filtering
    results = filter_pipelines(framework, complexity, features, tags)
    
    # Apply compiler type filter
    if compiler_type:
        results = [
            p for p in results 
            if p.get("compiler_type", "standard") == compiler_type
        ]
    
    # Apply MODS features filter
    if mods_features:
        results = [
            p for p in results
            if p.get("compiler_type") == "mods" and
            all(
                feature in p.get("mods_metadata", {}).keys() or
                feature in p.get("features", [])
                for feature in mods_features
            )
        ]
    
    return results


def get_pipeline_pairs() -> List[Tuple[PipelineEntry, Optional[PipelineEntry]]]:
    """
    Get pairs of standard and MODS pipelines that share the same DAG.
    
    Returns:
        List[Tuple[PipelineEntry, Optional[PipelineEntry]]]: 
        List of (standard_pipeline, mods_pipeline) tuples
    """
    standard_pipelines = get_standard_pipelines()
    mods_pipelines = get_mods_pipelines()
    
    pairs = []
    
    for std_pipeline in standard_pipelines:
        std_dag = std_pipeline.get("shared_dag")
        if not std_dag:
            pairs.append((std_pipeline, None))
            continue
        
        # Find matching MODS pipeline
        mods_match = None
        for mods_pipeline in mods_pipelines:
            if mods_pipeline.get("shared_dag") == std_dag:
                mods_match = mods_pipeline
                break
        
        pairs.append((std_pipeline, mods_match))
    
    return pairs


def validate_index_schema() -> Dict[str, Any]:
    """
    Validate the catalog index schema.
    
    Returns:
        Dict[str, Any]: Validation results with errors and warnings
    """
    try:
        index = load_index()
    except Exception as e:
        return {
            "valid": False,
            "errors": [f"Failed to load index: {e}"],
            "warnings": []
        }
    
    errors = []
    warnings = []
    
    # Check schema version
    schema_version = index.get("schema_version")
    if not schema_version:
        warnings.append("Missing schema_version field")
    elif schema_version != "2.0":
        warnings.append(f"Unexpected schema version: {schema_version}")
    
    # Validate pipelines
    pipelines = index.get("pipelines", [])
    if not pipelines:
        errors.append("No pipelines found in index")
    
    required_fields = ["id", "name", "path", "framework", "compiler_type"]
    pipeline_ids = set()
    
    for i, pipeline in enumerate(pipelines):
        # Check required fields
        for field in required_fields:
            if field not in pipeline:
                errors.append(f"Pipeline {i}: Missing required field '{field}'")
        
        # Check unique IDs
        pipeline_id = pipeline.get("id")
        if pipeline_id:
            if pipeline_id in pipeline_ids:
                errors.append(f"Duplicate pipeline ID: {pipeline_id}")
            pipeline_ids.add(pipeline_id)
        
        # Validate compiler type
        compiler_type = pipeline.get("compiler_type", "standard")
        if compiler_type not in ["standard", "mods"]:
            errors.append(f"Pipeline {pipeline_id}: Invalid compiler_type '{compiler_type}'")
        
        # Validate MODS metadata
        if compiler_type == "mods":
            if "mods_metadata" not in pipeline:
                warnings.append(f"MODS pipeline {pipeline_id}: Missing mods_metadata")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "pipeline_count": len(pipelines),
        "standard_count": len([p for p in pipelines if p.get("compiler_type", "standard") == "standard"]),
        "mods_count": len([p for p in pipelines if p.get("compiler_type") == "mods"])
    }


# ============================================================================
# MODS Global Registry Integration
# ============================================================================

def get_mods_registry_integration():
    """
    Import and return MODS registry integration functions.
    
    Returns:
        Module: MODS registry module or None if not available
    """
    try:
        from . import mods_registry
        return mods_registry
    except ImportError as e:
        logger.warning(f"MODS registry integration not available: {e}")
        return None


def get_mods_registry_status() -> Dict[str, Any]:
    """
    Get MODS registry status with fallback.
    
    Returns:
        Dict[str, Any]: Registry status information
    """
    registry_module = get_mods_registry_integration()
    if registry_module:
        return registry_module.get_mods_registry_status()
    
    return {
        "available": False,
        "connection_status": "unavailable",
        "error": "MODS registry integration not available"
    }


def get_mods_registered_templates() -> List[Dict[str, Any]]:
    """
    Get MODS registered templates with fallback.
    
    Returns:
        List[Dict[str, Any]]: List of registered templates
    """
    registry_module = get_mods_registry_integration()
    if registry_module:
        return registry_module.get_mods_registered_templates()
    
    return []


def get_registry_template_info(template_id: str) -> Optional[Dict[str, Any]]:
    """
    Get registry template info with fallback.
    
    Args:
        template_id: Template ID to retrieve
        
    Returns:
        Optional[Dict[str, Any]]: Template information
    """
    registry_module = get_mods_registry_integration()
    if registry_module:
        return registry_module.get_registry_template_info(template_id)
    
    return None


def check_mods_integration() -> Dict[str, Any]:
    """
    Check MODS integration status with fallback.
    
    Returns:
        Dict[str, Any]: Integration status report
    """
    registry_module = get_mods_registry_integration()
    if registry_module:
        return registry_module.check_mods_integration()
    
    return {
        "mods_available": False,
        "registry_status": {"available": False},
        "integration_status": "unavailable",
        "error": "MODS registry integration not available"
    }


def get_mods_summary() -> Dict[str, Any]:
    """
    Get MODS summary with fallback.
    
    Returns:
        Dict[str, Any]: MODS summary information
    """
    registry_module = get_mods_registry_integration()
    if registry_module:
        return registry_module.get_mods_summary()
    
    return {
        "available": False,
        "message": "MODS registry integration not available"
    }

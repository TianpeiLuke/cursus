"""
Utility Functions for Pipeline Execution Document Generation

This module provides utility functions for mapping pipeline names to
configurations, DAGs, and execution document templates using the existing
catalog registry infrastructure.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

from ...api.dag.base_dag import PipelineDAG
from ..core.catalog_registry import CatalogRegistry

logger = logging.getLogger(__name__)

# Initialize catalog registry
_catalog_registry = None

def _get_catalog_registry() -> CatalogRegistry:
    """Get or create catalog registry instance."""
    global _catalog_registry
    if _catalog_registry is None:
        _catalog_registry = CatalogRegistry()
    return _catalog_registry


def get_config_path_for_pipeline(pipeline_name: str) -> str:
    """
    Get the default configuration file path for a pipeline using catalog registry.
    
    Args:
        pipeline_name: Name of the pipeline
        
    Returns:
        str: Path to the configuration file
        
    Raises:
        ValueError: If pipeline name is not recognized
    """
    try:
        registry = _get_catalog_registry()
        pipeline_node = registry.get_pipeline_node(pipeline_name)
        
        if pipeline_node is None:
            available_pipelines = registry.get_all_pipelines()
            raise ValueError(
                f"Unknown pipeline name: {pipeline_name}. "
                f"Available pipelines: {available_pipelines}"
            )
        
        # Use source_file from registry to derive config path
        source_file = pipeline_node.get("source_file", f"pipelines/{pipeline_name}.py")
        # Convert pipeline file path to config path
        config_path = source_file.replace("pipelines/", "configs/").replace(".py", ".json")
        
        logger.debug(f"Config path for {pipeline_name}: {config_path}")
        return config_path
        
    except Exception as e:
        logger.error(f"Failed to get config path for {pipeline_name}: {e}")
        raise ValueError(f"Could not determine config path for pipeline {pipeline_name}: {e}")


def load_shared_dag_for_pipeline(pipeline_name: str) -> PipelineDAG:
    """
    Load the shared DAG for a pipeline using catalog registry and dynamic import.
    
    Args:
        pipeline_name: Name of the pipeline
        
    Returns:
        PipelineDAG: The loaded DAG instance
        
    Raises:
        ValueError: If pipeline name is not recognized
        ImportError: If pipeline class cannot be imported
    """
    try:
        registry = _get_catalog_registry()
        pipeline_node = registry.get_pipeline_node(pipeline_name)
        
        if pipeline_node is None:
            available_pipelines = registry.get_all_pipelines()
            raise ValueError(
                f"Unknown pipeline name: {pipeline_name}. "
                f"Available pipelines: {available_pipelines}"
            )
        
        # Get source file from registry
        source_file = pipeline_node.get("source_file", f"pipelines/{pipeline_name}.py")
        logger.debug(f"Loading DAG for {pipeline_name} from {source_file}")
        
        # Convert source file path to module path and class name
        module_path = source_file.replace("/", ".").replace(".py", "")
        if module_path.startswith("pipelines."):
            module_path = f"...{module_path}"  # Relative import from pipeline_catalog
        
        # Derive class name from pipeline name (convert snake_case to PascalCase)
        class_name = _snake_to_pascal_case(pipeline_name) + "Pipeline"
        
        try:
            # Dynamic import of the pipeline class
            import importlib
            module = importlib.import_module(module_path, package=__package__)
            pipeline_class = getattr(module, class_name)
            
            # Create a temporary instance to get the DAG
            # Use minimal parameters to avoid requiring config file
            temp_instance = pipeline_class.__new__(pipeline_class)
            dag = temp_instance.create_dag()
            
            logger.info(f"Successfully loaded DAG for {pipeline_name}: {len(dag.nodes)} nodes, {len(dag.edges)} edges")
            return dag
            
        except (ImportError, AttributeError) as e:
            logger.warning(f"Failed to load pipeline class {class_name}, trying direct DAG import: {e}")
            
            # Fallback: try to load DAG directly from shared_dags
            return _load_dag_from_shared_dags(pipeline_name)
        
    except Exception as e:
        logger.error(f"Failed to load DAG for {pipeline_name}: {e}")
        raise ImportError(f"Could not load DAG for pipeline {pipeline_name}: {e}")


def _snake_to_pascal_case(snake_str: str) -> str:
    """Convert snake_case to PascalCase."""
    return ''.join(word.capitalize() for word in snake_str.split('_'))


def _load_dag_from_shared_dags(pipeline_name: str) -> PipelineDAG:
    """
    Fallback method to load DAG directly from shared_dags.
    
    Args:
        pipeline_name: Name of the pipeline
        
    Returns:
        PipelineDAG: The loaded DAG instance
    """
    try:
        # Map pipeline names to their DAG creation functions
        if pipeline_name == "xgb_e2e_comprehensive":
            from ..shared_dags.xgboost.complete_e2e_dag import create_xgboost_complete_e2e_dag
            return create_xgboost_complete_e2e_dag()
        elif pipeline_name == "xgb_training_simple":
            from ..shared_dags.xgboost.simple_dag import create_xgboost_simple_dag
            return create_xgboost_simple_dag()
        elif pipeline_name == "xgb_training_evaluation":
            from ..shared_dags.xgboost.training_with_evaluation_dag import create_xgboost_training_with_evaluation_dag
            return create_xgboost_training_with_evaluation_dag()
        elif pipeline_name == "xgb_training_calibrated":
            from ..shared_dags.xgboost.training_with_calibration_dag import create_xgboost_training_with_calibration_dag
            return create_xgboost_training_with_calibration_dag()
        elif pipeline_name == "pytorch_e2e_standard":
            from ..shared_dags.pytorch.standard_e2e_dag import create_pytorch_standard_e2e_dag
            return create_pytorch_standard_e2e_dag()
        elif pipeline_name == "pytorch_training_basic":
            from ..shared_dags.pytorch.training_dag import create_pytorch_training_dag
            return create_pytorch_training_dag()
        elif pipeline_name == "dummy_e2e_basic":
            from ..shared_dags.dummy.e2e_basic_dag import create_dummy_e2e_basic_dag
            return create_dummy_e2e_basic_dag()
        else:
            raise ImportError(f"No shared DAG found for pipeline: {pipeline_name}")
            
    except ImportError as e:
        logger.error(f"Failed to import shared DAG for {pipeline_name}: {e}")
        raise


def create_execution_doc_template_for_pipeline(pipeline_name: str) -> Dict[str, Any]:
    """
    Create a default execution document template for a pipeline.
    
    This function creates a basic execution document template with the
    PIPELINE_STEP_CONFIGS structure that can be filled by the execution
    document generator.
    
    Args:
        pipeline_name: Name of the pipeline
        
    Returns:
        Dict[str, Any]: Execution document template
    """
    try:
        # Load the DAG to get step names
        dag = load_shared_dag_for_pipeline(pipeline_name)
        
        # Create basic template structure
        template = {
            "PIPELINE_STEP_CONFIGS": {}
        }
        
        # Add each step from the DAG
        for step_name in dag.nodes:
            template["PIPELINE_STEP_CONFIGS"][step_name] = {
                "STEP_TYPE": ["PROCESSING_STEP"]  # Default step type
            }
        
        logger.info(f"Created execution document template for {pipeline_name} with {len(dag.nodes)} steps")
        return template
        
    except Exception as e:
        logger.warning(f"Failed to create specific template for {pipeline_name}, using generic template: {e}")
        
        # Fallback to generic template
        return {
            "PIPELINE_STEP_CONFIGS": {
                "generic_step": {
                    "STEP_TYPE": ["PROCESSING_STEP"]
                }
            }
        }


def get_pipeline_metadata(pipeline_name: str) -> Dict[str, Any]:
    """
    Get metadata for a pipeline using catalog registry.
    
    Args:
        pipeline_name: Name of the pipeline
        
    Returns:
        Dict[str, Any]: Pipeline metadata
    """
    try:
        registry = _get_catalog_registry()
        pipeline_node = registry.get_pipeline_node(pipeline_name)
        
        if pipeline_node is None:
            return {
                "pipeline_name": pipeline_name,
                "error": "Pipeline not found in catalog registry"
            }
        
        # Get DAG and config path
        dag = load_shared_dag_for_pipeline(pipeline_name)
        config_path = get_config_path_for_pipeline(pipeline_name)
        
        # Combine registry metadata with runtime metadata
        metadata = {
            "pipeline_name": pipeline_name,
            "title": pipeline_node.get("title", pipeline_name),
            "description": pipeline_node.get("description", ""),
            "framework": pipeline_node.get("zettelkasten_metadata", {}).get("framework", "unknown"),
            "complexity": pipeline_node.get("zettelkasten_metadata", {}).get("complexity", "unknown"),
            "features": pipeline_node.get("zettelkasten_metadata", {}).get("features", []),
            "source_file": pipeline_node.get("source_file", ""),
            "config_path": config_path,
            "node_count": len(dag.nodes),
            "edge_count": len(dag.edges),
            "nodes": list(dag.nodes),
            "edges": list(dag.edges),
            "mods_compatible": pipeline_node.get("zettelkasten_metadata", {}).get("mods_compatible", False),
        }
        
        return metadata
        
    except Exception as e:
        logger.error(f"Failed to get metadata for {pipeline_name}: {e}")
        return {
            "pipeline_name": pipeline_name,
            "error": str(e)
        }


def list_available_pipelines() -> Dict[str, Dict[str, Any]]:
    """
    List all available pipelines with their metadata using catalog registry.
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping pipeline names to their metadata
    """
    try:
        registry = _get_catalog_registry()
        available_pipeline_names = registry.get_all_pipelines()
        
        pipelines = {}
        for pipeline_name in available_pipeline_names:
            try:
                pipelines[pipeline_name] = get_pipeline_metadata(pipeline_name)
            except Exception as e:
                logger.warning(f"Failed to get metadata for {pipeline_name}: {e}")
                pipelines[pipeline_name] = {
                    "pipeline_name": pipeline_name,
                    "error": str(e)
                }
        
        logger.info(f"Found {len(pipelines)} available pipelines")
        return pipelines
        
    except Exception as e:
        logger.error(f"Failed to list available pipelines: {e}")
        return {}


def validate_pipeline_setup(pipeline_name: str) -> Dict[str, Any]:
    """
    Validate that a pipeline is properly set up using catalog registry.
    
    Args:
        pipeline_name: Name of the pipeline to validate
        
    Returns:
        Dict[str, Any]: Validation results
    """
    validation_result = {
        "pipeline_name": pipeline_name,
        "is_valid": True,
        "errors": [],
        "warnings": []
    }
    
    try:
        registry = _get_catalog_registry()
        
        # Check if pipeline name is recognized in registry
        pipeline_node = registry.get_pipeline_node(pipeline_name)
        if pipeline_node is None:
            available_pipelines = registry.get_all_pipelines()
            validation_result["is_valid"] = False
            validation_result["errors"].append(
                f"Unknown pipeline name: {pipeline_name}. Available: {available_pipelines}"
            )
            return validation_result
        
        # Check if DAG can be loaded
        try:
            dag = load_shared_dag_for_pipeline(pipeline_name)
            validation_result["dag_nodes"] = len(dag.nodes)
            validation_result["dag_edges"] = len(dag.edges)
        except Exception as e:
            validation_result["is_valid"] = False
            validation_result["errors"].append(f"Failed to load DAG: {e}")
        
        # Check if config path exists
        try:
            config_path = get_config_path_for_pipeline(pipeline_name)
            if not Path(config_path).exists():
                validation_result["warnings"].append(f"Config file does not exist: {config_path}")
            validation_result["config_path"] = config_path
        except Exception as e:
            validation_result["warnings"].append(f"Failed to get config path: {e}")
        
        # Check if execution document template can be created
        try:
            template = create_execution_doc_template_for_pipeline(pipeline_name)
            validation_result["template_steps"] = len(template.get("PIPELINE_STEP_CONFIGS", {}))
        except Exception as e:
            validation_result["warnings"].append(f"Failed to create execution document template: {e}")
        
        # Add registry metadata to validation result
        validation_result["registry_metadata"] = {
            "framework": pipeline_node.get("zettelkasten_metadata", {}).get("framework"),
            "complexity": pipeline_node.get("zettelkasten_metadata", {}).get("complexity"),
            "mods_compatible": pipeline_node.get("zettelkasten_metadata", {}).get("mods_compatible"),
        }
        
    except Exception as e:
        validation_result["is_valid"] = False
        validation_result["errors"].append(f"Validation failed: {e}")
    
    return validation_result

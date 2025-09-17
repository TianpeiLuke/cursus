"""
Utility functions for execution document generation.

This module provides common utility functions used across the execution
document generation system.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def determine_step_type(step_name: str, config: Any) -> List[str]:
    """
    Determine the step type for execution document based on step name and config.
    
    Uses the existing registry system to determine step types accurately.
    
    Args:
        step_name: Name of the step
        config: Configuration object for the step
        
    Returns:
        List of step types for the execution document
    """
    try:
        # Import the existing registry system
        from ...registry.step_names import (
            get_config_step_registry,
            get_sagemaker_step_type,
            CONFIG_STEP_REGISTRY,
        )
        
        # Get config class name
        config_class_name = type(config).__name__
        
        # Try to find the canonical step name using the registry
        canonical_step_name = None
        
        # Method 1: Direct lookup by config class name
        config_registry = get_config_step_registry()
        if config_class_name in config_registry:
            canonical_step_name = config_registry[config_class_name]
        
        # Method 2: Fallback to legacy CONFIG_STEP_REGISTRY
        elif config_class_name in CONFIG_STEP_REGISTRY:
            canonical_step_name = CONFIG_STEP_REGISTRY[config_class_name]
        
        # Method 3: Try to resolve from step name using existing utilities
        if not canonical_step_name:
            try:
                from ...registry.step_names import get_canonical_name_from_file_name
                canonical_step_name = get_canonical_name_from_file_name(step_name)
            except ValueError:
                # If all registry methods fail, use fallback logic
                pass
        
        # If we found the canonical step name, get the SageMaker step type
        if canonical_step_name:
            try:
                sagemaker_step_type = get_sagemaker_step_type(canonical_step_name)
                
                # Map SageMaker step types to execution document step types
                step_types = ["PROCESSING_STEP"]  # Default base type
                
                if sagemaker_step_type == "CradleDataLoading":
                    step_types.append("CradleDataLoading")
                elif sagemaker_step_type == "MimsModelRegistrationProcessing":
                    step_types.append("ModelRegistration")
                elif sagemaker_step_type == "Training":
                    step_types.append("Training")
                elif sagemaker_step_type == "Processing":
                    # For processing steps, try to be more specific
                    if "eval" in canonical_step_name.lower():
                        step_types.append("Evaluation")
                    elif "preprocess" in canonical_step_name.lower():
                        step_types.append("Preprocessing")
                    else:
                        step_types.append("Processing")
                elif sagemaker_step_type == "CreateModel":
                    step_types.append("CreateModel")
                elif sagemaker_step_type == "Transform":
                    step_types.append("Transform")
                elif sagemaker_step_type == "Lambda":
                    step_types.append("Lambda")
                else:
                    # For other types, use the SageMaker type directly
                    step_types.append(sagemaker_step_type)
                
                logger.debug(f"Determined step types for {step_name} ({config_class_name}): {step_types}")
                return step_types
                
            except Exception as e:
                logger.warning(f"Failed to get SageMaker step type for {canonical_step_name}: {e}")
        
    except Exception as e:
        logger.warning(f"Failed to use registry system for step type determination: {e}")
    
    # Fallback logic if registry system fails
    logger.debug(f"Using fallback logic for step type determination: {step_name}")
    return _determine_step_type_fallback(step_name, config)


def _determine_step_type_fallback(step_name: str, config: Any) -> List[str]:
    """
    Fallback step type determination logic when registry system is unavailable.
    
    Args:
        step_name: Name of the step
        config: Configuration object for the step
        
    Returns:
        List of step types for the execution document
    """
    # Default step type
    step_types = ["PROCESSING_STEP"]
    
    # Determine specific step type based on config type
    config_type_name = type(config).__name__.lower()
    
    if "cradle" in config_type_name or "cradle" in step_name.lower():
        step_types.append("CradleDataLoading")
    elif "registration" in config_type_name or "registration" in step_name.lower():
        step_types.append("ModelRegistration")
    elif "training" in config_type_name or "training" in step_name.lower():
        step_types.append("Training")
    elif "evaluation" in config_type_name or "evaluation" in step_name.lower():
        step_types.append("Evaluation")
    elif "processing" in config_type_name or "processing" in step_name.lower():
        step_types.append("Processing")
    
    return step_types


def validate_execution_document_structure(execution_document: Dict[str, Any]) -> bool:
    """
    Validate that the execution document has the expected structure.
    
    Args:
        execution_document: Execution document to validate
        
    Returns:
        True if structure is valid, False otherwise
    """
    if not isinstance(execution_document, dict):
        logger.error("Execution document must be a dictionary")
        return False
    
    if "PIPELINE_STEP_CONFIGS" not in execution_document:
        logger.error("Execution document missing 'PIPELINE_STEP_CONFIGS' key")
        return False
    
    pipeline_configs = execution_document["PIPELINE_STEP_CONFIGS"]
    if not isinstance(pipeline_configs, dict):
        logger.error("'PIPELINE_STEP_CONFIGS' must be a dictionary")
        return False
    
    return True


def create_execution_document_template(step_names: List[str]) -> Dict[str, Any]:
    """
    Create a basic execution document template with the given step names.
    
    Args:
        step_names: List of step names to include in the template
        
    Returns:
        Basic execution document template
    """
    template = {
        "PIPELINE_STEP_CONFIGS": {}
    }
    
    for step_name in step_names:
        template["PIPELINE_STEP_CONFIGS"][step_name] = {
            "STEP_TYPE": ["PROCESSING_STEP"],
            "STEP_CONFIG": {}
        }
    
    return template


def merge_execution_documents(base_doc: Dict[str, Any], 
                            additional_doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two execution documents, with additional_doc taking precedence.
    
    Args:
        base_doc: Base execution document
        additional_doc: Additional execution document to merge
        
    Returns:
        Merged execution document
    """
    if not validate_execution_document_structure(base_doc):
        raise ValueError("Invalid base execution document structure")
    
    if not validate_execution_document_structure(additional_doc):
        raise ValueError("Invalid additional execution document structure")
    
    # Create a deep copy of the base document
    import copy
    merged_doc = copy.deepcopy(base_doc)
    
    # Merge pipeline step configs
    base_configs = merged_doc["PIPELINE_STEP_CONFIGS"]
    additional_configs = additional_doc["PIPELINE_STEP_CONFIGS"]
    
    for step_name, step_config in additional_configs.items():
        if step_name in base_configs:
            # Merge step configurations - need to handle nested STEP_CONFIG properly
            for key, value in step_config.items():
                if key == "STEP_CONFIG" and key in base_configs[step_name]:
                    # Merge STEP_CONFIG dictionaries
                    base_configs[step_name][key].update(value)
                else:
                    # Update other keys directly
                    base_configs[step_name][key] = value
        else:
            # Add new step configuration
            base_configs[step_name] = step_config
    
    return merged_doc

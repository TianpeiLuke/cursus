"""
Simple Pipeline Execution Document Generation

This module provides simple functions for generating execution documents
for pipelines in the pipeline catalog using the standalone execution document generator.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

from ...mods.exe_doc.generator import ExecutionDocumentGenerator
from .utils import (
    get_config_path_for_pipeline,
    load_shared_dag_for_pipeline,
    create_execution_doc_template_for_pipeline,
)

logger = logging.getLogger(__name__)


def generate_execution_document_for_pipeline(
    pipeline_name: str,
    config_path: Optional[str] = None,
    execution_doc_template: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate execution document for a specific pipeline.
    
    This is the main entry point for pipeline execution document generation.
    It provides a simple interface that:
    1. Loads the appropriate configuration for the pipeline
    2. Loads the shared DAG for the pipeline
    3. Creates or uses provided execution document template
    4. Uses the standalone execution document generator to fill the document
    
    Args:
        pipeline_name: Name of the pipeline (e.g., "xgb_e2e_comprehensive")
        config_path: Optional path to configuration file (overrides default)
        execution_doc_template: Optional execution document template (creates default if not provided)
        **kwargs: Additional arguments passed to ExecutionDocumentGenerator
        
    Returns:
        Dict[str, Any]: Filled execution document ready for pipeline execution
        
    Raises:
        ValueError: If pipeline name is not recognized or configuration not found
        FileNotFoundError: If configuration file or DAG not found
        
    Example:
        >>> execution_doc = generate_execution_document_for_pipeline(
        ...     pipeline_name="xgb_e2e_comprehensive",
        ...     config_path="/path/to/config.json"
        ... )
        >>> print(execution_doc["PIPELINE_STEP_CONFIGS"].keys())
    """
    try:
        logger.info(f"Generating execution document for pipeline: {pipeline_name}")
        
        # Get configuration path for pipeline
        if config_path is None:
            config_path = get_config_path_for_pipeline(pipeline_name)
            logger.info(f"Using default config path: {config_path}")
        
        # Validate configuration file exists
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load shared DAG for pipeline
        dag = load_shared_dag_for_pipeline(pipeline_name)
        logger.info(f"Loaded DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges")
        
        # Create execution document template if not provided
        if execution_doc_template is None:
            execution_doc_template = create_execution_doc_template_for_pipeline(pipeline_name)
            logger.info("Created default execution document template")
        
        # Create standalone execution document generator
        generator = ExecutionDocumentGenerator(config_path=config_path, **kwargs)
        logger.info("Created execution document generator")
        
        # Generate execution document
        filled_execution_doc = generator.fill_execution_document(dag, execution_doc_template)
        logger.info("Successfully generated execution document")
        
        return filled_execution_doc
        
    except Exception as e:
        logger.error(f"Failed to generate execution document for pipeline {pipeline_name}: {e}")
        raise


def generate_execution_document_for_pipeline_with_base_pipeline(
    base_pipeline_instance,
    execution_doc_template: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate execution document using a BasePipeline instance.
    
    This function provides integration with existing BasePipeline instances,
    using the new standalone execution document generator instead of the
    old template-based approach.
    
    Args:
        base_pipeline_instance: Instance of BasePipeline or its subclass
        execution_doc_template: Optional execution document template
        
    Returns:
        Dict[str, Any]: Filled execution document
        
    Example:
        >>> pipeline_instance = XGBoostE2EComprehensivePipeline(config_path="config.json")
        >>> execution_doc = generate_execution_document_for_pipeline_with_base_pipeline(
        ...     pipeline_instance,
        ...     execution_doc_template
        ... )
    """
    try:
        logger.info("Generating execution document using BasePipeline instance")
        
        # Get configuration path from pipeline instance
        config_path = base_pipeline_instance.config_path
        if not config_path:
            raise ValueError("BasePipeline instance must have config_path set")
        
        # Get DAG from pipeline instance
        dag = base_pipeline_instance.dag
        
        # Create execution document template if not provided
        if execution_doc_template is None:
            # Try to get pipeline name from metadata
            try:
                metadata = base_pipeline_instance.get_enhanced_dag_metadata()
                pipeline_name = metadata.zettelkasten_metadata.atomic_id
                execution_doc_template = create_execution_doc_template_for_pipeline(pipeline_name)
            except Exception:
                # Fallback to generic template
                execution_doc_template = create_execution_doc_template_for_pipeline("generic")
            logger.info("Created execution document template from pipeline metadata")
        
        # Create standalone execution document generator
        generator = ExecutionDocumentGenerator(
            config_path=config_path,
            sagemaker_session=base_pipeline_instance.sagemaker_session,
            role=base_pipeline_instance.execution_role,
        )
        logger.info("Created execution document generator from BasePipeline instance")
        
        # Generate execution document
        filled_execution_doc = generator.fill_execution_document(dag, execution_doc_template)
        logger.info("Successfully generated execution document using BasePipeline instance")
        
        return filled_execution_doc
        
    except Exception as e:
        logger.error(f"Failed to generate execution document using BasePipeline instance: {e}")
        raise


def update_base_pipeline_fill_execution_document():
    """
    Update BasePipeline.fill_execution_document to use the new standalone generator.
    
    This function can be used to monkey-patch the existing BasePipeline class
    to use the new standalone execution document generator instead of the old
    template-based approach.
    
    This provides a migration path for existing code that uses BasePipeline.fill_execution_document().
    """
    try:
        from ..core.base_pipeline import BasePipeline
        
        def new_fill_execution_document(self, execution_doc: Dict[str, Any]) -> Dict[str, Any]:
            """
            Fill an execution document using the new standalone generator.
            
            This method replaces the old template-based approach with the new
            standalone execution document generator.
            """
            return generate_execution_document_for_pipeline_with_base_pipeline(
                self, execution_doc
            )
        
        # Replace the method
        BasePipeline.fill_execution_document = new_fill_execution_document
        logger.info("Successfully updated BasePipeline.fill_execution_document to use standalone generator")
        
    except Exception as e:
        logger.error(f"Failed to update BasePipeline.fill_execution_document: {e}")
        raise

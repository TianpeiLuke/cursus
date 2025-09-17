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


# Note: BasePipeline integration functions removed to achieve complete independence
# between pipeline generation and execution document generation modules.
#
# The two modules are now completely independent:
# 1. Pipeline generation: cursus.pipeline_catalog.core.base_pipeline
# 2. Execution document generation: cursus.mods.exe_doc.generator
#
# Users should use the modules independently:
#
# For pipeline generation:
# pipeline_instance = XGBoostE2EComprehensivePipeline(config_path="config.json")
# pipeline = pipeline_instance.generate_pipeline()
#
# For execution document generation:
# from cursus.mods.exe_doc.generator import ExecutionDocumentGenerator
# generator = ExecutionDocumentGenerator(config_path="config.json")
# filled_doc = generator.fill_execution_document(dag, execution_doc_template)
#
# Or use the pipeline catalog integration:
# from cursus.pipeline_catalog.pipeline_exe import generate_execution_document_for_pipeline
# filled_doc = generate_execution_document_for_pipeline("xgb_e2e_comprehensive", "config.json", execution_doc)

"""
MODS PyTorch Standard End-to-End Pipeline

This pipeline implements a MODS-enhanced version of the standard PyTorch end-to-end workflow:
1) Data Loading (training)
2) Preprocessing (training)
3) PyTorch Model Training
4) Data Loading (evaluation)
5) Preprocessing (evaluation)
6) Model Evaluation
7) Model Registration
8) Model Deployment

This MODS variant provides enhanced functionality including:
- Automatic template registration in MODS global registry
- Enhanced metadata extraction and validation
- Integration with MODS operational tools
- Advanced pipeline tracking and monitoring

The pipeline uses the same shared DAG definition as the standard version,
ensuring consistency while providing MODS-specific features.

Example:
    ```python
    from cursus.pipeline_catalog.mods_frameworks.pytorch.end_to_end.standard_e2e_mods import create_pipeline
    from sagemaker import Session
    from sagemaker.workflow.pipeline_context import PipelineSession
    
    # Initialize session
    sagemaker_session = Session()
    role = sagemaker_session.get_caller_identity_arn()
    pipeline_session = PipelineSession()
    
    # Create MODS pipeline (automatically registers with MODS global registry)
    pipeline, report, dag_compiler, mods_template = create_pipeline(
        config_path="path/to/config.json",
        session=pipeline_session,
        role=role
    )
    
    # Execute the pipeline
    pipeline.upsert()
    execution = pipeline.start()
    ```
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession

from .....api.dag.base_dag import PipelineDAG
from ....shared_dags.pytorch.standard_e2e_dag import create_pytorch_standard_e2e_dag
from ... import check_mods_requirements, get_mods_compiler_class, MODSNotAvailableError

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dag() -> PipelineDAG:
    """
    Create a standard PyTorch end-to-end pipeline DAG.
    
    This function uses the shared DAG definition to ensure consistency
    between regular and MODS pipeline variants.
    
    Returns:
        PipelineDAG: The directed acyclic graph for the pipeline
    """
    dag = create_pytorch_standard_e2e_dag()
    logger.info(f"Created MODS PyTorch standard end-to-end DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges")
    return dag


def create_pipeline(
    config_path: str,
    session: PipelineSession,
    role: str,
    pipeline_name: Optional[str] = None,
    pipeline_description: Optional[str] = None
) -> Tuple[Pipeline, Dict[str, Any], Any, Any]:
    """
    Create a SageMaker Pipeline from the DAG for standard PyTorch end-to-end workflow with MODS features.
    
    Args:
        config_path: Path to the configuration file
        session: SageMaker pipeline session
        role: IAM role for pipeline execution
        pipeline_name: Custom name for the pipeline (optional)
        pipeline_description: Description for the pipeline (optional)
        
    Returns:
        Tuple containing:
            - Pipeline: The created SageMaker pipeline
            - Dict: Conversion report with details about the compilation
            - MODSPipelineDAGCompiler: The MODS compiler instance for further operations
            - Any: The MODS decorated template instance for global registry
            
    Raises:
        MODSNotAvailableError: If MODS is not available in the environment
    """
    # Check MODS availability
    check_mods_requirements()
    
    # Get MODS compiler class
    MODSPipelineDAGCompiler = get_mods_compiler_class()
    if MODSPipelineDAGCompiler is None:
        raise MODSNotAvailableError("MODSPipelineDAGCompiler is not available")
    
    dag = create_dag()
    
    # Create MODS compiler with the configuration
    dag_compiler = MODSPipelineDAGCompiler(
        config_path=config_path,
        sagemaker_session=session,
        role=role
    )
    
    # Set optional pipeline properties
    if pipeline_name:
        dag_compiler.pipeline_name = pipeline_name
    else:
        dag_compiler.pipeline_name = "MODS-PyTorch-Standard-E2E-Pipeline"
        
    if pipeline_description:
        dag_compiler.pipeline_description = pipeline_description
    else:
        dag_compiler.pipeline_description = "MODS-enhanced standard PyTorch end-to-end pipeline with training, evaluation, registration, and deployment"
    
    # Preview resolution (optional)
    try:
        preview = dag_compiler.preview_resolution(dag)
        logger.info("DAG node resolution preview:")
        for node, config_type in preview.node_config_map.items():
            confidence = preview.resolution_confidence.get(node, 0.0)
            logger.info(f"  {node} → {config_type} (confidence: {confidence:.2f})")
    except Exception as e:
        logger.warning(f"Preview resolution failed: {e}")
    
    # Compile the DAG into a pipeline with MODS features
    pipeline, report = dag_compiler.compile_with_report(dag=dag)
    
    # Get the MODS decorated template instance for global registry
    mods_template = dag_compiler.get_last_template()
    if mods_template is None:
        logger.warning("MODS template instance not found after compilation")
    else:
        logger.info("MODS decorated template instance retrieved for global registry")
    
    logger.info(f"MODS Pipeline '{pipeline.name}' created successfully")
    logger.info("MODS features enabled: template registration, enhanced metadata, operational integration")
    
    return pipeline, report, dag_compiler, mods_template


def fill_execution_document(
    pipeline: Pipeline,
    document: Dict[str, Any],
    dag_compiler: Any
) -> Dict[str, Any]:
    """
    Fill an execution document for the pipeline with all necessary parameters.
    
    Args:
        pipeline: The compiled SageMaker pipeline
        document: Initial parameter document with user-provided values
        dag_compiler: The MODS DAG compiler used to create the pipeline
    
    Returns:
        Dict: Complete execution document ready for pipeline execution
    """
    # Create execution document with all required parameters
    execution_doc = dag_compiler.create_execution_document(document)
    return execution_doc


if __name__ == "__main__":
    # Example usage
    import os
    from sagemaker import Session
    
    try:
        sagemaker_session = Session()
        role = sagemaker_session.get_caller_identity_arn()
        pipeline_session = PipelineSession()
        
        # Assuming config file is in a standard location
        config_dir = Path.cwd().parent.parent / "pipeline_config"
        config_path = os.path.join(config_dir, "config.json")
        
        pipeline, report, dag_compiler, mods_template = create_pipeline(
            config_path=config_path,
            session=pipeline_session,
            role=role,
            pipeline_name="MODS-PyTorch-Standard-E2E-Pipeline",
            pipeline_description="MODS-enhanced standard PyTorch end-to-end pipeline with training, evaluation, registration, and deployment"
        )
        
        logger.info("MODS pipeline created successfully!")
        logger.info(f"Pipeline name: {pipeline.name}")
        logger.info(f"MODS template available: {mods_template is not None}")
        logger.info(f"MODS features: Template registration, enhanced metadata, operational integration")
        
        # You can now upsert and execute the pipeline
        # pipeline.upsert()
        # execution_doc = fill_execution_document(
        #     pipeline=pipeline, 
        #     document={
        #         "training_dataset": "my-dataset", 
        #         "evaluation_dataset": "my-eval-dataset",
        #         "model_package_group_name": "my-model-group"
        #     }, 
        #     dag_compiler=dag_compiler
        # )
        # execution = pipeline.start(execution_input=execution_doc)
        
    except MODSNotAvailableError as e:
        logger.error(f"MODS not available: {e}")
        logger.info("Please install MODS or use the standard PyTorch standard end-to-end pipeline instead")
    except Exception as e:
        logger.error(f"Failed to create MODS pipeline: {e}")

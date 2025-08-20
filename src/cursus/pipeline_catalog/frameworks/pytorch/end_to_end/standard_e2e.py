"""
PyTorch End-to-End Pipeline

This pipeline implements a complete workflow for training, evaluating, packaging, 
and registering a PyTorch model:
1) Data Loading (training)
2) Preprocessing (training)
3) PyTorch Model Training
4) Package Model
5) Payload Generation
6) Model Registration
7) Data Loading (validation)
8) Preprocessing (validation) 
9) Model Evaluation

This comprehensive pipeline covers the entire ML lifecycle from data loading to
model registration and evaluation. Use this when you need a production-ready PyTorch
pipeline that handles model training through deployment.

Example:
    ```python
    from cursus.pipeline_catalog.frameworks.pytorch.end_to_end.standard_e2e import create_pipeline
    from sagemaker import Session
    from sagemaker.workflow.pipeline_context import PipelineSession
    
    # Initialize session
    sagemaker_session = Session()
    role = sagemaker_session.get_caller_identity_arn()
    pipeline_session = PipelineSession()
    
    # Create the pipeline
    pipeline, report, dag_compiler = create_pipeline(
        config_path="path/to/config_pytorch.json",
        session=pipeline_session,
        role=role
    )
    
    # Execute the pipeline
    pipeline.upsert()
    execution = pipeline.start()
    ```
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession

from src.cursus.api.dag.base_dag import PipelineDAG
from src.cursus.core.compiler.dag_compiler import PipelineDAGCompiler

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dag() -> PipelineDAG:
    """
    Create a complete end-to-end PyTorch pipeline DAG.
    
    This DAG represents a comprehensive workflow for training,
    evaluating, packaging, and registering a PyTorch model.
    
    Returns:
        PipelineDAG: The directed acyclic graph for the pipeline
    """
    dag = PipelineDAG()
    
    # Add all nodes - named to match configuration names exactly
    dag.add_node("CradleDataLoading_training")       # Data load for training
    dag.add_node("TabularPreprocessing_training")    # Tabular preprocessing for training
    dag.add_node("PyTorchTraining")                  # PyTorch training step
    dag.add_node("Package")                          # Package step
    dag.add_node("Payload")                          # Payload step
    dag.add_node("Registration")                     # Model registration step
    dag.add_node("CradleDataLoading_validation")     # Data load for validation
    dag.add_node("TabularPreprocessing_validation")  # Tabular preprocessing for validation
    dag.add_node("PyTorchModelEval")                 # Model evaluation step
    
    # Training flow
    dag.add_edge("CradleDataLoading_training", "TabularPreprocessing_training")
    dag.add_edge("TabularPreprocessing_training", "PyTorchTraining")
    
    # Output flow
    dag.add_edge("PyTorchTraining", "Package")      # Model is packaged
    dag.add_edge("PyTorchTraining", "Payload")      # Model is used for payload generation
    dag.add_edge("Package", "Registration")         # Packaged model is registered
    dag.add_edge("Payload", "Registration")         # Payload is needed for registration
    
    # Evaluation flow
    dag.add_edge("CradleDataLoading_validation", "TabularPreprocessing_validation")
    dag.add_edge("TabularPreprocessing_validation", "PyTorchModelEval")
    dag.add_edge("PyTorchTraining", "PyTorchModelEval")  # Model is input to evaluation
    dag.add_edge("PyTorchModelEval", "Registration")     # Evaluation results are needed for registration
    
    logger.info(f"Created DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges")
    return dag


def create_pipeline(
    config_path: str,
    session: PipelineSession,
    role: str,
    pipeline_name: Optional[str] = None,
    pipeline_description: Optional[str] = None,
    validate: bool = True
) -> Tuple[Pipeline, Dict[str, Any], PipelineDAGCompiler]:
    """
    Create a SageMaker Pipeline from the DAG for a complete PyTorch workflow.
    
    Args:
        config_path: Path to the configuration file
        session: SageMaker pipeline session
        role: IAM role for pipeline execution
        pipeline_name: Custom name for the pipeline (optional)
        pipeline_description: Description for the pipeline (optional)
        validate: Whether to validate the DAG before compilation
        
    Returns:
        Tuple containing:
            - Pipeline: The created SageMaker pipeline
            - Dict: Conversion report with details about the compilation
            - PipelineDAGCompiler: The compiler instance for further operations
    """
    dag = create_dag()
    
    # Create compiler with the configuration
    dag_compiler = PipelineDAGCompiler(
        config_path=config_path,
        sagemaker_session=session,
        role=role
    )
    
    # Set optional pipeline properties
    if pipeline_name:
        dag_compiler.pipeline_name = pipeline_name
    if pipeline_description:
        dag_compiler.pipeline_description = pipeline_description
    
    # Validate the DAG if requested
    if validate:
        validation = dag_compiler.validate_dag_compatibility(dag)
        if not validation.is_valid:
            logger.warning(f"DAG validation failed: {validation.summary()}")
            if validation.missing_configs:
                logger.warning(f"Missing configs: {validation.missing_configs}")
            if validation.unresolvable_builders:
                logger.warning(f"Unresolvable builders: {validation.unresolvable_builders}")
            if validation.config_errors:
                logger.warning(f"Config errors: {validation.config_errors}")
            if validation.dependency_issues:
                logger.warning(f"Dependency issues: {validation.dependency_issues}")
    
    # Compile the DAG into a pipeline
    pipeline, report = dag_compiler.compile_with_report(dag=dag)
    
    logger.info(f"Pipeline '{pipeline.name}' created successfully")
    logger.info(f"Average resolution confidence: {report.avg_confidence:.2f}")
    
    return pipeline, report, dag_compiler


def fill_execution_document(
    pipeline: Pipeline,
    document: Dict[str, Any],
    dag_compiler: PipelineDAGCompiler
) -> Dict[str, Any]:
    """
    Fill an execution document for the pipeline with all necessary parameters.
    
    Args:
        pipeline: The compiled SageMaker pipeline
        document: Initial parameter document with user-provided values
        dag_compiler: The DAG compiler used to create the pipeline
    
    Returns:
        Dict: Complete execution document ready for pipeline execution
    """
    # Create execution document with all required parameters
    execution_doc = dag_compiler.create_execution_document(document)
    return execution_doc


def save_execution_document(document: Dict[str, Any], output_path: str) -> None:
    """
    Save the execution document to a file.
    
    Args:
        document: The execution document to save
        output_path: Path where to save the document
    """
    import json
    
    # Ensure directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the document
    with open(output_path, "w") as f:
        json.dump(document, f, indent=2)
    
    logger.info(f"Execution document saved to: {output_path}")


def get_security_config():
    """
    Create security configuration for pipeline execution.
    
    Returns:
        Optional[SecurityConfig]: Security configuration if available
    """
    try:
        from secure_ai_sandbox_python_lib.session import Session as SaisSession
        from mods_workflow_helper.utils.secure_session import create_secure_session_config
        from mods_workflow_helper.sagemaker_pipeline_helper import SecurityConfig
        
        # Get security config
        sais_session = SaisSession(".")
        security_config = SecurityConfig(
            kms_key=sais_session.get_team_owned_bucket_kms_key(),
            security_group=sais_session.sandbox_vpc_security_group(),
            vpc_subnets=sais_session.sandbox_vpc_subnets()
        )
        return security_config
    except ImportError:
        logger.warning("Secure AI sandbox libraries not available")
        return None
    except Exception as e:
        logger.warning(f"Failed to create security config: {e}")
        return None


def execute_pipeline(
    pipeline: Pipeline, 
    execution_doc: Dict[str, Any],
    session,
    secure_config = None
) -> str:
    """
    Execute the pipeline with the given execution document.
    
    Args:
        pipeline: The SageMaker pipeline to execute
        execution_doc: The filled execution document
        session: SageMaker session
        secure_config: Security configuration (optional)
        
    Returns:
        str: ARN of the pipeline execution
    """
    try:
        from mods_workflow_helper.sagemaker_pipeline_helper import SagemakerPipelineHelper
        
        # Start pipeline execution
        if secure_config:
            execution_arn = SagemakerPipelineHelper.start_pipeline_execution(
                pipeline=pipeline,
                secure_config=secure_config,
                sagemaker_session=session,
                preparation_space_local_root="/tmp",
                pipeline_execution_document=execution_doc
            )
        else:
            pipeline.upsert()
            execution = pipeline.start(execution_input=execution_doc)
            execution_arn = execution.arn
            
        logger.info(f"Started pipeline execution: {execution_arn}")
        return execution_arn
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to execute pipeline: {e}")
        raise


if __name__ == "__main__":
    # Example usage
    import argparse
    from sagemaker import Session
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create a complete PyTorch end-to-end pipeline')
    parser.add_argument('--config-path', type=str, help='Path to the configuration file')
    parser.add_argument('--output-doc', type=str, help='Path to save the execution document')
    parser.add_argument('--upsert', action='store_true', help='Upsert the pipeline after creation')
    parser.add_argument('--execute', action='store_true', help='Execute the pipeline after creation')
    args = parser.parse_args()
    
    # Initialize session
    sagemaker_session = Session()
    role = sagemaker_session.get_caller_identity_arn()
    pipeline_session = PipelineSession()
    
    # Use provided config path or fallback to default
    config_path = args.config_path
    if not config_path:
        config_dir = Path.cwd().parent / "pipeline_config"
        config_path = os.path.join(config_dir, "config_pytorch.json")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Default config file not found: {config_path}")
    
    # Create the pipeline
    logger.info(f"Creating pipeline with config: {config_path}")
    pipeline, report, dag_compiler = create_pipeline(
        config_path=config_path,
        session=pipeline_session,
        role=role,
        pipeline_name="PyTorch-End-To-End",
        pipeline_description="Complete PyTorch pipeline with training, evaluation, and registration"
    )
    
    # Process execution documents and pipeline operations if requested
    execution_doc = None
    if args.output_doc or args.execute:
        execution_doc = fill_execution_document(
            pipeline=pipeline,
            document={
                "training_dataset": "dataset-training",
                "validation_dataset": "dataset-validation"
            },
            dag_compiler=dag_compiler
        )
        
        # Save the execution document if requested
        if args.output_doc:
            save_execution_document(
                document=execution_doc,
                output_path=args.output_doc
            )
    
    # Upsert if requested
    if args.upsert and not args.execute:
        pipeline.upsert()
        logger.info(f"Pipeline '{pipeline.name}' upserted successfully")
    
    # Execute if requested
    if args.execute and execution_doc:
        try:
            # Get security config if available
            secure_config = get_security_config()
            
            # Execute the pipeline
            execute_pipeline(
                pipeline=pipeline,
                execution_doc=execution_doc,
                session=sagemaker_session,
                secure_config=secure_config
            )
        except Exception as e:
            logger.error(f"Failed to execute pipeline: {e}")

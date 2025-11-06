import json
import logging
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, Union, List

# SageMaker imports
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString
from sagemaker.network import NetworkConfig
from sagemaker import Session

from mods_workflow_core.utils.constants import (
    PIPELINE_EXECUTION_TEMP_DIR,
    KMS_ENCRYPTION_KEY_PARAM,
    PROCESSING_JOB_SHARED_NETWORK_CONFIG,
    SECURITY_GROUP_ID,
    VPC_SUBNET,
)

pipeline_parameters = [
    PIPELINE_EXECUTION_TEMP_DIR,
    KMS_ENCRYPTION_KEY_PARAM,
    SECURITY_GROUP_ID,
    VPC_SUBNET,
]


# Cursus imports - adjusted for root repository location
from cursus.api.dag.base_dag import PipelineDAG
from cursus.core.compiler.dag_compiler import (
    PipelineDAGCompiler,
)

# MODS template import
from mods.mods_template import MODSTemplate

# Configure logging
import logging

logger = logging.getLogger(__name__)

# Default model and region settings
DEFAULT_MODEL_CLASS = "xgboost"
DEFAULT_REGION = "NA"
DEFAULT_SERVICE_NAME = "AtoZ"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_xgboost_complete_e2e_dag() -> PipelineDAG:
    """
    Create a DAG for a complete XGBoost end-to-end workflow.

    This DAG represents a complete end-to-end workflow including training,
    calibration, packaging, registration, and evaluation of an XGBoost model.

    Returns:
        PipelineDAG: The directed acyclic graph for the pipeline
    """
    dag = PipelineDAG()

    # Add all nodes
    dag.add_node("CradleDataLoading_training")  # Data load for training
    dag.add_node("TabularPreprocessing_training")  # Tabular preprocessing for training
    dag.add_node("XGBoostTraining")  # XGBoost training step
    dag.add_node(
        "ModelCalibration_calibration"
    )  # Model calibration step with calibration variant
    dag.add_node("Package")  # Package step
    dag.add_node("Registration")  # MIMS registration step
    dag.add_node("Payload")  # Payload step
    dag.add_node("CradleDataLoading_calibration")  # Data load for calibration
    dag.add_node(
        "TabularPreprocessing_calibration"
    )  # Tabular preprocessing for calibration
    dag.add_node("XGBoostModelEval_calibration")  # Model evaluation step

    # Training flow
    dag.add_edge("CradleDataLoading_training", "TabularPreprocessing_training")
    dag.add_edge("TabularPreprocessing_training", "XGBoostTraining")

    # Calibration flow
    dag.add_edge("CradleDataLoading_calibration", "TabularPreprocessing_calibration")

    # Evaluation flow
    dag.add_edge("XGBoostTraining", "XGBoostModelEval_calibration")
    dag.add_edge("TabularPreprocessing_calibration", "XGBoostModelEval_calibration")

    # Model calibration flow - depends on model evaluation
    dag.add_edge("XGBoostModelEval_calibration", "ModelCalibration_calibration")

    # Output flow
    dag.add_edge("ModelCalibration_calibration", "Package")
    dag.add_edge("XGBoostTraining", "Package")  # Raw model is also input to packaging
    dag.add_edge("XGBoostTraining", "Payload")  # Payload test uses the raw model
    dag.add_edge("Package", "Registration")
    dag.add_edge("Payload", "Registration")

    logger.info(
        f"Created XGBoost complete E2E DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges"
    )
    return dag


# Define constants
AUTHOR = "lukexie"
PIPELINE_VERSION = "1.3.14"
PIPELINE_DESCRIPTION = "XGBoost End-to-End Pipeline using Cursus DAG Compiler"


@MODSTemplate(author=AUTHOR, description=PIPELINE_DESCRIPTION, version=PIPELINE_VERSION)
class XGBoostAtoZPipeline:
    """
    Adapter class that bridges between Cursus DAG-based pipeline architecture and MODS Template structure.

    This class specifically creates an XGBoost end-to-end pipeline using the DAG compiler.
    """

    def __init__(
        self,
        sagemaker_session: Optional[Session] = None,
        execution_role: Optional[str] = None,
        regional_alias: str = DEFAULT_REGION,
        pipeline_parameters: Optional[List[Union[str, ParameterString]]] = None,
    ) -> None:
        """
        Initialize the adapter with configuration and session details.

        Args:
            sagemaker_session: SageMaker pipeline session
            execution_role: IAM role for pipeline execution
            regional_alias: Region code (NA, EU, FE, etc.) for configuration path
        """
        # Set defaults if not provided
        self.sagemaker_session = sagemaker_session or Session()
        self.execution_role = (
            execution_role or self.sagemaker_session.get_caller_identity_arn()
        )
        self.pipeline_parameters = pipeline_parameters

        # Use fixed values for the parameters that were previously configurable
        model_class = DEFAULT_MODEL_CLASS
        service_name = DEFAULT_SERVICE_NAME
        pipeline_name = None
        pipeline_description = None

        # Build fixed config path similar to how regional_xgboost_na does it
        module_dir = Path(__file__).resolve().parent
        config_dir = module_dir / "pipeline_config"
        pipeline_config_name = f"config.json"
        config_path = str(config_dir / pipeline_config_name)
        logger.info(f"Using config path: {config_path}")

        self.config_path = config_path
        self.pipeline_name = pipeline_name
        self.pipeline_description = pipeline_description

        # Create XGBoost DAG
        self.dag = create_xgboost_complete_e2e_dag()
        logger.info(
            f"Created XGBoost DAG with {len(self.dag.nodes)} nodes and {len(self.dag.edges)} edges"
        )

        # Initialize compiler (but don't compile yet)
        self.dag_compiler = PipelineDAGCompiler(
            config_path=self.config_path,
            sagemaker_session=self.sagemaker_session,
            role=self.execution_role,
            pipeline_parameters=self.pipeline_parameters,
        )
        logger.info("Initialized DAG compiler")

    def generate_pipeline(self) -> Pipeline:
        """
        Generate a SageMaker Pipeline using the DAG Compiler.

        This method fulfills the MODS Template interface requirement while using our
        DAG compilation process internally.

        Returns:
            Pipeline: Compiled SageMaker Pipeline
        """
        # Set optional pipeline properties if provided
        kwargs: Dict[str, str] = {}
        if self.pipeline_name:
            kwargs["pipeline_name"] = self.pipeline_name
        if self.pipeline_description:
            kwargs["pipeline_description"] = self.pipeline_description

        # Use the compiler to build the pipeline
        pipeline, report = self.dag_compiler.compile_with_report(dag=self.dag, **kwargs)

        # Log compilation details
        logger.info(f"Pipeline '{pipeline.name}' created successfully")
        logger.info(f"Average resolution confidence: {report.avg_confidence:.2f}")

        return pipeline

    def validate_dag_compatibility(self) -> Dict[str, Any]:
        """
        Validate that the DAG is compatible with the configuration.

        Returns:
            Dict: Validation results
        """
        validation = self.dag_compiler.validate_dag_compatibility(self.dag)
        return {
            "is_valid": validation.is_valid,
            "missing_configs": validation.missing_configs,
            "unresolvable_builders": validation.unresolvable_builders,
            "config_errors": validation.config_errors,
            "dependency_issues": validation.dependency_issues,
            "warnings": validation.warnings,
        }

    def preview_resolution(self) -> Dict[str, Any]:
        """
        Preview how DAG nodes will be resolved to configs and builders.

        Returns:
            Dict: Preview of node resolution
        """
        preview = self.dag_compiler.preview_resolution(self.dag)
        return {
            "node_config_map": preview.node_config_map,
            "config_builder_map": preview.config_builder_map,
            "resolution_confidence": preview.resolution_confidence,
            "ambiguous_resolutions": preview.ambiguous_resolutions,
            "recommendations": preview.recommendations,
        }

    def get_last_template(self) -> Any:
        """
        Get the last template used during compilation.

        Returns:
            Any: The template object
        """
        return self.dag_compiler.get_last_template()


# Example usage:
#
# from mods_pipeline_adapter import XGBoostCursusPipelineAdapter
# from sagemaker import Session
# from sagemaker.workflow.pipeline_context import PipelineSession
# from mods_workflow_helper.sagemaker_pipeline_helper import SagemakerPipelineHelper, SecurityConfig
# from secure_ai_sandbox_python_lib.session import Session as SaisSession
# from mods_workflow_helper.utils.secure_session import create_secure_session_config
#
# # Initialize SAIS session for security configuration
# sais_session = SaisSession(".")
#
# # Create security config for pipeline execution
# security_config = SecurityConfig(
#     kms_key=sais_session.get_team_owned_bucket_kms_key(),
#     security_group=sais_session.sandbox_vpc_security_group(),
#     vpc_subnets=sais_session.sandbox_vpc_subnets()
# )
#
# # Create secure SageMaker config
# sagemaker_config = create_secure_session_config(
#     role_arn=PipelineSession().get_caller_identity_arn(),
#     bucket_name=sais_session.team_owned_s3_bucket_name(),
#     kms_key=sais_session.get_team_owned_bucket_kms_key(),
#     vpc_subnet_ids=sais_session.sandbox_vpc_subnets(),
#     vpc_security_groups=[sais_session.sandbox_vpc_security_group()]
# )
#
# # Create pipeline session with security config
# pipeline_session = PipelineSession(
#     default_bucket=sais_session.team_owned_s3_bucket_name(),
#     sagemaker_config=sagemaker_config
# )
#
# # Get IAM role for execution
# role = pipeline_session.get_caller_identity_arn()
#
# # Create adapter instance with default configuration path
# adapter = XGBoostCursusPipelineAdapter(
#     sagemaker_session=pipeline_session,
#     execution_role=role,
#     pipeline_name="XGBoost-MODS-Pipeline",
#     pipeline_description="XGBoost pipeline using MODS adapter",
#     regional_alias="NA",
#     model_class="xgboost",
#     service_name="AtoZ"
# )
#
# # Generate pipeline
# pipeline = adapter.generate_pipeline()
#
# # Get default execution document and fill it with pipeline-specific parameters
# default_execution_doc = SagemakerPipelineHelper.get_pipeline_default_execution_document(pipeline)
# filled_execution_doc = adapter.fill_execution_document(default_execution_doc)
#
# # Save execution document for later use
# with open("execution_doc.json", "w") as f:
#     json.dump(filled_execution_doc, f, indent=2)
#
# # Execute pipeline with filled document and security config
# execution = SagemakerPipelineHelper.start_pipeline_execution(
#     pipeline=pipeline,
#     secure_config=security_config,
#     sagemaker_session=pipeline_session,
#     preparation_space_local_root="/tmp",
#     pipeline_execution_document=filled_execution_doc
# )

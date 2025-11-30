import json
import logging
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, Union, List

# Configure logging
import logging

logger = logging.getLogger(__name__)

from sagemaker import Session
from mods.mods_template import MODSTemplate
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

# SageMaker imports
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString
from sagemaker.network import NetworkConfig

# Cursus imports - adjusted for root repository location
from cursus.api.dag.base_dag import PipelineDAG
from cursus.core.compiler.dag_compiler import (
    PipelineDAGCompiler,
)

# Default model and region settings
DEFAULT_MODEL_CLASS = "pytorch"
DEFAULT_REGION = "NA"
DEFAULT_SERVICE_NAME = "BuyerAbuseRnR"

# Define constants
AUTHOR = "lukexie"
PIPELINE_VERSION = "0.0.1"
PIPELINE_DESCRIPTION = "Bedrock Processing Pipeline for Classification"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_bedrock_batch_pytorch_with_label_ruleset_e2e_dag() -> PipelineDAG:
    """
    Create a DAG for Bedrock Batch-enhanced PyTorch E2E pipeline with Label Ruleset steps.

    This DAG represents a complete end-to-end workflow that uses:
    1. Bedrock prompt template generation and batch processing for LLM-enhanced data
    2. Label ruleset generation and execution for transparent label transformation
    3. PyTorch training, followed by calibration, packaging, and registration

    The label ruleset steps sit between Bedrock processing and training/evaluation,
    providing transparent, rule-based label transformation that's easy to modify.

    Returns:
        PipelineDAG: The directed acyclic graph for the pipeline
    """
    dag = PipelineDAG()

    # Add all nodes - incorporating Bedrock batch processing and label ruleset steps
    dag.add_node("DummyDataLoading_training")  # Dummy data load for training
    dag.add_node("TabularPreprocessing_training")  # Tabular preprocessing for training
    dag.add_node(
        "BedrockPromptTemplateGeneration"
    )  # Bedrock prompt template generation (shared)
    dag.add_node(
        "BedrockBatchProcessing_training"
    )  # Bedrock batch processing step for training
    dag.add_node(
        "LabelRulesetGeneration"
    )  # Label ruleset generation (shared for training and calibration)
    dag.add_node(
        "LabelRulesetExecution_training"
    )  # Label ruleset execution for training data
    dag.add_node("PyTorchTraining")  # PyTorch training step
    dag.add_node(
        "ModelCalibration_calibration"
    )  # Model calibration step with calibration variant
    dag.add_node("Package")  # Package step
    dag.add_node("Registration")  # MIMS registration step
    dag.add_node("Payload")  # Payload step
    dag.add_node("DummyDataLoading_calibration")  # Dummy data load for calibration
    dag.add_node(
        "TabularPreprocessing_calibration"
    )  # Tabular preprocessing for calibration
    dag.add_node(
        "BedrockBatchProcessing_calibration"
    )  # Bedrock batch processing step for calibration
    dag.add_node(
        "LabelRulesetExecution_calibration"
    )  # Label ruleset execution for calibration data
    dag.add_node("PyTorchModelEval_calibration")  # Model evaluation step

    # Training flow with Bedrock batch processing and label ruleset integration
    dag.add_edge("DummyDataLoading_training", "TabularPreprocessing_training")

    # Bedrock batch processing flow for training - two inputs to BedrockBatchProcessing_training
    dag.add_edge(
        "TabularPreprocessing_training", "BedrockBatchProcessing_training"
    )  # Data input
    dag.add_edge(
        "BedrockPromptTemplateGeneration", "BedrockBatchProcessing_training"
    )  # Template input

    # Label ruleset execution for training - two inputs to LabelRulesetExecution_training
    dag.add_edge(
        "BedrockBatchProcessing_training", "LabelRulesetExecution_training"
    )  # Data input
    dag.add_edge(
        "LabelRulesetGeneration", "LabelRulesetExecution_training"
    )  # Ruleset input

    # Labeled data flows to PyTorch training
    dag.add_edge("LabelRulesetExecution_training", "PyTorchTraining")

    # Calibration flow with Bedrock batch processing and label ruleset integration
    dag.add_edge("DummyDataLoading_calibration", "TabularPreprocessing_calibration")

    # Bedrock batch processing flow for calibration - two inputs to BedrockBatchProcessing_calibration
    dag.add_edge(
        "TabularPreprocessing_calibration", "BedrockBatchProcessing_calibration"
    )  # Data input
    dag.add_edge(
        "BedrockPromptTemplateGeneration", "BedrockBatchProcessing_calibration"
    )  # Template input

    # Label ruleset execution for calibration - two inputs to LabelRulesetExecution_calibration
    dag.add_edge(
        "BedrockBatchProcessing_calibration", "LabelRulesetExecution_calibration"
    )  # Data input
    dag.add_edge(
        "LabelRulesetGeneration", "LabelRulesetExecution_calibration"
    )  # Ruleset input

    # Evaluation flow
    dag.add_edge("PyTorchTraining", "PyTorchModelEval_calibration")
    dag.add_edge(
        "LabelRulesetExecution_calibration", "PyTorchModelEval_calibration"
    )  # Use labeled calibration data

    # Model calibration flow - depends on model evaluation
    dag.add_edge("PyTorchModelEval_calibration", "ModelCalibration_calibration")

    # Output flow
    dag.add_edge("ModelCalibration_calibration", "Package")
    dag.add_edge("PyTorchTraining", "Package")  # Raw model is also input to packaging
    dag.add_edge("PyTorchTraining", "Payload")  # Payload test uses the raw model
    dag.add_edge("Package", "Registration")
    dag.add_edge("Payload", "Registration")

    logger.info(
        f"Created Bedrock Batch-PyTorch with Label Ruleset E2E DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges"
    )
    return dag


@MODSTemplate(author=AUTHOR, description=PIPELINE_DESCRIPTION, version=PIPELINE_VERSION)
class RnRPytorchBedRockPipeline:
    """
    Simple Pipline class that bridges between Cursus DAG-based pipeline architecture and MODS Template structure.

    This class specifically creates an one-step pipeline using the DAG compiler.
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
        self.dag = create_bedrock_batch_pytorch_with_label_ruleset_e2e_dag()
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

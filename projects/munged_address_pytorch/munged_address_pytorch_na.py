import logging
from pathlib import Path
from typing import Optional, Dict, Any

from sagemaker import Session
from sagemaker.workflow.pipeline import Pipeline

from cursus.api.dag.base_dag import PipelineDAG
from cursus.core.compiler.dag_compiler import PipelineDAGCompiler

DEFAULT_MODEL_CLASS = "pytorch"
DEFAULT_REGION = "NA"
DEFAULT_SERVICE_NAME = "MungedAddressDetection"

AUTHOR = "bjjin"
PIPELINE_VERSION = "0.0.1"
PIPELINE_DESCRIPTION = "Munged Address Detection DistilBERT Training Pipeline"

logger = logging.getLogger(__name__)


def create_munged_address_training_dag() -> PipelineDAG:
    """
    Create DAG for Munged Address Detection training pipeline.

    Phase 1: Tag generation (extract munged order IDs → upload to EDX)
    Phase 2: Address extraction (pull addresses using tags)
    Phase 3: LLM scoring (Bedrock strangeness rating)
    Phase 4: Preprocessing (label flip + merge + split + tokenize)
    Phase 5: Training (DistilBERT fine-tuning)
    Phase 6: Evaluation + Registration

    Returns:
        PipelineDAG: The directed acyclic graph for the pipeline
    """
    dag = PipelineDAG()

    # Phase 1: Tag generation (extract munged order IDs → upload to EDX)
    dag.add_node("CradleDataLoading_tagging")
    dag.add_node("EdxUploading_tagging")

    # Phase 2: Address extraction (pull addresses using EDX tags as join key)
    dag.add_node("CradleDataLoading_munged")
    dag.add_node("CradleDataLoading_normal")

    # Phase 3: Preprocessing (parse shards + dedup + compute reference_counts.json)
    dag.add_node("TabularPreprocessing_sampling")

    # Phase 3b: Stratified sampling (5× good, pass bad through via filter)
    dag.add_node("StratifiedSampling_sampling")

    # Phase 4: LLM scoring (single node scores ALL addresses, self-contained mode)
    dag.add_node("BedrockProcessing_scoring")

    # Phase 5: Preprocessing (label flip using __cohort__ + score, split)
    dag.add_node("TabularPreprocessing_training")

    # Phase 6: Training
    dag.add_node("PyTorchTraining")

    # Phase 7: Calibration (label-free: inference → percentile mapping)
    dag.add_node("CradleDataLoading_calibration")
    dag.add_node("TabularPreprocessing_calibration")
    dag.add_node("PyTorchModelInference_calibration")
    dag.add_node("PercentileModelCalibration_calibration")

    # Phase 8: Package + Payload → Registration
    dag.add_node("Package")
    dag.add_node("Payload")
    dag.add_node("Registration")

    # Phase 1 edges: Tag generation → upload
    dag.add_edge("CradleDataLoading_tagging", "EdxUploading_tagging")

    # Note: No edge from EdxUploading_tagging → CradleDataLoading_munged.
    # DataUploading is a SINK node (no outputs). CradleDataLoading_munged uses
    # the EDX ARN statically in config — Cradle waits for EDX data to be ready.

    # Phase 3 edges: Cradle → preprocessing (parse + dedup + reference_counts)
    dag.add_edge("CradleDataLoading_munged", "TabularPreprocessing_sampling")
    dag.add_edge("CradleDataLoading_normal", "TabularPreprocessing_sampling")

    # Phase 3b edges: Preprocessing → stratified sampling (filter: sample good, pass bad)
    dag.add_edge("TabularPreprocessing_sampling", "StratifiedSampling_sampling")

    # Phase 4 edges: Sampled data → single LLM scoring node (scores all addresses)
    dag.add_edge("StratifiedSampling_sampling", "BedrockProcessing_scoring")

    # Phase 5 edges: LLM scores → final preprocessing (label flip + split)
    dag.add_edge("BedrockProcessing_scoring", "TabularPreprocessing_training")

    # Phase 6 edges: Preprocessed → training
    dag.add_edge("TabularPreprocessing_training", "PyTorchTraining")

    # Phase 7 edges: Calibration (label-free)
    dag.add_edge("CradleDataLoading_calibration", "TabularPreprocessing_calibration")
    dag.add_edge("PyTorchTraining", "PyTorchModelInference_calibration")
    dag.add_edge(
        "TabularPreprocessing_calibration", "PyTorchModelInference_calibration"
    )
    dag.add_edge(
        "PyTorchModelInference_calibration", "PercentileModelCalibration_calibration"
    )

    # Phase 8 edges: Package + Payload → Registration
    dag.add_edge("PercentileModelCalibration_calibration", "Package")
    dag.add_edge("PyTorchTraining", "Package")
    dag.add_edge("PyTorchTraining", "Payload")
    dag.add_edge("Package", "Registration")
    dag.add_edge("Payload", "Registration")

    logger.info(
        f"Created Munged Address Training DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges"
    )
    return dag


class MungedAddressTrainingPipelineNA:
    """
    Cursus DAG-based training pipeline for Munged Address Detection.
    Development version — not yet @MODSTemplate decorated (will be added
    when migrating to BuyerAbuseModsTemplate).
    """

    def __init__(
        self,
        sagemaker_session: Optional[Session] = None,
        execution_role: Optional[str] = None,
        regional_alias: str = DEFAULT_REGION,
    ) -> None:
        self.sagemaker_session = sagemaker_session or Session()
        self.execution_role = (
            execution_role or self.sagemaker_session.get_caller_identity_arn()
        )

        module_dir = Path(__file__).resolve().parent
        config_dir = module_dir / "pipeline_configs"
        config_path = str(config_dir / f"config_{regional_alias}.json")
        logger.info(f"Using config path: {config_path}")

        self.config_path = config_path
        self.dag = create_munged_address_training_dag()
        logger.info(
            f"Created pipeline DAG with {len(self.dag.nodes)} nodes and {len(self.dag.edges)} edges"
        )

        self.dag_compiler = PipelineDAGCompiler(
            config_path=self.config_path,
            sagemaker_session=self.sagemaker_session,
            role=self.execution_role,
        )
        logger.info("Initialized DAG compiler")

    def generate_pipeline(self) -> Pipeline:
        pipeline, report = self.dag_compiler.compile_with_report(dag=self.dag)
        logger.info(f"Pipeline '{pipeline.name}' created successfully")
        logger.info(f"Average resolution confidence: {report.avg_confidence:.2f}")
        return pipeline

    def validate_dag_compatibility(self) -> Dict[str, Any]:
        validation = self.dag_compiler.validate_dag_compatibility(self.dag)
        return {
            "is_valid": validation.is_valid,
            "missing_configs": validation.missing_configs,
            "unresolvable_builders": validation.unresolvable_builders,
            "config_errors": validation.config_errors,
            "dependency_issues": validation.dependency_issues,
            "warnings": validation.warnings,
        }

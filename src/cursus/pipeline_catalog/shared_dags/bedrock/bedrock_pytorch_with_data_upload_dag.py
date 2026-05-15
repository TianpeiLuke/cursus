"""
Shared DAG definition for PyTorch E2E Pipeline with LLM Inference → Andes Upload → EDX Training

This module provides two separate pipeline DAGs that form a data pipeline:

Pipeline 1 (Inference + Upload to Andes):
1) CradleDataLoading_inference - load raw data from MDS
2) TabularPreprocessing_inference - preprocess for inference
3) BedrockBatchProcessing_inference - run LLM batch inference
4) DataUploading - upload inference results to Andes (SINK)

Pipeline 2 (Training from EDX + Registration):
1) CradleDataLoading_training - load LLM-enriched data FROM EDX (written by Pipeline 1)
2) TabularPreprocessing_training - preprocess for training
3) PyTorch Training - train the model
4) PyTorch Model Eval (calibration) - evaluate model
5) Payload Generation - generate payload samples
6) Package - package model artifacts
7) Registration - register with MIMS (SINK)

Key Design:
- Pipeline 1 produces data INTO Andes via DataUploading
- Pipeline 2 consumes data FROM Andes/EDX via CradleDataLoading_training (EDX source type)
- No DAG edge between the two pipelines — they are independent SageMaker pipelines
- Synchronization is implicit: Cradle2 polls EDX until data is available
- Each CradleDataLoading has a different job_type and data source:
  - _inference: pulls from MDS (raw data)
  - _training: pulls from EDX (LLM-enriched data uploaded by Pipeline 1)
"""

import logging

from ....api.dag.base_dag import PipelineDAG
from .. import DAGMetadata

logger = logging.getLogger(__name__)


def create_inference_upload_dag() -> PipelineDAG:
    """
    Create DAG for LLM inference pipeline that uploads results to Andes.

    Flow: CradleDataLoading (MDS) → TabPrep → BedrockBatch → DataUploading (Andes SINK)

    Returns:
        PipelineDAG: Inference + upload pipeline
    """
    dag = PipelineDAG()

    dag.add_node("CradleDataLoading_inference")
    dag.add_node("TabularPreprocessing_inference")
    dag.add_node("BedrockBatchProcessing_inference")
    dag.add_node("DataUploading")

    dag.add_edge("CradleDataLoading_inference", "TabularPreprocessing_inference")
    dag.add_edge("TabularPreprocessing_inference", "BedrockBatchProcessing_inference")
    dag.add_edge("BedrockBatchProcessing_inference", "DataUploading")

    logger.info(
        f"Created Inference+Upload DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges"
    )
    return dag


def create_training_from_edx_dag() -> PipelineDAG:
    """
    Create DAG for training pipeline that consumes LLM-enriched data from EDX.

    Flow: CradleDataLoading (EDX) → TabPrep → PyTorchTraining → Eval/Package/Registration

    The CradleDataLoading_training step is configured with EDX data source type,
    pointing to the Andes table populated by the inference_upload pipeline.

    Returns:
        PipelineDAG: Training + registration pipeline
    """
    dag = PipelineDAG()

    dag.add_node("CradleDataLoading_training")
    dag.add_node("TabularPreprocessing_training")
    dag.add_node("PyTorchTraining")
    dag.add_node("PyTorchModelEval_calibration")
    dag.add_node("ModelCalibration_calibration")
    dag.add_node("Payload")
    dag.add_node("Package")
    dag.add_node("Registration")

    dag.add_edge("CradleDataLoading_training", "TabularPreprocessing_training")
    dag.add_edge("TabularPreprocessing_training", "PyTorchTraining")
    dag.add_edge("PyTorchTraining", "PyTorchModelEval_calibration")
    dag.add_edge("PyTorchModelEval_calibration", "ModelCalibration_calibration")
    dag.add_edge("ModelCalibration_calibration", "Package")
    dag.add_edge("PyTorchTraining", "Package")
    dag.add_edge("PyTorchTraining", "Payload")
    dag.add_edge("Package", "Registration")
    dag.add_edge("Payload", "Registration")

    logger.info(
        f"Created Training-from-EDX DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges"
    )
    return dag


def create_combined_dag() -> PipelineDAG:
    """
    Create a single DAG containing both pipelines as disconnected subgraphs.

    Use this when both flows run within a single SageMaker Pipeline.
    SageMaker runs disconnected roots in parallel. CradleDataLoading_training
    will poll EDX until data is available (implicit sync via external system).

    job_type is now any lowercase alphanumeric string (not restricted to
    training/validation/testing/calibration). We use:
    - CradleDataLoading_inference → loads from MDS for LLM inference
    - CradleDataLoading_training → loads from EDX for training

    Returns:
        PipelineDAG: Combined DAG with two disconnected subgraphs
    """
    dag = PipelineDAG()

    # Subgraph 1: Inference + Upload to Andes
    dag.add_node("CradleDataLoading_inference")
    dag.add_node("TabularPreprocessing_inference")
    dag.add_node("BedrockBatchProcessing_inference")
    dag.add_node("DataUploading")

    dag.add_edge("CradleDataLoading_inference", "TabularPreprocessing_inference")
    dag.add_edge("TabularPreprocessing_inference", "BedrockBatchProcessing_inference")
    dag.add_edge("BedrockBatchProcessing_inference", "DataUploading")

    # Subgraph 2: Training from EDX + Registration
    dag.add_node("CradleDataLoading_training")
    dag.add_node("TabularPreprocessing_training")
    dag.add_node("PyTorchTraining")
    dag.add_node("PyTorchModelEval_calibration")
    dag.add_node("ModelCalibration_calibration")
    dag.add_node("Payload")
    dag.add_node("Package")
    dag.add_node("Registration")

    dag.add_edge("CradleDataLoading_training", "TabularPreprocessing_training")
    dag.add_edge("TabularPreprocessing_training", "PyTorchTraining")
    dag.add_edge("PyTorchTraining", "PyTorchModelEval_calibration")
    dag.add_edge("PyTorchModelEval_calibration", "ModelCalibration_calibration")
    dag.add_edge("ModelCalibration_calibration", "Package")
    dag.add_edge("PyTorchTraining", "Package")
    dag.add_edge("PyTorchTraining", "Payload")
    dag.add_edge("Package", "Registration")
    dag.add_edge("Payload", "Registration")

    # NOTE: No edge between subgraph 1 and subgraph 2.
    # CradleDataLoading_training polls EDX for data availability (implicit sync).

    logger.info(
        f"Created Combined DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges "
        "(2 disconnected subgraphs)"
    )
    return dag


def get_dag_metadata() -> DAGMetadata:
    """
    Get metadata for the Bedrock PyTorch with Data Upload DAGs.

    Returns:
        DAGMetadata: Metadata describing the DAG structure and purpose
    """
    return DAGMetadata(
        description="Two-pipeline pattern: LLM inference uploads to Andes, training consumes from EDX",
        complexity="comprehensive",
        features=[
            "cradle_data_loading",
            "tabular_preprocessing",
            "bedrock_batch_processing",
            "data_uploading",
            "training",
            "evaluation",
            "calibration",
            "packaging",
            "registration",
            "edx_round_trip",
        ],
        framework="pytorch",
        node_count=12,
        edge_count=12,
        extra_metadata={
            "name": "bedrock_pytorch_with_data_upload",
            "task_type": "inference_upload_then_train_from_edx",
            "pipeline_count": 2,
            "pipeline_1": {
                "name": "inference_upload",
                "entry_points": ["CradleDataLoading_inference"],
                "exit_points": ["DataUploading"],
                "cradle_source": "MDS",
                "sink": "Andes",
            },
            "pipeline_2": {
                "name": "training_from_edx",
                "entry_points": ["CradleDataLoading_training"],
                "exit_points": ["Registration"],
                "cradle_source": "EDX",
                "consumes_from": "Pipeline 1 Andes output",
            },
            "required_configs": [
                "CradleDataLoading_inference",
                "TabularPreprocessing_inference",
                "BedrockBatchProcessing_inference",
                "DataUploading",
                "CradleDataLoading_training",
                "TabularPreprocessing_training",
                "PyTorchTraining",
                "PyTorchModelEval_calibration",
                "ModelCalibration_calibration",
                "Payload",
                "Package",
                "Registration",
            ],
            "sink_nodes": ["DataUploading", "Registration"],
            "synchronization": "implicit via EDX data availability (Cradle polls)",
        },
    )

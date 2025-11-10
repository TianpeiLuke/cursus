"""
Shared DAG definition for PyTorch End-to-End Pipeline with Bedrock Processing and Label Ruleset

This module provides the shared DAG definition for a complete PyTorch workflow
that incorporates:
1. Bedrock prompt template generation and real-time processing for LLM data processing
2. Label ruleset generation and execution for transparent, rule-based label transformation
3. PyTorch training and calibration

The DAG includes label ruleset steps between Bedrock processing and training/evaluation:

Training Flow:
1) Dummy Data Loading (training)
2) Tabular Preprocessing (training)
3) Bedrock Prompt Template Generation (shared)
4) Bedrock Processing (training) - receives data + templates (REAL-TIME)
5) Label Ruleset Generation (shared) - generates rulesets for label transformation
6) Label Ruleset Execution (training) - applies rulesets to training data
7) PyTorch Model Training

Calibration Flow:
8) Dummy Data Loading (calibration)
9) Tabular Preprocessing (calibration)
10) Bedrock Processing (calibration) - receives data + templates (shared, REAL-TIME)
11) Label Ruleset Execution (calibration) - applies rulesets to calibration data
12) PyTorch Model Evaluation (calibration)

Final Steps:
13) Model Calibration
14) Package Model
15) Payload Generation
16) Model Registration

Key Features:
- Separate Bedrock real-time processing for training and calibration data
- Shared prompt template generation and label ruleset generation for consistency
- Transparent rule-based label transformation with validation
- Real-time LLM-enhanced data processing using AWS Bedrock API
- Suitable for smaller datasets or latency-sensitive workflows
- Job type variants (training vs calibration) for different processing behaviors
- Complete end-to-end workflow from data loading to model registration
"""

import logging
from typing import Dict, Any

from ....api.dag.base_dag import PipelineDAG
from .. import DAGMetadata

logger = logging.getLogger(__name__)


def create_bedrock_pytorch_with_label_ruleset_e2e_dag() -> PipelineDAG:
    """
    Create a DAG for Bedrock Real-time-enhanced PyTorch E2E pipeline with Label Ruleset steps.

    This DAG represents a complete end-to-end workflow that uses:
    1. Bedrock prompt template generation and real-time processing for LLM-enhanced data
    2. Label ruleset generation and execution for transparent label transformation
    3. PyTorch training, followed by calibration, packaging, and registration

    The label ruleset steps sit between Bedrock processing and training/evaluation,
    providing transparent, rule-based label transformation that's easy to modify.

    Returns:
        PipelineDAG: The directed acyclic graph for the pipeline
    """
    dag = PipelineDAG()

    # Add all nodes - incorporating Bedrock real-time processing and label ruleset steps
    dag.add_node("DummyDataLoading_training")  # Dummy data load for training
    dag.add_node("TabularPreprocessing_training")  # Tabular preprocessing for training
    dag.add_node(
        "BedrockPromptTemplateGeneration"
    )  # Bedrock prompt template generation (shared)
    dag.add_node(
        "BedrockProcessing_training"
    )  # Bedrock real-time processing step for training
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
        "BedrockProcessing_calibration"
    )  # Bedrock real-time processing step for calibration
    dag.add_node(
        "LabelRulesetExecution_calibration"
    )  # Label ruleset execution for calibration data
    dag.add_node("PyTorchModelEval_calibration")  # Model evaluation step

    # Training flow with Bedrock real-time processing and label ruleset integration
    dag.add_edge("DummyDataLoading_training", "TabularPreprocessing_training")

    # Bedrock real-time processing flow for training - two inputs to BedrockProcessing_training
    dag.add_edge(
        "TabularPreprocessing_training", "BedrockProcessing_training"
    )  # Data input
    dag.add_edge(
        "BedrockPromptTemplateGeneration", "BedrockProcessing_training"
    )  # Template input

    # Label ruleset execution for training - two inputs to LabelRulesetExecution_training
    dag.add_edge(
        "BedrockProcessing_training", "LabelRulesetExecution_training"
    )  # Data input
    dag.add_edge(
        "LabelRulesetGeneration", "LabelRulesetExecution_training"
    )  # Ruleset input

    # Labeled data flows to PyTorch training
    dag.add_edge("LabelRulesetExecution_training", "PyTorchTraining")

    # Calibration flow with Bedrock real-time processing and label ruleset integration
    dag.add_edge("DummyDataLoading_calibration", "TabularPreprocessing_calibration")

    # Bedrock real-time processing flow for calibration - two inputs to BedrockProcessing_calibration
    dag.add_edge(
        "TabularPreprocessing_calibration", "BedrockProcessing_calibration"
    )  # Data input
    dag.add_edge(
        "BedrockPromptTemplateGeneration", "BedrockProcessing_calibration"
    )  # Template input

    # Label ruleset execution for calibration - two inputs to LabelRulesetExecution_calibration
    dag.add_edge(
        "BedrockProcessing_calibration", "LabelRulesetExecution_calibration"
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
        f"Created Bedrock Real-time-PyTorch with Label Ruleset E2E DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges"
    )
    return dag


def get_dag_metadata() -> DAGMetadata:
    """
    Get metadata for the Bedrock Real-time-enhanced PyTorch with Label Ruleset end-to-end DAG.

    Returns:
        DAGMetadata: Metadata describing the DAG structure and purpose
    """
    return DAGMetadata(
        description="Bedrock Real-time-enhanced PyTorch end-to-end pipeline with label ruleset generation/execution for transparent rule-based label transformation, training, calibration, packaging, and registration",
        complexity="comprehensive",
        features=[
            "dummy_data_loading",
            "bedrock_prompt_generation",
            "bedrock_realtime_processing",
            "label_ruleset_generation",
            "label_ruleset_execution",
            "transparent_labeling",
            "training",
            "calibration",
            "packaging",
            "registration",
            "evaluation",
        ],
        framework="pytorch",
        node_count=16,
        edge_count=20,
        extra_metadata={
            "name": "bedrock_pytorch_with_label_ruleset_e2e",
            "task_type": "end_to_end_with_realtime_llm_and_label_ruleset",
            "entry_points": [
                "DummyDataLoading_training",
                "DummyDataLoading_calibration",
                "BedrockPromptTemplateGeneration",
                "LabelRulesetGeneration",
            ],
            "exit_points": ["Registration"],
            "required_configs": [
                "DummyDataLoading_training",
                "DummyDataLoading_calibration",
                "TabularPreprocessing_training",
                "TabularPreprocessing_calibration",
                "BedrockPromptTemplateGeneration",
                "BedrockProcessing_training",
                "BedrockProcessing_calibration",
                "LabelRulesetGeneration",
                "LabelRulesetExecution_training",
                "LabelRulesetExecution_calibration",
                "PyTorchTraining",
                "PyTorchModelEval_calibration",
                "ModelCalibration_calibration",
                "Package",
                "Payload",
                "Registration",
            ],
            "bedrock_integration": {
                "template_generation": "BedrockPromptTemplateGeneration",
                "training_processing": "BedrockProcessing_training",
                "calibration_processing": "BedrockProcessing_calibration",
                "processing_mode": "real-time",
                "training_flow": {
                    "input_sources": [
                        "TabularPreprocessing_training",
                        "BedrockPromptTemplateGeneration",
                    ],
                    "output_target": "LabelRulesetExecution_training",
                },
                "calibration_flow": {
                    "input_sources": [
                        "TabularPreprocessing_calibration",
                        "BedrockPromptTemplateGeneration",
                    ],
                    "output_target": "LabelRulesetExecution_calibration",
                },
            },
            "label_ruleset_integration": {
                "ruleset_generation": "LabelRulesetGeneration",
                "training_execution": "LabelRulesetExecution_training",
                "calibration_execution": "LabelRulesetExecution_calibration",
                "training_flow": {
                    "input_sources": [
                        "BedrockProcessing_training",
                        "LabelRulesetGeneration",
                    ],
                    "output_target": "PyTorchTraining",
                },
                "calibration_flow": {
                    "input_sources": [
                        "BedrockProcessing_calibration",
                        "LabelRulesetGeneration",
                    ],
                    "output_target": "PyTorchModelEval_calibration",
                },
                "key_features": {
                    "transparent_rules": "Easy-to-read format for rulesets",
                    "validation": "Field availability and label category validation",
                    "shared_rulesets": "Same ruleset used for training and calibration consistency",
                    "format_preservation": "Maintains CSV/TSV/Parquet format through pipeline",
                },
            },
        },
    )

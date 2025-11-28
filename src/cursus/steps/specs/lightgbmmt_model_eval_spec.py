"""
LightGBMMT Multi-Task Model Evaluation Step Specification.

This module defines the declarative specification for LightGBMMT multi-task model evaluation steps,
including their dependencies and outputs based on the actual implementation.
"""

from ...core.base.specification_base import (
    StepSpecification,
    DependencySpec,
    OutputSpec,
    DependencyType,
    NodeType,
)
from ...registry.step_names import get_spec_step_type
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..contracts.lightgbmmt_model_eval_contract import (
        LIGHTGBMMT_MODEL_EVAL_CONTRACT,
    )


# Import the contract at runtime to avoid circular imports
def _get_model_evaluation_contract():
    from ..contracts.lightgbmmt_model_eval_contract import (
        LIGHTGBMMT_MODEL_EVAL_CONTRACT,
    )

    return LIGHTGBMMT_MODEL_EVAL_CONTRACT


# LightGBMMT Multi-Task Model Evaluation Step Specification
LIGHTGBMMT_MODEL_EVAL_SPEC = StepSpecification(
    step_type=get_spec_step_type("LightGBMMTModelEval"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_model_evaluation_contract(),
    dependencies=[
        DependencySpec(
            logical_name="model_input",
            dependency_type=DependencyType.MODEL_ARTIFACTS,
            required=True,
            compatible_sources=[
                "LightGBMMTTraining",
                "LightGBMTraining",
                "LightGBMMTModel",
                "LightGBMModel",
                "DummyTraining",
            ],
            semantic_keywords=[
                "model",
                "artifacts",
                "trained",
                "output",
                "ModelArtifacts",
                "multi-task",
                "multitask",
            ],
            data_type="S3Uri",
            description="Trained multi-task model artifacts to be evaluated (includes hyperparameters.json with task_label_names)",
        ),
        DependencySpec(
            logical_name="processed_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=[
                "TabularPreprocessing",
                "CradleDataLoading",
                "RiskTableMapping",
                "CurrencyConversion",
                "LabelRulesetExecution",
                "BedrockBatchProcessing",
                "BedrockProcessing",
            ],
            semantic_keywords=[
                "data",
                "evaluation",
                "calibration",
                "validation",
                "test",
                "processed",
                "multi-task",
                "multitask",
            ],
            data_type="S3Uri",
            description="Evaluation dataset for multi-task model assessment (must contain all task label columns)",
        ),
    ],
    outputs=[
        OutputSpec(
            logical_name="eval_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['eval_output'].S3Output.S3Uri",
            data_type="S3Uri",
            aliases=[
                "inference_output",
                "predictions",
                "model_predictions",
                "inference_results",
                "prediction_data",
                "evaluation_data",
                "multi_task_predictions",
                "multitask_predictions",
            ],
            description="Multi-task model evaluation results including per-task predictions",
        ),
        OutputSpec(
            logical_name="metrics_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['metrics_output'].S3Output.S3Uri",
            data_type="S3Uri",
            aliases=[
                "multi_task_metrics",
                "multitask_metrics",
            ],
            description="Multi-task model evaluation metrics (per-task and aggregate AUC, precision, recall, etc.)",
        ),
    ],
)

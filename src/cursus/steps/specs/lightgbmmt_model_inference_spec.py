"""
LightGBMMT Multi-Task Model Inference Step Specification.

This module defines the declarative specification for LightGBMMT multi-task model inference steps,
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
    from ..contracts.lightgbmmt_model_inference_contract import (
        LIGHTGBMMT_MODEL_INFERENCE_CONTRACT,
    )


# Import the contract at runtime to avoid circular imports
def _get_model_inference_contract():
    from ..contracts.lightgbmmt_model_inference_contract import (
        LIGHTGBMMT_MODEL_INFERENCE_CONTRACT,
    )

    return LIGHTGBMMT_MODEL_INFERENCE_CONTRACT


# LightGBMMT Multi-Task Model Inference Step Specification
LIGHTGBMMT_MODEL_INFERENCE_SPEC = StepSpecification(
    step_type=get_spec_step_type("LightGBMMTModelInference"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_model_inference_contract(),
    dependencies={
        "model_input": DependencySpec(
            logical_name="model_input",
            dependency_type=DependencyType.MODEL_ARTIFACTS,
            required=True,
            compatible_sources=[
                "LightGBMMTTraining",
                "LightGBMTraining",
                "LightGBMMTModel",
                "LightGBMModel",
                "XGBoostTraining",
                "PyTorchTraining",
                "DummyTraining",
                "XGBoostModel",
                "PyTorchModel",
            ],
            semantic_keywords=[
                "model",
                "artifacts",
                "trained",
                "output",
                "ModelArtifacts",
                "multi-task",
                "multitask",
                "lightgbm",
                "lightgbmmt",
                "training",
            ],
            data_type="S3Uri",
            description="Trained multi-task model artifacts for inference (includes hyperparameters.json with task_label_names)",
        ),
        "processed_data": DependencySpec(
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
                "inference",
                "input",
                "multi-task",
                "multitask",
                "unlabeled",
                "prediction",
            ],
            data_type="S3Uri",
            description="Input dataset for multi-task model inference (task labels are optional)",
        ),
    },
    outputs={
        "eval_output": OutputSpec(
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
                "multi_task_inference",
                "multitask_inference",
            ],
            description="Multi-task model inference results with per-task predictions (no metrics or plots)",
        ),
    },
)

"""
XGBoost Model Inference Step Specification.

This module defines the declarative specification for XGBoost model inference steps,
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
    from ..contracts.xgboost_model_inference_contract import XGBOOST_MODEL_INFERENCE_CONTRACT


# Import the contract at runtime to avoid circular imports
def _get_model_inference_contract():
    from ..contracts.xgboost_model_inference_contract import XGBOOST_MODEL_INFERENCE_CONTRACT

    return XGBOOST_MODEL_INFERENCE_CONTRACT


# XGBoost Model Inference Step Specification
MODEL_INFERENCE_SPEC = StepSpecification(
    step_type=get_spec_step_type("XGBoostModelInference"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_model_inference_contract(),
    dependencies={
        "model_input": DependencySpec(
            logical_name="model_input",
            dependency_type=DependencyType.MODEL_ARTIFACTS,
            required=True,
            compatible_sources=[
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
                "xgboost",
                "training",
            ],
            data_type="S3Uri",
            description="Trained model artifacts for inference (includes hyperparameters.json)",
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
            ],
            data_type="S3Uri",
            description="Evaluation dataset for model inference",
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
            ],
            description="Model inference results with predictions and original data",
        ),
    },
)

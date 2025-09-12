"""
XGBoost Model Evaluation Step Specification.

This module defines the declarative specification for XGBoost model evaluation steps,
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
    from ..contracts.xgboost_model_eval_contract import XGBOOST_MODEL_EVAL_CONTRACT


# Import the contract at runtime to avoid circular imports
def _get_model_evaluation_contract():
    from ..contracts.xgboost_model_eval_contract import XGBOOST_MODEL_EVAL_CONTRACT

    return XGBOOST_MODEL_EVAL_CONTRACT


# XGBoost Model Evaluation Step Specification
MODEL_EVAL_SPEC = StepSpecification(
    step_type=get_spec_step_type("XGBoostModelEval"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_model_evaluation_contract(),
    dependencies=[
        DependencySpec(
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
            ],
            data_type="S3Uri",
            description="Trained model artifacts to be evaluated (includes hyperparameters.json)",
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
            ],
            semantic_keywords=[
                "data",
                "evaluation",
                "calibration",
                "validation",
                "test",
                "processed",
            ],
            data_type="S3Uri",
            description="Evaluation dataset for model assessment",
        ),
    ],
    outputs=[
        OutputSpec(
            logical_name="eval_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['eval_output'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Model evaluation results including predictions",
        ),
        OutputSpec(
            logical_name="metrics_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['metrics_output'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Model evaluation metrics (AUC, precision, recall, etc.)",
        ),
    ],
)

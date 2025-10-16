"""
XGBoost Training Step Specification.

This module defines the declarative specification for XGBoost training steps,
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


# Import the contract at runtime to avoid circular imports
def _get_xgboost_train_contract():
    from ..contracts.xgboost_training_contract import XGBOOST_TRAIN_CONTRACT

    return XGBOOST_TRAIN_CONTRACT


# XGBoost Training Step Specification
XGBOOST_TRAINING_SPEC = StepSpecification(
    step_type=get_spec_step_type("XGBoostTraining"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_xgboost_train_contract(),
    dependencies=[
        DependencySpec(
            logical_name="input_path",
            dependency_type=DependencyType.TRAINING_DATA,
            required=True,
            compatible_sources=[
                "TabularPreprocessing",
                "StratifiedSampling",
                "RiskTableMapping",
                "MissingValueImputation",
                "ProcessingStep",
                "DataLoad",
            ],
            semantic_keywords=[
                "data",
                "input",
                "training",
                "dataset",
                "processed",
                "train",
                "tabular",
            ],
            data_type="S3Uri",
            description="Training dataset S3 location with train/val/test subdirectories",
        ),
        DependencySpec(
            logical_name="hyperparameters_s3_uri",
            dependency_type=DependencyType.HYPERPARAMETERS,
            required=False,  # Can be generated internally
            compatible_sources=["HyperparameterPrep", "ProcessingStep"],
            semantic_keywords=[
                "config",
                "params",
                "hyperparameters",
                "settings",
                "hyperparams",
            ],
            data_type="S3Uri",
            description="S3 URI containing hyperparameters configuration file (optional - falls back to source directory)",
        ),
    ],
    outputs=[
        OutputSpec(
            logical_name="model_output",
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ModelArtifacts.S3ModelArtifacts",
            data_type="S3Uri",
            description="Trained XGBoost model artifacts",
            aliases=[
                "ModelOutputPath",
                "ModelArtifacts",
                "model_data",
                "output_path",
                "model_input",
            ],
        ),
        OutputSpec(
            logical_name="evaluation_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.OutputDataConfig.S3OutputPath",
            data_type="S3Uri",
            description="Model evaluation results and predictions (val.tar.gz, test.tar.gz)",
            aliases=[
                "evaluation_data",
                "eval_data",
                "validation_output",
                "test_output",
                "prediction_results",
            ],
        ),
    ],
)

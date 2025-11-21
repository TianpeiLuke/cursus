"""
LightGBM Training Step Specification.

This module defines the declarative specification for LightGBM training steps,
following the same patterns as XGBoost but adapted for LightGBM algorithm.
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
def _get_lightgbm_train_contract():
    from ..contracts.lightgbm_training_contract import LIGHTGBM_TRAIN_CONTRACT

    return LIGHTGBM_TRAIN_CONTRACT


# LightGBM Training Step Specification
LIGHTGBM_TRAINING_SPEC = StepSpecification(
    step_type=get_spec_step_type("LightGBMTraining"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_lightgbm_train_contract(),
    dependencies=[
        DependencySpec(
            logical_name="input_path",
            dependency_type=DependencyType.TRAINING_DATA,
            required=True,
            compatible_sources=[
                "TabularPreprocessing",
                "BedrockProcessing",
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
        DependencySpec(
            logical_name="model_artifacts_input",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=False,  # Optional - used for pre-computed preprocessing artifacts or pretrained models
            compatible_sources=[
                "LightGBMTraining",
                "MissingValueImputation",
                "RiskTableMapping",
                "FeatureSelection",
            ],
            semantic_keywords=[
                "artifacts",
                "model_artifacts",
                "preprocessing",
                "imputation",
                "risk_tables",
                "features",
                "parameters",
                "model_artifacts_output",
                "pretrain",
                "pretrained",
            ],
            data_type="S3Uri",
            description="Optional pre-computed preprocessing artifacts (impute_dict.pkl, risk_table_map.pkl, selected_features.json). When provided with USE_PRECOMPUTED_* environment variables, skips inline computation and uses pre-computed artifacts.",
        ),
    ],
    outputs=[
        OutputSpec(
            logical_name="model_output",
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ModelArtifacts.S3ModelArtifacts",
            data_type="S3Uri",
            description="Trained LightGBM model artifacts",
            aliases=[
                "ModelOutputPath",
                "ModelArtifacts",
                "model_data",
                "output_path",
                "model_input",
                "model_artifacts_input",
                "lightgbm_model",
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

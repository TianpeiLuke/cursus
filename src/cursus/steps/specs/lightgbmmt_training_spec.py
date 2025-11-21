"""
LightGBMMT Training Step Specification.

This module defines the declarative specification for LightGBMMT multi-task training steps,
extending LightGBM patterns for multi-label/multi-task learning scenarios.
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
def _get_lightgbmmt_train_contract():
    from ..contracts.lightgbmmt_training_contract import LIGHTGBMMT_TRAIN_CONTRACT

    return LIGHTGBMMT_TRAIN_CONTRACT


# Lightgbmmt Training Step Specification
LIGHTGBMMT_TRAINING_SPEC = StepSpecification(
    step_type=get_spec_step_type("LightGBMMTTraining"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_lightgbmmt_train_contract(),
    dependencies=[
        DependencySpec(
            logical_name="input_path",
            dependency_type=DependencyType.TRAINING_DATA,
            required=True,
            compatible_sources=[
                "TabularPreprocessing",
                "StratifiedSampling",
                "ProcessingStep",
                "DataLoad",
            ],
            semantic_keywords=[
                "data",
                "input",
                "training",
                "dataset",
                "multi_label",
                "multi_task",
                "processed",
                "train",
                "tabular",
            ],
            data_type="S3Uri",
            description="Multi-label training dataset S3 location with train/val/test subdirectories",
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
                "multi_task_config",
            ],
            data_type="S3Uri",
            description="S3 URI containing hyperparameters configuration file (optional - falls back to source directory)",
        ),
        DependencySpec(
            logical_name="model_artifacts_input",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=False,  # Optional - used for pre-computed preprocessing artifacts or pretrained models
            compatible_sources=[
                "LightGBMMTTraining",
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
            description="Trained LightGBMMT multi-task model artifacts",
            aliases=[
                "ModelOutputPath",
                "ModelArtifacts",
                "model_data",
                "output_path",
                "model_input",
                "model_artifacts_input",
                "lightgbmmt_model",
                "multi_task_model",
                "mtgbm_model",
            ],
        ),
        OutputSpec(
            logical_name="evaluation_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.OutputDataConfig.S3OutputPath",
            data_type="S3Uri",
            description="Multi-task evaluation results, predictions, and metrics (per-task and aggregate)",
            aliases=[
                "evaluation_data",
                "eval_data",
                "validation_output",
                "test_output",
                "prediction_results",
                "multi_task_results",
                "task_predictions",
            ],
        ),
    ],
)

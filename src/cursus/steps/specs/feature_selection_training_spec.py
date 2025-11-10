"""
Feature Selection Training Step Specification.

This module defines the declarative specification for feature selection steps
specifically for training data, including their dependencies and outputs.
"""

from ...core.base.specification_base import (
    StepSpecification,
    DependencySpec,
    OutputSpec,
    DependencyType,
    NodeType,
)
from ...registry.step_names import get_spec_step_type_with_job_type


# Import the contract at runtime to avoid circular imports
def _get_feature_selection_contract():
    from ..contracts.feature_selection_contract import FEATURE_SELECTION_CONTRACT

    return FEATURE_SELECTION_CONTRACT


# Feature Selection Training Step Specification
FEATURE_SELECTION_TRAINING_SPEC = StepSpecification(
    step_type=get_spec_step_type_with_job_type("FeatureSelection", "training"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_feature_selection_contract(),
    dependencies=[
        DependencySpec(
            logical_name="input_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=[
                "TabularPreprocessing",
                "StratifiedSampling",
                "RiskTableMapping",
                "MissingValueImputation",
                "ProcessingStep",
            ],
            semantic_keywords=[
                "training",
                "train",
                "processed_data",
                "preprocessed",
                "cleaned",
                "tabular",
                "data",
                "input",
                "dataset",
                "splits",
                "feature_engineering",
                "model_training",
                "input_data",
                "output_data",
            ],
            data_type="S3Uri",
            description="Processed training data from preprocessing steps for feature selection",
        ),
        # Selected features dependency - optional for training mode since training creates them
        DependencySpec(
            logical_name="model_artifacts_input",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=False,
            compatible_sources=["FeatureSelection_Training", "ProcessingStep"],
            semantic_keywords=[
                "selected_features",
                "feature_selection",
                "feature_metadata",
                "feature_artifacts",
                "training_artifacts",
                "selected_features_input",
                "selected_features_output",
                "model_artifacts_output",
            ],
            data_type="S3Uri",
            description="Optional pre-existing selected features (training mode creates new ones if not provided)",
        ),
    ],
    outputs=[
        OutputSpec(
            logical_name="processed_data",
            aliases=[
                "selected_data",
                "feature_selected_data",
                "training_data",
                "model_input_data",
                "input_path",
                "output_data",
                "input_data",
                "input_path",
            ],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Training data with selected features applied using statistical and ML methods",
        ),
        OutputSpec(
            logical_name="model_artifacts_output",
            aliases=[
                "selected_features",
                "feature_selection",
                "feature_metadata",
                "feature_artifacts",
                "selection_results",
                "training_artifacts",
                "selected_features_input",
                "model_artifacts_input",
            ],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['model_artifacts_output'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Selected features metadata and scores from training data for inference mode",
        ),
    ],
)

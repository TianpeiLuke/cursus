"""
Feature Selection Calibration Step Specification.

This module defines the declarative specification for feature selection steps
specifically for calibration data, including their dependencies and outputs.
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


# Feature Selection Calibration Step Specification
FEATURE_SELECTION_CALIBRATION_SPEC = StepSpecification(
    step_type=get_spec_step_type_with_job_type("FeatureSelection", "calibration"),
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
                "calibration",
                "calib",
                "processed_data",
                "preprocessed",
                "cleaned",
                "tabular",
                "data",
                "input",
                "dataset",
                "input_data",
                "calibration_data",
                "model_calibration",
                "output_data",
            ],
            data_type="S3Uri",
            description="Processed calibration data from preprocessing steps for feature selection",
        ),
        DependencySpec(
            logical_name="model_artifacts_input",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["FeatureSelection_Training"],
            semantic_keywords=[
                "selected_features",
                "feature_selection",
                "feature_metadata",
                "feature_artifacts",
                "training_artifacts",
                "selected_features_input",
                "feature_list",
                "feature_scores",
                "selected_features_output",
                "model_artifacts_output",
            ],
            data_type="S3Uri",
            description="Selected features metadata and scores from training step",
        ),
    ],
    outputs=[
        OutputSpec(
            logical_name="processed_data",
            aliases=[
                "selected_data",
                "feature_selected_data",
                "calibration_data",
                "model_calibration_data",
                "output_data",
                "filtered_data",
                "input_data",
                "input_path",
            ],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Calibration data with selected features applied using pre-computed selection",
        ),
        OutputSpec(
            logical_name="model_artifacts_output",
            aliases=[
                "selected_features",
                "feature_selection",
                "feature_metadata",
                "feature_artifacts",
                "selection_results",
                "feature_list",
                "selected_features_input",
                "model_artifacts_input",
            ],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['model_artifacts_output'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Selected features metadata and scores (passthrough from training)",
        ),
    ],
)

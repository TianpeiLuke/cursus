"""
Stratified Sampling Calibration Step Specification.

This module defines the declarative specification for stratified sampling steps
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
def _get_stratified_sampling_contract():
    from ..contracts.stratified_sampling_contract import STRATIFIED_SAMPLING_CONTRACT

    return STRATIFIED_SAMPLING_CONTRACT


# Stratified Sampling Calibration Step Specification
STRATIFIED_SAMPLING_CALIBRATION_SPEC = StepSpecification(
    step_type=get_spec_step_type_with_job_type("StratifiedSampling", "calibration"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_stratified_sampling_contract(),
    dependencies=[
        DependencySpec(
            logical_name="processed_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["TabularPreprocessing", "ProcessingStep"],
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
                "model_calibration",
            ],
            data_type="S3Uri",
            description="Processed calibration data from preprocessing step for stratified sampling",
        )
    ],
    outputs=[
        OutputSpec(
            logical_name="processed_data",
            aliases=[
                "sampled_data",
                "stratified_data",
                "calibration_data",
                "model_input_data",
                "input_path",
            ],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Stratified sampled calibration data",
        )
    ],
)

"""
Stratified Sampling Step Specification.

This module defines the declarative specification for stratified sampling steps,
including their dependencies and outputs based on the actual implementation.
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


# Stratified Sampling Step Specification
STRATIFIED_SAMPLING_SPEC = StepSpecification(
    step_type=get_spec_step_type_with_job_type("StratifiedSampling", "training"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_stratified_sampling_contract(),
    dependencies=[
        DependencySpec(
            logical_name="processed_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["TabularPreprocessing", "ProcessingStep"],
            semantic_keywords=[
                "processed_data",
                "preprocessed",
                "cleaned",
                "tabular",
                "data",
                "input",
                "dataset",
                "training",
                "train",
                "splits",
            ],
            data_type="S3Uri",
            description="Processed tabular data from preprocessing step for stratified sampling",
        )
    ],
    outputs=[
        OutputSpec(
            logical_name="processed_data",
            aliases=[
                "sampled_data",
                "stratified_data",
                "training_data",
                "model_input_data",
                "input_path",
            ],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Stratified sampled data with balanced train/val/test splits",
        )
    ],
)

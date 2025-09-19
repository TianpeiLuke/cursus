"""
Specification for the DummyTraining step.

This module defines the DummyTraining step specification, including its dependencies and outputs.
DummyTraining is designed to take a pretrained model and hyperparameters, add the hyperparameters
to the model.tar.gz file, and make it available for downstream packaging and payload steps.
"""

from ...core.base.specification_base import (
    StepSpecification,
    NodeType,
    DependencySpec,
    OutputSpec,
    DependencyType,
)
from ...registry.step_names import get_spec_step_type


def _get_dummy_training_contract():
    from ..contracts.dummy_training_contract import DUMMY_TRAINING_CONTRACT

    return DUMMY_TRAINING_CONTRACT


DUMMY_TRAINING_SPEC = StepSpecification(
    step_type=get_spec_step_type("DummyTraining"),
    node_type=NodeType.SOURCE,  # Changed from INTERNAL to SOURCE
    script_contract=_get_dummy_training_contract(),
    dependencies=[
        # Remove all dependencies - SOURCE node needs no external inputs
    ],
    outputs=[
        OutputSpec(
            logical_name="model_input",  # Matches contract output path name for consistency
            output_type=DependencyType.MODEL_ARTIFACTS,  # Using MODEL_ARTIFACTS for packaging compatibility
            property_path="properties.ProcessingOutputConfig.Outputs['model_input'].S3Output.S3Uri",
            data_type="S3Uri",
            description="S3 path to model artifacts with integrated hyperparameters (from source directory)",
            aliases=["ModelOutputPath", "ModelArtifacts", "model_data", "output_path"],
        )
    ],
)

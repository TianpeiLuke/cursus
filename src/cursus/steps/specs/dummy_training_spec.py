"""
Specification for the DummyTraining step with flexible input modes.

This module defines the DummyTraining step specification, including its dependencies and outputs.
DummyTraining can operate in two modes:

1. INTERNAL mode: Accepts optional inputs from previous steps
   - Can chain after PyTorchTraining, XGBoostTraining, or other training steps
   - Can accept hyperparameters from preprocessing or configuration steps

2. SOURCE mode (fallback): Reads from source directory when no inputs provided
   - Backward compatible with existing SOURCE node usage

The step processes the model by adding hyperparameters.json to model.tar.gz for
downstream packaging and payload steps.
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
    node_type=NodeType.INTERNAL,  # Changed from SOURCE to INTERNAL for flexible input support
    script_contract=_get_dummy_training_contract(),
    dependencies=[
        DependencySpec(
            logical_name="hyperparameters_s3_uri",
            dependency_type=DependencyType.HYPERPARAMETERS,
            required=False,  # Optional - falls back to source directory
            compatible_sources=["HyperparameterPrep", "ProcessingStep"],
            semantic_keywords=[
                "config",
                "params",
                "hyperparameters",
                "settings",
                "hyperparams",
            ],
            data_type="S3Uri",
            description="S3 URI containing hyperparameters configuration file (optional - falls back to code directory or source directory)",
        ),
        DependencySpec(
            logical_name="model_artifacts_input",
            dependency_type=DependencyType.MODEL_ARTIFACTS,
            required=False,  # Optional - falls back to source directory
            compatible_sources=[
                "PyTorchTraining",
                "XGBoostTraining",
                "LightGBMTraining",
                "ProcessingStep",
            ],
            semantic_keywords=[
                "model",
                "artifacts",
                "model_artifacts",
                "pretrained",
                "trained_model",
                "model_output",
            ],
            data_type="S3Uri",
            description="Optional model artifacts (model.tar.gz) from previous training step or S3. When not provided, reads from source directory.",
        ),
    ],
    outputs=[
        OutputSpec(
            logical_name="model_output",  # Updated to match contract
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ProcessingOutputConfig.Outputs['model_output'].S3Output.S3Uri",
            data_type="S3Uri",
            description="S3 path to model artifacts with integrated hyperparameters",
            aliases=[
                "ModelOutputPath",
                "ModelArtifacts",
                "model_data",
                "output_path",
                "model_input",
            ],
        )
    ],
)

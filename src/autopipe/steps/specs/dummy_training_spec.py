"""
Specification for the DummyTraining step.

This module defines the DummyTraining step specification, including its dependencies and outputs.
DummyTraining is designed to take a pretrained model and hyperparameters, add the hyperparameters
to the model.tar.gz file, and make it available for downstream packaging and payload steps.
"""

from ...core.base.specification_base import StepSpecification, NodeType, DependencySpec, OutputSpec, DependencyType
from ..registry.step_names import get_spec_step_type

def _get_dummy_training_contract():
    from ..contracts.dummy_training_contract import DUMMY_TRAINING_CONTRACT
    return DUMMY_TRAINING_CONTRACT

DUMMY_TRAINING_SPEC = StepSpecification(
    step_type=get_spec_step_type("DummyTraining"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_dummy_training_contract(),
    dependencies=[
        DependencySpec(
            logical_name="pretrained_model_path",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["ProcessingStep", "XGBoostTraining", "PytorchTraining", "TabularPreprocessing"],
            semantic_keywords=["model", "pretrained", "artifact", "weights", "training_output", "model_data"],
            data_type="S3Uri",
            description="Path to pretrained model.tar.gz file"
        ),
        DependencySpec(
            logical_name="hyperparameters_s3_uri",
            dependency_type=DependencyType.HYPERPARAMETERS,
            required=True,  # Now required for integration with downstream steps
            compatible_sources=["HyperparameterPrep", "ProcessingStep"],
            semantic_keywords=["config", "params", "hyperparameters", "settings", "hyperparams"],
            data_type="S3Uri",
            description="Hyperparameters configuration file for inclusion in the model package"
        )
    ],
    outputs=[
        OutputSpec(
            logical_name="model_input",  # Matches contract output path name for consistency
            output_type=DependencyType.MODEL_ARTIFACTS,  # Using MODEL_ARTIFACTS for packaging compatibility
            property_path="properties.ProcessingOutputConfig.Outputs['model_input'].S3Output.S3Uri",
            data_type="S3Uri",
            description="S3 path to model artifacts with integrated hyperparameters",
            aliases=["ModelOutputPath", "ModelArtifacts", "model_data", "output_path"]
        )
    ]
)

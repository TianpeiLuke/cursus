"""
PyTorch Training Step Specification.

This module defines the declarative specification for PyTorch training steps,
including their dependencies and outputs based on the actual implementation.
"""

from ...core.base.specification_base import StepSpecification, DependencySpec, OutputSpec, DependencyType, NodeType
from ..registry.step_names import get_spec_step_type

# Import the contract at runtime to avoid circular imports
def _get_pytorch_train_contract():
    from ..contracts.pytorch_train_contract import PYTORCH_TRAIN_CONTRACT
    return PYTORCH_TRAIN_CONTRACT

# PyTorch Training Step Specification
PYTORCH_TRAINING_SPEC = StepSpecification(
    step_type=get_spec_step_type("PyTorchTraining"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_pytorch_train_contract(),
    dependencies=[
        DependencySpec(
            logical_name="input_path",
            dependency_type=DependencyType.TRAINING_DATA,
            required=True,
            compatible_sources=["TabularPreprocessing", "ProcessingStep", "DataLoad", "RiskTableMapping"],
            semantic_keywords=["data", "input", "training", "dataset", "processed", "train", "pytorch"],
            data_type="S3Uri",
            description="Training dataset S3 location with train/val/test subdirectories"
        )
        # Note: Removed "config" dependency as PyTorch estimator accepts hyperparameters directly
        # and doesn't need an external hyperparameters file - this is handled internally
    ],
    outputs=[
        OutputSpec(
            logical_name="model_output",
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ModelArtifacts.S3ModelArtifacts",
            data_type="S3Uri",
            description="Trained PyTorch model artifacts",
            aliases=["ModelArtifacts", "model_data", "output_path", "model_input"]
        ),
        OutputSpec(
            logical_name="data_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.TrainingJobOutput.S3Output",
            data_type="S3Uri",
            description="Training evaluation results and predictions"
        )
    ]
)

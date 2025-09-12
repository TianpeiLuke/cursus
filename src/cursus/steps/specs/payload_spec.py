"""
MIMS Payload Step Specification.

This module defines the declarative specification for MIMS payload generation steps,
including their dependencies and outputs based on the actual implementation.
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
def _get_payload_contract():
    from ..contracts.payload_contract import PAYLOAD_CONTRACT

    return PAYLOAD_CONTRACT


# MIMS Payload Step Specification
PAYLOAD_SPEC = StepSpecification(
    step_type=get_spec_step_type("Payload"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_payload_contract(),
    dependencies=[
        DependencySpec(
            logical_name="model_input",
            dependency_type=DependencyType.MODEL_ARTIFACTS,
            required=True,
            compatible_sources=["XGBoostTraining", "TrainingStep", "ModelStep"],
            semantic_keywords=[
                "model",
                "artifacts",
                "trained",
                "output",
                "ModelArtifacts",
            ],
            data_type="S3Uri",
            description="Trained model artifacts for payload generation",
        )
    ],
    outputs=[
        OutputSpec(
            logical_name="payload_sample",
            aliases=["GeneratedPayloadSamples"],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['payload_sample'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Generated payload samples archive (payload.tar.gz)",
        )
    ],
)

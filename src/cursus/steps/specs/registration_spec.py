"""
Model Registration Step Specification.

This module defines the declarative specification for MIMS model registration steps,
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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..contracts.mims_registration_contract import MIMS_REGISTRATION_CONTRACT


# Import the contract at runtime to avoid circular imports
def _get_mims_registration_contract():
    from ..contracts.mims_registration_contract import MIMS_REGISTRATION_CONTRACT

    return MIMS_REGISTRATION_CONTRACT


# Model Registration Step Specification
REGISTRATION_SPEC = StepSpecification(
    step_type=get_spec_step_type("Registration"),
    node_type=NodeType.SINK,
    script_contract=_get_mims_registration_contract(),  # Add reference to the script contract
    dependencies=[
        DependencySpec(
            logical_name="PackagedModel",
            dependency_type=DependencyType.MODEL_ARTIFACTS,
            required=True,
            compatible_sources=["PackagingStep", "Package", "ProcessingStep"],
            semantic_keywords=["model", "package", "packaged", "artifacts", "tar"],
            data_type="S3Uri",
            description="Packaged model artifacts for registration",
        ),
        DependencySpec(
            logical_name="GeneratedPayloadSamples",
            dependency_type=DependencyType.PAYLOAD_SAMPLES,
            required=True,
            compatible_sources=["PayloadTestStep", "PayloadStep", "ProcessingStep"],
            semantic_keywords=["payload", "samples", "test", "generated", "inference"],
            data_type="S3Uri",
            description="Generated payload samples for model testing",
        ),
    ],
    outputs=[
        # Note: MIMS Registration step doesn't produce accessible outputs
        # It registers the model as a side effect but doesn't create
        # output properties that can be referenced by subsequent steps
    ],
)

"""
Cradle Data Loading Step Specification.

This module defines the declarative specification for Cradle data loading steps,
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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..contracts.cradle_data_loading_contract import CRADLE_DATA_LOADING_CONTRACT


# Import the contract at runtime to avoid circular imports
def _get_cradle_data_loading_contract():
    from ..contracts.cradle_data_loading_contract import CRADLE_DATA_LOADING_CONTRACT

    return CRADLE_DATA_LOADING_CONTRACT


# Cradle Data Loading Step Specification
DATA_LOADING_SPEC = StepSpecification(
    step_type=get_spec_step_type_with_job_type("CradleDataLoading", "training"),
    node_type=NodeType.SOURCE,
    script_contract=_get_cradle_data_loading_contract(),  # Add reference to the script contract
    dependencies=[
        # Note: CradleDataLoading is typically the first step in a pipeline
        # and doesn't depend on other pipeline steps - it loads data from external sources
    ],
    outputs=[
        OutputSpec(
            logical_name="DATA",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['DATA'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Main data output from Cradle data loading",
        ),
        OutputSpec(
            logical_name="METADATA",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['METADATA'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Metadata output from Cradle data loading",
        ),
        OutputSpec(
            logical_name="SIGNATURE",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['SIGNATURE'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Signature output from Cradle data loading",
        ),
    ],
)

"""
Dummy Data Loading Testing Step Specification.

This module defines the declarative specification for Dummy data loading steps
specifically for testing data, including their dependencies and outputs.
This step serves as a drop-in replacement for CradleDataLoadingStep but processes 
user-provided data instead of calling internal Cradle services.
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
    from ..contracts.dummy_data_loading_contract import DUMMY_DATA_LOADING_CONTRACT


# Import the contract at runtime to avoid circular imports
def _get_dummy_data_loading_contract():
    from ..contracts.dummy_data_loading_contract import DUMMY_DATA_LOADING_CONTRACT

    return DUMMY_DATA_LOADING_CONTRACT


# Dummy Data Loading Testing Step Specification
DUMMY_DATA_LOADING_TESTING_SPEC = StepSpecification(
    step_type=get_spec_step_type_with_job_type("DummyDataLoading", "testing"),
    node_type=NodeType.INTERNAL,  # INTERNAL node with dependencies
    script_contract=_get_dummy_data_loading_contract(),  # Add reference to the script contract
    dependencies=[
        DependencySpec(
            logical_name="INPUT_DATA",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["DataUploadStep", "S3DataStep", "LocalDataStep"],
            semantic_keywords=["data", "dataset", "input", "raw_data", "testing", "test"],
            data_type="S3Uri",
            description="Testing input data to be processed (from local or S3 source)",
        )
    ],
    outputs=[
        OutputSpec(
            logical_name="DATA",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['DATA'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Testing data output from dummy data loading",
            semantic_keywords=[
                "testing",
                "test",
                "data",
                "input",
                "raw",
                "dataset",
                "model_testing",
                "source",
            ],
        ),
        OutputSpec(
            logical_name="METADATA",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['METADATA'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Testing metadata output from dummy data loading",
            semantic_keywords=[
                "testing",
                "test",
                "metadata",
                "schema",
                "info",
                "description",
                "model_testing",
            ],
        ),
        OutputSpec(
            logical_name="SIGNATURE",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['SIGNATURE'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Testing signature output from dummy data loading",
            semantic_keywords=[
                "testing",
                "test",
                "signature",
                "validation",
                "checksum",
                "model_testing",
            ],
        ),
    ],
)

"""
Dummy Data Loading Step Specification.

This module defines the declarative specification for Dummy data loading steps,
including their dependencies and outputs. This step serves as a drop-in replacement
for CradleDataLoadingStep but processes user-provided data instead of calling
internal Cradle services.
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
    from ..contracts.dummy_data_loading_contract import DUMMY_DATA_LOADING_CONTRACT


# Import the contract at runtime to avoid circular imports
def _get_dummy_data_loading_contract():
    from ..contracts.dummy_data_loading_contract import DUMMY_DATA_LOADING_CONTRACT

    return DUMMY_DATA_LOADING_CONTRACT


# Dummy Data Loading Step Specification
DUMMY_DATA_LOADING_SPEC = StepSpecification(
    step_type=get_spec_step_type("DummyDataLoading"),
    node_type=NodeType.INTERNAL,  # INTERNAL node with dependencies
    script_contract=_get_dummy_data_loading_contract(),  # Add reference to the script contract
    dependencies=[
        DependencySpec(
            logical_name="INPUT_DATA",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["DataUploadStep", "S3DataStep", "LocalDataStep"],
            semantic_keywords=[
                "data",
                "dataset",
                "input",
                "raw_data",
                "training",
                "train",
                "validation",
                "val",
                "testing",
                "test",
                "calibration",
                "calib",
            ],
            data_type="S3Uri",
            description="Input data to be processed (from local or S3 source). Supports all job types: training, validation, testing, and calibration",
        )
    ],
    outputs=[
        # Identical outputs to CradleDataLoadingStep for drop-in replacement
        OutputSpec(
            logical_name="DATA",
            aliases=["input_data", "raw_data", "dataset"],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['DATA'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Main data output from dummy data loading. Compatible with all job types (training, validation, testing, calibration)",
        ),
        OutputSpec(
            logical_name="METADATA",
            aliases=["meta", "schema", "data_metadata"],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['METADATA'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Metadata output from dummy data loading. Compatible with all job types (training, validation, testing, calibration)",
        ),
        OutputSpec(
            logical_name="SIGNATURE",
            aliases=["sig", "column_signature", "data_signature"],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['SIGNATURE'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Signature output from dummy data loading. Compatible with all job types (training, validation, testing, calibration)",
        ),
    ],
)

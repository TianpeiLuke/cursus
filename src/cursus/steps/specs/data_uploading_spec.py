"""
Data Uploading Step Specification.

This module defines the declarative specification for the DataUploading step,
which uploads data from S3 to BDT (EDX/Andes). This is a SINK node —
it has dependencies (upstream S3 data) but no outputs.
"""

from ...core.base.specification_base import (
    StepSpecification,
    DependencySpec,
    NodeType,
)
from ...core.base.enums import DependencyType
from ...registry.step_names import get_spec_step_type


def _get_data_uploading_contract():
    from ..contracts.data_uploading_contract import DATA_UPLOADING_CONTRACT

    return DATA_UPLOADING_CONTRACT


DATA_UPLOADING_SPEC = StepSpecification(
    step_type=get_spec_step_type("DataUploading"),
    node_type=NodeType.SINK,
    script_contract=_get_data_uploading_contract(),
    dependencies=[
        DependencySpec(
            logical_name="input_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=[
                "TabularPreprocessing",
                "StratifiedSampling",
                "XGBoostTraining",
                "CradleDataLoading",
                "Processing",
            ],
            semantic_keywords=["data", "processed"],
            data_type="S3Uri",
            description="S3 data to upload to BDT (EDX/Andes)",
        ),
    ],
    outputs=[
        # SINK node — no outputs. Data goes to BDT external system, not S3.
        # Confirmed: DataUploadingStep.get_output_locations() returns None.
    ],
)

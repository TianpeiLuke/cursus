"""
EDX Uploading Step Specification.

This module defines the declarative specification for the EdxUploading step,
which uploads data from S3 to EDX. This is a SINK node —
it has dependencies (upstream S3 data) but no outputs.
"""

from ...core.base.specification_base import (
    StepSpecification,
    DependencySpec,
    NodeType,
)
from ...core.base.enums import DependencyType
from ...registry.step_names import get_spec_step_type


def _get_edx_uploading_contract():
    from ..contracts.edx_uploading_contract import EDX_UPLOADING_CONTRACT

    return EDX_UPLOADING_CONTRACT


EDX_UPLOADING_SPEC = StepSpecification(
    step_type=get_spec_step_type("EdxUploading"),
    node_type=NodeType.SINK,
    script_contract=_get_edx_uploading_contract(),
    dependencies=[
        DependencySpec(
            logical_name="input_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=[
                "CradleDataLoading",
                "RedshiftDataLoading",
                "TabularPreprocessing",
                "StratifiedSampling",
                "BedrockProcessing",
                "ProcessingStep",
            ],
            semantic_keywords=["data", "processed", "output", "input"],
            data_type="S3Uri",
            description="S3 data to upload to EDX",
        ),
    ],
    outputs=[
        # SINK node — data goes to EDX external system, not S3.
    ],
)

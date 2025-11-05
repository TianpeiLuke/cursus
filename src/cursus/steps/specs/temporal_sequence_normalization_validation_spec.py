"""
Temporal Sequence Normalization Validation Step Specification.

This module defines the declarative specification for temporal sequence normalization steps
specifically for validation data, including their dependencies and outputs.
"""

from ...core.base.specification_base import (
    StepSpecification,
    DependencySpec,
    OutputSpec,
    DependencyType,
    NodeType,
)
from ...registry.step_names import get_spec_step_type_with_job_type


# Import the contract at runtime to avoid circular imports
def _get_temporal_sequence_normalization_contract():
    from ..contracts.temporal_sequence_normalization_contract import (
        TEMPORAL_SEQUENCE_NORMALIZATION_CONTRACT,
    )

    return TEMPORAL_SEQUENCE_NORMALIZATION_CONTRACT


# Temporal Sequence Normalization Validation Step Specification
TEMPORAL_SEQUENCE_NORMALIZATION_VALIDATION_SPEC = StepSpecification(
    step_type=get_spec_step_type_with_job_type(
        "TemporalSequenceNormalization", "validation"
    ),
    node_type=NodeType.INTERNAL,
    script_contract=_get_temporal_sequence_normalization_contract(),
    dependencies=[
        DependencySpec(
            logical_name="DATA",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=[
                "CradleDataLoading",
                "DummyDataLoading",
                "DataLoad",
                "ProcessingStep",
                "TabularPreprocessing",
            ],
            semantic_keywords=[
                "validation",
                "val",
                "data",
                "input",
                "raw",
                "dataset",
                "source",
                "temporal",
                "sequence",
                "time_series",
                "sequential",
                "model_validation",
            ],
            data_type="S3Uri",
            description="Raw validation temporal sequence data for normalization",
        ),
        DependencySpec(
            logical_name="SIGNATURE",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=False,
            compatible_sources=["CradleDataLoading", "DummyDataLoading"],
            semantic_keywords=[
                "signature",
                "schema",
                "columns",
                "column_names",
                "metadata",
                "header",
            ],
            data_type="S3Uri",
            description="Column signature file for CSV/TSV data preprocessing",
        ),
    ],
    outputs=[
        OutputSpec(
            logical_name="normalized_sequences",
            aliases=[
                "sequences",
                "temporal_data",
                "sequence_data",
                "normalized_data",
                "processed_sequences",
                "validation_sequences",
            ],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['normalized_sequences'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Normalized validation temporal sequences with consistent length and format",
        )
    ],
)

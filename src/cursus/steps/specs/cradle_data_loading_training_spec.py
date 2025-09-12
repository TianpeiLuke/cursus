"""
Cradle Data Loading Training Step Specification.

This module defines the declarative specification for Cradle data loading steps
specifically for training data, including their dependencies and outputs.
"""

from ...core.base.specification_base import (
    StepSpecification,
    DependencySpec,
    OutputSpec,
    DependencyType,
    NodeType,
)
from ...registry.step_names import get_spec_step_type_with_job_type

# Cradle Data Loading Training Step Specification
DATA_LOADING_TRAINING_SPEC = StepSpecification(
    step_type=get_spec_step_type_with_job_type("CradleDataLoading", "training"),
    node_type=NodeType.SOURCE,
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
            description="Training data output from Cradle data loading",
            semantic_keywords=[
                "training",
                "train",
                "data",
                "input",
                "raw",
                "dataset",
                "model_training",
                "source",
            ],
        ),
        OutputSpec(
            logical_name="METADATA",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['METADATA'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Training metadata output from Cradle data loading",
            semantic_keywords=[
                "training",
                "train",
                "metadata",
                "schema",
                "info",
                "description",
                "model_training",
            ],
        ),
        OutputSpec(
            logical_name="SIGNATURE",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['SIGNATURE'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Training signature output from Cradle data loading",
            semantic_keywords=[
                "training",
                "train",
                "signature",
                "validation",
                "checksum",
                "model_training",
            ],
        ),
    ],
)

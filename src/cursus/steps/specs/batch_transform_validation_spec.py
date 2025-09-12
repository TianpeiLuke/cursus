"""
Batch Transform Validation Step Specification.

This module defines the declarative specification for batch transform validation steps,
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

# Batch Transform Validation Step Specification
BATCH_TRANSFORM_VALIDATION_SPEC = StepSpecification(
    step_type=get_spec_step_type_with_job_type("BatchTransform", "validation"),
    node_type=NodeType.INTERNAL,
    dependencies=[
        DependencySpec(
            logical_name="model_name",
            dependency_type=DependencyType.CUSTOM_PROPERTY,
            required=True,
            compatible_sources=["PytorchModel", "XgboostModel"],
            semantic_keywords=["model", "name", "model_name", "ModelName"],
            data_type="String",
            description="SageMaker model name from a model step",
        ),
        DependencySpec(
            logical_name="processed_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["TabularPreprocessing-Validation"],
            semantic_keywords=[
                "validation",
                "data",
                "features",
                "preprocessed",
                "tabular",
                "input_data",
                "model_input_data",
            ],
            data_type="S3Uri",
            description="Processed validation data for batch transform from preprocessing step",
        ),
    ],
    outputs=[
        OutputSpec(
            logical_name="transform_output",
            output_type=DependencyType.CUSTOM_PROPERTY,
            property_path="properties.TransformOutput.S3OutputPath",
            data_type="S3Uri",
            description="S3 location of the batch transform output",
        )
    ],
)

"""
Tabular Preprocessing Validation Step Specification.

This module defines the declarative specification for tabular preprocessing steps
specifically for validation data, including their dependencies and outputs.
"""

from ...core.base.specification_base import StepSpecification, DependencySpec, OutputSpec, DependencyType, NodeType
from ..registry.step_names import get_spec_step_type

# Tabular Preprocessing Validation Step Specification
PREPROCESSING_VALIDATION_SPEC = StepSpecification(
    step_type=get_spec_step_type("TabularPreprocessing") + "_Validation",
    node_type=NodeType.INTERNAL,
    dependencies=[
        DependencySpec(
            logical_name="DATA",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["CradleDataLoading", "DataLoad", "ProcessingStep"],
            semantic_keywords=["validation", "val", "data", "input", "raw", "dataset", "source", "tabular", "model_validation", "holdout"],
            data_type="S3Uri",
            description="Raw validation data for preprocessing"
        )
    ],
    outputs=[
        OutputSpec(
            logical_name="processed_data",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Processed validation data"
        )
    ]
)

"""
Tabular Preprocessing Step Specification.

This module defines the declarative specification for tabular preprocessing steps,
including their dependencies and outputs based on the actual implementation.
"""

from ...core.base.specification_base import StepSpecification, DependencySpec, OutputSpec, DependencyType, NodeType
from ..registry.step_names import get_spec_step_type

# Tabular Preprocessing Step Specification
PREPROCESSING_SPEC = StepSpecification(
    step_type=get_spec_step_type("TabularPreprocessing") + "_Training",
    node_type=NodeType.INTERNAL,
    dependencies=[
        DependencySpec(
            logical_name="DATA",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["CradleDataLoading", "DataLoad", "ProcessingStep"],
            semantic_keywords=["data", "input", "raw", "dataset", "source", "tabular"],
            data_type="S3Uri",
            description="Raw tabular data for preprocessing"
        )
    ],
    outputs=[
        OutputSpec(
            logical_name="processed_data",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Processed tabular data with train/val/test splits"
        )
    ]
)

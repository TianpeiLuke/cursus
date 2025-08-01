"""
Tabular Preprocessing Calibration Step Specification.

This module defines the declarative specification for tabular preprocessing steps
specifically for calibration data, including their dependencies and outputs.
"""

from ...core.base.specification_base import StepSpecification, DependencySpec, OutputSpec, DependencyType, NodeType
from ..registry.step_names import get_spec_step_type

# Tabular Preprocessing Calibration Step Specification
PREPROCESSING_CALIBRATION_SPEC = StepSpecification(
    step_type=get_spec_step_type("TabularPreprocessing") + "_Calibration",
    node_type=NodeType.INTERNAL,
    dependencies=[
        DependencySpec(
            logical_name="DATA",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["CradleDataLoading", "DataLoad", "ProcessingStep"],
            semantic_keywords=["calibration", "calib", "eval", "data", "input", "raw", "dataset", "source", "tabular", "evaluation", "model_eval"],
            data_type="S3Uri",
            description="Raw calibration data for preprocessing"
        )
    ],
    outputs=[
        OutputSpec(
            logical_name="processed_data",
            aliases=["eval_data_input", "calibration_data", "validation_data"],  # Added aliases for better matching
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Processed calibration data for model evaluation"
        )
    ]
)

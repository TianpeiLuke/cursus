"""
Tabular Preprocessing Training Step Specification.

This module defines the declarative specification for tabular preprocessing steps
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


# Import the contract at runtime to avoid circular imports
def _get_tabular_preprocess_contract():
    from ..contracts.tabular_preprocessing_contract import TABULAR_PREPROCESSING_CONTRACT

    return TABULAR_PREPROCESSING_CONTRACT


# Tabular Preprocessing Training Step Specification
TABULAR_PREPROCESSING_TRAINING_SPEC = StepSpecification(
    step_type=get_spec_step_type_with_job_type("TabularPreprocessing", "training"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_tabular_preprocess_contract(),
    dependencies=[
        DependencySpec(
            logical_name="DATA",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["CradleDataLoading", "DataLoad", "ProcessingStep"],
            semantic_keywords=[
                "training",
                "train",
                "data",
                "input",
                "raw",
                "dataset",
                "source",
                "tabular",
                "model_training",
            ],
            data_type="S3Uri",
            description="Raw training data for preprocessing",
        )
    ],
    outputs=[
        OutputSpec(
            logical_name="processed_data",
            aliases=[
                "input_path",
                "training_data",
                "model_input_data",
            ],  # Added aliases for better matching
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Processed training data with train/val/test splits",
        )
    ],
)

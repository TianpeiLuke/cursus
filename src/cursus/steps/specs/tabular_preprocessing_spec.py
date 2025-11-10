"""
Tabular Preprocessing Step Specification.

This module defines the declarative specification for tabular preprocessing steps,
including their dependencies and outputs based on the actual implementation.
"""

from ...core.base.specification_base import (
    StepSpecification,
    DependencySpec,
    OutputSpec,
    DependencyType,
    NodeType,
)
from ...registry.step_names import get_spec_step_type


# Import the contract at runtime to avoid circular imports
def _get_tabular_preprocess_contract():
    from ..contracts.tabular_preprocessing_contract import (
        TABULAR_PREPROCESSING_CONTRACT,
    )

    return TABULAR_PREPROCESSING_CONTRACT


# Tabular Preprocessing Step Specification
TABULAR_PREPROCESSING_SPEC = StepSpecification(
    step_type=get_spec_step_type("TabularPreprocessing"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_tabular_preprocess_contract(),
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
            ],
            semantic_keywords=[
                "data",
                "input",
                "raw",
                "dataset",
                "source",
                "tabular",
                "training",
                "train",
                "model_training",
                "validation",
                "val",
                "model_validation",
                "holdout",
                "testing",
                "test",
                "model_testing",
                "calibration",
                "calib",
                "model_calibration",
            ],
            data_type="S3Uri",
            description="Raw tabular data for preprocessing. Supports all job types: training, validation, testing, and calibration",
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
            logical_name="processed_data",
            aliases=[
                "input_path",
                "training_data",
                "model_input_data",
                "input_data",
                "validation_data",
                "testing_data",
                "calibration_data",
                "processed_training_data",
                "processed_validation_data",
                "processed_testing_data",
                "processed_calibration_data",
            ],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Processed tabular data. Compatible with all job types (training, validation, testing, calibration)",
        )
    ],
)

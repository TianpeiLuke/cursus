"""
Temporal Split Preprocessing Step Specification.

This module defines the declarative specification for temporal split preprocessing steps,
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
def _get_temporal_split_preprocess_contract():
    from ..contracts.temporal_split_preprocessing_contract import (
        TEMPORAL_SPLIT_PREPROCESSING_CONTRACT,
    )

    return TEMPORAL_SPLIT_PREPROCESSING_CONTRACT


# Temporal Split Preprocessing Step Specification
TEMPORAL_SPLIT_PREPROCESSING_SPEC = StepSpecification(
    step_type=get_spec_step_type("TemporalSplitPreprocessing"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_temporal_split_preprocess_contract(),
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
                "temporal",
                "time_series",
                "group",
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
                "oot",
                "out_of_time",
            ],
            data_type="S3Uri",
            description="Raw tabular data with temporal and group information for temporal split preprocessing. Supports all job types: training, validation, testing, and calibration",
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
            logical_name="training_data",
            aliases=[
                "input_path",
                "training_input",
                "model_input_data",
                "train_val_data",
                "processed_training_data",
                "temporal_training_data",
            ],
            output_type=DependencyType.TRAINING_DATA,
            property_path="properties.ProcessingOutputConfig.Outputs['training_data'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Training data with temporal and group-level splits (train/val subdirectories) for lightgbmmt_training",
        ),
        OutputSpec(
            logical_name="oot_data",
            aliases=[
                "processed_data",
                "evaluation_data",
                "out_of_time_data",
                "oot_evaluation_data",
                "temporal_oot_data",
                "model_eval_data",
            ],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['oot_data'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Out-of-time evaluation data for lightgbmmt_model_eval (post-training evaluation)",
        ),
    ],
)

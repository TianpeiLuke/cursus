"""
Missing Value Imputation Calibration Step Specification.

This module defines the declarative specification for missing value imputation steps
specifically for calibration data, including their dependencies and outputs.
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
def _get_missing_value_imputation_contract():
    from ..contracts.missing_value_imputation_contract import (
        MISSING_VALUE_IMPUTATION_CONTRACT,
    )

    return MISSING_VALUE_IMPUTATION_CONTRACT


# Missing Value Imputation Calibration Step Specification
MISSING_VALUE_IMPUTATION_CALIBRATION_SPEC = StepSpecification(
    step_type=get_spec_step_type_with_job_type("MissingValueImputation", "calibration"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_missing_value_imputation_contract(),
    dependencies=[
        DependencySpec(
            logical_name="input_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=[
                "TabularPreprocessing",
                "StratifiedSampling",
                "RiskTableMapping",
                "ProcessingStep",
            ],
            semantic_keywords=[
                "calibration",
                "calib",
                "processed_data",
                "preprocessed",
                "cleaned",
                "tabular",
                "data",
                "input",
                "dataset",
                "missing_values",
                "imputation",
                "na_values",
                "model_calibration",
                "data_input",
                "output_data",
            ],
            data_type="S3Uri",
            description="Processed calibration data from preprocessing steps for missing value imputation",
        ),
        DependencySpec(
            logical_name="model_artifacts_input",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["MissingValueImputation_Training", "ProcessingStep"],
            semantic_keywords=[
                "imputation_parameters",
                "fitted_imputers",
                "imputation_artifacts",
                "imputation_model",
                "training_artifacts",
                "parameters",
                "artifacts",
                "training",
                "fitted",
                "imputation_params_input",
                "imputation_params_output",
                "model_artifacts_output",
            ],
            data_type="S3Uri",
            description="Pre-trained imputation parameters from training job",
        ),
    ],
    outputs=[
        OutputSpec(
            logical_name="processed_data",
            aliases=[
                "imputed_data",
                "input_data",
                "cleaned_data",
                "filled_data",
                "calibration_data",
                "calib_data",
                "model_input_data",
                "input_path",
                "data_input",
            ],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Calibration data with missing values imputed using pre-trained parameters",
        ),
        OutputSpec(
            logical_name="model_artifacts_output",
            aliases=[
                "imputation_params",
                "imputation_parameters",
                "fitted_imputers",
                "imputation_artifacts",
                "imputation_model",
                "training_artifacts",
                "imputation_params_input",
                "model_artifacts_input",
            ],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['model_artifacts_output'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Imputation parameters (passthrough from training)",
        ),
    ],
)

"""
Missing Value Imputation Step Specification.

This module defines the declarative specification for missing value imputation steps,
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


# Import the contract at runtime to avoid circular imports
def _get_missing_value_imputation_contract():
    from ..contracts.missing_value_imputation_contract import (
        MISSING_VALUE_IMPUTATION_CONTRACT,
    )

    return MISSING_VALUE_IMPUTATION_CONTRACT


# Missing Value Imputation Step Specification (Default)
MISSING_VALUE_IMPUTATION_SPEC = StepSpecification(
    step_type=get_spec_step_type_with_job_type("MissingValueImputation", "training"),
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
                "processed_data",
                "preprocessed",
                "cleaned",
                "tabular",
                "data",
                "input",
                "dataset",
                "training",
                "train",
                "splits",
                "missing_values",
                "imputation",
                "na_values",
            ],
            data_type="S3Uri",
            description="Processed tabular data from preprocessing steps for missing value imputation",
        ),
        # Model artifacts dependency - optional for training mode since training creates them
        DependencySpec(
            logical_name="model_artifacts_input",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=False,
            compatible_sources=["MissingValueImputation_Training", "ProcessingStep"],
            semantic_keywords=[
                "model_artifacts",
                "imputation_parameters",
                "fitted_imputers",
                "imputation_artifacts",
                "imputation_model",
                "training_artifacts",
                "model_artifacts_input",
                "imputation_params_input",
                "imputation_params_output",
                "model_artifacts_output",
            ],
            data_type="S3Uri",
            description="Optional pre-existing model artifacts (training mode creates new ones if not provided)",
        ),
    ],
    outputs=[
        OutputSpec(
            logical_name="processed_data",
            aliases=[
                "data_input",
                "input_data",
                "imputed_data",
                "cleaned_data",
                "filled_data",
                "training_data",
                "model_input_data",
                "input_path",
            ],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Data with missing values imputed using statistical methods",
        ),
        OutputSpec(
            logical_name="model_artifacts_output",
            aliases=[
                "imputation_params",
                "imputation_parameters",
                "fitted_imputers",
                "imputation_artifacts",
                "imputation_model",
                "model_artifacts",
                "model_artifacts_input",
            ],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['model_artifacts_output'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Fitted imputation parameters for inference mode",
        ),
    ],
)

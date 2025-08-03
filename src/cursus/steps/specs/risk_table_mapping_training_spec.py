"""
Risk Table Mapping Training Step Specification.

This module defines the declarative specification for risk table mapping steps
specifically for training data, including their dependencies and outputs.
"""

from ...core.base.specification_base import StepSpecification, DependencySpec, OutputSpec, DependencyType, NodeType
from ..registry.step_names import get_spec_step_type

# Import the contract at runtime to avoid circular imports
def _get_risk_table_mapping_contract():
    from ..contracts.risk_table_mapping_contract import RISK_TABLE_MAPPING_CONTRACT
    return RISK_TABLE_MAPPING_CONTRACT

# Risk Table Mapping Training Step Specification
RISK_TABLE_MAPPING_TRAINING_SPEC = StepSpecification(
    step_type=get_spec_step_type("RiskTableMapping") + "_Training",
    node_type=NodeType.INTERNAL,
    script_contract=_get_risk_table_mapping_contract(),
    dependencies=[
        DependencySpec(
            logical_name="data_input",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["TabularPreprocessing", "ProcessingStep"],
            semantic_keywords=["training", "train", "data", "input", "preprocessed", "tabular"],
            data_type="S3Uri",
            description="Preprocessed training data from tabular preprocessing step"
        ),
        # Hyperparameters are optional as they can be generated internally,
        # but we still support external hyperparameters being provided
        DependencySpec(
            logical_name="hyperparameters_s3_uri",
            dependency_type=DependencyType.HYPERPARAMETERS,
            required=False,
            compatible_sources=[
                "HyperparameterPrep", "ProcessingStep", "ConfigurationStep",
                "DataPrep", "ModelTraining", "FeatureEngineering", "DataQuality"
            ],
            semantic_keywords=["config", "params", "hyperparameters", "settings", "hyperparams"],
            data_type="S3Uri",
            description="Optional external hyperparameters configuration file (will be overridden by internal generation)"
        ),
    ],
    outputs=[
        OutputSpec(
            logical_name="processed_data",
            aliases=["training_data", "model_input_data"],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Processed data with risk table mappings applied"
        ),
        OutputSpec(
            logical_name="risk_tables",
            aliases=["bin_mapping", "risk_table_artifacts", "categorical_mappings"],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['risk_tables'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Risk tables and imputation models for categorical features"
        )
    ]
)

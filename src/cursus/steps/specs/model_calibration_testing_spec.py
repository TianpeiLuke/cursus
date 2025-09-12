#!/usr/bin/env python
"""
Model Calibration Testing Step Specification.

This module defines the declarative specification for model calibration steps
specifically for testing data, including dependencies and outputs.
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
def _get_model_calibration_contract():
    from ..contracts.model_calibration_contract import MODEL_CALIBRATION_CONTRACT

    return MODEL_CALIBRATION_CONTRACT


# Model Calibration Testing Step Specification
MODEL_CALIBRATION_TESTING_SPEC = StepSpecification(
    step_type=get_spec_step_type_with_job_type("ModelCalibration", "testing"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_model_calibration_contract(),
    dependencies={
        "evaluation_data": DependencySpec(
            logical_name="evaluation_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=[
                "PytorchTraining",
                "XGBoostTraining",
                "XGBoostModelEval",
                "ModelEvaluation",
                "TrainingEvaluation",
                "CrossValidation",
            ],
            semantic_keywords=[
                "testing",
                "test",
                "evaluate",
                "evaluation",
                "predictions",
                "scores",
                "results",
                "model_output",
                "performance",
                "testing_evaluation",
                "test_predictions",
                "testing_results",
            ],
            data_type="S3Uri",
            description="Testing evaluation dataset with ground truth labels and model predictions",
        )
    },
    outputs={
        "calibration_output": OutputSpec(
            logical_name="calibration_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['calibration_output'].S3Output.S3Uri",
            aliases=[
                "calibration_model",
                "testing_calibration",
                "test_calibrator",
                "testing_probability_calibration",
                "testing_calibration_artifacts",
            ],
            data_type="S3Uri",
            description="Testing calibration mapping and artifacts",
        ),
        "metrics_output": OutputSpec(
            logical_name="metrics_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['metrics_output'].S3Output.S3Uri",
            aliases=[
                "testing_calibration_metrics",
                "testing_reliability_metrics",
                "testing_probability_metrics",
                "testing_calibration_performance",
                "testing_calibration_evaluation",
            ],
            data_type="S3Uri",
            description="Testing calibration quality metrics and visualizations",
        ),
        "calibrated_data": OutputSpec(
            logical_name="calibrated_data",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['calibrated_data'].S3Output.S3Uri",
            aliases=[
                "testing_calibrated_predictions",
                "testing_calibrated_probabilities",
                "testing_probability_scores",
                "testing_adjusted_predictions",
                "testing_calibrated_outputs",
            ],
            data_type="S3Uri",
            description="Testing dataset with calibrated probabilities",
        ),
    },
)

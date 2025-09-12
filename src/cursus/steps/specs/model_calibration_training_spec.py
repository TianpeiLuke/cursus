#!/usr/bin/env python
"""
Model Calibration Training Step Specification.

This module defines the declarative specification for model calibration steps
specifically for training data, including dependencies and outputs.
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


# Model Calibration Training Step Specification
MODEL_CALIBRATION_TRAINING_SPEC = StepSpecification(
    step_type=get_spec_step_type_with_job_type("ModelCalibration", "training"),
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
                "training",
                "train",
                "evaluation",
                "predictions",
                "scores",
                "results",
                "model_output",
                "performance",
                "training_evaluation",
                "train_predictions",
                "training_results",
            ],
            data_type="S3Uri",
            description="Training evaluation dataset with ground truth labels and model predictions",
        )
    },
    outputs={
        "calibration_output": OutputSpec(
            logical_name="calibration_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['calibration_output'].S3Output.S3Uri",
            aliases=[
                "calibration_model",
                "training_calibration",
                "train_calibrator",
                "training_probability_calibration",
                "training_calibration_artifacts",
            ],
            data_type="S3Uri",
            description="Training calibration mapping and artifacts",
        ),
        "metrics_output": OutputSpec(
            logical_name="metrics_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['metrics_output'].S3Output.S3Uri",
            aliases=[
                "training_calibration_metrics",
                "training_reliability_metrics",
                "training_probability_metrics",
                "training_calibration_performance",
                "training_calibration_evaluation",
            ],
            data_type="S3Uri",
            description="Training calibration quality metrics and visualizations",
        ),
        "calibrated_data": OutputSpec(
            logical_name="calibrated_data",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['calibrated_data'].S3Output.S3Uri",
            aliases=[
                "training_calibrated_predictions",
                "training_calibrated_probabilities",
                "training_probability_scores",
                "training_adjusted_predictions",
                "training_calibrated_outputs",
            ],
            data_type="S3Uri",
            description="Training dataset with calibrated probabilities",
        ),
    },
)

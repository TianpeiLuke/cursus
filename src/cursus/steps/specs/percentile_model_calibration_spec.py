#!/usr/bin/env python
"""Step Specification for Percentile Model Calibration Step.

This file defines the step specification for the percentile model calibration processing step,
including dependencies, outputs, and other metadata needed for pipeline integration.
"""

from ...core.base.specification_base import (
    StepSpecification,
    NodeType,
    DependencySpec,
    OutputSpec,
    DependencyType,
)
from ...registry.step_names import get_spec_step_type


def _get_percentile_model_calibration_contract():
    """Get the script contract for the PercentileModelCalibration step.

    Returns:
        ScriptContract: The contract defining input/output paths and environment variables.
    """
    from ..contracts.percentile_model_calibration_contract import (
        PERCENTILE_MODEL_CALIBRATION_CONTRACT,
    )

    return PERCENTILE_MODEL_CALIBRATION_CONTRACT


PERCENTILE_MODEL_CALIBRATION_SPEC = StepSpecification(
    step_type=get_spec_step_type("PercentileModelCalibration"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_percentile_model_calibration_contract(),
    dependencies={
        "evaluation_data": DependencySpec(
            logical_name="evaluation_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=[
                "XGBoostTraining",
                "XGBoostModelEval",
                "XGBoostModelInference",
                "LightGBMTraining",
                "LightGBMModelEval",
                "LightGBMModelInference",
                "LightGBMMTTraining",
                "LightGBMMTModelEval",
                "PyTorchTraining",
                "PyTorchModelEval",
                "PyTorchModelInference",
                "ModelEvaluation",
                "TrainingEvaluation",
                "CrossValidation",
                "ModelCalibration",
            ],
            semantic_keywords=[
                "evaluation",
                "predictions",
                "scores",
                "validation",
                "test",
                "results",
                "model_output",
                "performance",
                "inference",
                "output_data",
                "prediction_results",
                "calibrated_data",
            ],
            data_type="S3Uri",
            description="Evaluation dataset with ground truth labels and model predictions",
        ),
        "calibration_config": DependencySpec(
            logical_name="calibration_config",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=False,
            compatible_sources=[
                "ConfigurationStep",
                "DataPreprocessing",
                "FeatureEngineering",
                "ModelConfiguration",
            ],
            semantic_keywords=[
                "configuration",
                "config",
                "calibration",
                "parameters",
                "settings",
                "dictionary",
                "mapping",
                "thresholds",
                "percentiles",
            ],
            data_type="S3Uri",
            description="Optional calibration configuration directory containing standard_calibration_dictionary.json",
        ),
    },
    outputs={
        "calibration_output": OutputSpec(
            logical_name="calibration_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['calibration_output'].S3Output.S3Uri",
            aliases=[
                "percentile_mapping",
                "score_percentiles",
                "percentile_calibration",
                "risk_percentiles",
                "score_ranking",
                "percentile_scores",
                "calibration_mapping",
                "score_calibration",
            ],
            data_type="S3Uri",
            description="Percentile score mapping (percentile_score.pkl)",
        ),
        "metrics_output": OutputSpec(
            logical_name="metrics_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['metrics_output'].S3Output.S3Uri",
            aliases=[
                "percentile_metrics",
                "calibration_metrics",
                "percentile_performance",
                "score_distribution_metrics",
                "percentile_evaluation",
                "ranking_metrics",
            ],
            data_type="S3Uri",
            description="Percentile calibration quality metrics and performance statistics",
        ),
        "calibrated_data": OutputSpec(
            logical_name="calibrated_data",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['calibrated_data'].S3Output.S3Uri",
            aliases=[
                "percentile_data",
                "scored_percentiles",
                "percentile_predictions",
                "ranked_data",
                "percentile_scores_data",
                "calibrated_percentiles",
            ],
            data_type="S3Uri",
            description="Dataset with percentile scores and calibrated predictions",
        ),
    },
)

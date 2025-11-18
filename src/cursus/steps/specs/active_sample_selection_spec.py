"""
Active Sample Selection Step Specification.

This module defines the declarative specification for Active Sample Selection steps,
which intelligently select high-value samples from model predictions for
Semi-Supervised Learning or Active Learning workflows.
"""

from ...core.base.specification_base import (
    StepSpecification,
    DependencySpec,
    OutputSpec,
    DependencyType,
    NodeType,
)
from ...registry.step_names import get_spec_step_type
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..contracts.active_sample_selection_contract import (
        ACTIVE_SAMPLE_SELECTION_CONTRACT,
    )


# Import the contract at runtime to avoid circular imports
def _get_active_sample_selection_contract():
    from ..contracts.active_sample_selection_contract import (
        ACTIVE_SAMPLE_SELECTION_CONTRACT,
    )

    return ACTIVE_SAMPLE_SELECTION_CONTRACT


# Active Sample Selection Step Specification
ACTIVE_SAMPLE_SELECTION_SPEC = StepSpecification(
    step_type=get_spec_step_type("ActiveSampleSelection"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_active_sample_selection_contract(),
    dependencies={
        "evaluation_data": DependencySpec(
            logical_name="evaluation_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=[
                # Model inference outputs
                "XGBoostModelInference",
                "LightGBMModelInference",
                "PyTorchModelInference",
                # Model evaluation outputs (includes predictions)
                "XGBoostModelEval",
                "LightGBMModelEval",
                "PyTorchModelEval",
                # Bedrock/LLM outputs (with probability extraction)
                "BedrockBatchProcessing",
                "BedrockProcessing",
                # Label ruleset execution (classification outputs)
                "LabelRulesetExecution",
            ],
            semantic_keywords=[
                "evaluation",
                "predictions",
                "inference",
                "model_predictions",
                "inference_results",
                "eval_output",
                "prediction_data",
                "processed_data",
                "classification_output",
                "probabilities",
            ],
            data_type="S3Uri",
            description="Model predictions with probability columns for sample selection",
        ),
    },
    outputs={
        "selected_samples": OutputSpec(
            logical_name="selected_samples",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['selected_samples'].S3Output.S3Uri",
            data_type="S3Uri",
            aliases=[
                "active_samples",
                "high_confidence_samples",
                "pseudo_labeled_samples",
                "uncertain_samples",
                "selection_output",
                # Training data aliases for XGBoost/LightGBM/PyTorch Training compatibility
                "input_path",  # Exact match for XGBoost/LightGBM/PyTorch Training input
                "input_data",
                "processed_data",
                "eval_output",
                "augmentation_data",
            ],
            description="Selected samples with confidence scores and metadata",
        ),
        "selection_metadata": OutputSpec(
            logical_name="selection_metadata",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['selection_metadata'].S3Output.S3Uri",
            data_type="S3Uri",
            aliases=[
                "metadata",
                "selection_info",
                "sampling_metadata",
            ],
            description="Selection metadata including strategy config, counts, and timestamp",
        ),
    },
)

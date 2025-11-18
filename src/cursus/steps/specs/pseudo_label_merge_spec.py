"""
Pseudo Label Merge Step Specification.

This module defines the declarative specification for Pseudo Label Merge steps,
which intelligently merge labeled base data with pseudo-labeled or augmented samples
for Semi-Supervised Learning (SSL) and Active Learning workflows.
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
    from ..contracts.pseudo_label_merge_contract import PSEUDO_LABEL_MERGE_CONTRACT


# Import the contract at runtime to avoid circular imports
def _get_pseudo_label_merge_contract():
    from ..contracts.pseudo_label_merge_contract import PSEUDO_LABEL_MERGE_CONTRACT

    return PSEUDO_LABEL_MERGE_CONTRACT


# Pseudo Label Merge Step Specification
PSEUDO_LABEL_MERGE_SPEC = StepSpecification(
    step_type=get_spec_step_type("PseudoLabelMerge"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_pseudo_label_merge_contract(),
    dependencies={
        "base_data": DependencySpec(
            logical_name="base_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=[
                # Data preprocessing steps
                "TabularPreprocessing",
                # Risk and feature engineering steps
                "RiskTableMapping",
                "MissingValueImputation",
                "FeatureSelection",
                "StratifiedSampling",
                # Temporal processing steps
                "TemporalSequenceNormalization",
                "TemporalFeatureEngineering",
                # Label generation steps
                "LabelRulesetExecution",
            ],
            semantic_keywords=[
                "base_data",
                "labeled_data",
                "training_data",
                "processed_data",
                "preprocessed_data",
                "output",
                "data",
                "input_path",
                "train_data",
                "features",
                "processed_features",
            ],
            data_type="S3Uri",
            description="Base labeled training data (original dataset) with optional train/test/val split structure",
        ),
        "augmentation_data": DependencySpec(
            logical_name="augmentation_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=[
                # Active sample selection (primary SSL/AL source)
                "ActiveSampleSelection",
                # Model inference outputs (pseudo-labels)
                "XGBoostModelInference",
                "LightGBMModelInference",
                "PyTorchModelInference",
                # Model evaluation outputs (includes predictions)
                "XGBoostModelEval",
                "LightGBMModelEval",
                "PyTorchModelEval",
                # Bedrock/LLM outputs (for text classification pseudo-labels)
                "BedrockBatchProcessing",
                "BedrockProcessing",
                # Label ruleset execution (rule-based pseudo-labels)
                "LabelRulesetExecution",
            ],
            semantic_keywords=[
                "augmentation_data",
                "pseudo_labeled_data",
                "selected_samples",
                "active_samples",
                "predictions",
                "inference",
                "inference_results",
                "eval_output",
                "processed_data",
                "output",
                "high_confidence_samples",
                "unlabeled_predictions",
                "model_predictions",
            ],
            data_type="S3Uri",
            description="Augmentation data (pseudo-labeled samples, actively selected samples, or generated labels) to merge with base data",
        ),
    },
    outputs={
        "merged_data": OutputSpec(
            logical_name="merged_data",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['merged_data'].S3Output.S3Uri",
            data_type="S3Uri",
            aliases=[
                # Training data aliases for model training compatibility
                "training_data",
                "input_path",
                "input_data",
                "processed_data",
                "data",
                # SSL/AL specific aliases
                "combined_data",
                "augmented_training_data",
                "merged_training_data",
                "ssl_training_data",
                "enriched_data",
                # Generic output aliases
                "output",
                "processed_output",
            ],
            description="Merged dataset combining base labeled data and augmentation data with provenance tracking",
        ),
    },
)

"""
Tokenizer Training Step Specification.

This module defines the declarative specification for tokenizer training steps,
including their dependencies and outputs for BPE tokenizer training on customer name data.
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
def _get_tokenizer_training_contract():
    from ..contracts.tokenizer_training_contract import TOKENIZER_TRAINING_CONTRACT

    return TOKENIZER_TRAINING_CONTRACT


# Tokenizer Training Step Specification
TOKENIZER_TRAINING_SPEC = StepSpecification(
    step_type=get_spec_step_type("TokenizerTraining"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_tokenizer_training_contract(),
    dependencies=[
        DependencySpec(
            logical_name="input_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=[
                "TabularPreprocessing",
                "ProcessingStep",
            ],
            semantic_keywords=[
                "input_data",
                "training_data",
                "train",
                "processed_data",
                "text_data",
                "tokenizer_input",
                "training",
                "model_training",
                "preprocessed",
            ],
            data_type="S3Uri",
            description="Processed training data containing text field for tokenizer training. Expects parquet format with text column specified by TEXT_FIELD environment variable",
        ),
    ],
    outputs=[
        OutputSpec(
            logical_name="model_artifacts_output",
            aliases=[
                "tokenizer_output",
                "tokenizer_artifacts",
                "tokenizer",
                "bpe_tokenizer",
                "trained_tokenizer",
                "tokenizer_model",
                "model_artifacts_input",
            ],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['model_artifacts_output'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Trained BPE tokenizer artifacts including tokenizer.json, vocab.json, and tokenizer_metadata.json",
        )
    ],
)

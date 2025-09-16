"""
Shared enums for the cursus core base classes.

This module contains enums that are used across multiple base classes
to avoid circular imports and provide a single source of truth.
"""

from enum import Enum


class DependencyType(Enum):
    """Types of dependencies in the pipeline."""

    MODEL_ARTIFACTS = "model_artifacts"
    PROCESSING_OUTPUT = "processing_output"
    TRAINING_DATA = "training_data"
    HYPERPARAMETERS = "hyperparameters"
    PAYLOAD_SAMPLES = "payload_samples"
    CUSTOM_PROPERTY = "custom_property"

    def __eq__(self, other: object) -> bool:
        """Compare enum instances by value."""
        if isinstance(other, DependencyType):
            return self.value == other.value
        return super().__eq__(other)

    def __hash__(self) -> int:
        """Ensure hashability is maintained when used as dictionary keys."""
        return hash(self.value)


class NodeType(Enum):
    """Types of nodes in the pipeline based on their dependency/output characteristics."""

    SOURCE = "source"  # No dependencies, has outputs (e.g., data loading)
    INTERNAL = (
        "internal"  # Has both dependencies and outputs (e.g., processing, training)
    )
    SINK = "sink"  # Has dependencies, no outputs (e.g., model registration)
    SINGULAR = "singular"  # No dependencies, no outputs (e.g., standalone operations)

    def __eq__(self, other: object) -> bool:
        """Compare enum instances by value."""
        if isinstance(other, NodeType):
            return self.value == other.value
        return super().__eq__(other)

    def __hash__(self) -> int:
        """Ensure hashability is maintained when used as dictionary keys."""
        return hash(self.value)

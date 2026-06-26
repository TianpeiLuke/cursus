import pytest
from typing import Dict, Set
import logging

from cursus.core.base.enums import DependencyType, NodeType


class TestDependencyType:
    """Test cases for DependencyType enum."""

    def test_enum_values(self):
        """Test that all expected enum values exist."""
        expected_values = {
            "model_artifacts",
            "processing_output",
            "training_data",
            "hyperparameters",
            "payload_samples",
            "custom_property",
        }

        actual_values = {dep_type.value for dep_type in DependencyType}
        assert actual_values == expected_values

    def test_enum_members(self):
        """Test that all expected enum members exist."""
        expected_members = {
            "MODEL_ARTIFACTS",
            "PROCESSING_OUTPUT",
            "TRAINING_DATA",
            "HYPERPARAMETERS",
            "PAYLOAD_SAMPLES",
            "CUSTOM_PROPERTY",
        }

        actual_members = {dep_type.name for dep_type in DependencyType}
        assert actual_members == expected_members

    def test_enum_access_by_name(self):
        """Test accessing enum members by name."""
        assert DependencyType.MODEL_ARTIFACTS.value == "model_artifacts"
        assert DependencyType.PROCESSING_OUTPUT.value == "processing_output"
        assert DependencyType.TRAINING_DATA.value == "training_data"
        assert DependencyType.HYPERPARAMETERS.value == "hyperparameters"
        assert DependencyType.PAYLOAD_SAMPLES.value == "payload_samples"
        assert DependencyType.CUSTOM_PROPERTY.value == "custom_property"

    def test_enum_access_by_value(self):
        """Test accessing enum members by value."""
        assert DependencyType("model_artifacts") == DependencyType.MODEL_ARTIFACTS
        assert DependencyType("processing_output") == DependencyType.PROCESSING_OUTPUT
        assert DependencyType("training_data") == DependencyType.TRAINING_DATA
        assert DependencyType("hyperparameters") == DependencyType.HYPERPARAMETERS
        assert DependencyType("payload_samples") == DependencyType.PAYLOAD_SAMPLES
        assert DependencyType("custom_property") == DependencyType.CUSTOM_PROPERTY

    def test_equality_with_same_enum(self):
        """Test equality comparison between same enum instances."""
        dep1 = DependencyType.MODEL_ARTIFACTS
        dep2 = DependencyType.MODEL_ARTIFACTS

        assert dep1 == dep2
        assert dep1 == dep2

    def test_equality_with_different_enum(self):
        """Test equality comparison between different enum instances."""
        dep1 = DependencyType.MODEL_ARTIFACTS
        dep2 = DependencyType.TRAINING_DATA

        assert dep1 != dep2
        assert not (dep1 == dep2)

    def test_equality_with_non_enum(self):
        """Test equality comparison with non-enum values."""
        dep = DependencyType.MODEL_ARTIFACTS

        # Should not be equal to string value
        assert dep != "model_artifacts"

        # Should not be equal to other types
        assert dep != 1
        assert dep != None
        assert dep != []

    def test_hashability(self):
        """Test that enum instances are hashable."""
        dep1 = DependencyType.MODEL_ARTIFACTS
        dep2 = DependencyType.TRAINING_DATA
        dep3 = DependencyType.MODEL_ARTIFACTS

        # Should be able to use as dictionary keys
        dep_dict = {dep1: "artifacts", dep2: "data"}

        assert dep_dict[dep1] == "artifacts"
        assert dep_dict[dep2] == "data"
        assert dep_dict[dep3] == "artifacts"  # Same as dep1

    def test_hashability_in_set(self):
        """Test that enum instances work correctly in sets."""
        dep_set = {
            DependencyType.MODEL_ARTIFACTS,
            DependencyType.TRAINING_DATA,
            DependencyType.MODEL_ARTIFACTS,  # Duplicate
        }

        # Set should contain only unique values
        assert len(dep_set) == 2
        assert DependencyType.MODEL_ARTIFACTS in dep_set
        assert DependencyType.TRAINING_DATA in dep_set

    def test_hash_consistency(self):
        """Test that hash values are consistent."""
        dep1 = DependencyType.MODEL_ARTIFACTS
        dep2 = DependencyType.MODEL_ARTIFACTS

        # Same enum instances should have same hash
        assert hash(dep1) == hash(dep2)

        # Hash should be based on value
        assert hash(dep1) == hash(dep1.value)

    def test_string_representation(self):
        """Test string representation of enum."""
        dep = DependencyType.MODEL_ARTIFACTS

        # Should contain enum name and value
        str_repr = str(dep)
        assert "DependencyType" in str_repr
        assert "MODEL_ARTIFACTS" in str_repr

    def test_iteration(self):
        """Test iterating over enum members."""
        all_types = list(DependencyType)

        assert len(all_types) == 6
        assert DependencyType.MODEL_ARTIFACTS in all_types
        assert DependencyType.PROCESSING_OUTPUT in all_types
        assert DependencyType.TRAINING_DATA in all_types
        assert DependencyType.HYPERPARAMETERS in all_types
        assert DependencyType.PAYLOAD_SAMPLES in all_types
        assert DependencyType.CUSTOM_PROPERTY in all_types


class TestNodeType:
    """Test cases for NodeType enum."""

    def test_enum_values(self):
        """Test that all expected enum values exist."""
        expected_values = {"source", "internal", "sink", "singular"}

        actual_values = {node_type.value for node_type in NodeType}
        assert actual_values == expected_values

    def test_enum_members(self):
        """Test that all expected enum members exist."""
        expected_members = {"SOURCE", "INTERNAL", "SINK", "SINGULAR"}

        actual_members = {node_type.name for node_type in NodeType}
        assert actual_members == expected_members

    def test_enum_access_by_name(self):
        """Test accessing enum members by name."""
        assert NodeType.SOURCE.value == "source"
        assert NodeType.INTERNAL.value == "internal"
        assert NodeType.SINK.value == "sink"
        assert NodeType.SINGULAR.value == "singular"

    def test_enum_access_by_value(self):
        """Test accessing enum members by value."""
        assert NodeType("source") == NodeType.SOURCE
        assert NodeType("internal") == NodeType.INTERNAL
        assert NodeType("sink") == NodeType.SINK
        assert NodeType("singular") == NodeType.SINGULAR

    def test_equality_with_same_enum(self):
        """Test equality comparison between same enum instances."""
        node1 = NodeType.SOURCE
        node2 = NodeType.SOURCE

        assert node1 == node2
        assert node1 == node2

    def test_equality_with_different_enum(self):
        """Test equality comparison between different enum instances."""
        node1 = NodeType.SOURCE
        node2 = NodeType.SINK

        assert node1 != node2
        assert not (node1 == node2)

    def test_equality_with_non_enum(self):
        """Test equality comparison with non-enum values."""
        node = NodeType.SOURCE

        # Should not be equal to string value
        assert node != "source"

        # Should not be equal to other types
        assert node != 1
        assert node != None
        assert node != []

    def test_hashability(self):
        """Test that enum instances are hashable."""
        node1 = NodeType.SOURCE
        node2 = NodeType.SINK
        node3 = NodeType.SOURCE

        # Should be able to use as dictionary keys
        node_dict = {node1: "source_node", node2: "sink_node"}

        assert node_dict[node1] == "source_node"
        assert node_dict[node2] == "sink_node"
        assert node_dict[node3] == "source_node"  # Same as node1

    def test_hashability_in_set(self):
        """Test that enum instances work correctly in sets."""
        node_set = {NodeType.SOURCE, NodeType.SINK, NodeType.SOURCE}  # Duplicate

        # Set should contain only unique values
        assert len(node_set) == 2
        assert NodeType.SOURCE in node_set
        assert NodeType.SINK in node_set

    def test_hash_consistency(self):
        """Test that hash values are consistent."""
        node1 = NodeType.SOURCE
        node2 = NodeType.SOURCE

        # Same enum instances should have same hash
        assert hash(node1) == hash(node2)

        # Hash should be based on value
        assert hash(node1) == hash(node1.value)

    def test_string_representation(self):
        """Test string representation of enum."""
        node = NodeType.SOURCE

        # Should contain enum name and value
        str_repr = str(node)
        assert "NodeType" in str_repr
        assert "SOURCE" in str_repr

    def test_iteration(self):
        """Test iterating over enum members."""
        all_types = list(NodeType)

        assert len(all_types) == 4
        assert NodeType.SOURCE in all_types
        assert NodeType.INTERNAL in all_types
        assert NodeType.SINK in all_types
        assert NodeType.SINGULAR in all_types

    def test_node_type_semantics(self):
        """Test the semantic meaning of node types."""
        # SOURCE: No dependencies, has outputs
        source = NodeType.SOURCE
        assert source.value == "source"

        # INTERNAL: Has both dependencies and outputs
        internal = NodeType.INTERNAL
        assert internal.value == "internal"

        # SINK: Has dependencies, no outputs
        sink = NodeType.SINK
        assert sink.value == "sink"

        # SINGULAR: No dependencies, no outputs
        singular = NodeType.SINGULAR
        assert singular.value == "singular"


class TestEnumInteraction:
    """Test cases for interactions between different enums."""

    def test_different_enums_not_equal(self):
        """Test that different enum types are not equal even with same values."""
        # This would only apply if we had enums with overlapping values
        # Currently our enums don't have overlapping values, but test the principle
        dep = DependencyType.CUSTOM_PROPERTY
        node = NodeType.SOURCE

        assert dep != node
        assert not (dep == node)

    def test_mixed_enum_dictionary(self):
        """Test using different enum types as dictionary keys."""
        mixed_dict = {
            DependencyType.MODEL_ARTIFACTS: "dependency",
            NodeType.SOURCE: "node",
        }

        assert len(mixed_dict) == 2
        assert mixed_dict[DependencyType.MODEL_ARTIFACTS] == "dependency"
        assert mixed_dict[NodeType.SOURCE] == "node"

    def test_mixed_enum_set(self):
        """Test using different enum types in the same set."""
        mixed_set = {
            DependencyType.MODEL_ARTIFACTS,
            NodeType.SOURCE,
            DependencyType.TRAINING_DATA,
            NodeType.SINK,
        }

        assert len(mixed_set) == 4
        assert DependencyType.MODEL_ARTIFACTS in mixed_set
        assert NodeType.SOURCE in mixed_set
        assert DependencyType.TRAINING_DATA in mixed_set
        assert NodeType.SINK in mixed_set

    def test_enum_type_checking(self):
        """Test type checking with enums."""
        dep = DependencyType.MODEL_ARTIFACTS
        node = NodeType.SOURCE

        assert isinstance(dep, DependencyType)
        assert isinstance(node, NodeType)

        assert not isinstance(dep, NodeType)
        assert not isinstance(node, DependencyType)


class TestEnumEdgeCases:
    """Test edge cases and error conditions."""

    def test_invalid_enum_value_dependency_type(self):
        """Test creating enum with invalid value."""
        with pytest.raises(ValueError):
            DependencyType("invalid_value")

    def test_invalid_enum_value_node_type(self):
        """Test creating enum with invalid value."""
        with pytest.raises(ValueError):
            NodeType("invalid_value")

    def test_enum_comparison_with_none(self):
        """Test enum comparison with None."""
        dep = DependencyType.MODEL_ARTIFACTS
        node = NodeType.SOURCE

        assert dep != None
        assert node != None
        assert not (dep == None)
        assert not (node == None)

    def test_enum_boolean_context(self):
        """Test enum instances in boolean context."""
        dep = DependencyType.MODEL_ARTIFACTS
        node = NodeType.SOURCE

        # Enum instances should be truthy
        assert dep
        assert node
        assert bool(dep)
        assert bool(node)

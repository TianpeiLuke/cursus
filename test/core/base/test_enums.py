import unittest
from typing import Dict, Set
import logging

# Add the project root to the Python path to allow for absolute imports
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.cursus.core.base.enums import DependencyType, NodeType


class TestDependencyType(unittest.TestCase):
    """Test cases for DependencyType enum."""
    
    def test_enum_values(self):
        """Test that all expected enum values exist."""
        expected_values = {
            "model_artifacts",
            "processing_output", 
            "training_data",
            "hyperparameters",
            "payload_samples",
            "custom_property"
        }
        
        actual_values = {dep_type.value for dep_type in DependencyType}
        self.assertEqual(actual_values, expected_values)
    
    def test_enum_members(self):
        """Test that all expected enum members exist."""
        expected_members = {
            "MODEL_ARTIFACTS",
            "PROCESSING_OUTPUT",
            "TRAINING_DATA", 
            "HYPERPARAMETERS",
            "PAYLOAD_SAMPLES",
            "CUSTOM_PROPERTY"
        }
        
        actual_members = {dep_type.name for dep_type in DependencyType}
        self.assertEqual(actual_members, expected_members)
    
    def test_enum_access_by_name(self):
        """Test accessing enum members by name."""
        self.assertEqual(DependencyType.MODEL_ARTIFACTS.value, "model_artifacts")
        self.assertEqual(DependencyType.PROCESSING_OUTPUT.value, "processing_output")
        self.assertEqual(DependencyType.TRAINING_DATA.value, "training_data")
        self.assertEqual(DependencyType.HYPERPARAMETERS.value, "hyperparameters")
        self.assertEqual(DependencyType.PAYLOAD_SAMPLES.value, "payload_samples")
        self.assertEqual(DependencyType.CUSTOM_PROPERTY.value, "custom_property")
    
    def test_enum_access_by_value(self):
        """Test accessing enum members by value."""
        self.assertEqual(DependencyType("model_artifacts"), DependencyType.MODEL_ARTIFACTS)
        self.assertEqual(DependencyType("processing_output"), DependencyType.PROCESSING_OUTPUT)
        self.assertEqual(DependencyType("training_data"), DependencyType.TRAINING_DATA)
        self.assertEqual(DependencyType("hyperparameters"), DependencyType.HYPERPARAMETERS)
        self.assertEqual(DependencyType("payload_samples"), DependencyType.PAYLOAD_SAMPLES)
        self.assertEqual(DependencyType("custom_property"), DependencyType.CUSTOM_PROPERTY)
    
    def test_equality_with_same_enum(self):
        """Test equality comparison between same enum instances."""
        dep1 = DependencyType.MODEL_ARTIFACTS
        dep2 = DependencyType.MODEL_ARTIFACTS
        
        self.assertEqual(dep1, dep2)
        self.assertTrue(dep1 == dep2)
    
    def test_equality_with_different_enum(self):
        """Test equality comparison between different enum instances."""
        dep1 = DependencyType.MODEL_ARTIFACTS
        dep2 = DependencyType.TRAINING_DATA
        
        self.assertNotEqual(dep1, dep2)
        self.assertFalse(dep1 == dep2)
    
    def test_equality_with_non_enum(self):
        """Test equality comparison with non-enum values."""
        dep = DependencyType.MODEL_ARTIFACTS
        
        # Should not be equal to string value
        self.assertNotEqual(dep, "model_artifacts")
        
        # Should not be equal to other types
        self.assertNotEqual(dep, 1)
        self.assertNotEqual(dep, None)
        self.assertNotEqual(dep, [])
    
    def test_hashability(self):
        """Test that enum instances are hashable."""
        dep1 = DependencyType.MODEL_ARTIFACTS
        dep2 = DependencyType.TRAINING_DATA
        dep3 = DependencyType.MODEL_ARTIFACTS
        
        # Should be able to use as dictionary keys
        dep_dict = {
            dep1: "artifacts",
            dep2: "data"
        }
        
        self.assertEqual(dep_dict[dep1], "artifacts")
        self.assertEqual(dep_dict[dep2], "data")
        self.assertEqual(dep_dict[dep3], "artifacts")  # Same as dep1
    
    def test_hashability_in_set(self):
        """Test that enum instances work correctly in sets."""
        dep_set = {
            DependencyType.MODEL_ARTIFACTS,
            DependencyType.TRAINING_DATA,
            DependencyType.MODEL_ARTIFACTS  # Duplicate
        }
        
        # Set should contain only unique values
        self.assertEqual(len(dep_set), 2)
        self.assertIn(DependencyType.MODEL_ARTIFACTS, dep_set)
        self.assertIn(DependencyType.TRAINING_DATA, dep_set)
    
    def test_hash_consistency(self):
        """Test that hash values are consistent."""
        dep1 = DependencyType.MODEL_ARTIFACTS
        dep2 = DependencyType.MODEL_ARTIFACTS
        
        # Same enum instances should have same hash
        self.assertEqual(hash(dep1), hash(dep2))
        
        # Hash should be based on value
        self.assertEqual(hash(dep1), hash(dep1.value))
    
    def test_string_representation(self):
        """Test string representation of enum."""
        dep = DependencyType.MODEL_ARTIFACTS
        
        # Should contain enum name and value
        str_repr = str(dep)
        self.assertIn("DependencyType", str_repr)
        self.assertIn("MODEL_ARTIFACTS", str_repr)
    
    def test_iteration(self):
        """Test iterating over enum members."""
        all_types = list(DependencyType)
        
        self.assertEqual(len(all_types), 6)
        self.assertIn(DependencyType.MODEL_ARTIFACTS, all_types)
        self.assertIn(DependencyType.PROCESSING_OUTPUT, all_types)
        self.assertIn(DependencyType.TRAINING_DATA, all_types)
        self.assertIn(DependencyType.HYPERPARAMETERS, all_types)
        self.assertIn(DependencyType.PAYLOAD_SAMPLES, all_types)
        self.assertIn(DependencyType.CUSTOM_PROPERTY, all_types)


class TestNodeType(unittest.TestCase):
    """Test cases for NodeType enum."""
    
    def test_enum_values(self):
        """Test that all expected enum values exist."""
        expected_values = {
            "source",
            "internal",
            "sink", 
            "singular"
        }
        
        actual_values = {node_type.value for node_type in NodeType}
        self.assertEqual(actual_values, expected_values)
    
    def test_enum_members(self):
        """Test that all expected enum members exist."""
        expected_members = {
            "SOURCE",
            "INTERNAL",
            "SINK",
            "SINGULAR"
        }
        
        actual_members = {node_type.name for node_type in NodeType}
        self.assertEqual(actual_members, expected_members)
    
    def test_enum_access_by_name(self):
        """Test accessing enum members by name."""
        self.assertEqual(NodeType.SOURCE.value, "source")
        self.assertEqual(NodeType.INTERNAL.value, "internal")
        self.assertEqual(NodeType.SINK.value, "sink")
        self.assertEqual(NodeType.SINGULAR.value, "singular")
    
    def test_enum_access_by_value(self):
        """Test accessing enum members by value."""
        self.assertEqual(NodeType("source"), NodeType.SOURCE)
        self.assertEqual(NodeType("internal"), NodeType.INTERNAL)
        self.assertEqual(NodeType("sink"), NodeType.SINK)
        self.assertEqual(NodeType("singular"), NodeType.SINGULAR)
    
    def test_equality_with_same_enum(self):
        """Test equality comparison between same enum instances."""
        node1 = NodeType.SOURCE
        node2 = NodeType.SOURCE
        
        self.assertEqual(node1, node2)
        self.assertTrue(node1 == node2)
    
    def test_equality_with_different_enum(self):
        """Test equality comparison between different enum instances."""
        node1 = NodeType.SOURCE
        node2 = NodeType.SINK
        
        self.assertNotEqual(node1, node2)
        self.assertFalse(node1 == node2)
    
    def test_equality_with_non_enum(self):
        """Test equality comparison with non-enum values."""
        node = NodeType.SOURCE
        
        # Should not be equal to string value
        self.assertNotEqual(node, "source")
        
        # Should not be equal to other types
        self.assertNotEqual(node, 1)
        self.assertNotEqual(node, None)
        self.assertNotEqual(node, [])
    
    def test_hashability(self):
        """Test that enum instances are hashable."""
        node1 = NodeType.SOURCE
        node2 = NodeType.SINK
        node3 = NodeType.SOURCE
        
        # Should be able to use as dictionary keys
        node_dict = {
            node1: "source_node",
            node2: "sink_node"
        }
        
        self.assertEqual(node_dict[node1], "source_node")
        self.assertEqual(node_dict[node2], "sink_node")
        self.assertEqual(node_dict[node3], "source_node")  # Same as node1
    
    def test_hashability_in_set(self):
        """Test that enum instances work correctly in sets."""
        node_set = {
            NodeType.SOURCE,
            NodeType.SINK,
            NodeType.SOURCE  # Duplicate
        }
        
        # Set should contain only unique values
        self.assertEqual(len(node_set), 2)
        self.assertIn(NodeType.SOURCE, node_set)
        self.assertIn(NodeType.SINK, node_set)
    
    def test_hash_consistency(self):
        """Test that hash values are consistent."""
        node1 = NodeType.SOURCE
        node2 = NodeType.SOURCE
        
        # Same enum instances should have same hash
        self.assertEqual(hash(node1), hash(node2))
        
        # Hash should be based on value
        self.assertEqual(hash(node1), hash(node1.value))
    
    def test_string_representation(self):
        """Test string representation of enum."""
        node = NodeType.SOURCE
        
        # Should contain enum name and value
        str_repr = str(node)
        self.assertIn("NodeType", str_repr)
        self.assertIn("SOURCE", str_repr)
    
    def test_iteration(self):
        """Test iterating over enum members."""
        all_types = list(NodeType)
        
        self.assertEqual(len(all_types), 4)
        self.assertIn(NodeType.SOURCE, all_types)
        self.assertIn(NodeType.INTERNAL, all_types)
        self.assertIn(NodeType.SINK, all_types)
        self.assertIn(NodeType.SINGULAR, all_types)
    
    def test_node_type_semantics(self):
        """Test the semantic meaning of node types."""
        # SOURCE: No dependencies, has outputs
        source = NodeType.SOURCE
        self.assertEqual(source.value, "source")
        
        # INTERNAL: Has both dependencies and outputs  
        internal = NodeType.INTERNAL
        self.assertEqual(internal.value, "internal")
        
        # SINK: Has dependencies, no outputs
        sink = NodeType.SINK
        self.assertEqual(sink.value, "sink")
        
        # SINGULAR: No dependencies, no outputs
        singular = NodeType.SINGULAR
        self.assertEqual(singular.value, "singular")


class TestEnumInteraction(unittest.TestCase):
    """Test cases for interactions between different enums."""
    
    def test_different_enums_not_equal(self):
        """Test that different enum types are not equal even with same values."""
        # This would only apply if we had enums with overlapping values
        # Currently our enums don't have overlapping values, but test the principle
        dep = DependencyType.CUSTOM_PROPERTY
        node = NodeType.SOURCE
        
        self.assertNotEqual(dep, node)
        self.assertFalse(dep == node)
    
    def test_mixed_enum_dictionary(self):
        """Test using different enum types as dictionary keys."""
        mixed_dict = {
            DependencyType.MODEL_ARTIFACTS: "dependency",
            NodeType.SOURCE: "node"
        }
        
        self.assertEqual(len(mixed_dict), 2)
        self.assertEqual(mixed_dict[DependencyType.MODEL_ARTIFACTS], "dependency")
        self.assertEqual(mixed_dict[NodeType.SOURCE], "node")
    
    def test_mixed_enum_set(self):
        """Test using different enum types in the same set."""
        mixed_set = {
            DependencyType.MODEL_ARTIFACTS,
            NodeType.SOURCE,
            DependencyType.TRAINING_DATA,
            NodeType.SINK
        }
        
        self.assertEqual(len(mixed_set), 4)
        self.assertIn(DependencyType.MODEL_ARTIFACTS, mixed_set)
        self.assertIn(NodeType.SOURCE, mixed_set)
        self.assertIn(DependencyType.TRAINING_DATA, mixed_set)
        self.assertIn(NodeType.SINK, mixed_set)
    
    def test_enum_type_checking(self):
        """Test type checking with enums."""
        dep = DependencyType.MODEL_ARTIFACTS
        node = NodeType.SOURCE
        
        self.assertIsInstance(dep, DependencyType)
        self.assertIsInstance(node, NodeType)
        
        self.assertNotIsInstance(dep, NodeType)
        self.assertNotIsInstance(node, DependencyType)


class TestEnumEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def test_invalid_enum_value_dependency_type(self):
        """Test creating enum with invalid value."""
        with self.assertRaises(ValueError):
            DependencyType("invalid_value")
    
    def test_invalid_enum_value_node_type(self):
        """Test creating enum with invalid value."""
        with self.assertRaises(ValueError):
            NodeType("invalid_value")
    
    def test_enum_comparison_with_none(self):
        """Test enum comparison with None."""
        dep = DependencyType.MODEL_ARTIFACTS
        node = NodeType.SOURCE
        
        self.assertNotEqual(dep, None)
        self.assertNotEqual(node, None)
        self.assertFalse(dep == None)
        self.assertFalse(node == None)
    
    def test_enum_boolean_context(self):
        """Test enum instances in boolean context."""
        dep = DependencyType.MODEL_ARTIFACTS
        node = NodeType.SOURCE
        
        # Enum instances should be truthy
        self.assertTrue(dep)
        self.assertTrue(node)
        self.assertTrue(bool(dep))
        self.assertTrue(bool(node))


if __name__ == '__main__':
    # Set up logging for tests
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()

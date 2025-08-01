"""
Unit tests for edge_types module.

Tests the Pydantic V2 implementation of edge types including validation,
serialization, and edge collection functionality.
"""

import unittest
import logging
from unittest.mock import patch
from pydantic import ValidationError

from src.pipeline_dag.edge_types import (
    EdgeType,
    DependencyEdge,
    ConditionalEdge,
    ParallelEdge,
    EdgeCollection
)


class TestEdgeType(unittest.TestCase):
    """Test EdgeType enum."""
    
    def test_edge_type_values(self):
        """Test that EdgeType enum has correct values."""
        self.assertEqual(EdgeType.DEPENDENCY.value, "dependency")
        self.assertEqual(EdgeType.CONDITIONAL.value, "conditional")
        self.assertEqual(EdgeType.PARALLEL.value, "parallel")
        self.assertEqual(EdgeType.SEQUENTIAL.value, "sequential")


class TestDependencyEdge(unittest.TestCase):
    """Test DependencyEdge Pydantic model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.valid_edge_data = {
            'source_step': 'step1',
            'target_step': 'step2',
            'source_output': 'output1',
            'target_input': 'input1'
        }
    
    def test_create_valid_dependency_edge(self):
        """Test creating a valid DependencyEdge."""
        edge = DependencyEdge(**self.valid_edge_data)
        
        self.assertEqual(edge.source_step, 'step1')
        self.assertEqual(edge.target_step, 'step2')
        self.assertEqual(edge.source_output, 'output1')
        self.assertEqual(edge.target_input, 'input1')
        self.assertEqual(edge.confidence, 1.0)  # Default value
        self.assertEqual(edge.edge_type, EdgeType.DEPENDENCY)  # Default value
        self.assertEqual(edge.metadata, {})  # Default empty dict
    
    def test_create_dependency_edge_with_custom_values(self):
        """Test creating DependencyEdge with custom values."""
        custom_data = {
            **self.valid_edge_data,
            'confidence': 0.8,
            'edge_type': EdgeType.SEQUENTIAL,
            'metadata': {'custom': 'value'}
        }
        edge = DependencyEdge(**custom_data)
        
        self.assertEqual(edge.confidence, 0.8)
        self.assertEqual(edge.edge_type, 'sequential')  # Pydantic stores enum values as strings
        self.assertEqual(edge.metadata, {'custom': 'value'})
    
    def test_empty_string_validation(self):
        """Test that empty strings are rejected."""
        test_cases = [
            {'source_step': ''},
            {'target_step': ''},
            {'source_output': ''},
            {'target_input': ''}
        ]
        
        for invalid_field in test_cases:
            with self.subTest(invalid_field=invalid_field):
                data = {**self.valid_edge_data, **invalid_field}
                with self.assertRaises(ValidationError):
                    DependencyEdge(**data)
    
    def test_confidence_validation(self):
        """Test confidence value validation."""
        # Test valid confidence values
        valid_confidences = [0.0, 0.5, 1.0]
        for confidence in valid_confidences:
            with self.subTest(confidence=confidence):
                data = {**self.valid_edge_data, 'confidence': confidence}
                edge = DependencyEdge(**data)
                self.assertEqual(edge.confidence, confidence)
        
        # Test invalid confidence values
        invalid_confidences = [-0.1, 1.1, -1.0, 2.0]
        for confidence in invalid_confidences:
            with self.subTest(confidence=confidence):
                data = {**self.valid_edge_data, 'confidence': confidence}
                with self.assertRaises(ValidationError):
                    DependencyEdge(**data)
    
    def test_to_property_reference_dict(self):
        """Test property reference dictionary generation."""
        edge = DependencyEdge(**self.valid_edge_data)
        prop_ref = edge.to_property_reference_dict()
        
        expected = {"Get": "Steps.step1.output1"}
        self.assertEqual(prop_ref, expected)
    
    def test_is_high_confidence(self):
        """Test high confidence detection."""
        # Test with default threshold (0.8)
        high_conf_edge = DependencyEdge(**{**self.valid_edge_data, 'confidence': 0.9})
        low_conf_edge = DependencyEdge(**{**self.valid_edge_data, 'confidence': 0.7})
        
        self.assertTrue(high_conf_edge.is_high_confidence())
        self.assertFalse(low_conf_edge.is_high_confidence())
        
        # Test with custom threshold
        self.assertTrue(low_conf_edge.is_high_confidence(threshold=0.6))
        self.assertFalse(low_conf_edge.is_high_confidence(threshold=0.8))
    
    def test_is_auto_resolved(self):
        """Test auto-resolved detection."""
        manual_edge = DependencyEdge(**{**self.valid_edge_data, 'confidence': 1.0})
        auto_edge = DependencyEdge(**{**self.valid_edge_data, 'confidence': 0.9})
        
        self.assertFalse(manual_edge.is_auto_resolved())
        self.assertTrue(auto_edge.is_auto_resolved())
    
    def test_string_representations(self):
        """Test __str__ and __repr__ methods."""
        edge = DependencyEdge(**self.valid_edge_data)
        
        expected_str = "step1.output1 -> step2.input1"
        self.assertEqual(str(edge), expected_str)
        
        expected_repr = "DependencyEdge(source='step1.output1', target='step2.input1', confidence=1.000)"
        self.assertEqual(repr(edge), expected_repr)
    
    def test_pydantic_serialization(self):
        """Test Pydantic model serialization."""
        edge = DependencyEdge(**self.valid_edge_data)
        serialized = edge.model_dump()
        
        expected_keys = {
            'source_step', 'target_step', 'source_output', 'target_input',
            'confidence', 'edge_type', 'metadata'
        }
        self.assertEqual(set(serialized.keys()), expected_keys)
        self.assertEqual(serialized['source_step'], 'step1')
        self.assertEqual(serialized['confidence'], 1.0)
        self.assertEqual(serialized['edge_type'], EdgeType.DEPENDENCY)
    
    def test_pydantic_deserialization(self):
        """Test Pydantic model deserialization."""
        data = {
            'source_step': 'step1',
            'target_step': 'step2',
            'source_output': 'output1',
            'target_input': 'input1',
            'confidence': 0.8,
            'edge_type': 'sequential',
            'metadata': {'test': 'value'}
        }
        
        edge = DependencyEdge.model_validate(data)
        self.assertEqual(edge.source_step, 'step1')
        self.assertEqual(edge.confidence, 0.8)
        self.assertEqual(edge.edge_type, 'sequential')  # Pydantic stores enum values as strings
        self.assertEqual(edge.metadata, {'test': 'value'})


class TestConditionalEdge(unittest.TestCase):
    """Test ConditionalEdge Pydantic model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.valid_edge_data = {
            'source_step': 'step1',
            'target_step': 'step2',
            'source_output': 'output1',
            'target_input': 'input1'
        }
    
    def test_create_conditional_edge(self):
        """Test creating ConditionalEdge."""
        edge = ConditionalEdge(**self.valid_edge_data)
        
        self.assertEqual(edge.condition, "")  # Default empty condition
        self.assertEqual(edge.edge_type, EdgeType.CONDITIONAL)
        self.assertIsInstance(edge, DependencyEdge)  # Inheritance check
    
    def test_create_conditional_edge_with_condition(self):
        """Test creating ConditionalEdge with condition."""
        data = {**self.valid_edge_data, 'condition': 'x > 0'}
        edge = ConditionalEdge(**data)
        
        self.assertEqual(edge.condition, 'x > 0')
        self.assertEqual(edge.edge_type, EdgeType.CONDITIONAL)
    
    @patch('src.pipeline_dag.edge_types.logger')
    def test_empty_condition_warning(self, mock_logger):
        """Test that empty condition triggers warning."""
        ConditionalEdge(**self.valid_edge_data)
        
        # Check that warning was logged
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args[0][0]
        self.assertIn("ConditionalEdge", call_args)
        self.assertIn("no condition specified", call_args)
    
    @patch('src.pipeline_dag.edge_types.logger')
    def test_non_empty_condition_no_warning(self, mock_logger):
        """Test that non-empty condition doesn't trigger warning."""
        data = {**self.valid_edge_data, 'condition': 'x > 0'}
        ConditionalEdge(**data)
        
        # Check that no warning was logged
        mock_logger.warning.assert_not_called()


class TestParallelEdge(unittest.TestCase):
    """Test ParallelEdge Pydantic model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.valid_edge_data = {
            'source_step': 'step1',
            'target_step': 'step2',
            'source_output': 'output1',
            'target_input': 'input1'
        }
    
    def test_create_parallel_edge(self):
        """Test creating ParallelEdge."""
        edge = ParallelEdge(**self.valid_edge_data)
        
        self.assertIsNone(edge.max_parallel)  # Default None
        self.assertEqual(edge.edge_type, EdgeType.PARALLEL)
        self.assertIsInstance(edge, DependencyEdge)  # Inheritance check
    
    def test_create_parallel_edge_with_max_parallel(self):
        """Test creating ParallelEdge with max_parallel."""
        data = {**self.valid_edge_data, 'max_parallel': 4}
        edge = ParallelEdge(**data)
        
        self.assertEqual(edge.max_parallel, 4)
        self.assertEqual(edge.edge_type, EdgeType.PARALLEL)
    
    def test_max_parallel_validation(self):
        """Test max_parallel validation."""
        # Valid values
        valid_values = [1, 2, 10, 100]
        for value in valid_values:
            with self.subTest(value=value):
                data = {**self.valid_edge_data, 'max_parallel': value}
                edge = ParallelEdge(**data)
                self.assertEqual(edge.max_parallel, value)
        
        # Invalid values (less than 1)
        invalid_values = [0, -1, -10]
        for value in invalid_values:
            with self.subTest(value=value):
                data = {**self.valid_edge_data, 'max_parallel': value}
                with self.assertRaises(ValidationError):
                    ParallelEdge(**data)


class TestEdgeCollection(unittest.TestCase):
    """Test EdgeCollection functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.collection = EdgeCollection()
        self.edge1 = DependencyEdge(
            source_step='step1',
            target_step='step2',
            source_output='output1',
            target_input='input1'
        )
        self.edge2 = DependencyEdge(
            source_step='step2',
            target_step='step3',
            source_output='output2',
            target_input='input2'
        )
    
    def test_empty_collection(self):
        """Test empty collection properties."""
        self.assertEqual(len(self.collection), 0)
        self.assertEqual(list(self.collection), [])
        self.assertNotIn('any_id', self.collection)
    
    def test_add_edge(self):
        """Test adding edges to collection."""
        edge_id = self.collection.add_edge(self.edge1)
        
        expected_id = "step1:output1->step2:input1"
        self.assertEqual(edge_id, expected_id)
        self.assertEqual(len(self.collection), 1)
        self.assertIn(expected_id, self.collection)
    
    def test_add_duplicate_edge_higher_confidence(self):
        """Test adding duplicate edge with higher confidence."""
        # Add original edge with lower confidence
        low_conf_edge = DependencyEdge(
            source_step='step1',
            target_step='step2',
            source_output='output1',
            target_input='input1',
            confidence=0.7
        )
        self.collection.add_edge(low_conf_edge)
        
        # Create edge with same connection but higher confidence
        high_conf_edge = DependencyEdge(
            source_step='step1',
            target_step='step2',
            source_output='output1',
            target_input='input1',
            confidence=0.9
        )
        
        with patch('src.pipeline_dag.edge_types.logger') as mock_logger:
            edge_id = self.collection.add_edge(high_conf_edge)
            
            # Should replace the original edge
            self.assertEqual(len(self.collection), 1)
            stored_edge = self.collection.edges[edge_id]
            self.assertEqual(stored_edge.confidence, 0.9)
            
            # Should log replacement
            mock_logger.info.assert_called_once()
    
    def test_add_duplicate_edge_lower_confidence(self):
        """Test adding duplicate edge with lower confidence."""
        # Add high confidence edge first
        high_conf_edge = DependencyEdge(
            source_step='step1',
            target_step='step2',
            source_output='output1',
            target_input='input1',
            confidence=0.9
        )
        self.collection.add_edge(high_conf_edge)
        
        # Try to add higher confidence edge (1.0 > 0.9, so it should replace)
        with patch('src.pipeline_dag.edge_types.logger') as mock_logger:
            edge_id = self.collection.add_edge(self.edge1)  # confidence=1.0 should replace 0.9
            
            # Should replace with higher confidence edge
            stored_edge = self.collection.edges[edge_id]
            self.assertEqual(stored_edge.confidence, 1.0)
            
            # Should log replacement
            mock_logger.info.assert_called_once()
    
    def test_remove_edge(self):
        """Test removing edges from collection."""
        edge_id = self.collection.add_edge(self.edge1)
        
        # Remove existing edge
        result = self.collection.remove_edge(edge_id)
        self.assertTrue(result)
        self.assertEqual(len(self.collection), 0)
        self.assertNotIn(edge_id, self.collection)
        
        # Try to remove non-existent edge
        result = self.collection.remove_edge('non_existent')
        self.assertFalse(result)
    
    def test_get_edges_from_step(self):
        """Test getting edges from a specific step."""
        self.collection.add_edge(self.edge1)
        self.collection.add_edge(self.edge2)
        
        edges_from_step1 = self.collection.get_edges_from_step('step1')
        self.assertEqual(len(edges_from_step1), 1)
        self.assertEqual(edges_from_step1[0], self.edge1)
        
        edges_from_step2 = self.collection.get_edges_from_step('step2')
        self.assertEqual(len(edges_from_step2), 1)
        self.assertEqual(edges_from_step2[0], self.edge2)
        
        edges_from_nonexistent = self.collection.get_edges_from_step('nonexistent')
        self.assertEqual(len(edges_from_nonexistent), 0)
    
    def test_get_edges_to_step(self):
        """Test getting edges to a specific step."""
        self.collection.add_edge(self.edge1)
        self.collection.add_edge(self.edge2)
        
        edges_to_step2 = self.collection.get_edges_to_step('step2')
        self.assertEqual(len(edges_to_step2), 1)
        self.assertEqual(edges_to_step2[0], self.edge1)
        
        edges_to_step3 = self.collection.get_edges_to_step('step3')
        self.assertEqual(len(edges_to_step3), 1)
        self.assertEqual(edges_to_step3[0], self.edge2)
        
        edges_to_nonexistent = self.collection.get_edges_to_step('nonexistent')
        self.assertEqual(len(edges_to_nonexistent), 0)
    
    def test_get_edge(self):
        """Test getting specific edge by components."""
        self.collection.add_edge(self.edge1)
        
        # Get existing edge
        edge = self.collection.get_edge('step1', 'output1', 'step2', 'input1')
        self.assertEqual(edge, self.edge1)
        
        # Get non-existent edge
        edge = self.collection.get_edge('step1', 'output1', 'step3', 'input1')
        self.assertIsNone(edge)
    
    def test_list_methods(self):
        """Test various list methods."""
        # Create edges with different confidence levels and different connections
        high_conf_edge = DependencyEdge(
            source_step='step3', target_step='step4',
            source_output='output3', target_input='input3',
            confidence=0.9
        )
        low_conf_edge = DependencyEdge(
            source_step='step4', target_step='step5',
            source_output='output4', target_input='input4',
            confidence=0.5
        )
        
        self.collection.add_edge(self.edge1)  # confidence=1.0
        self.collection.add_edge(self.edge2)  # confidence=1.0
        self.collection.add_edge(high_conf_edge)
        self.collection.add_edge(low_conf_edge)
        
        # Test list_all_edges
        all_edges = self.collection.list_all_edges()
        self.assertEqual(len(all_edges), 4)
        
        # Test list_auto_resolved_edges
        auto_edges = self.collection.list_auto_resolved_edges()
        self.assertEqual(len(auto_edges), 2)  # high_conf_edge and low_conf_edge
        
        # Test list_high_confidence_edges
        high_conf_edges = self.collection.list_high_confidence_edges()
        self.assertEqual(len(high_conf_edges), 3)  # edge1, edge2, and high_conf_edge
        
        # Test list_low_confidence_edges
        low_conf_edges = self.collection.list_low_confidence_edges()
        self.assertEqual(len(low_conf_edges), 1)  # low_conf_edge
    
    def test_get_step_dependencies(self):
        """Test getting step dependencies as dictionary."""
        self.collection.add_edge(self.edge1)
        
        dependencies = self.collection.get_step_dependencies('step2')
        expected = {'input1': self.edge1}
        self.assertEqual(dependencies, expected)
        
        # Test step with no dependencies
        dependencies = self.collection.get_step_dependencies('step1')
        self.assertEqual(dependencies, {})
    
    def test_validate_edges(self):
        """Test edge validation."""
        # Add valid edges
        self.collection.add_edge(self.edge1)
        self.collection.add_edge(self.edge2)
        
        errors = self.collection.validate_edges()
        self.assertEqual(len(errors), 0)
        
        # Add self-dependency edge manually (bypassing normal validation)
        self_dep_edge = DependencyEdge(
            source_step='step1', target_step='step1',
            source_output='output1', target_input='input1'
        )
        edge_id = "step1:output1->step1:input1"
        self.collection.edges[edge_id] = self_dep_edge
        
        errors = self.collection.validate_edges()
        self.assertEqual(len(errors), 1)
        self.assertIn("Self-dependency detected", errors[0])
    
    def test_get_statistics(self):
        """Test statistics generation."""
        # Test empty collection
        stats = self.collection.get_statistics()
        expected_empty = {
            'total_edges': 0,
            'auto_resolved_edges': 0,
            'high_confidence_edges': 0,
            'low_confidence_edges': 0,
            'average_confidence': 0.0,
            'edge_types': {}
        }
        self.assertEqual(stats, expected_empty)
        
        # Add edges with different types and confidence levels using different connections
        conditional_edge = ConditionalEdge(
            source_step='step3', target_step='step4',
            source_output='output3', target_input='input3',
            condition='x > 0', confidence=0.8
        )
        parallel_edge = ParallelEdge(
            source_step='step4', target_step='step5',
            source_output='output4', target_input='input4',
            max_parallel=4, confidence=0.6
        )
        
        self.collection.add_edge(self.edge1)  # confidence=1.0, dependency
        self.collection.add_edge(conditional_edge)
        self.collection.add_edge(parallel_edge)
        
        stats = self.collection.get_statistics()
        
        self.assertEqual(stats['total_edges'], 3)
        self.assertEqual(stats['auto_resolved_edges'], 2)  # conditional and parallel
        self.assertEqual(stats['high_confidence_edges'], 2)  # edge1 and conditional
        self.assertEqual(stats['low_confidence_edges'], 0)  # parallel edge has 0.6 confidence, but default threshold is 0.6
        self.assertAlmostEqual(stats['average_confidence'], 0.8)  # (1.0 + 0.8 + 0.6) / 3
        self.assertEqual(stats['min_confidence'], 0.6)
        self.assertEqual(stats['max_confidence'], 1.0)
        
        expected_edge_types = {
            'dependency': 1,
            'conditional': 1,
            'parallel': 1
        }
        self.assertEqual(stats['edge_types'], expected_edge_types)
        self.assertEqual(stats['unique_source_steps'], 3)  # step1, step3, and step4
        self.assertEqual(stats['unique_target_steps'], 3)  # step2, step4, and step5


if __name__ == '__main__':
    unittest.main()

import unittest
from collections import deque
from unittest.mock import patch, MagicMock

# Add the project root to the Python path to allow for absolute imports
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.pipeline_builder.pipeline_dag import PipelineDAG


class TestPipelineDAG(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple DAG for testing
        self.nodes = ['A', 'B', 'C', 'D']
        self.edges = [('A', 'B'), ('B', 'C'), ('A', 'C'), ('C', 'D')]
        self.dag = PipelineDAG(nodes=self.nodes, edges=self.edges)

    def test_init_empty(self):
        """Test initialization with empty nodes and edges."""
        dag = PipelineDAG()
        self.assertEqual(dag.nodes, [])
        self.assertEqual(dag.edges, [])
        self.assertEqual(dag.adj_list, {})
        self.assertEqual(dag.reverse_adj, {})

    def test_init_with_nodes_edges(self):
        """Test initialization with nodes and edges."""
        # Check nodes
        self.assertEqual(set(self.dag.nodes), set(self.nodes))
        
        # Check edges
        self.assertEqual(set(self.dag.edges), set(self.edges))
        
        # Check adjacency list
        self.assertEqual(self.dag.adj_list['A'], ['B', 'C'])
        self.assertEqual(self.dag.adj_list['B'], ['C'])
        self.assertEqual(self.dag.adj_list['C'], ['D'])
        self.assertEqual(self.dag.adj_list['D'], [])
        
        # Check reverse adjacency list
        self.assertEqual(self.dag.reverse_adj['A'], [])
        self.assertEqual(self.dag.reverse_adj['B'], ['A'])
        self.assertEqual(self.dag.reverse_adj['C'], ['B', 'A'])
        self.assertEqual(self.dag.reverse_adj['D'], ['C'])

    def test_add_node(self):
        """Test adding a node to the DAG."""
        dag = PipelineDAG()
        
        # Add a node
        dag.add_node('X')
        self.assertIn('X', dag.nodes)
        self.assertEqual(dag.adj_list['X'], [])
        self.assertEqual(dag.reverse_adj['X'], [])
        
        # Add the same node again (should not duplicate)
        dag.add_node('X')
        self.assertEqual(dag.nodes.count('X'), 1)

    def test_add_edge(self):
        """Test adding an edge to the DAG."""
        dag = PipelineDAG()
        
        # Add an edge between non-existent nodes (should create nodes)
        dag.add_edge('X', 'Y')
        self.assertIn('X', dag.nodes)
        self.assertIn('Y', dag.nodes)
        self.assertIn(('X', 'Y'), dag.edges)
        self.assertEqual(dag.adj_list['X'], ['Y'])
        self.assertEqual(dag.reverse_adj['Y'], ['X'])
        
        # Add the same edge again (should not duplicate)
        dag.add_edge('X', 'Y')
        self.assertEqual(dag.edges.count(('X', 'Y')), 1)
        
        # Add edge where one node exists
        dag.add_edge('X', 'Z')
        self.assertIn('Z', dag.nodes)
        self.assertIn(('X', 'Z'), dag.edges)
        self.assertEqual(dag.adj_list['X'], ['Y', 'Z'])
        self.assertEqual(dag.reverse_adj['Z'], ['X'])

    def test_get_dependencies(self):
        """Test getting dependencies of a node."""
        # Test existing node
        self.assertEqual(set(self.dag.get_dependencies('C')), {'A', 'B'})
        
        # Test node with no dependencies
        self.assertEqual(self.dag.get_dependencies('A'), [])
        
        # Test non-existent node
        self.assertEqual(self.dag.get_dependencies('Z'), [])

    def test_topological_sort(self):
        """Test topological sorting of the DAG."""
        # Get the topological order
        order = self.dag.topological_sort()
        
        # Check that the order is valid
        self.assertEqual(len(order), len(self.nodes))
        self.assertIn('A', order)
        self.assertIn('B', order)
        self.assertIn('C', order)
        self.assertIn('D', order)
        
        # Check that dependencies come before dependents
        self.assertLess(order.index('A'), order.index('B'))
        self.assertLess(order.index('A'), order.index('C'))
        self.assertLess(order.index('B'), order.index('C'))
        self.assertLess(order.index('C'), order.index('D'))

    def test_topological_sort_with_cycle(self):
        """Test topological sorting with a cycle (should raise ValueError)."""
        # Create a DAG with a cycle
        dag = PipelineDAG(
            nodes=['A', 'B', 'C'],
            edges=[('A', 'B'), ('B', 'C'), ('C', 'A')]
        )
        
        # Topological sort should raise ValueError
        with self.assertRaises(ValueError):
            dag.topological_sort()

    def test_topological_sort_disconnected(self):
        """Test topological sorting with disconnected nodes."""
        # Create a DAG with disconnected nodes
        dag = PipelineDAG(
            nodes=['A', 'B', 'C', 'D', 'E'],
            edges=[('A', 'B'), ('B', 'C')]  # D and E are disconnected
        )
        
        # Get the topological order
        order = dag.topological_sort()
        
        # Check that all nodes are included
        self.assertEqual(set(order), {'A', 'B', 'C', 'D', 'E'})
        
        # Check that dependencies come before dependents
        self.assertLess(order.index('A'), order.index('B'))
        self.assertLess(order.index('B'), order.index('C'))


if __name__ == '__main__':
    unittest.main()

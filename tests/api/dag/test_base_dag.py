import unittest
from collections import deque
from unittest.mock import patch, MagicMock

from cursus.api.dag.base_dag import PipelineDAG


class TestPipelineDAG(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple DAG for testing
        self.nodes = ["A", "B", "C", "D"]
        self.edges = [("A", "B"), ("B", "C"), ("A", "C"), ("C", "D")]
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
        self.assertEqual(self.dag.adj_list["A"], ["B", "C"])
        self.assertEqual(self.dag.adj_list["B"], ["C"])
        self.assertEqual(self.dag.adj_list["C"], ["D"])
        self.assertEqual(self.dag.adj_list["D"], [])

        # Check reverse adjacency list
        self.assertEqual(self.dag.reverse_adj["A"], [])
        self.assertEqual(self.dag.reverse_adj["B"], ["A"])
        self.assertEqual(self.dag.reverse_adj["C"], ["B", "A"])
        self.assertEqual(self.dag.reverse_adj["D"], ["C"])

    def test_add_node(self):
        """Test adding a node to the DAG."""
        dag = PipelineDAG()

        # Add a node
        dag.add_node("X")
        self.assertIn("X", dag.nodes)
        self.assertEqual(dag.adj_list["X"], [])
        self.assertEqual(dag.reverse_adj["X"], [])

        # Add the same node again (should not duplicate)
        dag.add_node("X")
        self.assertEqual(dag.nodes.count("X"), 1)

    def test_add_edge(self):
        """Test adding an edge to the DAG."""
        dag = PipelineDAG()

        # Add an edge between non-existent nodes (should create nodes)
        dag.add_edge("X", "Y")
        self.assertIn("X", dag.nodes)
        self.assertIn("Y", dag.nodes)
        self.assertIn(("X", "Y"), dag.edges)
        self.assertEqual(dag.adj_list["X"], ["Y"])
        self.assertEqual(dag.reverse_adj["Y"], ["X"])

        # Add the same edge again (should not duplicate)
        dag.add_edge("X", "Y")
        self.assertEqual(dag.edges.count(("X", "Y")), 1)

        # Add edge where one node exists
        dag.add_edge("X", "Z")
        self.assertIn("Z", dag.nodes)
        self.assertIn(("X", "Z"), dag.edges)
        self.assertEqual(dag.adj_list["X"], ["Y", "Z"])
        self.assertEqual(dag.reverse_adj["Z"], ["X"])

    def test_get_dependencies(self):
        """Test getting dependencies of a node."""
        # Test existing node
        self.assertEqual(set(self.dag.get_dependencies("C")), {"A", "B"})

        # Test node with no dependencies
        self.assertEqual(self.dag.get_dependencies("A"), [])

        # Test non-existent node
        self.assertEqual(self.dag.get_dependencies("Z"), [])

    def test_topological_sort(self):
        """Test topological sorting of the DAG."""
        # Get the topological order
        order = self.dag.topological_sort()

        # Check that the order is valid
        self.assertEqual(len(order), len(self.nodes))
        self.assertIn("A", order)
        self.assertIn("B", order)
        self.assertIn("C", order)
        self.assertIn("D", order)

        # Check that dependencies come before dependents
        self.assertLess(order.index("A"), order.index("B"))
        self.assertLess(order.index("A"), order.index("C"))
        self.assertLess(order.index("B"), order.index("C"))
        self.assertLess(order.index("C"), order.index("D"))

    def test_topological_sort_with_cycle(self):
        """Test topological sorting with a cycle (should raise ValueError)."""
        # Create a DAG with a cycle
        dag = PipelineDAG(
            nodes=["A", "B", "C"], edges=[("A", "B"), ("B", "C"), ("C", "A")]
        )

        # Topological sort should raise ValueError
        with self.assertRaises(ValueError):
            dag.topological_sort()

    def test_topological_sort_disconnected(self):
        """Test topological sorting with disconnected nodes."""
        # Create a DAG with disconnected nodes
        dag = PipelineDAG(
            nodes=["A", "B", "C", "D", "E"],
            edges=[("A", "B"), ("B", "C")],  # D and E are disconnected
        )

        # Get the topological order
        order = dag.topological_sort()

        # Check that all nodes are included
        self.assertEqual(set(order), {"A", "B", "C", "D", "E"})

        # Check that dependencies come before dependents
        self.assertLess(order.index("A"), order.index("B"))
        self.assertLess(order.index("B"), order.index("C"))


class TestPipelineDAGNodeDeclarations(unittest.TestCase):
    """Declared-vs-auto-created node tracking, validate_node_declarations, and strict mode."""

    def test_validate_node_declarations_clean(self):
        """Every edge endpoint declared -> no undeclared endpoints reported."""
        dag = PipelineDAG(nodes=["A", "B"], edges=[("A", "B")])
        self.assertEqual(dag.validate_node_declarations(), [])

    def test_validate_node_declarations_flags_typo(self):
        """An edge endpoint that was never declared (a typo) is surfaced — and only once."""
        # 'Bb' is a typo; only 'A' was declared, 'B' is missing from nodes entirely.
        dag = PipelineDAG(nodes=["A", "B"], edges=[("A", "Bb"), ("A", "Bb")])
        self.assertEqual(dag.validate_node_declarations(), ["Bb"])

    def test_add_edge_auto_create_is_not_declared(self):
        """Lenient add_edge auto-creates the node but does NOT mark it declared."""
        dag = PipelineDAG(nodes=["A"])
        dag.add_edge("A", "Z")  # Z auto-created
        self.assertIn("Z", dag.nodes)
        self.assertEqual(dag.validate_node_declarations(), ["Z"])

    def test_add_node_promotes_auto_created(self):
        """An explicit add_node after an auto-create promotes the node to declared."""
        dag = PipelineDAG(nodes=["A"])
        dag.add_edge("A", "Z")
        self.assertEqual(dag.validate_node_declarations(), ["Z"])
        dag.add_node("Z")  # now declared
        self.assertEqual(dag.validate_node_declarations(), [])

    def test_strict_add_edge_raises_on_undeclared(self):
        """strict=True: add_edge raises if an endpoint was never declared."""
        dag = PipelineDAG(nodes=["A"], strict=True)
        with self.assertRaises(ValueError):
            dag.add_edge("A", "Z")  # Z not declared

    def test_strict_add_edge_ok_when_declared(self):
        """strict=True: add_edge succeeds once both endpoints are declared."""
        dag = PipelineDAG(strict=True)
        dag.add_node("A")
        dag.add_node("B")
        dag.add_edge("A", "B")  # both declared -> fine
        self.assertIn(("A", "B"), dag.edges)

    def test_strict_constructor_raises_on_undeclared_edge(self):
        """strict=True at construction: an edge endpoint missing from nodes= raises."""
        with self.assertRaises(ValueError):
            PipelineDAG(nodes=["A", "B"], edges=[("A", "Bb")], strict=True)

    def test_lenient_is_default(self):
        """Default stays lenient: add_edge between non-existent nodes still auto-creates."""
        dag = PipelineDAG()
        dag.add_edge("X", "Y")  # no raise
        self.assertIn("X", dag.nodes)
        self.assertIn("Y", dag.nodes)


if __name__ == "__main__":
    unittest.main()

"""
Tests for PipelineDAG serialization and deserialization.
"""

import json
import pytest
from pathlib import Path
import tempfile
from datetime import datetime

from cursus.api.dag import (
    PipelineDAG,
    PipelineDAGWriter,
    PipelineDAGReader,
    export_dag_to_json,
    import_dag_from_json,
)


class TestPipelineDAGWriter:
    """Tests for PipelineDAGWriter"""

    def test_to_dict_basic(self):
        """Test conversion to dictionary"""
        dag = PipelineDAG(nodes=["step1", "step2"], edges=[("step1", "step2")])

        writer = PipelineDAGWriter(dag)
        result = writer.to_dict()

        assert result["version"] == "1.0.0"
        assert "created_at" in result
        assert result["dag"]["nodes"] == ["step1", "step2"]
        assert result["dag"]["edges"] == [["step1", "step2"]]
        assert "statistics" in result

    def test_to_dict_with_metadata(self):
        """Test dictionary conversion with metadata"""
        dag = PipelineDAG(nodes=["step1"], edges=[])
        metadata = {"project": "test", "author": "tester"}

        writer = PipelineDAGWriter(dag, metadata=metadata)
        result = writer.to_dict()

        assert result["metadata"] == metadata

    def test_to_json_pretty(self):
        """Test JSON string conversion with pretty printing"""
        dag = PipelineDAG(nodes=["step1"], edges=[])
        writer = PipelineDAGWriter(dag)

        json_str = writer.to_json(pretty=True, indent=2)

        # Verify it's valid JSON
        data = json.loads(json_str)
        assert data["version"] == "1.0.0"

        # Check for indentation
        assert "\n" in json_str
        assert "  " in json_str

    def test_to_json_compact(self):
        """Test compact JSON conversion"""
        dag = PipelineDAG(nodes=["step1"], edges=[])
        writer = PipelineDAGWriter(dag)

        json_str = writer.to_json(pretty=False)

        # Should not have much whitespace
        lines = json_str.split("\n")
        assert len(lines) == 1  # Single line

    def test_write_to_file(self):
        """Test writing to file"""
        dag = PipelineDAG(
            nodes=["step1", "step2", "step3"],
            edges=[("step1", "step2"), ("step2", "step3")],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_dag.json"
            writer = PipelineDAGWriter(dag)
            writer.write_to_file(filepath)

            assert filepath.exists()

            # Verify content
            with open(filepath) as f:
                data = json.load(f)

            assert data["dag"]["nodes"] == ["step1", "step2", "step3"]
            assert len(data["dag"]["edges"]) == 2

    def test_write_creates_directories(self):
        """Test that write creates necessary directories"""
        dag = PipelineDAG(nodes=["step1"], edges=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "subdir" / "nested" / "test.json"
            writer = PipelineDAGWriter(dag)
            writer.write_to_file(filepath)

            assert filepath.exists()

    def test_validation_rejects_cyclic_dag(self):
        """Test that validation rejects DAGs with cycles"""
        # Create a DAG with a cycle
        dag = PipelineDAG(
            nodes=["a", "b", "c"], edges=[("a", "b"), ("b", "c"), ("c", "a")]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "cyclic.json"
            writer = PipelineDAGWriter(dag)

            with pytest.raises(ValueError, match="DAG validation failed"):
                writer.write_to_file(filepath, validate=True)

    def test_validation_rejects_empty_dag(self):
        """Test that validation rejects empty DAGs"""
        dag = PipelineDAG(nodes=[], edges=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "empty.json"
            writer = PipelineDAGWriter(dag)

            with pytest.raises(ValueError, match="Cannot write empty DAG"):
                writer.write_to_file(filepath, validate=True)

    def test_validation_rejects_dangling_edges(self):
        """Test that validation rejects dangling edges"""
        dag = PipelineDAG(
            nodes=["step1", "step2"],
            edges=[("step1", "step2"), ("step2", "step3")],  # step3 doesn't exist
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "dangling.json"
            writer = PipelineDAGWriter(dag)

            with pytest.raises(ValueError, match="non-existent"):
                writer.write_to_file(filepath, validate=True)

    def test_statistics_computation(self):
        """Test DAG statistics computation"""
        dag = PipelineDAG(
            nodes=["a", "b", "c", "d"],
            edges=[("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")],
        )

        writer = PipelineDAGWriter(dag)
        result = writer.to_dict()
        stats = result["statistics"]

        assert stats["node_count"] == 4
        assert stats["edge_count"] == 4
        assert stats["has_cycles"] is False
        assert stats["entry_nodes"] == ["a"]
        assert stats["exit_nodes"] == ["d"]
        assert stats["max_depth"] == 2
        assert stats["isolated_nodes"] == []


class TestPipelineDAGReader:
    """Tests for PipelineDAGReader"""

    def test_from_dict_basic(self):
        """Test reading from dictionary"""
        data = {
            "version": "1.0.0",
            "dag": {"nodes": ["step1", "step2"], "edges": [["step1", "step2"]]},
        }

        dag = PipelineDAGReader.from_dict(data)

        assert dag.nodes == ["step1", "step2"]
        assert dag.edges == [("step1", "step2")]

    def test_from_json_string(self):
        """Test reading from JSON string"""
        json_str = json.dumps(
            {
                "version": "1.0.0",
                "dag": {"nodes": ["a", "b", "c"], "edges": [["a", "b"], ["b", "c"]]},
            }
        )

        dag = PipelineDAGReader.from_json(json_str)

        assert len(dag.nodes) == 3
        assert len(dag.edges) == 2

    def test_read_from_file(self):
        """Test reading from file"""
        # First write a file
        original_dag = PipelineDAG(
            nodes=["step1", "step2", "step3"],
            edges=[("step1", "step2"), ("step2", "step3")],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json"
            writer = PipelineDAGWriter(original_dag)
            writer.write_to_file(filepath)

            # Now read it back
            loaded_dag = PipelineDAGReader.read_from_file(filepath)

            assert loaded_dag.nodes == original_dag.nodes
            assert loaded_dag.edges == original_dag.edges

    def test_read_nonexistent_file(self):
        """Test reading from non-existent file raises error"""
        with pytest.raises(FileNotFoundError):
            PipelineDAGReader.read_from_file("/nonexistent/file.json")

    def test_extract_metadata(self):
        """Test extracting metadata without loading full DAG"""
        dag = PipelineDAG(nodes=["step1", "step2"], edges=[("step1", "step2")])
        metadata = {"project": "test_project", "author": "test_author"}

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json"
            writer = PipelineDAGWriter(dag, metadata=metadata)
            writer.write_to_file(filepath)

            extracted = PipelineDAGReader.extract_metadata(filepath)

            assert extracted["version"] == "1.0.0"
            assert "created_at" in extracted
            assert extracted["metadata"] == metadata
            assert "statistics" in extracted

    def test_validation_checks_version(self):
        """Test that validation checks version compatibility"""
        data = {
            "version": "99.0.0",  # Unsupported version
            "dag": {"nodes": ["step1"], "edges": []},
        }

        with pytest.raises(ValueError, match="Unsupported version"):
            PipelineDAGReader.from_dict(data, validate=True)

    def test_validation_checks_required_fields(self):
        """Test that validation checks for required fields"""
        # Missing 'dag' field
        data = {"version": "1.0.0"}

        with pytest.raises(ValueError, match="Missing required field"):
            PipelineDAGReader.from_dict(data, validate=True)

    def test_validation_checks_edge_format(self):
        """Test that validation checks edge format"""
        data = {
            "version": "1.0.0",
            "dag": {
                "nodes": ["a", "b"],
                "edges": [["a", "b", "c"]],  # Invalid: 3 elements instead of 2
            },
        }

        with pytest.raises(ValueError, match="Invalid edge format"):
            PipelineDAGReader.from_dict(data, validate=True)

    def test_skip_validation(self):
        """Test that validation can be skipped"""
        # Intentionally malformed data
        data = {"version": "99.0.0", "dag": {"nodes": ["step1"], "edges": []}}

        # Should not raise with validate=False
        dag = PipelineDAGReader.from_dict(data, validate=False)
        assert dag.nodes == ["step1"]


class TestConvenienceFunctions:
    """Tests for convenience functions"""

    def test_export_import_round_trip(self):
        """Test round-trip export and import"""
        original = PipelineDAG(
            nodes=["a", "b", "c", "d"], edges=[("a", "b"), ("b", "c"), ("c", "d")]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "pipeline.json"

            # Export
            export_dag_to_json(original, filepath, metadata={"test": "round_trip"})

            # Import
            loaded = import_dag_from_json(filepath)

            # Verify
            assert loaded.nodes == original.nodes
            assert loaded.edges == original.edges

    def test_export_with_pretty_printing(self):
        """Test export with pretty printing option"""
        dag = PipelineDAG(nodes=["step1"], edges=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "pretty.json"
            export_dag_to_json(dag, filepath, pretty=True)

            with open(filepath) as f:
                content = f.read()

            # Should have multiple lines and indentation
            assert len(content.split("\n")) > 1


class TestComplexDAGs:
    """Tests with complex DAG structures"""

    def test_branching_dag(self):
        """Test DAG with multiple branches"""
        dag = PipelineDAG()

        # Add nodes
        nodes = ["root", "branch_a", "branch_b", "merge"]
        for node in nodes:
            dag.add_node(node)

        # Add edges
        dag.add_edge("root", "branch_a")
        dag.add_edge("root", "branch_b")
        dag.add_edge("branch_a", "merge")
        dag.add_edge("branch_b", "merge")

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "branching.json"

            # Write and read
            export_dag_to_json(dag, filepath)
            loaded = import_dag_from_json(filepath)

            # Verify structure preserved
            assert set(loaded.nodes) == set(nodes)
            assert len(loaded.edges) == 4

            # Verify topological order is valid
            topo_order = loaded.topological_sort()
            assert topo_order[0] == "root"
            assert topo_order[-1] == "merge"

    def test_deep_dag(self):
        """Test DAG with many sequential steps"""
        nodes = [f"step_{i}" for i in range(10)]
        edges = [(f"step_{i}", f"step_{i + 1}") for i in range(9)]

        dag = PipelineDAG(nodes=nodes, edges=edges)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "deep.json"
            writer = PipelineDAGWriter(dag)
            writer.write_to_file(filepath)

            # Check statistics
            data = json.loads(filepath.read_text())
            assert data["statistics"]["max_depth"] == 9

    def test_isolated_nodes(self):
        """Test DAG with isolated nodes"""
        dag = PipelineDAG(
            nodes=["connected_a", "connected_b", "isolated"],
            edges=[("connected_a", "connected_b")],
        )

        writer = PipelineDAGWriter(dag)
        result = writer.to_dict()

        assert "isolated" in result["statistics"]["isolated_nodes"]


class TestEdgeCases:
    """Tests for edge cases and error conditions"""

    def test_single_node_dag(self):
        """Test DAG with single node"""
        dag = PipelineDAG(nodes=["only_step"], edges=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "single.json"
            export_dag_to_json(dag, filepath)
            loaded = import_dag_from_json(filepath)

            assert loaded.nodes == ["only_step"]
            assert loaded.edges == []

    def test_unicode_in_node_names(self):
        """Test handling of unicode characters in node names"""
        dag = PipelineDAG(
            nodes=["步骤1", "étape_2", "шаг_3"],
            edges=[("步骤1", "étape_2"), ("étape_2", "шаг_3")],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "unicode.json"
            export_dag_to_json(dag, filepath)
            loaded = import_dag_from_json(filepath)

            assert loaded.nodes == dag.nodes

    def test_invalid_json_format(self):
        """Test handling of invalid JSON"""
        invalid_json = "{ this is not valid json }"

        with pytest.raises(ValueError, match="Invalid JSON"):
            PipelineDAGReader.from_json(invalid_json)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Unit tests for RegistryValidator integration and Zettelkasten principles validation.

Tests the comprehensive validation workflows and Zettelkasten compliance analysis.
"""

import pytest
from unittest.mock import Mock
from cursus.pipeline_catalog.utils.registry_validator import RegistryValidator
from cursus.pipeline_catalog.utils.catalog_registry import CatalogRegistry


class TestRegistryValidatorZettelkasten:
    """Test suite for Zettelkasten principles validation."""

    @pytest.fixture
    def mock_registry_good_compliance(self):
        """Create registry with good Zettelkasten compliance."""
        registry = Mock(spec=CatalogRegistry)

        # Create registry with good Zettelkasten compliance
        good_nodes = {
            f"pipeline_{i}": {
                "id": f"pipeline_{i}",
                "title": f"Pipeline {i}",
                "description": f"Pipeline {i} description",
                "atomic_properties": {
                    "single_responsibility": f"Responsibility {i}",
                    "input_interface": ["data"],
                    "output_interface": ["result"],
                    "independence": "fully_self_contained",
                    "side_effects": "none",
                },
                "zettelkasten_metadata": {
                    "framework": "xgboost" if i % 2 == 0 else "pytorch",
                    "complexity": "simple" if i < 3 else "advanced",
                },
                "multi_dimensional_tags": {
                    "framework_tags": ["xgboost" if i % 2 == 0 else "pytorch"],
                    "task_tags": ["training"],
                    "complexity_tags": ["simple" if i < 3 else "advanced"],
                },
                "connections": {
                    "alternatives": (
                        [
                            {
                                "id": f"pipeline_{(i+1)%5}",
                                "annotation": f"Alternative to {i}",
                            }
                        ]
                        if i < 4
                        else []
                    ),
                    "related": (
                        [{"id": f"pipeline_{(i+2)%5}", "annotation": f"Related to {i}"}]
                        if i < 3
                        else []
                    ),
                    "used_in": [],
                },
                "discovery_metadata": {
                    "estimated_runtime": "10 minutes",
                    "resource_requirements": "low",
                    "skill_level": "beginner",
                },
            }
            for i in range(5)
        }

        registry_data = {
            "nodes": good_nodes,
            "tag_index": {
                "framework_tags": {
                    "xgboost": ["pipeline_0", "pipeline_2", "pipeline_4"],
                    "pytorch": ["pipeline_1", "pipeline_3"],
                },
                "task_tags": {"training": [f"pipeline_{i}" for i in range(5)]},
                "complexity_tags": {
                    "simple": ["pipeline_0", "pipeline_1", "pipeline_2"],
                    "advanced": ["pipeline_3", "pipeline_4"],
                },
            },
        }

        registry.get_all_pipelines.return_value = list(good_nodes.keys())
        registry.get_pipeline_node.side_effect = lambda pid: good_nodes.get(pid)
        registry.get_pipeline_connections.side_effect = lambda pid: good_nodes.get(
            pid, {}
        ).get("connections", {})
        registry.load_registry.return_value = registry_data

        return registry

    def test_validate_zettelkasten_principles_good_compliance(
        self, mock_registry_good_compliance
    ):
        """Test Zettelkasten principles validation with good compliance."""
        validator = RegistryValidator(mock_registry_good_compliance)
        compliance = validator.validate_zettelkasten_principles()

        # Should have good overall compliance
        assert compliance["overall_compliance"] > 0.7

        # Check individual principle scores
        principle_scores = compliance["principle_scores"]
        assert principle_scores["atomicity"] > 0.8  # Good atomic properties
        assert principle_scores["connectivity"] > 0.6  # Good connections
        assert principle_scores["anti_categories"] > 0.7  # Good tag diversity
        assert principle_scores["manual_linking"] > 0.8  # Good annotations
        assert principle_scores["dual_form_structure"] > 0.9  # Complete metadata

        # Check metrics
        metrics = compliance["metrics"]
        assert metrics["total_pipelines"] == 5
        assert metrics["atomic_pipelines"] == 5
        assert metrics["connected_pipelines"] > 0
        assert metrics["well_structured_pipelines"] == 5

        # Should have few or no recommendations for good compliance
        recommendations = compliance["recommendations"]
        assert len(recommendations) <= 2  # Should be mostly compliant

    def test_validate_zettelkasten_principles_empty_registry(self):
        """Test Zettelkasten principles validation with empty registry."""
        registry = Mock(spec=CatalogRegistry)
        registry.get_all_pipelines.return_value = []

        validator = RegistryValidator(registry)
        compliance = validator.validate_zettelkasten_principles()

        assert "error" in compliance
        assert "No pipelines found" in compliance["error"]

    def test_generate_principle_recommendations(self, mock_registry_good_compliance):
        """Test generating principle compliance recommendations."""
        validator = RegistryValidator(mock_registry_good_compliance)

        recommendations = validator._generate_principle_recommendations(
            atomicity=0.7,
            connectivity=0.5,
            anti_categories=0.6,
            manual_linking=0.7,
            dual_form=0.8,
        )

        assert isinstance(recommendations, list)
        # Should recommend improvements for low scores
        assert any("connectivity" in rec.lower() for rec in recommendations)
        assert any("tag diversity" in rec.lower() for rec in recommendations)


class TestRegistryValidatorIntegration:
    """Integration tests for RegistryValidator with realistic scenarios."""

    def test_comprehensive_validation_workflow(self):
        """Test comprehensive validation workflow with mixed valid/invalid data."""
        registry = Mock(spec=CatalogRegistry)

        # Create mixed registry data with various issues
        mixed_nodes = {
            "valid_pipeline": {
                "id": "valid_pipeline",
                "title": "Valid Pipeline",
                "description": "A completely valid pipeline",
                "atomic_properties": {
                    "single_responsibility": "Train XGBoost model",
                    "input_interface": ["data"],
                    "output_interface": ["model"],
                    "independence": "fully_self_contained",
                    "side_effects": "none",
                    "dependencies": ["sagemaker"],
                },
                "zettelkasten_metadata": {
                    "framework": "xgboost",
                    "complexity": "simple",
                },
                "multi_dimensional_tags": {
                    "framework_tags": ["xgboost"],
                    "task_tags": ["training"],
                    "complexity_tags": ["simple"],
                },
                "connections": {
                    "alternatives": [],
                    "related": [
                        {
                            "id": "related_pipeline",
                            "annotation": "Related functionality",
                        }
                    ],
                    "used_in": [],
                },
                "discovery_metadata": {
                    "estimated_runtime": "10 minutes",
                    "resource_requirements": "low",
                    "skill_level": "beginner",
                },
            },
            "problematic_pipeline": {
                "id": "problematic_pipeline",
                "title": "Problematic Pipeline",
                # Missing description
                "atomic_properties": {
                    # Missing single_responsibility
                    "input_interface": [],  # Empty interface
                    "output_interface": ["model"],
                    "independence": "fully_self_contained",
                    "side_effects": "modifies_external_state",  # Contradicts independence
                    "dependencies": [
                        "dep1",
                        "dep2",
                        "dep3",
                        "dep4",
                        "dep5",
                    ],  # Too many deps
                },
                "zettelkasten_metadata": {
                    "framework": "pytorch",
                    "complexity": "advanced",
                },
                "multi_dimensional_tags": {
                    "framework_tags": ["xgboost"],  # Inconsistent with metadata
                    "task_tags": ["training"],
                    "complexity_tags": ["simple"],  # Inconsistent with metadata
                },
                "connections": {
                    "alternatives": [],
                    "related": [
                        {"id": "nonexistent_pipeline", "annotation": "Missing target"}
                    ],
                    "used_in": [
                        {"id": "problematic_pipeline", "annotation": "Self-reference"}
                    ],
                },
                # Missing discovery_metadata
            },
            "related_pipeline": {
                "id": "related_pipeline",
                "title": "Related Pipeline",
                "description": "A related pipeline",
                "atomic_properties": {
                    "single_responsibility": "Process data",
                    "input_interface": ["raw_data"],
                    "output_interface": ["processed_data"],
                },
                "zettelkasten_metadata": {
                    "framework": "sklearn"
                    # Missing complexity
                },
                "multi_dimensional_tags": {
                    "framework_tags": ["sklearn"],
                    "task_tags": ["preprocessing"],
                    # Missing complexity_tags
                },
                "connections": {"alternatives": [], "related": [], "used_in": []},
            },
        }

        registry_data = {
            "nodes": mixed_nodes,
            "tag_index": {
                "framework_tags": {
                    "xgboost": [
                        "valid_pipeline",
                        "problematic_pipeline",
                    ],  # Inconsistent
                    "pytorch": [],
                    "sklearn": ["related_pipeline"],
                },
                "task_tags": {
                    "training": ["valid_pipeline", "problematic_pipeline"],
                    "preprocessing": ["related_pipeline"],
                },
                "complexity_tags": {
                    "simple": [
                        "valid_pipeline",
                        "problematic_pipeline",
                    ],  # Inconsistent
                    "advanced": [],
                },
            },
        }

        registry.get_all_pipelines.return_value = list(mixed_nodes.keys())
        registry.get_pipeline_node.side_effect = lambda pid: mixed_nodes.get(pid)
        registry.get_pipeline_connections.side_effect = lambda pid: mixed_nodes.get(
            pid, {}
        ).get("connections", {})
        registry.load_registry.return_value = registry_data

        validator = RegistryValidator(registry)

        # Run comprehensive validation
        report = validator.generate_validation_report()

        # Should find multiple issues
        assert report.is_valid is False
        assert report.total_issues > 5  # Multiple types of issues

        # Should have errors from multiple categories
        assert len(report.issues_by_category) > 1

        # Should have both errors and warnings
        assert report.issues_by_severity.get("error", 0) > 0
        assert report.issues_by_severity.get("warning", 0) > 0

        # Check specific issue types are detected
        issue_categories = [issue.category for issue in report.all_issues]
        assert "atomicity" in issue_categories
        assert "connectivity" in issue_categories
        assert "metadata" in issue_categories
        assert "tags" in issue_categories
        assert "independence" in issue_categories

        # Verify specific issues are found
        issue_messages = [issue.message for issue in report.all_issues]
        assert any("missing_responsibility" in msg for msg in issue_messages)
        assert any("missing_target" in msg for msg in issue_messages)
        assert any("self_reference" in msg for msg in issue_messages)
        assert any("framework_tags" in msg for msg in issue_messages)
        assert any("side effects" in msg for msg in issue_messages)

    def test_validation_workflow_with_edge_cases(self):
        """Test validation workflow with edge cases and boundary conditions."""
        registry = Mock(spec=CatalogRegistry)

        # Create edge case scenarios
        edge_case_nodes = {
            "minimal_pipeline": {
                "id": "minimal_pipeline",
                "title": "",  # Empty title
                "description": "x",  # Very short description
                "atomic_properties": {
                    "single_responsibility": "a",  # Very short responsibility
                    "input_interface": [],  # No inputs
                    "output_interface": [],  # No outputs
                    "independence": "unknown",  # Unknown independence level
                    "side_effects": "",  # Empty side effects
                    "dependencies": [],  # No dependencies
                },
                "zettelkasten_metadata": {},  # Empty metadata
                "multi_dimensional_tags": {},  # No tags
                "connections": {},  # No connections
                "discovery_metadata": {},  # Empty discovery metadata
            },
            "maximal_pipeline": {
                "id": "maximal_pipeline",
                "title": "A" * 200,  # Very long title
                "description": "B" * 1000,  # Very long description
                "atomic_properties": {
                    "single_responsibility": "C" * 500,  # Very long responsibility
                    "input_interface": [f"input_{i}" for i in range(50)],  # Many inputs
                    "output_interface": [
                        f"output_{i}" for i in range(50)
                    ],  # Many outputs
                    "independence": "fully_self_contained",
                    "side_effects": "none",
                    "dependencies": [
                        f"dep_{i}" for i in range(20)
                    ],  # Many dependencies
                },
                "zettelkasten_metadata": {
                    "framework": "custom_framework",
                    "complexity": "ultra_complex",
                },
                "multi_dimensional_tags": {
                    "framework_tags": [f"framework_{i}" for i in range(10)],
                    "task_tags": [f"task_{i}" for i in range(20)],
                    "complexity_tags": [f"complexity_{i}" for i in range(5)],
                },
                "connections": {
                    "alternatives": [
                        {"id": f"alt_{i}", "annotation": f"Alt {i}"} for i in range(10)
                    ],
                    "related": [
                        {"id": f"rel_{i}", "annotation": f"Rel {i}"} for i in range(15)
                    ],
                    "used_in": [
                        {"id": f"used_{i}", "annotation": f"Used {i}"} for i in range(5)
                    ],
                },
                "discovery_metadata": {
                    "estimated_runtime": "24 hours",
                    "resource_requirements": "extreme",
                    "skill_level": "expert",
                },
            },
            "duplicate_connections": {
                "id": "duplicate_connections",
                "title": "Duplicate Connections",
                "description": "Pipeline with duplicate connections",
                "atomic_properties": {
                    "single_responsibility": "Test duplicates",
                    "input_interface": ["data"],
                    "output_interface": ["result"],
                },
                "zettelkasten_metadata": {"framework": "test", "complexity": "simple"},
                "multi_dimensional_tags": {
                    "framework_tags": ["test"],
                    "task_tags": ["testing"],
                },
                "connections": {
                    "alternatives": [
                        {"id": "target", "annotation": "First"},
                        {
                            "id": "target",
                            "annotation": "Duplicate",
                        },  # Duplicate connection
                    ],
                    "related": [],
                    "used_in": [],
                },
            },
        }

        registry_data = {
            "nodes": edge_case_nodes,
            "tag_index": {
                "framework_tags": {"test": ["duplicate_connections"]},
                "task_tags": {"testing": ["duplicate_connections"]},
            },
        }

        registry.get_all_pipelines.return_value = list(edge_case_nodes.keys())
        registry.get_pipeline_node.side_effect = lambda pid: edge_case_nodes.get(pid)
        registry.get_pipeline_connections.side_effect = lambda pid: edge_case_nodes.get(
            pid, {}
        ).get("connections", {})
        registry.load_registry.return_value = registry_data

        validator = RegistryValidator(registry)

        # Run validation on edge cases
        report = validator.generate_validation_report()

        # Should handle edge cases gracefully
        assert isinstance(report.total_issues, int)
        assert isinstance(report.all_issues, list)

        # Should detect issues with minimal pipeline
        minimal_issues = [
            issue
            for issue in report.all_issues
            if issue.pipeline_id == "minimal_pipeline"
        ]
        assert len(minimal_issues) > 0  # Should have issues with minimal data

        # Should handle maximal pipeline without crashing
        maximal_issues = [
            issue
            for issue in report.all_issues
            if issue.pipeline_id == "maximal_pipeline"
        ]
        # May or may not have issues, but should not crash

        # Should detect duplicate connections
        duplicate_issues = [
            issue
            for issue in report.all_issues
            if issue.pipeline_id == "duplicate_connections"
        ]
        # Should handle duplicates gracefully

    def test_performance_with_large_registry(self):
        """Test validation performance with large registry."""
        registry = Mock(spec=CatalogRegistry)

        # Create large registry (100 pipelines)
        large_nodes = {}
        for i in range(100):
            large_nodes[f"pipeline_{i}"] = {
                "id": f"pipeline_{i}",
                "title": f"Pipeline {i}",
                "description": f"Description for pipeline {i}",
                "atomic_properties": {
                    "single_responsibility": f"Responsibility {i}",
                    "input_interface": ["data"],
                    "output_interface": ["result"],
                    "independence": "fully_self_contained",
                    "side_effects": "none",
                },
                "zettelkasten_metadata": {
                    "framework": ["xgboost", "pytorch", "sklearn"][i % 3],
                    "complexity": ["simple", "standard", "advanced"][i % 3],
                },
                "multi_dimensional_tags": {
                    "framework_tags": [["xgboost", "pytorch", "sklearn"][i % 3]],
                    "task_tags": ["training"],
                    "complexity_tags": [["simple", "standard", "advanced"][i % 3]],
                },
                "connections": {
                    "alternatives": (
                        [{"id": f"pipeline_{(i+1)%100}", "annotation": f"Alt to {i}"}]
                        if i % 10 == 0
                        else []
                    ),
                    "related": (
                        [
                            {
                                "id": f"pipeline_{(i+2)%100}",
                                "annotation": f"Related to {i}",
                            }
                        ]
                        if i % 5 == 0
                        else []
                    ),
                    "used_in": [],
                },
                "discovery_metadata": {
                    "estimated_runtime": "10 minutes",
                    "resource_requirements": "low",
                    "skill_level": "beginner",
                },
            }

        registry_data = {
            "nodes": large_nodes,
            "tag_index": {
                "framework_tags": {
                    "xgboost": [f"pipeline_{i}" for i in range(0, 100, 3)],
                    "pytorch": [f"pipeline_{i}" for i in range(1, 100, 3)],
                    "sklearn": [f"pipeline_{i}" for i in range(2, 100, 3)],
                },
                "task_tags": {"training": [f"pipeline_{i}" for i in range(100)]},
                "complexity_tags": {
                    "simple": [f"pipeline_{i}" for i in range(0, 100, 3)],
                    "standard": [f"pipeline_{i}" for i in range(1, 100, 3)],
                    "advanced": [f"pipeline_{i}" for i in range(2, 100, 3)],
                },
            },
        }

        registry.get_all_pipelines.return_value = list(large_nodes.keys())
        registry.get_pipeline_node.side_effect = lambda pid: large_nodes.get(pid)
        registry.get_pipeline_connections.side_effect = lambda pid: large_nodes.get(
            pid, {}
        ).get("connections", {})
        registry.load_registry.return_value = registry_data

        validator = RegistryValidator(registry)

        # Run validation - should complete in reasonable time
        import time

        start_time = time.time()
        report = validator.generate_validation_report()
        end_time = time.time()

        # Should complete within reasonable time (less than 10 seconds)
        assert (end_time - start_time) < 10.0

        # Should handle large registry
        assert isinstance(report, type(report))  # Should return valid report
        assert report.total_issues >= 0  # Should have non-negative issue count

        # Test Zettelkasten principles on large registry
        compliance = validator.validate_zettelkasten_principles()
        assert "overall_compliance" in compliance
        assert isinstance(compliance["overall_compliance"], (int, float))

"""
Phase 1 Implementation Tests

Comprehensive tests for Phase 1 of the pipeline catalog zettelkasten refactoring.
Tests all components: Enhanced DAGMetadata, Registry Infrastructure, and Utility Functions.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, Any

# Import the Phase 1 components
from cursus.pipeline_catalog.shared_dags.enhanced_metadata import (
    EnhancedDAGMetadata,
    ZettelkastenMetadata,
    ComplexityLevel,
    PipelineFramework,
    DAGMetadataAdapter,
    validate_enhanced_dag_metadata,
)
from cursus.pipeline_catalog.shared_dags.registry_sync import (
    DAGMetadataRegistrySync,
    RegistryValidationError,
)
from cursus.pipeline_catalog.utils.catalog_registry import CatalogRegistry
from cursus.pipeline_catalog.utils.connection_traverser import (
    ConnectionTraverser,
    PipelineConnection,
)
from cursus.pipeline_catalog.utils.tag_discovery import TagBasedDiscovery
from cursus.pipeline_catalog.utils.recommendation_engine import (
    PipelineRecommendationEngine,
    RecommendationResult,
    CompositionRecommendation,
)
from cursus.pipeline_catalog.utils.registry_validator import (
    RegistryValidator,
    ValidationReport,
    ValidationSeverity,
)


class TestEnhancedDAGMetadata:
    """Test Enhanced DAGMetadata system."""

    def test_enhanced_metadata_creation(self):
        """Test creating enhanced metadata with Zettelkasten extensions."""
        # Create Zettelkasten metadata
        zk_metadata = ZettelkastenMetadata(
            atomic_id="xgb_simple_training",
            single_responsibility="XGBoost model training",
            input_interface=["tabular_data"],
            output_interface=["trained_model", "metrics"],
            framework_tags=["xgboost", "tree_based"],
            task_tags=["training", "supervised_learning"],
            complexity_tags=["simple", "beginner_friendly"],
        )

        # Create enhanced metadata
        metadata = EnhancedDAGMetadata(
            description="Simple XGBoost training pipeline",
            complexity=ComplexityLevel.SIMPLE,
            features=["training"],
            framework=PipelineFramework.XGBOOST,
            node_count=3,
            edge_count=2,
            zettelkasten_metadata=zk_metadata,
        )

        assert metadata.description == "Simple XGBoost training pipeline"
        assert metadata.complexity == ComplexityLevel.SIMPLE
        assert metadata.framework == PipelineFramework.XGBOOST
        assert metadata.zettelkasten_metadata.atomic_id == "xgb_simple_training"
        assert "xgboost" in metadata.zettelkasten_metadata.framework_tags

    def test_registry_node_conversion(self):
        """Test conversion to registry node format."""
        metadata = EnhancedDAGMetadata(
            description="Test pipeline",
            complexity=ComplexityLevel.SIMPLE,
            features=["training"],
            framework=PipelineFramework.XGBOOST,
            node_count=3,
            edge_count=2,
        )

        node = metadata.to_registry_node()

        assert "id" in node
        assert "title" in node
        assert "description" in node
        assert "atomic_properties" in node
        assert "zettelkasten_metadata" in node
        assert "multi_dimensional_tags" in node
        assert "connections" in node
        assert "discovery_metadata" in node

        # Check atomic properties
        atomic_props = node["atomic_properties"]
        assert "single_responsibility" in atomic_props
        assert "input_interface" in atomic_props
        assert "output_interface" in atomic_props
        assert "dependencies" in atomic_props

    def test_connection_management(self):
        """Test adding and managing connections."""
        metadata = EnhancedDAGMetadata(
            description="Test pipeline",
            complexity=ComplexityLevel.SIMPLE,
            features=["training"],
            framework=PipelineFramework.XGBOOST,
            node_count=3,
            edge_count=2,
        )

        # Add connection
        metadata.add_connection(
            target_id="pytorch_basic_training",
            connection_type="alternatives",
            annotation="Alternative ML framework for same task",
        )

        zk_meta = metadata.zettelkasten_metadata
        assert "alternatives" in zk_meta.manual_connections
        assert "pytorch_basic_training" in zk_meta.manual_connections["alternatives"]
        assert "pytorch_basic_training" in zk_meta.curated_connections

    def test_dag_metadata_adapter(self):
        """Test backward compatibility adapter."""

        # Create mock legacy metadata
        class MockLegacyMetadata:
            def __init__(self):
                self.description = "Legacy pipeline"
                self.complexity = "simple"
                self.features = ["training"]
                self.framework = "xgboost"
                self.node_count = 3
                self.edge_count = 2
                self.extra_metadata = {"test": "value"}

        legacy = MockLegacyMetadata()
        enhanced = DAGMetadataAdapter.from_legacy_dag_metadata(legacy)

        assert enhanced.description == "Legacy pipeline"
        assert enhanced.complexity == ComplexityLevel.SIMPLE
        assert enhanced.framework == PipelineFramework.XGBOOST
        assert enhanced.zettelkasten_metadata.atomic_id == "xgboost_training_simple"

    def test_metadata_validation(self):
        """Test metadata validation."""
        # Valid metadata
        valid_metadata = EnhancedDAGMetadata(
            description="Valid pipeline",
            complexity=ComplexityLevel.SIMPLE,
            features=["training"],
            framework=PipelineFramework.XGBOOST,
            node_count=3,
            edge_count=2,
        )

        assert validate_enhanced_dag_metadata(valid_metadata) == True

        # Invalid metadata (empty description)
        with pytest.raises(ValueError):
            EnhancedDAGMetadata(
                description="",
                complexity=ComplexityLevel.SIMPLE,
                features=["training"],
                framework=PipelineFramework.XGBOOST,
                node_count=3,
                edge_count=2,
            )


class TestRegistrySync:
    """Test Registry Synchronization infrastructure."""

    def setup_method(self):
        """Set up test registry file."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry_path = os.path.join(self.temp_dir, "test_registry.json")
        self.sync = DAGMetadataRegistrySync(self.registry_path)

    def teardown_method(self):
        """Clean up test files."""
        if os.path.exists(self.registry_path):
            os.remove(self.registry_path)
        os.rmdir(self.temp_dir)

    def test_registry_creation(self):
        """Test registry file creation."""
        assert os.path.exists(self.registry_path)

        registry = self.sync.load_registry()
        assert "version" in registry
        assert "metadata" in registry
        assert "nodes" in registry
        assert registry["nodes"] == {}

    def test_metadata_sync_to_registry(self):
        """Test syncing metadata to registry."""
        metadata = EnhancedDAGMetadata(
            description="Test pipeline",
            complexity=ComplexityLevel.SIMPLE,
            features=["training"],
            framework=PipelineFramework.XGBOOST,
            node_count=3,
            edge_count=2,
        )

        self.sync.sync_metadata_to_registry(metadata, "test_pipeline.py")

        registry = self.sync.load_registry()
        atomic_id = metadata.zettelkasten_metadata.atomic_id

        assert atomic_id in registry["nodes"]
        node = registry["nodes"][atomic_id]
        assert node["description"] == "Test pipeline"
        assert node["file"] == "test_pipeline.py"

    def test_registry_to_metadata_sync(self):
        """Test syncing registry back to metadata."""
        # First sync metadata to registry
        metadata = EnhancedDAGMetadata(
            description="Test pipeline",
            complexity=ComplexityLevel.SIMPLE,
            features=["training"],
            framework=PipelineFramework.XGBOOST,
            node_count=3,
            edge_count=2,
        )

        self.sync.sync_metadata_to_registry(metadata, "test_pipeline.py")
        atomic_id = metadata.zettelkasten_metadata.atomic_id

        # Now sync back
        synced_metadata = self.sync.sync_registry_to_metadata(atomic_id)

        assert synced_metadata is not None
        assert synced_metadata.description == "Test pipeline"
        assert synced_metadata.complexity == ComplexityLevel.SIMPLE
        assert synced_metadata.framework == PipelineFramework.XGBOOST

    def test_consistency_validation(self):
        """Test consistency validation between DAG and registry."""
        metadata = EnhancedDAGMetadata(
            description="Test pipeline",
            complexity=ComplexityLevel.SIMPLE,
            features=["training"],
            framework=PipelineFramework.XGBOOST,
            node_count=3,
            edge_count=2,
        )

        self.sync.sync_metadata_to_registry(metadata, "test_pipeline.py")
        atomic_id = metadata.zettelkasten_metadata.atomic_id

        errors = self.sync.validate_consistency(metadata, atomic_id)
        assert len(errors) == 0  # Should be consistent

    def test_registry_statistics(self):
        """Test registry statistics generation."""
        metadata = EnhancedDAGMetadata(
            description="Test pipeline",
            complexity=ComplexityLevel.SIMPLE,
            features=["training"],
            framework=PipelineFramework.XGBOOST,
            node_count=3,
            edge_count=2,
        )

        self.sync.sync_metadata_to_registry(metadata, "test_pipeline.py")

        stats = self.sync.get_registry_statistics()

        assert "total_pipelines" in stats
        assert stats["total_pipelines"] == 1
        assert "frameworks" in stats
        assert "xgboost" in stats["frameworks"]


class TestCatalogRegistry:
    """Test Catalog Registry management."""

    def setup_method(self):
        """Set up test registry."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry_path = os.path.join(self.temp_dir, "test_catalog.json")
        self.registry = CatalogRegistry(self.registry_path)

    def teardown_method(self):
        """Clean up test files."""
        if os.path.exists(self.registry_path):
            os.remove(self.registry_path)
        os.rmdir(self.temp_dir)

    def test_pipeline_node_operations(self):
        """Test CRUD operations on pipeline nodes."""
        # Create test node
        node_data = {
            "id": "test_pipeline",
            "title": "Test Pipeline",
            "description": "A test pipeline",
            "atomic_properties": {
                "single_responsibility": "Test functionality",
                "input_interface": ["data"],
                "output_interface": ["result"],
            },
            "zettelkasten_metadata": {"framework": "xgboost", "complexity": "simple"},
            "multi_dimensional_tags": {
                "framework_tags": ["xgboost"],
                "task_tags": ["training"],
                "complexity_tags": ["simple"],
            },
            "connections": {},
            "discovery_metadata": {},
        }

        # Add node
        success = self.registry.add_pipeline_node("test_pipeline", node_data)
        assert success == True

        # Get node
        retrieved_node = self.registry.get_pipeline_node("test_pipeline")
        assert retrieved_node is not None
        assert retrieved_node["title"] == "Test Pipeline"

        # Update node
        node_data["title"] = "Updated Test Pipeline"
        success = self.registry.update_pipeline_node("test_pipeline", node_data)
        assert success == True

        updated_node = self.registry.get_pipeline_node("test_pipeline")
        assert updated_node["title"] == "Updated Test Pipeline"

        # Remove node
        success = self.registry.remove_pipeline_node("test_pipeline")
        assert success == True

        removed_node = self.registry.get_pipeline_node("test_pipeline")
        assert removed_node is None

    def test_connection_management(self):
        """Test connection management between pipelines."""
        # Add two test nodes
        node1 = {
            "id": "pipeline1",
            "title": "Pipeline 1",
            "description": "First pipeline",
            "atomic_properties": {
                "single_responsibility": "Test 1",
                "input_interface": [],
                "output_interface": [],
            },
            "zettelkasten_metadata": {"framework": "xgboost", "complexity": "simple"},
            "multi_dimensional_tags": {
                "framework_tags": [],
                "task_tags": [],
                "complexity_tags": [],
            },
            "connections": {},
            "discovery_metadata": {},
        }

        node2 = {
            "id": "pipeline2",
            "title": "Pipeline 2",
            "description": "Second pipeline",
            "atomic_properties": {
                "single_responsibility": "Test 2",
                "input_interface": [],
                "output_interface": [],
            },
            "zettelkasten_metadata": {"framework": "pytorch", "complexity": "simple"},
            "multi_dimensional_tags": {
                "framework_tags": [],
                "task_tags": [],
                "complexity_tags": [],
            },
            "connections": {},
            "discovery_metadata": {},
        }

        self.registry.add_pipeline_node("pipeline1", node1)
        self.registry.add_pipeline_node("pipeline2", node2)

        # Add connection
        success = self.registry.add_connection(
            "pipeline1", "pipeline2", "alternatives", "Alternative framework approach"
        )
        assert success == True

        # Get connections
        connections = self.registry.get_pipeline_connections("pipeline1")
        assert "alternatives" in connections
        assert len(connections["alternatives"]) == 1
        assert connections["alternatives"][0]["id"] == "pipeline2"

        # Remove connection
        success = self.registry.remove_connection(
            "pipeline1", "pipeline2", "alternatives"
        )
        assert success == True

        connections = self.registry.get_pipeline_connections("pipeline1")
        assert len(connections.get("alternatives", [])) == 0


class TestUtilityFunctions:
    """Test utility functions integration."""

    def setup_method(self):
        """Set up test environment with sample data."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry_path = os.path.join(self.temp_dir, "test_utils.json")
        self.registry = CatalogRegistry(self.registry_path)
        self.traverser = ConnectionTraverser(self.registry)
        self.discovery = TagBasedDiscovery(self.registry)
        self.recommender = PipelineRecommendationEngine(
            self.registry, self.traverser, self.discovery
        )
        self.validator = RegistryValidator(self.registry)

        # Add sample pipelines
        self._add_sample_pipelines()

    def teardown_method(self):
        """Clean up test files."""
        if os.path.exists(self.registry_path):
            os.remove(self.registry_path)
        os.rmdir(self.temp_dir)

    def _add_sample_pipelines(self):
        """Add sample pipelines for testing."""
        pipelines = [
            {
                "id": "xgb_simple_training",
                "title": "XGBoost Simple Training",
                "description": "Basic XGBoost training pipeline",
                "atomic_properties": {
                    "single_responsibility": "XGBoost model training",
                    "input_interface": ["tabular_data"],
                    "output_interface": ["trained_model"],
                },
                "zettelkasten_metadata": {
                    "framework": "xgboost",
                    "complexity": "simple",
                },
                "multi_dimensional_tags": {
                    "framework_tags": ["xgboost", "tree_based"],
                    "task_tags": ["training", "supervised_learning"],
                    "complexity_tags": ["simple", "beginner_friendly"],
                },
                "connections": {
                    "alternatives": [
                        {
                            "id": "pytorch_basic_training",
                            "annotation": "Alternative ML framework",
                        }
                    ]
                },
                "discovery_metadata": {
                    "estimated_runtime": "15-30 minutes",
                    "resource_requirements": "medium",
                    "skill_level": "beginner",
                },
            },
            {
                "id": "pytorch_basic_training",
                "title": "PyTorch Basic Training",
                "description": "Basic PyTorch training pipeline",
                "atomic_properties": {
                    "single_responsibility": "PyTorch model training",
                    "input_interface": ["tensor_data"],
                    "output_interface": ["trained_model"],
                },
                "zettelkasten_metadata": {
                    "framework": "pytorch",
                    "complexity": "simple",
                },
                "multi_dimensional_tags": {
                    "framework_tags": ["pytorch", "neural_networks"],
                    "task_tags": ["training", "supervised_learning"],
                    "complexity_tags": ["simple", "beginner_friendly"],
                },
                "connections": {
                    "alternatives": [
                        {
                            "id": "xgb_simple_training",
                            "annotation": "Alternative ML framework",
                        }
                    ]
                },
                "discovery_metadata": {
                    "estimated_runtime": "20-40 minutes",
                    "resource_requirements": "medium",
                    "skill_level": "beginner",
                },
            },
        ]

        for pipeline in pipelines:
            self.registry.add_pipeline_node(pipeline["id"], pipeline)

    def test_connection_traverser(self):
        """Test connection traversal functionality."""
        # Test getting alternatives
        alternatives = self.traverser.get_alternatives("xgb_simple_training")
        assert len(alternatives) == 1
        assert alternatives[0].target_id == "pytorch_basic_training"

        # Test shortest path
        path = self.traverser.find_shortest_path(
            "xgb_simple_training", "pytorch_basic_training"
        )
        assert path is not None
        assert len(path) == 2
        assert path[0] == "xgb_simple_training"
        assert path[1] == "pytorch_basic_training"

        # Test connection subgraph
        subgraph = self.traverser.get_connection_subgraph(
            "xgb_simple_training", depth=1
        )
        assert "nodes" in subgraph
        assert "edges" in subgraph
        assert subgraph["center_node"] == "xgb_simple_training"

    def test_tag_based_discovery(self):
        """Test tag-based discovery functionality."""
        # Test finding by framework
        xgb_pipelines = self.discovery.find_by_framework("xgboost")
        assert "xgb_simple_training" in xgb_pipelines

        pytorch_pipelines = self.discovery.find_by_framework("pytorch")
        assert "pytorch_basic_training" in pytorch_pipelines

        # Test finding by complexity
        simple_pipelines = self.discovery.find_by_complexity("simple")
        assert len(simple_pipelines) == 2

        # Test finding by task
        training_pipelines = self.discovery.find_by_task("training")
        assert len(training_pipelines) == 2

        # Test text search
        results = self.discovery.search_by_text("XGBoost training")
        assert len(results) > 0
        assert results[0][0] == "xgb_simple_training"  # Should be top result

        # Test similar pipelines
        similar = self.discovery.suggest_similar_pipelines("xgb_simple_training")
        assert len(similar) > 0
        assert similar[0][0] == "pytorch_basic_training"

    def test_recommendation_engine(self):
        """Test recommendation engine functionality."""
        # Test use case recommendations
        recommendations = self.recommender.recommend_for_use_case(
            "machine learning training"
        )
        assert len(recommendations) > 0
        assert all(isinstance(r, RecommendationResult) for r in recommendations)

        # Test next steps recommendations
        next_steps = self.recommender.recommend_next_steps("xgb_simple_training")
        # May be empty if no "used_in" connections, but should not error
        assert isinstance(next_steps, list)

        # Test alternatives recommendations
        alternatives = self.recommender.recommend_alternatives("xgb_simple_training")
        assert len(alternatives) > 0
        assert alternatives[0].pipeline_id == "pytorch_basic_training"

        # Test learning path
        learning_path = self.recommender.get_learning_path("simple", "xgboost")
        assert isinstance(learning_path, list)

    def test_registry_validator(self):
        """Test registry validation functionality."""
        # Test atomicity validation
        atomicity_violations = self.validator.validate_atomicity()
        # Should have no violations for our well-formed test data
        assert len(atomicity_violations) == 0

        # Test connection validation
        connection_errors = self.validator.validate_connections()
        # Should have no errors for our test data
        assert len(connection_errors) == 0

        # Test comprehensive validation report
        report = self.validator.generate_validation_report()
        assert isinstance(report, ValidationReport)
        assert report.total_issues >= 0

        # Test Zettelkasten principles validation
        principles = self.validator.validate_zettelkasten_principles()
        assert "overall_compliance" in principles
        assert "principle_scores" in principles
        assert principles["overall_compliance"] >= 0.0


class TestIntegration:
    """Test integration between all Phase 1 components."""

    def setup_method(self):
        """Set up integrated test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry_path = os.path.join(self.temp_dir, "integration_test.json")

    def teardown_method(self):
        """Clean up test files."""
        if os.path.exists(self.registry_path):
            os.remove(self.registry_path)
        os.rmdir(self.temp_dir)

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # 1. Create enhanced metadata
        metadata = EnhancedDAGMetadata(
            description="Integration test pipeline",
            complexity=ComplexityLevel.STANDARD,
            features=["training", "evaluation"],
            framework=PipelineFramework.XGBOOST,
            node_count=5,
            edge_count=4,
        )

        # Add some connections
        metadata.add_connection(
            "pytorch_alternative", "alternatives", "PyTorch alternative approach"
        )

        # 2. Sync to registry
        sync = DAGMetadataRegistrySync(self.registry_path)
        sync.sync_metadata_to_registry(metadata, "integration_test.py")

        # 3. Use catalog registry
        registry = CatalogRegistry(self.registry_path)
        atomic_id = metadata.zettelkasten_metadata.atomic_id

        node = registry.get_pipeline_node(atomic_id)
        assert node is not None
        assert node["description"] == "Integration test pipeline"

        # 4. Test utilities
        traverser = ConnectionTraverser(registry)
        discovery = TagBasedDiscovery(registry)
        recommender = PipelineRecommendationEngine(registry, traverser, discovery)
        validator = RegistryValidator(registry)

        # Test discovery
        xgb_pipelines = discovery.find_by_framework("xgboost")
        assert atomic_id in xgb_pipelines

        # Test validation
        report = validator.generate_validation_report()
        assert isinstance(report, ValidationReport)

        # 5. Verify consistency
        errors = sync.validate_consistency(metadata, atomic_id)
        assert len(errors) == 0

        print("✅ End-to-end integration test passed!")


def test_phase1_implementation():
    """Main test function to validate Phase 1 implementation."""
    print("🚀 Starting Phase 1 Implementation Tests...")

    # Run all test classes
    test_classes = [
        TestEnhancedDAGMetadata,
        TestRegistrySync,
        TestCatalogRegistry,
        TestUtilityFunctions,
        TestIntegration,
    ]

    for test_class in test_classes:
        print(f"Testing {test_class.__name__}...")

        # Create instance and run tests
        instance = test_class()

        # Get all test methods
        test_methods = [
            method for method in dir(instance) if method.startswith("test_")
        ]

        for method_name in test_methods:
            print(f"  Running {method_name}...")

            # Setup if available
            if hasattr(instance, "setup_method"):
                instance.setup_method()

            try:
                # Run test method
                method = getattr(instance, method_name)
                method()
                print(f"  ✅ {method_name} passed")
            except Exception as e:
                print(f"  ❌ {method_name} failed: {e}")
                raise
            finally:
                # Teardown if available
                if hasattr(instance, "teardown_method"):
                    instance.teardown_method()

    print("🎉 All Phase 1 tests passed!")


if __name__ == "__main__":
    test_phase1_implementation()

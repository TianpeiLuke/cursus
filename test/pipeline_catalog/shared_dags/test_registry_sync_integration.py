"""
Integration tests for registry synchronization infrastructure.

Tests complex workflows and realistic scenarios for the DAGMetadataRegistrySync
system including complete workflows, multi-pipeline scenarios, and error recovery.
"""

import pytest
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock, mock_open
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from src.cursus.pipeline_catalog.shared_dags.registry_sync import (
    DAGMetadataRegistrySync, RegistryValidationError,
    create_empty_registry, validate_registry_file
)
from src.cursus.pipeline_catalog.shared_dags.enhanced_metadata import (
    EnhancedDAGMetadata, ZettelkastenMetadata, ComplexityLevel, PipelineFramework
)


class TestRegistrySyncIntegration:
    """Integration tests for registry synchronization with realistic scenarios."""
    
    @pytest.fixture
    def temp_registry_path(self):
        """Create temporary registry path for integration testing."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        yield temp_path
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)
    
    def test_complete_sync_workflow(self, temp_registry_path):
        """Test complete synchronization workflow."""
        sync = DAGMetadataRegistrySync(temp_registry_path)
        
        # Create enhanced metadata
        enhanced = EnhancedDAGMetadata(
            description="Complete XGBoost training pipeline",
            complexity=ComplexityLevel.ADVANCED,
            features=["training", "evaluation"],
            framework=PipelineFramework.XGBOOST,
            node_count=5,
            edge_count=4
        )
        
        # Customize Zettelkasten metadata
        zm = enhanced.zettelkasten_metadata
        zm.title = "Advanced XGBoost Pipeline"
        zm.domain_tags = ["tabular", "supervised_learning"]
        zm.pattern_tags = ["end_to_end", "production_ready"]
        zm.source_file = "xgboost/advanced_pipeline.py"
        zm.priority = "high"
        
        # Add connections
        enhanced.add_connection(
            target_id="simple_xgb_training",
            connection_type="alternatives",
            annotation="Simpler alternative",
            confidence=0.9
        )
        
        # Sync to registry
        sync.sync_metadata_to_registry(enhanced, "pipelines/xgboost_advanced.py")
        
        # Verify registry was updated
        registry = sync.load_registry()
        atomic_id = enhanced.zettelkasten_metadata.atomic_id
        
        assert atomic_id in registry["nodes"]
        node = registry["nodes"][atomic_id]
        assert node["title"] == "Advanced XGBoost Pipeline"
        assert node["file"] == "pipelines/xgboost_advanced.py"
        assert node["priority"] == "high"
        
        # Verify connections
        connections = node["connections"]
        assert len(connections["alternatives"]) == 1
        assert connections["alternatives"][0]["id"] == "simple_xgb_training"
        assert connections["alternatives"][0]["annotation"] == "Simpler alternative"
        
        # Verify tag index was updated
        tag_index = registry["tag_index"]
        assert atomic_id in tag_index["framework_tags"]["xgboost"]
        assert atomic_id in tag_index["task_tags"]["training"]
        assert atomic_id in tag_index["domain_tags"]["tabular"]
        
        # Test round-trip sync
        synced_back = sync.sync_registry_to_metadata(atomic_id)
        assert synced_back is not None
        assert synced_back.description == enhanced.description
        assert synced_back.framework == enhanced.framework
        assert synced_back.complexity == enhanced.complexity
        
        # Test consistency validation
        errors = sync.validate_consistency(enhanced, atomic_id)
        assert len(errors) == 0
        
        # Test registry statistics
        stats = sync.get_registry_statistics()
        assert stats["total_pipelines"] == 1
        assert "xgboost" in stats["frameworks"]
        assert "advanced" in stats["complexity_levels"]
        assert stats["total_connections"] == 1  # One connection added
        
        # Test pipeline removal
        removed = sync.remove_pipeline_from_registry(atomic_id)
        assert removed is True
        
        # Verify removal
        final_registry = sync.load_registry()
        assert atomic_id not in final_registry["nodes"]
        assert final_registry["metadata"]["total_pipelines"] == 0
    
    def test_multi_pipeline_sync_workflow(self, temp_registry_path):
        """Test synchronization workflow with multiple pipelines."""
        sync = DAGMetadataRegistrySync(temp_registry_path)
        
        # Create multiple enhanced metadata instances
        pipelines = []
        
        # XGBoost pipeline
        xgb_pipeline = EnhancedDAGMetadata(
            description="XGBoost training pipeline",
            complexity=ComplexityLevel.SIMPLE,
            features=["training"],
            framework=PipelineFramework.XGBOOST,
            node_count=3,
            edge_count=2
        )
        xgb_pipeline.zettelkasten_metadata.domain_tags = ["tabular"]
        pipelines.append(("xgb_pipeline", xgb_pipeline))
        
        # PyTorch pipeline
        pytorch_pipeline = EnhancedDAGMetadata(
            description="PyTorch NLP training pipeline",
            complexity=ComplexityLevel.STANDARD,
            features=["preprocessing", "training"],
            framework=PipelineFramework.PYTORCH,
            node_count=6,
            edge_count=5
        )
        pytorch_pipeline.zettelkasten_metadata.domain_tags = ["nlp"]
        pipelines.append(("pytorch_pipeline", pytorch_pipeline))
        
        # Sklearn pipeline
        sklearn_pipeline = EnhancedDAGMetadata(
            description="Sklearn evaluation pipeline",
            complexity=ComplexityLevel.ADVANCED,
            features=["evaluation", "calibration"],
            framework=PipelineFramework.SKLEARN,
            node_count=4,
            edge_count=3
        )
        sklearn_pipeline.zettelkasten_metadata.domain_tags = ["evaluation"]
        pipelines.append(("sklearn_pipeline", sklearn_pipeline))
        
        # Sync all pipelines
        for file_name, pipeline in pipelines:
            sync.sync_metadata_to_registry(pipeline, f"pipelines/{file_name}.py")
        
        # Add cross-connections
        xgb_pipeline.add_connection(
            target_id=pytorch_pipeline.zettelkasten_metadata.atomic_id,
            connection_type="alternatives",
            annotation="Alternative framework for different data types"
        )
        
        pytorch_pipeline.add_connection(
            target_id=sklearn_pipeline.zettelkasten_metadata.atomic_id,
            connection_type="used_in",
            annotation="Evaluation of trained models"
        )
        
        # Re-sync with connections
        sync.sync_metadata_to_registry(xgb_pipeline, "pipelines/xgb_pipeline.py")
        sync.sync_metadata_to_registry(pytorch_pipeline, "pipelines/pytorch_pipeline.py")
        
        # Verify registry state
        registry = sync.load_registry()
        assert len(registry["nodes"]) == 3
        
        # Verify metadata statistics
        metadata = registry["metadata"]
        assert metadata["total_pipelines"] == 3
        assert set(metadata["frameworks"]) == {"xgboost", "pytorch", "sklearn"}
        assert set(metadata["complexity_levels"]) == {"simple", "standard", "advanced"}
        
        # Verify connections
        xgb_node = registry["nodes"][xgb_pipeline.zettelkasten_metadata.atomic_id]
        assert len(xgb_node["connections"]["alternatives"]) == 1
        
        pytorch_node = registry["nodes"][pytorch_pipeline.zettelkasten_metadata.atomic_id]
        assert len(pytorch_node["connections"]["used_in"]) == 1
        
        # Verify tag index
        tag_index = registry["tag_index"]
        assert len(tag_index["framework_tags"]["xgboost"]) == 1
        assert len(tag_index["framework_tags"]["pytorch"]) == 1
        assert len(tag_index["framework_tags"]["sklearn"]) == 1
        
        # Verify domain tags
        assert len(tag_index["domain_tags"]["tabular"]) == 1
        assert len(tag_index["domain_tags"]["nlp"]) == 1
        assert len(tag_index["domain_tags"]["evaluation"]) == 1
        
        # Test registry statistics
        stats = sync.get_registry_statistics()
        assert stats["total_pipelines"] == 3
        assert stats["total_connections"] == 2
        assert stats["connection_density"] > 0
        
        # Test tag statistics
        tag_stats = stats["tag_statistics"]
        assert tag_stats["framework_tags"]["total_tags"] == 3
        assert tag_stats["domain_tags"]["total_tags"] == 3
        
        # Test consistency validation for all pipelines
        for file_name, pipeline in pipelines:
            atomic_id = pipeline.zettelkasten_metadata.atomic_id
            errors = sync.validate_consistency(pipeline, atomic_id)
            assert len(errors) == 0, f"Consistency errors for {atomic_id}: {errors}"
        
        # Test registry validation
        validation_errors = validate_registry_file(temp_registry_path)
        assert len(validation_errors) == 0, f"Registry validation errors: {validation_errors}"
    
    def test_error_recovery_workflow(self, temp_registry_path):
        """Test error recovery and resilience in sync workflow."""
        sync = DAGMetadataRegistrySync(temp_registry_path)
        
        # Create valid pipeline
        valid_pipeline = EnhancedDAGMetadata(
            description="Valid pipeline",
            complexity=ComplexityLevel.SIMPLE,
            features=["training"],
            framework=PipelineFramework.XGBOOST,
            node_count=1,
            edge_count=0
        )
        
        # Sync valid pipeline
        sync.sync_metadata_to_registry(valid_pipeline, "valid_pipeline.py")
        
        # Verify initial state
        registry = sync.load_registry()
        assert len(registry["nodes"]) == 1
        
        # Test recovery from corrupted registry
        with open(temp_registry_path, 'w') as f:
            f.write("corrupted json content")
        
        # Should raise validation error
        with pytest.raises(RegistryValidationError):
            sync.load_registry()
        
        # Recreate registry and verify recovery
        sync._create_empty_registry()
        registry = sync.load_registry()
        assert registry["metadata"]["total_pipelines"] == 0
        assert len(registry["nodes"]) == 0
        
        # Re-sync pipeline after recovery
        sync.sync_metadata_to_registry(valid_pipeline, "valid_pipeline.py")
        
        # Verify recovery worked
        registry = sync.load_registry()
        assert len(registry["nodes"]) == 1
        atomic_id = valid_pipeline.zettelkasten_metadata.atomic_id
        assert atomic_id in registry["nodes"]
        
        # Test statistics after recovery
        stats = sync.get_registry_statistics()
        assert stats["total_pipelines"] == 1
        assert "error" not in stats
    
    def test_complex_connection_workflow(self, temp_registry_path):
        """Test complex connection scenarios with multiple relationship types."""
        sync = DAGMetadataRegistrySync(temp_registry_path)
        
        # Create a network of related pipelines
        pipelines = {}
        
        # Data preprocessing pipeline
        preprocessing = EnhancedDAGMetadata(
            description="Data preprocessing and feature engineering",
            complexity=ComplexityLevel.STANDARD,
            features=["preprocessing"],
            framework=PipelineFramework.SKLEARN,
            node_count=4,
            edge_count=3
        )
        preprocessing.zettelkasten_metadata.domain_tags = ["preprocessing", "feature_engineering"]
        pipelines["preprocessing"] = preprocessing
        
        # Training pipelines
        xgb_training = EnhancedDAGMetadata(
            description="XGBoost model training",
            complexity=ComplexityLevel.SIMPLE,
            features=["training"],
            framework=PipelineFramework.XGBOOST,
            node_count=3,
            edge_count=2
        )
        xgb_training.zettelkasten_metadata.domain_tags = ["training", "gradient_boosting"]
        pipelines["xgb_training"] = xgb_training
        
        pytorch_training = EnhancedDAGMetadata(
            description="PyTorch neural network training",
            complexity=ComplexityLevel.ADVANCED,
            features=["training"],
            framework=PipelineFramework.PYTORCH,
            node_count=8,
            edge_count=7
        )
        pytorch_training.zettelkasten_metadata.domain_tags = ["training", "deep_learning"]
        pipelines["pytorch_training"] = pytorch_training
        
        # Evaluation pipeline
        evaluation = EnhancedDAGMetadata(
            description="Model evaluation and comparison",
            complexity=ComplexityLevel.STANDARD,
            features=["evaluation"],
            framework=PipelineFramework.SKLEARN,
            node_count=5,
            edge_count=4
        )
        evaluation.zettelkasten_metadata.domain_tags = ["evaluation", "model_comparison"]
        pipelines["evaluation"] = evaluation
        
        # Deployment pipeline
        deployment = EnhancedDAGMetadata(
            description="Model deployment and serving",
            complexity=ComplexityLevel.COMPREHENSIVE,
            features=["deployment"],
            framework=PipelineFramework.FRAMEWORK_AGNOSTIC,
            node_count=10,
            edge_count=9
        )
        deployment.zettelkasten_metadata.domain_tags = ["deployment", "serving"]
        pipelines["deployment"] = deployment
        
        # Sync all pipelines first
        for name, pipeline in pipelines.items():
            sync.sync_metadata_to_registry(pipeline, f"pipelines/{name}.py")
        
        # Set up complex connection network
        # Preprocessing is used by training pipelines
        xgb_training.add_connection(
            target_id=preprocessing.zettelkasten_metadata.atomic_id,
            connection_type="depends_on",
            annotation="Requires preprocessed data",
            confidence=0.95
        )
        
        pytorch_training.add_connection(
            target_id=preprocessing.zettelkasten_metadata.atomic_id,
            connection_type="depends_on",
            annotation="Requires preprocessed data",
            confidence=0.95
        )
        
        # Training pipelines are alternatives to each other
        xgb_training.add_connection(
            target_id=pytorch_training.zettelkasten_metadata.atomic_id,
            connection_type="alternatives",
            annotation="Alternative ML approach",
            confidence=0.8
        )
        
        pytorch_training.add_connection(
            target_id=xgb_training.zettelkasten_metadata.atomic_id,
            connection_type="alternatives",
            annotation="Alternative ML approach",
            confidence=0.8
        )
        
        # Evaluation uses trained models
        evaluation.add_connection(
            target_id=xgb_training.zettelkasten_metadata.atomic_id,
            connection_type="depends_on",
            annotation="Evaluates XGBoost models",
            confidence=0.9
        )
        
        evaluation.add_connection(
            target_id=pytorch_training.zettelkasten_metadata.atomic_id,
            connection_type="depends_on",
            annotation="Evaluates PyTorch models",
            confidence=0.9
        )
        
        # Deployment uses evaluated models
        deployment.add_connection(
            target_id=evaluation.zettelkasten_metadata.atomic_id,
            connection_type="depends_on",
            annotation="Deploys best performing model",
            confidence=0.85
        )
        
        # Training pipelines are used in deployment
        deployment.add_connection(
            target_id=xgb_training.zettelkasten_metadata.atomic_id,
            connection_type="used_in",
            annotation="XGBoost model deployment",
            confidence=0.8
        )
        
        deployment.add_connection(
            target_id=pytorch_training.zettelkasten_metadata.atomic_id,
            connection_type="used_in",
            annotation="PyTorch model deployment",
            confidence=0.8
        )
        
        # Re-sync all pipelines with connections
        for name, pipeline in pipelines.items():
            sync.sync_metadata_to_registry(pipeline, f"pipelines/{name}.py")
        
        # Verify complex connection network
        registry = sync.load_registry()
        
        # Check preprocessing connections
        prep_node = registry["nodes"][preprocessing.zettelkasten_metadata.atomic_id]
        assert len(prep_node["connections"]["used_in"]) == 2  # Used by both training pipelines
        
        # Check training pipeline connections
        xgb_node = registry["nodes"][xgb_training.zettelkasten_metadata.atomic_id]
        assert len(xgb_node["connections"]["depends_on"]) == 1  # Depends on preprocessing
        assert len(xgb_node["connections"]["alternatives"]) == 1  # Alternative to PyTorch
        assert len(xgb_node["connections"]["used_in"]) == 2  # Used in evaluation and deployment
        
        pytorch_node = registry["nodes"][pytorch_training.zettelkasten_metadata.atomic_id]
        assert len(pytorch_node["connections"]["depends_on"]) == 1  # Depends on preprocessing
        assert len(pytorch_node["connections"]["alternatives"]) == 1  # Alternative to XGBoost
        assert len(pytorch_node["connections"]["used_in"]) == 2  # Used in evaluation and deployment
        
        # Check evaluation connections
        eval_node = registry["nodes"][evaluation.zettelkasten_metadata.atomic_id]
        assert len(eval_node["connections"]["depends_on"]) == 2  # Depends on both training pipelines
        assert len(eval_node["connections"]["used_in"]) == 1  # Used in deployment
        
        # Check deployment connections
        deploy_node = registry["nodes"][deployment.zettelkasten_metadata.atomic_id]
        assert len(deploy_node["connections"]["depends_on"]) == 1  # Depends on evaluation
        assert len(deploy_node["connections"]["used_in"]) == 2  # Uses both training pipelines
        
        # Verify connection graph metadata
        graph_meta = registry["connection_graph_metadata"]
        assert graph_meta["total_connections"] == 9  # Total connections in the network
        assert graph_meta["independent_pipelines"] == 5  # All pipelines are connected
        assert graph_meta["connection_density"] > 0
        
        # Test registry statistics
        stats = sync.get_registry_statistics()
        assert stats["total_pipelines"] == 5
        assert stats["total_connections"] == 9
        assert len(stats["isolated_nodes"]) == 0  # No isolated nodes
        
        # Verify all connection types are represented
        connection_types = set()
        for node in registry["nodes"].values():
            for conn_type, connections in node["connections"].items():
                if connections:
                    connection_types.add(conn_type)
        
        assert "depends_on" in connection_types
        assert "alternatives" in connection_types
        assert "used_in" in connection_types
        
        # Test consistency validation for all pipelines
        for name, pipeline in pipelines.items():
            atomic_id = pipeline.zettelkasten_metadata.atomic_id
            errors = sync.validate_consistency(pipeline, atomic_id)
            assert len(errors) == 0, f"Consistency errors for {name}: {errors}"
        
        # Test registry validation
        validation_errors = validate_registry_file(temp_registry_path)
        assert len(validation_errors) == 0, f"Registry validation errors: {validation_errors}"
    
    def test_tag_evolution_workflow(self, temp_registry_path):
        """Test tag evolution and management over time."""
        sync = DAGMetadataRegistrySync(temp_registry_path)
        
        # Create initial pipeline with basic tags
        pipeline = EnhancedDAGMetadata(
            description="Evolving ML pipeline",
            complexity=ComplexityLevel.SIMPLE,
            features=["training"],
            framework=PipelineFramework.XGBOOST,
            node_count=3,
            edge_count=2
        )
        
        # Initial tags
        zm = pipeline.zettelkasten_metadata
        zm.domain_tags = ["tabular"]
        zm.pattern_tags = ["basic_workflow"]
        zm.quality_tags = ["experimental"]
        
        # Sync initial version
        sync.sync_metadata_to_registry(pipeline, "pipeline_v1.py")
        
        # Verify initial tag state
        registry = sync.load_registry()
        tag_index = registry["tag_index"]
        atomic_id = zm.atomic_id
        
        assert atomic_id in tag_index["domain_tags"]["tabular"]
        assert atomic_id in tag_index["pattern_tags"]["basic_workflow"]
        assert atomic_id in tag_index["quality_tags"]["experimental"]
        
        # Evolve pipeline - add evaluation capability
        pipeline.complexity = ComplexityLevel.STANDARD
        pipeline.features = ["training", "evaluation"]
        
        # Update tags to reflect evolution
        zm.complexity_tags = ["standard"]
        zm.task_tags = ["training", "evaluation"]
        zm.pattern_tags = ["end_to_end"]  # Replace basic_workflow
        zm.quality_tags = ["tested"]  # Replace experimental
        zm.domain_tags.append("model_evaluation")  # Add new domain
        
        # Sync evolved version
        sync.sync_metadata_to_registry(pipeline, "pipeline_v2.py")
        
        # Verify tag evolution
        registry = sync.load_registry()
        tag_index = registry["tag_index"]
        
        # Check new tags were added
        assert atomic_id in tag_index["complexity_tags"]["standard"]
        assert atomic_id in tag_index["task_tags"]["evaluation"]
        assert atomic_id in tag_index["pattern_tags"]["end_to_end"]
        assert atomic_id in tag_index["quality_tags"]["tested"]
        assert atomic_id in tag_index["domain_tags"]["model_evaluation"]
        
        # Check old tags were removed
        assert "basic_workflow" not in tag_index["pattern_tags"]
        assert "experimental" not in tag_index["quality_tags"]
        
        # Original domain tag should still be there
        assert atomic_id in tag_index["domain_tags"]["tabular"]
        
        # Further evolution - add calibration and make production ready
        pipeline.complexity = ComplexityLevel.ADVANCED
        pipeline.features = ["training", "evaluation", "calibration"]
        
        # Update tags for production readiness
        zm.complexity_tags = ["advanced"]
        zm.task_tags = ["training", "evaluation", "calibration"]
        zm.pattern_tags = ["production_ready", "comprehensive"]
        zm.quality_tags = ["production_ready", "well_tested", "documented"]
        zm.domain_tags.append("model_calibration")
        
        # Sync production version
        sync.sync_metadata_to_registry(pipeline, "pipeline_v3.py")
        
        # Verify final tag state
        registry = sync.load_registry()
        tag_index = registry["tag_index"]
        
        # Check all current tags
        assert atomic_id in tag_index["complexity_tags"]["advanced"]
        assert atomic_id in tag_index["task_tags"]["calibration"]
        assert atomic_id in tag_index["pattern_tags"]["production_ready"]
        assert atomic_id in tag_index["pattern_tags"]["comprehensive"]
        assert atomic_id in tag_index["quality_tags"]["production_ready"]
        assert atomic_id in tag_index["quality_tags"]["well_tested"]
        assert atomic_id in tag_index["quality_tags"]["documented"]
        assert atomic_id in tag_index["domain_tags"]["model_calibration"]
        
        # Check intermediate tags were cleaned up
        assert "standard" not in tag_index["complexity_tags"]
        assert "end_to_end" not in tag_index["pattern_tags"]
        assert "tested" not in tag_index["quality_tags"]
        
        # Test tag statistics evolution
        stats = sync.get_registry_statistics()
        tag_stats = stats["tag_statistics"]
        
        # Should have multiple tags per category
        assert tag_stats["pattern_tags"]["total_tags"] == 2
        assert tag_stats["quality_tags"]["total_tags"] == 3
        assert tag_stats["domain_tags"]["total_tags"] == 3
        
        # Test registry validation
        validation_errors = validate_registry_file(temp_registry_path)
        assert len(validation_errors) == 0

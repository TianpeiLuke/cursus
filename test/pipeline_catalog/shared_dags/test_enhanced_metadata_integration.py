"""
Integration tests for EnhancedDAGMetadata system.

Tests complex workflows and realistic scenarios for the enhanced metadata
system including complete workflows, migration scenarios, and end-to-end testing.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
from datetime import datetime

from cursus.pipeline_catalog.shared_dags.enhanced_metadata import (
    EnhancedDAGMetadata, ZettelkastenMetadata, ComplexityLevel, PipelineFramework,
    DAGMetadataAdapter, validate_enhanced_dag_metadata
)
from cursus.pipeline_catalog.shared_dags import DAGMetadata

class TestEnhancedDAGMetadataIntegration:
    """Integration tests for EnhancedDAGMetadata with realistic scenarios."""
    
    def test_complete_workflow(self):
        """Test complete workflow from creation to registry conversion."""
        # Create enhanced metadata
        enhanced = EnhancedDAGMetadata(
            description="Complete XGBoost training and evaluation pipeline",
            complexity=ComplexityLevel.ADVANCED,
            features=["training", "evaluation", "calibration"],
            framework=PipelineFramework.XGBOOST,
            node_count=8,
            edge_count=7
        )
        
        # Customize Zettelkasten metadata
        zm = enhanced.zettelkasten_metadata
        zm.title = "Advanced XGBoost ML Pipeline"
        zm.input_interface = ["training_data", "validation_data", "config"]
        zm.output_interface = ["trained_model", "evaluation_metrics", "calibrated_model"]
        zm.domain_tags = ["tabular", "supervised_learning"]
        zm.pattern_tags = ["end_to_end", "production_ready"]
        zm.source_file = "xgboost/advanced_training_evaluation.py"
        zm.created_date = "2024-01-15"
        zm.priority = "high"
        
        # Add connections
        enhanced.add_connection(
            target_id="xgb_simple_training",
            connection_type="alternatives",
            annotation="Simpler version without evaluation",
            confidence=0.9
        )
        
        enhanced.add_connection(
            target_id="model_deployment_pipeline",
            connection_type="used_in",
            annotation="Deploy trained and calibrated model",
            confidence=0.8
        )
        
        # Update tags
        enhanced.update_tags("quality_tags", ["production_ready", "well_tested"])
        
        # Convert to registry node
        registry_node = enhanced.to_registry_node()
        
        # Verify registry node structure
        assert registry_node["id"] == "xgboost_training_advanced"
        assert registry_node["title"] == "Advanced XGBoost ML Pipeline"
        assert registry_node["description"] == "Complete XGBoost training and evaluation pipeline"
        
        # Verify atomic properties
        atomic_props = registry_node["atomic_properties"]
        assert atomic_props["single_responsibility"] == "Complete XGBoost training and evaluation pipeline"
        assert atomic_props["independence_level"] == "fully_self_contained"
        assert atomic_props["node_count"] == 1  # From Zettelkasten metadata
        assert atomic_props["edge_count"] == 0  # From Zettelkasten metadata
        
        # Verify zettelkasten metadata
        zk_meta = registry_node["zettelkasten_metadata"]
        assert zk_meta["framework"] == "xgboost"
        assert zk_meta["complexity"] == "advanced"
        assert zk_meta["features"] == ["training", "evaluation", "calibration"]
        assert zk_meta["mods_compatible"] is False
        
        # Verify multi-dimensional tags
        tags = registry_node["multi_dimensional_tags"]
        assert tags["framework_tags"] == ["xgboost"]
        assert tags["task_tags"] == ["training", "evaluation", "calibration"]
        assert tags["complexity_tags"] == ["advanced"]
        
        # Verify connections
        connections = registry_node["connections"]
        assert len(connections["alternatives"]) == 1
        assert connections["alternatives"][0]["id"] == "xgb_simple_training"
        assert connections["alternatives"][0]["annotation"] == "Simpler version without evaluation"
        
        assert len(connections["used_in"]) == 1
        assert connections["used_in"][0]["id"] == "model_deployment_pipeline"
        assert connections["used_in"][0]["annotation"] == "Deploy trained and calibrated model"
        
        # Verify file tracking
        assert registry_node["source_file"] == "xgboost/advanced_training_evaluation.py"
        assert registry_node["created_date"] == "2024-01-15"
        assert registry_node["priority"] == "high"
        
        # Test conversion to dictionary
        data_dict = enhanced.to_dict()
        assert "zettelkasten_metadata" in data_dict
        assert data_dict["zettelkasten_metadata"]["quality_tags"] == ["production_ready", "well_tested"]
        
        # Test conversion to legacy format
        legacy = enhanced.to_legacy_dag_metadata()
        assert legacy.description == "Complete XGBoost training and evaluation pipeline"
        assert legacy.complexity == "advanced"
        assert legacy.framework == "xgboost"
        assert legacy.features == ["training", "evaluation", "calibration"]
        
        # Test validation
        assert validate_enhanced_dag_metadata(enhanced) is True
    
    def test_migration_scenario(self):
        """Test migration from legacy to enhanced metadata."""
        # Create legacy metadata
        legacy = DAGMetadata(
            description="Legacy PyTorch NLP training pipeline",
            complexity="standard",
            features=["preprocessing", "training", "evaluation"],
            framework="pytorch",
            node_count=6,
            edge_count=5,
            extra_metadata={
                "input_interface": ["text_data", "tokenizer_config"],
                "output_interface": ["trained_model", "evaluation_results"],
                "domain": "nlp",
                "task_type": "text_classification"
            }
        )
        
        # Convert to enhanced metadata
        enhanced = DAGMetadataAdapter.from_legacy_dag_metadata(legacy)
        
        # Verify conversion
        assert enhanced.description == "Legacy PyTorch NLP training pipeline"
        assert enhanced.complexity == ComplexityLevel.STANDARD
        assert enhanced.framework == PipelineFramework.PYTORCH
        assert enhanced.features == ["preprocessing", "training", "evaluation"]
        assert enhanced.node_count == 6
        assert enhanced.edge_count == 5
        
        # Verify Zettelkasten metadata was properly created
        zm = enhanced.zettelkasten_metadata
        assert zm.atomic_id == "pytorch_preprocessing_standard"
        assert zm.single_responsibility == "Legacy PyTorch NLP training pipeline"
        assert zm.input_interface == ["text_data", "tokenizer_config"]
        assert zm.output_interface == ["trained_model", "evaluation_results"]
        assert zm.framework_tags == ["pytorch"]
        assert zm.task_tags == ["preprocessing", "training", "evaluation"]
        assert zm.complexity_tags == ["standard"]
        
        # Enhance with Zettelkasten principles
        zm.title = "PyTorch NLP Training Pipeline"
        zm.domain_tags = ["nlp", "text_classification"]
        zm.pattern_tags = ["end_to_end", "multi_stage"]
        zm.source_file = "pytorch/nlp_training_pipeline.py"
        
        # Add manual connections
        enhanced.add_connection(
            target_id="bert_fine_tuning",
            connection_type="alternatives",
            annotation="BERT-based alternative approach",
            confidence=0.85
        )
        
        enhanced.add_connection(
            target_id="text_preprocessing_utils",
            connection_type="depends_on",
            annotation="Shared text preprocessing utilities",
            confidence=0.95
        )
        
        # Convert to registry node
        registry_node = enhanced.to_registry_node()
        
        # Verify registry node has all enhanced features
        assert registry_node["id"] == "pytorch_preprocessing_standard"
        assert registry_node["title"] == "PyTorch NLP Training Pipeline"
        
        # Verify connections were preserved
        connections = registry_node["connections"]
        assert "alternatives" in connections
        assert "depends_on" in connections
        assert len(connections["alternatives"]) == 1
        assert len(connections["depends_on"]) == 1
        
        # Test round-trip conversion
        converted_back = enhanced.to_legacy_dag_metadata()
        assert converted_back.description == legacy.description
        assert converted_back.complexity == legacy.complexity
        assert converted_back.framework == legacy.framework
        assert converted_back.features == legacy.features
        assert converted_back.node_count == legacy.node_count
        assert converted_back.edge_count == legacy.edge_count
        
        # Validate the enhanced metadata
        assert validate_enhanced_dag_metadata(enhanced) is True
    
    def test_multi_framework_pipeline_scenario(self):
        """Test scenario with multiple framework dependencies."""
        # Create a complex pipeline that uses multiple frameworks
        enhanced = EnhancedDAGMetadata(
            description="Multi-framework ensemble pipeline with preprocessing, training, and evaluation",
            complexity=ComplexityLevel.COMPREHENSIVE,
            features=["preprocessing", "training", "evaluation", "ensemble"],
            framework=PipelineFramework.FRAMEWORK_AGNOSTIC,
            node_count=12,
            edge_count=15
        )
        
        # Customize for multi-framework scenario
        zm = enhanced.zettelkasten_metadata
        zm.title = "Multi-Framework Ensemble Pipeline"
        zm.input_interface = ["raw_data", "feature_config", "model_configs"]
        zm.output_interface = ["ensemble_model", "performance_metrics", "feature_importance"]
        zm.framework_tags = ["sklearn", "xgboost", "pytorch", "framework_agnostic"]
        zm.domain_tags = ["ensemble_learning", "multi_modal"]
        zm.pattern_tags = ["complex_workflow", "multi_stage", "production_ready"]
        zm.quality_tags = ["well_tested", "documented", "scalable"]
        zm.source_file = "ensemble/multi_framework_pipeline.py"
        
        # Add connections to component pipelines
        enhanced.add_connection(
            target_id="sklearn_preprocessing",
            connection_type="depends_on",
            annotation="Feature preprocessing and selection",
            confidence=0.95
        )
        
        enhanced.add_connection(
            target_id="xgboost_training",
            connection_type="depends_on",
            annotation="XGBoost base model training",
            confidence=0.9
        )
        
        enhanced.add_connection(
            target_id="pytorch_training",
            connection_type="depends_on",
            annotation="Neural network base model training",
            confidence=0.9
        )
        
        enhanced.add_connection(
            target_id="simple_xgboost_pipeline",
            connection_type="alternatives",
            annotation="Simpler single-framework alternative",
            confidence=0.7
        )
        
        enhanced.add_connection(
            target_id="model_deployment_service",
            connection_type="used_in",
            annotation="Deploy ensemble model to production",
            confidence=0.85
        )
        
        # Convert to registry node
        registry_node = enhanced.to_registry_node()
        
        # Verify complex pipeline structure
        assert registry_node["id"] == "framework_agnostic_preprocessing_comprehensive"
        assert registry_node["title"] == "Multi-Framework Ensemble Pipeline"
        
        # Verify multi-dimensional tags include all frameworks
        tags = registry_node["multi_dimensional_tags"]
        assert "sklearn" in tags["framework_tags"]
        assert "xgboost" in tags["framework_tags"]
        assert "pytorch" in tags["framework_tags"]
        assert "framework_agnostic" in tags["framework_tags"]
        
        # Verify all connection types are present
        connections = registry_node["connections"]
        assert len(connections["depends_on"]) == 3
        assert len(connections["alternatives"]) == 1
        assert len(connections["used_in"]) == 1
        
        # Verify dependencies include all frameworks
        deps = enhanced._extract_dependencies()
        assert "sklearn" in deps
        assert "xgboost" in deps
        assert "torch" in deps
        assert "sagemaker" in deps
        
        # Test validation
        assert validate_enhanced_dag_metadata(enhanced) is True
    
    def test_pipeline_evolution_scenario(self):
        """Test scenario showing pipeline evolution over time."""
        # Start with simple pipeline
        simple_enhanced = EnhancedDAGMetadata(
            description="Simple XGBoost training pipeline",
            complexity=ComplexityLevel.SIMPLE,
            features=["training"],
            framework=PipelineFramework.XGBOOST,
            node_count=3,
            edge_count=2
        )
        
        # Evolve to standard pipeline with evaluation
        standard_enhanced = EnhancedDAGMetadata(
            description="XGBoost training and evaluation pipeline",
            complexity=ComplexityLevel.STANDARD,
            features=["training", "evaluation"],
            framework=PipelineFramework.XGBOOST,
            node_count=5,
            edge_count=4
        )
        
        # Evolve to advanced pipeline with calibration
        advanced_enhanced = EnhancedDAGMetadata(
            description="Advanced XGBoost pipeline with training, evaluation, and calibration",
            complexity=ComplexityLevel.ADVANCED,
            features=["training", "evaluation", "calibration"],
            framework=PipelineFramework.XGBOOST,
            node_count=8,
            edge_count=7
        )
        
        # Set up evolution connections
        standard_enhanced.add_connection(
            target_id=simple_enhanced.zettelkasten_metadata.atomic_id,
            connection_type="evolved_from",
            annotation="Added evaluation capabilities",
            confidence=0.95
        )
        
        advanced_enhanced.add_connection(
            target_id=standard_enhanced.zettelkasten_metadata.atomic_id,
            connection_type="evolved_from",
            annotation="Added calibration and advanced features",
            confidence=0.9
        )
        
        # Add reverse connections
        simple_enhanced.add_connection(
            target_id=standard_enhanced.zettelkasten_metadata.atomic_id,
            connection_type="evolved_to",
            annotation="Extended with evaluation",
            confidence=0.95
        )
        
        standard_enhanced.add_connection(
            target_id=advanced_enhanced.zettelkasten_metadata.atomic_id,
            connection_type="evolved_to",
            annotation="Extended with calibration",
            confidence=0.9
        )
        
        # Verify evolution chain
        simple_node = simple_enhanced.to_registry_node()
        standard_node = standard_enhanced.to_registry_node()
        advanced_node = advanced_enhanced.to_registry_node()
        
        # Check evolution connections
        assert len(simple_node["connections"]["evolved_to"]) == 1
        assert simple_node["connections"]["evolved_to"][0]["id"] == "xgboost_training_standard"
        
        assert len(standard_node["connections"]["evolved_from"]) == 1
        assert standard_node["connections"]["evolved_from"][0]["id"] == "xgboost_training_simple"
        assert len(standard_node["connections"]["evolved_to"]) == 1
        assert standard_node["connections"]["evolved_to"][0]["id"] == "xgboost_training_advanced"
        
        assert len(advanced_node["connections"]["evolved_from"]) == 1
        assert advanced_node["connections"]["evolved_from"][0]["id"] == "xgboost_training_standard"
        
        # Verify complexity progression
        assert simple_enhanced.complexity == ComplexityLevel.SIMPLE
        assert standard_enhanced.complexity == ComplexityLevel.STANDARD
        assert advanced_enhanced.complexity == ComplexityLevel.ADVANCED
        
        # Verify feature progression
        assert simple_enhanced.features == ["training"]
        assert standard_enhanced.features == ["training", "evaluation"]
        assert advanced_enhanced.features == ["training", "evaluation", "calibration"]
        
        # Validate all pipelines
        assert validate_enhanced_dag_metadata(simple_enhanced) is True
        assert validate_enhanced_dag_metadata(standard_enhanced) is True
        assert validate_enhanced_dag_metadata(advanced_enhanced) is True
    
    def test_cross_framework_alternatives_scenario(self):
        """Test scenario with cross-framework alternatives."""
        # Create XGBoost pipeline
        xgb_enhanced = EnhancedDAGMetadata(
            description="XGBoost tabular data training pipeline",
            complexity=ComplexityLevel.STANDARD,
            features=["training", "evaluation"],
            framework=PipelineFramework.XGBOOST,
            node_count=5,
            edge_count=4
        )
        
        # Create PyTorch alternative
        pytorch_enhanced = EnhancedDAGMetadata(
            description="PyTorch neural network training pipeline",
            complexity=ComplexityLevel.STANDARD,
            features=["training", "evaluation"],
            framework=PipelineFramework.PYTORCH,
            node_count=6,
            edge_count=5
        )
        
        # Create sklearn alternative
        sklearn_enhanced = EnhancedDAGMetadata(
            description="Scikit-learn ensemble training pipeline",
            complexity=ComplexityLevel.STANDARD,
            features=["training", "evaluation"],
            framework=PipelineFramework.SKLEARN,
            node_count=4,
            edge_count=3
        )
        
        # Set up cross-framework alternatives
        xgb_enhanced.add_connection(
            target_id=pytorch_enhanced.zettelkasten_metadata.atomic_id,
            connection_type="alternatives",
            annotation="Neural network approach for same task",
            confidence=0.8
        )
        
        xgb_enhanced.add_connection(
            target_id=sklearn_enhanced.zettelkasten_metadata.atomic_id,
            connection_type="alternatives",
            annotation="Ensemble learning approach",
            confidence=0.85
        )
        
        pytorch_enhanced.add_connection(
            target_id=xgb_enhanced.zettelkasten_metadata.atomic_id,
            connection_type="alternatives",
            annotation="Gradient boosting approach",
            confidence=0.8
        )
        
        pytorch_enhanced.add_connection(
            target_id=sklearn_enhanced.zettelkasten_metadata.atomic_id,
            connection_type="alternatives",
            annotation="Traditional ML approach",
            confidence=0.75
        )
        
        sklearn_enhanced.add_connection(
            target_id=xgb_enhanced.zettelkasten_metadata.atomic_id,
            connection_type="alternatives",
            annotation="Specialized gradient boosting",
            confidence=0.85
        )
        
        sklearn_enhanced.add_connection(
            target_id=pytorch_enhanced.zettelkasten_metadata.atomic_id,
            connection_type="alternatives",
            annotation="Deep learning approach",
            confidence=0.75
        )
        
        # Convert to registry nodes
        xgb_node = xgb_enhanced.to_registry_node()
        pytorch_node = pytorch_enhanced.to_registry_node()
        sklearn_node = sklearn_enhanced.to_registry_node()
        
        # Verify cross-framework connections
        assert len(xgb_node["connections"]["alternatives"]) == 2
        assert len(pytorch_node["connections"]["alternatives"]) == 2
        assert len(sklearn_node["connections"]["alternatives"]) == 2
        
        # Verify framework-specific IDs
        assert xgb_node["id"] == "xgboost_training_standard"
        assert pytorch_node["id"] == "pytorch_training_standard"
        assert sklearn_node["id"] == "sklearn_training_standard"
        
        # Verify framework-specific dependencies
        xgb_deps = xgb_enhanced._extract_dependencies()
        pytorch_deps = pytorch_enhanced._extract_dependencies()
        sklearn_deps = sklearn_enhanced._extract_dependencies()
        
        assert "xgboost" in xgb_deps
        assert "torch" in pytorch_deps
        assert "sklearn" in sklearn_deps
        
        # All should have sagemaker
        assert "sagemaker" in xgb_deps
        assert "sagemaker" in pytorch_deps
        assert "sagemaker" in sklearn_deps
        
        # Validate all alternatives
        assert validate_enhanced_dag_metadata(xgb_enhanced) is True
        assert validate_enhanced_dag_metadata(pytorch_enhanced) is True
        assert validate_enhanced_dag_metadata(sklearn_enhanced) is True

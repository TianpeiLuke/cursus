"""
Unit tests for PipelineRecommendationEngine integration scenarios.

Tests comprehensive recommendation workflows and composition algorithms
with realistic data scenarios.
"""

import pytest
from unittest.mock import Mock
from src.cursus.pipeline_catalog.utils.recommendation_engine import (
    PipelineRecommendationEngine, CompositionRecommendation
)
from src.cursus.pipeline_catalog.utils.catalog_registry import CatalogRegistry
from src.cursus.pipeline_catalog.utils.connection_traverser import ConnectionTraverser, PipelineConnection
from src.cursus.pipeline_catalog.utils.tag_discovery import TagBasedDiscovery


class TestPipelineRecommendationEngineCompositions:
    """Test suite for composition recommendation algorithms."""
    
    @pytest.fixture
    def engine_with_mocks(self):
        """Create engine with comprehensive mock data."""
        registry = Mock(spec=CatalogRegistry)
        traverser = Mock(spec=ConnectionTraverser)
        discovery = Mock(spec=TagBasedDiscovery)
        
        return PipelineRecommendationEngine(registry, traverser, discovery)
    
    def test_recommend_sequential_compositions(self, engine_with_mocks):
        """Test sequential composition recommendations."""
        engine = engine_with_mocks
        pipeline_ids = ["sklearn_prep", "xgb_simple", "xgb_advanced"]
        
        # Mock pipeline info
        pipeline_info = {
            "sklearn_prep": {
                "framework": "sklearn",
                "complexity": "simple",
                "task_tags": ["preprocessing"]
            },
            "xgb_simple": {
                "framework": "xgboost",
                "complexity": "simple",
                "task_tags": ["training"]
            },
            "xgb_advanced": {
                "framework": "xgboost",
                "complexity": "advanced",
                "task_tags": ["training", "evaluation"]
            }
        }
        
        compositions = engine._recommend_sequential_compositions(pipeline_ids, pipeline_info)
        
        assert isinstance(compositions, list)
        if compositions:
            comp = compositions[0]
            assert comp.composition_type == "sequential"
            assert len(comp.pipeline_sequence) >= 2
    
    def test_recommend_parallel_compositions(self, engine_with_mocks):
        """Test parallel composition recommendations."""
        engine = engine_with_mocks
        pipeline_ids = ["xgb_simple", "pytorch_nlp"]
        
        # Mock pipeline info with same task (parallel candidates)
        pipeline_info = {
            "xgb_simple": {
                "framework": "xgboost",
                "complexity": "simple",
                "task_tags": ["training"]
            },
            "pytorch_nlp": {
                "framework": "pytorch",
                "complexity": "standard",
                "task_tags": ["training"]
            }
        }
        
        compositions = engine._recommend_parallel_compositions(pipeline_ids, pipeline_info)
        
        assert isinstance(compositions, list)
        if compositions:
            comp = compositions[0]
            assert comp.composition_type == "parallel"
    
    def test_recommend_conditional_compositions(self, engine_with_mocks):
        """Test conditional composition recommendations."""
        engine = engine_with_mocks
        pipeline_ids = ["xgb_simple", "xgb_advanced"]
        
        # Mock pipeline info with different complexities
        pipeline_info = {
            "xgb_simple": {
                "framework": "xgboost",
                "complexity": "simple",
                "task_tags": ["training"]
            },
            "xgb_advanced": {
                "framework": "xgboost",
                "complexity": "advanced",
                "task_tags": ["training", "evaluation"]
            }
        }
        
        compositions = engine._recommend_conditional_compositions(pipeline_ids, pipeline_info)
        
        assert isinstance(compositions, list)
        if compositions:
            comp = compositions[0]
            assert comp.composition_type == "conditional"
            assert comp.estimated_complexity == "adaptive"
    
    def test_select_best_learning_candidate(self, engine_with_mocks):
        """Test selecting best candidate for learning path."""
        engine = engine_with_mocks
        candidates = ["xgb_simple", "xgb_advanced"]
        
        # Mock nodes with different characteristics
        def mock_get_node(pipeline_id):
            nodes = {
                "xgb_simple": {
                    "description": "Simple XGBoost training",
                    "atomic_properties": {
                        "single_responsibility": "Train model"
                    },
                    "multi_dimensional_tags": {
                        "complexity_tags": ["beginner_friendly"]
                    }
                },
                "xgb_advanced": {
                    "description": "Advanced XGBoost pipeline",
                    "atomic_properties": {
                        "single_responsibility": "Complete ML workflow with multiple steps"
                    },
                    "multi_dimensional_tags": {
                        "complexity_tags": ["advanced"]
                    }
                }
            }
            return nodes.get(pipeline_id)
        
        engine.registry.get_pipeline_node.side_effect = mock_get_node
        
        best = engine._select_best_learning_candidate(candidates, None)
        
        # Should prefer beginner-friendly pipeline
        assert best == "xgb_simple"


class TestPipelineRecommendationEngineIntegration:
    """Integration tests for PipelineRecommendationEngine with realistic scenarios."""
    
    def test_comprehensive_recommendation_workflow(self):
        """Test comprehensive recommendation workflow."""
        # Create mocks with realistic data
        registry = Mock(spec=CatalogRegistry)
        traverser = Mock(spec=ConnectionTraverser)
        discovery = Mock(spec=TagBasedDiscovery)
        
        # Mock comprehensive pipeline data
        pipeline_data = {
            "data_prep": {
                "id": "data_prep",
                "title": "Data Preprocessing Pipeline",
                "description": "Comprehensive data preprocessing and feature engineering",
                "zettelkasten_metadata": {
                    "framework": "sklearn",
                    "complexity": "simple",
                    "use_case": "Data preparation"
                },
                "multi_dimensional_tags": {
                    "framework_tags": ["sklearn"],
                    "task_tags": ["preprocessing", "feature_engineering"],
                    "complexity_tags": ["simple"]
                }
            },
            "xgb_training": {
                "id": "xgb_training",
                "title": "XGBoost Training Pipeline",
                "description": "XGBoost model training with hyperparameter tuning",
                "zettelkasten_metadata": {
                    "framework": "xgboost",
                    "complexity": "standard",
                    "use_case": "Model training"
                },
                "multi_dimensional_tags": {
                    "framework_tags": ["xgboost"],
                    "task_tags": ["training", "hyperparameter_tuning"],
                    "complexity_tags": ["standard"]
                }
            },
            "model_evaluation": {
                "id": "model_evaluation",
                "title": "Model Evaluation Pipeline",
                "description": "Comprehensive model evaluation and validation",
                "zettelkasten_metadata": {
                    "framework": "generic",
                    "complexity": "standard",
                    "use_case": "Model validation"
                },
                "multi_dimensional_tags": {
                    "framework_tags": ["generic"],
                    "task_tags": ["evaluation", "validation"],
                    "complexity_tags": ["standard"]
                }
            },
            "model_deployment": {
                "id": "model_deployment",
                "title": "Model Deployment Pipeline",
                "description": "Production model deployment and monitoring",
                "zettelkasten_metadata": {
                    "framework": "generic",
                    "complexity": "advanced",
                    "use_case": "Production deployment"
                },
                "multi_dimensional_tags": {
                    "framework_tags": ["generic"],
                    "task_tags": ["deployment", "monitoring"],
                    "complexity_tags": ["advanced"]
                }
            }
        }
        
        registry.get_pipeline_node.side_effect = lambda pid: pipeline_data.get(pid)
        registry.get_all_pipelines.return_value = list(pipeline_data.keys())
        
        # Mock connections for logical workflow
        connections = {
            "data_prep": {
                "used_in": [
                    PipelineConnection(target_id="xgb_training", connection_type="used_in",
                                     annotation="Feeds into training", source_id="data_prep")
                ]
            },
            "xgb_training": {
                "used_in": [
                    PipelineConnection(target_id="model_evaluation", connection_type="used_in",
                                     annotation="Model needs evaluation", source_id="xgb_training")
                ]
            },
            "model_evaluation": {
                "used_in": [
                    PipelineConnection(target_id="model_deployment", connection_type="used_in",
                                     annotation="Validated model for deployment", source_id="model_evaluation")
                ]
            }
        }
        
        traverser.get_all_connections.side_effect = lambda pid: connections.get(pid, {"alternatives": [], "related": [], "used_in": []})
        traverser.get_compositions.side_effect = lambda pid: connections.get(pid, {}).get("used_in", [])
        
        # Mock discovery for text search
        discovery.search_by_text.return_value = [
            ("xgb_training", 3.0),
            ("data_prep", 2.0),
            ("model_evaluation", 1.5)
        ]
        discovery.find_by_tags.return_value = ["xgb_training", "data_prep"]
        discovery.suggest_similar_pipelines.return_value = [("model_evaluation", 0.4)]
        
        engine = PipelineRecommendationEngine(registry, traverser, discovery)
        
        # Test 1: Use case recommendation
        use_case_recs = engine.recommend_for_use_case("machine learning training")
        assert len(use_case_recs) > 0
        assert any(rec.pipeline_id == "xgb_training" for rec in use_case_recs)
        
        # Test 2: Next steps recommendation
        next_steps = engine.recommend_next_steps("data_prep")
        assert len(next_steps) > 0
        # Should recommend xgb_training as next step
        assert any(rec.pipeline_id == "xgb_training" for rec in next_steps)
        
        # Test 3: Composition recommendation
        all_pipelines = list(pipeline_data.keys())
        compositions = engine.recommend_compositions(all_pipelines)
        assert len(compositions) > 0
        # Should find sequential composition
        assert any(comp.composition_type == "sequential" for comp in compositions)
        
        # Test 4: Learning path
        learning_path = engine.get_learning_path("simple", "any")
        assert len(learning_path) > 0
    
    def test_recommendation_scoring_consistency(self):
        """Test that recommendation scoring is consistent and meaningful."""
        registry = Mock(spec=CatalogRegistry)
        traverser = Mock(spec=ConnectionTraverser)
        discovery = Mock(spec=TagBasedDiscovery)
        
        # Mock high-relevance vs low-relevance scenarios
        registry.get_pipeline_node.return_value = {
            "id": "test_pipeline",
            "title": "Test Pipeline",
            "description": "Test description",
            "zettelkasten_metadata": {"framework": "xgboost", "complexity": "simple"}
        }
        
        # High relevance: direct text match + tag match
        discovery.search_by_text.return_value = [("test_pipeline", 3.0)]
        discovery.find_by_tags.return_value = ["test_pipeline"]
        
        engine = PipelineRecommendationEngine(registry, traverser, discovery)
        
        high_relevance_recs = engine.recommend_for_use_case("test pipeline training")
        
        # Low relevance: no text match, weak tag match
        discovery.search_by_text.return_value = []
        discovery.find_by_tags.return_value = ["test_pipeline"]
        
        low_relevance_recs = engine.recommend_for_use_case("unrelated query")
        
        # High relevance should have higher scores
        if high_relevance_recs and low_relevance_recs:
            high_score = max(rec.score for rec in high_relevance_recs)
            low_score = max(rec.score for rec in low_relevance_recs)
            assert high_score > low_score
    
    def test_end_to_end_ml_workflow_recommendations(self):
        """Test end-to-end ML workflow recommendation scenario."""
        registry = Mock(spec=CatalogRegistry)
        traverser = Mock(spec=ConnectionTraverser)
        discovery = Mock(spec=TagBasedDiscovery)
        
        # Create complete ML workflow pipeline data
        ml_workflow_data = {
            "data_ingestion": {
                "id": "data_ingestion",
                "title": "Data Ingestion Pipeline",
                "zettelkasten_metadata": {"framework": "generic", "complexity": "simple"},
                "multi_dimensional_tags": {"task_tags": ["ingestion"], "complexity_tags": ["simple"]}
            },
            "data_validation": {
                "id": "data_validation", 
                "title": "Data Validation Pipeline",
                "zettelkasten_metadata": {"framework": "generic", "complexity": "simple"},
                "multi_dimensional_tags": {"task_tags": ["validation"], "complexity_tags": ["simple"]}
            },
            "feature_engineering": {
                "id": "feature_engineering",
                "title": "Feature Engineering Pipeline", 
                "zettelkasten_metadata": {"framework": "sklearn", "complexity": "standard"},
                "multi_dimensional_tags": {"task_tags": ["preprocessing"], "complexity_tags": ["standard"]}
            },
            "model_training": {
                "id": "model_training",
                "title": "Model Training Pipeline",
                "zettelkasten_metadata": {"framework": "xgboost", "complexity": "standard"}, 
                "multi_dimensional_tags": {"task_tags": ["training"], "complexity_tags": ["standard"]}
            },
            "model_validation": {
                "id": "model_validation",
                "title": "Model Validation Pipeline",
                "zettelkasten_metadata": {"framework": "generic", "complexity": "standard"},
                "multi_dimensional_tags": {"task_tags": ["evaluation"], "complexity_tags": ["standard"]}
            },
            "model_deployment": {
                "id": "model_deployment", 
                "title": "Model Deployment Pipeline",
                "zettelkasten_metadata": {"framework": "generic", "complexity": "advanced"},
                "multi_dimensional_tags": {"task_tags": ["deployment"], "complexity_tags": ["advanced"]}
            }
        }
        
        registry.get_pipeline_node.side_effect = lambda pid: ml_workflow_data.get(pid)
        registry.get_all_pipelines.return_value = list(ml_workflow_data.keys())
        
        # Mock logical connections between workflow stages
        workflow_connections = {
            "data_ingestion": {"used_in": [PipelineConnection("data_validation", "used_in", "Validate ingested data", "data_ingestion")]},
            "data_validation": {"used_in": [PipelineConnection("feature_engineering", "used_in", "Engineer features from validated data", "data_validation")]},
            "feature_engineering": {"used_in": [PipelineConnection("model_training", "used_in", "Train on engineered features", "feature_engineering")]},
            "model_training": {"used_in": [PipelineConnection("model_validation", "used_in", "Validate trained model", "model_training")]},
            "model_validation": {"used_in": [PipelineConnection("model_deployment", "used_in", "Deploy validated model", "model_validation")]}
        }
        
        traverser.get_all_connections.side_effect = lambda pid: workflow_connections.get(pid, {"alternatives": [], "related": [], "used_in": []})
        traverser.get_compositions.side_effect = lambda pid: workflow_connections.get(pid, {}).get("used_in", [])
        
        # Mock discovery to find relevant pipelines
        discovery.search_by_text.side_effect = lambda query, **kwargs: [
            (pid, 2.0) for pid in ml_workflow_data.keys() 
            if any(word in ml_workflow_data[pid]["title"].lower() for word in query.lower().split())
        ]
        discovery.find_by_tags.side_effect = lambda tags, **kwargs: [
            pid for pid in ml_workflow_data.keys()
            if any(tag in ml_workflow_data[pid]["multi_dimensional_tags"].get("task_tags", []) for tag in tags)
        ]
        
        engine = PipelineRecommendationEngine(registry, traverser, discovery)
        
        # Test complete workflow composition
        all_workflow_pipelines = list(ml_workflow_data.keys())
        compositions = engine.recommend_compositions(all_workflow_pipelines)
        
        # Should find sequential composition for complete ML workflow
        assert len(compositions) > 0
        sequential_comps = [comp for comp in compositions if comp.composition_type == "sequential"]
        assert len(sequential_comps) > 0
        
        # Best sequential composition should include most/all workflow stages
        best_sequential = max(sequential_comps, key=lambda x: x.total_score)
        assert len(best_sequential.pipeline_sequence) >= 4  # Should include multiple stages
        
        # Test learning path from simple to advanced
        learning_path = engine.get_learning_path("simple", "any")
        assert len(learning_path) > 0
        
        # Learning path should start with simple pipelines
        first_pipeline = learning_path[0]
        first_pipeline_data = ml_workflow_data.get(first_pipeline, {})
        if first_pipeline_data:
            complexity = first_pipeline_data.get("zettelkasten_metadata", {}).get("complexity", "")
            assert complexity in ["simple", "standard"]  # Should start with simpler pipelines

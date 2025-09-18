"""
Unit tests for PipelineRecommendationEngine class.

Tests intelligent pipeline recommendation combining Zettelkasten principles.
Integrates manual linking (connections) with emergent organization (tags).
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any, List

from cursus.pipeline_catalog.core.recommendation_engine import (
    PipelineRecommendationEngine,
    RecommendationResult,
    CompositionRecommendation
)
from cursus.pipeline_catalog.core.catalog_registry import CatalogRegistry
from cursus.pipeline_catalog.core.connection_traverser import ConnectionTraverser, PipelineConnection
from cursus.pipeline_catalog.core.tag_discovery import TagBasedDiscovery


class TestPipelineRecommendationEngine:
    """Test suite for PipelineRecommendationEngine class."""

    @pytest.fixture
    def mock_registry(self):
        """Create mock CatalogRegistry."""
        return Mock(spec=CatalogRegistry)

    @pytest.fixture
    def mock_traverser(self):
        """Create mock ConnectionTraverser."""
        return Mock(spec=ConnectionTraverser)

    @pytest.fixture
    def mock_discovery(self):
        """Create mock TagBasedDiscovery."""
        return Mock(spec=TagBasedDiscovery)

    @pytest.fixture
    def sample_nodes(self):
        """Sample node data for testing."""
        return {
            "training_pipeline": {
                "id": "training_pipeline",
                "title": "XGBoost Training Pipeline",
                "description": "Train XGBoost models on tabular data",
                "zettelkasten_metadata": {
                    "framework": "xgboost",
                    "complexity": "simple"
                },
                "multi_dimensional_tags": {
                    "framework_tags": ["xgboost"],
                    "task_tags": ["training"],
                    "complexity_tags": ["simple"]
                },
                "atomic_properties": {
                    "single_responsibility": "Train XGBoost models"
                }
            },
            "evaluation_pipeline": {
                "id": "evaluation_pipeline",
                "title": "Model Evaluation Pipeline",
                "description": "Evaluate trained models",
                "zettelkasten_metadata": {
                    "framework": "xgboost",
                    "complexity": "standard"
                },
                "multi_dimensional_tags": {
                    "framework_tags": ["xgboost"],
                    "task_tags": ["evaluation"],
                    "complexity_tags": ["standard"]
                },
                "atomic_properties": {
                    "single_responsibility": "Evaluate models"
                }
            },
            "pytorch_training": {
                "id": "pytorch_training",
                "title": "PyTorch Training Pipeline",
                "description": "Train PyTorch models for deep learning",
                "zettelkasten_metadata": {
                    "framework": "pytorch",
                    "complexity": "advanced"
                },
                "multi_dimensional_tags": {
                    "framework_tags": ["pytorch"],
                    "task_tags": ["training"],
                    "complexity_tags": ["advanced"]
                },
                "atomic_properties": {
                    "single_responsibility": "Train PyTorch models"
                }
            }
        }

    @pytest.fixture
    def engine(self, mock_registry, mock_traverser, mock_discovery):
        """Create PipelineRecommendationEngine instance."""
        return PipelineRecommendationEngine(mock_registry, mock_traverser, mock_discovery)

    def test_init(self, mock_registry, mock_traverser, mock_discovery):
        """Test PipelineRecommendationEngine initialization."""
        engine = PipelineRecommendationEngine(mock_registry, mock_traverser, mock_discovery)
        
        assert engine.registry == mock_registry
        assert engine.traverser == mock_traverser
        assert engine.discovery == mock_discovery
        assert engine._recommendation_cache == {}
        assert not engine._cache_valid

    def test_recommend_for_use_case(self, engine, mock_discovery, mock_registry, sample_nodes):
        """Test recommending pipelines for use case."""
        # Mock text search results
        mock_discovery.search_by_text.return_value = [
            ("training_pipeline", 2.5),
            ("evaluation_pipeline", 1.8)
        ]
        
        # Mock tag search results
        mock_discovery.find_by_tags.return_value = ["training_pipeline", "pytorch_training"]
        
        # Mock registry responses
        mock_registry.get_pipeline_node.side_effect = lambda pid: sample_nodes.get(pid)
        
        # Mock constraint checking
        engine._meets_constraints = Mock(return_value=True)
        
        result = engine.recommend_for_use_case("machine learning training")
        
        assert len(result) > 0
        assert all(isinstance(r, RecommendationResult) for r in result)
        assert all(r.score > 0 for r in result)
        
        # Results should be sorted by score (descending)
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_recommend_for_use_case_with_constraints(self, engine, mock_discovery, mock_registry, sample_nodes):
        """Test recommending pipelines with constraints."""
        mock_discovery.search_by_text.return_value = [("training_pipeline", 2.0)]
        mock_discovery.find_by_tags.return_value = ["training_pipeline"]
        mock_registry.get_pipeline_node.side_effect = lambda pid: sample_nodes.get(pid)
        
        # Mock constraint checking to filter results
        def mock_meets_constraints(pipeline_id, constraints):
            if constraints.get("framework") == "xgboost":
                return pipeline_id in ["training_pipeline", "evaluation_pipeline"]
            return True
        
        engine._meets_constraints = mock_meets_constraints
        
        result = engine.recommend_for_use_case(
            "training models",
            constraints={"framework": "xgboost"}
        )
        
        assert len(result) > 0
        # Should only include xgboost pipelines
        for r in result:
            assert r.framework == "xgboost"

    def test_recommend_for_use_case_error(self, engine, mock_discovery):
        """Test recommend for use case with error."""
        mock_discovery.search_by_text.side_effect = Exception("Search error")
        
        result = engine.recommend_for_use_case("test use case")
        assert result == []

    def test_recommend_next_steps(self, engine, mock_traverser, mock_registry, sample_nodes):
        """Test recommending next steps."""
        # Mock connections
        used_in_conn = PipelineConnection(
            target_id="evaluation_pipeline",
            connection_type="used_in",
            annotation="Evaluate trained model",
            source_id="training_pipeline"
        )
        
        related_conn = PipelineConnection(
            target_id="pytorch_training",
            connection_type="related",
            annotation="Alternative training approach",
            source_id="training_pipeline"
        )
        
        mock_traverser.get_all_connections.return_value = {
            "used_in": [used_in_conn],
            "related": [related_conn],
            "alternatives": []
        }
        
        # Mock registry responses
        mock_registry.get_pipeline_node.side_effect = lambda pid: sample_nodes.get(pid)
        
        # Mock logical progression check
        engine._is_logical_progression = Mock(return_value=True)
        
        # Mock discovery for similar pipelines (empty to focus on connections)
        engine.discovery.suggest_similar_pipelines = Mock(return_value=[])
        
        result = engine.recommend_next_steps("training_pipeline")
        
        assert len(result) > 0
        assert any(r.pipeline_id == "evaluation_pipeline" for r in result)
        
        # Used_in connections should have higher scores
        eval_rec = next(r for r in result if r.pipeline_id == "evaluation_pipeline")
        assert eval_rec.score == 3.0

    def test_recommend_next_steps_complexity_progression(self, engine, mock_traverser, mock_discovery, mock_registry, sample_nodes):
        """Test next steps with complexity progression."""
        mock_traverser.get_all_connections.return_value = {"used_in": [], "related": [], "alternatives": []}
        
        # Mock similar pipelines with higher complexity
        mock_discovery.suggest_similar_pipelines.return_value = [
            ("evaluation_pipeline", 0.6),  # standard complexity
            ("pytorch_training", 0.4)      # advanced complexity
        ]
        
        mock_registry.get_pipeline_node.side_effect = lambda pid: sample_nodes.get(pid)
        
        result = engine.recommend_next_steps("training_pipeline")  # simple complexity
        
        assert len(result) > 0
        # Should recommend more complex similar pipelines
        assert any(r.pipeline_id == "evaluation_pipeline" for r in result)
        assert any(r.pipeline_id == "pytorch_training" for r in result)

    def test_recommend_alternatives(self, engine, mock_traverser, mock_registry, sample_nodes):
        """Test recommending alternatives."""
        # Mock direct alternatives
        alt_conn = PipelineConnection(
            target_id="pytorch_training",
            connection_type="alternatives",
            annotation="Deep learning alternative",
            source_id="training_pipeline"
        )
        
        mock_traverser.get_alternatives.return_value = [alt_conn]
        mock_registry.get_pipeline_node.side_effect = lambda pid: sample_nodes.get(pid)
        
        # Mock discovery for similar pipelines (empty to focus on direct alternatives)
        engine.discovery.suggest_similar_pipelines = Mock(return_value=[])
        
        result = engine.recommend_alternatives("training_pipeline")
        
        assert len(result) > 0
        assert any(r.pipeline_id == "pytorch_training" for r in result)
        
        # Direct alternatives should have high scores
        pytorch_rec = next(r for r in result if r.pipeline_id == "pytorch_training")
        assert pytorch_rec.score >= 3.0

    def test_recommend_alternatives_different_frameworks(self, engine, mock_traverser, mock_discovery, mock_registry, sample_nodes):
        """Test recommending alternatives with different frameworks."""
        mock_traverser.get_alternatives.return_value = []
        
        # Mock similar pipelines with different frameworks
        mock_discovery.suggest_similar_pipelines.return_value = [
            ("pytorch_training", 0.5)  # Different framework
        ]
        
        mock_registry.get_pipeline_node.side_effect = lambda pid: sample_nodes.get(pid)
        
        result = engine.recommend_alternatives("training_pipeline")  # xgboost framework
        
        assert len(result) > 0
        pytorch_rec = next(r for r in result if r.pipeline_id == "pytorch_training")
        assert "Alternative framework" in pytorch_rec.reasoning

    def test_recommend_alternatives_with_reason(self, engine, mock_traverser, mock_registry, sample_nodes):
        """Test recommending alternatives with specific reason."""
        alt_conn = PipelineConnection(
            target_id="evaluation_pipeline",
            connection_type="alternatives",
            annotation="Simpler evaluation approach",
            source_id="pytorch_training"
        )
        
        mock_traverser.get_alternatives.return_value = [alt_conn]
        mock_registry.get_pipeline_node.side_effect = lambda pid: sample_nodes.get(pid)
        engine._is_simpler = Mock(return_value=True)
        
        # Mock discovery for similar pipelines (empty to focus on direct alternatives)
        engine.discovery.suggest_similar_pipelines = Mock(return_value=[])
        
        result = engine.recommend_alternatives("pytorch_training", reason="simplicity")
        
        assert len(result) > 0
        # Should get bonus score for simplicity
        eval_rec = next(r for r in result if r.pipeline_id == "evaluation_pipeline")
        assert eval_rec.score > 3.0

    def test_recommend_compositions(self, engine, mock_registry, sample_nodes):
        """Test recommending pipeline compositions."""
        mock_registry.get_pipeline_node.side_effect = lambda pid: sample_nodes.get(pid)
        
        # Mock composition recommendation methods
        engine._recommend_sequential_compositions = Mock(return_value=[
            CompositionRecommendation(
                pipeline_sequence=["training_pipeline", "evaluation_pipeline"],
                composition_type="sequential",
                description="Training -> Evaluation workflow",
                estimated_complexity="standard",
                total_score=3.5
            )
        ])
        
        engine._recommend_parallel_compositions = Mock(return_value=[])
        engine._recommend_conditional_compositions = Mock(return_value=[])
        
        result = engine.recommend_compositions(["training_pipeline", "evaluation_pipeline"])
        
        assert len(result) > 0
        assert all(isinstance(r, CompositionRecommendation) for r in result)
        assert result[0].composition_type == "sequential"

    def test_recommend_compositions_insufficient_pipelines(self, engine):
        """Test recommending compositions with insufficient pipelines."""
        result = engine.recommend_compositions(["single_pipeline"])
        assert result == []

    def test_get_learning_path(self, engine, mock_discovery):
        """Test getting learning path."""
        # Mock complexity-based discovery
        mock_discovery.find_by_complexity.side_effect = lambda complexity: {
            "simple": ["training_pipeline"],
            "standard": ["evaluation_pipeline"],
            "advanced": ["pytorch_training"]
        }.get(complexity, [])
        
        # Mock best candidate selection
        engine._select_best_learning_candidate = Mock(side_effect=lambda candidates, prev: candidates[0] if candidates else None)
        
        result = engine.get_learning_path(start_complexity="simple", target_framework="any")
        
        assert len(result) > 0
        assert result[0] == "training_pipeline"  # Should start with simple

    def test_get_learning_path_specific_framework(self, engine, mock_discovery):
        """Test getting learning path for specific framework."""
        mock_discovery.find_by_complexity.return_value = ["training_pipeline", "evaluation_pipeline"]
        mock_discovery.find_by_framework.return_value = ["training_pipeline"]
        
        engine._select_best_learning_candidate = Mock(return_value="training_pipeline")
        
        result = engine.get_learning_path(target_framework="xgboost")
        
        assert len(result) > 0
        # Should only include xgboost pipelines

    def test_meets_constraints(self, engine, mock_registry, sample_nodes):
        """Test constraint checking."""
        mock_registry.get_pipeline_node.return_value = sample_nodes["training_pipeline"]
        
        # Test framework constraint
        assert engine._meets_constraints("training_pipeline", {"framework": "xgboost"})
        assert not engine._meets_constraints("training_pipeline", {"framework": "pytorch"})
        
        # Test complexity constraint
        assert engine._meets_constraints("training_pipeline", {"complexity": "simple"})
        assert not engine._meets_constraints("training_pipeline", {"complexity": "advanced"})

    def test_meets_constraints_no_node(self, engine, mock_registry):
        """Test constraint checking with non-existent node."""
        mock_registry.get_pipeline_node.return_value = None
        
        result = engine._meets_constraints("nonexistent", {"framework": "xgboost"})
        assert result is False

    def test_is_logical_progression(self, engine):
        """Test logical progression checking."""
        training_node = {
            "multi_dimensional_tags": {
                "task_tags": ["training"]
            }
        }
        
        evaluation_node = {
            "multi_dimensional_tags": {
                "task_tags": ["evaluation"]
            }
        }
        
        preprocessing_node = {
            "multi_dimensional_tags": {
                "task_tags": ["preprocessing"]
            }
        }
        
        # Training -> Evaluation should be logical
        assert engine._is_logical_progression(training_node, evaluation_node)
        
        # Preprocessing -> Training should be logical
        assert engine._is_logical_progression(preprocessing_node, training_node)
        
        # Evaluation -> Training should not be logical
        assert not engine._is_logical_progression(evaluation_node, training_node)

    def test_is_simpler(self, engine):
        """Test simplicity comparison."""
        simple_node = {
            "zettelkasten_metadata": {
                "complexity": "simple"
            }
        }
        
        advanced_node = {
            "zettelkasten_metadata": {
                "complexity": "advanced"
            }
        }
        
        # Simple should be simpler than advanced
        assert engine._is_simpler(simple_node, advanced_node)
        
        # Advanced should not be simpler than simple
        assert not engine._is_simpler(advanced_node, simple_node)

    def test_recommend_sequential_compositions(self, engine, sample_nodes):
        """Test sequential composition recommendations."""
        pipeline_info = {
            "training_pipeline": {
                "node": sample_nodes["training_pipeline"],
                "framework": "xgboost",
                "complexity": "simple",
                "task_tags": ["training"]
            },
            "evaluation_pipeline": {
                "node": sample_nodes["evaluation_pipeline"],
                "framework": "xgboost",
                "complexity": "standard",
                "task_tags": ["evaluation"]
            }
        }
        
        result = engine._recommend_sequential_compositions(
            ["training_pipeline", "evaluation_pipeline"],
            pipeline_info
        )
        
        assert len(result) > 0
        assert result[0].composition_type == "sequential"
        assert result[0].total_score > 0

    def test_recommend_parallel_compositions(self, engine, sample_nodes):
        """Test parallel composition recommendations."""
        pipeline_info = {
            "training_pipeline": {
                "node": sample_nodes["training_pipeline"],
                "framework": "xgboost",
                "complexity": "simple",
                "task_tags": ["training"]
            },
            "pytorch_training": {
                "node": sample_nodes["pytorch_training"],
                "framework": "pytorch",
                "complexity": "advanced",
                "task_tags": ["training"]
            }
        }
        
        result = engine._recommend_parallel_compositions(
            ["training_pipeline", "pytorch_training"],
            pipeline_info
        )
        
        assert len(result) > 0
        assert result[0].composition_type == "parallel"
        assert "ensemble" in result[0].description.lower()

    def test_recommend_conditional_compositions(self, engine, sample_nodes):
        """Test conditional composition recommendations."""
        pipeline_info = {
            "training_pipeline": {
                "node": sample_nodes["training_pipeline"],
                "framework": "xgboost",
                "complexity": "simple",
                "task_tags": ["training"]
            },
            "pytorch_training": {
                "node": sample_nodes["pytorch_training"],
                "framework": "pytorch",
                "complexity": "advanced",
                "task_tags": ["training"]
            }
        }
        
        result = engine._recommend_conditional_compositions(
            ["training_pipeline", "pytorch_training"],
            pipeline_info
        )
        
        assert len(result) > 0
        assert result[0].composition_type == "conditional"
        assert "adaptive" in result[0].description.lower()

    def test_select_best_learning_candidate(self, engine, mock_registry, mock_traverser, sample_nodes):
        """Test selecting best learning candidate."""
        candidates = ["training_pipeline", "evaluation_pipeline"]
        
        mock_registry.get_pipeline_node.side_effect = lambda pid: sample_nodes.get(pid)
        mock_traverser.get_all_connections.return_value = {
            "related": [PipelineConnection(
                target_id="evaluation_pipeline",
                connection_type="related",
                annotation="Related pipeline"
            )]
        }
        
        # Test with previous pipeline (should prefer connected ones)
        result = engine._select_best_learning_candidate(candidates, "training_pipeline")
        assert result == "evaluation_pipeline"  # Connected to training_pipeline
        
        # Test without previous pipeline
        result = engine._select_best_learning_candidate(candidates, None)
        assert result in candidates

    def test_select_best_learning_candidate_empty(self, engine):
        """Test selecting from empty candidates."""
        result = engine._select_best_learning_candidate([], None)
        assert result is None

    def test_select_best_learning_candidate_single(self, engine):
        """Test selecting from single candidate."""
        result = engine._select_best_learning_candidate(["single_pipeline"], None)
        assert result == "single_pipeline"

    def test_clear_cache(self, engine):
        """Test clearing recommendation cache."""
        engine._recommendation_cache = {"test": "data"}
        engine._cache_valid = True
        
        engine.clear_cache()
        
        assert engine._recommendation_cache == {}
        assert not engine._cache_valid

    def test_error_handling_in_methods(self, engine, mock_registry, mock_traverser, mock_discovery):
        """Test error handling in various methods."""
        # Mock errors in dependencies
        mock_discovery.search_by_text.side_effect = Exception("Discovery error")
        mock_traverser.get_all_connections.side_effect = Exception("Traverser error")
        mock_registry.get_pipeline_node.side_effect = Exception("Registry error")
        
        # Test methods handle errors gracefully
        assert engine.recommend_for_use_case("test") == []
        assert engine.recommend_next_steps("test") == []
        assert engine.recommend_alternatives("test") == []
        assert engine.recommend_compositions(["test1", "test2"]) == []
        assert engine.get_learning_path() == []

    def test_recommendation_result_model(self):
        """Test RecommendationResult model."""
        result = RecommendationResult(
            pipeline_id="test_pipeline",
            title="Test Pipeline",
            score=2.5,
            reasoning="Test reasoning",
            connection_path=["source", "target"],
            tag_overlap=0.8,
            framework="xgboost",
            complexity="simple"
        )
        
        assert result.pipeline_id == "test_pipeline"
        assert result.title == "Test Pipeline"
        assert result.score == 2.5
        assert result.reasoning == "Test reasoning"
        assert result.connection_path == ["source", "target"]
        assert result.tag_overlap == 0.8
        assert result.framework == "xgboost"
        assert result.complexity == "simple"

    def test_composition_recommendation_model(self):
        """Test CompositionRecommendation model."""
        composition = CompositionRecommendation(
            pipeline_sequence=["pipeline1", "pipeline2"],
            composition_type="sequential",
            description="Test composition",
            estimated_complexity="standard",
            total_score=3.0
        )
        
        assert composition.pipeline_sequence == ["pipeline1", "pipeline2"]
        assert composition.composition_type == "sequential"
        assert composition.description == "Test composition"
        assert composition.estimated_complexity == "standard"
        assert composition.total_score == 3.0

    def test_complex_recommendation_scenario(self, engine, mock_registry, mock_traverser, mock_discovery, sample_nodes):
        """Test complex recommendation scenario with multiple factors."""
        # Setup complex scenario
        mock_discovery.search_by_text.return_value = [("training_pipeline", 3.0)]
        mock_discovery.find_by_tags.return_value = ["training_pipeline", "evaluation_pipeline"]
        mock_registry.get_pipeline_node.side_effect = lambda pid: sample_nodes.get(pid)
        
        # Mock connections
        mock_traverser.get_all_connections.return_value = {
            "used_in": [PipelineConnection(
                target_id="evaluation_pipeline",
                connection_type="used_in",
                annotation="Natural next step"
            )],
            "alternatives": [PipelineConnection(
                target_id="pytorch_training",
                connection_type="alternatives",
                annotation="Deep learning alternative"
            )],
            "related": []
        }
        
        # Mock alternatives method specifically
        mock_traverser.get_alternatives.return_value = [PipelineConnection(
            target_id="pytorch_training",
            connection_type="alternatives",
            annotation="Deep learning alternative"
        )]
        
        # Mock similarity
        mock_discovery.suggest_similar_pipelines.return_value = [
            ("evaluation_pipeline", 0.7),
            ("pytorch_training", 0.5)
        ]
        
        engine._meets_constraints = Mock(return_value=True)
        engine._is_logical_progression = Mock(return_value=True)
        
        # Test use case recommendation
        use_case_results = engine.recommend_for_use_case("machine learning training")
        assert len(use_case_results) > 0
        
        # Test next steps
        next_steps = engine.recommend_next_steps("training_pipeline")
        assert len(next_steps) > 0
        assert any(r.pipeline_id == "evaluation_pipeline" for r in next_steps)
        
        # Test alternatives
        alternatives = engine.recommend_alternatives("training_pipeline")
        assert len(alternatives) > 0
        assert any(r.pipeline_id == "pytorch_training" for r in alternatives)

    def test_constraint_edge_cases(self, engine, mock_registry):
        """Test constraint checking edge cases."""
        # Test with missing metadata
        node_no_metadata = {"id": "test", "title": "Test"}
        mock_registry.get_pipeline_node.return_value = node_no_metadata
        
        # Should handle missing metadata gracefully
        assert engine._meets_constraints("test", {"framework": "xgboost"}) is False
        
        # Test with resource constraints
        node_with_resources = {
            "id": "test",
            "discovery_metadata": {
                "resource_requirements": "high"
            }
        }
        mock_registry.get_pipeline_node.return_value = node_with_resources
        
        assert engine._meets_constraints("test", {"max_resources": "medium"}) is False
        assert engine._meets_constraints("test", {"max_resources": "high"}) is True

    def test_learning_path_edge_cases(self, engine, mock_discovery):
        """Test learning path edge cases."""
        # Test with invalid start complexity
        mock_discovery.find_by_complexity.return_value = ["test_pipeline"]
        engine._select_best_learning_candidate = Mock(return_value="test_pipeline")
        
        result = engine.get_learning_path(start_complexity="invalid")
        assert len(result) > 0  # Should default to "simple"
        
        # Test with no candidates at any level
        mock_discovery.find_by_complexity.return_value = []
        
        result = engine.get_learning_path()
        assert result == []

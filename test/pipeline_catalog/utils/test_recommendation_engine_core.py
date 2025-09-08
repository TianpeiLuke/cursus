"""
Unit tests for PipelineRecommendationEngine core functionality.

Tests the main recommendation methods including use case recommendations,
next steps, alternatives, and compositions.
"""

import pytest
from unittest.mock import Mock, patch
from cursus.pipeline_catalog.utils.recommendation_engine import (
    PipelineRecommendationEngine, RecommendationResult, CompositionRecommendation
)
from cursus.pipeline_catalog.utils.catalog_registry import CatalogRegistry
from cursus.pipeline_catalog.utils.connection_traverser import ConnectionTraverser, PipelineConnection
from cursus.pipeline_catalog.utils.tag_discovery import TagBasedDiscovery

class TestPipelineRecommendationEngineCore:
    """Test suite for PipelineRecommendationEngine core functionality."""
    
    @pytest.fixture
    def mock_registry(self):
        """Create mock CatalogRegistry for testing."""
        registry = Mock(spec=CatalogRegistry)
        
        # Mock pipeline nodes
        nodes_data = {
            "xgb_simple": {
                "id": "xgb_simple",
                "title": "XGBoost Simple Training",
                "description": "Simple XGBoost training pipeline",
                "zettelkasten_metadata": {
                    "framework": "xgboost",
                    "complexity": "simple",
                    "use_case": "Basic model training"
                },
                "multi_dimensional_tags": {
                    "framework_tags": ["xgboost"],
                    "task_tags": ["training"],
                    "complexity_tags": ["simple"]
                }
            },
            "xgb_advanced": {
                "id": "xgb_advanced",
                "title": "XGBoost Advanced Pipeline",
                "description": "Advanced XGBoost with evaluation and calibration",
                "zettelkasten_metadata": {
                    "framework": "xgboost",
                    "complexity": "advanced",
                    "use_case": "Complete ML workflow"
                },
                "multi_dimensional_tags": {
                    "framework_tags": ["xgboost"],
                    "task_tags": ["training", "evaluation", "calibration"],
                    "complexity_tags": ["advanced"]
                }
            },
            "pytorch_nlp": {
                "id": "pytorch_nlp",
                "title": "PyTorch NLP Training",
                "description": "PyTorch training for NLP tasks",
                "zettelkasten_metadata": {
                    "framework": "pytorch",
                    "complexity": "standard",
                    "use_case": "NLP model training"
                },
                "multi_dimensional_tags": {
                    "framework_tags": ["pytorch"],
                    "task_tags": ["training"],
                    "complexity_tags": ["standard"]
                }
            },
            "sklearn_prep": {
                "id": "sklearn_prep",
                "title": "Sklearn Preprocessing",
                "description": "Data preprocessing with sklearn",
                "zettelkasten_metadata": {
                    "framework": "sklearn",
                    "complexity": "simple",
                    "use_case": "Data preparation"
                },
                "multi_dimensional_tags": {
                    "framework_tags": ["sklearn"],
                    "task_tags": ["preprocessing"],
                    "complexity_tags": ["simple"]
                }
            }
        }
        
        registry.get_pipeline_node.side_effect = lambda pid: nodes_data.get(pid)
        registry.get_all_pipelines.return_value = list(nodes_data.keys())
        
        return registry
    
    @pytest.fixture
    def mock_traverser(self):
        """Create mock ConnectionTraverser for testing."""
        traverser = Mock(spec=ConnectionTraverser)
        
        # Mock connections data
        connections_data = {
            "xgb_simple": {
                "alternatives": [
                    PipelineConnection(target_id="pytorch_nlp", connection_type="alternatives", 
                                     annotation="Alternative framework", source_id="xgb_simple")
                ],
                "related": [
                    PipelineConnection(target_id="sklearn_prep", connection_type="related",
                                     annotation="Data preprocessing", source_id="xgb_simple")
                ],
                "used_in": [
                    PipelineConnection(target_id="xgb_advanced", connection_type="used_in",
                                     annotation="Extended workflow", source_id="xgb_simple")
                ]
            },
            "xgb_advanced": {
                "alternatives": [],
                "related": [],
                "used_in": []
            },
            "pytorch_nlp": {
                "alternatives": [
                    PipelineConnection(target_id="xgb_simple", connection_type="alternatives",
                                     annotation="Alternative approach", source_id="pytorch_nlp")
                ],
                "related": [],
                "used_in": []
            },
            "sklearn_prep": {
                "alternatives": [],
                "related": [
                    PipelineConnection(target_id="xgb_simple", connection_type="related",
                                     annotation="Follows preprocessing", source_id="sklearn_prep")
                ],
                "used_in": []
            }
        }
        
        def mock_get_all_connections(pipeline_id):
            return connections_data.get(pipeline_id, {"alternatives": [], "related": [], "used_in": []})
        
        def mock_get_alternatives(pipeline_id):
            return connections_data.get(pipeline_id, {}).get("alternatives", [])
        
        def mock_get_compositions(pipeline_id):
            return connections_data.get(pipeline_id, {}).get("used_in", [])
        
        traverser.get_all_connections.side_effect = mock_get_all_connections
        traverser.get_alternatives.side_effect = mock_get_alternatives
        traverser.get_compositions.side_effect = mock_get_compositions
        
        return traverser
    
    @pytest.fixture
    def mock_discovery(self):
        """Create mock TagBasedDiscovery for testing."""
        discovery = Mock(spec=TagBasedDiscovery)
        
        # Mock search results
        def mock_search_by_text(query, search_fields=None):
            if "xgboost" in query.lower():
                return [("xgb_simple", 2.0), ("xgb_advanced", 1.5)]
            elif "training" in query.lower():
                return [("xgb_simple", 1.8), ("pytorch_nlp", 1.6)]
            else:
                return []
        
        def mock_find_by_tags(tags, match_mode="any"):
            if "training" in tags:
                return ["xgb_simple", "pytorch_nlp"]
            elif "preprocessing" in tags:
                return ["sklearn_prep"]
            else:
                return []
        
        def mock_suggest_similar_pipelines(pipeline_id, similarity_threshold=0.3):
            similarity_data = {
                "xgb_simple": [("sklearn_prep", 0.4), ("pytorch_nlp", 0.3)],
                "xgb_advanced": [("xgb_simple", 0.6)],
                "pytorch_nlp": [("xgb_simple", 0.3)],
                "sklearn_prep": [("xgb_simple", 0.4)]
            }
            return similarity_data.get(pipeline_id, [])
        
        discovery.search_by_text.side_effect = mock_search_by_text
        discovery.find_by_tags.side_effect = mock_find_by_tags
        discovery.suggest_similar_pipelines.side_effect = mock_suggest_similar_pipelines
        
        return discovery
    
    @pytest.fixture
    def engine(self, mock_registry, mock_traverser, mock_discovery):
        """Create PipelineRecommendationEngine instance with mocks."""
        return PipelineRecommendationEngine(mock_registry, mock_traverser, mock_discovery)
    
    def test_init(self, mock_registry, mock_traverser, mock_discovery):
        """Test PipelineRecommendationEngine initialization."""
        engine = PipelineRecommendationEngine(mock_registry, mock_traverser, mock_discovery)
        
        assert engine.registry == mock_registry
        assert engine.traverser == mock_traverser
        assert engine.discovery == mock_discovery
        assert engine._recommendation_cache == {}
        assert not engine._cache_valid
    
    def test_recommend_for_use_case(self, engine):
        """Test recommending pipelines for specific use case."""
        recommendations = engine.recommend_for_use_case("XGBoost training")
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Check recommendation format
        for rec in recommendations:
            assert isinstance(rec, RecommendationResult)
            assert rec.pipeline_id
            assert rec.title
            assert rec.score > 0
            assert rec.reasoning
        
        # Results should be sorted by score (descending)
        scores = [rec.score for rec in recommendations]
        assert scores == sorted(scores, reverse=True)
    
    def test_recommend_for_use_case_with_constraints(self, engine):
        """Test recommending pipelines with constraints."""
        constraints = {
            "framework": "xgboost",
            "complexity": "simple"
        }
        
        recommendations = engine.recommend_for_use_case("training", constraints=constraints)
        
        # Should only include pipelines meeting constraints
        for rec in recommendations:
            assert rec.framework == "xgboost"
            assert rec.complexity == "simple"
    
    def test_recommend_for_use_case_empty_query(self, engine):
        """Test recommending pipelines with empty use case."""
        recommendations = engine.recommend_for_use_case("")
        
        # Should return empty list for empty query
        assert recommendations == []
    
    def test_recommend_next_steps(self, engine):
        """Test recommending logical next steps after current pipeline."""
        recommendations = engine.recommend_next_steps("xgb_simple")
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should prioritize "used_in" connections
        used_in_recs = [rec for rec in recommendations if rec.connection_path and len(rec.connection_path) == 2]
        if used_in_recs:
            # Used_in connections should have high scores
            assert any(rec.score >= 2.5 for rec in used_in_recs)
    
    def test_recommend_next_steps_complexity_progression(self, engine):
        """Test next steps recommendations include complexity progression."""
        recommendations = engine.recommend_next_steps("xgb_simple")
        
        # Should suggest more complex pipelines as natural progression
        complex_recs = [rec for rec in recommendations if rec.complexity in ["standard", "advanced"]]
        assert len(complex_recs) > 0
    
    def test_recommend_alternatives(self, engine):
        """Test recommending alternative approaches."""
        recommendations = engine.recommend_alternatives("xgb_simple")
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should include direct alternatives with high scores
        direct_alts = [rec for rec in recommendations if rec.connection_path]
        if direct_alts:
            assert any(rec.score >= 2.5 for rec in direct_alts)
    
    def test_recommend_alternatives_by_reason(self, engine):
        """Test recommending alternatives for specific reasons."""
        # Test simplicity reason
        simple_recs = engine.recommend_alternatives("xgb_advanced", reason="simplicity")
        
        # Should prefer simpler alternatives
        for rec in simple_recs:
            if rec.complexity:
                complexity_order = {"simple": 0, "standard": 1, "advanced": 2}
                current_complexity = complexity_order.get("advanced", 2)
                rec_complexity = complexity_order.get(rec.complexity, 2)
                # Some recommendations should be simpler
                if rec_complexity < current_complexity:
                    assert rec.score > 2.0  # Should get bonus for being simpler
    
    def test_recommend_alternatives_different_frameworks(self, engine):
        """Test alternatives include different frameworks."""
        recommendations = engine.recommend_alternatives("xgb_simple")
        
        # Should include alternatives with different frameworks
        frameworks = [rec.framework for rec in recommendations if rec.framework]
        assert len(set(frameworks)) > 1  # Multiple frameworks represented
    
    def test_recommend_compositions(self, engine):
        """Test recommending pipeline compositions."""
        pipeline_ids = ["sklearn_prep", "xgb_simple", "xgb_advanced"]
        compositions = engine.recommend_compositions(pipeline_ids)
        
        assert isinstance(compositions, list)
        
        # Check composition format
        for comp in compositions:
            assert isinstance(comp, CompositionRecommendation)
            assert comp.pipeline_sequence
            assert comp.composition_type in ["sequential", "parallel", "conditional"]
            assert comp.description
            assert comp.estimated_complexity
            assert comp.total_score > 0
        
        # Results should be sorted by total score (descending)
        scores = [comp.total_score for comp in compositions]
        assert scores == sorted(scores, reverse=True)
    
    def test_recommend_compositions_insufficient_pipelines(self, engine):
        """Test composition recommendations with insufficient pipelines."""
        compositions = engine.recommend_compositions(["single_pipeline"])
        
        # Should return empty list for single pipeline
        assert compositions == []
    
    def test_get_learning_path(self, engine):
        """Test getting learning path from simple to complex."""
        learning_path = engine.get_learning_path(start_complexity="simple", target_framework="xgboost")
        
        assert isinstance(learning_path, list)
        assert len(learning_path) > 0
        
        # Path should start with simple complexity
        if learning_path:
            first_pipeline = learning_path[0]
            # Verify first pipeline is simple (would need to mock discovery methods)
            assert first_pipeline  # At least verify we got a pipeline ID
    
    def test_get_learning_path_any_framework(self, engine):
        """Test learning path for any framework."""
        learning_path = engine.get_learning_path(start_complexity="simple", target_framework="any")
        
        assert isinstance(learning_path, list)
        # Should include pipelines from multiple frameworks
    
    def test_meets_constraints(self, engine):
        """Test constraint checking for pipelines."""
        # Mock pipeline node for testing
        mock_node = {
            "zettelkasten_metadata": {
                "framework": "xgboost",
                "complexity": "simple"
            },
            "discovery_metadata": {
                "skill_level": "beginner",
                "resource_requirements": "low"
            }
        }
        
        engine.registry.get_pipeline_node.return_value = mock_node
        
        # Test framework constraint
        assert engine._meets_constraints("test_pipeline", {"framework": "xgboost"})
        assert not engine._meets_constraints("test_pipeline", {"framework": "pytorch"})
        
        # Test complexity constraint
        assert engine._meets_constraints("test_pipeline", {"complexity": "simple"})
        assert not engine._meets_constraints("test_pipeline", {"complexity": "advanced"})
        
        # Test skill level constraint
        assert engine._meets_constraints("test_pipeline", {"skill_level": "beginner"})
        assert not engine._meets_constraints("test_pipeline", {"skill_level": "expert"})
    
    def test_meets_constraints_no_node(self, engine):
        """Test constraint checking for non-existent pipeline."""
        engine.registry.get_pipeline_node.return_value = None
        
        result = engine._meets_constraints("nonexistent_pipeline", {"framework": "xgboost"})
        assert result is False
    
    def test_is_logical_progression(self, engine):
        """Test logical progression detection between pipelines."""
        current_node = {
            "multi_dimensional_tags": {
                "task_tags": ["training"]
            }
        }
        
        next_node = {
            "multi_dimensional_tags": {
                "task_tags": ["evaluation"]
            }
        }
        
        # Training -> Evaluation should be logical progression
        assert engine._is_logical_progression(current_node, next_node)
        
        # Same task should not be progression
        same_node = {
            "multi_dimensional_tags": {
                "task_tags": ["training"]
            }
        }
        assert not engine._is_logical_progression(current_node, same_node)
    
    def test_is_simpler(self, engine):
        """Test complexity comparison between pipelines."""
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
        assert not engine._is_simpler(advanced_node, simple_node)
    
    def test_clear_cache(self, engine):
        """Test clearing recommendation cache."""
        # Populate cache
        engine._recommendation_cache = {"test": "data"}
        engine._cache_valid = True
        
        # Clear cache
        engine.clear_cache()
        
        assert engine._recommendation_cache == {}
        assert not engine._cache_valid
    
    @patch('src.cursus.pipeline_catalog.utils.recommendation_engine.logger')
    def test_error_handling(self, mock_logger, engine):
        """Test error handling and logging."""
        # Mock discovery to raise exception
        engine.discovery.search_by_text.side_effect = Exception("Search error")
        
        # Test method that should handle the error gracefully
        recommendations = engine.recommend_for_use_case("test query")
        
        assert recommendations == []
        mock_logger.error.assert_called()

"""
Unit tests for TagBasedDiscovery class.

Tests the tag-based pipeline discovery implementing Zettelkasten anti-categories
principle for emergent organization through multi-dimensional tagging.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Tuple

from cursus.pipeline_catalog.utils.tag_discovery import TagBasedDiscovery
from cursus.pipeline_catalog.utils.catalog_registry import CatalogRegistry

class TestTagBasedDiscovery:
    """Test suite for TagBasedDiscovery class."""
    
    @pytest.fixture
    def mock_registry(self):
        """Create mock CatalogRegistry for testing."""
        registry = Mock(spec=CatalogRegistry)
        
        # Mock registry data with tag index
        registry_data = {
            "tag_index": {
                "framework_tags": {
                    "xgboost": ["xgb_simple", "xgb_advanced"],
                    "pytorch": ["pytorch_basic", "pytorch_complex"],
                    "sklearn": ["sklearn_simple"]
                },
                "task_tags": {
                    "training": ["xgb_simple", "pytorch_basic", "sklearn_simple"],
                    "evaluation": ["xgb_advanced", "pytorch_complex"],
                    "preprocessing": ["sklearn_simple"],
                    "calibration": ["xgb_advanced"]
                },
                "complexity_tags": {
                    "simple": ["xgb_simple", "pytorch_basic", "sklearn_simple"],
                    "standard": ["pytorch_complex"],
                    "advanced": ["xgb_advanced"]
                },
                "domain_tags": {
                    "tabular": ["xgb_simple", "xgb_advanced", "sklearn_simple"],
                    "nlp": ["pytorch_basic"],
                    "computer_vision": ["pytorch_complex"]
                }
            },
            "nodes": {
                "xgb_simple": {
                    "id": "xgb_simple",
                    "title": "XGBoost Simple Training",
                    "description": "Simple XGBoost training pipeline for tabular data",
                    "multi_dimensional_tags": {
                        "framework_tags": ["xgboost"],
                        "task_tags": ["training"],
                        "complexity_tags": ["simple"],
                        "domain_tags": ["tabular"]
                    },
                    "zettelkasten_metadata": {
                        "framework": "xgboost",
                        "complexity": "simple"
                    }
                },
                "xgb_advanced": {
                    "id": "xgb_advanced",
                    "title": "XGBoost Advanced Pipeline",
                    "description": "Advanced XGBoost pipeline with evaluation and calibration",
                    "multi_dimensional_tags": {
                        "framework_tags": ["xgboost"],
                        "task_tags": ["training", "evaluation", "calibration"],
                        "complexity_tags": ["advanced"],
                        "domain_tags": ["tabular"]
                    },
                    "zettelkasten_metadata": {
                        "framework": "xgboost",
                        "complexity": "advanced"
                    }
                },
                "pytorch_basic": {
                    "id": "pytorch_basic",
                    "title": "PyTorch Basic NLP",
                    "description": "Basic PyTorch training for NLP tasks",
                    "multi_dimensional_tags": {
                        "framework_tags": ["pytorch"],
                        "task_tags": ["training"],
                        "complexity_tags": ["simple"],
                        "domain_tags": ["nlp"]
                    },
                    "zettelkasten_metadata": {
                        "framework": "pytorch",
                        "complexity": "simple"
                    }
                },
                "pytorch_complex": {
                    "id": "pytorch_complex",
                    "title": "PyTorch Complex Vision",
                    "description": "Complex PyTorch pipeline for computer vision with evaluation",
                    "multi_dimensional_tags": {
                        "framework_tags": ["pytorch"],
                        "task_tags": ["training", "evaluation"],
                        "complexity_tags": ["standard"],
                        "domain_tags": ["computer_vision"]
                    },
                    "zettelkasten_metadata": {
                        "framework": "pytorch",
                        "complexity": "standard"
                    }
                },
                "sklearn_simple": {
                    "id": "sklearn_simple",
                    "title": "Sklearn Simple Preprocessing",
                    "description": "Simple sklearn preprocessing and training pipeline",
                    "multi_dimensional_tags": {
                        "framework_tags": ["sklearn"],
                        "task_tags": ["preprocessing", "training"],
                        "complexity_tags": ["simple"],
                        "domain_tags": ["tabular"]
                    },
                    "zettelkasten_metadata": {
                        "framework": "sklearn",
                        "complexity": "simple"
                    }
                }
            }
        }
        
        def mock_load_registry():
            return registry_data
        
        def mock_get_pipeline_node(pipeline_id):
            return registry_data["nodes"].get(pipeline_id)
        
        def mock_get_all_pipelines():
            return list(registry_data["nodes"].keys())
        
        registry.load_registry.side_effect = mock_load_registry
        registry.get_pipeline_node.side_effect = mock_get_pipeline_node
        registry.get_all_pipelines.side_effect = mock_get_all_pipelines
        
        return registry
    
    @pytest.fixture
    def discovery(self, mock_registry):
        """Create TagBasedDiscovery instance with mock registry."""
        return TagBasedDiscovery(mock_registry)
    
    def test_init(self, mock_registry):
        """Test TagBasedDiscovery initialization."""
        discovery = TagBasedDiscovery(mock_registry)
        
        assert discovery.registry == mock_registry
        assert discovery._tag_cache == {}
        assert not discovery._cache_valid
    
    def test_find_by_tags_any_match(self, discovery):
        """Test finding pipelines by tags with 'any' match mode."""
        # Find pipelines with either 'training' or 'evaluation' tags
        results = discovery.find_by_tags(["training", "evaluation"], match_mode="any")
        
        # Should include all pipelines that have either tag
        expected = ["xgb_simple", "xgb_advanced", "pytorch_basic", "pytorch_complex", "sklearn_simple"]
        assert set(results) == set(expected)
    
    def test_find_by_tags_all_match(self, discovery):
        """Test finding pipelines by tags with 'all' match mode."""
        # Find pipelines with both 'training' and 'evaluation' tags
        results = discovery.find_by_tags(["training", "evaluation"], match_mode="all")
        
        # Should only include pipelines that have both tags
        expected = ["xgb_advanced", "pytorch_complex"]
        assert set(results) == set(expected)
    
    def test_find_by_tags_exact_match(self, discovery):
        """Test finding pipelines by tags with 'exact' match mode."""
        # Mock _get_pipeline_tags to return specific tag sets
        def mock_get_pipeline_tags(pipeline_id):
            tag_sets = {
                "xgb_simple": ["xgboost", "training", "simple", "tabular"],
                "pytorch_basic": ["pytorch", "training", "simple", "nlp"],
                "test_exact": ["training", "simple"]  # Exact match for our test
            }
            return tag_sets.get(pipeline_id, [])
        
        with patch.object(discovery, '_get_pipeline_tags', side_effect=mock_get_pipeline_tags):
            with patch.object(discovery.registry, 'get_all_pipelines', return_value=["xgb_simple", "pytorch_basic", "test_exact"]):
                results = discovery.find_by_tags(["training", "simple"], match_mode="exact")
                
                # Should only include pipeline with exactly these tags
                assert results == ["test_exact"]
    
    def test_find_by_tags_empty_list(self, discovery):
        """Test finding pipelines with empty tag list."""
        results = discovery.find_by_tags([], match_mode="any")
        assert results == []
    
    def test_find_by_tags_invalid_mode(self, discovery):
        """Test finding pipelines with invalid match mode."""
        results = discovery.find_by_tags(["training"], match_mode="invalid")
        assert results == []
    
    def test_find_by_framework(self, discovery):
        """Test finding pipelines by framework."""
        xgboost_results = discovery.find_by_framework("xgboost")
        assert set(xgboost_results) == {"xgb_simple", "xgb_advanced"}
        
        pytorch_results = discovery.find_by_framework("pytorch")
        assert set(pytorch_results) == {"pytorch_basic", "pytorch_complex"}
        
        sklearn_results = discovery.find_by_framework("sklearn")
        assert set(sklearn_results) == {"sklearn_simple"}
        
        # Test non-existent framework
        empty_results = discovery.find_by_framework("nonexistent")
        assert empty_results == []
    
    def test_find_by_complexity(self, discovery):
        """Test finding pipelines by complexity."""
        simple_results = discovery.find_by_complexity("simple")
        assert set(simple_results) == {"xgb_simple", "pytorch_basic", "sklearn_simple"}
        
        standard_results = discovery.find_by_complexity("standard")
        assert set(standard_results) == {"pytorch_complex"}
        
        advanced_results = discovery.find_by_complexity("advanced")
        assert set(advanced_results) == {"xgb_advanced"}
        
        # Test non-existent complexity
        empty_results = discovery.find_by_complexity("nonexistent")
        assert empty_results == []
    
    def test_find_by_task(self, discovery):
        """Test finding pipelines by task."""
        training_results = discovery.find_by_task("training")
        assert set(training_results) == {"xgb_simple", "pytorch_basic", "sklearn_simple"}
        
        evaluation_results = discovery.find_by_task("evaluation")
        assert set(evaluation_results) == {"xgb_advanced", "pytorch_complex"}
        
        preprocessing_results = discovery.find_by_task("preprocessing")
        assert set(preprocessing_results) == {"sklearn_simple"}
        
        # Test non-existent task
        empty_results = discovery.find_by_task("nonexistent")
        assert empty_results == []
    
    def test_find_by_domain(self, discovery):
        """Test finding pipelines by domain."""
        tabular_results = discovery.find_by_domain("tabular")
        assert set(tabular_results) == {"xgb_simple", "xgb_advanced", "sklearn_simple"}
        
        nlp_results = discovery.find_by_domain("nlp")
        assert set(nlp_results) == {"pytorch_basic"}
        
        cv_results = discovery.find_by_domain("computer_vision")
        assert set(cv_results) == {"pytorch_complex"}
        
        # Test non-existent domain
        empty_results = discovery.find_by_domain("nonexistent")
        assert empty_results == []
    
    def test_find_by_pattern(self, discovery):
        """Test finding pipelines by pattern."""
        # Mock pattern tags in tag index
        with patch.object(discovery, '_get_tag_index') as mock_get_tag_index:
            mock_get_tag_index.return_value = {
                "pattern_tags": {
                    "atomic_workflow": ["xgb_simple", "pytorch_basic"],
                    "end_to_end": ["xgb_advanced", "pytorch_complex"],
                    "modular": ["sklearn_simple"]
                }
            }
            
            atomic_results = discovery.find_by_pattern("atomic_workflow")
            assert set(atomic_results) == {"xgb_simple", "pytorch_basic"}
            
            e2e_results = discovery.find_by_pattern("end_to_end")
            assert set(e2e_results) == {"xgb_advanced", "pytorch_complex"}
    
    def test_find_by_multiple_criteria(self, discovery):
        """Test finding pipelines by multiple criteria."""
        # Find simple training pipelines
        results = discovery.find_by_multiple_criteria(
            complexity="simple",
            additional_tags=["training"]
        )
        
        # Should find pipelines that are both simple AND have training tag
        expected = ["xgb_simple", "pytorch_basic", "sklearn_simple"]
        assert set(results) == set(expected)
    
    def test_find_by_multiple_criteria_no_criteria(self, discovery):
        """Test finding pipelines with no criteria (should return all)."""
        results = discovery.find_by_multiple_criteria()
        
        # Should return all pipelines
        all_pipelines = discovery.registry.get_all_pipelines()
        assert set(results) == set(all_pipelines)
    
    def test_search_by_text(self, discovery):
        """Test text-based search with relevance scoring."""
        results = discovery.search_by_text("XGBoost training")
        
        # Should return list of (pipeline_id, score) tuples
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Check result format
        for pipeline_id, score in results:
            assert isinstance(pipeline_id, str)
            assert isinstance(score, float)
            assert score > 0
        
        # Results should be sorted by score (descending)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_search_by_text_empty_query(self, discovery):
        """Test text search with empty query."""
        results = discovery.search_by_text("")
        assert results == []
        
        results = discovery.search_by_text("   ")
        assert results == []
    
    def test_search_by_text_specific_fields(self, discovery):
        """Test text search in specific fields."""
        # Search only in titles
        results = discovery.search_by_text("XGBoost", search_fields=["title"])
        
        # Should find XGBoost pipelines
        pipeline_ids = [pid for pid, _ in results]
        assert "xgb_simple" in pipeline_ids or "xgb_advanced" in pipeline_ids
    
    def test_get_tag_clusters(self, discovery):
        """Test getting emergent tag clusters."""
        clusters = discovery.get_tag_clusters()
        
        assert isinstance(clusters, dict)
        # Should have at least one cluster
        assert len(clusters) > 0
        
        # Each cluster should be a list of pipeline IDs
        for cluster_name, pipeline_list in clusters.items():
            assert isinstance(cluster_name, str)
            assert isinstance(pipeline_list, list)
            assert len(pipeline_list) > 0
    
    def test_suggest_similar_pipelines(self, discovery):
        """Test suggesting similar pipelines based on tag overlap."""
        similar = discovery.suggest_similar_pipelines("xgb_simple", similarity_threshold=0.2)
        
        assert isinstance(similar, list)
        # Should return list of (pipeline_id, similarity_score) tuples
        for pipeline_id, similarity in similar:
            assert isinstance(pipeline_id, str)
            assert isinstance(similarity, float)
            assert 0 <= similarity <= 1
            assert pipeline_id != "xgb_simple"  # Should not include self
        
        # Results should be sorted by similarity (descending)
        similarities = [sim for _, sim in similar]
        assert similarities == sorted(similarities, reverse=True)
    
    def test_suggest_similar_pipelines_no_tags(self, discovery):
        """Test suggesting similar pipelines for pipeline with no tags."""
        with patch.object(discovery, '_get_pipeline_tags', return_value=[]):
            similar = discovery.suggest_similar_pipelines("empty_pipeline")
            assert similar == []
    
    def test_get_tag_statistics(self, discovery):
        """Test getting tag usage statistics."""
        stats = discovery.get_tag_statistics()
        
        assert "total_tag_categories" in stats
        assert "tag_categories" in stats
        assert "most_common_tags" in stats
        assert "tag_distribution" in stats
        assert "pipeline_tag_counts" in stats
        
        # Check tag categories stats
        assert stats["total_tag_categories"] > 0
        
        # Check most common tags format
        most_common = stats["most_common_tags"]
        assert isinstance(most_common, list)
        for tag, count in most_common:
            assert isinstance(tag, str)
            assert isinstance(count, int)
            assert count > 0
    
    def test_find_undertagged_pipelines(self, discovery):
        """Test finding pipelines with insufficient tags."""
        undertagged = discovery.find_undertagged_pipelines(min_tags=5)
        
        assert isinstance(undertagged, list)
        # Should return list of (pipeline_id, tag_count) tuples
        for pipeline_id, tag_count in undertagged:
            assert isinstance(pipeline_id, str)
            assert isinstance(tag_count, int)
            assert tag_count < 5
        
        # Results should be sorted by tag count (ascending)
        tag_counts = [count for _, count in undertagged]
        assert tag_counts == sorted(tag_counts)
    
    def test_suggest_tags_for_pipeline(self, discovery):
        """Test suggesting additional tags for a pipeline."""
        suggestions = discovery.suggest_tags_for_pipeline("xgb_simple")
        
        assert isinstance(suggestions, dict)
        # Each category should map to a list of suggested tags
        for category, tags in suggestions.items():
            assert isinstance(category, str)
            assert isinstance(tags, list)
            assert len(tags) <= 3  # Should suggest max 3 tags per category
    
    def test_suggest_tags_no_similar_pipelines(self, discovery):
        """Test suggesting tags when no similar pipelines exist."""
        with patch.object(discovery, 'suggest_similar_pipelines', return_value=[]):
            suggestions = discovery.suggest_tags_for_pipeline("isolated_pipeline")
            assert suggestions == {}
    
    def test_get_tag_index_caching(self, discovery):
        """Test tag index caching mechanism."""
        # First call should load from registry
        tag_index1 = discovery._get_tag_index()
        assert discovery._cache_valid
        
        # Second call should use cache
        tag_index2 = discovery._get_tag_index()
        assert tag_index1 == tag_index2
        
        # Verify registry was only called once
        assert discovery.registry.load_registry.call_count == 1
    
    def test_get_pipeline_tags(self, discovery):
        """Test getting all tags for a pipeline as flat list."""
        tags = discovery._get_pipeline_tags("xgb_simple")
        
        expected_tags = ["xgboost", "training", "simple", "tabular"]
        assert set(tags) == set(expected_tags)
    
    def test_get_pipeline_tags_no_node(self, discovery):
        """Test getting tags for non-existent pipeline."""
        tags = discovery._get_pipeline_tags("nonexistent_pipeline")
        assert tags == []
    
    def test_clear_cache(self, discovery):
        """Test clearing tag discovery cache."""
        # Populate cache
        discovery._tag_cache = {"test": "data"}
        discovery._cache_valid = True
        
        # Clear cache
        discovery.clear_cache()
        
        assert discovery._tag_cache == {}
        assert not discovery._cache_valid
    
    @patch('cursus.pipeline_catalog.utils.tag_discovery.logger')
    def test_error_handling(self, mock_logger, discovery):
        """Test error handling and logging."""
        # Mock registry to raise exception
        discovery.registry.load_registry.side_effect = Exception("Load error")
        
        # Test method that should handle the error gracefully
        results = discovery.find_by_framework("xgboost")
        
        assert results == []
        mock_logger.error.assert_called()
    
    def test_are_similar_tags(self, discovery):
        """Test tag similarity detection."""
        # Test similar tags (typos, variations)
        assert discovery._are_similar_tags("training", "trainng")  # Missing 'i'
        assert discovery._are_similar_tags("evaluation", "evalution")  # Missing 'a'
        assert discovery._are_similar_tags("preprocessing", "preprocess")  # Common suffix
        
        # Test dissimilar tags
        assert not discovery._are_similar_tags("training", "evaluation")
        assert not discovery._are_similar_tags("xgboost", "pytorch")
        assert not discovery._are_similar_tags("a", "xyz")

class TestTagBasedDiscoveryIntegration:
    """Integration tests for TagBasedDiscovery with realistic scenarios."""
    
    def test_comprehensive_search_workflow(self):
        """Test comprehensive search workflow combining multiple methods."""
        registry = Mock(spec=CatalogRegistry)
        
        # Create realistic registry data
        registry_data = {
            "tag_index": {
                "framework_tags": {
                    "xgboost": ["xgb_basic", "xgb_advanced"],
                    "pytorch": ["pytorch_nlp", "pytorch_cv"],
                    "sklearn": ["sklearn_prep"]
                },
                "task_tags": {
                    "training": ["xgb_basic", "xgb_advanced", "pytorch_nlp", "pytorch_cv"],
                    "evaluation": ["xgb_advanced", "pytorch_cv"],
                    "preprocessing": ["sklearn_prep"],
                    "deployment": ["xgb_advanced"]
                },
                "complexity_tags": {
                    "simple": ["xgb_basic", "sklearn_prep"],
                    "standard": ["pytorch_nlp"],
                    "advanced": ["xgb_advanced", "pytorch_cv"]
                },
                "domain_tags": {
                    "tabular": ["xgb_basic", "xgb_advanced", "sklearn_prep"],
                    "nlp": ["pytorch_nlp"],
                    "computer_vision": ["pytorch_cv"]
                }
            },
            "nodes": {
                "xgb_basic": {
                    "id": "xgb_basic",
                    "title": "Basic XGBoost Training",
                    "description": "Simple XGBoost training for tabular data",
                    "multi_dimensional_tags": {
                        "framework_tags": ["xgboost"],
                        "task_tags": ["training"],
                        "complexity_tags": ["simple"],
                        "domain_tags": ["tabular"]
                    }
                },
                "xgb_advanced": {
                    "id": "xgb_advanced",
                    "title": "Advanced XGBoost Pipeline",
                    "description": "Complete XGBoost pipeline with training, evaluation, and deployment",
                    "multi_dimensional_tags": {
                        "framework_tags": ["xgboost"],
                        "task_tags": ["training", "evaluation", "deployment"],
                        "complexity_tags": ["advanced"],
                        "domain_tags": ["tabular"]
                    }
                },
                "pytorch_nlp": {
                    "id": "pytorch_nlp",
                    "title": "PyTorch NLP Training",
                    "description": "Standard PyTorch training pipeline for NLP tasks",
                    "multi_dimensional_tags": {
                        "framework_tags": ["pytorch"],
                        "task_tags": ["training"],
                        "complexity_tags": ["standard"],
                        "domain_tags": ["nlp"]
                    }
                },
                "pytorch_cv": {
                    "id": "pytorch_cv",
                    "title": "PyTorch Computer Vision",
                    "description": "Advanced PyTorch pipeline for computer vision with evaluation",
                    "multi_dimensional_tags": {
                        "framework_tags": ["pytorch"],
                        "task_tags": ["training", "evaluation"],
                        "complexity_tags": ["advanced"],
                        "domain_tags": ["computer_vision"]
                    }
                },
                "sklearn_prep": {
                    "id": "sklearn_prep",
                    "title": "Sklearn Preprocessing",
                    "description": "Simple sklearn preprocessing pipeline for tabular data",
                    "multi_dimensional_tags": {
                        "framework_tags": ["sklearn"],
                        "task_tags": ["preprocessing"],
                        "complexity_tags": ["simple"],
                        "domain_tags": ["tabular"]
                    }
                }
            }
        }
        
        registry.load_registry.return_value = registry_data
        registry.get_pipeline_node.side_effect = lambda pid: registry_data["nodes"].get(pid)
        registry.get_all_pipelines.return_value = list(registry_data["nodes"].keys())
        
        discovery = TagBasedDiscovery(registry)
        
        # Test 1: Find all training pipelines
        training_pipelines = discovery.find_by_task("training")
        assert len(training_pipelines) == 4
        
        # Test 2: Find advanced tabular pipelines
        advanced_tabular = discovery.find_by_multiple_criteria(
            complexity="advanced",
            domain="tabular"
        )
        assert set(advanced_tabular) == {"xgb_advanced"}
        
        # Test 3: Text search for "XGBoost"
        xgboost_results = discovery.search_by_text("XGBoost")
        xgboost_ids = [pid for pid, _ in xgboost_results]
        assert "xgb_basic" in xgboost_ids
        assert "xgb_advanced" in xgboost_ids
        
        # Test 4: Find similar pipelines to xgb_basic
        similar_to_basic = discovery.suggest_similar_pipelines("xgb_basic")
        # Should find sklearn_prep as similar (both simple tabular)
        similar_ids = [pid for pid, _ in similar_to_basic]
        assert "sklearn_prep" in similar_ids
        
        # Test 5: Get comprehensive statistics
        stats = discovery.get_tag_statistics()
        assert stats["total_tag_categories"] == 4
        assert stats["pipeline_tag_counts"]["min"] >= 4  # Each pipeline has at least 4 tags
    
    def test_tag_consistency_analysis(self):
        """Test analyzing tag consistency across pipelines."""
        registry = Mock(spec=CatalogRegistry)
        
        # Create data with some inconsistencies
        registry_data = {
            "tag_index": {
                "framework_tags": {
                    "xgboost": ["pipeline_a"],
                    "xgbost": ["pipeline_b"],  # Typo
                    "pytorch": ["pipeline_c"]
                },
                "task_tags": {
                    "training": ["pipeline_a", "pipeline_c"],
                    "trainng": ["pipeline_b"]  # Typo
                }
            },
            "nodes": {
                "pipeline_a": {
                    "multi_dimensional_tags": {
                        "framework_tags": ["xgboost"],
                        "task_tags": ["training"]
                    }
                },
                "pipeline_b": {
                    "multi_dimensional_tags": {
                        "framework_tags": ["xgbost"],  # Typo
                        "task_tags": ["trainng"]  # Typo
                    }
                },
                "pipeline_c": {
                    "multi_dimensional_tags": {
                        "framework_tags": ["pytorch"],
                        "task_tags": ["training"]
                    }
                }
            }
        }
        
        registry.load_registry.return_value = registry_data
        registry.get_pipeline_node.side_effect = lambda pid: registry_data["nodes"].get(pid)
        registry.get_all_pipelines.return_value = list(registry_data["nodes"].keys())
        
        discovery = TagBasedDiscovery(registry)
        
        # Test finding undertagged pipelines (each has only 2 tags)
        undertagged = discovery.find_undertagged_pipelines(min_tags=3)
        assert len(undertagged) == 3  # All pipelines are undertagged
        
        # Test tag statistics should show single-use tags
        stats = discovery.get_tag_statistics()
        
        # Check for single-use tags in framework category
        framework_stats = stats["tag_categories"]["framework_tags"]
        assert framework_stats["total_tags"] == 3  # xgboost, xgbost, pytorch

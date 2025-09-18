"""
Unit tests for TagBasedDiscovery class.

Tests tag-based pipeline discovery implementing Zettelkasten anti-categories principle.
Enables emergent organization through multi-dimensional tagging.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any, List
from collections import Counter

from cursus.pipeline_catalog.core.tag_discovery import TagBasedDiscovery
from cursus.pipeline_catalog.core.catalog_registry import CatalogRegistry


class TestTagBasedDiscovery:
    """Test suite for TagBasedDiscovery class."""

    @pytest.fixture
    def mock_registry(self):
        """Create mock CatalogRegistry."""
        return Mock(spec=CatalogRegistry)

    @pytest.fixture
    def sample_tag_index(self):
        """Sample tag index data for testing."""
        return {
            "framework_tags": {
                "xgboost": ["pipeline1", "pipeline2"],
                "pytorch": ["pipeline3", "pipeline4"],
                "sklearn": ["pipeline1", "pipeline5"]
            },
            "task_tags": {
                "training": ["pipeline1", "pipeline3"],
                "evaluation": ["pipeline2", "pipeline4"],
                "preprocessing": ["pipeline5"]
            },
            "complexity_tags": {
                "simple": ["pipeline1", "pipeline5"],
                "standard": ["pipeline2", "pipeline3"],
                "advanced": ["pipeline4"]
            },
            "domain_tags": {
                "tabular": ["pipeline1", "pipeline2"],
                "nlp": ["pipeline3"],
                "computer_vision": ["pipeline4"]
            }
        }

    @pytest.fixture
    def sample_nodes(self):
        """Sample node data for testing."""
        return {
            "pipeline1": {
                "id": "pipeline1",
                "title": "XGBoost Training Pipeline",
                "description": "Simple XGBoost training for tabular data",
                "multi_dimensional_tags": {
                    "framework_tags": ["xgboost", "sklearn"],
                    "task_tags": ["training"],
                    "complexity_tags": ["simple"],
                    "domain_tags": ["tabular"]
                },
                "atomic_properties": {
                    "single_responsibility": "Train XGBoost models on tabular data"
                }
            },
            "pipeline2": {
                "id": "pipeline2",
                "title": "XGBoost Evaluation Pipeline",
                "description": "Standard XGBoost evaluation for tabular data",
                "multi_dimensional_tags": {
                    "framework_tags": ["xgboost"],
                    "task_tags": ["evaluation"],
                    "complexity_tags": ["standard"],
                    "domain_tags": ["tabular"]
                },
                "atomic_properties": {
                    "single_responsibility": "Evaluate XGBoost models"
                }
            },
            "pipeline3": {
                "id": "pipeline3",
                "title": "PyTorch NLP Training",
                "description": "Standard PyTorch training for NLP tasks",
                "multi_dimensional_tags": {
                    "framework_tags": ["pytorch"],
                    "task_tags": ["training"],
                    "complexity_tags": ["standard"],
                    "domain_tags": ["nlp"]
                },
                "atomic_properties": {
                    "single_responsibility": "Train PyTorch models for NLP"
                }
            }
        }

    @pytest.fixture
    def discovery(self, mock_registry):
        """Create TagBasedDiscovery instance."""
        return TagBasedDiscovery(mock_registry)

    def test_init(self, mock_registry):
        """Test TagBasedDiscovery initialization."""
        discovery = TagBasedDiscovery(mock_registry)
        
        assert discovery.registry == mock_registry
        assert discovery._tag_cache == {}
        assert not discovery._cache_valid

    def test_find_by_tags_any_mode(self, discovery, mock_registry, sample_tag_index):
        """Test finding pipelines by tags with 'any' match mode."""
        discovery._tag_cache = sample_tag_index
        discovery._cache_valid = True
        
        result = discovery.find_by_tags(["xgboost", "pytorch"], match_mode="any")
        
        # Should return pipelines that have either xgboost OR pytorch
        expected = {"pipeline1", "pipeline2", "pipeline3", "pipeline4"}
        assert set(result) == expected

    def test_find_by_tags_all_mode(self, discovery, mock_registry, sample_tag_index):
        """Test finding pipelines by tags with 'all' match mode."""
        discovery._tag_cache = sample_tag_index
        discovery._cache_valid = True
        
        result = discovery.find_by_tags(["xgboost", "training"], match_mode="all")
        
        # Should return pipelines that have both xgboost AND training
        expected = {"pipeline1"}  # Only pipeline1 has both tags
        assert set(result) == expected

    def test_find_by_tags_exact_mode(self, discovery, mock_registry, sample_tag_index, sample_nodes):
        """Test finding pipelines by tags with 'exact' match mode."""
        discovery._tag_cache = sample_tag_index
        discovery._cache_valid = True
        
        mock_registry.get_all_pipelines.return_value = ["pipeline1", "pipeline2", "pipeline3"]
        mock_registry.get_pipeline_node.side_effect = lambda pid: sample_nodes.get(pid)
        
        # Mock _get_pipeline_tags method
        def mock_get_pipeline_tags(pipeline_id):
            node = sample_nodes.get(pipeline_id, {})
            tags = []
            for tag_list in node.get("multi_dimensional_tags", {}).values():
                tags.extend(tag_list)
            return list(set(tags))
        
        discovery._get_pipeline_tags = mock_get_pipeline_tags
        
        result = discovery.find_by_tags(["xgboost", "training", "simple", "tabular"], match_mode="exact")
        
        # Should return pipelines that have exactly these tags
        # pipeline1 has: xgboost, sklearn, training, simple, tabular (extra sklearn)
        # So no exact matches in this case
        assert len(result) == 0

    def test_find_by_tags_empty_list(self, discovery):
        """Test finding pipelines with empty tag list."""
        result = discovery.find_by_tags([])
        assert result == []

    def test_find_by_tags_invalid_mode(self, discovery, sample_tag_index):
        """Test finding pipelines with invalid match mode."""
        discovery._tag_cache = sample_tag_index
        discovery._cache_valid = True
        
        result = discovery.find_by_tags(["xgboost"], match_mode="invalid")
        assert result == []

    def test_find_by_tags_error(self, discovery, mock_registry):
        """Test finding pipelines with error."""
        mock_registry.load_registry.side_effect = Exception("Registry error")
        
        result = discovery.find_by_tags(["xgboost"])
        assert result == []

    def test_find_by_framework(self, discovery, sample_tag_index):
        """Test finding pipelines by framework."""
        discovery._tag_cache = sample_tag_index
        discovery._cache_valid = True
        
        result = discovery.find_by_framework("xgboost")
        
        assert set(result) == {"pipeline1", "pipeline2"}

    def test_find_by_framework_not_found(self, discovery, sample_tag_index):
        """Test finding pipelines by non-existent framework."""
        discovery._tag_cache = sample_tag_index
        discovery._cache_valid = True
        
        result = discovery.find_by_framework("nonexistent")
        assert result == []

    def test_find_by_complexity(self, discovery, sample_tag_index):
        """Test finding pipelines by complexity."""
        discovery._tag_cache = sample_tag_index
        discovery._cache_valid = True
        
        result = discovery.find_by_complexity("simple")
        
        assert set(result) == {"pipeline1", "pipeline5"}

    def test_find_by_task(self, discovery, sample_tag_index):
        """Test finding pipelines by task."""
        discovery._tag_cache = sample_tag_index
        discovery._cache_valid = True
        
        result = discovery.find_by_task("training")
        
        assert set(result) == {"pipeline1", "pipeline3"}

    def test_find_by_domain(self, discovery, sample_tag_index):
        """Test finding pipelines by domain."""
        discovery._tag_cache = sample_tag_index
        discovery._cache_valid = True
        
        result = discovery.find_by_domain("tabular")
        
        assert set(result) == {"pipeline1", "pipeline2"}

    def test_find_by_pattern(self, discovery, sample_tag_index):
        """Test finding pipelines by pattern."""
        # Add pattern tags to sample data
        sample_tag_index["pattern_tags"] = {
            "atomic_workflow": ["pipeline1", "pipeline3"],
            "end_to_end": ["pipeline2"]
        }
        discovery._tag_cache = sample_tag_index
        discovery._cache_valid = True
        
        result = discovery.find_by_pattern("atomic_workflow")
        
        assert set(result) == {"pipeline1", "pipeline3"}

    def test_find_by_multiple_criteria(self, discovery, sample_tag_index):
        """Test finding pipelines by multiple criteria."""
        discovery._tag_cache = sample_tag_index
        discovery._cache_valid = True
        
        # Mock find_by_tags to return intersection
        def mock_find_by_tags(tags, match_mode):
            if match_mode == "all" and set(tags) == {"xgboost", "simple"}:
                return ["pipeline1"]
            return []
        
        discovery.find_by_tags = mock_find_by_tags
        
        result = discovery.find_by_multiple_criteria(
            framework="xgboost",
            complexity="simple"
        )
        
        assert result == ["pipeline1"]

    def test_find_by_multiple_criteria_no_criteria(self, discovery, mock_registry):
        """Test finding pipelines with no criteria."""
        mock_registry.get_all_pipelines.return_value = ["pipeline1", "pipeline2"]
        
        result = discovery.find_by_multiple_criteria()
        
        assert result == ["pipeline1", "pipeline2"]

    def test_search_by_text(self, discovery, mock_registry, sample_nodes):
        """Test searching pipelines by text query."""
        mock_registry.get_all_pipelines.return_value = ["pipeline1", "pipeline2", "pipeline3"]
        mock_registry.get_pipeline_node.side_effect = lambda pid: sample_nodes.get(pid)
        
        # Mock _get_pipeline_tags
        def mock_get_pipeline_tags(pipeline_id):
            node = sample_nodes.get(pipeline_id, {})
            tags = []
            for tag_list in node.get("multi_dimensional_tags", {}).values():
                tags.extend(tag_list)
            return list(set(tags))
        
        discovery._get_pipeline_tags = mock_get_pipeline_tags
        
        result = discovery.search_by_text("xgboost training")
        
        # Should return results sorted by relevance
        assert len(result) > 0
        assert all(isinstance(item, tuple) and len(item) == 2 for item in result)
        assert all(isinstance(item[1], float) for item in result)
        
        # Results should be sorted by score (descending)
        scores = [item[1] for item in result]
        assert scores == sorted(scores, reverse=True)

    def test_search_by_text_empty_query(self, discovery):
        """Test searching with empty query."""
        result = discovery.search_by_text("")
        assert result == []

    def test_search_by_text_specific_fields(self, discovery, mock_registry, sample_nodes):
        """Test searching in specific fields."""
        mock_registry.get_all_pipelines.return_value = ["pipeline1"]
        mock_registry.get_pipeline_node.side_effect = lambda pid: sample_nodes.get(pid)
        discovery._get_pipeline_tags = lambda pid: ["xgboost", "training"]
        
        result = discovery.search_by_text("xgboost", search_fields=["title"])
        
        assert len(result) > 0
        assert result[0][0] == "pipeline1"

    def test_get_tag_clusters(self, discovery, mock_registry, sample_nodes):
        """Test getting emergent tag clusters."""
        mock_registry.get_all_pipelines.return_value = ["pipeline1", "pipeline2", "pipeline3"]
        
        # Mock _get_pipeline_tags to return different tag sets
        def mock_get_pipeline_tags(pipeline_id):
            tag_sets = {
                "pipeline1": ["xgboost", "training", "tabular"],
                "pipeline2": ["xgboost", "evaluation", "tabular"],  # Similar to pipeline1
                "pipeline3": ["pytorch", "training", "nlp"]  # Different
            }
            return tag_sets.get(pipeline_id, [])
        
        discovery._get_pipeline_tags = mock_get_pipeline_tags
        
        result = discovery.get_tag_clusters()
        
        assert isinstance(result, dict)
        # Should have clusters based on tag similarity
        assert len(result) > 0

    def test_suggest_similar_pipelines(self, discovery, mock_registry):
        """Test suggesting similar pipelines."""
        mock_registry.get_all_pipelines.return_value = ["pipeline1", "pipeline2", "pipeline3"]
        
        # Mock _get_pipeline_tags
        def mock_get_pipeline_tags(pipeline_id):
            tag_sets = {
                "pipeline1": ["xgboost", "training", "tabular"],
                "pipeline2": ["xgboost", "evaluation", "tabular"],  # 2/4 overlap = 0.5
                "pipeline3": ["pytorch", "training", "nlp"]  # 1/5 overlap = 0.2
            }
            return tag_sets.get(pipeline_id, [])
        
        discovery._get_pipeline_tags = mock_get_pipeline_tags
        
        result = discovery.suggest_similar_pipelines("pipeline1", similarity_threshold=0.3)
        
        assert len(result) == 1  # Only pipeline2 meets threshold
        assert result[0][0] == "pipeline2"
        assert result[0][1] == 0.5  # 2 intersection / 4 union

    def test_suggest_similar_pipelines_no_tags(self, discovery):
        """Test suggesting similar pipelines when source has no tags."""
        discovery._get_pipeline_tags = lambda pid: []
        
        result = discovery.suggest_similar_pipelines("pipeline1")
        assert result == []

    def test_get_tag_statistics(self, discovery, mock_registry, sample_tag_index):
        """Test getting tag statistics."""
        discovery._tag_cache = sample_tag_index
        discovery._cache_valid = True
        
        mock_registry.get_all_pipelines.return_value = ["pipeline1", "pipeline2", "pipeline3"]
        
        # Mock _get_pipeline_tags
        def mock_get_pipeline_tags(pipeline_id):
            tag_counts = {
                "pipeline1": ["tag1", "tag2", "tag3"],  # 3 tags
                "pipeline2": ["tag1", "tag2"],  # 2 tags
                "pipeline3": ["tag1"]  # 1 tag
            }
            return tag_counts.get(pipeline_id, [])
        
        discovery._get_pipeline_tags = mock_get_pipeline_tags
        
        result = discovery.get_tag_statistics()
        
        assert "total_tag_categories" in result
        assert "tag_categories" in result
        assert "most_common_tags" in result
        assert "tag_distribution" in result
        assert "pipeline_tag_counts" in result
        
        assert result["total_tag_categories"] == 4  # framework, task, complexity, domain
        assert "framework_tags" in result["tag_categories"]

    def test_get_tag_statistics_error(self, discovery, mock_registry):
        """Test getting tag statistics with error."""
        mock_registry.load_registry.side_effect = Exception("Registry error")
        
        result = discovery.get_tag_statistics()
        assert "error" in result

    def test_find_undertagged_pipelines(self, discovery, mock_registry):
        """Test finding undertagged pipelines."""
        mock_registry.get_all_pipelines.return_value = ["pipeline1", "pipeline2", "pipeline3"]
        
        # Mock _get_pipeline_tags
        def mock_get_pipeline_tags(pipeline_id):
            tag_counts = {
                "pipeline1": ["tag1", "tag2", "tag3", "tag4"],  # 4 tags (sufficient)
                "pipeline2": ["tag1", "tag2"],  # 2 tags (undertagged)
                "pipeline3": ["tag1"]  # 1 tag (undertagged)
            }
            return tag_counts.get(pipeline_id, [])
        
        discovery._get_pipeline_tags = mock_get_pipeline_tags
        
        result = discovery.find_undertagged_pipelines(min_tags=3)
        
        assert len(result) == 2
        assert ("pipeline3", 1) in result
        assert ("pipeline2", 2) in result
        
        # Should be sorted by tag count (ascending)
        assert result[0][1] <= result[1][1]

    def test_suggest_tags_for_pipeline(self, discovery, mock_registry, sample_nodes):
        """Test suggesting tags for a pipeline."""
        # Mock suggest_similar_pipelines
        discovery.suggest_similar_pipelines = Mock(return_value=[
            ("pipeline2", 0.8),
            ("pipeline3", 0.6)
        ])
        
        mock_registry.get_pipeline_node.side_effect = lambda pid: sample_nodes.get(pid)
        discovery._get_pipeline_tags = lambda pid: ["xgboost", "training"] if pid == "pipeline1" else []
        
        result = discovery.suggest_tags_for_pipeline("pipeline1")
        
        assert isinstance(result, dict)
        # Should suggest tags from similar pipelines that aren't already present

    def test_suggest_tags_for_pipeline_no_similar(self, discovery):
        """Test suggesting tags when no similar pipelines found."""
        discovery.suggest_similar_pipelines = Mock(return_value=[])
        
        result = discovery.suggest_tags_for_pipeline("pipeline1")
        assert result == {}

    def test_get_tag_index_cache_valid(self, discovery):
        """Test getting tag index when cache is valid."""
        sample_index = {"framework_tags": {"xgboost": ["pipeline1"]}}
        discovery._tag_cache = sample_index
        discovery._cache_valid = True
        
        result = discovery._get_tag_index()
        assert result == sample_index

    def test_get_tag_index_cache_invalid(self, discovery, mock_registry):
        """Test getting tag index when cache is invalid."""
        sample_registry = {"tag_index": {"framework_tags": {"xgboost": ["pipeline1"]}}}
        mock_registry.load_registry.return_value = sample_registry
        
        result = discovery._get_tag_index()
        
        assert result == sample_registry["tag_index"]
        assert discovery._cache_valid

    def test_get_pipeline_tags(self, discovery, mock_registry, sample_nodes):
        """Test getting all tags for a pipeline."""
        mock_registry.get_pipeline_node.return_value = sample_nodes["pipeline1"]
        
        result = discovery._get_pipeline_tags("pipeline1")
        
        expected_tags = {"xgboost", "sklearn", "training", "simple", "tabular"}
        assert set(result) == expected_tags

    def test_get_pipeline_tags_no_node(self, discovery, mock_registry):
        """Test getting tags for non-existent pipeline."""
        mock_registry.get_pipeline_node.return_value = None
        
        result = discovery._get_pipeline_tags("nonexistent")
        assert result == []

    def test_get_pipeline_tags_no_tags(self, discovery, mock_registry):
        """Test getting tags for pipeline with no tags."""
        mock_registry.get_pipeline_node.return_value = {"id": "pipeline1"}
        
        result = discovery._get_pipeline_tags("pipeline1")
        assert result == []

    def test_clear_cache(self, discovery):
        """Test clearing tag cache."""
        discovery._tag_cache = {"test": "data"}
        discovery._cache_valid = True
        
        discovery.clear_cache()
        
        assert discovery._tag_cache == {}
        assert not discovery._cache_valid

    def test_error_handling_in_methods(self, discovery, mock_registry):
        """Test error handling in various methods."""
        mock_registry.load_registry.side_effect = Exception("Registry error")
        
        # Test various methods handle errors gracefully
        assert discovery.find_by_framework("xgboost") == []
        assert discovery.find_by_complexity("simple") == []
        assert discovery.find_by_task("training") == []
        assert discovery.find_by_domain("tabular") == []
        assert discovery.find_by_pattern("atomic") == []
        assert discovery.find_by_multiple_criteria(framework="xgboost") == []
        assert discovery.search_by_text("query") == []
        assert discovery.get_tag_clusters() == {}
        assert discovery.suggest_similar_pipelines("pipeline1") == []
        assert discovery.find_undertagged_pipelines() == []
        assert discovery.suggest_tags_for_pipeline("pipeline1") == {}

    def test_complex_tag_scenarios(self, discovery, mock_registry):
        """Test complex tag matching scenarios."""
        complex_tag_index = {
            "framework_tags": {
                "xgboost": ["p1", "p2"],
                "pytorch": ["p2", "p3"],  # p2 has both xgboost and pytorch
                "sklearn": ["p1", "p4"]
            },
            "task_tags": {
                "training": ["p1", "p3"],
                "evaluation": ["p2", "p4"]
            }
        }
        
        discovery._tag_cache = complex_tag_index
        discovery._cache_valid = True
        
        # Test finding pipelines with multiple frameworks
        result_any = discovery.find_by_tags(["xgboost", "pytorch"], match_mode="any")
        assert set(result_any) == {"p1", "p2", "p3"}
        
        # Test finding pipelines with both frameworks (only p2)
        result_all = discovery.find_by_tags(["xgboost", "pytorch"], match_mode="all")
        assert set(result_all) == {"p2"}

    def test_jaccard_similarity_calculation(self, discovery, mock_registry):
        """Test Jaccard similarity calculation in suggest_similar_pipelines."""
        mock_registry.get_all_pipelines.return_value = ["p1", "p2", "p3"]
        
        def mock_get_pipeline_tags(pipeline_id):
            tag_sets = {
                "p1": ["a", "b", "c"],  # Base pipeline
                "p2": ["a", "b"],       # 2/3 = 0.67 similarity
                "p3": ["a", "d", "e"]   # 1/5 = 0.2 similarity
            }
            return tag_sets.get(pipeline_id, [])
        
        discovery._get_pipeline_tags = mock_get_pipeline_tags
        
        result = discovery.suggest_similar_pipelines("p1", similarity_threshold=0.5)
        
        assert len(result) == 1
        assert result[0][0] == "p2"
        assert abs(result[0][1] - 0.6666666666666666) < 0.001  # 2/3 similarity

    def test_text_search_scoring(self, discovery, mock_registry):
        """Test text search relevance scoring."""
        sample_nodes = {
            "p1": {
                "title": "XGBoost Training Pipeline",  # Title match = 2.0 points
                "description": "Train models",
                "multi_dimensional_tags": {"framework_tags": ["xgboost"]}
            },
            "p2": {
                "title": "Model Pipeline",
                "description": "XGBoost evaluation pipeline",  # Description match = 1.0 point
                "multi_dimensional_tags": {"framework_tags": ["pytorch"]}
            }
        }
        
        mock_registry.get_all_pipelines.return_value = ["p1", "p2"]
        mock_registry.get_pipeline_node.side_effect = lambda pid: sample_nodes.get(pid)
        discovery._get_pipeline_tags = lambda pid: sample_nodes[pid]["multi_dimensional_tags"]["framework_tags"]
        
        result = discovery.search_by_text("xgboost")
        
        # p1 should score higher due to title match
        assert len(result) == 2
        assert result[0][0] == "p1"  # Higher score
        assert result[0][1] > result[1][1]  # p1 score > p2 score

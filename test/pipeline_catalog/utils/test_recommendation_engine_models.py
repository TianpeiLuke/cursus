"""
Unit tests for PipelineRecommendationEngine model classes.

Tests the recommendation result models and data structures used by the
recommendation engine.
"""

import pytest
from src.cursus.pipeline_catalog.utils.recommendation_engine import (
    RecommendationResult, CompositionRecommendation
)


class TestRecommendationResult:
    """Test suite for RecommendationResult model."""
    
    def test_recommendation_result_creation(self):
        """Test creating RecommendationResult instance."""
        result = RecommendationResult(
            pipeline_id="test_pipeline",
            title="Test Pipeline",
            score=2.5,
            reasoning="High relevance match",
            connection_path=["source", "test_pipeline"],
            tag_overlap=0.8,
            framework="xgboost",
            complexity="simple"
        )
        
        assert result.pipeline_id == "test_pipeline"
        assert result.title == "Test Pipeline"
        assert result.score == 2.5
        assert result.reasoning == "High relevance match"
        assert result.connection_path == ["source", "test_pipeline"]
        assert result.tag_overlap == 0.8
        assert result.framework == "xgboost"
        assert result.complexity == "simple"
    
    def test_recommendation_result_minimal(self):
        """Test creating RecommendationResult with minimal fields."""
        result = RecommendationResult(
            pipeline_id="minimal_pipeline",
            title="Minimal Pipeline",
            score=1.0,
            reasoning="Basic match"
        )
        
        assert result.pipeline_id == "minimal_pipeline"
        assert result.title == "Minimal Pipeline"
        assert result.score == 1.0
        assert result.reasoning == "Basic match"
        assert result.connection_path is None
        assert result.tag_overlap is None
        assert result.framework is None
        assert result.complexity is None


class TestCompositionRecommendation:
    """Test suite for CompositionRecommendation model."""
    
    def test_composition_recommendation_creation(self):
        """Test creating CompositionRecommendation instance."""
        composition = CompositionRecommendation(
            pipeline_sequence=["pipeline_a", "pipeline_b", "pipeline_c"],
            composition_type="sequential",
            description="Sequential ML workflow",
            estimated_complexity="standard",
            total_score=4.5
        )
        
        assert composition.pipeline_sequence == ["pipeline_a", "pipeline_b", "pipeline_c"]
        assert composition.composition_type == "sequential"
        assert composition.description == "Sequential ML workflow"
        assert composition.estimated_complexity == "standard"
        assert composition.total_score == 4.5

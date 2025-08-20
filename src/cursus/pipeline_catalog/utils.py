"""
Utility functions for the pipeline catalog.

This module provides functions for loading and working with the pipeline catalog index.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

# Type alias for pipeline entries
PipelineEntry = Dict[str, Any]


def get_catalog_root() -> Path:
    """
    Get the path to the root of the pipeline catalog.
    
    Returns:
        Path: The path to the root of the pipeline catalog
    """
    return Path(os.path.dirname(os.path.abspath(__file__)))


def load_index() -> Dict[str, List[PipelineEntry]]:
    """
    Load the pipeline catalog index.
    
    Returns:
        Dict[str, List[PipelineEntry]]: The pipeline catalog index
    """
    index_path = get_catalog_root() / "index.json"
    try:
        with open(index_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to load pipeline catalog index: {e}")


def get_pipeline_by_id(pipeline_id: str) -> Optional[PipelineEntry]:
    """
    Get a pipeline entry by its ID.
    
    Args:
        pipeline_id: The ID of the pipeline to retrieve
        
    Returns:
        Optional[PipelineEntry]: The pipeline entry, or None if not found
    """
    index = load_index()
    for pipeline in index.get("pipelines", []):
        if pipeline.get("id") == pipeline_id:
            return pipeline
    return None


def filter_pipelines(
    framework: Optional[str] = None, 
    complexity: Optional[str] = None,
    features: Optional[List[str]] = None,
    tags: Optional[List[str]] = None
) -> List[PipelineEntry]:
    """
    Filter pipelines by criteria.
    
    Args:
        framework: Filter by framework (e.g., "xgboost", "pytorch")
        complexity: Filter by complexity (e.g., "simple", "standard", "advanced")
        features: Filter by features (e.g., ["training", "evaluation"])
        tags: Filter by tags
        
    Returns:
        List[PipelineEntry]: List of pipeline entries matching the criteria
    """
    index = load_index()
    pipelines = index.get("pipelines", [])
    
    results = pipelines
    
    # Apply filters
    if framework:
        results = [p for p in results if p.get("framework") == framework]
    
    if complexity:
        results = [p for p in results if p.get("complexity") == complexity]
    
    if features:
        # Pipeline must have ALL specified features
        results = [
            p for p in results 
            if all(feature in p.get("features", []) for feature in features)
        ]
    
    if tags:
        # Pipeline must have ANY of the specified tags
        results = [
            p for p in results 
            if any(tag in p.get("tags", []) for tag in tags)
        ]
    
    return results


def get_pipeline_path(pipeline_id: str) -> Optional[Path]:
    """
    Get the full path to a pipeline module.
    
    Args:
        pipeline_id: The ID of the pipeline
        
    Returns:
        Optional[Path]: The path to the pipeline module, or None if not found
    """
    pipeline = get_pipeline_by_id(pipeline_id)
    if not pipeline:
        return None
    
    return get_catalog_root() / pipeline.get("path", "")


def get_all_frameworks() -> List[str]:
    """
    Get a list of all available frameworks in the catalog.
    
    Returns:
        List[str]: List of framework names
    """
    index = load_index()
    frameworks = set()
    
    for pipeline in index.get("pipelines", []):
        if "framework" in pipeline:
            frameworks.add(pipeline["framework"])
    
    return sorted(list(frameworks))


def get_all_features() -> List[str]:
    """
    Get a list of all available features across pipelines in the catalog.
    
    Returns:
        List[str]: List of feature names
    """
    index = load_index()
    features = set()
    
    for pipeline in index.get("pipelines", []):
        if "features" in pipeline:
            features.update(pipeline["features"])
    
    return sorted(list(features))

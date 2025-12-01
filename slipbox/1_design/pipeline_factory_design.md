---
tags:
  - design
  - implementation
  - pipeline_catalog
  - factory_pattern
  - refactoring
keywords:
  - pipeline factory
  - catalog registry
  - shared dags
  - metadata-driven
  - dynamic creation
  - code elimination
topics:
  - pipeline management
  - factory pattern
  - code refactoring
  - metadata infrastructure
language: python
date of note: 2025-11-30
---

# Pipeline Factory Design: Eliminating Pipeline Class Redundancy

## Overview

This document outlines the design for a Pipeline Factory that eliminates redundant pipeline implementation classes by leveraging the existing metadata infrastructure in `pipeline_catalog`. The factory pattern replaces ~500 lines of boilerplate code across 7 pipeline files with a single, metadata-driven creation system.

## Problem Statement

### Current Redundancy

The `pipeline_catalog/pipelines` directory contains multiple pipeline classes that follow an identical pattern:

```python
class XGBoostE2EComprehensivePipeline(BasePipeline):
    def create_dag(self) -> PipelineDAG:
        return create_xgboost_complete_e2e_dag()  # Just calls shared DAG function
    
    def get_enhanced_dag_metadata(self) -> EnhancedDAGMetadata:
        return EnhancedDAGMetadata(...)  # 50+ lines of hardcoded metadata
```

**Key Issues:**

1. **Boilerplate Duplication**: Each pipeline class is a thin wrapper around:
   - A shared DAG creation function (already exists in `shared_dags/`)
   - Enhanced metadata (already exists in `catalog_index.json`)

2. **Redundant Metadata**: Pipeline metadata is duplicated in three places:
   - `catalog_index.json` (registry)
   - Pipeline class `get_enhanced_dag_metadata()` method
   - Shared DAG `get_dag_metadata()` function

3. **Maintenance Overhead**: Adding a new pipeline requires:
   - Creating a new pipeline class file (~70 lines)
   - Duplicating metadata across multiple locations
   - Maintaining consistency between registry and code

4. **No Dynamic Discovery**: Cannot programmatically list or create pipelines without importing individual classes

## Existing Infrastructure Analysis

The solution already exists within the codebase - we just need to connect the pieces:

### 1. CatalogRegistry (`core/catalog_registry.py`)

**Purpose**: Central registry for all pipeline metadata

**Key Methods:**
```python
registry.get_all_pipelines() -> List[str]
registry.get_pipeline_node(pipeline_id) -> Dict[str, Any]
registry.get_pipelines_by_framework(framework) -> List[str]
registry.get_pipelines_by_complexity(complexity) -> List[str]
```

**Contains:**
- Complete pipeline metadata
- Enhanced Zettelkasten metadata
- DAG structure information
- Source file references
- Connections between pipelines

### 2. Shared DAGs (`shared_dags/`)

**Purpose**: Reusable DAG creation functions

**Structure:**
```python
# In shared_dags/xgboost/complete_e2e_dag.py
def create_xgboost_complete_e2e_dag() -> PipelineDAG:
    """Creates the DAG structure"""
    
def get_dag_metadata() -> DAGMetadata:
    """Returns DAG metadata"""
```

**Discovery:**
```python
from shared_dags import get_all_shared_dags
all_dags = get_all_shared_dags()  # Returns registry of all DAG functions
```

### 3. TagBasedDiscovery (`core/tag_discovery.py`)

**Purpose**: Intelligent pipeline search and filtering

**Key Features:**
- Multi-dimensional tag search
- Framework/complexity/task filtering
- Similarity-based recommendations
- Text-based search with relevance scoring

### 4. EnhancedDAGMetadata (`shared_dags/enhanced_metadata.py`)

**Purpose**: Rich metadata with Zettelkasten principles

**Structure:**
- Inherits from `DAGMetadata`
- Contains `ZettelkastenMetadata`
- Converts to registry node format
- Validates metadata integrity

## Factory Design

### Core Concept

**Replace individual pipeline classes with a factory that:**
1. Looks up pipeline metadata from `CatalogRegistry`
2. Dynamically imports DAG creation function from `shared_dags`
3. Creates anonymous `BasePipeline` subclass at runtime
4. Provides discovery and filtering capabilities

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    PipelineFactory                          │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Registry   │  │  Discovery   │  │  Metadata    │    │
│  │   Lookup     │  │  Search      │  │  Creation    │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                  │                  │            │
│         └──────────────────┴──────────────────┘            │
│                          │                                 │
└──────────────────────────┼─────────────────────────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │  Dynamic BasePipeline   │
              │       Subclass          │
              │                         │
              │  create_dag() ────────► │ Calls shared_dags function
              │  get_metadata() ──────► │ Returns registry metadata
              └─────────────────────────┘
```

### Key Components

#### 1. PipelineFactory Class

**Responsibilities:**
- Create pipelines by ID or discovery criteria
- Manage DAG function imports
- Convert registry nodes to metadata objects
- Provide listing and search capabilities

**Interface:**
```python
class PipelineFactory:
    def __init__(self, registry_path: str = "catalog_index.json")
    
    @classmethod
    def create_pipeline(cls, pipeline_id: str, config_path: str, **kwargs) -> BasePipeline
    
    @classmethod
    def create_by_criteria(cls, framework: str, complexity: str, **kwargs) -> BasePipeline
    
    @classmethod
    def list_available_pipelines(cls) -> List[Dict[str, Any]]
    
    @classmethod
    def discover_pipelines(cls, **criteria) -> List[str]
```

#### 2. DAG Function Mapper

**Purpose**: Map pipeline IDs to shared DAG module paths

**Implementation Options:**

**Option A: Static Mapping** (Simple, Explicit)
```python
DAG_MODULE_MAP = {
    "xgb_e2e_comprehensive": "cursus.pipeline_catalog.shared_dags.xgboost.complete_e2e_dag",
    "pytorch_e2e_standard": "cursus.pipeline_catalog.shared_dags.pytorch.standard_e2e_dag",
    # ... etc
}
```

**Option B: Convention-Based** (Dynamic, Requires Naming Convention)
```python
def _resolve_dag_module(pipeline_id: str) -> str:
    # Extract framework and dag type from pipeline_id
    # e.g., "xgb_e2e_comprehensive" -> "xgboost.complete_e2e_dag"
    parts = pipeline_id.split('_')
    framework_map = {'xgb': 'xgboost', 'pytorch': 'pytorch'}
    framework = framework_map.get(parts[0], parts[0])
    dag_type = '_'.join(parts[1:]) + '_dag'
    return f"cursus.pipeline_catalog.shared_dags.{framework}.{dag_type}"
```

**Recommendation**: Use **Option A** initially for reliability, migrate to Option B once naming conventions are standardized.

#### 3. Metadata Converter

**Purpose**: Convert registry nodes to `EnhancedDAGMetadata` objects

```python
def _create_metadata_from_node(self, node: Dict[str, Any]) -> EnhancedDAGMetadata:
    """Convert CatalogRegistry node to EnhancedDAGMetadata"""
    zm_data = node.get("zettelkasten_metadata", {})
    atomic_props = node.get("atomic_properties", {})
    
    # Create ZettelkastenMetadata
    zettelkasten_metadata = ZettelkastenMetadata(
        atomic_id=node["id"],
        title=node.get("title", ""),
        single_responsibility=node.get("description", ""),
        framework=zm_data.get("framework", "generic"),
        complexity=zm_data.get("complexity", "standard"),
        features=zm_data.get("features", []),
        node_count=atomic_props.get("node_count", 1),
        edge_count=atomic_props.get("edge_count", 0),
        # ... populate remaining fields from node
    )
    
    # Create EnhancedDAGMetadata
    return EnhancedDAGMetadata(
        dag_id=node["id"],
        description=node["description"],
        complexity=zm_data.get("complexity", "standard"),
        features=zm_data.get("features", []),
        framework=zm_data.get("framework", "generic"),
        node_count=atomic_props.get("node_count", 1),
        edge_count=atomic_props.get("edge_count", 0),
        zettelkasten_metadata=zettelkasten_metadata
    )
```

## Implementation

### Complete PipelineFactory Implementation

```python
"""
Pipeline Factory Implementation

Eliminates redundant pipeline classes by creating pipelines dynamically
from catalog registry metadata and shared DAG functions.
"""

import logging
from typing import Dict, Any, List, Optional, Type, Callable
from pathlib import Path

from ..core.catalog_registry import CatalogRegistry
from ..core.tag_discovery import TagBasedDiscovery
from ..core.connection_traverser import ConnectionTraverser
from ..core.base_pipeline import BasePipeline
from ..shared_dags.enhanced_metadata import EnhancedDAGMetadata, ZettelkastenMetadata
from ...api.dag.base_dag import PipelineDAG

logger = logging.getLogger(__name__)


class PipelineFactory:
    """
    Factory for creating pipelines dynamically from registry metadata.
    
    Eliminates the need for individual pipeline classes by:
    1. Looking up metadata from CatalogRegistry
    2. Dynamically importing DAG creation functions
    3. Creating anonymous BasePipeline subclasses at runtime
    
    Example:
        # Create by ID
        pipeline = PipelineFactory.create_pipeline(
            pipeline_id="xgb_e2e_comprehensive",
            config_path="config.json"
        )
        
        # Create by criteria
        pipeline = PipelineFactory.create_by_criteria(
            framework="xgboost",
            complexity="comprehensive",
            config_path="config.json"
        )
        
        # List available
        pipelines = PipelineFactory.list_available_pipelines()
    """
    
    # Static mapping of pipeline IDs to DAG module paths
    # This could be generated automatically once naming conventions are standardized
    DAG_MODULE_MAP = {
        "xgb_e2e_comprehensive": "cursus.pipeline_catalog.shared_dags.xgboost.complete_e2e_dag",
        "xgb_training_simple": "cursus.pipeline_catalog.shared_dags.xgboost.simple_dag",
        "xgb_training_calibrated": "cursus.pipeline_catalog.shared_dags.xgboost.training_with_calibration_dag",
        "xgb_training_evaluation": "cursus.pipeline_catalog.shared_dags.xgboost.training_with_evaluation_dag",
        "pytorch_e2e_standard": "cursus.pipeline_catalog.shared_dags.pytorch.standard_e2e_dag",
        "pytorch_training_basic": "cursus.pipeline_catalog.shared_dags.pytorch.training_dag",
        "dummy_e2e_basic": "cursus.pipeline_catalog.shared_dags.dummy.e2e_basic_dag",
    }
    
    def __init__(self, registry_path: str = "catalog_index.json"):
        """
        Initialize factory with registry.
        
        Args:
            registry_path: Path to catalog registry JSON file
        """
        self.registry = CatalogRegistry(registry_path)
        self.discovery = TagBasedDiscovery(self.registry)
        self._dag_func_cache: Dict[str, Callable] = {}
    
    @classmethod
    def create_pipeline(
        cls,
        pipeline_id: str,
        config_path: str,
        **base_pipeline_kwargs
    ) -> BasePipeline:
        """
        Create a pipeline from registry by ID.
        
        Args:
            pipeline_id: Pipeline identifier (e.g., "xgb_e2e_comprehensive")
            config_path: Path to configuration file
            **base_pipeline_kwargs: Additional arguments for BasePipeline constructor
            
        Returns:
            BasePipeline instance with DAG and metadata from registry
            
        Raises:
            ValueError: If pipeline_id not found in registry
            
        Example:
            pipeline = PipelineFactory.create_pipeline(
                pipeline_id="xgb_e2e_comprehensive",
                config_path="config.json",
                sagemaker_session=session,
                execution_role=role
            )
        """
        factory = cls()
        
        # Get pipeline node from registry
        node = factory.registry.get_pipeline_node(pipeline_id)
        if not node:
            available = factory.registry.get_all_pipelines()
            raise ValueError(
                f"Pipeline '{pipeline_id}' not found in registry. "
                f"Available pipelines: {', '.join(available)}"
            )
        
        # Import DAG creation function
        dag_func = factory._import_dag_function(pipeline_id)
        
        # Create metadata from registry node
        metadata = factory._create_metadata_from_node(node)
        
        # Create dynamic pipeline class
        class _DynamicPipeline(BasePipeline):
            """Dynamically generated pipeline class"""
            
            def create_dag(self) -> PipelineDAG:
                """Create DAG using shared DAG function"""
                return dag_func()
            
            def get_enhanced_dag_metadata(self) -> EnhancedDAGMetadata:
                """Return metadata from registry"""
                return metadata
        
        # Set class name for better debugging
        _DynamicPipeline.__name__ = f"Dynamic_{pipeline_id}"
        _DynamicPipeline.__qualname__ = f"PipelineFactory.Dynamic_{pipeline_id}"
        
        logger.info(f"Created dynamic pipeline class for '{pipeline_id}'")
        
        return _DynamicPipeline(config_path=config_path, **base_pipeline_kwargs)
    
    @classmethod
    def create_by_criteria(
        cls,
        framework: str,
        complexity: str,
        config_path: str,
        task: Optional[str] = None,
        **base_pipeline_kwargs
    ) -> BasePipeline:
        """
        Create a pipeline by discovery criteria.
        
        Uses TagBasedDiscovery to find matching pipeline, then creates it.
        
        Args:
            framework: Framework name (e.g., "xgboost", "pytorch")
            complexity: Complexity level (e.g., "simple", "comprehensive")
            config_path: Path to configuration file
            task: Optional task filter (e.g., "training", "end_to_end")
            **base_pipeline_kwargs: Additional arguments for BasePipeline
            
        Returns:
            BasePipeline instance for first matching pipeline
            
        Raises:
            ValueError: If no pipeline matches criteria
            
        Example:
            pipeline = PipelineFactory.create_by_criteria(
                framework="xgboost",
                complexity="comprehensive",
                task="end_to_end",
                config_path="config.json"
            )
        """
        factory = cls()
        
        # Build search criteria
        criteria = {
            "framework": framework,
            "complexity": complexity
        }
        if task:
            criteria["task"] = task
        
        # Find matching pipelines
        candidates = factory.discovery.find_by_multiple_criteria(**criteria)
        
        if not candidates:
            raise ValueError(
                f"No pipeline found for criteria: {criteria}. "
                f"Use PipelineFactory.discover_pipelines() to explore available options."
            )
        
        # Take first match
        pipeline_id = candidates[0]
        logger.info(
            f"Found {len(candidates)} pipeline(s) matching criteria. "
            f"Creating: {pipeline_id}"
        )
        
        return cls.create_pipeline(pipeline_id, config_path, **base_pipeline_kwargs)
    
    @classmethod
    def list_available_pipelines(
        cls,
        include_metadata: bool = False
    ) -> List[Dict[str, Any]]:
        """
        List all available pipelines from registry.
        
        Args:
            include_metadata: If True, include full metadata for each pipeline
            
        Returns:
            List of pipeline information dictionaries
            
        Example:
            pipelines = PipelineFactory.list_available_pipelines()
            for p in pipelines:
                print(f"{p['id']}: {p['title']} ({p['framework']}/{p['complexity']})")
        """
        factory = cls()
        all_ids = factory.registry.get_all_pipelines()
        
        pipelines = []
        for pid in all_ids:
            node = factory.registry.get_pipeline_node(pid)
            if node:
                zm = node.get("zettelkasten_metadata", {})
                pipeline_info = {
                    "id": pid,
                    "title": node.get("title"),
                    "framework": zm.get("framework"),
                    "complexity": zm.get("complexity"),
                    "description": node.get("description"),
                    "features": zm.get("features", [])
                }
                
                if include_metadata:
                    pipeline_info["full_metadata"] = node
                
                pipelines.append(pipeline_info)
        
        return pipelines
    
    @classmethod
    def discover_pipelines(
        cls,
        framework: Optional[str] = None,
        complexity: Optional[str] = None,
        task: Optional[str] = None,
        search_text: Optional[str] = None
    ) -> List[str]:
        """
        Discover pipelines matching criteria.
        
        Args:
            framework: Optional framework filter
            complexity: Optional complexity filter
            task: Optional task filter
            search_text: Optional text search across titles and descriptions
            
        Returns:
            List of matching pipeline IDs
            
        Example:
            # Find all XGBoost pipelines
            xgb_pipelines = PipelineFactory.discover_pipelines(framework="xgboost")
            
            # Find simple training pipelines
            simple_training = PipelineFactory.discover_pipelines(
                complexity="simple",
                task="training"
            )
            
            # Text search
            evaluation_pipelines = PipelineFactory.discover_pipelines(
                search_text="evaluation"
            )
        """
        factory = cls()
        
        # Text search if provided
        if search_text:
            results = factory.discovery.search_by_text(search_text)
            return [pid for pid, score in results]
        
        # Criteria-based search
        criteria = {}
        if framework:
            criteria["framework"] = framework
        if complexity:
            criteria["complexity"] = complexity
        if task:
            criteria["task"] = task
        
        if criteria:
            return factory.discovery.find_by_multiple_criteria(**criteria)
        
        # No criteria - return all pipelines
        return factory.registry.get_all_pipelines()
    
    @classmethod
    def get_pipeline_info(cls, pipeline_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific pipeline.
        
        Args:
            pipeline_id: Pipeline identifier
            
        Returns:
            Dictionary with complete pipeline information
            
        Raises:
            ValueError: If pipeline not found
        """
        factory = cls()
        node = factory.registry.get_pipeline_node(pipeline_id)
        
        if not node:
            raise ValueError(f"Pipeline '{pipeline_id}' not found in registry")
        
        return node
    
    def _import_dag_function(self, pipeline_id: str) -> Callable[[], PipelineDAG]:
        """
        Dynamically import DAG creation function.
        
        Args:
            pipeline_id: Pipeline identifier
            
        Returns:
            Callable that creates PipelineDAG
            
        Raises:
            ValueError: If DAG module not found or invalid
        """
        # Check cache first
        if pipeline_id in self._dag_func_cache:
            return self._dag_func_cache[pipeline_id]
        
        # Get module path from mapping
        module_path = self.DAG_MODULE_MAP.get(pipeline_id)
        if not module_path:
            raise ValueError(
                f"No DAG module mapping for '{pipeline_id}'. "
                f"Add to PipelineFactory.DAG_MODULE_MAP."
            )
        
        try:
            # Dynamic import
            module = __import__(module_path, fromlist=['create'])
            
            # Find the create_*_dag function
            dag_func = None
            for name in dir(module):
                if name.startswith('create_') and name.endswith('_dag'):
                    dag_func = getattr(module, name)
                    break
            
            if dag_func is None:
                raise ValueError(
                    f"No create_*_dag function found in {module_path}"
                )
            
            # Cache for future use
            self._dag_func_cache[pipeline_id] = dag_func
            logger.debug(f"Imported DAG function for '{pipeline_id}': {dag_func.__name__}")
            
            return dag_func
            
        except ImportError as e:
            raise ValueError(
                f"Failed to import DAG module '{module_path}': {e}"
            )
    
    def _create_metadata_from_node(
        self,
        node: Dict[str, Any]
    ) -> EnhancedDAGMetadata:
        """
        Convert CatalogRegistry node to EnhancedDAGMetadata.
        
        Args:
            node: Registry node dictionary
            
        Returns:
            EnhancedDAGMetadata instance
        """
        zm_data = node.get("zettelkasten_metadata", {})
        atomic_props = node.get("atomic_properties", {})
        multi_tags = node.get("multi_dimensional_tags", {})
        
        # Create ZettelkastenMetadata
        zettelkasten_metadata = ZettelkastenMetadata(
            atomic_id=node["id"],
            title=node.get("title", ""),
            single_responsibility=node.get("description", ""),
            framework=zm_data.get("framework", "generic"),
            complexity=zm_data.get("complexity", "standard"),
            use_case=zm_data.get("use_case", ""),
            features=zm_data.get("features", []),
            mods_compatible=zm_data.get("mods_compatible", False),
            node_count=atomic_props.get("node_count", 1),
            edge_count=atomic_props.get("edge_count", 0),
            source_file=node.get("source_file", ""),
            migration_source=node.get("migration_source", ""),
            created_date=node.get("created_date", ""),
            priority=node.get("priority", "standard"),
            framework_tags=multi_tags.get("framework_tags", []),
            task_tags=multi_tags.get("task_tags", []),
            complexity_tags=multi_tags.get("complexity_tags", []),
        )
        
        # Create EnhancedDAGMetadata
        return EnhancedDAGMetadata(
            dag_id=node["id"],
            description=node["description"],
            complexity=zm_data.get("complexity", "standard"),
            features=zm_data.get("features", []),
            framework=zm_data.get("framework", "generic"),
            node_count=atomic_props.get("node_count", 1),
            edge_count=atomic_props.get("edge_count", 0),
            zettelkasten_metadata=zettelkasten_metadata
        )


# Convenience functions for backward compatibility and ease of use

def create_pipeline(
    pipeline_id: str,
    config_path: str,
    **kwargs
) -> BasePipeline:
    """
    Convenience function to create a pipeline.
    
    Equivalent to PipelineFactory.create_pipeline()
    """
    return PipelineFactory.create_pipeline(pipeline_id, config_path, **kwargs)


def list_pipelines() -> List[Dict[str, Any]]:
    """
    Convenience function to list available pipelines.
    
    Equivalent to PipelineFactory.list_available_pipelines()
    """
    return PipelineFactory.list_available_pipelines()


def discover_pipelines(**criteria) -> List[str]:
    """
    Convenience function to discover pipelines.
    
    Equivalent to PipelineFactory.discover_pipelines()
    """
    return PipelineFactory.discover_pipelines(**criteria)
```

## Design Patterns Used

### 1. Factory Pattern

**Pattern**: Create objects without specifying exact class

**Implementation**:
- `PipelineFactory.create_pipeline()` creates `BasePipeline` instances
- Hides complexity of DAG function importing and metadata conversion
- Returns consistent interface regardless of pipeline type

**Benefits**:
- Single creation point for all pipelines
- Consistent API across all pipeline types
- Easy to extend with new creation methods

### 2. Registry Pattern

**Pattern**: Central registry for object metadata

**Implementation**:
- Leverages existing `CatalogRegistry` for pipeline metadata
- Single source of truth for all pipeline information
- Supports discovery and querying

**Benefits**:
- No metadata duplication
- Dynamic pipeline discovery
- Centralized metadata management

### 3. Strategy Pattern

**Pattern**: Define family of algorithms, make them interchangeable

**Implementation**:
- Each pipeline has different DAG creation strategy (different `create_*_dag` function)
- Factory selects appropriate strategy based on pipeline_id
- Strategies are encapsulated in `shared_dags` modules

**Benefits**:
- Reusable DAG creation logic
- Easy to add new strategies
- Decoupled from factory implementation

### 4. Dynamic Class Creation

**Pattern**: Create classes at runtime

**Implementation**:
- Anonymous `BasePipeline` subclass created for each pipeline
- Implements required abstract methods with closures over DAG function and metadata
- Named dynamically for debugging

**Benefits**:
- No need for explicit pipeline classes
- Maximum code reuse
- Maintains type safety and interface compliance

### 5. Lazy Evaluation

**Pattern**: Defer computation until needed

**Implementation**:
- DAG functions imported on-demand
- Results cached for reuse
- Metadata converted only when pipeline created

**Benefits**:
- Faster startup time
- Reduced memory usage
- Only load what's needed

## Benefits Over Current Approach

### 1. Code Elimination

**Before**: 7 pipeline files × ~70 lines = ~500 lines of boilerplate
```python
# xgb_e2e_comprehensive.py (70 lines)
# pytorch_e2e_standard.py (70 lines)
# xgb_training_simple.py (70 lines)
# ... 4 more files
```

**After**: Single factory implementation (~300 lines) + mapping dict (~10 lines)
```python
# core/pipeline_factory.py (310 lines total)
# Handles ALL pipelines dynamically
```

**Reduction**: ~190 lines saved, ~62% reduction

### 2. Single Source of Truth

**Before**: Metadata in 3 places
- `catalog_index.json`
- Pipeline class `get_enhanced_dag_metadata()`
- Shared DAG `get_dag_metadata()`

**After**: Single source
- `catalog_index.json` (authoritative)

### 3. Dynamic Discovery

**Before**: Must know exact class to import
```python
from cursus.pipeline_catalog.pipelines.xgb_e2e_comprehensive import XGBoostE2EComprehensivePipeline
```

**After**: Discover and create programmatically
```python
# List all pipelines
pipelines = PipelineFactory.list_available_pipelines()

# Search by criteria
xgb_pipelines = PipelineFactory.discover_pipelines(framework="xgboost")

# Create dynamically
pipeline = PipelineFactory.create_by_criteria(
    framework="xgboost",
    complexity="comprehensive"
)
```

### 4. Easier Maintenance

**Before**: Adding new pipeline
1. Create new shared DAG function
2. Create new pipeline class file
3. Update catalog_index.json
4. Duplicate metadata in all 3 places

**After**: Adding new pipeline
1. Create new shared DAG function
2. Update catalog_index.json
3. Add entry to `DAG_MODULE_MAP`

### 5. Better Integration

**Factory provides:**
- Direct integration with `CatalogRegistry`
- Leverages `TagBasedDiscovery` for intelligent search
- Compatible with `RecommendationEngine`
- Works with existing `ConnectionTraverser`

## Migration Path

### Phase 1: Add Factory (Week 1)

**Goal**: Introduce factory alongside existing classes

**Tasks**:
1. Create `core/pipeline_factory.py` with complete implementation
2. Add `DAG_MODULE_MAP` for all existing pipelines
3. Write unit tests for factory
4. Document usage in README

**Outcome**: Factory available for new code, existing code unchanged

### Phase 2: Update Documentation (Week 2)

**Goal**: Encourage factory usage

**Tasks**:
1. Update pipeline_catalog README to recommend factory
2. Add factory examples to documentation
3. Create migration guide for existing users
4. Update Jupyter notebooks to demonstrate factory

**Outcome**: New users adopt factory pattern

### Phase 3: Deprecate Pipeline Classes (Week 3-4)

**Goal**: Phase out individual pipeline classes

**Tasks**:
1. Add deprecation warnings to existing pipeline classes
2. Update all internal code to use factory
3. Migrate examples and tests to factory
4. Document breaking changes

**Outcome**: All new code uses factory

### Phase 4: Remove Pipeline Classes (Week 5)

**Goal**: Complete migration

**Tasks**:
1. Remove deprecated pipeline class files
2. Update imports throughout codebase
3. Clean up tests
4. Update final documentation

**Outcome**: Single factory-based implementation

### Backward Compatibility Strategy

**During Migration**: Keep existing classes as thin wrappers
```python
# In pipelines/xgb_e2e_comprehensive.py
import warnings
from ..core.pipeline_factory import PipelineFactory

class XGBoostE2EComprehensivePipeline:
    """DEPRECATED: Use PipelineFactory.create_pipeline('xgb_e2e_comprehensive') instead"""
    
    def __new__(cls, *args, **kwargs):
        warnings.warn(
            "XGBoostE2EComprehensivePipeline is deprecated. "
            "Use PipelineFactory.create_pipeline('xgb_e2e_comprehensive') instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return PipelineFactory.create_pipeline(
            'xgb_e2e_comprehensive',
            *args,
            **kwargs
        )
```

## Usage Examples

### Basic Creation

```python
from cursus.pipeline_catalog.core.pipeline_factory import PipelineFactory
from sagemaker import Session
from sagemaker.workflow.pipeline_context import PipelineSession

# Initialize sessions
sagemaker_session = Session()
role = sagemaker_session.get_caller_identity_arn()
pipeline_session = PipelineSession()

# Create pipeline by ID
pipeline = PipelineFactory.create_pipeline(
    pipeline_id="xgb_e2e_comprehensive",
    config_path="config.json",
    sagemaker_session=pipeline_session,
    execution_role=role
)

# Generate and execute
sm_pipeline = pipeline.generate_pipeline()
sm_pipeline.upsert()
execution = sm_pipeline.start()
```

### Discovery-Based Creation

```python
# Find pipelines by criteria
xgb_pipelines = PipelineFactory.discover_pipelines(framework="xgboost")
print(f"Found {len(xgb_pipelines)} XGBoost pipelines")

# Create by criteria
pipeline = PipelineFactory.create_by_criteria(
    framework="xgboost",
    complexity="comprehensive",
    config_path="config.json",
    sagemaker_session=pipeline_session,
    execution_role=role
)
```

### Listing and Exploration

```python
# List all available pipelines
pipelines = PipelineFactory.list_available_pipelines()

for p in pipelines:
    print(f"{p['id']}: {p['title']}")
    print(f"  Framework: {p['framework']}, Complexity: {p['complexity']}")
    print(f"  Features: {', '.join(p['features'])}")
    print()

# Search by text
evaluation_pipelines = PipelineFactory.discover_pipelines(
    search_text="evaluation"
)

# Filter by multiple criteria
simple_training = PipelineFactory.discover_pipelines(
    complexity="simple",
    task="training"
)
```

### Integration with Existing Tools

```python
from cursus.pipeline_catalog.core.recommendation_engine import PipelineRecommendationEngine
from cursus.pipeline_catalog.core.catalog_registry import CatalogRegistry
from cursus.pipeline_catalog.core.tag_discovery import TagBasedDiscovery
from cursus.pipeline_catalog.core.connection_traverser import ConnectionTraverser

# Set up discovery infrastructure
registry = CatalogRegistry()
discovery = TagBasedDiscovery(registry)
traverser = ConnectionTraverser(registry)
engine = PipelineRecommendationEngine(registry, traverser, discovery)

# Get recommendations
recommendations = engine.recommend_for_use_case(
    "I need to train an XGBoost model for production"
)

# Create recommended pipeline
if recommendations:
    top_recommendation = recommendations[0]
    pipeline = PipelineFactory.create_pipeline(
        pipeline_id=top_recommendation.pipeline_id,
        config_path="config.json"
    )
```

### Convenience Functions

```python
from cursus.pipeline_catalog.core.pipeline_factory import (
    create_pipeline,
    list_pipelines,
    discover_pipelines
)

# Simplified API
pipeline = create_pipeline("xgb_e2e_comprehensive", "config.json")
all_pipelines = list_pipelines()
xgb_pipelines = discover_pipelines(framework="xgboost")
```

## Testing Strategy

### Unit Tests

```python
import pytest
from cursus.pipeline_catalog.core.pipeline_factory import PipelineFactory

def test_create_pipeline_by_id():
    """Test creating pipeline by ID"""
    pipeline = PipelineFactory.create_pipeline(
        pipeline_id="xgb_e2e_comprehensive",
        config_path="test_config.json"
    )
    
    assert pipeline is not None
    assert hasattr(pipeline, 'create_dag')
    assert hasattr(pipeline, 'get_enhanced_dag_metadata')

def test_list_available_pipelines():
    """Test listing all pipelines"""
    pipelines = PipelineFactory.list_available_pipelines()
    
    assert len(pipelines) > 0
    assert all('id' in p for p in pipelines)
    assert all('framework' in p for p in pipelines)

def test_discover_by_framework():
    """Test discovery by framework"""
    xgb_pipelines = PipelineFactory.discover_pipelines(framework="xgboost")
    
    assert len(xgb_pipelines) > 0
    
    # Verify all are actually XGBoost pipelines
    for pid in xgb_pipelines:
        info = PipelineFactory.get_pipeline_info(pid)
        assert info['zettelkasten_metadata']['framework'] == 'xgboost'

def test_create_by_criteria():
    """Test creating pipeline by criteria"""
    pipeline = PipelineFactory.create_by_criteria(
        framework="xgboost",
        complexity="simple",
        config_path="test_config.json"
    )
    
    assert pipeline is not None

def test_invalid_pipeline_id():
    """Test error handling for invalid pipeline ID"""
    with pytest.raises(ValueError, match="not found in registry"):
        PipelineFactory.create_pipeline(
            pipeline_id="nonexistent_pipeline",
            config_path="test_config.json"
        )

def test_dag_function_caching():
    """Test that DAG functions are cached"""
    factory = PipelineFactory()
    
    # First call imports and caches
    func1 = factory._import_dag_function("xgb_e2e_comprehensive")
    
    # Second call should return cached function
    func2 = factory._import_dag_function("xgb_e2e_comprehensive")
    
    assert func1 is func2  # Same object reference

def test_metadata_conversion():
    """Test registry node to metadata conversion"""
    factory = PipelineFactory()
    node = factory.registry.get_pipeline_node("xgb_e2e_comprehensive")
    
    metadata = factory._create_metadata_from_node(node)
    
    assert metadata is not None
    assert metadata.dag_id == "xgb_e2e_comprehensive"
    assert metadata.zettelkasten_metadata is not None
```

### Integration Tests

```python
def test_end_to_end_pipeline_creation():
    """Test complete pipeline creation and execution"""
    from sagemaker.workflow.pipeline_context import PipelineSession
    
    # Create pipeline
    pipeline = PipelineFactory.create_pipeline(
        pipeline_id="dummy_e2e_basic",
        config_path="test_config.json",
        sagemaker_session=PipelineSession()
    )
    
    # Generate SageMaker pipeline
    sm_pipeline = pipeline.generate_pipeline()
    
    assert sm_pipeline is not None
    assert sm_pipeline.name is not None

def test_discovery_integration():
    """Test integration with discovery systems"""
    # Should work seamlessly with existing discovery tools
    pipelines = PipelineFactory.discover_pipelines(
        framework="pytorch",
        complexity="standard"
    )
    
    # Create one of the discovered pipelines
    if pipelines:
        pipeline = PipelineFactory.create_pipeline(
            pipeline_id=pipelines[0],
            config_path="test_config.json"
        )
        assert pipeline is not None
```

## Error Handling

### Common Errors and Solutions

#### 1. Pipeline Not Found

**Error**: `ValueError: Pipeline 'xyz' not found in registry`

**Solution**: 
- Check pipeline ID spelling
- Use `list_available_pipelines()` to see valid IDs
- Verify `catalog_index.json` is up to date

#### 2. DAG Module Not Found

**Error**: `ValueError: No DAG module mapping for 'xyz'`

**Solution**:
- Add entry to `DAG_MODULE_MAP` in `pipeline_factory.py`
- Verify shared DAG module exists
- Check module path is correct

#### 3. DAG Function Not Found

**Error**: `ValueError: No create_*_dag function found in module`

**Solution**:
- Verify DAG module has function named `create_*_dag`
- Check function naming convention
- Ensure function is not private (doesn't start with `_`)

#### 4. No Pipelines Match Criteria

**Error**: `ValueError: No pipeline found for criteria`

**Solution**:
- Use `discover_pipelines()` without criteria to see all options
- Try broader search criteria
- Check if pipelines exist for that combination

## Performance Considerations

### Lazy Loading

- DAG functions imported only when needed
- Results cached for repeated use
- Registry loaded once per factory instance

### Memory Usage

- Registry loaded into memory (~few MB)
- DAG functions cached (~KB per function)
- Minimal overhead per pipeline creation

### Benchmark Results

Estimated performance on typical hardware:

| Operation | Time | Notes |
|-----------|------|-------|
| First pipeline creation | ~100ms | Includes registry load and DAG import |
| Subsequent creations | ~10ms | Uses cached functions |
| List all pipelines | ~5ms | Registry already loaded |
| Discovery search | ~20ms | Tag-based search |
| Text search | ~50ms | Full-text across descriptions |

## Future Enhancements

### 1. Auto-Generate DAG Module Map

**Goal**: Eliminate manual `DAG_MODULE_MAP` maintenance

**Approach**:
- Scan `shared_dags/` directory for all DAG modules
- Parse module names to extract pipeline IDs
- Build map automatically at runtime

### 2. Pipeline Composition Support

**Goal**: Enable combining multiple pipelines

**Approach**:
- Add `compose_pipelines()` method
- Merge DAGs from multiple pipelines
- Handle metadata aggregation

### 3. Template-Based Creation

**Goal**: Create pipelines from templates

**Approach**:
- Add template definitions to registry
- Support parameter substitution
- Enable rapid pipeline prototyping

### 4. Version Management

**Goal**: Support pipeline versioning

**Approach**:
- Add version field to registry
- Support creating specific versions
- Maintain version history

## Conclusion

The Pipeline Factory design eliminates ~500 lines of redundant code by leveraging existing infrastructure in `pipeline_catalog`. By connecting `CatalogRegistry`, `shared_dags`, `TagBasedDiscovery`, and `EnhancedDAGMetadata`, we create a powerful, maintainable system for dynamic pipeline creation.

**Key Benefits:**
- **62% code reduction** - From ~500 lines to ~310 lines
- **Single source of truth** - Metadata only in `catalog_index.json`
- **Dynamic discovery** - Programmatic pipeline exploration
- **Better integration** - Works seamlessly with existing tools
- **Easier maintenance** - Add pipelines with 3 steps instead of 4

**Next Steps:**
1. Implement `core/pipeline_factory.py` 
2. Add unit tests
3. Update documentation
4. Migrate existing code
5. Remove redundant pipeline classes

This design represents a significant architectural improvement that simplifies the codebase while enhancing functionality and maintainability.

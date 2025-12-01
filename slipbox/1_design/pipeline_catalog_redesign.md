---
tags:
  - design
  - architecture
  - pipeline_catalog
  - zettelkasten
  - discovery
  - factory_pattern
  - user_experience
keywords:
  - pipeline factory
  - dag discovery
  - knowledge graph
  - auto-discovery
  - search-driven creation
  - workspace support
  - metadata enrichment
topics:
  - pipeline management
  - knowledge systems
  - factory pattern
  - discovery mechanisms
  - user experience design
language: python
date of note: 2025-11-30
---

# Pipeline Catalog System Redesign: Knowledge-Driven Pipeline Discovery & Creation

## Overview

This document presents a comprehensive redesign of the `pipeline_catalog` system, transforming it from a manual, class-based approach into an intelligent, search-driven knowledge system. The new design integrates:

1. **Zettelkasten Knowledge Base** - Rich metadata graph on top of shared_dags
2. **DAGAutoDiscovery** - AST-based automatic discovery following step_catalog patterns
3. **PipelineFactory** - Search-query-driven dynamic pipeline creation
4. **Knowledge Graph Navigation** - Relationship and evolution tracking
5. **Interactive Exploration** - Multiple discovery and selection interfaces

**Core Philosophy**: Transform pipeline discovery from "you must know the exact class name" to "describe what you need and the system finds it for you."

## Problem Statement

### Current System Limitations

**1. Manual Discovery Burden**
```python
# User must know exact import path and class name
from cursus.pipeline_catalog.pipelines.xgb_e2e_comprehensive import (
    XGBoostE2EComprehensivePipeline
)
```

**2. Code Redundancy**
- 7 pipeline wrapper classes (~500 lines of boilerplate)
- Each class just calls shared DAG function + returns metadata
- Metadata duplicated in 3 places (registry, class, DAG function)

**3. Poor Discoverability**
- No way to explore available pipelines without reading code
- No search functionality
- No relationship visualization
- No complexity evolution tracking

**4. Limited Extensibility**
- Adding new pipeline requires creating new class file
- No support for workspace-local pipelines
- Cannot dynamically compose pipelines

**5. Weak Knowledge Management**
- Rich metadata exists but not leveraged
- No knowledge graph connections
- No pipeline evolution tracking
- Missing use case → pipeline recommendations

## User Experience Vision

### Target User Journeys

#### Journey 1: First-Time User - Discovery
```python
from cursus.pipeline_catalog import PipelineFactory

factory = PipelineFactory()

# Natural exploration
available = factory.search("xgboost training")
# Returns ranked results with descriptions and features

# Create by description
pipeline = factory.create_by_search(
    query="comprehensive xgboost with evaluation",
    config_path="config.json"
)
```

**UX Benefits**:
- No need to know exact pipeline names
- Natural language search
- Ranked, relevant results
- One-line creation

#### Journey 2: Experienced User - Structured Selection
```python
from cursus.pipeline_catalog import PipelineFactory

# Precise criteria-based creation
pipeline = factory.create(
    framework="xgboost",
    complexity="comprehensive",
    features=["training", "evaluation", "calibration"],
    config_path="config.json"
)
```

**UX Benefits**:
- Structured, predictable API
- Type-safe parameters
- Clear feature selection
- No class imports needed

#### Journey 3: Developer - Local Development
```python
# Step 1: Create custom DAG in workspace
# workspace/my_project/dags/custom_financial_dag.py

def create_custom_financial_risk_dag() -> PipelineDAG:
    """Custom pipeline for financial risk models."""
    dag = PipelineDAG()
    dag.add_node("FinancialDataPrep", ...)
    dag.add_node("RiskCalibration", ...)
    return dag

# Step 2: Automatic discovery (no registration needed!)
factory = PipelineFactory(workspace_dir="workspace/my_project")

# Step 3: Use immediately
pipeline = factory.create("custom_financial_risk", config_path="config.json")
```

**UX Benefits**:
- Zero-friction local development
- Automatic discovery of workspace DAGs
- No manual registration
- Immediate usability

#### Journey 4: Data Scientist - Interactive Exploration
```python
from cursus.pipeline_catalog import PipelineExplorer

explorer = PipelineExplorer()

# Question-driven exploration
xgb_options = explorer.list_by_framework("xgboost")
simple_pipelines = explorer.list_by_complexity("simple")

# Detailed investigation
info = explorer.get_info("xgb_e2e_comprehensive")
print(info.features)  # ['training', 'evaluation', 'calibration']
print(info.dag_structure)  # Visual DAG representation
print(info.similar_pipelines)  # Recommendations

# Find related pipelines
similar = explorer.find_similar("xgb_e2e_comprehensive")
```

**UX Benefits**:
- Interactive discovery
- Multiple exploration dimensions
- Visual representations
- Smart recommendations

#### Journey 5: ML Engineer - Knowledge Navigation
```python
from cursus.pipeline_catalog import PipelineKnowledgeGraph

kg = PipelineKnowledgeGraph()

# Understand relationships
relationships = kg.get_relationships("xgb_e2e_comprehensive")
# Shows:
# - Extends: xgb_training_simple
# - Similar to: pytorch_e2e_standard
# - Used by: 12 projects
# - Complexity evolution: simple -> standard -> comprehensive

# Visualize ecosystem
kg.visualize("xgboost")  # Interactive graph
```

**UX Benefits**:
- Understand pipeline evolution
- See relationships visually
- Track usage patterns
- Navigate complexity paths

#### Journey 6: Product Manager - Requirements to Pipeline
```python
from cursus.pipeline_catalog import PipelineAdvisor

advisor = PipelineAdvisor()

# Natural language requirements
recommendations = advisor.recommend(
    use_case="I need to train an XGBoost model for production",
    requirements=["model_registration", "calibration", "monitoring"]
)

# Returns:
# 1. xgb_e2e_comprehensive (95% match)
#    - All required features included
#    - Production-ready
# 2. xgb_training_calibrated (75% match)
#    - Missing registration, can be extended

# Gap analysis
gap = advisor.analyze_gap(
    selected="xgb_training_simple",
    requirements=["calibration", "registration"]
)
print(gap.missing_features)
print(gap.upgrade_path)  # Suggests better pipeline
```

**UX Benefits**:
- Requirements → recommendations
- Gap analysis
- Upgrade paths
- Clear justifications

### UX Comparison Table

| Aspect | Current System | New System |
|--------|---------------|------------|
| **Discovery** | Must know exact class name | Natural language + structured search |
| **Learning Curve** | Steep (15+ classes) | Gentle (1 factory, multiple interfaces) |
| **Exploration** | Manual code reading | Interactive tools + visualizations |
| **Local Dev** | Complex setup | Auto-discovery, zero config |
| **Documentation** | Scattered | Auto-generated, comprehensive |
| **Error Messages** | Generic "not found" | Smart suggestions + corrections |
| **Selection** | Trial and error | Guided recommendations |
| **Relationships** | Hidden | Explicit knowledge graph |
| **Extensibility** | Create class file | Drop DAG in workspace |
| **Composition** | Not supported | Dynamic composition |

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        User Interfaces                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │ PipelineFactory│ │PipelineExplorer│ │KnowledgeGraph│             │
│  │ (Creation)   │  │ (Discovery)   │  │ (Navigation) │             │
│  └──────┬───────┘  └──────┬────────┘  └──────┬───────┘             │
└─────────┼──────────────────┼────────────────────┼──────────────────┘
          │                  │                    │
          ▼                  ▼                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Core Discovery Layer                           │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │              DAGAutoDiscovery                              │    │
│  │  • AST-based scanning                                      │    │
│  │  • Workspace prioritization                                │    │
│  │  • Registry integration                                    │    │
│  │  • Function extraction & caching                           │    │
│  └────────────┬────────────────────────────────┬──────────────┘    │
└───────────────┼────────────────────────────────┼────────────────────┘
                │                                │
                ▼                                ▼
┌─────────────────────────────────┐  ┌──────────────────────────────┐
│   Zettelkasten Knowledge Base   │  │    Existing Infrastructure   │
│                                  │  │                              │
│  ┌────────────────────────────┐ │  │  • CatalogRegistry          │
│  │   catalog_index.json       │ │  │  • TagBasedDiscovery        │
│  │   (Enhanced Metadata)      │ │  │  • ConnectionTraverser      │
│  │                            │ │  │  • RecommendationEngine     │
│  │  • Pipeline nodes          │ │  │  • BasePipeline             │
│  │  • Relationships           │ │  │                              │
│  │  • Evolution tracking      │ │  └──────────────────────────────┘
│  │  • Feature metadata        │ │
│  │  • Usage patterns          │ │
│  └────────────────────────────┘ │
└─────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DAG Repository                                 │
│                                                                     │
│  Package DAGs              Workspace DAGs (Priority)               │
│  ┌────────────────────┐    ┌────────────────────┐                 │
│  │ shared_dags/       │    │ workspace/dags/    │                 │
│  │  xgboost/          │    │  custom_dags/      │                 │
│  │  pytorch/          │    │  experimental/     │                 │
│  │  lightgbm/         │    │                    │                 │
│  └────────────────────┘    └────────────────────┘                 │
│                                                                     │
│  Convention: create_*_dag() + get_dag_metadata()                   │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

```
User Query: "xgboost comprehensive training"
    │
    ▼
PipelineFactory.create_by_search()
    │
    ├─► TagBasedDiscovery.search_by_text()
    │   └─► Returns: ["xgb_e2e_comprehensive"]
    │
    ├─► DAGAutoDiscovery.load_dag_info()
    │   ├─► Find DAG file (workspace first, then package)
    │   ├─► Extract create_*_dag function using AST
    │   ├─► Extract metadata using AST
    │   └─► Returns: DAGInfo object
    │
    ├─► CatalogRegistry.get_pipeline_node()
    │   └─► Returns: Enhanced metadata from catalog_index.json
    │
    ├─► Enrich DAGInfo with registry metadata
    │
    └─► Create dynamic BasePipeline subclass
        ├─► Implements create_dag() → calls discovered function
        ├─► Implements get_metadata() → returns enriched metadata
        └─► Returns: BasePipeline instance
```

## Core Components

### 1. DAGAutoDiscovery

**Purpose**: AST-based automatic discovery of pipeline DAGs following step_catalog patterns

**Key Features**:
- AST parsing (no imports, no circular dependencies)
- Workspace prioritization (local overrides package)
- Registry integration (validation + enrichment)
- Function caching (performance)
- Convention enforcement (create_*_dag + get_dag_metadata)

**Architecture**:
```python
class DAGInfo:
    """Rich metadata about a discovered DAG."""
    dag_id: str
    dag_name: str
    dag_path: Path
    workspace_id: str
    framework: str
    complexity: str
    features: List[str]
    node_count: int
    edge_count: int
    create_function: Optional[Callable]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""

class DAGAutoDiscovery:
    """
    AST-based DAG discovery with workspace support.
    
    Discovery Strategy:
    1. Scan workspace_dirs/dags/ (highest priority)
    2. Scan package_root/pipeline_catalog/shared_dags/
    3. Parse files using AST (no imports)
    4. Extract create_*_dag() functions
    5. Extract get_dag_metadata() functions
    6. Cross-reference with CatalogRegistry
    7. Cache results for performance
    """
    
    def __init__(self, 
                 package_root: Path,
                 workspace_dirs: Optional[List[Path]] = None,
                 registry_path: str = "catalog_index.json"):
        self.registry = CatalogRegistry(registry_path)
        self._dag_cache: Dict[str, DAGInfo] = {}
        self._discovery_complete = False
    
    # Discovery Methods
    def discover_all_dags(self) -> Dict[str, DAGInfo]:
        """Discover all DAGs from package and workspaces."""
        
    def load_dag_info(self, dag_id: str) -> Optional[DAGInfo]:
        """Load specific DAG with workspace-aware priority."""
        
    def discover_by_framework(self, framework: str) -> Dict[str, DAGInfo]:
        """Find all DAGs for a specific framework."""
        
    def discover_by_criteria(self, **criteria) -> Dict[str, DAGInfo]:
        """Find DAGs matching multiple criteria."""
    
    # Core Discovery Implementation
    def _run_discovery(self):
        """Execute complete discovery process."""
        
    def _scan_dag_directory(self, dir_path: Path, workspace_id: str) -> Dict[str, DAGInfo]:
        """Scan directory for DAG files."""
        
    # AST Analysis
    def _extract_dag_from_ast(self, file_path: Path) -> Optional[DAGInfo]:
        """Extract DAG information using AST parsing."""
        
    def _find_create_dag_functions(self, tree: ast.AST) -> List[ast.FunctionDef]:
        """Find create_*_dag() functions in AST."""
        
    def _extract_metadata_from_ast(self, tree: ast.AST) -> Dict[str, Any]:
        """Extract metadata from get_dag_metadata() function."""
        
    # Registry Integration
    def _enrich_with_registry(self, dag_info: DAGInfo) -> DAGInfo:
        """Enrich DAGInfo with CatalogRegistry metadata."""
        
    def _validate_against_registry(self, dag_id: str) -> bool:
        """Validate DAG exists and matches registry."""
    
    # Dynamic Loading
    def _load_create_function(self, file_path: Path, func_name: str) -> Callable:
        """Dynamically load create_*_dag function."""
```

**Integration Points**:
- **Input**: File system (shared_dags/, workspace/dags/)
- **Input**: CatalogRegistry (validation, enrichment)
- **Output**: DAGInfo objects (consumed by PipelineFactory)
- **Pattern**: Mirrors BuilderAutoDiscovery and ScriptAutoDiscovery

### 2. PipelineFactory

**Purpose**: Search-driven dynamic pipeline creation without class definitions

**Key Features**:
- Multiple creation interfaces (search, criteria, direct ID)
- Uses DAGAutoDiscovery for DAG resolution
- Creates anonymous BasePipeline subclasses at runtime
- Caches results for performance
- No hardcoded mappings needed

**Architecture**:
```python
class PipelineFactory:
    """
    Factory for creating pipelines dynamically from discovered DAGs.
    
    Creation Methods:
    1. create(pipeline_id, ...) - Direct by ID
    2. create_by_search(query, ...) - Natural language search
    3. create_by_criteria(framework, complexity, ...) - Structured
    4. create_multiple(...) - Batch creation
    """
    
    def __init__(self, 
                 registry_path: str = "catalog_index.json",
                 workspace_dir: Optional[Path] = None):
        self.registry = CatalogRegistry(registry_path)
        self.discovery = TagBasedDiscovery(self.registry)
        self.dag_discovery = DAGAutoDiscovery(
            package_root=Path(__file__).parent.parent,
            workspace_dirs=[workspace_dir] if workspace_dir else [],
            registry_path=registry_path
        )
        self._pipeline_cache: Dict[str, Type[BasePipeline]] = {}
    
    # Creation Methods
    @classmethod
    def create(cls, 
               pipeline_id: str, 
               config_path: str,
               **base_pipeline_kwargs) -> BasePipeline:
        """
        Create pipeline by ID.
        
        Example:
            pipeline = PipelineFactory.create(
                pipeline_id="xgb_e2e_comprehensive",
                config_path="config.json"
            )
        """
        
    @classmethod
    def create_by_search(cls,
                        query: str,
                        config_path: str,
                        **base_pipeline_kwargs) -> BasePipeline:
        """
        Create pipeline using natural language search.
        
        Example:
            pipeline = PipelineFactory.create_by_search(
                query="comprehensive xgboost with evaluation",
                config_path="config.json"
            )
        """
        
    @classmethod
    def create_by_criteria(cls,
                          framework: str,
                          complexity: str,
                          config_path: str,
                          features: Optional[List[str]] = None,
                          **base_pipeline_kwargs) -> BasePipeline:
        """
        Create pipeline using structured criteria.
        
        Example:
            pipeline = PipelineFactory.create_by_criteria(
                framework="xgboost",
                complexity="comprehensive",
                features=["training", "evaluation"],
                config_path="config.json"
            )
        """
    
    # Discovery Methods
    @classmethod
    def search(cls, query: str) -> List[Dict[str, Any]]:
        """
        Search for pipelines using natural language.
        
        Returns ranked results with metadata.
        """
        
    @classmethod
    def list_available(cls, 
                      framework: Optional[str] = None,
                      complexity: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available pipelines with optional filtering."""
        
    @classmethod
    def get_info(cls, pipeline_id: str) -> Dict[str, Any]:
        """Get detailed information about a pipeline."""
    
    # Internal Implementation
    def _create_dynamic_pipeline_class(self,
                                      dag_info: DAGInfo,
                                      metadata: EnhancedDAGMetadata) -> Type[BasePipeline]:
        """
        Create anonymous BasePipeline subclass at runtime.
        
        The class:
        - Implements create_dag() using discovered function
        - Implements get_metadata() using enriched metadata
        - Is cached for reuse
        """
        
    def _resolve_pipeline_id(self, query: str) -> str:
        """Resolve search query to pipeline ID."""
        
    def _enrich_metadata(self, 
                        dag_info: DAGInfo,
                        registry_node: Dict[str, Any]) -> EnhancedDAGMetadata:
        """Combine DAG metadata with registry metadata."""
```

**Integration Points**:
- **Uses**: DAGAutoDiscovery (DAG resolution)
- **Uses**: CatalogRegistry (metadata source)
- **Uses**: TagBasedDiscovery (search functionality)
- **Returns**: BasePipeline instances (standard interface)
- **Replaces**: All individual pipeline classes in pipelines/

### 3. PipelineExplorer

**Purpose**: Interactive pipeline discovery and exploration

**Key Features**:
- Multiple filtering dimensions
- Detailed pipeline information
- Similarity recommendations
- Visual representations
- Jupyter notebook support

**Architecture**:
```python
class PipelineExplorer:
    """
    Interactive pipeline exploration interface.
    
    Provides multiple ways to discover and understand pipelines.
    """
    
    def __init__(self, workspace_dir: Optional[Path] = None):
        self.factory = PipelineFactory(workspace_dir=workspace_dir)
        self.registry = self.factory.registry
        self.discovery = self.factory.discovery
    
    # Listing Methods
    def list_all(self) -> List[Dict[str, Any]]:
        """List all available pipelines."""
        
    def list_by_framework(self, framework: str) -> List[Dict[str, Any]]:
        """List pipelines for specific framework."""
        
    def list_by_complexity(self, complexity: str) -> List[Dict[str, Any]]:
        """List pipelines of specific complexity."""
        
    def list_by_features(self, features: List[str]) -> List[Dict[str, Any]]:
        """List pipelines with specific features."""
    
    # Information Methods
    def get_info(self, pipeline_id: str) -> PipelineInfo:
        """
        Get comprehensive information about a pipeline.
        
        Returns:
            PipelineInfo object with:
            - Basic metadata
            - DAG structure visualization
            - Feature list
            - Similar pipelines
            - Usage examples
            - Configuration requirements
        """
        
    def get_dag_structure(self, pipeline_id: str) -> DAGStructure:
        """Get visual representation of DAG structure."""
    
    # Similarity Methods
    def find_similar(self, pipeline_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find pipelines similar to the given one."""
        
    def compare(self, pipeline_ids: List[str]) -> ComparisonResult:
        """
        Compare multiple pipelines side-by-side.
        
        Returns comparison table with:
        - Features
        - Complexity
        - Step counts
        - Estimated costs
        - Use case recommendations
        """
    
    # Interactive Methods (Jupyter support)
    def display(self) -> IPythonWidget:
        """Display interactive pipeline browser widget."""
        
    def preview(self, pipeline_id: str) -> IPythonDisplay:
        """Preview pipeline with visualization and details."""
```

**Integration Points**:
- **Uses**: PipelineFactory (creation backend)
- **Uses**: CatalogRegistry (metadata)
- **Uses**: TagBasedDiscovery (search/filter)
- **Provides**: User-friendly exploration interface

### 4. PipelineKnowledgeGraph

**Purpose**: Navigate pipeline relationships and evolution

**Key Features**:
- Relationship tracking (extends, similar_to, used_by)
- Evolution paths (simple → standard → comprehensive)
- Usage analytics
- Visual graph generation
- Connection traversal

**Architecture**:
```python
class PipelineKnowledgeGraph:
    """
    Knowledge graph for pipeline relationships and evolution.
    
    Builds on existing ConnectionTraverser with enhanced semantics.
    """
    
    def __init__(self):
        self.registry = CatalogRegistry()
        self.traverser = ConnectionTraverser(self.registry)
    
    # Relationship Methods
    def get_relationships(self, pipeline_id: str) -> PipelineRelationships:
        """
        Get all relationships for a pipeline.
        
        Returns:
            - extends: Pipelines this extends
            - extended_by: Pipelines that extend this
            - similar_to: Similar pipelines (different framework)
            - used_by: Projects/users using this
            - related_by_feature: Pipelines sharing features
        """
        
    def get_evolution_path(self, pipeline_id: str) -> EvolutionPath:
        """
        Get complexity evolution path.
        
        Returns sequence: simple → standard → comprehensive
        """
        
    def get_framework_ecosystem(self, framework: str) -> FrameworkEcosystem:
        """Get all pipelines and their relationships for a framework."""
    
    # Navigation Methods
    def find_path(self, from_id: str, to_id: str) -> List[str]:
        """Find connection path between two pipelines."""
        
    def get_neighbors(self, pipeline_id: str, degree: int = 1) -> List[str]:
        """Get pipelines within N degrees of connection."""
    
    # Visualization Methods
    def visualize(self, 
                  root: Optional[str] = None,
                  framework: Optional[str] = None) -> GraphVisualization:
        """
        Generate interactive graph visualization.
        
        Uses existing visualization tools enhanced with:
        - Node coloring by framework
        - Edge styling by relationship type
        - Interactive exploration
        - Filtering controls
        """
        
    def export_graph(self, format: str = "json") -> str:
        """Export graph in various formats (JSON, DOT, GraphML)."""
```

**Integration Points**:
- **Extends**: ConnectionTraverser (relationship tracking)
- **Uses**: CatalogRegistry (metadata + connections)
- **Provides**: Knowledge navigation interface

### 5. PipelineAdvisor

**Purpose**: Intelligent pipeline recommendations and gap analysis

**Key Features**:
- Requirements → recommendations
- Gap analysis
- Upgrade path suggestions
- Use case matching
- Confidence scoring

**Architecture**:
```python
class PipelineAdvisor:
    """
    Intelligent pipeline recommendation engine.
    
    Builds on existing RecommendationEngine with enhanced semantics.
    """
    
    def __init__(self):
        self.registry = CatalogRegistry()
        self.discovery = TagBasedDiscovery(self.registry)
        self.kg = PipelineKnowledgeGraph()
        self.recommender = PipelineRecommendationEngine(...)
    
    # Recommendation Methods
    def recommend(self,
                  use_case: str,
                  requirements: Optional[List[str]] = None) -> List[Recommendation]:
        """
        Get ranked recommendations for a use case.
        
        Args:
            use_case: Natural language description
            requirements: Optional feature requirements
            
        Returns:
            Ranked recommendations with:
            - Pipeline ID
            - Match score
            - Explanation
            - Missing features (if any)
            - Alternative options
        """
        
    def analyze_gap(self,
                   selected: str,
                   requirements: List[str]) -> GapAnalysis:
        """
        Analyze gap between selected pipeline and requirements.
        
        Returns:
            - Missing features
            - Upgrade path suggestions
            - Alternative pipelines
            - Extension options
        """
        
    def suggest_upgrade(self, 
                       current: str,
                       desired_features: List[str]) -> UpgradeSuggestion:
        """Suggest upgrade path to get desired features."""
    
    # Matching Methods
    def match_requirements(self, 
                          requirements: List[str]) -> List[Dict[str, Any]]:
        """Find all pipelines matching requirements."""
        
    def score_match(self,
                   pipeline_id: str,
                   requirements: List[str]) -> float:
        """Score how well a pipeline matches requirements."""
```

**Integration Points**:
- **Extends**: RecommendationEngine (recommendation logic)
- **Uses**: TagBasedDiscovery (search)
- **Uses**: PipelineKnowledgeGraph (relationships)
- **Provides**: Intelligent guidance interface

## Integration with Existing System

### Existing Components (Keep + Enhance)

#### 1. CatalogRegistry
**Status**: Keep, enhance metadata

**Current Role**:
- Central metadata registry (catalog_index.json)
- Pipeline node storage
- Connection management

**Enhancements**:
```json
{
  "nodes": {
    "xgb_e2e_comprehensive": {
      "id": "xgb_e2e_comprehensive",
      "title": "XGBoost E2E Comprehensive Pipeline",
      
      // NEW: Source file reference for discovery
      "source_file": "shared_dags/xgboost/complete_e2e_dag.py",
      
      // NEW: Discovery metadata
      "discovery_metadata": {
        "create_function": "create_xgboost_complete_e2e_dag",
        "last_discovered": "2025-11-30T22:00:00Z",
        "workspace_override": false
      },
      
      // EXISTING: Rich Zettelkasten metadata
      "zettelkasten_metadata": { ... },
      "atomic_properties": { ... },
      "multi_dimensional_tags": { ... }
    }
  }
}
```

#### 2. TagBasedDiscovery
**Status**: Keep, use as-is

**Role**:
- Text-based search
- Multi-criteria filtering
- Tag-based discovery

**Integration**:
- Used by PipelineFactory for search-driven creation
- Used by PipelineExplorer for filtering
- Used by PipelineAdvisor for requirement matching

#### 3. ConnectionTraverser
**Status**: Keep, extend

**Role**:
- Relationship navigation
- Path finding
- Connection analysis

**Integration**:
- Extended by PipelineKnowledgeGraph
- Used for evolution path tracking
- Used for similarity discovery

#### 4. RecommendationEngine
**Status**: Keep, extend

**Role**:
- Use case → pipeline matching
- Recommendation generation
- Scoring logic

**Integration**:
- Extended by PipelineAdvisor
- Used for intelligent recommendations
- Enhanced with gap analysis

#### 5. BasePipeline
**Status**: Keep as-is

**Role**:
- Abstract base class for pipelines
- Standard interface (create_dag, get_metadata)
- SageMaker integration

**Integration**:
- PipelineFactory creates anonymous subclasses
- All existing functionality preserved
- No changes to consuming code

### New Components (Add)

#### 1. DAGAutoDiscovery
**Location**: `core/dag_discovery.py`

**Purpose**: AST-based DAG discovery

**Integration**:
- Scans shared_dags/ and workspace/dags/
- Integrates with CatalogRegistry for validation
- Follows step_catalog discovery patterns
- Used by PipelineFactory for DAG resolution

#### 2. PipelineFactory
**Location**: `core/pipeline_factory.py`

**Purpose**: Dynamic pipeline creation

**Integration**:
- Uses DAGAutoDiscovery for DAG finding
- Uses CatalogRegistry for metadata
- Uses TagBasedDiscovery for search
- Creates BasePipeline instances

#### 3. PipelineExplorer
**Location**: `core/pipeline_explorer.py`

**Purpose**: Interactive exploration interface

**Integration**:
- Uses PipelineFactory as backend
- Uses TagBasedDiscovery for filtering
- Provides user-friendly API
- Jupyter notebook integration

#### 4. PipelineKnowledgeGraph
**Location**: `core/pipeline_knowledge_graph.py`

**Purpose**: Relationship navigation

**Integration**:
- Extends ConnectionTraverser
- Uses CatalogRegistry for metadata
- Generates visualizations
- Tracks evolution paths

#### 5. PipelineAdvisor
**Location**: `core/pipeline_advisor.py`

**Purpose**: Intelligent recommendations

**Integration**:
- Extends RecommendationEngine
- Uses TagBasedDiscovery for matching
- Uses PipelineKnowledgeGraph for relationships
- Provides gap analysis

### Components to Remove

#### pipelines/ Directory
**Status**: Delete after migration

**Reason**: All functionality replaced by PipelineFactory

**Files to Remove**:
- `pipelines/xgb_e2e_comprehensive.py`
- `pipelines/pytorch_e2e_standard.py`
- `pipelines/xgb_training_simple.py`
- `pipelines/xgb_training_calibrated.py`
- `pipelines/xgb_training_evaluation.py`
- `pipelines/pytorch_training_basic.py`
- `pipelines/dummy_e2e_basic.py`

**Migration Strategy**: Keep with deprecation warnings during transition

## DAG Convention Requirements

### File Naming Convention
```
shared_dags/
  <framework>/
    <type>_dag.py
    
Examples:
  shared_dags/xgboost/complete_e2e_dag.py
  shared_dags/pytorch/standard_e2e_dag.py
  shared_dags/xgboost/simple_dag.py
```

### Function Requirements

Every DAG module MUST implement:

```python
def create_<framework>_<type>_dag() -> PipelineDAG:
    """
    Create the pipeline DAG.
    
    Returns:
        PipelineDAG instance with all nodes and edges
    """
    dag = PipelineDAG()
    # Add nodes
    # Add edges
    return dag

def get_dag_metadata() -> DAGMetadata:
    """
    Return metadata about this DAG.
    
    Returns:
        DAGMetadata with description, complexity, features, etc.
    """
    return DAGMetadata(
        description="...",
        complexity="simple|standard|comprehensive",
        features=["training", "evaluation", ...],
        framework="xgboost|pytorch|lightgbm|...",
        node_count=5,
        edge_count=4
    )
```

### Validation Rules

1. **File must be named** `*_dag.py`
2. **Must have** `create_*_dag()` function
3. **Must have** `get_dag_metadata()` function
4. **Function names must match** file naming pattern
5. **Must return correct types** (PipelineDAG, DAGMetadata)

## Transformation Strategy

### Phase 1: Add New Components (Week 1)

**Goal**: Introduce factory + discovery alongside existing code

**Actions**:
1. Create `core/dag_discovery.py` (DAGAutoDiscovery)
2. Create `core/pipeline_factory.py` (PipelineFactory)
3. Create `core/pipeline_explorer.py` (PipelineExplorer)
4. Create `core/pipeline_knowledge_graph.py` (PipelineKnowledgeGraph)
5. Create `core/pipeline_advisor.py` (PipelineAdvisor)
6. Add comprehensive unit tests
7. Document new interfaces

**Deliverables**:
- All new components implemented
- 100% test coverage
- Documentation complete
- No breaking changes to existing code

### Phase 2: Reorganize shared_dags (Week 2)

**Goal**: Ensure all DAGs follow discovery conventions

**Actions**:
1. Audit all existing DAG modules
2. Rename functions to follow convention
3. Add/fix `get_dag_metadata()` where missing
4. Update `shared_dags/__init__.py` to use discovery
5. Validate all DAGs with convention checker
6. Add source_file references to catalog_index.json

**Validation Script**:
```python
from cursus.pipeline_catalog.core.dag_discovery import DAGAutoDiscovery

def validate_all_dags():
    """Validate all DAGs follow conventions."""
    discovery = DAGAutoDiscovery(...)
    all_dags = discovery.discover_all_dags()
    
    for dag_id, dag_info in all_dags.items():
        # Check has create function
        assert dag_info.create_function is not None, \
            f"{dag_id}: Missing create_*_dag function"
        
        # Check has metadata
        assert dag_info.metadata is not None, \
            f"{dag_id}: Missing get_dag_metadata function"
        
        # Check registry match
        assert discovery._validate_against_registry(dag_id), \
            f"{dag_id}: Not in registry"
    
    print(f"✓ All {len(all_dags)} DAGs validated successfully")
```

### Phase 3: Migrate Consumers (Week 3)

**Goal**: Update all code to use factory

**Migration Guide**:
```python
# BEFORE - Direct class import
from cursus.pipeline_catalog.pipelines.xgb_e2e_comprehensive import (
    XGBoostE2EComprehensivePipeline
)

pipeline = XGBoostE2EComprehensivePipeline(
    config_path="config.json",
    sagemaker_session=session
)

# AFTER - Factory pattern
from cursus.pipeline_catalog import PipelineFactory

pipeline = PipelineFactory.create(
    pipeline_id="xgb_e2e_comprehensive",
    config_path="config.json",
    sagemaker_session=session
)
```

**Update Locations**:
- Jupyter notebooks in `projects/*/`
- Test files in `tests/pipeline_catalog/`
- Example code in documentation
- Internal tools and scripts

### Phase 4: Deprecate Pipeline Classes (Week 4)

**Goal**: Signal deprecation, maintain backward compatibility

**Deprecation Wrapper**:
```python
# pipelines/xgb_e2e_comprehensive.py
import warnings
from ..core.pipeline_factory import PipelineFactory

class XGBoostE2EComprehensivePipeline:
    """
    DEPRECATED: Use PipelineFactory instead.
    
    This class will be removed in version 2.0.
    
    Migration:
        OLD: XGBoostE2EComprehensivePipeline(config_path, ...)
        NEW: PipelineFactory.create("xgb_e2e_comprehensive", config_path, ...)
        
    See: tiny.amazon.com/pipeline-factory-migration
    """
    
    def __new__(cls, *args, **kwargs):
        warnings.warn(
            "XGBoostE2EComprehensivePipeline is deprecated. "
            "Use PipelineFactory.create('xgb_e2e_comprehensive') instead. "
            "This class will be removed in version 2.0.",
            DeprecationWarning,
            stacklevel=2
        )
        return PipelineFactory.create(
            'xgb_e2e_comprehensive',
            *args,
            **kwargs
        )
```

### Phase 5: Remove Pipeline Classes (Week 5)

**Goal**: Complete cleanup

**Actions**:
1. Remove `pipeline_catalog/pipelines/` directory
2. Update all imports (should already be done)
3. Remove deprecation wrappers
4. Update documentation to remove old examples
5. Announce breaking change in release notes

## Usage Examples

### Example 1: Basic Pipeline Creation

```python
from cursus.pipeline_catalog import PipelineFactory

# Create by ID
pipeline = PipelineFactory.create(
    pipeline_id="xgb_e2e_comprehensive",
    config_path="config.json"
)

# Generate and execute
sm_pipeline = pipeline.generate_pipeline()
sm_pipeline.upsert()
execution = sm_pipeline.start()
```

### Example 2: Search-Driven Creation

```python
from cursus.pipeline_catalog import PipelineFactory

factory = PipelineFactory()

# Search first
results = factory.search("xgboost comprehensive training")
for r in results:
    print(f"{r['id']}: {r['title']}")
    print(f"  Features: {', '.join(r['features'])}")
    print(f"  Complexity: {r['complexity']}")

# Create from search
pipeline = factory.create_by_search(
    query="xgboost comprehensive with calibration",
    config_path="config.json"
)
```

### Example 3: Interactive Exploration

```python
from cursus.pipeline_catalog import PipelineExplorer

explorer = PipelineExplorer()

# List by framework
xgb_pipelines = explorer.list_by_framework("xgboost")
print(f"Found {len(xgb_pipelines)} XGBoost pipelines:")
for p in xgb_pipelines:
    print(f"  - {p['id']}: {p['title']}")

# Get detailed info
info = explorer.get_info("xgb_e2e_comprehensive")
print(f"Features: {info.features}")
print(f"Node count: {info.node_count}")
print(f"Similar: {[s['id'] for s in info.similar_pipelines]}")

# Compare pipelines
comparison = explorer.compare([
    "xgb_training_simple",
    "xgb_e2e_comprehensive"
])
print(comparison.summary_table)
```

### Example 4: Workspace Development

```python
# Step 1: Create custom DAG in workspace
# workspace/my_project/dags/custom_financial_dag.py

from cursus.api.dag.base_dag import PipelineDAG
from cursus.pipeline_catalog.shared_dags.base_metadata import DAGMetadata

def create_custom_financial_risk_dag() -> PipelineDAG:
    """Custom XGBoost pipeline for financial risk models."""
    dag = PipelineDAG()
    
    # Custom nodes for financial domain
    dag.add_node("FinancialDataPreparation", ...)
    dag.add_node("XGBoostTraining", ...)
    dag.add_node("RiskCalibration", ...)  # Custom!
    dag.add_node("ComplianceValidation", ...)  # Custom!
    
    dag.add_edge("FinancialDataPreparation", "XGBoostTraining")
    dag.add_edge("XGBoostTraining", "RiskCalibration")
    dag.add_edge("RiskCalibration", "ComplianceValidation")
    
    return dag

def get_dag_metadata() -> DAGMetadata:
    return DAGMetadata(
        description="Financial risk assessment pipeline with compliance validation",
        complexity="advanced",
        features=["training", "risk_calibration", "compliance"],
        framework="xgboost",
        node_count=4,
        edge_count=3
    )

# Step 2: Auto-discover and use
from cursus.pipeline_catalog import PipelineFactory

factory = PipelineFactory(workspace_dir="workspace/my_project")

# List shows custom DAG!
all_pipelines = factory.list_available()
# Includes: "custom_financial_risk"

# Use immediately
pipeline = factory.create(
    pipeline_id="custom_financial_risk",
    config_path="financial_config.json"
)
```

### Example 5: Knowledge Graph Navigation

```python
from cursus.pipeline_catalog import PipelineKnowledgeGraph

kg = PipelineKnowledgeGraph()

# Explore relationships
rel = kg.get_relationships("xgb_e2e_comprehensive")
print(f"Extends: {rel.extends}")
print(f"Extended by: {rel.extended_by}")
print(f"Similar to: {rel.similar_to}")
print(f"Used by: {len(rel.used_by)} projects")

# Get evolution path
path = kg.get_evolution_path("xgb_e2e_comprehensive")
print(f"Evolution: {' → '.join(path.stages)}")
# Output: simple → calibrated → evaluation → comprehensive

# Visualize ecosystem
viz = kg.visualize(framework="xgboost")
viz.show()  # Interactive graph
viz.export("xgboost_ecosystem.html")
```

### Example 6: Requirements to Pipeline

```python
from cursus.pipeline_catalog import PipelineAdvisor

advisor = PipelineAdvisor()

# Get recommendations
recommendations = advisor.recommend(
    use_case="I need to train an XGBoost model for production with calibration",
    requirements=["model_registration", "calibration", "evaluation"]
)

for i, rec in enumerate(recommendations[:3], 1):
    print(f"{i}. {rec.pipeline_id} (score: {rec.score:.0%})")
    print(f"   {rec.explanation}")
    if rec.missing_features:
        print(f"   Missing: {', '.join(rec.missing_features)}")
    print()

# Analyze gap
gap = advisor.analyze_gap(
    selected="xgb_training_simple",
    requirements=["calibration", "registration", "evaluation"]
)
print(f"Missing features: {gap.missing_features}")
print(f"Suggested upgrade: {gap.upgrade_path.recommended}")
print(f"Why: {gap.upgrade_path.rationale}")
```

## Testing Strategy

### Unit Tests

```python
# test_dag_discovery.py
def test_discover_all_dags():
    """Test discovering all package DAGs."""
    discovery = DAGAutoDiscovery(package_root=get_package_root())
    dags = discovery.discover_all_dags()
    
    assert len(dags) >= 7  # At least 7 existing DAGs
    assert "xgb_e2e_comprehensive" in dags
    assert all(d.create_function is not None for d in dags.values())

def test_workspace_priority():
    """Test workspace DAGs override package DAGs."""
    discovery = DAGAutoDiscovery(
        package_root=get_package_root(),
        workspace_dirs=[get_test_workspace()]
    )
    
    dag_info = discovery.load_dag_info("custom_test_dag")
    assert dag_info.workspace_id == "test_workspace"

# test_pipeline_factory.py
def test_create_by_id():
    """Test creating pipeline by ID."""
    pipeline = PipelineFactory.create(
        pipeline_id="xgb_e2e_comprehensive",
        config_path="test_config.json"
    )
    
    assert pipeline is not None
    assert hasattr(pipeline, 'create_dag')
    assert hasattr(pipeline, 'get_enhanced_dag_metadata')

def test_search():
    """Test natural language search."""
    results = PipelineFactory.search("xgboost comprehensive")
    
    assert len(results) > 0
    assert any(r['id'] == 'xgb_e2e_comprehensive' for r in results)

# test_pipeline_explorer.py
def test_list_by_framework():
    """Test listing by framework."""
    explorer = PipelineExplorer()
    xgb_pipelines = explorer.list_by_framework("xgboost")
    
    assert len(xgb_pipelines) >= 4
    assert all(p['framework'] == 'xgboost' for p in xgb_pipelines)
```

### Integration Tests

```python
def test_end_to_end_creation():
    """Test complete pipeline creation and execution."""
    pipeline = PipelineFactory.create(
        pipeline_id="dummy_e2e_basic",
        config_path="test_config.json"
    )
    
    sm_pipeline = pipeline.generate_pipeline()
    assert sm_pipeline is not None
    assert sm_pipeline.name is not None

def test_workspace_dag_discovery():
    """Test workspace DAG discovery and usage."""
    factory = PipelineFactory(workspace_dir="test_workspace")
    
    all_pipelines = factory.list_available()
    assert "custom_test_dag" in [p['id'] for p in all_pipelines]
    
    pipeline = factory.create(
        pipeline_id="custom_test_dag",
        config_path="test_config.json"
    )
    assert pipeline is not None
```

## Performance Considerations

### Caching Strategy

1. **DAG Function Caching**: Functions loaded once, cached for reuse
2. **Discovery Results**: Complete discovery cached after first run
3. **Pipeline Class Caching**: Dynamic classes cached by ID
4. **Metadata Caching**: Registry metadata cached in memory

### Lazy Loading

- Discovery triggered only when needed
- DAG functions loaded on-demand
- Workspace scanning deferred until access

### Benchmark Targets

| Operation | Target Time | Notes |
|-----------|-------------|-------|
| First discovery | < 500ms | Full package + workspace scan |
| Cached discovery | < 10ms | From cached results |
| Pipeline creation (first) | < 100ms | Includes DAG import |
| Pipeline creation (cached) | < 10ms | Using cached function |
| Search query | < 50ms | Text search across metadata |
| List all pipelines | < 20ms | From cached discovery |

## Documentation Plan

### User Documentation

1. **Quick Start Guide**
   - Basic pipeline creation
   - Search-driven workflow
   - Common use cases

2. **User Guide**
   - All creation methods
   - Exploration tools
   - Knowledge graph navigation
   - Workspace development

3. **API Reference**
   - Complete API documentation
   - All public methods
   - Parameter descriptions
   - Return type specifications

4. **Migration Guide**
   - Old → new mapping
   - Breaking changes
   - Compatibility notes
   - Troubleshooting

### Developer Documentation

1. **Architecture Guide**
   - Component overview
   - Integration points
   - Extension points

2. **DAG Convention Guide**
   - Naming requirements
   - Function signatures
   - Metadata specification
   - Validation rules

3. **Contributing Guide**
   - Adding new DAGs
   - Testing requirements
   - Documentation standards

## Benefits Summary

### For Users

| Benefit | Before | After | Improvement |
|---------|--------|-------|-------------|
| **Discovery** | Know exact class name | Natural language search | 10x easier |
| **Learning Curve** | 15+ classes to learn | 1 factory interface | 93% reduction |
| **Exploration** | Read source code | Interactive tools | Qualitative leap |
| **Local Dev** | Complex setup | Drop file in workspace | Zero friction |
| **Selection Confidence** | Trial and error | Guided recommendations | Higher success rate |

### For System

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Code Volume** | ~500 lines (7 files) | ~310 lines (1 file) | 62% reduction |
| **Metadata Duplication** | 3 locations | 1 location (registry) | 67% reduction |
| **Maintenance Overhead** | 4 steps to add pipeline | 3 steps | 25% reduction |
| **Extensibility** | Limited | High (workspace support) | Qualitative leap |
| **Knowledge Leverage** | Low (unused metadata) | High (full utilization) | Maximum value |

## Conclusion

This redesign transforms the `pipeline_catalog` from a manual, class-based system into an intelligent knowledge system that:

1. **Eliminates Redundancy**: Single factory replaces 7 wrapper classes
2. **Enables Discovery**: Natural language + structured search
3. **Leverages Knowledge**: Full utilization of Zettelkasten metadata
4. **Supports Development**: Workspace DAGs with auto-discovery
5. **Guides Users**: Intelligent recommendations and navigation

**Core Achievement**: Transform "you must know the class name" into "describe what you need."

**Next Steps**:
1. Implement Phase 1 (new components)
2. Validate with existing DAGs
3. Migrate consumers
4. Complete transformation

This design represents a fundamental architectural improvement that enhances both user experience and system maintainability while preserving all existing functionality through the `BasePipeline` interface.

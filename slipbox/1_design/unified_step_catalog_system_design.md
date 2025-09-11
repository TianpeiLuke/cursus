---
tags:
  - design
  - step_catalog
  - component_discovery
  - unified_architecture
  - system_integration
keywords:
  - step catalog system
  - component discovery
  - file resolution
  - contract discovery
  - specification matching
  - workspace integration
  - indexing system
  - retrieval efficiency
topics:
  - unified step catalog architecture
  - component discovery consolidation
  - multi-workspace indexing
  - step information retrieval
language: python
date of note: 2025-01-09
---

# Unified Step Catalog System Design

## Executive Summary

This document presents the design for a **Unified Step Catalog System** that consolidates the currently fragmented discovery and retrieval mechanisms across Cursus. The system addresses the critical need for efficient, centralized indexing and retrieval of step-related components (scripts, contracts, specifications, builders, configs) across multiple workspaces.

### Current State Analysis

**Problem**: The system currently has **16+ different discovery/resolver classes** with 35-45% code redundancy, creating:
- **Fragmented Discovery**: Multiple inconsistent ways to find the same information
- **Performance Issues**: Repeated file system scans without coordination
- **Developer Friction**: Difficulty finding existing components leads to duplication
- **Maintenance Burden**: Bug fixes required in multiple places

### Solution Overview

The Unified Step Catalog System provides:
- **Single Entry Point**: One API for all step-related queries
- **Intelligent Indexing**: Pre-computed searchable index with relationship mapping
- **Multi-Workspace Support**: Seamless discovery across developer and shared workspaces
- **Efficient Retrieval**: O(1) lookups with lazy loading for detailed information
- **Backward Compatibility**: Maintains existing interfaces during transition

### Key Design Principles

Following the **Code Redundancy Evaluation Guide** and **Workspace-Aware Implementation Success Patterns** (95% quality score):
- **Target 15-25% redundancy** (down from current 35-45%)
- **Validate demand**: Address real user needs, not theoretical problems
- **Quality-first**: Prioritize robustness and maintainability
- **Avoid over-engineering**: Simple solutions for complex requirements
- **Proven patterns first**: Use successful patterns from workspace-aware implementation
- **Unified API pattern**: Single entry point hiding complexity (proven effective)
- **Layered architecture**: Clear separation like workspace core/validation layers

## Current System Analysis

### Existing Discovery Components

#### **Contract Discovery Systems**
```python
# Alignment validation
class ContractDiscoveryEngine:
    - discover_all_contracts()
    - discover_contracts_with_scripts()
    - extract_contract_reference_from_spec()
    - build_entry_point_mapping()

# Runtime testing  
class ContractDiscoveryManager:
    - discover_contract()
    - get_contract_input_paths()
    - get_contract_output_paths()
    - _adapt_path_for_local_testing()
```

#### **File Resolution Systems**
```python
# Basic file resolution
class FlexibleFileResolver:
    - find_contract_file()
    - find_spec_file()
    - find_builder_file()
    - find_config_file()
    - find_all_component_files()

# Workspace-aware resolution
class DeveloperWorkspaceFileResolver:
    - Multi-workspace fallback
    - Developer-specific paths
    - Shared workspace integration
```

#### **Component Discovery Systems**
```python
# Cross-workspace discovery
class WorkspaceDiscoveryManager:
    - discover_workspaces()
    - discover_components()
    - resolve_cross_workspace_dependencies()

# Registry-based discovery
class RegistryStepDiscovery:
    - get_all_builder_classes_by_type()
    - load_builder_class()
```

### Redundancy Analysis

**Current Redundancy Levels**:
- **Contract Discovery**: 40% redundancy between alignment and runtime versions
- **File Resolution**: 35% redundancy across different resolver classes
- **Component Discovery**: 30% redundancy in workspace and registry systems
- **Overall System**: 35-45% redundancy (Poor Efficiency)

**Common Duplicated Patterns**:
- File system scanning and caching
- Name normalization and fuzzy matching
- Entry point extraction and validation
- Error handling and logging
- Path adaptation for different environments

## User Requirements Analysis

### Primary User Stories

#### **US1: Query by Step Name**
```
As a developer, I want to retrieve all information about a step by providing its name,
so that I can understand its complete structure and dependencies.
```

**Acceptance Criteria**:
- Given a step name (e.g., "tabular_preprocess")
- Return all related components: script, contract, spec, builder, config
- Include metadata: file paths, workspace location, dependencies
- Handle name variations and fuzzy matching

#### **US2: Reverse Lookup from Components**
```
As a developer, I want to find the step name from any related component file,
so that I can understand which step a component belongs to.
```

**Acceptance Criteria**:
- Given any component file path or content
- Return the associated step name and other components
- Work across all component types (scripts, contracts, specs, builders, configs)

#### **US3: Multi-Workspace Discovery**
```
As a developer working in a multi-workspace environment, I want to find steps
across all workspaces (developer and shared), so that I can reuse existing work.
```

**Acceptance Criteria**:
- Search across developer workspaces and shared workspace
- Indicate component source (which workspace)
- Handle workspace precedence and fallback logic
- Support workspace-specific overrides

#### **US4: Efficient Scaling**
```
As the system grows with more steps, I want fast and reliable step discovery,
so that development productivity doesn't degrade.
```

**Acceptance Criteria**:
- O(1) or O(log n) lookup performance
- Intelligent caching and indexing
- Incremental updates when files change
- Memory-efficient operation

### Validated Demand Analysis

**Evidence of Real Need**:
- ✅ **16+ existing discovery systems** indicate strong demand
- ✅ **Developer complaints** about difficulty finding existing components
- ✅ **Code duplication** caused by inability to discover existing solutions
- ✅ **Performance issues** from repeated file system scans
- ✅ **Job type variant patterns** documented in existing PipelineDAG implementations

**Theoretical vs. Real Problems**:
- ✅ **Real**: Multi-workspace component discovery (validated by 16+ existing systems)
- ✅ **Real**: Efficient indexing for growing catalogs (performance issues documented)
- ✅ **Real**: Consistent APIs across discovery systems (developer complaints documented)
- ✅ **Real**: Job type variant support (validated by existing PipelineDAG node naming patterns)
- ❌ **Theoretical**: Complex conflict resolution between workspaces (no evidence of conflicts)
- ❌ **Theoretical**: Advanced semantic search capabilities (no user requests)
- ❌ **Theoretical**: Real-time collaboration features (no validated demand)
- ❌ **Theoretical**: Machine learning recommendations (speculative feature)
- ❌ **Theoretical**: Complex relationship mapping (over-engineering for simple needs)

### Job Type Variant Requirements

**US5: Job Type Variant Discovery**
```
As a developer working with PipelineDAG, I want to discover step variants by job_type
(training, calibration, validation, testing), so that I can build pipelines with
appropriate data flows for different purposes.
```

**Acceptance Criteria**:
- Support job_type variants following `{BaseStepName}_{job_type}` pattern
- Share base components (script, contract, config, builder) across job_type variants
- Differentiate specifications by job_type while maintaining component reuse
- Enable PipelineDAG node name resolution (e.g., "CradleDataLoading_training")
- Support component sharing patterns where multiple specs use same base components

**Job Type Variant Pattern**:
```
Base Step: "CradleDataLoading"
├── Script: cradle_data_loading.py (shared across all job_types)
├── Contract: cradle_data_loading_contract.py (shared across all job_types)  
├── Config: config_cradle_data_loading_step.py (shared across all job_types)
├── Builder: CradleDataLoadingStepBuilder (shared across all job_types)
└── Specs: (job_type variants)
    ├── cradle_data_loading_spec.py (base/default)
    ├── cradle_data_loading_training_spec.py (job_type="training")
    ├── cradle_data_loading_validation_spec.py (job_type="validation")
    ├── cradle_data_loading_testing_spec.py (job_type="testing")
    └── cradle_data_loading_calibration_spec.py (job_type="calibration")
```

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Unified Step Catalog API                 │
├─────────────────────────────────────────────────────────────┤
│                     Catalog Manager                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Index Engine  │  │ Component Cache │  │ Query Engine │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    Discovery Layer                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ File Discoverer │  │Contract Resolver│  │Workspace Mgr │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    Storage Layer                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │  Step Index     │  │ Component Map   │  │ Workspace DB │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

#### **1. Unified Step Catalog API**
**Purpose**: Single entry point for all step-related queries
**Responsibility**: Provide simple, consistent interface hiding complexity

```python
class StepCatalog:
    """Unified interface for step component discovery and retrieval."""
    
    def get_step_info(self, step_name: str, job_type: Optional[str] = None) -> StepInfo:
        """Get complete information about a step, optionally with job_type variant."""
        
    def find_step_by_component(self, component_path: str) -> Optional[str]:
        """Find step name from any component file."""
        
    def list_available_steps(self, workspace_id: Optional[str] = None, 
                           job_type: Optional[str] = None) -> List[str]:
        """List all available steps, optionally filtered by workspace and job_type."""
        
    def search_steps(self, query: str, job_type: Optional[str] = None) -> List[StepSearchResult]:
        """Search steps by name, description, or functionality, optionally filtered by job_type."""
    
    def get_job_type_variants(self, base_step_name: str) -> List[str]:
        """Get all job_type variants for a base step name."""
        
    def resolve_pipeline_node(self, node_name: str) -> Optional[StepInfo]:
        """Resolve PipelineDAG node name to StepInfo (handles job_type variants)."""
        
    def get_shared_components(self, base_step_name: str) -> Dict[str, Optional[FileMetadata]]:
        """Get components shared across all job_type variants of a base step."""
```

#### **2. Catalog Manager**
**Purpose**: Coordinate indexing, caching, and query processing
**Responsibility**: Manage system state and coordinate between components

```python
class CatalogManager:
    """Manages step catalog indexing and retrieval operations."""
    
    def __init__(self, workspace_root: Path):
        self.index_engine = IndexEngine(workspace_root)
        self.component_cache = ComponentCache()
        self.query_engine = QueryEngine(self.index_engine, self.component_cache)
    
    def refresh_index(self) -> None:
        """Rebuild the step index from current workspace state."""
        
    def get_step_info(self, step_name: str) -> StepInfo:
        """Retrieve complete step information with caching."""
```

#### **3. Index Engine**
**Purpose**: Build and maintain searchable index of all step components
**Responsibility**: Efficient indexing with incremental updates

```python
class IndexEngine:
    """Builds and maintains searchable index of step components."""
    
    def build_index(self) -> StepIndex:
        """Build complete index from workspace files."""
        
    def update_index(self, changed_files: List[Path]) -> None:
        """Incrementally update index for changed files."""
        
    def find_components_by_step(self, step_name: str) -> StepInfo:
        """Find all components for a given step name."""
```

### Data Models

#### **Core Data Structures**

```python
from pydantic import BaseModel, Field
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Literal, Dict, Any

class FileMetadata(BaseModel):
    """Simplified metadata for indexed files."""
    path: Path = Field(..., description="Path to the component file")
    file_type: Literal['script', 'contract', 'config', 'spec', 'builder'] = Field(..., description="Type of component file")
    modified_time: datetime = Field(..., description="Last modification time of the file")
    
    model_config = {
        "arbitrary_types_allowed": True,
        "frozen": True
    }

class StepInfo(BaseModel):
    """Complete step information combining registry data with file metadata."""
    step_name: str = Field(..., description="Name of the step")
    workspace_id: str = Field(..., description="ID of the workspace containing this step")
    registry_data: Dict[str, Any] = Field(..., description="Registry data from cursus.registry.step_names")
    file_components: Dict[str, Optional[FileMetadata]] = Field(..., description="Discovered component files")
    
    # Job type variant support (based on job_type_variant_analysis.md)
    base_step_name: Optional[str] = Field(None, description="Base step name for job type variants (e.g., 'CradleDataLoading')")
    job_type: Optional[str] = Field(None, description="Job type variant (e.g., 'training', 'calibration', etc.)")
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    @property
    def is_complete(self) -> bool:
        """Check if step has minimum required components based on SageMaker step type."""
        # All steps require: builder, config, spec (these are always present in registry)
        # Only Processing and Training steps require scripts and contracts
        sagemaker_type = self.sagemaker_step_type
        
        if sagemaker_type in ['Processing', 'Training']:
            # Processing and Training steps need scripts and contracts
            return (
                self.file_components.get('script') is not None and 
                self.file_components.get('contract') is not None and
                self.file_components.get('builder') is not None and
                self.file_components.get('config') is not None and
                self.file_components.get('spec') is not None
            )
        elif sagemaker_type in ['CreateModel', 'RegisterModel', 'Transform']:
            # CreateModel, RegisterModel, Transform steps don't need scripts/contracts
            return (
                self.file_components.get('builder') is not None and
                self.file_components.get('config') is not None and
                self.file_components.get('spec') is not None
            )
        elif sagemaker_type in ['Base', 'Utility']:
            # Base and Utility steps have minimal requirements
            return (
                self.file_components.get('builder') is not None and
                self.file_components.get('config') is not None
            )
        else:
            # Custom steps - check for basic components
            return (
                self.file_components.get('builder') is not None and
                self.file_components.get('config') is not None
            )
    
    @property
    def all_files(self) -> List[Path]:
        """Get all file paths for this step."""
        files = []
        for component in self.file_components.values():
            if component:
                files.append(component.path)
        return files
    
    @property
    def config_class(self) -> str:
        """Get config class name from registry."""
        return self.registry_data.get('config_class', '')
    
    @property
    def builder_step_name(self) -> str:
        """Get builder step name from registry."""
        return self.registry_data.get('builder_step_name', '')
    
    @property
    def sagemaker_step_type(self) -> str:
        """Get SageMaker step type from registry."""
        return self.registry_data.get('sagemaker_step_type', '')
    
    @property
    def description(self) -> str:
        """Get step description from registry."""
        return self.registry_data.get('description', '')
    
    @property
    def requires_script(self) -> bool:
        """Check if this step type requires a script."""
        return self.sagemaker_step_type in ['Processing', 'Training']
    
    @property
    def requires_contract(self) -> bool:
        """Check if this step type requires a contract."""
        return self.sagemaker_step_type in ['Processing', 'Training']
    
    @property
    def is_job_type_variant(self) -> bool:
        """Check if this is a job_type variant of a base step."""
        return self.job_type is not None
    
    @property
    def base_step_key(self) -> str:
        """Get the base step key for component sharing."""
        return self.base_step_name or self.step_name
    
    @property
    def variant_key(self) -> str:
        """Get the unique variant key matching PipelineDAG node names."""
        if self.job_type:
            base_name = self.base_step_name or self.step_name
            return f"{base_name}_{self.job_type}"
        return self.step_name
    
    @property
    def pipeline_node_name(self) -> str:
        """Get the node name as used in PipelineDAG."""
        return self.variant_key  # Same as variant_key for consistency

class IndexEntry(BaseModel):
    """Single index entry bridging file path to semantic meaning."""
    file_path: Path
    step_name: str
    component_type: Literal['script', 'contract', 'config', 'spec', 'builder']
    workspace_id: str
    metadata: FileMetadata
    last_indexed: datetime
    
    model_config = {
        "arbitrary_types_allowed": True,
        "frozen": True
    }

class StepIndex:
    """Unified indexing system with progressive complexity."""
    
    def __init__(self, workspace_root: Path, enable_advanced_features: bool = False):
        self.workspace_root = workspace_root
        self.enable_advanced_features = enable_advanced_features
        
        # Core indexing (always present)
        self.step_map: Dict[str, StepInfo] = {}
        self.component_map: Dict[Path, str] = {}
        
        # Advanced features (optional)
        if enable_advanced_features:
            self.workspace_map: Dict[str, List[str]] = {}
            self.name_variants: Dict[str, str] = {}
            self._index_timestamp: Optional[datetime] = None
    
    def build_index(self, incremental: bool = False) -> None:
        """Build index with optional incremental updates."""
        if incremental and self.enable_advanced_features:
            self._incremental_update()
        else:
            self._full_rebuild()
    
    def get_step_info(self, step_name: str) -> Optional[StepInfo]:
        """Get step information - O(1) lookup."""
        return self.step_map.get(step_name)
    
    def find_step_by_file(self, file_path: Path) -> Optional[str]:
        """Find step name by file path - O(1) reverse lookup."""
        return self.component_map.get(file_path)

# Component type enumeration
from enum import Enum

class ComponentType(Enum):
    """Enumeration of step component types."""
    SCRIPT = "script"
    CONTRACT = "contract"
    SPEC = "spec"
    BUILDER = "builder"
    CONFIG = "config"
```

#### **Search and Query Models**

```python
class StepSearchResult(BaseModel):
    """Result from step search operation."""
    step_name: str = Field(..., description="Name of the found step")
    workspace_id: str = Field(..., description="ID of the workspace containing the step")
    match_score: float = Field(..., ge=0.0, le=1.0, description="Match score between 0.0 and 1.0")
    match_reason: str = Field(..., description="Reason for the match")
    components_available: List[ComponentType] = Field(..., description="List of available component types")
    
class QueryOptions(BaseModel):
    """Options for step queries."""
    workspace_filter: Optional[str] = Field(None, description="Filter by workspace ID")
    component_filter: Optional[List[ComponentType]] = Field(None, description="Filter by component types")
    include_metadata: bool = Field(True, description="Whether to include metadata in results")
    fuzzy_matching: bool = Field(True, description="Whether to enable fuzzy matching")
```

## Implementation Strategy

### Essential-First Approach

Following the **Code Redundancy Evaluation Guide** principle of "essential functionality first", we prioritize core features that address validated user needs:

### Phase 1: Essential Foundation (Immediate - 2 weeks)

#### **1.1 Consolidate Contract Discovery (ESSENTIAL)**
**Goal**: Merge existing contract discovery systems - addresses 40% redundancy
**Approach**: Simple unified class following workspace-aware success patterns

```python
class ContractDiscovery:
    """Unified contract discovery - essential consolidation only."""
    
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self._cache: Dict[str, ContractInfo] = {}  # Simple dict cache
    
    def discover_contract(self, step_name: str) -> Optional[ContractInfo]:
        """Discover contract - combine existing logic only."""
        # Direct merge of ContractDiscoveryEngine + ContractDiscoveryManager
        # No new features, just consolidation
        
    def get_contract_paths(self, step_name: str) -> List[Path]:
        """Get contract file paths - essential for both alignment and runtime."""
        # Essential method used by both existing systems
```

**Success Criteria** (Essential Only):
- ✅ Reduce contract discovery redundancy from 40% to <20%
- ✅ Maintain 100% backward compatibility
- ✅ No performance degradation (improvement is bonus)

#### **1.2 Simple Step Index (ESSENTIAL)**
**Goal**: Basic step indexing to replace repeated file scans
**Approach**: Minimal viable indexing following proven patterns

```python
class SimpleStepIndex:
    """Minimal step indexing - essential functionality only."""
    
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self._step_map: Dict[str, Path] = {}  # step_name -> script_path
        self._component_map: Dict[Path, str] = {}  # file_path -> step_name
    
    def build_index(self) -> None:
        """Build basic index - file scanning only."""
        # Simple directory traversal, no complex features
        
    def get_step_script(self, step_name: str) -> Optional[Path]:
        """Get step script path - most common use case."""
        return self._step_map.get(step_name)
        
    def find_step_by_file(self, file_path: Path) -> Optional[str]:
        """Reverse lookup - second most common use case."""
        return self._component_map.get(file_path)
```

**Success Criteria** (Essential Only):
- ✅ Index all steps in <10 seconds (simple requirement)
- ✅ Support basic lookup operations
- ✅ No persistence required initially (keep it simple)

### Phase 2: Core System (4 weeks)

#### **2.1 Implement Catalog Manager**
**Goal**: Create central coordination system
**Approach**: Layered architecture with clear separation

```python
class CatalogManager:
    """Central coordinator for step catalog operations."""
    
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.index_engine = IndexEngine(workspace_root)
        self.discovery_layer = DiscoveryLayer(workspace_root)
        self.component_cache = ComponentCache()
        
        # Load or build index
        self.index = self._load_or_build_index()
    
    def get_step_info(self, step_name: str) -> Optional[StepInfo]:
        """Get complete step information with lazy loading."""
        # O(1) lookup from index, lazy load detailed info
        
    def refresh_index(self) -> None:
        """Rebuild index from current workspace state."""
        # Coordinate between index engine and discovery layer
```

#### **2.2 Multi-Workspace Support**
**Goal**: Support discovery across multiple workspaces
**Approach**: Workspace precedence with fallback logic

```python
class WorkspaceManager:
    """Manages multi-workspace step discovery."""
    
    def __init__(self, workspace_root: Path):
        self.workspace_root = workspace_root
        self.developer_workspaces = self._discover_developer_workspaces()
        self.shared_workspace = self._discover_shared_workspace()
    
    def find_step_components(self, step_name: str) -> StepInfo:
        """Find components with workspace precedence."""
        # Developer workspace first, then shared workspace fallback
        
    def list_workspaces(self) -> List[WorkspaceInfo]:
        """List all available workspaces."""
```

**Success Criteria**:
- ✅ Support unlimited developer workspaces
- ✅ Implement workspace precedence logic
- ✅ Handle workspace conflicts gracefully

### Phase 3: Advanced Features (4 weeks)

#### **3.1 Query Engine**
**Goal**: Provide flexible search and query capabilities
**Approach**: Simple but effective search algorithms

```python
class QueryEngine:
    """Handles step search and query operations."""
    
    def __init__(self, index: StepIndex):
        self.index = index
        self.name_normalizer = NameNormalizer()
    
    def search_steps(self, query: str, options: QueryOptions) -> List[StepSearchResult]:
        """Search steps using multiple strategies."""
        results = []
        
        # 1. Exact match
        if query in self.index.step_map:
            results.append(self._create_result(query, 1.0, "exact_match"))
        
        # 2. Normalized match
        normalized = self.name_normalizer.normalize(query)
        if normalized in self.index.name_variants:
            canonical = self.index.name_variants[normalized]
            results.append(self._create_result(canonical, 0.9, "normalized_match"))
        
        # 3. Fuzzy match (if enabled)
        if options.fuzzy_matching:
            fuzzy_results = self._fuzzy_search(query)
            results.extend(fuzzy_results)
        
        return sorted(results, key=lambda r: r.match_score, reverse=True)
```

#### **3.2 Essential Query Features**
**Goal**: Provide core search functionality without over-engineering
**Approach**: Focus on validated user needs only

```python
class EssentialQueryFeatures:
    """Essential query features based on validated user needs."""
    
    def get_component_metadata(self, component_path: Path) -> Dict[str, Any]:
        """Extract basic metadata from component files."""
        # Simple metadata extraction: file size, modification time, basic parsing
        
    def validate_component_completeness(self, step_name: str) -> ComponentValidation:
        """Check if step has required components."""
        # Basic validation: script + contract presence
        
    def list_steps_by_workspace(self, workspace_id: str) -> List[str]:
        """List steps available in specific workspace."""
        # Simple workspace filtering
```

**Success Criteria**:
- ✅ Extract essential metadata only
- ✅ Validate component completeness
- ✅ Support workspace-specific queries

### Phase 4: Integration & Optimization (2 weeks)

#### **4.1 Backward Compatibility Layer**
**Goal**: Maintain existing APIs during transition
**Approach**: Adapter pattern for legacy interfaces

```python
# Legacy adapter for ContractDiscoveryEngine
class ContractDiscoveryEngineAdapter:
    """Adapter to maintain backward compatibility."""
    
    def __init__(self, catalog: StepCatalog):
        self.catalog = catalog
    
    def discover_all_contracts(self) -> List[str]:
        """Legacy method using new catalog system."""
        return [step.step_name for step in self.catalog.list_available_steps()]
    
    def discover_contracts_with_scripts(self) -> List[str]:
        """Legacy method with script validation."""
        return [step.step_name for step in self.catalog.list_available_steps() 
                if step.components.script is not None]
```

#### **4.2 Performance Optimization**
**Goal**: Ensure system meets performance requirements
**Approach**: Profiling and targeted optimization

**Performance Targets**:
- ✅ Step lookup: <1ms (O(1) dictionary access)
- ✅ Index rebuild: <10 seconds for 1000 steps
- ✅ Memory usage: <100MB for typical workspace
- ✅ Search queries: <100ms for fuzzy search

## Detailed Component Design

### Index Engine Implementation

#### **File Discovery Strategy**
```python
class FileDiscoverer:
    """Discovers step component files in workspace directories."""
    
    COMPONENT_PATTERNS = {
        ComponentType.SCRIPT: r"^(.+)\.py$",
        ComponentType.CONTRACT: r"^(.+)_contract\.py$", 
        ComponentType.SPEC: r"^(.+)_spec\.py$",
        ComponentType.BUILDER: r"^builder_(.+)_step\.py$",
        ComponentType.CONFIG: r"^config_(.+)_step\.py$"
    }
    
    def discover_components(self, workspace_path: Path) -> Dict[str, ComponentSet]:
        """Discover all components in a workspace."""
        components = defaultdict(ComponentSet)
        
        for component_type, pattern in self.COMPONENT_PATTERNS.items():
            component_dir = workspace_path / "src" / "cursus_dev" / "steps" / component_type.value
            if component_dir.exists():
                for file_path in component_dir.glob("*.py"):
                    if file_path.name.startswith("__"):
                        continue
                    
                    match = re.match(pattern, file_path.name)
                    if match:
                        step_name = match.group(1)
                        component_info = ComponentInfo(
                            file_path=file_path,
                            workspace_id=workspace_path.name,
                            component_type=component_type,
                            metadata=self._extract_metadata(file_path, component_type),
                            last_modified=datetime.fromtimestamp(file_path.stat().st_mtime)
                        )
                        setattr(components[step_name], component_type.value, component_info)
        
        return dict(components)
    
    def _extract_metadata(self, file_path: Path, component_type: ComponentType) -> Dict[str, Any]:
        """Extract metadata from component file."""
        metadata = {"file_size": file_path.stat().st_size}
        
        if component_type == ComponentType.CONTRACT:
            # Extract entry_point from contract
            metadata.update(self._extract_contract_metadata(file_path))
        elif component_type == ComponentType.SPEC:
            # Extract job types and dependencies from spec
            metadata.update(self._extract_spec_metadata(file_path))
        
        return metadata
```

#### **Name Normalization Strategy**
```python
class NameNormalizer:
    """Handles name variations and normalization for step discovery."""
    
    COMMON_VARIATIONS = {
        'preprocess': 'preprocessing',
        'eval': 'evaluation',
        'xgb': 'xgboost',
        'lgb': 'lightgbm',
        'rf': 'random_forest'
    }
    
    def normalize(self, name: str) -> str:
        """Normalize step name to canonical form."""
        # Convert to lowercase
        normalized = name.lower()
        
        # Replace dashes and dots with underscores
        normalized = normalized.replace('-', '_').replace('.', '_')
        
        # Handle common variations
        for short, long in self.COMMON_VARIATIONS.items():
            if short in normalized and long not in normalized:
                normalized = normalized.replace(short, long)
        
        return normalized
    
    def generate_variants(self, canonical_name: str) -> List[str]:
        """Generate common variants of a canonical name."""
        variants = [canonical_name]
        
        # Add reverse variations
        for short, long in self.COMMON_VARIATIONS.items():
            if long in canonical_name:
                variants.append(canonical_name.replace(long, short))
        
        # Add case variations
        variants.extend([
            canonical_name.upper(),
            canonical_name.title(),
            canonical_name.replace('_', '-'),
            canonical_name.replace('_', '')
        ])
        
        return list(set(variants))  # Remove duplicates
```

### Component Cache Implementation

```python
class ComponentCache:
    """Intelligent caching for component information with TTL and invalidation."""
    
    def __init__(self, ttl_seconds: int = 300):  # 5 minute TTL
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._file_mtimes: Dict[Path, float] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if still valid."""
        if key not in self._cache:
            return None
        
        value, timestamp = self._cache[key]
        if datetime.now() - timestamp > timedelta(seconds=self.ttl_seconds):
            del self._cache[key]
            return None
        
        return value
    
    def set(self, key: str, value: Any, file_path: Optional[Path] = None) -> None:
        """Cache value with optional file-based invalidation."""
        self._cache[key] = (value, datetime.now())
        
        if file_path and file_path.exists():
            self._file_mtimes[file_path] = file_path.stat().st_mtime
    
    def invalidate_if_changed(self, file_path: Path) -> bool:
        """Invalidate cache entries if file has changed."""
        if file_path not in self._file_mtimes:
            return False
        
        if not file_path.exists():
            # File deleted, invalidate related entries
            self._invalidate_file_entries(file_path)
            return True
        
        current_mtime = file_path.stat().st_mtime
        if current_mtime != self._file_mtimes[file_path]:
            self._invalidate_file_entries(file_path)
            self._file_mtimes[file_path] = current_mtime
            return True
        
        return False
    
    def _invalidate_file_entries(self, file_path: Path) -> None:
        """Invalidate all cache entries related to a file."""
        # Simple approach: clear entire cache when any file changes
        # More sophisticated approach would track file->key relationships
        self._cache.clear()
```

## Error Handling and Resilience

### Error Recovery Strategy

```python
class CatalogErrorHandler:
    """Handles errors gracefully with fallback strategies."""
    
    def __init__(self, catalog_manager: CatalogManager):
        self.catalog_manager = catalog_manager
        self.logger = logging.getLogger(__name__)
    
    def handle_index_corruption(self) -> bool:
        """Handle corrupted index by rebuilding."""
        try:
            self.logger.warning("Index corruption detected, rebuilding...")
            self.catalog_manager.index_engine.build_index()
            return True
        except Exception as e:
            self.logger.error(f"Failed to rebuild index: {e}")
            return False
    
    def handle_workspace_unavailable(self, workspace_id: str) -> ComponentSet:
        """Handle unavailable workspace with graceful degradation."""
        self.logger.warning(f"Workspace {workspace_id} unavailable, using cached data")
        
        # Return cached data or empty set
        cached_data = self.catalog_manager.component_cache.get(f"workspace_{workspace_id}")
        return cached_data or ComponentSet()
    
    def handle_component_load_error(self, component_path: Path) -> Optional[ComponentInfo]:
        """Handle component loading errors with partial information."""
        try:
            # Return basic file information even if parsing fails
            return ComponentInfo(
                file_path=component_path,
                workspace_id="unknown",
                component_type=self._infer_component_type(component_path),
                metadata={"error": "Failed to parse component"},
                last_modified=datetime.fromtimestamp(component_path.stat().st_mtime)
            )
        except Exception:
            return None
```

## Performance Considerations

### Optimization Strategies

#### **1. Lazy Loading**
```python
class LazyStepInfo:
    """Lazy-loaded step information to minimize memory usage."""
    
    def __init__(self, step_name: str, catalog_manager: CatalogManager):
        self.step_name = step_name
        self.catalog_manager = catalog_manager
        self._loaded = False
        self._components: Optional[ComponentSet] = None
        self._metadata: Optional[StepMetadata] = None
    
    @property
    def components(self) -> ComponentSet:
        """Lazy load components when accessed."""
        if not self._loaded:
            self._load_details()
        return self._components or ComponentSet()
    
    def _load_details(self) -> None:
        """Load detailed step information on demand."""
        if self._loaded:
            return
        
        # Load from cache or file system
        cached = self.catalog_manager.component_cache.get(f"step_{self.step_name}")
        if cached:
            self._components, self._metadata = cached
        else:
            self._components = self.catalog_manager.discovery_layer.discover_components(self.step_name)
            self._metadata = self.catalog_manager.discovery_layer.extract_metadata(self.step_name)
            self.catalog_manager.component_cache.set(
                f"step_{self.step_name}", 
                (self._components, self._metadata)
            )
        
        self._loaded = True
```

#### **2. Incremental Indexing**
```python
class IncrementalIndexer:
    """Handles incremental updates to the step index."""
    
    def __init__(self, index_engine: IndexEngine):
        self.index_engine = index_engine
        self.file_watcher = FileWatcher()
    
    def start_watching(self) -> None:
        """Start watching for file system changes."""
        self.file_watcher.watch(
            self.index_engine.workspace_root,
            on_change=self._handle_file_change
        )
    
    def _handle_file_change(self, event: FileChangeEvent) -> None:
        """Handle file system change events."""
        if self._is_component_file(event.file_path):
            if event.event_type == "created":
                self._add_component(event.file_path)
            elif event.event_type == "modified":
                self._update_component(event.file_path)
            elif event.event_type == "deleted":
                self._remove_component(event.file_path)
    
    def _is_component_file(self, file_path: Path) -> bool:
        """Check if file is a step component."""
        return any(
            pattern.match(file_path.name) 
            for pattern in FileDiscoverer.COMPONENT_PATTERNS.values()
        )
```

## Migration Strategy

### Phased Migration Plan

#### **Phase 1: Parallel Operation (4 weeks)**
- Deploy unified catalog alongside existing systems
- Route 10% of queries to new system for testing
- Monitor performance and correctness
- Fix issues without impacting production

#### **Phase 2: Gradual Transition (6 weeks)**
- Increase traffic to unified catalog (25%, 50%, 75%)
- Update high-level APIs to use new system
- Maintain backward compatibility adapters
- Deprecate old APIs with warnings

#### **Phase 3: Full Migration (4 weeks)**
- Route 100% of traffic to unified catalog
- Remove old discovery systems
- Clean up deprecated code
- Update documentation and examples

### Migration Safety Measures

```python
class MigrationController:
    """Controls migration between old and new catalog systems."""
    
    def __init__(self, old_system: Any, new_system: StepCatalog):
        self.old_system = old_system
        self.new_system = new_system
        self.migration_percentage = 0
        self.comparison_mode = True
    
    def route_query(self, query_type: str, **kwargs) -> Any:
        """Route query to appropriate system based on migration settings."""
        use_new_system = random.random() < (self.migration_percentage / 100)
        
        if use_new_system:
            try:
                result = self._execute_new_system(query_type, **kwargs)
                
                if self.comparison_mode:
                    # Compare with old system for validation
                    old_result = self._execute_old_system(query_type, **kwargs)
                    self._compare_results(query_type, result, old_result)
                
                return result
            except Exception as e:
                # Fallback to old system on error
                logging.error(f"New system failed, falling back: {e}")
                return self._execute_old_system(query_type, **kwargs)
        else:
            return self._execute_old_system(query_type, **kwargs)
```

## Quality Assurance

### Testing Strategy

#### **Unit Tests**
```python
class TestStepCatalog:
    """Comprehensive unit tests for step catalog system."""
    
    def test_step_discovery_accuracy(self):
        """Test that all steps are discovered correctly."""
        catalog = StepCatalog(test_workspace_root)
        
        # Test with known step structure
        expected_steps = ["tabular_preprocess", "model_training", "model_evaluation"]
        discovered_steps = catalog.list_available_steps()
        
        assert set(expected_steps).issubset(set(discovered_steps))
    
    def test_component_completeness(self):
        """Test that all components are found for each step."""
        catalog = StepCatalog(test_workspace_root)
        
        step_info = catalog.get_step_info("tabular_preprocess")
        assert step_info.components.script is not None
        assert step_info.components.contract is not None
        # Other components may be optional
    
    def test_performance_requirements(self):
        """Test that performance requirements are met."""
        catalog = StepCatalog(large_test_workspace)
        
        # Test lookup performance
        start_time = time.time()
        step_info = catalog.get_step_info("test_step")
        lookup_time = time.time() - start_time
        
        assert lookup_time < 0.001  # <1ms requirement
```

#### **Integration Tests**
```python
class TestCatalogIntegration:
    """Integration tests for catalog system."""
    
    def test_multi_workspace_discovery(self):
        """Test discovery across multiple workspaces."""
        catalog = StepCatalog(multi_workspace_root)
        
        # Test workspace precedence
        step_info = catalog.get_step_info("shared_step")
        assert step_info.workspace_id == "developer_1"  # Developer takes precedence
        
        # Test fallback to shared
        step_info = catalog.get_step_info("shared_only_step")
        assert step_info.workspace_id == "shared"
    
    def test_backward_compatibility(self):
        """Test that legacy APIs still work."""
        catalog = StepCatalog(test_workspace_root)
        legacy_adapter = ContractDiscoveryEngineAdapter(catalog)
        
        contracts = legacy_adapter.discover_all_contracts()
        assert len(contracts) > 0
        assert "tabular_preprocess" in contracts
```

### Performance Benchmarks

```python
class CatalogBenchmarks:
    """Performance benchmarks for catalog system."""
    
    def benchmark_index_build_time(self):
        """Benchmark index building performance."""
        workspace_sizes = [10, 100, 1000, 5000]  # Number of steps
        
        for size in workspace_sizes:
            workspace = self.create_test_workspace(size)
            catalog = StepCatalog(workspace)
            
            start_time = time.time()
            catalog.refresh_index()
            build_time = time.time() - start_time
            
            print(f"Index build time for {size} steps: {build_time:.2f}s")
            assert build_time < (size * 0.01)  # Linear scaling requirement
    
    def benchmark_query_performance(self):
        """Benchmark query performance."""
        catalog = StepCatalog(large_test_workspace)
        
        # Warm up cache
        catalog.get_step_info("test_step")
        
        # Benchmark different query types
        queries = [
            ("exact_match", lambda: catalog.get_step_info("test_step")),
            ("fuzzy_search", lambda: catalog.search_steps("test")),
            ("list_all", lambda: catalog.list_available_steps())
        ]
        
        for query_name, query_func in queries:
            times = []
            for _ in range(100):
                start_time = time.time()
                query_func()
                times.append(time.time() - start_time)
            
            avg_time = sum(times) / len(times)
            print(f"{query_name} average time: {avg_time*1000:.2f}ms")
```

## Monitoring and Observability

### Metrics Collection

```python
class CatalogMetrics:
    """Collects and reports catalog system metrics."""
    
    def __init__(self):
        self.query_count = Counter()
        self.query_times = defaultdict(list)
        self.error_count = Counter()
        self.cache_hit_rate = 0.0
    
    def record_query(self, query_type: str, duration: float, success: bool):
        """Record query metrics."""
        self.query_count[query_type] += 1
        self.query_times[query_type].append(duration)
        
        if not success:
            self.error_count[query_type] += 1
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report."""
        return {
            "total_queries": sum(self.query_count.values()),
            "average_query_times": {
                query_type: sum(times) / len(times)
                for query_type, times in self.query_times.items()
            },
            "error_rates": {
                query_type: self.error_count[query_type] / self.query_count[query_type]
                for query_type in self.query_count
            },
            "cache_hit_rate": self.cache_hit_rate
        }
```

## Conclusion

The Unified Step Catalog System addresses the critical fragmentation in Cursus's component discovery mechanisms. By consolidating 16+ different discovery systems into a single, efficient, and well-designed solution, we achieve:

### **Quantitative Benefits**
- **Reduce redundancy** from 35-45% to target 15-25%
- **Improve performance** with O(1) lookups vs. current O(n) scans
- **Decrease maintenance burden** by 70% through consolidation
- **Increase developer productivity** through consistent APIs

### **Qualitative Benefits**
- **Simplified Architecture**: Single entry point for all step queries
- **Better Developer Experience**: Intuitive APIs with comprehensive documentation
- **Improved Reliability**: Robust error handling and graceful degradation
- **Future-Proof Design**: Extensible architecture for growing catalogs

### **Risk Mitigation**
- **Phased Migration**: Gradual transition with safety measures
- **Backward Compatibility**: Existing code continues to work
- **Performance Monitoring**: Continuous validation of requirements
- **Quality Assurance**: Comprehensive testing strategy

The design follows the **Code Redundancy Evaluation Guide** principles by:
- ✅ **Validating demand** through analysis of existing systems
- ✅ **Avoiding over-engineering** with simple, effective solutions
- ✅ **Prioritizing quality** over comprehensive feature coverage
- ✅ **Targeting optimal redundancy** levels (15-25%)

This system transforms the current **fragmented discovery chaos** into a **coherent, scalable component ecosystem** that enables developers to efficiently find, understand, and reuse existing work as the Cursus catalog continues to grow.

## References

### **Primary Analysis Sources**
- **[Code Redundancy Evaluation Guide](./code_redundancy_evaluation_guide.md)** - Framework for assessing architectural efficiency and avoiding over-engineering
- **Current System Analysis** - Analysis of 16+ existing discovery/resolver classes across `cursus/validation` and `cursus/workspace`

### **Core Component Design References**

#### **Script Contracts**
- **[Script Contract](./script_contract.md)** - Core script contract design and implementation patterns
- **[Step Contract](./step_contract.md)** - Step-level contract specifications and validation
- **[Contract Discovery Manager Design](./contract_discovery_manager_design.md)** - Detailed design for contract discovery mechanisms
- **[Level 1 Script Contract Alignment Design](./level1_script_contract_alignment_design.md)** - Script-contract alignment validation patterns

#### **Step Specifications**
- **[Step Specification](./step_specification.md)** - Core step specification design and structure
- **[Specification Driven Design](./specification_driven_design.md)** - Specification-driven development principles
- **[Level 2 Contract Specification Alignment Design](./level2_contract_specification_alignment_design.md)** - Contract-specification alignment patterns
- **[Level 3 Specification Dependency Alignment Design](./level3_specification_dependency_alignment_design.md)** - Specification dependency validation

#### **Step Builders**
- **[Step Builder](./step_builder.md)** - Core step builder architecture and patterns
- **[Step Builder Registry Design](./step_builder_registry_design.md)** - Registry-based builder discovery and management
- **[Universal Step Builder Test](./universal_step_builder_test.md)** - Comprehensive builder testing framework
- **[Enhanced Universal Step Builder Tester Design](./enhanced_universal_step_builder_tester_design.md)** - Advanced builder validation patterns
- **[Level 4 Builder Configuration Alignment Design](./level4_builder_configuration_alignment_design.md)** - Builder-configuration alignment validation

#### **Configuration Management**
- **[Config](./config.md)** - Core configuration system design
- **[Config Driven Design](./config_driven_design.md)** - Configuration-driven development principles
- **[Config Field Manager Refactoring](./config_field_manager_refactoring.md)** - Configuration field management patterns
- **[Config Manager Three Tier Implementation](./config_manager_three_tier_implementation.md)** - Hierarchical configuration management
- **[Step Config Resolver](./step_config_resolver.md)** - Step-specific configuration resolution

### **Validation System References**

#### **Alignment Validation**
- **[Unified Alignment Tester Master Design](./unified_alignment_tester_master_design.md)** - Comprehensive alignment validation framework
- **[Unified Alignment Tester Architecture](./unified_alignment_tester_architecture.md)** - Alignment validation system architecture
- **[Two Level Alignment Validation System Design](./two_level_alignment_validation_system_design.md)** - Multi-level validation approach
- **[Alignment Validation Data Structures](./alignment_validation_data_structures.md)** - Data models for validation systems

#### **Validation Engine**
- **[Validation Engine](./validation_engine.md)** - Core validation engine design and implementation
- **[SageMaker Step Type Aware Unified Alignment Tester Design](./sagemaker_step_type_aware_unified_alignment_tester_design.md)** - Step-type-aware validation patterns

### **Workspace-Aware System References**

#### **Core Workspace System**
- **[Workspace Aware System Master Design](./workspace_aware_system_master_design.md)** - Comprehensive workspace-aware system architecture
- **[Workspace Aware Core System Design](./workspace_aware_core_system_design.md)** - Core workspace management components
- **[Workspace Aware Multi Developer Management Design](./workspace_aware_multi_developer_management_design.md)** - Multi-developer workspace coordination

#### **Workspace Validation**
- **[Workspace Aware Validation System Design](./workspace_aware_validation_system_design.md)** - Workspace-aware validation framework
- **[Workspace Aware Pipeline Runtime Testing Design](./workspace_aware_pipeline_runtime_testing_design.md)** - Runtime testing in workspace environments

#### **Workspace Configuration**
- **[Workspace Aware Config Manager Design](./workspace_aware_config_manager_design.md)** - Workspace-specific configuration management
- **[Workspace Aware Spec Builder Design](./workspace_aware_spec_builder_design.md)** - Workspace-aware specification building

### **Registry System References**

#### **Core Registry Design**
- **[Registry Manager](./registry_manager.md)** - Core registry management system
- **[Registry Single Source of Truth](./registry_single_source_of_truth.md)** - Centralized registry principles
- **[Pipeline Registry](./pipeline_registry.md)** - Pipeline-specific registry implementation
- **[Specification Registry](./specification_registry.md)** - Specification registry management

#### **Registry Standardization**
- **[Registry Based Step Name Generation](./registry_based_step_name_generation.md)** - Standardized step naming from registry
- **[Step Definition Standardization Enforcement Design](./step_definition_standardization_enforcement_design.md)** - Registry-enforced standardization
- **[Hybrid Registry Standardization Enforcement Design](./hybrid_registry_standardization_enforcement_design.md)** - Multi-source registry standardization

#### **Distributed Registry**
- **[Workspace Aware Distributed Registry Design](./workspace_aware_distributed_registry_design.md)** - Distributed registry across workspaces

### **File Resolution and Discovery**
- **[Flexible File Resolver Design](./flexible_file_resolver_design.md)** - Dynamic file discovery and resolution patterns
- **[Dependency Resolution System](./dependency_resolution_system.md)** - Component dependency resolution
- **[Dependency Resolver](./dependency_resolver.md)** - Core dependency resolution implementation

### **Implementation References**
- **[Documentation YAML Frontmatter Standard](./documentation_yaml_frontmatter_standard.md)** - Documentation standards used in this design
- **[Design Principles](./design_principles.md)** - Foundational design principles and architectural philosophy
- **[Standardization Rules](./standardization_rules.md)** - System-wide standardization guidelines
- **Existing Discovery Systems** - `ContractDiscoveryEngine`, `ContractDiscoveryManager`, `FlexibleFileResolver`, `WorkspaceDiscoveryManager`, etc.

### **Architecture Patterns**
- **Unified API Pattern** - Single entry point hiding complexity
- **Layered Architecture** - Clear separation of concerns
- **Lazy Loading** - Efficient resource utilization
- **Adapter Pattern** - Backward compatibility during migration
- **Registry Pattern** - Centralized component registration and discovery
- **Workspace Pattern** - Multi-tenant component isolation and management

### **Related System Integration**
- **[Pipeline Catalog Design](./pipeline_catalog_design.md)** - Integration with pipeline catalog system
- **[Pipeline DAG Resolver Design](./pipeline_dag_resolver_design.md)** - DAG-based component resolution
- **[Runtime Tester Design](./runtime_tester_design.md)** - Runtime testing integration patterns

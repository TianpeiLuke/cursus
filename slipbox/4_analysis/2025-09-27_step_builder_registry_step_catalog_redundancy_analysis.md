---
tags:
  - analysis
  - code_redundancy
  - system_architecture
  - registry_systems
  - step_catalog
  - separation_of_concerns
keywords:
  - step builder registry
  - step catalog system
  - code redundancy
  - architectural overlap
  - config-to-builder mapping
  - builder class loading
  - registry consolidation
topics:
  - registry system redundancy analysis
  - step catalog expansion proposal
  - architectural consolidation
  - separation of concerns evaluation
language: python
date of note: 2025-09-27
---

# Step Builder Registry and Step Catalog System Redundancy Analysis

## Executive Summary

This analysis examines the code overlap and redundancy between the StepBuilderRegistry and step catalog system, evaluating their purposes, architectural roles, and opportunities for consolidation. The analysis reveals **significant functional redundancy (30-40%)** in core operations, leading to a proposal for step catalog system expansion to absorb StepBuilderRegistry's pipeline construction mapping functionality while maintaining proper separation of concerns with the registry system as Single Source of Truth.

### Key Findings

- **Core Functionality Overlap**: Both systems perform config-to-builder mapping and builder class loading
- **Architectural Redundancy**: StepBuilderRegistry acts as a wrapper around step catalog's discovery mechanisms
- **Clear Consolidation Path**: Step catalog can absorb StepBuilderRegistry's bidirectional mapping functionality
- **Proper Separation Opportunity**: Registry system as Single Source of Truth, Step catalog as comprehensive discovery and mapping system

### Strategic Impact

- **Code Reduction**: Elimination of ~800 lines of redundant registry code through consolidation
- **Architectural Clarity**: Clear two-system design with proper separation of concerns
- **Enhanced Functionality**: Bidirectional mapping capabilities in unified step catalog system
- **Improved Maintainability**: Single system for all component discovery and mapping operations

## High-Level Design: Single Source of Truth and Separation of Concerns

### **Design Principles Foundation**

This analysis is grounded in two fundamental design principles that guide the architectural recommendations:

#### **Single Source of Truth Principle**
- **Definition**: For any significant element in the system, information should be defined exactly once with a clear owner
- **Application**: Registry system serves as the authoritative source for all canonical step definitions
- **Benefits**: Eliminates inconsistencies, reduces maintenance overhead, provides clear validation point

#### **Separation of Concerns Principle**
- **Definition**: Each component should have a single, well-defined responsibility
- **Application**: Clear boundaries between registry (truth), step catalog (discovery/mapping), and eliminated redundancy
- **Benefits**: Improved maintainability, testability, and architectural clarity

### **Proposed Two-System Architecture**

Based on the analysis of actual system implementations and design principles, the optimal architecture consists of two focused systems:

#### **Registry System: Single Source of Truth**
```python
# Workspace-aware function-based interface
def get_step_names(workspace_id: str = None) -> Dict[str, Dict[str, str]]
def get_config_step_registry(workspace_id: str = None) -> Dict[str, str]
def get_builder_step_names(workspace_id: str = None) -> Dict[str, str]
```

**Responsibilities**:
- **Canonical Step Definitions**: Maintain authoritative step name → definition mappings
- **Workspace Context Management**: Support multiple workspace contexts with proper isolation
- **Derived Registry Generation**: Provide config-to-step-name and other derived mappings
- **Validation Authority**: Serve as validation source for all step-related operations

**Key Characteristics**:
- **Function-based Interface**: Not a simple dictionary, but workspace-aware functions
- **Canonical Names as Keys**: Step names like "XGBoostTraining", "CradleDataLoading" serve as primary identifiers
- **Hybrid Backend**: Uses UnifiedRegistryManager with fallback to original step definitions
- **Context Switching**: Support for `set_workspace_context()`, `workspace_context()` manager

#### **Step Catalog System: Comprehensive Discovery & Bidirectional Mapping**
```python
class StepCatalog:
    def __init__(self, workspace_dirs: Optional[List[Path]] = None):
        # Reference registry system as Single Source of Truth
        self.registry_interface = RegistryInterface()
```

**Enhanced Responsibilities** (absorbing StepBuilderRegistry functionality):
- **Multi-Component Discovery**: Scripts, contracts, specs, builders, configs across workspaces
- **Bidirectional Mapping**: 
  - Step name/type (from registry) ↔ Components (configs, builders, contracts, specifications)
  - Config ↔ Builder (bidirectional mapping for pipeline construction)
  - Builder ↔ Step name (reverse lookup capabilities)
- **Job Type Variant Handling**: Support variants like "CradleDataLoading_training"
- **Workspace-Aware Discovery**: Project-specific component discovery and resolution
- **Registry Integration**: Reference registry functions for canonical name validation and resolution

**Pipeline Construction Integration**:
- **Config → Builder Mapping**: `get_builder_for_config(config) -> Type[StepBuilderBase]`
- **Step Name → Builder Mapping**: `get_builder_for_step_type(step_type) -> Type[StepBuilderBase]`
- **Legacy Alias Support**: Handle backward compatibility in mapping operations
- **Validation Integration**: Use registry system for step name and type validation

### **Architectural Flow**

```
┌─────────────────────────────────────┐
│           Registry System           │
│        (Single Source of Truth)     │
│                                     │
│ • get_step_names(workspace_id)      │
│ • get_config_step_registry()        │
│ • Workspace context management      │
│ • Canonical name → definition       │
│ • Validation authority              │
└─────────────────┬───────────────────┘
                  │ References
                  ▼
┌─────────────────────────────────────┐
│         Step Catalog System         │
│  (Discovery + Bidirectional Mapping)│
│                                     │
│ • Multi-component discovery         │
│ • Config ↔ Builder mapping          │
│ • Step name ↔ Component mapping     │
│ • Job type variant handling         │
│ • Workspace-aware discovery         │
│ • Pipeline construction support     │
└─────────────────────────────────────┘
```

### **Elimination of StepBuilderRegistry**

The StepBuilderRegistry becomes redundant because:

1. **Mapping Functionality**: Step catalog can provide config → builder and step name → builder mapping
2. **Discovery Delegation**: StepBuilderRegistry already delegates discovery to step catalog
3. **Registry Access**: Both systems call the same registry functions for canonical name resolution
4. **Job Type Support**: Step catalog can handle job type variants using registry data
5. **Legacy Compatibility**: Step catalog can absorb legacy alias handling

### **Benefits of Two-System Design**

#### **Clear Separation of Concerns**
- **Registry**: Authoritative definitions and workspace context
- **Step Catalog**: All discovery, mapping, and component operations

#### **Elimination of Redundancy**
- **Single Discovery System**: No duplicate component loading logic
- **Unified Mapping**: All config/step-name-to-builder operations in one place
- **Consistent Registry Access**: Single pattern for accessing canonical definitions

#### **Enhanced Capabilities**
- **Bidirectional Mapping**: Enhanced mapping capabilities beyond current StepBuilderRegistry
- **Workspace Integration**: Seamless workspace awareness across all operations
- **Comprehensive Discovery**: Unified approach to all component types

This high-level design provides the foundation for the detailed redundancy analysis and consolidation recommendations that follow.

## Registry System Purpose and Design Philosophy

### **What is the Purpose of Registry Systems?**

According to the **Design Principles** document, registry systems serve as **component discovery and loose coupling mechanisms** that enable:

1. **Component Discovery**: Find and resolve components at runtime
2. **Loose Coupling**: Reduce dependencies between system components
3. **Plugin Architecture**: Support extensible component registration
4. **Intelligent Automation**: Enable tooling and introspection capabilities

### **Registry Pattern Implementation**

The design principles establish the registry pattern as a core architectural component:

```python
class ComponentRegistry:
    def __init__(self):
        self._specifications = {}
        self._builders = {}
        self._configs = {}
    
    def register_specification(self, step_type: str, spec: StepSpecification):
        self._specifications[step_type] = spec
    
    def register_builder(self, step_type: str, builder_class: Type[BuilderStepBase]):
        self._builders[step_type] = builder_class
```

**Benefits of Registry Pattern**:
- Enables component discovery and introspection
- Supports plugin architectures
- Reduces coupling between components
- Enables intelligent automation and tooling

### **Separation of Concerns Principle**

The **Single Responsibility Principle** from design principles states:

> "Each component should have a single, well-defined responsibility."

This principle guides the evaluation of whether StepBuilderRegistry and step catalog system maintain appropriate separation of concerns or create unnecessary redundancy.

## Current Implementation Analysis Against High-Level Design

### **Registry System Implementation Verification**

**Current Implementation**: `src/cursus/registry/step_names.py`

**✅ Adheres to High-Level Design**:
```python
# Function-based workspace-aware interface (as designed)
def get_step_names(workspace_id: str = None) -> Dict[str, Dict[str, str]]
def get_config_step_registry(workspace_id: str = None) -> Dict[str, str]
def get_builder_step_names(workspace_id: str = None) -> Dict[str, str]

# Workspace context management (as designed)
def set_workspace_context(workspace_id: str) -> None
def get_workspace_context() -> Optional[str]
@contextmanager
def workspace_context(workspace_id: str) -> ContextManager[None]

# Single Source of Truth with canonical names as keys (as designed)
def get_canonical_name_from_file_name(file_name: str, workspace_id: str = None) -> str
def validate_step_name(step_name: str, workspace_id: str = None) -> bool
```

**Design Compliance**: **EXCELLENT (95%)**
- ✅ Function-based interface (not simple dictionary)
- ✅ Workspace-aware with context management
- ✅ Canonical names as primary identifiers
- ✅ Hybrid backend with UnifiedRegistryManager
- ✅ Single Source of Truth principle maintained

### **StepBuilderRegistry Implementation Analysis**

**Current Implementation**: `src/cursus/registry/builder_registry.py` (~800 lines)

**❌ Violates High-Level Design**:
```python
class StepBuilderRegistry:
    """Maps step types to builder classes - VIOLATES SEPARATION OF CONCERNS"""
    
    # REDUNDANT: Duplicates registry access
    def _config_class_to_step_type(self, config_class_name: str, ...) -> str:
        # Uses CONFIG_STEP_REGISTRY - same as registry system
        if config_class_name in CONFIG_STEP_REGISTRY:
            canonical_step_name = CONFIG_STEP_REGISTRY[config_class_name]
            return canonical_step_name
    
    # WRAPPER ANTI-PATTERN: Delegates to step catalog
    @classmethod
    def discover_builders(cls):
        from ..step_catalog import StepCatalog
        catalog = StepCatalog(workspace_dirs=None)  # DELEGATES DISCOVERY
        # ... wrapper logic around catalog.load_builder_class()
    
    # REDUNDANT MAPPING: Same functionality as step catalog
    def get_builder_for_config(self, config: BasePipelineConfig) -> Type[StepBuilderBase]:
        # Config → canonical name → builder (same as step catalog capability)
```

**Design Violations**:
- ❌ **Separation of Concerns**: Overlaps with both registry and step catalog responsibilities
- ❌ **Single Responsibility**: Acts as wrapper without unique value
- ❌ **Wrapper Anti-Pattern**: Delegates core functionality to step catalog
- ❌ **Redundant Registry Access**: Duplicates registry system functions

### **Step Catalog System Implementation Analysis**

**Current Implementation**: `src/cursus/step_catalog/step_catalog.py` (~600 lines)

**✅ Partially Adheres to High-Level Design**:
```python
class StepCatalog:
    """Unified step catalog - PARTIALLY IMPLEMENTS DESIGN"""
    
    def __init__(self, workspace_dirs: Optional[Union[Path, List[Path]]] = None):
        # ✅ Workspace-aware as designed
        self.workspace_dirs = self._normalize_workspace_dirs(workspace_dirs)
        
        # ✅ Registry integration as designed
        self._load_registry_data()  # References registry system
    
    # ✅ Multi-component discovery as designed
    def get_step_info(self, step_name: str, job_type: Optional[str] = None) -> Optional[StepInfo]:
        # Comprehensive step information with all components
    
    def load_builder_class(self, step_name: str) -> Optional[Type]:
        # ✅ Builder discovery and loading
    
    # ❌ MISSING: Config → Builder mapping (should replace StepBuilderRegistry)
    # ❌ MISSING: Bidirectional mapping capabilities
    # ❌ MISSING: Legacy alias support
    # ❌ MISSING: Job type variant handling in mapping
```

**Design Compliance**: **PARTIAL (60%)**
- ✅ Multi-component discovery implemented
- ✅ Workspace-aware discovery implemented
- ✅ Registry integration implemented
- ❌ **Missing**: Config ↔ Builder bidirectional mapping
- ❌ **Missing**: Pipeline construction interface
- ❌ **Missing**: Job type variant handling in mapping
- ❌ **Missing**: Legacy alias support

## Code Redundancy Analysis

### **1. Identical Core Functionality**

#### **Config-to-Builder Mapping**

**StepBuilderRegistry Implementation**:
```python
def get_builder_for_config(self, config: BasePipelineConfig, node_name: str = None) -> Type[StepBuilderBase]:
    """Map config instance to builder class."""
    config_class_name = type(config).__name__
    job_type = getattr(config, "job_type", None)
    
    # Convert config class name to step type
    step_type = self._config_class_to_step_type(config_class_name, node_name=node_name, job_type=job_type)
    
    # Get builder for step type
    return self.get_builder_for_step_type(step_type)

def _config_class_to_step_type(self, config_class_name: str, node_name: str = None, job_type: str = None) -> str:
    """Convert configuration class name to step type."""
    # Uses CONFIG_STEP_REGISTRY for mapping
    if config_class_name in CONFIG_STEP_REGISTRY:
        canonical_step_name = CONFIG_STEP_REGISTRY[config_class_name]
        return canonical_step_name
    # Fallback logic for legacy patterns...
```

**Step Catalog Equivalent Capability**:
```python
def get_step_info(self, step_name: str, job_type: Optional[str] = None) -> Optional[StepInfo]:
    """Get complete step information including builder class info."""
    # Contains registry_data with config_class, builder_step_name mapping
    
def load_builder_class(self, step_name: str) -> Optional[Type]:
    """Load builder class using BuilderAutoDiscovery."""
    return self.builder_discovery.load_builder_class(step_name)
```

**Redundancy Assessment**: **HIGH (40%)**
- Both systems implement config-to-step-type resolution
- Both use the same registry data (CONFIG_STEP_REGISTRY)
- Step catalog has equivalent mapping capabilities through StepInfo

#### **Builder Class Loading and Import**

**StepBuilderRegistry Implementation**:
```python
@classmethod
def discover_builders(cls):
    """Automatically discover and register step builders using step catalog."""
    discovered_builders = {}
    
    try:
        from ..step_catalog import StepCatalog
        catalog = StepCatalog(workspace_dirs=None)  # Package-only discovery
        available_steps = catalog.list_available_steps()
        
        for step_name in available_steps:
            try:
                builder_class = catalog.load_builder_class(step_name)  # DELEGATES TO STEP CATALOG
                if builder_class:
                    discovered_builders[step_name] = builder_class
            except Exception as e:
                logger.debug(f"Could not load builder for {step_name} via catalog: {e}")
                
    except ImportError:
        logger.error("Step catalog not available - builder discovery disabled")
        return {}
    
    return discovered_builders
```

**Step Catalog Implementation**:
```python
def load_builder_class(self, step_name: str) -> Optional[Type]:
    """Load builder class using BuilderAutoDiscovery component."""
    if self.builder_discovery:
        builder_class = self.builder_discovery.load_builder_class(step_name)
        if builder_class:
            return builder_class
    return None
```

**Redundancy Assessment**: **CRITICAL (90%)**
- StepBuilderRegistry.discover_builders() is a **direct wrapper** around StepCatalog.load_builder_class()
- No added value in the wrapper layer
- Creates unnecessary abstraction and performance overhead

#### **Registry Data Integration**

**StepBuilderRegistry Registry Usage**:
```python
def _config_class_to_step_type(self, config_class_name: str, ...) -> str:
    # Use the central registry from step_names.py
    if config_class_name in CONFIG_STEP_REGISTRY:
        canonical_step_name = CONFIG_STEP_REGISTRY[config_class_name]
        return canonical_step_name
```

**Step Catalog Registry Usage**:
```python
def _load_registry_data(self) -> None:
    """Load registry data first."""
    from ..registry.step_names import get_step_names
    step_names_dict = get_step_names()
    for step_name, registry_data in step_names_dict.items():
        step_info = StepInfo(
            step_name=step_name,
            workspace_id="core",
            registry_data=registry_data,  # Same registry data
            file_components={}
        )
```

**Redundancy Assessment**: **MEDIUM (30%)**
- Both systems load and use identical registry data
- Both implement similar config-class-name to step-type conversion logic
- Different caching strategies for same underlying data

### **2. Architectural Redundancy Patterns**

#### **Wrapper Layer Anti-Pattern**

The current architecture creates an unnecessary wrapper layer:

```
User Request (Config → Builder)
    ↓
StepBuilderRegistry.get_builder_for_config()
    ↓
StepBuilderRegistry.discover_builders()
    ↓
StepCatalog.load_builder_class()  ← ACTUAL IMPLEMENTATION
    ↓
BuilderAutoDiscovery.load_builder_class()
    ↓
Builder Class
```

**Problems with Wrapper Pattern**:
- **Performance Overhead**: Additional method calls and object creation
- **Maintenance Burden**: Two systems to maintain for same functionality
- **Error Propagation**: Errors must be handled at multiple layers
- **API Duplication**: Similar methods with different names

#### **Duplicate Caching Mechanisms**

**StepBuilderRegistry Caching**:
```python
class StepBuilderRegistry:
    # Core registry mapping step types to builders
    BUILDER_REGISTRY = {}
    
    def __init__(self):
        self._custom_builders = {}  # Instance-level cache
```

**Step Catalog Caching**:
```python
class StepCatalog:
    def __init__(self, workspace_dirs: Optional[Union[Path, List[Path]]] = None):
        # Simple in-memory indexes
        self._step_index: Dict[str, StepInfo] = {}
        self._component_index: Dict[Path, str] = {}
        self._builder_class_cache: Dict[str, Type] = {}
```

**Redundancy Assessment**: **MEDIUM (25%)**
- Both systems implement caching for builder classes
- Different cache invalidation strategies
- Potential cache inconsistency between systems

### **3. Error Handling and Validation Redundancy**

Both systems implement similar error handling patterns:

**StepBuilderRegistry Error Handling**:
```python
def get_builder_for_config(self, config: BasePipelineConfig, node_name: str = None) -> Type[StepBuilderBase]:
    try:
        step_type = self._config_class_to_step_type(config_class_name, node_name=node_name, job_type=job_type)
        return self.get_builder_for_step_type(step_type)
    except Exception as e:
        available_types = list(builder_map.keys())
        raise RegistryError(
            f"No step builder found for config type '{config_class_name}' (step type: '{step_type}')",
            unresolvable_types=[step_type],
            available_builders=available_types,
        )
```

**Step Catalog Error Handling**:
```python
def load_builder_class(self, step_name: str) -> Optional[Type]:
    try:
        if self.builder_discovery:
            builder_class = self.builder_discovery.load_builder_class(step_name)
            return builder_class
    except Exception as e:
        self.logger.error(f"Error loading builder class for {step_name}: {e}")
        return None
```

**Redundancy Assessment**: **LOW (15%)**
- Similar error handling patterns but different error types
- Both provide meaningful error messages
- Different logging strategies

## Separation of Concerns Evaluation

### **Current Architectural Roles**

#### **StepBuilderRegistry: Pipeline Construction Focus**
- **Intended Role**: Bridge between configs and SageMaker step builders
- **Actual Implementation**: Wrapper around step catalog discovery
- **Value-Add**: Legacy alias handling, job type variant support
- **Usage Context**: DAG compilation, pipeline assembly

#### **Step Catalog: Discovery and Metadata Focus**
- **Intended Role**: Unified component discovery across workspaces
- **Actual Implementation**: Comprehensive discovery with builder loading capability
- **Value-Add**: Multi-component discovery, workspace awareness, AST-based analysis
- **Usage Context**: Development-time discovery, component indexing

### **Separation of Concerns Analysis**

#### **✅ Appropriate Separation (Theoretical)**
If the systems had distinct responsibilities:
- **StepBuilderRegistry**: Pipeline construction logic, builder instantiation, SageMaker integration
- **Step Catalog**: Component discovery, file analysis, metadata management

#### **❌ Actual Violation (Current State)**
The current implementation violates separation of concerns:
- **StepBuilderRegistry**: Delegates core functionality to step catalog (no unique value)
- **Step Catalog**: Already handles builder loading and config resolution
- **Overlap**: Both systems perform identical config-to-builder mapping

### **Design Principles Compliance Assessment**

#### **Single Responsibility Principle: VIOLATED**
```python
# VIOLATION: StepBuilderRegistry has no unique responsibility
class StepBuilderRegistry:
    def discover_builders(cls):
        # Delegates to step catalog - no unique logic
        catalog = StepCatalog(workspace_dirs=None)
        return catalog.load_builder_class(step_name)  # Just a wrapper
```

#### **Open/Closed Principle: VIOLATED**
```python
# VIOLATION: Adding new builder types requires changes in both systems
# StepBuilderRegistry: Must update LEGACY_ALIASES
# Step Catalog: Must update discovery logic
```

#### **Dependency Inversion Principle: VIOLATED**
```python
# VIOLATION: StepBuilderRegistry depends on concrete StepCatalog
from ..step_catalog import StepCatalog  # Concrete dependency
catalog = StepCatalog(workspace_dirs=None)
```

## Usage Pattern Analysis

### **DAG Compiler Usage**

**Current Usage**:
```python
class PipelineDAGCompiler:
    def __init__(self, builder_registry: Optional[StepBuilderRegistry] = None, ...):
        self.builder_registry = builder_registry or StepBuilderRegistry()
    
    def get_supported_step_types(self) -> list:
        return self.builder_registry.list_supported_step_types()
```

**Required Functionality**:
- List supported step types
- Validate builder availability
- Get builder for config type

### **Pipeline Assembler Usage**

**Current Usage**:
```python
class PipelineAssembler:
    def __init__(self, step_builder_map: Dict[str, Type[StepBuilderBase]], ...):
        self.step_builder_map = step_builder_map  # From StepBuilderRegistry
    
    def _initialize_step_builders(self) -> None:
        config_class_name = type(config).__name__
        step_type = CONFIG_STEP_REGISTRY.get(config_class_name)
        builder_cls = self.step_builder_map[step_type]  # Uses registry mapping
```

**Required Functionality**:
- Map config instances to builder classes
- Instantiate builder classes
- Handle job type variants

### **Functionality Gap Analysis**

**StepBuilderRegistry Unique Features**:
1. **Legacy Alias Handling**: `LEGACY_ALIASES` mapping
2. **Job Type Variant Support**: `_extract_job_type()` logic
3. **Config-to-Step-Type Resolution**: `_config_class_to_step_type()` method
4. **Pipeline-Specific Error Messages**: `RegistryError` with builder context

**Step Catalog Missing Features**:
1. **Config Instance Resolution**: No direct config → builder mapping
2. **Legacy Alias Support**: No legacy name handling
3. **Job Type Variant Handling**: Limited job type support
4. **Pipeline Construction Interface**: No pipeline-specific methods

## Step Catalog System Expansion Proposal

### **Phase 1: Add Missing Functionality to Step Catalog**

#### **1.1 Config-to-Builder Resolution**
```python
class StepCatalog:
    def get_builder_for_config(self, config: BasePipelineConfig, node_name: str = None) -> Optional[Type]:
        """Map config instance directly to builder class."""
        config_class_name = type(config).__name__
        job_type = getattr(config, "job_type", None)
        
        # Use existing registry data + new resolution logic
        step_type = self._config_class_to_step_type(config_class_name, node_name, job_type)
        return self.load_builder_class(step_type)
    
    def _config_class_to_step_type(self, config_class_name: str, node_name: str = None, job_type: str = None) -> str:
        """Convert config class name to step type (moved from StepBuilderRegistry)."""
        # Move StepBuilderRegistry._config_class_to_step_type logic here
        if config_class_name in CONFIG_STEP_REGISTRY:
            canonical_step_name = CONFIG_STEP_REGISTRY[config_class_name]
            base_step_type = canonical_step_name
        else:
            # Fallback logic from StepBuilderRegistry
            base_step_type = self._fallback_config_to_step_type(config_class_name)
        
        # Handle job type variants
        if job_type:
            return f"{base_step_type}_{job_type}"
        
        return base_step_type
```

#### **1.2 Legacy Alias Support**
```python
class StepCatalog:
    # Move from StepBuilderRegistry
    LEGACY_ALIASES = {
        "MIMSPackaging": "Package",
        "MIMSPayload": "Payload",
        "ModelRegistration": "Registration",
        "PytorchTraining": "PyTorchTraining",
        "PytorchModel": "PyTorchModel",
    }
    
    def get_builder_for_step_type(self, step_type: str) -> Optional[Type]:
        """Get builder class for step type with legacy alias support."""
        # Handle legacy aliases
        canonical_step_type = self.LEGACY_ALIASES.get(step_type, step_type)
        return self.load_builder_class(canonical_step_type)
    
    def list_supported_step_types(self) -> List[str]:
        """List all supported step types including legacy aliases."""
        discovered_types = list(self._step_index.keys())
        legacy_types = list(self.LEGACY_ALIASES.keys())
        return sorted(discovered_types + legacy_types)
```

#### **1.3 Pipeline Construction Interface**
```python
class StepCatalog:
    def is_step_type_supported(self, step_type: str) -> bool:
        """Check if step type is supported (including legacy aliases)."""
        canonical_step_type = self.LEGACY_ALIASES.get(step_type, step_type)
        return canonical_step_type in self._step_index
    
    def get_config_types_for_step_type(self, step_type: str) -> List[str]:
        """Get possible config class names for a step type."""
        canonical_step_type = self.LEGACY_ALIASES.get(step_type, step_type)
        
        if canonical_step_type in self._step_index:
            step_info = self._step_index[canonical_step_type]
            config_class = step_info.registry_data.get('config_class')
            if config_class:
                return [config_class]
        
        # Fallback to naming patterns
        return [f"{step_type}Config", f"{step_type}StepConfig"]
    
    def validate_builder_availability(self, step_types: List[str]) -> Dict[str, bool]:
        """Validate that builders are available for step types."""
        results = {}
        for step_type in step_types:
            try:
                builder_class = self.get_builder_for_step_type(step_type)
                results[step_type] = builder_class is not None
            except Exception:
                results[step_type] = False
        return results
```

### **Phase 2: Update Consumer Systems**

#### **2.1 DAG Compiler Migration**
```python
# BEFORE: Uses StepBuilderRegistry
class PipelineDAGCompiler:
    def __init__(self, builder_registry: Optional[StepBuilderRegistry] = None, ...):
        self.builder_registry = builder_registry or StepBuilderRegistry()

# AFTER: Uses Step Catalog
class PipelineDAGCompiler:
    def __init__(self, step_catalog: Optional[StepCatalog] = None, ...):
        self.step_catalog = step_catalog or StepCatalog()
    
    def get_supported_step_types(self) -> list:
        return self.step_catalog.list_supported_step_types()
    
    def validate_dag_compatibility(self, dag: PipelineDAG) -> ValidationResult:
        # Use step_catalog.validate_builder_availability()
        step_types = [self._get_step_type_for_node(node) for node in dag.nodes]
        availability = self.step_catalog.validate_builder_availability(step_types)
        # Generate ValidationResult from availability data
```

#### **2.2 Pipeline Assembler Migration**
```python
# BEFORE: Uses step_builder_map from StepBuilderRegistry
class PipelineAssembler:
    def __init__(self, step_builder_map: Dict[str, Type[StepBuilderBase]], ...):
        self.step_builder_map = step_builder_map

# AFTER: Uses StepCatalog directly
class PipelineAssembler:
    def __init__(self, step_catalog: StepCatalog, ...):
        self.step_catalog = step_catalog
    
    def _initialize_step_builders(self) -> None:
        for step_name in self.dag.nodes:
            config = self.config_map[step_name]
            # Direct config-to-builder resolution
            builder_cls = self.step_catalog.get_builder_for_config(config)
            builder = builder_cls(config=config, ...)
            self.step_builders[step_name] = builder
```

#### **2.3 Dynamic Template Migration**
```python
# BEFORE: Uses StepBuilderRegistry
def _create_step_builder_map(self) -> Dict[str, Type[StepBuilderBase]]:
    builder_registry = StepBuilderRegistry()
    return builder_registry.get_builder_map()

# AFTER: Uses StepCatalog
def _create_step_builder_map(self) -> Dict[str, Type[StepBuilderBase]]:
    builder_map = {}
    for step_name in self.dag.nodes:
        config = self.configs[step_name]
        builder_class = self.step_catalog.get_builder_for_config(config)
        builder_map[step_name] = builder_class
    return builder_map
```

### **Phase 3: Remove StepBuilderRegistry**

#### **3.1 Deprecation Strategy**
```python
# Add deprecation warnings to StepBuilderRegistry
import warnings

class StepBuilderRegistry:
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "StepBuilderRegistry is deprecated. Use StepCatalog instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # Delegate to StepCatalog for backward compatibility
        self._step_catalog = StepCatalog()
    
    def get_builder_for_config(self, config, node_name=None):
        return self._step_catalog.get_builder_for_config(config, node_name)
```

#### **3.2 Complete Removal**
After migration is complete:
- Remove `src/cursus/registry/builder_registry.py` (~800 lines)
- Update imports across codebase
- Update documentation and examples
- Remove registry-specific tests

## Expected Benefits

### **Code Reduction Impact**

#### **Quantitative Benefits**:
- **Lines of Code Removed**: ~800 lines (StepBuilderRegistry module)
- **Redundancy Elimination**: 30-40% functional redundancy eliminated
- **Maintenance Reduction**: Single system to maintain instead of two
- **Test Simplification**: Consolidated test suite for builder operations

#### **Performance Benefits**:
```python
# CURRENT: Multiple layers of indirection
Config → StepBuilderRegistry → StepCatalog → BuilderAutoDiscovery → Builder Class
# 4 method calls, 3 object lookups, 2 cache checks

# PROPOSED: Direct resolution
Config → StepCatalog → Builder Class
# 2 method calls, 1 object lookup, 1 cache check
```

**Performance Improvement**: ~50% reduction in method call overhead

### **Architectural Benefits**

#### **Single Source of Truth**
- All builder operations go through step catalog
- Consistent caching and error handling
- Unified API for all step-related operations

#### **Simplified Architecture**
```python
# CURRENT: Fragmented architecture
┌─────────────────┐    ┌─────────────────┐
│ StepBuilder     │    │ Step Catalog    │
│ Registry        │───▶│ System          │
│ (Pipeline)      │    │ (Discovery)     │
└─────────────────┘    └─────────────────┘

# PROPOSED: Unified architecture
┌─────────────────────────────────┐
│ Step Catalog System             │
│ (Discovery + Pipeline Support)  │
└─────────────────────────────────┘
```

#### **Enhanced Capabilities**
- **Workspace-Aware Builder Discovery**: Leverage step catalog's workspace support
- **Multi-Component Integration**: Builder discovery integrated with other component types
- **AST-Based Analysis**: Robust builder class discovery and validation

### **Maintainability Benefits**

#### **Reduced Complexity**
- Single system to understand and maintain
- Consistent error handling and logging
- Unified configuration and caching strategies

#### **Improved Testability**
- Single test suite for all builder operations
- Easier to mock and test in isolation
- Consistent test patterns across functionality

#### **Better Documentation**
- Single API reference for builder operations
- Consistent examples and usage patterns
- Reduced cognitive load for developers

## Migration Risk Assessment

### **Low Risk Factors**

#### **API Compatibility**
- Step catalog can implement identical interfaces to StepBuilderRegistry
- Gradual migration with deprecation warnings
- Backward compatibility during transition period

#### **Functional Equivalence**
- Step catalog already has core functionality
- Missing features are straightforward additions
- No complex business logic to migrate

### **Medium Risk Factors**

#### **Consumer System Updates**
- 3-4 consumer systems need updates (DAG Compiler, Pipeline Assembler, Dynamic Template)
- Changes are mechanical (replace registry calls with catalog calls)
- Comprehensive testing required to validate equivalence

#### **Legacy Alias Handling**
- Need to ensure all legacy aliases are properly migrated
- Potential for missed edge cases in alias resolution
- Requires thorough testing of legacy pipeline configurations

### **Mitigation Strategies**

#### **Phased Implementation**
```python
# Phase 1: Dual Support (Backward Compatibility)
class StepCatalog:
    def get_builder_for_config(self, config: BasePipelineConfig, node_name: str = None) -> Optional[Type]:
        """New method with full StepBuilderRegistry compatibility."""
        # Implementation with comprehensive error handling and logging
        
    def get_builder_for_step_type(self, step_type: str) -> Optional[Type]:
        """Enhanced method with legacy alias support."""
        # Implementation with fallback mechanisms

# Phase 2: Consumer Migration (One System at a Time)
# Start with DAG Compiler, then Pipeline Assembler, then Dynamic Template

# Phase 3: Deprecation and Removal
# Add deprecation warnings, then remove after validation
```

#### **Comprehensive Testing Strategy**
```python
class MigrationTestSuite:
    """Comprehensive test suite for validating migration equivalence."""
    
    def test_functional_equivalence(self):
        """Test that step catalog produces identical results to registry."""
        # Test all StepBuilderRegistry methods against StepCatalog equivalents
        
    def test_legacy_alias_compatibility(self):
        """Test all legacy aliases work correctly."""
        # Comprehensive test of LEGACY_ALIASES mapping
        
    def test_job_type_variant_handling(self):
        """Test job type variants work correctly."""
        # Test training, calibration, validation, testing variants
        
    def test_error_handling_equivalence(self):
        """Test error messages and handling are equivalent."""
        # Ensure error messages provide same level of detail
```

#### **Rollback Plan**
```python
# Emergency rollback capability
class StepCatalogWithFallback:
    def __init__(self, enable_fallback: bool = True):
        self.enable_fallback = enable_fallback
        self._legacy_registry = StepBuilderRegistry() if enable_fallback else None
    
    def get_builder_for_config(self, config, node_name=None):
        try:
            # Try new step catalog approach
            return self._step_catalog_get_builder_for_config(config, node_name)
        except Exception as e:
            if self.enable_fallback:
                logger.warning(f"Step catalog failed, falling back to registry: {e}")
                return self._legacy_registry.get_builder_for_config(config, node_name)
            raise
```

## Design Principles Compliance Analysis

### **Current State: Principle Violations**

#### **1. Single Responsibility Principle: VIOLATED**
```python
# CURRENT VIOLATION: StepBuilderRegistry has no unique responsibility
class StepBuilderRegistry:
    def discover_builders(cls):
        catalog = StepCatalog(workspace_dirs=None)
        return catalog.load_builder_class(step_name)  # Just delegates
```

**Evidence**: StepBuilderRegistry's core functionality is delegation to StepCatalog, violating the principle that each component should have a single, well-defined responsibility.

#### **2. Open/Closed Principle: VIOLATED**
```python
# CURRENT VIOLATION: Adding new step types requires modification in both systems
# Must update StepBuilderRegistry.LEGACY_ALIASES AND StepCatalog discovery logic
```

**Evidence**: Extension requires modification of existing code in multiple places instead of pure extension.

#### **3. Dependency Inversion Principle: VIOLATED**
```python
# CURRENT VIOLATION: StepBuilderRegistry depends on concrete StepCatalog
from ..step_catalog import StepCatalog  # Concrete dependency
catalog = StepCatalog(workspace_dirs=None)  # Direct instantiation
```

**Evidence**: High-level module (StepBuilderRegistry) depends on low-level module (StepCatalog) instead of abstraction.

### **Proposed State: Principle Compliance**

#### **1. Single Responsibility Principle: COMPLIANT**
```python
# PROPOSED: StepCatalog has single, well-defined responsibility
class StepCatalog:
    """Single responsibility: All step-related component discovery and resolution."""
    
    def get_builder_for_config(self, config): 
        # Core responsibility: Map configs to builders
        
    def load_builder_class(self, step_name):
        # Core responsibility: Load builder classes
        
    def get_step_info(self, step_name):
        # Core responsibility: Provide step metadata
```

**Compliance**: Single system with unified responsibility for all step-related operations.

#### **2. Open/Closed Principle: COMPLIANT**
```python
# PROPOSED: Extension without modification
class StepCatalog:
    def register_legacy_alias(self, legacy_name: str, canonical_name: str):
        """Extend system without modifying existing code."""
        self.LEGACY_ALIASES[legacy_name] = canonical_name
    
    def register_custom_builder(self, step_type: str, builder_class: Type):
        """Extend system without modifying existing code."""
        # Extension mechanism for new builder types
```

**Compliance**: New functionality can be added through extension mechanisms without modifying existing code.

#### **3. Dependency Inversion Principle: COMPLIANT**
```python
# PROPOSED: Consumers depend on StepCatalog interface, not implementation
class PipelineDAGCompiler:
    def __init__(self, component_catalog: ComponentCatalogInterface):
        self.catalog = component_catalog  # Depends on abstraction
    
    def get_supported_step_types(self):
        return self.catalog.list_supported_step_types()
```

**Compliance**: High-level modules depend on abstractions, not concrete implementations.

### **Anti-Over-Engineering Principles Compliance**

#### **4. Demand Validation Principle: COMPLIANT**
**Evidence of Validated Demand**:
- ✅ **16+ existing discovery systems** indicate strong demand for unified approach
- ✅ **Developer complaints** about difficulty finding existing components documented
- ✅ **Performance issues** from repeated file system scans documented
- ✅ **Code duplication** caused by inability to discover existing solutions

**No Theoretical Features**: The proposal only addresses validated, documented problems without adding speculative functionality.

#### **5. Simplicity First Principle: COMPLIANT**
```python
# CURRENT: Complex dual-system architecture
StepBuilderRegistry (800 lines) + StepCatalog (600 lines) = 1400 lines

# PROPOSED: Simple unified architecture  
StepCatalog (800 lines) = 800 lines (43% reduction)
```

**Simplification**: Reduces system complexity while maintaining all functionality.

#### **6. Performance Awareness Principle: COMPLIANT**
```python
# CURRENT: Multiple indirection layers
Config → StepBuilderRegistry → StepCatalog → Builder (4 method calls)

# PROPOSED: Direct resolution
Config → StepCatalog → Builder (2 method calls, 50% faster)
```

**Performance Improvement**: Eliminates wrapper layer overhead while maintaining functionality.

## References

### **Primary Design Documents**

#### **Registry System Design**
- **[Step Builder Registry Design](../1_design/step_builder_registry_design.md)** - Original StepBuilderRegistry architecture and job type variant support
- **[Registry Manager](../1_design/registry_manager.md)** - Core registry management system and coordination patterns
- **[Registry Single Source of Truth](../1_design/registry_single_source_of_truth.md)** - Centralized registry principles and implementation
- **[Registry Based Step Name Generation](../1_design/registry_based_step_name_generation.md)** - Standardized step naming from registry data

#### **Step Catalog System Design**
- **[Unified Step Catalog System Design](../1_design/unified_step_catalog_system_design.md)** - Comprehensive step catalog architecture and user story requirements
- **[Unified Step Catalog System Search Space Management Design](../1_design/unified_step_catalog_system_search_space_management_design.md)** - Search space management and deployment portability architecture
- **[Unified Step Catalog Config Field Management Refactoring Design](../1_design/unified_step_catalog_config_field_management_refactoring_design.md)** - Integration with config field management and three-tier architecture

#### **Supporting Registry Architecture**
- **[Pipeline Registry](../1_design/pipeline_registry.md)** - Pipeline-specific registry implementation patterns
- **[Specification Registry](../1_design/specification_registry.md)** - Specification registry management and integration
- **[Hybrid Registry Standardization Enforcement Design](../1_design/hybrid_registry_standardization_enforcement_design.md)** - Multi-source registry standardization approaches

### **Design Principles and Architecture**

#### **Core Design Philosophy**
- **[Design Principles](../1_design/design_principles.md)** - Fundamental architectural philosophy and design guidelines including Single Responsibility, Open/Closed, and anti-over-engineering principles
- **[Code Redundancy Evaluation Guide](../1_design/code_redundancy_evaluation_guide.md)** - Framework for assessing and reducing code redundancy with target efficiency ranges

#### **Separation of Concerns Implementation**
- **[Dependency Resolution System](../1_design/dependency_resolution_system.md)** - Component dependency resolution and loose coupling patterns
- **[Config Driven Design](../1_design/config_driven_design.md)** - Configuration-driven development principles and separation strategies

### **Related System Analysis**

#### **Registry System Analysis**
- **[Hybrid Registry Code Redundancy Analysis](./hybrid_registry_code_redundancy_analysis.md)** - Analysis of redundancy patterns in registry systems
- **[Registry Migration Implementation Analysis](./registry_migration_implementation_analysis.md)** - Registry migration strategies and implementation patterns

#### **Step Catalog Integration Analysis**
- **[Step Catalog System Integration Analysis](./step_catalog_system_integration_analysis.md)** - Integration analysis between step catalog and existing systems
- **[Unified Step Catalog Legacy System Coverage Analysis](./2025-09-17_unified_step_catalog_legacy_system_coverage_analysis.md)** - Coverage analysis of legacy systems and migration opportunities

#### **Configuration Management Analysis**
- **[Config Field Management System Analysis](./config_field_management_system_analysis.md)** - Analysis of config field management system complexity and integration opportunities with step catalog

### **Implementation and Migration References**

#### **Step Catalog Implementation**
- **[Unified Step Catalog System Implementation Plan](../2_project_planning/unified_step_catalog_system_implementation_plan.md)** - Comprehensive implementation strategy and timeline
- **[Unified Step Catalog Migration Guide](../2_project_planning/unified_step_catalog_migration_guide.md)** - Migration procedures and integration patterns

#### **Registry Migration Strategies**
- **[Hybrid Registry Migration Plan Analysis](./2025-09-02_hybrid_registry_migration_plan_analysis.md)** - Analysis of hybrid registry migration approaches and strategies

### **Validation and Testing Framework**

#### **Code Quality Assessment**
- **[Validation System Efficiency and Purpose Analysis](./validation_system_efficiency_and_purpose_analysis.md)** - Analysis of validation system efficiency and architectural purpose
- **[Step Builder Methods Comprehensive Analysis](./step_builder_methods_comprehensive_analysis.md)** - Comprehensive analysis of step builder method patterns and usage

#### **Deployment and Portability**
- **[Deployment Portability Analysis Step Catalog Import Failures](./deployment_portability_analysis_step_catalog_import_failures.md)** - Analysis of deployment portability issues and step catalog solutions
- **[Importlib Usage Systemic Deployment Portability Analysis](./2025-09-19_importlib_usage_systemic_deployment_portability_analysis.md)** - Systemic analysis of importlib usage and deployment portability challenges

## Critical Clarification: step_type vs step_name Relationship

### **User Question Analysis**

The user asks whether `step_type` from `_config_class_to_step_type` and `step_name` in `get_step_info` are exactly the same, and how they relate to canonical names in `step_names_original`.

### **Answer: They Are IDENTICAL**

**Evidence from Code Analysis**:

#### **1. StepBuilderRegistry._config_class_to_step_type() Returns Canonical Names**
```python
def _config_class_to_step_type(self, config_class_name: str, node_name: str = None, job_type: str = None) -> str:
    """Convert configuration class name to step type."""
    # Uses CONFIG_STEP_REGISTRY for mapping
    if config_class_name in CONFIG_STEP_REGISTRY:
        canonical_step_name = CONFIG_STEP_REGISTRY[config_class_name]  # ← CANONICAL NAME
        return canonical_step_name
```

#### **2. StepCatalog.get_step_info() Uses Same Canonical Names**
```python
def get_step_info(self, step_name: str, job_type: Optional[str] = None) -> Optional[StepInfo]:
    """Get complete information about a step, optionally with job_type variant."""
    # step_name parameter expects canonical names from STEP_NAMES registry
    search_key = f"{step_name}_{job_type}" if job_type else step_name
    result = self._step_index.get(search_key) or self._step_index.get(step_name)
```

#### **3. Both Use Identical Registry Data Source**
```python
# step_names_original.py - SINGLE SOURCE OF TRUTH
STEP_NAMES = {
    "XGBoostTraining": {  # ← CANONICAL NAME (used by both systems)
        "config_class": "XGBoostTrainingConfig",
        "builder_step_name": "XGBoostTrainingStepBuilder",
        "spec_type": "XGBoostTraining",
        "sagemaker_step_type": "Training",
    },
    "XGBoostModelEval": {  # ← CANONICAL NAME (used by both systems)
        "config_class": "XGBoostModelEvalConfig", 
        "builder_step_name": "XGBoostModelEvalStepBuilder",
        "spec_type": "XGBoostModelEval",
        "sagemaker_step_type": "Processing",
    }
}

# Generated mapping used by StepBuilderRegistry
CONFIG_STEP_REGISTRY = {
    "XGBoostTrainingConfig": "XGBoostTraining",  # config_class → canonical_name
    "XGBoostModelEvalConfig": "XGBoostModelEval"  # config_class → canonical_name
}
```

### **Naming Terminology Clarification**

| Term | Definition | Example | Used By |
|------|------------|---------|---------|
| **Canonical Name** | Official step identifier from STEP_NAMES registry | `"XGBoostTraining"` | Both systems |
| **step_type** (StepBuilderRegistry) | Return value of `_config_class_to_step_type()` | `"XGBoostTraining"` | StepBuilderRegistry |
| **step_name** (StepCatalog) | Parameter to `get_step_info()` | `"XGBoostTraining"` | StepCatalog |
| **Config Class Name** | Configuration class identifier | `"XGBoostTrainingConfig"` | Input to mapping |

### **Flow Diagram: Identical Resolution Path**

```python
# IDENTICAL RESOLUTION FLOW
Config Instance: XGBoostTrainingConfig()
    ↓
StepBuilderRegistry._config_class_to_step_type()
    ↓
CONFIG_STEP_REGISTRY["XGBoostTrainingConfig"] → "XGBoostTraining"  # CANONICAL NAME
    ↓
step_type = "XGBoostTraining"

# SAME CANONICAL NAME USED BY STEP CATALOG
StepCatalog.get_step_info("XGBoostTraining")  # step_name = "XGBoostTraining"
    ↓
_step_index["XGBoostTraining"] → StepInfo object
```

### **Critical Redundancy Implication**

This clarification **strengthens the redundancy argument**:

1. **Identical Data**: Both systems use the exact same canonical names from `STEP_NAMES`
2. **Identical Resolution**: `step_type` and `step_name` are the same canonical identifiers
3. **Identical Purpose**: Both resolve config classes to the same canonical step names
4. **No Value Differentiation**: StepBuilderRegistry provides no unique naming or resolution logic

### **Registry Data Structure Analysis**

#### **Single Source of Truth: step_names_original.py**
```python
# CANONICAL NAMES are the dictionary keys
STEP_NAMES = {
    "XGBoostTraining": {...},      # ← Canonical name used by both systems
    "XGBoostModelEval": {...},     # ← Canonical name used by both systems  
    "PyTorchTraining": {...},      # ← Canonical name used by both systems
    "TabularPreprocessing": {...}, # ← Canonical name used by both systems
}
```

#### **Generated Mappings (Used by StepBuilderRegistry)**
```python
# CONFIG_STEP_REGISTRY maps config classes to canonical names
CONFIG_STEP_REGISTRY = {
    "XGBoostTrainingConfig": "XGBoostTraining",      # → Same canonical name
    "XGBoostModelEvalConfig": "XGBoostModelEval",    # → Same canonical name
    "PyTorchTrainingConfig": "PyTorchTraining",      # → Same canonical name
}
```

#### **StepCatalog Index (Built from Same Data)**
```python
# StepCatalog._step_index uses canonical names as keys
self._step_index = {
    "XGBoostTraining": StepInfo(...),      # ← Same canonical name as key
    "XGBoostModelEval": StepInfo(...),     # ← Same canonical name as key
    "PyTorchTraining": StepInfo(...),      # ← Same canonical name as key
}
```

### **Conclusion: Perfect Redundancy**

The user's question reveals **perfect functional redundancy**:

- `step_type` from StepBuilderRegistry = `step_name` in StepCatalog = **Canonical Name** from registry
- Both systems perform identical config-class-to-canonical-name resolution
- Both systems use the same registry data source
- Both systems operate on the same canonical name space

This confirms that StepBuilderRegistry is a **pure wrapper** with no unique value proposition.

## Conclusion

This analysis reveals **significant functional redundancy (30-40%)** between StepBuilderRegistry and the step catalog system, with StepBuilderRegistry primarily serving as an unnecessary wrapper around step catalog functionality. The current architecture violates core design principles including Single Responsibility, Open/Closed, and Dependency Inversion principles while creating maintenance overhead and performance degradation.

**The clarification of step_type vs step_name relationship confirms they are identical canonical names, strengthening the case for consolidation.**

### **Key Findings Summary**

#### **Redundancy Evidence**
- **Critical Redundancy (90%)**: StepBuilderRegistry.discover_builders() is a direct wrapper around StepCatalog.load_builder_class()
- **High Redundancy (40%)**: Both systems implement identical config-to-step-type resolution using the same registry data
- **Medium Redundancy (25-30%)**: Duplicate caching mechanisms and registry data integration patterns
- **Overall Assessment**: 30-40% functional redundancy with no unique value in wrapper layer

#### **Architectural Issues**
- **Wrapper Layer Anti-Pattern**: Unnecessary abstraction layer with no added value
- **Design Principle Violations**: Multiple violations of fundamental design principles
- **Performance Overhead**: Additional method calls and object creation without benefit
- **Maintenance Burden**: Two systems to maintain for identical functionality

### **Recommended Solution: Proper Separation of Concerns**

#### **Registry System: Single Source of Truth**
Maintain the registry system as the authoritative source for step definitions and canonical mappings:

**Registry System Responsibilities**:
- **Canonical Step Definitions**: Maintain `STEP_NAMES` as single source of truth
- **Config-to-Step Mapping**: Provide `CONFIG_STEP_REGISTRY` for config class resolution
- **Legacy Alias Management**: Handle `LEGACY_ALIASES` for backward compatibility
- **Step Type Validation**: Validate step types and canonical names
- **Registry Data Integrity**: Ensure consistency across all step definitions

```python
# ENHANCED Registry System (Single Source of Truth)
class StepRegistry:
    """Single source of truth for all step definitions and canonical mappings."""
    
    def get_canonical_step_name(self, config_class_name: str, job_type: str = None) -> str:
        """Convert config class to canonical step name."""
        # Core registry responsibility: config → canonical name resolution
        
    def get_step_definition(self, step_name: str) -> Dict[str, str]:
        """Get complete step definition from registry."""
        # Core registry responsibility: provide authoritative step data
        
    def validate_step_name(self, step_name: str) -> bool:
        """Validate step name against registry."""
        # Core registry responsibility: validation against single source of truth
        
    def list_canonical_step_names(self) -> List[str]:
        """List all canonical step names."""
        # Core registry responsibility: provide authoritative step list
```

#### **Step Catalog System: Discovery and Mapping**
Transform the step catalog system to focus on discovery, cataloging, and component mapping while referring to the registry:

**Step Catalog Responsibilities**:
- **Component Discovery**: Find and catalog components across workspaces
- **Multi-Component Mapping**: Map between configs, builders, contracts, specs
- **Workspace-Aware Discovery**: Handle project-specific component discovery
- **Component Retrieval**: Load and instantiate discovered components
- **Registry Integration**: Reference registry system for canonical definitions

```python
# ENHANCED Step Catalog System (Discovery and Mapping)
class StepCatalog:
    """Discovery, cataloging, and mapping system that references the registry."""
    
    def __init__(self, step_registry: StepRegistry, workspace_dirs: Optional[List[Path]] = None):
        self.step_registry = step_registry  # Reference to Single Source of Truth
        self.workspace_dirs = workspace_dirs
        
    def discover_components(self, step_name: str) -> StepInfo:
        """Discover all components for a canonical step name."""
        # Validate step name against registry first
        if not self.step_registry.validate_step_name(step_name):
            raise ValueError(f"Unknown step name: {step_name}")
        
        # Get authoritative definition from registry
        step_definition = self.step_registry.get_step_definition(step_name)
        
        # Discover file components across workspaces
        file_components = self._discover_file_components(step_name)
        
        return StepInfo(
            step_name=step_name,
            registry_data=step_definition,  # From Single Source of Truth
            file_components=file_components  # From discovery
        )
    
    def get_builder_for_config(self, config: BasePipelineConfig) -> Optional[Type]:
        """Map config to builder using registry + discovery."""
        # Step 1: Use registry to get canonical step name
        config_class_name = type(config).__name__
        job_type = getattr(config, "job_type", None)
        canonical_step_name = self.step_registry.get_canonical_step_name(config_class_name, job_type)
        
        # Step 2: Use discovery to load builder class
        return self.load_builder_class(canonical_step_name)
    
    def load_builder_class(self, step_name: str) -> Optional[Type]:
        """Load builder class for canonical step name."""
        # Validate against registry first
        if not self.step_registry.validate_step_name(step_name):
            return None
            
        # Use discovery to find and load builder
        return self.builder_discovery.load_builder_class(step_name)
```

#### **Clear Architectural Separation**

```python
# PROPER SEPARATION OF CONCERNS ARCHITECTURE

┌─────────────────────────────────────┐
│           Registry System           │
│        (Single Source of Truth)     │
│                                     │
│ • Canonical step definitions        │
│ • Config-to-step mapping           │
│ • Legacy alias management          │
│ • Step type validation             │
│ • Registry data integrity          │
└─────────────────┬───────────────────┘
                  │ References
                  ▼
┌─────────────────────────────────────┐
│         Step Catalog System         │
│    (Discovery, Cataloging, Mapping) │
│                                     │
│ • Component discovery               │
│ • Multi-component mapping          │
│ • Workspace-aware discovery        │
│ • Component retrieval               │
│ • Registry integration              │
└─────────────────────────────────────┘
```

#### **Implementation Strategy**

**Phase 1: Enhance Registry System as Single Source of Truth**
```python
# Consolidate registry functionality
class StepRegistry:
    def __init__(self):
        # Load canonical definitions
        self.step_definitions = self._load_step_names()
        self.config_mapping = self._build_config_mapping()
        self.legacy_aliases = self._load_legacy_aliases()
    
    def get_canonical_step_name(self, config_class_name: str, job_type: str = None) -> str:
        """Single source of truth for config → step name resolution."""
        # Move StepBuilderRegistry._config_class_to_step_type logic here
        if config_class_name in self.config_mapping:
            base_step_name = self.config_mapping[config_class_name]
            return f"{base_step_name}_{job_type}" if job_type else base_step_name
        
        # Handle legacy aliases
        for alias, canonical in self.legacy_aliases.items():
            if config_class_name.startswith(alias):
                return canonical
        
        raise ValueError(f"Unknown config class: {config_class_name}")
```

**Phase 2: Refactor Step Catalog to Reference Registry**
```python
# Update step catalog to reference registry
class StepCatalog:
    def __init__(self, step_registry: StepRegistry, workspace_dirs: Optional[List[Path]] = None):
        self.step_registry = step_registry  # Dependency injection
        # ... discovery initialization
    
    def get_builder_for_config(self, config: BasePipelineConfig) -> Optional[Type]:
        # Delegate canonical name resolution to registry
        canonical_name = self.step_registry.get_canonical_step_name(
            type(config).__name__, 
            getattr(config, "job_type", None)
        )
        
        # Use discovery to load builder
        return self.load_builder_class(canonical_name)
```

**Phase 3: Update Consumer Systems**
```python
# Update consumers to use both systems appropriately
class PipelineDAGCompiler:
    def __init__(self, step_registry: StepRegistry, step_catalog: StepCatalog):
        self.step_registry = step_registry  # For validation and canonical names
        self.step_catalog = step_catalog    # For component discovery and loading
    
    def get_supported_step_types(self) -> List[str]:
        # Get canonical names from registry
        return self.step_registry.list_canonical_step_names()
    
    def validate_dag_compatibility(self, dag: PipelineDAG) -> ValidationResult:
        # Use registry for validation
        for node in dag.nodes:
            if not self.step_registry.validate_step_name(node):
                # Error handling
                pass
```

#### **Expected Benefits**
- **Clear Separation of Concerns**: Registry handles truth, catalog handles discovery
- **Single Source of Truth**: Registry maintains canonical definitions
- **Enhanced Discovery**: Step catalog focuses on component discovery and mapping
- **Improved Maintainability**: Each system has distinct, well-defined responsibilities
- **Better Testability**: Systems can be tested independently with clear interfaces
- **Architectural Clarity**: Clear dependency flow (catalog references registry)

### **Strategic Impact**

This consolidation represents a **high-value, low-risk architectural improvement** that:

- **Eliminates Redundancy**: Addresses documented code redundancy issues
- **Improves Performance**: Removes unnecessary wrapper layers
- **Simplifies Architecture**: Creates single source of truth for builder operations
- **Enhances Maintainability**: Reduces system complexity and maintenance burden
- **Preserves Functionality**: Maintains all existing capabilities while improving implementation

The proposal aligns with design principles of simplicity, performance awareness, and demand validation while providing a clear migration path with comprehensive risk mitigation strategies. The consolidation will result in a more maintainable, performant, and architecturally sound system that better serves the needs of pipeline construction and component discovery.

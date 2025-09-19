---
tags:
  - project
  - implementation
  - config_management
  - step_catalog_integration
  - refactoring
  - system_architecture
  - deployment_portability
keywords:
  - config field management
  - step catalog integration
  - redundancy reduction
  - deployment agnostic
  - backward compatibility
  - execution document generator
  - AST-based discovery
topics:
  - config field management refactoring
  - step catalog integration
  - data structure simplification
  - deployment portability
  - code redundancy reduction
language: python
date of note: 2025-09-19
---

# Config Field Management System Refactoring Implementation Plan

## Executive Summary

This implementation plan provides a detailed roadmap for refactoring the config field management system to integrate with the unified step catalog architecture while addressing critical issues: 83% config discovery failure rate, deployment portability problems, and 47% code redundancy in data structures. The refactoring will be executed in three phases over 3 weeks, maintaining complete backward compatibility and preserving the existing config JSON file structure.

### Key Objectives

#### **Primary Objectives**
- **Fix Critical Discovery Failures**: Resolve 83% failure rate in `build_complete_config_classes()`
- **Achieve Deployment Portability**: Eliminate hardcoded module paths causing Lambda/Docker failures
- **Reduce Code Redundancy**: Target 87% reduction in ConfigClassStore, TierRegistry, CircularReferenceTracker
- **Maintain Output Format**: Preserve existing shared/specific field organization and metadata structure

#### **Secondary Objectives**
- **Enable Workspace Awareness**: Support project-specific configuration discovery
- **Enhance System Reliability**: Implement robust error handling with fallbacks
- **Improve Developer Experience**: Clear logging, error messages, and debugging capabilities

### Strategic Impact

- **ExecutionDocumentGenerator**: Direct fix for malfunction caused by missing config classes
- **Universal Deployment**: Same config files work across development, AWS Lambda, Docker, PyPI packages
- **Code Quality**: Achieve optimal 15-25% redundancy levels while maintaining functionality
- **Future-Ready Architecture**: Foundation for advanced workspace-aware config management

## Analysis Summary

### **Current System Issues** (from [Config Field Management System Analysis](../4_analysis/config_field_management_system_analysis.md))

#### **Critical Discovery Failures**
- **83% Import Failure Rate**: Only 3/18 config classes successfully imported
- **Hardcoded Module Paths**: `src.pipeline_steps.*` paths don't exist in deployment environments
- **Silent Failures**: No error reporting or fallback mechanisms

#### **Massive Code Redundancy**
- **ConfigClassStore**: 200 lines, 85% redundant (duplicates step catalog functionality)
- **TierRegistry**: 150 lines, 90% redundant (duplicates config class methods)
- **CircularReferenceTracker**: 600 lines, 95% redundant (over-engineered for use case)
- **Total Impact**: 950 lines (47% of system complexity) can be reduced to ~120 lines

#### **Deployment Portability Issues**
- **Environment-Specific Paths**: Hardcoded paths fail in Lambda, Docker, PyPI packages
- **Import Logic Fragility**: Relies on specific directory structures that vary by deployment
- **No Runtime Discovery**: Cannot adapt to different deployment environments

### **Solution Architecture** (from [Unified Step Catalog Config Field Management Refactoring Design](../1_design/unified_step_catalog_config_field_management_refactoring_design.md))

#### **Step Catalog Integration**
- **AST-Based Discovery**: Robust file analysis instead of broken import logic
- **Deployment Agnostic**: Runtime class discovery works in any environment
- **Workspace Awareness**: Project-specific configuration discovery support

#### **Data Structure Simplification**
- **Eliminate ConfigClassStore**: Replace with step catalog automatic discovery
- **Eliminate TierRegistry**: Use config classes' own `categorize_fields()` methods
- **Simplify CircularReferenceTracker**: Reduce to minimal tier-aware tracking

#### **Format Preservation**
- **Exact JSON Structure**: Maintain shared/specific field organization
- **Metadata Compatibility**: Preserve existing metadata format and content
- **Backward Compatibility**: All existing APIs continue working unchanged

## Implementation Phases

### **Phase 1: Discovery Layer Integration** (Week 1)

#### **Objective**: Fix critical discovery failures and establish step catalog integration

#### **Day 1-2: Replace build_complete_config_classes()**

**Target File**: `src/cursus/steps/configs/utils.py` (Lines 545-600)

**Current Broken Implementation**:
```python
def build_complete_config_classes() -> Dict[str, Type[BaseModel]]:
    """BROKEN: 83% failure rate due to incorrect import logic."""
    # Wrong module paths, silent failures, no fallbacks
```

**New Implementation Strategy**:
```python
def build_complete_config_classes(project_id: Optional[str] = None) -> Dict[str, Type[BaseModel]]:
    """
    REFACTORED: Step catalog integration with multiple fallback strategies.
    
    Success Rate: 83% failure → 100% success
    Deployment: Works in all environments (dev, Lambda, Docker, PyPI)
    Workspace: Optional project-specific discovery
    """
    try:
        # Primary: Step catalog unified discovery
        from ...step_catalog import StepCatalog
        catalog = StepCatalog(workspace_root)
        return catalog.build_complete_config_classes(project_id)
        
    except ImportError:
        # Fallback 1: ConfigAutoDiscovery directly
        from ...step_catalog.config_discovery import ConfigAutoDiscovery
        return ConfigAutoDiscovery(workspace_root).build_complete_config_classes(project_id)
        
    except Exception:
        # Fallback 2: Legacy implementation (for safety)
        return _legacy_build_complete_config_classes()
```

**Implementation Tasks**:
- [ ] Create new implementation with step catalog integration
- [ ] Add comprehensive error handling and logging
- [ ] Implement multiple fallback strategies
- [ ] Preserve exact function signature for backward compatibility
- [ ] Add optional `project_id` parameter for workspace awareness

**Testing Requirements**:
- [ ] Verify 100% config class discovery success rate
- [ ] Test in development environment
- [ ] Test fallback mechanisms
- [ ] Validate backward compatibility (existing code unchanged)

#### **Day 3-4: Fix load_configs Module Path Issues**

**Target File**: `src/cursus/core/config_fields/__init__.py`

**Current Issues**:
- Hardcoded module paths fail in deployment environments
- No integration with step catalog discovery
- Fragile import logic

**New Implementation Strategy**:
```python
def load_configs(input_file: str, config_classes: Optional[Dict[str, Type]] = None, 
                project_id: Optional[str] = None) -> Dict[str, Any]:
    """
    ENHANCED: Step catalog integration for deployment-agnostic loading.
    
    Portability: Works across all deployment environments
    Discovery: Automatic config class resolution
    Workspace: Project-specific loading support
    """
    if not config_classes:
        # Use step catalog for robust discovery
        config_classes = build_complete_config_classes(project_id)
    
    # Use correct module paths from actual classes (not hardcoded)
    # Implement deployment-agnostic deserialization
```

**Implementation Tasks**:
- [ ] Integrate step catalog discovery for config class resolution
- [ ] Replace hardcoded module paths with runtime class information
- [ ] Add project_id parameter for workspace-aware loading
- [ ] Implement robust error handling
- [ ] Maintain exact return format and structure

**Testing Requirements**:
- [ ] Test loading in development environment
- [ ] Simulate Lambda/Docker deployment scenarios
- [ ] Verify config file format preservation
- [ ] Validate workspace-specific loading

#### **Day 5: Simplified Serialization with Format Preservation**

**Target File**: `src/cursus/core/config_fields/type_aware_config_serializer.py`

**Current Issues**:
- Complex type preservation logic (300+ lines)
- Hardcoded module paths in serialization
- Over-engineered for actual use cases

**New Implementation Strategy**:
```python
class OptimizedTypeAwareConfigSerializer:
    """
    SIMPLIFIED: Essential type preservation with exact format compatibility.
    
    Complexity: 300+ lines → ~100 lines
    Portability: No hardcoded module paths
    Format: Exact same JSON output structure
    """
    def serialize(self, obj: Any) -> Any:
        # Handle primitives, lists, dicts with minimal metadata
        # Use step catalog for deployment-agnostic class information
        # Maintain exact JSON structure for backward compatibility
```

**Implementation Tasks**:
- [ ] Simplify serialization logic while preserving output format
- [ ] Remove hardcoded module path dependencies
- [ ] Implement minimal circular reference tracking
- [ ] Maintain exact JSON structure compatibility
- [ ] Add comprehensive testing for format preservation

**Testing Requirements**:
- [ ] Verify identical JSON output structure
- [ ] Test serialization/deserialization round-trip
- [ ] Validate metadata format preservation
- [ ] Test in multiple deployment environments

#### **Phase 1 Success Criteria**
- ✅ Config discovery success rate: 17% → 100%
- ✅ ExecutionDocumentGenerator functional (primary objective achieved)
- ✅ Zero breaking changes in existing code
- ✅ Deployment portability established
- ✅ Comprehensive logging and error handling implemented

### **Phase 2: Data Structure Simplification and Integration** (Week 2)

#### **Objective**: Eliminate redundant data structures and achieve 87% code reduction

#### **Day 1-2: Eliminate ConfigClassStore**

**Target File**: `src/cursus/core/config_fields/config_class_store.py` (200 lines)

**Current Redundancy**: 85% redundant - duplicates step catalog functionality

**Elimination Strategy**:
```python
# BEFORE: Manual registration system
class ConfigClassStore:
    _classes = {}  # Manual storage
    
    @classmethod
    def register(cls, config_class):
        # Manual registration required
    
    @classmethod 
    def get_all_classes(cls):
        # Returns manually registered classes only

# AFTER: Step catalog integration
def get_config_classes(project_id: Optional[str] = None) -> Dict[str, Type]:
    """Replace ConfigClassStore with step catalog automatic discovery."""
    return StepCatalog(workspace_root).build_complete_config_classes(project_id)
```

**Implementation Tasks**:
- [ ] Identify all ConfigClassStore usage locations
- [ ] Replace manual registration with step catalog discovery
- [ ] Update import statements throughout codebase
- [ ] Remove ConfigClassStore class and related files
- [ ] Add migration compatibility layer if needed

**Code Reduction**: 200 lines → 0 lines (100% elimination)

#### **Day 2-3: Eliminate TierRegistry**

**Target File**: `src/cursus/core/config_fields/tier_registry.py` (150 lines)

**Current Redundancy**: 90% redundant - duplicates config class methods

**Elimination Strategy**:
```python
# BEFORE: External tier storage
class TierRegistry:
    _tier_mappings = {}  # External storage
    
    @classmethod
    def register_tier_info(cls, class_name, tier_info):
        # Store tier info externally
    
    @classmethod
    def get_tier_info(cls, class_name):
        # Retrieve from external storage

# AFTER: Config class self-contained methods
def get_field_tiers(config_instance) -> Dict[str, List[str]]:
    """Use config's own categorize_fields() method."""
    return config_instance.categorize_fields()  # Already available!
```

**Implementation Tasks**:
- [ ] Identify all TierRegistry usage locations
- [ ] Replace external tier storage with config class methods
- [ ] Update field categorization logic to use self-contained methods
- [ ] Remove TierRegistry class and related files
- [ ] Verify tier information accuracy and completeness

**Code Reduction**: 150 lines → 0 lines (100% elimination)

#### **Day 3-4: Simplify CircularReferenceTracker**

**Target File**: `src/cursus/core/config_fields/circular_reference_tracker.py` (600 lines)

**Current Redundancy**: 95% redundant - over-engineered for use case

**Simplification Strategy**:
```python
# BEFORE: Complex circular reference detection (600+ lines)
class CircularReferenceTracker:
    # Complex graph analysis
    # Sophisticated detection algorithms
    # Extensive edge case handling
    # Theoretical problem handling

# AFTER: Simple tier-aware tracking (~70 lines)
class SimpleTierAwareTracker:
    def __init__(self):
        self.visited = set()  # Simple tracking
    
    def track_serialization(self, obj):
        # Tier-based prevention + minimal tracking
        # Three-tier architecture prevents most circular references
```

**Implementation Tasks**:
- [ ] Analyze actual circular reference occurrences in config objects
- [ ] Implement minimal tracking based on three-tier architecture constraints
- [ ] Replace complex detection with simple tier-aware prevention
- [ ] Remove unnecessary edge case handling
- [ ] Maintain essential circular reference protection

**Code Reduction**: 600 lines → ~70 lines (88% reduction)

#### **Day 4-5: Implement UnifiedConfigManager**

**New File**: `src/cursus/core/config_fields/unified_config_manager.py` (~120 lines)

**Integration Strategy**:
```python
class UnifiedConfigManager:
    """
    Single integrated component replacing three separate systems.
    
    Replaces: ConfigClassStore + TierRegistry + CircularReferenceTracker
    Total Reduction: 950 lines → 120 lines (87% reduction)
    """
    def __init__(self, step_catalog):
        self.step_catalog = step_catalog
        self.simple_tracker = set()  # Minimal circular reference tracking
    
    def get_config_classes(self, project_id: Optional[str] = None):
        # Step catalog integration (replaces ConfigClassStore)
        return self.step_catalog.build_complete_config_classes(project_id)
    
    def get_field_tiers(self, config_instance):
        # Config self-contained methods (replaces TierRegistry)
        return config_instance.categorize_fields()
    
    def serialize_with_tier_awareness(self, obj):
        # Simple tier-aware serialization (replaces CircularReferenceTracker)
        return self._tier_aware_serialize(obj)
```

**Implementation Tasks**:
- [ ] Design unified interface combining all three functionalities
- [ ] Implement step catalog integration for config discovery
- [ ] Add tier-aware serialization with minimal tracking
- [ ] Create comprehensive test suite for unified functionality
- [ ] Document migration path from separate components

#### **Phase 2 Success Criteria**
- ✅ 87% code reduction achieved (950 lines → 120 lines)
- ✅ ConfigClassStore eliminated (200 lines removed)
- ✅ TierRegistry eliminated (150 lines removed)
- ✅ CircularReferenceTracker simplified (600 → 70 lines)
- ✅ UnifiedConfigManager operational
- ✅ All functionality preserved with simplified architecture

### **Phase 3: Enhanced Public API and Advanced Features** (Week 3)

#### **Objective**: Enhance public APIs with workspace awareness and advanced features

#### **Day 1-2: Enhanced merge_and_save_configs**

**Target File**: `src/cursus/core/config_fields/__init__.py`

**Enhancement Strategy**:
```python
def merge_and_save_configs(
    config_list: List[Any],
    output_file: str,
    project_id: Optional[str] = None,  # NEW: Workspace awareness
    step_catalog: Optional[Any] = None  # NEW: Step catalog integration
) -> Dict[str, Any]:
    """
    ENHANCED: Workspace-aware merging with step catalog integration.
    
    Backward Compatible: Original signature preserved
    Enhanced: Optional workspace and step catalog parameters
    Format: Exact same JSON output structure maintained
    """
    # Create step catalog aware merger if parameters provided
    # Enhance metadata with workspace and framework information
    # Maintain exact output format for backward compatibility
```

**Implementation Tasks**:
- [ ] Add optional project_id parameter for workspace awareness
- [ ] Add optional step_catalog parameter for enhanced processing
- [ ] Implement StepCatalogAwareConfigMerger integration
- [ ] Enhance metadata with workspace and framework information
- [ ] Maintain complete backward compatibility

#### **Day 2-3: Enhanced load_configs with Advanced Discovery**

**Enhancement Strategy**:
```python
def load_configs(
    input_file: str, 
    config_classes: Optional[Dict[str, Type]] = None,
    project_id: Optional[str] = None  # NEW: Workspace awareness
) -> Dict[str, Any]:
    """
    ENHANCED: Workspace-aware loading with automatic project detection.
    
    Auto-Detection: Extract project_id from file metadata
    Workspace: Project-specific config class discovery
    Fallbacks: Multiple discovery strategies with graceful degradation
    """
    # Auto-detect project_id from file metadata if not provided
    # Use workspace-aware config discovery
    # Implement multiple fallback strategies
```

**Implementation Tasks**:
- [ ] Add project_id parameter for workspace-specific loading
- [ ] Implement automatic project detection from file metadata
- [ ] Add workspace-aware config class discovery
- [ ] Implement graceful fallback strategies
- [ ] Enhance error reporting and debugging information

#### **Day 3-4: Workspace-Aware Field Categorization**

**New File**: `src/cursus/core/config_fields/step_catalog_aware_categorizer.py`

**Enhancement Strategy**:
```python
class StepCatalogAwareConfigFieldCategorizer(ConfigFieldCategorizer):
    """
    Enhanced categorizer with workspace and framework awareness.
    
    Workspace: Project-specific field categorization
    Framework: Framework-specific field handling
    Preserved: All existing categorization rules and logic
    """
    def _categorize_field_with_step_catalog_context(self, field_name: str):
        # Base categorization from existing sophisticated logic
        # Enhanced with workspace-specific field detection
        # Framework-specific field handling
        # Maintain all existing rules and precedence
```

**Implementation Tasks**:
- [ ] Extend existing ConfigFieldCategorizer with step catalog integration
- [ ] Add workspace-specific field detection
- [ ] Implement framework-specific field handling
- [ ] Preserve all existing categorization rules and logic
- [ ] Add comprehensive testing for enhanced categorization

#### **Day 4-5: Performance Optimization and Testing**

**Optimization Areas**:
- **Caching**: Implement config class discovery caching
- **Performance**: Optimize AST parsing and class loading
- **Memory**: Reduce memory footprint through simplified data structures
- **Logging**: Optimize logging for production environments

**Comprehensive Testing**:
- **Unit Tests**: All individual components and functions
- **Integration Tests**: End-to-end config processing workflows
- **Deployment Tests**: Lambda, Docker, PyPI package scenarios
- **Performance Tests**: Large config file processing benchmarks
- **Compatibility Tests**: Backward compatibility validation

**Implementation Tasks**:
- [ ] Implement config class discovery caching
- [ ] Optimize AST parsing performance
- [ ] Add comprehensive unit test coverage
- [ ] Create integration test suite
- [ ] Test deployment scenarios (Lambda, Docker, PyPI)
- [ ] Benchmark performance improvements
- [ ] Validate complete backward compatibility

#### **Phase 3 Success Criteria**
- ✅ Workspace-aware field categorization operational
- ✅ Enhanced public APIs with optional advanced parameters
- ✅ Performance optimization achieved (target: 90% faster loading)
- ✅ Comprehensive test coverage (>95%)
- ✅ Universal deployment compatibility validated
- ✅ Complete backward compatibility maintained

## Risk Management

### **High Risk Items**

#### **Risk 1: Backward Compatibility Breakage**
- **Probability**: Medium
- **Impact**: High
- **Mitigation**: 
  - Preserve exact function signatures
  - Maintain identical JSON output format
  - Implement comprehensive compatibility testing
  - Create rollback plan with legacy implementation

#### **Risk 2: Performance Regression**
- **Probability**: Low
- **Impact**: Medium
- **Mitigation**:
  - Implement performance benchmarking
  - Add caching for expensive operations
  - Monitor memory usage during refactoring
  - Optimize AST parsing and class loading

#### **Risk 3: Deployment Environment Issues**
- **Probability**: Medium
- **Impact**: High
- **Mitigation**:
  - Test in all target deployment environments
  - Implement robust fallback mechanisms
  - Use deployment-agnostic discovery methods
  - Create environment-specific test suites

### **Medium Risk Items**

#### **Risk 4: Complex Integration Issues**
- **Probability**: Medium
- **Impact**: Medium
- **Mitigation**:
  - Implement phased integration approach
  - Create comprehensive integration tests
  - Maintain clear separation of concerns
  - Document all integration points

#### **Risk 5: Data Structure Migration Complexity**
- **Probability**: Low
- **Impact**: Medium
- **Mitigation**:
  - Create detailed migration documentation
  - Implement gradual replacement strategy
  - Maintain compatibility layers during transition
  - Validate data integrity throughout process

## Success Metrics

### **Immediate Success Metrics** (Week 1)
- **Config Discovery Success Rate**: 17% → 100%
- **ExecutionDocumentGenerator Functionality**: Broken → Operational
- **Deployment Compatibility**: Environment-specific → Universal
- **Error Handling**: Silent failures → Comprehensive logging

### **Intermediate Success Metrics** (Week 2)
- **Code Redundancy Reduction**: 950 lines → 120 lines (87% reduction)
- **System Complexity**: 3 separate components → 1 unified manager
- **Maintenance Overhead**: 3 components → 1 integrated component
- **Architecture Quality**: Fragmented → Unified and coherent

### **Final Success Metrics** (Week 3)
- **Performance Improvement**: Target 90% faster config loading
- **Test Coverage**: >95% comprehensive coverage
- **Workspace Support**: Basic → Full workspace-aware functionality
- **Developer Experience**: Enhanced logging, error messages, debugging

### **Long-term Success Metrics**
- **System Reliability**: Elimination of silent failures
- **Deployment Portability**: Universal compatibility across environments
- **Code Maintainability**: Simplified, unified architecture
- **Future Extensibility**: Foundation for advanced features

## Dependencies and Prerequisites

### **Required Dependencies**
- **Unified Step Catalog System**: Must be operational and tested
- **AST-based Config Discovery**: ConfigAutoDiscovery implementation complete
- **Three-Tier Architecture**: Existing config classes with `categorize_fields()` methods

### **Development Environment**
- **Python 3.8+**: Required for AST parsing and type hints
- **Testing Framework**: pytest with comprehensive test coverage
- **Development Tools**: Code coverage, performance profiling, linting

### **Deployment Environments**
- **Local Development**: Standard Python environment
- **AWS Lambda**: Serverless deployment testing
- **Docker Containers**: Containerized deployment testing
- **PyPI Package**: Package distribution testing

## Quality Assurance

### **Code Quality Standards**
- **Test Coverage**: Minimum 95% line coverage
- **Documentation**: Comprehensive docstrings and inline comments
- **Type Hints**: Full type annotation for all public APIs
- **Code Style**: Consistent formatting and linting compliance

### **Testing Strategy**
- **Unit Tests**: Individual component functionality
- **Integration Tests**: End-to-end workflow validation
- **Performance Tests**: Benchmarking and optimization validation
- **Compatibility Tests**: Backward compatibility and deployment testing

### **Review Process**
- **Code Review**: Peer review for all changes
- **Architecture Review**: Design validation and approval
- **Testing Review**: Test coverage and quality validation
- **Documentation Review**: Completeness and accuracy verification

## Rollback Plan

### **Rollback Triggers**
- **Critical Functionality Breakage**: Core config processing fails
- **Performance Regression**: >20% performance degradation
- **Deployment Failures**: Universal deployment compatibility lost
- **Data Integrity Issues**: Config file format corruption

### **Rollback Procedure**
1. **Immediate Rollback**: Revert to previous working version
2. **Issue Analysis**: Identify root cause of failure
3. **Fix Implementation**: Address issues in development environment
4. **Re-deployment**: Gradual re-introduction with enhanced testing

### **Rollback Assets**
- **Legacy Implementation**: Preserved as final fallback
- **Version Control**: Complete change history and rollback points
- **Test Suites**: Comprehensive validation for rollback verification
- **Documentation**: Rollback procedures and troubleshooting guides

## Conclusion

This implementation plan provides a comprehensive roadmap for refactoring the config field management system to achieve:

- **Critical Problem Resolution**: Fix 83% discovery failure rate and deployment portability issues
- **Code Quality Improvement**: Achieve 87% reduction in redundant data structures
- **System Reliability Enhancement**: Implement robust error handling and fallback mechanisms
- **Future-Ready Architecture**: Foundation for workspace-aware and advanced config management

The phased approach ensures minimal risk while delivering immediate value through the ExecutionDocumentGenerator fix, followed by systematic code quality improvements and advanced feature enhancements. Complete backward compatibility and format preservation ensure seamless integration with existing systems.

## References

### **Analysis Documents**
- **[Config Field Management System Analysis](../4_analysis/config_field_management_system_analysis.md)** - Comprehensive analysis of current system issues, redundancy patterns, and improvement opportunities

### **Design Documents**
- **[Unified Step Catalog Config Field Management Refactoring Design](../1_design/unified_step_catalog_config_field_management_refactoring_design.md)** - Complete architectural design for the refactoring approach
- **[Unified Step Catalog System Design](../1_design/unified_step_catalog_system_design.md)** - Base step catalog architecture and integration principles
- **[Config Field Categorization Consolidated](../1_design/config_field_categorization_consolidated.md)** - Sophisticated field categorization rules and three-tier architecture
- **[Config Manager Three-Tier Implementation](../1_design/config_manager_three_tier_implementation.md)** - Three-tier field classification and property-based derivation

### **Supporting Documents**
- **[Code Redundancy Evaluation Guide](../1_design/code_redundancy_evaluation_guide.md)** - Principles for optimal redundancy reduction
- **[Config Tiered Design](../1_design/config_tiered_design.md)** - Tiered configuration architecture principles
- **[Type-Aware Serializer](../1_design/type_aware_serializer.md)** - Advanced serialization with type preservation

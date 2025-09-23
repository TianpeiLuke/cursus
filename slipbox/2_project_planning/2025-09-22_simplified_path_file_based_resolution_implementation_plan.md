---
tags:
  - project
  - implementation
  - path_resolution
  - simplification
  - step_builders
  - deployment_agnostic
  - cleanup
keywords:
  - Path(__file__) approach
  - simplified path resolution
  - step builder relative paths
  - deployment context agnostic
  - complex solution cleanup
  - runtime path resolution
topics:
  - simplified path resolution implementation
  - Path(__file__) based approach
  - complex infrastructure cleanup
  - step builder modernization
  - deployment agnostic architecture
language: python
date of note: 2025-09-22
---

# Simplified Path(__file__) Based Resolution Implementation Plan

## Executive Summary

This implementation plan provides a roadmap for **dramatically simplifying** the path resolution system by adopting a **`Path(__file__)` based approach** that eliminates the complex portable path infrastructure while achieving the same deployment portability goals. The new approach uses relative paths in configuration and resolves them at runtime using the step builder's file location as the reference point.

### Key Objectives

#### **Primary Objectives**
- **Eliminate Complex Infrastructure**: Remove portable path properties, conversion utilities, and complex resolution logic
- **Simplify Configuration**: Use simple relative paths like `"dockers/xgboost_atoz/"` in configs
- **Runtime Resolution**: Use `Path(__file__)` in step builders to resolve paths at execution time
- **Maintain Deployment Portability**: Ensure same configs work across all deployment contexts
- **Zero Breaking Changes**: Preserve all existing functionality and APIs

#### **Secondary Objectives**
- **Improve Maintainability**: Reduce code complexity and maintenance burden
- **Enhance Reliability**: Eliminate edge cases and complex fallback mechanisms
- **Simplify Testing**: Straightforward testing without complex mocking
- **Future-Proof Architecture**: Simple, robust foundation for future enhancements

### Strategic Impact

- **Simplified Architecture**: Clean, understandable path resolution system
- **Deployment Agnostic**: Works reliably across all deployment contexts
- **Reduced Complexity**: Eliminates complex portable path infrastructure
- **Enhanced Reliability**: Simple approach reduces failure modes
- **Improved Developer Experience**: Easy to understand and maintain

## Problem Analysis

### **Current Complex Solution Issues**

#### **Over-Engineering Problems**
- **Complex Path Conversion**: Multiple strategies, fallbacks, and edge case handling
- **Portable Path Properties**: Additional properties and caching mechanisms
- **Runtime Context Dependencies**: Complex logic based on execution context
- **Multi-Strategy Resolution**: Child vs sibling directory detection logic
- **Maintenance Burden**: Complex codebase difficult to understand and maintain

#### **Current Complex Implementation**
```python
# Complex portable path properties
@property
def portable_source_dir(self) -> Optional[str]:
    if self._portable_source_dir is None:
        self._portable_source_dir = self._convert_to_relative_path(self.source_dir)
    return self._portable_source_dir

# Complex conversion logic
def _convert_to_relative_path(self, path: str) -> str:
    # Multiple strategies, fallbacks, error handling...
    
# Complex resolution utilities
def resolve_package_relative_path(relative_path: str) -> str:
    # Child vs sibling detection, multiple strategies...
```

#### **Step Builder Complexity**
```python
# Current complex approach
portable_source_dir = self.config.portable_source_dir
if portable_source_dir:
    source_dir = self.config.get_resolved_path(portable_source_dir)
    self.log_info("Resolved portable source dir %s to %s", portable_source_dir, source_dir)
else:
    source_dir = self.config.source_dir
    self.log_info("Using source directory: %s (portable: no)", source_dir)
```

### **Proposed Simple Solution**

#### **Simple Configuration Approach**
```python
# Simple relative paths in configuration
source_dir = "dockers/xgboost_atoz/"
processing_source_dir = "dockers/xgboost_atoz/scripts"
```

#### **Simple Step Builder Resolution**
```python
# Simple Path(__file__) based resolution
def _get_source_dir(self) -> str:
    """Get absolute source directory using step builder location as reference."""
    if self.config.source_dir:
        # Resolve relative to step builder file location
        builder_file = Path(__file__)
        project_root = builder_file.parent.parent.parent.parent  # Navigate to project root
        return str(project_root / self.config.source_dir)
    return None
```

### **Key Insight: Lambda Working Directory Independence**

The user correctly identified that since the absolute path is constructed at runtime relative to the step builder file location, it will **always stay within `/tmp/*/buyer_abuse_mods_template`** regardless of the Lambda working directory. This approach **completely ignores the lambda working directory** and uses the package installation location as the reference point.

#### **Lambda Context Example**
```
Step Builder Location: /tmp/buyer_abuse_mods_template/cursus/steps/builders/builder_tabular_preprocessing_step.py
Relative Config Path: dockers/xgboost_atoz/scripts
Resolved Absolute Path: /tmp/buyer_abuse_mods_template/dockers/xgboost_atoz/scripts
```

This approach works because:
1. **Step builder file location is always within the package installation**
2. **Relative paths are resolved from package root, not working directory**
3. **Lambda working directory becomes irrelevant**
4. **Same logic works across all deployment contexts**

## Implementation Phases

### **Phase 1: Configuration Simplification** (Week 1)

#### **Objective**: Update configuration classes to use simple relative paths

#### **Day 1-2: Update Base Configuration Classes**

**Target Files**:
- `src/cursus/core/base/config_base.py`
- `src/cursus/steps/configs/config_processing_step_base.py`

**Current Complex Implementation**:
```python
class BasePipelineConfig(BaseModel):
    source_dir: Optional[str] = Field(default=None)
    
    # Complex portable path infrastructure
    _portable_source_dir: Optional[str] = PrivateAttr(default=None)
    
    @property
    def portable_source_dir(self) -> Optional[str]:
        # Complex conversion logic...
    
    def _convert_to_relative_path(self, path: str) -> str:
        # Complex multi-strategy conversion...
    
    def get_resolved_path(self, relative_path: str) -> str:
        # Complex resolution logic...
```

**Simplified Implementation**:
```python
class BasePipelineConfig(BaseModel):
    """Base configuration with simple relative path support."""
    
    # Simple relative path field - users provide relative paths directly
    source_dir: Optional[str] = Field(
        default=None,
        description="Relative source directory path (e.g., 'dockers/xgboost_atoz/')"
    )
    
    # Remove all complex portable path infrastructure:
    # - No _portable_source_dir private field
    # - No portable_source_dir property
    # - No _convert_to_relative_path method
    # - No get_resolved_path method
    # - No complex path resolution utilities
```

**Implementation Tasks**:
- [x] **Remove Complex Infrastructure**: Delete portable path properties and conversion methods
- [x] **Update Field Documentation**: Clarify that paths should be relative
- [x] **Simplify Serialization**: Remove portable path fields from model_dump
- [ ] **Update Configuration Examples**: Show relative path usage patterns
- [ ] **Create Migration Guide**: Help users convert from absolute to relative paths

#### **Day 3-4: Update Processing Configuration Classes**

**Target Files**:
- `src/cursus/steps/configs/config_processing_step_base.py`
- All derived processing config classes

**Simplified Processing Config**:
```python
class ProcessingStepConfigBase(BasePipelineConfig):
    """Processing configuration with simple relative paths."""
    
    # Simple relative path field
    processing_source_dir: Optional[str] = Field(
        default=None,
        description="Relative processing source directory (e.g., 'dockers/xgboost_atoz/scripts')"
    )
    
    # Remove all complex portable path infrastructure:
    # - No _portable_processing_source_dir private field
    # - No portable_processing_source_dir property
    # - No portable_effective_source_dir property
    # - No get_portable_script_path method
    # - No complex path conversion logic
    
    # Keep existing effective source directory logic (simplified)
    @property
    def effective_source_dir(self) -> Optional[str]:
        """Get effective source directory (processing_source_dir or source_dir)."""
        return self.processing_source_dir or self.source_dir
```

**Implementation Tasks**:
- [x] **Remove Complex Processing Path Infrastructure**: Delete all portable processing path methods
- [x] **Simplify Effective Source Directory**: Keep simple fallback logic
- [x] **Update All Processing Configs**: Ensure all derived classes work with simplified approach
- [ ] **Update Processing Config Tests**: Simplify tests to match new approach
- [ ] **Validate Backward Compatibility**: Ensure existing configs continue working

#### **Day 5: Configuration Migration Strategy**

**Migration Approach**: Gradual migration with backward compatibility

**Backward Compatibility Strategy**:
```python
class BasePipelineConfig(BaseModel):
    """Base configuration with backward compatibility for absolute paths."""
    
    source_dir: Optional[str] = Field(default=None)
    
    @field_validator('source_dir')
    @classmethod
    def convert_absolute_to_relative(cls, v: str) -> str:
        """Automatically convert absolute paths to relative paths."""
        if v and Path(v).is_absolute():
            # Simple conversion: extract meaningful relative part
            if 'dockers' in v:
                # Extract from 'dockers' onwards
                parts = Path(v).parts
                docker_index = parts.index('dockers')
                relative_parts = parts[docker_index:]
                return str(Path(*relative_parts))
        return v
```

**Implementation Tasks**:
- [ ] **Add Automatic Conversion**: Convert absolute paths to relative in validators
- [ ] **Create Migration Utilities**: Tools to help users migrate existing configs
- [ ] **Update Documentation**: Guide users on new relative path approach
- [ ] **Test Migration Scenarios**: Ensure smooth transition from existing configs

#### **Phase 1 Success Criteria**
- [x] **Complex Infrastructure Removed**: All portable path properties and methods deleted
- [x] **Simple Relative Paths**: Configuration classes use simple relative path fields
- [ ] **Backward Compatibility**: Automatic conversion from absolute to relative paths
- [ ] **Documentation Updated**: Clear guidance on new relative path approach
- [x] **Zero Breaking Changes**: Existing functionality preserved

### **Phase 2: Step Builder Simplification** (Week 1-2)

#### **Objective**: Update all step builders to use simple `Path(__file__)` based resolution

#### **Day 1-2: Create Base Path Resolution Helper**

**Target File**: `src/cursus/core/base/builder_base.py`

**Simple Path Resolution Helper**:
```python
class StepBuilderBase:
    """Base step builder with simple path resolution."""
    
    def _resolve_source_dir(self, relative_path: Optional[str]) -> Optional[str]:
        """
        Resolve relative source directory to absolute path using step builder location.
        
        Args:
            relative_path: Relative path from project root (e.g., 'dockers/xgboost_atoz/')
            
        Returns:
            Absolute path resolved from step builder location
        """
        if not relative_path:
            return None
            
        # Get step builder file location
        builder_file = Path(__file__)
        
        # Navigate to project root from step builder location
        # Step builders are at: src/cursus/steps/builders/
        # So we need to go up 4 levels to reach project root
        project_root = builder_file.parent.parent.parent.parent
        
        # Resolve relative path from project root
        absolute_path = project_root / relative_path
        
        return str(absolute_path.resolve())
    
    def _resolve_script_path(self, relative_source_dir: Optional[str], entry_point: str) -> Optional[str]:
        """
        Resolve script path using step builder location.
        
        Args:
            relative_source_dir: Relative source directory
            entry_point: Script filename
            
        Returns:
            Absolute script path
        """
        if not relative_source_dir or not entry_point:
            return None
            
        source_dir = self._resolve_source_dir(relative_source_dir)
        if source_dir:
            return str(Path(source_dir) / entry_point)
        
        return None
```

**Implementation Tasks**:
- [ ] **Add Base Path Resolution Methods**: Create helper methods in StepBuilderBase
- [ ] **Determine Navigation Levels**: Confirm correct number of parent levels to project root
- [ ] **Add Error Handling**: Handle cases where paths don't exist
- [ ] **Add Logging**: Log path resolution for debugging
- [ ] **Test Path Resolution**: Verify correct resolution across deployment contexts

#### **Day 2-3: Update Training Step Builders**

**Target Files**:
- `src/cursus/steps/builders/builder_xgboost_training_step.py`
- `src/cursus/steps/builders/builder_pytorch_training_step.py`

**Current Complex Implementation**:
```python
def _create_estimator(self, output_path=None) -> XGBoost:
    # Complex portable path resolution
    portable_source_dir = self.config.portable_source_dir
    if portable_source_dir:
        source_dir = self.config.get_resolved_path(portable_source_dir)
        self.log_info("Resolved portable source dir %s to %s", portable_source_dir, source_dir)
    else:
        source_dir = self.config.source_dir
        self.log_info("Using source directory: %s (portable: no)", source_dir)
    
    return XGBoost(
        entry_point=self.config.training_entry_point,
        source_dir=source_dir,
        # ... other parameters
    )
```

**Simplified Implementation**:
```python
def _create_estimator(self, output_path=None) -> XGBoost:
    # Simple Path(__file__) based resolution
    source_dir = self._resolve_source_dir(self.config.source_dir)
    
    return XGBoost(
        entry_point=self.config.training_entry_point,
        source_dir=source_dir,
        # ... other parameters
    )
```

**Implementation Tasks**:
- [ ] **Update XGBoostTrainingStepBuilder**: Replace complex resolution with simple helper
- [ ] **Update PyTorchTrainingStepBuilder**: Replace complex resolution with simple helper
- [ ] **Remove Complex Logic**: Delete all portable path resolution code
- [ ] **Add Simple Logging**: Log resolved paths for debugging
- [ ] **Test Training Step Creation**: Verify estimators are created correctly

#### **Day 3-4: Update Model Step Builders**

**Target Files**:
- `src/cursus/steps/builders/builder_xgboost_model_step.py`
- `src/cursus/steps/builders/builder_pytorch_model_step.py`

**Simplified Model Step Implementation**:
```python
def _create_model(self, model_data: str) -> XGBoostModel:
    # Simple Path(__file__) based resolution
    source_dir = self._resolve_source_dir(self.config.source_dir)
    
    return XGBoostModel(
        model_data=model_data,
        entry_point=self.config.entry_point,
        source_dir=source_dir,
        # ... other parameters
    )
```

**Implementation Tasks**:
- [ ] **Update XGBoostModelStepBuilder**: Replace complex resolution with simple helper
- [ ] **Update PyTorchModelStepBuilder**: Replace complex resolution with simple helper
- [ ] **Remove Complex Logic**: Delete all portable path resolution code
- [ ] **Test Model Creation**: Verify models are created correctly

#### **Day 4-5: Update Processing Step Builders**

**Target Files**:
- `src/cursus/steps/builders/builder_tabular_preprocessing_step.py`
- `src/cursus/steps/builders/builder_model_calibration_step.py`
- `src/cursus/steps/builders/builder_package_step.py`
- All other processing step builders

**Current Complex Implementation**:
```python
def create_step(self, **kwargs) -> ProcessingStep:
    # Complex portable script path resolution
    portable_script_path = self.config.get_portable_script_path()
    if portable_script_path:
        script_path = self.config.get_resolved_path(portable_script_path)
        self.log_info("Resolved portable script path %s to %s", portable_script_path, script_path)
    else:
        script_path = self.config.get_script_path()
        self.log_info("Using script path: %s (portable: no)", script_path)
    
    # For processor.run(), use entry point and source_dir separately
    return processor.run(
        code=self.config.processing_entry_point,  # Just filename
        source_dir=source_dir,  # Resolved directory
        # ... other parameters
    )
```

**Simplified Implementation**:
```python
def create_step(self, **kwargs) -> ProcessingStep:
    # Simple Path(__file__) based resolution
    source_dir = self._resolve_source_dir(self.config.effective_source_dir)
    
    # For processor.run(), use entry point and source_dir separately
    return processor.run(
        code=self.config.processing_entry_point,  # Just filename
        source_dir=source_dir,  # Resolved directory
        # ... other parameters
    )
```

**Implementation Tasks**:
- [ ] **Update All Processing Step Builders**: Replace complex resolution with simple helper
- [ ] **Use effective_source_dir**: Leverage simplified effective source directory property
- [ ] **Remove Complex Logic**: Delete all portable path resolution code
- [ ] **Maintain SageMaker Compatibility**: Keep entry point and source_dir separation
- [ ] **Test Processing Step Creation**: Verify all processing steps work correctly

#### **Phase 2 Success Criteria**
- [ ] **Base Helper Methods**: Path resolution helpers added to StepBuilderBase
- [ ] **All Step Builders Updated**: 13 step builders using simple Path(__file__) approach
- [ ] **Complex Logic Removed**: All portable path resolution code deleted
- [ ] **SageMaker Compatibility**: Proper entry point and source_dir usage maintained
- [ ] **Zero Breaking Changes**: All step builders continue working correctly

### **Phase 3: Infrastructure Cleanup** (Week 2)

#### **Objective**: Remove all complex portable path infrastructure

#### **Day 1-2: Remove Path Resolution Utilities**

**Target Files**:
- `src/cursus/core/utils/path_resolution.py` (DELETE)
- `test/core/utils/test_path_resolution.py` (DELETE)

**Files to Clean Up**:
```python
# DELETE ENTIRE FILE: src/cursus/core/utils/path_resolution.py
# Contains:
# - get_package_relative_path()
# - resolve_package_relative_path()
# - Complex multi-strategy resolution logic
# - Child vs sibling directory detection

# DELETE ENTIRE FILE: test/core/utils/test_path_resolution.py
# Contains:
# - 16 complex path resolution tests
# - Lambda simulation tests
# - Multi-strategy resolution tests
```

**Implementation Tasks**:
- [ ] **Delete Path Resolution Utilities**: Remove entire path_resolution.py file
- [ ] **Delete Complex Tests**: Remove test_path_resolution.py file
- [ ] **Update Imports**: Remove imports of deleted utilities from other files
- [ ] **Clean Up __init__.py**: Remove path resolution exports
- [ ] **Verify No Dependencies**: Ensure no other code depends on deleted utilities

#### **Day 2-3: Remove Portable Path Properties**

**Target Files**:
- `src/cursus/core/base/config_base.py`
- `src/cursus/steps/configs/config_processing_step_base.py`

**Properties to Remove from BasePipelineConfig**:
```python
# DELETE these from BasePipelineConfig:
_portable_source_dir: Optional[str] = PrivateAttr(default=None)

@property
def portable_source_dir(self) -> Optional[str]:
    # DELETE entire method

def _convert_to_relative_path(self, path: str) -> str:
    # DELETE entire method

def get_resolved_path(self, relative_path: str) -> str:
    # DELETE entire method

def _convert_via_common_parent(self, path: str, reference_location: Optional[Path] = None) -> str:
    # DELETE entire method

def _find_common_parent(self, path1: Path, path2: Path) -> Optional[Path]:
    # DELETE entire method
```

**Properties to Remove from ProcessingStepConfigBase**:
```python
# DELETE these from ProcessingStepConfigBase:
_portable_processing_source_dir: Optional[str] = PrivateAttr(default=None)
_portable_script_path: Optional[str] = PrivateAttr(default=None)

@property
def portable_processing_source_dir(self) -> Optional[str]:
    # DELETE entire method

@property
def portable_effective_source_dir(self) -> Optional[str]:
    # DELETE entire method

def get_portable_script_path(self, default_path: Optional[str] = None) -> Optional[str]:
    # DELETE entire method

def get_resolved_effective_source_dir(self) -> Optional[str]:
    # DELETE entire method
```

**Implementation Tasks**:
- [ ] **Remove Portable Path Properties**: Delete all portable path properties and methods
- [ ] **Remove Private Fields**: Delete all portable path private attributes
- [ ] **Simplify model_dump**: Remove portable path fields from serialization
- [ ] **Clean Up Imports**: Remove imports related to deleted functionality
- [ ] **Update Documentation**: Remove references to deleted portable path features

#### **Day 3-4: Remove Complex Tests**

**Target Files**:
- `test/integration/test_portable_path_resolution_integration.py` (DELETE or SIMPLIFY)
- Various unit tests with complex portable path testing

**Test Cleanup Strategy**:
```python
# BEFORE: Complex portable path integration tests
class TestPortablePathResolutionIntegration:
    def test_mods_pipeline_error_scenario_simulation(self):
        # Complex test with temp directories, config serialization,
        # portable path conversion, step builder mocking, etc.

# AFTER: Simple path resolution tests
class TestSimplePathResolution:
    def test_step_builder_path_resolution(self):
        # Simple test that verifies Path(__file__) based resolution works
        config = TabularPreprocessingConfig(
            processing_source_dir="dockers/xgboost_atoz/scripts",
            processing_entry_point="tabular_preprocessing.py"
        )
        
        builder = TabularPreprocessingStepBuilder(config)
        source_dir = builder._resolve_source_dir(config.processing_source_dir)
        
        assert source_dir is not None
        assert source_dir.endswith("dockers/xgboost_atoz/scripts")
        assert Path(source_dir).is_absolute()
```

**Implementation Tasks**:
- [ ] **Delete Complex Integration Tests**: Remove or drastically simplify integration tests
- [ ] **Create Simple Path Tests**: Add basic tests for Path(__file__) resolution
- [ ] **Test Step Builder Helpers**: Verify base path resolution methods work
- [ ] **Remove Mocking Complexity**: Eliminate complex mocking and temp directory setup
- [ ] **Focus on Core Functionality**: Test that step builders can resolve paths correctly

#### **Day 4-5: Documentation Cleanup**

**Target Files**:
- `slipbox/1_design/config_portability_path_resolution_design.md` (UPDATE)
- `slipbox/1_design/deployment_context_agnostic_path_resolution_design.md` (UPDATE)
- Various documentation references to portable paths

**Documentation Updates**:
- [ ] **Update Design Documents**: Reflect simplified Path(__file__) approach
- [ ] **Remove Complex Architecture References**: Delete references to portable path infrastructure
- [ ] **Add Simple Approach Documentation**: Document new Path(__file__) based approach
- [ ] **Update Examples**: Show simple relative path configuration examples
- [ ] **Create Migration Guide**: Help users transition from complex to simple approach

#### **Phase 3 Success Criteria**
- [ ] **Complex Infrastructure Deleted**: All portable path utilities and properties removed
- [ ] **Tests Simplified**: Complex integration tests replaced with simple path resolution tests
- [ ] **Documentation Updated**: All references to complex approach removed or updated
- [ ] **Clean Codebase**: No remaining references to deleted portable path infrastructure
- [ ] **Reduced Complexity**: Significantly simplified codebase with same functionality

### **Phase 4: Testing and Validation** (Week 2-3)

#### **Objective**: Comprehensive testing of simplified approach

#### **Day 1-2: Simple Path Resolution Testing**

**Test Strategy**: Focus on core functionality without complex mocking

**Simple Test Examples**:
```python
class TestSimplePathResolution:
    """Test simple Path(__file__) based path resolution."""
    
    def test_resolve_source_dir_basic(self):
        """Test basic source directory resolution."""
        builder = TabularPreprocessingStepBuilder(config=mock_config)
        
        # Test with relative path
        result = builder._resolve_source_dir("dockers/xgboost_atoz")
        
        assert result is not None
        assert Path(result).is_absolute()
        assert result.endswith("dockers/xgboost_atoz")
    
    def test_resolve_script_path_basic(self):
        """Test basic script path resolution."""
        builder = TabularPreprocessingStepBuilder(config=mock_config)
        
        result = builder._resolve_script_path("dockers/xgboost_atoz/scripts", "tabular_preprocessing.py")
        
        assert result is not None
        assert Path(result).is_absolute()
        assert result.endswith("dockers/xgboost_atoz/scripts/tabular_preprocessing.py")
    
    def test_step_creation_with_relative_paths(self):
        """Test step creation works with relative paths."""
        config = TabularPreprocessingConfig(
            processing_source_dir="dockers/xgboost_atoz/scripts",
            processing_entry_point="tabular_preprocessing.py",
            # ... other required fields
        )
        
        builder = TabularPreprocessingStepBuilder(config)
        
        # Should not raise any exceptions
        step = builder.create_step()
        assert step is not None
```

**Implementation Tasks**:
- [ ] **Create Simple Test Suite**: Focus on core path resolution functionality
- [ ] **Test All Step Builder Types**: Verify each step builder can resolve paths
- [ ] **Test Edge Cases**: Handle None paths, empty paths, invalid paths
- [ ] **Test Cross-Platform**: Verify works on different operating systems
- [ ] **Performance Testing**: Ensure Path(__file__) approach is performant

#### **Day 2-3: Deployment Context Testing**

**Deployment Testing Strategy**: Test in realistic deployment contexts

**Test Scenarios**:
```python
class TestDeploymentContexts:
    """Test path resolution across deployment contexts."""
    
    def test_development_environment(self):
        """Test in development environment structure."""
        # Test with typical development directory structure
        pass
    
    def test_lambda_environment_simulation(self):
        """Test Lambda-like environment structure."""
        # Simulate Lambda package installation structure
        # /tmp/buyer_abuse_mods_template/cursus/
        # /tmp/buyer_abuse_mods_template/dockers/
        pass
    
    def test_container_environment_simulation(self):
        """Test container-like environment structure."""
        # Simulate container package structure
        pass
    
    def test_pypi_package_simulation(self):
        """Test PyPI package installation structure."""
        # Simulate site-packages installation
        pass
```

**Implementation Tasks**:
- [ ] **Simulate Deployment Contexts**: Create realistic test environments
- [ ] **Test Lambda Structure**: Verify works in Lambda-like directory structure
- [ ] **Test Container Structure**: Verify works in container environments
- [ ] **Test PyPI Structure**: Verify works with PyPI package installations
- [ ] **Cross-Context Compatibility**: Ensure same config works across all contexts

#### **Day 3-4: End-to-End Pipeline Testing**

**Pipeline Testing Strategy**: Test complete pipeline creation with simplified approach

**End-to-End Test Examples**:
```python
class TestEndToEndPipeline:
    """Test complete pipeline creation with simplified path resolution."""
    
    def test_complete_pipeline_creation(self):
        """Test creating a complete pipeline with relative paths."""
        # Create configs with relative paths
        base_config = BasePipelineConfig(
            source_dir="dockers/xgboost_atoz",
            # ... other fields
        )
        
        processing_config = ProcessingStepConfigBase.from_base_config(
            base_config,
            processing_source_dir="dockers/xgboost_atoz/scripts"
        )
        
        # Create multiple step types
        training_config = XGBoostTrainingConfig.from_base_config(base_config, ...)
        processing_config = TabularPreprocessingConfig.from_base_config(processing_config, ...)
        
        # Create step builders
        training_builder = XGBoostTrainingStepBuilder(training_config)
        processing_builder = TabularPreprocessingStepBuilder(processing_config)
        
        # Create steps
        training_step = training_builder.create_step()
        processing_step = processing_builder.create_step()
        
        # Verify steps created successfully
        assert training_step is not None
        assert processing_step is not None
    
    def test_config_serialization_deserialization(self):
        """Test config save/load cycle with relative paths."""
        # Create config with relative paths
        config = TabularPreprocessingConfig(
            processing_source_dir="dockers/xgboost_atoz/scripts",
            # ... other fields
        )
        
        # Serialize to JSON
        config_data = config.model_dump()
        
        # Deserialize from JSON
        loaded_config = TabularPreprocessingConfig(**config_data)
        
        # Verify paths are preserved
        assert loaded_config.processing_source_dir == "dockers/xgboost_atoz/scripts"
        
        # Verify step creation works with loaded config
        builder = TabularPreprocessingStepBuilder(loaded_config)
        step = builder.create_step()
        assert step is not None
```

**Implementation Tasks**:
- [ ] **Test Complete Pipeline Creation**: Verify end-to-end pipeline creation works
- [ ] **Test Config Serialization**: Verify save/load cycle preserves relative paths
- [ ] **Test Multiple Step Types**: Verify all step types work with simplified approach
- [ ] **Test Step Dependencies**: Verify step dependencies work correctly
- [ ] **Test Error Handling**: Verify graceful handling of path resolution errors

#### **Day 4-5: Performance and Reliability Testing**

**Performance Testing**:
```python
class TestPerformanceAndReliability:
    """Test performance and reliability of simplified approach."""
    
    def test_path_resolution_performance(self):
        """Test Path(__file__) resolution performance."""
        builder = TabularPreprocessingStepBuilder(config)
        
        # Benchmark path resolution
        start_time = time.time()
        for _ in range(1000):
            result = builder._resolve_source_dir("dockers/xgboost_atoz/scripts")
        end_time = time.time()
        
        # Should be very fast
        assert (end_time - start_time) < 0.1
    
    def test_reliability_with_missing_paths(self):
        """Test behavior when paths don't exist."""
        builder = TabularPreprocessingStepBuilder(config)
        
        # Test with non-existent path
        result = builder._resolve_source_dir("non/existent/path")
        
        # Should still return a path (may not exist)
        assert result is not None
        assert Path(result).is_absolute()
    
    def test_concurrent_path_resolution(self):
        """Test path resolution under concurrent access."""
        # Test multiple threads resolving paths simultaneously
        pass
```

**Implementation Tasks**:
- [ ] **Performance Benchmarking**: Measure Path(__file__) resolution performance
- [ ] **Reliability Testing**: Test behavior with edge cases and errors
- [ ] **Concurrent Access Testing**: Verify thread safety
- [ ] **Memory Usage Testing**: Ensure no memory leaks or excessive usage
- [ ] **Stress Testing**: Test with large numbers of step builders and configs

#### **Phase 4 Success Criteria**
- [ ] **Simple Test Suite**: Comprehensive tests without complex mocking
- [ ] **Deployment Context Validation**: Works across all deployment environments
- [ ] **End-to-End Testing**: Complete pipeline creation works correctly
- [ ] **Performance Validation**: Path resolution is fast and efficient
- [ ] **Reliability Confirmation**: Robust error handling and edge case management

## Migration Strategy

### **Backward Compatibility Approach**

#### **Automatic Path Conversion**
```python
@field_validator('source_dir')
@classmethod
def convert_absolute_to_relative(cls, v: str) -> str:
    """Automatically convert absolute paths to relative paths."""
    if v and Path(v).is_absolute():
        # Extract meaningful relative part
        if 'dockers' in v:
            parts = Path(v).parts
            docker_index = parts.index('dockers')
            relative_parts = parts[docker_index:]
            return str(Path(*relative_parts))
    return v
```

#### **Migration Utilities**
```python
def migrate_config_to_relative_paths(config_file: str) -> None:
    """Migrate existing config file to use relative paths."""
    with open(config_file, 'r') as f:
        config_data = json.load(f)
    
    # Convert absolute paths to relative
    for key, value in config_data.items():
        if key.endswith('_dir') and isinstance(value, str) and Path(value).is_absolute():
            if 'dockers' in value:
                parts = Path(value).parts
                docker_index = parts.index('dockers')
                relative_parts = parts[docker_index:]
                config_data[key] = str(Path(*relative_parts))
    
    # Save updated config
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)
```

### **User Migration Guide**

#### **Step 1: Update Configuration Files**
```python
# BEFORE: Absolute paths
config = BasePipelineConfig(
    source_dir="/home/user/project/dockers/xgboost_atoz"
)

# AFTER: Relative paths
config = BasePipelineConfig(
    source_dir="dockers/xgboost_atoz"
)
```

#### **Step 2: Verify Step Creation**
```python
# Test that step creation still works
builder = XGBoostTrainingStepBuilder(config)
step = builder.create_step()  # Should work without changes
```

#### **Step 3: Update Documentation and Examples**
- Update all configuration examples to use relative paths
- Update developer guides to explain new Path(__file__) approach
- Create troubleshooting guide for path resolution issues

## Cleanup Strategy

### **Complex Infrastructure to Remove**

#### **Files to Delete**
- `src/cursus/core/utils/path_resolution.py` - Complex multi-strategy path resolution utilities
- `test/core/utils/test_path_resolution.py` - 16 complex path resolution tests
- `test/integration/test_portable_path_resolution_integration.py` - Complex integration tests

#### **Methods to Remove from BasePipelineConfig**
```python
# DELETE these methods and properties:
_portable_source_dir: Optional[str] = PrivateAttr(default=None)
@property
def portable_source_dir(self) -> Optional[str]:
def _convert_to_relative_path(self, path: str) -> str:
def get_resolved_path(self, relative_path: str) -> str:
def _convert_via_common_parent(self, path: str, reference_location: Optional[Path] = None) -> str:
def _find_common_parent(self, path1: Path, path2: Path) -> Optional[Path]:
```

#### **Methods to Remove from ProcessingStepConfigBase**
```python
# DELETE these methods and properties:
_portable_processing_source_dir: Optional[str] = PrivateAttr(default=None)
_portable_script_path: Optional[str] = PrivateAttr(default=None)
@property
def portable_processing_source_dir(self) -> Optional[str]:
@property
def portable_effective_source_dir(self) -> Optional[str]:
def get_portable_script_path(self, default_path: Optional[str] = None) -> Optional[str]:
def get_resolved_effective_source_dir(self) -> Optional[str]:
```

#### **Step Builder Code to Remove**
```python
# DELETE complex portable path resolution from all step builders:
portable_source_dir = self.config.portable_source_dir
if portable_source_dir:
    source_dir = self.config.get_resolved_path(portable_source_dir)
    self.log_info("Resolved portable source dir %s to %s", portable_source_dir, source_dir)
else:
    source_dir = self.config.source_dir
    self.log_info("Using source directory: %s (portable: no)", source_dir)
```

### **Cleanup Verification Checklist**
- [ ] **No Imports**: Verify no remaining imports of deleted path resolution utilities
- [ ] **No References**: Search codebase for any remaining references to portable path methods
- [ ] **No Tests**: Ensure no tests depend on deleted portable path functionality
- [ ] **Documentation Clean**: Remove all references to complex portable path approach
- [ ] **Examples Updated**: All configuration examples use simple relative paths

## Risk Management

### **Low Risk Items**

#### **Risk 1: Path Navigation Level Errors**
- **Probability**: Low
- **Impact**: Medium
- **Mitigation**: 
  - Carefully verify step builder location relative to project root
  - Test path resolution across different deployment contexts
  - Add error handling for incorrect navigation levels

#### **Risk 2: Cross-Platform Path Issues**
- **Probability**: Low
- **Impact**: Low
- **Mitigation**:
  - Use pathlib.Path for cross-platform compatibility
  - Test on Windows, macOS, and Linux
  - Handle path separator differences automatically

### **Very Low Risk Items**

#### **Risk 3: Performance Impact**
- **Probability**: Very Low
- **Impact**: Very Low
- **Mitigation**:
  - Path(__file__) resolution is extremely fast
  - No complex logic or multiple strategies
  - Simple string operations and path joining

#### **Risk 4: Backward Compatibility**
- **Probability**: Very Low
- **Impact**: Medium
- **Mitigation**:
  - Add automatic conversion from absolute to relative paths
  - Preserve all existing APIs and functionality
  - Gradual migration with fallback support

## Success Metrics

### **Immediate Success Metrics** (Week 1)
- [ ] **Complex Infrastructure Removed**: All portable path properties and utilities deleted
- [ ] **Simple Configuration**: Configs use simple relative paths like "dockers/xgboost_atoz/"
- [ ] **Base Helper Methods**: Path resolution helpers added to StepBuilderBase
- [ ] **Zero Breaking Changes**: All existing functionality preserved

### **Intermediate Success Metrics** (Week 2)
- [ ] **All Step Builders Updated**: 13 step builders using simple Path(__file__) approach
- [ ] **Complex Tests Removed**: Integration tests simplified or deleted
- [ ] **Documentation Updated**: All references to complex approach removed
- [ ] **Migration Tools**: Utilities to help users transition to relative paths

### **Final Success Metrics** (Week 3)
- [ ] **Clean Codebase**: No remaining complex portable path infrastructure
- [ ] **Simple Testing**: Straightforward tests without complex mocking
- [ ] **Universal Portability**: Same configs work across all deployment contexts
- [ ] **Improved Maintainability**: Significantly reduced code complexity

### **Long-term Success Metrics**
- **Reduced Maintenance Burden**: Simpler codebase easier to understand and maintain
- **Enhanced Reliability**: Fewer failure modes and edge cases
- **Improved Developer Experience**: Clear, understandable path resolution approach
- **Future-Proof Architecture**: Simple foundation for future enhancements

## Key Benefits of Simplified Approach

### **Technical Benefits**
- **Eliminates Complex Logic**: No multi-strategy resolution, fallbacks, or edge case handling
- **Reduces Code Complexity**: Removes hundreds of lines of complex path conversion code
- **Improves Reliability**: Simple approach has fewer failure modes
- **Enhances Performance**: Direct Path(__file__) resolution is faster than complex conversion
- **Simplifies Testing**: No need for complex mocking or temp directory setup

### **Operational Benefits**
- **Easier Maintenance**: Simple code is easier to understand and modify
- **Reduced Bug Surface**: Fewer complex interactions means fewer potential bugs
- **Clearer Documentation**: Simple approach is easier to document and explain
- **Faster Development**: Less time spent debugging complex path resolution issues
- **Better Onboarding**: New developers can understand the system more quickly

### **Deployment Benefits**
- **Lambda Working Directory Independence**: Completely ignores Lambda working directory
- **Universal Compatibility**: Same logic works across all deployment contexts
- **Package Installation Agnostic**: Works regardless of where package is installed
- **Runtime Context Agnostic**: Path resolution based on package location, not execution context

## Conclusion

This simplified `Path(__file__)` based approach achieves the same deployment portability goals as the complex portable path infrastructure while dramatically reducing code complexity and maintenance burden. The key insight is that using the step builder's file location as the reference point for path resolution provides universal deployment compatibility without the need for complex runtime context detection or multi-strategy resolution logic.

**Key Innovation**: By resolving paths relative to the step builder file location using `Path(__file__).parent.parent.parent.parent`, we ensure that paths are always resolved from within the package installation directory, making the Lambda working directory completely irrelevant and providing universal deployment portability.

**Final Deliverables**:
- **Simplified Configuration**: Users provide simple relative paths like "dockers/xgboost_atoz/"
- **Runtime Path Resolution**: Step builders resolve paths using Path(__file__) at execution time
- **Universal Deployment Portability**: Same configs work across all deployment contexts
- **Dramatically Reduced Complexity**: Elimination of complex portable path infrastructure
- **Zero Breaking Changes**: All existing functionality preserved with simpler implementation
- **Enhanced Maintainability**: Clean, understandable codebase for future development

## References

### **Previous Implementation Plans**
- **[2025-09-20 Config Portability Path Resolution Implementation Plan](./2025-09-20_config_portability_path_resolution_implementation_plan.md)** - Original complex approach that this plan replaces
- **[2025-09-22 MODS Lambda Sibling Directory Path Resolution Fix Completion](./2025-09-22_mods_lambda_sibling_directory_path_resolution_fix_completion.md)** - Complex fix that revealed the need for simplification

### **Design Documents**
- **[Config Portability Path Resolution Design](../1_design/config_portability_path_resolution_design.md)** - Original design document
- **[Deployment Context Agnostic Path Resolution Design](../1_design/deployment_context_agnostic_path_resolution_design.md)** - Enhanced design document

### **Analysis Documents**
- **[MODS Pipeline Path Resolution Error Analysis](.internal/mods_pipeline_path_resolution_error_analysis.md)** - Root cause analysis that led to this simplified approach

### **Implementation Standards**
- **[Documentation YAML Frontmatter Standard](../1_design/documentation_yaml_frontmatter_standard.md)** - Documentation formatting standards
- **[Config Tiered Design](../1_design/config_tiered_design.md)** - Three-tier configuration architecture principles
